"""Plan-level optimizer for complete evidence retrieval slates.

This module gives a compiled ontology plan first-class semantics. Its
signature-grounded plan treats semantic/fact retrieval as an access operator,
binds scan tuples to logical evidence slots, materializes a provenance-valid
chain, and lets a calibrated cost model decide whether to commit the complete
physical slate. Gold labels are never consumed at runtime.
"""

from __future__ import annotations

import json
import hashlib
import math
import os
import re
from bisect import bisect_right
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from .query_plan_nonlinear_delta_admission import predict_ensemble
from .query_plan_supervised_evidence_ranker import final_state_hash


MODEL_TYPE = "counterfactual_complete_evidence_slate_cbo"
GLOBAL_MODEL_KEY = "__global_plan_outcome__"
STRUCTURAL_EXPERT_PREFIX = "__structural_expert__::"
OBJECTIVE_CLASS_EXPERT_PREFIX = "__objective_class_expert__::"
OBJECTIVE_FAMILY_EXPERT_PREFIX = "__objective_family_expert__::"
TOPOLOGY_ROUTING_STRICT_V280 = "strict_v280"
TOPOLOGY_ROUTING_RECALL_SAFE_V281 = "recall_safe_v281"
TOPOLOGY_ROUTING_BOUNDED_V282 = "bounded_v282"
TOPOLOGY_ROUTING_BOUNDED_V283 = "bounded_v283"
TOPOLOGY_ROUTING_WAREHOUSE_V284 = "warehouse_v284"
TOPOLOGY_ROUTING_COMBINED_V285 = "combined_v285"
TOPOLOGY_ROUTING_HIERARCHICAL_V286 = "hierarchical_v286"
TOPOLOGY_ROUTING_OBJECTIVE_FAMILY_V287 = "objective_family_v287"
TOPOLOGY_ROUTING_MULTI_ACCESS_V289 = "multi_access_v289"
TOPOLOGY_ROUTING_STRUCTURAL_RECOVERY_V293 = "structural_recovery_v293"
SCHEMA_VERSION = "counterfactual_complete_evidence_slate_v19"
PLAN_SEMANTIC = "semantic_fact_scan"
PLAN_SEMANTIC_TOP2_NOOP = "semantic_fact_scan_top2_noop"
PLAN_RAW_SEMANTIC = "raw_semantic_graph_scan"
PLAN_ONTOLOGY_PHYSICAL = "ontology_physical_complete_slate"
PLAN_STRUCTURAL_EXECUTION = "ontology_structural_execution_slate"
PLAN_CHAIN_COMPLETION = "incumbent_preserving_chain_completion"
PLAN_ONTOLOGY = "ontology_proof_scan"
PLAN_HYBRID = "proof_aware_topk_merge"
PLAN_PROOF_CHAIN = "proof_chain_materialization"
PLAN_TOP2_SEMIJOIN = "missing_hop_top2_semijoin"
PLAN_TOP5_SEMIJOIN = "r2_preserving_top5_semijoin"
PLAN_BOUND_PROOF = "entity_bound_proof_join"
PLAN_QUERY_LOCAL_TOP2_PAIR = "query_local_provenance_top2_pair"
PLAN_QUERY_LOCAL_TOP5 = "query_local_provenance_top5_completion"
PLAN_SIGNATURE_GROUNDED = "signature_grounded_unified_slate"
PLAN_BRANCH_COMPLETE_TOP2 = "branch_complete_atomic_top2"
PLAN_RANK_PARTITIONED_TOP2 = "rank_partitioned_top2_ordering"
PLAN_RANK_PARTITIONED_TOP5 = "rank_partitioned_top5_completion"
PLAN_RANK_PARTITIONED_NESTED = "rank_partitioned_nested_topk"
PLAN_LEARNED_RESIDUAL_TOP5 = "learned_residual_top5_completion"
PLAN_ONTOLOGY_STRUCTURAL_TOP5 = "ontology_structural_rank5_completion"
PLAN_ONTOLOGY_WAREHOUSE_TOP5 = "ontology_warehouse_rank5_completion"
PLAN_TOPOLOGY_MULTI_ACCESS_TOP5 = "topology_conditioned_multi_access_top5"
ALTERNATIVE_PLANS = (
    PLAN_SEMANTIC_TOP2_NOOP,
    PLAN_RAW_SEMANTIC,
    PLAN_ONTOLOGY_PHYSICAL,
    PLAN_STRUCTURAL_EXECUTION,
    PLAN_CHAIN_COMPLETION,
    PLAN_SIGNATURE_GROUNDED,
    PLAN_BRANCH_COMPLETE_TOP2,
    PLAN_BOUND_PROOF,
    PLAN_QUERY_LOCAL_TOP2_PAIR,
    PLAN_QUERY_LOCAL_TOP5,
    PLAN_RANK_PARTITIONED_TOP2,
    PLAN_RANK_PARTITIONED_TOP5,
    PLAN_RANK_PARTITIONED_NESTED,
    PLAN_LEARNED_RESIDUAL_TOP5,
    PLAN_ONTOLOGY_STRUCTURAL_TOP5,
    PLAN_ONTOLOGY_WAREHOUSE_TOP5,
    PLAN_TOPOLOGY_MULTI_ACCESS_TOP5,
    PLAN_ONTOLOGY,
    PLAN_PROOF_CHAIN,
    PLAN_TOP2_SEMIJOIN,
    PLAN_TOP5_SEMIJOIN,
    PLAN_HYBRID,
)

FEATURE_NAMES = (
    "matched_plan_count_log",
    "strict_executable_path_count_log",
    "executable_path_count_log",
    "relation_slot_count",
    "explicit_relation_slot_count",
    "latent_contract_recovered",
    "logical_required_hop_count",
    "logical_required_branch_count",
    "query_relative_clause_count",
    "query_comparison_cue",
    "query_nested_bridge_cue",
    "anchor_count_log",
    "proof_signal_candidate_count_log",
    "complete_proof_candidate_count_log",
    "local_proof_path_count_log",
    "local_proof_path_passage_count_log",
    "bound_proof_path_count_log",
    "bound_proof_path_passage_count_log",
    "proof_signal_fraction",
    "answer_type_compatible_fraction",
    "mean_relation_coverage",
    "max_relation_coverage",
    "max_anchor_coverage",
    "semantic_score_margin_top2",
    "semantic_score_margin_top5",
    "semantic_ontology_top2_overlap",
    "semantic_ontology_top5_overlap",
    "ontology_top2_novelty",
    "ontology_top5_novelty",
    "plan_semantic_top2_overlap",
    "plan_semantic_top5_overlap",
    "plan_ontology_top2_overlap",
    "plan_ontology_top5_overlap",
    "plan_complete_proof_top2_fraction",
    "plan_complete_proof_top5_fraction",
    "plan_bound_proof_top2_fraction",
    "plan_bound_proof_top5_fraction",
    "plan_bound_terminal_type_top2",
    "plan_bound_terminal_type_top5",
    "plan_relation_coverage_top2",
    "plan_relation_coverage_top5",
    "plan_is_ontology",
    "plan_is_proof_chain",
    "plan_is_top2_semijoin",
    "plan_is_top5_semijoin",
    "plan_is_hybrid",
    "plan_is_bound_proof",
    "plan_is_query_local_top2_pair",
    "plan_is_query_local_top5_completion",
    "plan_is_signature_grounded",
    "plan_is_branch_complete_top2",
    "plan_is_rank_partitioned_top2_ordering",
    "plan_is_rank_partitioned_top5_completion",
    "plan_is_rank_partitioned_nested_topk",
    "plan_is_learned_residual_top5_completion",
    "plan_is_ontology_structural_rank5_completion",
    "plan_is_ontology_warehouse_rank5_completion",
    "plan_is_topology_conditioned_multi_access_top5",
    "plan_is_semantic_fact_scan",
    "plan_is_semantic_top2_noop",
    "plan_is_raw_semantic_graph_scan",
    "plan_is_ontology_physical_complete_slate",
    "plan_is_ontology_structural_execution_slate",
    "plan_is_incumbent_preserving_chain_completion",
    "plan_objective_top2_noop",
    "plan_top2_membership_invariant",
    "plan_top5_membership_invariant",
    "signature_topology_ordered_chain",
    "signature_topology_branch_comparison",
    "plan_signature_bound_top2_fraction",
    "plan_signature_bound_top5_fraction",
    "plan_semantic_access_top2_fraction",
    "plan_semantic_access_top5_fraction",
    "plan_untyped_residual_top5_fraction",
    "plan_semantic_top2_retained_fraction",
    "plan_semantic_top5_retained_fraction",
    "plan_top2_displacement_count",
    "plan_top5_displacement_count",
    "plan_introduced_top2_proof_fraction",
    "plan_introduced_top5_proof_fraction",
    "plan_introduced_complete_proof_fraction",
    "plan_introduced_relation_coverage_mean",
    "plan_introduced_original_rank_mean_log",
    "estimated_cost_log",
    "plan_pair_path_connected",
    "plan_completion_path_connected",
    "plan_promoted_semantic_rank_log",
    "plan_completion_original_rank_log",
    "warehouse_replacement_rank_log",
    "residual_candidate_pool_rank_log",
    "residual_candidate_original_rank_log",
    "residual_candidate_semantic_rank_log",
    "residual_candidate_score",
    "residual_candidate_confidence",
    "residual_candidate_query_coverage",
    "residual_candidate_anchor_coverage",
    "residual_candidate_relation_coverage",
    "residual_candidate_answer_type_score",
    "residual_candidate_has_proof_signal",
    "residual_candidate_complete_provenance",
    "residual_candidate_binding",
    "residual_candidate_typed_slot_binding",
    "residual_candidate_terminal_relation_match",
    "residual_candidate_exact_entity_key",
    "residual_candidate_role_consistent",
    "residual_candidate_access_source_count_log",
    "residual_candidate_path_connected",
    "residual_displaced_query_coverage",
    "residual_displaced_anchor_coverage",
    "residual_displaced_relation_coverage",
    "residual_displaced_answer_type_score",
    "residual_delta_query_coverage",
    "residual_delta_anchor_coverage",
    "residual_delta_relation_coverage",
    "residual_delta_answer_type_score",
    "residual_displaced_score",
    "residual_delta_score",
    "residual_score_ratio",
    "residual_displaced_has_proof_signal",
    "residual_candidate_from_semantic_access",
    "residual_candidate_from_ontology_access",
    "query_candidate_count_log",
    "query_family_candidate_count_log",
    "query_structural_quality_rank_fraction",
    "query_score_delta_rank_fraction",
    "query_safety_rank_fraction",
    "query_plan_novelty_fraction",
    "signature_anchor_count_log",
    "signature_is_comparison",
    "signature_answer_type_person",
    "signature_answer_type_location",
    "signature_answer_type_date",
    "signature_answer_type_number",
    "signature_answer_type_boolean",
    "signature_answer_type_organization",
    "signature_answer_type_other",
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        return default if not math.isfinite(number) else number
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def recover_latent_query_contract(
    query_plan_log: Mapping[str, Any],
    proof_slate: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Recover implicit multihop obligations missed by surface predicates.

    The relation extractor often sees only the terminal predicate in nested
    questions (for example, ``the person who ...``) and records a one-hop
    signature.  The physical optimizer needs the query contract rather than
    that lossy count, so this routine combines the explicit signature, the
    logical compiler, branch constraints, and syntax-level bridge cues.  It
    never consults dataset identity or answer/evidence labels.
    """
    slate = proof_slate or (
        query_plan_log.get("full_aperture_proof_slate", {}) or {}
    )
    signature = slate.get("query_structure_signature", {}) or {}
    logical = (
        (query_plan_log.get("factorized_query_executor", {}) or {}).get(
            "logical_evidence_plan", {}
        )
        or {}
    )
    explicit_slots = max(
        _safe_int(signature.get("relation_slot_count"), 0),
        _safe_int(slate.get("relation_slot_count"), 0),
    )
    logical_hops = max(_safe_int(logical.get("required_hop_count"), 0), 0)
    logical_branches = max(
        _safe_int(logical.get("required_branch_count"), 0), 0
    )
    logical_confidence = _safe_float(logical.get("contract_confidence"), 0.0)
    anchors = signature.get("anchors", []) or slate.get("anchors", []) or []
    anchor_count = max(_safe_int(signature.get("anchor_count"), 0), len(anchors))

    query = str(query_plan_log.get("query", "") or "").strip().lower()
    word_matches = list(re.finditer(r"\b(who|whom|whose|that|which|where|when)\b", query))
    relative_clause_count = sum(
        int(match.start() > 0 and query[:match.start()].strip() != "")
        for match in word_matches
    )


    nested_bridge_cue = bool(
        relative_clause_count > 0
        or re.search(
            r"\b(person|people|man|woman|film|movie|book|work|place|city|country|"
            r"company|organization|team|player|actor|author|director|song|album)\s+"
            r"(who|whom|whose|that|which)\b",
            query,
        )
    )
    comparison_cue = bool(
        signature.get("comparison")
        or str(signature.get("topology", "") or "") == "branch_comparison"
        or str(logical.get("query_shape", "") or "") in {
            "comparison", "branch_comparison", "parallel_comparison"
        }
        or logical_branches >= 2
        or (
            anchor_count >= 2
            and re.search(
                r"\b(compared|earlier|later|first|older|younger|more|less|same|"
                r"both|before|after)\b|\bor\b",
                query,
            )
        )
    )

    effective_slots = explicit_slots
    sources: List[str] = ["explicit_signature"] if explicit_slots else []
    if logical_confidence >= 0.60 and logical_hops > effective_slots:
        effective_slots = logical_hops
        sources.append("logical_compiler")
    if comparison_cue and anchor_count >= 2 and effective_slots < 2:
        effective_slots = 2
        sources.append("comparison_branches")
    if nested_bridge_cue:
        recovered_hops = min(max(explicit_slots, 1) + relative_clause_count, 4)
        if recovered_hops > effective_slots:
            effective_slots = recovered_hops
            sources.append("relative_clause_bridge")

    return {
        "explicit_relation_slot_count": explicit_slots,
        "effective_relation_slot_count": effective_slots,
        "latent_contract_recovered": effective_slots > explicit_slots,
        "logical_required_hop_count": logical_hops,
        "logical_required_branch_count": logical_branches,
        "logical_contract_confidence": round(logical_confidence, 6),
        "anchor_count": anchor_count,
        "relative_clause_count": relative_clause_count,
        "comparison_cue": comparison_cue,
        "nested_bridge_cue": nested_bridge_cue,
        "recovery_sources": list(dict.fromkeys(sources)),
    }


def query_contract_regime(features: Mapping[str, Any]) -> str:
    """Route action estimation by query structure, never by dataset name."""
    if (
        _safe_float(features.get("query_comparison_cue")) > 0.0
        or _safe_float(features.get("signature_topology_branch_comparison")) > 0.0
        or _safe_float(features.get("logical_required_branch_count")) >= 2.0
    ):
        return "comparison_branch"
    if (
        _safe_float(features.get("query_nested_bridge_cue")) > 0.0
        or _safe_float(features.get("query_relative_clause_count")) > 0.0
    ):
        return "nested_bridge"
    relation_slots = _safe_float(features.get("relation_slot_count"), 0.0)
    if relation_slots >= 3.0:
        return "long_chain"
    if relation_slots >= 2.0:
        return "explicit_two_hop"
    return "undercompiled_or_single"


def query_plan_topology(features: Mapping[str, Any]) -> str:
    """Return the compiled query topology used for physical-plan routing.

    Unlike ``query_contract_regime``, this is a deliberately small and strict
    contract.  It routes only signatures that the logical compiler explicitly
    recognized; unresolved queries keep the semantic incumbent.
    """
    if _safe_float(features.get("signature_topology_branch_comparison")) > 0.0:
        return "branch_comparison"
    if _safe_float(features.get("signature_topology_ordered_chain")) > 0.0:
        return "ordered_multihop_chain"
    return "unresolved"


def topology_partitioned_plan_families(
    profile: str = TOPOLOGY_ROUTING_STRICT_V280,
) -> Dict[str, List[str]]:
    """Return a dataset-agnostic physical-plan routing contract.

    The strict profile encodes one plan family per topology. The recall-safe
    profile keeps the
    topology contract, but exposes every rank-compatible access path that can
    be independently certified by the objective-class expert.  Unresolved
    queries remain outside both profiles and therefore retain the incumbent.
    """
    if profile == TOPOLOGY_ROUTING_STRICT_V280:
        return {
            "ordered_multihop_chain": [
                PLAN_SIGNATURE_GROUNDED,
                PLAN_QUERY_LOCAL_TOP2_PAIR,
            ],
            "branch_comparison": [
                PLAN_CHAIN_COMPLETION,
                PLAN_QUERY_LOCAL_TOP5,
            ],
        }
    if profile == TOPOLOGY_ROUTING_RECALL_SAFE_V281:
        return {
            "ordered_multihop_chain": [
                PLAN_SIGNATURE_GROUNDED,
                PLAN_QUERY_LOCAL_TOP2_PAIR,
                PLAN_LEARNED_RESIDUAL_TOP5,
            ],
            "branch_comparison": [
                PLAN_QUERY_LOCAL_TOP2_PAIR,
                PLAN_CHAIN_COMPLETION,
                PLAN_QUERY_LOCAL_TOP5,
                PLAN_LEARNED_RESIDUAL_TOP5,
            ],
        }
    if profile == TOPOLOGY_ROUTING_BOUNDED_V282:




        return {
            "ordered_multihop_chain": [
                PLAN_ONTOLOGY_PHYSICAL,
                PLAN_QUERY_LOCAL_TOP2_PAIR,
            ],
            "branch_comparison": [
                PLAN_ONTOLOGY_PHYSICAL,
                PLAN_LEARNED_RESIDUAL_TOP5,
            ],
        }
    if profile == TOPOLOGY_ROUTING_BOUNDED_V283:




        return {
            "ordered_multihop_chain": [
                PLAN_QUERY_LOCAL_TOP2_PAIR,
                PLAN_ONTOLOGY_STRUCTURAL_TOP5,
            ],
            "branch_comparison": [
                PLAN_QUERY_LOCAL_TOP2_PAIR,
                PLAN_ONTOLOGY_STRUCTURAL_TOP5,
            ],
        }
    if profile == TOPOLOGY_ROUTING_WAREHOUSE_V284:



        return {
            "ordered_multihop_chain": [
                PLAN_QUERY_LOCAL_TOP2_PAIR,
                PLAN_ONTOLOGY_WAREHOUSE_TOP5,
            ],
            "branch_comparison": [
                PLAN_QUERY_LOCAL_TOP2_PAIR,
                PLAN_ONTOLOGY_WAREHOUSE_TOP5,
            ],
        }
    if profile in {
        TOPOLOGY_ROUTING_COMBINED_V285,
        TOPOLOGY_ROUTING_HIERARCHICAL_V286,
        TOPOLOGY_ROUTING_OBJECTIVE_FAMILY_V287,
    }:



        return {
            "ordered_multihop_chain": [
                PLAN_SIGNATURE_GROUNDED,
                PLAN_QUERY_LOCAL_TOP2_PAIR,
                PLAN_LEARNED_RESIDUAL_TOP5,
                PLAN_ONTOLOGY_WAREHOUSE_TOP5,
            ],
            "branch_comparison": [
                PLAN_QUERY_LOCAL_TOP2_PAIR,
                PLAN_CHAIN_COMPLETION,
                PLAN_QUERY_LOCAL_TOP5,
                PLAN_LEARNED_RESIDUAL_TOP5,
                PLAN_ONTOLOGY_WAREHOUSE_TOP5,
            ],
        }
    if profile == TOPOLOGY_ROUTING_MULTI_ACCESS_V289:




        return {
            "ordered_multihop_chain": [
                PLAN_SIGNATURE_GROUNDED,
                PLAN_QUERY_LOCAL_TOP2_PAIR,
                PLAN_TOPOLOGY_MULTI_ACCESS_TOP5,
                PLAN_LEARNED_RESIDUAL_TOP5,
                PLAN_ONTOLOGY_WAREHOUSE_TOP5,
            ],
            "branch_comparison": [
                PLAN_QUERY_LOCAL_TOP2_PAIR,
                PLAN_CHAIN_COMPLETION,
                PLAN_QUERY_LOCAL_TOP5,
                PLAN_TOPOLOGY_MULTI_ACCESS_TOP5,
                PLAN_LEARNED_RESIDUAL_TOP5,
                PLAN_ONTOLOGY_WAREHOUSE_TOP5,
            ],
        }
    if profile == TOPOLOGY_ROUTING_STRUCTURAL_RECOVERY_V293:





        return {
            "ordered_multihop_chain": [
                PLAN_SEMANTIC_TOP2_NOOP,
                PLAN_RAW_SEMANTIC,
                PLAN_ONTOLOGY_PHYSICAL,
                PLAN_STRUCTURAL_EXECUTION,
                PLAN_SIGNATURE_GROUNDED,
                PLAN_QUERY_LOCAL_TOP2_PAIR,
                PLAN_TOPOLOGY_MULTI_ACCESS_TOP5,
                PLAN_LEARNED_RESIDUAL_TOP5,
                PLAN_ONTOLOGY_WAREHOUSE_TOP5,
            ],
            "branch_comparison": [
                PLAN_SEMANTIC_TOP2_NOOP,
                PLAN_RAW_SEMANTIC,
                PLAN_ONTOLOGY_PHYSICAL,
                PLAN_STRUCTURAL_EXECUTION,
                PLAN_QUERY_LOCAL_TOP2_PAIR,
                PLAN_CHAIN_COMPLETION,
                PLAN_QUERY_LOCAL_TOP5,
                PLAN_TOPOLOGY_MULTI_ACCESS_TOP5,
                PLAN_LEARNED_RESIDUAL_TOP5,
                PLAN_ONTOLOGY_WAREHOUSE_TOP5,
            ],
        }
    raise ValueError(f"unknown topology routing profile: {profile}")


def plan_allowed_for_topology(
    plan_name: str,
    features: Mapping[str, Any],
    topology_plan_families: Mapping[str, Sequence[str]],
) -> bool:
    """Apply a model-owned physical-plan routing contract."""
    if not topology_plan_families:
        return True
    allowed = topology_plan_families.get(query_plan_topology(features), ())
    return str(plan_name) in {str(value) for value in allowed}


def _ids(values: Iterable[Any]) -> List[int]:
    result: List[int] = []
    seen = set()
    for value in values:
        passage_idx = _safe_int(value, -1)
        if passage_idx < 0 or passage_idx in seen:
            continue
        seen.add(passage_idx)
        result.append(passage_idx)
    return result


def _piecewise_linear_score(value: float, mapping: Mapping[str, Any]) -> float:
    source = [float(item) for item in mapping.get("source_knots", []) or []]
    target = [float(item) for item in mapping.get("target_knots", []) or []]
    if len(source) < 2 or len(source) != len(target) or not math.isfinite(value):
        return value
    if value <= source[0]:
        return target[0]
    if value >= source[-1]:
        return target[-1]
    right = bisect_right(source, value)
    left = max(0, right - 1)
    right = min(right, len(source) - 1)
    width = source[right] - source[left]
    if width <= 1e-12:
        return target[right]
    fraction = (value - source[left]) / width
    return target[left] + fraction * (target[right] - target[left])


def compose_hierarchical_topk(
    stage1_doc_ids: Sequence[int],
    completion_doc_ids: Sequence[int],
    fallback_doc_ids: Sequence[int],
    *,
    top_k: int = 5,
) -> List[int]:
    """Freeze the Stage-1 Top-2 and fill only the residual Top-K slots.

    A Top-5 completion plan is materialized against the semantic incumbent, so
    its first two documents merely repeat that incumbent prefix.  Hierarchical
    execution discards that stale prefix, keeps the independently selected
    Stage-1 pair, and consumes only the completion plan's residual tail.
    """
    limit = max(2, int(top_k))
    stage1 = _ids(stage1_doc_ids)
    completion = _ids(completion_doc_ids)
    fallback = _ids(fallback_doc_ids)
    return _ids([
        *stage1[:2],
        *completion[2:limit],
        *stage1[2:],
        *fallback,
    ])[:limit]


def _top5_slate_representative_key(
    plan_instance: str,
    plan: Mapping[str, Any],
) -> Tuple[float, ...]:
    """Rank equivalent Top-5 realizations without using evaluation labels."""
    features = plan.get("features", {}) or {}
    plan_family = str(plan.get("plan_family") or plan_instance)
    proof_coherence = sum(_safe_float(features.get(name), 0.0) for name in (
        "plan_completion_path_connected",
        "plan_introduced_complete_proof_fraction",
        "plan_introduced_relation_coverage_mean",
        "residual_candidate_complete_provenance",
        "residual_candidate_terminal_relation_match",
    ))
    query_coverage = sum(_safe_float(features.get(name), 0.0) for name in (
        "residual_candidate_query_coverage",
        "residual_candidate_anchor_coverage",
        "residual_candidate_relation_coverage",
        "residual_candidate_answer_type_score",
    ))
    displacement = (
        2.0 * _safe_float(features.get("plan_top2_displacement_count"), 0.0)
        + _safe_float(features.get("plan_top5_displacement_count"), 0.0)
    )
    return (
        float(plan_family == PLAN_SEMANTIC),
        float(bool(plan.get("valid"))),
        proof_coherence,
        query_coverage,
        -displacement,
        -_safe_float(features.get("estimated_cost_log"), 0.0),
        str(plan_instance),
    )


def deduplicate_top5_plan_instances(
    plans: Mapping[str, Mapping[str, Any]],
    plan_instances: Sequence[str] | None = None,
) -> Tuple[List[str], Dict[str, Any]]:
    """Keep one label-free representative per residual Top-5 completion.

    Physical operators often rediscover the same rank-three-to-five payload
    under different family names. Comparing all aliases gives prolific families
    extra votes and makes training/runtime candidate distributions disagree.
    Top-2 operators remain untouched because their composition is stage dependent.
    """
    instances = list(plan_instances) if plan_instances is not None else list(plans)
    top5_groups: Dict[Tuple[int, ...], List[str]] = {}
    passthrough: set[str] = set()
    for plan_instance in instances:
        plan = plans.get(plan_instance, {}) or {}
        features = plan.get("features", {}) or {}
        if _plan_objective_class(features) != "top5_completion":
            passthrough.add(plan_instance)
            continue




        slate = tuple(_ids(plan.get("doc_ids", []) or [])[2:5])
        if not slate:
            passthrough.add(plan_instance)
            continue
        top5_groups.setdefault(slate, []).append(plan_instance)

    selected = set(passthrough)
    duplicate_groups = 0
    dropped: List[str] = []
    for group in top5_groups.values():
        representative = max(
            group,
            key=lambda value: _top5_slate_representative_key(value, plans[value]),
        )
        selected.add(representative)
        if len(group) > 1:
            duplicate_groups += 1
            dropped.extend(value for value in group if value != representative)
    retained = [value for value in instances if value in selected]
    return retained, {
        "enabled": True,
        "candidate_count_before": len(instances),
        "candidate_count_after": len(retained),
        "duplicate_slate_group_count": duplicate_groups,
        "dropped_alias_count": len(dropped),
        "dropped_plan_instances": dropped,
    }


def objective_family_model_key(objective_class: str, plan_name: str) -> str:
    """Return the serialized key for one rank objective and plan family."""
    return (
        OBJECTIVE_FAMILY_EXPERT_PREFIX
        + str(objective_class)
        + "::"
        + str(plan_name)
    )


def _runtime_regret_score(
    prediction: Mapping[str, Any], objective_class: str
) -> float:
    """Return the regret owned by one hierarchical optimization stage."""
    if objective_class == "top5_completion":
        return _safe_float(prediction.get("regret_r5_ucb"), 1.0)
    return (
        2.0 * _safe_float(prediction.get("regret_r2_ucb"), 1.0)
        + _safe_float(prediction.get("regret_r5_ucb"), 1.0)
    )


def select_proof_connected_stage2_candidate(
    candidate_instances: Sequence[str],
    predictions: Mapping[str, Mapping[str, Any]],
    plans: Mapping[str, Mapping[str, Any]],
    *,
    rescue_advantage: float,
) -> Tuple[str, Dict[str, Any]]:
    """Prefer an executable proof slate and admit an open-path rescue safely.

    Connectivity is a strong precision signal, but making it an absolute gate
    discards useful late evidence that has not yet been joined into a complete
    path.  The primary lane therefore contains the semantic no-op and connected
    slates; an unconnected slate may win only by an OOF-calibrated regret margin.
    """
    instances = [
        value for value in candidate_instances
        if value in predictions and value in plans
    ]
    if not instances:
        return "", {
            "enabled": True,
            "selected_lane": "none",
            "rescue_advantage": 0.0,
            "required_rescue_advantage": float(rescue_advantage),
        }

    def key(value: str) -> Tuple[float, ...]:
        prediction = predictions[value]
        features = plans[value].get("features", {}) or {}
        proof_coherence = sum(_safe_float(features.get(name), 0.0) for name in (
            "plan_completion_path_connected",
            "plan_introduced_complete_proof_fraction",
            "plan_introduced_relation_coverage_mean",
            "residual_candidate_complete_provenance",
            "residual_candidate_terminal_relation_match",
        ))
        displacement = (
            2.0 * _safe_float(
                features.get("plan_top2_displacement_count"), 0.0
            )
            + _safe_float(features.get("plan_top5_displacement_count"), 0.0)
        )
        return (
            -_runtime_regret_score(prediction, "top5_completion"),
            proof_coherence,
            -displacement,
            -_safe_float(features.get("estimated_cost_log"), 0.0),
            float(str(plans[value].get("plan_family") or value) == PLAN_SEMANTIC),
            value,
        )

    primary = []
    rescue = []
    for value in instances:
        family = str(plans[value].get("plan_family") or value)
        connected = bool(_safe_float(
            (plans[value].get("features", {}) or {}).get(
                "plan_completion_path_connected"
            ),
            0.0,
        ))
        if family == PLAN_SEMANTIC or connected:
            primary.append(value)
        else:
            rescue.append(value)

    primary_best = max(primary or instances, key=key)
    rescue_best = max(rescue, key=key) if rescue else ""
    primary_regret = _runtime_regret_score(
        predictions[primary_best], "top5_completion"
    )
    rescue_regret = (
        _runtime_regret_score(predictions[rescue_best], "top5_completion")
        if rescue_best
        else float("inf")
    )
    advantage = primary_regret - rescue_regret
    use_rescue = bool(
        rescue_best
        and advantage + 1e-12 >= max(float(rescue_advantage), 0.0)
    )
    selected = rescue_best if use_rescue else primary_best
    return selected, {
        "enabled": True,
        "selected_lane": "open_path_rescue" if use_rescue else "connected_primary",
        "primary_plan_instance": primary_best,
        "rescue_plan_instance": rescue_best,
        "primary_regret_r5_ucb": primary_regret,
        "rescue_regret_r5_ucb": rescue_regret if rescue_best else None,
        "rescue_advantage": advantage if rescue_best else 0.0,
        "required_rescue_advantage": float(rescue_advantage),
    }


def _overlap(left: Sequence[int], right: Sequence[int], limit: int) -> float:
    left_set = set(left[:limit])
    right_set = set(right[:limit])
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / max(len(left_set), 1)


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def _query_characteristic(features: Mapping[str, float]) -> str:
    strict_paths = _safe_float(
        features.get("strict_executable_path_count_log"), 0.0
    )
    relation_slots = _safe_float(features.get("relation_slot_count"), 0.0)
    if strict_paths <= 0.0:
        return "no_strict_path"
    if relation_slots <= 1.0:
        return "single_relation"
    if relation_slots <= 2.0:
        return "two_relation"
    return "branch_or_long_chain"


def _plan_objective_class(features: Mapping[str, float]) -> str:
    if _safe_float(features.get("plan_objective_top2_noop"), 0.0) >= 0.5:
        return "top2_ordering"
    if _safe_float(features.get("plan_top2_membership_invariant"), 0.0) >= 0.5:
        return "top5_completion"
    return "top2_ordering"


def _candidate_quality(candidate: Mapping[str, Any]) -> Tuple[Any, ...]:
    """Lexicographic proof order; semantic rank is only the final tie-break."""
    return (
        int(bool(candidate.get("bound_proof_path_complete"))),
        int(bool(candidate.get("local_proof_path_complete"))),
        int(bool(candidate.get("complete_provenance"))),
        int(bool(candidate.get("binding"))),
        _safe_float(candidate.get("relation_coverage_score")),
        int(bool(candidate.get("terminal_relation_match"))),
        _safe_float(candidate.get("answer_type_score")),
        _safe_float(candidate.get("anchor_coverage")),
        _safe_float(candidate.get("query_coverage")),
        -max(_safe_int(candidate.get("original_rank"), 10_000), 1),
    )


def _ontology_order(
    proof_slate: Mapping[str, Any],
) -> Tuple[List[int], Dict[int, Mapping[str, Any]]]:
    candidates = [
        value
        for value in proof_slate.get("candidates", []) or []
        if isinstance(value, Mapping)
        and _safe_int(value.get("passage_idx"), -1) >= 0
        and bool(value.get("content_available", True))
        and bool(value.get("has_proof_signal"))
    ]
    by_id = {
        _safe_int(candidate.get("passage_idx"), -1): candidate
        for candidate in candidates
    }




    paths = [
        value
        for value in proof_slate.get("local_proof_paths", []) or []
        if isinstance(value, Mapping) and value.get("passage_ids")
    ]
    paths.sort(
        key=lambda value: (
            int(bool(value.get("query_local_compiler_validated"))),
            _safe_float(value.get("min_relation_score")),
            len(_ids(value.get("passage_ids", []) or [])),
        ),
        reverse=True,
    )
    result: List[int] = []
    if paths:
        result.extend(
            passage_idx
            for passage_idx in _ids(paths[0].get("passage_ids", []) or [])
            if passage_idx in by_id
        )
    ranked = sorted(candidates, key=_candidate_quality, reverse=True)
    result.extend(_safe_int(value.get("passage_idx"), -1) for value in ranked)
    return _ids(result), by_id


def _full_candidate_lookup(
    proof_slate: Mapping[str, Any],
) -> Dict[int, Mapping[str, Any]]:
    """Return every content-bearing candidate, including incomplete proofs."""
    result: Dict[int, Mapping[str, Any]] = {}
    for value in proof_slate.get("candidates", []) or []:
        if not isinstance(value, Mapping):
            continue
        passage_idx = _safe_int(value.get("passage_idx"), -1)
        if passage_idx < 0 or not bool(value.get("content_available", True)):
            continue
        incumbent = result.get(passage_idx)
        if incumbent is None or _candidate_quality(value) > _candidate_quality(
            incumbent
        ):
            result[passage_idx] = value
    return result


def _residual_top5_candidate_pool(
    proof_slate: Mapping[str, Any],
    semantic_ids: Sequence[int],
    connected_external_ids: Sequence[int],
    *,
    max_candidates: int = 32,
) -> Tuple[List[int], Dict[int, Mapping[str, Any]]]:
    """Enumerate a lossless late-materialization reservoir for rank five.

    The semantic prefix supplies high-recall candidates while the structural
    order adds candidates that satisfy ontology slots but were retrieved at a
    deeper semantic rank.  Candidate selection is learned later; this helper
    only bounds the physical-plan search space and never consumes gold labels.
    """
    candidate_by_id = _full_candidate_lookup(proof_slate)
    incumbent_top5 = set(semantic_ids[:5])
    semantic_residual = [
        passage_idx for passage_idx in semantic_ids[5:29]
        if passage_idx not in incumbent_top5
    ]
    structural_residual = sorted(
        (
            passage_idx for passage_idx in candidate_by_id
            if passage_idx not in incumbent_top5
        ),
        key=lambda passage_idx: _candidate_quality(candidate_by_id[passage_idx]),
        reverse=True,
    )[:16]
    connected_residual = [
        passage_idx for passage_idx in connected_external_ids[:8]
        if passage_idx not in incumbent_top5
    ]
    return (
        _complete_order(
            semantic_residual,
            structural_residual,
            connected_residual,
        )[: max(1, int(max_candidates))],
        candidate_by_id,
    )


def _ontology_warehouse_top5_candidate_pool(
    proof_slate: Mapping[str, Any],
    semantic_ids: Sequence[int],
    *,
    max_candidates: int = 16,
) -> Tuple[List[int], Dict[int, Mapping[str, Any]]]:
    """Build a diverse ontology-backed rank-five reservoir.

    The earlier high-recall pipeline succeeded because it retained deep
    warehouse tuples selected by complementary text, relation, and terminal
    signals.  A semantic-rank-first pool can crowd all of those tuples out.
    We therefore round-robin three deterministic index orders and leave the
    actual utility/risk decision to the learned complete-slate CBO.  No
    dataset identity, hand-weighted utility, or runtime label is used here.
    """
    candidate_by_id = _full_candidate_lookup(proof_slate)
    incumbent_top5 = set(semantic_ids[:5])
    candidates = [
        value for passage_idx, value in candidate_by_id.items()
        if passage_idx not in incumbent_top5
        and (
            bool(value.get("has_proof_signal"))
            or _safe_int(value.get("physical_access_rank"), 0) > 0
            or "ontology_physical_operator" in {
                str(source) for source in value.get("access_sources", []) or []
            }
        )
    ]

    proof_order = sorted(candidates, key=_candidate_quality, reverse=True)
    semantic_order = sorted(
        candidates,
        key=lambda value: (
            _safe_float(value.get("query_coverage")),
            _safe_float(value.get("anchor_coverage")),
            _safe_float(value.get("score")),
            _safe_float(value.get("confidence")),
            -max(_safe_int(value.get("original_rank"), 10_000), 1),
        ),
        reverse=True,
    )
    terminal_order = sorted(
        candidates,
        key=lambda value: (
            int(bool(value.get("terminal_relation_match"))),
            _safe_float(value.get("answer_type_score")),
            _safe_float(value.get("relation_coverage_score")),
            int(bool(value.get("typed_slot_binding"))),
            int(bool(value.get("binding"))),
            _safe_float(value.get("score")),
        ),
        reverse=True,
    )

    result: List[int] = []
    seen = set()
    for offset in range(max(map(len, (proof_order, semantic_order, terminal_order)), default=0)):
        for order in (proof_order, semantic_order, terminal_order):
            if offset >= len(order):
                continue
            passage_idx = _safe_int(order[offset].get("passage_idx"), -1)
            if passage_idx < 0 or passage_idx in seen:
                continue
            seen.add(passage_idx)
            result.append(passage_idx)
            if len(result) >= max(1, int(max_candidates)):
                return result, candidate_by_id
    return result, candidate_by_id


def _topology_conditioned_multi_access_slates(
    topology: str,
    semantic_ids: Sequence[int],
    strongest_query_local_path_ids: Sequence[int],
    strongest_branch_complete_path_ids: Sequence[int],
    connected_external_ids: Sequence[int],
    warehouse_candidate_ids: Sequence[int],
    candidate_by_id: Mapping[int, Mapping[str, Any]],
) -> List[Tuple[str, List[int], Dict[str, Any]]]:
    """Compile a few complete Top-5 alternatives from complementary indexes.

    Earlier portfolios exposed one tuple swap at a time. That is unable to
    represent a multi-hop proof whose useful passages must be committed as a
    set. This compiler keeps the incumbent Top-2 immutable and materializes
    three dataset-agnostic set alternatives: provenance closure, terminal-slot
    projection, and multi-index consensus. Gold evidence is never consulted.
    """
    semantic_top2 = set(semantic_ids[:2])
    semantic_top5 = set(semantic_ids[:5])

    def external(values: Sequence[int]) -> List[int]:
        return [value for value in _ids(values) if value not in semantic_top2]

    proof_order = sorted(
        (
            passage_idx for passage_idx in candidate_by_id
            if passage_idx not in semantic_top2
        ),
        key=lambda passage_idx: _candidate_quality(candidate_by_id[passage_idx]),
        reverse=True,
    )
    terminal_order = sorted(
        external(warehouse_candidate_ids),
        key=lambda passage_idx: (
            int(bool(candidate_by_id.get(passage_idx, {}).get(
                "terminal_relation_match"
            ))),
            _safe_float(candidate_by_id.get(passage_idx, {}).get(
                "answer_type_score"
            )),
            int(bool(candidate_by_id.get(passage_idx, {}).get(
                "typed_slot_binding"
            ))),
            _safe_float(candidate_by_id.get(passage_idx, {}).get(
                "relation_coverage_score"
            )),
            _candidate_quality(candidate_by_id.get(passage_idx, {})),
        ),
        reverse=True,
    )
    consensus_order = sorted(
        external(warehouse_candidate_ids),
        key=lambda passage_idx: (
            len(candidate_by_id.get(passage_idx, {}).get("access_sources", []) or []),
            int(bool(candidate_by_id.get(passage_idx, {}).get("exact_entity_key"))),
            int(bool(candidate_by_id.get(passage_idx, {}).get("role_consistent"))),
            int(bool(candidate_by_id.get(passage_idx, {}).get("complete_provenance"))),
            _candidate_quality(candidate_by_id.get(passage_idx, {})),
        ),
        reverse=True,
    )
    topology_path = (
        strongest_branch_complete_path_ids
        if topology == "branch_comparison" and strongest_branch_complete_path_ids
        else strongest_query_local_path_ids
    )
    variants = (
        (
            "provenance_closure",
            _complete_order(
                semantic_ids[:2],
                external(topology_path),
                external(connected_external_ids),
                proof_order,
                semantic_ids[2:],
            ),
        ),
        (
            "terminal_projection",
            _complete_order(
                semantic_ids[:2],
                terminal_order,
                external(topology_path),
                semantic_ids[2:],
            ),
        ),
        (
            "multi_index_consensus",
            _complete_order(
                semantic_ids[:2],
                consensus_order,
                external(connected_external_ids),
                semantic_ids[2:],
            ),
        ),
    )
    result: List[Tuple[str, List[int], Dict[str, Any]]] = []
    seen_top5 = set()
    for variant, doc_ids in variants:
        top5 = tuple(doc_ids[:5])
        introduced = set(top5) - semantic_top5
        if len(top5) < 5 or not introduced or top5 in seen_top5:
            continue
        seen_top5.add(top5)
        result.append((
            variant,
            doc_ids,
            {
                "multi_access_variant": variant,
                "multi_access_external_count": len(introduced),
                "completion_path_connected": bool(
                    introduced & set(connected_external_ids)
                ),
                "access_path": (
                    "topology_conditioned_semantic_entity_graph_warehouse_join"
                ),
                "complete_slate_transaction": True,
            },
        ))
    return result


def _rank_merge(semantic_ids: Sequence[int], ontology_ids: Sequence[int]) -> List[int]:
    """Equal-weight reciprocal-rank merge with deterministic tie-breaking."""
    semantic_rank = {value: rank for rank, value in enumerate(semantic_ids, start=1)}
    ontology_rank = {value: rank for rank, value in enumerate(ontology_ids, start=1)}
    union = _ids([*semantic_ids, *ontology_ids])
    return sorted(
        union,
        key=lambda value: (
            1.0 / (1.0 + semantic_rank[value]) if value in semantic_rank else 0.0
        ) + (
            1.0 / (1.0 + ontology_rank[value]) if value in ontology_rank else 0.0
        ),
        reverse=True,
    )


def _complete_order(*parts: Sequence[int]) -> List[int]:
    return _ids(value for part in parts for value in part)


def _descending_rank_fraction(
    items: Sequence[Tuple[str, Tuple[Any, ...]]],
) -> Dict[str, float]:
    """Return a stable query-local rank in [0, 1], where one is best."""
    ordered = sorted(items, key=lambda item: (item[1], item[0]), reverse=True)
    denominator = max(len(ordered) - 1, 1)
    return {
        plan_key: 1.0 - (rank / denominator)
        for rank, (plan_key, _) in enumerate(ordered)
    }


def _annotate_query_relative_features(
    plans: Dict[str, Dict[str, Any]], *, include_semantic: bool = False
) -> None:
    """Add within-query comparisons shared by replay and online inference."""
    valid = [
        (plan_key, plan)
        for plan_key, plan in plans.items()
        if (include_semantic or plan_key != PLAN_SEMANTIC)
        and bool(plan.get("valid"))
    ]
    if not valid:
        return
    family_counts: Dict[str, int] = {}
    for _, plan in valid:
        family = str(plan.get("plan_family", ""))
        family_counts[family] = family_counts.get(family, 0) + 1

    structural_items: List[Tuple[str, Tuple[Any, ...]]] = []
    score_items: List[Tuple[str, Tuple[Any, ...]]] = []
    safety_items: List[Tuple[str, Tuple[Any, ...]]] = []
    for plan_key, plan in valid:
        features = plan.get("features", {}) or {}
        structural_items.append((plan_key, (
            _safe_float(features.get("plan_completion_path_connected")),
            _safe_float(features.get("residual_candidate_complete_provenance")),
            _safe_float(features.get("residual_candidate_terminal_relation_match")),
            _safe_float(features.get("plan_introduced_complete_proof_fraction")),
            _safe_float(features.get("plan_introduced_relation_coverage_mean")),
            _safe_float(features.get("residual_delta_relation_coverage")),
            _safe_float(features.get("residual_delta_answer_type_score")),
        )))
        score_items.append((plan_key, (
            _safe_float(features.get("residual_delta_score")),
            _safe_float(features.get("residual_score_ratio")),
            _safe_float(features.get("residual_candidate_score")),
            -_safe_float(features.get("residual_candidate_pool_rank_log")),
        )))
        safety_items.append((plan_key, (
            _safe_float(features.get("plan_semantic_top2_retained_fraction")),
            _safe_float(features.get("plan_semantic_top5_retained_fraction")),
            _safe_float(features.get("plan_introduced_complete_proof_fraction")),
            -_safe_float(features.get("plan_top5_displacement_count")),
            _safe_float(features.get("residual_delta_score")),
        )))

    structural_rank = _descending_rank_fraction(structural_items)
    score_rank = _descending_rank_fraction(score_items)
    safety_rank = _descending_rank_fraction(safety_items)
    for plan_key, plan in valid:
        features = plan.get("features", {}) or {}
        family = str(plan.get("plan_family", ""))
        features.update({
            "query_candidate_count_log": math.log1p(len(valid)),
            "query_family_candidate_count_log": math.log1p(
                family_counts.get(family, 0)
            ),
            "query_structural_quality_rank_fraction": structural_rank[plan_key],
            "query_score_delta_rank_fraction": score_rank[plan_key],
            "query_safety_rank_fraction": safety_rank[plan_key],
            "query_plan_novelty_fraction": 1.0 - _safe_float(
                features.get("plan_semantic_top5_overlap")
            ),
        })


def _shortlist_query_grouped_plans(
    plans: Dict[str, Dict[str, Any]], limit_per_family: int
) -> Dict[str, Any]:
    """Bound large swap families using complementary, label-free orderings."""
    target_families = {
        PLAN_LEARNED_RESIDUAL_TOP5,
        PLAN_ONTOLOGY_WAREHOUSE_TOP5,
    }
    diagnostics: Dict[str, Any] = {
        "enabled": limit_per_family > 0,
        "limit_per_family": max(limit_per_family, 0),
        "families": {},
    }
    if limit_per_family <= 0:
        return diagnostics
    grouped: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
    for plan_key, plan in plans.items():
        family = str(plan.get("plan_family", ""))
        if family in target_families and bool(plan.get("valid")):
            grouped.setdefault(family, []).append((plan_key, plan))

    keep = set(plans)
    for family, family_plans in grouped.items():
        if len(family_plans) <= limit_per_family:
            diagnostics["families"][family] = {
                "before": len(family_plans),
                "after": len(family_plans),
            }
            continue

        def feature(plan: Dict[str, Any], name: str) -> float:
            return _safe_float((plan.get("features", {}) or {}).get(name))

        selectors = (
            lambda item: (
                feature(item[1], "query_structural_quality_rank_fraction"),
                feature(item[1], "query_safety_rank_fraction"),
            ),
            lambda item: (
                feature(item[1], "query_score_delta_rank_fraction"),
                feature(item[1], "residual_delta_score"),
            ),
            lambda item: (
                feature(item[1], "residual_delta_relation_coverage"),
                feature(item[1], "residual_delta_query_coverage"),
                feature(item[1], "residual_delta_anchor_coverage"),
            ),
            lambda item: (
                feature(item[1], "residual_candidate_terminal_relation_match"),
                feature(item[1], "residual_delta_answer_type_score"),
                feature(item[1], "residual_candidate_typed_slot_binding"),
            ),
            lambda item: (
                feature(item[1], "residual_candidate_access_source_count_log"),
                feature(item[1], "residual_candidate_exact_entity_key"),
                feature(item[1], "residual_candidate_role_consistent"),
            ),
        )
        selected: List[str] = []
        recall_reserved: List[str] = []

        def add(plan_key: str) -> None:
            if plan_key not in selected and len(selected) < limit_per_family:
                selected.append(plan_key)

        for selector in selectors:
            add(max(family_plans, key=selector)[0])





        recall_reserve = max(2, limit_per_family // 3)
        shallow_candidates = sorted(
            family_plans,
            key=lambda item: (
                _safe_int(
                    (item[1].get("plan_metadata", {}) or {}).get(
                        "residual_candidate_original_rank"
                    ),
                    10_000,
                ),
                -feature(item[1], "query_structural_quality_rank_fraction"),
                item[0],
            ),
        )
        for plan_key, plan in shallow_candidates:
            original_rank = _safe_int(
                (plan.get("plan_metadata", {}) or {}).get(
                    "residual_candidate_original_rank"
                ),
                0,
            )
            if original_rank <= 0:
                continue
            before = len(selected)
            add(plan_key)
            if len(selected) > before:
                recall_reserved.append(plan_key)
            if len(recall_reserved) >= recall_reserve:
                break

        replacement_groups: Dict[int, List[Tuple[str, Dict[str, Any]]]] = {}
        for item in family_plans:
            replacement_rank = _safe_int(
                (item[1].get("plan_metadata", {}) or {}).get(
                    "warehouse_replacement_rank"
                ),
                0,
            )
            if replacement_rank:
                replacement_groups.setdefault(replacement_rank, []).append(item)
        for replacement_rank in sorted(replacement_groups):
            add(max(
                replacement_groups[replacement_rank],
                key=lambda item: (
                    feature(item[1], "query_structural_quality_rank_fraction"),
                    feature(item[1], "query_score_delta_rank_fraction"),
                ),
            )[0])

        ranked = sorted(
            family_plans,
            key=lambda item: (
                feature(item[1], "query_structural_quality_rank_fraction")
                + feature(item[1], "query_score_delta_rank_fraction")
                + feature(item[1], "query_safety_rank_fraction"),
                feature(item[1], "residual_delta_relation_coverage"),
                feature(item[1], "residual_delta_score"),
            ),
            reverse=True,
        )
        for plan_key, _ in ranked:
            add(plan_key)
        selected_set = set(selected)
        for plan_key, _ in family_plans:
            if plan_key not in selected_set:
                keep.discard(plan_key)
        diagnostics["families"][family] = {
            "before": len(family_plans),
            "after": len(selected),
            "selected_plan_instances": selected,
            "recall_reserved_plan_instances": recall_reserved,
        }

    for plan_key in list(plans):
        if plan_key not in keep:
            del plans[plan_key]
    diagnostics["candidate_count_after"] = len(plans)
    return diagnostics


def _proof_ranked_incumbent_top5(
    semantic_ids: Sequence[int],
    candidate_by_id: Mapping[int, Mapping[str, Any]],
) -> List[int]:
    """Rank proof-bearing incumbent members without changing Top-5 membership."""
    incumbent_top5 = list(semantic_ids[:5])
    proof_members = [
        passage_idx
        for passage_idx in incumbent_top5
        if passage_idx in candidate_by_id
        and bool(candidate_by_id[passage_idx].get("has_proof_signal"))
    ]
    proof_members.sort(
        key=lambda passage_idx: _candidate_quality(candidate_by_id[passage_idx]),
        reverse=True,
    )
    return _complete_order(proof_members, incumbent_top5)


def _connected_external_candidates(
    proof_slate: Mapping[str, Any],
    semantic_ids: Sequence[int],
    candidate_by_id: Mapping[int, Mapping[str, Any]],
) -> List[int]:
    """Return proof candidates joined to at least one incumbent Top-5 tuple.

    This is deliberately stricter than taking the globally highest ontology
    candidate.  A candidate may enter the rank-partitioned Top-5 plan only if
    a materialized local/bound proof path connects it to evidence already in
    the incumbent Top-5.  The plan therefore represents an index nested-loop
    completion over an incumbent binding, rather than an unrelated scan hit.
    """
    incumbent_top5 = set(semantic_ids[:5])
    paths: List[Tuple[Tuple[Any, ...], List[int]]] = []
    for source in ("bound_proof_paths", "local_proof_paths"):
        for path in proof_slate.get(source, []) or []:
            if not isinstance(path, Mapping):
                continue
            passage_ids = [
                passage_idx
                for passage_idx in _ids(path.get("passage_ids", []) or [])
                if passage_idx in candidate_by_id
            ]
            if not passage_ids or not (set(passage_ids) & incumbent_top5):
                continue
            external = [
                passage_idx
                for passage_idx in passage_ids
                if passage_idx not in incumbent_top5
            ]
            if not external:
                continue
            path_key = (
                int(bool(path.get("query_local_compiler_validated"))),
                int(bool(path.get("strict_chain_certificate"))),
                int(bool(path.get("anchor_endpoint_bound"))),
                int(bool(path.get("binding_consistent"))),
                _safe_float(path.get("proof_relation_coverage")),
                _safe_float(path.get("proof_provenance_coverage")),
                _safe_float(path.get("min_relation_score")),
            )
            paths.append((path_key, external))
    paths.sort(key=lambda item: item[0], reverse=True)
    connected = _ids(
        passage_idx
        for _, passage_ids in paths
        for passage_idx in passage_ids
    )
    return sorted(
        connected,
        key=lambda passage_idx: _candidate_quality(candidate_by_id[passage_idx]),
        reverse=True,
    )


def _incumbent_top2_pair_candidates(
    proof_slate: Mapping[str, Any],
    semantic_ids: Sequence[int],
    candidate_by_id: Mapping[int, Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Enumerate proof-bearing Top-2 promotions and their join contracts."""
    semantic_top2 = set(semantic_ids[:2])
    promotable = set(semantic_ids[2:5])
    semantic_rank = {
        passage_idx: rank for rank, passage_idx in enumerate(semantic_ids, start=1)
    }
    connected_by_partner: Dict[int, Tuple[Tuple[Any, ...], int]] = {}
    for source in ("bound_proof_paths", "local_proof_paths"):
        for path in proof_slate.get(source, []) or []:
            if not isinstance(path, Mapping):
                continue
            passage_ids = [
                passage_idx
                for passage_idx in _ids(path.get("passage_ids", []) or [])
                if passage_idx in candidate_by_id
            ]
            anchors = [
                passage_idx for passage_idx in passage_ids
                if passage_idx in semantic_top2
            ]
            partners = [
                passage_idx for passage_idx in passage_ids
                if passage_idx in promotable
            ]
            if not anchors or not partners:
                continue
            anchor = min(anchors, key=lambda value: semantic_rank[value])
            partner = max(
                partners,
                key=lambda value: _candidate_quality(candidate_by_id[value]),
            )
            path_key = (
                int(bool(path.get("query_local_compiler_validated"))),
                int(bool(path.get("strict_chain_certificate"))),
                int(bool(path.get("anchor_endpoint_bound"))),
                int(bool(path.get("binding_consistent"))),
                _safe_float(path.get("proof_relation_coverage")),
                _safe_float(path.get("proof_provenance_coverage")),
                _safe_float(path.get("min_relation_score")),
                *_candidate_quality(candidate_by_id[partner]),
            )
            previous = connected_by_partner.get(partner)
            if previous is None or path_key > previous[0]:
                connected_by_partner[partner] = (path_key, anchor)

    result = []
    default_anchor = semantic_ids[0] if semantic_ids else None
    for partner in semantic_ids[2:5]:
        candidate = candidate_by_id.get(partner, {}) or {}
        if not candidate or not bool(candidate.get("has_proof_signal")):
            continue
        connected = partner in connected_by_partner
        anchor = (
            connected_by_partner[partner][1]
            if connected
            else default_anchor
        )
        if anchor is None or anchor == partner:
            continue
        result.append({
            "pair": [anchor, partner],
            "partner": partner,
            "pair_path_connected": connected,
            "promoted_semantic_rank": semantic_rank.get(partner, 0),
            "quality": (
                int(connected),
                *_candidate_quality(candidate),
            ),
        })
    result.sort(key=lambda value: value["quality"], reverse=True)
    return result


def build_unified_access_window(
    semantic_ids: Sequence[int],
    semantic_scores: Sequence[float],
    physical_ids: Sequence[int],
    physical_scores: Sequence[float],
    *,
    limit: int = 200,
) -> Dict[str, Any]:
    """Build one scan aperture from semantic/fact and ontology operators.

    The semantic result is an access method inside the evidence plan, not a
    competing final ranking.  We retain the head of both scans, then
    round-robin their tails so one source cannot consume the entire aperture.
    """
    semantic = _ids(semantic_ids)
    physical = _ids(physical_ids)
    limit = max(int(limit), 5)
    semantic_score_by_id = {
        passage_idx: _safe_float(score)
        for passage_idx, score in zip(semantic, semantic_scores)
    }
    physical_score_by_id = {
        passage_idx: _safe_float(score)
        for passage_idx, score in zip(physical, physical_scores)
    }
    semantic_rank = {value: rank for rank, value in enumerate(semantic, start=1)}
    physical_rank = {value: rank for rank, value in enumerate(physical, start=1)}

    ordered: List[int] = []
    seen = set()

    def append(values: Sequence[int]) -> None:
        for passage_idx in values:
            if passage_idx < 0 or passage_idx in seen or len(ordered) >= limit:
                continue
            seen.add(passage_idx)
            ordered.append(passage_idx)



    append(semantic[:5])
    append(physical[:5])
    for offset in range(5, max(len(semantic), len(physical))):
        if len(ordered) >= limit:
            break
        if offset < len(semantic):
            append([semantic[offset]])
        if offset < len(physical):
            append([physical[offset]])

    provenance = []
    scores = []
    for passage_idx in ordered:
        sources = []
        if passage_idx in semantic_rank:
            sources.append("semantic_fact_scan")
        if passage_idx in physical_rank:
            sources.append("ontology_physical_operator")
        semantic_score = semantic_score_by_id.get(passage_idx)
        physical_score = physical_score_by_id.get(passage_idx)
        source_scores = [
            value for value in (semantic_score, physical_score)
            if value is not None
        ]
        scores.append(max(source_scores, default=0.0))
        provenance.append({
            "passage_idx": passage_idx,
            "access_sources": sources,
            "semantic_rank": semantic_rank.get(passage_idx),
            "physical_rank": physical_rank.get(passage_idx),
        })
    return {
        "schema_version": "unified_evidence_access_window_v1",
        "doc_ids": ordered,
        "doc_scores": scores,
        "semantic_count": len(semantic),
        "physical_count": len(physical),
        "window_count": len(ordered),
        "provenance": provenance,
    }


def _strongest_local_path(
    proof_slate: Mapping[str, Any],
    candidate_by_id: Mapping[int, Mapping[str, Any]],
) -> List[int]:
    paths = [
        value
        for value in proof_slate.get("local_proof_paths", []) or []
        if isinstance(value, Mapping) and value.get("passage_ids")
    ]
    paths.sort(
        key=lambda value: (
            int(bool(value.get("query_local_compiler_validated"))),
            _safe_float(value.get("min_relation_score")),
            len(_ids(value.get("passage_ids", []) or [])),
        ),
        reverse=True,
    )
    if not paths:
        return []
    return [
        passage_idx
        for passage_idx in _ids(paths[0].get("passage_ids", []) or [])
        if passage_idx in candidate_by_id
    ]


def _strongest_bound_path(
    proof_slate: Mapping[str, Any],
    candidate_by_id: Mapping[int, Mapping[str, Any]],
) -> List[int]:
    paths = [
        value
        for value in proof_slate.get("bound_proof_paths", []) or []
        if isinstance(value, Mapping)
        and value.get("passage_ids")
        and bool(value.get("anchor_endpoint_bound"))
        and bool(value.get("binding_consistent"))
    ]
    paths.sort(
        key=lambda value: (
            _safe_float(value.get("min_relation_score")),
            len(_ids(value.get("passage_ids", []) or [])),
        ),
        reverse=True,
    )
    if not paths:
        return []
    return [
        passage_idx
        for passage_idx in _ids(paths[0].get("passage_ids", []) or [])
        if passage_idx in candidate_by_id
    ]


def _strongest_query_local_path(
    proof_slate: Mapping[str, Any],
    candidate_by_id: Mapping[int, Mapping[str, Any]],
) -> List[int]:
    paths = [
        value
        for value in proof_slate.get("bound_proof_paths", []) or []
        if isinstance(value, Mapping)
        and value.get("passage_ids")
        and bool(value.get("query_local_compiler_validated"))
        and bool(value.get("strict_chain_certificate"))
        and bool(value.get("anchor_endpoint_bound"))
        and bool(value.get("binding_consistent"))
    ]
    paths.sort(
        key=lambda value: (
            _safe_float(value.get("proof_relation_coverage")),
            _safe_float(value.get("proof_provenance_coverage")),
            _safe_float(value.get("min_relation_score")),
            len(_ids(value.get("passage_ids", []) or [])),
        ),
        reverse=True,
    )
    if not paths:
        return []
    return [
        passage_idx
        for passage_idx in _ids(paths[0].get("passage_ids", []) or [])
        if passage_idx in candidate_by_id
    ]


def _strongest_branch_complete_path(
    proof_slate: Mapping[str, Any],
    candidate_by_id: Mapping[int, Mapping[str, Any]],
) -> List[int]:
    """Return a certified two-branch proof that fits atomically in Top-2.

    Comparison questions may have one relation slot applied independently to
    two anchors.  Treating ``relation_slot_count == 1`` as single-hop drops the
    exact two-passage proof even though the physical join has completed both
    branches.  The compiler already emits one joined path with a strict
    certificate; this operator exposes that path as a first-class Top-2 plan.
    """
    signature = proof_slate.get("query_structure_signature", {}) or {}
    if (
        str(signature.get("topology", "") or "") != "branch_comparison"
        or _safe_int(signature.get("anchor_count"), 0) < 2
    ):
        return []
    paths = []
    for value in proof_slate.get("bound_proof_paths", []) or []:
        if not isinstance(value, Mapping):
            continue
        passage_ids = _ids(value.get("passage_ids", []) or [])
        certificate = value.get("strict_chain_certificate_details", {}) or {}
        expected_steps = _safe_int(
            certificate.get("expected_step_count"), len(passage_ids)
        )
        if (
            len(passage_ids) != 2
            or expected_steps < 2
            or not bool(value.get("query_local_compiler_validated"))
            or not bool(value.get("strict_chain_certificate"))
            or not bool(value.get("anchor_endpoint_bound"))
            or not bool(value.get("binding_consistent"))
            or not bool(value.get("exact_anchor"))
            or not all(passage_idx in candidate_by_id for passage_idx in passage_ids)
        ):
            continue
        paths.append((value, passage_ids))
    paths.sort(
        key=lambda item: (
            _safe_float(item[0].get("proof_relation_coverage")),
            _safe_float(item[0].get("proof_provenance_coverage")),
            _safe_float(item[0].get("min_relation_score")),
        ),
        reverse=True,
    )
    return list(paths[0][1]) if paths else []


def _monotone_scores(
    selected_ids: Sequence[int],
    score_sources: Sequence[Tuple[Sequence[int], Sequence[float]]],
) -> List[float]:
    source_scores: Dict[int, float] = {}
    for ids, scores in score_sources:
        for passage_idx, score in zip(ids, scores):
            source_scores.setdefault(int(passage_idx), _safe_float(score))
    observed = list(source_scores.values())
    ceiling = max(observed, default=1.0)
    floor = min(observed, default=0.0)
    step = max(abs(ceiling - floor), 1.0) / max(len(selected_ids), 1)
    return [ceiling - rank * step for rank in range(len(selected_ids))]


def _plan_features(
    query_plan_log: Mapping[str, Any],
    proof_slate: Mapping[str, Any],
    semantic_ids: Sequence[int],
    semantic_scores: Sequence[float],
    ontology_ids: Sequence[int],
    plan_name: str,
    plan_ids: Sequence[int],
    candidate_by_id: Mapping[int, Mapping[str, Any]],
    plan_metadata: Mapping[str, Any] | None = None,
) -> Dict[str, float]:
    plan_metadata = plan_metadata or {}
    query_signature = dict(
        proof_slate.get("query_structure_signature", {}) or {}
    )
    latent_contract = recover_latent_query_contract(
        query_plan_log, proof_slate
    )
    signature_answer_type = str(
        query_signature.get("answer_type", "") or ""
    ).strip().lower()
    candidates = list(candidate_by_id.values())
    relation_coverages = [
        _safe_float(value.get("relation_coverage_score")) for value in candidates
    ]
    signal_count = len(candidates)
    all_candidate_count = max(
        _safe_int(proof_slate.get("candidate_count"), signal_count), 1
    )
    answer_compatible = sum(
        int(bool(value.get("answer_type_compatible"))) for value in candidates
    )
    complete_ids = {
        passage_idx
        for passage_idx, value in candidate_by_id.items()
        if bool(value.get("complete_provenance"))
        or bool(value.get("local_proof_path_complete"))
    }
    bound_ids = {
        passage_idx
        for passage_idx, value in candidate_by_id.items()
        if bool(value.get("bound_proof_path_complete"))
    }
    signature_bound_ids = {
        passage_idx
        for passage_idx, value in candidate_by_id.items()
        if bool(value.get("query_local_compiler_validated"))
        and bool(value.get("bound_proof_path_complete"))
    }
    semantic_set = set(semantic_ids)
    semantic_top2 = set(semantic_ids[:2])
    semantic_top5 = set(semantic_ids[:5])
    introduced_top2 = [value for value in plan_ids[:2] if value not in semantic_top2]
    introduced_top5 = [value for value in plan_ids[:5] if value not in semantic_top5]
    introduced_candidates = [
        candidate_by_id[value]
        for value in introduced_top5
        if value in candidate_by_id
    ]

    def plan_mean(key: str, limit: int) -> float:
        return _mean(
            _safe_float(candidate_by_id.get(value, {}).get(key))
            for value in plan_ids[:limit]
        )

    score_margin_top2 = (
        _safe_float(semantic_scores[0]) - _safe_float(semantic_scores[1])
        if len(semantic_scores) >= 2 else 0.0
    )
    score_margin_top5 = (
        _safe_float(semantic_scores[0]) - _safe_float(semantic_scores[4])
        if len(semantic_scores) >= 5 else 0.0
    )
    estimated_cost = (
        signal_count
        + 2 * _safe_int(proof_slate.get("local_proof_path_count"), 0)
        + 3 * _safe_int(proof_slate.get("bound_proof_path_count"), 0)
        + _safe_int(latent_contract.get("effective_relation_slot_count"), 0)
    )
    features = {
        "matched_plan_count_log": math.log1p(
            _safe_int(query_plan_log.get("matched_plan_count"), 0)
        ),
        "strict_executable_path_count_log": math.log1p(
            _safe_int(query_plan_log.get("strict_executable_path_count"), 0)
        ),
        "executable_path_count_log": math.log1p(
            _safe_int(query_plan_log.get("executable_path_count"), 0)
        ),
        "relation_slot_count": float(
            _safe_int(latent_contract.get("effective_relation_slot_count"), 0)
        ),
        "explicit_relation_slot_count": float(
            _safe_int(latent_contract.get("explicit_relation_slot_count"), 0)
        ),
        "latent_contract_recovered": float(bool(
            latent_contract.get("latent_contract_recovered")
        )),
        "logical_required_hop_count": float(
            _safe_int(latent_contract.get("logical_required_hop_count"), 0)
        ),
        "logical_required_branch_count": float(
            _safe_int(latent_contract.get("logical_required_branch_count"), 0)
        ),
        "query_relative_clause_count": float(
            _safe_int(latent_contract.get("relative_clause_count"), 0)
        ),
        "query_comparison_cue": float(bool(
            latent_contract.get("comparison_cue")
        )),
        "query_nested_bridge_cue": float(bool(
            latent_contract.get("nested_bridge_cue")
        )),
        "anchor_count_log": math.log1p(
            len(proof_slate.get("anchors", []) or [])
        ),
        "proof_signal_candidate_count_log": math.log1p(signal_count),
        "complete_proof_candidate_count_log": math.log1p(len(complete_ids)),
        "local_proof_path_count_log": math.log1p(
            _safe_int(proof_slate.get("local_proof_path_count"), 0)
        ),
        "local_proof_path_passage_count_log": math.log1p(
            _safe_int(proof_slate.get("local_proof_path_passage_count"), 0)
        ),
        "bound_proof_path_count_log": math.log1p(
            _safe_int(proof_slate.get("bound_proof_path_count"), 0)
        ),
        "bound_proof_path_passage_count_log": math.log1p(
            _safe_int(proof_slate.get("bound_proof_path_passage_count"), 0)
        ),
        "proof_signal_fraction": signal_count / all_candidate_count,
        "answer_type_compatible_fraction": answer_compatible / max(signal_count, 1),
        "mean_relation_coverage": _mean(relation_coverages),
        "max_relation_coverage": max(relation_coverages, default=0.0),
        "max_anchor_coverage": max(
            (_safe_float(value.get("anchor_coverage")) for value in candidates),
            default=0.0,
        ),
        "semantic_score_margin_top2": score_margin_top2,
        "semantic_score_margin_top5": score_margin_top5,
        "semantic_ontology_top2_overlap": _overlap(semantic_ids, ontology_ids, 2),
        "semantic_ontology_top5_overlap": _overlap(semantic_ids, ontology_ids, 5),
        "ontology_top2_novelty": 1.0 - _overlap(ontology_ids, semantic_ids, 2),
        "ontology_top5_novelty": 1.0 - _overlap(ontology_ids, semantic_ids, 5),
        "plan_semantic_top2_overlap": _overlap(plan_ids, semantic_ids, 2),
        "plan_semantic_top5_overlap": _overlap(plan_ids, semantic_ids, 5),
        "plan_ontology_top2_overlap": _overlap(plan_ids, ontology_ids, 2),
        "plan_ontology_top5_overlap": _overlap(plan_ids, ontology_ids, 5),
        "plan_complete_proof_top2_fraction": sum(
            int(value in complete_ids) for value in plan_ids[:2]
        ) / 2.0,
        "plan_complete_proof_top5_fraction": sum(
            int(value in complete_ids) for value in plan_ids[:5]
        ) / 5.0,
        "plan_bound_proof_top2_fraction": sum(
            int(value in bound_ids) for value in plan_ids[:2]
        ) / 2.0,
        "plan_bound_proof_top5_fraction": sum(
            int(value in bound_ids) for value in plan_ids[:5]
        ) / 5.0,
        "plan_bound_terminal_type_top2": plan_mean(
            "bound_terminal_answer_type_score", 2
        ),
        "plan_bound_terminal_type_top5": plan_mean(
            "bound_terminal_answer_type_score", 5
        ),
        "plan_relation_coverage_top2": plan_mean("relation_coverage_score", 2),
        "plan_relation_coverage_top5": plan_mean("relation_coverage_score", 5),
        "plan_is_ontology": float(plan_name == PLAN_ONTOLOGY),
        "plan_is_proof_chain": float(plan_name == PLAN_PROOF_CHAIN),
        "plan_is_top2_semijoin": float(plan_name == PLAN_TOP2_SEMIJOIN),
        "plan_is_top5_semijoin": float(plan_name == PLAN_TOP5_SEMIJOIN),
        "plan_is_hybrid": float(plan_name == PLAN_HYBRID),
        "plan_is_bound_proof": float(
            plan_name in {
                PLAN_BOUND_PROOF,
                PLAN_QUERY_LOCAL_TOP2_PAIR,
                PLAN_QUERY_LOCAL_TOP5,
                PLAN_SIGNATURE_GROUNDED,
                PLAN_BRANCH_COMPLETE_TOP2,
            }
        ),
        "plan_is_query_local_top2_pair": float(
            plan_name == PLAN_QUERY_LOCAL_TOP2_PAIR
        ),
        "plan_is_query_local_top5_completion": float(
            plan_name == PLAN_QUERY_LOCAL_TOP5
        ),
        "plan_is_signature_grounded": float(
            plan_name == PLAN_SIGNATURE_GROUNDED
        ),
        "plan_is_branch_complete_top2": float(
            plan_name == PLAN_BRANCH_COMPLETE_TOP2
        ),
        "plan_is_rank_partitioned_top2_ordering": float(
            plan_name == PLAN_RANK_PARTITIONED_TOP2
        ),
        "plan_is_rank_partitioned_top5_completion": float(
            plan_name == PLAN_RANK_PARTITIONED_TOP5
        ),
        "plan_is_rank_partitioned_nested_topk": float(
            plan_name == PLAN_RANK_PARTITIONED_NESTED
        ),
        "plan_is_learned_residual_top5_completion": float(
            plan_name == PLAN_LEARNED_RESIDUAL_TOP5
        ),
        "plan_is_ontology_structural_rank5_completion": float(
            plan_name == PLAN_ONTOLOGY_STRUCTURAL_TOP5
        ),
        "plan_is_ontology_warehouse_rank5_completion": float(
            plan_name == PLAN_ONTOLOGY_WAREHOUSE_TOP5
        ),
        "plan_is_topology_conditioned_multi_access_top5": float(
            plan_name == PLAN_TOPOLOGY_MULTI_ACCESS_TOP5
        ),
        "plan_is_semantic_fact_scan": float(
            plan_name in {PLAN_SEMANTIC, PLAN_SEMANTIC_TOP2_NOOP}
        ),
        "plan_is_semantic_top2_noop": float(
            plan_name == PLAN_SEMANTIC_TOP2_NOOP
        ),
        "plan_is_raw_semantic_graph_scan": float(
            plan_name == PLAN_RAW_SEMANTIC
        ),
        "plan_is_ontology_physical_complete_slate": float(
            plan_name == PLAN_ONTOLOGY_PHYSICAL
        ),
        "plan_is_ontology_structural_execution_slate": float(
            plan_name == PLAN_STRUCTURAL_EXECUTION
        ),
        "plan_is_incumbent_preserving_chain_completion": float(
            plan_name == PLAN_CHAIN_COMPLETION
        ),
        "plan_objective_top2_noop": float(
            plan_name == PLAN_SEMANTIC_TOP2_NOOP
        ),
        "plan_top2_membership_invariant": float(
            set(plan_ids[:2]) == set(semantic_ids[:2])
        ),
        "plan_top5_membership_invariant": float(
            set(plan_ids[:5]) == set(semantic_ids[:5])
        ),
        "signature_topology_ordered_chain": float(
            str(query_signature.get("topology", ""))
            == "ordered_multihop_chain"
        ),
        "signature_topology_branch_comparison": float(
            str(query_signature.get("topology", ""))
            == "branch_comparison"
        ),
        "plan_signature_bound_top2_fraction": sum(
            int(value in signature_bound_ids) for value in plan_ids[:2]
        ) / 2.0,
        "plan_signature_bound_top5_fraction": sum(
            int(value in signature_bound_ids) for value in plan_ids[:5]
        ) / 5.0,
        "plan_semantic_access_top2_fraction": sum(
            int(value in semantic_set) for value in plan_ids[:2]
        ) / 2.0,
        "plan_semantic_access_top5_fraction": sum(
            int(value in semantic_set) for value in plan_ids[:5]
        ) / 5.0,
        "plan_untyped_residual_top5_fraction": sum(
            int(value not in candidate_by_id) for value in plan_ids[:5]
        ) / 5.0,
        "plan_semantic_top2_retained_fraction": (
            len(set(plan_ids[:2]) & semantic_top2) / max(len(semantic_top2), 1)
        ),
        "plan_semantic_top5_retained_fraction": (
            len(set(plan_ids[:5]) & semantic_top5) / max(len(semantic_top5), 1)
        ),
        "plan_top2_displacement_count": float(len(introduced_top2)),
        "plan_top5_displacement_count": float(len(introduced_top5)),
        "plan_introduced_top2_proof_fraction": (
            sum(int(value in candidate_by_id) for value in introduced_top2)
            / max(len(introduced_top2), 1)
        ),
        "plan_introduced_top5_proof_fraction": (
            sum(int(value in candidate_by_id) for value in introduced_top5)
            / max(len(introduced_top5), 1)
        ),
        "plan_introduced_complete_proof_fraction": (
            sum(int(value in complete_ids) for value in introduced_top5)
            / max(len(introduced_top5), 1)
        ),
        "plan_introduced_relation_coverage_mean": _mean(
            _safe_float(value.get("relation_coverage_score"))
            for value in introduced_candidates
        ),
        "plan_introduced_original_rank_mean_log": _mean(
            math.log1p(max(_safe_int(value.get("original_rank"), 0), 0))
            for value in introduced_candidates
        ),
        "estimated_cost_log": math.log1p(estimated_cost),
        "plan_pair_path_connected": float(
            bool(plan_metadata.get("pair_path_connected"))
        ),
        "plan_completion_path_connected": float(
            bool(plan_metadata.get("completion_path_connected"))
        ),
        "plan_promoted_semantic_rank_log": math.log1p(max(
            _safe_int(plan_metadata.get("promoted_semantic_rank"), 0), 0
        )),
        "plan_completion_original_rank_log": math.log1p(max(
            _safe_int(plan_metadata.get("completion_original_rank"), 0), 0
        )),
        "warehouse_replacement_rank_log": math.log1p(max(
            _safe_int(plan_metadata.get("warehouse_replacement_rank"), 0), 0
        )),
        "residual_candidate_pool_rank_log": math.log1p(max(
            _safe_int(plan_metadata.get("residual_candidate_pool_rank"), 0), 0
        )),
        "residual_candidate_original_rank_log": math.log1p(max(
            _safe_int(plan_metadata.get("residual_candidate_original_rank"), 0), 0
        )),
        "residual_candidate_semantic_rank_log": math.log1p(max(
            _safe_int(plan_metadata.get("residual_candidate_semantic_rank"), 0), 0
        )),
        "residual_candidate_score": _safe_float(
            plan_metadata.get("residual_candidate_score")
        ),
        "residual_candidate_confidence": _safe_float(
            plan_metadata.get("residual_candidate_confidence")
        ),
        "residual_candidate_query_coverage": _safe_float(
            plan_metadata.get("residual_candidate_query_coverage")
        ),
        "residual_candidate_anchor_coverage": _safe_float(
            plan_metadata.get("residual_candidate_anchor_coverage")
        ),
        "residual_candidate_relation_coverage": _safe_float(
            plan_metadata.get("residual_candidate_relation_coverage")
        ),
        "residual_candidate_answer_type_score": _safe_float(
            plan_metadata.get("residual_candidate_answer_type_score")
        ),
        "residual_candidate_has_proof_signal": float(bool(
            plan_metadata.get("residual_candidate_has_proof_signal")
        )),
        "residual_candidate_complete_provenance": float(bool(
            plan_metadata.get("residual_candidate_complete_provenance")
        )),
        "residual_candidate_binding": float(bool(
            plan_metadata.get("residual_candidate_binding")
        )),
        "residual_candidate_typed_slot_binding": float(bool(
            plan_metadata.get("residual_candidate_typed_slot_binding")
        )),
        "residual_candidate_terminal_relation_match": float(bool(
            plan_metadata.get("residual_candidate_terminal_relation_match")
        )),
        "residual_candidate_exact_entity_key": float(bool(
            plan_metadata.get("residual_candidate_exact_entity_key")
        )),
        "residual_candidate_role_consistent": float(bool(
            plan_metadata.get("residual_candidate_role_consistent")
        )),
        "residual_candidate_access_source_count_log": math.log1p(max(
            _safe_int(
                plan_metadata.get("residual_candidate_access_source_count"), 0
            ),
            0,
        )),
        "residual_candidate_path_connected": float(bool(
            plan_metadata.get("residual_candidate_path_connected")
        )),
        "residual_displaced_query_coverage": _safe_float(
            plan_metadata.get("residual_displaced_query_coverage")
        ),
        "residual_displaced_anchor_coverage": _safe_float(
            plan_metadata.get("residual_displaced_anchor_coverage")
        ),
        "residual_displaced_relation_coverage": _safe_float(
            plan_metadata.get("residual_displaced_relation_coverage")
        ),
        "residual_displaced_answer_type_score": _safe_float(
            plan_metadata.get("residual_displaced_answer_type_score")
        ),
        "residual_delta_query_coverage": _safe_float(
            plan_metadata.get("residual_delta_query_coverage")
        ),
        "residual_delta_anchor_coverage": _safe_float(
            plan_metadata.get("residual_delta_anchor_coverage")
        ),
        "residual_delta_relation_coverage": _safe_float(
            plan_metadata.get("residual_delta_relation_coverage")
        ),
        "residual_delta_answer_type_score": _safe_float(
            plan_metadata.get("residual_delta_answer_type_score")
        ),
        "residual_displaced_score": _safe_float(
            plan_metadata.get("residual_displaced_score")
        ),
        "residual_delta_score": _safe_float(
            plan_metadata.get("residual_delta_score")
        ),
        "residual_score_ratio": _safe_float(
            plan_metadata.get("residual_score_ratio")
        ),
        "residual_displaced_has_proof_signal": float(bool(
            plan_metadata.get("residual_displaced_has_proof_signal")
        )),
        "residual_candidate_from_semantic_access": float(bool(
            plan_metadata.get("residual_candidate_from_semantic_access")
        )),
        "residual_candidate_from_ontology_access": float(bool(
            plan_metadata.get("residual_candidate_from_ontology_access")
        )),
        "signature_anchor_count_log": math.log1p(max(
            _safe_int(query_signature.get("anchor_count"), 0), 0
        )),
        "signature_is_comparison": float(bool(
            query_signature.get("comparison")
        )),
        "signature_answer_type_person": float(
            signature_answer_type == "person"
        ),
        "signature_answer_type_location": float(
            signature_answer_type in {"location", "place", "country", "city"}
        ),
        "signature_answer_type_date": float(
            signature_answer_type in {"date", "year", "time"}
        ),
        "signature_answer_type_number": float(
            signature_answer_type in {"number", "count", "quantity"}
        ),
        "signature_answer_type_boolean": float(
            signature_answer_type in {"boolean", "bool", "yes_no"}
        ),
        "signature_answer_type_organization": float(
            signature_answer_type in {"organization", "organisation", "company"}
        ),
        "signature_answer_type_other": float(
            signature_answer_type not in {
                "person", "location", "place", "country", "city", "date",
                "year", "time", "number", "count", "quantity", "boolean",
                "bool", "yes_no", "organization", "organisation", "company",
            }
        ),
    }
    return {name: _safe_float(features.get(name)) for name in FEATURE_NAMES}


def materialize_complete_plan_portfolio(
    query_plan_log: Mapping[str, Any],
    current_doc_ids: Sequence[int],
    current_doc_scores: Sequence[float],
    *,
    top_k: int = 5,
    allowed_plan_families: Iterable[str] | None = None,
    shortlist_per_family: int | None = None,
    query_relative_semantic_competitor: bool = False,
) -> Dict[str, Any]:



    final_incumbent = query_plan_log.get("final_ranking_incumbent", {}) or {}
    semantic = query_plan_log.get("semantic_plan_incumbent", {}) or {}
    current_ids = _ids(current_doc_ids)
    semantic_ids = _ids(
        final_incumbent.get("candidate_window_doc_ids", [])
        or final_incumbent.get("doc_ids", [])
        or current_ids
    )
    semantic_scores = [
        _safe_float(value)
        for value in (
            final_incumbent.get("candidate_window_doc_scores", [])
            or final_incumbent.get("doc_scores", [])
            or current_doc_scores
        )
    ][: len(semantic_ids)]
    semantic_access_ids = _ids(
        semantic.get("candidate_window_doc_ids", [])
        or semantic.get("doc_ids", [])
    )
    semantic_access_scores = [
        _safe_float(value)
        for value in (
            semantic.get("candidate_window_doc_scores", [])
            or semantic.get("doc_scores", [])
        )
    ][: len(semantic_access_ids)]
    ontology_physical = query_plan_log.get(
        "ontology_physical_plan_incumbent", {}
    ) or {}
    ontology_physical_ids = _ids(
        ontology_physical.get("candidate_window_doc_ids", [])
        or ontology_physical.get("doc_ids", [])
    )
    structural_execution = query_plan_log.get(
        "pre_complete_structural_plan_incumbent", {}
    ) or query_plan_log.get("candidate_admission_incumbent", {}) or {}
    structural_snapshot_ids = _ids(
        structural_execution.get("candidate_window_doc_ids", [])
        or structural_execution.get("doc_ids", [])
    )
    factorized_execution = query_plan_log.get("factorized_query_executor", {}) or {}
    factorized_provenance = dict(
        factorized_execution.get("schema_execution_provenance", {}) or {}
    )
    terminal_projection = dict(
        factorized_execution.get("terminal_projection_certification", {}) or {}
    )
    certified_terminal_doc_ids = {
        int(value)
        for value in terminal_projection.get("candidate_doc_ids", []) or []
        if str(value).lstrip("-").isdigit() and int(value) >= 0
    }
    strict_endpoint_execution = bool(
        factorized_provenance.get("route_contract_matched", False)
        and factorized_provenance.get("logical_plan_compiled", False)
        and factorized_provenance.get("logical_plan_admissible", False)
        and factorized_provenance.get("relation_realization_resolved", False)
        and factorized_provenance.get("physical_operator_selected", False)
        and factorized_provenance.get("physical_execution_completed", False)
        and factorized_provenance.get("endpoint_grounded_source_schema_ids")
        and terminal_projection.get("certified", False)
    )
    endpoint_provenance_candidate_ids = _ids([
        _safe_int(candidate.get("passage_idx", -1), -1)
        for candidate in factorized_execution.get("candidates", []) or []
        if isinstance(candidate, Mapping)
        and _safe_int(candidate.get("passage_idx", -1), -1) in certified_terminal_doc_ids
        and bool(candidate.get("exact_entity_key", False))
        and bool(candidate.get("provenance_validated", False))
        and bool(candidate.get("candidate_window_path", False))
        and int(candidate.get("contextual_relation_count", 0) or 0) > 0
    ]) if strict_endpoint_execution else []
    if endpoint_provenance_candidate_ids:




        structural_execution_ids = _ids([
            *semantic_ids[:2],
            *endpoint_provenance_candidate_ids[:1],
            *semantic_ids,
        ])
    else:
        structural_execution_ids = _ids([
            *(factorized_execution.get("proposed_top5", []) or []),
            *structural_snapshot_ids,
        ])
    ontology_physical_scores = [
        _safe_float(value)
        for value in (
            ontology_physical.get("candidate_window_doc_scores", [])
            or ontology_physical.get("doc_scores", [])
        )
    ][: len(ontology_physical_ids)]
    requested_families = (
        {str(value) for value in allowed_plan_families}
        if allowed_plan_families is not None
        else None
    )

    def family_requested(plan_name: str) -> bool:
        return (
            plan_name == PLAN_SEMANTIC
            or requested_families is None
            or plan_name in requested_families
        )
    proof_slate = query_plan_log.get("full_aperture_proof_slate", {}) or {}
    ontology_ids, candidate_by_id = _ontology_order(proof_slate)
    hybrid_ids = _rank_merge(semantic_ids, ontology_ids)
    strongest_path_ids = _strongest_local_path(proof_slate, candidate_by_id)
    strongest_bound_path_ids = _strongest_bound_path(
        proof_slate, candidate_by_id
    )
    strongest_query_local_path_ids = _strongest_query_local_path(
        proof_slate, candidate_by_id
    )
    strongest_branch_complete_path_ids = _strongest_branch_complete_path(
        proof_slate, candidate_by_id
    )
    query_structure_signature = dict(
        proof_slate.get("query_structure_signature", {}) or {}
    )
    signature_path_set = set(strongest_query_local_path_ids)
    semantic_set = set(semantic_ids)
    typed_semantic_residual_ids = [
        passage_idx
        for passage_idx, candidate in sorted(
            candidate_by_id.items(),
            key=lambda item: _candidate_quality(item[1]),
            reverse=True,
        )
        if passage_idx in semantic_set and passage_idx not in signature_path_set
    ]
    untyped_semantic_residual_ids = [
        passage_idx
        for passage_idx in semantic_ids
        if passage_idx not in signature_path_set
        and passage_idx not in candidate_by_id
    ]
    semantic_residual_ids = [
        passage_idx
        for passage_idx in semantic_ids
        if passage_idx not in signature_path_set
    ]
    typed_physical_residual_ids = [
        passage_idx
        for passage_idx, candidate in sorted(
            candidate_by_id.items(),
            key=lambda item: _candidate_quality(item[1]),
            reverse=True,
        )
        if passage_idx not in semantic_set
        and passage_idx not in signature_path_set
    ]
    physical_residual_ids = [
        passage_idx
        for passage_idx in current_ids
        if passage_idx not in signature_path_set
        and passage_idx not in semantic_set
        and passage_idx not in candidate_by_id
    ]
    all_bound_candidate_ids = [
        passage_idx
        for passage_idx, candidate in sorted(
            candidate_by_id.items(),
            key=lambda item: _candidate_quality(item[1]),
            reverse=True,
        )
        if bool(candidate.get("bound_proof_path_complete"))
    ]
    bound_proof_join_ids = _complete_order(
        strongest_bound_path_ids,
        all_bound_candidate_ids,
        ontology_ids,
    )
    query_local_pair_source = (
        strongest_query_local_path_ids[0]
        if len(strongest_query_local_path_ids) >= 2
        else None
    )
    query_local_pair_terminal = (
        strongest_query_local_path_ids[-1]
        if len(strongest_query_local_path_ids) >= 2
        else None
    )
    query_local_pair_source_in_semantic_top2 = bool(
        query_local_pair_source is not None
        and query_local_pair_source in set(semantic_ids[:2])
    )




    query_local_top2_pair_ids = _complete_order(
        [] if query_local_pair_source is None else [query_local_pair_source],
        [] if query_local_pair_terminal is None else [query_local_pair_terminal],
        strongest_query_local_path_ids[1:-1],
        semantic_ids,
        all_bound_candidate_ids,
        ontology_ids,
    )



    query_local_top5_ids = _complete_order(
        semantic_ids[:2],
        strongest_query_local_path_ids,
        semantic_ids[2:],
        all_bound_candidate_ids,
        ontology_ids,
    )



    incumbent_preserving_chain_ids = _complete_order(
        semantic_ids[:2],
        strongest_query_local_path_ids,
        semantic_access_ids[2:],
        semantic_ids[2:],
        all_bound_candidate_ids,
        ontology_ids,
    )
    branch_complete_top2_ids = _complete_order(
        strongest_branch_complete_path_ids,
        semantic_ids,
        semantic_access_ids,
        all_bound_candidate_ids,
        ontology_ids,
    )
    topology = str(query_structure_signature.get("topology", "") or "")
    if topology == "ordered_multihop_chain":
        signature_route = PLAN_QUERY_LOCAL_TOP2_PAIR
        signature_grounded_ids = query_local_top2_pair_ids
    elif topology == "branch_comparison":
        if strongest_branch_complete_path_ids:
            signature_route = PLAN_BRANCH_COMPLETE_TOP2
            signature_grounded_ids = branch_complete_top2_ids
        else:
            signature_route = PLAN_QUERY_LOCAL_TOP5
            signature_grounded_ids = query_local_top5_ids
    else:
        signature_route = PLAN_SEMANTIC
        signature_grounded_ids = list(semantic_ids)
    proof_chain_ids = _complete_order(
        strongest_path_ids, semantic_ids, ontology_ids
    )
    top2_novel = next(
        (value for value in ontology_ids if value not in set(semantic_ids[:2])),
        None,
    )
    top2_semijoin_ids = _complete_order(
        semantic_ids[:1],
        [] if top2_novel is None else [top2_novel],
        semantic_ids[1:],
        ontology_ids,
    )
    top5_novel = [value for value in ontology_ids if value not in set(semantic_ids[:5])]
    top5_semijoin_ids = _complete_order(
        semantic_ids[:2], top5_novel[:3], semantic_ids[2:], ontology_ids
    )
    proof_ranked_incumbent_top5 = _proof_ranked_incumbent_top5(
        semantic_ids, candidate_by_id
    )
    top2_pair_candidates = _incumbent_top2_pair_candidates(
        proof_slate, semantic_ids, candidate_by_id
    )
    connected_external_ids = _connected_external_candidates(
        proof_slate, semantic_ids, candidate_by_id
    )
    all_external_ids = [
        passage_idx
        for passage_idx in ontology_ids
        if passage_idx not in set(semantic_ids[:5])
    ]
    completion_candidates = _complete_order(
        connected_external_ids, all_external_ids
    )[:4]
    residual_top5_candidates, full_candidate_by_id = (
        _residual_top5_candidate_pool(
            proof_slate,
            semantic_ids,
            connected_external_ids,
        )
    )
    warehouse_top5_candidates, _ = _ontology_warehouse_top5_candidate_pool(
        proof_slate,
        semantic_ids,
    )
    multi_access_slates = _topology_conditioned_multi_access_slates(
        topology,
        semantic_ids,
        strongest_query_local_path_ids,
        strongest_branch_complete_path_ids,
        connected_external_ids,
        warehouse_top5_candidates,
        full_candidate_by_id,
    )
    plans: Dict[str, Dict[str, Any]] = {}
    raw_plans: Dict[str, Dict[str, Any]] = {
        PLAN_SEMANTIC: {"family": PLAN_SEMANTIC, "doc_ids": semantic_ids},
        PLAN_SEMANTIC_TOP2_NOOP: {
            "family": PLAN_SEMANTIC_TOP2_NOOP,
            "doc_ids": semantic_ids,
            "metadata": {
                "access_path": "semantic_fact_scan",
                "objective_stage": "top2_ordering",
                "zero_modification": True,
            },
        },
        PLAN_RAW_SEMANTIC: {
            "family": PLAN_RAW_SEMANTIC,
            "doc_ids": semantic_access_ids,
            "metadata": {"access_path": "pre_ontology_semantic_fact_graph"},
        },
        PLAN_ONTOLOGY_PHYSICAL: {
            "family": PLAN_ONTOLOGY_PHYSICAL,
            "doc_ids": ontology_physical_ids,
            "metadata": {
                "access_path": "ontology_ppr_graph_then_complete_topk",
                "complete_execution": True,
            },
        },
        PLAN_STRUCTURAL_EXECUTION: {
            "family": PLAN_STRUCTURAL_EXECUTION,
            "doc_ids": structural_execution_ids,
            "metadata": {
                "access_path": (
                    "endpoint_provenance_terminal_projection_then_complete_topk"
                    if endpoint_provenance_candidate_ids
                    else "late_structural_execution_then_complete_topk"
                ),
                "complete_execution": True,
                "endpoint_provenance_candidate_count": len(
                    endpoint_provenance_candidate_ids
                ),
                "endpoint_provenance_candidate_ids": list(
                    endpoint_provenance_candidate_ids
                ),
                "incumbent_top2_preserved": bool(
                    structural_execution_ids[:2] == semantic_ids[:2]
                ),
            },
        },
        PLAN_CHAIN_COMPLETION: {
            "family": PLAN_CHAIN_COMPLETION,
            "doc_ids": incumbent_preserving_chain_ids,
            "metadata": {
                "completion_path_connected": bool(
                    strongest_query_local_path_ids
                ),
                "access_path": "incumbent_top2_then_provenance_chain",
            },
        },
        PLAN_SIGNATURE_GROUNDED: {
            "family": PLAN_SIGNATURE_GROUNDED,
            "doc_ids": signature_grounded_ids,
        },
        PLAN_BRANCH_COMPLETE_TOP2: {
            "family": PLAN_BRANCH_COMPLETE_TOP2,
            "doc_ids": branch_complete_top2_ids,
        },
        PLAN_BOUND_PROOF: {"family": PLAN_BOUND_PROOF, "doc_ids": bound_proof_join_ids},
        PLAN_QUERY_LOCAL_TOP2_PAIR: {
            "family": PLAN_QUERY_LOCAL_TOP2_PAIR,
            "doc_ids": query_local_top2_pair_ids,
        },
        PLAN_QUERY_LOCAL_TOP5: {
            "family": PLAN_QUERY_LOCAL_TOP5,
            "doc_ids": query_local_top5_ids,
        },
        PLAN_ONTOLOGY: {"family": PLAN_ONTOLOGY, "doc_ids": ontology_ids},
        PLAN_PROOF_CHAIN: {"family": PLAN_PROOF_CHAIN, "doc_ids": proof_chain_ids},
        PLAN_TOP2_SEMIJOIN: {"family": PLAN_TOP2_SEMIJOIN, "doc_ids": top2_semijoin_ids},
        PLAN_TOP5_SEMIJOIN: {"family": PLAN_TOP5_SEMIJOIN, "doc_ids": top5_semijoin_ids},
        PLAN_HYBRID: {"family": PLAN_HYBRID, "doc_ids": hybrid_ids},
    }
    if family_requested(PLAN_TOPOLOGY_MULTI_ACCESS_TOP5):
        for variant, doc_ids, metadata in multi_access_slates:
            raw_plans[f"{PLAN_TOPOLOGY_MULTI_ACCESS_TOP5}::{variant}"] = {
                "family": PLAN_TOPOLOGY_MULTI_ACCESS_TOP5,
                "doc_ids": doc_ids,
                "metadata": metadata,
            }
    raw_plans = {
        key: value
        for key, value in raw_plans.items()
        if family_requested(str(value.get("family") or key))
    }
    for pair_index, pair_spec in enumerate(
        top2_pair_candidates[:3] if (
            family_requested(PLAN_RANK_PARTITIONED_TOP2)
            or family_requested(PLAN_RANK_PARTITIONED_NESTED)
        ) else [],
        start=1,
    ):
        pair = list(pair_spec["pair"])
        metadata = {
            "pair_path_connected": bool(pair_spec["pair_path_connected"]),
            "promoted_semantic_rank": int(pair_spec["promoted_semantic_rank"]),
        }
        top2_key = f"{PLAN_RANK_PARTITIONED_TOP2}::{pair_index}"
        if family_requested(PLAN_RANK_PARTITIONED_TOP2):
            raw_plans[top2_key] = {
                "family": PLAN_RANK_PARTITIONED_TOP2,
                "doc_ids": _complete_order(pair, semantic_ids[:5], semantic_ids[5:]),
                "metadata": metadata,
            }
        for completion_index, completion_id in enumerate(
            completion_candidates, start=1
        ):
            preserved = [
                passage_idx
                for passage_idx in semantic_ids[:5]
                if passage_idx not in set(pair)
            ][:2]
            nested_key = (
                f"{PLAN_RANK_PARTITIONED_NESTED}::{pair_index}:{completion_index}"
            )
            completion_candidate = candidate_by_id.get(completion_id, {}) or {}
            if not family_requested(PLAN_RANK_PARTITIONED_NESTED):
                continue
            raw_plans[nested_key] = {
                "family": PLAN_RANK_PARTITIONED_NESTED,
                "doc_ids": _complete_order(
                    pair,
                    preserved,
                    [completion_id],
                    semantic_ids[:5],
                    semantic_ids[5:],
                    completion_candidates,
                ),
                "metadata": {
                    **metadata,
                    "completion_path_connected": completion_id in set(
                        connected_external_ids
                    ),
                    "completion_original_rank": _safe_int(
                        completion_candidate.get("original_rank"), 0
                    ),
                },
            }
    for completion_index, completion_id in enumerate(
        completion_candidates
        if family_requested(PLAN_RANK_PARTITIONED_TOP5)
        else [],
        start=1,
    ):
        completion_candidate = candidate_by_id.get(completion_id, {}) or {}
        top5_key = f"{PLAN_RANK_PARTITIONED_TOP5}::{completion_index}"
        raw_plans[top5_key] = {
            "family": PLAN_RANK_PARTITIONED_TOP5,
            "doc_ids": _complete_order(
                semantic_ids[:4],
                [completion_id],
                semantic_ids[4:],
                completion_candidates,
            ),
            "metadata": {
                "completion_path_connected": completion_id in set(
                    connected_external_ids
                ),
                "completion_original_rank": _safe_int(
                    completion_candidate.get("original_rank"), 0
                ),
            },
        }

    semantic_rank_by_id = {
        passage_idx: rank
        for rank, passage_idx in enumerate(semantic_ids, start=1)
    }
    warehouse_stage1_top2_admissible = bool(
        len(strongest_query_local_path_ids) >= 2
        and query_local_pair_source != query_local_pair_terminal
        and query_local_pair_source_in_semantic_top2
    )
    warehouse_stage1_ids = (
        list(query_local_top2_pair_ids)
        if warehouse_stage1_top2_admissible
        else list(semantic_ids)
    )
    warehouse_stage1_top5 = warehouse_stage1_ids[:5]
    default_displacement_index = 4
    connected_external_set = set(connected_external_ids)
    shallow_semantic_residual_set = set(semantic_ids[5:29])
    ontology_structural_candidate_id = next(
        (
            candidate_id
            for candidate_id in residual_top5_candidates
            if candidate_id not in shallow_semantic_residual_set
        ),
        None,
    )
    residual_candidate_items: List[Tuple[str, int, int, int]] = []
    if family_requested(PLAN_LEARNED_RESIDUAL_TOP5):
        residual_candidate_items.extend(
            (
                PLAN_LEARNED_RESIDUAL_TOP5,
                pool_rank,
                candidate_id,
                default_displacement_index,
            )
            for pool_rank, candidate_id in enumerate(
                residual_top5_candidates, start=1
            )
        )
    if (
        family_requested(PLAN_ONTOLOGY_STRUCTURAL_TOP5)
        and ontology_structural_candidate_id is not None
    ):
        residual_candidate_items.append((
            PLAN_ONTOLOGY_STRUCTURAL_TOP5,
            residual_top5_candidates.index(ontology_structural_candidate_id) + 1,
            ontology_structural_candidate_id,
            default_displacement_index,
        ))
    if family_requested(PLAN_ONTOLOGY_WAREHOUSE_TOP5):
        residual_candidate_items.extend(
            (
                PLAN_ONTOLOGY_WAREHOUSE_TOP5,
                pool_rank,
                candidate_id,
                displacement_index,
            )
            for pool_rank, candidate_id in enumerate(
                warehouse_top5_candidates, start=1
            )
            for displacement_index in range(2, min(len(semantic_ids), 5))
        )
    for (
        candidate_family,
        pool_rank,
        candidate_id,
        displacement_index,
    ) in residual_candidate_items:
        if (
            candidate_family == PLAN_ONTOLOGY_WAREHOUSE_TOP5
            and candidate_id in set(warehouse_stage1_top5)
        ):
            continue
        candidate = full_candidate_by_id.get(candidate_id, {}) or {}
        displacement_base = (
            warehouse_stage1_ids
            if candidate_family == PLAN_ONTOLOGY_WAREHOUSE_TOP5
            else semantic_ids
        )
        displaced_id = (
            displacement_base[displacement_index]
            if displacement_index < len(displacement_base)
            else -1
        )
        displaced_candidate = full_candidate_by_id.get(displaced_id, {}) or {}
        displaced_score = _safe_float(displaced_candidate.get("score"))
        if (
            displaced_score == 0.0
            and displacement_index < len(semantic_scores)
        ):
            displaced_score = _safe_float(
                semantic_scores[displacement_index]
            )
        candidate_semantic_rank = semantic_rank_by_id.get(
            candidate_id,
            _safe_int(candidate.get("semantic_access_rank"), 0),
        )
        candidate_score = _safe_float(candidate.get("score"))
        if candidate_score == 0.0 and 0 < candidate_semantic_rank <= len(
            semantic_scores
        ):
            candidate_score = _safe_float(
                semantic_scores[candidate_semantic_rank - 1]
            )
        candidate_query_coverage = _safe_float(candidate.get("query_coverage"))
        candidate_anchor_coverage = _safe_float(candidate.get("anchor_coverage"))
        candidate_relation_coverage = _safe_float(
            candidate.get("relation_coverage_score")
        )
        candidate_answer_type_score = _safe_float(
            candidate.get("answer_type_score")
        )
        displaced_query_coverage = _safe_float(
            displaced_candidate.get("query_coverage")
        )
        displaced_anchor_coverage = _safe_float(
            displaced_candidate.get("anchor_coverage")
        )
        displaced_relation_coverage = _safe_float(
            displaced_candidate.get("relation_coverage_score")
        )
        displaced_answer_type_score = _safe_float(
            displaced_candidate.get("answer_type_score")
        )
        candidate_tail = (
            warehouse_top5_candidates
            if candidate_family == PLAN_ONTOLOGY_WAREHOUSE_TOP5
            else residual_top5_candidates
        )
        replacement_prefix = list(displacement_base[:5])
        if displacement_index < len(replacement_prefix):
            replacement_prefix[displacement_index] = candidate_id
        plan_doc_ids = _complete_order(
            replacement_prefix,
            semantic_ids,
            candidate_tail,
        )
        plan_metadata = {
                "completion_path_connected": (
                    candidate_id in connected_external_set
                ),
                "completion_original_rank": _safe_int(
                    candidate.get("original_rank"), candidate_semantic_rank
                ),
                "residual_candidate_pool_rank": pool_rank,
                "warehouse_replacement_rank": displacement_index + 1,
                "warehouse_stage1_top2_pair": bool(
                    candidate_family == PLAN_ONTOLOGY_WAREHOUSE_TOP5
                    and warehouse_stage1_top2_admissible
                ),
                "warehouse_stage1_doc_ids": list(
                    warehouse_stage1_top5
                    if candidate_family == PLAN_ONTOLOGY_WAREHOUSE_TOP5
                    else semantic_ids[:5]
                ),
                "residual_candidate_original_rank": _safe_int(
                    candidate.get("original_rank"), candidate_semantic_rank
                ),
                "residual_candidate_semantic_rank": candidate_semantic_rank,
                "residual_candidate_score": candidate_score,
                "residual_candidate_confidence": _safe_float(
                    candidate.get("confidence")
                ),
                "residual_candidate_query_coverage": candidate_query_coverage,
                "residual_candidate_anchor_coverage": candidate_anchor_coverage,
                "residual_candidate_relation_coverage": (
                    candidate_relation_coverage
                ),
                "residual_candidate_answer_type_score": (
                    candidate_answer_type_score
                ),
                "residual_candidate_has_proof_signal": bool(
                    candidate.get("has_proof_signal")
                ),
                "residual_candidate_complete_provenance": bool(
                    candidate.get("complete_provenance")
                ),
                "residual_candidate_binding": bool(candidate.get("binding")),
                "residual_candidate_typed_slot_binding": bool(
                    candidate.get("typed_slot_binding")
                ),
                "residual_candidate_terminal_relation_match": bool(
                    candidate.get("terminal_relation_match")
                ),
                "residual_candidate_exact_entity_key": bool(
                    candidate.get("exact_entity_key")
                ),
                "residual_candidate_role_consistent": bool(
                    candidate.get("role_consistent")
                ),
                "residual_candidate_access_source_count": len(
                    candidate.get("access_sources", []) or []
                ),
                "residual_candidate_path_connected": (
                    candidate_id in connected_external_set
                ),
                "residual_displaced_query_coverage": displaced_query_coverage,
                "residual_displaced_anchor_coverage": displaced_anchor_coverage,
                "residual_displaced_relation_coverage": (
                    displaced_relation_coverage
                ),
                "residual_displaced_answer_type_score": (
                    displaced_answer_type_score
                ),
                "residual_delta_query_coverage": (
                    candidate_query_coverage - displaced_query_coverage
                ),
                "residual_delta_anchor_coverage": (
                    candidate_anchor_coverage - displaced_anchor_coverage
                ),
                "residual_delta_relation_coverage": (
                    candidate_relation_coverage - displaced_relation_coverage
                ),
                "residual_delta_answer_type_score": (
                    candidate_answer_type_score - displaced_answer_type_score
                ),
                "residual_displaced_score": displaced_score,
                "residual_delta_score": candidate_score - displaced_score,
                "residual_score_ratio": (
                    candidate_score / max(abs(displaced_score), 1e-12)
                ),
                "residual_displaced_has_proof_signal": bool(
                    displaced_candidate.get("has_proof_signal")
                ),
                "residual_candidate_from_semantic_access": (
                    candidate_id in semantic_rank_by_id
                ),
                "residual_candidate_from_ontology_access": bool(
                    candidate.get("has_proof_signal")
                    or candidate.get("physical_access_rank")
                ),
                "access_path": (
                    "ontology_warehouse_diverse_scan"
                    if candidate_family == PLAN_ONTOLOGY_WAREHOUSE_TOP5
                    else "late_materialized_candidate_reservoir"
                ),
        }
        if candidate_family == PLAN_LEARNED_RESIDUAL_TOP5:
            plan_key = f"{PLAN_LEARNED_RESIDUAL_TOP5}::{pool_rank}"
            raw_plans[plan_key] = {
                "family": PLAN_LEARNED_RESIDUAL_TOP5,
                "doc_ids": plan_doc_ids,
                "metadata": plan_metadata,
            }
        if (
            candidate_family == PLAN_ONTOLOGY_STRUCTURAL_TOP5
            and candidate_id == ontology_structural_candidate_id
        ):
            raw_plans[PLAN_ONTOLOGY_STRUCTURAL_TOP5] = {
                "family": PLAN_ONTOLOGY_STRUCTURAL_TOP5,
                "doc_ids": plan_doc_ids,
                "metadata": {
                    **plan_metadata,
                    "access_path": (
                        "first_incremental_ontology_structural_candidate"
                    ),
                    "bounded_single_action": True,
                },
            }
        if candidate_family == PLAN_ONTOLOGY_WAREHOUSE_TOP5:
            replacement_rank = displacement_index + 1
            plan_key = (
                f"{PLAN_ONTOLOGY_WAREHOUSE_TOP5}::"
                f"{pool_rank}:r{replacement_rank}"
            )
            raw_plans[plan_key] = {
                "family": PLAN_ONTOLOGY_WAREHOUSE_TOP5,
                "doc_ids": plan_doc_ids,
                "metadata": {
                    **plan_metadata,
                    "warehouse_tuple_rank": pool_rank,
                    "warehouse_replacement_rank": replacement_rank,
                    "complete_slate_transaction": True,
                },
            }

    for plan_key, raw_plan in raw_plans.items():
        plan_name = str(raw_plan.get("family") or plan_key)
        ordered = list(raw_plan.get("doc_ids", []) or [])
        plan_metadata = dict(raw_plan.get("metadata", {}) or {})
        changed = ordered[:top_k] != semantic_ids[:top_k]
        prerequisite = True
        reason = ""
        if plan_name == PLAN_BOUND_PROOF and not strongest_bound_path_ids:
            prerequisite = False
            reason = "no_anchor_bound_entity_join_path"
        elif plan_name == PLAN_SIGNATURE_GROUNDED:
            signature_path_ids = (
                strongest_branch_complete_path_ids
                if topology == "branch_comparison"
                else strongest_query_local_path_ids
            )
            if len(signature_path_ids) < 2:
                prerequisite = False
                reason = "no_signature_bound_multihop_path"
            elif len(signature_path_ids) > top_k:
                prerequisite = False
                reason = "signature_bound_path_exceeds_topk_budget"
            elif topology == "ordered_multihop_chain" and _safe_int(
                query_structure_signature.get("relation_slot_count"), 0
            ) < 2:
                prerequisite = False
                reason = "query_signature_is_not_multihop"
            elif topology == "branch_comparison" and _safe_int(
                query_structure_signature.get("anchor_count"), 0
            ) < 2:
                prerequisite = False
                reason = "comparison_signature_has_fewer_than_two_anchors"
            elif topology == "ordered_multihop_chain" and not query_local_pair_source_in_semantic_top2:
                prerequisite = False
                reason = "ordered_chain_source_outside_semantic_top2"
            elif topology == "ordered_multihop_chain" and not set(
                strongest_query_local_path_ids
            ).issubset(set(signature_grounded_ids[:top_k])):
                prerequisite = False
                reason = "ordered_chain_exceeds_complete_topk_budget"
            elif topology == "branch_comparison" and not set(
                signature_path_ids
            ).issubset(set(signature_grounded_ids[:top_k])):
                prerequisite = False
                reason = "comparison_branches_exceed_complete_topk_budget"
            elif topology not in {"ordered_multihop_chain", "branch_comparison"}:
                prerequisite = False
                reason = "unsupported_query_signature_topology"
        elif plan_name == PLAN_QUERY_LOCAL_TOP2_PAIR:
            if len(strongest_query_local_path_ids) < 2:
                prerequisite = False
                reason = "no_complete_query_local_provenance_pair"
            elif query_local_pair_source == query_local_pair_terminal:
                prerequisite = False
                reason = "degenerate_query_local_provenance_pair"
            elif not query_local_pair_source_in_semantic_top2:
                prerequisite = False
                reason = "query_local_source_outside_semantic_top2"
        elif plan_name == PLAN_BRANCH_COMPLETE_TOP2:
            if topology != "branch_comparison":
                prerequisite = False
                reason = "query_signature_is_not_branch_comparison"
            elif len(strongest_branch_complete_path_ids) != 2:
                prerequisite = False
                reason = "no_strict_two_branch_top2_proof"
            elif not set(strongest_branch_complete_path_ids).issubset(
                set(branch_complete_top2_ids[:2])
            ):
                prerequisite = False
                reason = "branch_proof_not_atomic_in_top2"
        elif (
            plan_name == PLAN_QUERY_LOCAL_TOP5
            and not strongest_query_local_path_ids
        ):
            prerequisite = False
            reason = "no_query_local_provenance_path"
        elif plan_name == PLAN_RAW_SEMANTIC and not semantic_access_ids:
            prerequisite = False
            reason = "raw_semantic_snapshot_missing"
        elif plan_name == PLAN_ONTOLOGY_PHYSICAL:
            if not ontology_physical_ids:
                prerequisite = False
                reason = "ontology_physical_snapshot_missing"
            elif len(ontology_physical_ids) < top_k:
                prerequisite = False
                reason = "ontology_physical_snapshot_too_short"
        elif plan_name == PLAN_STRUCTURAL_EXECUTION:
            if not structural_execution_ids:
                prerequisite = False
                reason = "structural_execution_snapshot_missing"
            elif len(structural_execution_ids) < top_k:
                prerequisite = False
                reason = "structural_execution_snapshot_too_short"
        elif plan_name == PLAN_CHAIN_COMPLETION:
            if not strongest_query_local_path_ids:
                prerequisite = False
                reason = "no_query_local_provenance_path"
            elif ordered[:2] != semantic_ids[:2]:
                prerequisite = False
                reason = "chain_completion_changed_incumbent_top2"
        elif plan_name == PLAN_TOPOLOGY_MULTI_ACCESS_TOP5:
            if topology not in {"ordered_multihop_chain", "branch_comparison"}:
                prerequisite = False
                reason = "unsupported_query_signature_topology"
            elif ordered[:2] != semantic_ids[:2]:
                prerequisite = False
                reason = "multi_access_slate_changed_incumbent_top2"
            elif not (set(ordered[:5]) - set(semantic_ids[:5])):
                prerequisite = False
                reason = "multi_access_slate_has_no_external_evidence"
        elif plan_name == PLAN_RANK_PARTITIONED_TOP2:
            if set(ordered[:5]) != set(semantic_ids[:5]):
                prerequisite = False
                reason = "top2_ordering_changed_top5_membership"
            elif set(ordered[:2]) == set(semantic_ids[:2]):
                prerequisite = False
                reason = "no_proof_backed_top2_promotion"
        elif plan_name == PLAN_RANK_PARTITIONED_TOP5:
            if ordered[:2] != semantic_ids[:2]:
                prerequisite = False
                reason = "top5_completion_changed_top2_order"
        elif plan_name in {
            PLAN_LEARNED_RESIDUAL_TOP5,
            PLAN_ONTOLOGY_STRUCTURAL_TOP5,
        }:
            if ordered[:4] != semantic_ids[:4]:
                prerequisite = False
                reason = "residual_completion_changed_incumbent_top4"
            elif len(ordered) < 5 or ordered[4] == semantic_ids[4]:
                prerequisite = False
                reason = "residual_completion_is_rank5_noop"
        elif plan_name == PLAN_ONTOLOGY_WAREHOUSE_TOP5:
            warehouse_stage1 = _ids(
                plan_metadata.get("warehouse_stage1_doc_ids", []) or []
            )
            if ordered[:2] != warehouse_stage1[:2]:
                prerequisite = False
                reason = "warehouse_completion_changed_stage1_top2"
            elif len(set(ordered[:5]) - set(warehouse_stage1[:5])) != 1:
                prerequisite = False
                reason = "warehouse_completion_is_not_one_tuple_swap"
        elif plan_name == PLAN_RANK_PARTITIONED_NESTED:
            if set(ordered[:2]) == set(semantic_ids[:2]):
                prerequisite = False
                reason = "no_proof_backed_top2_promotion"
        elif plan_name == PLAN_PROOF_CHAIN and not strongest_path_ids:
            prerequisite = False
            reason = "no_complete_local_proof_path"
        elif plan_name == PLAN_TOP2_SEMIJOIN and top2_novel is None:
            prerequisite = False
            reason = "no_missing_hop_candidate_outside_semantic_top2"
        elif plan_name == PLAN_TOP5_SEMIJOIN and not top5_novel:
            prerequisite = False
            reason = "no_proof_candidate_outside_semantic_top5"
        valid = len(ordered) >= top_k and prerequisite
        if not valid and not reason:
            reason = "insufficient_independent_topk"
        elif (
            valid
            and plan_name not in {PLAN_SEMANTIC, PLAN_SEMANTIC_TOP2_NOOP}
            and not changed
        ):
            valid = False
            reason = "physical_plan_is_semantic_noop"
        plan_ids = ordered[:top_k]
        features = (
            {}
            if plan_name == PLAN_SEMANTIC
            and not query_relative_semantic_competitor
            else _plan_features(
                query_plan_log,
                proof_slate,
                semantic_ids,
                semantic_scores,
                ontology_ids,
                plan_name,
                plan_ids,
                candidate_by_id,
                plan_metadata,
            )
        )
        plans[plan_key] = {
            "valid": valid,
            "invalid_reason": reason,
            "plan_family": plan_name,
            "plan_instance": plan_key,
            "plan_metadata": plan_metadata,
            "doc_ids": plan_ids,
            "state_hash": final_state_hash(plan_ids),
            "features": features,
        }
        if plan_name == PLAN_SIGNATURE_GROUNDED:
            plans[plan_key]["execution_contract"] = {
                "query_structure_signature": query_structure_signature,
                "signature_route": signature_route,
                "access_methods": [
                    "semantic_fact_scan",
                    "ontology_physical_operator",
                    "signature_slot_binding",
                    "query_local_provenance_join",
                    "residual_candidate_fill",
                ],
                "bound_path_doc_ids": strongest_query_local_path_ids,
                "branch_complete_doc_ids": strongest_branch_complete_path_ids,
                "typed_semantic_residual_doc_ids": typed_semantic_residual_ids,
                "untyped_semantic_residual_doc_ids": (
                    untyped_semantic_residual_ids
                ),
                "typed_physical_residual_doc_ids": typed_physical_residual_ids,
                "untyped_physical_residual_doc_ids": physical_residual_ids,
                "final_order_owned_by": "ontology_guided_evidence_executor",
            }
    plans[PLAN_SEMANTIC]["valid"] = len(semantic_ids) >= top_k
    plans[PLAN_SEMANTIC]["invalid_reason"] = (
        "" if len(semantic_ids) >= top_k else "semantic_snapshot_too_short"
    )
    _annotate_query_relative_features(
        plans, include_semantic=query_relative_semantic_competitor
    )
    candidate_count_before_shortlist = len(plans)
    shortlist_limit = max(
        _safe_int(
            shortlist_per_family
            if shortlist_per_family is not None
            else os.environ.get(
                "QUERY_PLAN_ONTOLOGY_COMPLETE_PLAN_SHORTLIST_PER_FAMILY", "0"
            )
        ),
        0,
    )
    shortlist_diagnostics = _shortlist_query_grouped_plans(
        plans, shortlist_limit
    )
    shortlist_diagnostics["candidate_count_before"] = (
        candidate_count_before_shortlist
    )
    return {
        "enabled": True,
        "schema_version": SCHEMA_VERSION,
        "gold_free": True,
        "top_k": top_k,
        "semantic_snapshot_hash": final_state_hash(semantic_access_ids[:top_k]),
        "raw_semantic_snapshot_doc_ids": semantic_access_ids[:top_k],
        "raw_semantic_snapshot_scores": semantic_access_scores[:top_k],
        "ontology_physical_snapshot_hash": final_state_hash(
            ontology_physical_ids[:top_k]
        ),
        "ontology_physical_snapshot_doc_ids": ontology_physical_ids[:top_k],
        "ontology_physical_snapshot_scores": ontology_physical_scores[:top_k],
        "final_incumbent_hash": plans[PLAN_SEMANTIC]["state_hash"],
        "current_upstream_hash": final_state_hash(current_ids[:top_k]),
        "endpoint_contract_matched": (
            plans[PLAN_SEMANTIC]["state_hash"]
            == final_state_hash(current_ids[:top_k])
        ),
        "ontology_candidate_count": len(ontology_ids),
        "strongest_local_proof_path": strongest_path_ids,
        "strongest_bound_proof_path": strongest_bound_path_ids,
        "strongest_query_local_provenance_path": strongest_query_local_path_ids,
        "strongest_branch_complete_path": strongest_branch_complete_path_ids,
        "proof_ranked_incumbent_top5": proof_ranked_incumbent_top5,
        "rank_partitioned_top2_candidate_count": len(top2_pair_candidates),
        "connected_external_candidate_count": len(connected_external_ids),
        "rank_partitioned_top5_candidate_count": len(completion_candidates),
        "learned_residual_top5_candidate_count": len(
            residual_top5_candidates
        ),
        "ontology_warehouse_top5_candidate_count": len(
            warehouse_top5_candidates
        ),
        "topology_multi_access_slate_count": len(multi_access_slates),
        "query_grouped_shortlist": shortlist_diagnostics,
        "ontology_structural_rank5_candidate_available": bool(
            ontology_structural_candidate_id is not None
        ),
        "query_structure_signature": query_structure_signature,
        "query_local_pair_source": query_local_pair_source,
        "query_local_pair_terminal": query_local_pair_terminal,
        "query_local_pair_source_in_semantic_top2": (
            query_local_pair_source_in_semantic_top2
        ),
        "physical_plan_catalog": list(raw_plans),
        "plans": plans,
    }


class CompleteEvidencePlanPortfolioOptimizer:
    """Select one complete physical evidence plan using a calibrated model."""

    def __init__(
        self,
        model_path: str = "",
        *,
        top_k: int = 5,
        allowed_plan_families: Iterable[str] | None = None,
        selector_mode: str = "full_cbo",
    ) -> None:
        self.model_path = str(model_path or "")
        self.top_k = max(2, int(top_k))
        self.allowed_plan_families = {
            str(value)
            for value in (allowed_plan_families or [])
            if str(value) in ALTERNATIVE_PLANS or str(value) == PLAN_SEMANTIC
        }
        self.selector_mode = str(selector_mode or "full_cbo").strip().lower()
        if self.selector_mode not in {
            "full_cbo", "fixed", "utility_only", "random", "offline_oracle",
        }:
            raise ValueError(f"unsupported complete-plan selector: {selector_mode}")
        self.model: Dict[str, Any] = {}
        if self.model_path:
            path = Path(self.model_path).expanduser()
            if path.is_file():
                with path.open("r", encoding="utf-8") as handle:
                    value = json.load(handle)
                if isinstance(value, dict):
                    self.model = value

    def is_ready(self) -> bool:
        return bool(
            self.model.get("type") == MODEL_TYPE
            and self.model.get("feature_names")
            and self.model.get("plans")
            and (self.model.get("holdout_certificate", {}) or {}).get("passed", False)
        )

    def _predict(self, plan_name: str, features: Mapping[str, float]) -> Dict[str, float]:
        models = self.model.get("plans", {}) or {}
        objective_class = _plan_objective_class(features)
        objective_family_key = objective_family_model_key(
            objective_class, plan_name
        )
        structural_key = (
            STRUCTURAL_EXPERT_PREFIX + query_contract_regime(features)
        )
        objective_class_key = (
            OBJECTIVE_CLASS_EXPERT_PREFIX + objective_class
        )
        model = (
            models.get(objective_family_key, {})
            or models.get(structural_key, {})
            or models.get(GLOBAL_MODEL_KEY, {})
            or models.get(objective_class_key, {})
            or models.get(plan_name, {})
        )
        result: Dict[str, float] = {}
        for metric in ("delta_r2", "delta_r5", "loss_probability"):
            mean, std = predict_ensemble(model.get(f"{metric}_head", {}) or {}, features)
            result[metric] = mean
            result[f"{metric}_std"] = std
        gain_head = model.get("gain_probability_head", {}) or {}
        if gain_head:
            gain_mean, gain_std = predict_ensemble(gain_head, features)
        else:
            gain_mean = float(
                result["delta_r2"] > 1e-12 or result["delta_r5"] > 1e-12
            )
            gain_std = 0.0
        result["gain_probability"] = gain_mean
        result["gain_probability_std"] = gain_std
        result["delta_r2_lcb"] = (
            result["delta_r2"]
            + _safe_float(model.get("delta_r2_residual_lcb"), 0.0)
        )
        result["delta_r5_lcb"] = (
            result["delta_r5"]
            + _safe_float(model.get("delta_r5_residual_lcb"), 0.0)
        )
        result["utility_lcb"] = (
            2.0 * result["delta_r2_lcb"] + result["delta_r5_lcb"]
        )
        result["utility_mean"] = 2.0 * result["delta_r2"] + result["delta_r5"]
        result["safety_margin"] = (
            result["gain_probability"] - result["loss_probability"]
        )
        quality_heads = {
            metric: model.get(f"{metric}_head", {}) or {}
            for metric in (
                "quality_r2",
                "quality_r5",
                "optimal_r2_probability",
                "optimal_r5_probability",
            )
        }
        if all(quality_heads.values()):
            for metric, head in quality_heads.items():
                mean, std = predict_ensemble(head, features)
                if metric.startswith("optimal_"):
                    mean = min(1.0, max(0.0, mean))
                result[metric] = mean
                result[f"{metric}_std"] = std
            result["quality_r2_lcb"] = (
                result["quality_r2"]
                + _safe_float(model.get("quality_r2_residual_lcb"), 0.0)
            )
            result["quality_r5_lcb"] = (
                result["quality_r5"]
                + _safe_float(model.get("quality_r5_residual_lcb"), 0.0)
            )
            result["regret_r2_probability"] = (
                1.0 - result["optimal_r2_probability"]
            )
            result["regret_r5_probability"] = (
                1.0 - result["optimal_r5_probability"]
            )
        regret_heads = {
            metric: model.get(f"{metric}_head", {}) or {}
            for metric in ("regret_r2", "regret_r5")
        }
        if all(regret_heads.values()):
            for metric, head in regret_heads.items():
                mean, std = predict_ensemble(head, features)
                result[metric] = max(0.0, mean)
                result[f"{metric}_std"] = std
                result[f"{metric}_ucb"] = max(
                    0.0,
                    result[metric]
                    + _safe_float(model.get(f"{metric}_residual_ucb"), 0.0),
                )
        return result

    def _align_prediction_to_oof(
        self,
        prediction: Mapping[str, float],
        objective_class: str,
        plan_name: str,
    ) -> Dict[str, float]:
        alignment_root = self.model.get("grouped_oof_score_alignment", {}) or {}
        family_alignment = (
            (alignment_root.get("alignments", {}) or {})
            .get(objective_class, {})
            .get(plan_name, {})
        )
        fields = family_alignment.get("fields", {}) or {}
        if not fields:
            return dict(prediction)
        result = dict(prediction)
        result["raw_utility_mean"] = float(result.get("utility_mean", 0.0))
        result["raw_gain_probability"] = float(
            result.get("gain_probability", 0.0)
        )
        result["raw_loss_probability"] = float(
            result.get("loss_probability", 0.0)
        )
        for field, mapping in fields.items():
            if field not in result or not isinstance(mapping, Mapping):
                continue
            aligned = _piecewise_linear_score(float(result[field]), mapping)
            if field in {
                "gain_probability",
                "loss_probability",
                "optimal_r2_probability",
                "optimal_r5_probability",
                "regret_r2_probability",
                "regret_r5_probability",
            }:
                aligned = min(1.0, max(0.0, aligned))
            result[field] = aligned
        result["score_alignment_applied"] = True
        return result

    def optimize(
        self,
        current_doc_ids: Sequence[int],
        current_doc_scores: Sequence[float],
        query_plan_log: Dict[str, Any],
        *,
        shadow_only: bool = False,
    ) -> Tuple[List[int], List[float], Dict[str, Any]]:
        policy = self.model.get("policy", {}) or {}
        configured_plans = policy.get(
            "allowed_plans",
            self.allowed_plan_families or ALTERNATIVE_PLANS,
        ) or []
        allowed_plans = {
            str(value) for value in configured_plans
            if str(value) in ALTERNATIVE_PLANS or str(value) == PLAN_SEMANTIC
        }
        if not allowed_plans:
            allowed_plans = set(ALTERNATIVE_PLANS)
        raw_topology_plan_families = policy.get(
            "topology_plan_families", {}
        ) or {}
        topology_plan_families = {
            str(topology): [
                str(plan_name)
                for plan_name in plan_names or []
                if str(plan_name) in ALTERNATIVE_PLANS
            ]
            for topology, plan_names in raw_topology_plan_families.items()
            if isinstance(plan_names, Sequence) and not isinstance(plan_names, str)
        }
        materialization_families = set(allowed_plans)
        proof_slate = query_plan_log.get("full_aperture_proof_slate", {}) or {}
        signature = proof_slate.get("query_structure_signature", {}) or {}
        topology = str(signature.get("topology", "") or "")
        if topology_plan_families:
            materialization_families.intersection_update(
                topology_plan_families.get(topology, ())
            )
        portfolio = materialize_complete_plan_portfolio(
            query_plan_log,
            current_doc_ids,
            current_doc_scores,
            top_k=self.top_k,
            allowed_plan_families=(
                materialization_families
                if self.model or self.allowed_plan_families
                else None
            ),
            shortlist_per_family=(
                _safe_int(policy.get("query_grouped_shortlist_per_family"), 0)
                if "query_grouped_shortlist_per_family" in policy
                else None
            ),
            query_relative_semantic_competitor=(
                str(policy.get("selection_mode", "") or "")
                == "query_relative_slate_regret"
            ),
        )
        plans = portfolio["plans"]
        semantic_ids = list(plans[PLAN_SEMANTIC]["doc_ids"])
        selected_plan_instance = PLAN_SEMANTIC
        selected_plan_family = PLAN_SEMANTIC
        predictions: Dict[str, Any] = {}
        candidates: List[Tuple[float, ...]] = []
        candidates_by_objective: Dict[str, List[Tuple[float, ...]]] = {
            "top2_ordering": [],
            "top5_completion": [],
        }
        objective_class_policies = policy.get(
            "objective_class_policies", {}
        ) or {}
        hierarchical_objective_execution = bool(
            policy.get("hierarchical_objective_execution", False)
        )
        selection_mode = str(policy.get("selection_mode", "pareto_lcb") or "pareto_lcb")
        query_relative_slate_regret = (
            selection_mode == "query_relative_slate_regret"
        )
        max_loss_probability = _safe_float(
            policy.get("max_loss_probability"), 0.05
        )
        min_utility_lcb = _safe_float(policy.get("min_utility_lcb"), 0.0)
        min_utility_mean = _safe_float(policy.get("min_utility_mean"), 0.0)
        min_absolute_quality_lcb = _safe_float(
            policy.get("min_absolute_quality_lcb"), 0.0
        )
        min_regret_advantage = _safe_float(
            policy.get("min_regret_advantage"), 0.0
        )
        min_r2_lcb = _safe_float(policy.get("min_delta_r2_lcb"), 0.0)
        min_r5_lcb = _safe_float(policy.get("min_delta_r5_lcb"), 0.0)
        min_r2_mean = _safe_float(policy.get("min_delta_r2_mean"), 0.0)


        min_r5_mean = _safe_float(policy.get("min_delta_r5_mean"), float("-inf"))
        min_gain_probability = _safe_float(
            policy.get("min_gain_probability"), 0.0
        )
        allowed_characteristics = set(
            str(value)
            for value in policy.get("allowed_query_characteristics", []) or []
        )
        stage2_slate_dedup = bool(policy.get("stage2_slate_dedup", False))
        plan_instances = list(plans)
        slate_dedup_diagnostics: Dict[str, Any] = {
            "enabled": False,
            "candidate_count_before": len(plan_instances),
            "candidate_count_after": len(plan_instances),
            "duplicate_slate_group_count": 0,
            "dropped_alias_count": 0,
            "dropped_plan_instances": [],
        }
        if stage2_slate_dedup:
            plan_instances, slate_dedup_diagnostics = (
                deduplicate_top5_plan_instances(plans, plan_instances)
            )

        if self.is_ready():
            model_plans = self.model.get("plans", {}) or {}
            shared_outcome_model = bool(
                GLOBAL_MODEL_KEY in model_plans
                or any(
                    str(key).startswith(STRUCTURAL_EXPERT_PREFIX)
                    or str(key).startswith(OBJECTIVE_CLASS_EXPERT_PREFIX)
                    or str(key).startswith(OBJECTIVE_FAMILY_EXPERT_PREFIX)
                    for key in model_plans
                )
            )
            for plan_instance in plan_instances:
                plan = plans[plan_instance]
                plan_name = str(plan.get("plan_family") or plan_instance)
                plan_features = plan.get("features", {}) or {}
                if plan_name != PLAN_SEMANTIC and not plan_allowed_for_topology(
                    plan_name, plan_features, topology_plan_families
                ):
                    continue
                objective_class = _plan_objective_class(plan_features)
                class_policy = objective_class_policies.get(objective_class)
                if objective_class_policies:
                    if not isinstance(class_policy, Mapping) or not bool(
                        class_policy.get("calibration_feasible", False)
                    ):
                        continue
                    family_policies = class_policy.get(
                        "plan_family_policies", {}
                    ) or {}
                    if family_policies:
                        family_policy = family_policies.get(plan_name)
                        if not isinstance(family_policy, Mapping) or not bool(
                            family_policy.get("calibration_feasible", False)
                        ):
                            continue
                        nested_regime_policies = family_policy.get(
                            "structural_regime_policies", {}
                        ) or {}
                        if nested_regime_policies:
                            structural_regime = query_contract_regime(
                                plan_features
                            )
                            routed_policy = nested_regime_policies.get(
                                structural_regime
                            )
                            if not isinstance(routed_policy, Mapping) or not bool(
                                routed_policy.get("calibration_feasible", False)
                            ):
                                continue
                            plan_policy = routed_policy
                        else:
                            structural_regime = ""
                            plan_policy = family_policy
                    else:
                        regime_policies = class_policy.get(
                            "structural_regime_policies", {}
                        ) or {}
                        if regime_policies:
                            structural_regime = query_contract_regime(plan_features)
                            routed_policy = regime_policies.get(structural_regime)
                            if not isinstance(routed_policy, Mapping) or not bool(
                                routed_policy.get("calibration_feasible", False)
                            ):
                                continue
                            plan_policy = routed_policy
                        else:
                            structural_regime = ""
                            plan_policy = class_policy
                else:
                    structural_regime = ""
                    plan_policy = policy
                plan_selection_mode = str(
                    plan_policy.get("selection_mode", selection_mode)
                    or selection_mode
                )
                plan_max_loss_probability = _safe_float(
                    plan_policy.get("max_loss_probability"),
                    max_loss_probability,
                )
                plan_min_utility_lcb = _safe_float(
                    plan_policy.get("min_utility_lcb"), min_utility_lcb
                )
                plan_min_utility_mean = _safe_float(
                    plan_policy.get("min_utility_mean"), min_utility_mean
                )
                plan_min_absolute_quality_lcb = _safe_float(
                    plan_policy.get("min_absolute_quality_lcb"),
                    min_absolute_quality_lcb,
                )
                plan_min_r2_lcb = _safe_float(
                    plan_policy.get("min_delta_r2_lcb"), min_r2_lcb
                )
                plan_min_r5_lcb = _safe_float(
                    plan_policy.get("min_delta_r5_lcb"), min_r5_lcb
                )
                plan_min_r2_mean = _safe_float(
                    plan_policy.get("min_delta_r2_mean"), min_r2_mean
                )
                plan_min_r5_mean = _safe_float(
                    plan_policy.get("min_delta_r5_mean"), min_r5_mean
                )
                plan_min_gain_probability = _safe_float(
                    plan_policy.get("min_gain_probability"),
                    min_gain_probability,
                )
                plan_min_safety_margin = _safe_float(
                    plan_policy.get("min_safety_margin"), 0.0
                )
                plan_min_regret_advantage = _safe_float(
                    plan_policy.get("min_regret_advantage"), 0.0
                )
                plan_allowed_characteristics = set(
                    str(value)
                    for value in plan_policy.get(
                        "allowed_query_characteristics", []
                    ) or []
                )
                if (
                    plan_name not in allowed_plans
                    or
                    not plan.get("valid")
                    or (
                        not shared_outcome_model
                        and plan_name not in model_plans
                    )
                    or (
                        plan_allowed_characteristics
                        and _query_characteristic(plan_features)
                        not in plan_allowed_characteristics
                    )
                ):
                    continue
                prediction = self._predict(plan_name, plan_features)
                prediction = self._align_prediction_to_oof(
                    prediction, objective_class, plan_name
                )
                prediction["plan_family"] = plan_name
                prediction["plan_instance"] = plan_instance
                prediction["objective_class"] = objective_class
                prediction["query_contract_regime"] = structural_regime
                prediction["query_plan_topology"] = query_plan_topology(
                    plan_features
                )
                if plan_selection_mode == "query_relative_slate_regret":
                    has_regret_prediction = bool(
                        "regret_r2_ucb" in prediction
                        and "regret_r5_ucb" in prediction
                    )
                    prediction["regret_score"] = (
                        2.0 * prediction.get("regret_r2_ucb", 1.0)
                        + prediction.get("regret_r5_ucb", 1.0)
                    )
                    prediction["eligible"] = has_regret_prediction
                    if (
                        has_regret_prediction
                        and objective_class == "top2_ordering"
                        and plan_name not in {
                            PLAN_SEMANTIC, PLAN_SEMANTIC_TOP2_NOOP,
                        }
                    ):
                        comparator_instance = next((
                            value
                            for value, comparator in plans.items()
                            if str(
                                comparator.get("plan_family") or value
                            ) == PLAN_SEMANTIC_TOP2_NOOP
                            and comparator.get("valid")
                        ), "")
                        comparator_prediction: Dict[str, Any] = {}
                        if comparator_instance:
                            comparator_plan = plans[comparator_instance]
                            comparator_features = (
                                comparator_plan.get("features", {}) or {}
                            )
                            comparator_prediction = self._predict(
                                PLAN_SEMANTIC_TOP2_NOOP,
                                comparator_features,
                            )
                            comparator_prediction = self._align_prediction_to_oof(
                                comparator_prediction,
                                "top2_ordering",
                                PLAN_SEMANTIC_TOP2_NOOP,
                            )
                        if (
                            "regret_r2_ucb" not in comparator_prediction
                            or "regret_r5_ucb" not in comparator_prediction
                        ):
                            prediction["eligible"] = False
                            prediction["regret_advantage_over_noop"] = float("-inf")
                        else:
                            comparator_regret = _runtime_regret_score(
                                comparator_prediction, "top2_ordering"
                            )
                            plan_regret = _runtime_regret_score(
                                prediction, "top2_ordering"
                            )
                            advantage = comparator_regret - plan_regret
                            prediction["regret_advantage_over_noop"] = advantage
                            prediction["required_regret_advantage"] = (
                                plan_min_regret_advantage
                            )
                            prediction["eligible"] = bool(
                                advantage + 1e-12
                                >= plan_min_regret_advantage
                            )
                elif plan_selection_mode == "absolute_quality_regret":
                    metric = (
                        "r5" if objective_class == "top5_completion" else "r2"
                    )
                    prediction["eligible"] = bool(
                        prediction.get(f"quality_{metric}_lcb", -1.0)
                        > plan_min_absolute_quality_lcb
                        and prediction.get(f"regret_{metric}_probability", 1.0)
                        <= plan_max_loss_probability
                    )
                elif plan_selection_mode == "safe_gain_listwise":
                    prediction["eligible"] = bool(
                        prediction["loss_probability"]
                        <= plan_max_loss_probability
                        and prediction["gain_probability"]
                        >= plan_min_gain_probability
                        and prediction["safety_margin"]
                        > plan_min_safety_margin
                    )
                elif plan_selection_mode == "risk_budgeted_lexicographic":
                    prediction["eligible"] = bool(
                        prediction["loss_probability"]
                        <= plan_max_loss_probability
                        and prediction["gain_probability"]
                        >= plan_min_gain_probability
                        and prediction["delta_r2"] >= plan_min_r2_mean
                        and prediction["delta_r5"] >= plan_min_r5_mean
                        and prediction["utility_mean"] > plan_min_utility_mean
                    )
                else:
                    prediction["eligible"] = bool(
                        prediction["loss_probability"]
                        <= plan_max_loss_probability
                        and prediction["gain_probability"]
                        >= plan_min_gain_probability
                        and prediction["delta_r2_lcb"] >= plan_min_r2_lcb
                        and prediction["delta_r5_lcb"] >= plan_min_r5_lcb
                        and prediction["utility_lcb"] > plan_min_utility_lcb
                    )
                predictions[plan_instance] = prediction
                if prediction["eligible"]:
                    if plan_selection_mode == "query_relative_slate_regret":
                        proof_coherence = sum(_safe_float(
                            plan_features.get(name), 0.0
                        ) for name in (
                            "plan_completion_path_connected",
                            "plan_introduced_complete_proof_fraction",
                            "plan_introduced_relation_coverage_mean",
                            "residual_candidate_complete_provenance",
                            "residual_candidate_terminal_relation_match",
                        ))
                        displacement = (
                            2.0 * _safe_float(
                                plan_features.get("plan_top2_displacement_count"), 0.0
                            )
                            + _safe_float(
                                plan_features.get("plan_top5_displacement_count"), 0.0
                            )
                        )
                        candidate = (
                            -prediction["regret_r2_ucb"],
                            -prediction["regret_r5_ucb"],
                            proof_coherence,
                            -displacement,
                            -_safe_float(plan_features.get("estimated_cost_log"), 0.0),
                            float(plan_name in {
                                PLAN_SEMANTIC, PLAN_SEMANTIC_TOP2_NOOP,
                            }),
                            plan_instance,
                        )
                    elif plan_selection_mode == "absolute_quality_regret":
                        metric = (
                            "r5" if objective_class == "top5_completion" else "r2"
                        )
                        other_metric = "r2" if metric == "r5" else "r5"
                        candidate = (
                            prediction[f"quality_{metric}_lcb"],
                            prediction[f"optimal_{metric}_probability"],
                            -prediction[f"regret_{metric}_probability"],
                            prediction[f"quality_{other_metric}_lcb"],
                            plan_instance,
                        )
                    elif plan_selection_mode == "safe_gain_listwise":
                        candidate = (
                            prediction["safety_margin"],
                            prediction["gain_probability"],
                            -prediction["loss_probability"],
                            prediction["delta_r2"],
                            prediction["delta_r5"],
                            plan_instance,
                        )
                    elif plan_selection_mode == "risk_budgeted_lexicographic":
                        if objective_class == "top5_completion":
                            candidate = (
                                prediction["delta_r5"],
                                prediction["utility_mean"],
                                prediction["gain_probability"],
                                -prediction["loss_probability"],
                                prediction["delta_r2"],
                                plan_instance,
                            )
                        else:
                            candidate = (
                                prediction["delta_r2"],
                                prediction["delta_r5"],
                                prediction["utility_mean"],
                                prediction["gain_probability"],
                                -prediction["loss_probability"],
                                plan_instance,
                            )
                    else:
                        if objective_class == "top5_completion":
                            candidate = (
                                prediction["delta_r5_lcb"],
                                prediction["utility_lcb"],
                                prediction["delta_r2_lcb"],
                                plan_instance,
                            )
                        else:
                            candidate = (
                                prediction["utility_lcb"],
                                prediction["delta_r2_lcb"],
                                prediction["delta_r5_lcb"],
                                plan_instance,
                            )
                    candidates.append(candidate)
                    candidates_by_objective[objective_class].append(candidate)
            if candidates and not hierarchical_objective_execution:
                selected_plan_instance = max(candidates)[-1]
                if (
                    query_relative_slate_regret
                    and selected_plan_instance != PLAN_SEMANTIC
                    and PLAN_SEMANTIC in predictions
                ):
                    semantic_regret = _safe_float(
                        predictions[PLAN_SEMANTIC].get("regret_score"), 3.0
                    )
                    selected_regret = _safe_float(
                        predictions[selected_plan_instance].get("regret_score"),
                        3.0,
                    )
                    if (
                        semantic_regret - selected_regret + 1e-12
                        < min_regret_advantage
                    ):
                        selected_plan_instance = PLAN_SEMANTIC
                selected_plan_family = str(
                    plans[selected_plan_instance].get("plan_family")
                    or selected_plan_instance
                )

        selector_override = self.selector_mode != "full_cbo"
        if selector_override:
            valid_instances = [
                name for name in plan_instances
                if bool((plans.get(name, {}) or {}).get("valid", False))
            ]
            if not valid_instances:
                valid_instances = [PLAN_SEMANTIC]
            if self.selector_mode == "fixed":
                fixed_preference = (
                    PLAN_ONTOLOGY_PHYSICAL,
                    PLAN_STRUCTURAL_EXECUTION,
                    PLAN_SEMANTIC,
                )
                selected_plan_instance = next(
                    (
                        name for family in fixed_preference
                        for name in valid_instances
                        if str((plans.get(name, {}) or {}).get("plan_family") or name) == family
                    ),
                    sorted(valid_instances)[0],
                )
            elif self.selector_mode == "utility_only":
                scored = [
                    (float((predictions.get(name, {}) or {}).get("utility_mean", float("-inf"))), name)
                    for name in valid_instances
                    if name in predictions
                ]
                selected_plan_instance = max(scored)[1] if scored else PLAN_SEMANTIC
            elif self.selector_mode == "random":
                query_key = str(query_plan_log.get("query") or query_plan_log.get("query_idx") or "")
                digest = hashlib.sha256(query_key.encode("utf-8")).digest()
                selected_plan_instance = sorted(valid_instances)[
                    int.from_bytes(digest[:8], "big") % len(valid_instances)
                ]
            else:
                labels = (
                    (query_plan_log.get("complete_evidence_plan_training", {}) or {})
                    .get("plans", {})
                    or {}
                )
                scored = []
                for name in valid_instances:
                    label = labels.get(name, {}) or {}
                    if not label:
                        continue
                    cost = float(
                        ((plans.get(name, {}) or {}).get("features", {}) or {})
                        .get("estimated_cost_log", 0.0)
                        or 0.0
                    )
                    scored.append((
                        2.0 * float(label.get("recall_at2", 0.0) or 0.0)
                        + float(label.get("recall_at5", 0.0) or 0.0),
                        -cost,
                        name,
                    ))
                selected_plan_instance = max(scored)[-1] if scored else PLAN_SEMANTIC
            selected_plan_family = str(
                (plans.get(selected_plan_instance, {}) or {}).get("plan_family")
                or selected_plan_instance
            )
            hierarchical_objective_execution = False

        stage1_plan_instance = PLAN_SEMANTIC
        stage1_plan_family = PLAN_SEMANTIC
        stage2_plan_instance = ""
        stage2_plan_family = ""
        stage2_admission = {
            "enabled": False,
            "selected_lane": "legacy",
        }
        stage1_topk = list(semantic_ids[: self.top_k])
        selected_topk = list(stage1_topk)
        if self.is_ready() and hierarchical_objective_execution:
            if candidates_by_objective["top2_ordering"]:
                stage1_plan_instance = max(
                    candidates_by_objective["top2_ordering"]
                )[-1]
                stage1_plan_family = str(
                    plans[stage1_plan_instance].get("plan_family")
                    or stage1_plan_instance
                )
                stage1_topk = _ids([
                    *plans[stage1_plan_instance]["doc_ids"][:2],
                    *semantic_ids,
                ])[: self.top_k]
            if candidates_by_objective["top5_completion"]:
                top5_policy = objective_class_policies.get(
                    "top5_completion", {}
                ) or {}
                top5_candidate_instances = [
                    value[-1]
                    for value in candidates_by_objective["top5_completion"]
                ]
                if (
                    query_relative_slate_regret
                    and bool(top5_policy.get(
                        "proof_connected_primary_admission", False
                    ))
                ):
                    stage2_plan_instance, stage2_admission = (
                        select_proof_connected_stage2_candidate(
                            top5_candidate_instances,
                            predictions,
                            plans,
                            rescue_advantage=_safe_float(
                                top5_policy.get(
                                    "min_unconnected_regret_advantage"
                                ),
                                float("inf"),
                            ),
                        )
                    )
                else:
                    stage2_plan_instance = max(
                        candidates_by_objective["top5_completion"]
                    )[-1]
                if (
                    query_relative_slate_regret
                    and stage2_plan_instance
                    and stage2_plan_instance != PLAN_SEMANTIC
                    and PLAN_SEMANTIC in predictions
                ):
                    semantic_regret = _runtime_regret_score(
                        predictions[PLAN_SEMANTIC], "top5_completion"
                    )
                    selected_regret = _runtime_regret_score(
                        predictions[stage2_plan_instance], "top5_completion"
                    )
                    class_min_advantage = _safe_float(
                        top5_policy.get("min_regret_advantage"), 0.0
                    )
                    if (
                        semantic_regret - selected_regret + 1e-12
                        < class_min_advantage
                    ):
                        stage2_plan_instance = PLAN_SEMANTIC
                        stage2_admission = {
                            **stage2_admission,
                            "selected_lane": "semantic_noop_guard",
                            "semantic_advantage": semantic_regret - selected_regret,
                            "required_semantic_advantage": class_min_advantage,
                        }
                stage2_plan_family = str(
                    plans[stage2_plan_instance].get("plan_family")
                    or stage2_plan_instance
                )
                selected_topk = compose_hierarchical_topk(
                    stage1_topk,
                    plans[stage2_plan_instance]["doc_ids"],
                    semantic_ids,
                    top_k=self.top_k,
                )
            else:
                selected_topk = list(stage1_topk)
            selected_plan_instance = (
                stage2_plan_instance or stage1_plan_instance
            )
            selected_plan_family = stage2_plan_family or stage1_plan_family

        ready = self.is_ready()
        endpoint_contract_matched = bool(
            portfolio.get("endpoint_contract_matched", False)
        )
        preserve_upstream = bool(
            shadow_only or not ready or not endpoint_contract_matched
        )
        execution_plan = (
            "upstream_shadow" if shadow_only
            else "upstream_model_unavailable" if not ready
            else "upstream_endpoint_contract_mismatch"
            if not endpoint_contract_matched
            else selected_plan_family
        )
        if preserve_upstream:
            selected_topk = _ids(current_doc_ids)[: self.top_k]
        elif not hierarchical_objective_execution:
            selected_topk = list(plans[selected_plan_instance]["doc_ids"])
        changed = selected_topk[: self.top_k] != semantic_ids[: self.top_k]
        committed = bool(not preserve_upstream and changed)
        if preserve_upstream:
            tail = _ids(current_doc_ids)
            output_scores = [
                _safe_float(value) for value in current_doc_scores[: len(tail)]
            ]
        else:
            tail = _ids([
                *selected_topk,
                *current_doc_ids,
                *(
                    (query_plan_log.get("semantic_plan_incumbent", {}) or {}).get(
                        "candidate_window_doc_ids", []
                    )
                ),
            ])
            output_scores = _monotone_scores(
                tail,
                [
                    (list(current_doc_ids), list(current_doc_scores)),
                    (
                        (query_plan_log.get("semantic_plan_incumbent", {}) or {}).get(
                            "candidate_window_doc_ids", []
                        ),
                        (query_plan_log.get("semantic_plan_incumbent", {}) or {}).get(
                            "candidate_window_doc_scores", []
                        ),
                    ),
                    (
                        (query_plan_log.get(
                            "ontology_physical_plan_incumbent", {}
                        ) or {}).get("candidate_window_doc_ids", []),
                        (query_plan_log.get(
                            "ontology_physical_plan_incumbent", {}
                        ) or {}).get("candidate_window_doc_scores", []),
                    ),
                ],
            )
        log = {
            **portfolio,
            "component": "complete_evidence_plan_portfolio_optimizer",
            "model_path": self.model_path,
            "model_ready": self.is_ready(),
            "shadow_only": bool(shadow_only),
            "predictions": predictions,
            "allowed_plans": sorted(allowed_plans),
            "allowed_query_characteristics": sorted(allowed_characteristics),
            "selection_mode": selection_mode,
            "selector_mode": self.selector_mode,
            "selector_override": selector_override,
            "hierarchical_objective_execution": hierarchical_objective_execution,
            "objective_class_policies": objective_class_policies,
            "topology_plan_families": topology_plan_families,
            "materialized_plan_families": sorted(materialization_families),
            "stage2_slate_dedup": slate_dedup_diagnostics,
            "admission_thresholds": {
                "max_loss_probability": max_loss_probability,
                "max_regret_probability": max_loss_probability,
                "min_absolute_quality_lcb": min_absolute_quality_lcb,
                "min_utility_mean": min_utility_mean,
                "min_delta_r2_mean": min_r2_mean,
                "min_delta_r5_mean": min_r5_mean,
                "min_gain_probability": min_gain_probability,
                "min_safety_margin": _safe_float(
                    policy.get("min_safety_margin"), 0.0
                ),
            },
            "selected_plan": execution_plan,
            "selected_plan_instance": (
                execution_plan if preserve_upstream else selected_plan_instance
            ),
            "proposed_plan": selected_plan_family,
            "proposed_plan_instance": selected_plan_instance,
            "no_op": preserve_upstream or selected_topk == semantic_ids[: self.top_k],
            "selection_contract": (
                "hierarchical_top2_then_top5_completion"
                if hierarchical_objective_execution
                else "whole_topk_physical_plan"
            ),
            "endpoint_contract": "exact_final_ranking_incumbent",
            "endpoint_contract_matched": endpoint_contract_matched,
            "upstream_ranker_preserved": False,
            "hierarchical_execution": {
                "stage1_plan": stage1_plan_family,
                "stage1_plan_instance": stage1_plan_instance,
                "stage1_top5": stage1_topk,
                "stage2_plan": stage2_plan_family,
                "stage2_plan_instance": stage2_plan_instance,
                "stage2_top5": selected_topk,
                "stage2_admission": stage2_admission,
                "top2_frozen_during_stage2": bool(
                    hierarchical_objective_execution
                ),
            },
            "atomic_execution": {
                "executed": committed,
                "action_kind": (
                    "select_top2_then_complete_top5"
                    if hierarchical_objective_execution
                    else "select_complete_top5_slate"
                ),
                "selected_plan": execution_plan,
                "selected_plan_instance": (
                    execution_plan
                    if preserve_upstream
                    else selected_plan_instance
                ),
                "before_top5": list(semantic_ids[: self.top_k]),
                "after_top5": list(selected_topk[: self.top_k]),
            },
        }
        return tail, output_scores, log
