"""Proof-carrying nested top-k evidence optimizer.

The compiler exposes two disjoint physical action spaces over one final
retrieval slate:

* membership actions replace rank 5 while preserving ranks 1-2 exactly;
* ordering actions choose a top-2 subset from the resulting top 5 while
  preserving top-5 membership exactly.

Every action carries a passage-keyed provenance profile derived from the
unified evidence relation. Gold labels are attached only by offline replay.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

try:
    from . import query_plan_content_complete_set_comparator as content_complete
    from . import query_plan_end_to_end_slate_optimizer as end_to_end
    from . import query_plan_rank_partitioned_set_optimizer as rank_partitioned
except ImportError:
    import query_plan_content_complete_set_comparator as content_complete
    import query_plan_end_to_end_slate_optimizer as end_to_end
    import query_plan_rank_partitioned_set_optimizer as rank_partitioned


MODEL_TYPE = "proof_carrying_nested_topk_optimizer"
STAGE_MODEL_TYPE = "proof_carrying_nested_topk_stage_optimizer"
FEATURE_ROUTED_STAGE_MODEL_TYPE = (
    "feature_routed_residual_opportunity_stage_optimizer"
)
ONTOLOGY_SIGNATURE_STAGE_MODEL_TYPE = (
    "ontology_signature_constrained_stage_optimizer"
)
NESTED_TOPK_MODES = {
    "proof_nested_topk",
    "proof_nested_top5_only",
    "proof_nested_top2_only",
    "proof_full_aperture_nested_topk",
}

V225_PROOF_FEATURE_NAMES = [
    "nested_stage_top5_membership",
    "nested_stage_top2_ordering",
    "top2_membership_invariant",
    "selected_real_proof_fraction",
    "selected_derived_proof_fraction",
    "selected_proof_certificate_strength_mean",
    "selected_proof_obligation_coverage_mean",
    "selected_proof_obligation_coverage_min",
    "selected_branch_coverage_mean",
    "selected_plan_identity_fraction",
    "selected_exact_entity_key_fraction",
    "selected_binding_fraction",
    "selected_provenance_source_count_log",
    "selected_provenance_agreement",
    "displaced_real_proof_fraction",
    "displaced_derived_proof_fraction",
    "displaced_proof_obligation_coverage_mean",
    "selected_minus_displaced_proof_coverage",
    "proof_catalog_coverage_fraction",
    "proof_catalog_passage_count_log",
    "selected_full_aperture_fraction",
    "selected_aperture_anchor_coverage_mean",
    "selected_aperture_query_coverage_mean",
    "selected_aperture_answer_type_score_mean",
    "selected_aperture_terminal_relation_fraction",
    "selected_aperture_relation_slot_coverage_mean",
    "displaced_aperture_relation_slot_coverage_mean",
    "selected_minus_displaced_aperture_slot_coverage",
    "selected_aperture_structured_anchor_coverage_mean",
    "selected_aperture_structured_triple_count_log_mean",
    "result_slate_aperture_relation_slot_coverage",
    "result_slate_aperture_binding_join_count_log",
    "result_slate_aperture_connected_passage_fraction",
    "selected_aperture_local_proof_path_fraction",
    "selected_aperture_local_path_score_mean",
    "result_slate_aperture_shared_path_count_log",
    "proof_gap_full_set_action",
    "proof_gap_member_count_log",
    "proof_gap_external_member_fraction",
    "proof_gap_before_coverage",
    "proof_gap_coverage_gain",
    "proof_gap_confidence",
    "proof_gap_probe_cost_log",
]

FULL_APERTURE_FEATURE_NAMES = {
    "selected_full_aperture_fraction",
    "selected_aperture_anchor_coverage_mean",
    "selected_aperture_query_coverage_mean",
    "selected_aperture_answer_type_score_mean",
    "selected_aperture_terminal_relation_fraction",
    "selected_aperture_relation_slot_coverage_mean",
    "displaced_aperture_relation_slot_coverage_mean",
    "selected_minus_displaced_aperture_slot_coverage",
    "selected_aperture_structured_anchor_coverage_mean",
    "selected_aperture_structured_triple_count_log_mean",
    "result_slate_aperture_relation_slot_coverage",
    "result_slate_aperture_binding_join_count_log",
    "result_slate_aperture_connected_passage_fraction",
    "selected_aperture_local_proof_path_fraction",
    "selected_aperture_local_path_score_mean",
    "result_slate_aperture_shared_path_count_log",
}


def feature_names(base_module) -> List[str]:
    return [
        *rank_partitioned.feature_names(base_module),
        *V225_PROOF_FEATURE_NAMES,
    ]


def proof_feature_names() -> Set[str]:
    return {
        *V225_PROOF_FEATURE_NAMES,
        "selected_proof_fraction",
        "selected_role_consistent_fraction",
        "selected_complete_provenance_fraction",
        "selected_relation_coverage_mean",
        "selected_answer_type_compatible_fraction",
    }


def full_aperture_feature_names() -> Set[str]:
    return set(FULL_APERTURE_FEATURE_NAMES)


def content_feature_names(base_module) -> Set[str]:
    return rank_partitioned.content_feature_names(base_module)


def is_nested_topk_mode(candidate_mode: str) -> bool:
    return str(candidate_mode or "").strip().lower() in NESTED_TOPK_MODES


def stages_for_mode(candidate_mode: str) -> Tuple[str, ...]:
    mode = str(candidate_mode or "").strip().lower()
    if mode == "proof_nested_top5_only":
        return ("top5",)
    if mode == "proof_nested_top2_only":
        return ("top2",)
    return (
        ("top5", "top2")
        if mode in {"proof_nested_topk", "proof_full_aperture_nested_topk"}
        else ()
    )


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        return default if math.isnan(number) or math.isinf(number) else number
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _mean(values: Iterable[float]) -> float:
    rows = list(values)
    return sum(rows) / len(rows) if rows else 0.0


def _context_rows(candidate: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    return [
        value for value in candidate.get("contexts", []) or []
        if isinstance(value, Mapping)
    ]


def _truthy(candidate: Mapping[str, Any], *names: str) -> bool:
    if any(bool(candidate.get(name)) for name in names):
        return True
    return any(
        any(bool(context.get(name)) for name in names)
        for context in _context_rows(candidate)
    )


def _real_certificate(candidate: Mapping[str, Any]) -> bool:
    if bool(candidate.get("proof_valid")) and bool(candidate.get("proof_carrying")):
        return True
    for name in ("proof_certificate", "typed_proof"):
        value = candidate.get(name)
        if not value:
            continue
        if isinstance(value, Mapping) and value and all(
            str(key).endswith("_enumeration_only") for key in value
        ):
            continue
        return True
    return any(
        bool(context.get("proof") or context.get("certificate") or context.get("typed_proof"))
        for context in _context_rows(candidate)
    )


def _logical_contract(query_plan_log: Mapping[str, Any]) -> Mapping[str, Any]:
    factorized = query_plan_log.get("factorized_query_executor", {}) or {}
    logical = factorized.get("logical_evidence_plan", {}) or {}
    return logical if isinstance(logical, Mapping) else {}


def _proof_obligations(query_plan_log: Mapping[str, Any]) -> List[str]:
    obligations = _logical_contract(query_plan_log).get("proof_obligations", []) or []
    return sorted({str(value).strip().lower() for value in obligations if str(value).strip()})


def _candidate_relation_coverage(candidate: Mapping[str, Any]) -> float:
    values = [
        _safe_float(candidate.get("relation_coverage_score")),
        _safe_float(candidate.get("query_relation_coverage")),
        _safe_float(candidate.get("query_predicate_coverage")),
    ]
    for context in _context_rows(candidate):
        values.extend([
            _safe_float(context.get("query_relation_coverage")),
            float(bool(context.get("relations") or context.get("relation_path"))),
        ])
    values.append(float(bool(candidate.get("relation") or candidate.get("relations"))))
    return min(max(values, default=0.0), 1.0)


def _candidate_obligations(
    candidate: Mapping[str, Any], required: Sequence[str]
) -> Set[str]:
    real = _real_certificate(candidate)
    complete = _truthy(candidate, "complete_provenance", "provenance")
    relation = _candidate_relation_coverage(candidate) > 0.0
    join = _truthy(
        candidate,
        "exact_entity_key",
        "binding",
        "support_key",
        "group_id",
    )
    typed = _truthy(candidate, "typed_proof", "answer_type")
    satisfied: Set[str] = set()
    for obligation in required:
        normalized = obligation.replace("-", "_").replace(" ", "_")
        if "provenance" in normalized and (complete or real):
            satisfied.add(obligation)
        elif ("predicate" in normalized or "relation" in normalized) and relation:
            satisfied.add(obligation)
        elif ("join" in normalized or "connect" in normalized) and join:
            satisfied.add(obligation)
        elif ("type" in normalized or "answer" in normalized) and typed:
            satisfied.add(obligation)
        elif real:
            satisfied.add(obligation)
    return satisfied


def materialize_passage_proof_catalog(
    query_plan_log: Dict[str, Any], *, base_module
) -> Dict[int, Dict[str, Any]]:
    """Build a gold-free, passage-keyed catalog from all evidence operators."""
    cached = query_plan_log.get("_v225_passage_proof_catalog")
    if isinstance(cached, dict):
        return cached
    grouped = end_to_end.group_unified_candidate_refs(
        query_plan_log,
        base_module=base_module,
        candidate_mode="unified",
    )
    required = _proof_obligations(query_plan_log)
    required_branches = max(
        _safe_int(_logical_contract(query_plan_log).get("required_branch_count"), 0),
        0,
    )
    catalog: Dict[int, Dict[str, Any]] = {}
    for passage_idx, refs in grouped.items():
        candidates = [candidate for _, candidate in refs]
        sources = sorted({str(source).strip().lower() for source, _ in refs if str(source).strip()})
        real = any(_real_certificate(candidate) for candidate in candidates)
        role = any(_truthy(candidate, "role_consistent") for candidate in candidates)
        complete = any(
            _truthy(candidate, "complete_provenance", "provenance")
            for candidate in candidates
        )
        exact_key = any(_truthy(candidate, "exact_entity_key") for candidate in candidates)
        binding = any(_truthy(candidate, "binding", "support_key") for candidate in candidates)
        derived = any(
            (
                _truthy(candidate, "has_proof_signal") and role and complete
            )
            or (
                bool(candidate.get("triple")) and _truthy(candidate, "exact_entity_key")
            )
            for candidate in candidates
        )
        satisfied = set().union(*(
            _candidate_obligations(candidate, required) for candidate in candidates
        )) if candidates else set()
        branch_ids = {
            str(value)
            for candidate in candidates
            for value in [candidate.get("branch_idx")]
            if value is not None
        }
        plan_ids = {
            str(candidate.get("plan_id") or context.get("plan_id") or "")
            for candidate in candidates
            for context in [*(_context_rows(candidate) or [{}])]
            if str(candidate.get("plan_id") or context.get("plan_id") or "")
        }
        relation_coverage = max(
            (_candidate_relation_coverage(candidate) for candidate in candidates),
            default=0.0,
        )
        answer_type = any(_truthy(candidate, "answer_type") for candidate in candidates)
        original_ranks = [
            _safe_int(candidate.get("original_rank"), 10**9)
            for candidate in candidates
            if _safe_int(candidate.get("original_rank"), 10**9) > 0
        ]
        obligation_coverage = (
            len(satisfied) / len(required) if required else float(real or derived)
        )
        branch_coverage = (
            min(len(branch_ids) / required_branches, 1.0)
            if required_branches > 0 else float(bool(branch_ids))
        )
        aperture_candidates = [
            candidate
            for source, candidate in refs
            if str(source).strip().lower() == "full_aperture"
        ]
        proof_gap_candidates = [
            candidate
            for _, candidate in refs
            if str(candidate.get("operator") or "")
            == "proof_gap_bound_access_path"
            and bool(candidate.get("proof_valid"))
        ]
        proof_gap_slot_count = max(
            (
                _safe_int(candidate.get("proof_relation_slot_count"), 0)
                for candidate in proof_gap_candidates
            ),
            default=0,
        )
        aperture_slot_count = max(
            (
                len(candidate.get("relation_slot_hits", []) or [])
                for candidate in aperture_candidates
            ),
            default=0,
        )
        combined_slot_count = max(aperture_slot_count, proof_gap_slot_count)
        catalog[passage_idx] = {
            "passage_idx": passage_idx,
            "sources": sources,
            "source_count": len(sources),
            "real_proof": real,
            "derived_proof": derived,
            "certificate_strength": 1.0 if real else 0.65 if derived else 0.0,
            "role_consistent": role,
            "complete_provenance": complete,
            "exact_entity_key": exact_key,
            "binding": binding,
            "relation_coverage": relation_coverage,
            "answer_type_compatible": answer_type,
            "original_rank": min(original_ranks, default=10**9),
            "semantic_score": max(
                (
                    _safe_float(candidate.get("semantic_score"))
                    for candidate in candidates
                ),
                default=0.0,
            ),
            "required_obligations": required,
            "satisfied_obligations": sorted(satisfied),
            "obligation_coverage": obligation_coverage,
            "branch_coverage": branch_coverage,
            "branch_ids": sorted(branch_ids),
            "plan_ids": sorted(plan_ids),
            "full_aperture": bool(aperture_candidates),
            "aperture_anchor_coverage": max(
                (
                    _safe_float(candidate.get("anchor_coverage"))
                    for candidate in aperture_candidates
                ),
                default=0.0,
            ),
            "aperture_query_coverage": max(
                (
                    _safe_float(candidate.get("query_coverage"))
                    for candidate in aperture_candidates
                ),
                default=0.0,
            ),
            "aperture_answer_type_score": max(
                (
                    _safe_float(candidate.get("answer_type_score"))
                    for candidate in aperture_candidates
                ),
                default=0.0,
            ),
            "aperture_terminal_relation": any(
                bool(candidate.get("terminal_relation_match"))
                for candidate in [*aperture_candidates, *proof_gap_candidates]
            ),
            "aperture_relation_slot_coverage": max(
                (
                    _safe_float(candidate.get("relation_coverage_score"))
                    for candidate in aperture_candidates
                ),
                default=0.0,
            ),
            "aperture_structured_anchor_coverage": max(
                (
                    _safe_float(candidate.get("structured_anchor_coverage"))
                    for candidate in aperture_candidates
                ),
                default=0.0,
            ),
            "aperture_structured_triple_count_log": max(
                (
                    _safe_float(candidate.get("structured_triple_count_log"))
                    for candidate in aperture_candidates
                ),
                default=0.0,
            ),
            "aperture_relation_slot_hits": [
                int(any(
                    index < len(candidate.get("relation_slot_hits", []) or [])
                    and bool((candidate.get("relation_slot_hits", []) or [])[index])
                    for candidate in aperture_candidates
                ) or any(
                    _safe_int(candidate.get("proof_relation_slot_index")) == index
                    for candidate in proof_gap_candidates
                ))
                for index in range(combined_slot_count)
            ],
            "aperture_binding_entities": sorted({
                str(value)
                for candidate in [*aperture_candidates, *proof_gap_candidates]
                for value in [
                    *(candidate.get("binding_entities", []) or []),
                    *(candidate.get("proof_binding_entities", []) or []),
                ]
                if str(value)
            }),
            "aperture_local_proof_path": any(
                bool(candidate.get("local_proof_path_complete"))
                for candidate in aperture_candidates
            ) or bool(proof_gap_candidates),
            "aperture_local_path_score": max(
                [
                    _safe_float(candidate.get("local_proof_path_min_relation_score"))
                    for candidate in aperture_candidates
                ] + [
                    _safe_float(candidate.get("proof_confidence"))
                    for candidate in proof_gap_candidates
                ],
                default=0.0,
            ),
            "aperture_local_path_ids": sorted({
                str(value)
                for candidate in [*aperture_candidates, *proof_gap_candidates]
                for value in [
                    *(candidate.get("local_proof_path_ids", []) or []),
                    candidate.get("group_id"),
                ]
                if str(value)
            }),
            "query_required_clause_count": max(
                (
                    _safe_int(candidate.get("query_required_clause_count"), 1)
                    for candidate in proof_gap_candidates
                ),
                default=1,
            ),
            "query_clause_coverage": max(
                (
                    _safe_float(candidate.get("query_clause_coverage"))
                    for candidate in proof_gap_candidates
                ),
                default=0.0,
            ),
            "query_clause_complete": any(
                bool(candidate.get("query_clause_complete", False))
                for candidate in proof_gap_candidates
            ),
        }
    query_plan_log["_v225_passage_proof_catalog"] = catalog
    return catalog


def _aggregate_proof_features(
    selected_ids: Sequence[int],
    displaced_ids: Sequence[int],
    catalog: Mapping[int, Mapping[str, Any]],
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    selected = [catalog.get(value, {}) or {} for value in selected_ids]
    displaced = [catalog.get(value, {}) or {} for value in displaced_ids]

    def fraction(rows: Sequence[Mapping[str, Any]], name: str) -> float:
        return _mean(float(bool(row.get(name))) for row in rows)

    def average(rows: Sequence[Mapping[str, Any]], name: str) -> float:
        return _mean(_safe_float(row.get(name)) for row in rows)

    source_union = {
        str(source)
        for row in selected
        for source in row.get("sources", []) or []
    }
    selected_coverage = average(selected, "obligation_coverage")
    displaced_coverage = average(displaced, "obligation_coverage")
    selected_aperture_coverage = average(
        selected, "aperture_relation_slot_coverage"
    )
    displaced_aperture_coverage = average(
        displaced, "aperture_relation_slot_coverage"
    )
    catalog_covered = sum(bool(catalog.get(value)) for value in selected_ids)
    features = {
        "selected_ontology_backed_fraction": catalog_covered / max(len(selected_ids), 1),
        "selected_ontology_source_count_log": math.log1p(len(source_union)),
        "selected_ontology_source_agreement": min(len(source_union) / 4.0, 1.0),
        "selected_proof_fraction": fraction(selected, "real_proof")
        + 0.65 * fraction(selected, "derived_proof"),
        "selected_role_consistent_fraction": fraction(selected, "role_consistent"),
        "selected_complete_provenance_fraction": fraction(selected, "complete_provenance"),
        "selected_relation_coverage_mean": average(selected, "relation_coverage"),
        "selected_answer_type_compatible_fraction": fraction(
            selected, "answer_type_compatible"
        ),
        "selected_real_proof_fraction": fraction(selected, "real_proof"),
        "selected_derived_proof_fraction": fraction(selected, "derived_proof"),
        "selected_proof_certificate_strength_mean": average(
            selected, "certificate_strength"
        ),
        "selected_proof_obligation_coverage_mean": selected_coverage,
        "selected_proof_obligation_coverage_min": min(
            (_safe_float(row.get("obligation_coverage")) for row in selected),
            default=0.0,
        ),
        "selected_branch_coverage_mean": average(selected, "branch_coverage"),
        "selected_plan_identity_fraction": _mean(
            float(bool(row.get("plan_ids"))) for row in selected
        ),
        "selected_exact_entity_key_fraction": fraction(selected, "exact_entity_key"),
        "selected_binding_fraction": fraction(selected, "binding"),
        "selected_provenance_source_count_log": math.log1p(len(source_union)),
        "selected_provenance_agreement": min(len(source_union) / 4.0, 1.0),
        "displaced_real_proof_fraction": fraction(displaced, "real_proof"),
        "displaced_derived_proof_fraction": fraction(displaced, "derived_proof"),
        "displaced_proof_obligation_coverage_mean": displaced_coverage,
        "selected_minus_displaced_proof_coverage": (
            selected_coverage - displaced_coverage
        ),
        "proof_catalog_coverage_fraction": catalog_covered / max(len(selected_ids), 1),
        "proof_catalog_passage_count_log": math.log1p(len(catalog)),
        "selected_full_aperture_fraction": fraction(selected, "full_aperture"),
        "selected_aperture_anchor_coverage_mean": average(
            selected, "aperture_anchor_coverage"
        ),
        "selected_aperture_query_coverage_mean": average(
            selected, "aperture_query_coverage"
        ),
        "selected_aperture_answer_type_score_mean": average(
            selected, "aperture_answer_type_score"
        ),
        "selected_aperture_terminal_relation_fraction": fraction(
            selected, "aperture_terminal_relation"
        ),
        "selected_aperture_relation_slot_coverage_mean": selected_aperture_coverage,
        "displaced_aperture_relation_slot_coverage_mean": displaced_aperture_coverage,
        "selected_minus_displaced_aperture_slot_coverage": (
            selected_aperture_coverage - displaced_aperture_coverage
        ),
        "selected_aperture_structured_anchor_coverage_mean": average(
            selected, "aperture_structured_anchor_coverage"
        ),
        "selected_aperture_structured_triple_count_log_mean": average(
            selected, "aperture_structured_triple_count_log"
        ),
        "selected_aperture_local_proof_path_fraction": fraction(
            selected, "aperture_local_proof_path"
        ),
        "selected_aperture_local_path_score_mean": average(
            selected, "aperture_local_path_score"
        ),
    }
    contract = {
        "selected": [dict(catalog.get(value, {}) or {"passage_idx": value}) for value in selected_ids],
        "displaced": [dict(catalog.get(value, {}) or {"passage_idx": value}) for value in displaced_ids],
    }
    return features, contract


def _aperture_result_set_features(
    passage_ids: Sequence[int],
    catalog: Mapping[int, Mapping[str, Any]],
) -> Dict[str, float]:
    rows = [catalog.get(value, {}) or {} for value in passage_ids]
    slot_count = max(
        (len(row.get("aperture_relation_slot_hits", []) or []) for row in rows),
        default=0,
    )
    covered_slots = {
        index
        for row in rows
        for index, hit in enumerate(row.get("aperture_relation_slot_hits", []) or [])
        if hit
    }
    entity_rows: Dict[str, Set[int]] = {}
    for row_index, row in enumerate(rows):
        for entity in row.get("aperture_binding_entities", []) or []:
            entity_rows.setdefault(str(entity), set()).add(row_index)
    join_pairs = {
        tuple(sorted((left, right)))
        for indices in entity_rows.values()
        if len(indices) > 1
        for left in indices
        for right in indices
        if left < right
    }
    connected_rows = {
        value for pair in join_pairs for value in pair
    }
    path_counts: Dict[str, int] = {}
    for row in rows:
        for path_id in row.get("aperture_local_path_ids", []) or []:
            path_counts[str(path_id)] = path_counts.get(str(path_id), 0) + 1
    shared_path_count = sum(count > 1 for count in path_counts.values())
    return {
        "result_slate_aperture_relation_slot_coverage": (
            len(covered_slots) / slot_count if slot_count else 0.0
        ),
        "result_slate_aperture_binding_join_count_log": math.log1p(
            len(join_pairs)
        ),
        "result_slate_aperture_connected_passage_fraction": (
            len(connected_rows) / len(rows) if rows else 0.0
        ),
        "result_slate_aperture_shared_path_count_log": math.log1p(
            shared_path_count
        ),
    }


def required_passage_ids(
    query_plan_log: Dict[str, Any],
    *,
    base_module,
    max_single_candidates: int = 24,
    max_pair_candidates: int = 12,
    candidate_mode: str = "proof_nested_topk",
    enforce_cq: bool = True,
) -> Set[int]:
    stages = stages_for_mode(candidate_mode)
    result: Set[int] = set()
    incumbent = query_plan_log.get("candidate_admission_incumbent", {}) or {}
    result.update(
        _safe_int(value) for value in incumbent.get("doc_ids", []) or []
        if _safe_int(value) >= 0
    )
    if "top5" in stages:
        result.update(content_complete.required_passage_ids(
            query_plan_log,
            base_module=base_module,
            max_single_candidates=max_single_candidates,
            max_pair_candidates=max_pair_candidates,
            candidate_mode="content_complete_unified",
            enforce_cq=enforce_cq,
        ))
        factorized = query_plan_log.get("factorized_query_executor", {}) or {}
        for group in factorized.get("groups", []) or []:
            if not isinstance(group, Mapping):
                continue
            if str(group.get("operator") or "") != "proof_gap_bound_access_path":
                continue
            if not bool(group.get("proof_valid")):
                continue
            result.update(
                _safe_int(value)
                for value in group.get("passage_indices", []) or []
                if _safe_int(value) >= 0
            )
    return result


def _proof_gap_membership_actions(
    query_plan_log: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Compile feasible, atomic proof-set insertions into the top-5 budget."""
    factorized = query_plan_log.get("factorized_query_executor", {}) or {}
    incumbent = query_plan_log.get("candidate_admission_incumbent", {}) or {}
    incumbent_ids = [
        _safe_int(value)
        for value in incumbent.get("doc_ids", []) or []
        if _safe_int(value) >= 0
    ][:5]
    if len(incumbent_ids) < 5:
        return []
    candidate_window = [
        _safe_int(value)
        for value in incumbent.get("candidate_window_doc_ids", []) or []
        if _safe_int(value) >= 0
    ]
    rank_by_id = {value: index + 1 for index, value in enumerate(candidate_window)}
    for candidate in factorized.get("candidates", []) or []:
        if not isinstance(candidate, Mapping):
            continue
        passage_idx = _safe_int(candidate.get("passage_idx"))
        original_rank = _safe_int(candidate.get("original_rank"), 10**9)
        if passage_idx >= 0:
            rank_by_id.setdefault(passage_idx, original_rank)
    local_context = query_plan_log.get("candidate_local_context", {}) or {}
    passage_context = local_context.get("passages", {}) or {}
    fixed_top2 = incumbent_ids[:2]
    actions: List[Dict[str, Any]] = []
    seen: Set[Tuple[int, ...]] = set()

    for group in factorized.get("groups", []) or []:
        if not isinstance(group, Mapping):
            continue
        if str(group.get("operator") or "") != "proof_gap_bound_access_path":
            continue
        if not (
            bool(group.get("proof_valid"))
            and bool(group.get("full_proof_set_projection"))
        ):
            continue
        proof_ids = list(dict.fromkeys(
            _safe_int(value)
            for value in group.get("passage_indices", []) or []
            if _safe_int(value) >= 0
        ))
        proof_tail = [value for value in proof_ids if value not in fixed_top2]
        if not proof_tail or len(fixed_top2) + len(proof_tail) > 5:
            continue
        result_top5 = list(dict.fromkeys([*fixed_top2, *proof_tail]))
        result_top5.extend(
            value for value in incumbent_ids if value not in set(result_top5)
        )
        result_top5 = result_top5[:5]
        if set(result_top5) == set(incumbent_ids):
            continue
        key = tuple(result_top5)
        if key in seen:
            continue
        seen.add(key)

        displaced = [value for value in incumbent_ids if value not in result_top5]
        preserved = [value for value in result_top5 if value in incumbent_ids]
        context_rows = [
            passage_context.get(str(value), {}) or {} for value in proof_ids
        ]
        fingerprints = [
            str(row.get("content_fingerprint") or "") for row in context_rows
        ]
        if not all(
            bool(row.get("content_available")) and fingerprint
            for row, fingerprint in zip(context_rows, fingerprints)
        ):
            continue
        displaced_fingerprints = [
            str((passage_context.get(str(value), {}) or {}).get(
                "content_fingerprint"
            ) or "")
            for value in displaced
        ]
        if not all(displaced_fingerprints):
            continue
        incumbent_fingerprints = [
            str((passage_context.get(str(value), {}) or {}).get(
                "content_fingerprint"
            ) or "")
            for value in incumbent_ids
        ]
        if not all(incumbent_fingerprints):
            continue
        content_state_fingerprint = hashlib.sha256(
            json.dumps(
                incumbent_fingerprints, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()[:24]

        member_count = len(proof_ids)
        external_count = sum(value not in incumbent_ids for value in proof_ids)
        features = {
            "proof_gap_full_set_action": 1.0,
            "proof_gap_member_count_log": math.log1p(member_count),
            "proof_gap_external_member_fraction": (
                external_count / max(member_count, 1)
            ),
            "proof_gap_before_coverage": _safe_float(
                group.get("proof_gap_before")
            ),
            "proof_gap_coverage_gain": _safe_float(group.get("proof_gap_gain")),
            "proof_gap_confidence": _safe_float(group.get("proof_confidence")),
            "proof_gap_probe_cost_log": math.log1p(
                max(_safe_float(group.get("probe_cost")), 0.0)
            ),
        }
        actions.append({
            "action_kind": "replace_top5_proof_set",
            "passage_ids": proof_ids,
            "candidate_ids": proof_ids,
            "preserved_incumbent_ids": preserved,
            "displaced_incumbent_ids": displaced,
            "result_top5_ids": result_top5,
            "final_ranks": [rank_by_id.get(value, 10**9) for value in proof_ids],
            "sources": ["factorized", "proof_gap_bound_access_path"],
            "expected_state_hash": str(incumbent.get("state_hash") or ""),
            "state_version": str(incumbent.get("state_version") or ""),
            "features": features,
            "query_fingerprint": str(local_context.get("query_fingerprint") or ""),
            "candidate_fingerprints": fingerprints,
            "displaced_fingerprints": displaced_fingerprints,
            "content_state_fingerprint": content_state_fingerprint,
            "content_identity_complete": True,
            "semantic_identity_complete": True,
            "top2_membership_invariant": True,
            "top5_membership_invariant": False,
            "proof_gap_group_id": str(group.get("group_id") or ""),
            "proof_gap_atomic_set": True,
        })
    return actions


def _enrich_action(
    action: Dict[str, Any],
    *,
    stage: str,
    catalog: Mapping[int, Mapping[str, Any]],
    base_module,
) -> Dict[str, Any]:
    names = feature_names(base_module)
    selected_ids = [
        _safe_int(value) for value in action.get("candidate_ids", []) or []
        if _safe_int(value) >= 0
    ]
    if stage == "top2":
        selected_ids = [
            _safe_int(value) for value in action.get("passage_ids", []) or []
            if _safe_int(value) >= 0
        ]
    displaced_ids = [
        _safe_int(value) for value in action.get("displaced_incumbent_ids", []) or []
        if _safe_int(value) >= 0
    ]
    proof_features, proof_contract = _aggregate_proof_features(
        selected_ids, displaced_ids, catalog
    )
    explicit_result = [
        _safe_int(value)
        for value in action.get("result_top5_ids", []) or []
        if _safe_int(value) >= 0
    ]
    result_slate_ids = explicit_result or list(dict.fromkeys([
        *selected_ids,
        *[
            _safe_int(value)
            for value in action.get("preserved_incumbent_ids", []) or []
            if _safe_int(value) >= 0
        ],
    ]))
    proof_features.update(_aperture_result_set_features(result_slate_ids, catalog))
    features = {
        name: _safe_float((action.get("features", {}) or {}).get(name))
        for name in names
    }
    features.update(proof_features)
    features.update({
        "nested_stage_top5_membership": float(stage == "top5"),
        "nested_stage_top2_ordering": float(stage == "top2"),
        "top2_membership_invariant": float(stage == "top5"),
    })
    payload = {
        "schema": "proof_carrying_nested_topk_v1",
        "stage": stage,
        "base": str(action.get("action_fingerprint") or ""),
        "selected": selected_ids,
        "displaced": displaced_ids,
        "proof": proof_contract,
    }
    return {
        **action,
        "features": {name: _safe_float(features.get(name)) for name in names},
        "nested_stage": stage,
        "top2_membership_invariant": stage == "top5",
        "top5_membership_invariant": stage == "top2",
        "proof_carrying_action": True,
        "proof_contract": proof_contract,
        "content_schema_version": "proof_carrying_nested_topk_v1",
        "action_fingerprint": hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:24],
    }


def _top5_membership_actions(
    query_plan_log: Dict[str, Any],
    *,
    base_module,
    max_single_candidates: int,
    max_pair_candidates: int,
    enforce_cq: bool,
) -> List[Dict[str, Any]]:
    incumbent = query_plan_log.get("candidate_admission_incumbent", {}) or {}
    incumbent_ids = [
        _safe_int(value) for value in incumbent.get("doc_ids", []) or []
        if _safe_int(value) >= 0
    ][:5]
    if len(incumbent_ids) < 5:
        return []
    catalog = materialize_passage_proof_catalog(
        query_plan_log, base_module=base_module
    )
    result = []
    for action in content_complete.generate_content_complete_post_physical_actions(
        query_plan_log,
        base_module=base_module,
        max_single_candidates=max_single_candidates,
        max_pair_candidates=max_pair_candidates,
        candidate_mode="content_complete_unified",
        enforce_cq=enforce_cq,
    ):
        if str(action.get("action_kind") or "") != "replace_top5_slot5":
            continue
        action = dict(action)
        action["displaced_incumbent_ids"] = incumbent_ids[4:5]
        result.append(_enrich_action(
            action, stage="top5", catalog=catalog, base_module=base_module
        ))
    for action in _proof_gap_membership_actions(query_plan_log):
        result.append(_enrich_action(
            action, stage="top5", catalog=catalog, base_module=base_module
        ))
    return result


def _top2_ordering_actions(
    query_plan_log: Dict[str, Any], *, base_module
) -> List[Dict[str, Any]]:
    catalog = materialize_passage_proof_catalog(
        query_plan_log, base_module=base_module
    )
    return [
        _enrich_action(
            action, stage="top2", catalog=catalog, base_module=base_module
        )
        for action in rank_partitioned._rank_partition_actions(
            query_plan_log, base_module=base_module
        )
    ]


def generate_proof_carrying_stage_actions(
    query_plan_log: Dict[str, Any],
    *,
    base_module,
    stage: str,
    max_single_candidates: int = 24,
    max_pair_candidates: int = 12,
    enforce_cq: bool = True,
) -> List[Dict[str, Any]]:
    normalized = str(stage or "").strip().lower()
    if normalized == "top5":
        actions = _top5_membership_actions(
            query_plan_log,
            base_module=base_module,
            max_single_candidates=max_single_candidates,
            max_pair_candidates=max_pair_candidates,
            enforce_cq=enforce_cq,
        )
    elif normalized == "top2":
        actions = _top2_ordering_actions(query_plan_log, base_module=base_module)
    else:
        return []
    rank_partitioned._annotate_action_pool(actions)
    return actions
