"""End-to-end counterfactual evidence-slate optimization.

The optimizer exposes one candidate relation over every evidence-producing
operator, compiles atomic top-2/top-5 actions, and augments each action with
logical-contract, physical-plan, provenance, cost, and displacement features.
Gold labels are deliberately absent from this module; exact counterfactual
labels are attached by the offline training-data builder only.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from typing import Any, Callable, Dict, Iterable, List, Mapping, Tuple


V219_EXTRA_FEATURE_NAMES = [
    "source_lifted",
    "source_factorized",
    "source_warehouse",
    "source_base_operator",
    "source_full_aperture",
    "unified_source_count_log",
    "unified_source_agreement",
    "candidate_multi_source_fraction",
    "candidate_lifted_fraction",
    "candidate_factorized_fraction",
    "candidate_warehouse_fraction",
    "candidate_base_operator_fraction",
    "candidate_full_aperture_fraction",
    "candidate_plan_precision_max",
    "candidate_branch_score_max",
    "candidate_utility_max",
    "candidate_estimated_cost_mean",
    "candidate_proof_fraction",
    "candidate_role_consistent_fraction",
    "candidate_answer_type_compatible_fraction",
    "physical_plan_enabled",
    "physical_strategy_adaptive",
    "physical_strategy_index_only",
    "physical_strategy_dense_first",
    "physical_estimated_cardinality",
    "physical_estimated_quality",
    "physical_estimated_recall_delta",
    "physical_estimated_proof_coverage",
    "physical_estimated_cost_log",
    "physical_estimated_loss_probability",
    "physical_objective",
    "physical_catalog_sample_count_log",
    "physical_catalog_exact_match",
    "cq_logical_plan_present",
    "cq_logical_admissible",
    "cq_contract_confidence",
    "cq_predicate_coverage",
    "cq_required_branch_count_log",
    "cq_required_hop_count_log",
    "cq_proof_obligation_count_log",
    "cq_typed_admissible",
    "cq_typed_contract_confidence",
    "cq_typed_predicate_coverage",
    "cq_hard_admissible",
    "action_estimated_cost_log",
    "action_displacement_risk_proxy",
    "factorized_x_physical_recall",
    "factorized_x_physical_loss",
    "warehouse_x_physical_quality",
    "pair_x_physical_proof",
    "source_agreement_x_catalog_support",
]


def feature_names(base_module) -> List[str]:
    return list(base_module.V213_FEATURE_NAMES) + list(V219_EXTRA_FEATURE_NAMES)


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


def _canonical_answer_type(value: Any) -> str:
    text = str(value or "").strip().lower().replace("_", " ")
    aliases = {
        "place": "location",
        "city": "location",
        "country": "location",
        "geographic location": "location",
        "human": "person",
        "people": "person",
        "organization": "organisation",
        "org": "organisation",
        "time": "date",
        "year": "date",
        "number": "numeric",
        "quantity": "numeric",
    }
    return aliases.get(text, text)


def _context_values(candidate: Mapping[str, Any], key: str) -> Iterable[Any]:
    for context in candidate.get("contexts", []) or []:
        if isinstance(context, Mapping) and context.get(key) is not None:
            yield context.get(key)


def _normalize_candidate(candidate: Mapping[str, Any], source: str) -> Dict[str, Any]:
    normalized = dict(candidate)
    normalized["_v219_source"] = source
    if source == "lifted":
        normalized.setdefault("score", normalized.get("raw_lift_score", 0.0))
        precision = max(
            (_safe_float(value) for value in _context_values(normalized, "plan_effective_precision")),
            default=0.0,
        )
        normalized.setdefault("confidence", precision)
        normalized.setdefault(
            "matched_anchor",
            any(bool(value) for value in _context_values(normalized, "anchor_entity")),
        )
        normalized.setdefault(
            "matched_terminal",
            any(bool(value) for value in _context_values(normalized, "end_entity")),
        )
    elif source == "factorized":
        normalized.setdefault(
            "score",
            normalized.get("utility", normalized.get("branch_score", normalized.get("raw_score", 0.0))),
        )
        normalized.setdefault(
            "confidence",
            max(
                _safe_float(normalized.get("branch_score")),
                _safe_float(normalized.get("typed_contract_confidence")),
                _safe_float(normalized.get("query_predicate_coverage")),
            ),
        )
        normalized.setdefault("matched_anchor", bool(normalized.get("anchor_entity")))
        normalized.setdefault("matched_terminal", bool(normalized.get("terminal_entity")))
        normalized.setdefault("role_consistent", bool(normalized.get("exact_entity_key")))
        normalized.setdefault(
            "complete_provenance",
            bool(normalized.get("triple")) and bool(normalized.get("exact_entity_key")),
        )
        normalized.setdefault("contexts", [{
            "anchor_entity": normalized.get("anchor_entity"),
            "end_entity": normalized.get("terminal_entity"),
            "relations": normalized.get("relations") or [normalized.get("relation")],
            "plan_effective_precision": normalized.get("typed_contract_confidence", 0.0),
            "score": normalized.get("branch_score", normalized.get("utility", 0.0)),
            "branch_idx": normalized.get("branch_idx"),
            "group_id": normalized.get("group_id"),
        }])
    return normalized


def iter_unified_candidate_refs(
    query_plan_log: Dict[str, Any],
    *,
    base_module,
    candidate_mode: str = "unified",
) -> Iterable[Tuple[str, Dict[str, Any]]]:
    """Yield one normalized relation over every runtime evidence source."""
    for source, candidate in base_module.iter_query_plan_evidence_candidate_refs(
        query_plan_log
    ):
        yield source, _normalize_candidate(candidate, source)
    if candidate_mode == "base_only":
        return

    for candidate in query_plan_log.get("lifted_passages", []) or []:
        if isinstance(candidate, Mapping):
            yield "lifted", _normalize_candidate(candidate, "lifted")

    factorized = query_plan_log.get("factorized_query_executor", {}) or {}
    if isinstance(factorized, Mapping):
        for candidate in factorized.get("candidates", []) or []:
            if isinstance(candidate, Mapping):
                yield "factorized", _normalize_candidate(candidate, "factorized")

    evidence_set = query_plan_log.get("evidence_set_retriever", {}) or {}
    if isinstance(evidence_set, Mapping):
        for key, source in (
            ("selected_ontology_passages", "warehouse"),
            ("candidate_pool_samples", "warehouse"),
            ("optimizer_candidate_pool_samples", "warehouse_optimizer"),
        ):
            for candidate in evidence_set.get(key, []) or []:
                if isinstance(candidate, Mapping):
                    yield source, _normalize_candidate(candidate, source)

    full_aperture = query_plan_log.get("full_aperture_proof_slate", {}) or {}
    if isinstance(full_aperture, Mapping):
        for candidate in full_aperture.get("candidates", []) or []:
            if isinstance(candidate, Mapping):
                yield "full_aperture", _normalize_candidate(
                    candidate, "full_aperture"
                )


def group_unified_candidate_refs(
    query_plan_log: Dict[str, Any],
    *,
    base_module,
    candidate_mode: str = "unified",
) -> Dict[int, List[Tuple[str, Dict[str, Any]]]]:
    grouped: Dict[int, List[Tuple[str, Dict[str, Any]]]] = {}
    seen = set()
    for source, candidate in iter_unified_candidate_refs(
        query_plan_log,
        base_module=base_module,
        candidate_mode=candidate_mode,
    ):
        passage_idx = _safe_int(candidate.get("passage_idx"))
        if passage_idx < 0:
            continue
        identity = (
            passage_idx,
            source,
            str(candidate.get("group_id") or ""),
            str(candidate.get("support_key") or ""),
        )
        if identity in seen:
            continue
        seen.add(identity)
        grouped.setdefault(passage_idx, []).append((source, candidate))
    return grouped


def _has_real_proof(refs: List[Tuple[str, Dict[str, Any]]]) -> bool:
    for _, candidate in refs:
        if candidate.get("proof_certificate") or candidate.get("typed_proof"):
            return True
        if (
            bool(candidate.get("has_proof_signal"))
            and bool(candidate.get("complete_provenance"))
            and bool(candidate.get("role_consistent"))
        ):
            return True
        if candidate.get("triple") and candidate.get("exact_entity_key"):
            return True
    return False


def _augment_for_branch_enumeration(
    query_plan_log: Dict[str, Any],
    *,
    base_module,
    candidate_mode: str,
    allowed_passage_ids: set,
) -> Dict[str, Any]:
    augmented = dict(query_plan_log)
    for source, log_key in base_module.SOURCE_TO_LOG_KEY.items():
        source_log = query_plan_log.get(log_key, {}) or {}
        if not isinstance(source_log, Mapping):
            continue
        cloned_log = dict(source_log)
        list_keys = ("added_candidates",)
        if source == "terminal":
            list_keys = ("selected_passages", "inserted_passages")
        for key in list_keys:
            cloned = []
            for candidate in source_log.get(key, []) or []:
                if not isinstance(candidate, Mapping):
                    cloned.append(candidate)
                    continue
                if _safe_int(candidate.get("passage_idx")) not in allowed_passage_ids:
                    continue
                item = dict(candidate)
                item["proof_certificate"] = {"v219_enumeration_only": True}
                cloned.append(item)
            cloned_log[key] = cloned
        if source == "terminal":
            verifier = source_log.get("verifier", {}) or {}
            if isinstance(verifier, Mapping):
                cloned_verifier = dict(verifier)
                cloned_verifier["inserted_passages"] = [
                    {
                        **dict(candidate),
                        "proof_certificate": {"v219_enumeration_only": True},
                    }
                    for candidate in verifier.get("inserted_passages", []) or []
                    if isinstance(candidate, Mapping)
                    and _safe_int(candidate.get("passage_idx")) in allowed_passage_ids
                ]
                cloned_log["verifier"] = cloned_verifier
        augmented[log_key] = cloned_log

    if candidate_mode != "base_only":
        structural = dict(augmented.get("structural_candidate_retriever", {}) or {})
        candidates = list(structural.get("added_candidates", []) or [])
        for source, candidate in iter_unified_candidate_refs(
            query_plan_log,
            base_module=base_module,
            candidate_mode=candidate_mode,
        ):
            if source in base_module.SOURCE_TO_LOG_KEY:
                continue
            if _safe_int(candidate.get("passage_idx")) not in allowed_passage_ids:
                continue
            item = dict(candidate)
            item["proof_certificate"] = {"v219_enumeration_only": True}
            candidates.append(item)
        structural["added_candidates"] = candidates
        augmented["structural_candidate_retriever"] = structural
    return augmented


def _source_stratified_candidate_ids(
    query_plan_log: Dict[str, Any],
    grouped: Dict[int, List[Tuple[str, Dict[str, Any]]]],
    limit: int,
) -> List[int]:
    incumbent = query_plan_log.get("candidate_admission_incumbent", {}) or {}
    window = [_safe_int(value) for value in incumbent.get("candidate_window_doc_ids", []) or []]
    rank = {passage_idx: index + 1 for index, passage_idx in enumerate(window)}
    eligible = {
        passage_idx for passage_idx in grouped
        if passage_idx in rank and rank[passage_idx] > 2
    }

    def priority(passage_idx: int):
        refs = grouped[passage_idx]
        proof = float(_has_real_proof(refs))
        aperture_rows = [
            candidate for source, candidate in refs if source == "full_aperture"
        ]
        aperture_slot_coverage = max(
            (
                _safe_float(candidate.get("relation_coverage_score"))
                for candidate in aperture_rows
            ),
            default=0.0,
        )
        aperture_anchor_coverage = max(
            (_safe_float(candidate.get("anchor_coverage")) for candidate in aperture_rows),
            default=0.0,
        )
        aperture_answer_type = max(
            (_safe_float(candidate.get("answer_type_score")) for candidate in aperture_rows),
            default=0.0,
        )
        confidence = max(
            (_safe_float(candidate.get("confidence")) for _, candidate in refs),
            default=0.0,
        )
        score = max(
            (_safe_float(candidate.get("score")) for _, candidate in refs),
            default=0.0,
        )
        return (
            proof,
            aperture_slot_coverage,
            aperture_anchor_coverage,
            aperture_answer_type,
            confidence,
            score,
            -rank[passage_idx],
        )

    source_order = [
        "full_aperture", "factorized", "lifted", "warehouse_optimizer", "warehouse",
        "bind_probe", "path_executor", "evidence_grounded", "support",
        "schema_slot", "schema_route", "evidence_index", "projected_graph",
        "terminal", "missing", "subgraph", "failure_evidence", "structural",
    ]
    buckets: Dict[str, List[int]] = {source: [] for source in source_order}
    for passage_idx in eligible:
        for source, _ in grouped[passage_idx]:
            buckets.setdefault(source, []).append(passage_idx)
    for source in buckets:
        buckets[source] = sorted(
            set(buckets[source]), key=priority, reverse=True
        )

    selected: List[int] = []




    aperture_values = buckets.get("full_aperture", [])
    if aperture_values:
        return aperture_values[: max(int(limit), 1)]
    cursor = 0
    while len(selected) < max(int(limit), 1):
        added = False
        for source in source_order + sorted(set(buckets) - set(source_order)):
            values = buckets.get(source, [])
            if cursor < len(values) and values[cursor] not in selected:
                selected.append(values[cursor])
                added = True
                if len(selected) >= max(int(limit), 1):
                    break
        if not added:
            break
        cursor += 1
    if len(selected) < max(int(limit), 1):
        for passage_idx in sorted(eligible, key=priority, reverse=True):
            if passage_idx not in selected:
                selected.append(passage_idx)
            if len(selected) >= max(int(limit), 1):
                break
    return selected


def _candidate_answer_type_compatible(
    refs: List[Tuple[str, Dict[str, Any]]], expected: str
) -> bool:
    if not expected:
        return True
    observed = {
        _canonical_answer_type(candidate.get("answer_type"))
        for _, candidate in refs
        if _canonical_answer_type(candidate.get("answer_type"))
    }
    return not observed or expected in observed


def _cq_admission(
    action: Dict[str, Any],
    grouped: Dict[int, List[Tuple[str, Dict[str, Any]]]],
    logical: Mapping[str, Any],
    typed: Mapping[str, Any],
) -> Tuple[bool, str]:
    candidate_ids = [
        _safe_int(value) for value in action.get("candidate_ids", []) or []
    ]
    refs = [grouped.get(value, []) for value in candidate_ids]
    sources = {source for rows in refs for source, _ in rows}
    uses_factorized = "factorized" in sources
    if uses_factorized and logical and logical.get("admissible") is False:
        return False, "logical_plan_inadmissible"
    if uses_factorized and typed and typed.get("admissible") is False:
        return False, "typed_contract_inadmissible"

    expected = _canonical_answer_type(
        logical.get("answer_type") or typed.get("answer_type")
    )
    if any(not _candidate_answer_type_compatible(rows, expected) for rows in refs):
        return False, "answer_type_contract_mismatch"

    required_branches = max(_safe_int(logical.get("required_branch_count"), 0), 0)
    if required_branches >= 2 and len(candidate_ids) >= 2:
        branch_ids = set()
        factorized_members = 0
        for rows in refs:
            member_branches = {
                str(candidate.get("branch_idx"))
                for source, candidate in rows
                if source == "factorized" and candidate.get("branch_idx") is not None
            }
            if member_branches:
                factorized_members += 1
                branch_ids.update(member_branches)
        if factorized_members == len(refs) and len(branch_ids) < 2:
            return False, "comparison_branch_not_complete"
    return True, ""


def _structured_fingerprint(candidate: Mapping[str, Any]) -> str:
    payload = {
        "passage_idx": _safe_int(candidate.get("passage_idx")),
        "source": str(candidate.get("_v219_source") or ""),
        "original_rank": _safe_int(candidate.get("original_rank")),
        "group_id": str(candidate.get("group_id") or ""),
        "plan_id": str(candidate.get("plan_id") or ""),
        "triple": candidate.get("triple") or [],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:24]


def generate_end_to_end_counterfactual_actions(
    query_plan_log: Dict[str, Any],
    *,
    base_module,
    max_single_candidates: int = 24,
    max_pair_candidates: int = 12,
    candidate_mode: str = "unified",
    enforce_cq: bool = True,
    zero_physical_features: bool = False,
    zero_cq_features: bool = False,
) -> List[Dict[str, Any]]:
    """Compile unified, branch-complete actions over one immutable final state."""
    grouped = group_unified_candidate_refs(
        query_plan_log,
        base_module=base_module,
        candidate_mode=candidate_mode,
    )
    selected_candidate_ids = _source_stratified_candidate_ids(
        query_plan_log, grouped, max_single_candidates
    )
    augmented = _augment_for_branch_enumeration(
        query_plan_log,
        base_module=base_module,
        candidate_mode=candidate_mode,
        allowed_passage_ids=set(selected_candidate_ids),
    )
    actions = base_module.generate_incumbent_aware_slate_actions(
        augmented,
        max_single_candidates=max_single_candidates,
        max_pair_candidates=max_pair_candidates,
    )
    factorized = query_plan_log.get("factorized_query_executor", {}) or {}
    logical = factorized.get("logical_evidence_plan", {}) or {}
    typed = factorized.get("typed_contract", {}) or {}
    physical = factorized.get("physical_plan_optimizer", {}) or {}
    selected_physical = physical.get("selected", {}) or {}
    strategy = str(selected_physical.get("access_strategy") or "").lower()
    expected_type = _canonical_answer_type(
        logical.get("answer_type") or typed.get("answer_type")
    )
    names = feature_names(base_module)

    compiled: List[Dict[str, Any]] = []
    seen = set()
    for action in actions:
        candidate_ids = [
            _safe_int(value)
            for value in action.get("candidate_ids", []) or []
            if _safe_int(value) >= 0
        ]
        if not candidate_ids:
            continue
        candidate_refs = [grouped.get(value, []) for value in candidate_ids]
        if any(not rows for rows in candidate_refs):
            continue
        cq_ok, cq_reason = _cq_admission(action, grouped, logical, typed)
        if enforce_cq and not cq_ok:
            continue

        source_sets = [{source for source, _ in rows} for rows in candidate_refs]
        source_union = set().union(*source_sets)
        flat_candidates = [candidate for rows in candidate_refs for _, candidate in rows]
        source_counts = Counter(source for rows in candidate_refs for source, _ in rows)
        candidate_count = max(len(candidate_ids), 1)
        physical_recall = _safe_float(selected_physical.get("estimated_recall_delta"))
        physical_quality = _safe_float(selected_physical.get("estimated_quality"))
        physical_proof = _safe_float(selected_physical.get("estimated_proof_coverage"))
        physical_cost = _safe_float(selected_physical.get("estimated_cost"))
        physical_loss = _safe_float(selected_physical.get("estimated_loss_probability"))
        if zero_physical_features:
            strategy = ""
            physical_recall = 0.0
            physical_quality = 0.0
            physical_proof = 0.0
            physical_cost = 0.0
            physical_loss = 0.0
        factorized_fraction = sum(
            1 for rows in candidate_refs if any(source == "factorized" for source, _ in rows)
        ) / candidate_count
        warehouse_fraction = sum(
            1 for rows in candidate_refs if any(source.startswith("warehouse") for source, _ in rows)
        ) / candidate_count
        lifted_fraction = sum(
            1 for rows in candidate_refs if any(source == "lifted" for source, _ in rows)
        ) / candidate_count
        base_fraction = sum(
            1 for rows in candidate_refs
            if any(source in base_module.SOURCE_TO_LOG_KEY for source, _ in rows)
        ) / candidate_count
        full_aperture_fraction = sum(
            1 for rows in candidate_refs
            if any(source == "full_aperture" for source, _ in rows)
        ) / candidate_count
        multi_source_fraction = sum(len(values) > 1 for values in source_sets) / candidate_count
        compatible_fraction = sum(
            _candidate_answer_type_compatible(rows, expected_type)
            for rows in candidate_refs
        ) / candidate_count
        estimated_candidate_cost = sum(
            _safe_float(candidate.get("estimated_cost")) for candidate in flat_candidates
        ) / max(len(flat_candidates), 1)
        action_cost = estimated_candidate_cost + factorized_fraction * physical_cost
        base_features = action.get("features", {}) or {}
        all_new = _safe_float(base_features.get("all_new_top2_pair"))
        displaced_score = _safe_float(base_features.get("displaced_incumbent_score"))
        candidate_score = max(
            (_safe_float(candidate.get("score")) for candidate in flat_candidates),
            default=0.0,
        )
        displacement_proxy = (
            factorized_fraction * physical_loss
            + 0.20 * all_new
            + 0.10 * max(displaced_score - candidate_score, 0.0)
        )

        extra = {
            "source_lifted": float("lifted" in source_union),
            "source_factorized": float("factorized" in source_union),
            "source_warehouse": float(any(value.startswith("warehouse") for value in source_union)),
            "source_base_operator": float(any(value in base_module.SOURCE_TO_LOG_KEY for value in source_union)),
            "source_full_aperture": float("full_aperture" in source_union),
            "unified_source_count_log": math.log1p(len(source_union)),
            "unified_source_agreement": min(len(source_union) / 5.0, 1.0),
            "candidate_multi_source_fraction": multi_source_fraction,
            "candidate_lifted_fraction": lifted_fraction,
            "candidate_factorized_fraction": factorized_fraction,
            "candidate_warehouse_fraction": warehouse_fraction,
            "candidate_base_operator_fraction": base_fraction,
            "candidate_full_aperture_fraction": full_aperture_fraction,
            "candidate_plan_precision_max": max(
                (
                    max(
                        [_safe_float(value) for value in _context_values(candidate, "plan_effective_precision")]
                        or [0.0]
                    )
                    for candidate in flat_candidates
                ),
                default=0.0,
            ),
            "candidate_branch_score_max": max(
                (_safe_float(candidate.get("branch_score")) for candidate in flat_candidates),
                default=0.0,
            ),
            "candidate_utility_max": max(
                (_safe_float(candidate.get("utility")) for candidate in flat_candidates),
                default=0.0,
            ),
            "candidate_estimated_cost_mean": estimated_candidate_cost,
            "candidate_proof_fraction": sum(_has_real_proof(rows) for rows in candidate_refs) / candidate_count,
            "candidate_role_consistent_fraction": sum(
                any(bool(candidate.get("role_consistent")) for _, candidate in rows)
                for rows in candidate_refs
            ) / candidate_count,
            "candidate_answer_type_compatible_fraction": compatible_fraction,
            "physical_plan_enabled": float(bool(physical.get("enabled", False))),
            "physical_strategy_adaptive": float(strategy == "adaptive"),
            "physical_strategy_index_only": float(strategy == "index_only"),
            "physical_strategy_dense_first": float(strategy == "dense_first"),
            "physical_estimated_cardinality": _safe_float(selected_physical.get("estimated_cardinality")),
            "physical_estimated_quality": physical_quality,
            "physical_estimated_recall_delta": physical_recall,
            "physical_estimated_proof_coverage": physical_proof,
            "physical_estimated_cost_log": math.log1p(max(physical_cost, 0.0)),
            "physical_estimated_loss_probability": physical_loss,
            "physical_objective": _safe_float(selected_physical.get("objective")),
            "physical_catalog_sample_count_log": math.log1p(
                max(_safe_int(selected_physical.get("catalog_sample_count"), 0), 0)
            ),
            "physical_catalog_exact_match": float(
                selected_physical.get("catalog_match_level") == "query_feature_exact"
            ),
            "cq_logical_plan_present": float(bool(logical)),
            "cq_logical_admissible": float(logical.get("admissible", True)) if logical else 0.0,
            "cq_contract_confidence": _safe_float(logical.get("contract_confidence")),
            "cq_predicate_coverage": _safe_float(logical.get("predicate_coverage")),
            "cq_required_branch_count_log": math.log1p(max(_safe_int(logical.get("required_branch_count"), 0), 0)),
            "cq_required_hop_count_log": math.log1p(max(_safe_int(logical.get("required_hop_count"), 0), 0)),
            "cq_proof_obligation_count_log": math.log1p(len(logical.get("proof_obligations", []) or [])),
            "cq_typed_admissible": float(typed.get("admissible", True)) if typed else 0.0,
            "cq_typed_contract_confidence": _safe_float(typed.get("contract_confidence")),
            "cq_typed_predicate_coverage": _safe_float(typed.get("query_predicate_coverage")),
            "cq_hard_admissible": float(cq_ok),
            "action_estimated_cost_log": math.log1p(max(action_cost, 0.0)),
            "action_displacement_risk_proxy": displacement_proxy,
            "factorized_x_physical_recall": factorized_fraction * physical_recall,
            "factorized_x_physical_loss": factorized_fraction * physical_loss,
            "warehouse_x_physical_quality": warehouse_fraction * physical_quality,
            "pair_x_physical_proof": float(len(candidate_ids) > 1) * physical_proof,
            "source_agreement_x_catalog_support": min(len(source_union) / 5.0, 1.0)
            * math.log1p(max(_safe_int(selected_physical.get("catalog_sample_count"), 0), 0)),
        }
        if zero_physical_features:
            for name in list(extra):
                if name.startswith("physical_") or "_x_physical_" in name or name == "source_agreement_x_catalog_support":
                    extra[name] = 0.0
        if zero_cq_features:
            for name in list(extra):
                if name.startswith("cq_") or name == "candidate_answer_type_compatible_fraction":
                    extra[name] = 0.0

        features = {
            name: _safe_float(base_features.get(name), 0.0)
            for name in base_module.V213_FEATURE_NAMES
        }
        features["pair_complete_proof"] = float(
            len(candidate_ids) == 2 and all(_has_real_proof(rows) for rows in candidate_refs)
        )
        features["local_comparator_proxy"] = (
            _safe_float(base_features.get("candidate_score"))
            + 0.35 * max((_safe_float(candidate.get("confidence")) for candidate in flat_candidates), default=0.0)
            + 0.25 * extra["candidate_proof_fraction"]
            + 0.20 * extra["candidate_answer_type_compatible_fraction"]
            - 0.20 * displacement_proxy
        )
        features.update(extra)
        sources = sorted(source_union)
        key = (
            str(action.get("action_kind") or ""),
            tuple(action.get("passage_ids", []) or []),
            _safe_int(action.get("preserved_incumbent_rank"), 0),
        )
        if key in seen:
            continue
        seen.add(key)
        candidate_fingerprints = [
            _structured_fingerprint(candidate)
            for rows in candidate_refs
            for _, candidate in rows
        ]
        identity_payload = {
            "state": action.get("expected_state_hash", ""),
            "kind": action.get("action_kind", ""),
            "passages": action.get("passage_ids", []),
            "preserved_rank": action.get("preserved_incumbent_rank", 0),
            "candidates": sorted(candidate_fingerprints),
        }
        compiled_action = {
            **action,
            "sources": sources,
            "source_signature": base_module.normalized_action_source_signature(sources),
            "provenance_signature": base_module.exact_action_source_signature(sources),
            "features": {name: _safe_float(features.get(name), 0.0) for name in names},
            "candidate_fingerprints": candidate_fingerprints,
            "structured_identity_complete": bool(candidate_fingerprints),
            "content_identity_complete": False,
            "cq_hard_admissible": cq_ok,
            "cq_rejection_reason": cq_reason,
            "action_fingerprint": hashlib.sha256(
                json.dumps(identity_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()[:24],
            "v219_unified_candidate_action": True,
        }
        compiled_action["source_risk_group"] = base_module.source_risk_group_key(
            compiled_action
        )
        compiled.append(compiled_action)
    return compiled


def rank_end_to_end_counterfactual_slate(
    ranker,
    query_plan_log: Dict[str, Any],
    *,
    base_module,
    listwise_ranker: Callable[..., Tuple[Dict[str, Any], Dict[str, Any]]],
    candidate_mode: str = "unified",
    enforce_cq: bool = True,
    zero_physical_features: bool = False,
    zero_cq_features: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    generator = lambda log, **kwargs: generate_end_to_end_counterfactual_actions(
        log,
        base_module=base_module,
        candidate_mode=candidate_mode,
        enforce_cq=enforce_cq,
        zero_physical_features=zero_physical_features,
        zero_cq_features=zero_cq_features,
        **kwargs,
    )
    result, log = listwise_ranker(
        ranker,
        query_plan_log,
        action_generator=generator,
        safe_float=base_module._safe_float,
        safe_int=base_module._safe_int,
        candidate_compiler="end_to_end_counterfactual_evidence_relation",
    )
    grouped = group_unified_candidate_refs(
        query_plan_log,
        base_module=base_module,
        candidate_mode=candidate_mode,
    )
    counts = Counter(
        source for refs in grouped.values() for source, _ in refs
    )
    log.update({
        "mode": "end_to_end_counterfactual_slate",
        "candidate_mode": candidate_mode,
        "cq_constraints_enforced": bool(enforce_cq),
        "physical_features_enabled": not bool(zero_physical_features),
        "cq_features_enabled": not bool(zero_cq_features),
        "unified_candidate_count": len(grouped),
        "candidate_source_counts": dict(sorted(counts.items())),
    })
    return result, log
