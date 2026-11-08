"""Executable proof semijoin for ontology-guided nested top-k actions.

The proof-carrying compiler intentionally exposes a wide evidence aperture.
This module adds a relational access-path layer before learned admission.
Strong actions participate in an executable proof through relation-slot
coverage and a realized entity/path join. Because relation realization can
be incomplete, a bounded signal-wise fallback preserves alternative access
paths for the learned counterfactual admission model. The contract is
gold-free and independent of dataset identity.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

try:
    from . import query_plan_proof_carrying_nested_topk_optimizer as proof_nested
    from .query_plan_clause_complete_lineage import (
        lineage_replacement_profile,
        query_clause_profile,
        row_lineage_tokens,
    )
except ImportError:
    import query_plan_proof_carrying_nested_topk_optimizer as proof_nested
    from query_plan_clause_complete_lineage import (
        lineage_replacement_profile,
        query_clause_profile,
        row_lineage_tokens,
    )


CANDIDATE_MODE = "executable_proof_semijoin_nested_topk"
CLAUSE_COMPLETE_LINEAGE_MODE = "clause_complete_lineage_nested_topk"
SCHEMA_VERSION = "executable_proof_semijoin_v1"
CLAUSE_LINEAGE_SCHEMA_VERSION = "clause_complete_lineage_semijoin_v1"

SEMIJOIN_FEATURE_NAMES = [
    "semijoin_required_slot_count_log",
    "semijoin_required_branch_count_log",
    "semijoin_before_slot_coverage",
    "semijoin_after_slot_coverage",
    "semijoin_missing_slot_gain",
    "semijoin_before_branch_coverage",
    "semijoin_after_branch_coverage",
    "semijoin_missing_branch_gain",
    "semijoin_shared_binding_count_log",
    "semijoin_shared_path_count_log",
    "semijoin_anchor_connected",
    "semijoin_terminal_supported",
    "semijoin_answer_type_supported",
    "semijoin_selected_participation",
    "semijoin_proof_coverage_margin",
    "semijoin_displacement_safe",
    "semijoin_executable",
    "semijoin_strong_executable",
    "semijoin_fallback_eligible",
]

CLAUSE_LINEAGE_FEATURE_NAMES = [
    "clause_required_count_log",
    "clause_surface_lower_bound_log",
    "clause_logical_relation_count_log",
    "clause_logical_coverage",
    "clause_logical_complete",
    "clause_long_chain",
    "clause_set_level_required",
    "clause_selected_path_coverage",
    "clause_selected_path_complete",
    "lineage_before_coverage_log",
    "lineage_after_coverage_log",
    "lineage_coverage_gain_log",
    "lineage_unique_obligation_loss_log",
    "lineage_witness_preserved",
    "lineage_set_action",
]

_TOKEN = re.compile(r"[a-z0-9]+")
_LOW_INFORMATION_KEYS = {
    "a", "an", "article", "book", "census", "city", "company", "country",
    "district", "film", "group", "movie", "person", "place", "province",
    "region", "state", "team", "the", "town", "village", "work",
}


def feature_names(
    base_module,
    candidate_mode: str = CANDIDATE_MODE,
) -> List[str]:
    names = [*proof_nested.feature_names(base_module), *SEMIJOIN_FEATURE_NAMES]
    if str(candidate_mode or "").strip().lower() == CLAUSE_COMPLETE_LINEAGE_MODE:
        names.extend(CLAUSE_LINEAGE_FEATURE_NAMES)
    return names


def is_executable_semijoin_mode(value: Any) -> bool:
    return str(value or "").strip().lower() in {
        CANDIDATE_MODE,
        CLAUSE_COMPLETE_LINEAGE_MODE,
    }


def stages_for_mode(value: Any) -> Tuple[str, ...]:
    return ("top5", "top2") if is_executable_semijoin_mode(value) else ()


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


def _entity_key(value: Any) -> str:
    return " ".join(_TOKEN.findall(str(value or "").lower()))


def _informative_entity_key(value: Any) -> str:
    key = _entity_key(value)
    tokens = key.split()
    if not key or (len(tokens) == 1 and tokens[0] in _LOW_INFORMATION_KEYS):
        return ""
    if all(token.isdigit() for token in tokens):
        return ""
    return key


def _entity_match(left: str, right: str) -> bool:
    if not left or not right:
        return False
    if left == right:
        return True
    left_tokens = set(left.split())
    right_tokens = set(right.split())
    shorter = min(len(left_tokens), len(right_tokens))
    if shorter == 0:
        return False
    overlap = len(left_tokens & right_tokens)
    return bool(shorter >= 2 and overlap / shorter >= 0.8)


def _logical_contract(query_plan_log: Mapping[str, Any]) -> Mapping[str, Any]:
    factorized = query_plan_log.get("factorized_query_executor", {}) or {}
    logical = factorized.get("logical_evidence_plan", {}) or {}
    return logical if isinstance(logical, Mapping) else {}


def _anchors(query_plan_log: Mapping[str, Any]) -> List[str]:
    logical = _logical_contract(query_plan_log)
    factorized = query_plan_log.get("factorized_query_executor", {}) or {}
    typed = factorized.get("typed_contract", {}) or {}
    values = logical.get("anchors", []) or typed.get("anchors", []) or []
    result = [_informative_entity_key(value) for value in values]
    return list(dict.fromkeys(value for value in result if value))


def _required_branch_count(query_plan_log: Mapping[str, Any]) -> int:
    logical = _logical_contract(query_plan_log)
    configured = max(_safe_int(logical.get("required_branch_count"), 0), 0)
    return max(configured, min(len(_anchors(query_plan_log)), 1))


def _row_slots(row: Mapping[str, Any]) -> Set[int]:
    return {
        index
        for index, hit in enumerate(row.get("aperture_relation_slot_hits", []) or [])
        if bool(hit)
    }


def _slot_count(rows: Sequence[Mapping[str, Any]]) -> int:
    return max(
        (len(row.get("aperture_relation_slot_hits", []) or []) for row in rows),
        default=0,
    )


def _row_entities(row: Mapping[str, Any]) -> Set[str]:
    return {
        value
        for raw in row.get("aperture_binding_entities", []) or []
        for value in [_informative_entity_key(raw)]
        if value
    }


def _anchor_hits(rows: Sequence[Mapping[str, Any]], anchors: Sequence[str]) -> Set[int]:
    hits: Set[int] = set()
    for anchor_index, anchor in enumerate(anchors):
        if any(
            any(_entity_match(anchor, entity) for entity in _row_entities(row))
            for row in rows
        ):
            hits.add(anchor_index)
    if len(anchors) == 1 and not hits and any(
        _safe_float(row.get("aperture_anchor_coverage")) > 0.0 for row in rows
    ):
        hits.add(0)
    return hits


def _shared_values(
    rows: Sequence[Mapping[str, Any]], field: str, *, normalize_entities: bool = False
) -> Set[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        values = row.get(field, []) or []
        local = {
            (_informative_entity_key(value) if normalize_entities else str(value))
            for value in values
        }
        counts.update(value for value in local if value)
    return {value for value, count in counts.items() if count >= 2}


def _selected_participates(
    result_rows: Sequence[Tuple[int, Mapping[str, Any]]], selected_ids: Set[int]
) -> Tuple[bool, int, int]:
    entity_owners: Dict[str, Set[int]] = {}
    path_owners: Dict[str, Set[int]] = {}
    selected_local_proof = False
    for passage_id, row in result_rows:
        for entity in _row_entities(row):
            entity_owners.setdefault(entity, set()).add(passage_id)
        for path_id in row.get("aperture_local_path_ids", []) or []:
            path_owners.setdefault(str(path_id), set()).add(passage_id)
        if passage_id in selected_ids and (
            bool(row.get("aperture_local_proof_path"))
            or (
                bool(row.get("complete_provenance"))
                and bool(row.get("binding"))
            )
        ):
            selected_local_proof = True
    shared_entities = {
        value for value, owners in entity_owners.items()
        if len(owners) >= 2 and bool(owners & selected_ids)
    }
    shared_paths = {
        value for value, owners in path_owners.items()
        if len(owners) >= 2 and bool(owners & selected_ids)
    }
    return (
        bool(selected_local_proof or shared_entities or shared_paths),
        len(shared_entities),
        len(shared_paths),
    )


def evaluate_executable_semijoin(
    action: Mapping[str, Any],
    query_plan_log: Mapping[str, Any],
    catalog: Mapping[int, Mapping[str, Any]],
    *,
    stage: str,
    candidate_mode: str = CANDIDATE_MODE,
) -> Dict[str, Any]:
    """Return the gold-free semijoin profile for one physical action."""
    normalized_mode = str(candidate_mode or "").strip().lower()
    clause_lineage_mode = normalized_mode == CLAUSE_COMPLETE_LINEAGE_MODE
    factorized = query_plan_log.get("factorized_query_executor", {}) or {}
    clause_profile = query_clause_profile(
        str(query_plan_log.get("query", "") or ""),
        query_plan_log,
        factorized,
    )
    incumbent = query_plan_log.get("candidate_admission_incumbent", {}) or {}
    incumbent_ids = [
        _safe_int(value) for value in incumbent.get("doc_ids", []) or []
        if _safe_int(value) >= 0
    ][:5]
    selected_ids = {
        _safe_int(value)
        for value in (
            action.get("passage_ids", [])
            if stage == "top2" else action.get("candidate_ids", [])
        ) or []
        if _safe_int(value) >= 0
    }
    if stage == "top5":
        explicit_result = [
            _safe_int(value)
            for value in action.get("result_top5_ids", []) or []
            if _safe_int(value) >= 0
        ]
        after_ids = explicit_result[:5] or list(dict.fromkeys([
            *[
                _safe_int(value)
                for value in action.get("preserved_incumbent_ids", []) or []
                if _safe_int(value) >= 0
            ],
            *selected_ids,
        ]))[:5]
        before_ids = incumbent_ids[:5]
    else:
        after_ids = [
            _safe_int(value) for value in action.get("passage_ids", []) or []
            if _safe_int(value) >= 0
        ][:2]
        before_ids = incumbent_ids[:2]

    before_rows = [catalog.get(value, {}) or {} for value in before_ids]
    after_pairs = [(value, catalog.get(value, {}) or {}) for value in after_ids]
    after_rows = [row for _, row in after_pairs]
    selected_rows = [catalog.get(value, {}) or {} for value in selected_ids]
    displaced_rows = [
        catalog.get(_safe_int(value), {}) or {}
        for value in action.get("displaced_incumbent_ids", []) or []
        if _safe_int(value) >= 0
    ]

    required_slots = max(_slot_count(before_rows), _slot_count(after_rows))
    if clause_lineage_mode:
        required_slots = max(
            required_slots,
            _safe_int(clause_profile.get("required_clause_count"), 1),
        )
    before_slots = set().union(*(_row_slots(row) for row in before_rows)) if before_rows else set()
    after_slots = set().union(*(_row_slots(row) for row in after_rows)) if after_rows else set()
    before_slot_coverage = len(before_slots) / required_slots if required_slots else 0.0
    after_slot_coverage = len(after_slots) / required_slots if required_slots else 0.0

    anchors = _anchors(query_plan_log)
    required_branches = max(_required_branch_count(query_plan_log), 1 if anchors else 0)
    before_anchor_hits = _anchor_hits(before_rows, anchors)
    after_anchor_hits = _anchor_hits(after_rows, anchors)
    before_branch_coverage = (
        min(len(before_anchor_hits) / required_branches, 1.0)
        if required_branches else 0.0
    )
    after_branch_coverage = (
        min(len(after_anchor_hits) / required_branches, 1.0)
        if required_branches else 0.0
    )

    participation, shared_bindings, shared_paths = _selected_participates(
        after_pairs, selected_ids
    )
    anchor_connected = bool(after_anchor_hits) if anchors else any(
        bool(row.get("exact_entity_key")) for row in after_rows
    )
    terminal_supported = any(
        bool(row.get("aperture_terminal_relation")) for row in after_rows
    )
    answer_type_supported = any(
        bool(row.get("answer_type_compatible"))
        or _safe_float(row.get("aperture_answer_type_score")) > 0.0
        for row in after_rows
    )
    local_or_single_proof = any(
        bool(row.get("aperture_local_proof_path"))
        or (bool(row.get("complete_provenance")) and bool(row.get("binding")))
        for row in selected_rows
    )
    logical = _logical_contract(query_plan_log)
    comparison = max(_safe_int(logical.get("required_branch_count"), 0), 0) > 1
    slot_complete = bool(required_slots and len(after_slots) == required_slots)
    branch_complete = bool(
        not required_branches or len(after_anchor_hits) >= required_branches
    )
    join_realized = bool(shared_bindings or shared_paths or local_or_single_proof)
    executable = bool(
        participation
        and slot_complete
        and join_realized
        and (
            branch_complete
            if comparison
            else anchor_connected and terminal_supported
        )
    )

    selected_coverage = _mean(
        _safe_float(row.get("obligation_coverage")) for row in selected_rows
    )
    displaced_coverage = _mean(
        _safe_float(row.get("obligation_coverage")) for row in displaced_rows
    )
    proof_margin = selected_coverage - displaced_coverage
    slot_gain = max(after_slot_coverage - before_slot_coverage, 0.0)
    branch_gain = max(after_branch_coverage - before_branch_coverage, 0.0)
    lineage = lineage_replacement_profile(before_ids, after_ids, catalog)
    if clause_lineage_mode:
        displacement_safe = bool(
            lineage.get("witness_preserved", False)
            and (
                proof_margin >= -1e-12
                or _safe_int(lineage.get("coverage_gain"), 0) > 0
                or slot_gain > 0.0
                or branch_gain > 0.0
            )
        )
    else:
        displacement_safe = bool(
            proof_margin >= -1e-12 or slot_gain > 0.0 or branch_gain > 0.0
        )
    eligible = bool(executable and displacement_safe)

    if clause_lineage_mode and not lineage.get("witness_preserved", False):
        reason = "lineage_witness_displaced"
    elif not participation:
        reason = "selected_passage_not_in_join"
    elif not slot_complete:
        reason = "relation_slots_incomplete"
    elif comparison and not branch_complete:
        reason = "comparison_branches_incomplete"
    elif not comparison and not anchor_connected:
        reason = "anchor_not_connected"
    elif not comparison and not terminal_supported:
        reason = "terminal_relation_missing"
    elif not join_realized:
        reason = "no_realized_join_key"
    elif not displacement_safe:
        reason = "negative_proof_displacement_margin"
    else:
        reason = "executable"

    features = {
        "semijoin_required_slot_count_log": math.log1p(required_slots),
        "semijoin_required_branch_count_log": math.log1p(required_branches),
        "semijoin_before_slot_coverage": before_slot_coverage,
        "semijoin_after_slot_coverage": after_slot_coverage,
        "semijoin_missing_slot_gain": slot_gain,
        "semijoin_before_branch_coverage": before_branch_coverage,
        "semijoin_after_branch_coverage": after_branch_coverage,
        "semijoin_missing_branch_gain": branch_gain,
        "semijoin_shared_binding_count_log": math.log1p(shared_bindings),
        "semijoin_shared_path_count_log": math.log1p(shared_paths),
        "semijoin_anchor_connected": float(anchor_connected),
        "semijoin_terminal_supported": float(terminal_supported),
        "semijoin_answer_type_supported": float(answer_type_supported),
        "semijoin_selected_participation": float(participation),
        "semijoin_proof_coverage_margin": proof_margin,
        "semijoin_displacement_safe": float(displacement_safe),
        "semijoin_executable": float(executable),
        "semijoin_strong_executable": float(eligible),
        "semijoin_fallback_eligible": 0.0,
    }
    selected_clause_coverage = max(
        (
            _safe_float(row.get("query_clause_coverage"))
            for row in selected_rows
        ),
        default=0.0,
    )
    selected_clause_complete = any(
        bool(row.get("query_clause_complete", False))
        for row in selected_rows
    )
    if clause_lineage_mode:
        features.update({
            "clause_required_count_log": math.log1p(
                _safe_int(clause_profile.get("required_clause_count"), 1)
            ),
            "clause_surface_lower_bound_log": math.log1p(
                _safe_int(clause_profile.get("surface_clause_lower_bound"), 1)
            ),
            "clause_logical_relation_count_log": math.log1p(
                _safe_int(clause_profile.get("logical_relation_count"), 0)
            ),
            "clause_logical_coverage": _safe_float(
                clause_profile.get("logical_clause_coverage")
            ),
            "clause_logical_complete": float(
                bool(clause_profile.get("logical_clause_complete", False))
            ),
            "clause_long_chain": float(
                bool(clause_profile.get("long_chain", False))
            ),
            "clause_set_level_required": float(
                bool(clause_profile.get("set_level_evidence_required", False))
            ),
            "clause_selected_path_coverage": selected_clause_coverage,
            "clause_selected_path_complete": float(selected_clause_complete),
            "lineage_before_coverage_log": math.log1p(
                _safe_int(lineage.get("before_coverage_count"), 0)
            ),
            "lineage_after_coverage_log": math.log1p(
                _safe_int(lineage.get("after_coverage_count"), 0)
            ),
            "lineage_coverage_gain_log": math.log1p(
                _safe_int(lineage.get("coverage_gain"), 0)
            ),
            "lineage_unique_obligation_loss_log": math.log1p(
                _safe_int(lineage.get("unique_obligation_loss"), 0)
            ),
            "lineage_witness_preserved": float(
                bool(lineage.get("witness_preserved", False))
            ),
            "lineage_set_action": float(
                str(action.get("action_kind", "") or "")
                == "replace_top5_clause_complete_set"
            ),
        })
    return {
        "schema_version": (
            CLAUSE_LINEAGE_SCHEMA_VERSION if clause_lineage_mode else SCHEMA_VERSION
        ),
        "gold_free": True,
        "dataset_identity_used": False,
        "eligible": eligible,
        "strong_executable": eligible,
        "admission_tier": "strong_executable" if eligible else "rejected",
        "reason": reason,
        "stage": stage,
        "comparison_contract": comparison,
        "required_relation_slots": required_slots,
        "covered_relation_slots": sorted(after_slots),
        "required_branches": required_branches,
        "covered_anchor_branches": sorted(after_anchor_hits),
        "shared_binding_count": shared_bindings,
        "shared_path_count": shared_paths,
        "query_clause_profile": clause_profile,
        "selected_query_clause_coverage": selected_clause_coverage,
        "selected_query_clause_complete": selected_clause_complete,
        "lineage_replacement": lineage,
        "features": features,
    }


def _clause_complete_set_actions(
    query_plan_log: Dict[str, Any],
    catalog: Mapping[int, Mapping[str, Any]],
    *,
    max_single_candidates: int,
    max_pair_candidates: int,
) -> List[Dict[str, Any]]:
    """Enumerate complementary top-5 sets under a lineage no-loss contract."""
    query_profile = query_clause_profile(
        str(query_plan_log.get("query", "") or ""),
        query_plan_log,
        query_plan_log.get("factorized_query_executor", {}) or {},
    )
    if not bool(query_profile.get("set_level_evidence_required", False)):
        return []
    incumbent = query_plan_log.get("candidate_admission_incumbent", {}) or {}
    incumbent_ids = [
        _safe_int(value)
        for value in incumbent.get("doc_ids", []) or []
        if _safe_int(value) >= 0
    ][:5]
    if len(incumbent_ids) < 5:
        return []
    local_context = query_plan_log.get("candidate_local_context", {}) or {}
    passage_context = local_context.get("passages", {}) or {}
    incumbent_fingerprints = [
        str((passage_context.get(str(value), {}) or {}).get(
            "content_fingerprint"
        ) or "")
        for value in incumbent_ids
    ]
    if not all(incumbent_fingerprints):
        return []
    content_state_fingerprint = hashlib.sha256(
        json.dumps(incumbent_fingerprints, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:24]
    incumbent_set = set(incumbent_ids)
    candidate_window = [
        _safe_int(value)
        for value in incumbent.get("candidate_window_doc_ids", []) or []
        if _safe_int(value) >= 0
    ]
    rank_by_id = {value: index + 1 for index, value in enumerate(candidate_window)}

    def candidate_priority(passage_id: int) -> Tuple[float, ...]:
        row = catalog.get(passage_id, {}) or {}
        return (
            float(bool(row.get("query_clause_complete", False))),
            _safe_float(row.get("query_clause_coverage")),
            float(len(row_lineage_tokens(row))),
            float(bool(row.get("aperture_local_proof_path", False))),
            _safe_float(row.get("aperture_local_path_score")),
            _safe_float(row.get("aperture_query_coverage")),
            _safe_float(row.get("semantic_score")),
            1.0 / max(rank_by_id.get(passage_id, 10**9), 1),
        )

    candidates = []
    for passage_id in catalog:
        if passage_id in incumbent_set:
            continue
        context = passage_context.get(str(passage_id), {}) or {}
        row = catalog.get(passage_id, {}) or {}
        if not (
            bool(context.get("content_available"))
            and str(context.get("content_fingerprint") or "")
        ):
            continue
        if not (
            row_lineage_tokens(row)
            or bool(row.get("aperture_local_proof_path", False))
            or _safe_float(row.get("aperture_query_coverage")) > 0.0
        ):
            continue
        candidates.append(int(passage_id))
    candidates.sort(key=candidate_priority, reverse=True)
    candidates = candidates[: max(1, int(max_single_candidates))]
    if not candidates:
        return []

    fixed_top2 = incumbent_ids[:2]
    beam: List[Tuple[int, ...]] = [tuple()]
    selected_states: List[Tuple[int, ...]] = []
    max_additions = min(3, 5 - len(fixed_top2))
    beam_width = max(2, int(max_pair_candidates))

    def materialize(selected: Sequence[int]) -> List[int]:
        result = list(dict.fromkeys([*fixed_top2, *selected]))
        result.extend(value for value in incumbent_ids if value not in set(result))
        return result[:5]

    def state_priority(selected: Sequence[int]) -> Tuple[float, ...]:
        after_ids = materialize(selected)
        lineage = lineage_replacement_profile(incumbent_ids, after_ids, catalog)
        rows = [catalog.get(value, {}) or {} for value in selected]
        slot_union = set().union(*(_row_slots(row) for row in rows)) if rows else set()
        return (
            float(bool(lineage.get("witness_preserved", False))),
            float(_safe_int(lineage.get("coverage_gain"), 0)),
            float(len(slot_union)),
            float(sum(bool(row.get("query_clause_complete")) for row in rows)),
            sum(_safe_float(row.get("query_clause_coverage")) for row in rows),
            sum(_safe_float(row.get("aperture_local_path_score")) for row in rows),
            -sum(rank_by_id.get(value, 10**9) for value in selected),
        )

    for _ in range(max_additions):
        expanded: Set[Tuple[int, ...]] = set()
        for selected in beam:
            for passage_id in candidates:
                if passage_id in selected:
                    continue
                expanded.add(tuple([*selected, passage_id]))
        valid = []
        for selected in expanded:
            after_ids = materialize(selected)
            if set(after_ids) == incumbent_set:
                continue
            lineage = lineage_replacement_profile(incumbent_ids, after_ids, catalog)
            if not bool(lineage.get("witness_preserved", False)):
                continue
            selected_rows = [catalog.get(value, {}) or {} for value in selected]
            adds_evidence = bool(
                _safe_int(lineage.get("coverage_gain"), 0) > 0
                or any(
                    bool(row.get("aperture_local_proof_path", False))
                    or _safe_float(row.get("query_clause_coverage")) > 0.0
                    for row in selected_rows
                )
            )
            if adds_evidence:
                valid.append(selected)
        valid.sort(key=state_priority, reverse=True)
        beam = valid[:beam_width]
        selected_states.extend(beam)
        if not beam:
            break

    actions: List[Dict[str, Any]] = []
    seen_results: Set[Tuple[int, ...]] = set()
    for selected in sorted(selected_states, key=state_priority, reverse=True):
        result_top5 = materialize(selected)
        result_key = tuple(result_top5)
        if result_key in seen_results:
            continue
        seen_results.add(result_key)
        displaced = [value for value in incumbent_ids if value not in result_top5]
        preserved = [value for value in result_top5 if value in incumbent_set]
        lineage = lineage_replacement_profile(incumbent_ids, result_top5, catalog)
        candidate_fingerprints = [
            str((passage_context.get(str(value), {}) or {}).get(
                "content_fingerprint"
            ) or "")
            for value in selected
        ]
        displaced_fingerprints = [
            str((passage_context.get(str(value), {}) or {}).get(
                "content_fingerprint"
            ) or "")
            for value in displaced
        ]
        if not all(candidate_fingerprints) or not all(displaced_fingerprints):
            continue
        sources = sorted({
            str(source)
            for value in selected
            for source in (catalog.get(value, {}) or {}).get("sources", []) or []
            if str(source)
        })
        actions.append({
            "action_kind": "replace_top5_clause_complete_set",
            "passage_ids": list(selected),
            "candidate_ids": list(selected),
            "preserved_incumbent_ids": preserved,
            "displaced_incumbent_ids": displaced,
            "result_top5_ids": result_top5,
            "final_ranks": [rank_by_id.get(value, 10**9) for value in selected],
            "sources": sources or ["clause_complete_lineage"],
            "expected_state_hash": str(incumbent.get("state_hash") or ""),
            "state_version": str(incumbent.get("state_version") or ""),
            "features": {
                "lineage_set_action": 1.0,
                "lineage_coverage_gain_log": math.log1p(
                    _safe_int(lineage.get("coverage_gain"), 0)
                ),
                "lineage_unique_obligation_loss_log": math.log1p(
                    _safe_int(lineage.get("unique_obligation_loss"), 0)
                ),
                "lineage_witness_preserved": 1.0,
            },
            "query_fingerprint": str(local_context.get("query_fingerprint") or ""),
            "candidate_fingerprints": candidate_fingerprints,
            "displaced_fingerprints": displaced_fingerprints,
            "content_state_fingerprint": content_state_fingerprint,
            "content_identity_complete": True,
            "semantic_identity_complete": True,
            "top2_membership_invariant": True,
            "top5_membership_invariant": False,
            "query_clause_profile": query_profile,
            "lineage_replacement": lineage,
        })
    return actions


def generate_executable_semijoin_stage_actions(
    query_plan_log: Dict[str, Any],
    *,
    base_module,
    stage: str,
    max_single_candidates: int = 24,
    max_pair_candidates: int = 12,
    enforce_cq: bool = True,
    enforce_semijoin: bool = True,
    max_fallback_per_signal: int = 20,
    candidate_mode: str = CANDIDATE_MODE,
) -> List[Dict[str, Any]]:
    normalized_stage = str(stage or "").strip().lower()
    normalized_mode = str(candidate_mode or "").strip().lower()
    actions = proof_nested.generate_proof_carrying_stage_actions(
        query_plan_log,
        base_module=base_module,
        stage=normalized_stage,
        max_single_candidates=max_single_candidates,
        max_pair_candidates=max_pair_candidates,
        enforce_cq=enforce_cq,
    )
    catalog = proof_nested.materialize_passage_proof_catalog(
        query_plan_log, base_module=base_module
    )
    if (
        normalized_mode == CLAUSE_COMPLETE_LINEAGE_MODE
        and normalized_stage == "top5"
    ):
        actions.extend(
            proof_nested._enrich_action(
                action,
                stage="top5",
                catalog=catalog,
                base_module=base_module,
            )
            for action in _clause_complete_set_actions(
                query_plan_log,
                catalog,
                max_single_candidates=max_single_candidates,
                max_pair_candidates=max_pair_candidates,
            )
        )
        proof_nested.rank_partitioned._annotate_action_pool(actions)
    names = feature_names(base_module, normalized_mode)
    compiled: List[Dict[str, Any]] = []
    for action in actions:
        profile = evaluate_executable_semijoin(
            action,
            query_plan_log,
            catalog,
            stage=normalized_stage,
            candidate_mode=normalized_mode,
        )
        features = dict(action.get("features", {}) or {})
        features.update(profile["features"])
        payload = {
            "schema": profile.get("schema_version", SCHEMA_VERSION),
            "base": str(action.get("action_fingerprint") or ""),
            "stage": normalized_stage,
            "semijoin": profile,
        }
        enriched = {
            **action,
            "features": {name: _safe_float(features.get(name)) for name in names},
            "executable_semijoin": profile,
            "content_schema_version": profile.get(
                "schema_version", SCHEMA_VERSION
            ),
            "action_fingerprint": hashlib.sha256(
                json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()[:24],
        }
        compiled.append(enriched)

    fallback_pool = [
        action for action in compiled
        if not (action.get("executable_semijoin", {}) or {}).get(
            "strong_executable", False
        )
        and not (
            normalized_mode == CLAUSE_COMPLETE_LINEAGE_MODE
            and not bool(((action.get("executable_semijoin", {}) or {}).get(
                "lineage_replacement", {}
            ) or {}).get("witness_preserved", False))
        )
    ]
    fallback_ids: Set[str] = set()
    fallback_limit = max(0, int(max_fallback_per_signal))
    fallback_signals = (
        "candidate_semantic_score_max",
        "candidate_query_coverage_max",
        "original_rank_inv",
        "semijoin_proof_coverage_margin",
        "semijoin_shared_binding_count_log",
    )
    if normalized_mode == CLAUSE_COMPLETE_LINEAGE_MODE:
        fallback_signals = (
            *fallback_signals,
            "clause_selected_path_coverage",
            "lineage_coverage_gain_log",
            "lineage_witness_preserved",
        )
    for signal in fallback_signals:
        ranked = sorted(
            fallback_pool,
            key=lambda action: (
                _safe_float((action.get("features", {}) or {}).get(signal)),
                str(action.get("action_fingerprint") or ""),
            ),
            reverse=True,
        )[:fallback_limit]
        fallback_ids.update(str(action.get("action_fingerprint") or "") for action in ranked)

    result: List[Dict[str, Any]] = []
    for action in compiled:
        profile = action.get("executable_semijoin", {}) or {}
        fingerprint = str(action.get("action_fingerprint") or "")
        fallback = bool(
            not profile.get("strong_executable", False)
            and fingerprint in fallback_ids
        )
        if fallback:
            original_reason = str(profile.get("reason") or "unknown")
            profile["eligible"] = True
            profile["fallback_eligible"] = True
            profile["fallback_reason"] = original_reason
            profile["reason"] = f"calibrated_fallback:{original_reason}"
            profile["admission_tier"] = "calibrated_fallback"
            profile["features"]["semijoin_fallback_eligible"] = 1.0
            action["features"]["semijoin_fallback_eligible"] = 1.0
        else:
            profile["fallback_eligible"] = False
        profile["fallback_budget_per_signal"] = fallback_limit
        profile["fallback_signals"] = list(fallback_signals)
        if not enforce_semijoin or profile.get("eligible", False):
            result.append(action)
    proof_nested.rank_partitioned._annotate_action_pool(result)
    return result


def required_passage_ids(
    query_plan_log: Dict[str, Any],
    *,
    base_module,
    max_single_candidates: int = 24,
    max_pair_candidates: int = 12,
    candidate_mode: str = CANDIDATE_MODE,
    enforce_cq: bool = True,
) -> Set[int]:
    result = proof_nested.required_passage_ids(
        query_plan_log,
        base_module=base_module,
        max_single_candidates=max_single_candidates,
        max_pair_candidates=max_pair_candidates,
        candidate_mode="proof_full_aperture_nested_topk",
        enforce_cq=enforce_cq,
    )
    if str(candidate_mode or "").strip().lower() == CLAUSE_COMPLETE_LINEAGE_MODE:
        catalog = proof_nested.materialize_passage_proof_catalog(
            query_plan_log, base_module=base_module
        )
        result.update(int(value) for value in catalog if int(value) >= 0)
    return result
