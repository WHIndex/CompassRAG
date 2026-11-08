import hashlib
import json
import math
import os
import re
from itertools import combinations
from typing import Any, Dict, Iterable, List, Optional, Tuple


FEATURE_NAMES = [
    "original_rank_inv",
    "original_rank_log",
    "original_rank_le_2",
    "original_rank_le_5",
    "original_rank_le_10",
    "original_rank_le_20",
    "original_rank_le_80",
    "dense_rank_inv",
    "dense_rank_le_5",
    "candidate_confidence",
    "candidate_score",
    "was_in_baseline",
    "matched_anchor",
    "matched_terminal",
    "answer_hit",
    "relation_hit_count",
    "relation_hit_any",
    "context_count",
    "context_support_evidence",
    "context_plan_precision_max",
    "context_lift_weight_max",
    "context_score_max",
    "context_has_anchor",
    "context_has_terminal",
    "context_has_relation",
    "llm_approved",
    "llm_score",
    "eg_plan_match_score",
    "eg_estimated_precision",
    "eg_candidate_precision",
    "eg_gt5_precision",
    "eg_selected_precision",
    "eg_action_tier_strong",
    "eg_action_tier_moderate",
    "eg_action_tier_weak",
    "signature_hit_count",
    "signature_hit_any",
    "feedback_signature_rescue",
    "matched_plan_count_log",
    "strict_executable_path_count_log",
    "executable_path_count_log",
    "query_condition_multihop",
    "query_condition_simple",
    "source_support",
    "source_evidence_grounded",
    "source_missing",
    "source_subgraph",
    "source_terminal",
    "source_failure_evidence",
    "source_schema_route",
    "source_schema_slot",
    "source_bind_probe",
    "source_path_executor",
    "source_evidence_index",
    "source_projected_graph",
    "source_structural",
    "dataset_2wikimultihopqa",
    "dataset_hotpotqa",
    "dataset_musique",
    "dataset_nq_rear",
    "dataset_popqa",
    "original_rank_gt_2",
    "original_rank_gt_5",
    "original_rank_3_to_5",
    "original_rank_6_to_20",
    "original_rank_gt_20",
    "effective_min_confidence",
    "candidate_margin_over_min_confidence",
    "matched_anchor_terminal_pair",
    "relation_coverage_score",
    "query_relation_coverage",
    "terminal_type_score",
    "path_length_log",
    "path_length_ge_2",
    "binding_score",
    "terminal_relation_score",
    "role_consistent",
    "complete_provenance",
    "has_proof_signal",
    "has_chain_signal",
    "has_source_certificate",
    "context_plan_precision_avg",
    "context_lift_weight_avg",
    "context_relation_count_log",
    "context_fact_count_log",
]


V209_AGGREGATE_FEATURE_NAMES = [
    "candidate_record_count_log",
    "candidate_source_count_log",
    "candidate_source_agreement",
    "candidate_confidence_mean",
    "candidate_score_mean",
    "candidate_in_incumbent_top2",
    "candidate_in_incumbent_top5",
    "action_target_top2",
    "action_target_top5",
    "strong_source_count_log",
]

V209_FEATURE_NAMES = FEATURE_NAMES + V209_AGGREGATE_FEATURE_NAMES


V210_SLATE_FEATURE_NAMES = [
    "final_rank_1_inv",
    "final_rank_1_log",
    "final_rank_1_le_2",
    "final_rank_1_le_5",
    "final_rank_1_le_20",
    "final_rank_2_inv",
    "final_rank_2_log",
    "final_rank_2_le_2",
    "final_rank_2_le_5",
    "final_rank_2_le_20",
    "action_size",
    "action_single_top2",
    "action_single_top5",
    "action_pair_top2",
    "pair_source_union_log",
    "pair_source_overlap",
    "pair_both_strong_source",
    "pair_both_proof_signal",
    "incumbent_top2_score_gap",
    "incumbent_top5_score_spread",
    "candidate_1_score_gap_to_slot",
    "candidate_2_score_gap_to_slot",
    "candidate_1_in_final_top5",
    "candidate_2_in_final_top5",
]

V210_FEATURE_NAMES = V209_FEATURE_NAMES + V210_SLATE_FEATURE_NAMES

V210_STATE_VERSION = "post_base_operators_v210"


V211_CONSTRAINED_FEATURE_NAMES = [
    "incumbent_preserved_count",
    "preserve_rank1_incumbent",
    "preserve_rank2_incumbent",
    "candidate_member_count",
    "all_new_top2_pair",
    "pair_complete_proof",
    "preserved_incumbent_score",
    "displaced_incumbent_score",
]

V211_FEATURE_NAMES = V210_FEATURE_NAMES + V211_CONSTRAINED_FEATURE_NAMES


V213_CANDIDATE_LOCAL_FEATURE_NAMES = [
    "candidate_content_available_fraction",
    "displaced_content_available_fraction",
    "candidate_query_coverage_max",
    "candidate_query_coverage_mean",
    "candidate_title_query_coverage_max",
    "candidate_title_query_coverage_mean",
    "candidate_query_bigram_coverage_max",
    "displaced_query_coverage_max",
    "displaced_title_query_coverage_max",
    "candidate_minus_displaced_query_coverage",
    "candidate_minus_displaced_title_coverage",
    "candidate_displaced_token_jaccard_max",
    "candidate_displaced_token_novelty_min",
    "candidate_pair_query_union_coverage",
    "candidate_pair_complementarity",
    "candidate_text_length_log_mean",
    "displaced_text_length_log_mean",
    "content_identity_complete",
    "local_comparator_proxy",
    "local_comparator_margin_to_group_mean",
    "local_comparator_rank_inv",
    "local_comparator_is_group_best",
]

V213_FEATURE_NAMES = V211_FEATURE_NAMES + V213_CANDIDATE_LOCAL_FEATURE_NAMES
V213_CONTENT_SCHEMA_VERSION = "content_addressed_candidate_local_v1"

_CONTENT_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def _content_tokens(value: Any) -> List[str]:
    return _CONTENT_TOKEN_PATTERN.findall(str(value or "").lower())


def _token_hash(token: str) -> int:
    return int(hashlib.sha1(token.encode("utf-8")).hexdigest()[:8], 16)


def _token_hashes(tokens: Iterable[str], limit: int = 384) -> List[int]:
    return sorted({_token_hash(token) for token in tokens if token})[: max(1, int(limit))]


def _set_overlap(left: Iterable[int], right: Iterable[int]) -> float:
    left_set = set(left or [])
    right_set = set(right or [])
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / max(len(left_set), 1)


def _set_jaccard(left: Iterable[int], right: Iterable[int]) -> float:
    left_set = set(left or [])
    right_set = set(right or [])
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / max(len(left_set | right_set), 1)


def build_content_addressed_passage_record(query: str, passage_text: str) -> Dict[str, Any]:
    """Build a compact, deterministic passage identity and query-local sketch."""
    normalized_text = " ".join(_content_tokens(passage_text))
    query_tokens = _content_tokens(query)
    passage_tokens = _content_tokens(passage_text)
    title = str(passage_text or "").split("\n", 1)[0]
    title_tokens = _content_tokens(title)
    query_bigrams = {
        _token_hash(f"{query_tokens[index]}::{query_tokens[index + 1]}")
        for index in range(max(0, len(query_tokens) - 1))
    }
    passage_bigrams = {
        _token_hash(f"{passage_tokens[index]}::{passage_tokens[index + 1]}")
        for index in range(max(0, len(passage_tokens) - 1))
    }
    query_hashes = _token_hashes(query_tokens)
    passage_hashes = _token_hashes(passage_tokens)
    title_hashes = _token_hashes(title_tokens)
    fingerprint = ""
    if normalized_text:
        fingerprint = hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()[:24]
    return {
        "content_fingerprint": fingerprint,
        "token_hashes": passage_hashes,
        "title_token_hashes": title_hashes,
        "query_token_hashes": query_hashes,
        "query_coverage": _set_overlap(query_hashes, passage_hashes),
        "title_query_coverage": _set_overlap(query_hashes, title_hashes),
        "query_bigram_coverage": _set_overlap(query_bigrams, passage_bigrams),
        "token_count_log": math.log1p(len(passage_tokens)),
        "content_available": bool(fingerprint),
    }


def build_candidate_local_context(
    query_plan_log: Dict[str, Any],
    get_passage_text,
    *,
    additional_passage_ids: Iterable[int] = (),
    get_passage_semantic_score=None,
) -> Dict[str, Any]:
    """Materialize content identities for incumbent and ontology candidates."""
    incumbent = query_plan_log.get("candidate_admission_incumbent", {}) or {}
    passage_ids = {
        _safe_int(value, -1)
        for value in incumbent.get("doc_ids", []) or []
        if _safe_int(value, -1) >= 0
    }
    for _, candidate in iter_query_plan_evidence_candidate_refs(query_plan_log):
        passage_idx = _safe_int(candidate.get("passage_idx"), -1)
        if passage_idx >= 0:
            passage_ids.add(passage_idx)
    for value in additional_passage_ids or ():
        passage_idx = _safe_int(value, -1)
        if passage_idx >= 0:
            passage_ids.add(passage_idx)
    query = str(query_plan_log.get("query") or "")
    passages: Dict[str, Any] = {}
    for passage_idx in sorted(passage_ids):
        try:
            passage_text = str(get_passage_text(passage_idx) or "")
        except Exception:
            passage_text = ""
        record = build_content_addressed_passage_record(
            query, passage_text
        )
        if get_passage_semantic_score is not None:
            try:
                semantic_score = _safe_float(
                    get_passage_semantic_score(passage_idx), 0.0
                )
                record["semantic_score"] = semantic_score
                record["semantic_score_available"] = True
            except Exception:
                record["semantic_score"] = 0.0
                record["semantic_score_available"] = False
        passages[str(passage_idx)] = record
    query_normalized = " ".join(_content_tokens(query))
    return {
        "schema_version": V213_CONTENT_SCHEMA_VERSION,
        "query_fingerprint": hashlib.sha256(
            query_normalized.encode("utf-8")
        ).hexdigest()[:24],
        "passages": passages,
        "semantic_scores_available": bool(get_passage_semantic_score is not None),
    }


def merge_candidate_local_context(
    existing: Optional[Dict[str, Any]],
    materialized: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Merge repeated materializations without downgrading richer records."""
    existing = dict(existing or {})
    materialized = dict(materialized or {})
    if not existing:
        return materialized
    if not materialized:
        return existing

    existing_query = str(existing.get("query_fingerprint") or "")
    materialized_query = str(materialized.get("query_fingerprint") or "")
    if existing_query and materialized_query and existing_query != materialized_query:
        return materialized

    existing_passages = existing.get("passages", {}) or {}
    materialized_passages = materialized.get("passages", {}) or {}
    passages: Dict[str, Any] = {
        str(key): dict(value or {})
        for key, value in existing_passages.items()
        if isinstance(value, dict)
    }
    for key, value in materialized_passages.items():
        if not isinstance(value, dict):
            continue
        key = str(key)
        previous = dict(passages.get(key, {}) or {})
        merged = {**previous, **value}
        if previous.get("content_available") and not value.get("content_available"):
            merged.update(previous)
        elif (
            previous.get("semantic_score_available")
            and not value.get("semantic_score_available")
        ):
            merged["semantic_score"] = previous.get("semantic_score", 0.0)
            merged["semantic_score_available"] = True
        passages[key] = merged

    return {
        **existing,
        **materialized,
        "passages": passages,
        "semantic_scores_available": bool(
            existing.get("semantic_scores_available")
            or materialized.get("semantic_scores_available")
            or any(
                passage.get("semantic_score_available")
                for passage in passages.values()
            )
        ),
    }


SOURCE_RISK_PRIORITY = [
    "full_aperture",
    "bind_probe",
    "evidence_grounded",
    "support",
    "schema_slot",
    "schema_route",
    "path_executor",
    "projected_graph",
    "evidence_index",
    "terminal",
    "subgraph",
    "missing",
    "failure_evidence",
    "structural",
]


def normalized_action_source_signature(sources: Iterable[Any]) -> str:
    """Map a provenance set to a stable source family for risk calibration."""
    normalized = sorted({str(value).strip().lower() for value in sources or [] if str(value).strip()})
    if not normalized:
        return "unknown"
    for source in SOURCE_RISK_PRIORITY:
        if source in normalized:
            return source
    return normalized[0]


def exact_action_source_signature(sources: Iterable[Any]) -> str:
    normalized = sorted({str(value).strip().lower() for value in sources or [] if str(value).strip()})
    return "+".join(normalized) if normalized else "unknown"


def source_risk_group_key(action: Dict[str, Any]) -> str:
    signature = str(
        action.get("source_signature")
        or normalized_action_source_signature(action.get("sources", []))
    ).strip().lower()
    preserved_rank = max(0, _safe_int(action.get("preserved_incumbent_rank"), 0))
    return f"{signature}|rank{preserved_rank}"


SOURCE_TO_LOG_KEY = {
    "support": "support_evidence_retriever",
    "evidence_grounded": "evidence_grounded_support_retriever",
    "missing": "missing_evidence_completer",
    "subgraph": "subgraph_retriever",
    "terminal": "terminal_evidence_retriever",
    "failure_evidence": "failure_evidence_retriever",
    "schema_route": "schema_route_retriever",
    "schema_slot": "schema_slot_recall_expander",
    "bind_probe": "bind_probe_join",
    "path_executor": "path_executor_retriever",
    "evidence_index": "evidence_index_retriever",
    "projected_graph": "projected_graph_retriever",
    "structural": "structural_candidate_retriever",
}

SOURCE_FEATURE_NAME = {
    "support": "source_support",
    "evidence_grounded": "source_evidence_grounded",
    "missing": "source_missing",
    "subgraph": "source_subgraph",
    "terminal": "source_terminal",
    "failure_evidence": "source_failure_evidence",
    "schema_route": "source_schema_route",
    "schema_slot": "source_schema_slot",
    "bind_probe": "source_bind_probe",
    "path_executor": "source_path_executor",
    "evidence_index": "source_evidence_index",
    "projected_graph": "source_projected_graph",
    "structural": "source_structural",
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return default
        return number
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def _safe_bool(value: Any) -> float:
    return 1.0 if bool(value) else 0.0


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def iter_query_plan_evidence_candidate_refs(
    query_plan_log: Dict[str, Any],
) -> Iterable[Tuple[str, Dict[str, Any]]]:
    if not isinstance(query_plan_log, dict):
        return
    for source, log_key in SOURCE_TO_LOG_KEY.items():
        source_log = query_plan_log.get(log_key, {}) or {}
        if not isinstance(source_log, dict):
            continue
        list_keys = ("added_candidates",)
        if source == "terminal":
            list_keys = ("selected_passages", "inserted_passages")
            verifier_log = source_log.get("verifier", {}) or {}
            for item in verifier_log.get("inserted_passages", []) or []:
                if isinstance(item, dict):
                    yield source, item
        for list_key in list_keys:
            for item in source_log.get(list_key, []) or []:
                if isinstance(item, dict):
                    yield source, item


def group_query_plan_evidence_candidate_refs(
    query_plan_log: Dict[str, Any],
) -> Dict[int, List[Tuple[str, Dict[str, Any]]]]:
    grouped: Dict[int, List[Tuple[str, Dict[str, Any]]]] = {}
    for source, candidate in iter_query_plan_evidence_candidate_refs(query_plan_log):
        passage_idx = _safe_int(candidate.get("passage_idx"), -1)
        if passage_idx < 0:
            continue
        grouped.setdefault(passage_idx, []).append((source, candidate))
    return grouped


def final_state_hash(doc_ids: Iterable[int], version: str = V210_STATE_VERSION) -> str:
    payload = {
        "version": str(version),
        "top5": [int(value) for value in list(doc_ids)[:5]],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:20]


def build_final_state_snapshot(
    doc_ids: Iterable[int],
    doc_scores: Iterable[float],
    candidate_window: int = 120,
    defer_decision: bool = False,
) -> Dict[str, Any]:
    all_ids = [int(value) for value in list(doc_ids)]
    all_scores = [float(value) for value in list(doc_scores)]
    window = max(5, int(candidate_window))
    top_ids = all_ids[:5]
    return {
        "doc_ids": top_ids,
        "doc_scores": all_scores[:5],
        "candidate_window_doc_ids": all_ids[:window],
        "candidate_window_doc_scores": all_scores[:window],
        "state_version": V210_STATE_VERSION,
        "state_hash": final_state_hash(top_ids, V210_STATE_VERSION),
        "defer_decision": bool(defer_decision),
    }


def _materialize_slate_action_ids(
    doc_ids: List[int], action: Dict[str, Any]
) -> List[int]:
    action_kind = str(action.get("action_kind") or "").strip().lower()
    if not action_kind and action.get("target"):
        action_kind = (
            "replace_top2_slot2"
            if str(action.get("target")).strip().lower() == "top2"
            else "replace_top5_slot5"
        )
    passage_ids = [
        _safe_int(value, -1)
        for value in action.get("passage_ids", []) or []
        if _safe_int(value, -1) >= 0
    ]
    if not passage_ids and action.get("passage_idx") is not None:
        passage_ids = [_safe_int(action.get("passage_idx"), -1)]
    passage_ids = [value for value in passage_ids if value >= 0]
    if action_kind in {"", "noop"} or not passage_ids:
        return list(doc_ids)

    remaining = [value for value in doc_ids if value not in set(passage_ids)]
    if action_kind in {"replace_top2_pair", "select_top2_subset"}:
        return passage_ids[:2] + remaining
    if action_kind == "replace_top5_proof_set":



        fixed_top2 = list(doc_ids[:2])
        proof_tail = [value for value in passage_ids if value not in fixed_top2]
        if len(fixed_top2) + len(proof_tail) > 5:
            return list(doc_ids)
        result_top5 = list(dict.fromkeys([*fixed_top2, *proof_tail]))
        result_top5.extend(
            value
            for value in doc_ids
            if value not in set(result_top5)
        )
        result_top5 = result_top5[:5]
        return result_top5 + [
            value for value in doc_ids if value not in set(result_top5)
        ]
    if action_kind == "replace_top5_slot5":
        return remaining[:4] + passage_ids[:1] + remaining[4:]
    return remaining[:1] + passage_ids[:1] + remaining[1:]


def materialize_final_state_slate_action(
    doc_ids: Iterable[int], action: Dict[str, Any]
) -> List[int]:
    """Materialize one action without mutating the incumbent slate."""
    return _materialize_slate_action_ids(
        [int(value) for value in list(doc_ids)], action
    )


def _rank_feature_values(rank: int, suffix: str) -> Dict[str, float]:
    safe_rank = max(int(rank), 1)
    return {
        f"final_rank_{suffix}_inv": 1.0 / safe_rank,
        f"final_rank_{suffix}_log": math.log1p(safe_rank),
        f"final_rank_{suffix}_le_2": 1.0 if safe_rank <= 2 else 0.0,
        f"final_rank_{suffix}_le_5": 1.0 if safe_rank <= 5 else 0.0,
        f"final_rank_{suffix}_le_20": 1.0 if safe_rank <= 20 else 0.0,
    }


def _candidate_sort_key(
    passage_idx: int,
    refs: List[Tuple[str, Dict[str, Any]]],
    final_rank: int,
) -> Tuple[float, float, float, float]:
    sources = {source for source, _ in refs}
    strong = len(sources.intersection({"bind_probe", "path_executor"}))
    proof = max(
        (_safe_bool(item.get("has_proof_signal") or item.get("proof_certificate")) for _, item in refs),
        default=0.0,
    )
    confidence = max((_safe_float(item.get("confidence"), 0.0) for _, item in refs), default=0.0)
    return float(strong), proof, confidence, -float(final_rank)


def generate_final_state_slate_actions(
    query_plan_log: Dict[str, Any],
    max_single_candidates: int = 10,
    max_pair_candidates: int = 6,
) -> List[Dict[str, Any]]:
    incumbent = query_plan_log.get("candidate_admission_incumbent", {}) or {}
    top5_ids = [_safe_int(value, -1) for value in incumbent.get("doc_ids", []) or []]
    window_ids = [
        _safe_int(value, -1)
        for value in incumbent.get("candidate_window_doc_ids", []) or top5_ids
    ]
    window_scores = [
        _safe_float(value, 0.0)
        for value in incumbent.get("candidate_window_doc_scores", [])
        or incumbent.get("doc_scores", [])
        or []
    ]
    rank_by_id = {passage_idx: index + 1 for index, passage_idx in enumerate(window_ids)}
    score_by_id = {
        passage_idx: window_scores[index] if index < len(window_scores) else 0.0
        for index, passage_idx in enumerate(window_ids)
    }
    grouped = group_query_plan_evidence_candidate_refs(query_plan_log)
    eligible = [
        (passage_idx, refs, rank_by_id[passage_idx])
        for passage_idx, refs in grouped.items()
        if passage_idx in rank_by_id and rank_by_id[passage_idx] > 2
    ]
    eligible.sort(key=lambda item: _candidate_sort_key(*item), reverse=True)
    eligible = eligible[: max(1, int(max_single_candidates))]
    state_hash = str(incumbent.get("state_hash") or final_state_hash(top5_ids))
    state_version = str(incumbent.get("state_version") or V210_STATE_VERSION)
    top_scores = [_safe_float(value, 0.0) for value in incumbent.get("doc_scores", []) or []]
    top2_gap = abs(top_scores[0] - top_scores[1]) if len(top_scores) >= 2 else 0.0
    top5_spread = abs(top_scores[0] - top_scores[4]) if len(top_scores) >= 5 else 0.0

    def action_features(
        action_kind: str,
        members: List[Tuple[int, List[Tuple[str, Dict[str, Any]]], int]],
    ) -> Dict[str, float]:
        target = "top5" if action_kind == "replace_top5_slot5" else "top2"
        member_features = [
            aggregate_passage_candidate_features(query_plan_log, passage_idx, refs, target)
            for passage_idx, refs, _ in members
        ]
        features: Dict[str, float] = {}
        for name in V209_FEATURE_NAMES:
            values = [row.get(name, 0.0) for row in member_features]
            features[name] = max(values) if values else 0.0
        for suffix, member in zip(("1", "2"), members):
            features.update(_rank_feature_values(member[2], suffix))
        if len(members) < 2:
            features.update(_rank_feature_values(10_000, "2"))
        source_sets = [
            {source for source, _ in refs}
            for _, refs, _ in members
        ]
        source_union = set().union(*source_sets) if source_sets else set()
        source_overlap = set.intersection(*source_sets) if len(source_sets) > 1 else set()
        both_strong = bool(members) and all(
            sources.intersection({"bind_probe", "path_executor"}) for sources in source_sets
        )
        both_proof = len(members) == 2 and all(
            max(
                (_safe_bool(item.get("has_proof_signal") or item.get("proof_certificate")) for _, item in refs),
                default=0.0,
            )
            > 0.0
            for _, refs, _ in members
        )
        slot_index = 4 if action_kind == "replace_top5_slot5" else 1
        slot_score = top_scores[slot_index] if slot_index < len(top_scores) else 0.0
        features.update({
            "action_size": float(len(members)),
            "action_single_top2": 1.0 if action_kind == "replace_top2_slot2" else 0.0,
            "action_single_top5": 1.0 if action_kind == "replace_top5_slot5" else 0.0,
            "action_pair_top2": 1.0 if action_kind == "replace_top2_pair" else 0.0,
            "pair_source_union_log": math.log1p(len(source_union)),
            "pair_source_overlap": min(len(source_overlap) / 3.0, 1.0),
            "pair_both_strong_source": 1.0 if both_strong else 0.0,
            "pair_both_proof_signal": 1.0 if both_proof else 0.0,
            "incumbent_top2_score_gap": top2_gap,
            "incumbent_top5_score_spread": top5_spread,
            "candidate_1_score_gap_to_slot": (
                score_by_id.get(members[0][0], 0.0) - slot_score if members else 0.0
            ),
            "candidate_2_score_gap_to_slot": (
                score_by_id.get(members[1][0], 0.0) - slot_score if len(members) > 1 else 0.0
            ),
            "candidate_1_in_final_top5": (
                1.0 if members and members[0][0] in top5_ids else 0.0
            ),
            "candidate_2_in_final_top5": (
                1.0 if len(members) > 1 and members[1][0] in top5_ids else 0.0
            ),
        })
        return {name: float(features.get(name, 0.0)) for name in V210_FEATURE_NAMES}

    actions: List[Dict[str, Any]] = []
    for passage_idx, refs, final_rank in eligible:
        members = [(passage_idx, refs, final_rank)]
        actions.append({
            "action_kind": "replace_top2_slot2",
            "passage_ids": [passage_idx],
            "final_ranks": [final_rank],
            "sources": sorted({source for source, _ in refs}),
            "expected_state_hash": state_hash,
            "state_version": state_version,
            "features": action_features("replace_top2_slot2", members),
        })
        if final_rank > 5:
            actions.append({
                "action_kind": "replace_top5_slot5",
                "passage_ids": [passage_idx],
                "final_ranks": [final_rank],
                "sources": sorted({source for source, _ in refs}),
                "expected_state_hash": state_hash,
                "state_version": state_version,
                "features": action_features("replace_top5_slot5", members),
            })

    pair_pool = eligible[: max(2, int(max_pair_candidates))]
    for left, right in combinations(pair_pool, 2):
        members = [left, right]
        passage_ids = [left[0], right[0]]
        if set(passage_ids) == set(top5_ids[:2]):
            continue
        actions.append({
            "action_kind": "replace_top2_pair",
            "passage_ids": passage_ids,
            "final_ranks": [left[2], right[2]],
            "sources": sorted(
                {source for _, refs, _ in members for source, _ in refs}
            ),
            "expected_state_hash": state_hash,
            "state_version": state_version,
            "features": action_features("replace_top2_pair", members),
        })
    return actions


def generate_incumbent_aware_slate_actions(
    query_plan_log: Dict[str, Any],
    max_single_candidates: int = 10,
    max_pair_candidates: int = 6,
) -> List[Dict[str, Any]]:
    """Compile displacement-aware actions from one immutable final state."""
    incumbent = query_plan_log.get("candidate_admission_incumbent", {}) or {}
    incumbent_ids = [
        _safe_int(value, -1) for value in incumbent.get("doc_ids", []) or []
    ]
    incumbent_scores = [
        _safe_float(value, 0.0) for value in incumbent.get("doc_scores", []) or []
    ]
    if len(incumbent_ids) < 2:
        return []
    grouped = group_query_plan_evidence_candidate_refs(query_plan_log)

    def complete_proof(passage_idx: int) -> bool:
        for _, candidate in grouped.get(passage_idx, []):
            if candidate.get("proof_certificate"):
                return True
            if (
                _safe_bool(candidate.get("has_proof_signal")) > 0.0
                and _safe_bool(candidate.get("complete_provenance")) > 0.0
                and _safe_bool(candidate.get("role_consistent")) > 0.0
            ):
                return True
        return False

    def extend_features(
        base: Dict[str, Any],
        preserved_count: int,
        preserve_rank1: bool,
        preserve_rank2: bool,
        candidate_count: int,
        all_new_pair: bool,
        pair_complete: bool,
        preserved_score: float,
        displaced_score: float,
    ) -> Dict[str, float]:
        features = {name: _safe_float(base.get(name), 0.0) for name in V210_FEATURE_NAMES}
        features.update({
            "incumbent_preserved_count": float(preserved_count),
            "preserve_rank1_incumbent": 1.0 if preserve_rank1 else 0.0,
            "preserve_rank2_incumbent": 1.0 if preserve_rank2 else 0.0,
            "candidate_member_count": float(candidate_count),
            "all_new_top2_pair": 1.0 if all_new_pair else 0.0,
            "pair_complete_proof": 1.0 if pair_complete else 0.0,
            "preserved_incumbent_score": float(preserved_score),
            "displaced_incumbent_score": float(displaced_score),
        })
        return {name: float(features.get(name, 0.0)) for name in V211_FEATURE_NAMES}

    compiled: List[Dict[str, Any]] = []
    for action in generate_final_state_slate_actions(
        query_plan_log,
        max_single_candidates=max_single_candidates,
        max_pair_candidates=max_pair_candidates,
    ):
        action_kind = str(action.get("action_kind") or "")
        candidate_ids = [
            _safe_int(value, -1) for value in action.get("passage_ids", []) or []
        ]
        candidate_ids = [value for value in candidate_ids if value >= 0]
        if action_kind == "replace_top2_slot2" and len(candidate_ids) == 1:
            candidate_idx = candidate_ids[0]
            for preserved_rank in (0, 1):
                preserved_idx = incumbent_ids[preserved_rank]
                if candidate_idx == preserved_idx:
                    continue
                displaced_rank = 1 - preserved_rank
                compiled.append({
                    **action,
                    "action_kind": "select_top2_subset",
                    "passage_ids": [preserved_idx, candidate_idx],
                    "candidate_ids": [candidate_idx],
                    "preserved_incumbent_ids": [preserved_idx],
                    "preserved_incumbent_rank": preserved_rank + 1,
                    "features": extend_features(
                        action.get("features", {}) or {},
                        preserved_count=1,
                        preserve_rank1=preserved_rank == 0,
                        preserve_rank2=preserved_rank == 1,
                        candidate_count=1,
                        all_new_pair=False,
                        pair_complete=False,
                        preserved_score=(
                            incumbent_scores[preserved_rank]
                            if preserved_rank < len(incumbent_scores)
                            else 0.0
                        ),
                        displaced_score=(
                            incumbent_scores[displaced_rank]
                            if displaced_rank < len(incumbent_scores)
                            else 0.0
                        ),
                    ),
                })
        elif action_kind == "replace_top5_slot5" and len(candidate_ids) == 1:
            compiled.append({
                **action,
                "candidate_ids": candidate_ids,
                "preserved_incumbent_ids": incumbent_ids[:4],
                "features": extend_features(
                    action.get("features", {}) or {},
                    preserved_count=4,
                    preserve_rank1=True,
                    preserve_rank2=True,
                    candidate_count=1,
                    all_new_pair=False,
                    pair_complete=False,
                    preserved_score=(
                        min(incumbent_scores[:4]) if incumbent_scores[:4] else 0.0
                    ),
                    displaced_score=(
                        incumbent_scores[4] if len(incumbent_scores) > 4 else 0.0
                    ),
                ),
            })
        elif action_kind == "replace_top2_pair" and len(candidate_ids) == 2:
            pair_complete = all(complete_proof(value) for value in candidate_ids)
            if not pair_complete:
                continue
            compiled.append({
                **action,
                "candidate_ids": candidate_ids,
                "preserved_incumbent_ids": [],
                "features": extend_features(
                    action.get("features", {}) or {},
                    preserved_count=0,
                    preserve_rank1=False,
                    preserve_rank2=False,
                    candidate_count=2,
                    all_new_pair=True,
                    pair_complete=True,
                    preserved_score=0.0,
                    displaced_score=(
                        max(incumbent_scores[:2]) if incumbent_scores[:2] else 0.0
                    ),
                ),
            })
    for action in compiled:
        sources = action.get("sources", []) or []
        action["source_signature"] = normalized_action_source_signature(sources)
        action["provenance_signature"] = exact_action_source_signature(sources)
        action["source_risk_group"] = source_risk_group_key(action)
    return compiled


def generate_content_addressed_candidate_actions(
    query_plan_log: Dict[str, Any],
    max_single_candidates: int = 10,
    max_pair_candidates: int = 6,
) -> List[Dict[str, Any]]:
    """Extend actions with stable identities and candidate-local comparisons."""
    context = query_plan_log.get("candidate_local_context", {}) or {}
    passage_context = context.get("passages", {}) or {}
    incumbent = query_plan_log.get("candidate_admission_incumbent", {}) or {}
    incumbent_ids = [
        _safe_int(value, -1) for value in incumbent.get("doc_ids", []) or []
    ]

    def record(passage_idx: int) -> Dict[str, Any]:
        value = passage_context.get(str(passage_idx), {}) or {}
        return value if isinstance(value, dict) else {}

    def displaced_ids(action: Dict[str, Any]) -> List[int]:
        action_kind = str(action.get("action_kind") or "")
        if action_kind == "replace_top5_slot5":
            return incumbent_ids[4:5]
        if action_kind == "replace_top2_pair":
            return incumbent_ids[:2]
        preserved = {
            _safe_int(value, -1)
            for value in action.get("preserved_incumbent_ids", []) or []
        }
        return [value for value in incumbent_ids[:2] if value not in preserved]

    def mean_value(rows: List[Dict[str, Any]], key: str) -> float:
        if not rows:
            return 0.0
        return sum(_safe_float(row.get(key), 0.0) for row in rows) / len(rows)

    actions = generate_incumbent_aware_slate_actions(
        query_plan_log,
        max_single_candidates=max_single_candidates,
        max_pair_candidates=max_pair_candidates,
    )
    compiled: List[Dict[str, Any]] = []
    for action in actions:
        candidate_ids = [
            _safe_int(value, -1)
            for value in action.get("candidate_ids", []) or []
            if _safe_int(value, -1) >= 0
        ]
        current_displaced_ids = displaced_ids(action)
        candidate_rows = [record(value) for value in candidate_ids]
        displaced_rows = [record(value) for value in current_displaced_ids]
        candidate_available = [
            row for row in candidate_rows if bool(row.get("content_available", False))
        ]
        displaced_available = [
            row for row in displaced_rows if bool(row.get("content_available", False))
        ]
        candidate_query_max = max(
            (_safe_float(row.get("query_coverage"), 0.0) for row in candidate_rows),
            default=0.0,
        )
        displaced_query_max = max(
            (_safe_float(row.get("query_coverage"), 0.0) for row in displaced_rows),
            default=0.0,
        )
        candidate_title_max = max(
            (_safe_float(row.get("title_query_coverage"), 0.0) for row in candidate_rows),
            default=0.0,
        )
        displaced_title_max = max(
            (_safe_float(row.get("title_query_coverage"), 0.0) for row in displaced_rows),
            default=0.0,
        )
        query_hashes = next(
            (
                row.get("query_token_hashes", []) or []
                for row in candidate_rows + displaced_rows
                if row.get("query_token_hashes")
            ),
            [],
        )
        candidate_union = set()
        for row in candidate_rows:
            candidate_union.update(row.get("token_hashes", []) or [])
        pair_union_coverage = _set_overlap(query_hashes, candidate_union)
        pair_complementarity = max(0.0, pair_union_coverage - candidate_query_max)
        similarities = [
            _set_jaccard(
                candidate.get("token_hashes", []) or [],
                displaced.get("token_hashes", []) or [],
            )
            for candidate in candidate_rows
            for displaced in displaced_rows
        ]
        max_similarity = max(similarities, default=0.0)
        candidate_fraction = len(candidate_available) / max(len(candidate_rows), 1)
        displaced_fraction = len(displaced_available) / max(len(displaced_rows), 1)
        identity_complete = bool(candidate_rows and displaced_rows) and (
            len(candidate_available) == len(candidate_rows)
            and len(displaced_available) == len(displaced_rows)
        )
        local_features = {
            "candidate_content_available_fraction": candidate_fraction,
            "displaced_content_available_fraction": displaced_fraction,
            "candidate_query_coverage_max": candidate_query_max,
            "candidate_query_coverage_mean": mean_value(candidate_rows, "query_coverage"),
            "candidate_title_query_coverage_max": candidate_title_max,
            "candidate_title_query_coverage_mean": mean_value(
                candidate_rows, "title_query_coverage"
            ),
            "candidate_query_bigram_coverage_max": max(
                (
                    _safe_float(row.get("query_bigram_coverage"), 0.0)
                    for row in candidate_rows
                ),
                default=0.0,
            ),
            "displaced_query_coverage_max": displaced_query_max,
            "displaced_title_query_coverage_max": displaced_title_max,
            "candidate_minus_displaced_query_coverage": (
                candidate_query_max - displaced_query_max
            ),
            "candidate_minus_displaced_title_coverage": (
                candidate_title_max - displaced_title_max
            ),
            "candidate_displaced_token_jaccard_max": max_similarity,
            "candidate_displaced_token_novelty_min": 1.0 - max_similarity,
            "candidate_pair_query_union_coverage": pair_union_coverage,
            "candidate_pair_complementarity": pair_complementarity,
            "candidate_text_length_log_mean": mean_value(
                candidate_rows, "token_count_log"
            ),
            "displaced_text_length_log_mean": mean_value(
                displaced_rows, "token_count_log"
            ),
            "content_identity_complete": 1.0 if identity_complete else 0.0,
        }
        features = {
            name: _safe_float((action.get("features", {}) or {}).get(name), 0.0)
            for name in V211_FEATURE_NAMES
        }
        features.update(local_features)
        candidate_fingerprints = [
            str(row.get("content_fingerprint") or "") for row in candidate_rows
        ]
        displaced_fingerprints = [
            str(row.get("content_fingerprint") or "") for row in displaced_rows
        ]
        incumbent_fingerprints = [
            str(record(value).get("content_fingerprint") or "")
            for value in incumbent_ids[:5]
        ]
        identity_payload = {
            "schema": V213_CONTENT_SCHEMA_VERSION,
            "query": str(context.get("query_fingerprint") or ""),
            "action_kind": str(action.get("action_kind") or ""),
            "source_group": str(action.get("source_risk_group") or ""),
            "preserved_rank": _safe_int(action.get("preserved_incumbent_rank"), 0),
            "candidates": sorted(candidate_fingerprints),
            "displaced": sorted(displaced_fingerprints),
        }
        action_fingerprint = hashlib.sha256(
            json.dumps(
                identity_payload, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()[:24]
        content_state_fingerprint = hashlib.sha256(
            json.dumps(
                incumbent_fingerprints, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()[:24]
        compiled.append({
            **action,
            "features": {name: float(features.get(name, 0.0)) for name in V213_FEATURE_NAMES},
            "content_schema_version": str(context.get("schema_version") or ""),
            "query_fingerprint": str(context.get("query_fingerprint") or ""),
            "candidate_fingerprints": candidate_fingerprints,
            "displaced_incumbent_ids": current_displaced_ids,
            "displaced_fingerprints": displaced_fingerprints,
            "content_state_fingerprint": content_state_fingerprint,
            "action_fingerprint": action_fingerprint,
            "content_identity_complete": identity_complete,
        })
    grouped_actions: Dict[Tuple[str, str, int], List[Dict[str, Any]]] = {}
    for action in compiled:
        key = (
            str(action.get("action_kind") or ""),
            str(action.get("source_risk_group") or ""),
            _safe_int(action.get("preserved_incumbent_rank"), 0),
        )
        grouped_actions.setdefault(key, []).append(action)
    for group_actions in grouped_actions.values():
        scored = []
        for action in group_actions:
            features = action.get("features", {}) or {}
            proxy = (
                _safe_float(features.get("candidate_query_coverage_max"), 0.0)
                + 0.50
                * _safe_float(
                    features.get("candidate_title_query_coverage_max"), 0.0
                )
                + 0.25
                * _safe_float(
                    features.get("candidate_query_bigram_coverage_max"), 0.0
                )
                + 0.25
                * _safe_float(features.get("candidate_pair_complementarity"), 0.0)
                + 0.20
                * _safe_float(
                    features.get("candidate_minus_displaced_query_coverage"), 0.0
                )
            )
            scored.append((proxy, action))
        mean_proxy = sum(value for value, _ in scored) / max(len(scored), 1)
        ordered = sorted(scored, key=lambda item: item[0], reverse=True)
        rank_by_identity = {
            id(action): rank for rank, (_, action) in enumerate(ordered, start=1)
        }
        for proxy, action in scored:
            rank = rank_by_identity[id(action)]
            features = action.get("features", {}) or {}
            features.update({
                "local_comparator_proxy": proxy,
                "local_comparator_margin_to_group_mean": proxy - mean_proxy,
                "local_comparator_rank_inv": 1.0 / rank,
                "local_comparator_is_group_best": 1.0 if rank == 1 else 0.0,
            })
            action["features"] = {
                name: float(features.get(name, 0.0)) for name in V213_FEATURE_NAMES
            }
    return compiled


def apply_passage_admission_action_to_slate(
    doc_ids: List[int],
    doc_scores: List[float],
    ranker_log: Dict[str, Any],
) -> Tuple[List[int], List[float], Dict[str, Any]]:
    action = ranker_log.get("selected_action", {}) or {}
    if not action or bool(ranker_log.get("no_op", True)):
        ranker_log["atomic_execution"] = {"executed": False, "reason": "no_op"}
        return doc_ids, doc_scores, ranker_log
    passage_ids = action.get("passage_ids", []) or []
    if not passage_ids and action.get("passage_idx") is not None:
        passage_ids = [action.get("passage_idx")]
    try:
        passage_ids = [int(value) for value in passage_ids]
    except (TypeError, ValueError):
        ranker_log["atomic_execution"] = {
            "executed": False,
            "reason": "invalid_passage_idx",
        }
        return doc_ids, doc_scores, ranker_log
    if not passage_ids or any(passage_idx not in doc_ids for passage_idx in passage_ids):
        ranker_log["atomic_execution"] = {
            "executed": False,
            "reason": "passage_not_in_retrieval_slate",
            "passage_ids": passage_ids,
        }
        ranker_log["promoted_count"] = 0
        ranker_log["committed_top2_count"] = 0
        ranker_log["committed_top5_count"] = 0
        ranker_log["no_op"] = True
        return doc_ids, doc_scores, ranker_log

    doc_ids = list(doc_ids)
    doc_scores = list(doc_scores)
    original_doc_ids = list(doc_ids)
    original_doc_scores = list(doc_scores)
    before_top5 = list(doc_ids[:5])
    state_version = str(action.get("state_version") or V210_STATE_VERSION)
    expected_hash = str(action.get("expected_state_hash") or "")
    observed_hash = final_state_hash(before_top5, state_version)
    if expected_hash and expected_hash != observed_hash:
        ranker_log["atomic_execution"] = {
            "executed": False,
            "reason": "stale_final_state",
            "expected_state_hash": expected_hash,
            "observed_state_hash": observed_hash,
        }
        ranker_log["state_conflict"] = True
        ranker_log["promoted_count"] = 0
        ranker_log["no_op"] = True
        return doc_ids, doc_scores, ranker_log

    old_ranks = {passage_idx: doc_ids.index(passage_idx) + 1 for passage_idx in passage_ids}
    doc_ids = _materialize_slate_action_ids(doc_ids, action)
    if bool(action.get("top2_membership_invariant", False)) and (
        doc_ids[:2] != before_top5[:2]
    ):
        ranker_log["atomic_execution"] = {
            "executed": False,
            "reason": "top2_membership_invariant_violation",
            "before_top2": before_top5[:2],
            "proposed_top2": list(doc_ids[:2]),
        }
        ranker_log["promoted_count"] = 0
        ranker_log["committed_top2_count"] = 0
        ranker_log["committed_top5_count"] = 0
        ranker_log["no_op"] = True
        return original_doc_ids, original_doc_scores, ranker_log
    if bool(action.get("top5_membership_invariant", False)) and (
        set(doc_ids[:5]) != set(before_top5)
    ):
        ranker_log["atomic_execution"] = {
            "executed": False,
            "reason": "top5_membership_invariant_violation",
            "before_top5": before_top5,
            "proposed_top5": list(doc_ids[:5]),
        }
        ranker_log["promoted_count"] = 0
        ranker_log["committed_top2_count"] = 0
        ranker_log["committed_top5_count"] = 0
        ranker_log["no_op"] = True
        return original_doc_ids, original_doc_scores, ranker_log
    ranked_scores = sorted(doc_scores, reverse=True)
    doc_scores = ranked_scores[: len(doc_ids)]
    after_top5 = list(doc_ids[:5])
    action_kind = str(action.get("action_kind") or "").strip().lower()
    if not action_kind:
        action_kind = (
            "replace_top2_slot2"
            if str(action.get("target") or "top5").strip().lower() == "top2"
            else "replace_top5_slot5"
        )
    target = (
        "top5"
        if action_kind in {"replace_top5_slot5", "replace_top5_proof_set"}
        else "top2"
    )
    ranker_log["atomic_execution"] = {
        "executed": True,
        "action_kind": action_kind,
        "target": target,
        "passage_ids": passage_ids,
        "old_ranks": old_ranks,
        "new_ranks": {passage_idx: doc_ids.index(passage_idx) + 1 for passage_idx in passage_ids},
        "expected_state_hash": expected_hash,
        "observed_state_hash": observed_hash,
        "before_top5": before_top5,
        "after_top5": after_top5,
        "changed_top2": before_top5[:2] != after_top5[:2],
        "changed_top5": before_top5 != after_top5,
    }
    if len(passage_ids) == 1:
        passage_idx = passage_ids[0]
        ranker_log["atomic_execution"].update({
            "passage_idx": passage_idx,
            "old_rank": old_ranks[passage_idx],
            "new_rank": doc_ids.index(passage_idx) + 1,
        })
    return doc_ids, doc_scores, ranker_log


def _context_list(candidate: Dict[str, Any]) -> List[Dict[str, Any]]:
    contexts = candidate.get("contexts", []) or []
    if isinstance(contexts, list):
        return [item for item in contexts if isinstance(item, dict)]
    return []


def _max_context_value(contexts: List[Dict[str, Any]], keys: Tuple[str, ...]) -> float:
    value = 0.0
    for context in contexts:
        for key in keys:
            value = max(value, _safe_float(context.get(key), 0.0))
    return value


def _has_context_key(contexts: List[Dict[str, Any]], keys: Tuple[str, ...]) -> float:
    for context in contexts:
        for key in keys:
            if context.get(key):
                return 1.0
    return 0.0


def _avg_context_value(contexts: List[Dict[str, Any]], keys: Tuple[str, ...]) -> float:
    values: List[float] = []
    for context in contexts:
        for key in keys:
            if key in context:
                values.append(_safe_float(context.get(key), 0.0))
                break
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _count_context_values(contexts: List[Dict[str, Any]], keys: Tuple[str, ...]) -> int:
    count = 0
    for context in contexts:
        for key in keys:
            value = context.get(key)
            if isinstance(value, (list, tuple, set)):
                count += len(value)
                break
            if isinstance(value, dict):
                count += len(value)
                break
            if value:
                count += 1
                break
    return count


def _nested_float(mapping: Dict[str, Any], paths: Tuple[Tuple[str, ...], ...]) -> float:
    best = 0.0
    for path in paths:
        cursor: Any = mapping
        for key in path:
            if not isinstance(cursor, dict):
                cursor = None
                break
            cursor = cursor.get(key)
        best = max(best, _safe_float(cursor, 0.0))
    return best


def _truthy_nested(mapping: Dict[str, Any], paths: Tuple[Tuple[str, ...], ...]) -> float:
    for path in paths:
        cursor: Any = mapping
        for key in path:
            if not isinstance(cursor, dict):
                cursor = None
                break
            cursor = cursor.get(key)
        if cursor:
            return 1.0
    return 0.0


def _contains_key_fragment(value: Any, fragments: Tuple[str, ...]) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).lower()
            if any(fragment in lowered for fragment in fragments) and bool(child):
                return True
            if _contains_key_fragment(child, fragments):
                return True
    elif isinstance(value, list):
        for child in value:
            if _contains_key_fragment(child, fragments):
                return True
    return False


def _dataset_features(query_plan_log: Dict[str, Any]) -> Dict[str, float]:
    dataset = str(
        query_plan_log.get("dataset")
        or query_plan_log.get("_dataset")
        or query_plan_log.get("dataset_name")
        or ""
    ).strip().lower()
    return {
        "dataset_2wikimultihopqa": 1.0 if dataset == "2wikimultihopqa" else 0.0,
        "dataset_hotpotqa": 1.0 if dataset == "hotpotqa" else 0.0,
        "dataset_musique": 1.0 if dataset == "musique" else 0.0,
        "dataset_nq_rear": 1.0 if dataset == "nq_rear" else 0.0,
        "dataset_popqa": 1.0 if dataset == "popqa" else 0.0,
    }


def _query_condition_features(query_plan_log: Dict[str, Any]) -> Tuple[float, float]:
    condition = query_plan_log.get("query_condition", {}) or {}
    if not isinstance(condition, dict):
        return 0.0, 0.0
    text = " ".join(
        str(condition.get(key, "") or "")
        for key in ("query_type", "intent", "answer_type", "family", "mode")
    ).lower()
    multihop_terms = ("multi", "bridge", "comparison", "composition", "hop")
    simple_terms = ("simple", "factoid", "single")
    multihop = 1.0 if any(term in text for term in multihop_terms) else 0.0
    simple = 1.0 if any(term in text for term in simple_terms) else 0.0
    return multihop, simple


def extract_query_plan_candidate_features(
    query_plan_log: Dict[str, Any],
    candidate: Dict[str, Any],
    source: str,
) -> Dict[str, float]:
    original_rank = _safe_int(candidate.get("original_rank"), 10_000)
    if original_rank <= 0:
        original_rank = 10_000
    dense_rank = _safe_int(candidate.get("dense_rank"), 10_000)
    if dense_rank <= 0:
        dense_rank = 10_000
    contexts = _context_list(candidate)
    relation_hits = candidate.get("relation_hits", []) or []
    if not isinstance(relation_hits, list):
        relation_hits = []
    llm_log = candidate.get("llm_evidence_verifier", {}) or {}
    if not isinstance(llm_log, dict):
        llm_log = {}
    signature_hits = candidate.get("signature_hits", []) or []
    if not isinstance(signature_hits, list):
        signature_hits = []
    probe = candidate.get("probe", {}) or {}
    if not isinstance(probe, dict):
        probe = {}
    feedback = probe.get("retrieval_feedback", {}) or {}
    if not isinstance(feedback, dict):
        feedback = {}
    action_tier = str(candidate.get("action_tier") or (probe.get("action_policy", {}) or {}).get("tier") or "").lower()
    query_multihop, query_simple = _query_condition_features(query_plan_log)
    effective_min_confidence = _safe_float(candidate.get("effective_min_confidence"), 0.0)
    confidence = _safe_float(candidate.get("confidence"), 0.0)
    path_length = max(
        _safe_float(candidate.get("path_length"), 0.0),
        _max_context_value(contexts, ("path_length", "chain_length")),
    )
    relation_coverage_score = max(
        min(1.0, float(len(relation_hits)) / 3.0),
        _safe_float(candidate.get("relation_score"), 0.0),
        _safe_float(candidate.get("query_relation_coverage"), 0.0),
        _max_context_value(
            contexts,
            (
                "relation_score",
                "relation_overlap",
                "query_relation_coverage",
                "terminal_relation_score",
                "terminal_relation_overlap",
            ),
        ),
    )
    binding = candidate.get("binding", {}) or {}
    if not isinstance(binding, dict):
        binding = {}
    binding_score = max(
        _safe_float(candidate.get("binding_score"), 0.0),
        _safe_float(candidate.get("matched_binding"), 0.0),
        _nested_float(binding, (("score",), ("match_score",), ("support_score",))),
        _max_context_value(contexts, ("binding_score", "binding_match_score", "source_binding_score")),
    )
    terminal_type_score = max(
        _safe_float(candidate.get("terminal_type_score"), 0.0),
        _max_context_value(contexts, ("terminal_type_score", "answer_type_score")),
    )
    terminal_relation_score = max(
        _safe_float(candidate.get("terminal_relation_score"), 0.0),
        _safe_float(candidate.get("relation_score"), 0.0),
        _max_context_value(contexts, ("terminal_relation_score", "terminal_relation_overlap")),
    )
    complete_provenance = max(
        _safe_bool(candidate.get("complete_provenance")),
        _safe_bool(_safe_int(candidate.get("binding_source_passage_idx"), -1) >= 0),
        _truthy_nested(candidate, (("provenance",), ("proof",), ("certificate",))),
        _has_context_key(
            contexts,
            (
                "provenance",
                "proof",
                "certificate",
                "source_passage_idx",
                "binding_source_passage_idx",
            ),
        ),
    )
    has_proof_signal = max(
        _truthy_nested(candidate, (("proof",), ("certificate",), ("typed_proof",), ("proof_certificate",))),
        1.0 if _contains_key_fragment(candidate, ("proof", "certificate")) else 0.0,
        1.0 if any(_contains_key_fragment(context, ("proof", "certificate")) for context in contexts) else 0.0,
    )
    has_chain_signal = max(
        1.0 if path_length >= 2.0 else 0.0,
        1.0 if _contains_key_fragment(candidate, ("chain", "path")) else 0.0,
        1.0 if any(_contains_key_fragment(context, ("chain", "path")) for context in contexts) else 0.0,
    )
    has_source_certificate = max(
        complete_provenance,
        1.0 if candidate.get("was_in_baseline") else 0.0,
        1.0 if candidate.get("matched_anchor") and (candidate.get("matched_terminal") or relation_hits) else 0.0,
    )
    features = {
        "original_rank_inv": 1.0 / float(original_rank),
        "original_rank_log": math.log1p(float(original_rank)),
        "original_rank_le_2": 1.0 if original_rank <= 2 else 0.0,
        "original_rank_le_5": 1.0 if original_rank <= 5 else 0.0,
        "original_rank_le_10": 1.0 if original_rank <= 10 else 0.0,
        "original_rank_le_20": 1.0 if original_rank <= 20 else 0.0,
        "original_rank_le_80": 1.0 if original_rank <= 80 else 0.0,
        "dense_rank_inv": 1.0 / float(dense_rank),
        "dense_rank_le_5": 1.0 if dense_rank <= 5 else 0.0,
        "candidate_confidence": confidence,
        "candidate_score": _safe_float(candidate.get("score"), 0.0),
        "was_in_baseline": _safe_bool(candidate.get("was_in_baseline")),
        "matched_anchor": _safe_bool(candidate.get("matched_anchor")),
        "matched_terminal": _safe_bool(candidate.get("matched_terminal")),
        "answer_hit": _safe_bool(candidate.get("answer_hit")),
        "relation_hit_count": float(len(relation_hits)),
        "relation_hit_any": 1.0 if relation_hits else 0.0,
        "context_count": float(len(contexts)),
        "context_support_evidence": _has_context_key(contexts, ("support_evidence",)),
        "context_plan_precision_max": _max_context_value(
            contexts,
            ("plan_precision", "effective_precision", "plan_effective_precision"),
        ),
        "context_lift_weight_max": _max_context_value(
            contexts,
            ("lift_weight", "plan_lift_weight_scale", "weight"),
        ),
        "context_score_max": _max_context_value(contexts, ("score", "raw_score")),
        "context_has_anchor": _has_context_key(contexts, ("anchor_entity", "anchor", "subject")),
        "context_has_terminal": _has_context_key(contexts, ("end_entity", "terminal", "object")),
        "context_has_relation": _has_context_key(contexts, ("relations", "relation", "relation_path")),
        "llm_approved": _safe_bool(llm_log.get("approved")),
        "llm_score": _safe_float(llm_log.get("score"), 0.0),
        "eg_plan_match_score": _safe_float(probe.get("plan_match_score"), 0.0),
        "eg_estimated_precision": _safe_float(probe.get("estimated_precision"), 0.0),
        "eg_candidate_precision": _safe_float(feedback.get("candidate_precision"), 0.0),
        "eg_gt5_precision": _safe_float(feedback.get("candidate_gt5_precision"), 0.0),
        "eg_selected_precision": _safe_float(feedback.get("selected_precision"), 0.0),
        "eg_action_tier_strong": 1.0 if action_tier == "strong" else 0.0,
        "eg_action_tier_moderate": 1.0 if action_tier == "moderate" else 0.0,
        "eg_action_tier_weak": 1.0 if action_tier == "weak" else 0.0,
        "signature_hit_count": float(len(signature_hits)),
        "signature_hit_any": 1.0 if signature_hits else 0.0,
        "feedback_signature_rescue": _safe_bool(candidate.get("feedback_signature_rescue")),
        "matched_plan_count_log": math.log1p(_safe_float(query_plan_log.get("matched_plan_count"), 0.0)),
        "strict_executable_path_count_log": math.log1p(_safe_float(query_plan_log.get("strict_executable_path_count"), 0.0)),
        "executable_path_count_log": math.log1p(_safe_float(query_plan_log.get("executable_path_count"), 0.0)),
        "query_condition_multihop": query_multihop,
        "query_condition_simple": query_simple,
        "source_support": 1.0 if source == "support" else 0.0,
        "source_evidence_grounded": 1.0 if source == "evidence_grounded" else 0.0,
        "source_missing": 1.0 if source == "missing" else 0.0,
        "source_subgraph": 1.0 if source == "subgraph" else 0.0,
        "source_terminal": 1.0 if source == "terminal" else 0.0,
        "source_failure_evidence": 1.0 if source == "failure_evidence" else 0.0,
        "source_schema_route": 1.0 if source == "schema_route" else 0.0,
        "source_schema_slot": 1.0 if source == "schema_slot" else 0.0,
        "source_bind_probe": 1.0 if source == "bind_probe" else 0.0,
        "source_path_executor": 1.0 if source == "path_executor" else 0.0,
        "source_evidence_index": 1.0 if source == "evidence_index" else 0.0,
        "source_projected_graph": 1.0 if source == "projected_graph" else 0.0,
        "source_structural": 1.0 if source == "structural" else 0.0,
        "original_rank_gt_2": 1.0 if original_rank > 2 else 0.0,
        "original_rank_gt_5": 1.0 if original_rank > 5 else 0.0,
        "original_rank_3_to_5": 1.0 if 2 < original_rank <= 5 else 0.0,
        "original_rank_6_to_20": 1.0 if 5 < original_rank <= 20 else 0.0,
        "original_rank_gt_20": 1.0 if original_rank > 20 else 0.0,
        "effective_min_confidence": effective_min_confidence,
        "candidate_margin_over_min_confidence": confidence - effective_min_confidence,
        "matched_anchor_terminal_pair": (
            1.0 if candidate.get("matched_anchor") and candidate.get("matched_terminal") else 0.0
        ),
        "relation_coverage_score": relation_coverage_score,
        "query_relation_coverage": _safe_float(candidate.get("query_relation_coverage"), 0.0),
        "terminal_type_score": terminal_type_score,
        "path_length_log": math.log1p(path_length),
        "path_length_ge_2": 1.0 if path_length >= 2.0 else 0.0,
        "binding_score": binding_score,
        "terminal_relation_score": terminal_relation_score,
        "role_consistent": _safe_bool(candidate.get("role_consistent")),
        "complete_provenance": complete_provenance,
        "has_proof_signal": has_proof_signal,
        "has_chain_signal": has_chain_signal,
        "has_source_certificate": has_source_certificate,
        "context_plan_precision_avg": _avg_context_value(
            contexts,
            ("plan_precision", "effective_precision", "plan_effective_precision"),
        ),
        "context_lift_weight_avg": _avg_context_value(
            contexts,
            ("lift_weight", "plan_lift_weight_scale", "weight"),
        ),
        "context_relation_count_log": math.log1p(_count_context_values(contexts, ("relations", "relation_path"))),
        "context_fact_count_log": math.log1p(_count_context_values(contexts, ("facts", "support_facts", "triples"))),
    }
    features.update(_dataset_features(query_plan_log))
    return {name: float(features.get(name, 0.0)) for name in FEATURE_NAMES}


def aggregate_passage_candidate_features(
    query_plan_log: Dict[str, Any],
    passage_idx: int,
    candidate_refs: List[Tuple[str, Dict[str, Any]]],
    target: str,
) -> Dict[str, float]:
    feature_rows = [
        extract_query_plan_candidate_features(query_plan_log, candidate, source)
        for source, candidate in candidate_refs
    ]
    if not feature_rows:
        return {name: 0.0 for name in V209_FEATURE_NAMES}

    aggregated: Dict[str, float] = {}
    for name in FEATURE_NAMES:
        values = [float(row.get(name, 0.0)) for row in feature_rows]
        if name == "original_rank_log":
            aggregated[name] = min(values)
        else:
            aggregated[name] = max(values)

    sources = {str(source).strip().lower() for source, _ in candidate_refs if str(source).strip()}
    confidences = [_safe_float(candidate.get("confidence"), 0.0) for _, candidate in candidate_refs]
    scores = [_safe_float(candidate.get("score"), 0.0) for _, candidate in candidate_refs]
    incumbent = query_plan_log.get("candidate_admission_incumbent", {}) or {}
    incumbent_ids = [
        _safe_int(value, -1)
        for value in incumbent.get("doc_ids", []) or []
    ]
    strong_sources = sources.intersection({"bind_probe", "path_executor"})
    aggregated.update({
        "candidate_record_count_log": math.log1p(len(candidate_refs)),
        "candidate_source_count_log": math.log1p(len(sources)),
        "candidate_source_agreement": min(len(sources) / 4.0, 1.0),
        "candidate_confidence_mean": sum(confidences) / max(len(confidences), 1),
        "candidate_score_mean": sum(scores) / max(len(scores), 1),
        "candidate_in_incumbent_top2": 1.0 if passage_idx in incumbent_ids[:2] else 0.0,
        "candidate_in_incumbent_top5": 1.0 if passage_idx in incumbent_ids[:5] else 0.0,
        "action_target_top2": 1.0 if target == "top2" else 0.0,
        "action_target_top5": 1.0 if target == "top5" else 0.0,
        "strong_source_count_log": math.log1p(len(strong_sources)),
    })
    return {name: float(aggregated.get(name, 0.0)) for name in V209_FEATURE_NAMES}


def feature_vector(features: Dict[str, float], feature_names: List[str] = None) -> List[float]:
    names = feature_names or FEATURE_NAMES
    return [float(features.get(name, 0.0)) for name in names]


class QueryPlanSupervisedEvidenceRanker:
    def __init__(
        self,
        model_path: str,
        min_score: float = 0.55,
        promote_confidence: float = 0.90,
        reject_below: float = 0.20,
        rejected_confidence: float = 0.05,
        max_candidates: int = 16,
    ):
        self.model_path = model_path
        self.min_score = float(min_score)
        self.promote_confidence = float(promote_confidence)
        self.reject_below = float(reject_below)
        self.rejected_confidence = float(rejected_confidence)
        self.max_candidates = int(max_candidates)
        self.model = self._load_model(model_path)

    @staticmethod
    def _load_model(model_path: str) -> Dict[str, Any]:
        if not model_path or not os.path.exists(model_path):
            return {}
        with open(model_path, "r", encoding="utf-8") as model_file:
            model = json.load(model_file)
        return model if isinstance(model, dict) else {}

    def is_ready(self) -> bool:
        if self.model.get("type") in {
            "passage_counterfactual_admission_optimizer",
            "final_state_transactional_slate_optimizer",
            "incumbent_aware_constrained_slate_optimizer",
            "source_conditional_grouped_oof_slate_optimizer",
            "content_addressed_candidate_local_optimizer",
        }:
            return bool(self.model.get("sections")) and bool(self.model.get("feature_names"))
        return bool(self.model.get("weights")) and bool(self.model.get("feature_names"))

    def _metadata(self) -> Dict[str, Any]:
        metadata = self.model.get("metadata", {}) or {}
        return metadata if isinstance(metadata, dict) else {}

    def _metadata_set(self, name: str) -> set:
        value = self._metadata().get(name, []) or []
        if isinstance(value, str):
            return {item.strip().lower() for item in value.split(",") if item.strip()}
        if isinstance(value, (list, tuple, set)):
            return {str(item).strip().lower() for item in value if str(item).strip()}
        return set()

    def _metadata_dict(self, name: str) -> Dict[str, Any]:
        value = self._metadata().get(name, {}) or {}
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _dataset_name(query_plan_log: Dict[str, Any]) -> str:
        return str(
            query_plan_log.get("dataset")
            or query_plan_log.get("_dataset")
            or query_plan_log.get("dataset_name")
            or ""
        ).strip().lower()

    def _threshold_for_dataset(self, name: str, dataset: str, default: float) -> float:
        metadata = self._metadata()
        for container in (metadata, self.model):
            by_dataset = container.get(f"{name}_by_dataset", {}) or {}
            if isinstance(by_dataset, dict):
                for key in (dataset, dataset.lower(), dataset.upper(), "ALL", "all", "__default__"):
                    if key in by_dataset:
                        return _safe_float(by_dataset.get(key), default)
            if name in container:
                return _safe_float(container.get(name), default)
        return float(default)

    def _dataset_is_enabled(self, dataset: str) -> Tuple[bool, str]:
        metadata = self._metadata()
        if not bool(metadata.get("disable_untrained_datasets")):
            return True, ""
        trained_datasets = self._metadata_set("trained_datasets")
        if not trained_datasets or not dataset:
            return True, ""
        if dataset.lower() in trained_datasets:
            return True, ""
        return False, "dataset_not_seen_in_train_split"

    def _score_weight(self) -> float:
        metadata = self._metadata()
        return _safe_float(
            metadata.get("candidate_score_weight", metadata.get("score_weight", self.model.get("score_weight", 0.0))),
            0.0,
        )

    def _promote_confidence_for_dataset(self, dataset: str) -> float:
        return self._threshold_for_dataset("promote_confidence", dataset, self.promote_confidence)

    @staticmethod
    def _source_is_trained(source: str, feature_names: List[str]) -> bool:
        source_feature = SOURCE_FEATURE_NAME.get(source)
        return bool(source_feature and source_feature in feature_names)

    def _source_cap(self, source: str) -> int:
        caps = self._metadata_dict("source_caps")
        if source in caps:
            return max(0, _safe_int(caps.get(source), self.max_candidates))
        return max(0, self.max_candidates)

    def _source_risk_gate(
        self,
        source: str,
        features: Dict[str, float],
    ) -> Tuple[bool, int, List[str]]:
        metadata = self._metadata()
        if not bool(metadata.get("source_risk_gate")):
            return True, 0, []
        low_certification_sources = self._metadata_set("low_certification_sources")
        if source.lower() not in low_certification_sources:
            return True, 0, []

        signals: List[str] = []
        if max(features.get("matched_anchor", 0.0), features.get("context_has_anchor", 0.0)) > 0.0:
            signals.append("anchor")
        if max(
            features.get("matched_terminal", 0.0),
            features.get("relation_hit_any", 0.0),
            1.0 if features.get("relation_coverage_score", 0.0) >= 0.25 else 0.0,
            1.0 if features.get("terminal_relation_score", 0.0) >= 0.25 else 0.0,
        ) > 0.0:
            signals.append("terminal_or_relation")
        if max(features.get("has_proof_signal", 0.0), features.get("has_chain_signal", 0.0)) > 0.0:
            signals.append("proof_or_chain")
        if max(
            features.get("has_source_certificate", 0.0),
            features.get("complete_provenance", 0.0),
            features.get("was_in_baseline", 0.0),
        ) > 0.0:
            signals.append("provenance")
        if max(
            features.get("answer_hit", 0.0),
            features.get("llm_approved", 0.0),
            1.0 if features.get("llm_score", 0.0) >= 0.60 else 0.0,
        ) > 0.0:
            signals.append("text_or_reader")
        if (
            features.get("effective_min_confidence", 0.0) > 0.0
            and features.get("candidate_margin_over_min_confidence", -1.0) >= 0.0
        ) or features.get("candidate_confidence", 0.0) >= 0.86:
            signals.append("confidence_margin")

        required = max(1, _safe_int(metadata.get("source_risk_min_signals"), 2))
        return len(signals) >= required, len(signals), signals

    def score_features(self, features: Dict[str, float]) -> float:
        names = list(self.model.get("feature_names", FEATURE_NAMES) or FEATURE_NAMES)
        weights = [float(value) for value in self.model.get("weights", [])]
        means = [float(value) for value in self.model.get("means", [0.0] * len(names))]
        stds = [float(value) if float(value) != 0.0 else 1.0 for value in self.model.get("stds", [1.0] * len(names))]
        bias = float(self.model.get("bias", 0.0) or 0.0)
        values = feature_vector(features, names)
        total = bias
        for index, value in enumerate(values):
            if index >= len(weights):
                break
            total += weights[index] * ((value - means[index]) / stds[index])
        return _sigmoid(total)

    @staticmethod
    def _score_model_head(head: Dict[str, Any], features: Dict[str, float]) -> float:
        names = list(head.get("feature_names", V209_FEATURE_NAMES) or V209_FEATURE_NAMES)
        weights = [float(value) for value in head.get("weights", []) or []]
        means = [float(value) for value in head.get("means", [0.0] * len(names))]
        stds = [
            float(value) if float(value) != 0.0 else 1.0
            for value in head.get("stds", [1.0] * len(names))
        ]
        total = _safe_float(head.get("bias"), 0.0)
        for index, name in enumerate(names):
            if index >= len(weights):
                break
            mean = means[index] if index < len(means) else 0.0
            std = stds[index] if index < len(stds) else 1.0
            total += weights[index] * ((_safe_float(features.get(name), 0.0) - mean) / std)
        return _sigmoid(total)

    @staticmethod
    def _feature_support_distance(
        head: Dict[str, Any], features: Dict[str, float]
    ) -> float:
        """Return RMS standardized distance from one section's train support."""
        names = list(head.get("feature_names", V211_FEATURE_NAMES) or V211_FEATURE_NAMES)
        means = [float(value) for value in head.get("means", [0.0] * len(names))]
        stds = [
            float(value) if abs(float(value)) >= 1e-6 else 1.0
            for value in head.get("stds", [1.0] * len(names))
        ]
        if not names:
            return 0.0
        squared = 0.0
        for index, name in enumerate(names):
            mean = means[index] if index < len(means) else 0.0
            std = stds[index] if index < len(stds) else 1.0
            standardized = (_safe_float(features.get(name), 0.0) - mean) / std
            squared += standardized * standardized
        return math.sqrt(squared / len(names))

    @staticmethod
    def _score_linear_model_head(head: Dict[str, Any], features: Dict[str, float]) -> float:
        names = list(head.get("feature_names", V210_FEATURE_NAMES) or V210_FEATURE_NAMES)
        weights = [float(value) for value in head.get("weights", []) or []]
        means = [float(value) for value in head.get("means", [0.0] * len(names))]
        stds = [
            float(value) if float(value) != 0.0 else 1.0
            for value in head.get("stds", [1.0] * len(names))
        ]
        total = _safe_float(head.get("bias"), 0.0)
        for index, name in enumerate(names):
            if index >= len(weights):
                break
            mean = means[index] if index < len(means) else 0.0
            std = stds[index] if index < len(stds) else 1.0
            total += weights[index] * (
                (_safe_float(features.get(name), 0.0) - mean) / std
            )
        return float(total)

    def _counterfactual_section(self, dataset: str, target: str) -> Dict[str, Any]:
        sections = self.model.get("sections", {}) or {}
        section = sections.get(f"{dataset}|{target}", {}) or {}
        return section if isinstance(section, dict) else {}

    def _counterfactual_action_score(
        self,
        dataset: str,
        target: str,
        features: Dict[str, float],
    ) -> Dict[str, Any]:
        section = self._counterfactual_section(dataset, target)
        if not section or not bool(section.get("enabled", False)):
            return {
                "enabled": False,
                "target": target,
                "reason": str(section.get("disabled_reason", "section_missing") or "section_missing"),
            }
        gain_probability = self._score_model_head(section.get("gain_head", {}) or {}, features)
        loss_probability = self._score_model_head(section.get("loss_head", {}) or {}, features)
        mean_gain = _safe_float(section.get("mean_gain"), 0.0)
        mean_loss = _safe_float(section.get("mean_loss"), 0.0)
        action_cost = _safe_float(section.get("action_cost"), 0.0)
        net_utility = gain_probability * mean_gain - loss_probability * mean_loss - action_cost
        threshold = max(0.0, _safe_float(section.get("min_net_utility"), 0.0))
        return {
            "enabled": True,
            "target": target,
            "gain_probability": gain_probability,
            "loss_probability": loss_probability,
            "predicted_net_utility": net_utility,
            "min_net_utility": threshold,
            "approved": net_utility > threshold + 1e-12,
        }

    def _rank_passage_counterfactual(
        self,
        query_plan_log: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        dataset = self._dataset_name(query_plan_log)
        grouped = group_query_plan_evidence_candidate_refs(query_plan_log)
        candidate_count = sum(len(refs) for refs in grouped.values())
        log: Dict[str, Any] = {
            "enabled": True,
            "model_path": self.model_path,
            "model_ready": self.is_ready(),
            "mode": "passage_counterfactual_admission",
            "dataset": dataset,
            "candidate_count": candidate_count,
            "unique_passage_count": len(grouped),
            "duplicate_candidate_count": max(0, candidate_count - len(grouped)),
            "scored_count": 0,
            "promoted_count": 0,
            "committed_top2_count": 0,
            "committed_top5_count": 0,
            "no_op": True,
            "top_actions": [],
        }
        dataset_enabled, skip_reason = self._dataset_is_enabled(dataset)
        log["dataset_enabled"] = dataset_enabled
        if not dataset_enabled:
            log["skip_reason"] = skip_reason
            return query_plan_log, log

        incumbent = query_plan_log.get("candidate_admission_incumbent", {}) or {}
        if bool(incumbent.get("defer_decision", False)):
            log["skip_reason"] = "deferred_until_control_slate"
            log["deferred"] = True
            return query_plan_log, log
        incumbent_ids = [
            _safe_int(value, -1)
            for value in incumbent.get("doc_ids", []) or []
            if _safe_int(value, -1) >= 0
        ]
        incumbent_top2 = set(incumbent_ids[:2])
        incumbent_top5 = set(incumbent_ids[:5])
        max_original_rank = max(
            1,
            _safe_int(self._metadata().get("max_original_rank"), 200),
        )
        actions: List[Dict[str, Any]] = []
        for passage_idx, refs in grouped.items():
            original_rank = min(
                (_safe_int(candidate.get("original_rank"), 10_000) for _, candidate in refs),
                default=10_000,
            )


            if passage_idx in incumbent_top2 or original_rank <= 2 or original_rank > max_original_rank:
                continue
            targets = ["top2"]
            if passage_idx not in incumbent_top5 and original_rank > 5:
                targets.append("top5")
            for target in targets:
                features = aggregate_passage_candidate_features(
                    query_plan_log,
                    passage_idx,
                    refs,
                    target,
                )
                result = self._counterfactual_action_score(dataset, target, features)
                log["scored_count"] += int(bool(result.get("enabled")))
                if not result.get("enabled"):
                    continue
                actions.append({
                    **result,
                    "passage_idx": passage_idx,
                    "original_rank": original_rank,
                    "sources": sorted({source for source, _ in refs}),
                    "refs": refs,
                })

        actions.sort(
            key=lambda item: (
                _safe_float(item.get("predicted_net_utility"), -1.0),
                _safe_float(item.get("gain_probability"), 0.0),
            ),
            reverse=True,
        )
        approved = next((item for item in actions if item.get("approved")), None)
        if approved is not None:
            target = str(approved.get("target") or "top5")
            ranker_score = _safe_float(approved.get("gain_probability"), 0.0)
            for source, candidate in approved.get("refs", []):
                candidate["candidate_to_topk_ranker_score"] = round(ranker_score, 6)
                candidate["candidate_to_topk_ranker"] = {
                    "decision": "commit_pending_atomic_execution",
                    "target": target,
                    "source": source,
                    "passage_idx": approved.get("passage_idx"),
                    "predicted_net_utility": round(
                        _safe_float(approved.get("predicted_net_utility"), 0.0), 6
                    ),
                    "gain_probability": round(ranker_score, 6),
                    "loss_probability": round(
                        _safe_float(approved.get("loss_probability"), 0.0), 6
                    ),
                }
                candidate["supervised_evidence_ranker"] = candidate["candidate_to_topk_ranker"]
            log["promoted_count"] = 1
            log[f"committed_{target}_count"] = 1
            log["no_op"] = False
            log["selected_action"] = {
                key: value
                for key, value in approved.items()
                if key not in {"refs"}
            }

        for action in actions[:10]:
            log["top_actions"].append({
                "passage_idx": action.get("passage_idx"),
                "target": action.get("target"),
                "original_rank": action.get("original_rank"),
                "sources": action.get("sources", []),
                "gain_probability": round(_safe_float(action.get("gain_probability"), 0.0), 6),
                "loss_probability": round(_safe_float(action.get("loss_probability"), 0.0), 6),
                "predicted_net_utility": round(
                    _safe_float(action.get("predicted_net_utility"), 0.0), 6
                ),
                "min_net_utility": round(_safe_float(action.get("min_net_utility"), 0.0), 6),
                "approved": bool(action.get("approved", False)),
            })
        return query_plan_log, log

    def _rank_final_state_transactional_slate(
        self,
        query_plan_log: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        dataset = self._dataset_name(query_plan_log)
        incumbent = query_plan_log.get("candidate_admission_incumbent", {}) or {}
        grouped = group_query_plan_evidence_candidate_refs(query_plan_log)
        candidate_count = sum(len(refs) for refs in grouped.values())
        log: Dict[str, Any] = {
            "enabled": True,
            "model_path": self.model_path,
            "model_ready": self.is_ready(),
            "mode": "final_state_transactional_slate",
            "dataset": dataset,
            "candidate_count": candidate_count,
            "unique_passage_count": len(grouped),
            "duplicate_candidate_count": max(0, candidate_count - len(grouped)),
            "scored_count": 0,
            "promoted_count": 0,
            "committed_top2_count": 0,
            "committed_top5_count": 0,
            "committed_pair_count": 0,
            "state_conflict": False,
            "no_op": True,
            "top_actions": [],
            "state_hash": incumbent.get("state_hash", ""),
            "state_version": incumbent.get("state_version", V210_STATE_VERSION),
        }
        if bool(incumbent.get("defer_decision", False)):
            log["skip_reason"] = "deferred_until_final_state"
            log["deferred"] = True
            return query_plan_log, log
        dataset_enabled, skip_reason = self._dataset_is_enabled(dataset)
        log["dataset_enabled"] = dataset_enabled
        if not dataset_enabled:
            log["skip_reason"] = skip_reason
            return query_plan_log, log
        section = (self.model.get("sections", {}) or {}).get(dataset, {}) or {}
        if not bool(section.get("enabled", False)):
            log["skip_reason"] = str(
                section.get("disabled_reason", "dataset_section_disabled")
            )
            return query_plan_log, log

        metadata = self._metadata()
        actions = generate_final_state_slate_actions(
            query_plan_log,
            max_single_candidates=max(
                1, _safe_int(metadata.get("max_single_candidates"), 10)
            ),
            max_pair_candidates=max(
                2, _safe_int(metadata.get("max_pair_candidates"), 6)
            ),
        )
        threshold = max(0.0, _safe_float(section.get("min_net_utility"), 0.0))
        mean_gain = _safe_float(section.get("mean_gain"), 0.0)
        mean_loss = _safe_float(section.get("mean_loss"), mean_gain)
        action_costs = metadata.get("action_costs", {}) or {}
        direct_utility_weight = min(
            1.0, max(0.0, _safe_float(metadata.get("direct_utility_weight"), 0.65))
        )
        scored_actions: List[Dict[str, Any]] = []
        for action in actions:
            features = action.get("features", {}) or {}
            gain_probability = self._score_model_head(
                section.get("gain_head", {}) or {}, features
            )
            loss_probability = self._score_model_head(
                section.get("loss_head", {}) or {}, features
            )
            action_kind = str(action.get("action_kind") or "")
            action_cost = _safe_float(action_costs.get(action_kind), 0.0)
            expected_class_utility = (
                gain_probability * mean_gain
                - loss_probability * mean_loss
            )
            direct_utility = self._score_linear_model_head(
                section.get("utility_head", {}) or {}, features
            )
            predicted_utility = (
                direct_utility_weight * direct_utility
                + (1.0 - direct_utility_weight) * expected_class_utility
                - action_cost
            )
            scored_actions.append({
                **action,
                "gain_probability": gain_probability,
                "loss_probability": loss_probability,
                "direct_utility": direct_utility,
                "expected_class_utility": expected_class_utility,
                "predicted_net_utility": predicted_utility,
                "min_net_utility": threshold,
                "approved": predicted_utility > threshold + 1e-12,
            })
        log["scored_count"] = len(scored_actions)
        scored_actions.sort(
            key=lambda item: (
                _safe_float(item.get("predicted_net_utility"), -1.0),
                _safe_float(item.get("gain_probability"), 0.0),
                -_safe_float(item.get("loss_probability"), 1.0),
            ),
            reverse=True,
        )
        approved = next((item for item in scored_actions if item.get("approved")), None)
        if approved is not None:
            action_kind = str(approved.get("action_kind") or "")
            log["selected_action"] = {
                key: value
                for key, value in approved.items()
                if key != "features"
            }
            log["promoted_count"] = 1
            log["no_op"] = False
            if action_kind == "replace_top5_slot5":
                log["committed_top5_count"] = 1
            else:
                log["committed_top2_count"] = 1
            if action_kind == "replace_top2_pair":
                log["committed_pair_count"] = 1

        for action in scored_actions[:10]:
            log["top_actions"].append({
                "action_kind": action.get("action_kind"),
                "passage_ids": action.get("passage_ids", []),
                "final_ranks": action.get("final_ranks", []),
                "sources": action.get("sources", []),
                "gain_probability": round(
                    _safe_float(action.get("gain_probability"), 0.0), 6
                ),
                "loss_probability": round(
                    _safe_float(action.get("loss_probability"), 0.0), 6
                ),
                "direct_utility": round(
                    _safe_float(action.get("direct_utility"), 0.0), 6
                ),
                "predicted_net_utility": round(
                    _safe_float(action.get("predicted_net_utility"), 0.0), 6
                ),
                "min_net_utility": round(threshold, 6),
                "approved": bool(action.get("approved", False)),
            })
        return query_plan_log, log

    def _rank_incumbent_aware_constrained_slate(
        self,
        query_plan_log: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        dataset = self._dataset_name(query_plan_log)
        incumbent = query_plan_log.get("candidate_admission_incumbent", {}) or {}
        grouped = group_query_plan_evidence_candidate_refs(query_plan_log)
        candidate_count = sum(len(refs) for refs in grouped.values())
        log: Dict[str, Any] = {
            "enabled": True,
            "model_path": self.model_path,
            "model_ready": self.is_ready(),
            "mode": "incumbent_aware_constrained_slate",
            "dataset": dataset,
            "candidate_count": candidate_count,
            "unique_passage_count": len(grouped),
            "duplicate_candidate_count": max(0, candidate_count - len(grouped)),
            "scored_count": 0,
            "support_rejected_count": 0,
            "section_rejected_count": 0,
            "margin_rejected_count": 0,
            "utility_rejected_count": 0,
            "promoted_count": 0,
            "committed_top2_count": 0,
            "committed_top5_count": 0,
            "committed_pair_count": 0,
            "state_conflict": False,
            "no_op": True,
            "top_actions": [],
            "state_hash": incumbent.get("state_hash", ""),
            "state_version": incumbent.get("state_version", V210_STATE_VERSION),
        }
        if bool(incumbent.get("defer_decision", False)):
            log["skip_reason"] = "deferred_until_final_state"
            log["deferred"] = True
            return query_plan_log, log
        dataset_enabled, skip_reason = self._dataset_is_enabled(dataset)
        log["dataset_enabled"] = dataset_enabled
        if not dataset_enabled:
            log["skip_reason"] = skip_reason
            return query_plan_log, log

        metadata = self._metadata()
        actions = generate_incumbent_aware_slate_actions(
            query_plan_log,
            max_single_candidates=max(
                1, _safe_int(metadata.get("max_single_candidates"), 10)
            ),
            max_pair_candidates=max(
                2, _safe_int(metadata.get("max_pair_candidates"), 6)
            ),
        )
        sections = self.model.get("sections", {}) or {}
        action_costs = metadata.get("action_costs", {}) or {}
        min_gain_loss_margin = _safe_float(
            metadata.get("min_gain_loss_margin"), 0.05
        )
        scored_actions: List[Dict[str, Any]] = []
        for action in actions:
            action_kind = str(action.get("action_kind") or "")
            section_key = f"{dataset}|{action_kind}"
            section = sections.get(section_key, {}) or {}
            features = action.get("features", {}) or {}
            scored: Dict[str, Any] = {
                **action,
                "section_key": section_key,
                "approved": False,
            }
            if not bool(section.get("enabled", False)):
                scored["rejection_reason"] = str(
                    section.get("disabled_reason", "section_missing_or_disabled")
                )
                log["section_rejected_count"] += 1
                scored_actions.append(scored)
                continue

            gain_head = section.get("gain_head", {}) or {}
            loss_head = section.get("loss_head", {}) or {}
            support_distance = self._feature_support_distance(gain_head, features)
            max_support_distance = max(
                0.0, _safe_float(section.get("max_support_distance"), 0.0)
            )
            gain_probability = self._score_model_head(gain_head, features)
            loss_probability = self._score_model_head(loss_head, features)
            mean_gain = max(0.0, _safe_float(section.get("mean_gain"), 0.0))
            mean_loss = max(0.0, _safe_float(section.get("mean_loss"), mean_gain))
            action_cost = max(
                0.0,
                _safe_float(
                    section.get("action_cost", action_costs.get(action_kind)), 0.0
                ),
            )
            expected_utility = (
                gain_probability * mean_gain
                - loss_probability * mean_loss
                - action_cost
            )
            bounded_utility = min(
                mean_gain - action_cost,
                max(-mean_loss - action_cost, expected_utility),
            )
            threshold = max(0.0, _safe_float(section.get("min_net_utility"), 0.0))
            probability_margin = gain_probability - loss_probability
            support_ok = max_support_distance > 0.0 and (
                support_distance <= max_support_distance + 1e-12
            )
            margin_ok = probability_margin >= min_gain_loss_margin
            utility_ok = bounded_utility > threshold + 1e-12
            if not support_ok:
                rejection_reason = "outside_training_support"
                log["support_rejected_count"] += 1
            elif not margin_ok:
                rejection_reason = "gain_loss_margin_below_threshold"
                log["margin_rejected_count"] += 1
            elif not utility_ok:
                rejection_reason = "bounded_utility_below_threshold"
                log["utility_rejected_count"] += 1
            else:
                rejection_reason = ""
            scored.update({
                "gain_probability": gain_probability,
                "loss_probability": loss_probability,
                "gain_loss_margin": probability_margin,
                "support_distance": support_distance,
                "max_support_distance": max_support_distance,
                "predicted_net_utility": bounded_utility,
                "min_net_utility": threshold,
                "approved": support_ok and margin_ok and utility_ok,
                "rejection_reason": rejection_reason,
            })
            scored_actions.append(scored)

        log["scored_count"] = len(scored_actions)
        scored_actions.sort(
            key=lambda item: (
                bool(item.get("approved", False)),
                _safe_float(item.get("predicted_net_utility"), -1e9),
                _safe_float(item.get("gain_loss_margin"), -1e9),
                -_safe_float(item.get("support_distance"), 1e9),
            ),
            reverse=True,
        )
        approved = next((item for item in scored_actions if item.get("approved")), None)
        if approved is not None:
            action_kind = str(approved.get("action_kind") or "")
            log["selected_action"] = {
                key: value for key, value in approved.items() if key != "features"
            }
            log["promoted_count"] = 1
            log["no_op"] = False
            if action_kind == "replace_top5_slot5":
                log["committed_top5_count"] = 1
            else:
                log["committed_top2_count"] = 1
            if action_kind == "replace_top2_pair":
                log["committed_pair_count"] = 1

        for action in scored_actions[:10]:
            log["top_actions"].append({
                "action_kind": action.get("action_kind"),
                "passage_ids": action.get("passage_ids", []),
                "candidate_ids": action.get("candidate_ids", []),
                "preserved_incumbent_ids": action.get("preserved_incumbent_ids", []),
                "final_ranks": action.get("final_ranks", []),
                "sources": action.get("sources", []),
                "section_key": action.get("section_key", ""),
                "gain_probability": round(
                    _safe_float(action.get("gain_probability"), 0.0), 6
                ),
                "loss_probability": round(
                    _safe_float(action.get("loss_probability"), 0.0), 6
                ),
                "gain_loss_margin": round(
                    _safe_float(action.get("gain_loss_margin"), 0.0), 6
                ),
                "support_distance": round(
                    _safe_float(action.get("support_distance"), 0.0), 6
                ),
                "max_support_distance": round(
                    _safe_float(action.get("max_support_distance"), 0.0), 6
                ),
                "predicted_net_utility": round(
                    _safe_float(action.get("predicted_net_utility"), 0.0), 6
                ),
                "min_net_utility": round(
                    _safe_float(action.get("min_net_utility"), 0.0), 6
                ),
                "approved": bool(action.get("approved", False)),
                "rejection_reason": action.get("rejection_reason", ""),
            })
        if not actions:
            log["skip_reason"] = "no_incumbent_aware_actions"
        elif approved is None:
            log["skip_reason"] = "no_validation_safe_action"
        return query_plan_log, log

    def _rank_source_conditional_grouped_oof_slate(
        self,
        query_plan_log: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        dataset = self._dataset_name(query_plan_log)
        incumbent = query_plan_log.get("candidate_admission_incumbent", {}) or {}
        grouped = group_query_plan_evidence_candidate_refs(query_plan_log)
        candidate_count = sum(len(refs) for refs in grouped.values())
        content_addressed_mode = (
            self.model.get("type") == "content_addressed_candidate_local_optimizer"
        )
        log: Dict[str, Any] = {
            "enabled": True,
            "model_path": self.model_path,
            "model_ready": self.is_ready(),
            "mode": (
                "content_addressed_candidate_local"
                if content_addressed_mode
                else "source_conditional_grouped_oof_slate"
            ),
            "dataset": dataset,
            "candidate_count": candidate_count,
            "unique_passage_count": len(grouped),
            "duplicate_candidate_count": max(0, candidate_count - len(grouped)),
            "scored_count": 0,
            "section_rejected_count": 0,
            "source_rejected_count": 0,
            "source_distribution_rejected_count": 0,
            "content_identity_rejected_count": 0,
            "support_rejected_count": 0,
            "r2_risk_rejected_count": 0,
            "r2_utility_rejected_count": 0,
            "r5_utility_rejected_count": 0,
            "margin_rejected_count": 0,
            "utility_rejected_count": 0,
            "promoted_count": 0,
            "committed_top2_count": 0,
            "committed_top5_count": 0,
            "committed_pair_count": 0,
            "state_conflict": False,
            "no_op": True,
            "top_actions": [],
            "state_hash": incumbent.get("state_hash", ""),
            "state_version": incumbent.get("state_version", V210_STATE_VERSION),
        }
        if bool(incumbent.get("defer_decision", False)):
            log["skip_reason"] = "deferred_until_final_state"
            log["deferred"] = True
            return query_plan_log, log
        dataset_enabled, skip_reason = self._dataset_is_enabled(dataset)
        log["dataset_enabled"] = dataset_enabled
        if not dataset_enabled:
            log["skip_reason"] = skip_reason
            return query_plan_log, log

        metadata = self._metadata()
        source_group_mode = str(
            metadata.get("source_group_mode", "conditional")
        ).strip().lower()
        action_generator = (
            generate_content_addressed_candidate_actions
            if content_addressed_mode
            else generate_incumbent_aware_slate_actions
        )
        actions = action_generator(
            query_plan_log,
            max_single_candidates=max(
                1, _safe_int(metadata.get("max_single_candidates"), 10)
            ),
            max_pair_candidates=max(
                2, _safe_int(metadata.get("max_pair_candidates"), 6)
            ),
        )
        sections = self.model.get("sections", {}) or {}
        runtime_group_counts: Dict[str, int] = {}
        for action in actions:
            group_key = (
                "__all__|rank0"
                if source_group_mode == "global"
                else source_risk_group_key(action)
            )
            runtime_group_counts[group_key] = runtime_group_counts.get(group_key, 0) + 1
        scored_actions: List[Dict[str, Any]] = []
        for action in actions:
            action_kind = str(action.get("action_kind") or "")
            section_key = f"{dataset}|{action_kind}"
            section = sections.get(section_key, {}) or {}
            risk_group_key = (
                "__all__|rank0"
                if source_group_mode == "global"
                else source_risk_group_key(action)
            )
            source_groups = section.get("source_groups", {}) or {}
            source_group = source_groups.get(risk_group_key, {}) or {}
            features = dict(action.get("features", {}) or {})
            if content_addressed_mode and not bool(
                metadata.get("candidate_local_comparator", True)
            ):
                for name in V213_CANDIDATE_LOCAL_FEATURE_NAMES:
                    features[name] = 0.0
            scored: Dict[str, Any] = {
                **action,
                "section_key": section_key,
                "source_risk_group": risk_group_key,
                "approved": False,
            }
            if not bool(section.get("enabled", False)):
                scored["rejection_reason"] = str(
                    section.get("disabled_reason", "section_missing_or_disabled")
                )
                log["section_rejected_count"] += 1
                scored_actions.append(scored)
                continue
            if not bool(source_group.get("enabled", False)):
                scored["rejection_reason"] = str(
                    source_group.get("disabled_reason", "source_group_missing_or_disabled")
                )
                log["source_rejected_count"] += 1
                scored_actions.append(scored)
                continue

            support_profile = source_group.get("support_profile", {}) or {}
            support_distance = self._feature_support_distance(support_profile, features)
            max_support_distance = max(
                0.0, _safe_float(source_group.get("max_support_distance"), 0.0)
            )
            r2_gain_probability = self._score_model_head(
                section.get("r2_gain_head", {}) or {}, features
            )
            r2_loss_probability = self._score_model_head(
                section.get("r2_loss_head", {}) or {}, features
            )
            r5_gain_probability = self._score_model_head(
                section.get("r5_gain_head", {}) or {}, features
            )
            r5_loss_probability = self._score_model_head(
                section.get("r5_loss_head", {}) or {}, features
            )
            expected_r2 = (
                r2_gain_probability
                * max(0.0, _safe_float(source_group.get("mean_r2_gain"), 0.0))
                - r2_loss_probability
                * max(0.0, _safe_float(source_group.get("mean_r2_loss"), 0.0))
            )
            expected_r5 = (
                r5_gain_probability
                * max(0.0, _safe_float(source_group.get("mean_r5_gain"), 0.0))
                - r5_loss_probability
                * max(0.0, _safe_float(source_group.get("mean_r5_loss"), 0.0))
                - max(0.0, _safe_float(source_group.get("action_cost"), 0.0))
            )
            max_r2_loss_probability = min(
                1.0,
                max(
                    0.0,
                    _safe_float(
                        source_group.get("max_r2_loss_probability"), 0.0
                    ),
                ),
            )
            min_expected_r2 = _safe_float(
                source_group.get("min_expected_r2"), 0.0
            )
            min_expected_r5 = _safe_float(
                source_group.get("min_expected_r5"), 0.0
            )
            source_action_share = runtime_group_counts.get(risk_group_key, 0) / max(
                len(actions), 1
            )
            max_source_action_share = min(
                1.0,
                max(
                    0.0,
                    _safe_float(
                        source_group.get("max_source_action_share"), 1.0
                    )
                    + max(
                        0.0,
                        _safe_float(metadata.get("source_share_tolerance"), 0.05),
                    ),
                ),
            )
            support_ok = max_support_distance > 0.0 and (
                support_distance <= max_support_distance + 1e-12
            )
            if not bool(metadata.get("support_distance_gate", True)):
                support_ok = True
            source_distribution_ok = (
                source_action_share <= max_source_action_share + 1e-12
            )
            if not bool(metadata.get("source_share_prior_shift_gate", True)):
                source_distribution_ok = True
            content_identity_ok = (
                not bool(metadata.get("require_content_identity", False))
                or bool(action.get("content_identity_complete", False))
            )
            r2_risk_ok = r2_loss_probability <= max_r2_loss_probability + 1e-12
            r2_utility_ok = expected_r2 >= min_expected_r2 - 1e-12
            r5_utility_ok = expected_r5 >= min_expected_r5 - 1e-12
            if not content_identity_ok:
                rejection_reason = "content_identity_incomplete"
                log["content_identity_rejected_count"] += 1
            elif not support_ok:
                rejection_reason = "outside_source_training_support"
                log["support_rejected_count"] += 1
            elif not source_distribution_ok:
                rejection_reason = "source_action_share_above_training_support"
                log["source_distribution_rejected_count"] += 1
            elif not r2_risk_ok:
                rejection_reason = "r2_loss_risk_above_source_bound"
                log["r2_risk_rejected_count"] += 1
                log["margin_rejected_count"] += 1
            elif not r2_utility_ok:
                rejection_reason = "expected_r2_below_source_threshold"
                log["r2_utility_rejected_count"] += 1
                log["utility_rejected_count"] += 1
            elif not r5_utility_ok:
                rejection_reason = "expected_r5_below_source_threshold"
                log["r5_utility_rejected_count"] += 1
                log["utility_rejected_count"] += 1
            else:
                rejection_reason = ""
            approved = (
                content_identity_ok
                and support_ok
                and source_distribution_ok
                and r2_risk_ok
                and r2_utility_ok
                and r5_utility_ok
            )
            scored.update({
                "r2_gain_probability": r2_gain_probability,
                "r2_loss_probability": r2_loss_probability,
                "r5_gain_probability": r5_gain_probability,
                "r5_loss_probability": r5_loss_probability,
                "expected_r2": expected_r2,
                "expected_r5": expected_r5,
                "max_r2_loss_probability": max_r2_loss_probability,
                "min_expected_r2": min_expected_r2,
                "min_expected_r5": min_expected_r5,
                "source_action_share": source_action_share,
                "max_source_action_share": max_source_action_share,
                "support_distance": support_distance,
                "max_support_distance": max_support_distance,
                "content_identity_ok": content_identity_ok,
                "approved": approved,
                "rejection_reason": rejection_reason,
            })
            scored_actions.append(scored)

        log["scored_count"] = len(scored_actions)
        scored_actions.sort(
            key=lambda item: (
                bool(item.get("approved", False)),
                _safe_float(item.get("expected_r2"), -1e9),
                _safe_float(item.get("expected_r5"), -1e9),
                -_safe_float(item.get("r2_loss_probability"), 1e9),
                -_safe_float(item.get("support_distance"), 1e9),
            ),
            reverse=True,
        )
        approved_action = next(
            (item for item in scored_actions if item.get("approved")), None
        )
        if approved_action is not None:
            action_kind = str(approved_action.get("action_kind") or "")
            log["selected_action"] = {
                key: value
                for key, value in approved_action.items()
                if key != "features"
            }
            log["promoted_count"] = 1
            log["no_op"] = False
            if action_kind == "replace_top5_slot5":
                log["committed_top5_count"] = 1
            else:
                log["committed_top2_count"] = 1
            if action_kind == "replace_top2_pair":
                log["committed_pair_count"] = 1

        for action in scored_actions[:10]:
            log["top_actions"].append({
                "action_kind": action.get("action_kind"),
                "passage_ids": action.get("passage_ids", []),
                "candidate_ids": action.get("candidate_ids", []),
                "preserved_incumbent_ids": action.get("preserved_incumbent_ids", []),
                "preserved_incumbent_rank": action.get("preserved_incumbent_rank", 0),
                "final_ranks": action.get("final_ranks", []),
                "sources": action.get("sources", []),
                "source_signature": action.get("source_signature", "unknown"),
                "provenance_signature": action.get("provenance_signature", "unknown"),
                "source_risk_group": action.get("source_risk_group", ""),
                "section_key": action.get("section_key", ""),
                "action_fingerprint": action.get("action_fingerprint", ""),
                "content_state_fingerprint": action.get(
                    "content_state_fingerprint", ""
                ),
                "candidate_fingerprints": action.get(
                    "candidate_fingerprints", []
                ),
                "displaced_fingerprints": action.get(
                    "displaced_fingerprints", []
                ),
                "content_identity_ok": bool(
                    action.get("content_identity_ok", False)
                ),
                "r2_gain_probability": round(
                    _safe_float(action.get("r2_gain_probability"), 0.0), 6
                ),
                "r2_loss_probability": round(
                    _safe_float(action.get("r2_loss_probability"), 0.0), 6
                ),
                "r5_gain_probability": round(
                    _safe_float(action.get("r5_gain_probability"), 0.0), 6
                ),
                "r5_loss_probability": round(
                    _safe_float(action.get("r5_loss_probability"), 0.0), 6
                ),
                "expected_r2": round(
                    _safe_float(action.get("expected_r2"), 0.0), 6
                ),
                "expected_r5": round(
                    _safe_float(action.get("expected_r5"), 0.0), 6
                ),
                "max_r2_loss_probability": round(
                    _safe_float(action.get("max_r2_loss_probability"), 0.0), 6
                ),
                "min_expected_r2": round(
                    _safe_float(action.get("min_expected_r2"), 0.0), 6
                ),
                "min_expected_r5": round(
                    _safe_float(action.get("min_expected_r5"), 0.0), 6
                ),
                "source_action_share": round(
                    _safe_float(action.get("source_action_share"), 0.0), 6
                ),
                "max_source_action_share": round(
                    _safe_float(action.get("max_source_action_share"), 0.0), 6
                ),
                "support_distance": round(
                    _safe_float(action.get("support_distance"), 0.0), 6
                ),
                "max_support_distance": round(
                    _safe_float(action.get("max_support_distance"), 0.0), 6
                ),
                "approved": bool(action.get("approved", False)),
                "rejection_reason": action.get("rejection_reason", ""),
            })
        if not actions:
            log["skip_reason"] = "no_incumbent_aware_actions"
        elif approved_action is None:
            log["skip_reason"] = "no_source_conditionally_safe_action"
        return query_plan_log, log

    def rank(self, query_plan_log: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if self.model.get("type") in {
            "source_conditional_grouped_oof_slate_optimizer",
            "content_addressed_candidate_local_optimizer",
        }:
            return self._rank_source_conditional_grouped_oof_slate(query_plan_log)
        if self.model.get("type") == "incumbent_aware_constrained_slate_optimizer":
            return self._rank_incumbent_aware_constrained_slate(query_plan_log)
        if self.model.get("type") == "final_state_transactional_slate_optimizer":
            return self._rank_final_state_transactional_slate(query_plan_log)
        if self.model.get("type") == "passage_counterfactual_admission_optimizer":
            return self._rank_passage_counterfactual(query_plan_log)
        log = {
            "enabled": True,
            "model_path": self.model_path,
            "model_ready": self.is_ready(),
            "candidate_count": 0,
            "scored_count": 0,
            "promoted_count": 0,
            "rejected_count": 0,
            "skipped_count": 0,
            "top_candidates": [],
        }
        if not self.is_ready():
            log["skip_reason"] = "model_missing_or_invalid"
            return query_plan_log, log

        scored: List[Tuple[float, str, Dict[str, Any], Dict[str, float]]] = []
        model_feature_names = list(self.model.get("feature_names", FEATURE_NAMES) or FEATURE_NAMES)
        dataset = self._dataset_name(query_plan_log)
        min_score = self._threshold_for_dataset("min_score", dataset, self.min_score)
        reject_below = self._threshold_for_dataset("reject_below", dataset, self.reject_below)
        promote_confidence = self._promote_confidence_for_dataset(dataset)
        score_weight = self._score_weight()
        log["dataset"] = dataset
        dataset_enabled, dataset_skip_reason = self._dataset_is_enabled(dataset)
        log["dataset_enabled"] = dataset_enabled
        log["min_score"] = round(float(min_score), 6)
        log["reject_below"] = round(float(reject_below), 6)
        log["promote_confidence"] = round(float(promote_confidence), 6)
        log["candidate_score_weight"] = round(float(score_weight), 6)
        log["source_risk_gate"] = bool(self._metadata().get("source_risk_gate"))
        log["source_risk_blocked_count"] = 0
        log["source_cap_blocked_count"] = 0
        log["promoted_by_source"] = {}
        if not dataset_enabled:
            log["skip_reason"] = dataset_skip_reason
            return query_plan_log, log

        for source, candidate in iter_query_plan_evidence_candidate_refs(query_plan_log):
            log["candidate_count"] += 1
            if not self._source_is_trained(source, model_feature_names):
                candidate["supervised_evidence_ranker"] = {
                    "decision": "skip_untrained_source",
                    "source": source,
                    "reason": "model_has_no_source_feature",
                }
                log["skipped_count"] += 1
                continue
            features = extract_query_plan_candidate_features(query_plan_log, candidate, source)
            score = self.score_features(features)
            scored.append((score, source, candidate, features))
        scored.sort(key=lambda item: item[0], reverse=True)
        log["scored_count"] = len(scored)
        promoted = 0
        promoted_by_source: Dict[str, int] = {}
        for score, source, candidate, features in scored:
            decision = "keep"
            old_confidence = _safe_float(candidate.get("confidence"), 0.0)
            old_score = _safe_float(candidate.get("score"), 0.0)
            original_rank = _safe_int(candidate.get("original_rank"), 10_000)
            source_cap = self._source_cap(source)
            source_promoted = promoted_by_source.get(source, 0)
            source_risk_allowed, source_signal_count, source_signals = self._source_risk_gate(source, features)
            if 0 < original_rank <= 2:
                decision = "keep_incumbent"
            elif source_cap <= 0 and score >= min_score:
                decision = "keep_source_disabled"
                log["source_cap_blocked_count"] += 1
            elif score >= min_score and source_promoted >= source_cap:
                decision = "keep_source_cap"
                log["source_cap_blocked_count"] += 1
            elif score >= min_score and not source_risk_allowed:
                decision = "keep_source_risk"
                log["source_risk_blocked_count"] += 1
            elif score >= min_score and (self.max_candidates <= 0 or promoted < self.max_candidates):
                promoted += 1
                promoted_by_source[source] = source_promoted + 1
                decision = "promote"
                candidate["confidence"] = max(old_confidence, promote_confidence, score)
                candidate["score"] = max(old_score, old_score + score_weight * score, score)
                candidate["candidate_to_topk_ranker_score"] = round(float(score), 6)
            elif score < reject_below:
                decision = "reject"
                candidate["confidence"] = min(old_confidence, self.rejected_confidence)
                log["rejected_count"] += 1
            ranker_info = {
                "score": round(float(score), 6),
                "decision": decision,
                "source": source,
                "old_confidence": round(float(old_confidence), 6),
                "old_score": round(float(old_score), 6),
                "min_score": round(float(min_score), 6),
                "reject_below": round(float(reject_below), 6),
                "source_cap": int(source_cap),
                "source_signal_count": int(source_signal_count),
                "source_signals": source_signals[:6],
                "feature_snapshot": {
                    "original_rank_log": round(features.get("original_rank_log", 0.0), 4),
                    "candidate_confidence": round(features.get("candidate_confidence", 0.0), 4),
                    "matched_anchor": features.get("matched_anchor", 0.0),
                    "matched_terminal": features.get("matched_terminal", 0.0),
                    "relation_coverage_score": round(features.get("relation_coverage_score", 0.0), 4),
                    "terminal_type_score": round(features.get("terminal_type_score", 0.0), 4),
                    "has_proof_signal": features.get("has_proof_signal", 0.0),
                    "has_chain_signal": features.get("has_chain_signal", 0.0),
                    "llm_approved": features.get("llm_approved", 0.0),
                    "source_support": features.get("source_support", 0.0),
                    "source_evidence_grounded": features.get("source_evidence_grounded", 0.0),
                    "source_missing": features.get("source_missing", 0.0),
                    "source_subgraph": features.get("source_subgraph", 0.0),
                    "source_bind_probe": features.get("source_bind_probe", 0.0),
                    "source_path_executor": features.get("source_path_executor", 0.0),
                },
            }
            candidate["supervised_evidence_ranker"] = ranker_info
            candidate["candidate_to_topk_ranker"] = ranker_info
        log["promoted_count"] = promoted
        log["promoted_by_source"] = dict(sorted(promoted_by_source.items()))
        for score, source, candidate, _ in scored[:10]:
            log["top_candidates"].append({
                "passage_idx": candidate.get("passage_idx"),
                "source": source,
                "score": round(float(score), 6),
                "decision": (candidate.get("supervised_evidence_ranker", {}) or {}).get("decision", ""),
                "original_rank": candidate.get("original_rank"),
                "confidence": candidate.get("confidence"),
            })
        return query_plan_log, log
