import json
import math
import os
from typing import Any, Dict, List, Mapping

from .query_plan_evidence_set_policy import (
    FEATURE_NAMES as EVIDENCE_SET_FEATURE_NAMES,
    extract_evidence_set_policy_features,
    feature_vector,
    sigmoid,
)


BASE_FEATURE_NAMES = [f"candidate_{name}" for name in EVIDENCE_SET_FEATURE_NAMES]

PAIRWISE_EXTRA_FEATURE_NAMES = [
    "replacement_rank_inv",
    "replacement_rank_log",
    "replacement_rank_le_2",
    "replacement_rank_le_5",
    "replacement_value",
    "candidate_value",
    "value_margin",
    "confidence_margin",
    "score_margin",
    "rank_log_margin",
    "candidate_original_rank_after_rank2",
    "candidate_original_rank_after_top5",
    "source_score",
    "path_aware_score",
    "path_aware_approved",
    "rank2_rescue_score",
    "recall_swapper_score",
    "utility_policy_score",
    "window_selector_score",
    "text_verifier_score",
    "text_verifier_anchor_hits_log",
    "text_verifier_terminal_hits_log",
    "text_verifier_relation_hits_log",
    "text_verifier_query_hits_log",
    "text_verifier_answer_hits_log",
    "text_verifier_promoted",
    "text_verifier_rejected",
    "evidence_join_score",
    "evidence_join_text_score",
    "evidence_join_path_score",
    "evidence_join_source_precision",
    "evidence_join_slot_value",
    "evidence_join_query_slot_value",
    "evidence_join_rank_cost",
    "evidence_join_structural",
    "evidence_join_relation_only",
    "evidence_join_router_bypass",
    "answer_pairwise_score",
    "answer_pairwise_text_score",
    "answer_pairwise_path_score",
    "answer_pairwise_source_precision",
    "answer_pairwise_join_score",
    "answer_pairwise_slot_value",
    "answer_pairwise_query_slot_value",
    "answer_pairwise_rank_cost",
    "answer_pairwise_anchor_hits_log",
    "answer_pairwise_terminal_hits_log",
    "answer_pairwise_relation_hits_log",
    "answer_pairwise_query_hits_log",
    "answer_pairwise_answer_hits_log",
    "answer_pairwise_endpoint_signal",
    "answer_pairwise_answer_signal",
    "answer_pairwise_structural",
    "answer_pairwise_relation_only",
    "answer_pairwise_candidate_utility",
    "answer_pairwise_replacement_utility",
    "answer_pairwise_margin",
    "answer_pairwise_target_top2",
    "provenance_groupwise_score",
    "provenance_groupwise_source_precision",
    "provenance_groupwise_recall_yield",
    "provenance_groupwise_cost",
    "provenance_groupwise_text_score",
    "provenance_groupwise_anchor_hits_log",
    "provenance_groupwise_terminal_hits_log",
    "provenance_groupwise_relation_hits_log",
    "provenance_groupwise_answer_hits_log",
    "schema_slot_utility_score",
    "schema_slot_utility_top2_approved",
    "schema_slot_utility_has_schema_slot",
    "replacement_doc_known",
]

PAIRWISE_FEATURE_NAMES = BASE_FEATURE_NAMES + PAIRWISE_EXTRA_FEATURE_NAMES

SET_COUNTERFACTUAL_EXTRA_FEATURE_NAMES = [
    "set_target_top2",
    "set_target_top5",
    "set_selected_len",
    "set_final_top_k",
    "set_replacement_index",
    "set_replacement_index_inv",
    "set_replacement_index_log",
    "set_top2_value_sum",
    "set_top2_value_mean",
    "set_top2_value_min",
    "set_top5_value_sum",
    "set_top5_value_mean",
    "set_top5_value_min",
    "set_candidate_value_minus_top2_mean",
    "set_candidate_value_minus_top5_mean",
    "set_candidate_value_minus_top5_min",
    "set_candidate_text_minus_top5_mean",
    "set_candidate_source_seen_top5",
    "set_candidate_new_source_top5",
    "set_candidate_answer_or_endpoint",
    "set_candidate_relation_only",
    "set_candidate_structural",
    "set_replacement_relation_only",
    "set_replacement_answer_or_endpoint",
    "set_query_simple_factoid",
    "set_query_complex",
    "set_query_length_log",
]

SET_COUNTERFACTUAL_FEATURE_NAMES = PAIRWISE_FEATURE_NAMES + SET_COUNTERFACTUAL_EXTRA_FEATURE_NAMES


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


def _safe_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _bool_feature(value: Any) -> float:
    return 1.0 if bool(value) else 0.0


def _log_count(value: Any) -> float:
    return math.log1p(max(_safe_float(value, 0.0), 0.0))


def _source_score(candidate: Mapping[str, Any]) -> float:
    source_policy = _safe_mapping(candidate.get("source_policy"))
    return max(
        _safe_float(source_policy.get("selected_precision"), 0.0),
        _safe_float(source_policy.get("precision"), 0.0),
        _safe_float(source_policy.get("gt5_precision"), 0.0),
    )


def _candidate_value(candidate: Mapping[str, Any], base_features: Mapping[str, float]) -> float:
    source_score = _source_score(candidate)
    path_info = _safe_mapping(candidate.get("path_aware"))
    value = (
        0.40 * _safe_float(candidate.get("confidence"), 0.0)
        + 0.10 * _safe_float(candidate.get("score"), 0.0)
        + 0.24 * source_score
        + 0.08 * float(base_features.get("has_terminal", 0.0))
        + 0.07 * float(base_features.get("has_strict", 0.0))
        + 0.05 * float(base_features.get("has_multihop", 0.0))
        + 0.04 * float(base_features.get("has_relation", 0.0))
        + 0.04 * _safe_float(path_info.get("score"), 0.0)
    )
    return min(max(value, 0.0), 2.0)


def extract_rank2_pairwise_features(
    query_plan_log: Mapping[str, Any],
    candidate: Mapping[str, Any],
    replacement_info: Mapping[str, Any] = None,
) -> Dict[str, float]:
    replacement_info = _safe_mapping(replacement_info)
    base = extract_evidence_set_policy_features(query_plan_log, candidate)
    features: Dict[str, float] = {
        f"candidate_{name}": float(base.get(name, 0.0))
        for name in EVIDENCE_SET_FEATURE_NAMES
    }
    replacement_rank = _safe_int(replacement_info.get("rank"), 2)
    if replacement_rank <= 0:
        replacement_rank = 2
    replacement_value = _safe_float(replacement_info.get("value"), 0.0)
    original_rank = _safe_int(candidate.get("original_rank"), 10_000)
    if original_rank <= 0:
        original_rank = 10_000
    candidate_value = _candidate_value(candidate, base)
    path_info = _safe_mapping(candidate.get("path_aware"))
    rank2_rescue = _safe_mapping(candidate.get("rank2_rescue"))
    recall_swapper = _safe_mapping(candidate.get("recall_swapper"))
    utility_policy = _safe_mapping(candidate.get("utility_policy"))
    window_selector = _safe_mapping(candidate.get("window_selector"))
    text_verifier = _safe_mapping(candidate.get("text_verifier"))
    evidence_join = _safe_mapping(candidate.get("evidence_join_planner"))
    answer_pairwise = _safe_mapping(candidate.get("answer_pairwise_verifier"))
    answer_pairwise_gate = _safe_mapping(candidate.get("answer_pairwise_gate"))
    answer_pairwise_gate_features = _safe_mapping(answer_pairwise_gate.get("features"))
    provenance_groupwise = _safe_mapping(candidate.get("provenance_groupwise"))
    schema_slot_utility = _safe_mapping(candidate.get("schema_slot_utility"))
    features.update({
        "replacement_rank_inv": 1.0 / float(max(replacement_rank, 1)),
        "replacement_rank_log": math.log1p(float(replacement_rank)),
        "replacement_rank_le_2": 1.0 if replacement_rank <= 2 else 0.0,
        "replacement_rank_le_5": 1.0 if replacement_rank <= 5 else 0.0,
        "replacement_value": replacement_value,
        "candidate_value": candidate_value,
        "value_margin": candidate_value - replacement_value,
        "confidence_margin": _safe_float(candidate.get("confidence"), 0.0) - _safe_float(
            replacement_info.get("confidence"),
            0.0,
        ),
        "score_margin": _safe_float(candidate.get("score"), 0.0) - _safe_float(
            replacement_info.get("score"),
            0.0,
        ),
        "rank_log_margin": math.log1p(float(replacement_rank)) - math.log1p(float(original_rank)),
        "candidate_original_rank_after_rank2": 1.0 if original_rank > 2 else 0.0,
        "candidate_original_rank_after_top5": 1.0 if original_rank > 5 else 0.0,
        "source_score": _source_score(candidate),
        "path_aware_score": _safe_float(path_info.get("score"), 0.0),
        "path_aware_approved": 1.0 if bool(path_info.get("approved")) else 0.0,
        "rank2_rescue_score": _safe_float(rank2_rescue.get("score"), 0.0),
        "recall_swapper_score": _safe_float(recall_swapper.get("score"), 0.0),
        "utility_policy_score": _safe_float(utility_policy.get("score"), 0.0),
        "window_selector_score": _safe_float(window_selector.get("score"), 0.0),
        "text_verifier_score": _safe_float(text_verifier.get("score"), 0.0),
        "text_verifier_anchor_hits_log": _log_count(text_verifier.get("anchor_hits")),
        "text_verifier_terminal_hits_log": _log_count(text_verifier.get("terminal_hits")),
        "text_verifier_relation_hits_log": _log_count(len(text_verifier.get("relation_overlap", []) or [])),
        "text_verifier_query_hits_log": _log_count(len(text_verifier.get("query_overlap", []) or [])),
        "text_verifier_answer_hits_log": _log_count(len(text_verifier.get("answer_overlap", []) or [])),
        "text_verifier_promoted": _bool_feature(text_verifier.get("promoted")),
        "text_verifier_rejected": _bool_feature(text_verifier.get("rejected")),
        "evidence_join_score": _safe_float(evidence_join.get("score"), 0.0),
        "evidence_join_text_score": _safe_float(evidence_join.get("text_score"), 0.0),
        "evidence_join_path_score": _safe_float(evidence_join.get("path_score"), 0.0),
        "evidence_join_source_precision": _safe_float(evidence_join.get("source_precision"), 0.0),
        "evidence_join_slot_value": _safe_float(evidence_join.get("slot_value"), 0.0),
        "evidence_join_query_slot_value": _safe_float(evidence_join.get("query_slot_value"), 0.0),
        "evidence_join_rank_cost": _safe_float(evidence_join.get("rank_cost"), 0.0),
        "evidence_join_structural": _bool_feature(evidence_join.get("structural")),
        "evidence_join_relation_only": _bool_feature(evidence_join.get("relation_only")),
        "evidence_join_router_bypass": _bool_feature(evidence_join.get("router_bypass")),
        "answer_pairwise_score": _safe_float(answer_pairwise.get("score"), 0.0),
        "answer_pairwise_text_score": _safe_float(answer_pairwise.get("text_score"), 0.0),
        "answer_pairwise_path_score": _safe_float(answer_pairwise.get("path_score"), 0.0),
        "answer_pairwise_source_precision": _safe_float(answer_pairwise.get("source_precision"), 0.0),
        "answer_pairwise_join_score": _safe_float(answer_pairwise.get("join_score"), 0.0),
        "answer_pairwise_slot_value": _safe_float(answer_pairwise.get("slot_value"), 0.0),
        "answer_pairwise_query_slot_value": _safe_float(answer_pairwise.get("query_slot_value"), 0.0),
        "answer_pairwise_rank_cost": _safe_float(answer_pairwise.get("rank_cost"), 0.0),
        "answer_pairwise_anchor_hits_log": _log_count(answer_pairwise.get("anchor_hits")),
        "answer_pairwise_terminal_hits_log": _log_count(answer_pairwise.get("terminal_hits")),
        "answer_pairwise_relation_hits_log": _log_count(answer_pairwise.get("relation_hits")),
        "answer_pairwise_query_hits_log": _log_count(answer_pairwise.get("query_hits")),
        "answer_pairwise_answer_hits_log": _log_count(answer_pairwise.get("answer_hits")),
        "answer_pairwise_endpoint_signal": _bool_feature(answer_pairwise.get("endpoint_signal")),
        "answer_pairwise_answer_signal": _bool_feature(answer_pairwise.get("answer_signal")),
        "answer_pairwise_structural": _bool_feature(answer_pairwise.get("structural")),
        "answer_pairwise_relation_only": _bool_feature(answer_pairwise.get("relation_only")),
        "answer_pairwise_candidate_utility": _safe_float(
            answer_pairwise_gate_features.get("candidate_utility"),
            _safe_float(answer_pairwise.get("score"), 0.0),
        ),
        "answer_pairwise_replacement_utility": _safe_float(answer_pairwise_gate_features.get("replacement_utility"), 0.0),
        "answer_pairwise_margin": _safe_float(answer_pairwise_gate_features.get("margin"), 0.0),
        "answer_pairwise_target_top2": _bool_feature(answer_pairwise_gate_features.get("target_top2")),
        "provenance_groupwise_score": _safe_float(provenance_groupwise.get("score"), 0.0),
        "provenance_groupwise_source_precision": _safe_float(provenance_groupwise.get("source_precision"), 0.0),
        "provenance_groupwise_recall_yield": _safe_float(provenance_groupwise.get("recall_yield"), 0.0),
        "provenance_groupwise_cost": _safe_float(provenance_groupwise.get("cost"), 0.0),
        "provenance_groupwise_text_score": _safe_float(provenance_groupwise.get("text_score"), 0.0),
        "provenance_groupwise_anchor_hits_log": _log_count(provenance_groupwise.get("anchor_hits")),
        "provenance_groupwise_terminal_hits_log": _log_count(provenance_groupwise.get("terminal_hits")),
        "provenance_groupwise_relation_hits_log": _log_count(provenance_groupwise.get("relation_hits")),
        "provenance_groupwise_answer_hits_log": _log_count(provenance_groupwise.get("answer_hits")),
        "schema_slot_utility_score": _safe_float(schema_slot_utility.get("score"), 0.0),
        "schema_slot_utility_top2_approved": _bool_feature(schema_slot_utility.get("top2_approved")),
        "schema_slot_utility_has_schema_slot": _bool_feature(schema_slot_utility.get("has_schema_slot")),
        "replacement_doc_known": 1.0 if replacement_info.get("doc_id") is not None else 0.0,
    })
    return {name: float(features.get(name, 0.0)) for name in PAIRWISE_FEATURE_NAMES}


def _text_score(candidate: Mapping[str, Any]) -> float:
    text_verifier = _safe_mapping(candidate.get("counterfactual_text_verifier"))
    if not text_verifier:
        text_verifier = _safe_mapping(candidate.get("text_verifier"))
    return _safe_float(text_verifier.get("score"), 0.0)


def _relation_only(candidate: Mapping[str, Any]) -> float:
    text_verifier = _safe_mapping(candidate.get("counterfactual_text_verifier"))
    if not text_verifier:
        text_verifier = _safe_mapping(candidate.get("text_verifier"))
    if bool(text_verifier.get("relation_only")):
        return 1.0
    return 1.0 if bool(candidate.get("has_relation")) and not (
        bool(candidate.get("has_terminal"))
        or bool(candidate.get("has_strict"))
        or bool(candidate.get("has_multihop"))
    ) else 0.0


def _answer_or_endpoint(candidate: Mapping[str, Any]) -> float:
    text_verifier = _safe_mapping(candidate.get("counterfactual_text_verifier"))
    if not text_verifier:
        text_verifier = _safe_mapping(candidate.get("text_verifier"))
    return 1.0 if bool(text_verifier.get("answer_signal")) or bool(text_verifier.get("endpoint_signal")) else 0.0


def _structural(candidate: Mapping[str, Any]) -> float:
    return 1.0 if (
        bool(candidate.get("has_strict"))
        or bool(candidate.get("has_multihop"))
        or bool(candidate.get("has_terminal"))
        or bool(candidate.get("has_relation"))
        or _answer_or_endpoint(candidate) > 0.0
    ) else 0.0


def _query_text_complexity(query_plan_log: Mapping[str, Any]) -> Dict[str, float]:
    query_condition = _safe_mapping(query_plan_log.get("query_condition"))
    query = str(query_plan_log.get("query") or query_plan_log.get("question") or "")
    lowered = query.lower()
    tokens = [tok for tok in query.split() if tok]
    condition_text = " ".join(
        str(query_condition.get(key, "") or "")
        for key in ("query_type", "intent", "answer_type", "family", "mode")
    ).lower()
    families = " ".join(str(value) for value in query_condition.get("relation_families", []) or []).lower()
    signatures = " ".join(str(value) for value in query_condition.get("relation_signatures", []) or []).lower()
    combined = f"{lowered} {condition_text} {families} {signatures}"
    complex_condition = any(
        bool(query_condition.get(key))
        for key in (
            "complex",
            "multihop",
            "multi_hop",
            "comparative",
            "comparison",
            "bridge",
            "composition",
            "constraint",
        )
    )
    complex_terms = (
        "multi",
        "hop",
        "bridge",
        "comparison",
        "comparative",
        "composition",
        "constraint",
    )
    relative_markers = (
        " whose ",
        " which ",
        " that ",
        " where ",
        " when ",
        " before ",
        " after ",
        " larger than ",
        " smaller than ",
        " older than ",
        " younger than ",
        " more than ",
        " less than ",
        " greater than ",
        " compared ",
        " located ",
        " headquarters ",
        " headquarter ",
        " record label ",
        " signed by ",
        " born in ",
        " died in ",
        " founded by ",
        " part of ",
        " member of ",
        " the person who ",
        " the city where ",
        " the organization that ",
        " the company that ",
        " the country where ",
        " the film that ",
        " the album that ",
        " the team that ",
    )
    nested_who_when = (
        " who " in lowered.strip()[4:]
        or " when " in lowered.strip()[5:]
        or " where " in lowered.strip()[6:]
    )
    long_query = len(tokens) > 12
    marker_hit = any(marker in f" {lowered} " for marker in relative_markers)
    term_hit = any(term in combined for term in complex_terms)
    complexish = bool(complex_condition or long_query or marker_hit or nested_who_when or term_hit)
    return {
        "complex": 1.0 if complexish else 0.0,
        "long": 1.0 if long_query else 0.0,
        "marker": 1.0 if marker_hit else 0.0,
        "condition": 1.0 if complex_condition or term_hit else 0.0,
    }


def _query_simple_factoid(query_plan_log: Mapping[str, Any]) -> float:
    query_condition = _safe_mapping(query_plan_log.get("query_condition"))
    complexity = _query_text_complexity(query_plan_log)
    if complexity["complex"] > 0.0:
        return 0.0
    if "simple_factoid" in query_condition:
        return _bool_feature(query_condition.get("simple_factoid"))
    return 1.0


def _query_complex(query_plan_log: Mapping[str, Any]) -> float:
    complexity = _query_text_complexity(query_plan_log)
    if complexity["complex"] > 0.0:
        return 1.0
    return 1.0 - _query_simple_factoid(query_plan_log)


def extract_set_counterfactual_features(
    query_plan_log: Mapping[str, Any],
    candidate: Mapping[str, Any],
    replacement_info: Mapping[str, Any] = None,
    set_info: Mapping[str, Any] = None,
) -> Dict[str, float]:
    """Pairwise swap features plus runtime-observable top-k set state."""
    set_info = _safe_mapping(set_info)
    replacement_info = _safe_mapping(replacement_info)
    features = extract_rank2_pairwise_features(query_plan_log, candidate, replacement_info)
    selected_values = [
        _safe_float(value, 0.0)
        for value in (set_info.get("selected_values") or [])
    ]
    selected_text_scores = [
        _safe_float(value, 0.0)
        for value in (set_info.get("selected_text_scores") or [])
    ]
    final_top_k = max(_safe_int(set_info.get("final_top_k"), 5), 1)
    selected_len = _safe_int(set_info.get("selected_len"), len(selected_values))
    replacement_index = _safe_int(
        set_info.get("replacement_index"),
        _safe_int(replacement_info.get("rank"), 2) - 1,
    )
    replacement_rank = replacement_index + 1 if replacement_index >= 0 else _safe_int(
        replacement_info.get("rank"),
        2,
    )
    top2_values = selected_values[:2]
    top5_values = selected_values[:5]
    top5_text = selected_text_scores[:5]
    top2_sum = sum(top2_values)
    top5_sum = sum(top5_values)
    top2_mean = top2_sum / max(len(top2_values), 1)
    top5_mean = top5_sum / max(len(top5_values), 1)
    top2_min = min(top2_values) if top2_values else 0.0
    top5_min = min(top5_values) if top5_values else 0.0
    top5_text_mean = sum(top5_text) / max(len(top5_text), 1)
    candidate_value = _safe_float(features.get("candidate_value"), 0.0)
    replacement_value = _safe_float(features.get("replacement_value"), 0.0)
    if "candidate_value_override" in set_info:
        candidate_value = _safe_float(set_info.get("candidate_value_override"), candidate_value)
        features["candidate_value"] = candidate_value
    if "replacement_value_override" in set_info:
        replacement_value = _safe_float(set_info.get("replacement_value_override"), replacement_value)
        features["replacement_value"] = replacement_value
    features["value_margin"] = candidate_value - replacement_value
    candidate_source = str(set_info.get("candidate_source") or "")
    selected_sources = {str(value) for value in (set_info.get("selected_sources") or []) if str(value)}
    target_top2 = bool(set_info.get("target_top2"))
    query = str(query_plan_log.get("query") or query_plan_log.get("question") or "")
    features.update({
        "set_target_top2": 1.0 if target_top2 else 0.0,
        "set_target_top5": 0.0 if target_top2 else 1.0,
        "set_selected_len": float(selected_len),
        "set_final_top_k": float(final_top_k),
        "set_replacement_index": float(max(replacement_index, 0)),
        "set_replacement_index_inv": 1.0 / float(max(replacement_rank, 1)),
        "set_replacement_index_log": math.log1p(float(max(replacement_rank, 1))),
        "set_top2_value_sum": float(top2_sum),
        "set_top2_value_mean": float(top2_mean),
        "set_top2_value_min": float(top2_min),
        "set_top5_value_sum": float(top5_sum),
        "set_top5_value_mean": float(top5_mean),
        "set_top5_value_min": float(top5_min),
        "set_candidate_value_minus_top2_mean": float(candidate_value - top2_mean),
        "set_candidate_value_minus_top5_mean": float(candidate_value - top5_mean),
        "set_candidate_value_minus_top5_min": float(candidate_value - top5_min),
        "set_candidate_text_minus_top5_mean": float(_text_score(candidate) - top5_text_mean),
        "set_candidate_source_seen_top5": 1.0 if candidate_source and candidate_source in selected_sources else 0.0,
        "set_candidate_new_source_top5": 1.0 if candidate_source and candidate_source not in selected_sources else 0.0,
        "set_candidate_answer_or_endpoint": _answer_or_endpoint(candidate),
        "set_candidate_relation_only": _relation_only(candidate),
        "set_candidate_structural": _structural(candidate),
        "set_replacement_relation_only": _safe_float(set_info.get("replacement_relation_only"), 0.0),
        "set_replacement_answer_or_endpoint": _safe_float(set_info.get("replacement_answer_or_endpoint"), 0.0),
        "set_query_simple_factoid": _query_simple_factoid(query_plan_log),
        "set_query_complex": _query_complex(query_plan_log),
        "set_query_length_log": math.log1p(float(len([tok for tok in query.split() if tok]))),
    })
    return {name: float(features.get(name, 0.0)) for name in SET_COUNTERFACTUAL_FEATURE_NAMES}


class QueryPlanRank2PairwiseOptimizer:
    def __init__(self, model_path: str):
        self.model_path = str(model_path or "")
        self.model = self._load_model(self.model_path)

    @staticmethod
    def _load_model(model_path: str) -> Dict[str, Any]:
        if not model_path or not os.path.exists(model_path):
            return {}
        with open(model_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}

    def is_ready(self) -> bool:
        return bool(self.model.get("feature_names")) and bool(self.model.get("weights"))

    def score_features(self, features: Mapping[str, float]) -> float:
        if not self.is_ready():
            return 0.0
        names = list(self.model.get("feature_names", PAIRWISE_FEATURE_NAMES) or PAIRWISE_FEATURE_NAMES)
        weights = [float(value) for value in self.model.get("weights", [])]
        means = [float(value) for value in self.model.get("means", [0.0] * len(names))]
        stds = [
            float(value) if abs(float(value)) > 1e-9 else 1.0
            for value in self.model.get("stds", [1.0] * len(names))
        ]
        bias = float(self.model.get("bias", 0.0) or 0.0)
        values = feature_vector(features, names)
        total = bias
        for index, value in enumerate(values):
            if index >= len(weights):
                break
            total += weights[index] * ((value - means[index]) / stds[index])
        return sigmoid(total)

    def score(
        self,
        query_plan_log: Mapping[str, Any],
        candidate: Mapping[str, Any],
        replacement_info: Mapping[str, Any] = None,
    ) -> Dict[str, Any]:
        features = extract_rank2_pairwise_features(query_plan_log, candidate, replacement_info)
        score = self.score_features(features)
        return {
            "enabled": True,
            "model_path": self.model_path,
            "model_ready": self.is_ready(),
            "score": round(float(score), 6),
            "feature_snapshot": {
                "candidate_original_rank_log": round(features.get("candidate_original_rank_log", 0.0), 4),
                "candidate_confidence": round(features.get("candidate_candidate_confidence", 0.0), 4),
                "candidate_value": round(features.get("candidate_value", 0.0), 4),
                "replacement_value": round(features.get("replacement_value", 0.0), 4),
                "value_margin": round(features.get("value_margin", 0.0), 4),
            "source_score": round(features.get("source_score", 0.0), 4),
            "path_aware_score": round(features.get("path_aware_score", 0.0), 4),
            "text_verifier_score": round(features.get("text_verifier_score", 0.0), 4),
            "evidence_join_score": round(features.get("evidence_join_score", 0.0), 4),
            "answer_pairwise_score": round(features.get("answer_pairwise_score", 0.0), 4),
            "answer_pairwise_margin": round(features.get("answer_pairwise_margin", 0.0), 4),
            "provenance_groupwise_score": round(features.get("provenance_groupwise_score", 0.0), 4),
            "schema_slot_utility_score": round(features.get("schema_slot_utility_score", 0.0), 4),
            },
        }
