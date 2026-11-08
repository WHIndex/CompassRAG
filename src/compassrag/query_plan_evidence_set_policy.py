import json
import math
import os
from typing import Any, Dict, Iterable, List, Mapping


FEATURE_NAMES = [
    "original_rank_inv",
    "original_rank_log",
    "original_rank_le_5",
    "original_rank_le_10",
    "original_rank_le_20",
    "original_rank_le_80",
    "candidate_confidence",
    "candidate_score",
    "coverage_gain",
    "new_roles",
    "new_terminals",
    "new_relations",
    "new_plans",
    "new_facts",
    "has_strict",
    "has_multihop",
    "has_terminal",
    "has_relation",
    "effective_min_confidence",
    "source_policy_approved",
    "source_policy_precision",
    "source_policy_gt5_precision",
    "source_policy_selected_precision",
    "source_policy_tier_strong",
    "source_policy_tier_moderate",
    "source_policy_tier_weak",
    "intervention_approved",
    "intervention_path_or_endpoint",
    "intervention_anchored_relation",
    "intervention_source_aware",
    "counterfactual_approved",
    "counterfactual_candidate_value",
    "counterfactual_replacement_value",
    "counterfactual_margin",
    "context_count_log",
    "context_evidence_grounded_count",
    "context_support_count",
    "context_missing_count",
    "context_signature_rescue_any",
    "context_leaf_policy_score_max",
    "context_plan_effective_precision_max",
    "context_plan_feedback_gt5_max",
    "context_plan_feedback_selected_max",
    "source_lifted",
    "source_support",
    "source_evidence_grounded",
    "source_missing",
    "source_subgraph",
    "source_terminal",
    "source_structural",
    "source_failure_evidence",
    "source_schema_route",
    "source_evidence_index",
    "evidence_index_matched_anchor",
    "evidence_index_matched_terminal",
    "evidence_index_answer_hit",
    "evidence_index_relation_hit_count_log",
    "evidence_index_slot_score",
    "evidence_index_endpoint_overlap",
    "evidence_index_plan_precision",
    "matched_plan_count_log",
    "strict_executable_path_count_log",
    "executable_path_count_log",
    "query_condition_multihop",
    "query_condition_simple",
    "answer_type_date",
    "answer_type_location",
    "answer_type_person",
    "answer_type_number",
    "answer_type_entity",
    "recall_selector_score",
    "recall_selector_promoted",
    "recall_selector_high_precision",
    "evidence_set_policy_score",
    "evidence_set_policy_promoted",
    "utility_policy_score",
    "utility_policy_promoted",
    "utility_policy_scope_skipped",
    "counterfactual_text_score",
    "counterfactual_text_text_score",
    "counterfactual_text_answer_pairwise_score",
    "counterfactual_text_join_score",
    "counterfactual_text_path_score",
    "counterfactual_text_source_precision",
    "counterfactual_text_query_slot_value",
    "counterfactual_text_slot_value",
    "counterfactual_text_anchor_hits",
    "counterfactual_text_terminal_hits",
    "counterfactual_text_relation_hits",
    "counterfactual_text_query_hits",
    "counterfactual_text_answer_hits",
    "counterfactual_text_answer_signal",
    "counterfactual_text_endpoint_signal",
    "counterfactual_text_structural",
    "counterfactual_text_relation_only",
    "counterfactual_text_confidence",
    "counterfactual_text_rank_cost",
    "schema_route_score",
    "schema_route_entity_hit",
    "schema_route_triple_hit",
    "schema_route_path_hit",
    "schema_route_community_hit",
    "schema_route_answer_hit",
    "schema_route_plan_hit",
    "schema_route_subquery_hits",
    "schema_route_covered_kinds_log",
]


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


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _contexts(candidate: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    contexts = candidate.get("contexts", []) or []
    if not isinstance(contexts, list):
        return []
    return [context for context in contexts if isinstance(context, Mapping)]


def _max_context_value(contexts: Iterable[Mapping[str, Any]], keys: Iterable[str]) -> float:
    value = 0.0
    for context in contexts:
        for key in keys:
            value = max(value, _safe_float(context.get(key), 0.0))
    return value


def _count_context_flag(contexts: Iterable[Mapping[str, Any]], key: str) -> float:
    return float(sum(1 for context in contexts if bool(context.get(key))))


def _query_condition_features(query_plan_log: Mapping[str, Any]) -> Dict[str, float]:
    condition = query_plan_log.get("query_condition", {}) or {}
    if not isinstance(condition, Mapping):
        condition = {}
    answer_type = str(condition.get("answer_type", "") or "").lower()
    text = " ".join(
        str(condition.get(key, "") or "")
        for key in ("query_type", "intent", "answer_type", "family", "mode")
    ).lower()
    relation_families = " ".join(str(item) for item in condition.get("relation_families", []) or []).lower()
    relation_signatures = " ".join(str(item) for item in condition.get("relation_signatures", []) or []).lower()
    combined = f"{text} {relation_families} {relation_signatures}"
    return {
        "query_condition_multihop": 1.0 if any(token in combined for token in ("multi", "bridge", "hop", "composition", "comparison")) else 0.0,
        "query_condition_simple": 1.0 if any(token in combined for token in ("simple", "factoid", "single")) else 0.0,
        "answer_type_date": 1.0 if answer_type == "date" else 0.0,
        "answer_type_location": 1.0 if answer_type == "location" else 0.0,
        "answer_type_person": 1.0 if answer_type == "person" else 0.0,
        "answer_type_number": 1.0 if answer_type == "number" else 0.0,
        "answer_type_entity": 1.0 if answer_type not in {"date", "location", "person", "number"} else 0.0,
    }


def _candidate_source(candidate: Mapping[str, Any], contexts: List[Mapping[str, Any]]) -> str:
    source = str(candidate.get("source") or candidate.get("source_key") or "").lower()
    if not source:
        decision = candidate.get("reranker_decision", {}) or {}
        if isinstance(decision, Mapping):
            source = str(decision.get("source") or "").lower()
    if not source:
        for context in contexts:
            if context.get("failure_evidence"):
                return "failure_evidence"
            if context.get("ontology_evidence_index"):
                return "evidence_index"
            if context.get("schema_route_evidence"):
                return "schema_route"
            if context.get("evidence_grounded"):
                return "evidence_grounded"
            if context.get("support_evidence"):
                return "support"
            if context.get("missing_evidence"):
                return "missing"
            if context.get("subgraph_candidate"):
                return "subgraph"
            if context.get("terminal_evidence"):
                return "terminal"
            if context.get("structural_candidate"):
                return "structural"
        return "lifted"
    if "evidence_grounded" in source:
        return "evidence_grounded"
    if "failure" in source:
        return "failure_evidence"
    if "evidence_index" in source or "ontology_evidence_index" in source:
        return "evidence_index"
    if "schema_route" in source or "schema-route" in source:
        return "schema_route"
    if "support" in source:
        return "support"
    if "missing" in source:
        return "missing"
    if "subgraph" in source:
        return "subgraph"
    if "terminal" in source:
        return "terminal"
    if "structural" in source:
        return "structural"
    return "lifted"


def extract_evidence_set_policy_features(
    query_plan_log: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> Dict[str, float]:
    original_rank = _safe_int(candidate.get("original_rank"), 10_000)
    if original_rank <= 0:
        original_rank = 10_000
    delta = candidate.get("delta", {}) or {}
    if not isinstance(delta, Mapping):
        delta = {}
    source_policy = candidate.get("source_policy", {}) or {}
    if not isinstance(source_policy, Mapping):
        source_policy = {}
    intervention_policy = candidate.get("intervention_policy", {}) or {}
    if not isinstance(intervention_policy, Mapping):
        intervention_policy = {}
    counterfactual_policy = candidate.get("counterfactual_policy", {}) or {}
    if not isinstance(counterfactual_policy, Mapping):
        counterfactual_policy = {}
    recall_selector = candidate.get("recall_selector", {}) or {}
    if not isinstance(recall_selector, Mapping):
        recall_selector = {}
    evidence_set_policy = candidate.get("evidence_set_policy", {}) or {}
    if not isinstance(evidence_set_policy, Mapping):
        evidence_set_policy = {}
    utility_policy = candidate.get("utility_policy", {}) or {}
    if not isinstance(utility_policy, Mapping):
        utility_policy = {}
    counterfactual_text = candidate.get("counterfactual_text_verifier", {}) or {}
    if not isinstance(counterfactual_text, Mapping):
        counterfactual_text = {}
    schema_route = candidate.get("schema_route_adapter", {}) or {}
    if not isinstance(schema_route, Mapping):
        schema_route = {}
    schema_hits = schema_route.get("hits", {}) or {}
    if not isinstance(schema_hits, Mapping):
        schema_hits = {}
    intervention_reason = str(intervention_policy.get("reason", "") or "")
    counterfactual_features = counterfactual_policy.get("features", {}) or {}
    if not isinstance(counterfactual_features, Mapping):
        counterfactual_features = {}
    contexts = _contexts(candidate)
    relation_hits = candidate.get("relation_hits", []) or []
    if not isinstance(relation_hits, list):
        relation_hits = []
    evidence_index_slot_score = _max_context_value(contexts, ("slot_score",))
    evidence_index_endpoint_overlap = 0.0
    for context in contexts:
        reasons = context.get("fact_score_reasons", {}) or {}
        if isinstance(reasons, Mapping):
            evidence_index_endpoint_overlap = max(
                evidence_index_endpoint_overlap,
                _safe_float(reasons.get("endpoint_overlap"), 0.0),
            )
    evidence_index_plan_precision = _max_context_value(
        contexts,
        ("plan_effective_precision", "plan_precision", "effective_precision"),
    )
    leaf_scores = []
    for context in contexts:
        leaf_policy = context.get("leaf_policy", {}) or {}
        if isinstance(leaf_policy, Mapping):
            leaf_scores.append(_safe_float(leaf_policy.get("score"), 0.0))
    source_tier = str(source_policy.get("tier", "") or "").lower()
    source = _candidate_source(candidate, contexts)
    features = {
        "original_rank_inv": 1.0 / float(original_rank),
        "original_rank_log": math.log1p(float(original_rank)),
        "original_rank_le_5": 1.0 if original_rank <= 5 else 0.0,
        "original_rank_le_10": 1.0 if original_rank <= 10 else 0.0,
        "original_rank_le_20": 1.0 if original_rank <= 20 else 0.0,
        "original_rank_le_80": 1.0 if original_rank <= 80 else 0.0,
        "candidate_confidence": _safe_float(candidate.get("confidence"), 0.0),
        "candidate_score": _safe_float(candidate.get("score"), 0.0),
        "coverage_gain": _safe_float(candidate.get("coverage_gain"), 0.0),
        "new_roles": _safe_float(delta.get("new_roles"), 0.0),
        "new_terminals": _safe_float(delta.get("new_terminals"), 0.0),
        "new_relations": _safe_float(delta.get("new_relations"), 0.0),
        "new_plans": _safe_float(delta.get("new_plans"), 0.0),
        "new_facts": _safe_float(delta.get("new_facts"), 0.0),
        "has_strict": _safe_bool(candidate.get("has_strict")),
        "has_multihop": _safe_bool(candidate.get("has_multihop")),
        "has_terminal": _safe_bool(candidate.get("has_terminal")),
        "has_relation": _safe_bool(candidate.get("has_relation")),
        "effective_min_confidence": _safe_float(candidate.get("effective_min_confidence"), 0.0),
        "source_policy_approved": _safe_bool(source_policy.get("approved")),
        "source_policy_precision": _safe_float(source_policy.get("precision"), 0.0),
        "source_policy_gt5_precision": _safe_float(source_policy.get("gt5_precision"), 0.0),
        "source_policy_selected_precision": _safe_float(source_policy.get("selected_precision"), 0.0),
        "source_policy_tier_strong": 1.0 if source_tier == "strong" else 0.0,
        "source_policy_tier_moderate": 1.0 if source_tier == "moderate" else 0.0,
        "source_policy_tier_weak": 1.0 if source_tier == "weak" else 0.0,
        "intervention_approved": _safe_bool(intervention_policy.get("approved")),
        "intervention_path_or_endpoint": 1.0 if "path_or_endpoint" in intervention_reason else 0.0,
        "intervention_anchored_relation": 1.0 if "anchored_relation" in intervention_reason else 0.0,
        "intervention_source_aware": 1.0 if "source_aware" in intervention_reason else 0.0,
        "counterfactual_approved": _safe_bool(counterfactual_policy.get("approved")),
        "counterfactual_candidate_value": _safe_float(counterfactual_features.get("candidate_value"), 0.0),
        "counterfactual_replacement_value": _safe_float(counterfactual_features.get("replacement_value"), 0.0),
        "counterfactual_margin": _safe_float(counterfactual_features.get("margin"), 0.0),
        "context_count_log": math.log1p(float(len(contexts))),
        "context_evidence_grounded_count": _count_context_flag(contexts, "evidence_grounded"),
        "context_support_count": _count_context_flag(contexts, "support_evidence"),
        "context_missing_count": _count_context_flag(contexts, "missing_evidence"),
        "context_signature_rescue_any": 1.0 if any(bool(context.get("feedback_signature_rescue")) for context in contexts) else 0.0,
        "context_leaf_policy_score_max": max(leaf_scores) if leaf_scores else 0.0,
        "context_plan_effective_precision_max": _max_context_value(contexts, ("plan_effective_precision", "effective_precision", "plan_precision")),
        "context_plan_feedback_gt5_max": _max_context_value(contexts, ("plan_feedback_gt5_precision",)),
        "context_plan_feedback_selected_max": _max_context_value(contexts, ("plan_feedback_selected_precision",)),
        "source_lifted": 1.0 if source == "lifted" else 0.0,
        "source_support": 1.0 if source == "support" else 0.0,
        "source_evidence_grounded": 1.0 if source == "evidence_grounded" else 0.0,
        "source_missing": 1.0 if source == "missing" else 0.0,
        "source_subgraph": 1.0 if source == "subgraph" else 0.0,
        "source_terminal": 1.0 if source == "terminal" else 0.0,
        "source_structural": 1.0 if source == "structural" else 0.0,
        "source_failure_evidence": 1.0 if source == "failure_evidence" else 0.0,
        "source_schema_route": 1.0 if source == "schema_route" else 0.0,
        "source_evidence_index": 1.0 if source == "evidence_index" else 0.0,
        "evidence_index_matched_anchor": _safe_bool(candidate.get("matched_anchor")),
        "evidence_index_matched_terminal": _safe_bool(candidate.get("matched_terminal")),
        "evidence_index_answer_hit": _safe_bool(candidate.get("answer_hit")),
        "evidence_index_relation_hit_count_log": math.log1p(float(len(relation_hits))),
        "evidence_index_slot_score": evidence_index_slot_score,
        "evidence_index_endpoint_overlap": evidence_index_endpoint_overlap,
        "evidence_index_plan_precision": evidence_index_plan_precision,
        "matched_plan_count_log": math.log1p(_safe_float(query_plan_log.get("matched_plan_count"), 0.0)),
        "strict_executable_path_count_log": math.log1p(_safe_float(query_plan_log.get("strict_executable_path_count"), 0.0)),
        "executable_path_count_log": math.log1p(_safe_float(query_plan_log.get("executable_path_count"), 0.0)),
        "recall_selector_score": _safe_float(recall_selector.get("score"), 0.0),
        "recall_selector_promoted": 1.0 if str(recall_selector.get("decision", "") or "").lower() == "promote" else 0.0,
        "recall_selector_high_precision": 1.0 if _safe_float(recall_selector.get("score"), 0.0) >= 0.75 else 0.0,
        "evidence_set_policy_score": _safe_float(evidence_set_policy.get("score"), 0.0),
        "evidence_set_policy_promoted": 1.0 if str(evidence_set_policy.get("decision", "") or "").lower() == "promote" else 0.0,
        "utility_policy_score": _safe_float(utility_policy.get("score"), 0.0),
        "utility_policy_promoted": 1.0 if str(utility_policy.get("decision", "") or "").lower() == "promote" else 0.0,
        "utility_policy_scope_skipped": 1.0 if str(utility_policy.get("scope_reason", "") or "").strip() else 0.0,
        "counterfactual_text_score": _safe_float(counterfactual_text.get("score"), 0.0),
        "counterfactual_text_text_score": _safe_float(counterfactual_text.get("text_score"), 0.0),
        "counterfactual_text_answer_pairwise_score": _safe_float(counterfactual_text.get("answer_pairwise_score"), 0.0),
        "counterfactual_text_join_score": _safe_float(counterfactual_text.get("join_score"), 0.0),
        "counterfactual_text_path_score": _safe_float(counterfactual_text.get("path_score"), 0.0),
        "counterfactual_text_source_precision": _safe_float(counterfactual_text.get("source_precision"), 0.0),
        "counterfactual_text_query_slot_value": _safe_float(counterfactual_text.get("query_slot_value"), 0.0),
        "counterfactual_text_slot_value": _safe_float(counterfactual_text.get("slot_value"), 0.0),
        "counterfactual_text_anchor_hits": _safe_float(counterfactual_text.get("anchor_hits"), 0.0),
        "counterfactual_text_terminal_hits": _safe_float(counterfactual_text.get("terminal_hits"), 0.0),
        "counterfactual_text_relation_hits": _safe_float(counterfactual_text.get("relation_hits"), 0.0),
        "counterfactual_text_query_hits": _safe_float(counterfactual_text.get("query_hits"), 0.0),
        "counterfactual_text_answer_hits": _safe_float(counterfactual_text.get("answer_hits"), 0.0),
        "counterfactual_text_answer_signal": _safe_bool(counterfactual_text.get("answer_signal")),
        "counterfactual_text_endpoint_signal": _safe_bool(counterfactual_text.get("endpoint_signal")),
        "counterfactual_text_structural": _safe_bool(counterfactual_text.get("structural")),
        "counterfactual_text_relation_only": _safe_bool(counterfactual_text.get("relation_only")),
        "counterfactual_text_confidence": _safe_float(counterfactual_text.get("confidence"), 0.0),
        "counterfactual_text_rank_cost": _safe_float(counterfactual_text.get("rank_cost"), 0.0),
        "schema_route_score": _safe_float(schema_route.get("score"), 0.0),
        "schema_route_entity_hit": _safe_bool(schema_hits.get("entity")),
        "schema_route_triple_hit": _safe_bool(schema_hits.get("triple")),
        "schema_route_path_hit": _safe_bool(schema_hits.get("path")),
        "schema_route_community_hit": _safe_bool(schema_hits.get("community")),
        "schema_route_answer_hit": _safe_bool(schema_hits.get("answer")),
        "schema_route_plan_hit": _safe_bool(schema_hits.get("plan")),
        "schema_route_subquery_hits": _safe_float(schema_route.get("subquery_hits"), 0.0),
        "schema_route_covered_kinds_log": math.log1p(float(len(schema_route.get("covered_kinds", []) or []))),
    }
    features.update(_query_condition_features(query_plan_log))
    return {name: float(features.get(name, 0.0)) for name in FEATURE_NAMES}


def feature_vector(features: Mapping[str, float], feature_names: List[str] = None) -> List[float]:
    names = feature_names or FEATURE_NAMES
    return [float(features.get(name, 0.0)) for name in names]


class QueryPlanEvidenceSetPolicy:
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
        return sigmoid(total)

    def score(self, query_plan_log: Mapping[str, Any], candidate: Mapping[str, Any]) -> Dict[str, Any]:
        features = extract_evidence_set_policy_features(query_plan_log, candidate)
        score = self.score_features(features)
        return {
            "score": round(float(score), 6),
            "model_path": self.model_path,
            "model_ready": self.is_ready(),
            "feature_snapshot": {
                "original_rank_log": round(features.get("original_rank_log", 0.0), 4),
                "candidate_confidence": round(features.get("candidate_confidence", 0.0), 4),
                "coverage_gain": round(features.get("coverage_gain", 0.0), 4),
                "source_policy_gt5_precision": round(features.get("source_policy_gt5_precision", 0.0), 4),
                "context_leaf_policy_score_max": round(features.get("context_leaf_policy_score_max", 0.0), 4),
                "counterfactual_margin": round(features.get("counterfactual_margin", 0.0), 4),
            },
        }
