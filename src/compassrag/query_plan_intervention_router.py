import json
import math
import os
from typing import Any, Dict, Iterable, List, Mapping, Sequence


FEATURE_NAMES = [
    "candidate_count_log",
    "candidate_rank_le_5_frac",
    "candidate_rank_le_10_frac",
    "candidate_rank_le_20_frac",
    "min_original_rank_inv",
    "mean_original_rank_log",
    "max_confidence",
    "mean_confidence",
    "max_candidate_score",
    "mean_candidate_score",
    "max_coverage_gain",
    "mean_coverage_gain",
    "max_new_roles",
    "max_new_terminals",
    "max_new_relations",
    "max_new_plans",
    "strict_frac",
    "multihop_frac",
    "terminal_frac",
    "relation_frac",
    "evidence_grounded_context_count_log",
    "support_context_count_log",
    "missing_context_count_log",
    "max_source_precision",
    "max_source_gt5_precision",
    "max_source_selected_precision",
    "strong_source_frac",
    "max_leaf_policy_score",
    "max_plan_effective_precision",
    "max_plan_feedback_gt5",
    "max_plan_feedback_selected",
    "intervention_approved_frac",
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
        "query_condition_multihop": 1.0 if any(
            token in combined for token in ("multi", "bridge", "hop", "composition", "comparison")
        ) else 0.0,
        "query_condition_simple": 1.0 if any(token in combined for token in ("simple", "factoid", "single")) else 0.0,
        "answer_type_date": 1.0 if answer_type == "date" else 0.0,
        "answer_type_location": 1.0 if answer_type == "location" else 0.0,
        "answer_type_person": 1.0 if answer_type == "person" else 0.0,
        "answer_type_number": 1.0 if answer_type == "number" else 0.0,
        "answer_type_entity": 1.0 if answer_type not in {"date", "location", "person", "number"} else 0.0,
    }


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _frac(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def extract_query_intervention_router_features(
    query_plan_log: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
) -> Dict[str, float]:
    usable = [candidate for candidate in candidates if isinstance(candidate, Mapping)]
    ranks = []
    confidences = []
    candidate_scores = []
    coverage_gains = []
    new_roles = []
    new_terminals = []
    new_relations = []
    new_plans = []
    strict_flags = []
    multihop_flags = []
    terminal_flags = []
    relation_flags = []
    evidence_grounded_contexts = 0
    support_contexts = 0
    missing_contexts = 0
    source_precisions = []
    source_gt5_precisions = []
    source_selected_precisions = []
    strong_source_flags = []
    leaf_scores = []
    plan_effective_precisions = []
    plan_feedback_gt5 = []
    plan_feedback_selected = []
    intervention_flags = []
    counterfactual_flags = []
    counterfactual_values = []
    counterfactual_margins = []

    for candidate in usable:
        original_rank = _safe_int(candidate.get("original_rank"), 10_000)
        if original_rank <= 0:
            original_rank = 10_000
        ranks.append(float(original_rank))
        confidences.append(_safe_float(candidate.get("confidence"), 0.0))
        candidate_scores.append(_safe_float(candidate.get("score"), 0.0))
        coverage_gains.append(_safe_float(candidate.get("coverage_gain"), 0.0))
        delta = candidate.get("delta", {}) or {}
        if not isinstance(delta, Mapping):
            delta = {}
        new_roles.append(_safe_float(delta.get("new_roles"), 0.0))
        new_terminals.append(_safe_float(delta.get("new_terminals"), 0.0))
        new_relations.append(_safe_float(delta.get("new_relations"), 0.0))
        new_plans.append(_safe_float(delta.get("new_plans"), 0.0))
        strict_flags.append(_safe_bool(candidate.get("has_strict")))
        multihop_flags.append(_safe_bool(candidate.get("has_multihop")))
        terminal_flags.append(_safe_bool(candidate.get("has_terminal")))
        relation_flags.append(_safe_bool(candidate.get("has_relation")))
        source_policy = candidate.get("source_policy", {}) or {}
        if not isinstance(source_policy, Mapping):
            source_policy = {}
        source_precisions.append(_safe_float(source_policy.get("precision"), 0.0))
        source_gt5_precisions.append(_safe_float(source_policy.get("gt5_precision"), 0.0))
        source_selected_precisions.append(_safe_float(source_policy.get("selected_precision"), 0.0))
        strong_source_flags.append(1.0 if str(source_policy.get("tier", "") or "").lower() == "strong" else 0.0)
        intervention_policy = candidate.get("intervention_policy", {}) or {}
        if isinstance(intervention_policy, Mapping):
            intervention_flags.append(_safe_bool(intervention_policy.get("approved")))
        counterfactual_policy = candidate.get("counterfactual_policy", {}) or {}
        if isinstance(counterfactual_policy, Mapping):
            counterfactual_flags.append(_safe_bool(counterfactual_policy.get("approved")))
            cf_features = counterfactual_policy.get("features", {}) or {}
            if isinstance(cf_features, Mapping):
                counterfactual_values.append(_safe_float(cf_features.get("candidate_value"), 0.0))
                counterfactual_margins.append(_safe_float(cf_features.get("margin"), 0.0))
        contexts = _contexts(candidate)
        for context in contexts:
            if context.get("evidence_grounded"):
                evidence_grounded_contexts += 1
            if context.get("support_evidence"):
                support_contexts += 1
            if context.get("missing_evidence"):
                missing_contexts += 1
            leaf_policy = context.get("leaf_policy", {}) or {}
            if isinstance(leaf_policy, Mapping):
                leaf_scores.append(_safe_float(leaf_policy.get("score"), 0.0))
        plan_effective_precisions.append(_max_context_value(contexts, ("plan_effective_precision", "effective_precision", "plan_precision")))
        plan_feedback_gt5.append(_max_context_value(contexts, ("plan_feedback_gt5_precision",)))
        plan_feedback_selected.append(_max_context_value(contexts, ("plan_feedback_selected_precision",)))

    candidate_count = len(usable)
    min_rank = min(ranks) if ranks else 10_000.0
    features = {
        "candidate_count_log": math.log1p(float(candidate_count)),
        "candidate_rank_le_5_frac": _frac([1.0 if rank <= 5 else 0.0 for rank in ranks]),
        "candidate_rank_le_10_frac": _frac([1.0 if rank <= 10 else 0.0 for rank in ranks]),
        "candidate_rank_le_20_frac": _frac([1.0 if rank <= 20 else 0.0 for rank in ranks]),
        "min_original_rank_inv": 1.0 / max(min_rank, 1.0),
        "mean_original_rank_log": _mean([math.log1p(rank) for rank in ranks]),
        "max_confidence": max(confidences) if confidences else 0.0,
        "mean_confidence": _mean(confidences),
        "max_candidate_score": max(candidate_scores) if candidate_scores else 0.0,
        "mean_candidate_score": _mean(candidate_scores),
        "max_coverage_gain": max(coverage_gains) if coverage_gains else 0.0,
        "mean_coverage_gain": _mean(coverage_gains),
        "max_new_roles": max(new_roles) if new_roles else 0.0,
        "max_new_terminals": max(new_terminals) if new_terminals else 0.0,
        "max_new_relations": max(new_relations) if new_relations else 0.0,
        "max_new_plans": max(new_plans) if new_plans else 0.0,
        "strict_frac": _frac(strict_flags),
        "multihop_frac": _frac(multihop_flags),
        "terminal_frac": _frac(terminal_flags),
        "relation_frac": _frac(relation_flags),
        "evidence_grounded_context_count_log": math.log1p(float(evidence_grounded_contexts)),
        "support_context_count_log": math.log1p(float(support_contexts)),
        "missing_context_count_log": math.log1p(float(missing_contexts)),
        "max_source_precision": max(source_precisions) if source_precisions else 0.0,
        "max_source_gt5_precision": max(source_gt5_precisions) if source_gt5_precisions else 0.0,
        "max_source_selected_precision": max(source_selected_precisions) if source_selected_precisions else 0.0,
        "strong_source_frac": _frac(strong_source_flags),
        "max_leaf_policy_score": max(leaf_scores) if leaf_scores else 0.0,
        "max_plan_effective_precision": max(plan_effective_precisions) if plan_effective_precisions else 0.0,
        "max_plan_feedback_gt5": max(plan_feedback_gt5) if plan_feedback_gt5 else 0.0,
        "max_plan_feedback_selected": max(plan_feedback_selected) if plan_feedback_selected else 0.0,
        "intervention_approved_frac": _frac(intervention_flags),
        "counterfactual_approved_frac": _frac(counterfactual_flags),
        "max_counterfactual_candidate_value": max(counterfactual_values) if counterfactual_values else 0.0,
        "max_counterfactual_margin": max(counterfactual_margins) if counterfactual_margins else 0.0,
        "matched_plan_count_log": math.log1p(_safe_float(query_plan_log.get("matched_plan_count"), 0.0)),
        "strict_executable_path_count_log": math.log1p(_safe_float(query_plan_log.get("strict_executable_path_count"), 0.0)),
        "executable_path_count_log": math.log1p(_safe_float(query_plan_log.get("executable_path_count"), 0.0)),
    }
    features.update(_query_condition_features(query_plan_log))
    return {name: float(features.get(name, 0.0)) for name in FEATURE_NAMES}


def feature_vector(features: Mapping[str, float], feature_names: List[str] = None) -> List[float]:
    names = feature_names or FEATURE_NAMES
    return [float(features.get(name, 0.0)) for name in names]


class QueryPlanInterventionRouter:
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

    def score(self, query_plan_log: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        features = extract_query_intervention_router_features(query_plan_log, candidates)
        score = self.score_features(features)
        return {
            "score": round(float(score), 6),
            "model_path": self.model_path,
            "model_ready": self.is_ready(),
            "candidate_count": int(round(math.expm1(features.get("candidate_count_log", 0.0)))),
            "feature_snapshot": {
                "candidate_count_log": round(features.get("candidate_count_log", 0.0), 4),
                "max_confidence": round(features.get("max_confidence", 0.0), 4),
                "max_coverage_gain": round(features.get("max_coverage_gain", 0.0), 4),
                "max_source_gt5_precision": round(features.get("max_source_gt5_precision", 0.0), 4),
                "max_leaf_policy_score": round(features.get("max_leaf_policy_score", 0.0), 4),
                "max_counterfactual_margin": round(features.get("max_counterfactual_margin", 0.0), 4),
            },
        }
