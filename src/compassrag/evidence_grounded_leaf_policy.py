import json
import math
import os
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np


FEATURE_NAMES = [
    "match_score",
    "answer_type_match",
    "direct_overlap_count",
    "direct_overlap_ratio",
    "soft_overlap_count",
    "soft_overlap_ratio",
    "route_score",
    "active_leaf_count_log",
    "active_leaf_count_inv",
    "plan_support_log",
    "plan_priority",
    "plan_estimated_precision",
    "feedback_candidate_precision",
    "feedback_gt5_precision",
    "feedback_selected_precision",
    "feedback_candidate_count_log",
    "feedback_selected_count_log",
    "feedback_small_sample",
    "feedback_approved",
    "tree_leaf_plan",
    "failure_expanded_plan",
    "flat_schema_plan",
    "query_signature_plan",
    "relation_terms_count_log",
    "signature_terms_count_log",
    "bridge_terms_count_log",
    "required_terms_count_log",
    "min_direct_overlap",
    "min_required_overlap",
    "answer_type_date",
    "answer_type_location",
    "answer_type_person",
    "answer_type_number",
    "answer_type_entity",
    "intent_multihop",
    "intent_simple",
]

GENERIC_TERMS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "has",
    "have",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "whose",
    "with",
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


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _normalize(value: Any) -> str:
    text = str(value or "").lower().replace("_", " ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _tokens(value: Any) -> List[str]:
    output = []
    seen = set()
    for token in _normalize(value).split():
        if token in GENERIC_TERMS or token in seen:
            continue
        output.append(token)
        seen.add(token)
    return output


def _token_set(values: Iterable[Any]) -> set[str]:
    return {token for value in values or [] for token in _tokens(value)}


def infer_answer_type(query: str) -> str:
    q_tokens = set(re.findall(r"[a-z0-9]+", str(query or "").lower().replace("_", " ")))
    q = " ".join(sorted(q_tokens))
    if any(token in q_tokens for token in ("when", "date", "year")):
        return "date"
    if any(token in q_tokens for token in ("where", "country", "city", "state", "county", "place")):
        return "location"
    if q.startswith("who ") or " who " in f" {q} ":
        return "person"
    if "who" in q_tokens:
        return "person"
    if any(token in q_tokens for token in ("how", "many", "number", "count")):
        return "number"
    return "entity"


def _plan_overlap_features(query: str, plan: Mapping[str, Any]) -> tuple[float, float, float, float]:
    overlap = plan.get("query_overlap", {}) if isinstance(plan, Mapping) else {}
    if isinstance(overlap, dict) and ("direct" in overlap or "soft" in overlap):
        direct = {str(item) for item in overlap.get("direct", []) or [] if str(item)}
        soft = {str(item) for item in overlap.get("soft", []) or [] if str(item)}
        query_tokens = set(_tokens(query))
        return (
            float(len(direct)),
            float(len(direct)) / max(min(len(query_tokens), 8), 1),
            float(len(soft)),
            float(len(soft)) / max(min(len(query_tokens), 8), 1),
        )

    query_tokens = set(_tokens(query))
    trigger_tokens = _token_set(plan.get("triggers", []) or [])
    relation_tokens = _token_set(plan.get("relation_terms", []) or [])
    signature_tokens = _token_set(plan.get("support_passage_signatures", []) or [])
    bridge_tokens = _token_set(plan.get("bridge_terms", []) or [])
    intent_tokens = set(_tokens(plan.get("intent", "")))
    direct = query_tokens & (trigger_tokens | relation_tokens | intent_tokens)
    soft = query_tokens & (signature_tokens | bridge_tokens)
    return (
        float(len(direct)),
        float(len(direct)) / max(min(len(query_tokens), len(trigger_tokens | relation_tokens | intent_tokens)), 1),
        float(len(soft)),
        float(len(soft)) / max(min(len(query_tokens), len(signature_tokens | bridge_tokens)), 1),
    )


def _feature_answer_type(answer_type: str) -> Dict[str, float]:
    answer_type = str(answer_type or "entity").lower()
    return {
        "answer_type_date": 1.0 if answer_type == "date" else 0.0,
        "answer_type_location": 1.0 if answer_type == "location" else 0.0,
        "answer_type_person": 1.0 if answer_type == "person" else 0.0,
        "answer_type_number": 1.0 if answer_type == "number" else 0.0,
        "answer_type_entity": 1.0 if answer_type not in {"date", "location", "person", "number"} else 0.0,
    }


def extract_leaf_policy_features(query: str, plan: Mapping[str, Any]) -> Dict[str, float]:
    plan = plan if isinstance(plan, Mapping) else {}
    feedback = plan.get("retrieval_feedback", {}) if isinstance(plan.get("retrieval_feedback", {}), Mapping) else {}
    tree_route = plan.get("tree_route", {}) if isinstance(plan.get("tree_route", {}), Mapping) else {}
    query_answer_type = infer_answer_type(query)
    plan_answer_type = str(plan.get("answer_type") or "entity").lower()
    direct_count, direct_ratio, soft_count, soft_ratio = _plan_overlap_features(query, plan)

    plan_id = str(plan.get("id", "") or "")
    intent = str(plan.get("intent", "") or "").lower()
    relation_terms = plan.get("relation_terms", []) or []
    signatures = plan.get("support_passage_signatures", []) or []
    bridge_terms = plan.get("bridge_terms", []) or []
    required_terms = plan.get("required_query_terms", []) or []
    candidate_count = _safe_float(feedback.get("candidate_count"), 0.0)
    selected_count = _safe_float(feedback.get("selected_count"), 0.0)
    features = {
        "match_score": _safe_float(plan.get("match_score"), 0.0),
        "answer_type_match": 1.0 if query_answer_type == plan_answer_type else 0.0,
        "direct_overlap_count": direct_count,
        "direct_overlap_ratio": direct_ratio,
        "soft_overlap_count": soft_count,
        "soft_overlap_ratio": soft_ratio,
        "route_score": _safe_float(tree_route.get("route_score"), 0.0),
        "active_leaf_count_log": math.log1p(_safe_float(tree_route.get("active_leaf_count"), 0.0)),
        "active_leaf_count_inv": 1.0 / max(_safe_float(tree_route.get("active_leaf_count"), 1.0), 1.0),
        "plan_support_log": math.log1p(_safe_float(plan.get("support"), 0.0)),
        "plan_priority": _safe_float(plan.get("priority"), 0.0) + _safe_float(feedback.get("priority"), 0.0),
        "plan_estimated_precision": _safe_float(plan.get("estimated_precision", plan.get("confidence")), 0.0),
        "feedback_candidate_precision": _safe_float(feedback.get("candidate_precision"), 0.0),
        "feedback_gt5_precision": _safe_float(feedback.get("candidate_gt5_precision"), 0.0),
        "feedback_selected_precision": _safe_float(feedback.get("selected_precision"), 0.0),
        "feedback_candidate_count_log": math.log1p(candidate_count),
        "feedback_selected_count_log": math.log1p(selected_count),
        "feedback_small_sample": 1.0 if candidate_count < 5.0 and selected_count < 2.0 else 0.0,
        "feedback_approved": 1.0 if bool(feedback.get("approved", False)) else 0.0,
        "tree_leaf_plan": 1.0 if plan_id.startswith("egtree::") else 0.0,
        "failure_expanded_plan": 1.0 if "failure_expanded" in intent or "missed_gold" in plan_id else 0.0,
        "flat_schema_plan": 1.0 if plan_id.startswith("eg::") else 0.0,
        "query_signature_plan": 1.0 if signatures else 0.0,
        "relation_terms_count_log": math.log1p(len(relation_terms)),
        "signature_terms_count_log": math.log1p(len(signatures)),
        "bridge_terms_count_log": math.log1p(len(bridge_terms)),
        "required_terms_count_log": math.log1p(len(required_terms)),
        "min_direct_overlap": float(_safe_int(plan.get("min_direct_overlap"), 0)),
        "min_required_overlap": float(_safe_int(plan.get("min_required_query_overlap"), 0)),
        "intent_multihop": 1.0 if any(term in intent for term in ("multi", "hop", "bridge", "composition")) else 0.0,
        "intent_simple": 1.0 if "simple" in intent or "single" in intent else 0.0,
    }
    features.update(_feature_answer_type(plan_answer_type))
    return {name: float(features.get(name, 0.0)) for name in FEATURE_NAMES}


def feature_vector(features: Mapping[str, float], feature_names: Optional[List[str]] = None) -> List[float]:
    names = feature_names or FEATURE_NAMES
    return [float(features.get(name, 0.0)) for name in names]


class EvidenceGroundedLeafPolicy:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = self._load_model(model_path)

    @staticmethod
    def _load_model(model_path: str) -> Dict[str, Any]:
        if not model_path or not os.path.exists(model_path):
            return {}
        with open(model_path, "r", encoding="utf-8") as model_file:
            model = json.load(model_file)
        return model if isinstance(model, dict) else {}

    def is_ready(self) -> bool:
        return bool(self.model.get("weights")) and bool(self.model.get("feature_names"))

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
            mean = means[index] if index < len(means) else 0.0
            std = stds[index] if index < len(stds) and stds[index] != 0.0 else 1.0
            total += weights[index] * ((value - mean) / std)
        return _sigmoid(total)

    def score_plan(self, query: str, plan: Mapping[str, Any]) -> tuple[float, Dict[str, float]]:
        features = extract_leaf_policy_features(query, plan)
        return self.score_features(features), features


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-values))
