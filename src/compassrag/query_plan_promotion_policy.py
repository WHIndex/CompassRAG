import json
import os
from collections import Counter
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .query_plan_adapter import QueryPlanOntologyAdapter
from .query_plan_promotion_verifier import QueryPlanPassagePromotionVerifier
from .utils.misc_utils import text_processing


FEATURE_GROUPS = [
    [
        "query_type",
        "rank_bucket",
        "path_bucket",
        "match_mode",
        "fallback",
        "answer_type",
        "terminal_compatible",
        "concrete_anchor",
        "context_bucket",
        "confidence_bucket",
    ],
    [
        "query_type",
        "rank_bucket",
        "path_bucket",
        "match_mode",
        "fallback",
        "answer_type",
        "terminal_compatible",
        "confidence_bucket",
    ],
    [
        "query_type",
        "rank_bucket",
        "path_bucket",
        "match_mode",
        "fallback",
        "answer_type",
    ],
    [
        "query_type",
        "rank_bucket",
        "path_bucket",
        "fallback",
        "answer_type",
    ],
    [
        "query_type",
        "rank_bucket",
        "match_mode",
        "fallback",
    ],
]


def _bucket_rank(rank: int) -> str:
    if rank <= 2:
        return "rank_1_2"
    if rank <= 5:
        return "rank_3_5"
    if rank <= 10:
        return "rank_6_10"
    if rank <= 20:
        return "rank_11_20"
    return "rank_21_plus"


def _bucket_float(value: float, high: float, mid: float) -> str:
    if value >= high:
        return "high"
    if value >= mid:
        return "mid"
    return "low"


def _normalize_label(value: Any, default: str = "unknown") -> str:
    text = text_processing(str(value or "")).replace(" ", "_")
    return text or default


def _dominant(values: Iterable[str], default: str = "unknown") -> str:
    cleaned = [item for item in values if item]
    if not cleaned:
        return default
    counts = Counter(cleaned)
    if len(counts) > 1:
        top_two = counts.most_common(2)
        if top_two[0][1] == top_two[1][1]:
            return "mixed"
    return counts.most_common(1)[0][0]


class QueryPlanPassagePromotionPolicy:
    """Learned gate for query-plan passage promotion.

    The policy is intentionally a small JSON rule table rather than a model
    dependency. It learns which feature combinations have historically moved
    gold supporting passages, then gates online promotion with those rules.
    """

    def __init__(self, rules: Sequence[Mapping[str, Any]], metadata: Optional[Mapping[str, Any]] = None):
        self.rules = list(rules or [])
        self.metadata = dict(metadata or {})
        self.adapter = QueryPlanOntologyAdapter()
        self.verifier = QueryPlanPassagePromotionVerifier()

    @classmethod
    def load(cls, path: str) -> "QueryPlanPassagePromotionPolicy":
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls(rules=payload.get("rules", []) or [], metadata=payload.get("metadata", {}) or {})

    @staticmethod
    def exists(path: Optional[str]) -> bool:
        return bool(path) and os.path.exists(str(path))

    @staticmethod
    def _path_len(context: Mapping[str, Any]) -> int:
        return max(
            len(context.get("fact_indices", []) or []),
            len(context.get("relations", []) or []),
            1,
        )

    @staticmethod
    def _match_mode(contexts: Sequence[Mapping[str, Any]]) -> str:
        modes = {
            str(context.get("signature_match_mode", "") or "").strip().lower()
            for context in contexts
        }
        modes.discard("")
        if not modes:
            return "none"
        if modes == {"strict"}:
            return "strict"
        if modes == {"soft"}:
            return "soft"
        return "mixed"

    @staticmethod
    def _has_simple_fallback(contexts: Sequence[Mapping[str, Any]]) -> bool:
        return any(str(context.get("plan_id", "") or "").startswith("simple_fallback::") for context in contexts)

    @staticmethod
    def _has_concrete_anchor(contexts: Sequence[Mapping[str, Any]]) -> bool:
        anchors = {
            text_processing(str(context.get("anchor_entity", "") or ""))
            for context in contexts
        }
        anchors.discard("")
        if not anchors:
            return False
        generic = QueryPlanPassagePromotionVerifier.GENERIC_ANCHORS
        return any(anchor not in generic and len(anchor) >= 4 for anchor in anchors)

    def infer_answer_type(
        self,
        query: str,
        query_condition: Optional[Mapping[str, Any]],
        contexts: Sequence[Mapping[str, Any]],
    ) -> str:
        context_types = [
            _normalize_label(context.get("plan_answer_type"))
            for context in contexts
            if context.get("plan_answer_type")
        ]
        answer_type = _dominant(context_types, default="")
        if answer_type and answer_type != "mixed":
            return answer_type
        return _normalize_label(self.verifier.infer_answer_form(query, dict(query_condition or {})))

    def extract_features(
        self,
        query: str,
        query_condition: Optional[Mapping[str, Any]],
        query_type: str,
        original_rank: int,
        confidence: float,
        raw_score: float,
        max_raw_score: float,
        passage_scale: float,
        contexts: Sequence[Mapping[str, Any]],
    ) -> Dict[str, str]:
        contexts = list(contexts or [])
        max_raw_score = max(float(max_raw_score or 0.0), 1e-8)
        normalized_lift = min(max(float(raw_score or 0.0) / max_raw_score, 0.0), 1.0)
        max_path_len = max((self._path_len(context) for context in contexts), default=0)
        terminal_compatible = self.verifier.terminal_compatible(query, dict(query_condition or {}), list(contexts))
        features = {
            "query_type": _normalize_label(query_type),
            "rank_bucket": _bucket_rank(int(original_rank)),
            "path_bucket": "multi_hop" if max_path_len >= 2 else "single_hop",
            "match_mode": self._match_mode(contexts),
            "fallback": "simple_fallback" if self._has_simple_fallback(contexts) else "ontology_path",
            "answer_type": self.infer_answer_type(query, query_condition, contexts),
            "terminal_compatible": "yes" if terminal_compatible else "no",
            "concrete_anchor": "yes" if self._has_concrete_anchor(contexts) else "no",
            "context_bucket": "multi_context" if len(contexts) >= 2 else "single_context",
            "confidence_bucket": _bucket_float(float(confidence or 0.0), high=0.85, mid=0.65),
            "lift_bucket": _bucket_float(normalized_lift, high=0.80, mid=0.45),
            "scale_bucket": _bucket_float(float(passage_scale or 0.0), high=0.90, mid=0.60),
        }
        return features

    @staticmethod
    def signature(features: Mapping[str, str], group: Sequence[str]) -> str:
        return "|".join(f"{key}={features.get(key, 'unknown')}" for key in group)

    def evaluate(self, features: Mapping[str, str]) -> Dict[str, Any]:
        for rule in self.rules:
            group = rule.get("features", []) or []
            if all(str(features.get(key, "unknown")) == str(rule.get("values", {}).get(key, "")) for key in group):
                return {
                    "approved": True,
                    "reason": "policy_rule_matched",
                    "rule_id": rule.get("rule_id"),
                    "precision": rule.get("precision"),
                    "support": rule.get("support"),
                    "features": list(group),
                }
        return {
            "approved": False,
            "reason": "no_policy_rule_matched",
        }


def build_rule_values(features: Mapping[str, str], group: Sequence[str]) -> Dict[str, str]:
    return {key: str(features.get(key, "unknown")) for key in group}


def rank_rules(rules: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    return sorted(
        rules,
        key=lambda rule: (
            float(rule.get("precision", 0.0)),
            int(rule.get("gold", 0)),
            int(rule.get("support", 0)),
            len(rule.get("features", []) or []),
        ),
        reverse=True,
    )
