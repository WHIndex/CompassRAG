import json
import os
from collections import Counter
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

from .query_plan_promotion_policy import (
    QueryPlanPassagePromotionPolicy,
    _bucket_float,
    _bucket_rank,
    _normalize_label,
)


STATE_FEATURE_GROUPS = [
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
        "evidence_bucket",
        "coverage_bucket",
        "structural_gain_bucket",
        "passage_policy",
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
        "coverage_bucket",
        "structural_gain_bucket",
    ],
    [
        "query_type",
        "rank_bucket",
        "path_bucket",
        "match_mode",
        "fallback",
        "answer_type",
        "coverage_bucket",
    ],
    [
        "query_type",
        "rank_bucket",
        "path_bucket",
        "answer_type",
        "new_relation",
        "new_role",
    ],
    [
        "query_type",
        "rank_bucket",
        "match_mode",
        "fallback",
        "passage_policy",
    ],
]


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


def _bucket_count(value: int) -> str:
    value = int(value or 0)
    if value <= 0:
        return "zero"
    if value == 1:
        return "one"
    if value == 2:
        return "two"
    return "three_plus"


class QueryPlanStateRetrieverPolicy:
    """Small rule-table policy for state retriever intervention.

    The passage reranker policy answers "is this ontology passage generally
    promotable?". The state retriever needs a different decision: "should this
    passage be allowed to replace a baseline passage in the evidence state?".
    This class keeps that decision learnable from trace labels without adding a
    heavy ML dependency.
    """

    def __init__(self, rules: Sequence[Mapping[str, Any]], metadata: Optional[Mapping[str, Any]] = None):
        self.rules = list(rules or [])
        self.metadata = dict(metadata or {})
        self.passage_policy = QueryPlanPassagePromotionPolicy(rules=[], metadata={})

    @classmethod
    def load(cls, path: str) -> "QueryPlanStateRetrieverPolicy":
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
    def _has_answer_path_verified(contexts: Sequence[Mapping[str, Any]]) -> bool:
        return any(bool(context.get("answer_path_verified")) for context in contexts)

    def _has_concrete_anchor(self, contexts: Sequence[Mapping[str, Any]]) -> bool:
        return self.passage_policy._has_concrete_anchor(contexts)

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
        return _normalize_label(self.passage_policy.verifier.infer_answer_form(query, dict(query_condition or {})))

    @staticmethod
    def _passage_policy_label(passage_policy_decision: Optional[Mapping[str, Any]]) -> str:
        policy = dict(passage_policy_decision or {})
        if not policy:
            return "no_policy"
        if policy.get("approved"):
            return "approved"
        reason = str(policy.get("reason", "rejected") or "rejected")
        return f"rejected_{_normalize_label(reason, default='unknown')}"

    def extract_features(
        self,
        query: str,
        query_condition: Optional[Mapping[str, Any]],
        query_type: str,
        original_rank: int,
        confidence: float,
        evidence_score: float,
        baseline_score: float,
        coverage_gain: float,
        redundancy: float,
        added: Mapping[str, int],
        contexts: Sequence[Mapping[str, Any]],
        passage_policy_decision: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, str]:
        contexts = list(contexts or [])
        max_path_len = max((self._path_len(context) for context in contexts), default=0)
        terminal_compatible = self.passage_policy.verifier.terminal_compatible(
            query,
            dict(query_condition or {}),
            list(contexts),
        )
        structural_gain = (
            int(added.get("roles", 0) or 0)
            + int(added.get("relations", 0) or 0)
            + int(added.get("plans", 0) or 0)
            + int(added.get("verified_paths", 0) or 0)
        )
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
            "evidence_bucket": _bucket_float(float(evidence_score or 0.0), high=0.70, mid=0.35),
            "baseline_bucket": _bucket_float(float(baseline_score or 0.0), high=0.60, mid=0.35),
            "coverage_bucket": _bucket_float(float(coverage_gain or 0.0), high=1.40, mid=0.65),
            "redundancy_bucket": _bucket_float(float(redundancy or 0.0), high=1.00, mid=0.35),
            "structural_gain_bucket": _bucket_count(structural_gain),
            "role_gain_bucket": _bucket_count(int(added.get("roles", 0) or 0)),
            "relation_gain_bucket": _bucket_count(int(added.get("relations", 0) or 0)),
            "plan_gain_bucket": _bucket_count(int(added.get("plans", 0) or 0)),
            "terminal_gain_bucket": _bucket_count(int(added.get("terminals", 0) or 0)),
            "new_role": "yes" if int(added.get("roles", 0) or 0) > 0 else "no",
            "new_relation": "yes" if int(added.get("relations", 0) or 0) > 0 else "no",
            "new_plan": "yes" if int(added.get("plans", 0) or 0) > 0 else "no",
            "new_terminal": "yes" if int(added.get("terminals", 0) or 0) > 0 else "no",
            "answer_path_verified": "yes" if self._has_answer_path_verified(contexts) else "no",
            "passage_policy": self._passage_policy_label(passage_policy_decision),
        }
        return features

    @staticmethod
    def signature(features: Mapping[str, str], group: Sequence[str]) -> str:
        return "|".join(f"{key}={features.get(key, 'unknown')}" for key in group)

    def evaluate(self, features: Mapping[str, str]) -> Dict[str, Any]:
        for rule in self.rules:
            group = rule.get("features", []) or []
            values = rule.get("values", {}) or {}
            if all(str(features.get(key, "unknown")) == str(values.get(key, "")) for key in group):
                return {
                    "approved": True,
                    "reason": "state_policy_rule_matched",
                    "rule_id": rule.get("rule_id"),
                    "precision": rule.get("precision"),
                    "support": rule.get("support"),
                    "features": list(group),
                }
        return {
            "approved": False,
            "reason": "no_state_policy_rule_matched",
        }


def build_rule_values(features: Mapping[str, str], group: Sequence[str]) -> Dict[str, str]:
    return {key: str(features.get(key, "unknown")) for key in group}


def rank_rules(rules: Sequence[Mapping[str, Any]]) -> Sequence[Mapping[str, Any]]:
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
