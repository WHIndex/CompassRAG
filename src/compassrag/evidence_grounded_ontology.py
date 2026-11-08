import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from .evidence_grounded_leaf_policy import EvidenceGroundedLeafPolicy
from .query_ontology_prior import tokenize
from .query_plan_support_evidence_retriever import QueryPlanSupportEvidenceRetriever
from .utils.misc_utils import text_processing


@dataclass
class EvidenceGroundedResult:
    sorted_doc_ids: np.ndarray
    sorted_doc_scores: np.ndarray
    log: Dict[str, Any] = field(default_factory=dict)


class EvidenceGroundedOntologyMatcher:
    """Match queries to evidence-grounded ontology plans.

    These plans are not abstract relation schemas only. They contain evidence
    locator fields learned from training QA support passages: relation terms,
    support passage signatures, bridge terms, and successful probe templates.
    """

    def __init__(
        self,
        ontology: Dict[str, Any],
        leaf_policy: EvidenceGroundedLeafPolicy | None = None,
        leaf_policy_reject_below: float = 0.0,
        leaf_policy_boost: float = 0.0,
    ):
        self.ontology = ontology or {}
        self.plans = [
            plan for plan in self.ontology.get("plans", [])
            if isinstance(plan, dict)
        ]
        self.leaf_policy = leaf_policy if leaf_policy and leaf_policy.is_ready() else None
        self.leaf_policy_reject_below = min(max(float(leaf_policy_reject_below), 0.0), 1.0)
        self.leaf_policy_boost = max(float(leaf_policy_boost), 0.0)
        tree = self.ontology.get("tree", {})
        self.tree = tree if isinstance(tree, dict) else {}
        self.tree_enabled = bool(self.tree.get("enabled", False))
        self.tree_leaf_by_key = {
            str(leaf.get("key")): leaf
            for leaf in self.tree.get("leaves", []) or []
            if isinstance(leaf, dict) and leaf.get("key")
        }

    @classmethod
    def load(
        cls,
        path: str,
        leaf_policy_path: str = "",
        leaf_policy_reject_below: float = 0.0,
        leaf_policy_boost: float = 0.0,
    ) -> "EvidenceGroundedOntologyMatcher":
        leaf_policy = EvidenceGroundedLeafPolicy(leaf_policy_path) if leaf_policy_path else None
        with open(path, "r", encoding="utf-8") as handle:
            return cls(
                json.load(handle),
                leaf_policy=leaf_policy,
                leaf_policy_reject_below=leaf_policy_reject_below,
                leaf_policy_boost=leaf_policy_boost,
            )

    @staticmethod
    def _tokens(value: Any) -> List[str]:
        return [
            token for token in tokenize(text_processing(str(value or "").replace("_", " ")))
            if token and token not in QueryPlanSupportEvidenceRetriever.GENERIC_TERMS
        ]

    @staticmethod
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

    @staticmethod
    def _retrieval_feedback(plan: Mapping[str, Any]) -> Dict[str, Any]:
        feedback = plan.get("retrieval_feedback", {}) if isinstance(plan, Mapping) else {}
        return feedback if isinstance(feedback, dict) else {}

    @classmethod
    def _feedback_approved(cls, plan: Mapping[str, Any]) -> bool:
        """Return whether a calibrated plan is allowed to drive retrieval.

        Uncalibrated ontologies remain backward compatible. Once a builder has
        written retrieval_feedback, that field becomes the verifier's decision:
        unsupported or harmful plans are skipped instead of being rescued by the
        older support-frequency confidence.
        """
        feedback = cls._retrieval_feedback(plan)
        if not feedback:
            return True
        return bool(feedback.get("approved", False))

    @classmethod
    def _feedback_precision(cls, plan: Mapping[str, Any]) -> float:
        feedback = cls._retrieval_feedback(plan)
        for key in (
            "calibrated_precision",
            "candidate_precision",
            "selected_precision",
            "candidate_gt5_precision",
        ):
            try:
                value = float(feedback.get(key, 0.0))
            except (TypeError, ValueError):
                value = 0.0
            if value > 0:
                return min(max(value, 0.0), 1.0)
        try:
            return min(max(float(plan.get("estimated_precision", plan.get("confidence", 0.0)) or 0.0), 0.0), 1.0)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @classmethod
    def _tree_path(cls, plan: Mapping[str, Any]) -> Dict[str, Any]:
        path = plan.get("tree_path", {}) if isinstance(plan, Mapping) else {}
        return path if isinstance(path, dict) else {}

    @classmethod
    def _tree_leaf_key(cls, plan: Mapping[str, Any]) -> str:
        path = cls._tree_path(plan)
        for key in ("leaf_key", "key"):
            if path.get(key):
                return str(path.get(key))
        parts = [
            str(path.get("query_type", "unknown")),
            str(path.get("answer_type", plan.get("answer_type", "entity"))),
            str(path.get("relation_family", "generic")),
            str(path.get("evidence_slot", "generic")),
        ]
        return "::".join(parts)

    def _tree_leaf_route_scores(self, query_tokens: set[str], answer_type: str) -> Dict[str, float]:
        if not self.tree_enabled or not self.tree_leaf_by_key:
            return {}
        scored: List[Tuple[float, str]] = []
        for key, leaf in self.tree_leaf_by_key.items():
            leaf_tokens = {
                token for value in leaf.get("route_terms", []) or []
                for token in self._tokens(value)
            }
            relation_tokens = {
                token for value in leaf.get("relation_terms", []) or []
                for token in self._tokens(value)
            }
            signature_tokens = {
                token for value in leaf.get("signature_terms", []) or []
                for token in self._tokens(value)
            }
            answer_bonus = 0.22 if str(leaf.get("answer_type", "")) == answer_type else 0.0
            direct_overlap = query_tokens & (leaf_tokens | relation_tokens)
            soft_overlap = query_tokens & signature_tokens
            precision = self._safe_float(leaf.get("precision"), 0.0)
            gt5_precision = self._safe_float(leaf.get("gt5_precision"), 0.0)
            selected_precision = self._safe_float(leaf.get("selected_precision"), 0.0)
            support = self._safe_float(leaf.get("support"), 0.0)
            score = (
                answer_bonus
                + 0.42 * len(direct_overlap) / max(min(len(query_tokens), len(leaf_tokens | relation_tokens)), 1)
                + 0.18 * len(soft_overlap) / max(min(len(query_tokens), len(signature_tokens)), 1)
                + 0.16 * min(max(precision, selected_precision), 1.0)
                + 0.10 * min(max(gt5_precision, 0.0), 1.0)
                + 0.08 * min(support / 20.0, 1.0)
            )
            if score > 0:
                scored.append((score, key))
        scored.sort(reverse=True)
        max_leaves = int(self.tree.get("routing", {}).get("max_active_leaves", 12) or 12)
        min_leaf_score = float(self.tree.get("routing", {}).get("min_leaf_score", 0.16) or 0.16)
        return {
            key: score
            for score, key in scored[:max(max_leaves, 1)]
            if score >= min_leaf_score
        }

    def match(self, query: str, top_k: int = 6, min_score: float = 0.05) -> List[Dict[str, Any]]:
        query_tokens = set(self._tokens(query))
        if not query_tokens:
            return []
        answer_type = self.infer_answer_type(query)
        tree_route_scores = self._tree_leaf_route_scores(query_tokens, answer_type)
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for plan in self.plans:
            if not self._feedback_approved(plan):
                continue
            trigger_tokens = {
                token for value in plan.get("triggers", []) or []
                for token in self._tokens(value)
            }
            relation_tokens = {
                token for value in plan.get("relation_terms", []) or []
                for token in self._tokens(value)
            }
            signature_tokens = {
                token for value in plan.get("support_passage_signatures", []) or []
                for token in self._tokens(value)
            }
            bridge_tokens = {
                token for value in plan.get("bridge_terms", []) or []
                for token in self._tokens(value)
            }
            intent_tokens = set(self._tokens(plan.get("intent", "")))
            evidence_tokens = trigger_tokens | relation_tokens | signature_tokens | intent_tokens
            if not evidence_tokens:
                continue
            required_terms = {
                token for value in plan.get("required_query_terms", []) or []
                for token in self._tokens(value)
            }
            if required_terms:
                min_required = int(plan.get("min_required_query_overlap", 1) or 1)
                if len(query_tokens & required_terms) < max(min_required, 1):
                    continue
            if bool(plan.get("match_answer_type_required", False)) and answer_type != plan.get("answer_type"):
                continue
            direct_overlap = query_tokens & (trigger_tokens | relation_tokens | intent_tokens)
            soft_overlap = query_tokens & (signature_tokens | bridge_tokens)
            min_direct_overlap = int(plan.get("min_direct_overlap", 0) or 0)
            if min_direct_overlap > 0 and len(direct_overlap) < min_direct_overlap:
                continue
            if not direct_overlap and len(soft_overlap) < 2:
                continue
            tree_leaf_key = self._tree_leaf_key(plan)
            tree_route_score = 0.0
            if self.tree_enabled and tree_route_scores:
                tree_route_score = tree_route_scores.get(tree_leaf_key, 0.0)
                if tree_route_score <= 0.0:




                    if len(direct_overlap) < 2 or self._feedback_precision(plan) < 0.20:
                        continue
            support = float(plan.get("support", 0) or 0)
            precision = self._feedback_precision(plan)
            score = (
                len(direct_overlap) / max(min(len(query_tokens), len(evidence_tokens)), 1)
                + 0.35 * len(soft_overlap) / max(min(len(query_tokens), len(signature_tokens | bridge_tokens)), 1)
                + 0.12 * min(support / 40.0, 1.0)
                + 0.15 * min(max(precision, 0.0), 1.0)
            )
            if self._retrieval_feedback(plan):


                score += 0.10 * min(max(precision, 0.0), 1.0)
            if answer_type == plan.get("answer_type"):
                score += 0.16
            try:
                score += float(plan.get("priority", 0.0) or 0.0)
            except (TypeError, ValueError):
                pass
            feedback = self._retrieval_feedback(plan)
            try:
                score += float(feedback.get("priority", 0.0) or 0.0)
            except (TypeError, ValueError):
                pass
            if self.tree_enabled:
                if tree_route_score > 0.0:
                    score += 0.22 * min(tree_route_score, 1.2)
                elif tree_route_scores:
                    score -= 0.05
            if score < min_score:
                continue
            enriched = dict(plan)
            enriched["match_score"] = float(score)
            enriched["estimated_precision"] = float(precision)
            if self.tree_enabled:
                enriched["tree_route"] = {
                    "leaf_key": tree_leaf_key,
                    "route_score": float(tree_route_score),
                    "active_leaf_count": len(tree_route_scores),
                }
            enriched["query_overlap"] = {
                "direct": sorted(direct_overlap)[:16],
                "soft": sorted(soft_overlap)[:16],
                "answer_type": answer_type,
            }
            if self.leaf_policy is not None:
                leaf_score, leaf_features = self.leaf_policy.score_plan(query, enriched)
                enriched["leaf_policy"] = {
                    "score": round(float(leaf_score), 6),
                    "reject_below": self.leaf_policy_reject_below,
                    "boost": self.leaf_policy_boost,
                    "feature_snapshot": {
                        "direct_overlap_count": leaf_features.get("direct_overlap_count", 0.0),
                        "soft_overlap_count": leaf_features.get("soft_overlap_count", 0.0),
                        "route_score": round(leaf_features.get("route_score", 0.0), 4),
                        "feedback_gt5_precision": round(leaf_features.get("feedback_gt5_precision", 0.0), 4),
                        "feedback_candidate_precision": round(leaf_features.get("feedback_candidate_precision", 0.0), 4),
                    },
                }
                if leaf_score < self.leaf_policy_reject_below:
                    continue
                score += self.leaf_policy_boost * leaf_score
                enriched["match_score"] = float(score)
            scored.append((score, enriched))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [plan for _, plan in scored[:top_k]]


class EvidenceGroundedSupportRetriever:
    """Generate dense probes from evidence-grounded ontology plans."""

    def __init__(
        self,
        matcher: EvidenceGroundedOntologyMatcher,
        dense_top_k: int = 80,
        top_k_plans: int = 6,
        max_probes: int = 18,
        max_added_passages: int = 10,
        min_confidence: float = 0.58,
        preserve_top_k: int = 2,
        candidate_only: bool = True,
        append_score_margin: float = 1e-7,
        min_match_score: float = 0.05,
    ):
        self.matcher = matcher
        self.dense_top_k = max(int(dense_top_k), 1)
        self.top_k_plans = max(int(top_k_plans), 1)
        self.max_probes = max(int(max_probes), 0)
        self.max_added_passages = max(int(max_added_passages), 0)
        self.min_confidence = min(max(float(min_confidence), 0.0), 1.0)
        self.preserve_top_k = max(int(preserve_top_k), 0)
        self.candidate_only = bool(candidate_only)
        self.append_score_margin = max(float(append_score_margin), 0.0)
        self.min_match_score = float(min_match_score)

    @staticmethod
    def _norm(value: Any) -> str:
        return text_processing(str(value or ""))

    @staticmethod
    def _tokens(value: Any, max_tokens: int = 8) -> List[str]:
        output = []
        seen = set()
        for token in tokenize(text_processing(str(value or "").replace("_", " "))):
            if token in QueryPlanSupportEvidenceRetriever.GENERIC_TERMS or token in seen:
                continue
            output.append(token)
            seen.add(token)
            if len(output) >= max_tokens:
                break
        return output

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @classmethod
    def _contains_phrase(cls, text_norm: str, phrase: Any) -> bool:
        phrase_norm = cls._norm(phrase)
        if not phrase_norm:
            return False
        if f" {phrase_norm} " in f" {text_norm} ":
            return True
        phrase_tokens = cls._tokens(phrase_norm, max_tokens=8)
        if not phrase_tokens:
            return False
        text_tokens = set(tokenize(text_norm))
        matched = sum(1 for token in phrase_tokens if token in text_tokens)
        return matched / max(len(phrase_tokens), 1) >= 0.70

    @classmethod
    def _query_anchors(cls, query: str, max_items: int = 5) -> List[str]:
        spans = []


        spans.extend(re.findall(r"['\"]([^'\"]{3,80})['\"]", query))
        spans.extend(re.findall(r"\b[A-Z][\w'’.-]*(?:\s+[A-Z0-9][\w'’.-]*){0,5}", query))
        cleaned = []
        seen = set()
        for span in spans:
            text = text_processing(span)
            if not text or len(text) < 3 or text in seen:
                continue
            if text in {"what", "when", "where", "which", "who", "how"}:
                continue
            cleaned.append(text)
            seen.add(text)
            if len(cleaned) >= max_items:
                break
        return cleaned

    @staticmethod
    def _unique(values: Iterable[Any], max_items: int = 24) -> List[str]:
        output = []
        seen = set()
        for value in values or []:
            text = text_processing(str(value or ""))
            if not text or text in seen:
                continue
            output.append(text)
            seen.add(text)
            if len(output) >= max_items:
                break
        return output

    def _answer_hints(self, plan: Mapping[str, Any]) -> List[str]:
        answer_type = str(plan.get("answer_type") or "").lower()
        hints = list(plan.get("answer_terms", []) or [])[:4]
        hints.extend(QueryPlanSupportEvidenceRetriever.ANSWER_HINTS.get(answer_type, []))
        if answer_type:
            hints.append(answer_type)
        return self._unique(hints, max_items=8)

    @staticmethod
    def _probe_kind_allowed(plan: Mapping[str, Any], kind: str) -> bool:
        feedback = plan.get("retrieval_feedback", {}) if isinstance(plan, Mapping) else {}
        if not isinstance(feedback, dict):
            return True
        if "approved_probe_kinds" not in feedback:
            return True
        approved = {str(item) for item in feedback.get("approved_probe_kinds", []) or []}
        return kind in approved

    @staticmethod
    def _feedback_value(feedback: Mapping[str, Any], key: str, default: float = 0.0) -> float:
        try:
            return float(feedback.get(key, default))
        except (TypeError, ValueError):
            return default

    def _plan_action_policy(self, plan: Mapping[str, Any]) -> Dict[str, Any]:
        """Translate retrieval feedback into runtime retrieval permissions.

        Strong plans get a lower local threshold and can use query-signature
        probes as high-recall evidence locators, while weak-but-approved plans
        remain conservative. This keeps the policy plan-specific rather than
        dataset-specific.
        """
        feedback = plan.get("retrieval_feedback", {}) if isinstance(plan, Mapping) else {}
        if not isinstance(feedback, dict) or not feedback:
            return {
                "tier": "uncalibrated",
                "precision": self._safe_float(plan.get("estimated_precision", plan.get("confidence", 0.0)), 0.0),
                "gt5_precision": 0.0,
                "selected_precision": 0.0,
                "min_confidence": self.min_confidence,
                "allow_signature_rescue": False,
                "confidence_bonus": 0.0,
            }
        precision = max(
            self._feedback_value(feedback, "calibrated_precision"),
            self._feedback_value(feedback, "candidate_precision"),
        )
        gt5_precision = self._feedback_value(feedback, "candidate_gt5_precision")
        selected_precision = self._feedback_value(feedback, "selected_precision")
        candidate_count = int(self._feedback_value(feedback, "candidate_count"))
        selected_count = int(self._feedback_value(feedback, "selected_count"))
        if selected_precision >= 0.30 or precision >= 0.35 or gt5_precision >= 0.15:
            tier = "strong"
            local_min = self.min_confidence - 0.18
            bonus = 0.08
            signature_rescue = True
        elif selected_precision >= 0.15 or precision >= 0.15 or gt5_precision >= 0.06:
            tier = "moderate"
            local_min = self.min_confidence - 0.11
            bonus = 0.05
            signature_rescue = True
        else:
            tier = "weak"
            local_min = self.min_confidence - 0.04
            bonus = 0.02
            signature_rescue = False

        if candidate_count < 5 and selected_count < 2:
            local_min = max(local_min, self.min_confidence - 0.04)
            signature_rescue = False
            tier = f"{tier}_small_sample"
        return {
            "tier": tier,
            "precision": min(max(precision, 0.0), 1.0),
            "gt5_precision": min(max(gt5_precision, 0.0), 1.0),
            "selected_precision": min(max(selected_precision, 0.0), 1.0),
            "min_confidence": min(max(local_min, 0.38), 0.98),
            "allow_signature_rescue": signature_rescue,
            "confidence_bonus": bonus,
        }

    def _build_probes(self, query: str, plans: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
        query_anchors = self._query_anchors(query)
        probes = []
        seen = set()

        def add_probe(text: str, plan: Mapping[str, Any], kind: str, anchor: str = "", relation: str = "", signature: str = "") -> None:
            if not self._probe_kind_allowed(plan, kind):
                return
            key = self._norm(text)
            if not key or key in seen:
                return
            seen.add(key)
            probes.append({
                "query": text,
                "kind": kind,
                "anchor": anchor,
                "relation": relation,
                "signature": signature,
                "plan_id": plan.get("id", "evidence_grounded"),
                "plan_match_score": plan.get("match_score", 0.0),
                "estimated_precision": plan.get("estimated_precision", plan.get("confidence", 0.0)),
                "retrieval_feedback": plan.get("retrieval_feedback", {}) if isinstance(plan.get("retrieval_feedback", {}), dict) else {},
                "tree_route": plan.get("tree_route", {}) if isinstance(plan.get("tree_route", {}), dict) else {},
                "leaf_policy": plan.get("leaf_policy", {}) if isinstance(plan.get("leaf_policy", {}), dict) else {},
                "action_policy": self._plan_action_policy(plan),
                "answer_hints": self._answer_hints(plan),
            })

        for plan in plans:
            relation_terms = self._unique(plan.get("relation_terms", []) or [], max_items=8)
            signatures = self._unique(plan.get("support_passage_signatures", []) or [], max_items=8)
            bridge_terms = self._unique(plan.get("bridge_terms", []) or [], max_items=6)
            templates = self._unique(plan.get("successful_probe_templates", []) or [], max_items=6)
            answer_hints = self._answer_hints(plan)
            hint_text = " ".join(answer_hints[:2])




            for signature in signatures[:4]:
                add_probe(" ".join(x for x in [query, signature, hint_text] if x), plan, "query_signature", "", "", signature)
            if not signatures:
                for relation in relation_terms[:3]:
                    add_probe(" ".join(x for x in [query, relation, hint_text] if x), plan, "query_signature", "", "", relation)
            for anchor in query_anchors[:4]:
                for relation in relation_terms[:4]:
                    add_probe(" ".join(x for x in [query, anchor, relation] if x), plan, "query_anchor_relation", anchor, relation)
                    add_probe(" ".join(x for x in [anchor, relation, hint_text] if x), plan, "anchor_relation", anchor, relation)
                for signature in signatures[:4]:
                    add_probe(" ".join(x for x in [query, anchor, signature] if x), plan, "query_anchor_signature", anchor, "", signature)
                for bridge in bridge_terms[:3]:
                    add_probe(" ".join(x for x in [anchor, bridge, hint_text] if x), plan, "anchor_bridge", anchor, bridge)
            for template in templates[:4]:
                for anchor in query_anchors[:2] or [""]:
                    text = template.replace("{query}", query).replace("{anchor}", anchor).replace("{answer_hint}", hint_text)
                    add_probe(text, plan, "learned_template", anchor)
        return probes[: self.max_probes]

    def _answer_hit(self, text_norm: str, hints: Sequence[str]) -> bool:
        tokens = set(tokenize(text_norm))
        hint_tokens = {token for hint in hints for token in self._tokens(hint, max_tokens=3)}
        if hint_tokens and tokens.intersection(hint_tokens):
            return True
        if any(hint in {"date", "year", "month", "day"} for hint in hints):
            if any(token.isdigit() and len(token) == 4 for token in tokens):
                return True
        return False

    def _score_candidate(self, passage_text: str, dense_score: float, max_dense_score: float, probe: Mapping[str, Any]) -> Tuple[float, Dict[str, Any]]:
        text_norm = self._norm(passage_text)
        anchor_hit = self._contains_phrase(text_norm, probe.get("anchor", ""))
        relation_tokens = self._tokens(probe.get("relation", ""), max_tokens=5)
        signature_tokens = self._tokens(probe.get("signature", ""), max_tokens=8)
        text_tokens = set(tokenize(text_norm))
        relation_hits = [token for token in relation_tokens if token in text_tokens]
        signature_hits = [token for token in signature_tokens if token in text_tokens]
        relation_score = min((len(relation_hits) + 0.6 * len(signature_hits)) / max(min(len(relation_tokens) + len(signature_tokens), 5), 1), 1.0)
        answer_hit = self._answer_hit(text_norm, probe.get("answer_hints", []) or [])
        dense_norm = 0.0 if max_dense_score <= 0 else min(max(float(dense_score) / max_dense_score, 0.0), 1.0)
        plan_score = min(max(self._safe_float(probe.get("plan_match_score"), 0.0), 0.0), 1.5) / 1.5
        precision = min(max(self._safe_float(probe.get("estimated_precision"), 0.0), 0.0), 1.0)
        action_policy = probe.get("action_policy", {}) if isinstance(probe.get("action_policy", {}), Mapping) else {}
        confidence = (
            0.34 * dense_norm
            + 0.24 * (1.0 if anchor_hit else 0.0)
            + 0.20 * relation_score
            + 0.08 * (1.0 if answer_hit else 0.0)
            + 0.08 * plan_score
            + 0.06 * precision
            + self._safe_float(action_policy.get("confidence_bonus"), 0.0) * precision
        )
        return min(max(confidence, 0.0), 1.0), {
            "anchor_hit": anchor_hit,
            "relation_hits": relation_hits,
            "signature_hits": signature_hits,
            "answer_hit": answer_hit,
            "dense_norm": dense_norm,
            "plan_match_score": plan_score,
            "action_tier": action_policy.get("tier", ""),
            "local_min_confidence": self._safe_float(action_policy.get("min_confidence"), self.min_confidence),
            "allow_signature_rescue": bool(action_policy.get("allow_signature_rescue", False)),
            "feedback_precision": self._safe_float(action_policy.get("precision"), precision),
            "feedback_gt5_precision": self._safe_float(action_policy.get("gt5_precision"), 0.0),
            "feedback_selected_precision": self._safe_float(action_policy.get("selected_precision"), 0.0),
            "confidence": min(max(confidence, 0.0), 1.0),
        }

    def _allow_feedback_rescue(self, probe: Mapping[str, Any], detail: Mapping[str, Any], dense_rank: int, confidence: float) -> bool:
        if not bool(detail.get("allow_signature_rescue", False)):
            return False
        kind = str(probe.get("kind", ""))
        if kind not in {"query_signature", "query_anchor_signature", "query_anchor_relation", "anchor_relation"}:
            return False
        precision = self._safe_float(detail.get("feedback_precision"), 0.0)
        gt5_precision = self._safe_float(detail.get("feedback_gt5_precision"), 0.0)
        selected_precision = self._safe_float(detail.get("feedback_selected_precision"), 0.0)
        if max(precision, gt5_precision, selected_precision) < 0.12:
            return False



        return dense_rank <= 8 and confidence >= max(self._safe_float(detail.get("local_min_confidence"), self.min_confidence) - 0.04, 0.38)

    def retrieve(
        self,
        query: str,
        sorted_doc_ids: Sequence[int],
        sorted_doc_scores: Sequence[float],
        dense_retrieve: Callable[[str], Tuple[np.ndarray, np.ndarray]],
        get_passage_text: Callable[[int], str],
    ) -> EvidenceGroundedResult:
        base_doc_ids = [int(doc_id) for doc_id in list(sorted_doc_ids)]
        base_scores = np.array(sorted_doc_scores, dtype=np.float32)
        log: Dict[str, Any] = {
            "enabled": True,
            "dense_top_k": self.dense_top_k,
            "top_k_plans": self.top_k_plans,
            "max_probes": self.max_probes,
            "max_added_passages": self.max_added_passages,
            "min_confidence": self.min_confidence,
            "matched_plans": [],
            "probes": [],
            "added_candidates": [],
            "rejection_reasons": {},
        }
        if not self.matcher or not self.matcher.plans:
            log["skip_reason"] = "empty_evidence_grounded_ontology"
            return EvidenceGroundedResult(np.array(base_doc_ids, dtype=np.intp), base_scores, log)
        plans = self.matcher.match(query, top_k=self.top_k_plans, min_score=self.min_match_score)
        log["matched_plan_count"] = len(plans)
        log["matched_plans"] = [
            {
                "id": plan.get("id", ""),
                "intent": plan.get("intent", ""),
                "answer_type": plan.get("answer_type", ""),
                "match_score": round(float(plan.get("match_score", 0.0)), 6),
                "query_overlap": plan.get("query_overlap", {}),
                "tree_route": plan.get("tree_route", {}),
                "leaf_policy": plan.get("leaf_policy", {}),
            }
            for plan in plans
        ]
        if not plans:
            log["skip_reason"] = "no_evidence_grounded_plan_match"
            return EvidenceGroundedResult(np.array(base_doc_ids, dtype=np.intp), base_scores, log)
        probes = self._build_probes(query, plans)
        log["probes"] = probes
        log["raw_probe_count"] = len(probes)
        if not probes:
            log["skip_reason"] = "no_evidence_grounded_probes"
            return EvidenceGroundedResult(np.array(base_doc_ids, dtype=np.intp), base_scores, log)

        original_rank_by_doc = {int(doc_id): rank + 1 for rank, doc_id in enumerate(base_doc_ids)}
        preserved = set(base_doc_ids[: self.preserve_top_k])
        selected_by_passage: Dict[int, Dict[str, Any]] = {}
        rejection_reasons: Dict[str, int] = {}
        feedback_rescue_count = 0
        for probe in probes:
            dense_doc_ids, dense_scores = dense_retrieve(str(probe.get("query", "")))
            dense_doc_ids = list(dense_doc_ids[: self.dense_top_k])
            dense_scores = list(dense_scores[: self.dense_top_k])
            max_dense_score = max((self._safe_float(score) for score in dense_scores), default=0.0)
            for dense_rank, passage_idx in enumerate(dense_doc_ids, start=1):
                passage_idx = int(passage_idx)
                if passage_idx in preserved:
                    rejection_reasons["preserved_top"] = rejection_reasons.get("preserved_top", 0) + 1
                    continue
                passage_text = get_passage_text(passage_idx)
                confidence, detail = self._score_candidate(
                    passage_text,
                    self._safe_float(dense_scores[dense_rank - 1] if dense_rank - 1 < len(dense_scores) else 0.0),
                    max_dense_score,
                    probe,
                )
                if probe.get("anchor") and not detail["anchor_hit"]:
                    rejection_reasons["anchor_miss"] = rejection_reasons.get("anchor_miss", 0) + 1
                    continue
                rescued_by_feedback = False
                if not (detail["relation_hits"] or detail["signature_hits"] or detail["answer_hit"]):
                    rescued_by_feedback = self._allow_feedback_rescue(probe, detail, dense_rank, confidence)
                if not (detail["relation_hits"] or detail["signature_hits"] or detail["answer_hit"] or rescued_by_feedback):
                    rejection_reasons["evidence_signature_miss"] = rejection_reasons.get("evidence_signature_miss", 0) + 1
                    continue
                local_min_confidence = self._safe_float(detail.get("local_min_confidence"), self.min_confidence)
                if confidence < local_min_confidence:
                    rejection_reasons["low_confidence"] = rejection_reasons.get("low_confidence", 0) + 1
                    continue
                if rescued_by_feedback:
                    feedback_rescue_count += 1
                original_rank = int(original_rank_by_doc.get(passage_idx, 80))
                relation_values = []
                if probe.get("relation"):
                    relation_values.append(probe.get("relation", ""))
                if probe.get("signature"):
                    relation_values.append(probe.get("signature", ""))
                candidate = {
                    "passage_idx": passage_idx,
                    "confidence": float(confidence),
                    "dense_rank": int(dense_rank),
                    "original_rank": original_rank,
                    "was_in_baseline": passage_idx in original_rank_by_doc,
                    "score": float(confidence + 0.01 / max(dense_rank, 1)),
                    "probe": dict(probe),
                    "matched_anchor": bool(detail["anchor_hit"]),
                    "relation_hits": detail["relation_hits"],
                    "signature_hits": detail["signature_hits"],
                    "answer_hit": bool(detail["answer_hit"]),
                    "feedback_signature_rescue": bool(rescued_by_feedback),
                    "action_tier": detail.get("action_tier", ""),
                    "contexts": [{
                        "evidence_grounded": True,
                        "support_key": f"evidence_grounded::{probe.get('kind', '')}::{passage_idx}",
                        "plan_id": probe.get("plan_id", "evidence_grounded"),
                        "anchor_entity": probe.get("anchor", ""),
                        "end_entity": "",
                        "relations": relation_values,
                        "fact_indices": [],
                        "fact_position": 0,
                        "signature_match_mode": "evidence_grounded_dense",
                        "plan_effective_precision": detail.get("feedback_precision", probe.get("estimated_precision", 0.0)),
                        "plan_feedback_gt5_precision": detail.get("feedback_gt5_precision", 0.0),
                        "plan_feedback_selected_precision": detail.get("feedback_selected_precision", 0.0),
                        "plan_action_tier": detail.get("action_tier", ""),
                        "feedback_signature_rescue": bool(rescued_by_feedback),
                        "plan_lift_weight_scale": 0.95,
                        "score": float(confidence),
                    }],
                }
                existing = selected_by_passage.get(passage_idx)
                if existing is None or candidate["score"] > existing["score"]:
                    selected_by_passage[passage_idx] = candidate

        candidates = sorted(
            selected_by_passage.values(),
            key=lambda item: (
                float(item.get("score", 0.0)),
                -int(item.get("dense_rank", 10**9)),
                -int(item.get("original_rank", 10**9)),
            ),
            reverse=True,
        )
        selected = candidates[: self.max_added_passages]
        log["candidate_count"] = len(candidates)
        log["selected_count"] = len(selected)
        log["added_candidates"] = selected
        log["feedback_rescued_candidate_count"] = int(feedback_rescue_count)
        log["rejection_reasons"] = dict(sorted(rejection_reasons.items(), key=lambda kv: kv[1], reverse=True)[:8])
        if not selected:
            log["skip_reason"] = "no_evidence_grounded_candidates"
            return EvidenceGroundedResult(np.array(base_doc_ids, dtype=np.intp), base_scores, log)
        if self.candidate_only:
            appended = [int(item["passage_idx"]) for item in selected if int(item["passage_idx"]) not in set(base_doc_ids)]
            score_by_doc = {
                int(doc_id): float(base_scores[idx]) if idx < len(base_scores) else 0.0
                for idx, doc_id in enumerate(base_doc_ids)
            }
            min_tail_score = float(base_scores[-1]) if len(base_scores) else 0.0
            for rank, passage_idx in enumerate(appended, start=1):
                score_by_doc[passage_idx] = min_tail_score - self.append_score_margin * rank
            new_doc_ids = base_doc_ids + [idx for idx in appended if idx not in set(base_doc_ids)]
            new_scores = np.array([score_by_doc[int(idx)] for idx in new_doc_ids], dtype=np.float32)
            log["appended_passage_count"] = len(appended)
            log["skip_reason"] = "candidate_only"
            return EvidenceGroundedResult(np.array(new_doc_ids, dtype=np.intp), new_scores, log)
        return EvidenceGroundedResult(np.array(base_doc_ids, dtype=np.intp), base_scores, log)


def load_evidence_grounded_matcher(
    path: str,
    leaf_policy_path: str = "",
    leaf_policy_reject_below: float = 0.0,
    leaf_policy_boost: float = 0.0,
) -> EvidenceGroundedOntologyMatcher | None:
    if not path or not os.path.exists(path):
        return None
    return EvidenceGroundedOntologyMatcher.load(
        path,
        leaf_policy_path=leaf_policy_path,
        leaf_policy_reject_below=leaf_policy_reject_below,
        leaf_policy_boost=leaf_policy_boost,
    )
