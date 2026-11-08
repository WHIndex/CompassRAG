from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np

from .query_plan_adapter import QueryPlanOntologyAdapter
from .query_plan_promotion_policy import QueryPlanPassagePromotionPolicy
from .query_plan_promotion_verifier import QueryPlanPassagePromotionVerifier


@dataclass
class PassageRerankDecision:
    passage_idx: int
    confidence: float
    action: str
    reasons: List[str]
    verifier: Dict[str, Any] = field(default_factory=dict)
    policy: Dict[str, Any] = field(default_factory=dict)


class QueryPlanOntologyPassageReranker:
    """Turn executable query-plan paths into a calibrated passage ranking signal.

    The older passage lift used a small global interpolation weight. That is
    safe, but it often fails to move high-quality ontology-supported gold
    passages into top-2/top-5. This reranker keeps the soft score blend for
    ordinary evidence, while allowing high-confidence ontology-supported
    passages to be promoted near the top.
    """

    PROMOTABLE_QUERY_TYPES = {
        "comparative",
        "nested_bridge_factoid",
        "extractive_factoid",
        "direct_slot_factoid",
        "other",
    }

    def __init__(
        self,
        high_confidence_threshold: float = 0.70,
        promote_top_k: int = 1,
        promote_to_rank: int = 2,
        preserve_top_k: int = 1,
        enable_promotion_verifier: bool = False,
        promotion_verifier_min_confidence: float = 0.85,
        promotion_verifier_max_original_rank: int = 5,
        promotion_policy_path: str = "",
        promotion_policy_mode: str = "off",
        promotion_policy_enable_coverage_gate: bool = False,
        promotion_policy_coverage_top_k: int = 2,
    ):
        self.high_confidence_threshold = min(max(float(high_confidence_threshold), 0.0), 1.0)
        self.promote_top_k = max(int(promote_top_k), 0)
        self.promote_to_rank = max(int(promote_to_rank), 1)
        self.preserve_top_k = max(int(preserve_top_k), 0)
        self.adapter = QueryPlanOntologyAdapter()
        self.enable_promotion_verifier = bool(enable_promotion_verifier)
        self.promotion_verifier = QueryPlanPassagePromotionVerifier(
            min_confidence=promotion_verifier_min_confidence,
            max_original_rank=promotion_verifier_max_original_rank,
        )
        self.promotion_policy_path = str(promotion_policy_path or "")
        self.promotion_policy_mode = str(promotion_policy_mode or "off").strip().lower()
        self.promotion_policy_enable_coverage_gate = bool(promotion_policy_enable_coverage_gate)
        self.promotion_policy_coverage_top_k = max(int(promotion_policy_coverage_top_k), 1)
        self.promotion_policy: Optional[QueryPlanPassagePromotionPolicy] = None
        if self.promotion_policy_mode != "off" and QueryPlanPassagePromotionPolicy.exists(self.promotion_policy_path):
            self.promotion_policy = QueryPlanPassagePromotionPolicy.load(self.promotion_policy_path)

    @staticmethod
    def _modes(contexts: Iterable[Mapping[str, Any]]) -> set:
        modes = {
            str(context.get("signature_match_mode", "") or "").strip().lower()
            for context in contexts
        }
        modes.discard("")
        return modes

    @staticmethod
    def _path_lengths(contexts: Iterable[Mapping[str, Any]]) -> List[int]:
        lengths = []
        for context in contexts:
            fact_len = len(context.get("fact_indices", []) or [])
            rel_len = len(context.get("relations", []) or [])
            lengths.append(max(fact_len, rel_len, 1))
        return lengths

    @staticmethod
    def _norm_value(value: Any) -> str:
        return " ".join(str(value or "").strip().lower().split())

    def _context_signature(self, contexts: Iterable[Mapping[str, Any]]) -> Dict[str, set]:
        anchors = set()
        terminals = set()
        roles = set()
        plans = set()
        relations = set()
        facts = set()
        for context in contexts or []:
            plan_id = self._norm_value(context.get("plan_id"))
            anchor = self._norm_value(context.get("anchor_entity"))
            terminal = self._norm_value(context.get("end_entity"))
            if plan_id:
                plans.add(plan_id)
            if anchor:
                anchors.add(anchor)
            if terminal:
                terminals.add(terminal)
            fact_position = context.get("fact_position")
            if plan_id and fact_position is not None:
                roles.add((plan_id, str(fact_position)))
            for fact_idx in context.get("fact_indices", []) or []:
                facts.add(str(fact_idx))
            rel_tuple = tuple(
                self._norm_value(relation)
                for relation in (context.get("relations", []) or [])
                if self._norm_value(relation)
            )
            if rel_tuple:
                relations.add(rel_tuple)
        return {
            "anchors": anchors,
            "terminals": terminals,
            "roles": roles,
            "plans": plans,
            "relations": relations,
            "facts": facts,
        }

    @staticmethod
    def _merge_signatures(signatures: Iterable[Mapping[str, set]]) -> Dict[str, set]:
        merged = {
            "anchors": set(),
            "terminals": set(),
            "roles": set(),
            "plans": set(),
            "relations": set(),
            "facts": set(),
        }
        for signature in signatures:
            for key in merged:
                merged[key].update(signature.get(key, set()) or set())
        return merged

    def _promotion_coverage_gate(
        self,
        passage_idx: int,
        candidate_contexts: List[Dict[str, Any]],
        base_doc_ids: List[int],
        passage_contexts: Mapping[int, List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        if not self.promotion_policy_enable_coverage_gate:
            return {"approved": True, "reason": "coverage_gate_disabled"}
        if not candidate_contexts:
            return {"approved": False, "reason": "candidate_has_no_ontology_context"}

        insert_rank = max(self.promote_to_rank, self.preserve_top_k + 1, 1)
        target_doc_id = None
        if 0 <= insert_rank - 1 < len(base_doc_ids):
            target_doc_id = int(base_doc_ids[insert_rank - 1])

        top_k = max(self.promotion_policy_coverage_top_k, insert_rank)
        current_top_ids = [
            int(doc_id)
            for doc_id in base_doc_ids[:top_k]
            if int(doc_id) != int(passage_idx)
        ]
        current_context_groups = [
            passage_contexts.get(doc_id, []) or []
            for doc_id in current_top_ids
        ]
        top_context_doc_count = sum(1 for contexts in current_context_groups if contexts)
        candidate_sig = self._context_signature(candidate_contexts)
        current_sig = self._merge_signatures(
            self._context_signature(contexts)
            for contexts in current_context_groups
            if contexts
        )
        if target_doc_id is not None:
            target_contexts = passage_contexts.get(target_doc_id, []) or []
        else:
            target_contexts = []

        new_roles = candidate_sig["roles"] - current_sig["roles"]
        new_anchors = candidate_sig["anchors"] - current_sig["anchors"]
        new_plans = candidate_sig["plans"] - current_sig["plans"]
        new_relations = candidate_sig["relations"] - current_sig["relations"]
        adds_new_evidence_role = bool(new_roles or new_anchors or new_plans or new_relations)

        result = {
            "approved": True,
            "reason": "adds_new_evidence_role" if adds_new_evidence_role else "no_competing_top_evidence",
            "target_doc_id": target_doc_id,
            "top_k": top_k,
            "top_context_doc_count": top_context_doc_count,
            "candidate_new_roles": len(new_roles),
            "candidate_new_anchors": len(new_anchors),
            "candidate_new_plans": len(new_plans),
            "candidate_new_relations": len(new_relations),
            "target_has_ontology_context": bool(target_contexts),
        }
        if bool(target_contexts) and not adds_new_evidence_role:
            result["approved"] = False
            result["reason"] = "would_displace_ontology_supported_target_without_new_role"
        elif top_context_doc_count >= min(top_k, 2) and not adds_new_evidence_role:
            result["approved"] = False
            result["reason"] = "top_window_already_has_ontology_evidence_without_new_role"
        return result

    def score_passage(
        self,
        passage_idx: int,
        raw_score: float,
        max_raw_score: float,
        passage_scale: float,
        contexts: List[Dict[str, Any]],
        original_rank: int,
        max_original_rank: int,
        query_type: str,
    ) -> PassageRerankDecision:
        reasons: List[str] = []
        max_raw_score = max(float(max_raw_score), 1e-8)
        normalized_lift = min(max(float(raw_score) / max_raw_score, 0.0), 1.0)
        scale = min(max(float(passage_scale or 1.0), 0.0), 1.0)
        modes = self._modes(contexts)
        only_soft = bool(modes) and modes <= {"soft"}
        has_strict = "strict" in modes or not modes
        path_lengths = self._path_lengths(contexts)
        min_path_len = min(path_lengths) if path_lengths else 1
        context_strength = min(len(contexts) / 2.0, 1.0)
        if max_original_rank > 0:
            rank_window_score = min(max((max_original_rank - original_rank + 1) / max_original_rank, 0.0), 1.0)
        else:
            rank_window_score = 0.5

        confidence = (
            0.40 * normalized_lift
            + 0.25 * scale
            + 0.15 * (1.0 if has_strict else 0.0)
            + 0.10 * context_strength
            + 0.10 * rank_window_score
        )
        if only_soft:
            confidence *= 0.55
            reasons.append("soft_only_penalty")
        if min_path_len >= 2:
            confidence += 0.04
            reasons.append("multi_hop_path")
        if query_type in {"category_or_alias", "discourse_or_long_answer", "non_extractive_answer"}:
            confidence *= 0.55
            reasons.append(f"query_type_{query_type}_penalty")
        if has_strict:
            reasons.append("strict_relation")
        if scale >= 0.70:
            reasons.append("high_plan_scale")
        if normalized_lift >= 0.75:
            reasons.append("high_lift_score")
        if context_strength >= 1.0:
            reasons.append("multiple_contexts")
        if original_rank <= 2:
            reasons.append("already_top2")

        confidence = min(max(confidence, 0.0), 1.0)
        promotable = (
            query_type in self.PROMOTABLE_QUERY_TYPES
            and confidence >= self.high_confidence_threshold
            and original_rank > self.promote_to_rank
        )
        action = "promote" if promotable else "boost"
        return PassageRerankDecision(
            passage_idx=int(passage_idx),
            confidence=round(float(confidence), 6),
            action=action,
            reasons=reasons[:8],
        )

    def rerank(
        self,
        query: str,
        query_condition: Dict[str, Any],
        sorted_doc_ids: np.ndarray,
        combined_scores: np.ndarray,
        ranked_passages: List[Tuple[int, float]],
        passage_contexts: Mapping[int, List[Dict[str, Any]]],
        passage_scales: Mapping[int, float],
        original_rank_by_doc: Mapping[int, int],
        max_original_rank: int,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        base_doc_ids = [int(doc_id) for doc_id in np.argsort(combined_scores)[::-1]]
        query_type = self.adapter.classify_query(query, query_condition)
        max_raw_score = max((float(score) for _, score in ranked_passages), default=1.0)

        decisions: Dict[int, PassageRerankDecision] = {}
        promote_candidates: List[PassageRerankDecision] = []
        for passage_idx, raw_score in ranked_passages:
            passage_idx = int(passage_idx)
            original_rank = int(original_rank_by_doc.get(passage_idx, len(base_doc_ids) + 1))
            decision = self.score_passage(
                passage_idx=passage_idx,
                raw_score=float(raw_score),
                max_raw_score=max_raw_score,
                passage_scale=float(passage_scales.get(passage_idx, 1.0) or 1.0),
                contexts=passage_contexts.get(passage_idx, []) or [],
                original_rank=original_rank,
                max_original_rank=max_original_rank,
                query_type=query_type,
            )
            if self.promotion_policy is not None and original_rank > self.promote_to_rank:
                policy_features = self.promotion_policy.extract_features(
                    query=query,
                    query_condition=query_condition,
                    query_type=query_type,
                    original_rank=original_rank,
                    confidence=decision.confidence,
                    raw_score=float(raw_score),
                    max_raw_score=max_raw_score,
                    passage_scale=float(passage_scales.get(passage_idx, 1.0) or 1.0),
                    contexts=passage_contexts.get(passage_idx, []) or [],
                )
                policy_result = self.promotion_policy.evaluate(policy_features)
                decision.policy = {
                    **policy_result,
                    "mode": self.promotion_policy_mode,
                    "features": policy_features,
                }
                if policy_result.get("approved"):
                    coverage_gate = self._promotion_coverage_gate(
                        passage_idx=passage_idx,
                        candidate_contexts=passage_contexts.get(passage_idx, []) or [],
                        base_doc_ids=base_doc_ids,
                        passage_contexts=passage_contexts,
                    )
                    decision.policy["coverage_gate"] = coverage_gate
                    if coverage_gate.get("approved"):
                        decision.action = "promote"
                        decision.reasons = (
                            decision.reasons
                            + [
                                "promotion_policy_approved",
                                f"policy_{policy_result.get('rule_id')}",
                                f"policy_precision_{policy_result.get('precision')}",
                                f"coverage_{coverage_gate.get('reason')}",
                            ]
                        )[:12]
                    else:
                        decision.action = "boost"
                        decision.reasons = (
                            decision.reasons
                            + [
                                "promotion_policy_coverage_rejected",
                                str(coverage_gate.get("reason", "coverage_gate_rejected")),
                            ]
                        )[:12]
                elif decision.action == "promote" and self.promotion_policy_mode in {"gate", "override"}:
                    decision.action = "boost"
                    decision.reasons = (
                        decision.reasons
                        + [
                            "promotion_policy_rejected",
                            str(policy_result.get("reason", "no_policy_rule_matched")),
                        ]
                    )[:12]
            if self.enable_promotion_verifier and decision.action == "promote":
                verifier_result = self.promotion_verifier.verify(
                    query=query,
                    query_condition=query_condition,
                    query_type=query_type,
                    original_rank=original_rank,
                    confidence=decision.confidence,
                    contexts=passage_contexts.get(passage_idx, []) or [],
                )
                decision.verifier = verifier_result
                if not verifier_result.get("verified", False):
                    decision.action = "boost"
                    decision.reasons = (
                        decision.reasons
                        + ["promotion_verifier_rejected"]
                        + [
                            f"verifier_{penalty}"
                            for penalty in verifier_result.get("penalties", [])[:4]
                        ]
                    )[:12]
                else:
                    decision.reasons = (
                        decision.reasons
                        + ["promotion_verifier_verified"]
                        + [
                            f"verifier_{reason}"
                            for reason in verifier_result.get("reasons", [])[:3]
                        ]
                    )[:12]
            decisions[passage_idx] = decision
            if decision.action == "promote":
                promote_candidates.append(decision)

        promote_candidates.sort(
            key=lambda item: (
                float(item.policy.get("precision", 0.0) or 0.0),
                float(item.confidence),
            ),
            reverse=True,
        )
        selected_promotions = promote_candidates[:self.promote_top_k]
        promoted_ids = [decision.passage_idx for decision in selected_promotions]
        promoted_set = set(promoted_ids)

        if not promoted_ids:
            reranked_doc_ids = np.array(base_doc_ids, dtype=np.intp)
            return reranked_doc_ids, combined_scores[reranked_doc_ids], {
                "enabled": True,
                "mode": "adaptive_promote",
                "query_type": query_type,
                "promoted_passages": [],
                "promoted_count": 0,
                "high_confidence_threshold": self.high_confidence_threshold,
                "promote_top_k": self.promote_top_k,
                "promote_to_rank": self.promote_to_rank,
                "preserve_top_k": self.preserve_top_k,
                "promotion_verifier_enabled": self.enable_promotion_verifier,
                "promotion_policy_enabled": self.promotion_policy is not None,
                "promotion_policy_mode": self.promotion_policy_mode,
                "promotion_policy_path": self.promotion_policy_path,
                "promotion_policy_coverage_gate_enabled": self.promotion_policy_enable_coverage_gate,
                "promotion_policy_coverage_top_k": self.promotion_policy_coverage_top_k,
                "decisions": {
                    str(idx): decisions[idx].__dict__
                    for idx, _ in ranked_passages[:20]
                    if idx in decisions
                },
            }

        kept_doc_ids = [doc_id for doc_id in base_doc_ids if doc_id not in promoted_set]
        preserve_count = min(self.preserve_top_k, len(kept_doc_ids))
        insert_at = min(max(self.promote_to_rank - 1, preserve_count), len(kept_doc_ids))
        final_doc_ids = (
            kept_doc_ids[:insert_at]
            + promoted_ids
            + kept_doc_ids[insert_at:]
        )
        reranked_doc_ids = np.array(final_doc_ids, dtype=np.intp)
        reranked_scores = combined_scores[reranked_doc_ids]

        return reranked_doc_ids, reranked_scores, {
            "enabled": True,
            "mode": "adaptive_promote",
            "query_type": query_type,
            "promoted_passages": [decision.__dict__ for decision in selected_promotions],
            "promoted_count": len(selected_promotions),
            "high_confidence_threshold": self.high_confidence_threshold,
            "promote_top_k": self.promote_top_k,
            "promote_to_rank": self.promote_to_rank,
            "preserve_top_k": self.preserve_top_k,
            "promotion_verifier_enabled": self.enable_promotion_verifier,
            "promotion_policy_enabled": self.promotion_policy is not None,
            "promotion_policy_mode": self.promotion_policy_mode,
            "promotion_policy_path": self.promotion_policy_path,
            "promotion_policy_coverage_gate_enabled": self.promotion_policy_enable_coverage_gate,
            "promotion_policy_coverage_top_k": self.promotion_policy_coverage_top_k,
            "decisions": {
                str(idx): decisions[idx].__dict__
                for idx, _ in ranked_passages[:20]
                if idx in decisions
            },
        }
