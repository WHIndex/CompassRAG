from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


@dataclass
class ContextPassageDecision:
    passage_idx: int
    original_rank: int
    final_rank: int
    score: float
    selected: bool
    reason: str
    added_roles: int
    added_anchors: int
    added_plans: int


class QueryPlanEvidenceContextBuilder:
    """Build the final QA context from retrieval rank plus query-plan evidence.

    Passage promotion changes global rank, but the QA reader only sees the
    first few passages. This builder uses query-plan lifted passages as a
    deterministic context assembly signal: preserve the strongest baseline
    passage, add a small number of complementary ontology-supported evidence
    passages, then fill the rest from the original retrieval order.
    """

    def __init__(
        self,
        qa_top_k: int = 5,
        preserve_top_k: int = 1,
        max_evidence_passages: int = 2,
        max_original_rank: int = 20,
        min_candidate_confidence: float = 0.0,
        require_new_evidence_role: bool = True,
    ):
        self.qa_top_k = max(int(qa_top_k), 1)
        self.preserve_top_k = max(int(preserve_top_k), 0)
        self.max_evidence_passages = max(int(max_evidence_passages), 0)
        self.max_original_rank = int(max_original_rank)
        self.min_candidate_confidence = min(max(float(min_candidate_confidence), 0.0), 1.0)
        self.require_new_evidence_role = bool(require_new_evidence_role)

    @staticmethod
    def _norm(value: Any) -> str:
        return " ".join(str(value or "").strip().lower().split())

    @staticmethod
    def _safe_int(value: Any, default: int = 10**9) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _context_signature(self, contexts: Iterable[Mapping[str, Any]]) -> Dict[str, set]:
        signatures = {
            "roles": set(),
            "anchors": set(),
            "plans": set(),
            "relations": set(),
            "facts": set(),
        }
        for context in contexts or []:
            plan_id = self._norm(context.get("plan_id"))
            anchor = self._norm(context.get("anchor_entity"))
            if plan_id:
                signatures["plans"].add(plan_id)
            if anchor:
                signatures["anchors"].add(anchor)
            fact_position = context.get("fact_position")
            if plan_id and fact_position is not None:
                signatures["roles"].add((plan_id, str(fact_position)))
            for fact_idx in context.get("fact_indices", []) or []:
                signatures["facts"].add(str(fact_idx))
            rel_tuple = tuple(
                self._norm(relation)
                for relation in (context.get("relations", []) or [])
                if self._norm(relation)
            )
            if rel_tuple:
                signatures["relations"].add(rel_tuple)
        return signatures

    @staticmethod
    def _merge_signature(target: Dict[str, set], source: Mapping[str, set]) -> None:
        for key in target:
            target[key].update(source.get(key, set()) or set())

    def _candidate_score(self, passage: Mapping[str, Any], max_raw_score: float) -> float:
        contexts = passage.get("contexts", []) or []
        max_raw_score = max(float(max_raw_score or 0.0), 1e-8)
        raw_score = self._safe_float(passage.get("raw_lift_score"), 0.0)
        normalized_raw = min(max(raw_score / max_raw_score, 0.0), 1.0)
        original_rank = self._safe_int(passage.get("original_rank"))
        rank_bonus = 1.0 / max(original_rank, 1)
        decision = passage.get("reranker_decision", {}) or {}
        confidence = self._safe_float(decision.get("confidence"), 0.0)
        action = str(decision.get("action", "") or "")
        strict = any(
            str(context.get("signature_match_mode", "") or "").strip().lower() == "strict"
            for context in contexts
        )
        max_path_len = max(
            (
                max(
                    len(context.get("fact_indices", []) or []),
                    len(context.get("relations", []) or []),
                    1,
                )
                for context in contexts
            ),
            default=1,
        )
        max_plan_scale = max(
            (self._safe_float(context.get("plan_lift_weight_scale"), 1.0) for context in contexts),
            default=1.0,
        )
        score = (
            1.10 * normalized_raw
            + 0.75 * confidence
            + 0.45 * min(len(contexts) / 2.0, 1.0)
            + 0.35 * max_plan_scale
            + 0.25 * rank_bonus
        )
        if action == "promote":
            score += 0.60
        if strict:
            score += 0.35
        if max_path_len >= 2:
            score += 0.20
        return float(score)

    def build(
        self,
        sorted_doc_ids: Sequence[int],
        sorted_doc_scores: Sequence[float],
        query_plan_log: Mapping[str, Any],
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        base_doc_ids = [int(doc_id) for doc_id in list(sorted_doc_ids)]
        base_scores = np.array(sorted_doc_scores, dtype=np.float32)
        log: Dict[str, Any] = {
            "enabled": True,
            "mode": "qa_context_builder",
            "qa_top_k": self.qa_top_k,
            "preserve_top_k": self.preserve_top_k,
            "max_evidence_passages": self.max_evidence_passages,
            "max_original_rank": self.max_original_rank,
            "min_candidate_confidence": self.min_candidate_confidence,
            "require_new_evidence_role": self.require_new_evidence_role,
            "selected_evidence_passages": [],
            "selected_evidence_count": 0,
            "changed_top_context": False,
            "skip_reason": "",
        }
        if not base_doc_ids:
            log["skip_reason"] = "empty_retrieval"
            return np.array([], dtype=np.intp), base_scores, log
        if self.max_evidence_passages <= 0:
            log["skip_reason"] = "zero_max_evidence_passages"
            return np.array(base_doc_ids, dtype=np.intp), base_scores, log

        lifted = list(query_plan_log.get("lifted_passages", []) or [])
        if not lifted:
            log["skip_reason"] = "no_lifted_passages"
            return np.array(base_doc_ids, dtype=np.intp), base_scores, log

        passage_details = {}
        max_raw_score = max((self._safe_float(item.get("raw_lift_score"), 0.0) for item in lifted), default=1.0)
        for passage in lifted:
            passage_idx = self._safe_int(passage.get("passage_idx"), default=-1)
            if passage_idx < 0:
                continue

            old = passage_details.get(passage_idx)
            if old is None or len(passage.get("contexts", []) or []) > len(old.get("contexts", []) or []):
                passage_details[passage_idx] = passage

        preserved = base_doc_ids[: min(self.preserve_top_k, len(base_doc_ids))]
        selected = list(preserved)
        selected_set = set(selected)
        evidence_signature = {
            "roles": set(),
            "anchors": set(),
            "plans": set(),
            "relations": set(),
            "facts": set(),
        }
        for doc_id in preserved:
            passage = passage_details.get(int(doc_id))
            if passage:
                self._merge_signature(evidence_signature, self._context_signature(passage.get("contexts", []) or []))

        candidates = []
        for passage_idx, passage in passage_details.items():
            if passage_idx in selected_set:
                continue
            original_rank = self._safe_int(passage.get("original_rank"))
            if self.max_original_rank > 0 and original_rank > self.max_original_rank:
                continue
            contexts = passage.get("contexts", []) or []
            if not contexts:
                continue
            decision = passage.get("reranker_decision", {}) or {}
            confidence = self._safe_float(decision.get("confidence"), 0.0)
            if confidence < self.min_candidate_confidence:
                continue
            score = self._candidate_score(passage, max_raw_score=max_raw_score)
            candidates.append((score, -original_rank, passage_idx, passage))

        if not candidates:
            log["skip_reason"] = "no_context_candidates"
            return np.array(base_doc_ids, dtype=np.intp), base_scores, log

        decisions: List[ContextPassageDecision] = []
        evidence_selected = 0
        for score, _, passage_idx, passage in sorted(candidates, reverse=True):
            if evidence_selected >= self.max_evidence_passages:
                break
            signature = self._context_signature(passage.get("contexts", []) or [])
            added_roles = len(signature["roles"] - evidence_signature["roles"])
            added_anchors = len(signature["anchors"] - evidence_signature["anchors"])
            added_plans = len(signature["plans"] - evidence_signature["plans"])
            added_relations = len(signature["relations"] - evidence_signature["relations"])
            adds_new_evidence = bool(added_roles or added_anchors or added_plans or added_relations)
            if self.require_new_evidence_role and not adds_new_evidence:
                decisions.append(ContextPassageDecision(
                    passage_idx=int(passage_idx),
                    original_rank=self._safe_int(passage.get("original_rank")),
                    final_rank=-1,
                    score=round(float(score), 6),
                    selected=False,
                    reason="no_new_evidence_role",
                    added_roles=0,
                    added_anchors=0,
                    added_plans=0,
                ))
                continue
            selected.append(int(passage_idx))
            selected_set.add(int(passage_idx))
            evidence_selected += 1
            self._merge_signature(evidence_signature, signature)
            decisions.append(ContextPassageDecision(
                passage_idx=int(passage_idx),
                original_rank=self._safe_int(passage.get("original_rank")),
                final_rank=len(selected),
                score=round(float(score), 6),
                selected=True,
                reason="selected_new_evidence_role" if adds_new_evidence else "selected_by_score",
                added_roles=added_roles,
                added_anchors=added_anchors,
                added_plans=added_plans,
            ))

        if evidence_selected <= 0:
            log["skip_reason"] = "no_selected_evidence_passages"
            log["candidate_decisions"] = [decision.__dict__ for decision in decisions[:12]]
            return np.array(base_doc_ids, dtype=np.intp), base_scores, log

        for doc_id in base_doc_ids:
            if len(selected) >= len(base_doc_ids):
                break
            if int(doc_id) in selected_set:
                continue
            selected.append(int(doc_id))
            selected_set.add(int(doc_id))

        score_by_doc = {
            int(doc_id): float(base_scores[idx]) if idx < len(base_scores) else 0.0
            for idx, doc_id in enumerate(base_doc_ids)
        }
        final_doc_ids = np.array(selected, dtype=np.intp)
        final_scores = np.array([score_by_doc.get(int(doc_id), 0.0) for doc_id in selected], dtype=np.float32)
        before_top = base_doc_ids[: self.qa_top_k]
        after_top = selected[: self.qa_top_k]
        log.update({
            "selected_evidence_count": evidence_selected,
            "selected_evidence_passages": [decision.__dict__ for decision in decisions if decision.selected],
            "candidate_decisions": [decision.__dict__ for decision in decisions[:12]],
            "changed_top_context": before_top != after_top,
            "top_context_before": before_top,
            "top_context_after": after_top,
        })
        return final_doc_ids, final_scores, log
