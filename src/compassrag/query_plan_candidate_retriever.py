from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

from .query_plan_structural_candidate_retriever import QueryPlanOntologyStructuralCandidateRetriever
from .utils.misc_utils import text_processing


@dataclass
class QueryPlanCandidateRetrieverResult:
    sorted_doc_ids: np.ndarray
    sorted_doc_scores: np.ndarray
    log: Dict[str, Any] = field(default_factory=dict)


class QueryPlanOntologyCandidateRetriever:
    """Promote high-confidence ontology source passages before final retrieval.

    Existing query-plan passage lift mostly nudges scores after graph retrieval.
    This retriever is deliberately earlier and more direct: if executable
    ontology paths point to source passages with strong path evidence, it can
    insert those passages into the retrieval top window. The policy is still
    conservative: it preserves the top baseline passages and only promotes
    candidates whose structural confidence clears a threshold.
    """

    def __init__(
        self,
        top_k: int = 2,
        promote_to_rank: int = 5,
        preserve_top_k: int = 2,
        max_original_rank: int = 120,
        min_confidence: float = 0.72,
        score_margin: float = 1e-6,
        aggressive_promote_to_rank: int = 2,
        aggressive_min_confidence: float = 0.84,
        aggressive_require_strict: bool = True,
        aggressive_require_multihop_or_multi_context: bool = True,
        enable_anchor_guard: bool = True,
        allow_structural_only: bool = True,
        structural_only_min_confidence: float = 0.92,
    ):
        self.top_k = max(int(top_k), 0)
        self.promote_to_rank = max(int(promote_to_rank), 1)
        self.preserve_top_k = max(int(preserve_top_k), 0)
        self.max_original_rank = int(max_original_rank)
        self.min_confidence = min(max(float(min_confidence), 0.0), 1.0)
        self.score_margin = max(float(score_margin), 0.0)
        self.aggressive_promote_to_rank = max(int(aggressive_promote_to_rank), 1)
        self.aggressive_min_confidence = min(max(float(aggressive_min_confidence), 0.0), 1.0)
        self.aggressive_require_strict = bool(aggressive_require_strict)
        self.aggressive_require_multihop_or_multi_context = bool(aggressive_require_multihop_or_multi_context)
        self.enable_anchor_guard = bool(enable_anchor_guard)
        self.allow_structural_only = bool(allow_structural_only)
        self.structural_only_min_confidence = min(max(float(structural_only_min_confidence), 0.0), 1.0)

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _has_strict_context(contexts: Sequence[Mapping[str, Any]]) -> bool:
        for context in contexts or []:
            if str(context.get("signature_match_mode", "") or "").strip().lower() == "strict":
                return True
        return False

    @staticmethod
    def _max_context_value(contexts: Sequence[Mapping[str, Any]], key: str, default: float = 0.0) -> float:
        values = []
        for context in contexts or []:
            try:
                values.append(float(context.get(key, default) or default))
            except (TypeError, ValueError):
                continue
        return max(values, default=default)

    @staticmethod
    def _max_path_len(contexts: Sequence[Mapping[str, Any]]) -> int:
        max_len = 1
        for context in contexts or []:
            max_len = max(
                max_len,
                len(context.get("fact_indices", []) or []),
                len(context.get("relations", []) or []),
            )
        return max_len

    @staticmethod
    def _has_concrete_anchor_or_terminal(contexts: Sequence[Mapping[str, Any]]) -> bool:
        for context in contexts or []:
            anchor = text_processing(context.get("anchor_entity", ""))
            terminal = text_processing(context.get("end_entity", ""))
            if anchor and not QueryPlanOntologyStructuralCandidateRetriever.is_generic_anchor(anchor):
                return True
            if terminal and not QueryPlanOntologyStructuralCandidateRetriever.is_generic_anchor(terminal):
                return True
        return False

    @staticmethod
    def _has_terminal_relation_context(contexts: Sequence[Mapping[str, Any]]) -> bool:
        for context in contexts or []:
            terminal = text_processing(context.get("end_entity", ""))
            if not terminal or QueryPlanOntologyStructuralCandidateRetriever.is_generic_anchor(terminal):
                continue
            relations = [
                text_processing(relation)
                for relation in (context.get("relations", []) or [])
                if text_processing(relation)
            ]
            if relations:
                return True
        return False

    @staticmethod
    def _is_structural_only(contexts: Sequence[Mapping[str, Any]]) -> bool:
        contexts = list(contexts or [])
        if not contexts:
            return False
        return all(bool(context.get("structural_candidate", False)) for context in contexts)

    def _target_rank_for_candidate(self, confidence: float, contexts: Sequence[Mapping[str, Any]]) -> int:
        if confidence < self.aggressive_min_confidence:
            return self.promote_to_rank
        if self.aggressive_require_strict and not self._has_strict_context(contexts):
            return self.promote_to_rank
        if self.aggressive_require_multihop_or_multi_context:
            if self._max_path_len(contexts) < 2 and len(contexts or []) < 2:
                if not self._has_terminal_relation_context(contexts):
                    return self.promote_to_rank
        return self.aggressive_promote_to_rank

    def _confidence(
        self,
        raw_score: float,
        max_raw_score: float,
        contexts: Sequence[Mapping[str, Any]],
    ) -> float:
        norm_score = 0.0 if max_raw_score <= 0.0 else raw_score / max_raw_score
        norm_score = min(max(norm_score, 0.0), 1.0)
        precision = min(max(self._max_context_value(contexts, "plan_effective_precision", 0.0), 0.0), 1.0)
        scale = min(max(self._max_context_value(contexts, "plan_lift_weight_scale", 1.0), 0.0), 1.0)
        strict = 1.0 if self._has_strict_context(contexts) else 0.0
        context_strength = min(len(contexts or []) / 2.0, 1.0)
        confidence = (
            0.42 * norm_score
            + 0.24 * precision
            + 0.14 * scale
            + 0.12 * strict
            + 0.08 * context_strength
        )
        return min(max(confidence, 0.0), 1.0)

    def rerank(
        self,
        sorted_doc_ids: np.ndarray,
        sorted_doc_scores: np.ndarray,
        passage_scores: Mapping[int, float],
        passage_contexts: Mapping[int, Sequence[Mapping[str, Any]]],
        original_rank_by_doc: Mapping[int, int],
    ) -> QueryPlanCandidateRetrieverResult:
        log: Dict[str, Any] = {
            "enabled": True,
            "candidate_passage_count": 0,
            "scored_passage_count": 0,
            "promoted_passage_count": 0,
            "top_k": self.top_k,
            "promote_to_rank": self.promote_to_rank,
            "aggressive_promote_to_rank": self.aggressive_promote_to_rank,
            "preserve_top_k": self.preserve_top_k,
            "max_original_rank": self.max_original_rank,
            "min_confidence": self.min_confidence,
            "aggressive_min_confidence": self.aggressive_min_confidence,
            "enable_anchor_guard": self.enable_anchor_guard,
            "allow_structural_only": self.allow_structural_only,
            "structural_only_min_confidence": self.structural_only_min_confidence,
        }
        sorted_doc_ids = np.array(sorted_doc_ids, dtype=np.intp)
        sorted_doc_scores = np.array(sorted_doc_scores, dtype=np.float32)
        if self.top_k <= 0:
            log["skip_reason"] = "zero_top_k"
            return QueryPlanCandidateRetrieverResult(sorted_doc_ids, sorted_doc_scores, log)
        if sorted_doc_ids.size == 0:
            log["skip_reason"] = "empty_retrieval"
            return QueryPlanCandidateRetrieverResult(sorted_doc_ids, sorted_doc_scores, log)
        if not passage_scores:
            log["skip_reason"] = "no_passage_scores"
            return QueryPlanCandidateRetrieverResult(sorted_doc_ids, sorted_doc_scores, log)

        max_raw_score = max((self._safe_float(score) for score in passage_scores.values()), default=0.0)
        if max_raw_score <= 0.0:
            log["skip_reason"] = "nonpositive_passage_scores"
            return QueryPlanCandidateRetrieverResult(sorted_doc_ids, sorted_doc_scores, log)

        preserved = {int(doc_id) for doc_id in sorted_doc_ids[: self.preserve_top_k]}
        candidates = []
        rejected_before_preserve = 0
        rejected_after_rank = 0
        rejected_low_confidence = 0
        rejected_generic_anchor = 0
        rejected_structural_only = 0

        for passage_idx, raw_score_value in passage_scores.items():
            passage_idx = int(passage_idx)
            raw_score = self._safe_float(raw_score_value)
            original_rank = int(original_rank_by_doc.get(passage_idx, len(sorted_doc_ids) + 1))
            if passage_idx in preserved:
                rejected_before_preserve += 1
                continue
            if self.max_original_rank > 0 and original_rank > self.max_original_rank:
                rejected_after_rank += 1
                continue
            contexts = list(passage_contexts.get(passage_idx, []) or [])
            if self.enable_anchor_guard and not self._has_concrete_anchor_or_terminal(contexts):
                rejected_generic_anchor += 1
                continue
            confidence = self._confidence(raw_score, max_raw_score, contexts)
            structural_only = self._is_structural_only(contexts)
            if structural_only:
                if not self.allow_structural_only or confidence < self.structural_only_min_confidence:
                    rejected_structural_only += 1
                    continue
            if confidence < self.min_confidence:
                rejected_low_confidence += 1
                continue
            target_rank = self._target_rank_for_candidate(confidence, contexts)
            if structural_only:
                target_rank = max(target_rank, self.promote_to_rank)
            candidates.append({
                "passage_idx": passage_idx,
                "raw_score": raw_score,
                "confidence": confidence,
                "original_rank": original_rank,
                "target_rank": target_rank,
                "contexts": contexts[:3],
            })

        candidates.sort(
            key=lambda item: (
                float(item["confidence"]),
                float(item["raw_score"]),
                -int(item["original_rank"]),
            ),
            reverse=True,
        )
        selected = candidates[: self.top_k]
        log["candidate_passage_count"] = len(passage_scores)
        log["scored_passage_count"] = len(candidates)
        log["rejected_preserved_top_count"] = rejected_before_preserve
        log["rejected_after_rank_count"] = rejected_after_rank
        log["rejected_low_confidence_count"] = rejected_low_confidence
        log["rejected_generic_anchor_count"] = rejected_generic_anchor
        log["rejected_structural_only_count"] = rejected_structural_only
        log["promoted_passage_count"] = len(selected)
        log["promoted_passages"] = [
            {
                "passage_idx": int(item["passage_idx"]),
                "confidence": round(float(item["confidence"]), 6),
                "raw_score": round(float(item["raw_score"]), 6),
                "original_rank": int(item["original_rank"]),
                "target_rank": int(item["target_rank"]),
                "contexts": item["contexts"],
            }
            for item in selected
        ]
        if not selected:
            log["skip_reason"] = "no_promotable_passages"
            return QueryPlanCandidateRetrieverResult(sorted_doc_ids, sorted_doc_scores, log)

        score_by_doc = {int(doc_id): float(score) for doc_id, score in zip(sorted_doc_ids, sorted_doc_scores)}
        epsilon = abs(float(sorted_doc_scores[0])) * self.score_margin + self.score_margin
        for offset, item in enumerate(selected):
            passage_idx = int(item["passage_idx"])
            target_rank = int(item.get("target_rank", self.promote_to_rank))
            target_index = min(max(target_rank - 1, self.preserve_top_k), len(sorted_doc_scores) - 1)
            target_score = float(sorted_doc_scores[target_index]) if sorted_doc_scores.size else 0.0
            score_by_doc[passage_idx] = max(
                score_by_doc.get(passage_idx, 0.0),
                target_score + epsilon * float(len(selected) - offset),
            )

        reranked = sorted(score_by_doc.items(), key=lambda item: item[1], reverse=True)
        new_doc_ids = np.array([doc_id for doc_id, _ in reranked], dtype=np.intp)
        new_doc_scores = np.array([score for _, score in reranked], dtype=np.float32)
        final_rank_by_doc = {int(doc_id): rank + 1 for rank, doc_id in enumerate(new_doc_ids)}
        for item in log["promoted_passages"]:
            item["final_rank"] = int(final_rank_by_doc.get(int(item["passage_idx"]), len(new_doc_ids) + 1))
            item["rank_delta"] = int(item["original_rank"]) - int(item["final_rank"])
        return QueryPlanCandidateRetrieverResult(new_doc_ids, new_doc_scores, log)
