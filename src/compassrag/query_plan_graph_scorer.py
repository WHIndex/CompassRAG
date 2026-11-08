from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


@dataclass
class QueryPlanGraphScorerResult:
    top_k_fact_indices: List[int]
    top_k_facts: List[Tuple]
    query_fact_scores: np.ndarray
    log: Dict[str, Any] = field(default_factory=dict)


class QueryPlanOntologyGraphScorer:
    """Boost graph-search fact seeds with executable query-plan evidence.

    Passage-level reranking can only move documents after CompassRAG has already
    run PPR. This scorer acts earlier: it uses query-plan path facts as a
    structured prior over fact nodes, then lets the existing entity/PPR graph
    search propagate that signal to passages.
    """

    def __init__(
        self,
        score_weight: float = 0.28,
        top_k: int = 6,
        min_confidence: float = 0.45,
        require_anchor_or_dense_source: bool = True,
        boost_existing: bool = True,
        reorder_facts: bool = True,
    ):
        self.score_weight = min(max(float(score_weight), 0.0), 1.0)
        self.top_k = max(int(top_k), 0)
        self.min_confidence = min(max(float(min_confidence), 0.0), 1.0)
        self.require_anchor_or_dense_source = bool(require_anchor_or_dense_source)
        self.boost_existing = bool(boost_existing)
        self.reorder_facts = bool(reorder_facts)

    @staticmethod
    def _as_int(value: Any) -> Optional[int]:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _context_plan_precision(contexts: Iterable[Mapping[str, Any]]) -> float:
        precisions = []
        for context in contexts or []:
            try:
                precisions.append(float(context.get("plan_effective_precision", 0.0) or 0.0))
            except (TypeError, ValueError):
                continue
        return min(max(max(precisions, default=0.0), 0.0), 1.0)

    @staticmethod
    def _has_strict_context(contexts: Iterable[Mapping[str, Any]]) -> bool:
        for context in contexts or []:
            if str(context.get("signature_match_mode", "") or "").strip().lower() == "strict":
                return True
        return False

    @staticmethod
    def _candidate_key(candidate: Mapping[str, Any]) -> Optional[int]:
        return QueryPlanOntologyGraphScorer._as_int(candidate.get("fact_index"))

    def _candidate_confidence(self, candidate: Mapping[str, Any]) -> float:
        contexts = candidate.get("contexts", []) or []
        try:
            normalized_path_score = float(candidate.get("normalized_path_score", 0.0) or 0.0)
        except (TypeError, ValueError):
            normalized_path_score = 0.0
        plan_precision = self._context_plan_precision(contexts)
        has_strict = self._has_strict_context(contexts)
        has_anchor = bool(candidate.get("fact_has_query_anchor", False))
        has_dense_source = bool(candidate.get("source_in_dense_top_k", False))
        context_strength = min(len(contexts) / 2.0, 1.0)

        confidence = (
            0.42 * min(max(normalized_path_score, 0.0), 1.0)
            + 0.22 * plan_precision
            + 0.14 * (1.0 if has_strict else 0.0)
            + 0.10 * (1.0 if has_anchor else 0.0)
            + 0.07 * (1.0 if has_dense_source else 0.0)
            + 0.05 * context_strength
        )
        return min(max(confidence, 0.0), 1.0)

    def score(
        self,
        query_fact_scores: np.ndarray,
        top_k_fact_indices: Sequence[int],
        top_k_facts: Sequence[Tuple],
        fact_triples: Sequence[Tuple],
        query_plan_ppr_log: Optional[Mapping[str, Any]],
    ) -> QueryPlanGraphScorerResult:
        log: Dict[str, Any] = {
            "enabled": True,
            "candidate_fact_count": 0,
            "scored_fact_count": 0,
            "added_fact_count": 0,
            "boosted_existing_fact_count": 0,
        }
        if len(query_fact_scores) == 0:
            log["skip_reason"] = "empty_scores"
            return QueryPlanGraphScorerResult(list(top_k_fact_indices), list(top_k_facts), query_fact_scores, log)
        if not query_plan_ppr_log or not query_plan_ppr_log.get("enabled", False):
            log["skip_reason"] = "no_query_plan_ppr_log"
            return QueryPlanGraphScorerResult(list(top_k_fact_indices), list(top_k_facts), query_fact_scores, log)

        raw_candidates = list(query_plan_ppr_log.get("candidate_facts", []) or [])
        if not raw_candidates:
            raw_candidates = list(query_plan_ppr_log.get("added_facts", []) or [])
        if not raw_candidates:
            log["skip_reason"] = "no_candidate_facts"
            return QueryPlanGraphScorerResult(list(top_k_fact_indices), list(top_k_facts), query_fact_scores, log)

        selected_indices = {
            int(idx)
            for idx in top_k_fact_indices
            if self._as_int(idx) is not None
        }
        fact_by_idx = {
            int(idx): tuple(fact)
            for idx, fact in zip(top_k_fact_indices, top_k_facts)
            if self._as_int(idx) is not None
        }
        if top_k_fact_indices:
            anchor_score = float(np.max(query_fact_scores[list(selected_indices)])) if selected_indices else 0.0
        else:
            dense_anchor_count = min(5, len(query_fact_scores))
            anchor_score = float(np.max(np.sort(query_fact_scores)[-dense_anchor_count:])) if dense_anchor_count else 0.0
        if anchor_score <= 0.0:
            anchor_score = float(np.max(query_fact_scores)) if len(query_fact_scores) else 0.0

        deduped: Dict[int, Dict[str, Any]] = {}
        for candidate in raw_candidates:
            if not isinstance(candidate, Mapping):
                continue
            fact_idx = self._candidate_key(candidate)
            if fact_idx is None or fact_idx < 0 or fact_idx >= len(query_fact_scores) or fact_idx >= len(fact_triples):
                continue
            if fact_triples[fact_idx] is None:
                continue
            confidence = self._candidate_confidence(candidate)
            candidate_copy = dict(candidate)
            candidate_copy["confidence"] = confidence
            if fact_idx not in deduped or confidence > float(deduped[fact_idx].get("confidence", 0.0) or 0.0):
                deduped[fact_idx] = candidate_copy

        candidates = sorted(
            deduped.values(),
            key=lambda item: (
                float(item.get("confidence", 0.0) or 0.0),
                float(item.get("normalized_path_score", 0.0) or 0.0),
            ),
            reverse=True,
        )
        log["candidate_fact_count"] = len(candidates)
        if not candidates:
            log["skip_reason"] = "no_valid_candidate_facts"
            return QueryPlanGraphScorerResult(list(top_k_fact_indices), list(top_k_facts), query_fact_scores, log)

        updated_scores = np.array(query_fact_scores, copy=True)
        added_indices: List[int] = []
        selected_details: List[Dict[str, Any]] = []
        rejected_missing_anchor_or_source = 0
        rejected_low_confidence = 0
        boosted_existing = 0

        for candidate in candidates:
            if len(added_indices) >= self.top_k and int(candidate["fact_index"]) not in selected_indices:
                continue
            fact_idx = int(candidate["fact_index"])
            is_existing = fact_idx in selected_indices
            if is_existing and not self.boost_existing:
                continue

            has_anchor = bool(candidate.get("fact_has_query_anchor", False))
            has_dense_source = bool(candidate.get("source_in_dense_top_k", False))
            if self.require_anchor_or_dense_source and not (has_anchor or has_dense_source):
                rejected_missing_anchor_or_source += 1
                continue

            confidence = float(candidate.get("confidence", 0.0) or 0.0)
            if confidence < self.min_confidence:
                rejected_low_confidence += 1
                continue

            old_score = float(updated_scores[fact_idx])
            target_score = anchor_score * confidence
            raised_score = (1.0 - self.score_weight) * old_score + self.score_weight * target_score
            updated_scores[fact_idx] = max(old_score, raised_score)

            if is_existing:
                boosted_existing += 1
            else:
                added_indices.append(fact_idx)
                selected_indices.add(fact_idx)
                fact_by_idx[fact_idx] = tuple(fact_triples[fact_idx])

            selected_details.append({
                "fact_index": fact_idx,
                "confidence": round(confidence, 6),
                "old_score": round(old_score, 6),
                "new_score": round(float(updated_scores[fact_idx]), 6),
                "score_delta": round(float(updated_scores[fact_idx]) - old_score, 6),
                "already_selected": bool(is_existing),
                "fact_has_query_anchor": has_anchor,
                "source_in_dense_top_k": has_dense_source,
                "source_passages": list(candidate.get("source_passages", []) or [])[:10],
                "contexts": list(candidate.get("contexts", []) or [])[:3],
            })

        log["rejected_missing_anchor_or_source_count"] = rejected_missing_anchor_or_source
        log["rejected_low_confidence_count"] = rejected_low_confidence
        log["scored_fact_count"] = len(selected_details)
        log["added_fact_count"] = len(added_indices)
        log["boosted_existing_fact_count"] = boosted_existing
        log["selected_fact_indices"] = [int(item["fact_index"]) for item in selected_details]
        log["added_fact_indices"] = added_indices
        log["selected_facts"] = selected_details[:40]
        log["score_weight"] = self.score_weight
        log["min_confidence"] = self.min_confidence
        log["top_k"] = self.top_k
        log["require_anchor_or_dense_source"] = self.require_anchor_or_dense_source
        log["boost_existing"] = self.boost_existing
        log["anchor_score"] = round(anchor_score, 6)

        if not selected_details:
            log["skip_reason"] = "no_scored_facts"
            return QueryPlanGraphScorerResult(list(top_k_fact_indices), list(top_k_facts), query_fact_scores, log)

        merged_indices = list(dict.fromkeys([int(idx) for idx in top_k_fact_indices] + added_indices))
        if self.reorder_facts:
            merged_indices.sort(key=lambda idx: float(updated_scores[idx]), reverse=True)
        merged_facts = [fact_by_idx.get(idx, tuple(fact_triples[idx])) for idx in merged_indices]
        return QueryPlanGraphScorerResult(merged_indices, merged_facts, updated_scores, log)
