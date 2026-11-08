from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from .query_ontology_prior import tokenize
from .query_plan_structural_candidate_retriever import QueryPlanOntologyStructuralCandidateRetriever
from .utils.misc_utils import text_processing


@dataclass
class QueryPlanMissingEvidenceResult:
    sorted_doc_ids: np.ndarray
    sorted_doc_scores: np.ndarray
    log: Dict[str, Any] = field(default_factory=dict)


class QueryPlanMissingEvidenceCompleter:
    """Retrieve missing supporting evidence from ontology path fact slots.

    Earlier modules mostly score passages already surfaced by CompassRAG. This
    module starts from query-plan executable paths, turns uncovered path facts
    into dense subqueries, then verifies retrieved passages against the fact
    endpoints and relation tokens. It is designed as a high-precision candidate
    generator for the ontology evidence-set retriever.
    """

    GENERIC_RELATION_TOKENS = {
        "of", "in", "on", "at", "by", "to", "from", "for", "with", "and",
        "or", "the", "a", "an", "is", "was", "were", "be", "has", "have",
        "had", "as", "into", "during", "after", "before", "part",
    }

    def __init__(
        self,
        dense_top_k: int = 40,
        top_window_k: int = 5,
        max_missing_slots: int = 6,
        max_query_variants: int = 8,
        max_candidates: int = 3,
        min_confidence: float = 0.70,
        promote_to_rank: int = 5,
        preserve_top_k: int = 2,
        require_relation_hit: bool = True,
        require_both_endpoints: bool = False,
        candidate_only: bool = True,
        include_covered_slots: bool = False,
        max_covered_slots: int = 4,
        score_margin: float = 1e-6,
    ):
        self.dense_top_k = max(int(dense_top_k), 1)
        self.top_window_k = max(int(top_window_k), 1)
        self.max_missing_slots = max(int(max_missing_slots), 0)
        self.max_query_variants = max(int(max_query_variants), 1)
        self.max_candidates = max(int(max_candidates), 0)
        self.min_confidence = min(max(float(min_confidence), 0.0), 1.0)
        self.promote_to_rank = max(int(promote_to_rank), 1)
        self.preserve_top_k = max(int(preserve_top_k), 0)
        self.require_relation_hit = bool(require_relation_hit)
        self.require_both_endpoints = bool(require_both_endpoints)
        self.candidate_only = bool(candidate_only)
        self.include_covered_slots = bool(include_covered_slots)
        self.max_covered_slots = max(int(max_covered_slots), 0)
        self.score_margin = max(float(score_margin), 0.0)

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _norm(value: Any) -> str:
        return text_processing(str(value or ""))

    @classmethod
    def _contains_phrase(cls, text_norm: str, phrase: Any) -> bool:
        phrase_norm = cls._norm(phrase)
        if not phrase_norm:
            return False
        if f" {phrase_norm} " in f" {text_norm} ":
            return True
        phrase_tokens = tokenize(phrase_norm)
        if not phrase_tokens:
            return False
        text_tokens = set(tokenize(text_norm))
        matched = sum(1 for token in phrase_tokens if token in text_tokens)
        return matched / max(len(phrase_tokens), 1) >= 0.75

    @classmethod
    def _relation_tokens(cls, relation: Any) -> List[str]:
        tokens = []
        seen = set()
        for token in tokenize(cls._norm(str(relation or "").replace("_", " "))):
            if token in cls.GENERIC_RELATION_TOKENS or token in seen:
                continue
            tokens.append(token)
            seen.add(token)
        return tokens

    @classmethod
    def _relation_hits(cls, text_norm: str, relation_tokens: Sequence[str]) -> List[str]:
        text_tokens = set(tokenize(text_norm))
        return [token for token in relation_tokens if token in text_tokens]

    @staticmethod
    def _concrete_entity(value: Any) -> str:
        entity = str(value or "").strip()
        key = text_processing(entity)
        if not key or QueryPlanOntologyStructuralCandidateRetriever.is_generic_anchor(key):
            return ""
        return entity

    def _fact_covered_by_text(self, fact: Mapping[str, Any], text_norm: str) -> Tuple[bool, Dict[str, Any]]:
        subject = fact.get("subject", "")
        obj = fact.get("object", "")
        relation_tokens = fact.get("relation_tokens", []) or []
        subject_hit = self._contains_phrase(text_norm, subject)
        object_hit = self._contains_phrase(text_norm, obj)
        relation_hits = self._relation_hits(text_norm, relation_tokens)
        endpoint_hit_count = int(subject_hit) + int(object_hit)
        endpoint_ok = endpoint_hit_count >= (2 if self.require_both_endpoints else 1)
        relation_ok = bool(relation_hits) or not relation_tokens or not self.require_relation_hit
        covered = endpoint_ok and relation_ok
        return covered, {
            "subject_hit": subject_hit,
            "object_hit": object_hit,
            "relation_hits": relation_hits,
            "endpoint_hit_count": endpoint_hit_count,
        }

    def _build_fact_slots(
        self,
        query_plan_log: Mapping[str, Any],
        get_fact_triples: Callable[[Any], List[Tuple[str, str, str]]],
    ) -> List[Dict[str, Any]]:
        slots = []
        seen = set()
        paths = list(query_plan_log.get("sample_paths", []) or [])
        for passage in query_plan_log.get("lifted_passages", []) or []:
            for context in passage.get("contexts", []) or []:
                if isinstance(context, Mapping):
                    paths.append(context)

        for path in paths:
            if not isinstance(path, Mapping):
                continue
            fact_indices = list(path.get("fact_indices", []) or [])
            triples = get_fact_triples(fact_indices)
            for position, triple in enumerate(triples):
                if len(triple) != 3:
                    continue
                subject, relation, obj = [str(item or "").strip() for item in triple]
                subject = self._concrete_entity(subject)
                obj = self._concrete_entity(obj)
                relation_tokens = self._relation_tokens(relation)
                if not subject and not obj:
                    continue
                key = (
                    text_processing(subject),
                    text_processing(relation),
                    text_processing(obj),
                )
                if key in seen:
                    continue
                seen.add(key)
                slots.append({
                    "subject": subject,
                    "relation": relation,
                    "object": obj,
                    "relation_tokens": relation_tokens,
                    "fact_indices": fact_indices,
                    "fact_position": position,
                    "plan_id": path.get("plan_id", ""),
                    "anchor_entity": path.get("anchor_entity", ""),
                    "end_entity": path.get("end_entity") or path.get("entity") or obj,
                    "relations": path.get("relations", []) or [relation],
                    "signature_match_mode": path.get("signature_match_mode", ""),
                    "plan_effective_precision": self._safe_float(path.get("plan_effective_precision"), 0.0),
                    "plan_lift_weight_scale": self._safe_float(path.get("plan_lift_weight_scale"), 1.0),
                    "path_score": self._safe_float(path.get("score"), 0.0),
                })
        slots.sort(
            key=lambda item: (
                float(item.get("plan_effective_precision", 0.0)),
                float(item.get("path_score", 0.0)),
                1.0 if item.get("signature_match_mode") == "strict" else 0.0,
            ),
            reverse=True,
        )
        return slots

    def _split_slots_by_top_window_coverage(
        self,
        slots: Sequence[Mapping[str, Any]],
        top_texts: Sequence[str],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        missing = []
        covered_slots = []
        for slot in slots:
            covered = False
            best_detail = {}
            for text_norm in top_texts:
                covered, detail = self._fact_covered_by_text(slot, text_norm)
                if covered:
                    best_detail = detail
                    break
            if covered:
                slot_copy = dict(slot)
                slot_copy["coverage_detail"] = best_detail
                covered_slots.append(slot_copy)
                continue
            slot_copy = dict(slot)
            slot_copy["coverage_detail"] = best_detail
            missing.append(slot_copy)
            if len(missing) >= self.max_missing_slots:
                continue
        return missing[: self.max_missing_slots], covered_slots

    def _slot_subqueries(self, query: str, slot: Mapping[str, Any]) -> List[str]:
        subject = str(slot.get("subject") or "").strip()
        relation = str(slot.get("relation") or "").strip()
        obj = str(slot.get("object") or "").strip()
        relation_phrase = " ".join(slot.get("relation_tokens", []) or []) or relation
        subqueries = []
        if subject and obj and relation:
            subqueries.append(f"{subject} {relation} {obj}")
        if subject and relation_phrase:
            subqueries.append(f"{subject} {relation_phrase}")
        if obj and relation_phrase:
            subqueries.append(f"{obj} {relation_phrase}")
        if obj:
            subqueries.append(f"{query} {obj}")
        if subject:
            subqueries.append(f"{query} {subject}")
        return subqueries

    def _build_subqueries(self, query: str, missing_slots: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
        subqueries = []
        seen = set()
        for slot in missing_slots:
            for subquery in self._slot_subqueries(query, slot):
                key = text_processing(subquery)
                if not key or key in seen:
                    continue
                seen.add(key)
                subqueries.append({
                    "query": subquery,
                    "slot": dict(slot),
                })
                if len(subqueries) >= self.max_query_variants:
                    return subqueries
        return subqueries

    def _score_candidate(
        self,
        passage_text: str,
        dense_score: float,
        max_dense_score: float,
        slot: Mapping[str, Any],
    ) -> Tuple[float, Dict[str, Any]]:
        text_norm = text_processing(passage_text)
        covered, detail = self._fact_covered_by_text(slot, text_norm)
        relation_tokens = slot.get("relation_tokens", []) or []
        relation_score = min(len(detail.get("relation_hits", []) or []) / max(min(len(relation_tokens), 4), 1), 1.0)
        endpoint_score = min(detail.get("endpoint_hit_count", 0) / 2.0, 1.0)
        dense_norm = 0.0 if max_dense_score <= 0.0 else float(dense_score) / max_dense_score
        dense_norm = min(max(dense_norm, 0.0), 1.0)
        precision = min(max(self._safe_float(slot.get("plan_effective_precision"), 0.0), 0.0), 1.0)
        strict = 1.0 if str(slot.get("signature_match_mode", "")).lower() == "strict" else 0.0
        confidence = (
            0.30 * dense_norm
            + 0.30 * endpoint_score
            + 0.20 * relation_score
            + 0.12 * precision
            + 0.08 * strict
        )
        detail.update({
            "covered": covered,
            "dense_norm": dense_norm,
            "endpoint_score": endpoint_score,
            "relation_score": relation_score,
            "plan_precision": precision,
            "strict": bool(strict),
        })
        return min(max(confidence, 0.0), 1.0), detail

    def complete(
        self,
        query: str,
        sorted_doc_ids: Sequence[int],
        sorted_doc_scores: Sequence[float],
        query_plan_log: Mapping[str, Any],
        dense_retrieve: Callable[[str], Tuple[np.ndarray, np.ndarray]],
        get_passage_text: Callable[[int], str],
        get_fact_triples: Callable[[Any], List[Tuple[str, str, str]]],
    ) -> QueryPlanMissingEvidenceResult:
        sorted_doc_ids = np.array(sorted_doc_ids, dtype=np.intp)
        sorted_doc_scores = np.array(sorted_doc_scores, dtype=np.float32)
        log: Dict[str, Any] = {
            "enabled": True,
            "dense_top_k": self.dense_top_k,
            "top_window_k": self.top_window_k,
            "max_missing_slots": self.max_missing_slots,
            "max_query_variants": self.max_query_variants,
            "max_candidates": self.max_candidates,
            "min_confidence": self.min_confidence,
            "candidate_only": self.candidate_only,
            "include_covered_slots": self.include_covered_slots,
            "added_candidates": [],
            "subqueries": [],
        }
        if self.max_candidates <= 0:
            log["skip_reason"] = "zero_max_candidates"
            return QueryPlanMissingEvidenceResult(sorted_doc_ids, sorted_doc_scores, log)
        if sorted_doc_ids.size == 0:
            log["skip_reason"] = "empty_retrieval"
            return QueryPlanMissingEvidenceResult(sorted_doc_ids, sorted_doc_scores, log)
        if not query_plan_log or not query_plan_log.get("enabled", False):
            log["skip_reason"] = "no_query_plan_log"
            return QueryPlanMissingEvidenceResult(sorted_doc_ids, sorted_doc_scores, log)

        top_doc_ids = [int(doc_id) for doc_id in sorted_doc_ids[: self.top_window_k]]
        top_texts = [text_processing(get_passage_text(doc_id)) for doc_id in top_doc_ids]
        fact_slots = self._build_fact_slots(query_plan_log, get_fact_triples)
        log["fact_slot_count"] = len(fact_slots)
        if not fact_slots:
            log["skip_reason"] = "no_fact_slots"
            return QueryPlanMissingEvidenceResult(sorted_doc_ids, sorted_doc_scores, log)
        missing_slots, covered_slots = self._split_slots_by_top_window_coverage(fact_slots, top_texts)
        log["covered_slot_count"] = len(covered_slots)
        log["missing_slot_count"] = len(missing_slots)
        log["missing_slots_sample"] = [
            {
                "subject": item.get("subject", ""),
                "relation": item.get("relation", ""),
                "object": item.get("object", ""),
                "plan_id": item.get("plan_id", ""),
                "fact_position": item.get("fact_position", 0),
                "signature_match_mode": item.get("signature_match_mode", ""),
            }
            for item in missing_slots[:5]
        ]
        slots_to_expand = list(missing_slots)
        covered_probe_slots = []
        if self.include_covered_slots and self.max_covered_slots > 0:
            covered_probe_slots = [dict(item, covered_slot_probe=True) for item in covered_slots[: self.max_covered_slots]]
            slots_to_expand.extend(covered_probe_slots)
        log["covered_probe_slot_count"] = len(covered_probe_slots)
        if not slots_to_expand:
            log["skip_reason"] = "no_missing_slots"
            return QueryPlanMissingEvidenceResult(sorted_doc_ids, sorted_doc_scores, log)

        subqueries = self._build_subqueries(query, slots_to_expand)
        log["subqueries"] = [
            {
                "query": item["query"],
                "slot_subject": item["slot"].get("subject", ""),
                "slot_relation": item["slot"].get("relation", ""),
                "slot_object": item["slot"].get("object", ""),
            }
            for item in subqueries
        ]
        if not subqueries:
            log["skip_reason"] = "no_subqueries"
            return QueryPlanMissingEvidenceResult(sorted_doc_ids, sorted_doc_scores, log)

        original_rank_by_doc = {int(doc_id): rank + 1 for rank, doc_id in enumerate(sorted_doc_ids)}
        selected_by_passage: Dict[int, Dict[str, Any]] = {}
        rejection_reasons: Dict[str, int] = {}
        for item in subqueries:
            subquery = item["query"]
            slot = item["slot"]
            dense_doc_ids, dense_scores = dense_retrieve(subquery)
            dense_doc_ids = list(dense_doc_ids[: self.dense_top_k])
            dense_scores = list(dense_scores[: self.dense_top_k])
            max_dense_score = max((self._safe_float(score) for score in dense_scores), default=0.0)
            for dense_rank, passage_idx in enumerate(dense_doc_ids, start=1):
                passage_idx = int(passage_idx)
                if passage_idx in top_doc_ids[: self.preserve_top_k]:
                    rejection_reasons["preserved_top"] = rejection_reasons.get("preserved_top", 0) + 1
                    continue
                passage_text = get_passage_text(passage_idx)
                confidence, detail = self._score_candidate(
                    passage_text=passage_text,
                    dense_score=self._safe_float(dense_scores[dense_rank - 1] if dense_rank - 1 < len(dense_scores) else 0.0),
                    max_dense_score=max_dense_score,
                    slot=slot,
                )
                if not detail.get("covered"):
                    rejection_reasons["slot_not_covered"] = rejection_reasons.get("slot_not_covered", 0) + 1
                    continue
                if confidence < self.min_confidence:
                    rejection_reasons["low_confidence"] = rejection_reasons.get("low_confidence", 0) + 1
                    continue
                existing = selected_by_passage.get(passage_idx)
                candidate = {
                    "passage_idx": passage_idx,
                    "confidence": confidence,
                    "dense_rank": dense_rank,
                    "original_rank": int(original_rank_by_doc.get(passage_idx, len(sorted_doc_ids) + 1)),
                    "score": confidence + 0.01 / max(dense_rank, 1),
                    "subquery": subquery,
                    "matched_subject": bool(detail.get("subject_hit")),
                    "matched_object": bool(detail.get("object_hit")),
                    "relation_hits": detail.get("relation_hits", []),
                    "slot": {
                        "subject": slot.get("subject", ""),
                        "relation": slot.get("relation", ""),
                        "object": slot.get("object", ""),
                        "plan_id": slot.get("plan_id", ""),
                        "fact_position": slot.get("fact_position", 0),
                        "covered_slot_probe": bool(slot.get("covered_slot_probe", False)),
                    },
                    "contexts": [{
                        "missing_evidence": True,
                        "slot_alternative_support": bool(slot.get("covered_slot_probe", False)),
                        "support_key": f"missing_evidence::{slot.get('plan_id', '')}::{slot.get('fact_position', 0)}::{passage_idx}",
                        "plan_id": slot.get("plan_id", ""),
                        "anchor_entity": slot.get("anchor_entity") or slot.get("subject", ""),
                        "end_entity": slot.get("end_entity") or slot.get("object", ""),
                        "relations": slot.get("relations", []) or [slot.get("relation", "")],
                        "fact_indices": slot.get("fact_indices", []),
                        "fact_position": slot.get("fact_position", 0),
                        "signature_match_mode": slot.get("signature_match_mode", ""),
                        "plan_effective_precision": slot.get("plan_effective_precision", 0.0),
                        "plan_lift_weight_scale": slot.get("plan_lift_weight_scale", 1.0),
                        "score": confidence,
                    }],
                }
                if existing is None or candidate["score"] > existing["score"]:
                    selected_by_passage[passage_idx] = candidate

        candidates = sorted(
            selected_by_passage.values(),
            key=lambda candidate: (
                float(candidate["score"]),
                -int(candidate["dense_rank"]),
                -int(candidate["original_rank"]),
            ),
            reverse=True,
        )
        selected = candidates[: self.max_candidates]
        log["candidate_count"] = len(candidates)
        log["selected_count"] = len(selected)
        log["added_candidates"] = selected
        log["rejection_reasons"] = dict(sorted(rejection_reasons.items(), key=lambda kv: kv[1], reverse=True)[:8])
        if not selected:
            log["skip_reason"] = "no_verified_candidates"
            return QueryPlanMissingEvidenceResult(sorted_doc_ids, sorted_doc_scores, log)
        if self.candidate_only:
            log["skip_reason"] = "candidate_only"
            return QueryPlanMissingEvidenceResult(sorted_doc_ids, sorted_doc_scores, log)

        selected_ids = {int(item["passage_idx"]) for item in selected}
        preserved_ids = [int(doc_id) for doc_id in sorted_doc_ids[: self.preserve_top_k]]
        rest = [int(doc_id) for doc_id in sorted_doc_ids if int(doc_id) not in selected_ids and int(doc_id) not in preserved_ids]
        insert_at = min(max(self.promote_to_rank - 1, self.preserve_top_k), len(preserved_ids) + len(rest))
        selected_order = [int(item["passage_idx"]) for item in selected]
        reordered = preserved_ids + rest
        reordered = reordered[:insert_at] + selected_order + reordered[insert_at:]
        seen = set()
        final_ids = []
        for doc_id in reordered:
            if doc_id in seen:
                continue
            final_ids.append(doc_id)
            seen.add(doc_id)
        high = float(np.max(sorted_doc_scores)) if sorted_doc_scores.size else 1.0
        step = max(abs(high), 1.0) * self.score_margin
        final_scores = np.array([high - step * i for i in range(len(final_ids))], dtype=np.float32)
        final_rank = {int(doc_id): rank + 1 for rank, doc_id in enumerate(final_ids)}
        for item in selected:
            item["final_rank"] = int(final_rank.get(int(item["passage_idx"]), 0))
            item["rank_delta"] = int(item.get("original_rank", 0)) - int(item.get("final_rank", 0))
        log["changed_top5"] = list(sorted_doc_ids[:5]) != final_ids[:5]
        log["changed_top2"] = list(sorted_doc_ids[:2]) != final_ids[:2]
        return QueryPlanMissingEvidenceResult(np.array(final_ids, dtype=np.intp), final_scores, log)
