from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from .query_ontology_prior import tokenize
from .utils.misc_utils import text_processing


@dataclass
class QueryPlanTerminalEvidenceVerifierResult:
    sorted_doc_ids: np.ndarray
    sorted_doc_scores: np.ndarray
    log: Dict[str, Any] = field(default_factory=dict)


class QueryPlanTerminalEvidenceVerifier:
    """Verify terminal-evidence candidates before they can alter retrieval order.

    The terminal-evidence retriever is intentionally recall-oriented: it launches
    dense subqueries from ontology terminals and relation hints. This verifier is
    the precision gate. It only admits a candidate when the ontology evidence is
    concrete enough and the candidate adds coverage that the current top window
    does not already contain.
    """

    BLOCKED_QUERY_TYPES = {
        "comparative",
        "discourse_or_long_answer",
        "non_extractive_answer",
        "explanatory_or_measure",
        "long_compositional_factoid",
    }

    GENERIC_TERMS = {
        "what",
        "which",
        "who",
        "where",
        "when",
        "how",
        "name",
        "named",
        "person",
        "people",
        "place",
        "location",
        "city",
        "country",
        "state",
        "film",
        "movie",
        "book",
        "album",
        "song",
        "novel",
        "series",
        "company",
        "organization",
    }

    def __init__(
        self,
        promote_to_rank: int = 5,
        preserve_top_k: int = 2,
        max_inserted: int = 1,
        min_confidence: float = 0.74,
        min_relation_hits: int = 1,
        max_original_rank: int = 120,
        max_matched_plans: int = 3,
        max_executable_paths: int = 6,
        require_new_coverage: bool = True,
        score_margin: float = 1e-6,
    ):
        self.promote_to_rank = max(int(promote_to_rank), 1)
        self.preserve_top_k = max(int(preserve_top_k), 0)
        self.max_inserted = max(int(max_inserted), 0)
        self.min_confidence = min(max(float(min_confidence), 0.0), 1.0)
        self.min_relation_hits = max(int(min_relation_hits), 0)
        self.max_original_rank = max(int(max_original_rank), 0)
        self.max_matched_plans = max(int(max_matched_plans), 0)
        self.max_executable_paths = max(int(max_executable_paths), 0)
        self.require_new_coverage = bool(require_new_coverage)
        self.score_margin = max(float(score_margin), 0.0)

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _safe_int(value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _as_text_list(values: Any) -> List[str]:
        if not isinstance(values, (list, tuple, set)):
            return []
        result = []
        seen = set()
        for value in values:
            text = str(value or "").strip()
            key = text_processing(text)
            if not key or key in seen:
                continue
            result.append(text)
            seen.add(key)
        return result

    @classmethod
    def _content_terms(cls, item: Mapping[str, Any]) -> List[str]:
        terms: List[str] = []
        terms.extend(cls._as_text_list(item.get("matched_terminals", [])))
        terms.extend(cls._as_text_list(item.get("matched_anchors", [])))
        terms.extend(cls._as_text_list(item.get("relation_hits", [])))
        unique = []
        seen = set()
        for term in terms:
            key = text_processing(term)
            if not key or key in seen:
                continue
            if key in cls.GENERIC_TERMS:
                continue
            unique.append(term)
            seen.add(key)
        return unique

    @staticmethod
    def _contains_term(text: str, term: str) -> bool:
        text_norm = f" {text_processing(text)} "
        term_norm = text_processing(term)
        if not term_norm:
            return False
        if f" {term_norm} " in text_norm:
            return True
        term_tokens = tokenize(term_norm)
        if not term_tokens:
            return False
        text_tokens = set(tokenize(text_norm))
        matched = sum(1 for token in term_tokens if token in text_tokens)
        return matched / max(len(term_tokens), 1) >= 0.75

    def _adds_new_coverage(
        self,
        item: Mapping[str, Any],
        top_window_doc_ids: Sequence[int],
        get_passage_text: Callable[[int], str],
    ) -> Tuple[bool, List[str], List[str]]:
        candidate_terms = self._content_terms(item)
        if not candidate_terms:
            return False, [], []
        top_text = " ".join(get_passage_text(int(doc_id)) for doc_id in top_window_doc_ids)
        new_terms = [term for term in candidate_terms if not self._contains_term(top_text, term)]
        seen_terms = [term for term in candidate_terms if term not in new_terms]
        return bool(new_terms), new_terms[:8], seen_terms[:8]

    @staticmethod
    def _query_condition(query_plan_log: Mapping[str, Any]) -> Dict[str, Any]:
        condition = query_plan_log.get("query_condition", {}) or {}
        return condition if isinstance(condition, dict) else {}

    def _global_skip_reason(self, query_plan_log: Mapping[str, Any]) -> str:
        condition = self._query_condition(query_plan_log)
        query_type = str(condition.get("query_type", "") or "")
        if query_type in self.BLOCKED_QUERY_TYPES:
            return f"blocked_query_type_{query_type}"
        matched_plan_count = self._safe_int(query_plan_log.get("matched_plan_count", 0))
        executable_path_count = self._safe_int(query_plan_log.get("executable_path_count", 0))
        if self.max_matched_plans > 0 and matched_plan_count > self.max_matched_plans:
            return "too_many_matched_plans"
        if self.max_executable_paths > 0 and executable_path_count > self.max_executable_paths:
            return "too_many_executable_paths"
        return ""

    def verify_and_rerank(
        self,
        sorted_doc_ids: np.ndarray,
        sorted_doc_scores: np.ndarray,
        terminal_evidence_log: Mapping[str, Any],
        query_plan_log: Mapping[str, Any],
        get_passage_text: Callable[[int], str],
    ) -> QueryPlanTerminalEvidenceVerifierResult:
        sorted_doc_ids = np.array(sorted_doc_ids, dtype=np.intp)
        sorted_doc_scores = np.array(sorted_doc_scores, dtype=np.float32)
        log: Dict[str, Any] = {
            "enabled": True,
            "promote_to_rank": self.promote_to_rank,
            "preserve_top_k": self.preserve_top_k,
            "max_inserted": self.max_inserted,
            "min_confidence": self.min_confidence,
            "min_relation_hits": self.min_relation_hits,
            "max_original_rank": self.max_original_rank,
            "max_matched_plans": self.max_matched_plans,
            "max_executable_paths": self.max_executable_paths,
            "require_new_coverage": self.require_new_coverage,
        }
        if self.max_inserted <= 0:
            log["skip_reason"] = "zero_max_inserted"
            return QueryPlanTerminalEvidenceVerifierResult(sorted_doc_ids, sorted_doc_scores, log)
        if not terminal_evidence_log or not terminal_evidence_log.get("enabled", False):
            log["skip_reason"] = "no_terminal_evidence_log"
            return QueryPlanTerminalEvidenceVerifierResult(sorted_doc_ids, sorted_doc_scores, log)

        global_skip = self._global_skip_reason(query_plan_log)
        if global_skip:
            log["skip_reason"] = global_skip
            return QueryPlanTerminalEvidenceVerifierResult(sorted_doc_ids, sorted_doc_scores, log)

        candidates = [
            dict(item)
            for item in (terminal_evidence_log.get("selected_passages", []) or [])
            if isinstance(item, dict)
        ]
        log["candidate_count"] = len(candidates)
        if not candidates:
            log["skip_reason"] = "no_terminal_candidates"
            return QueryPlanTerminalEvidenceVerifierResult(sorted_doc_ids, sorted_doc_scores, log)

        current_top_ids = [int(doc_id) for doc_id in sorted_doc_ids[: self.promote_to_rank]]
        original_rank_by_doc = {int(doc_id): rank + 1 for rank, doc_id in enumerate(sorted_doc_ids)}
        rejection_reasons: Counter = Counter()
        accepted: List[Dict[str, Any]] = []
        for item in candidates:
            passage_idx = self._safe_int(item.get("passage_idx"), -1)
            if passage_idx < 0:
                rejection_reasons["missing_passage_idx"] += 1
                continue
            if passage_idx in current_top_ids:
                rejection_reasons["already_in_target_window"] += 1
                continue
            confidence = self._safe_float(item.get("confidence", 0.0))
            if confidence < self.min_confidence:
                rejection_reasons["low_confidence"] += 1
                continue
            original_rank = self._safe_int(
                item.get("original_rank"),
                original_rank_by_doc.get(passage_idx, len(sorted_doc_ids) + 1),
            )
            if self.max_original_rank > 0 and original_rank > self.max_original_rank:
                rejection_reasons["original_rank_after_window"] += 1
                continue
            relation_hits = self._as_text_list(item.get("relation_hits", []))
            if len(relation_hits) < self.min_relation_hits:
                rejection_reasons["insufficient_relation_hits"] += 1
                continue
            matched_entities = self._as_text_list(item.get("matched_terminals", []))
            matched_entities.extend(self._as_text_list(item.get("matched_anchors", [])))
            if not matched_entities:
                rejection_reasons["no_matched_entities"] += 1
                continue
            new_coverage_terms: List[str] = []
            seen_coverage_terms: List[str] = []
            if self.require_new_coverage:
                has_new_coverage, new_coverage_terms, seen_coverage_terms = self._adds_new_coverage(
                    item=item,
                    top_window_doc_ids=sorted_doc_ids[: self.promote_to_rank],
                    get_passage_text=get_passage_text,
                )
                if not has_new_coverage:
                    rejection_reasons["no_new_top_window_coverage"] += 1
                    continue
            accepted_item = dict(item)
            accepted_item["verifier"] = {
                "verified": True,
                "new_coverage_terms": new_coverage_terms,
                "seen_coverage_terms": seen_coverage_terms,
                "reasons": [
                    "confidence_above_threshold",
                    "relation_and_entity_matched",
                    "adds_top_window_coverage" if self.require_new_coverage else "coverage_gate_disabled",
                ],
            }
            accepted.append(accepted_item)

        accepted = sorted(
            accepted,
            key=lambda item: (
                self._safe_float(item.get("confidence")),
                len(item.get("relation_hits", []) or []),
                -self._safe_int(item.get("original_rank"), 10**9),
            ),
            reverse=True,
        )[: self.max_inserted]
        log["rejection_reasons"] = dict(rejection_reasons)
        log["verified_count"] = len(accepted)
        log["inserted_passages"] = []
        if not accepted:
            log["skip_reason"] = "no_verified_terminal_evidence"
            return QueryPlanTerminalEvidenceVerifierResult(sorted_doc_ids, sorted_doc_scores, log)

        selected_ids = [int(item["passage_idx"]) for item in accepted]
        selected_set = set(selected_ids)
        existing_scores = {int(doc_id): float(score) for doc_id, score in zip(sorted_doc_ids, sorted_doc_scores)}
        sorted_without_selected = [int(doc_id) for doc_id in sorted_doc_ids if int(doc_id) not in selected_set]
        insert_at = min(max(self.promote_to_rank - 1, self.preserve_top_k), len(sorted_without_selected))
        reordered = sorted_without_selected[:insert_at] + selected_ids + sorted_without_selected[insert_at:]
        base_scores = np.array([existing_scores.get(int(doc_id), 0.0) for doc_id in reordered], dtype=np.float32)
        for offset, item in enumerate(accepted):
            idx = insert_at + offset
            if idx >= len(base_scores):
                continue
            reference_idx = min(max(self.promote_to_rank - 1, 0), len(base_scores) - 1)
            base_scores[idx] = max(
                base_scores[idx],
                base_scores[reference_idx] + self.score_margin * (len(accepted) - offset),
            )
            item["final_rank"] = idx + 1
            item["rank_delta"] = self._safe_int(item.get("original_rank"), idx + 1) - (idx + 1)
        log["inserted_count"] = len(accepted)
        log["inserted_passages"] = accepted
        return QueryPlanTerminalEvidenceVerifierResult(np.array(reordered, dtype=np.intp), base_scores, log)
