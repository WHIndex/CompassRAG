from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from .query_ontology_prior import tokenize
from .query_plan_structural_candidate_retriever import QueryPlanOntologyStructuralCandidateRetriever
from .utils.misc_utils import text_processing


@dataclass
class QueryPlanTerminalEvidenceResult:
    sorted_doc_ids: np.ndarray
    sorted_doc_scores: np.ndarray
    log: Dict[str, Any] = field(default_factory=dict)


class QueryPlanTerminalEvidenceRetriever:
    """Retrieve answer-terminal passages from ontology-derived subqueries.

    Query-plan paths often identify the bridge page but leave the final
    answer-bearing page outside the top evidence window. This module converts
    the current query-plan state into short dense subqueries and admits only
    passages that match concrete plan entities plus relation clues.
    """

    RELATION_HINTS = {
        "date": ("date", "year", "when"),
        "birth": ("born", "birth", "birthplace", "place of birth"),
        "death": ("died", "death", "date of death", "place of death"),
        "location": ("where", "place", "city", "country", "county", "location"),
        "nationality": ("nationality", "citizenship", "country"),
        "creator": ("director", "writer", "author", "composer", "producer"),
        "release": ("released", "release", "published", "publication"),
        "family": ("father", "mother", "parent", "spouse", "husband", "wife"),
        "education": ("university", "college", "school", "educated"),
        "measurement": ("area", "population", "height", "length", "number"),
    }

    def __init__(
        self,
        dense_top_k: int = 40,
        max_query_variants: int = 4,
        max_selected: int = 1,
        promote_to_rank: int = 5,
        preserve_top_k: int = 2,
        min_confidence: float = 0.58,
        require_relation_hint: bool = True,
        require_entity_hint: bool = True,
        max_entity_terms: int = 6,
        score_margin: float = 1e-6,
        candidate_only: bool = False,
    ):
        self.dense_top_k = max(int(dense_top_k), 1)
        self.max_query_variants = max(int(max_query_variants), 1)
        self.max_selected = max(int(max_selected), 0)
        self.promote_to_rank = max(int(promote_to_rank), 1)
        self.preserve_top_k = max(int(preserve_top_k), 0)
        self.min_confidence = min(max(float(min_confidence), 0.0), 1.0)
        self.require_relation_hint = bool(require_relation_hint)
        self.require_entity_hint = bool(require_entity_hint)
        self.max_entity_terms = max(int(max_entity_terms), 1)
        self.score_margin = max(float(score_margin), 0.0)
        self.candidate_only = bool(candidate_only)

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _contains_phrase(text: str, phrase: str) -> bool:
        text_norm = f" {text_processing(text)} "
        phrase_norm = text_processing(phrase)
        if not phrase_norm:
            return False
        if f" {phrase_norm} " in text_norm:
            return True
        phrase_tokens = tokenize(phrase_norm)
        if not phrase_tokens:
            return False
        text_tokens = set(tokenize(text_norm))
        matched = sum(1 for token in phrase_tokens if token in text_tokens)
        return matched / max(len(phrase_tokens), 1) >= 0.75

    @classmethod
    def _relation_tokens_from_query(cls, query: str, query_condition: Mapping[str, Any]) -> List[str]:
        values: List[str] = []
        query_norm = text_processing(query)
        answer_type = text_processing(query_condition.get("answer_type", ""))
        values.extend(str(item) for item in query_condition.get("relation_terms", []) or [])
        values.extend(str(item).replace("_", " ") for item in query_condition.get("relation_signatures", []) or [])
        values.extend(str(item) for item in query_condition.get("relation_families", []) or [])
        for key, hints in cls.RELATION_HINTS.items():
            if key in query_norm or key in answer_type:
                values.extend(hints)
        if "when" in query_norm or answer_type == "date":
            values.extend(cls.RELATION_HINTS["date"])
        if "where" in query_norm or answer_type in {"location", "place"}:
            values.extend(cls.RELATION_HINTS["location"])
        if "place of birth" in query_norm or "born" in query_norm:
            values.extend(cls.RELATION_HINTS["birth"])
        if "date of death" in query_norm or "place of death" in query_norm or " died" in f" {query_norm} ":
            values.extend(cls.RELATION_HINTS["death"])
        unique = []
        seen = set()
        for value in values:
            for token in tokenize(str(value).replace("_", " ")):
                if token and token not in seen:
                    unique.append(token)
                    seen.add(token)
        return unique

    @staticmethod
    def _relation_phrase_from_tokens(tokens_: Sequence[str]) -> str:
        preferred = [
            token
            for token in tokens_
            if token not in {"what", "which", "who", "where", "when", "date", "place"}
        ]
        return " ".join((preferred or list(tokens_))[:6])

    @staticmethod
    def _concrete_entities(values: Iterable[Any]) -> List[str]:
        entities = []
        seen = set()
        for value in values:
            text = str(value or "").strip()
            key = text_processing(text)
            if not key or key in seen:
                continue
            if QueryPlanOntologyStructuralCandidateRetriever.is_generic_anchor(key):
                continue
            entities.append(text)
            seen.add(key)
        return entities

    def _collect_entities(self, query_plan_log: Mapping[str, Any]) -> Tuple[List[str], List[str]]:
        anchors = self._concrete_entities(query_plan_log.get("anchor_entities", []) or [])
        terminals = []
        for path in query_plan_log.get("sample_paths", []) or []:
            terminals.append(path.get("end_entity") or path.get("entity"))
        for passage in query_plan_log.get("lifted_passages", []) or []:
            for context in passage.get("contexts", []) or []:
                terminals.append(context.get("end_entity"))
                terminals.append(context.get("anchor_entity"))
        terminals = self._concrete_entities(terminals)
        return anchors[: self.max_entity_terms], terminals[: self.max_entity_terms]

    def _build_subqueries(self, query: str, query_plan_log: Mapping[str, Any]) -> Tuple[List[str], Dict[str, Any]]:
        query_condition = query_plan_log.get("query_condition", {}) or {}
        relation_tokens = self._relation_tokens_from_query(query, query_condition)
        relation_phrase = self._relation_phrase_from_tokens(relation_tokens)
        anchors, terminals = self._collect_entities(query_plan_log)
        entities = terminals or anchors

        subqueries = []
        for entity in entities:
            if relation_phrase:
                subqueries.append(f"{entity} {relation_phrase}")
            subqueries.append(f"{query} {entity}")
        if anchors and terminals and relation_phrase:
            for anchor in anchors[:2]:
                for terminal in terminals[:2]:
                    subqueries.append(f"{anchor} {terminal} {relation_phrase}")

        unique = []
        seen = set()
        for subquery in subqueries:
            key = text_processing(subquery)
            if key and key not in seen:
                unique.append(subquery)
                seen.add(key)
            if len(unique) >= self.max_query_variants:
                break
        debug = {
            "anchors": anchors,
            "terminals": terminals,
            "relation_tokens": relation_tokens,
            "relation_phrase": relation_phrase,
        }
        return unique, debug

    def _score_candidate(
        self,
        passage_text: str,
        dense_score: float,
        max_dense_score: float,
        anchors: Sequence[str],
        terminals: Sequence[str],
        relation_tokens: Sequence[str],
    ) -> Tuple[float, Dict[str, Any]]:
        passage_norm = text_processing(passage_text)
        matched_terminals = [entity for entity in terminals if self._contains_phrase(passage_norm, entity)]
        matched_anchors = [entity for entity in anchors if self._contains_phrase(passage_norm, entity)]
        passage_tokens = set(tokenize(passage_norm))
        relation_hits = [token for token in relation_tokens if token in passage_tokens]

        entity_score = 0.0
        if matched_terminals:
            entity_score += 0.55
        if matched_anchors:
            entity_score += 0.25
        relation_score = min(len(relation_hits) / max(min(len(relation_tokens), 4), 1), 1.0)
        dense_norm = 0.0 if max_dense_score <= 0.0 else dense_score / max_dense_score
        dense_norm = min(max(dense_norm, 0.0), 1.0)
        confidence = 0.42 * dense_norm + 0.36 * min(entity_score, 1.0) + 0.22 * relation_score
        detail = {
            "matched_terminals": matched_terminals[:4],
            "matched_anchors": matched_anchors[:4],
            "relation_hits": relation_hits[:8],
            "dense_norm": dense_norm,
            "relation_score": relation_score,
            "entity_score": min(entity_score, 1.0),
        }
        return min(max(confidence, 0.0), 1.0), detail

    def rerank(
        self,
        query: str,
        sorted_doc_ids: np.ndarray,
        sorted_doc_scores: np.ndarray,
        query_plan_log: Mapping[str, Any],
        dense_retrieve: Callable[[str], Tuple[np.ndarray, np.ndarray]],
        get_passage_text: Callable[[int], str],
    ) -> QueryPlanTerminalEvidenceResult:
        log: Dict[str, Any] = {
            "enabled": True,
            "dense_top_k": self.dense_top_k,
            "max_query_variants": self.max_query_variants,
            "max_selected": self.max_selected,
            "promote_to_rank": self.promote_to_rank,
            "preserve_top_k": self.preserve_top_k,
            "min_confidence": self.min_confidence,
            "require_relation_hint": self.require_relation_hint,
            "require_entity_hint": self.require_entity_hint,
            "candidate_only": self.candidate_only,
        }
        sorted_doc_ids = np.array(sorted_doc_ids, dtype=np.intp)
        sorted_doc_scores = np.array(sorted_doc_scores, dtype=np.float32)
        if self.max_selected <= 0:
            log["skip_reason"] = "zero_max_selected"
            return QueryPlanTerminalEvidenceResult(sorted_doc_ids, sorted_doc_scores, log)
        if not query_plan_log or not query_plan_log.get("enabled", False):
            log["skip_reason"] = "no_query_plan_log"
            return QueryPlanTerminalEvidenceResult(sorted_doc_ids, sorted_doc_scores, log)

        subqueries, debug = self._build_subqueries(query, query_plan_log)
        log.update(debug)
        log["subqueries"] = subqueries
        if not subqueries:
            log["skip_reason"] = "no_terminal_subqueries"
            return QueryPlanTerminalEvidenceResult(sorted_doc_ids, sorted_doc_scores, log)
        anchors = debug.get("anchors", [])
        terminals = debug.get("terminals", [])
        relation_tokens = debug.get("relation_tokens", [])
        if self.require_entity_hint and not anchors and not terminals:
            log["skip_reason"] = "no_entity_hints"
            return QueryPlanTerminalEvidenceResult(sorted_doc_ids, sorted_doc_scores, log)
        if self.require_relation_hint and not relation_tokens:
            log["skip_reason"] = "no_relation_hints"
            return QueryPlanTerminalEvidenceResult(sorted_doc_ids, sorted_doc_scores, log)

        original_rank_by_doc = {int(doc_id): rank + 1 for rank, doc_id in enumerate(sorted_doc_ids)}
        preserved = {int(doc_id) for doc_id in sorted_doc_ids[: self.preserve_top_k]}
        candidates: Dict[int, Dict[str, Any]] = {}
        rejection_reasons = {
            "preserved": 0,
            "low_confidence": 0,
            "missing_entity": 0,
            "missing_relation": 0,
        }
        for subquery in subqueries:
            dense_doc_ids, dense_scores = dense_retrieve(subquery)
            dense_doc_ids = list(dense_doc_ids[: self.dense_top_k])
            dense_scores = list(dense_scores[: self.dense_top_k])
            max_dense_score = max((self._safe_float(score) for score in dense_scores), default=0.0)
            for doc_id, dense_score in zip(dense_doc_ids, dense_scores):
                doc_id = int(doc_id)
                if doc_id in preserved:
                    rejection_reasons["preserved"] += 1
                    continue
                passage_text = get_passage_text(doc_id)
                confidence, detail = self._score_candidate(
                    passage_text=passage_text,
                    dense_score=self._safe_float(dense_score),
                    max_dense_score=max_dense_score,
                    anchors=anchors,
                    terminals=terminals,
                    relation_tokens=relation_tokens,
                )
                if self.require_entity_hint and not detail["matched_terminals"] and not detail["matched_anchors"]:
                    rejection_reasons["missing_entity"] += 1
                    continue
                if self.require_relation_hint and not detail["relation_hits"]:
                    rejection_reasons["missing_relation"] += 1
                    continue
                if confidence < self.min_confidence:
                    rejection_reasons["low_confidence"] += 1
                    continue
                prior = candidates.get(doc_id)
                item = {
                    "passage_idx": doc_id,
                    "confidence": confidence,
                    "dense_score": self._safe_float(dense_score),
                    "original_rank": int(original_rank_by_doc.get(doc_id, len(sorted_doc_ids) + 1)),
                    "subquery": subquery,
                    **detail,
                }
                if prior is None or confidence > prior["confidence"]:
                    candidates[doc_id] = item

        selected = sorted(
            candidates.values(),
            key=lambda item: (item["confidence"], item["dense_score"], -item["original_rank"]),
            reverse=True,
        )[: self.max_selected]
        log["candidate_count"] = len(candidates)
        log["selected_count"] = len(selected)
        log["rejection_reasons"] = rejection_reasons
        log["selected_passages"] = selected
        if not selected:
            log["skip_reason"] = "no_selected_terminal_evidence"
            return QueryPlanTerminalEvidenceResult(sorted_doc_ids, sorted_doc_scores, log)
        if self.candidate_only:
            log["skip_reason"] = "candidate_only_no_direct_insertion"
            return QueryPlanTerminalEvidenceResult(sorted_doc_ids, sorted_doc_scores, log)

        selected_ids = [int(item["passage_idx"]) for item in selected]
        selected_set = set(selected_ids)
        existing_scores = {int(doc_id): float(score) for doc_id, score in zip(sorted_doc_ids, sorted_doc_scores)}
        sorted_without_selected = [int(doc_id) for doc_id in sorted_doc_ids if int(doc_id) not in selected_set]
        insert_at = min(max(self.promote_to_rank - 1, self.preserve_top_k), len(sorted_without_selected))
        reordered = sorted_without_selected[:insert_at] + selected_ids + sorted_without_selected[insert_at:]
        base_scores = np.array([existing_scores.get(int(doc_id), 0.0) for doc_id in reordered], dtype=np.float32)
        for offset, item in enumerate(selected):
            idx = insert_at + offset
            if idx >= len(base_scores):
                continue
            reference_idx = min(max(self.promote_to_rank - 1, 0), len(base_scores) - 1)
            base_scores[idx] = max(
                base_scores[idx],
                base_scores[reference_idx] + self.score_margin * (len(selected) - offset),
            )
            item["final_rank"] = idx + 1
            item["rank_delta"] = int(item.get("original_rank", idx + 1)) - (idx + 1)
        return QueryPlanTerminalEvidenceResult(np.array(reordered, dtype=np.intp), base_scores, log)
