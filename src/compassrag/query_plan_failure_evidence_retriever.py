import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from .query_ontology_prior import tokenize
from .query_plan_structural_candidate_retriever import QueryPlanOntologyStructuralCandidateRetriever
from .query_plan_support_evidence_retriever import QueryPlanSupportEvidenceRetriever
from .utils.misc_utils import text_processing


@dataclass
class QueryPlanFailureEvidenceResult:
    sorted_doc_ids: np.ndarray
    sorted_doc_scores: np.ndarray
    log: Dict[str, Any] = field(default_factory=dict)


class QueryPlanFailureEvidenceOntology:
    """Dataset-local rules induced from retrieval failures.

    The evidence-grounded ontology captures useful query/path schemas, but
    some missing passages never enter the ontology candidate pool. These rules are
    intentionally
    retrieval-facing. They describe which query terms, relation terms, and
    answer-type hints should be used to form additional dense probes.
    """

    def __init__(self, data: Dict[str, Any]):
        self.data = data or {}
        self.rules = [
            rule for rule in self.data.get("rules", [])
            if isinstance(rule, dict)
        ]

    @classmethod
    def load(cls, path: str) -> "QueryPlanFailureEvidenceOntology":
        with open(path, "r", encoding="utf-8") as handle:
            return cls(json.load(handle))


class QueryPlanFailureEvidenceRetriever:
    GENERIC_TERMS = QueryPlanSupportEvidenceRetriever.GENERIC_TERMS | {
        "did", "does", "do", "which", "what", "when", "where", "who",
        "whom", "whose", "why", "how", "also", "same", "called",
    }
    ANSWER_HINTS = QueryPlanSupportEvidenceRetriever.ANSWER_HINTS

    def __init__(
        self,
        ontology: QueryPlanFailureEvidenceOntology,
        dense_top_k: int = 120,
        max_rules: int = 8,
        max_probes: int = 16,
        max_added_passages: int = 8,
        min_confidence: float = 0.56,
        preserve_top_k: int = 2,
        synthetic_original_rank: int = 60,
        require_anchor_hit: bool = False,
        require_rule_term_hit: bool = True,
        candidate_only: bool = True,
        append_score_margin: float = 1e-7,
        min_rule_gold_precision: float = 0.0,
        min_rule_gt5_precision: float = 0.0,
        min_rule_gold_hits: int = 0,
        min_rule_gt5_hits: int = 0,
    ):
        self.ontology = ontology
        self.dense_top_k = max(int(dense_top_k), 1)
        self.max_rules = max(int(max_rules), 0)
        self.max_probes = max(int(max_probes), 0)
        self.max_added_passages = max(int(max_added_passages), 0)
        self.min_confidence = min(max(float(min_confidence), 0.0), 1.0)
        self.preserve_top_k = max(int(preserve_top_k), 0)
        self.synthetic_original_rank = max(int(synthetic_original_rank), 1)
        self.require_anchor_hit = bool(require_anchor_hit)
        self.require_rule_term_hit = bool(require_rule_term_hit)
        self.candidate_only = bool(candidate_only)
        self.append_score_margin = max(float(append_score_margin), 0.0)
        self.min_rule_gold_precision = min(max(float(min_rule_gold_precision), 0.0), 1.0)
        self.min_rule_gt5_precision = min(max(float(min_rule_gt5_precision), 0.0), 1.0)
        self.min_rule_gold_hits = max(int(min_rule_gold_hits), 0)
        self.min_rule_gt5_hits = max(int(min_rule_gt5_hits), 0)

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
    def _tokens(cls, value: Any, max_tokens: int = 8) -> List[str]:
        output: List[str] = []
        seen = set()
        for token in tokenize(text_processing(str(value or "").replace("_", " "))):
            if token in cls.GENERIC_TERMS or token in seen:
                continue
            output.append(token)
            seen.add(token)
            if len(output) >= max_tokens:
                break
        return output

    @classmethod
    def _unique(cls, values: Iterable[Any], max_items: int = 32) -> List[str]:
        output: List[str] = []
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
    def _answer_type(cls, query_plan_log: Mapping[str, Any]) -> str:
        query_condition = query_plan_log.get("query_condition", {}) or {}
        return text_processing(str(query_condition.get("answer_type") or "entity"))

    @classmethod
    def _query_terms(cls, query: str, query_plan_log: Mapping[str, Any]) -> List[str]:
        query_condition = query_plan_log.get("query_condition", {}) or {}
        terms: List[str] = []
        terms.extend(query_condition.get("query_tokens", []) or [])
        terms.extend(cls._tokens(query, max_tokens=12))
        return cls._unique(terms, max_items=16)

    @classmethod
    def _relation_terms(cls, query_plan_log: Mapping[str, Any]) -> List[str]:
        query_condition = query_plan_log.get("query_condition", {}) or {}
        terms: List[str] = []
        terms.extend(query_condition.get("relation_terms", []) or [])
        terms.extend(query_condition.get("relation_signatures", []) or [])
        terms.extend(query_condition.get("relation_families", []) or [])
        for plan in query_plan_log.get("matched_plans", []) or []:
            if not isinstance(plan, Mapping):
                continue
            terms.extend(plan.get("relation_signatures", []) or [])
            terms.extend(plan.get("relation_families", []) or [])
            terms.extend(plan.get("triggers", []) or [])
        cleaned = []
        for term in cls._unique(terms, max_items=24):
            tokens = cls._tokens(term, max_tokens=5)
            if tokens:
                cleaned.append(" ".join(tokens))
        return cls._unique(cleaned, max_items=16)

    @classmethod
    def _anchors(cls, query_plan_log: Mapping[str, Any]) -> List[str]:
        values: List[str] = []
        values.extend(query_plan_log.get("anchor_entities", []) or [])
        for path in query_plan_log.get("sample_paths", []) or []:
            if isinstance(path, Mapping):
                values.append(path.get("anchor_entity", ""))
        for passage in query_plan_log.get("lifted_passages", []) or []:
            for context in passage.get("contexts", []) or []:
                if isinstance(context, Mapping):
                    values.append(context.get("anchor_entity", ""))
        anchors = []
        for anchor in cls._unique(values, max_items=16):
            if not QueryPlanOntologyStructuralCandidateRetriever.is_generic_anchor(anchor):
                anchors.append(anchor)
        return anchors

    def _score_rule(self, rule: Mapping[str, Any], query: str, query_plan_log: Mapping[str, Any]) -> Tuple[float, Dict[str, Any]]:
        answer_type = self._answer_type(query_plan_log)
        query_terms = set(self._query_terms(query, query_plan_log))
        relation_terms = set(
            token for term in self._relation_terms(query_plan_log)
            for token in self._tokens(term, max_tokens=6)
        )
        rule_query_terms = set(
            token for term in rule.get("query_terms", []) or []
            for token in self._tokens(term, max_tokens=6)
        )
        rule_relation_terms = set(
            token for term in rule.get("relation_terms", []) or []
            for token in self._tokens(term, max_tokens=6)
        )
        answer_bonus = 0.20 if str(rule.get("answer_type", "")) == answer_type else 0.0
        query_overlap = query_terms & rule_query_terms
        relation_overlap = relation_terms & rule_relation_terms
        query_score = len(query_overlap) / max(min(len(query_terms), len(rule_query_terms)), 1)
        relation_score = len(relation_overlap) / max(min(len(relation_terms), len(rule_relation_terms)), 1)
        no_candidate_rate = self._safe_float(rule.get("no_candidate_gold_rate"), 0.0)
        no_oracle_rate = self._safe_float(rule.get("no_oracle_gain_rate"), 0.0)
        support = self._safe_float(rule.get("failure_count"), 0.0)
        validation_gold_precision = self._safe_float(
            rule.get("validation_gold_doc_precision"),
            self._safe_float((rule.get("validation", {}) or {}).get("gold_doc_precision"), 0.0),
        )
        validation_gt5_precision = self._safe_float(
            rule.get("validation_gold_from_gt5_precision"),
            self._safe_float((rule.get("validation", {}) or {}).get("gold_from_gt5_precision"), 0.0),
        )
        validation_gt5_hits = self._safe_float(
            rule.get("validation_gold_from_gt5_hits"),
            self._safe_float((rule.get("validation", {}) or {}).get("gold_from_gt5_hits"), 0.0),
        )
        score = (
            answer_bonus
            + 0.34 * query_score
            + 0.30 * relation_score
            + 0.08 * min(support / 60.0, 1.0)
            + 0.04 * no_candidate_rate
            + 0.04 * no_oracle_rate
            + 0.12 * min(validation_gt5_precision / 0.08, 1.0)
            + 0.06 * min(validation_gold_precision / 0.12, 1.0)
            + 0.04 * min(validation_gt5_hits / 4.0, 1.0)
        )
        detail = {
            "answer_type": answer_type,
            "query_overlap": sorted(query_overlap),
            "relation_overlap": sorted(relation_overlap),
            "query_score": query_score,
            "relation_score": relation_score,
            "validation_gold_doc_precision": validation_gold_precision,
            "validation_gold_from_gt5_precision": validation_gt5_precision,
            "validation_gold_from_gt5_hits": validation_gt5_hits,
        }
        return score, detail

    def _rule_passes_validation(self, rule: Mapping[str, Any]) -> bool:
        validation = rule.get("validation", {}) or {}
        gold_precision = self._safe_float(
            rule.get("validation_gold_doc_precision"),
            self._safe_float(validation.get("gold_doc_precision"), 0.0),
        )
        gt5_precision = self._safe_float(
            rule.get("validation_gold_from_gt5_precision"),
            self._safe_float(validation.get("gold_from_gt5_precision"), 0.0),
        )
        gold_hits = int(self._safe_float(
            rule.get("validation_gold_doc_hits"),
            self._safe_float(validation.get("gold_doc_hits"), 0.0),
        ))
        gt5_hits = int(self._safe_float(
            rule.get("validation_gold_from_gt5_hits"),
            self._safe_float(validation.get("gold_from_gt5_hits"), 0.0),
        ))
        if gold_precision < self.min_rule_gold_precision:
            return False
        if gt5_precision < self.min_rule_gt5_precision:
            return False
        if gold_hits < self.min_rule_gold_hits:
            return False
        if gt5_hits < self.min_rule_gt5_hits:
            return False
        return True

    def _matched_rules(self, query: str, query_plan_log: Mapping[str, Any]) -> List[Dict[str, Any]]:
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for rule in self.ontology.rules:
            if not self._rule_passes_validation(rule):
                continue
            score, detail = self._score_rule(rule, query, query_plan_log)
            min_score = self._safe_float(rule.get("min_route_score"), 0.22)
            if score < min_score:
                continue
            matched = dict(rule)
            matched["route_score"] = float(score)
            matched["route_detail"] = detail
            scored.append((score, matched))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [rule for _, rule in scored[: self.max_rules]]

    def _answer_hints(self, query_plan_log: Mapping[str, Any], rule: Mapping[str, Any]) -> List[str]:
        hints: List[str] = []
        query_condition = query_plan_log.get("query_condition", {}) or {}
        answer_type = self._answer_type(query_plan_log)
        hints.extend(query_condition.get("answer_hints", []) or [])
        hints.extend(rule.get("answer_hints", []) or [])
        hints.extend(self.ANSWER_HINTS.get(answer_type, []))
        if answer_type:
            hints.append(answer_type)
        return self._unique(hints, max_items=8)

    def _build_probes(
        self,
        query: str,
        query_plan_log: Mapping[str, Any],
        matched_rules: Sequence[Mapping[str, Any]],
    ) -> List[Dict[str, Any]]:
        anchors = self._anchors(query_plan_log)
        query_terms = self._query_terms(query, query_plan_log)
        runtime_relations = self._relation_terms(query_plan_log)
        probes: List[Dict[str, Any]] = []
        seen = set()

        def add_probe(text: str, rule: Mapping[str, Any], kind: str, anchor: str = "", relation: str = "") -> None:
            key = self._norm(text)
            if not key or key in seen:
                return
            seen.add(key)
            probes.append({
                "query": text,
                "kind": kind,
                "rule_id": rule.get("id", ""),
                "anchor": anchor,
                "relation": relation,
                "answer_hints": self._answer_hints(query_plan_log, rule),
                "route_score": self._safe_float(rule.get("route_score"), 0.0),
                "rule_support": int(rule.get("failure_count", 0) or 0),
                "rule_precision_prior": self._safe_float(rule.get("precision_prior"), 0.0),
                "rule": {
                    "answer_type": rule.get("answer_type", ""),
                    "relation_key": rule.get("relation_key", ""),
                    "query_terms": list(rule.get("query_terms", []) or [])[:8],
                    "relation_terms": list(rule.get("relation_terms", []) or [])[:8],
                    "no_candidate_gold_rate": rule.get("no_candidate_gold_rate", 0.0),
                    "no_oracle_gain_rate": rule.get("no_oracle_gain_rate", 0.0),
                    "validation_gold_doc_precision": rule.get("validation_gold_doc_precision", 0.0),
                    "validation_gold_from_gt5_precision": rule.get("validation_gold_from_gt5_precision", 0.0),
                    "validation_gold_doc_hits": rule.get("validation_gold_doc_hits", 0),
                    "validation_gold_from_gt5_hits": rule.get("validation_gold_from_gt5_hits", 0),
                },
            })

        for rule in matched_rules:
            rule_relations = self._unique(
                list(rule.get("relation_terms", []) or []) + runtime_relations,
                max_items=8,
            )
            rule_query_terms = self._unique(
                list(rule.get("query_terms", []) or []) + query_terms,
                max_items=10,
            )
            hints = self._answer_hints(query_plan_log, rule)
            hint_text = " ".join(hints[:3])
            relation_text = " ".join(rule_relations[:3])
            query_term_text = " ".join(rule_query_terms[:5])
            add_probe(
                " ".join(piece for piece in [query, relation_text, hint_text] if piece),
                rule,
                "query_relation_answer",
                "",
                relation_text,
            )
            add_probe(
                " ".join(piece for piece in [query_term_text, relation_text, hint_text] if piece),
                rule,
                "failure_signature",
                "",
                relation_text,
            )
            for anchor in anchors[:4]:
                add_probe(
                    " ".join(piece for piece in [anchor, relation_text, hint_text] if piece),
                    rule,
                    "anchor_failure_relation",
                    anchor,
                    relation_text,
                )
                add_probe(
                    " ".join(piece for piece in [query, anchor, relation_text] if piece),
                    rule,
                    "query_anchor_failure_relation",
                    anchor,
                    relation_text,
                )
        return probes[: self.max_probes]

    def _answer_type_hit(self, text_norm: str, hints: Sequence[str]) -> bool:
        tokens = set(tokenize(text_norm))
        hint_tokens = {token for hint in hints for token in self._tokens(hint, max_tokens=3)}
        if hint_tokens and tokens.intersection(hint_tokens):
            return True
        if any(hint in {"date", "year", "month", "day"} for hint in hints):
            if any(token.isdigit() and len(token) == 4 for token in tokens):
                return True
        if any(hint in {"number", "count", "total", "age"} for hint in hints):
            if any(token.isdigit() for token in tokens):
                return True
        return False

    def _score_candidate(
        self,
        passage_text: str,
        dense_score: float,
        max_dense_score: float,
        probe: Mapping[str, Any],
    ) -> Tuple[float, Dict[str, Any]]:
        text_norm = self._norm(passage_text)
        text_tokens = set(tokenize(text_norm))
        anchor_hit = self._contains_phrase(text_norm, probe.get("anchor", ""))
        relation_tokens = self._tokens(probe.get("relation", ""), max_tokens=10)
        rule_payload = probe.get("rule", {}) or {}
        rule_tokens = {
            token
            for term in (list(rule_payload.get("query_terms", []) or []) + list(rule_payload.get("relation_terms", []) or []))
            for token in self._tokens(term, max_tokens=5)
        }
        relation_hits = [token for token in relation_tokens if token in text_tokens]
        rule_hits = sorted(token for token in rule_tokens if token in text_tokens)
        answer_hit = self._answer_type_hit(text_norm, probe.get("answer_hints", []) or [])
        dense_norm = 0.0 if max_dense_score <= 0.0 else float(dense_score) / max_dense_score
        dense_norm = min(max(dense_norm, 0.0), 1.0)
        route_score = min(max(self._safe_float(probe.get("route_score"), 0.0), 0.0), 1.0)
        relation_score = min(len(relation_hits) / max(min(len(relation_tokens), 4), 1), 1.0)
        rule_score = min(len(rule_hits) / max(min(len(rule_tokens), 5), 1), 1.0)
        confidence = (
            0.36 * dense_norm
            + 0.20 * route_score
            + 0.18 * relation_score
            + 0.12 * rule_score
            + 0.08 * (1.0 if anchor_hit else 0.0)
            + 0.06 * (1.0 if answer_hit else 0.0)
        )
        return min(max(confidence, 0.0), 1.0), {
            "anchor_hit": anchor_hit,
            "relation_hits": relation_hits,
            "rule_hits": rule_hits[:10],
            "answer_hit": answer_hit,
            "dense_norm": dense_norm,
            "route_score": route_score,
            "confidence": min(max(confidence, 0.0), 1.0),
        }

    def retrieve(
        self,
        query: str,
        sorted_doc_ids: Sequence[int],
        sorted_doc_scores: Sequence[float],
        query_plan_log: Mapping[str, Any],
        dense_retrieve: Callable[[str], Tuple[np.ndarray, np.ndarray]],
        get_passage_text: Callable[[int], str],
    ) -> QueryPlanFailureEvidenceResult:
        base_doc_ids = [int(doc_id) for doc_id in list(sorted_doc_ids)]
        base_scores = np.array(sorted_doc_scores, dtype=np.float32)
        log: Dict[str, Any] = {
            "enabled": True,
            "dense_top_k": self.dense_top_k,
            "max_rules": self.max_rules,
            "max_probes": self.max_probes,
            "max_added_passages": self.max_added_passages,
            "min_confidence": self.min_confidence,
            "min_rule_gold_precision": self.min_rule_gold_precision,
            "min_rule_gt5_precision": self.min_rule_gt5_precision,
            "min_rule_gold_hits": self.min_rule_gold_hits,
            "min_rule_gt5_hits": self.min_rule_gt5_hits,
            "candidate_only": self.candidate_only,
            "matched_rules": [],
            "probes": [],
            "added_candidates": [],
            "rejection_reasons": {},
        }
        if self.max_rules <= 0 or self.max_probes <= 0 or self.max_added_passages <= 0:
            log["skip_reason"] = "disabled_by_zero_limit"
            return QueryPlanFailureEvidenceResult(np.array(base_doc_ids, dtype=np.intp), base_scores, log)
        if not base_doc_ids:
            log["skip_reason"] = "empty_retrieval"
            return QueryPlanFailureEvidenceResult(np.array(base_doc_ids, dtype=np.intp), base_scores, log)
        if not query_plan_log or not query_plan_log.get("enabled", False):
            log["skip_reason"] = "no_query_plan_log"
            return QueryPlanFailureEvidenceResult(np.array(base_doc_ids, dtype=np.intp), base_scores, log)
        matched_rules = self._matched_rules(query, query_plan_log)
        if not matched_rules:
            log["skip_reason"] = "no_matched_failure_rules"
            return QueryPlanFailureEvidenceResult(np.array(base_doc_ids, dtype=np.intp), base_scores, log)
        probes = self._build_probes(query, query_plan_log, matched_rules)
        log["matched_rules"] = [
            {
                "id": rule.get("id", ""),
                "route_score": rule.get("route_score", 0.0),
                "answer_type": rule.get("answer_type", ""),
                "relation_key": rule.get("relation_key", ""),
                "failure_count": rule.get("failure_count", 0),
                "validation_gold_doc_precision": rule.get("validation_gold_doc_precision", 0.0),
                "validation_gold_from_gt5_precision": rule.get("validation_gold_from_gt5_precision", 0.0),
                "validation_gold_doc_hits": rule.get("validation_gold_doc_hits", 0),
                "validation_gold_from_gt5_hits": rule.get("validation_gold_from_gt5_hits", 0),
                "route_detail": rule.get("route_detail", {}),
            }
            for rule in matched_rules
        ]
        log["probes"] = [
            {
                "query": probe.get("query", ""),
                "kind": probe.get("kind", ""),
                "rule_id": probe.get("rule_id", ""),
                "anchor": probe.get("anchor", ""),
                "relation": probe.get("relation", ""),
                "route_score": probe.get("route_score", 0.0),
            }
            for probe in probes
        ]
        if not probes:
            log["skip_reason"] = "no_failure_evidence_probes"
            return QueryPlanFailureEvidenceResult(np.array(base_doc_ids, dtype=np.intp), base_scores, log)

        original_rank_by_doc = {int(doc_id): rank + 1 for rank, doc_id in enumerate(base_doc_ids)}
        preserved = set(base_doc_ids[: self.preserve_top_k])
        selected_by_passage: Dict[int, Dict[str, Any]] = {}
        rejection_reasons: Dict[str, int] = {}

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
                    passage_text=passage_text,
                    dense_score=self._safe_float(dense_scores[dense_rank - 1] if dense_rank - 1 < len(dense_scores) else 0.0),
                    max_dense_score=max_dense_score,
                    probe=probe,
                )
                if self.require_anchor_hit and probe.get("anchor") and not detail["anchor_hit"]:
                    rejection_reasons["anchor_miss"] = rejection_reasons.get("anchor_miss", 0) + 1
                    continue
                has_rule_evidence = bool(detail["relation_hits"] or detail["rule_hits"] or detail["answer_hit"])
                if self.require_rule_term_hit and not has_rule_evidence:
                    rejection_reasons["rule_term_miss"] = rejection_reasons.get("rule_term_miss", 0) + 1
                    continue
                if confidence < self.min_confidence:
                    rejection_reasons["low_confidence"] = rejection_reasons.get("low_confidence", 0) + 1
                    continue
                original_rank = int(original_rank_by_doc.get(passage_idx, self.synthetic_original_rank))
                candidate = {
                    "passage_idx": passage_idx,
                    "confidence": float(confidence),
                    "dense_rank": int(dense_rank),
                    "original_rank": original_rank,
                    "was_in_baseline": passage_idx in original_rank_by_doc,
                    "score": float(confidence + 0.01 / max(dense_rank, 1)),
                    "probe": {
                        "query": probe.get("query", ""),
                        "kind": probe.get("kind", ""),
                        "rule_id": probe.get("rule_id", ""),
                        "anchor": probe.get("anchor", ""),
                        "relation": probe.get("relation", ""),
                        "route_score": probe.get("route_score", 0.0),
                        "rule": probe.get("rule", {}),
                    },
                    "matched_anchor": bool(detail["anchor_hit"]),
                    "relation_hits": detail["relation_hits"],
                    "rule_hits": detail["rule_hits"],
                    "answer_hit": bool(detail["answer_hit"]),
                    "contexts": [{
                        "failure_evidence": True,
                        "support_key": f"failure_evidence::{probe.get('rule_id', '')}::{probe.get('kind', '')}::{passage_idx}",
                        "plan_id": probe.get("rule_id", "failure_evidence"),
                        "anchor_entity": probe.get("anchor", ""),
                        "end_entity": "",
                        "relations": [probe.get("relation", "")] if probe.get("relation") else [],
                        "fact_indices": [],
                        "fact_position": 0,
                        "signature_match_mode": "failure_evidence_dense",
                        "plan_effective_precision": probe.get("rule_precision_prior", 0.0),
                        "plan_route_score": probe.get("route_score", 0.0),
                        "plan_lift_weight_scale": 0.82,
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
        log["rejection_reasons"] = dict(sorted(rejection_reasons.items(), key=lambda kv: kv[1], reverse=True)[:8])
        if not selected:
            log["skip_reason"] = "no_verified_failure_evidence_candidates"
            return QueryPlanFailureEvidenceResult(np.array(base_doc_ids, dtype=np.intp), base_scores, log)

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
            return QueryPlanFailureEvidenceResult(np.array(new_doc_ids, dtype=np.intp), new_scores, log)

        selected_ids = {int(item["passage_idx"]) for item in selected}
        preserved_ids = [int(doc_id) for doc_id in base_doc_ids[: self.preserve_top_k]]
        rest = [int(doc_id) for doc_id in base_doc_ids if int(doc_id) not in selected_ids and int(doc_id) not in preserved_ids]
        reordered = preserved_ids + [int(item["passage_idx"]) for item in selected] + rest
        seen = set()
        final_ids = []
        for doc_id in reordered:
            if doc_id in seen:
                continue
            final_ids.append(doc_id)
            seen.add(doc_id)
        high = float(np.max(base_scores)) if base_scores.size else 1.0
        step = max(abs(high), 1.0) * self.append_score_margin
        final_scores = np.array([high - step * i for i in range(len(final_ids))], dtype=np.float32)
        final_rank = {int(doc_id): rank + 1 for rank, doc_id in enumerate(final_ids)}
        for item in selected:
            item["final_rank"] = int(final_rank.get(int(item["passage_idx"]), 0))
            item["rank_delta"] = int(item.get("original_rank", 0)) - int(item.get("final_rank", 0))
        log["changed_top5"] = list(base_doc_ids[:5]) != final_ids[:5]
        log["changed_top2"] = list(base_doc_ids[:2]) != final_ids[:2]
        return QueryPlanFailureEvidenceResult(np.array(final_ids, dtype=np.intp), final_scores, log)
