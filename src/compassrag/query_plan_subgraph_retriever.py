from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

import numpy as np

from .query_ontology_prior import _relation_family, tokenize
from .query_plan_ontology import relation_signature
from .utils.misc_utils import text_processing


@dataclass
class QueryPlanSubgraphRetrieverResult:
    sorted_doc_ids: np.ndarray
    sorted_doc_scores: np.ndarray
    log: Dict[str, Any] = field(default_factory=dict)


class QueryPlanOntologySubgraphRetriever:
    """Generate retrieval candidates from the query-plan ontology subgraph.

    Earlier query-plan modules mostly rerank passages already surfaced by
    dense/PPR retrieval. This retriever treats matched query plans as a schema
    over the CompassRAG fact graph: start from concrete query anchors, traverse
    relation-family-compatible facts, map the resulting facts back to source
    passages, and append those passages to the candidate pool for downstream
    evidence-set selection.
    """

    GENERIC_ENTITIES = {
        "answer", "entity", "name", "title", "person", "people", "place",
        "location", "city", "country", "state", "county", "region", "area",
        "film", "movie", "book", "song", "album", "work", "event", "group",
        "organization", "company", "school", "university", "team", "river",
        "mountain", "island", "lake", "first", "second", "third", "same",
        "different", "both", "north", "south", "east", "west",
    }

    def __init__(
        self,
        candidate_fact_top_k: int = 240,
        max_facts_per_anchor: int = 120,
        max_facts_per_relation: int = 160,
        max_source_passages_per_fact: int = 3,
        max_added_passages: int = 24,
        max_candidate_pool_rank: int = 260,
        min_fact_score: float = 0.10,
        min_passage_score: float = 0.18,
        relation_weight: float = 0.34,
        anchor_weight: float = 0.30,
        path_neighbor_weight: float = 0.22,
        query_fact_weight: float = 0.14,
        use_entity_fallback: bool = True,
        append_score_margin: float = 1e-7,
        min_original_rank: int = 1,
        max_original_rank: int = 0,
        require_strong_fact_gate: bool = False,
        relation_only_min_query_fact_score: float = 0.35,
    ):
        self.candidate_fact_top_k = max(int(candidate_fact_top_k), 0)
        self.max_facts_per_anchor = max(int(max_facts_per_anchor), 0)
        self.max_facts_per_relation = max(int(max_facts_per_relation), 0)
        self.max_source_passages_per_fact = max(int(max_source_passages_per_fact), 1)
        self.max_added_passages = max(int(max_added_passages), 0)
        self.max_candidate_pool_rank = max(int(max_candidate_pool_rank), 1)
        self.min_fact_score = max(float(min_fact_score), 0.0)
        self.min_passage_score = max(float(min_passage_score), 0.0)
        self.relation_weight = max(float(relation_weight), 0.0)
        self.anchor_weight = max(float(anchor_weight), 0.0)
        self.path_neighbor_weight = max(float(path_neighbor_weight), 0.0)
        self.query_fact_weight = max(float(query_fact_weight), 0.0)
        self.use_entity_fallback = bool(use_entity_fallback)
        self.append_score_margin = max(float(append_score_margin), 0.0)
        self.min_original_rank = max(int(min_original_rank), 1)
        self.max_original_rank = max(int(max_original_rank), 0)
        self.require_strong_fact_gate = bool(require_strong_fact_gate)
        self.relation_only_min_query_fact_score = max(float(relation_only_min_query_fact_score), 0.0)

    @staticmethod
    def _safe_int(value: Any, default: int = -1) -> int:
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

    @classmethod
    def _is_generic_entity(cls, value: Any) -> bool:
        text = text_processing(value)
        if not text:
            return True
        tokens = tokenize(text)
        if text in cls.GENERIC_ENTITIES:
            return True
        if len(tokens) <= 1 and (len(text) <= 2 or text in cls.GENERIC_ENTITIES):
            return True
        return bool(tokens) and all(token in cls.GENERIC_ENTITIES for token in tokens)

    @classmethod
    def _normalize_entities(cls, values: Iterable[Any], max_items: int = 32) -> List[str]:
        output = []
        seen = set()
        for value in values or []:
            text = text_processing(value)
            if not text or text in seen or cls._is_generic_entity(text):
                continue
            output.append(text)
            seen.add(text)
            if len(output) >= max_items:
                break
        return output

    @staticmethod
    def _relation_families_from_values(values: Iterable[Any]) -> List[str]:
        families = []
        seen = set()
        for value in values or []:
            raw = text_processing(str(value or "").replace("_", " "))
            if not raw:
                continue
            for candidate in (raw, relation_signature(raw).replace("_", " ")):
                family = _relation_family(candidate) or candidate
                family = text_processing(family)
                if family and family not in seen:
                    families.append(family)
                    seen.add(family)
        return families

    def _extract_terms(self, query_plan_log: Mapping[str, Any]) -> Dict[str, Any]:
        anchors = self._normalize_entities(query_plan_log.get("anchor_entities", []) or [])
        relation_values = []
        terminals = []
        seed_fact_indices = set()

        for plan in query_plan_log.get("matched_plans", []) or []:
            if not isinstance(plan, Mapping):
                continue
            relation_values.extend(plan.get("relation_families", []) or [])
            relation_values.extend(plan.get("relation_signatures", []) or [])

        for path in query_plan_log.get("sample_paths", []) or []:
            if not isinstance(path, Mapping):
                continue
            relation_values.extend(path.get("relations", []) or [])
            terminals.append(path.get("end_entity", ""))
            terminals.append(path.get("entity", ""))
            for fact_idx in path.get("fact_indices", []) or []:
                if self._safe_int(fact_idx) >= 0:
                    seed_fact_indices.add(self._safe_int(fact_idx))

        for passage in query_plan_log.get("lifted_passages", []) or []:
            if not isinstance(passage, Mapping):
                continue
            for context in passage.get("contexts", []) or []:
                if not isinstance(context, Mapping):
                    continue
                relation_values.extend(context.get("relations", []) or [])
                terminals.append(context.get("end_entity", ""))
                anchors.append(text_processing(context.get("anchor_entity", "")))
                for fact_idx in context.get("fact_indices", []) or []:
                    if self._safe_int(fact_idx) >= 0:
                        seed_fact_indices.add(self._safe_int(fact_idx))

        for fact_idx in query_plan_log.get("path_fact_indices", []) or []:
            if self._safe_int(fact_idx) >= 0:
                seed_fact_indices.add(self._safe_int(fact_idx))

        anchors = self._normalize_entities(anchors, max_items=24)
        terminals = self._normalize_entities(terminals, max_items=40)
        relation_families = self._relation_families_from_values(relation_values)
        return {
            "anchors": anchors,
            "terminals": terminals,
            "relation_families": relation_families,
            "seed_fact_indices": seed_fact_indices,
        }

    def _query_fact_score(self, query_fact_scores: Sequence[float], fact_idx: int) -> float:
        if query_fact_scores is None or fact_idx < 0 or fact_idx >= len(query_fact_scores):
            return 0.0
        return self._safe_float(query_fact_scores[fact_idx], 0.0)

    def _collect_fact_candidates(
        self,
        terms: Mapping[str, Any],
        query_fact_scores: Sequence[float],
        fact_index: Mapping[str, Any],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        entities = fact_index.get("entities", {}) or {}
        relation_families = fact_index.get("relation_families", {}) or {}
        fact_triples = fact_index.get("fact_triples", []) or []

        anchors = set(terms.get("anchors", []) or [])
        terminals = set(terms.get("terminals", []) or [])
        target_families = set(terms.get("relation_families", []) or [])
        seed_fact_indices = {int(idx) for idx in terms.get("seed_fact_indices", set()) or set()}

        fact_reasons: Dict[int, Counter] = defaultdict(Counter)
        rejected = Counter()

        for entity in anchors:
            indices = entities.get(entity)
            if indices is None:
                rejected["anchor_entity_not_found"] += 1
                continue
            for fact_idx in list(indices)[: self.max_facts_per_anchor]:
                fact_reasons[int(fact_idx)]["anchor"] += 1

        for family in target_families:
            indices = relation_families.get(family)
            if indices is None:
                rejected["relation_family_not_found"] += 1
                continue
            ranked = sorted(
                (int(idx) for idx in indices),
                key=lambda idx: self._query_fact_score(query_fact_scores, idx),
                reverse=True,
            )
            for fact_idx in ranked[: self.max_facts_per_relation]:
                fact_reasons[int(fact_idx)]["relation"] += 1

        for fact_idx in seed_fact_indices:
            if 0 <= fact_idx < len(fact_triples):
                fact_reasons[int(fact_idx)]["path_seed"] += 1
                triple = fact_triples[fact_idx]
                if triple:
                    subject, _, obj = triple
                    for entity in (subject, obj):
                        indices = entities.get(entity)
                        if indices is None:
                            continue
                        ranked = sorted(
                            (int(idx) for idx in indices),
                            key=lambda idx: self._query_fact_score(query_fact_scores, idx),
                            reverse=True,
                        )
                        for neighbor_idx in ranked[: max(self.max_facts_per_anchor // 3, 12)]:
                            fact_reasons[int(neighbor_idx)]["path_neighbor"] += 1

        candidates = []
        for fact_idx, reasons in fact_reasons.items():
            if fact_idx < 0 or fact_idx >= len(fact_triples):
                rejected["invalid_fact_idx"] += 1
                continue
            triple = fact_triples[fact_idx]
            if not triple:
                rejected["missing_fact_triple"] += 1
                continue
            subject, relation, obj = triple
            relation_family = _relation_family(relation) or relation
            anchor_hit = subject in anchors or obj in anchors
            terminal_hit = subject in terminals or obj in terminals
            relation_hit = text_processing(relation_family) in target_families
            query_score = self._query_fact_score(query_fact_scores, fact_idx)
            path_seed_hit = bool(reasons.get("path_seed"))
            strong_fact_gate = (
                path_seed_hit
                or (anchor_hit and relation_hit)
                or (relation_hit and query_score >= self.relation_only_min_query_fact_score)
            )
            if self.require_strong_fact_gate and not strong_fact_gate:
                rejected["weak_fact_gate"] += 1
                continue
            if query_score < self.min_fact_score and not anchor_hit and not relation_hit and not reasons.get("path_seed"):
                rejected["low_fact_score"] += 1
                continue
            score = (
                self.query_fact_weight * min(max(query_score, 0.0), 1.0)
                + self.anchor_weight * (1.0 if anchor_hit else min(reasons.get("anchor", 0), 1))
                + self.relation_weight * (1.0 if relation_hit else min(reasons.get("relation", 0), 1))
                + self.path_neighbor_weight * min(reasons.get("path_seed", 0) + reasons.get("path_neighbor", 0), 1)
            )
            if terminal_hit:
                score += 0.08
            score = min(max(score, 0.0), 1.0)
            if score < self.min_fact_score:
                rejected["low_combined_fact_score"] += 1
                continue
            candidates.append({
                "fact_idx": int(fact_idx),
                "score": float(score),
                "query_fact_score": float(query_score),
                "triple": triple,
                "relation_family": text_processing(relation_family),
                "anchor_hit": bool(anchor_hit),
                "terminal_hit": bool(terminal_hit),
                "relation_hit": bool(relation_hit),
                "path_seed_hit": bool(path_seed_hit),
                "strong_fact_gate": bool(strong_fact_gate),
                "reasons": dict(reasons),
            })

        candidates.sort(
            key=lambda item: (
                item["score"],
                item["anchor_hit"],
                item["relation_hit"],
                item["terminal_hit"],
                item["query_fact_score"],
            ),
            reverse=True,
        )
        return candidates[: self.candidate_fact_top_k], dict(rejected)

    def retrieve(
        self,
        sorted_doc_ids: Sequence[int],
        sorted_doc_scores: Sequence[float],
        query_fact_scores: Sequence[float],
        query_plan_log: Mapping[str, Any],
        fact_index: Mapping[str, Any],
        get_source_passages: Callable[[Iterable[int], bool], Mapping[int, Set[int]]],
    ) -> QueryPlanSubgraphRetrieverResult:
        base_doc_ids = [int(doc_id) for doc_id in list(sorted_doc_ids)]
        base_scores = np.array(sorted_doc_scores, dtype=np.float32)
        log: Dict[str, Any] = {
            "enabled": True,
            "candidate_fact_top_k": self.candidate_fact_top_k,
            "max_added_passages": self.max_added_passages,
            "max_candidate_pool_rank": self.max_candidate_pool_rank,
            "min_fact_score": self.min_fact_score,
            "min_passage_score": self.min_passage_score,
            "use_entity_fallback": self.use_entity_fallback,
            "min_original_rank": self.min_original_rank,
            "max_original_rank": self.max_original_rank,
            "require_strong_fact_gate": self.require_strong_fact_gate,
            "relation_only_min_query_fact_score": self.relation_only_min_query_fact_score,
            "added_candidates": [],
        }
        if self.candidate_fact_top_k <= 0 or self.max_added_passages <= 0:
            log["skip_reason"] = "disabled_by_zero_limit"
            return QueryPlanSubgraphRetrieverResult(np.array(base_doc_ids, dtype=np.intp), base_scores, log)
        if not base_doc_ids:
            log["skip_reason"] = "empty_retrieval"
            return QueryPlanSubgraphRetrieverResult(np.array(base_doc_ids, dtype=np.intp), base_scores, log)
        if not fact_index:
            log["skip_reason"] = "missing_fact_index"
            return QueryPlanSubgraphRetrieverResult(np.array(base_doc_ids, dtype=np.intp), base_scores, log)

        terms = self._extract_terms(query_plan_log)
        log["anchor_count"] = len(terms["anchors"])
        log["terminal_count"] = len(terms["terminals"])
        log["relation_family_count"] = len(terms["relation_families"])
        log["seed_fact_count"] = len(terms["seed_fact_indices"])
        log["anchors"] = terms["anchors"][:8]
        log["relation_families"] = terms["relation_families"][:8]
        if not terms["anchors"] and not terms["seed_fact_indices"]:
            log["skip_reason"] = "no_anchor_or_seed_facts"
            return QueryPlanSubgraphRetrieverResult(np.array(base_doc_ids, dtype=np.intp), base_scores, log)
        if not terms["relation_families"] and not terms["seed_fact_indices"]:
            log["skip_reason"] = "no_relation_family_or_seed_facts"
            return QueryPlanSubgraphRetrieverResult(np.array(base_doc_ids, dtype=np.intp), base_scores, log)

        fact_candidates, rejected_facts = self._collect_fact_candidates(terms, query_fact_scores, fact_index)
        log["candidate_fact_count"] = len(fact_candidates)
        log["fact_rejection_reasons"] = rejected_facts
        if not fact_candidates:
            log["skip_reason"] = "no_fact_candidates"
            return QueryPlanSubgraphRetrieverResult(np.array(base_doc_ids, dtype=np.intp), base_scores, log)

        fact_to_passages = get_source_passages(
            [item["fact_idx"] for item in fact_candidates],
            self.use_entity_fallback,
        )
        if not fact_to_passages:
            log["skip_reason"] = "no_source_passages"
            return QueryPlanSubgraphRetrieverResult(np.array(base_doc_ids, dtype=np.intp), base_scores, log)

        original_rank_by_doc = {int(doc_id): rank + 1 for rank, doc_id in enumerate(base_doc_ids)}
        passage_scores = defaultdict(float)
        passage_contexts = defaultdict(list)
        source_fact_count = 0
        passage_rejections = Counter()
        for item in fact_candidates:
            fact_idx = int(item["fact_idx"])
            source_passages = sorted(int(idx) for idx in fact_to_passages.get(fact_idx, set()) or set())
            if not source_passages:
                continue
            source_fact_count += 1
            for passage_idx in source_passages[: self.max_source_passages_per_fact]:
                original_rank = original_rank_by_doc.get(int(passage_idx), len(base_doc_ids) + 1)
                if original_rank > self.max_candidate_pool_rank:


                    original_rank = self.max_candidate_pool_rank
                if original_rank < self.min_original_rank:
                    passage_rejections["rank_before_min_original_rank"] += 1
                    continue
                if self.max_original_rank > 0 and original_rank > self.max_original_rank:
                    passage_rejections["rank_after_max_original_rank"] += 1
                    continue
                passage_scores[int(passage_idx)] += float(item["score"])
                subject, relation, obj = item["triple"]
                passage_contexts[int(passage_idx)].append({
                    "subgraph_candidate": True,
                    "plan_id": "ontology_subgraph",
                    "anchor_entity": subject if item["anchor_hit"] else "",
                    "end_entity": obj if item["terminal_hit"] else "",
                    "relations": [relation],
                    "relation_family": item["relation_family"],
                    "fact_indices": [fact_idx],
                    "fact_position": 0,
                    "signature_match_mode": "subgraph",
                    "plan_effective_precision": float(item["score"]),
                    "plan_lift_weight_scale": 0.80,
                    "score": float(item["score"]),
                    "original_rank": int(original_rank),
                    "fact_score_reasons": item["reasons"],
                    "path_seed_hit": bool(item.get("path_seed_hit", False)),
                    "relation_hit": bool(item.get("relation_hit", False)),
                    "strong_fact_gate": bool(item.get("strong_fact_gate", False)),
                    "support_key": f"subgraph::{fact_idx}::{int(passage_idx)}",
                })

        ranked_passages = sorted(passage_scores.items(), key=lambda item: item[1], reverse=True)
        added = []
        existing = set(base_doc_ids)
        for passage_idx, raw_score in ranked_passages:
            confidence = min(float(raw_score), 1.0)
            if confidence < self.min_passage_score:
                continue
            contexts = passage_contexts.get(int(passage_idx), []) or []
            original_rank = int(original_rank_by_doc.get(int(passage_idx), self.max_candidate_pool_rank))
            added.append({
                "passage_idx": int(passage_idx),
                "score": float(confidence),
                "raw_score": float(raw_score),
                "original_rank": original_rank,
                "contexts": contexts[:5],
                "fact_count": len({ctx.get("fact_indices", [None])[0] for ctx in contexts}),
                "already_retrieved": int(passage_idx) in existing,
            })
            if len(added) >= self.max_added_passages:
                break

        log["source_fact_count"] = int(source_fact_count)
        log["candidate_passage_count"] = len(ranked_passages)
        log["passage_rejection_reasons"] = dict(passage_rejections)
        log["added_candidate_count"] = len(added)
        log["added_candidates"] = added
        if not added:
            log["skip_reason"] = "no_added_candidates"
            return QueryPlanSubgraphRetrieverResult(np.array(base_doc_ids, dtype=np.intp), base_scores, log)

        score_by_doc = {
            int(doc_id): float(base_scores[idx]) if idx < len(base_scores) else 0.0
            for idx, doc_id in enumerate(base_doc_ids)
        }
        min_tail_score = float(base_scores[-1]) if len(base_scores) else 0.0
        appended = []
        for rank, item in enumerate(added, start=1):
            passage_idx = int(item["passage_idx"])
            if passage_idx not in score_by_doc:
                appended.append(passage_idx)


                score_by_doc[passage_idx] = min_tail_score - self.append_score_margin * rank

        new_doc_ids = base_doc_ids + [idx for idx in appended if idx not in set(base_doc_ids)]
        new_scores = np.array([score_by_doc[int(idx)] for idx in new_doc_ids], dtype=np.float32)
        log["appended_passage_count"] = len(appended)
        return QueryPlanSubgraphRetrieverResult(np.array(new_doc_ids, dtype=np.intp), new_scores, log)
