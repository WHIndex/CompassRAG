import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

import numpy as np

from .query_ontology_prior import _relation_family, tokenize
from .query_plan_evidence_set_policy import QueryPlanEvidenceSetPolicy
from .query_plan_ontology import relation_signature
from .query_plan_structural_candidate_retriever import QueryPlanOntologyStructuralCandidateRetriever
from .utils.misc_utils import text_processing


@dataclass
class QueryPlanEvidenceIndexResult:
    sorted_doc_ids: np.ndarray
    sorted_doc_scores: np.ndarray
    log: Dict[str, Any] = field(default_factory=dict)


class QueryPlanOntologyEvidenceIndexRetriever:
    """High-precision ontology-slot evidence index.

    The projected-graph retriever tries to traverse many schema paths. That
    improves coverage diagnostics but can flood the downstream evidence set
    with low-precision candidates. This retriever uses a narrower corpus index:
    ontology step/slot -> matching OpenIE facts -> source passages. A passage is
    exposed only when the query has a compatible schema slot and the candidate
    fact endpoints overlap a concrete query anchor or endpoint token.
    """

    GENERIC_TERMS = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
        "has", "have", "had", "in", "is", "of", "on", "or", "the", "to",
        "was", "were", "with", "what", "when", "where", "which", "who",
        "whom", "whose", "why", "how", "answer", "entity", "relation",
        "person", "place", "thing", "item", "name", "type",
    }

    def __init__(
        self,
        ontology: Mapping[str, Any],
        relation_realization_map: Mapping[str, Any],
        fact_index: Mapping[str, Any],
        get_source_passages: Callable[[Iterable[int]], Dict[int, Set[int]]],
        max_schema_slots: int = 1024,
        max_facts_per_slot: int = 5000,
        max_facts_per_query_slot: int = 400,
        max_candidates: int = 4,
        min_score: float = 0.84,
        preserve_top_k: int = 2,
        candidate_only: bool = True,
        append_score_margin: float = 1e-7,
        require_anchor_overlap: bool = True,
        min_relation_token_overlap: float = 0.35,
        synthetic_original_rank: int = 45,
        top2_min_score: float = 0.92,
        top2_max_original_rank: int = 30,
        top2_require_terminal: bool = True,
        recall_utility_model_path: str = "",
        recall_utility_mode: str = "off",
        recall_utility_min_score: float = 0.70,
        recall_utility_score_weight: float = 0.35,
        recall_utility_promote_confidence: float = 0.98,
        recall_utility_top2_min_score: float = 0.80,
    ) -> None:
        self.ontology = ontology or {}
        self.relation_realization_map = relation_realization_map or {}
        self.fact_index = fact_index or {}
        self.get_source_passages = get_source_passages
        self.max_schema_slots = max(int(max_schema_slots), 1)
        self.max_facts_per_slot = max(int(max_facts_per_slot), 1)
        self.max_facts_per_query_slot = max(int(max_facts_per_query_slot), 1)
        self.max_candidates = max(int(max_candidates), 0)
        self.min_score = min(max(float(min_score), 0.0), 1.0)
        self.preserve_top_k = max(int(preserve_top_k), 0)
        self.candidate_only = bool(candidate_only)
        self.append_score_margin = max(float(append_score_margin), 0.0)
        self.require_anchor_overlap = bool(require_anchor_overlap)
        self.min_relation_token_overlap = min(max(float(min_relation_token_overlap), 0.0), 1.0)
        self.synthetic_original_rank = max(int(synthetic_original_rank), 1)
        self.top2_min_score = min(max(float(top2_min_score), 0.0), 1.0)
        self.top2_max_original_rank = int(top2_max_original_rank)
        self.top2_require_terminal = bool(top2_require_terminal)
        self.recall_utility_mode = str(recall_utility_mode or "off").strip().lower()
        if self.recall_utility_mode not in {"off", "score", "selector"}:
            self.recall_utility_mode = "off"
        self.recall_utility_min_score = min(max(float(recall_utility_min_score), 0.0), 1.0)
        self.recall_utility_score_weight = max(float(recall_utility_score_weight), 0.0)
        self.recall_utility_promote_confidence = min(max(float(recall_utility_promote_confidence), 0.0), 1.0)
        self.recall_utility_top2_min_score = min(max(float(recall_utility_top2_min_score), 0.0), 1.0)
        self.recall_utility_policy = QueryPlanEvidenceSetPolicy(recall_utility_model_path)
        if not self.recall_utility_policy.is_ready():
            self.recall_utility_mode = "off"

        self.slots: Dict[str, Dict[str, Any]] = {}
        self.plan_to_slot_ids: Dict[str, List[str]] = defaultdict(list)
        self.signature_to_slot_ids: Dict[str, Set[str]] = defaultdict(set)
        self.family_to_slot_ids: Dict[str, Set[str]] = defaultdict(set)
        self.token_to_slot_ids: Dict[str, Set[str]] = defaultdict(set)
        self.slot_to_fact_indices: Dict[str, List[int]] = defaultdict(list)
        self.entity_to_slot_fact_indices: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
        self.fact_to_slot_ids: Dict[int, Set[str]] = defaultdict(set)
        self._fact_source_passages_cache: Dict[int, Set[int]] = {}
        self._build_index()

    @classmethod
    def _norm(cls, value: Any) -> str:
        return text_processing(str(value or "").replace("_", " ").replace("-", " "))

    @classmethod
    def _tokens(cls, value: Any, max_tokens: int = 12) -> List[str]:
        output: List[str] = []
        seen: Set[str] = set()
        for token in tokenize(cls._norm(value)):
            if token in cls.GENERIC_TERMS or token in seen:
                continue
            output.append(token)
            seen.add(token)
            if len(output) >= max_tokens:
                break
        return output

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            if value is None:
                return default
            value = float(value)
            if math.isnan(value) or math.isinf(value):
                return default
            return value
        except Exception:
            return default

    @staticmethod
    def _unique(values: Iterable[Any], max_items: int = 64) -> List[str]:
        output: List[str] = []
        seen: Set[str] = set()
        for value in values or []:
            text = text_processing(str(value or ""))
            if not text or text in seen:
                continue
            output.append(text)
            seen.add(text)
            if len(output) >= max_items:
                break
        return output

    def _relation_aliases_for_slot(self, plan_id: str, step_idx: int, step: Mapping[str, Any]) -> List[str]:
        aliases: List[Any] = [
            step.get("relation_family", ""),
            step.get("relation_signature", ""),
        ]
        aliases.extend(step.get("relation_aliases", []) or [])
        aliases.extend(step.get("required_relation_terms", []) or [])

        plans = self.relation_realization_map.get("plans", {}) if isinstance(self.relation_realization_map, Mapping) else {}
        plan_entry = plans.get(plan_id, {}) if isinstance(plans, Mapping) else {}
        steps = plan_entry.get("steps", {}) if isinstance(plan_entry, Mapping) else {}
        step_entry = steps.get(str(step_idx), {}) if isinstance(steps, Mapping) else {}
        for realization in step_entry.get("realizations", []) or []:
            if not isinstance(realization, Mapping):
                continue
            aliases.append(realization.get("relation", ""))
            aliases.append(realization.get("normalized_relation", ""))
            aliases.append(realization.get("relation_family", ""))
            aliases.append(realization.get("relation_signature", ""))
        return self._unique(aliases, max_items=128)

    def _build_slots(self) -> None:
        plans = [
            plan for plan in self.ontology.get("plans", []) or []
            if isinstance(plan, Mapping) and plan.get("path")
        ]
        for plan in plans:
            if len(self.slots) >= self.max_schema_slots:
                break
            plan_id = str(plan.get("id") or f"plan_{len(self.plan_to_slot_ids)}")
            precision = self._safe_float(plan.get("effective_precision", plan.get("precision", 0.0)), 0.0)
            support = self._safe_float(plan.get("support", 0.0), 0.0)
            answer_type = self._norm(plan.get("answer_type", ""))
            for step_idx, step in enumerate(plan.get("path", []) or []):
                if not isinstance(step, Mapping):
                    continue
                if len(self.slots) >= self.max_schema_slots:
                    break
                slot_id = f"{plan_id}:{step_idx}"
                aliases = self._relation_aliases_for_slot(plan_id, step_idx, step)
                alias_tokens = {
                    token
                    for alias in aliases
                    for token in self._tokens(alias, max_tokens=8)
                }
                signature = self._norm(step.get("relation_signature", ""))
                family = self._norm(step.get("relation_family", ""))
                if not aliases and not signature and not family:
                    continue
                self.slots[slot_id] = {
                    "slot_id": slot_id,
                    "plan_id": plan_id,
                    "step_idx": int(step_idx),
                    "from": self._norm(step.get("from", "")),
                    "to": self._norm(step.get("to", "")),
                    "direction": self._norm(step.get("direction", "")),
                    "aliases": aliases,
                    "alias_tokens": alias_tokens,
                    "relation_signature": signature,
                    "relation_family": family,
                    "required_relation_terms": self._unique(step.get("required_relation_terms", []) or [], max_items=32),
                    "reject_relation_terms": self._unique(step.get("reject_relation_terms", []) or [], max_items=32),
                    "terminal_object_allow_terms": self._unique(step.get("terminal_object_allow_terms", []) or [], max_items=64),
                    "terminal_object_reject_terms": self._unique(step.get("terminal_object_reject_terms", []) or [], max_items=32),
                    "answer_type": answer_type,
                    "precision": precision,
                    "support": support,
                    "triggers": list(plan.get("triggers", []) or []),
                }
                self.plan_to_slot_ids[plan_id].append(slot_id)
                if signature:
                    self.signature_to_slot_ids[signature].add(slot_id)
                    self.signature_to_slot_ids[relation_signature(signature)].add(slot_id)
                if family:
                    self.family_to_slot_ids[family].add(slot_id)
                for token in alias_tokens:
                    self.token_to_slot_ids[token].add(slot_id)

    def _candidate_slot_ids_for_relation(self, relation: Any) -> Set[str]:
        relation_norm = self._norm(relation)
        relation_tokens = set(self._tokens(relation_norm, max_tokens=12))
        relation_sig = relation_signature(relation_norm)
        relation_family = _relation_family(relation_norm)
        candidates: Set[str] = set()
        candidates.update(self.signature_to_slot_ids.get(relation_sig, set()))
        if relation_family:
            candidates.update(self.family_to_slot_ids.get(relation_family, set()))
        for token in relation_tokens:
            candidates.update(self.token_to_slot_ids.get(token, set()))

        matched: Set[str] = set()
        for slot_id in candidates:
            slot = self.slots.get(slot_id, {})
            alias_tokens = set(slot.get("alias_tokens", set()) or set())
            if not alias_tokens:
                matched.add(slot_id)
                continue
            overlap = relation_tokens.intersection(alias_tokens)
            token_score = len(overlap) / max(min(len(relation_tokens), len(alias_tokens)), 1)
            signature_match = bool(relation_sig and relation_sig == relation_signature(slot.get("relation_signature", "")))
            family_match = bool(relation_family and relation_family == slot.get("relation_family"))
            if signature_match or (family_match and token_score >= 0.20) or token_score >= self.min_relation_token_overlap:
                matched.add(slot_id)
        return matched

    def _build_index(self) -> None:
        self._build_slots()
        fact_triples = self.fact_index.get("fact_triples", []) or []
        for fact_idx, triple in enumerate(fact_triples):
            if triple is None or len(triple) != 3:
                continue
            subject, relation, obj = triple
            slot_ids = self._candidate_slot_ids_for_relation(relation)
            if not slot_ids:
                continue
            endpoint_tokens = set(self._tokens(subject, max_tokens=12)) | set(self._tokens(obj, max_tokens=12))
            for slot_id in slot_ids:
                if len(self.slot_to_fact_indices[slot_id]) >= self.max_facts_per_slot:
                    continue
                self.slot_to_fact_indices[slot_id].append(int(fact_idx))
                self.fact_to_slot_ids[int(fact_idx)].add(slot_id)
                for token in endpoint_tokens:
                    self.entity_to_slot_fact_indices[slot_id][token].append(int(fact_idx))

    def index_stats(self) -> Dict[str, Any]:
        projected_facts = len(self.fact_to_slot_ids)
        fact_count = len(self.fact_index.get("fact_triples", []) or [])
        non_empty_slots = sum(1 for indices in self.slot_to_fact_indices.values() if indices)
        return {
            "schema_slot_count": len(self.slots),
            "indexed_slot_count": non_empty_slots,
            "fact_count": fact_count,
            "indexed_fact_count": projected_facts,
            "index_coverage": round(projected_facts / max(fact_count, 1), 6),
        }

    def _anchors(self, query: str, query_plan_log: Mapping[str, Any]) -> List[str]:
        anchors: List[Any] = []
        anchors.extend(query_plan_log.get("anchor_entities", []) or [])
        for path in query_plan_log.get("sample_paths", []) or []:
            if isinstance(path, Mapping):
                anchors.append(path.get("anchor_entity", ""))
        for passage in query_plan_log.get("lifted_passages", []) or []:
            if not isinstance(passage, Mapping):
                continue
            for context in passage.get("contexts", []) or []:
                if isinstance(context, Mapping):
                    anchors.append(context.get("anchor_entity", ""))
                    anchors.append(context.get("end_entity", ""))
        cleaned = []
        for anchor in self._unique(anchors, max_items=32):
            if not QueryPlanOntologyStructuralCandidateRetriever.is_generic_anchor(anchor):
                cleaned.append(anchor)
        if cleaned:
            return cleaned
        tokens = self._tokens(query, max_tokens=8)
        return [" ".join(tokens[:4])] if tokens else []

    def _slot_ids_for_runtime(self, query_plan_log: Mapping[str, Any]) -> List[Dict[str, Any]]:
        slot_scores: Dict[str, Dict[str, Any]] = {}

        def add_slot(slot_id: str, score: float, source: str, plan_precision: float = 0.0) -> None:
            if slot_id not in self.slots:
                return
            current = slot_scores.get(slot_id)
            score = min(max(float(score), 0.0), 1.0)
            if current is None or score > float(current.get("score", 0.0)):
                slot = self.slots.get(slot_id, {})
                slot_scores[slot_id] = {
                    "slot_id": slot_id,
                    "score": score,
                    "source": source,
                    "plan_precision": max(plan_precision, self._safe_float(slot.get("precision"), 0.0)),
                    "plan_id": slot.get("plan_id", ""),
                }

        for plan in query_plan_log.get("matched_plans", []) or []:
            if not isinstance(plan, Mapping):
                continue
            plan_id = str(plan.get("id") or "")
            precision = self._safe_float(plan.get("effective_precision", plan.get("precision", 0.0)), 0.0)
            match_score = self._safe_float(plan.get("match_score", 0.0), 0.0)
            for slot_id in self.plan_to_slot_ids.get(plan_id, [])[:4]:
                add_slot(slot_id, 0.74 + 0.10 * min(match_score, 1.0) + 0.08 * precision, "matched_plan", precision)
            for step in plan.get("path", []) or []:
                if not isinstance(step, Mapping):
                    continue
                relation_values: List[Any] = [
                    step.get("relation_signature", ""),
                    step.get("relation_family", ""),
                ]
                relation_values.extend(step.get("relation_aliases", []) or [])
                relation_values.extend(step.get("required_relation_terms", []) or [])
                for relation in relation_values:
                    for slot_id in self._candidate_slot_ids_for_relation(relation):
                        add_slot(slot_id, 0.70 + 0.08 * precision, "matched_plan_relation", precision)

        for sample_path in query_plan_log.get("sample_paths", []) or []:
            if not isinstance(sample_path, Mapping):
                continue
            fact_indices = [
                int(idx) for idx in sample_path.get("fact_indices", []) or []
                if str(idx).lstrip("-").isdigit()
            ]
            precision = self._safe_float(sample_path.get("plan_effective_precision", 0.0), 0.0)
            for fact_idx in fact_indices[:8]:
                for slot_id in self.fact_to_slot_ids.get(fact_idx, set()):
                    add_slot(slot_id, 0.82 + 0.08 * precision, "sample_path_fact", precision)

        query_condition = query_plan_log.get("query_condition", {}) or {}
        relation_values: List[Any] = []
        relation_values.extend(query_condition.get("relation_signatures", []) or [])
        relation_values.extend(query_condition.get("relation_families", []) or [])
        relation_values.extend(query_condition.get("relation_terms", []) or [])
        for relation in relation_values:
            for slot_id in self._candidate_slot_ids_for_relation(relation):
                add_slot(slot_id, 0.72, "query_condition", self._safe_float(self.slots.get(slot_id, {}).get("precision"), 0.0))

        ranked = sorted(
            slot_scores.values(),
            key=lambda item: (
                float(item.get("score", 0.0)),
                float(item.get("plan_precision", 0.0)),
                self._safe_float(self.slots.get(item.get("slot_id", ""), {}).get("support"), 0.0),
            ),
            reverse=True,
        )
        return ranked[:16]

    def _endpoint_context(self, fact_indices: Sequence[int], anchor_tokens: Set[str], query_tokens: Set[str]) -> Dict[str, Any]:
        fact_triples = self.fact_index.get("fact_triples", []) or []
        triples_for_log: List[Tuple[str, str, str]] = []
        endpoint_records: List[Tuple[str, Set[str]]] = []
        for fact_idx in fact_indices or []:
            if fact_idx < 0 or fact_idx >= len(fact_triples):
                continue
            triple = fact_triples[fact_idx]
            if triple is None or len(triple) != 3:
                continue
            triples_for_log.append((str(triple[0]), str(triple[1]), str(triple[2])))
            for entity in (triple[0], triple[2]):
                entity_text = text_processing(str(entity or ""))
                if entity_text:
                    endpoint_records.append((entity_text, set(self._tokens(entity_text, max_tokens=12))))

        anchor_entity = ""
        for entity, tokens in endpoint_records:
            if tokens and tokens.intersection(anchor_tokens):
                anchor_entity = entity
                break
        terminal_entity = ""
        for entity, tokens in reversed(endpoint_records):
            if entity == anchor_entity:
                continue
            if tokens and (tokens.intersection(query_tokens) or not terminal_entity):
                terminal_entity = entity
                if tokens.intersection(query_tokens):
                    break
        return {
            "anchor_entity": anchor_entity,
            "end_entity": terminal_entity,
            "fact_triples": triples_for_log[:4],
        }

    def _endpoint_overlap(self, triple: Tuple[str, str, str], anchor_tokens: Set[str], query_tokens: Set[str]) -> float:
        subject, _, obj = triple
        endpoint_tokens = set(self._tokens(subject, max_tokens=12)) | set(self._tokens(obj, max_tokens=12))
        if not endpoint_tokens:
            return 0.0
        anchor_overlap = len(endpoint_tokens.intersection(anchor_tokens)) / max(len(anchor_tokens), 1)
        query_overlap = len(endpoint_tokens.intersection(query_tokens)) / max(min(len(query_tokens), 12), 1)
        return min(max(0.80 * anchor_overlap + 0.20 * query_overlap, 0.0), 1.0)

    def _cached_source_passages(self, fact_indices: Iterable[int]) -> Dict[int, Set[int]]:
        normalized: List[int] = []
        seen: Set[int] = set()
        for fact_idx in fact_indices or []:
            try:
                fact_idx = int(fact_idx)
            except Exception:
                continue
            if fact_idx < 0 or fact_idx in seen:
                continue
            normalized.append(fact_idx)
            seen.add(fact_idx)

        missing = [idx for idx in normalized if idx not in self._fact_source_passages_cache]
        if missing:
            fetched = self.get_source_passages(missing) or {}
            for idx in missing:
                passage_indices = fetched.get(idx, fetched.get(str(idx), set())) or set()
                cleaned: Set[int] = set()
                for passage_idx in passage_indices:
                    try:
                        cleaned.add(int(passage_idx))
                    except Exception:
                        continue
                self._fact_source_passages_cache[idx] = cleaned

        return {
            idx: self._fact_source_passages_cache.get(idx, set())
            for idx in normalized
            if self._fact_source_passages_cache.get(idx)
        }

    def _facts_for_slot(
        self,
        slot_id: str,
        anchor_tokens: Set[str],
        query_tokens: Set[str],
    ) -> List[Tuple[int, float]]:
        fact_triples = self.fact_index.get("fact_triples", []) or []
        candidate_indices: Set[int] = set()
        token_index = self.entity_to_slot_fact_indices.get(slot_id, {})
        for token in anchor_tokens:
            candidate_indices.update(token_index.get(token, [])[: self.max_facts_per_query_slot])
        if not candidate_indices and not self.require_anchor_overlap:
            candidate_indices.update(self.slot_to_fact_indices.get(slot_id, [])[: self.max_facts_per_query_slot])
        scored: List[Tuple[int, float]] = []
        for fact_idx in candidate_indices:
            if fact_idx < 0 or fact_idx >= len(fact_triples):
                continue
            triple = fact_triples[fact_idx]
            if triple is None or len(triple) != 3:
                continue
            overlap = self._endpoint_overlap(triple, anchor_tokens, query_tokens)
            if self.require_anchor_overlap and overlap <= 0.0:
                continue
            scored.append((int(fact_idx), overlap))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[: self.max_facts_per_query_slot]

    def _source_candidates_for_slot(
        self,
        slot_match: Mapping[str, Any],
        anchor_tokens: Set[str],
        query_tokens: Set[str],
    ) -> List[Dict[str, Any]]:
        slot_id = str(slot_match.get("slot_id") or "")
        slot = self.slots.get(slot_id, {})
        if not slot:
            return []
        facts = self._facts_for_slot(slot_id, anchor_tokens, query_tokens)
        if not facts:
            return []
        fact_to_passages = self._cached_source_passages([fact_idx for fact_idx, _ in facts])
        if not fact_to_passages:
            return []

        fact_overlap = {int(fact_idx): float(overlap) for fact_idx, overlap in facts}
        passage_hits: Dict[int, List[int]] = defaultdict(list)
        for fact_idx, passage_indices in fact_to_passages.items():
            for passage_idx in passage_indices:
                passage_hits[int(passage_idx)].append(int(fact_idx))

        candidates: Dict[int, Dict[str, Any]] = {}
        relation = slot.get("relation_signature") or slot.get("relation_family") or slot_id
        slot_score = self._safe_float(slot_match.get("score"), 0.0)
        plan_precision = self._safe_float(slot_match.get("plan_precision"), self._safe_float(slot.get("precision"), 0.0))
        for passage_idx, source_fact_indices in passage_hits.items():
            source_fact_indices = sorted(set(source_fact_indices))
            best_overlap = max((fact_overlap.get(fact_idx, 0.0) for fact_idx in source_fact_indices), default=0.0)
            coverage_bonus = min(len(source_fact_indices), 3) / 3.0
            score = (
                0.52
                + 0.25 * best_overlap
                + 0.10 * slot_score
                + 0.08 * plan_precision
                + 0.05 * coverage_bonus
            )
            score = min(max(score, 0.0), 1.0)
            if score < self.min_score:
                continue
            endpoint_ctx = self._endpoint_context(source_fact_indices, anchor_tokens, query_tokens)
            strong_endpoint = bool(endpoint_ctx.get("anchor_entity")) and (
                bool(endpoint_ctx.get("end_entity")) or best_overlap >= 0.45
            )
            context = {
                "ontology_evidence_index": True,
                "schema_route_source": "ontology_evidence_index",
                "support_key": f"ontology_evidence_index::{slot_id}::{passage_idx}",
                "plan_id": slot.get("plan_id", ""),
                "slot_id": slot_id,
                "anchor_entity": endpoint_ctx.get("anchor_entity", ""),
                "end_entity": endpoint_ctx.get("end_entity", ""),
                "relations": [relation],
                "fact_indices": source_fact_indices,
                "fact_position": 0,
                "signature_match_mode": "ontology_evidence_index",
                "plan_effective_precision": max(plan_precision, score * 0.70),
                "plan_lift_weight_scale": 0.98,
                "score": score,
                "slot_score": slot_score,
                "source": slot_match.get("source", ""),
                "fact_triples": endpoint_ctx.get("fact_triples", []),
                "path_seed_hit": strong_endpoint,
                "fact_score_reasons": {
                    "path_seed": 1 if strong_endpoint else 0,
                    "path_neighbor": max(len(source_fact_indices) - 1, 0),
                    "endpoint_overlap": round(best_overlap, 4),
                },
            }
            candidates[int(passage_idx)] = {
                "passage_idx": int(passage_idx),
                "confidence": score,
                "score": score,
                "route": "ontology_evidence_index",
                "matched_anchor": bool(endpoint_ctx.get("anchor_entity")),
                "matched_terminal": bool(endpoint_ctx.get("end_entity")),
                "relation_hits": [relation],
                "answer_hit": False,
                "slot_id": slot_id,
                "slot_source": slot_match.get("source", ""),
                "contexts": [context],
            }
        return sorted(candidates.values(), key=lambda item: item.get("score", 0.0), reverse=True)

    def _candidate_allowed_for_top2(self, candidate: Mapping[str, Any]) -> Tuple[bool, str]:
        original_rank = int(candidate.get("baseline_original_rank", candidate.get("original_rank", 10**9)) or 10**9)
        if self.top2_max_original_rank > 0 and original_rank > self.top2_max_original_rank:
            return False, "top2_rank_too_low"
        if self.top2_require_terminal and not (
            bool(candidate.get("matched_anchor", False)) and bool(candidate.get("matched_terminal", False))
        ):
            return False, "top2_missing_anchor_or_terminal"
        utility = candidate.get("recall_utility", {}) or {}
        if isinstance(utility, Mapping) and bool(utility.get("enabled", False)):
            utility_score = self._safe_float(utility.get("score"), 0.0)
            if self.recall_utility_mode in {"score", "selector"}:
                if utility_score < self.recall_utility_top2_min_score:
                    return False, "top2_recall_utility_low_score"
                return True, "top2_recall_utility"
        score = self._safe_float(candidate.get("score", candidate.get("confidence", 0.0)), 0.0)
        if score < self.top2_min_score:
            return False, "top2_low_score"
        return True, "top2_allowed"

    def _apply_recall_utility(
        self,
        candidate: Dict[str, Any],
        query_plan_log: Mapping[str, Any],
    ) -> None:
        if self.recall_utility_mode == "off" or not self.recall_utility_policy.is_ready():
            candidate["recall_utility"] = {
                "enabled": False,
                "mode": self.recall_utility_mode,
                "model_ready": self.recall_utility_policy.is_ready(),
            }
            return
        enriched = dict(candidate)
        enriched["source"] = "ontology_evidence_index"
        policy = self.recall_utility_policy.score(query_plan_log, enriched)
        score = self._safe_float(policy.get("score"), 0.0)
        decision = "promote" if score >= self.recall_utility_min_score else "keep"
        policy.update({
            "enabled": True,
            "mode": self.recall_utility_mode,
            "min_score": self.recall_utility_min_score,
            "top2_min_score": self.recall_utility_top2_min_score,
            "score_weight": self.recall_utility_score_weight,
            "decision": decision,
        })
        if score >= self.recall_utility_min_score:
            candidate["confidence"] = max(
                self._safe_float(candidate.get("confidence"), 0.0),
                self.recall_utility_promote_confidence,
                min(score, 0.995),
            )
        if score >= self.recall_utility_min_score and self.recall_utility_score_weight > 0.0:
            candidate["score"] = self._safe_float(candidate.get("score"), 0.0) + self.recall_utility_score_weight * score
        candidate["recall_utility"] = policy

    def retrieve(
        self,
        query: str,
        sorted_doc_ids: Sequence[int],
        sorted_doc_scores: Sequence[float],
        query_plan_log: Mapping[str, Any],
    ) -> QueryPlanEvidenceIndexResult:
        base_doc_ids = [int(doc_id) for doc_id in list(sorted_doc_ids)]
        base_scores = np.array(sorted_doc_scores, dtype=np.float32)
        log: Dict[str, Any] = {
            "enabled": True,
            "candidate_only": self.candidate_only,
            "max_candidates": self.max_candidates,
            "min_score": self.min_score,
            "preserve_top_k": self.preserve_top_k,
            "synthetic_original_rank": self.synthetic_original_rank,
            "top2_min_score": self.top2_min_score,
            "top2_max_original_rank": self.top2_max_original_rank,
            "top2_require_terminal": self.top2_require_terminal,
            "recall_utility_mode": self.recall_utility_mode,
            "recall_utility_model_path": self.recall_utility_policy.model_path,
            "recall_utility_model_ready": self.recall_utility_policy.is_ready(),
            "recall_utility_min_score": self.recall_utility_min_score,
            "recall_utility_top2_min_score": self.recall_utility_top2_min_score,
            "recall_utility_score_weight": self.recall_utility_score_weight,
            "index": self.index_stats(),
            "candidate_slot_count": 0,
            "candidate_count": 0,
            "selected_count": 0,
            "added_candidates": [],
            "rejection_reasons": {},
        }
        if self.max_candidates <= 0:
            log["skip_reason"] = "disabled_by_zero_limit"
            return QueryPlanEvidenceIndexResult(np.array(base_doc_ids, dtype=np.intp), base_scores, log)
        if not base_doc_ids:
            log["skip_reason"] = "empty_retrieval"
            return QueryPlanEvidenceIndexResult(np.array(base_doc_ids, dtype=np.intp), base_scores, log)
        if not query_plan_log or not query_plan_log.get("enabled", False):
            log["skip_reason"] = "no_query_plan_log"
            return QueryPlanEvidenceIndexResult(np.array(base_doc_ids, dtype=np.intp), base_scores, log)
        if not self.slots or not self.fact_to_slot_ids:
            log["skip_reason"] = "empty_evidence_index"
            return QueryPlanEvidenceIndexResult(np.array(base_doc_ids, dtype=np.intp), base_scores, log)

        slot_matches = self._slot_ids_for_runtime(query_plan_log)
        log["candidate_slot_count"] = len(slot_matches)
        log["candidate_slots"] = slot_matches[:12]
        if not slot_matches:
            log["skip_reason"] = "no_runtime_evidence_slots"
            return QueryPlanEvidenceIndexResult(np.array(base_doc_ids, dtype=np.intp), base_scores, log)

        anchors = self._anchors(query, query_plan_log)
        anchor_tokens: Set[str] = set()
        for anchor in anchors:
            anchor_tokens.update(self._tokens(anchor, max_tokens=12))
        query_tokens = set(self._tokens(query, max_tokens=24))
        log["anchors"] = anchors[:8]

        original_rank_by_doc = {int(doc_id): rank + 1 for rank, doc_id in enumerate(base_doc_ids)}
        preserved = set(base_doc_ids[: self.preserve_top_k])
        candidates_by_doc: Dict[int, Dict[str, Any]] = {}
        rejection_reasons: Dict[str, int] = {}
        for slot_match in slot_matches:
            slot_candidates = self._source_candidates_for_slot(slot_match, anchor_tokens, query_tokens)
            if not slot_candidates:
                rejection_reasons["slot_no_candidates"] = rejection_reasons.get("slot_no_candidates", 0) + 1
                continue
            for candidate in slot_candidates:
                passage_idx = int(candidate["passage_idx"])
                if passage_idx in preserved:
                    rejection_reasons["preserved_top"] = rejection_reasons.get("preserved_top", 0) + 1
                    continue
                actual_rank = int(original_rank_by_doc.get(passage_idx, 10**9))
                candidate["baseline_original_rank"] = actual_rank
                candidate["original_rank"] = min(actual_rank, self.synthetic_original_rank)
                candidate["was_in_baseline"] = passage_idx in original_rank_by_doc
                score = float(candidate.get("score", 0.0))
                existing = candidates_by_doc.get(passage_idx)
                if existing is None or score > float(existing.get("score", 0.0)):
                    candidates_by_doc[passage_idx] = candidate

        for candidate in candidates_by_doc.values():
            self._apply_recall_utility(candidate, query_plan_log)

        candidates = sorted(
            candidates_by_doc.values(),
            key=lambda item: (
                (
                    self._safe_float((item.get("recall_utility", {}) or {}).get("score"), 0.0)
                    if self._safe_float((item.get("recall_utility", {}) or {}).get("score"), 0.0)
                    >= self.recall_utility_min_score else 0.0
                )
                if self.recall_utility_mode in {"score", "selector"} else 0.0,
                float(item.get("score", 0.0)),
                -int(item.get("original_rank", 10**9)),
            ),
            reverse=True,
        )
        selected = candidates[: self.max_candidates]
        log["candidate_count"] = len(candidates)
        log["selected_count"] = len(selected)
        log["added_candidates"] = selected
        log["recall_utility_scored_candidates"] = sum(
            1 for item in candidates
            if bool((item.get("recall_utility", {}) or {}).get("enabled", False))
        )
        log["recall_utility_promoted_candidates"] = sum(
            1 for item in candidates
            if str((item.get("recall_utility", {}) or {}).get("decision", "") or "") == "promote"
        )
        log["recall_utility_top_score"] = round(
            max(
                (
                    self._safe_float((item.get("recall_utility", {}) or {}).get("score"), 0.0)
                    for item in candidates
                ),
                default=0.0,
            ),
            6,
        )
        log["rejection_reasons"] = dict(sorted(rejection_reasons.items(), key=lambda kv: kv[1], reverse=True)[:10])
        if not selected:
            log["skip_reason"] = "no_evidence_index_candidates"
            return QueryPlanEvidenceIndexResult(np.array(base_doc_ids, dtype=np.intp), base_scores, log)

        if self.candidate_only:
            score_by_doc = {
                int(doc_id): float(base_scores[idx]) if idx < len(base_scores) else 0.0
                for idx, doc_id in enumerate(base_doc_ids)
            }
            min_tail_score = float(base_scores[-1]) if len(base_scores) else 0.0
            appended: List[int] = []
            existing = set(base_doc_ids)
            for rank, item in enumerate(selected, start=1):
                passage_idx = int(item["passage_idx"])
                if passage_idx in existing:
                    continue
                appended.append(passage_idx)
                score_by_doc[passage_idx] = min_tail_score - self.append_score_margin * rank
            new_doc_ids = base_doc_ids + appended
            new_scores = np.array([score_by_doc[int(idx)] for idx in new_doc_ids], dtype=np.float32)
            log["appended_passage_count"] = len(appended)
            log["skip_reason"] = "candidate_only"
            return QueryPlanEvidenceIndexResult(np.array(new_doc_ids, dtype=np.intp), new_scores, log)

        selected_ids = {int(item["passage_idx"]) for item in selected}
        top2_candidates: List[Dict[str, Any]] = []
        later_candidates: List[Dict[str, Any]] = []
        top2_rejection_reasons: Dict[str, int] = {}
        for item in selected:
            allowed, reason = self._candidate_allowed_for_top2(item)
            item["top2_allowed"] = bool(allowed)
            item["top2_decision_reason"] = reason
            if allowed:
                top2_candidates.append(item)
            else:
                later_candidates.append(item)
                top2_rejection_reasons[reason] = top2_rejection_reasons.get(reason, 0) + 1

        preserved_ids = [int(doc_id) for doc_id in base_doc_ids[: self.preserve_top_k]]
        if self.preserve_top_k <= 1 and top2_candidates:
            protected_for_later = set(base_doc_ids[:1])
            top2_insert = [int(top2_candidates[0]["passage_idx"])]
            late_insert = [int(item["passage_idx"]) for item in top2_candidates[1:] + later_candidates]
        else:
            protected_for_later = set(base_doc_ids[: max(self.preserve_top_k, 2)])
            top2_insert = []
            late_insert = [int(item["passage_idx"]) for item in selected]

        rest = [
            int(doc_id)
            for doc_id in base_doc_ids
            if int(doc_id) not in selected_ids and int(doc_id) not in set(preserved_ids)
        ]
        if not top2_insert and self.preserve_top_k <= 1 and len(base_doc_ids) > 1:
            preserved_ids = [int(base_doc_ids[0]), int(base_doc_ids[1])]
            rest = [
                int(doc_id)
                for doc_id in base_doc_ids
                if int(doc_id) not in selected_ids and int(doc_id) not in set(preserved_ids)
            ]
        late_insert = [doc_id for doc_id in late_insert if doc_id not in protected_for_later]
        reordered = preserved_ids + top2_insert + late_insert + rest
        final_ids: List[int] = []
        seen: Set[int] = set()
        for doc_id in reordered:
            if doc_id in seen:
                continue
            final_ids.append(int(doc_id))
            seen.add(int(doc_id))
        high = float(np.max(base_scores)) if base_scores.size else 1.0
        step = max(abs(high), 1.0) * self.append_score_margin
        final_scores = np.array([high - step * index for index in range(len(final_ids))], dtype=np.float32)
        log["changed_top2"] = list(base_doc_ids[:2]) != final_ids[:2]
        log["changed_top5"] = list(base_doc_ids[:5]) != final_ids[:5]
        log["top2_inserted"] = bool(top2_insert)
        log["top2_inserted_passage"] = int(top2_insert[0]) if top2_insert else None
        log["top2_rejection_reasons"] = dict(sorted(top2_rejection_reasons.items(), key=lambda kv: kv[1], reverse=True))
        return QueryPlanEvidenceIndexResult(np.array(final_ids, dtype=np.intp), final_scores, log)
