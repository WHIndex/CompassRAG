import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

import numpy as np

from .query_ontology_prior import _relation_family, tokenize
from .query_plan_ontology import relation_signature
from .query_plan_structural_candidate_retriever import QueryPlanOntologyStructuralCandidateRetriever
from .utils.misc_utils import text_processing


@dataclass
class QueryPlanProjectedGraphResult:
    sorted_doc_ids: np.ndarray
    sorted_doc_scores: np.ndarray
    log: Dict[str, Any] = field(default_factory=dict)


class QueryPlanOntologyProjectedGraphRetriever:
    """Project the corpus fact graph onto query-plan ontology schemas.

    Earlier query-plan modules mostly create dense passage probes or rerank
    already surfaced passages. This module gives the ontology a more direct
    retrieval role: map each OpenIE fact relation to ontology path steps, then
    use matched query-plan paths to pull source passages from the projected
    fact graph.
    """

    GENERIC_TERMS = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
        "has", "have", "had", "in", "is", "of", "on", "or", "the", "to",
        "was", "were", "with", "what", "when", "where", "which", "who",
        "whom", "whose", "why", "how", "answer", "entity", "relation",
        "person", "place", "thing", "item",
    }

    def __init__(
        self,
        ontology: Mapping[str, Any],
        relation_realization_map: Mapping[str, Any],
        fact_index: Mapping[str, Any],
        get_source_passages: Callable[[Iterable[int]], Dict[int, Set[int]]],
        max_schema_relations: int = 512,
        max_facts_per_schema: int = 5000,
        max_facts_per_step_query: int = 600,
        max_candidates: int = 4,
        min_score: float = 0.62,
        preserve_top_k: int = 2,
        candidate_only: bool = True,
        append_score_margin: float = 1e-7,
        require_anchor_overlap: bool = True,
        min_relation_token_overlap: float = 0.50,
        enable_bridge_paths: bool = True,
    ) -> None:
        self.ontology = ontology or {}
        self.relation_realization_map = relation_realization_map or {}
        self.fact_index = fact_index or {}
        self.get_source_passages = get_source_passages
        self.max_schema_relations = max(int(max_schema_relations), 1)
        self.max_facts_per_schema = max(int(max_facts_per_schema), 1)
        self.max_facts_per_step_query = max(int(max_facts_per_step_query), 1)
        self.max_candidates = max(int(max_candidates), 0)
        self.min_score = min(max(float(min_score), 0.0), 1.0)
        self.preserve_top_k = max(int(preserve_top_k), 0)
        self.candidate_only = bool(candidate_only)
        self.append_score_margin = max(float(append_score_margin), 0.0)
        self.require_anchor_overlap = bool(require_anchor_overlap)
        self.min_relation_token_overlap = min(max(float(min_relation_token_overlap), 0.0), 1.0)
        self.enable_bridge_paths = bool(enable_bridge_paths)

        self.schema_steps: Dict[str, Dict[str, Any]] = {}
        self.plan_to_step_ids: Dict[str, List[str]] = defaultdict(list)
        self.signature_to_step_ids: Dict[str, Set[str]] = defaultdict(set)
        self.family_to_step_ids: Dict[str, Set[str]] = defaultdict(set)
        self.token_to_step_ids: Dict[str, Set[str]] = defaultdict(set)
        self.step_to_fact_indices: Dict[str, List[int]] = defaultdict(list)
        self.entity_to_step_fact_indices: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
        self.fact_to_step_ids: Dict[int, Set[str]] = defaultdict(set)
        self._step_bridge_entity_index_cache: Dict[str, Dict[str, List[int]]] = {}
        self._fact_source_passages_cache: Dict[int, Set[int]] = {}
        self._build_projection()

    @classmethod
    def _norm(cls, value: Any) -> str:
        return text_processing(str(value or "").replace("_", " ").replace("-", " "))

    @classmethod
    def _tokens(cls, value: Any, max_tokens: int = 12) -> List[str]:
        output: List[str] = []
        seen = set()
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

    def _relation_aliases_for_step(self, plan_id: str, step_idx: int, step: Mapping[str, Any]) -> List[str]:
        aliases: List[Any] = []
        aliases.append(step.get("relation_family", ""))
        aliases.append(step.get("relation_signature", ""))
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
        return self._unique(aliases, max_items=96)

    def _build_schema_steps(self) -> None:
        plans = [
            plan for plan in self.ontology.get("plans", []) or []
            if isinstance(plan, Mapping) and plan.get("path")
        ]
        for plan in plans:
            if len(self.schema_steps) >= self.max_schema_relations:
                break
            plan_id = str(plan.get("id") or f"plan_{len(self.plan_to_step_ids)}")
            precision = self._safe_float(plan.get("effective_precision", plan.get("precision", 0.0)), 0.0)
            support = self._safe_float(plan.get("support", 0.0), 0.0)
            for step_idx, step in enumerate(plan.get("path", []) or []):
                if not isinstance(step, Mapping):
                    continue
                if len(self.schema_steps) >= self.max_schema_relations:
                    break
                step_id = f"{plan_id}:{step_idx}"
                aliases = self._relation_aliases_for_step(plan_id, step_idx, step)
                alias_tokens = {
                    token
                    for alias in aliases
                    for token in self._tokens(alias, max_tokens=8)
                }
                signature = self._norm(step.get("relation_signature", ""))
                family = self._norm(step.get("relation_family", ""))
                if not aliases and not signature and not family:
                    continue
                self.schema_steps[step_id] = {
                    "step_id": step_id,
                    "plan_id": plan_id,
                    "step_idx": int(step_idx),
                    "aliases": aliases,
                    "alias_tokens": alias_tokens,
                    "relation_signature": signature,
                    "relation_family": family,
                    "answer_type": self._norm(plan.get("answer_type", "")),
                    "precision": precision,
                    "support": support,
                    "triggers": list(plan.get("triggers", []) or []),
                }
                self.plan_to_step_ids[plan_id].append(step_id)
                if signature:
                    self.signature_to_step_ids[signature].add(step_id)
                    self.signature_to_step_ids[relation_signature(signature)].add(step_id)
                if family:
                    self.family_to_step_ids[family].add(step_id)
                for token in alias_tokens:
                    self.token_to_step_ids[token].add(step_id)

    def _candidate_step_ids_for_relation(self, relation: Any) -> Set[str]:
        relation_norm = self._norm(relation)
        relation_tokens = set(self._tokens(relation_norm, max_tokens=12))
        relation_sig = relation_signature(relation_norm)
        relation_family = _relation_family(relation_norm)
        candidates: Set[str] = set()
        candidates.update(self.signature_to_step_ids.get(relation_sig, set()))
        if relation_family:
            candidates.update(self.family_to_step_ids.get(relation_family, set()))
        for token in relation_tokens:
            candidates.update(self.token_to_step_ids.get(token, set()))
        matched: Set[str] = set()
        for step_id in candidates:
            step = self.schema_steps.get(step_id, {})
            alias_tokens = step.get("alias_tokens", set()) or set()
            if not alias_tokens and step_id in candidates:
                matched.add(step_id)
                continue
            overlap = relation_tokens.intersection(alias_tokens)
            token_score = len(overlap) / max(min(len(relation_tokens), len(alias_tokens)), 1)
            signature_match = relation_sig and relation_sig == relation_signature(step.get("relation_signature", ""))
            family_match = bool(relation_family and relation_family == step.get("relation_family"))
            if signature_match or (family_match and token_score >= 0.25) or token_score >= self.min_relation_token_overlap:
                matched.add(step_id)
        return matched

    def _build_projection(self) -> None:
        self._build_schema_steps()
        fact_triples = self.fact_index.get("fact_triples", []) or []
        for fact_idx, triple in enumerate(fact_triples):
            if triple is None or len(triple) != 3:
                continue
            subject, relation, obj = triple
            step_ids = self._candidate_step_ids_for_relation(relation)
            if not step_ids:
                continue
            endpoint_tokens = set(self._tokens(subject, max_tokens=12)) | set(self._tokens(obj, max_tokens=12))
            for step_id in step_ids:
                if len(self.step_to_fact_indices[step_id]) >= self.max_facts_per_schema:
                    continue
                self.step_to_fact_indices[step_id].append(int(fact_idx))
                self.fact_to_step_ids[int(fact_idx)].add(step_id)
                for token in endpoint_tokens:
                    self.entity_to_step_fact_indices[step_id][token].append(int(fact_idx))

    def projection_stats(self) -> Dict[str, Any]:
        projected_facts = len(self.fact_to_step_ids)
        fact_count = len(self.fact_index.get("fact_triples", []) or [])
        non_empty_steps = sum(1 for indices in self.step_to_fact_indices.values() if indices)
        return {
            "schema_step_count": len(self.schema_steps),
            "projected_step_count": non_empty_steps,
            "fact_count": fact_count,
            "projected_fact_count": projected_facts,
            "projection_coverage": round(projected_facts / max(fact_count, 1), 6),
        }

    def _anchors(self, query: str, query_plan_log: Mapping[str, Any]) -> List[str]:
        anchors: List[Any] = []
        anchors.extend(query_plan_log.get("anchor_entities", []) or [])
        for path in query_plan_log.get("sample_paths", []) or []:
            if isinstance(path, Mapping):
                anchors.append(path.get("anchor_entity", ""))
        for passage in query_plan_log.get("lifted_passages", []) or []:
            for context in passage.get("contexts", []) or []:
                if isinstance(context, Mapping):
                    anchors.append(context.get("anchor_entity", ""))
        cleaned = []
        for anchor in self._unique(anchors, max_items=24):
            if not QueryPlanOntologyStructuralCandidateRetriever.is_generic_anchor(anchor):
                cleaned.append(anchor)
        if cleaned:
            return cleaned



        tokens = self._tokens(query, max_tokens=10)
        return [" ".join(tokens[:4])] if tokens else []

    def _step_ids_for_runtime_plan(self, plan: Mapping[str, Any]) -> List[str]:
        plan_id = str(plan.get("id") or "")
        if plan_id and self.plan_to_step_ids.get(plan_id):
            return list(self.plan_to_step_ids.get(plan_id, []))
        output: List[str] = []
        for step in plan.get("path", []) or []:
            if not isinstance(step, Mapping):
                continue
            relation_values = []
            relation_values.append(step.get("relation_signature", ""))
            relation_values.append(step.get("relation_family", ""))
            relation_values.extend(step.get("relation_aliases", []) or [])
            matched: Set[str] = set()
            for relation in relation_values:
                matched.update(self._candidate_step_ids_for_relation(relation))
            if matched:
                output.append(sorted(matched)[0])
        return output

    def _runtime_paths(self, query_plan_log: Mapping[str, Any]) -> List[Dict[str, Any]]:
        paths: List[Dict[str, Any]] = []
        for plan in query_plan_log.get("matched_plans", []) or []:
            if not isinstance(plan, Mapping):
                continue
            step_ids = self._step_ids_for_runtime_plan(plan)
            if not step_ids:
                continue
            paths.append({
                "plan_id": str(plan.get("id") or "matched_plan"),
                "step_ids": step_ids[:4],
                "precision": self._safe_float(plan.get("effective_precision", plan.get("precision", 0.0)), 0.0),
                "match_score": self._safe_float(plan.get("match_score", 0.0), 0.0),
                "source": "matched_plan",
            })
        for sample_path in query_plan_log.get("sample_paths", []) or []:
            if not isinstance(sample_path, Mapping):
                continue
            fact_indices = [int(idx) for idx in sample_path.get("fact_indices", []) or [] if str(idx).lstrip("-").isdigit()]
            step_ids: List[str] = []
            for fact_idx in fact_indices:
                for step_id in sorted(self.fact_to_step_ids.get(fact_idx, set())):
                    if step_id not in step_ids:
                        step_ids.append(step_id)
            if step_ids:
                paths.append({
                    "plan_id": str(sample_path.get("plan_id") or "sample_path"),
                    "step_ids": step_ids[:4],
                    "precision": self._safe_float(sample_path.get("plan_effective_precision", 0.0), 0.0),
                    "match_score": self._safe_float(sample_path.get("score", 0.0), 0.0),
                    "source": "sample_path",
                    "seed_fact_indices": fact_indices[:8],
                })
        if not paths:
            paths.extend(self._query_fallback_runtime_paths(query_plan_log))
        deduped: List[Dict[str, Any]] = []
        seen = set()
        for path in sorted(paths, key=lambda item: (len(item.get("step_ids", [])), item.get("precision", 0.0), item.get("match_score", 0.0)), reverse=True):
            key = tuple(path.get("step_ids", []))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(path)
            if len(deduped) >= 10:
                break
        return deduped

    def _query_fallback_runtime_paths(self, query_plan_log: Mapping[str, Any]) -> List[Dict[str, Any]]:
        """Infer lightweight projected-graph paths when no executable path exists.

        When the query-plan matcher has a useful query condition but no concrete
        sample path, use
        query/plan relation tokens to select ontology schema steps directly.
        This makes the projected graph an independent recall channel rather
        than a side-effect of already successful path instantiation.
        """
        query_condition = query_plan_log.get("query_condition", {}) or {}
        query_tokens = set(query_condition.get("query_tokens", []) or [])
        relation_values: List[Any] = []
        relation_values.extend(query_condition.get("relation_signatures", []) or [])
        relation_values.extend(query_condition.get("relation_families", []) or [])
        relation_values.extend(query_condition.get("relation_terms", []) or [])

        for plan in query_plan_log.get("matched_plans", []) or []:
            if not isinstance(plan, Mapping):
                continue
            relation_values.extend(plan.get("relation_signatures", []) or [])
            relation_values.extend(plan.get("relation_families", []) or [])
            relation_values.extend(plan.get("triggers", []) or [])
            for step in plan.get("path", []) or []:
                if not isinstance(step, Mapping):
                    continue
                relation_values.append(step.get("relation_signature", ""))
                relation_values.append(step.get("relation_family", ""))
                relation_values.extend(step.get("relation_aliases", []) or [])
                relation_values.extend(step.get("required_relation_terms", []) or [])

        step_scores: Dict[str, float] = defaultdict(float)
        for relation in relation_values:
            relation_tokens = set(self._tokens(relation, max_tokens=10))
            if not relation_tokens:
                continue
            for step_id in self._candidate_step_ids_for_relation(relation):
                step_scores[step_id] = max(step_scores[step_id], 0.72 + 0.04 * len(relation_tokens))

        if query_tokens:
            for step_id, step in self.schema_steps.items():
                alias_tokens = set(step.get("alias_tokens", set()) or set())
                if not alias_tokens:
                    continue
                overlap = alias_tokens.intersection(query_tokens)
                if not overlap:
                    continue
                overlap_score = len(overlap) / max(min(len(alias_tokens), len(query_tokens)), 1)
                if overlap_score >= 0.12:
                    precision_bonus = 0.08 * self._safe_float(step.get("precision"), 0.0)
                    step_scores[step_id] = max(step_scores[step_id], 0.52 + 0.28 * overlap_score + precision_bonus)

        if not step_scores:
            return []

        ranked_step_ids = [
            step_id
            for step_id, _ in sorted(
                step_scores.items(),
                key=lambda item: (
                    item[1],
                    self._safe_float(self.schema_steps.get(item[0], {}).get("precision"), 0.0),
                    self._safe_float(self.schema_steps.get(item[0], {}).get("support"), 0.0),
                ),
                reverse=True,
            )
        ][:8]

        output: List[Dict[str, Any]] = []
        seen = set()
        for plan_id, plan_step_ids in self.plan_to_step_ids.items():
            kept = [step_id for step_id in plan_step_ids if step_id in step_scores]
            if not kept:
                continue
            step_ids = kept[:2] if len(kept) >= 2 else kept[:1]
            key = tuple(step_ids)
            if key in seen:
                continue
            seen.add(key)
            score = max(step_scores.get(step_id, 0.0) for step_id in step_ids)
            precision = max(self._safe_float(self.schema_steps.get(step_id, {}).get("precision"), 0.0) for step_id in step_ids)
            output.append({
                "plan_id": str(plan_id),
                "step_ids": step_ids,
                "precision": precision,
                "match_score": min(score, 1.0),
                "source": "query_fallback",
            })
            if len(output) >= 4:
                break

        for step_id in ranked_step_ids:
            key = (step_id,)
            if key in seen:
                continue
            seen.add(key)
            step = self.schema_steps.get(step_id, {})
            output.append({
                "plan_id": str(step.get("plan_id") or "query_fallback"),
                "step_ids": [step_id],
                "precision": self._safe_float(step.get("precision"), 0.0),
                "match_score": min(step_scores.get(step_id, 0.0), 1.0),
                "source": "query_fallback",
            })
            if len(output) >= 8:
                break

        return output

    def _anchor_tokens(self, anchors: Sequence[str], query: str) -> Set[str]:
        tokens: Set[str] = set()
        for anchor in anchors:
            tokens.update(self._tokens(anchor, max_tokens=12))
        if not tokens:
            tokens.update(self._tokens(query, max_tokens=12))
        return tokens

    def _endpoint_overlap(self, triple: Tuple[str, str, str], anchor_tokens: Set[str], query_tokens: Set[str]) -> float:
        subject, _, obj = triple
        endpoint_tokens = set(self._tokens(subject, max_tokens=12)) | set(self._tokens(obj, max_tokens=12))
        if not endpoint_tokens:
            return 0.0
        anchor_overlap = len(endpoint_tokens.intersection(anchor_tokens)) / max(len(anchor_tokens), 1)
        query_overlap = len(endpoint_tokens.intersection(query_tokens)) / max(min(len(query_tokens), 12), 1)
        return min(max(0.75 * anchor_overlap + 0.25 * query_overlap, 0.0), 1.0)

    @staticmethod
    def _triple_entities(triple: Tuple[str, str, str]) -> Tuple[str, str]:
        return text_processing(triple[0]), text_processing(triple[2])

    def _projected_path_context_fields(
        self,
        fact_indices: Sequence[int],
        anchor_tokens: Set[str],
        query_tokens: Set[str],
        bridge: bool,
    ) -> Dict[str, Any]:
        fact_triples = self.fact_index.get("fact_triples", []) or []
        endpoint_records: List[Tuple[str, Set[str]]] = []
        triples_for_log: List[Tuple[str, str, str]] = []
        for fact_idx in fact_indices or []:
            if fact_idx < 0 or fact_idx >= len(fact_triples):
                continue
            triple = fact_triples[fact_idx]
            if triple is None or len(triple) != 3:
                continue
            triples_for_log.append((str(triple[0]), str(triple[1]), str(triple[2])))
            for entity in (triple[0], triple[2]):
                entity_text = text_processing(str(entity or ""))
                if not entity_text:
                    continue
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
            if bridge or (tokens and tokens.intersection(query_tokens)):
                terminal_entity = entity
                break

        fact_score_reasons = {}
        if bridge:
            fact_score_reasons["path_seed"] = 1
        if len(fact_indices or []) >= 2:
            fact_score_reasons["path_neighbor"] = len(fact_indices) - 1

        return {
            "anchor_entity": anchor_entity,
            "end_entity": terminal_entity,
            "fact_triples": triples_for_log[:4],
            "path_seed_hit": bool(bridge),
            "fact_score_reasons": fact_score_reasons,
        }

    def _bridge_entity_index_for_step(self, step_id: str) -> Dict[str, List[int]]:
        cached = self._step_bridge_entity_index_cache.get(step_id)
        if cached is not None:
            return cached

        fact_triples = self.fact_index.get("fact_triples", []) or []
        by_entity: Dict[str, List[int]] = defaultdict(list)
        for fact_idx in self.step_to_fact_indices.get(step_id, [])[: self.max_facts_per_schema]:
            if fact_idx < 0 or fact_idx >= len(fact_triples):
                continue
            triple = fact_triples[fact_idx]
            if triple is None:
                continue
            subject, obj = self._triple_entities(triple)
            by_entity[subject].append(int(fact_idx))
            by_entity[obj].append(int(fact_idx))

        output = dict(by_entity)
        self._step_bridge_entity_index_cache[step_id] = output
        return output

    def _cached_source_passages(self, fact_indices: Iterable[int]) -> Dict[int, Set[int]]:
        normalized: List[int] = []
        seen = set()
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

    def _facts_for_step(self, step_id: str, anchor_tokens: Set[str], query_tokens: Set[str]) -> List[Tuple[int, float]]:
        fact_triples = self.fact_index.get("fact_triples", []) or []
        candidate_indices: Set[int] = set()
        if anchor_tokens:
            token_index = self.entity_to_step_fact_indices.get(step_id, {})
            for token in anchor_tokens:
                candidate_indices.update(token_index.get(token, [])[: self.max_facts_per_step_query])
        if not candidate_indices:
            candidate_indices.update(self.step_to_fact_indices.get(step_id, [])[: self.max_facts_per_step_query])

        scored: List[Tuple[int, float]] = []
        for fact_idx in candidate_indices:
            if fact_idx < 0 or fact_idx >= len(fact_triples):
                continue
            triple = fact_triples[fact_idx]
            if triple is None:
                continue
            overlap = self._endpoint_overlap(triple, anchor_tokens, query_tokens)
            if self.require_anchor_overlap and overlap <= 0.0:
                continue
            scored.append((int(fact_idx), overlap))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[: self.max_facts_per_step_query]

    def _path_fact_sets(
        self,
        path: Mapping[str, Any],
        anchor_tokens: Set[str],
        query_tokens: Set[str],
    ) -> List[Dict[str, Any]]:
        fact_triples = self.fact_index.get("fact_triples", []) or []
        step_ids = list(path.get("step_ids", []) or [])
        if not step_ids:
            return []
        first_facts = self._facts_for_step(step_ids[0], anchor_tokens, query_tokens)
        if not first_facts:
            return []
        if len(step_ids) == 1 or not self.enable_bridge_paths:
            return [
                {
                    "fact_indices": [fact_idx],
                    "score": 0.55 + 0.35 * overlap,
                    "coverage": 1,
                    "bridge": False,
                }
                for fact_idx, overlap in first_facts[:80]
            ]

        second_step = step_ids[1]
        by_entity = self._bridge_entity_index_for_step(second_step)

        outputs: List[Dict[str, Any]] = []
        for first_idx, first_overlap in first_facts[:120]:
            first_triple = fact_triples[first_idx]
            if first_triple is None:
                continue
            endpoints = self._triple_entities(first_triple)
            linked: List[int] = []
            for endpoint in endpoints:
                linked.extend(by_entity.get(endpoint, [])[:16])
            seen_second = set()
            for second_idx in linked:
                if second_idx in seen_second or second_idx == first_idx:
                    continue
                seen_second.add(second_idx)
                second_triple = fact_triples[second_idx]
                if second_triple is None:
                    continue
                second_overlap = self._endpoint_overlap(second_triple, anchor_tokens, query_tokens)
                score = 0.68 + 0.16 * first_overlap + 0.08 * second_overlap + 0.08 * min(len(step_ids) / 2.0, 1.0)
                outputs.append({
                    "fact_indices": [int(first_idx), int(second_idx)],
                    "score": min(score, 1.0),
                    "coverage": 2,
                    "bridge": True,
                })
                if len(outputs) >= 200:
                    break
            if len(outputs) >= 200:
                break
        if outputs:
            outputs.sort(key=lambda item: item["score"], reverse=True)
            return outputs


        return [
            {
                "fact_indices": [fact_idx],
                "score": 0.50 + 0.30 * overlap,
                "coverage": 1,
                "bridge": False,
            }
            for fact_idx, overlap in first_facts[:60]
        ]

    def _source_candidates_for_path(
        self,
        path: Mapping[str, Any],
        anchor_tokens: Set[str],
        query_tokens: Set[str],
    ) -> List[Dict[str, Any]]:
        fact_sets = self._path_fact_sets(path, anchor_tokens, query_tokens)
        if not fact_sets:
            return []

        all_fact_indices: List[int] = []
        for fact_set in fact_sets:
            for fact_idx in fact_set.get("fact_indices", []) or []:
                try:
                    all_fact_indices.append(int(fact_idx))
                except Exception:
                    continue
        all_fact_to_passages = self._cached_source_passages(all_fact_indices)

        candidates: Dict[int, Dict[str, Any]] = {}
        for fact_set in fact_sets:
            fact_indices = list(fact_set.get("fact_indices", []) or [])
            if not fact_indices:
                continue
            bridge = bool(fact_set.get("bridge", False))
            path_context = self._projected_path_context_fields(
                fact_indices=fact_indices,
                anchor_tokens=anchor_tokens,
                query_tokens=query_tokens,
                bridge=bridge,
            )
            fact_to_passages = {
                int(fact_idx): all_fact_to_passages.get(int(fact_idx), set())
                for fact_idx in fact_indices
                if int(fact_idx) in all_fact_to_passages
            }
            if not fact_to_passages:
                continue
            passage_hits: Dict[int, List[int]] = defaultdict(list)
            for fact_idx, passage_indices in fact_to_passages.items():
                for passage_idx in passage_indices:
                    passage_hits[int(passage_idx)].append(int(fact_idx))
            for passage_idx, source_fact_indices in passage_hits.items():
                fact_coverage = len(set(source_fact_indices))
                score = float(fact_set.get("score", 0.0))
                score += 0.07 * min(fact_coverage, 2)
                score += 0.04 * self._safe_float(path.get("precision"), 0.0)
                score += 0.02 * min(self._safe_float(path.get("match_score"), 0.0), 1.0)
                existing = candidates.get(passage_idx)
                context = {
                    "projected_graph_evidence": True,
                    "schema_route_source": "projected_graph",
                    "support_key": f"projected_graph::{path.get('plan_id', '')}::{passage_idx}",
                    "plan_id": path.get("plan_id", "projected_graph"),
                    "anchor_entity": path_context.get("anchor_entity", ""),
                    "end_entity": path_context.get("end_entity", ""),
                    "schema_step_ids": list(path.get("step_ids", []) or []),
                    "relations": [
                        self.schema_steps.get(step_id, {}).get("relation_signature")
                        or self.schema_steps.get(step_id, {}).get("relation_family")
                        or step_id
                        for step_id in path.get("step_ids", []) or []
                    ],
                    "fact_indices": sorted(set(source_fact_indices)),
                    "fact_position": 0,
                    "signature_match_mode": "ontology_projected_graph",
                    "plan_effective_precision": path.get("precision", 0.0),
                    "path_score": path.get("match_score", 0.0),
                    "plan_lift_weight_scale": 0.96,
                    "score": min(score, 1.0),
                    "coverage": int(fact_set.get("coverage", 1)),
                    "bridge": bridge,
                    "source": path.get("source", "matched_plan"),
                    "fact_triples": path_context.get("fact_triples", []),
                    "path_seed_hit": path_context.get("path_seed_hit", False),
                    "fact_score_reasons": path_context.get("fact_score_reasons", {}),
                }
                candidate = {
                    "passage_idx": int(passage_idx),
                    "confidence": min(score, 1.0),
                    "score": min(score, 1.0),
                    "route": "projected_graph",
                    "matched_anchor": True,
                    "matched_terminal": bool(fact_set.get("bridge", False)),
                    "relation_hits": context["relations"],
                    "answer_hit": False,
                    "contexts": [context],
                }
                if existing is None or candidate["score"] > existing["score"]:
                    candidates[int(passage_idx)] = candidate
        output = sorted(candidates.values(), key=lambda item: item.get("score", 0.0), reverse=True)
        return output[: max(self.max_candidates * 4, self.max_candidates)]

    def retrieve(
        self,
        query: str,
        sorted_doc_ids: Sequence[int],
        sorted_doc_scores: Sequence[float],
        query_plan_log: Mapping[str, Any],
    ) -> QueryPlanProjectedGraphResult:
        base_doc_ids = [int(doc_id) for doc_id in list(sorted_doc_ids)]
        base_scores = np.array(sorted_doc_scores, dtype=np.float32)
        log: Dict[str, Any] = {
            "enabled": True,
            "candidate_only": self.candidate_only,
            "max_candidates": self.max_candidates,
            "min_score": self.min_score,
            "preserve_top_k": self.preserve_top_k,
            "projection": self.projection_stats(),
            "candidate_count": 0,
            "selected_count": 0,
            "added_candidates": [],
            "rejection_reasons": {},
        }
        if self.max_candidates <= 0:
            log["skip_reason"] = "disabled_by_zero_limit"
            return QueryPlanProjectedGraphResult(np.array(base_doc_ids, dtype=np.intp), base_scores, log)
        if not base_doc_ids:
            log["skip_reason"] = "empty_retrieval"
            return QueryPlanProjectedGraphResult(np.array(base_doc_ids, dtype=np.intp), base_scores, log)
        if not query_plan_log or not query_plan_log.get("enabled", False):
            log["skip_reason"] = "no_query_plan_log"
            return QueryPlanProjectedGraphResult(np.array(base_doc_ids, dtype=np.intp), base_scores, log)
        if not self.schema_steps or not self.fact_to_step_ids:
            log["skip_reason"] = "empty_projection"
            return QueryPlanProjectedGraphResult(np.array(base_doc_ids, dtype=np.intp), base_scores, log)

        runtime_paths = self._runtime_paths(query_plan_log)
        log["runtime_path_count"] = len(runtime_paths)
        log["runtime_paths"] = [
            {
                "plan_id": path.get("plan_id", ""),
                "step_ids": path.get("step_ids", []),
                "source": path.get("source", ""),
                "precision": path.get("precision", 0.0),
                "match_score": path.get("match_score", 0.0),
            }
            for path in runtime_paths[:8]
        ]
        if not runtime_paths:
            log["skip_reason"] = "no_runtime_projected_paths"
            return QueryPlanProjectedGraphResult(np.array(base_doc_ids, dtype=np.intp), base_scores, log)

        anchors = self._anchors(query, query_plan_log)
        anchor_tokens = self._anchor_tokens(anchors, query)
        query_tokens = set(self._tokens(query, max_tokens=24))
        log["anchors"] = anchors[:8]

        original_rank_by_doc = {int(doc_id): rank + 1 for rank, doc_id in enumerate(base_doc_ids)}
        preserved = set(base_doc_ids[: self.preserve_top_k])
        candidates_by_doc: Dict[int, Dict[str, Any]] = {}
        rejection_reasons: Dict[str, int] = {}
        for path in runtime_paths:
            for candidate in self._source_candidates_for_path(path, anchor_tokens, query_tokens):
                passage_idx = int(candidate["passage_idx"])
                if passage_idx in preserved:
                    rejection_reasons["preserved_top"] = rejection_reasons.get("preserved_top", 0) + 1
                    continue
                score = float(candidate.get("score", 0.0))
                if score < self.min_score:
                    rejection_reasons["low_score"] = rejection_reasons.get("low_score", 0) + 1
                    continue
                candidate["original_rank"] = int(original_rank_by_doc.get(passage_idx, 10**9))
                candidate["was_in_baseline"] = passage_idx in original_rank_by_doc
                existing = candidates_by_doc.get(passage_idx)
                if existing is None or score > float(existing.get("score", 0.0)):
                    candidates_by_doc[passage_idx] = candidate

        candidates = sorted(
            candidates_by_doc.values(),
            key=lambda item: (
                float(item.get("score", 0.0)),
                -int(item.get("original_rank", 10**9)),
            ),
            reverse=True,
        )
        selected = candidates[: self.max_candidates]
        log["candidate_count"] = len(candidates)
        log["selected_count"] = len(selected)
        log["added_candidates"] = selected
        log["rejection_reasons"] = dict(sorted(rejection_reasons.items(), key=lambda kv: kv[1], reverse=True)[:10])
        if not selected:
            log["skip_reason"] = "no_projected_graph_candidates"
            return QueryPlanProjectedGraphResult(np.array(base_doc_ids, dtype=np.intp), base_scores, log)

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
            return QueryPlanProjectedGraphResult(np.array(new_doc_ids, dtype=np.intp), new_scores, log)

        selected_ids = {int(item["passage_idx"]) for item in selected}
        preserved_ids = [int(doc_id) for doc_id in base_doc_ids[: self.preserve_top_k]]
        rest = [int(doc_id) for doc_id in base_doc_ids if int(doc_id) not in selected_ids and int(doc_id) not in preserved_ids]
        reordered = preserved_ids + [int(item["passage_idx"]) for item in selected] + rest
        final_ids: List[int] = []
        seen = set()
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
        return QueryPlanProjectedGraphResult(np.array(final_ids, dtype=np.intp), final_scores, log)
