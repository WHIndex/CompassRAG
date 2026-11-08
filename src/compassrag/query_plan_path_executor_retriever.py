import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

import numpy as np

from .query_plan_evidence_index_retriever import QueryPlanOntologyEvidenceIndexRetriever
from .utils.misc_utils import text_processing


@dataclass
class QueryPlanPathExecutorResult:
    sorted_doc_ids: np.ndarray
    sorted_doc_scores: np.ndarray
    log: Dict[str, Any] = field(default_factory=dict)


class QueryPlanOntologyPathExecutorRetriever(QueryPlanOntologyEvidenceIndexRetriever):
    """Execute matched ontology paths over the OpenIE fact graph.

    The evidence-index retriever is intentionally single-hop. It is precise, but
    it often stops at a related fact instead of following the schema path toward
    the missing support document. This retriever reuses the same slot->fact
    index, then runs a small beam search:

        query anchor -> ontology slot -> next entity -> next ontology slot ...

    Only the source passages of facts on executed chains are exposed downstream.
    The module is candidate-only by default so the existing evidence-set policy
    remains the final gate.
    """

    def __init__(
        self,
        ontology: Mapping[str, Any],
        relation_realization_map: Mapping[str, Any],
        fact_index: Mapping[str, Any],
        get_source_passages: Callable[[Iterable[int]], Dict[int, Set[int]]],
        max_schema_slots: int = 1024,
        max_facts_per_slot: int = 5000,
        max_facts_per_step: int = 160,
        max_runtime_paths: int = 8,
        max_candidates: int = 4,
        min_score: float = 0.78,
        preserve_top_k: int = 2,
        candidate_only: bool = True,
        append_score_margin: float = 1e-7,
        synthetic_original_rank: int = 35,
        beam_size: int = 8,
        max_steps: int = 3,
        min_relation_token_overlap: float = 0.35,
        enable_answer_type_completion: bool = False,
        enable_terminal_type_filter: bool = False,
        allow_query_condition_single: bool = True,
        require_terminal_type_for_single_step: bool = False,
        min_terminal_type_score: float = 0.45,
        direction_penalty: float = 1.0,
        max_relation_repeats: int = 0,
        strict_query_relation_binding: bool = False,
        min_query_relation_coverage: float = 1.0,
    ) -> None:
        super().__init__(
            ontology=ontology,
            relation_realization_map=relation_realization_map,
            fact_index=fact_index,
            get_source_passages=get_source_passages,
            max_schema_slots=max_schema_slots,
            max_facts_per_slot=max_facts_per_slot,
            max_facts_per_query_slot=max_facts_per_step,
            max_candidates=max_candidates,
            min_score=min_score,
            preserve_top_k=preserve_top_k,
            candidate_only=candidate_only,
            append_score_margin=append_score_margin,
            require_anchor_overlap=True,
            min_relation_token_overlap=min_relation_token_overlap,
            synthetic_original_rank=synthetic_original_rank,
        )
        self.max_facts_per_step = max(int(max_facts_per_step), 1)
        self.max_runtime_paths = max(int(max_runtime_paths), 1)
        self.beam_size = max(int(beam_size), 1)
        self.max_steps = max(int(max_steps), 1)
        self.enable_answer_type_completion = bool(enable_answer_type_completion)
        self.enable_terminal_type_filter = bool(enable_terminal_type_filter)
        self.allow_query_condition_single = bool(allow_query_condition_single)
        self.require_terminal_type_for_single_step = bool(require_terminal_type_for_single_step)
        self.min_terminal_type_score = min(max(float(min_terminal_type_score), 0.0), 1.0)
        self.direction_penalty = min(max(float(direction_penalty), 0.05), 1.0)
        self.max_relation_repeats = max(int(max_relation_repeats), 0)
        self.strict_query_relation_binding = bool(strict_query_relation_binding)
        self.min_query_relation_coverage = min(max(float(min_query_relation_coverage), 0.0), 1.0)
        self._last_runtime_rejection_reasons: Dict[str, int] = {}

    DATE_TOKENS = {
        "date", "day", "month", "year", "january", "february", "march", "april",
        "may", "june", "july", "august", "september", "october", "november",
        "december", "born", "birth", "died", "death",
    }
    LOCATION_TOKENS = {
        "city", "town", "county", "state", "country", "province", "location",
        "place", "village", "district", "region", "island", "river", "mountain",
        "park", "arena", "stadium", "university",
    }
    PERSON_TOKENS = {
        "person", "people", "actor", "artist", "director", "writer", "author",
        "singer", "composer", "performer", "player", "politician", "president",
    }
    ORG_TOKENS = {
        "organization", "organisation", "company", "label", "team", "group",
        "club", "agency", "school", "university", "publisher", "studio",
    }
    RELATION_STOP_TOKENS = {
        "a", "an", "and", "answer", "attribute", "by", "entity", "fact",
        "family", "for", "from", "has", "in", "is", "name", "of", "on",
        "person", "place", "query", "relation", "related", "the", "to",
        "type", "with", "year", "date", "time",
    }
    RELATION_TOKEN_ALIASES = {
        "born": "birth",
        "birthplace": "birth",
        "died": "death",
        "die": "death",
        "dead": "death",
        "deceased": "death",
        "mom": "mother",
        "mum": "mother",
        "dad": "father",
        "spouse": "spouse",
        "wife": "spouse",
        "husband": "spouse",
        "married": "spouse",
        "headquartered": "headquarters",
        "headquarter": "headquarters",
        "hq": "headquarters",
        "located": "location",
        "locate": "location",
        "capital": "capital",
        "director": "director",
        "directed": "director",
        "author": "author",
        "wrote": "author",
        "writer": "author",
        "producer": "producer",
        "produced": "producer",
        "label": "label",
        "publisher": "publisher",
        "released": "release",
        "release": "release",
        "founded": "founder",
        "founder": "founder",
    }
    RELATION_HINT_TOKENS = {
        "author", "birth", "capital", "child", "daughter", "death", "director",
        "father", "founder", "headquarters", "husband", "label", "location",
        "mother", "parent", "publisher", "release", "sibling", "sister",
        "son", "spouse", "wife", "writer", "country", "nationality",
        "occupation", "genre", "record", "language", "population", "owner",
        "member", "league", "team",
    }

    def _query_answer_type(self, query: str, query_plan_log: Mapping[str, Any]) -> str:
        condition = query_plan_log.get("query_condition", {}) or {}
        answer_type = self._norm(condition.get("answer_type", ""))
        if answer_type and answer_type != "entity":
            return answer_type
        query_tokens = set(self._tokens(query, max_tokens=24))
        if "when" in query_tokens:
            return "date"
        if "where" in query_tokens:
            return "location"
        if "who" in query_tokens or "whom" in query_tokens or "whose" in query_tokens:
            return "person"
        if "many" in query_tokens or "number" in query_tokens or "count" in query_tokens:
            return "number"
        return answer_type or "entity"

    def _canonical_relation_token(self, token: Any) -> str:
        text = self._norm(token)
        if not text:
            return ""
        return self.RELATION_TOKEN_ALIASES.get(text, text)

    def _relation_tokens_from_value(self, value: Any) -> Set[str]:
        output: Set[str] = set()
        for token in self._tokens(value, max_tokens=16):
            canonical = self._canonical_relation_token(token)
            if canonical and canonical not in self.RELATION_STOP_TOKENS:
                output.add(canonical)
        return output

    def _slot_relation_tokens(self, slot_id: str) -> Set[str]:
        slot = self.slots.get(slot_id, {}) or {}
        values: List[Any] = [
            slot.get("relation_signature", ""),
            slot.get("relation_family", ""),
        ]
        values.extend(slot.get("required_relation_terms", []) or [])
        values.extend(slot.get("aliases", []) or [])
        output: Set[str] = set()
        for value in values:
            output.update(self._relation_tokens_from_value(value))
        return output

    def _query_relation_tokens(self, query: str, query_plan_log: Mapping[str, Any]) -> Set[str]:
        condition = query_plan_log.get("query_condition", {}) or {}
        term_values: List[Any] = []
        signature_values: List[Any] = []
        if isinstance(condition, Mapping):
            signature_values.extend(condition.get("relation_signatures", []) or [])
            signature_values.extend(condition.get("relation_families", []) or [])
            term_values.extend(condition.get("relation_terms", []) or [])
        output: Set[str] = set()
        for value in term_values:
            output.update(self._relation_tokens_from_value(value))
        for value in signature_values:
            for token in self._relation_tokens_from_value(value):
                if token in self.RELATION_HINT_TOKENS:
                    output.add(token)
        for token in self._tokens(query, max_tokens=48):
            canonical = self._canonical_relation_token(token)
            if canonical in self.RELATION_HINT_TOKENS:
                output.add(canonical)
        return set(sorted(output)[:8])

    def _path_query_relation_coverage(
        self,
        slot_ids: Sequence[str],
        query_relation_tokens: Set[str],
    ) -> Tuple[float, List[str]]:
        if not query_relation_tokens:
            return 1.0, []
        path_tokens: Set[str] = set()
        for slot_id in slot_ids or []:
            path_tokens.update(self._slot_relation_tokens(slot_id))
        missing = sorted(token for token in query_relation_tokens if token not in path_tokens)
        coverage = (len(query_relation_tokens) - len(missing)) / max(len(query_relation_tokens), 1)
        return min(max(float(coverage), 0.0), 1.0), missing

    def _slot_answer_type_compatible(self, slot_id: str, answer_type: str) -> bool:
        if not answer_type or answer_type == "entity":
            return True
        slot = self.slots.get(slot_id, {})
        slot_answer_type = self._norm(slot.get("answer_type", ""))
        relation_text = " ".join([
            self._norm(slot.get("relation_signature", "")),
            self._norm(slot.get("relation_family", "")),
            " ".join(slot.get("terminal_object_allow_terms", []) or []),
        ])
        relation_tokens = set(self._tokens(relation_text, max_tokens=32))
        if slot_answer_type == answer_type:
            return True
        if answer_type in {"date", "time"}:
            return bool(relation_tokens.intersection(self.DATE_TOKENS) or "date" in slot_answer_type or "time" in slot_answer_type)
        if answer_type in {"location", "place"}:
            return bool(relation_tokens.intersection(self.LOCATION_TOKENS) or slot_answer_type in {"location", "place"})
        if answer_type == "person":
            return bool(relation_tokens.intersection(self.PERSON_TOKENS) or slot_answer_type == "person")
        if answer_type in {"number", "count"}:
            return "number" in slot_answer_type or "count" in slot_answer_type or bool({"number", "count", "population"}.intersection(relation_tokens))
        return True

    def _complete_slot_path(self, slot_id: str, answer_type: str) -> List[str]:
        if slot_id not in self.slots:
            return []
        slot = self.slots.get(slot_id, {})
        plan_id = str(slot.get("plan_id", ""))
        plan_slots = list(self.plan_to_slot_ids.get(plan_id, []) or [])
        if slot_id not in plan_slots:
            return [slot_id]
        start = plan_slots.index(slot_id)
        suffix = plan_slots[start: start + self.max_steps]
        if not suffix:
            return [slot_id]
        if len(suffix) == 1:
            return suffix
        for end in range(1, len(suffix)):
            if self._slot_answer_type_compatible(suffix[end], answer_type):
                return suffix[: end + 1]
        return suffix

    def _terminal_type_score(self, entity: Any, answer_type: str, slot_ids: Sequence[str]) -> float:
        answer_type = self._norm(answer_type)
        if not answer_type or answer_type == "entity":
            return 0.50
        text = self._norm(entity)
        tokens = set(self._tokens(text, max_tokens=16))
        if not text:
            return 0.0

        allow_tokens: Set[str] = set()
        reject_tokens: Set[str] = set()
        for slot_id in slot_ids or []:
            slot = self.slots.get(slot_id, {})
            for term in slot.get("terminal_object_allow_terms", []) or []:
                allow_tokens.update(self._tokens(term, max_tokens=8))
            for term in slot.get("terminal_object_reject_terms", []) or []:
                reject_tokens.update(self._tokens(term, max_tokens=8))
        if reject_tokens and tokens.intersection(reject_tokens):
            return 0.0
        allow_overlap = self._overlap_score(tokens, allow_tokens) if allow_tokens else 0.0

        if answer_type in {"date", "time"}:
            has_year = any(token.isdigit() and len(token) == 4 for token in tokens)
            has_month = bool(tokens.intersection(self.DATE_TOKENS - {"date", "day", "month", "year", "born", "birth", "died", "death"}))
            has_numeric = any(any(char.isdigit() for char in token) for token in tokens)
            return min(max(0.10 + 0.45 * float(has_year or has_month) + 0.25 * float(has_numeric) + 0.30 * allow_overlap, 0.0), 1.0)
        if answer_type in {"location", "place"}:
            comma_bonus = 0.20 if "," in str(entity or "") else 0.0
            type_bonus = 0.45 if tokens.intersection(self.LOCATION_TOKENS) else 0.0
            return min(max(0.18 + comma_bonus + type_bonus + 0.35 * allow_overlap, 0.0), 1.0)
        if answer_type == "person":
            non_generic = [token for token in tokens if token not in self.GENERIC_TERMS]
            name_like = 0.40 if len(non_generic) >= 2 else 0.18
            type_bonus = 0.25 if tokens.intersection(self.PERSON_TOKENS) else 0.0
            bad_type = bool(tokens.intersection(self.DATE_TOKENS | self.LOCATION_TOKENS | self.ORG_TOKENS))
            return min(max(name_like + type_bonus + 0.25 * allow_overlap - (0.25 if bad_type else 0.0), 0.0), 1.0)
        if answer_type in {"number", "count"}:
            has_number = any(any(char.isdigit() for char in token) for token in tokens)
            return 0.85 if has_number else min(0.35 + 0.30 * allow_overlap, 1.0)
        if answer_type in {"organization", "org"}:
            return min(max(0.25 + 0.40 * float(bool(tokens.intersection(self.ORG_TOKENS))) + 0.30 * allow_overlap, 0.0), 1.0)
        return min(max(0.35 + 0.40 * allow_overlap, 0.0), 1.0)

    @staticmethod
    def _as_fact_indices(value: Any, limit: int = 16) -> List[int]:
        output: List[int] = []
        seen: Set[int] = set()
        for item in value or []:
            try:
                idx = int(item)
            except Exception:
                continue
            if idx < 0 or idx in seen:
                continue
            output.append(idx)
            seen.add(idx)
            if len(output) >= limit:
                break
        return output

    def _runtime_paths(self, query: str, query_plan_log: Mapping[str, Any]) -> List[Dict[str, Any]]:
        paths: List[Dict[str, Any]] = []
        seen: Set[Tuple[str, ...]] = set()
        answer_type = self._query_answer_type(query, query_plan_log)
        query_relation_tokens = self._query_relation_tokens(query, query_plan_log)
        runtime_rejections: Counter = Counter()

        def add_path(slot_ids: Sequence[str], source: str, score: float, plan_id: str = "", precision: float = 0.0) -> None:
            cleaned = [slot_id for slot_id in slot_ids if slot_id in self.slots]
            if not cleaned:
                return
            cleaned = cleaned[: self.max_steps]
            query_relation_coverage, missing_query_relation_tokens = self._path_query_relation_coverage(
                cleaned,
                query_relation_tokens,
            )
            if (
                self.strict_query_relation_binding
                and query_relation_coverage < self.min_query_relation_coverage
            ):
                runtime_rejections["strict_query_relation_miss"] += 1
                return
            key = tuple(cleaned)
            if key in seen:
                return
            seen.add(key)
            paths.append({
                "slot_ids": cleaned,
                "source": source,
                "score": min(max(float(score), 0.0), 1.0),
                "plan_id": plan_id or str(self.slots.get(cleaned[0], {}).get("plan_id", "")),
                "plan_precision": min(max(float(precision), 0.0), 1.0),
                "answer_type": answer_type,
                "query_relation_tokens": sorted(query_relation_tokens),
                "query_relation_coverage": query_relation_coverage,
                "missing_query_relation_tokens": missing_query_relation_tokens,
            })

        for plan in query_plan_log.get("matched_plans", []) or []:
            if not isinstance(plan, Mapping):
                continue
            plan_id = str(plan.get("id") or "")
            slot_ids = self.plan_to_slot_ids.get(plan_id, [])
            precision = self._safe_float(plan.get("effective_precision", plan.get("precision", 0.0)), 0.0)
            match_score = self._safe_float(plan.get("match_score", 0.0), 0.0)
            if len(slot_ids) >= 2:
                add_path(slot_ids, "matched_plan_path", 0.76 + 0.10 * match_score + 0.08 * precision, plan_id, precision)
            elif slot_ids:
                add_path(slot_ids, "matched_plan_single", 0.70 + 0.08 * precision, plan_id, precision)
            for step in plan.get("path", []) or []:
                if not isinstance(step, Mapping):
                    continue
                relation_values: List[Any] = [
                    step.get("relation_signature", ""),
                    step.get("relation_family", ""),
                ]
                relation_values.extend(step.get("relation_aliases", []) or [])
                relation_values.extend(step.get("required_relation_terms", []) or [])
                step_slots: List[str] = []
                for relation in relation_values:
                    for slot_id in self._candidate_slot_ids_for_relation(relation):
                        if slot_id not in step_slots:
                            step_slots.append(slot_id)
                if step_slots:
                    for slot_id in step_slots[:2]:
                        completed = self._complete_slot_path(slot_id, answer_type) if self.enable_answer_type_completion else [slot_id]
                        if len(completed) > 1 or self.allow_query_condition_single:
                            add_path(completed, "matched_plan_relation", 0.68 + 0.08 * precision, plan_id, precision)

        for sample_path in query_plan_log.get("sample_paths", []) or []:
            if not isinstance(sample_path, Mapping):
                continue
            path_slot_ids: List[str] = []
            for fact_idx in self._as_fact_indices(sample_path.get("fact_indices", []), limit=8):
                for slot_id in sorted(self.fact_to_slot_ids.get(fact_idx, set())):
                    if slot_id not in path_slot_ids:
                        path_slot_ids.append(slot_id)
                        break
            if path_slot_ids:
                precision = self._safe_float(sample_path.get("plan_effective_precision", 0.0), 0.0)
                add_path(path_slot_ids, "sample_path_fact_chain", 0.80 + 0.08 * precision, precision=precision)

        query_condition = query_plan_log.get("query_condition", {}) or {}
        condition_slots: List[str] = []
        relation_values: List[Any] = []
        relation_values.extend(query_condition.get("relation_signatures", []) or [])
        relation_values.extend(query_condition.get("relation_families", []) or [])
        relation_values.extend(query_condition.get("relation_terms", []) or [])
        for relation in relation_values:
            for slot_id in self._candidate_slot_ids_for_relation(relation):
                if slot_id not in condition_slots:
                    condition_slots.append(slot_id)
        for slot_id in condition_slots[:4]:
            precision = self._safe_float(self.slots.get(slot_id, {}).get("precision"), 0.0)
            completed = self._complete_slot_path(slot_id, answer_type) if self.enable_answer_type_completion else [slot_id]
            if len(completed) > 1:
                add_path(completed, "query_condition_completed", 0.72 + 0.08 * precision, precision=precision)
            elif self.allow_query_condition_single:
                add_path([slot_id], "query_condition", 0.68 + 0.06 * precision, precision=precision)

        paths.sort(
            key=lambda item: (
                len(item.get("slot_ids", [])),
                float(item.get("score", 0.0)),
                float(item.get("plan_precision", 0.0)),
            ),
            reverse=True,
        )
        self._last_runtime_rejection_reasons = dict(runtime_rejections)
        return paths[: self.max_runtime_paths]

    def _entity_tokens_for_fact(self, fact_idx: int) -> Tuple[Set[str], Set[str], Tuple[str, str, str]]:
        fact_triples = self.fact_index.get("fact_triples", []) or []
        if fact_idx < 0 or fact_idx >= len(fact_triples):
            return set(), set(), ("", "", "")
        triple = fact_triples[fact_idx]
        if triple is None or len(triple) != 3:
            return set(), set(), ("", "", "")
        subject, relation, obj = triple
        return (
            set(self._tokens(subject, max_tokens=12)),
            set(self._tokens(obj, max_tokens=12)),
            (str(subject), str(relation), str(obj)),
        )

    @staticmethod
    def _overlap_score(tokens: Set[str], target: Set[str]) -> float:
        if not tokens or not target:
            return 0.0
        return len(tokens.intersection(target)) / max(min(len(tokens), len(target)), 1)

    def _facts_for_path_step(
        self,
        slot_id: str,
        current_tokens: Set[str],
        query_tokens: Set[str],
    ) -> List[Dict[str, Any]]:
        token_index = self.entity_to_slot_fact_indices.get(slot_id, {})
        candidate_indices: Set[int] = set()
        for token in current_tokens:
            candidate_indices.update(token_index.get(token, [])[: self.max_facts_per_step])
        if not candidate_indices:
            return []

        scored: List[Dict[str, Any]] = []
        for fact_idx in candidate_indices:
            subject_tokens, object_tokens, triple = self._entity_tokens_for_fact(int(fact_idx))
            if not triple[1]:
                continue
            slot = self.slots.get(slot_id, {})
            direction = self._norm(slot.get("direction", ""))
            subject_anchor = self._overlap_score(subject_tokens, current_tokens)
            object_anchor = self._overlap_score(object_tokens, current_tokens)
            anchor_score = max(subject_anchor, object_anchor)
            if anchor_score <= 0.0:
                continue
            direction_score = 1.0
            if direction == "forward":
                if subject_anchor > 0.0:
                    next_entity = triple[2]
                    next_tokens = object_tokens
                    direction_score = 1.0
                elif object_anchor > 0.0:
                    next_entity = triple[0]
                    next_tokens = subject_tokens
                    direction_score = self.direction_penalty
                else:
                    continue
            elif direction == "backward":
                if object_anchor > 0.0:
                    next_entity = triple[0]
                    next_tokens = subject_tokens
                    direction_score = 1.0
                elif subject_anchor > 0.0:
                    next_entity = triple[2]
                    next_tokens = object_tokens
                    direction_score = self.direction_penalty
                else:
                    continue
            elif subject_anchor >= object_anchor:
                next_entity = triple[2]
                next_tokens = object_tokens
            else:
                next_entity = triple[0]
                next_tokens = subject_tokens
            query_endpoint_score = max(
                self._overlap_score(subject_tokens, query_tokens),
                self._overlap_score(object_tokens, query_tokens),
            )
            step_score = min(max((0.76 * anchor_score + 0.24 * query_endpoint_score) * direction_score, 0.0), 1.0)
            scored.append({
                "fact_idx": int(fact_idx),
                "score": step_score,
                "anchor_score": anchor_score,
                "query_endpoint_score": query_endpoint_score,
                "direction_score": direction_score,
                "next_entity": text_processing(str(next_entity or "")),
                "next_tokens": set(next_tokens),
                "triple": triple,
            })
        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[: self.max_facts_per_step]

    def _execute_path(
        self,
        runtime_path: Mapping[str, Any],
        anchors: Sequence[str],
        query_tokens: Set[str],
    ) -> List[Dict[str, Any]]:
        slot_ids = list(runtime_path.get("slot_ids", []) or [])[: self.max_steps]
        if not slot_ids:
            return []
        states: List[Dict[str, Any]] = []
        for anchor in anchors:
            anchor_tokens = set(self._tokens(anchor, max_tokens=12))
            if not anchor_tokens:
                continue
            states.append({
                "anchor_entity": text_processing(str(anchor or "")),
                "current_entity": text_processing(str(anchor or "")),
                "current_tokens": anchor_tokens,
                "fact_indices": [],
                "fact_triples": [],
                "relations": [],
                "step_scores": [],
                "score": 0.0,
            })
        if not states:
            return []

        executed: List[Dict[str, Any]] = []
        for step_idx, slot_id in enumerate(slot_ids):
            next_states: List[Dict[str, Any]] = []
            for state in states:
                facts = self._facts_for_path_step(slot_id, state["current_tokens"], query_tokens)
                for fact in facts:
                    fact_idx = int(fact["fact_idx"])
                    if fact_idx in state["fact_indices"]:
                        continue
                    if not fact.get("next_tokens"):
                        continue
                    slot = self.slots.get(slot_id, {})
                    relation_label = slot.get("relation_signature") or slot.get("relation_family") or slot_id
                    existing_relations = list(state.get("relations", []))
                    if self.max_relation_repeats > 0:
                        relation_counts = defaultdict(int)
                        for relation in existing_relations:
                            relation_counts[self._norm(relation)] += 1
                        if relation_counts[self._norm(relation_label)] >= self.max_relation_repeats:
                            continue
                    slot_precision = self._safe_float(slot.get("precision"), 0.0)
                    step_score = float(fact.get("score", 0.0))
                    path_score = (
                        float(state.get("score", 0.0))
                        + step_score
                        + 0.08 * slot_precision
                        + (0.05 if step_idx > 0 else 0.0)
                    )
                    next_state = {
                        "anchor_entity": state.get("anchor_entity", ""),
                        "current_entity": fact.get("next_entity", ""),
                        "current_tokens": set(fact.get("next_tokens", set())),
                        "fact_indices": list(state.get("fact_indices", [])) + [fact_idx],
                        "fact_triples": list(state.get("fact_triples", [])) + [fact.get("triple")],
                        "relations": existing_relations + [relation_label],
                        "slot_ids": slot_ids[: step_idx + 1],
                        "step_scores": list(state.get("step_scores", [])) + [step_score],
                        "score": path_score,
                        "runtime_source": runtime_path.get("source", ""),
                        "plan_id": runtime_path.get("plan_id", ""),
                        "plan_precision": runtime_path.get("plan_precision", 0.0),
                        "answer_type": runtime_path.get("answer_type", ""),
                        "query_relation_tokens": list(runtime_path.get("query_relation_tokens", []) or []),
                        "query_relation_coverage": runtime_path.get("query_relation_coverage", 1.0),
                        "missing_query_relation_tokens": list(runtime_path.get("missing_query_relation_tokens", []) or []),
                    }
                    next_states.append(next_state)
            if not next_states:
                break
            next_states.sort(
                key=lambda item: (
                    float(item.get("score", 0.0)),
                    len(item.get("fact_indices", [])),
                ),
                reverse=True,
            )
            states = next_states[: self.beam_size]
            executed.extend(states)

        executed.sort(
            key=lambda item: (
                len(item.get("fact_indices", [])),
                float(item.get("score", 0.0)),
            ),
            reverse=True,
        )
        if self.enable_terminal_type_filter:
            filtered: List[Dict[str, Any]] = []
            for state in executed:
                terminal_score = self._terminal_type_score(
                    state.get("current_entity", ""),
                    state.get("answer_type", ""),
                    state.get("slot_ids", []) or [],
                )
                state["terminal_type_score"] = terminal_score
                if len(state.get("fact_indices", []) or []) <= 1 and self.require_terminal_type_for_single_step:
                    if terminal_score < self.min_terminal_type_score:
                        continue
                elif terminal_score < max(self.min_terminal_type_score * 0.70, 0.20):
                    continue
                filtered.append(state)
            executed = filtered
        return executed[: self.beam_size]

    def _source_candidates_for_state(
        self,
        state: Mapping[str, Any],
        query_tokens: Set[str],
        original_rank_by_doc: Mapping[int, int],
    ) -> List[Dict[str, Any]]:
        fact_indices = self._as_fact_indices(state.get("fact_indices", []), limit=8)
        if not fact_indices:
            return []
        fact_to_passages = self._cached_source_passages(fact_indices)
        if not fact_to_passages:
            return []

        passage_hits: Dict[int, List[int]] = defaultdict(list)
        for fact_idx, passage_indices in fact_to_passages.items():
            for passage_idx in passage_indices:
                passage_hits[int(passage_idx)].append(int(fact_idx))

        path_len = max(len(fact_indices), 1)
        step_scores = [float(value) for value in state.get("step_scores", []) or []]
        avg_step_score = sum(step_scores) / max(len(step_scores), 1)
        plan_precision = self._safe_float(state.get("plan_precision", 0.0), 0.0)
        terminal_tokens = set(self._tokens(state.get("current_entity", ""), max_tokens=12))
        terminal_overlap = self._overlap_score(terminal_tokens, query_tokens)
        terminal_type_score = self._safe_float(
            state.get("terminal_type_score"),
            self._terminal_type_score(state.get("current_entity", ""), state.get("answer_type", ""), state.get("slot_ids", []) or []),
        )
        if self.enable_terminal_type_filter and terminal_type_score < self.min_terminal_type_score:
            if path_len <= 1 or self.require_terminal_type_for_single_step:
                return []
        length_bonus = min(path_len, self.max_steps) / max(self.max_steps, 1)

        candidates: List[Dict[str, Any]] = []
        for passage_idx, passage_fact_indices in passage_hits.items():
            passage_fact_indices = sorted(set(passage_fact_indices))
            coverage = len(passage_fact_indices) / path_len
            score = (
                0.44
                + 0.16 * length_bonus
                + 0.14 * coverage
                + 0.12 * avg_step_score
                + 0.10 * plan_precision
                + 0.04 * terminal_overlap
                + 0.10 * terminal_type_score
            )
            score = min(max(score, 0.0), 1.0)
            if score < self.min_score:
                continue
            fact_triples = [
                triple for triple in state.get("fact_triples", []) or []
                if isinstance(triple, tuple) and len(triple) == 3
            ]
            context = {
                "ontology_path_executor": True,
                "schema_route_source": "ontology_path_executor",
                "support_key": f"ontology_path_executor::{state.get('plan_id', '')}::{passage_idx}",
                "plan_id": state.get("plan_id", "ontology_path_executor") or "ontology_path_executor",
                "slot_ids": list(state.get("slot_ids", []) or []),
                "anchor_entity": state.get("anchor_entity", ""),
                "end_entity": state.get("current_entity", ""),
                "relations": list(state.get("relations", []) or []),
                "fact_indices": fact_indices,
                "source_fact_indices": passage_fact_indices,
                "fact_position": max(len(fact_indices) - 1, 0),
                "signature_match_mode": "ontology_path_executor",
                "plan_effective_precision": max(plan_precision, score * 0.74),
                "plan_lift_weight_scale": 1.0,
                "score": score,
                "runtime_source": state.get("runtime_source", ""),
                "path_seed_hit": bool(state.get("anchor_entity")) and path_len >= 1,
                "path_terminal_hit": bool(terminal_overlap > 0.0),
                "answer_type": state.get("answer_type", ""),
                "terminal_type_score": terminal_type_score,
                "query_relation_tokens": list(state.get("query_relation_tokens", []) or []),
                "query_relation_coverage": self._safe_float(state.get("query_relation_coverage"), 1.0),
                "missing_query_relation_tokens": list(state.get("missing_query_relation_tokens", []) or []),
                "fact_triples": fact_triples[:5],
                "fact_score_reasons": {
                    "path_seed": 1,
                    "path_neighbor": max(path_len - 1, 0),
                    "path_length": path_len,
                    "source_coverage": round(coverage, 4),
                    "terminal_overlap": round(terminal_overlap, 4),
                    "terminal_type_score": round(terminal_type_score, 4),
                    "query_relation_coverage": round(self._safe_float(state.get("query_relation_coverage"), 1.0), 4),
                },
            }
            actual_rank = int(original_rank_by_doc.get(int(passage_idx), 10**9))
            candidates.append({
                "passage_idx": int(passage_idx),
                "confidence": score,
                "score": score,
                "route": "ontology_path_executor",
                "matched_anchor": bool(state.get("anchor_entity")),
                "matched_terminal": bool(terminal_overlap > 0.0 or terminal_type_score >= self.min_terminal_type_score or path_len >= 2),
                "relation_hits": list(state.get("relations", []) or []),
                "answer_hit": bool(terminal_overlap > 0.0 or terminal_type_score >= self.min_terminal_type_score),
                "slot_id": ",".join(state.get("slot_ids", []) or []),
                "slot_source": state.get("runtime_source", ""),
                "path_length": path_len,
                "answer_type": state.get("answer_type", ""),
                "terminal_type_score": terminal_type_score,
                "query_relation_tokens": list(state.get("query_relation_tokens", []) or []),
                "query_relation_coverage": self._safe_float(state.get("query_relation_coverage"), 1.0),
                "missing_query_relation_tokens": list(state.get("missing_query_relation_tokens", []) or []),
                "baseline_original_rank": actual_rank,
                "original_rank": min(actual_rank, self.synthetic_original_rank),
                "was_in_baseline": int(passage_idx) in original_rank_by_doc,
                "contexts": [context],
            })
        return candidates

    def retrieve(
        self,
        query: str,
        sorted_doc_ids: Sequence[int],
        sorted_doc_scores: Sequence[float],
        query_plan_log: Mapping[str, Any],
    ) -> QueryPlanPathExecutorResult:
        base_doc_ids = [int(doc_id) for doc_id in list(sorted_doc_ids)]
        base_scores = np.array(sorted_doc_scores, dtype=np.float32)
        log: Dict[str, Any] = {
            "enabled": True,
            "candidate_only": self.candidate_only,
            "max_candidates": self.max_candidates,
            "min_score": self.min_score,
            "preserve_top_k": self.preserve_top_k,
            "synthetic_original_rank": self.synthetic_original_rank,
            "beam_size": self.beam_size,
            "max_steps": self.max_steps,
            "enable_answer_type_completion": self.enable_answer_type_completion,
            "enable_terminal_type_filter": self.enable_terminal_type_filter,
            "allow_query_condition_single": self.allow_query_condition_single,
            "require_terminal_type_for_single_step": self.require_terminal_type_for_single_step,
            "min_terminal_type_score": self.min_terminal_type_score,
            "direction_penalty": self.direction_penalty,
            "max_relation_repeats": self.max_relation_repeats,
            "strict_query_relation_binding": self.strict_query_relation_binding,
            "min_query_relation_coverage": self.min_query_relation_coverage,
            "index": self.index_stats(),
            "runtime_path_count": 0,
            "executed_path_count": 0,
            "candidate_count": 0,
            "selected_count": 0,
            "added_candidates": [],
            "rejection_reasons": {},
        }
        if self.max_candidates <= 0:
            log["skip_reason"] = "disabled_by_zero_limit"
            return QueryPlanPathExecutorResult(np.array(base_doc_ids, dtype=np.intp), base_scores, log)
        if not base_doc_ids:
            log["skip_reason"] = "empty_retrieval"
            return QueryPlanPathExecutorResult(np.array(base_doc_ids, dtype=np.intp), base_scores, log)
        if not query_plan_log or not query_plan_log.get("enabled", False):
            log["skip_reason"] = "no_query_plan_log"
            return QueryPlanPathExecutorResult(np.array(base_doc_ids, dtype=np.intp), base_scores, log)
        if not self.slots or not self.fact_to_slot_ids:
            log["skip_reason"] = "empty_path_executor_index"
            return QueryPlanPathExecutorResult(np.array(base_doc_ids, dtype=np.intp), base_scores, log)

        runtime_paths = self._runtime_paths(query, query_plan_log)
        log["runtime_path_count"] = len(runtime_paths)
        log["runtime_paths"] = runtime_paths[:8]
        log["runtime_rejection_reasons"] = dict(getattr(self, "_last_runtime_rejection_reasons", {}) or {})
        if not runtime_paths:
            log["skip_reason"] = "no_runtime_paths"
            return QueryPlanPathExecutorResult(np.array(base_doc_ids, dtype=np.intp), base_scores, log)

        anchors = self._anchors(query, query_plan_log)
        query_tokens = set(self._tokens(query, max_tokens=32))
        log["anchors"] = anchors[:8]
        original_rank_by_doc = {int(doc_id): rank + 1 for rank, doc_id in enumerate(base_doc_ids)}
        preserved = set(base_doc_ids[: self.preserve_top_k])

        candidates_by_doc: Dict[int, Dict[str, Any]] = {}
        rejection_reasons: Dict[str, int] = {}
        executed_state_count = 0
        for runtime_path in runtime_paths:
            states = self._execute_path(runtime_path, anchors, query_tokens)
            executed_state_count += len(states)
            if not states:
                rejection_reasons["path_no_states"] = rejection_reasons.get("path_no_states", 0) + 1
                continue
            for state in states:
                for candidate in self._source_candidates_for_state(state, query_tokens, original_rank_by_doc):
                    passage_idx = int(candidate["passage_idx"])
                    if passage_idx in preserved:
                        rejection_reasons["preserved_top"] = rejection_reasons.get("preserved_top", 0) + 1
                        continue
                    score = float(candidate.get("score", 0.0))
                    existing = candidates_by_doc.get(passage_idx)
                    if existing is None or score > float(existing.get("score", 0.0)):
                        candidates_by_doc[passage_idx] = candidate

        candidates = sorted(
            candidates_by_doc.values(),
            key=lambda item: (
                float(item.get("score", 0.0)),
                int(item.get("path_length", 0)),
                -int(item.get("original_rank", 10**9)),
            ),
            reverse=True,
        )
        selected = candidates[: self.max_candidates]
        log["executed_path_count"] = executed_state_count
        log["candidate_count"] = len(candidates)
        log["selected_count"] = len(selected)
        log["added_candidates"] = selected
        log["rejection_reasons"] = dict(sorted(rejection_reasons.items(), key=lambda kv: kv[1], reverse=True)[:10])
        if not selected:
            log["skip_reason"] = "no_path_executor_candidates"
            return QueryPlanPathExecutorResult(np.array(base_doc_ids, dtype=np.intp), base_scores, log)

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
            return QueryPlanPathExecutorResult(np.array(new_doc_ids, dtype=np.intp), new_scores, log)

        selected_ids = {int(item["passage_idx"]) for item in selected}
        preserved_ids = [int(doc_id) for doc_id in base_doc_ids[: self.preserve_top_k]]
        rest = [int(doc_id) for doc_id in base_doc_ids if int(doc_id) not in selected_ids and int(doc_id) not in preserved_ids]
        reordered = preserved_ids + [int(item["passage_idx"]) for item in selected] + rest
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
        return QueryPlanPathExecutorResult(np.array(final_ids, dtype=np.intp), final_scores, log)
