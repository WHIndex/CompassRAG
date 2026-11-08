"""Failure-driven ontology subqueries with provenance-aware dependent joins."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

from .query_plan_bind_probe_join import QueryPlanOntologyBindProbeJoin
from .query_plan_factorized_executor import QueryPlanFactorizedExecutor


@dataclass
class MissingSlotExecutorResult:
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    groups: List[Dict[str, Any]] = field(default_factory=list)
    log: Dict[str, Any] = field(default_factory=dict)


class QueryPlanMissingSlotExecutor:
    """Recover failed query plans by binding missing ontology slots.

    Unlike a reranker, this operator can issue new dense probes. Intermediate
    passages are retained as provenance, while only the answer-producing
    terminal passage is exposed as a top-k candidate.
    """

    _GENERIC_RELATIONS = {
        "be", "been", "being", "from", "had", "has", "have", "in", "is",
        "of", "was", "were",
    }
    _FAILURE_REASONS = {
        "insufficient_query_predicate_coverage",
        "no_exact_query_anchor",
        "no_relational_access_path",
        "no_validated_candidates",
    }

    def __init__(
        self,
        dense_top_k: int = 64,
        max_probes: int = 12,
        max_bridge_entities: int = 6,
        max_candidates: int = 12,
        max_hops: int = 3,
        min_relation_score: float = 0.20,
        min_terminal_type_score: float = 0.30,
        min_candidate_utility: float = 0.38,
        trigger_reasons: Iterable[str] = (),
    ) -> None:
        self.dense_top_k = max(int(dense_top_k), 1)
        self.max_probes = max(int(max_probes), 1)
        self.max_bridge_entities = max(int(max_bridge_entities), 1)
        self.max_candidates = max(int(max_candidates), 1)
        self.max_hops = min(max(int(max_hops), 2), 4)
        self.min_relation_score = min(max(float(min_relation_score), 0.0), 1.0)
        self.min_terminal_type_score = min(
            max(float(min_terminal_type_score), 0.0),
            1.0,
        )
        self.min_candidate_utility = min(
            max(float(min_candidate_utility), 0.0),
            1.0,
        )
        configured = {
            str(value or "").strip().lower()
            for value in trigger_reasons or []
            if str(value or "").strip()
        }
        self.trigger_reasons = configured or set(self._FAILURE_REASONS)
        self._helper = QueryPlanFactorizedExecutor(
            dense_top_k=self.dense_top_k,
            max_probes=self.max_probes,
            max_candidates=self.max_candidates,
            max_hops=self.max_hops,
            beam_width=self.max_bridge_entities,
            min_relation_score=self.min_relation_score,
            min_type_score=self.min_terminal_type_score,
            min_candidate_utility=0.0,
            cost_weight=0.0,
            enable_typed_semantic_compiler=True,
            require_exact_entity_keys=False,
            require_typed_join_keys=True,
        )

    @staticmethod
    def _clamp(value: Any) -> float:
        try:
            return min(max(float(value), 0.0), 1.0)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _unique(values: Iterable[Any], limit: int = 16) -> List[str]:
        return QueryPlanFactorizedExecutor._unique(values, limit=limit)

    @classmethod
    def _relation_is_generic(cls, relation: Any) -> bool:
        tokens = set(QueryPlanFactorizedExecutor._identity_tokens(relation))
        return not tokens or tokens.issubset(cls._GENERIC_RELATIONS)

    @classmethod
    def _relation_equivalent(cls, left: Any, right: Any) -> bool:
        return max(
            QueryPlanFactorizedExecutor._typed_relation_overlap(left, right)[0],
            QueryPlanFactorizedExecutor._typed_relation_overlap(right, left)[0],
        ) >= 0.60

    @staticmethod
    def _humanize_relation(value: Any) -> str:
        relation = QueryPlanOntologyBindProbeJoin._norm(
            str(value or "").replace("_of_", " ").replace("_", " ")
        )
        for suffix in (" of person", " of entity", " of film", " of place"):
            if relation.endswith(suffix):
                relation = relation[: -len(suffix)].strip()
        return relation

    @classmethod
    def _surface_terminal_relation(
        cls,
        query: str,
        answer_type: str,
    ) -> Dict[str, Any]:
        query_norm = QueryPlanOntologyBindProbeJoin._norm(query)
        profile = QueryPlanOntologyBindProbeJoin._terminal_relation_profile(
            query,
            {"answer_type": answer_type},
        )
        if profile.get("resolved") and profile.get("canonical_relation"):
            return dict(profile)


        if " birthday " in f" {query_norm} " or query_norm.endswith(" birthday"):
            return {
                "resolved": True,
                "source": "missing_slot_surface_repair",
                "reason": "birthday_alias",
                "answer_type": "date",
                "canonical_relation": "date of birth",
                "aliases": ["date of birth", "birth date", "born on", "born"],
            }
        if any(cue in f" {query_norm} " for cue in (" country ", " nationality ")) and any(
            cue in f" {query_norm} " for cue in (" from ", " citizen ", " nationality ")
        ):
            return {
                "resolved": True,
                "source": "missing_slot_surface_repair",
                "reason": "country_from_alias",
                "answer_type": "location",
                "canonical_relation": "country of citizenship",
                "aliases": ["country of citizenship", "nationality", "citizen of", "from"],
            }
        if any(cue in f" {query_norm} " for cue in (" county ", " where ", " based in ")):
            return {
                "resolved": True,
                "source": "missing_slot_surface_repair",
                "reason": "location_alias",
                "answer_type": "location",
                "canonical_relation": "located in",
                "aliases": ["located in", "location", "county", "based in"],
            }
        return dict(profile)

    @classmethod
    def _plan_relation_paths(
        cls,
        query_plan_log: Mapping[str, Any],
    ) -> List[Dict[str, Any]]:
        paths: List[Dict[str, Any]] = []
        for plan in query_plan_log.get("matched_plans", []) or []:
            if not isinstance(plan, Mapping):
                continue
            relations: List[Any] = []
            for step in plan.get("path", []) or []:
                if isinstance(step, Mapping):
                    relations.append(
                        step.get("relation")
                        or step.get("relation_signature")
                        or step.get("relation_family")
                        or ""
                    )
                else:
                    relations.append(step)
            if not relations:
                relations = list(plan.get("relation_signatures", []) or [])
            if not relations:
                relations = list(plan.get("relation_families", []) or [])
            normalized = cls._unique(
                (cls._humanize_relation(value) for value in relations),
                limit=6,
            )
            if not normalized:
                continue
            paths.append({
                "relations": normalized,
                "source": "matched_plan",
                "plan_id": str(plan.get("id", "") or ""),
                "score": float(
                    plan.get("match_score", plan.get("effective_precision", 0.0)) or 0.0
                ),
            })
        for path in query_plan_log.get("sample_paths", []) or []:
            if not isinstance(path, Mapping):
                continue
            normalized = cls._unique(
                (cls._humanize_relation(value) for value in path.get("relations", []) or []),
                limit=6,
            )
            if normalized:
                paths.append({
                    "relations": normalized,
                    "source": "sample_path",
                    "plan_id": str(path.get("plan_id", "") or ""),
                    "score": float(path.get("score", 0.0) or 0.0),
                })
        paths.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
        return paths

    @classmethod
    def _compile_relation_paths(
        cls,
        query: str,
        query_plan_log: Mapping[str, Any],
        factorized_log: Mapping[str, Any],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        contract = factorized_log.get("typed_contract", {}) or {}
        answer_type = str(
            factorized_log.get("answer_type")
            or (query_plan_log.get("query_condition", {}) or {}).get("answer_type")
            or "entity"
        )
        terminal = cls._surface_terminal_relation(query, answer_type)
        terminal_relation = cls._humanize_relation(
            terminal.get("canonical_relation", "")
        )
        candidates: List[Dict[str, Any]] = []
        contract_relations = cls._unique(
            (cls._humanize_relation(value) for value in contract.get("relations", []) or []),
            limit=6,
        )
        if contract_relations:
            candidates.append({
                "relations": contract_relations,
                "source": "typed_contract",
                "plan_id": str(contract.get("source_plan_id", "") or ""),
                "score": float(contract.get("contract_confidence", 0.0) or 0.0),
            })
        candidates.extend(cls._plan_relation_paths(query_plan_log))

        compiled: List[Dict[str, Any]] = []
        seen: Set[Tuple[str, ...]] = set()
        for item in candidates:
            relations = list(item.get("relations", []) or [])[: cls._path_limit()]
            if terminal_relation:
                if len(relations) >= 2 and cls._relation_is_generic(relations[-1]):
                    relations[-1] = terminal_relation
                elif not any(
                    cls._relation_equivalent(terminal_relation, relation)
                    for relation in relations
                ):
                    relations.append(terminal_relation)
            relations = cls._semantic_unique(relations)[: cls._path_limit()]
            if len(relations) < 2:
                continue
            key = tuple(relations)
            if key in seen:
                continue
            seen.add(key)
            compiled.append({
                **dict(item),
                "relations": relations,
                "terminal_relation_repaired": bool(
                    terminal_relation and cls._relation_equivalent(relations[-1], terminal_relation)
                ),
            })
            if len(compiled) >= 4:
                break
        return compiled, terminal

    @classmethod
    def _path_limit(cls) -> int:
        return 4

    @staticmethod
    def _fit_relation_budget(relations: Sequence[Any], max_hops: int) -> List[str]:
        """Keep the answer-producing terminal relation when a path is truncated."""
        values = [str(value or "").strip() for value in relations if str(value or "").strip()]
        limit = max(int(max_hops), 2)
        if len(values) <= limit:
            return values
        return values[: limit - 1] + [values[-1]]

    @classmethod
    def _semantic_unique(cls, relations: Iterable[Any]) -> List[str]:
        output: List[str] = []
        for relation in cls._unique(relations, limit=8):
            if output and cls._relation_equivalent(relation, output[-1]):
                continue
            output.append(relation)
        return output

    @classmethod
    def _anchors(
        cls,
        query: str,
        query_plan_log: Mapping[str, Any],
        factorized_log: Mapping[str, Any],
    ) -> List[str]:
        values = (
            list((factorized_log.get("typed_contract", {}) or {}).get("anchors", []) or [])
            + list(factorized_log.get("anchors", []) or [])
            + list(query_plan_log.get("anchor_entities", []) or [])
        )
        exact = QueryPlanFactorizedExecutor._canonical_anchors(query, values)
        return exact or cls._unique(values, limit=6)

    @staticmethod
    def _existing_candidate_ids(query_plan_log: Mapping[str, Any]) -> Set[int]:
        ids: Set[int] = set()
        for key in (
            "bind_probe_join", "candidate_retriever", "evidence_index_retriever",
            "evidence_set_retriever", "failure_evidence_retriever",
            "missing_evidence_completer", "path_executor_retriever",
            "schema_slot_recall_expander", "structural_candidate_retriever",
            "support_evidence_retriever", "terminal_evidence_retriever",
        ):
            payload = query_plan_log.get(key, {}) or {}
            if not isinstance(payload, Mapping):
                continue
            for list_key in (
                "added_candidates", "candidate_decisions", "candidate_pool_samples",
                "promoted_passages", "selected_ontology_passages", "selected_passages",
            ):
                for item in payload.get(list_key, []) or []:
                    if not isinstance(item, Mapping):
                        continue
                    try:
                        ids.add(int(item.get("passage_idx")))
                    except (TypeError, ValueError):
                        continue
        return ids

    @staticmethod
    def _subquery(binding: str, relation: str, terminal: bool) -> str:
        if terminal:
            return f"What is the {relation} of {binding}?"
        return f"Who or what is connected to {binding} by {relation}?"

    def _scan_passages(
        self,
        passage_ids: Sequence[int],
        binding: str,
        relation: str,
        answer_type: str,
        terminal: bool,
        get_openie_record: Callable[[int], Mapping[str, Any]],
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for passage_idx in passage_ids:
            record = get_openie_record(int(passage_idx)) or {}
            for triple in record.get("extracted_triples", []) or []:
                if not isinstance(triple, (list, tuple)) or len(triple) != 3:
                    continue
                subject, predicate, obj = [str(value or "") for value in triple]
                profile = self._helper._score_entry(
                    {
                        "passage_idx": int(passage_idx),
                        "subject": subject,
                        "predicate": predicate,
                        "object": obj,
                        "entity_match_score": 0.85,
                    },
                    binding,
                    relation,
                    answer_type,
                    terminal,
                )
                if not profile.get("terminal_entity"):
                    continue
                if float(profile.get("relation_score", 0.0)) < self.min_relation_score:
                    continue
                if terminal and float(profile.get("type_score", 0.0)) < self.min_terminal_type_score:
                    continue
                profile["access_path"] = "protected_prefix_scan"
                rows.append(profile)
        return rows

    @staticmethod
    def _deduplicate_rows(rows: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
        best: Dict[Tuple[int, str], Dict[str, Any]] = {}
        for value in rows:
            row = dict(value)
            try:
                passage_idx = int(row.get("passage_idx", -1))
            except (TypeError, ValueError):
                continue
            terminal = QueryPlanOntologyBindProbeJoin._norm(row.get("terminal_entity", ""))
            if passage_idx < 0 or not terminal:
                continue
            key = (passage_idx, terminal)
            if key not in best or float(row.get("raw_score", 0.0)) > float(
                best[key].get("raw_score", 0.0)
            ):
                best[key] = row
        return sorted(
            best.values(),
            key=lambda item: (
                float(item.get("raw_score", 0.0)),
                float(item.get("relation_score", 0.0)),
                float(item.get("type_score", 0.0)),
            ),
            reverse=True,
        )

    def _probe(
        self,
        query: str,
        base_doc_ids: Sequence[int],
        binding: str,
        relation: str,
        answer_type: str,
        terminal: bool,
        dense_retrieve: Callable[[str], Tuple[Sequence[int], Sequence[float]]],
        get_openie_record: Callable[[int], Mapping[str, Any]],
        lookup_entity_relations: Callable[[str, int], Sequence[Mapping[str, Any]]],
        probes: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if len(probes) >= self.max_probes:
            return []
        subquery = self._subquery(binding, relation, terminal)
        indexed = self._helper._indexed_access(
            binding,
            relation,
            answer_type,
            terminal,
            lookup_entity_relations,
        )
        for row in indexed:
            row["access_path"] = "entity_relation_index"
        prefix_rows = self._scan_passages(
            list(base_doc_ids[:10]),
            binding,
            relation,
            answer_type,
            terminal,
            get_openie_record,
        )
        dense_rows, dense_log = self._helper._dense_access(
            subquery,
            binding,
            relation,
            answer_type,
            terminal,
            dense_retrieve,
            get_openie_record,
        )
        for row in dense_rows:
            row["access_path"] = "missing_slot_dense_probe"
        rows = self._deduplicate_rows(list(indexed) + prefix_rows + list(dense_rows))
        probes.append({
            "probe_id": len(probes),
            "subquery": subquery,
            "binding": binding,
            "relation": relation,
            "terminal": terminal,
            "index_match_count": len(indexed),
            "prefix_match_count": len(prefix_rows),
            "dense_count": int(dense_log.get("dense_count", 0) or 0),
            "dense_match_count": len(dense_rows),
            "matched_count": len(rows),
            "dense_error": str(dense_log.get("error", "") or ""),
        })
        return rows[: self.max_bridge_entities]

    def run(
        self,
        query: str,
        base_doc_ids: Sequence[int],
        query_plan_log: Mapping[str, Any],
        factorized_log: Mapping[str, Any],
        dense_retrieve: Callable[[str], Tuple[Sequence[int], Sequence[float]]],
        get_openie_record: Callable[[int], Mapping[str, Any]],
        lookup_entity_relations: Callable[[str, int], Sequence[Mapping[str, Any]]],
    ) -> MissingSlotExecutorResult:
        failure_reason = str(factorized_log.get("skip_reason", "") or "").lower()
        log: Dict[str, Any] = {
            "enabled": True,
            "triggered": False,
            "trigger_reason": failure_reason,
            "compiled_paths": [],
            "terminal_relation_profile": {},
            "probes": [],
            "candidate_count": 0,
            "group_count": 0,
            "new_action_space_candidate_count": 0,
            "skip_reason": "",
        }
        if failure_reason not in self.trigger_reasons:
            log["skip_reason"] = "factorized_route_not_recoverable"
            return MissingSlotExecutorResult(log=log)

        anchors = self._anchors(query, query_plan_log, factorized_log)
        compiled_paths, terminal_profile = self._compile_relation_paths(
            query,
            query_plan_log,
            factorized_log,
        )
        compiled_paths = [
            {
                **item,
                "relations": self._fit_relation_budget(
                    list(item.get("relations", []) or []),
                    self.max_hops,
                ),
            }
            for item in compiled_paths
        ]
        log["triggered"] = True
        log["anchors"] = anchors
        log["compiled_paths"] = compiled_paths
        log["terminal_relation_profile"] = terminal_profile
        if not anchors:
            log["skip_reason"] = "no_recovery_anchor"
            return MissingSlotExecutorResult(log=log)
        if not compiled_paths:
            log["skip_reason"] = "no_missing_slot_path"
            return MissingSlotExecutorResult(log=log)

        answer_type = str(
            terminal_profile.get("answer_type")
            or factorized_log.get("answer_type")
            or (query_plan_log.get("query_condition", {}) or {}).get("answer_type")
            or "entity"
        )
        existing_ids = self._existing_candidate_ids(query_plan_log)
        probes: List[Dict[str, Any]] = []
        terminal_states: List[Dict[str, Any]] = []
        for path_idx, path in enumerate(compiled_paths):
            relations = list(path.get("relations", []) or [])
            states: List[Dict[str, Any]] = [{
                "binding": anchor,
                "bindings": [anchor],
                "passage_indices": [],
                "triples": [],
                "step_profiles": [],
            } for anchor in anchors[:2]]
            for step_idx, relation in enumerate(relations):
                terminal = step_idx == len(relations) - 1
                next_states: List[Dict[str, Any]] = []
                for state in states:
                    rows = self._probe(
                        query,
                        base_doc_ids,
                        str(state["binding"]),
                        relation,
                        answer_type,
                        terminal,
                        dense_retrieve,
                        get_openie_record,
                        lookup_entity_relations,
                        probes,
                    )
                    for row in rows:
                        next_entity = str(row.get("terminal_entity", "") or "")
                        if not next_entity:
                            continue
                        if not terminal and not self._helper._valid_intermediate_join_key(next_entity):
                            continue
                        normalized = QueryPlanOntologyBindProbeJoin._norm(next_entity)
                        if normalized in {
                            QueryPlanOntologyBindProbeJoin._norm(value)
                            for value in state.get("bindings", [])
                        }:
                            continue
                        next_states.append({
                            "binding": next_entity,
                            "bindings": list(state["bindings"]) + [next_entity],
                            "passage_indices": list(state["passage_indices"]) + [
                                int(row["passage_idx"])
                            ],
                            "triples": list(state["triples"]) + [list(row.get("triple", []))],
                            "step_profiles": list(state["step_profiles"]) + [dict(row)],
                            "path_idx": path_idx,
                            "relations": relations,
                            "path_source": str(path.get("source", "") or ""),
                            "path_plan_id": str(path.get("plan_id", "") or ""),
                        })
                if not next_states:
                    states = []
                    break
                next_states.sort(
                    key=lambda item: sum(
                        float(row.get("raw_score", 0.0))
                        for row in item.get("step_profiles", [])
                    ) / max(len(item.get("step_profiles", [])), 1),
                    reverse=True,
                )
                states = next_states[: self.max_bridge_entities]
            terminal_states.extend(states)
            if len(probes) >= self.max_probes:
                break

        rank_by_doc = {int(doc_id): rank + 1 for rank, doc_id in enumerate(base_doc_ids)}
        groups: List[Dict[str, Any]] = []
        candidates: List[Dict[str, Any]] = []
        for state_idx, state in enumerate(terminal_states):
            profiles = list(state.get("step_profiles", []) or [])
            if len(profiles) < 2:
                continue
            terminal_profile = dict(profiles[-1])
            terminal_doc = int(terminal_profile.get("passage_idx", -1))
            if terminal_doc < 0 or terminal_doc not in rank_by_doc:
                continue
            scores = [float(item.get("raw_score", 0.0)) for item in profiles]
            relation_score = self._clamp(terminal_profile.get("relation_score"))
            type_score = self._clamp(terminal_profile.get("type_score"))
            chain_mean = sum(scores) / max(len(scores), 1)
            new_to_action_space = terminal_doc not in existing_ids
            probe_cost = len(profiles) + sum(
                1 for item in profiles if item.get("access_path") == "missing_slot_dense_probe"
            )
            utility = (
                0.38 * self._clamp(terminal_profile.get("raw_score"))
                + 0.22 * self._clamp(chain_mean)
                + 0.16 * relation_score
                + 0.14 * type_score
                + 0.10 * float(new_to_action_space)
                - 0.02 * min(probe_cost / 4.0, 1.0)
            )
            if utility < self.min_candidate_utility:
                continue
            passage_indices = list(dict.fromkeys(
                int(value) for value in state.get("passage_indices", [])
            ))
            group_id = f"missing_slot_dependent_join::{state_idx}"
            group = {
                "group_id": group_id,
                "operator": "missing_slot_dependent_join",
                "passage_indices": passage_indices,
                "provenance_passage_indices": passage_indices[:-1],
                "terminal_passage_idx": terminal_doc,
                "bindings": list(state.get("bindings", []) or []),
                "relations": list(state.get("relations", []) or []),
                "triples": list(state.get("triples", []) or []),
                "path_source": state.get("path_source", ""),
                "path_plan_id": state.get("path_plan_id", ""),
                "chain_completeness": 1.0,
                "terminal_only_projection": True,
                "new_to_action_space": new_to_action_space,
                "probe_cost": probe_cost,
                "utility": round(self._clamp(utility), 6),
            }
            groups.append(group)
            candidates.append({
                **terminal_profile,
                "passage_idx": terminal_doc,
                "operator": "missing_slot_dependent_join",
                "group_id": group_id,
                "anchor_entity": (state.get("bindings", []) or [""])[0],
                "bindings": list(state.get("bindings", []) or []),
                "relations": list(state.get("relations", []) or []),
                "answer_type": answer_type,
                "provenance_passage_indices": passage_indices[:-1],
                "terminal_only_projection": True,
                "new_to_action_space": new_to_action_space,
                "probe_cost": probe_cost,
                "utility": round(self._clamp(utility), 6),
                "original_rank": int(rank_by_doc.get(terminal_doc, 10**9)),
                "was_in_baseline": terminal_doc in rank_by_doc,
                "missing_slot_executor": True,
            })

        best_by_doc: Dict[int, Dict[str, Any]] = {}
        for candidate in candidates:
            doc_id = int(candidate["passage_idx"])
            if doc_id not in best_by_doc or float(candidate["utility"]) > float(
                best_by_doc[doc_id]["utility"]
            ):
                best_by_doc[doc_id] = candidate
        candidates = sorted(
            best_by_doc.values(),
            key=lambda item: (
                bool(item.get("new_to_action_space", False)),
                float(item.get("utility", 0.0)),
            ),
            reverse=True,
        )[: self.max_candidates]
        selected_groups = {
            str(item.get("group_id", "") or "") for item in candidates
        }
        groups = [
            group for group in groups
            if str(group.get("group_id", "") or "") in selected_groups
        ]
        log["probes"] = probes
        log["probe_count"] = len(probes)
        log["candidates"] = candidates
        log["groups"] = groups
        log["candidate_count"] = len(candidates)
        log["group_count"] = len(groups)
        log["new_action_space_candidate_count"] = sum(
            1 for item in candidates if item.get("new_to_action_space", False)
        )
        log["skip_reason"] = "candidate_only" if candidates else "no_recovered_terminal"
        return MissingSlotExecutorResult(candidates=candidates, groups=groups, log=log)
