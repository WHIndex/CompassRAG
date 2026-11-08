"""Proof-gap driven bound access paths for incomplete top-k evidence sets.

The operator is deliberately a candidate generator, not an admission rule. It
compares the current top-k passages with a query's logical evidence paths,
identifies uncovered relation slots, and executes bound probes for complete
paths. Every result carries its bindings and source passages so the learned
post-physical optimizer can make the final, holdout-calibrated decision.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

from .query_plan_bind_probe_join import QueryPlanOntologyBindProbeJoin
from .query_plan_factorized_executor import QueryPlanFactorizedExecutor
from .query_plan_clause_complete_lineage import (
    path_clause_profile,
    query_clause_profile,
)
from .query_plan_missing_slot_executor import (
    MissingSlotExecutorResult,
    QueryPlanMissingSlotExecutor,
)


class QueryPlanProofGapAccessPathExecutor(QueryPlanMissingSlotExecutor):
    """Execute ontology paths only when the current top-k has a proof gap."""

    OPERATOR = "proof_gap_bound_access_path"

    def __init__(
        self,
        *args: Any,
        top_window_k: int = 5,
        max_paths: int = 4,
        min_missing_slot_fraction: float = 0.25,
        min_binding_score: float = 0.80,
        min_proof_confidence: float = 0.20,
        require_external_member: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.top_window_k = max(int(top_window_k), 2)
        self.max_paths = max(int(max_paths), 1)
        self.min_missing_slot_fraction = self._clamp(min_missing_slot_fraction)
        self.min_binding_score = self._clamp(min_binding_score)
        self.min_proof_confidence = self._clamp(min_proof_confidence)
        self.require_external_member = bool(require_external_member)
        self._helper.require_exact_entity_keys = False
        self._helper.require_typed_join_keys = True

    @staticmethod
    def _answer_type(query_plan_log: Mapping[str, Any], factorized_log: Mapping[str, Any]) -> str:
        typed = factorized_log.get("typed_contract", {}) or {}
        condition = query_plan_log.get("query_condition", {}) or {}
        return str(
            factorized_log.get("answer_type")
            or typed.get("answer_type")
            or condition.get("answer_type")
            or "entity"
        )

    @classmethod
    def _path_relations(cls, value: Mapping[str, Any]) -> List[str]:
        relations: List[Any] = list(value.get("relations", []) or [])
        if not relations:
            for step in value.get("path", []) or []:
                if not isinstance(step, Mapping):
                    relations.append(step)
                    continue
                aliases = list(step.get("relation_aliases", []) or [])
                relations.append(
                    step.get("relation")
                    or step.get("relation_signature")
                    or step.get("relation_family")
                    or (aliases[0] if aliases else "")
                )
        return cls._semantic_unique(
            cls._humanize_relation(relation) for relation in relations
        )

    def _candidate_paths(
        self,
        query: str,
        query_plan_log: Mapping[str, Any],
        factorized_log: Mapping[str, Any],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
        answer_type = self._answer_type(query_plan_log, factorized_log)
        terminal = self._surface_terminal_relation(query, answer_type)
        terminal_relation = self._humanize_relation(
            terminal.get("canonical_relation", "")
        )
        typed = factorized_log.get("typed_contract", {}) or {}
        logical = factorized_log.get("logical_evidence_plan", {}) or {}
        condition = query_plan_log.get("query_condition", {}) or {}
        clause_profile = query_clause_profile(
            query,
            query_plan_log,
            factorized_log,
        )
        raw_paths: List[Dict[str, Any]] = []

        best_ontology_path = typed.get("best_ontology_path", {}) or {}
        if isinstance(best_ontology_path, Mapping):
            raw_paths.append({
                "relations": self._path_relations(best_ontology_path),
                "source": "typed_contract_ontology_path",
                "plan_id": str(best_ontology_path.get("plan_id", "") or ""),
                "score": float(best_ontology_path.get("schema_link_score", 0.0) or 0.0),
            })

        logical_relations = self._semantic_unique(
            self._humanize_relation(value)
            for value in logical.get("relations", []) or []
        )
        if logical_relations:
            raw_paths.append({
                "relations": logical_relations,
                "source": "logical_evidence_plan",
                "plan_id": str(logical.get("plan_id", "") or ""),
                "score": float(logical.get("contract_confidence", 0.0) or 0.0),
            })

        for plan in condition.get("compiler_schema_plans", []) or []:
            if not isinstance(plan, Mapping):
                continue
            plan_answer_type = str(plan.get("answer_type", "") or "")
            if (
                plan_answer_type
                and answer_type not in {"", "entity"}
                and plan_answer_type not in {"entity", answer_type}
            ):
                continue
            raw_paths.append({
                "relations": self._path_relations(plan),
                "source": "schema_linked_compiler",
                "plan_id": str(plan.get("id", "") or ""),
                "score": float(
                    plan.get("schema_link_score", plan.get("effective_precision", 0.0))
                    or 0.0
                ),
            })

        inherited_paths, inherited_terminal = self._compile_relation_paths(
            query,
            query_plan_log,
            factorized_log,
        )
        raw_paths.extend(inherited_paths)
        if not terminal.get("resolved") and inherited_terminal.get("resolved"):
            terminal = dict(inherited_terminal)
            terminal_relation = self._humanize_relation(
                terminal.get("canonical_relation", "")
            )

        compiled: List[Dict[str, Any]] = []
        seen: Set[Tuple[str, ...]] = set()
        for raw in raw_paths:
            relations = self._semantic_unique(raw.get("relations", []) or [])
            if terminal_relation and not any(
                self._relation_equivalent(terminal_relation, relation)
                for relation in relations
            ):
                relations.append(terminal_relation)
            relations = self._fit_relation_budget(relations, self.max_hops)
            if len(relations) < 2:
                continue
            key = tuple(QueryPlanOntologyBindProbeJoin._norm(value) for value in relations)
            if key in seen:
                continue
            seen.add(key)
            compiled.append({
                **dict(raw),
                "relations": relations,
                **path_clause_profile(relations, clause_profile),
            })

        compiled.sort(
            key=lambda item: (
                bool(item.get("query_clause_complete", False)),
                float(item.get("query_clause_coverage", 0.0) or 0.0),
                float(item.get("score", 0.0) or 0.0),
                len(item.get("relations", []) or []),
            ),
            reverse=True,
        )
        return compiled[: self.max_paths], terminal, clause_profile

    @staticmethod
    def _record_triples(record: Mapping[str, Any]) -> Iterable[Sequence[Any]]:
        for triple in record.get("extracted_triples", []) or []:
            if isinstance(triple, (list, tuple)) and len(triple) == 3:
                yield triple

    @classmethod
    def _relation_score(cls, required: str, observed: str) -> float:
        forward = QueryPlanFactorizedExecutor._typed_relation_overlap(required, observed)[0]
        reverse = QueryPlanFactorizedExecutor._typed_relation_overlap(observed, required)[0]
        return max(float(forward), float(reverse))

    def _top_window_profile(
        self,
        base_doc_ids: Sequence[int],
        relations: Sequence[str],
        anchors: Sequence[str],
        get_openie_record: Callable[[int], Mapping[str, Any]],
    ) -> Dict[str, Any]:
        observed_predicates: List[str] = []
        observed_text: List[str] = []
        for doc_id in list(base_doc_ids)[: self.top_window_k]:
            record = get_openie_record(int(doc_id)) or {}
            observed_text.append(str(record.get("passage", "") or ""))
            for subject, predicate, obj in self._record_triples(record):
                observed_predicates.append(str(predicate or ""))
                observed_text.extend([str(subject or ""), str(obj or "")])

        slot_scores = [
            max(
                (self._relation_score(relation, predicate) for predicate in observed_predicates),
                default=0.0,
            )
            for relation in relations
        ]
        covered_slots = [
            index for index, score in enumerate(slot_scores)
            if score >= self.min_relation_score
        ]
        normalized_text = QueryPlanOntologyBindProbeJoin._norm(" ".join(observed_text))
        covered_anchors = [
            index for index, anchor in enumerate(anchors)
            if QueryPlanOntologyBindProbeJoin._phrase_hit(
                normalized_text,
                anchor,
                min_coverage=0.80,
            )
        ]
        required_slot_count = len(relations)
        coverage = len(covered_slots) / max(required_slot_count, 1)
        return {
            "required_relation_slots": list(relations),
            "slot_scores": [round(value, 6) for value in slot_scores],
            "covered_slot_indices": covered_slots,
            "missing_slot_indices": [
                index for index in range(required_slot_count)
                if index not in set(covered_slots)
            ],
            "slot_coverage": round(coverage, 6),
            "missing_slot_fraction": round(1.0 - coverage, 6),
            "covered_anchor_indices": covered_anchors,
            "top_window_ids": [int(value) for value in list(base_doc_ids)[: self.top_window_k]],
        }

    @staticmethod
    def _dedupe_states(states: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
        best: Dict[Tuple[str, Tuple[int, ...]], Dict[str, Any]] = {}
        for state_value in states:
            state = dict(state_value)
            key = (
                QueryPlanOntologyBindProbeJoin._norm(state.get("binding", "")),
                tuple(int(value) for value in state.get("passage_indices", []) or []),
            )
            score = sum(
                float(profile.get("raw_score", 0.0) or 0.0)
                for profile in state.get("step_profiles", []) or []
            )
            incumbent = best.get(key)
            incumbent_score = sum(
                float(profile.get("raw_score", 0.0) or 0.0)
                for profile in (incumbent or {}).get("step_profiles", []) or []
            )
            if incumbent is None or score > incumbent_score:
                best[key] = state
        return list(best.values())

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
        anchors = self._anchors(query, query_plan_log, factorized_log)
        paths, terminal_profile, clause_profile = self._candidate_paths(
            query,
            query_plan_log,
            factorized_log,
        )
        log: Dict[str, Any] = {
            "enabled": True,
            "operator": self.OPERATOR,
            "triggered": False,
            "trigger_reason": "",
            "gold_free": True,
            "dataset_identity_used": False,
            "anchors": anchors,
            "compiled_paths": paths,
            "query_clause_profile": clause_profile,
            "terminal_relation_profile": terminal_profile,
            "top_window_k": self.top_window_k,
            "candidate_count": 0,
            "group_count": 0,
            "proof_valid_count": 0,
            "probes": [],
            "skip_reason": "",
        }
        if not anchors:
            log["skip_reason"] = "no_recovery_anchor"
            return MissingSlotExecutorResult(log=log)
        if not paths:
            log["skip_reason"] = "no_multihop_ontology_path"
            return MissingSlotExecutorResult(log=log)

        path_profiles: List[Dict[str, Any]] = []
        for path in paths:
            profile = self._top_window_profile(
                base_doc_ids,
                list(path.get("relations", []) or []),
                anchors,
                get_openie_record,
            )
            path_profiles.append({**dict(path), "top_window_proof": profile})
        eligible_paths = [
            path for path in path_profiles
            if float((path.get("top_window_proof", {}) or {}).get(
                "missing_slot_fraction", 0.0
            )) >= self.min_missing_slot_fraction
        ]
        log["path_profiles"] = path_profiles
        if not eligible_paths:
            log["skip_reason"] = "top_window_proof_complete"
            return MissingSlotExecutorResult(log=log)

        log["triggered"] = True
        log["trigger_reason"] = "top5_proof_gap"
        answer_type = self._answer_type(query_plan_log, factorized_log)
        top_window_ids = {int(value) for value in list(base_doc_ids)[: self.top_window_k]}
        rank_by_doc = {int(doc_id): rank + 1 for rank, doc_id in enumerate(base_doc_ids)}
        existing_ids = self._existing_candidate_ids(query_plan_log)
        probes: List[Dict[str, Any]] = []
        rejections: Counter[str] = Counter()
        groups: List[Dict[str, Any]] = []
        candidates: List[Dict[str, Any]] = []

        for path_index, path in enumerate(eligible_paths):
            relations = list(path.get("relations", []) or [])
            states: List[Dict[str, Any]] = [{
                "binding": anchor,
                "bindings": [anchor],
                "passage_indices": [],
                "step_profiles": [],
            } for anchor in anchors[:2]]
            for relation_index, relation in enumerate(relations):
                terminal = relation_index == len(relations) - 1
                next_states: List[Dict[str, Any]] = []
                for state in states:
                    rows = self._probe(
                        query,
                        base_doc_ids,
                        str(state.get("binding", "") or ""),
                        relation,
                        answer_type if terminal else "entity",
                        terminal,
                        dense_retrieve,
                        get_openie_record,
                        lookup_entity_relations,
                        probes,
                    )
                    for row in rows:
                        next_entity = str(row.get("terminal_entity", "") or "")
                        if not next_entity:
                            rejections["empty_output_binding"] += 1
                            continue
                        if (
                            not row.get("exact_entity_key", False)
                            and float(row.get("entity_score", 0.0) or 0.0)
                            < self.min_binding_score
                        ):
                            rejections["weak_dependent_binding"] += 1
                            continue
                        normalized = QueryPlanOntologyBindProbeJoin._norm(next_entity)
                        if normalized in {
                            QueryPlanOntologyBindProbeJoin._norm(value)
                            for value in state.get("bindings", []) or []
                        }:
                            rejections["cyclic_binding"] += 1
                            continue
                        next_states.append({
                            "binding": next_entity,
                            "bindings": list(state.get("bindings", []) or []) + [next_entity],
                            "passage_indices": list(state.get("passage_indices", []) or [])
                            + [int(row.get("passage_idx", -1))],
                            "step_profiles": list(state.get("step_profiles", []) or [])
                            + [dict(row)],
                        })
                if not next_states:
                    states = []
                    break
                next_states = self._dedupe_states(next_states)
                next_states.sort(
                    key=lambda state: min(
                        (
                            float(profile.get("relation_score", 0.0) or 0.0)
                            for profile in state.get("step_profiles", []) or []
                        ),
                        default=0.0,
                    ),
                    reverse=True,
                )
                states = next_states[: self.max_bridge_entities]
            before_profile = path.get("top_window_proof", {}) or {}
            before_coverage = float(before_profile.get("slot_coverage", 0.0) or 0.0)
            for state_index, state in enumerate(states):
                proof_docs = list(dict.fromkeys(
                    int(value)
                    for value in state.get("passage_indices", []) or []
                    if int(value) >= 0
                ))
                if not proof_docs:
                    continue
                external_docs = [doc_id for doc_id in proof_docs if doc_id not in top_window_ids]
                if self.require_external_member and not external_docs:
                    rejections["no_external_proof_member"] += 1
                    continue
                profiles = list(state.get("step_profiles", []) or [])
                relation_scores = [
                    self._clamp(profile.get("relation_score")) for profile in profiles
                ]
                binding_scores = [
                    self._clamp(profile.get("entity_score")) for profile in profiles
                ]
                terminal_type_score = self._clamp(
                    (profiles[-1] if profiles else {}).get("type_score")
                )
                provenance_coverage = sum(
                    bool(profile.get("triple")) and int(profile.get("passage_idx", -1)) >= 0
                    for profile in profiles
                ) / max(len(profiles), 1)
                proof_confidence = min(
                    min(relation_scores or [0.0]),
                    min(binding_scores or [0.0]),
                    terminal_type_score,
                    provenance_coverage,
                )
                if proof_confidence < self.min_proof_confidence:
                    rejections["proof_certificate_below_threshold"] += 1
                    continue
                group_id = f"{self.OPERATOR}::{path_index}::{state_index}"
                gap_gain = max(1.0 - before_coverage, 0.0)
                provenance_validated = bool(
                    profiles
                    and len(profiles) == len(relations)
                    and provenance_coverage >= 1.0
                )
                strict_chain_certificate = bool(
                    provenance_validated
                    and proof_confidence >= self.min_proof_confidence
                )
                physical_access_methods = list(dict.fromkeys(
                    str(profile.get("access_path", "") or "")
                    for profile in profiles
                    if str(profile.get("access_path", "") or "")
                ))
                group = {
                    "group_id": group_id,
                    "operator": self.OPERATOR,
                    "passage_indices": proof_docs,
                    "provenance_passage_indices": proof_docs,
                    "external_passage_indices": external_docs,
                    "bindings": list(state.get("bindings", []) or []),
                    "relations": relations,
                    "proof_edges": [
                        {
                            "step_idx": index,
                            "relation": relations[index],
                            "passage_idx": int(profile.get("passage_idx", -1)),
                            "triple": list(profile.get("triple", []) or []),
                            "input_binding": (state.get("bindings", []) or [""])[index],
                            "output_binding": (state.get("bindings", []) or ["", ""])[index + 1],
                            "relation_score": float(profile.get("relation_score", 0.0) or 0.0),
                            "type_score": float(profile.get("type_score", 0.0) or 0.0),
                            "access_path": str(profile.get("access_path", "") or ""),
                        }
                        for index, profile in enumerate(profiles)
                    ],
                    "proof_carrying": True,
                    "proof_valid": True,
                    "provenance_validated": provenance_validated,
                    "strict_chain_certificate": strict_chain_certificate,
                    "strict_chain_certificate_details": {
                        "source": self.OPERATOR,
                        "expected_step_count": len(relations),
                        "observed_step_count": len(profiles),
                        "provenance_coverage": provenance_coverage,
                        "minimum_proof_confidence": self.min_proof_confidence,
                    },
                    "terminal_passage_indices": proof_docs[-1:],
                    "triples": [
                        list(profile.get("triple", []) or [])
                        for profile in profiles
                        if len(list(profile.get("triple", []) or [])) == 3
                    ],
                    "physical_access_methods": physical_access_methods,
                    "mixed_access_path": len(physical_access_methods) > 1,
                    "schema_to_evidence_grounding": True,
                    "exact_anchor": bool(
                        profiles and profiles[0].get("exact_entity_key", False)
                    ),
                    "branch_complete": True,
                    "chain_completeness": 1.0,
                    "proof_relation_coverage": 1.0,
                    "proof_type_consistency": terminal_type_score,
                    "proof_provenance_coverage": provenance_coverage,
                    "proof_exact_binding_coverage": sum(
                        bool(profile.get("exact_entity_key", False)) for profile in profiles
                    ) / max(len(profiles), 1),
                    "proof_return_projection_valid": True,
                    "proof_member_count": len(proof_docs),
                    "full_proof_set_projection": True,
                    "new_to_action_space": any(doc_id not in existing_ids for doc_id in proof_docs),
                    "proof_gap_before": before_coverage,
                    "proof_gap_after": 1.0,
                    "proof_gap_gain": gap_gain,
                    "missing_slot_indices": list(before_profile.get("missing_slot_indices", []) or []),
                    "probe_cost": len(profiles) + sum(
                        profile.get("access_path") == "missing_slot_dense_probe"
                        for profile in profiles
                    ),
                    "proof_confidence": round(proof_confidence, 6),
                    "utility": round(proof_confidence, 6),
                    "path_source": str(path.get("source", "") or ""),
                    "path_plan_id": str(path.get("plan_id", "") or ""),
                    "query_required_clause_count": int(
                        path.get("query_required_clause_count", 1) or 1
                    ),
                    "query_clause_coverage": float(
                        path.get("query_clause_coverage", 0.0) or 0.0
                    ),
                    "query_clause_complete": bool(
                        path.get("query_clause_complete", False)
                    ),
                    "query_long_chain": bool(
                        clause_profile.get("long_chain", False)
                    ),
                }
                groups.append(group)
                for member_index, profile in enumerate(profiles):
                    doc_id = int(profile.get("passage_idx", -1))
                    if doc_id < 0:
                        continue
                    bindings = list(state.get("bindings", []) or [])
                    candidates.append({
                        **dict(profile),
                        "passage_idx": doc_id,
                        "operator": self.OPERATOR,
                        "group_id": group_id,
                        "proof_member_idx": member_index,
                        "proof_member_count": len(proof_docs),
                        "proof_relation_slot_index": member_index,
                        "proof_relation_slot_count": len(relations),
                        "proof_binding_entities": [
                            value
                            for value in (
                                bindings[member_index]
                                if member_index < len(bindings) else "",
                                bindings[member_index + 1]
                                if member_index + 1 < len(bindings) else "",
                            )
                            if value
                        ],
                        "proof_carrying": True,
                        "proof_valid": True,
                        "complete_provenance": bool(profile.get("triple")),
                        "role_consistent": bool(profile.get("exact_entity_key")),
                        "binding": str(profile.get("terminal_entity", "") or ""),
                        "answer_type": answer_type if member_index == len(profiles) - 1 else "entity",
                        "terminal_relation_match": member_index == len(profiles) - 1,
                        "proof_confidence": round(proof_confidence, 6),
                        "proof_gap_before": before_coverage,
                        "proof_gap_gain": gap_gain,
                        "full_proof_set_projection": True,
                        "new_to_action_space": doc_id not in existing_ids,
                        "utility": round(proof_confidence, 6),
                        "original_rank": int(rank_by_doc.get(doc_id, 10**9)),
                        "was_in_baseline": doc_id in rank_by_doc,
                        "query_required_clause_count": int(
                            path.get("query_required_clause_count", 1) or 1
                        ),
                        "query_clause_coverage": float(
                            path.get("query_clause_coverage", 0.0) or 0.0
                        ),
                        "query_clause_complete": bool(
                            path.get("query_clause_complete", False)
                        ),
                        "query_long_chain": bool(
                            clause_profile.get("long_chain", False)
                        ),
                    })

        groups.sort(
            key=lambda group: (
                float(group.get("proof_gap_gain", 0.0) or 0.0),
                float(group.get("proof_confidence", 0.0) or 0.0),
                -int(group.get("probe_cost", 0) or 0),
            ),
            reverse=True,
        )
        selected_groups: List[Dict[str, Any]] = []
        selected_group_ids: Set[str] = set()
        selected_member_count = 0
        for group in groups:
            member_count = len(group.get("passage_indices", []) or [])
            if selected_member_count + member_count > self.max_candidates:
                continue
            selected_groups.append(group)
            selected_group_ids.add(str(group.get("group_id", "") or ""))
            selected_member_count += member_count
        candidates = [
            candidate for candidate in candidates
            if str(candidate.get("group_id", "") or "") in selected_group_ids
        ]
        candidates.sort(
            key=lambda candidate: (
                float(candidate.get("proof_gap_gain", 0.0) or 0.0),
                float(candidate.get("proof_confidence", 0.0) or 0.0),
                -int(candidate.get("original_rank", 10**9) or 10**9),
            ),
            reverse=True,
        )
        log["probes"] = probes
        log["probe_count"] = len(probes)
        log["runtime_rejections"] = dict(rejections)
        log["candidates"] = candidates
        log["groups"] = selected_groups
        log["candidate_count"] = len(candidates)
        log["group_count"] = len(selected_groups)
        log["proof_valid_count"] = len(selected_groups)
        log["query_clause_complete_group_count"] = sum(
            bool(group.get("query_clause_complete", False))
            for group in selected_groups
        )
        log["new_action_space_candidate_count"] = sum(
            bool(group.get("new_to_action_space", False)) for group in selected_groups
        )
        log["skip_reason"] = "candidate_only" if selected_groups else "no_validated_gap_closure"
        return MissingSlotExecutorResult(
            candidates=candidates,
            groups=selected_groups,
            log=log,
        )
