"""Typed query compilation and proof-carrying dependent joins."""

from __future__ import annotations

import itertools
import re
from collections import Counter
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

from .query_plan_bind_probe_join import QueryPlanOntologyBindProbeJoin
from .query_plan_factorized_executor import QueryPlanFactorizedExecutor
from .query_plan_missing_slot_executor import (
    MissingSlotExecutorResult,
    QueryPlanMissingSlotExecutor,
)


class QueryPlanProofCarryingJoinExecutor(QueryPlanMissingSlotExecutor):
    """Recover missing evidence with a typed, inspectable logical plan.

    Every admitted candidate carries the complete sequence of bindings, triples,
    source passages, and type transitions that produced it. Comparison queries
    are materialized atomically: a proof is valid only when every branch has a
    complete path.
    """

    OPERATOR = "proof_carrying_dependent_join"

    _RELATION_SCHEMAS: Dict[str, Dict[str, Any]] = {
        "director": {
            "patterns": (r"\bdirect(?:ed|or)\b",),
            "input_types": {"work"},
            "output_type": "person",
        },
        "author": {
            "patterns": (r"\b(?:author|writer|written|wrote)\b",),
            "input_types": {"work"},
            "output_type": "person",
        },
        "producer": {
            "patterns": (r"\b(?:producer|produced|maker|manufacturer)\b",),
            "input_types": {"work", "organization"},
            "output_type": "entity",
        },
        "performer": {
            "patterns": (r"\b(?:performer|performed|starred|starring|acted)\b",),
            "input_types": {"work"},
            "output_type": "person",
        },
        "father": {
            "patterns": (r"\bfather\b",),
            "input_types": {"person"},
            "output_type": "person",
        },
        "mother": {
            "patterns": (r"\b(?:mother|maternal)\b",),
            "input_types": {"person"},
            "output_type": "person",
        },
        "spouse": {
            "patterns": (r"\b(?:spouse|wife|husband|married)\b",),
            "input_types": {"person"},
            "output_type": "person",
        },
        "birth_date": {
            "patterns": (
                r"\b(?:date of birth|birth date|birthday|born on|was born on)\b",
                r"^(?:was )?born$",
            ),
            "input_types": {"person"},
            "output_type": "date",
        },
        "birth_place": {
            "patterns": (r"\b(?:place of birth|birth place|born in|was born in)\b",),
            "input_types": {"person"},
            "output_type": "location",
        },
        "death_date": {
            "patterns": (r"\b(?:date of death|death date|died on)\b",),
            "input_types": {"person"},
            "output_type": "date",
        },
        "death_place": {
            "patterns": (r"\b(?:place of death|death place|died in)\b",),
            "input_types": {"person"},
            "output_type": "location",
        },
        "nationality": {
            "patterns": (
                r"\b(?:country of citizenship|citizen of|nationality|is from)\b",
            ),
            "input_types": {"person"},
            "output_type": "location",
        },
        "education": {
            "patterns": (r"\b(?:educated at|studied at|graduated from|attended)\b",),
            "input_types": {"person"},
            "output_type": "organization",
        },
        "located": {
            "patterns": (r"\b(?:located in|is located in|situated in|based in)\b",),
            "input_types": {"location", "organization", "entity"},
            "output_type": "location",
        },
        "capital": {
            "patterns": (r"\bcapital\b",),
            "input_types": {"location"},
            "output_type": "location",
        },
        "population": {
            "patterns": (r"\b(?:population|inhabitants)\b",),
            "input_types": {"location"},
            "output_type": "number",
        },
        "release_date": {
            "patterns": (r"\b(?:released|release date|published|premiered|aired)\b",),
            "input_types": {"work"},
            "output_type": "date",
        },
        "founding_date": {
            "patterns": (r"\b(?:founded|established|formation date|inception)\b",),
            "input_types": {"organization"},
            "output_type": "date",
        },
        "award": {
            "patterns": (r"\b(?:won|winner|award)\b",),
            "input_types": {"person", "work", "organization", "entity"},
            "output_type": "entity",
        },
        "membership": {
            "patterns": (r"\b(?:member of|membership|belongs to)\b",),
            "input_types": {"person", "organization"},
            "output_type": "organization",
        },
        "occupation": {
            "patterns": (r"\b(?:occupation|profession|job)\b",),
            "input_types": {"person"},
            "output_type": "entity",
        },
    }

    _WORK_DESCRIPTORS = {
        "album", "book", "film", "movie", "novel", "record", "series", "show", "song",
    }
    _LOCATION_DESCRIPTORS = {"city", "country", "county", "place", "state", "village"}

    def __init__(
        self,
        *args: Any,
        require_branch_complete_proof: bool = True,
        require_provenance_validation: bool = True,
        require_strict_branch_identity: bool = False,
        project_full_proof_set: bool = True,
        min_proof_confidence: float = 0.72,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.require_branch_complete_proof = bool(require_branch_complete_proof)
        self.require_provenance_validation = bool(require_provenance_validation)
        self.require_strict_branch_identity = bool(require_strict_branch_identity)
        self.project_full_proof_set = bool(project_full_proof_set)
        self.min_proof_confidence = self._clamp(min_proof_confidence)


        self._helper.require_exact_entity_keys = True
        self._helper.require_typed_join_keys = True

    @classmethod
    def _branch_identity_profile(
        cls,
        branch: Mapping[str, Any],
        row: Mapping[str, Any],
        source_record: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Validate a first-hop row against the exact query branch identity."""
        anchor = str(branch.get("entity_key", "") or "")
        anchor_key = QueryPlanFactorizedExecutor._entity_key(anchor)
        triple = list(row.get("triple", []) or [])
        endpoints = [str(triple[idx] or "") for idx in (0, 2) if len(triple) > idx]
        endpoint_keys = [
            QueryPlanFactorizedExecutor._entity_key(value)
            for value in endpoints
        ]
        exact_endpoint_match = bool(anchor_key and anchor_key in endpoint_keys)

        qualifiers = [
            str(value or "").strip()
            for value in branch.get("qualifiers", []) or []
            if str(value or "").strip()
        ]
        source_text = " ".join([
            str(source_record.get("passage", "") or ""),
            " ".join(endpoints),
        ]).strip()
        qualifier_matches = [
            QueryPlanOntologyBindProbeJoin._phrase_hit(
                source_text,
                qualifier,
                min_coverage=0.72,
            )
            for qualifier in qualifiers
        ]
        qualifier_coverage = (
            sum(bool(value) for value in qualifier_matches) / len(qualifier_matches)
            if qualifier_matches else 1.0
        )
        return {
            "branch_idx": int(branch.get("branch_idx", 0) or 0),
            "entity_key": anchor,
            "qualifiers": qualifiers,
            "triple_endpoints": endpoints,
            "exact_endpoint_match": exact_endpoint_match,
            "qualifier_matches": qualifier_matches,
            "qualifier_coverage": round(float(qualifier_coverage), 6),
            "valid": bool(exact_endpoint_match and qualifier_coverage >= 1.0 - 1e-9),
        }

    @classmethod
    def _relation_schema(cls, relation: Any) -> Dict[str, Any]:
        normalized = QueryPlanOntologyBindProbeJoin._norm(relation)
        for family, schema in cls._RELATION_SCHEMAS.items():
            if any(re.search(pattern, normalized) for pattern in schema["patterns"]):
                return {"family": family, "relation": normalized, **schema}
        return {}

    @staticmethod
    def _types_compatible(actual: str, expected: Set[str]) -> bool:
        actual = str(actual or "entity")
        if actual in expected:
            return True


        return actual == "entity" and "entity" in expected

    @classmethod
    def _predicate_matches_edge(cls, predicate: Any, edge: Mapping[str, Any]) -> bool:
        observed = cls._relation_schema(predicate)
        return bool(
            observed
            and observed.get("family") == edge.get("relation_family")
        )

    @classmethod
    def _runtime_output_type_valid(cls, value: Any, output_type: Any) -> bool:
        text = str(value or "").strip()
        normalized = QueryPlanOntologyBindProbeJoin._norm(text)
        expected = str(output_type or "entity").strip().lower()
        if not normalized:
            return False
        date_like = bool(
            re.search(r"\b[0-9]{3,4}\b", normalized)
            or re.search(
                r"\b(?:january|february|march|april|may|june|july|august|"
                r"september|october|november|december)\b",
                normalized,
            )
            or re.fullmatch(r"[0-9]{1,4}(?:[-/.][0-9]{1,2}){1,2}", normalized)
        )
        number_like = bool(re.search(r"[-+]?\d+(?:[.,]\d+)?", normalized))
        if expected == "date":
            return date_like
        if expected == "number":
            return number_like
        if expected in {"person", "location", "organization", "work"}:
            return not date_like and not re.fullmatch(r"[-+]?\d+(?:[.,]\d+)?", normalized)
        if expected == "boolean":
            return normalized in {"yes", "no", "true", "false"}
        return True

    @classmethod
    def _descriptor_type(cls, descriptor: str) -> str:
        descriptor = QueryPlanOntologyBindProbeJoin._norm(descriptor)
        if descriptor in cls._WORK_DESCRIPTORS:
            return "work"
        if descriptor in cls._LOCATION_DESCRIPTORS:
            return "location"
        if descriptor in {"person", "director", "author", "actor", "performer"}:
            return "person"
        return "entity"

    @classmethod
    def _return_contract(cls, query: str, comparison: bool) -> Dict[str, Any]:
        query_norm = QueryPlanOntologyBindProbeJoin._norm(query)
        which = re.search(
            r"\bwhich\s+(album|book|city|country|county|film|movie|novel|person|place|"
            r"record|series|show|song|state|village)\b",
            query_norm,
        )
        descriptor = which.group(1) if which else ""
        if comparison and descriptor:
            return {
                "variable": "branch_anchor",
                "projection": "branch_anchor",
                "answer_type": cls._descriptor_type(descriptor),
                "descriptor": descriptor,
            }
        if re.search(r"\b(?:when|what date|which date|what year|which year)\b", query_norm):
            answer_type = "date"
        elif re.search(r"\b(?:where|which country|what country|which city|what city)\b", query_norm):
            answer_type = "location"
        elif re.search(r"\b(?:how many|what number|which number)\b", query_norm):
            answer_type = "number"
        elif re.search(r"\bwho\b", query_norm):
            answer_type = "person"
        elif re.match(r"^(?:are|can|did|do|does|has|have|is|was|were)\b", query_norm):
            answer_type = "boolean"
        else:
            answer_type = cls._descriptor_type(descriptor)
        return {
            "variable": "terminal_value",
            "projection": "terminal_value",
            "answer_type": answer_type,
            "descriptor": descriptor,
        }

    @classmethod
    def _comparison_operator(cls, query: str) -> str:
        query_norm = QueryPlanOntologyBindProbeJoin._norm(query)
        if re.search(r"\b(?:earlier|older|before|lower|less)\b", query_norm):
            return "argmin"
        if re.search(r"\b(?:later|younger|after|higher|more)\b", query_norm):
            return "argmax"
        if re.search(r"\b(?:same|both|equal)\b", query_norm):
            return "equality"
        return "compare"

    @classmethod
    def _attribute_type(cls, query: str, return_contract: Mapping[str, Any]) -> str:
        query_norm = QueryPlanOntologyBindProbeJoin._norm(query)
        if re.search(r"\b(?:born|birth|birthday|died|death)\b", query_norm) and re.search(
            r"\b(?:earlier|later|older|younger|before|after|when|date)\b",
            query_norm,
        ):
            return "date"
        if re.search(r"\b(?:country|nationality|citizen|where|county)\b", query_norm):
            return "location"
        if re.search(r"\b(?:population|how many|higher|lower|more|less)\b", query_norm):
            return "number"
        return str(return_contract.get("answer_type", "entity") or "entity")

    @classmethod
    def _strict_surface_relations(cls, query: str, attribute_type: str) -> List[str]:
        relations = QueryPlanFactorizedExecutor._surface_relations(query, attribute_type)
        query_norm = QueryPlanOntologyBindProbeJoin._norm(query)
        if "birthday" in query_norm and not any(
            cls._relation_schema(value).get("family") == "birth_date" for value in relations
        ):
            relations.append("date of birth")
        if (
            re.search(r"\b(?:which|what) country\b", query_norm)
            and re.search(r"\b(?:from|nationality|citizen)\b", query_norm)
            and not any(cls._relation_schema(value).get("family") == "nationality" for value in relations)
        ):
            relations.append("country of citizenship")
        return cls._semantic_unique(relations)

    @classmethod
    def _typed_edges(cls, relations: Sequence[str]) -> Tuple[List[Dict[str, Any]], str]:
        edges: List[Dict[str, Any]] = []
        current_type = ""
        for step_idx, relation in enumerate(relations):
            schema = cls._relation_schema(relation)
            if not schema:
                return [], f"unknown_relation:{relation}"
            input_types = set(schema.get("input_types", set()) or set())
            if current_type and not cls._types_compatible(current_type, input_types):
                return [], (
                    f"type_discontinuity:{current_type}->{schema.get('family', '')}"
                )
            if not current_type:
                current_type = sorted(input_types)[0] if len(input_types) == 1 else (
                    "work" if "work" in input_types else "person" if "person" in input_types else "entity"
                )
            output_type = str(schema.get("output_type", "entity") or "entity")
            edges.append({
                "step_idx": step_idx,
                "relation": str(relation),
                "relation_family": str(schema.get("family", "")),
                "input_type": current_type,
                "allowed_input_types": sorted(input_types),
                "output_type": output_type,
            })
            current_type = output_type
        return edges, ""

    def _compile_logical_contract(
        self,
        query: str,
        query_plan_log: Mapping[str, Any],
        factorized_log: Mapping[str, Any],
    ) -> Dict[str, Any]:
        typed = dict(factorized_log.get("typed_contract", {}) or {})
        anchors = self._anchors(query, query_plan_log, factorized_log)
        comparison = bool(factorized_log.get("comparison_query", False))
        if not comparison:
            comparison = QueryPlanFactorizedExecutor._is_comparison(query, anchors)
        return_contract = self._return_contract(query, comparison)
        attribute_type = self._attribute_type(query, return_contract)
        relations = self._strict_surface_relations(query, attribute_type)
        relations = self._fit_relation_budget(relations, self.max_hops)
        edges, rejection = self._typed_edges(relations)

        branch_contract = dict(
            typed.get("comparison_branches", {})
            or factorized_log.get("comparison_branch_contract", {})
            or {}
        )
        if comparison and not branch_contract.get("complete", False):
            branch_contract = QueryPlanFactorizedExecutor._compile_comparison_branches(
                query,
                anchors,
            )
        if comparison:
            branches = [
                {
                    "branch_idx": int(item.get("branch_idx", idx) or idx),
                    "entity_key": str(item.get("entity_key", "") or ""),
                    "qualifiers": list(item.get("qualifiers", []) or []),
                }
                for idx, item in enumerate(branch_contract.get("branches", []) or [])
                if str(item.get("entity_key", "") or "")
            ]
        else:
            branches = [{"branch_idx": 0, "entity_key": anchors[0], "qualifiers": []}] if anchors else []

        if not anchors:
            rejection = rejection or "no_recovery_anchor"
        elif len(edges) < 2:
            rejection = rejection or "no_typed_multihop_path"
        elif comparison and self.require_branch_complete_proof and len(branches) != 2:
            rejection = rejection or "comparison_branch_incomplete"

        return {
            "enabled": True,
            "admissible": not rejection,
            "anchors": anchors,
            "relations": relations,
            "typed_edges": edges,
            "branch_contracts": branches,
            "expected_branch_count": 2 if comparison else 1,
            "comparison_query": comparison,
            "comparison_operator": self._comparison_operator(query) if comparison else "none",
            "return_contract": return_contract,
            "attribute_type": attribute_type,
            "return_projection_valid": bool(
                return_contract.get("projection") in {"branch_anchor", "terminal_value"}
            ),
            "type_consistent": bool(edges and not rejection),
            "source": "query_surface_typed_proof",
            "rejection_reason": rejection,
        }

    def _execute_branch(
        self,
        branch: Mapping[str, Any],
        contract: Mapping[str, Any],
        query: str,
        base_doc_ids: Sequence[int],
        dense_retrieve: Callable[[str], Tuple[Sequence[int], Sequence[float]]],
        get_openie_record: Callable[[int], Mapping[str, Any]],
        lookup_entity_relations: Callable[[str, int], Sequence[Mapping[str, Any]]],
        probes: List[Dict[str, Any]],
        rejections: Counter,
    ) -> List[Dict[str, Any]]:
        anchor = str(branch.get("entity_key", "") or "")
        states: List[Dict[str, Any]] = [{
            "binding": anchor,
            "bindings": [anchor],
            "passage_indices": [],
            "proof_edges": [],
            "step_profiles": [],
        }]
        typed_edges = list(contract.get("typed_edges", []) or [])
        for edge in typed_edges:
            terminal = int(edge.get("step_idx", 0) or 0) == len(typed_edges) - 1
            next_states: List[Dict[str, Any]] = []
            for state in states:
                rows = self._probe(
                    query,
                    base_doc_ids,
                    str(state["binding"]),
                    str(edge.get("relation", "") or ""),
                    str(edge.get("output_type", "entity") or "entity"),
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
                    if self.require_provenance_validation and not row.get("exact_entity_key", False):
                        rejections["inexact_binding_provenance"] += 1
                        continue
                    observed_triple = list(row.get("triple", []) or [])
                    observed_predicate = (
                        observed_triple[1]
                        if len(observed_triple) >= 2 else row.get("predicate", "")
                    )
                    if self.require_provenance_validation and not self._predicate_matches_edge(
                        observed_predicate,
                        edge,
                    ):
                        rejections["predicate_family_mismatch"] += 1
                        continue
                    type_score = QueryPlanOntologyBindProbeJoin._terminal_type_score(
                        next_entity,
                        edge.get("output_type", "entity"),
                    )
                    if type_score < self.min_terminal_type_score:
                        rejections["typed_output_mismatch"] += 1
                        continue
                    if not self._runtime_output_type_valid(
                        next_entity,
                        edge.get("output_type", "entity"),
                    ):
                        rejections["runtime_output_type_mismatch"] += 1
                        continue
                    branch_identity = dict(state.get("branch_identity", {}) or {})
                    if int(edge.get("step_idx", 0) or 0) == 0:
                        branch_identity = self._branch_identity_profile(
                            branch,
                            row,
                            get_openie_record(int(row.get("passage_idx", -1))) or {},
                        )
                        if self.require_strict_branch_identity and not branch_identity.get(
                            "valid", False
                        ):
                            if not branch_identity.get("exact_endpoint_match", False):
                                rejections["branch_identity_endpoint_mismatch"] += 1
                            if float(branch_identity.get("qualifier_coverage", 0.0)) < 1.0:
                                rejections["branch_identity_qualifier_mismatch"] += 1
                            continue
                    normalized = QueryPlanOntologyBindProbeJoin._norm(next_entity)
                    if normalized in {
                        QueryPlanOntologyBindProbeJoin._norm(value)
                        for value in state.get("bindings", [])
                    }:
                        rejections["cyclic_binding"] += 1
                        continue
                    proof_edge = {
                        "step_idx": int(edge.get("step_idx", 0) or 0),
                        "relation": str(edge.get("relation", "") or ""),
                        "relation_family": str(edge.get("relation_family", "") or ""),
                        "input_binding": str(state.get("binding", "") or ""),
                        "input_type": str(edge.get("input_type", "entity") or "entity"),
                        "output_binding": next_entity,
                        "output_type": str(edge.get("output_type", "entity") or "entity"),
                        "passage_idx": int(row.get("passage_idx", -1)),
                        "triple": list(row.get("triple", []) or []),
                        "exact_entity_key": bool(row.get("exact_entity_key", False)),
                        "relation_score": float(row.get("relation_score", 0.0) or 0.0),
                        "type_score": float(type_score),
                        "access_path": str(row.get("access_path", "") or ""),
                        "branch_identity_valid": bool(
                            branch_identity.get("valid", False)
                        ),
                    }
                    next_states.append({
                        "binding": next_entity,
                        "bindings": list(state.get("bindings", [])) + [next_entity],
                        "passage_indices": list(state.get("passage_indices", [])) + [
                            int(row.get("passage_idx", -1))
                        ],
                        "proof_edges": list(state.get("proof_edges", [])) + [proof_edge],
                        "step_profiles": list(state.get("step_profiles", [])) + [dict(row)],
                        "branch_idx": int(branch.get("branch_idx", 0) or 0),
                        "anchor_entity": anchor,
                        "branch_identity": branch_identity,
                    })
            if not next_states:
                return []
            next_states.sort(
                key=lambda item: sum(
                    float(profile.get("raw_score", 0.0) or 0.0)
                    for profile in item.get("step_profiles", [])
                ) / max(len(item.get("step_profiles", [])), 1),
                reverse=True,
            )
            states = next_states[: self.max_bridge_entities]
        return states[:2]

    @staticmethod
    def _proof_signature(states: Iterable[Mapping[str, Any]]) -> Tuple[int, ...]:
        return tuple(dict.fromkeys(
            int(value)
            for state in states
            for value in state.get("passage_indices", []) or []
            if int(value) >= 0
        ))

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
            "operator": self.OPERATOR,
            "triggered": False,
            "trigger_reason": failure_reason,
            "logical_contract": {},
            "probes": [],
            "candidate_count": 0,
            "group_count": 0,
            "new_action_space_candidate_count": 0,
            "proof_valid_count": 0,
            "runtime_rejections": {},
            "skip_reason": "",
        }
        if failure_reason not in self.trigger_reasons:
            log["skip_reason"] = "factorized_route_not_recoverable"
            return MissingSlotExecutorResult(log=log)

        contract = self._compile_logical_contract(query, query_plan_log, factorized_log)
        log["triggered"] = True
        log["logical_contract"] = contract
        log["compiled_paths"] = [{
            "relations": list(contract.get("relations", []) or []),
            "typed_edges": list(contract.get("typed_edges", []) or []),
            "source": contract.get("source", ""),
        }] if contract.get("typed_edges") else []
        if not contract.get("admissible", False):
            log["skip_reason"] = str(contract.get("rejection_reason", "typed_proof_rejected"))
            return MissingSlotExecutorResult(log=log)

        probes: List[Dict[str, Any]] = []
        rejections: Counter = Counter()
        branch_states: List[List[Dict[str, Any]]] = []
        for branch in contract.get("branch_contracts", []) or []:
            states = self._execute_branch(
                branch,
                contract,
                query,
                base_doc_ids,
                dense_retrieve,
                get_openie_record,
                lookup_entity_relations,
                probes,
                rejections,
            )
            if not states and self.require_branch_complete_proof:
                log["probes"] = probes
                log["probe_count"] = len(probes)
                log["runtime_rejections"] = dict(rejections)
                log["skip_reason"] = "incomplete_proof_branch"
                return MissingSlotExecutorResult(log=log)
            if states:
                branch_states.append(states)

        expected_branches = int(contract.get("expected_branch_count", 1) or 1)
        if len(branch_states) != expected_branches:
            log["probes"] = probes
            log["probe_count"] = len(probes)
            log["runtime_rejections"] = dict(rejections)
            log["skip_reason"] = "incomplete_proof_set"
            return MissingSlotExecutorResult(log=log)

        combinations = list(itertools.product(*branch_states))[:4]
        rank_by_doc = {int(doc_id): rank + 1 for rank, doc_id in enumerate(base_doc_ids)}
        existing_ids = self._existing_candidate_ids(query_plan_log)
        groups: List[Dict[str, Any]] = []
        candidates: List[Dict[str, Any]] = []
        seen_proofs: Set[Tuple[int, ...]] = set()
        for combo_idx, states in enumerate(combinations):
            proof_docs = self._proof_signature(states)
            if not proof_docs or proof_docs in seen_proofs:
                continue
            seen_proofs.add(proof_docs)
            if any(doc_id not in rank_by_doc for doc_id in proof_docs):
                rejections["proof_doc_outside_ranking"] += 1
                continue
            proof_edges = [
                dict(edge)
                for state in states
                for edge in state.get("proof_edges", []) or []
            ]
            terminal_docs = tuple(dict.fromkeys(
                int((state.get("proof_edges", []) or [])[-1].get("passage_idx", -1))
                for state in states
                if state.get("proof_edges")
                and int((state.get("proof_edges", []) or [])[-1].get("passage_idx", -1)) >= 0
            ))
            bridge_docs = tuple(doc_id for doc_id in proof_docs if doc_id not in set(terminal_docs))
            relation_scores = [float(edge.get("relation_score", 0.0)) for edge in proof_edges]
            type_scores = [float(edge.get("type_score", 0.0)) for edge in proof_edges]
            relation_coverage = len(proof_edges) / max(
                expected_branches * len(contract.get("typed_edges", []) or []),
                1,
            )
            provenance_coverage = sum(
                bool(edge.get("passage_idx", -1) >= 0 and edge.get("triple"))
                for edge in proof_edges
            ) / max(len(proof_edges), 1)
            exact_coverage = sum(bool(edge.get("exact_entity_key")) for edge in proof_edges) / max(
                len(proof_edges), 1
            )
            branch_identity_profiles = [
                dict(state.get("branch_identity", {}) or {})
                for state in states
            ]
            branch_identity_coverage = sum(
                bool(profile.get("valid", False))
                for profile in branch_identity_profiles
            ) / max(len(branch_identity_profiles), 1)
            branch_qualifier_coverage = sum(
                float(profile.get("qualifier_coverage", 0.0) or 0.0)
                for profile in branch_identity_profiles
            ) / max(len(branch_identity_profiles), 1)
            branch_complete = len(states) == expected_branches
            type_consistency = bool(contract.get("type_consistent", False)) and min(
                type_scores or [0.0]
            ) >= self.min_terminal_type_score
            return_projection_valid = bool(contract.get("return_projection_valid", False))
            proof_confidence = (
                0.24 * min(relation_scores or [0.0])
                + 0.18 * min(type_scores or [0.0])
                + 0.20 * relation_coverage
                + 0.16 * provenance_coverage
                + 0.12 * exact_coverage
                + 0.10 * float(branch_complete and return_projection_valid)
            )
            proof_valid = bool(
                branch_complete
                and type_consistency
                and return_projection_valid
                and relation_coverage >= 1.0
                and (
                    not self.require_provenance_validation
                    or (provenance_coverage >= 1.0 and exact_coverage >= 1.0)
                )
                and (
                    not self.require_strict_branch_identity
                    or (
                        branch_identity_coverage >= 1.0
                        and branch_qualifier_coverage >= 1.0
                    )
                )
                and proof_confidence >= self.min_proof_confidence
            )
            if not proof_valid:
                rejections["proof_validation_failed"] += 1
                continue
            new_to_action_space = any(doc_id not in existing_ids for doc_id in proof_docs)
            probe_cost = len(proof_edges) + sum(
                edge.get("access_path") == "missing_slot_dense_probe" for edge in proof_edges
            )
            utility = self._clamp(
                0.42 * proof_confidence
                + 0.18 * min(relation_scores or [0.0])
                + 0.12 * min(type_scores or [0.0])
                + 0.10 * float(new_to_action_space)
                + 0.10 * float(branch_complete)
                + 0.08 * provenance_coverage
                - 0.02 * min(probe_cost / 8.0, 1.0)
            )
            if utility < self.min_candidate_utility:
                rejections["proof_utility_below_threshold"] += 1
                continue
            group_id = f"{self.OPERATOR}::{combo_idx}"
            group = {
                "group_id": group_id,
                "operator": self.OPERATOR,
                "passage_indices": list(proof_docs),
                "provenance_passage_indices": list(proof_docs),
                "terminal_passage_indices": list(terminal_docs),
                "bridge_passage_indices": list(bridge_docs),
                "bindings": [list(state.get("bindings", []) or []) for state in states],
                "relations": list(contract.get("relations", []) or []),
                "proof_edges": proof_edges,
                "logical_contract": contract,
                "proof_carrying": True,
                "proof_valid": True,
                "branch_complete": branch_complete,
                "branch_count": len(states),
                "expected_branch_count": expected_branches,
                "branch_confidence": 1.0 if branch_complete else 0.0,
                "chain_completeness": relation_coverage,
                "proof_relation_coverage": relation_coverage,
                "proof_type_consistency": 1.0 if type_consistency else 0.0,
                "proof_provenance_coverage": provenance_coverage,
                "proof_exact_binding_coverage": exact_coverage,
                "proof_identity_required": self.require_strict_branch_identity,
                "proof_branch_identity_coverage": branch_identity_coverage,
                "proof_branch_qualifier_coverage": branch_qualifier_coverage,
                "proof_branch_identity_profiles": branch_identity_profiles,
                "proof_return_projection_valid": return_projection_valid,
                "proof_member_count": len(proof_docs),
                "return_projection": (contract.get("return_contract", {}) or {}).get("projection", ""),
                "terminal_only_projection": False,
                "full_proof_set_projection": self.project_full_proof_set,
                "new_to_action_space": new_to_action_space,
                "probe_cost": probe_cost,
                "proof_confidence": round(proof_confidence, 6),
                "utility": round(utility, 6),
            }
            groups.append(group)
            profile_by_doc: Dict[int, Dict[str, Any]] = {}
            for state in states:
                for profile in state.get("step_profiles", []) or []:
                    doc_id = int(profile.get("passage_idx", -1))
                    if doc_id >= 0 and (
                        doc_id not in profile_by_doc
                        or float(profile.get("raw_score", 0.0))
                        > float(profile_by_doc[doc_id].get("raw_score", 0.0))
                    ):
                        profile_by_doc[doc_id] = dict(profile)
            for member_idx, doc_id in enumerate(proof_docs):
                profile = dict(profile_by_doc.get(doc_id, {}))
                candidates.append({
                    **profile,
                    "passage_idx": doc_id,
                    "operator": self.OPERATOR,
                    "group_id": group_id,
                    "proof_member_idx": member_idx,
                    "proof_member_count": len(proof_docs),
                    "proof_carrying": True,
                    "proof_valid": True,
                    "proof_confidence": round(proof_confidence, 6),
                    "full_proof_set_projection": self.project_full_proof_set,
                    "new_to_action_space": new_to_action_space,
                    "probe_cost": probe_cost,
                    "utility": round(utility, 6),
                    "original_rank": int(rank_by_doc.get(doc_id, 10**9)),
                    "was_in_baseline": doc_id in rank_by_doc,
                })

        candidates.sort(
            key=lambda item: (
                bool(item.get("new_to_action_space", False)),
                float(item.get("proof_confidence", 0.0)),
                float(item.get("utility", 0.0)),
                -int(item.get("original_rank", 10**9)),
            ),
            reverse=True,
        )
        candidates = candidates[: self.max_candidates]
        selected_group_ids = {str(item.get("group_id", "")) for item in candidates}
        groups = [group for group in groups if group["group_id"] in selected_group_ids]
        log["probes"] = probes
        log["probe_count"] = len(probes)
        log["candidates"] = candidates
        log["groups"] = groups
        log["candidate_count"] = len(candidates)
        log["group_count"] = len(groups)
        log["proof_valid_count"] = len(groups)
        log["new_action_space_candidate_count"] = sum(
            bool(group.get("new_to_action_space")) for group in groups
        )
        log["runtime_rejections"] = dict(rejections)
        log["skip_reason"] = "candidate_only" if groups else "no_validated_proof"
        return MissingSlotExecutorResult(candidates=candidates, groups=groups, log=log)


def materialize_provenance_proof_view(
    group: Mapping[str, Any],
    max_edges: int = 8,
) -> Dict[str, Any]:
    """Project a certified proof into a compact branch-balanced reader view."""
    if not group.get("proof_valid", False) or not group.get("branch_complete", False):
        return {"valid": False, "reason": "proof_not_complete", "text": ""}
    if group.get("proof_identity_required", False) and float(
        group.get("proof_branch_identity_coverage", 0.0) or 0.0
    ) < 1.0:
        return {"valid": False, "reason": "branch_identity_incomplete", "text": ""}

    contract = dict(group.get("logical_contract", {}) or {})
    branch_contracts = list(contract.get("branch_contracts", []) or [])
    bindings = list(group.get("bindings", []) or [])
    proof_edges = [
        dict(edge)
        for edge in group.get("proof_edges", []) or []
        if isinstance(edge, Mapping)
    ]
    edge_budget = max(int(max_edges), 1)
    required_edge_count = sum(
        max(len(list(binding_chain or [])) - 1, 0)
        for binding_chain in bindings
    )
    if required_edge_count > edge_budget:
        return {"valid": False, "reason": "proof_view_edge_budget_exceeded", "text": ""}
    branches: List[Dict[str, Any]] = []
    used_edge_indices: Set[int] = set()
    for branch_position, binding_chain in enumerate(bindings):
        binding_chain = [str(value or "").strip() for value in binding_chain if str(value or "").strip()]
        if len(binding_chain) < 2:
            continue
        branch_contract = (
            dict(branch_contracts[branch_position])
            if branch_position < len(branch_contracts) else {}
        )
        branch_edges: List[Dict[str, Any]] = []
        for input_binding, output_binding in zip(binding_chain, binding_chain[1:]):
            input_key = QueryPlanFactorizedExecutor._entity_key(input_binding)
            output_key = QueryPlanFactorizedExecutor._entity_key(output_binding)
            matched_idx = -1
            for edge_idx, edge in enumerate(proof_edges):
                if edge_idx in used_edge_indices:
                    continue
                if (
                    QueryPlanFactorizedExecutor._entity_key(edge.get("input_binding", "")) == input_key
                    and QueryPlanFactorizedExecutor._entity_key(edge.get("output_binding", "")) == output_key
                ):
                    matched_idx = edge_idx
                    break
            if matched_idx < 0:
                return {"valid": False, "reason": "proof_edge_projection_gap", "text": ""}
            used_edge_indices.add(matched_idx)
            edge = proof_edges[matched_idx]
            branch_edges.append({
                "input": input_binding,
                "relation": str(edge.get("relation", "") or ""),
                "output": output_binding,
                "passage_idx": int(edge.get("passage_idx", -1) or -1),
            })
        qualifiers = [
            str(value or "").strip()
            for value in branch_contract.get("qualifiers", []) or []
            if str(value or "").strip()
        ]
        branches.append({
            "branch_idx": int(branch_contract.get("branch_idx", branch_position) or branch_position),
            "entity_key": str(branch_contract.get("entity_key", binding_chain[0]) or binding_chain[0]),
            "qualifiers": qualifiers,
            "bindings": binding_chain,
            "edges": branch_edges,
        })

    expected_branches = int(group.get("expected_branch_count", 1) or 1)
    if len(branches) != expected_branches or any(not item["edges"] for item in branches):
        return {"valid": False, "reason": "branch_projection_incomplete", "text": ""}

    branch_texts: List[str] = []
    for position, branch in enumerate(branches, start=1):
        qualifier_text = (
            f"; qualifier: {', '.join(branch['qualifiers'])}"
            if branch["qualifiers"] else ""
        )
        edge_text = " ; ".join(
            (
                f"{edge['input']} ->[{edge['relation']}]-> {edge['output']} "
                f"[source doc {edge['passage_idx']}]"
            )
            for edge in branch["edges"]
        )
        branch_texts.append(
            f"branch {position} ({branch['entity_key']}{qualifier_text}): {edge_text}"
        )
    return {
        "valid": True,
        "reason": "",
        "text": "Certified proof view: " + " || ".join(branch_texts),
        "branch_count": len(branches),
        "edge_count": sum(len(item["edges"]) for item in branches),
        "branches": branches,
        "proof_confidence": float(group.get("proof_confidence", 0.0) or 0.0),
        "provenance_passage_indices": list(
            group.get("provenance_passage_indices", []) or []
        ),
    }
