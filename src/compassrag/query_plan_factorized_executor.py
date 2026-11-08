import itertools
import re
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

from .ontology_evidence_query_optimizer import (
    CostBasedEvidenceQueryOptimizer,
    EvidenceCatalog,
    OntologyEvidenceQueryCompiler,
)
from .query_plan_bind_probe_join import QueryPlanOntologyBindProbeJoin
from .query_local_evidence_path_compiler import QueryLocalEvidencePathCompiler


@dataclass
class FactorizedQueryExecutorResult:
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    groups: List[Dict[str, Any]] = field(default_factory=list)
    log: Dict[str, Any] = field(default_factory=dict)


def select_typed_certificate_top2(
    groups: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
    rank_by_doc: Mapping[int, int],
    min_utility: float,
    min_semantic_score: float,
) -> Tuple[Dict[str, Any], Dict[str, Any], List[int], float]:
    """Project a complete typed proof into two head passages.

    Linear chains retain their anchor and terminal evidence. Comparison plans
    retain the terminal evidence of each branch. The projection is only
    eligible when all selected passages are already in the executable
    candidate universe and the proof passes both utility and semantic gates.
    """
    log: Dict[str, Any] = {
        "eligible": False,
        "reason": "no_complete_certificate",
        "head_passage_indices": [],
        "proof_passage_indices": [],
    }
    complete_groups = [
        dict(group)
        for group in groups
        if bool(
            dict(group.get("binding_certificate", {}) or {}).get(
                "complete", False
            )
        )
    ]
    complete_groups.sort(
        key=lambda group: float(group.get("utility", 0.0) or 0.0),
        reverse=True,
    )
    if not complete_groups:
        return log, {}, [], 0.0

    group = complete_groups[0]
    certificate = dict(group.get("binding_certificate", {}) or {})
    lineage = [
        dict(item)
        for item in certificate.get("lineage", []) or []
        if int(
            item.get("passage_idx")
            if item.get("passage_idx") is not None
            else -1
        ) in rank_by_doc
    ]
    head_docs: List[int] = []
    if str(certificate.get("topology", "") or "") == "dual_branch_comparison":
        branch_rows: Dict[str, List[Dict[str, Any]]] = {}
        for item in lineage:
            branch_id = str(item.get("branch_id", "") or "")
            if branch_id:
                branch_rows.setdefault(branch_id, []).append(item)
        for branch_id in sorted(branch_rows)[:2]:
            branch_lineage = sorted(
                branch_rows[branch_id],
                key=lambda item: int(item.get("edge_index", 0) or 0),
            )
            if branch_lineage:
                head_docs.append(int(branch_lineage[-1]["passage_idx"]))
    else:
        ordered_lineage = sorted(
            lineage,
            key=lambda item: int(item.get("edge_index", 0) or 0),
        )
        if ordered_lineage:
            head_docs.extend([
                int(ordered_lineage[0]["passage_idx"]),
                int(ordered_lineage[-1]["passage_idx"]),
            ])
    head_docs = list(dict.fromkeys(head_docs))[:2]
    semantic_score = min(
        [
            float(contract.get("contract_confidence", 0.0) or 0.0),
            float(
                dict(contract.get("typed_ir_semantic_score", {}) or {}).get(
                    "score", 0.0
                )
                or 0.0
            ),
        ]
        + [float(item.get("relation_score", 0.0) or 0.0) for item in lineage]
    )
    utility = float(group.get("utility", 0.0) or 0.0)
    if len(head_docs) != 2:
        log["reason"] = "certificate_does_not_project_two_head_passages"
    elif not all(bool(item.get("exact_entity_key", False)) for item in lineage):
        log["reason"] = "proof_contains_inexact_entity_binding"
    elif utility < float(min_utility):
        log["reason"] = "proof_utility_below_threshold"
    elif semantic_score < float(min_semantic_score):
        log["reason"] = "proof_semantic_score_below_threshold"
    else:
        log.update({
            "eligible": True,
            "reason": "complete_typed_certificate",
            "head_passage_indices": list(head_docs),
            "proof_passage_indices": [
                int(value)
                for value in group.get("passage_indices", []) or []
                if int(value) in rank_by_doc
            ],
            "utility": round(utility, 6),
            "semantic_score": round(semantic_score, 6),
        })
    return log, group, head_docs, semantic_score


class QueryPlanFactorizedExecutor:
    """Execute ontology plans with bounded relational physical operators.

    The executor does not use gold labels. It compiles query-plan anchors,
    relations, and answer types into one of three access paths and validates
    every result against materialized OpenIE triples before admission.
    """

    _COMPARISON_CUES = (
        "earlier", "later", "older", "younger", "more", "less", "higher",
        "lower", "same", "both", "difference", "compare", "before", "after",
    )





    _TYPED_RELATION_PATTERNS = (
        ("father", r"\bfather\b"),
        ("mother", r"\bmother\b"),
        ("spouse", r"\b(?:spouse|wife|husband|married)\b"),
        ("graduated from", r"\bgraduat(?:e|ed|ing)\b"),
        ("educated at", r"\b(?:educated|study|studies|studied|studying|attended)\b"),
        ("directed by", r"\b(?:directors?|directed)\b"),
        ("written by", r"\b(?:authors?|writers?|written|wrote)\b"),
        ("performed by", r"\b(?:performers?|performed|singers?|sang|starred|starring|acted)\b"),
        ("composed by", r"\b(?:composers?|composed|music by)\b"),
        ("buried in", r"\b(?:burial|buried|interred)\b"),
        ("born", r"\b(?:born|birth)\b"),
        ("died", r"\b(?:died|death)\b"),
        ("issued", r"\b(?:issue|issued)\b"),
        ("founded", r"\b(?:founded|established|formed)\b"),
        ("released", r"\b(?:released|published|premiered|aired)\b"),
        ("located in", r"\b(?:located|location|situated)\b"),
        ("country of citizenship", r"\b(?:citizenship|nationality|same country)\b"),
        ("capital", r"\bcapital\b"),
        ("occupation", r"\b(?:occupation|profession|job)\b"),
        ("part of", r"\bpart of\b"),
        ("member of", r"\b(?:member|membership|belongs)\b"),
        ("produced by", r"\b(?:producer|produced|maker|manufacturer)\b"),
        ("won", r"\b(?:won|winner)\b"),
        ("population", r"\b(?:population|inhabitants)\b"),
    )

    _NON_EVENT_BIRTH_CONTEXTS = (
        r"\bbirth certificates?\b",
        r"\bbirth controls?\b",
        r"\bbirth rates?\b",
        r"\bbirth records?\b",
    )

    _ANSWER_DESCRIPTOR_TOKENS = {
        "answer", "city", "country", "date", "entity", "location",
        "number", "person", "place", "quantity", "state", "string",
        "time", "year",
    }

    _NON_ENTITY_ROLE_TOKENS = {
        "actor", "actress", "answer", "author", "birth", "born", "burial", "buried",
        "capital", "city", "country", "composer",
        "date", "death", "died", "director", "educated", "education",
        "father", "founded", "graduated", "location", "mother", "occupation",
        "performer", "person", "place", "population", "producer", "profession", "released", "school",
        "genre", "husband", "language", "nationality", "singer",
        "spouse", "state", "studies", "study", "studied", "studying", "university",
        "where", "which", "who", "wife", "writer", "year",
    }

    _ROLE_ANCHOR_CONNECTOR_TOKENS = {
        "a", "an", "for", "in", "of", "the", "to",
    }

    _ROLE_ANCHOR_OBJECT_TOKENS = {
        "album", "book", "film", "movie", "novel", "person", "series",
        "show", "song", "team", "work",
    }

    _QUESTION_PREFIX_TOKENS = {
        "are", "did", "do", "does", "how", "is", "name", "the", "what",
        "when", "where", "which", "who", "whose", "why",
    }

    def __init__(
        self,
        enable_entity_relation_index_scan: bool = True,
        enable_cascaded_bind_join: bool = True,
        enable_dual_branch_comparison_join: bool = True,
        dense_top_k: int = 48,
        max_probes: int = 10,
        max_candidates: int = 8,
        max_hops: int = 4,
        beam_width: int = 4,
        max_index_rows: int = 64,
        min_relation_score: float = 0.34,
        min_type_score: float = 0.55,
        min_candidate_utility: float = 0.68,
        cost_weight: float = 0.035,
        enable_typed_semantic_compiler: bool = False,
        require_exact_entity_keys: bool = False,
        require_typed_join_keys: bool = False,
        min_query_predicate_coverage: float = 0.75,
        enable_branch_complete_compiler: bool = False,
        require_branch_qualifier_evidence: bool = False,
        enable_evidence_query_compiler: bool = False,
        enable_cost_based_physical_planner: bool = False,
        evidence_catalog_path: str = "",
        physical_planner_quality_weight: float = 0.45,
        physical_planner_recall_weight: float = 0.25,
        physical_planner_proof_weight: float = 0.30,
        physical_planner_cost_weight: float = 0.15,
        physical_planner_risk_weight: float = 0.35,
        physical_planner_min_proof_coverage: float = 0.20,
        physical_planner_max_estimated_cost: float = 8.0,
        physical_planner_no_op_utility: float = 0.0,
        physical_planner_selection_mode: str = "weighted_utility",
        physical_planner_max_loss_probability: float = 0.25,
        physical_planner_confidence_delta: float = 0.05,
        physical_planner_min_catalog_samples: int = 5,
        physical_access_strategy: str = "adaptive",
        enable_executable_ontology_compiler: bool = False,
        enable_relation_realization: bool = False,
        relation_realization_map: Mapping[str, Any] = None,
        enable_whole_warehouse_probe: bool = False,
        require_relation_artifact_contract: bool = False,
        enable_relation_keyed_catalog: bool = False,
        relation_keyed_allow_matched_schema_fallback: bool = False,
        enable_query_relation_alternative_recovery: bool = False,
        max_query_relation_alternatives: int = 3,
        enable_execution_aware_contract: bool = False,
        enable_lossless_typed_plan_ir: bool = False,
        require_typed_variable_unification: bool = False,
        enable_certificate_aware_plan_portfolio: bool = False,
        enable_certificate_aware_access_path_reoptimization: bool = False,
        enable_provenance_plan_portfolio: bool = False,
        enable_inverse_direction_fallback: bool = False,
        inverse_direction_penalty: float = 0.12,
        enable_candidate_window_path_compiler: bool = False,
        candidate_window_path_size: int = 200,
        candidate_window_path_beam_width: int = 12,
        candidate_window_path_max_paths: int = 8,
        candidate_window_path_min_anchor_score: float = 0.86,
        candidate_window_path_min_relation_score: float = 0.50,
        candidate_window_path_min_path_score: float = 0.58,
        candidate_window_path_allow_contextual_terminal_edges: bool = True,
        candidate_window_path_enable_entity_index_expansion: bool = False,
        candidate_window_path_enable_bound_dense_expansion: bool = False,
        candidate_window_path_entity_index_top_k: int = 64,
        candidate_window_path_bound_dense_top_k: int = 32,
        candidate_window_path_max_access_probes: int = 8,
        candidate_window_path_max_external_passages: int = 96,
        candidate_window_path_enable_relation_agnostic_bridge: bool = False,
        candidate_window_path_bridge_max_paths: int = 8,
        candidate_window_path_bridge_max_edges_per_state: int = 48,
        candidate_window_path_bridge_min_path_score: float = 0.55,
    ) -> None:
        self.enable_entity_relation_index_scan = bool(enable_entity_relation_index_scan)
        self.enable_cascaded_bind_join = bool(enable_cascaded_bind_join)
        self.enable_dual_branch_comparison_join = bool(enable_dual_branch_comparison_join)
        self.dense_top_k = max(int(dense_top_k), 1)
        self.max_probes = max(int(max_probes), 1)
        self.max_candidates = max(int(max_candidates), 1)
        self.max_hops = min(max(int(max_hops), 1), 6)
        self.beam_width = max(int(beam_width), 1)
        self.max_index_rows = max(int(max_index_rows), 1)
        self.min_relation_score = min(max(float(min_relation_score), 0.0), 1.0)
        self.min_type_score = min(max(float(min_type_score), 0.0), 1.0)
        self.min_candidate_utility = min(max(float(min_candidate_utility), 0.0), 1.0)
        self.cost_weight = max(float(cost_weight), 0.0)
        self.enable_typed_semantic_compiler = bool(enable_typed_semantic_compiler)
        self.require_exact_entity_keys = bool(require_exact_entity_keys)
        self.require_typed_join_keys = bool(require_typed_join_keys)
        self.min_query_predicate_coverage = min(
            max(float(min_query_predicate_coverage), 0.0),
            1.0,
        )
        self.enable_branch_complete_compiler = bool(enable_branch_complete_compiler)
        self.require_branch_qualifier_evidence = bool(require_branch_qualifier_evidence)
        self.enable_evidence_query_compiler = bool(enable_evidence_query_compiler)
        self.enable_cost_based_physical_planner = bool(enable_cost_based_physical_planner)
        self.evidence_catalog_path = str(evidence_catalog_path or "").strip()
        self.physical_planner_quality_weight = max(float(physical_planner_quality_weight), 0.0)
        self.physical_planner_recall_weight = max(float(physical_planner_recall_weight), 0.0)
        self.physical_planner_proof_weight = max(float(physical_planner_proof_weight), 0.0)
        self.physical_planner_cost_weight = max(float(physical_planner_cost_weight), 0.0)
        self.physical_planner_risk_weight = max(float(physical_planner_risk_weight), 0.0)
        self.physical_planner_min_proof_coverage = min(
            max(float(physical_planner_min_proof_coverage), 0.0),
            1.0,
        )
        self.physical_planner_max_estimated_cost = max(
            float(physical_planner_max_estimated_cost),
            0.0,
        )
        self.physical_planner_no_op_utility = float(physical_planner_no_op_utility)
        self.physical_planner_selection_mode = str(
            physical_planner_selection_mode or "weighted_utility"
        ).strip().lower()
        self.physical_planner_max_loss_probability = min(
            max(float(physical_planner_max_loss_probability), 0.0), 1.0
        )
        self.physical_planner_confidence_delta = min(
            max(float(physical_planner_confidence_delta), 1e-6), 0.50
        )
        self.physical_planner_min_catalog_samples = max(
            int(physical_planner_min_catalog_samples), 1
        )
        normalized_strategy = str(physical_access_strategy or "adaptive").strip().lower()
        if normalized_strategy not in {"adaptive", "index_only", "dense_first"}:
            raise ValueError(f"unsupported factorized access strategy: {normalized_strategy}")
        self.physical_access_strategy = normalized_strategy
        self.enable_executable_ontology_compiler = bool(enable_executable_ontology_compiler)
        self.enable_relation_realization = bool(enable_relation_realization)
        self.relation_realization_map = dict(relation_realization_map or {})
        self.enable_whole_warehouse_probe = bool(enable_whole_warehouse_probe)
        self.require_relation_artifact_contract = bool(require_relation_artifact_contract)
        self.enable_relation_keyed_catalog = bool(enable_relation_keyed_catalog)
        self.relation_keyed_allow_matched_schema_fallback = bool(
            relation_keyed_allow_matched_schema_fallback
        )
        self.enable_query_relation_alternative_recovery = bool(
            enable_query_relation_alternative_recovery
        )
        self.max_query_relation_alternatives = max(
            int(max_query_relation_alternatives), 1
        )
        self.enable_execution_aware_contract = bool(enable_execution_aware_contract)
        self.enable_lossless_typed_plan_ir = bool(enable_lossless_typed_plan_ir)
        self.require_typed_variable_unification = bool(
            require_typed_variable_unification
        )
        self.enable_certificate_aware_plan_portfolio = bool(
            enable_certificate_aware_plan_portfolio
        )
        self.enable_certificate_aware_access_path_reoptimization = bool(
            enable_certificate_aware_access_path_reoptimization
        )
        self.enable_provenance_plan_portfolio = bool(enable_provenance_plan_portfolio)
        self.enable_inverse_direction_fallback = bool(enable_inverse_direction_fallback)
        self.inverse_direction_penalty = min(
            max(float(inverse_direction_penalty), 0.0),
            0.95,
        )
        self.enable_candidate_window_path_compiler = bool(
            enable_candidate_window_path_compiler
        )
        self.candidate_window_path_size = max(int(candidate_window_path_size), 1)
        self.candidate_window_path_beam_width = max(
            int(candidate_window_path_beam_width), 1
        )
        self.candidate_window_path_max_paths = max(
            int(candidate_window_path_max_paths), 1
        )
        self.candidate_window_path_min_anchor_score = min(
            max(float(candidate_window_path_min_anchor_score), 0.0), 1.0
        )
        self.candidate_window_path_min_relation_score = min(
            max(float(candidate_window_path_min_relation_score), 0.0), 1.0
        )
        self.candidate_window_path_min_path_score = min(
            max(float(candidate_window_path_min_path_score), 0.0), 1.0
        )
        self.candidate_window_path_allow_contextual_terminal_edges = bool(
            candidate_window_path_allow_contextual_terminal_edges
        )
        self.candidate_window_path_enable_entity_index_expansion = bool(
            candidate_window_path_enable_entity_index_expansion
        )
        self.candidate_window_path_enable_bound_dense_expansion = bool(
            candidate_window_path_enable_bound_dense_expansion
        )
        self.candidate_window_path_entity_index_top_k = max(
            int(candidate_window_path_entity_index_top_k), 1
        )
        self.candidate_window_path_bound_dense_top_k = max(
            int(candidate_window_path_bound_dense_top_k), 1
        )
        self.candidate_window_path_max_access_probes = max(
            int(candidate_window_path_max_access_probes), 1
        )
        self.candidate_window_path_max_external_passages = max(
            int(candidate_window_path_max_external_passages), 0
        )
        self.candidate_window_path_enable_relation_agnostic_bridge = bool(
            candidate_window_path_enable_relation_agnostic_bridge
        )
        self.candidate_window_path_bridge_max_paths = max(
            int(candidate_window_path_bridge_max_paths), 1
        )
        self.candidate_window_path_bridge_max_edges_per_state = max(
            int(candidate_window_path_bridge_max_edges_per_state), 8
        )
        self.candidate_window_path_bridge_min_path_score = min(
            max(float(candidate_window_path_bridge_min_path_score), 0.0), 1.0
        )
        self._active_relation_realizations: Dict[str, List[str]] = {}
        self._active_relation_directions: Dict[str, str] = {}
        self._active_relation_predicate_directions: Dict[str, Dict[str, str]] = {}
        self._strict_relation_predicates: set[str] = set()
        self._runtime_rejections: Dict[str, int] = {}

    @staticmethod
    def _unique(values: Iterable[Any], limit: int = 16) -> List[str]:
        output: List[str] = []
        seen = set()
        for value in values or []:
            normalized = QueryPlanFactorizedExecutor._entity_key(value)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            output.append(normalized)
            if len(output) >= limit:
                break
        return output

    @classmethod
    def _ordered_relations(cls, query_plan_log: Mapping[str, Any]) -> List[str]:
        relation_paths: List[List[Any]] = []
        for plan in query_plan_log.get("matched_plans", []) or []:
            if not isinstance(plan, Mapping):
                continue
            path_relations: List[Any] = []
            for step in plan.get("path", []) or []:
                if isinstance(step, Mapping):
                    path_relations.append(
                        step.get("relation")
                        or step.get("relation_signature")
                        or step.get("relation_family")
                        or ""
                    )
                else:
                    path_relations.append(step)
            if path_relations:
                relation_paths.append(path_relations)
        for path in query_plan_log.get("sample_paths", []) or []:
            if isinstance(path, Mapping) and path.get("relations"):
                relation_paths.append(list(path.get("relations", []) or []))
        relation_paths.sort(key=len, reverse=True)
        if relation_paths:
            return cls._unique(relation_paths[0], limit=6)
        return []

    @staticmethod
    def _entity_key(value: Any) -> str:
        normalized = unicodedata.normalize("NFKD", str(value or ""))
        ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
        return " ".join(re.sub(r"[^a-z0-9]+", " ", ascii_text.lower()).split())

    @classmethod
    def _entity_alias_keys(cls, value: Any) -> List[str]:
        """Return aliases that preserve entity identity without fuzzy matching."""
        raw = str(value or "")
        candidates = [raw]
        without_parenthetical = re.sub(r"\s*\([^)]*\)\s*", " ", raw)
        candidates.append(without_parenthetical)
        output: List[str] = []
        for candidate in candidates:
            normalized = cls._entity_key(candidate)
            if normalized.startswith("the "):
                normalized = normalized[4:].strip()
            if normalized.endswith(" s"):
                normalized = normalized[:-2].strip()
            if normalized and normalized not in output:
                output.append(normalized)
        return output

    @staticmethod
    def _identity_tokens(value: Any) -> List[str]:
        normalized = QueryPlanFactorizedExecutor._entity_key(value)
        return [
            token for token in normalized.split()
            if token and token not in QueryPlanOntologyBindProbeJoin._STOPWORDS
        ]

    @classmethod
    def _canonical_anchors(cls, query: str, anchors: Sequence[str]) -> List[str]:
        """Prefer the most specific query-grounded entity key.

        Initials are retained here even though the general content tokenizer
        drops short tokens. This distinguishes ``C. N. Annadurai`` from every
        other entity sharing the surname ``Annadurai``.
        """
        query_norm = cls._entity_key(query)
        profiles: List[Dict[str, Any]] = []
        for anchor in cls._unique(anchors, limit=16):
            tokens = cls._identity_tokens(anchor)
            if not tokens:
                continue
            normalized = cls._entity_key(anchor)
            profiles.append({
                "anchor": normalized,
                "tokens": tokens,
                "token_set": set(tokens),
                "query_exact": f" {normalized} " in f" {query_norm} ",
                "query_position": query_norm.find(normalized),
            })

        maximal: List[Dict[str, Any]] = []
        for profile in profiles:
            token_set = profile["token_set"]
            if any(
                token_set < other["token_set"]
                and bool(other["query_exact"])
                for other in profiles
            ):
                continue
            maximal.append(profile)



        best_by_tokens: Dict[Tuple[str, ...], Dict[str, Any]] = {}
        for profile in maximal:
            key = tuple(sorted(profile["token_set"]))
            previous = best_by_tokens.get(key)
            candidate_key = (
                int(bool(profile["query_exact"])),
                -len(str(profile["anchor"])),
            )
            previous_key = (
                int(bool(previous["query_exact"])),
                -len(str(previous["anchor"])),
            ) if previous else (-1, -10**9)
            if previous is None or candidate_key > previous_key:
                best_by_tokens[key] = profile

        selected = list(best_by_tokens.values())
        selected.sort(key=lambda item: (
            int(bool(item["query_exact"])),
            len(item["tokens"]),
            -max(int(item["query_position"]), 0),
        ), reverse=True)
        return [str(item["anchor"]) for item in selected]

    @classmethod
    def _anchor_spans(cls, query: str, anchors: Sequence[str]) -> List[Dict[str, Any]]:
        query_norm = cls._entity_key(query)
        query_raw = str(query or "").lower()
        spans: List[Dict[str, Any]] = []
        for anchor in cls._unique(anchors, limit=16):
            start = query_norm.find(anchor)
            if start < 0:
                continue
            raw_pattern = r"[\W_]+".join(
                re.escape(token) for token in anchor.split() if token
            )
            raw_match = re.search(raw_pattern, query_raw) if raw_pattern else None
            spans.append({
                "anchor": anchor,
                "start": start,
                "end": start + len(anchor),
                "raw_start": int(raw_match.start()) if raw_match else -1,
                "raw_end": int(raw_match.end()) if raw_match else -1,
                "tokens": cls._identity_tokens(anchor),
            })
        spans.sort(key=lambda item: (int(item["start"]), -len(str(item["anchor"]))))
        return spans

    @classmethod
    def _compile_comparison_branches(
        cls,
        query: str,
        anchors: Sequence[str],
    ) -> Dict[str, Any]:
        """Compile two coordinated entities without promoting their qualifiers.

        Query-plan anchors are an unordered entity bag. Comparison execution,
        however, needs two ordered branches. We recover them from the nearest
        anchors around a coordinating ``and``/``or`` and attach all remaining
        local anchors as disambiguating qualifiers.
        """
        query_norm = cls._entity_key(query)
        spans = cls._anchor_spans(query, anchors)
        parenthetical_values = [
            cls._entity_key(value)
            for value in re.findall(r"\(([^()]*)\)", str(query or ""))
            if cls._entity_key(value)
        ]
        for span in spans:
            anchor = str(span["anchor"])
            span["parenthetical"] = any(
                f" {anchor} " in f" {value} "
                for value in parenthetical_values
            )
        candidates: List[Dict[str, Any]] = []
        for match in re.finditer(r"\b(and|or)\b", query_norm):
            if any(
                int(span["start"]) < match.start() < int(span["end"])
                for span in spans
            ):
                continue
            left = [span for span in spans if int(span["end"]) <= match.start()]
            right = [span for span in spans if int(span["start"]) >= match.end()]
            if not left or not right:
                continue
            left_primary_pool = [item for item in left if not item.get("parenthetical")] or left
            right_primary_pool = [item for item in right if not item.get("parenthetical")] or right
            left_primary = min(
                left_primary_pool,
                key=lambda item: (
                    match.start() - int(item["end"]),
                    -len(item.get("tokens", [])),
                ),
            )
            ordered_left = sorted(left_primary_pool, key=lambda item: int(item["start"]))
            if len(ordered_left) >= 2:
                first, second = ordered_left[-2:]
                raw_start = int(first.get("raw_end", -1))
                raw_end = int(second.get("raw_start", -1))
                if raw_start >= 0 and raw_end >= raw_start and "," in str(query or "")[raw_start:raw_end]:
                    left_primary = first
            right_primary = min(
                right_primary_pool,
                key=lambda item: (
                    int(item["start"]) - match.end(),
                    -len(item.get("tokens", [])),
                ),
            )
            if left_primary["anchor"] == right_primary["anchor"]:
                continue
            distance = (
                match.start() - int(left_primary["end"])
                + int(right_primary["start"]) - match.end()
            )
            candidates.append({
                "separator": match.group(1),
                "separator_start": match.start(),
                "separator_end": match.end(),
                "left": left_primary,
                "right": right_primary,
                "distance": distance,
            })
        if not candidates:
            return {
                "enabled": True,
                "complete": False,
                "branches": [],
                "confidence": 0.0,
                "rejection_reason": "no_two_sided_coordination",
            }
        selected = min(
            candidates,
            key=lambda item: (
                int(item["distance"]),
                0 if item["separator"] == "or" else 1,
            ),
        )
        primary = [selected["left"], selected["right"]]
        primary_names = {str(item["anchor"]) for item in primary}
        branches: List[Dict[str, Any]] = []
        for branch_idx, item in enumerate(primary):
            if branch_idx == 0:
                local_start = int(item["end"])
                local_end = int(selected["separator_start"])
            else:
                local_start = int(item["end"])
                local_end = len(query_norm)
            qualifiers = []
            for span in spans:
                name = str(span["anchor"])
                if name in primary_names:
                    continue
                if local_start <= int(span["start"]) < local_end:
                    qualifiers.append(name)
            branches.append({
                "branch_idx": branch_idx,
                "entity_key": str(item["anchor"]),
                "qualifiers": cls._unique(qualifiers, limit=4),
                "query_start": int(item["start"]),
                "query_end": int(item["end"]),
            })
        complete = bool(
            len(branches) == 2
            and branches[0]["entity_key"] != branches[1]["entity_key"]
        )
        qualifier_count = sum(len(item["qualifiers"]) for item in branches)
        confidence = 1.0 if complete else 0.0
        if complete and len(spans) > 2 and qualifier_count == 0:
            confidence = 0.80
        return {
            "enabled": True,
            "complete": complete,
            "branches": branches,
            "confidence": round(confidence, 6),
            "separator": str(selected["separator"]),
            "anchor_count": len(spans),
            "qualifier_count": qualifier_count,
            "rejection_reason": "" if complete else "branch_entities_not_distinct",
        }

    @classmethod
    def _bind_query_surface_slots(
        cls,
        query: str,
        anchors: Sequence[str],
        enable_branch_compiler: bool,
    ) -> Dict[str, Any]:
        """Bind executable anchor slots only to entity mentions in the query."""
        query_norm = cls._entity_key(query)
        mention_pattern = re.compile(
            r"\b[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ0-9.'’'-]*"
            r"(?:\s+(?:[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ0-9.'’'-]*|of|the|de|da|van|von)){0,5}"
        )
        mentions = []
        for raw_mention in mention_pattern.findall(str(query or "")):
            mention = cls._entity_key(re.sub(r"['’]s$", "", raw_mention))
            mention_tokens = set(cls._identity_tokens(mention))
            if not mention_tokens or mention_tokens.issubset(cls._QUESTION_PREFIX_TOKENS):
                continue
            mentions.append(mention)
        canonical = cls._canonical_anchors(query, list(mentions) + list(anchors or []))
        rejected_role_anchors: List[str] = []
        grounded = [
            anchor for anchor in canonical
            if f" {anchor} " in f" {query_norm} "
            and not (
                cls._is_non_entity_role_anchor(anchor)
                or all(token.isdigit() for token in cls._identity_tokens(anchor))
            )
        ]
        for anchor in canonical:
            tokens = set(cls._identity_tokens(anchor))
            if (
                f" {anchor} " in f" {query_norm} "
                and tokens
                and (cls._is_non_entity_role_anchor(anchor) or all(token.isdigit() for token in tokens))
            ):
                rejected_role_anchors.append(anchor)
        comparison = cls._is_comparison(query, grounded)
        branch_contract: Dict[str, Any] = {
            "enabled": False,
            "complete": True,
            "branches": [],
        }
        executable = list(grounded)
        if comparison and enable_branch_compiler:
            branch_contract = cls._compile_comparison_branches(query, grounded)
            if branch_contract.get("complete", False):
                executable = [
                    str(branch.get("entity_key", "") or "")
                    for branch in branch_contract.get("branches", []) or []
                    if branch.get("entity_key")
                ]
        return {
            "enabled": True,
            "raw_anchor_count": len(list(anchors or [])),
            "canonical_anchors": canonical,
            "grounded_anchors": grounded,
            "executable_anchors": cls._unique(executable, limit=4),
            "dropped_non_query_anchors": [
                anchor for anchor in canonical if anchor not in grounded
            ],
            "dropped_non_entity_role_anchors": rejected_role_anchors,
            "invalid_role_anchor_count": len(rejected_role_anchors),
            "exact_anchor_success": bool(grounded),
            "comparison_query": comparison,
            "comparison_branches": branch_contract,
        }

    @classmethod
    def _is_non_entity_role_anchor(cls, value: Any) -> bool:
        tokens = set(cls._identity_tokens(value))
        if not tokens:
            return False
        allowed = (
            cls._NON_ENTITY_ROLE_TOKENS
            | cls._ROLE_ANCHOR_CONNECTOR_TOKENS
            | cls._ROLE_ANCHOR_OBJECT_TOKENS
        )
        numeric_type_label = bool(
            any(token.isdigit() for token in tokens)
            and {
                token for token in tokens if not token.isdigit()
            }.issubset(cls._ROLE_ANCHOR_CONNECTOR_TOKENS | cls._ROLE_ANCHOR_OBJECT_TOKENS)
        )
        return bool(
            numeric_type_label
            or
            tokens.intersection(cls._NON_ENTITY_ROLE_TOKENS)
            and tokens.issubset(allowed)
        )

    @classmethod
    def _ordered_relation_contract(
        cls,
        surface_relations: Sequence[Any],
        ontology_relations: Sequence[Any],
        terminal_profile: Mapping[str, Any],
        *,
        ontology_order_is_authoritative: bool = False,
    ) -> List[str]:
        """Compile relations in binding order and keep the answer edge last.

        A certified schema path already encodes edge direction and binding
        order.  Reordering such a path with a surface-level terminal guess can
        reverse a valid chain (for example, ``director -> birth date``).  The
        terminal heuristic is therefore only allowed to order an unstructured
        surface contract.
        """
        ordered = cls._semantic_unique_relations(
            ontology_relations or surface_relations,
            limit=6,
        )
        if ontology_order_is_authoritative and ontology_relations:
            return ordered
        terminal = str(terminal_profile.get("canonical_relation", "") or "")
        if not terminal_profile.get("resolved") or not terminal:
            return ordered
        aliases = [terminal] + list(terminal_profile.get("aliases", []) or [])

        def is_terminal(relation: str) -> bool:
            return any(
                max(
                    cls._typed_relation_overlap(relation, alias)[0],
                    cls._typed_relation_overlap(alias, relation)[0],
                ) >= 0.60
                for alias in aliases
            )

        prefix = [relation for relation in ordered if not is_terminal(relation)]
        return cls._semantic_unique_relations([*prefix, terminal], limit=6)

    @classmethod
    def _relation_output_type(cls, relation: Any, direction: Any) -> str:
        relation_key = cls._entity_key(relation)
        forward_type = "entity"
        if any(token in relation_key for token in ("birth date", "death date", "release date", "publication date", "formation date", "founded on")):
            forward_type = "date"
        elif any(token in relation_key for token in ("birth place", "death place", "location", "country", "nationality", "capital", "headquarters")):
            forward_type = "location"
        elif any(token in relation_key for token in ("director", "author", "writer", "composer", "performer", "actor", "father", "mother", "spouse", "founder", "president")):
            forward_type = "person"
        elif any(token in relation_key for token in ("population", "count", "quantity", "number")):
            forward_type = "number"
        elif any(token in relation_key for token in ("publisher", "company", "organization", "team", "university")):
            forward_type = "organization"
        if str(direction or "forward").lower() == "reverse":
            return "entity"
        return forward_type

    @classmethod
    def _smallest_repeating_step_period(
        cls,
        steps: Sequence[Mapping[str, Any]],
    ) -> int:
        signatures = [
            (
                cls._entity_key(
                    step.get("relation_signature")
                    or step.get("relation_family")
                    or step.get("relation")
                    or ""
                ),
                str(step.get("direction", "forward") or "forward").lower(),
            )
            for step in steps
        ]
        for period in range(1, len(signatures) // 2 + 1):
            if len(signatures) % period == 0 and signatures == (
                signatures[:period] * (len(signatures) // period)
            ):
                return period
        return len(signatures)

    @classmethod
    def _comparison_operator(cls, query: str) -> str:
        query_key = cls._entity_key(query)
        if re.search(r"\b(?:later|younger|more|higher|larger|greater)\b", query_key):
            return "argmax"
        if re.search(r"\b(?:earlier|older|less|lower|smaller|fewer)\b", query_key):
            return "argmin"
        if re.search(r"\b(?:same|both|equal)\b", query_key):
            return "equality"
        return "compare"

    @classmethod
    def _compile_lossless_plan_ir(
        cls,
        query: str,
        candidate: Mapping[str, Any],
        anchors: Sequence[str],
        answer_type: str,
        comparison: bool,
        comparison_branches: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Preserve schema variables, directions, and branch topology."""
        raw_steps = [
            dict(step)
            for step in candidate.get("path", []) or []
            if isinstance(step, Mapping)
        ]
        raw_steps.sort(key=lambda step: int(step.get("step", 0) or 0))
        plan_id = str(candidate.get("plan_id") or candidate.get("id") or "")
        errors: List[str] = []
        if not raw_steps:
            errors.append("empty_schema_path")

        def compile_edge(
            step: Mapping[str, Any],
            *,
            edge_index: int,
            input_variable: str,
            output_variable: str,
            branch_id: str,
        ) -> Dict[str, Any]:
            relation = str(
                step.get("relation_signature")
                or step.get("relation")
                or step.get("relation_family")
                or ""
            )
            direction = str(step.get("direction", "forward") or "forward").lower()
            if direction not in {"forward", "reverse"}:
                direction = "either"
            output_type = cls._relation_output_type(relation, direction)
            return {
                "edge_index": edge_index,
                "schema_step": int(step.get("step", edge_index) or edge_index),
                "input_variable": input_variable,
                "input_type": "entity",
                "relation": relation,
                "relation_family": str(step.get("relation_family", "") or ""),
                "relation_aliases": list(step.get("relation_aliases", []) or []),
                "direction": direction,
                "output_variable": output_variable,
                "output_type": output_type,
                "branch_id": branch_id,
            }

        edges: List[Dict[str, Any]] = []
        initial_bindings: Dict[str, str] = {}
        branch_ids: List[str] = []
        topology = "linear_chain"
        projection: Dict[str, Any] = {
            "operator": "project",
            "variable": "answer",
            "answer_type": answer_type,
        }
        if comparison and len(anchors) >= 2 and raw_steps:
            topology = "dual_branch_comparison"
            period = cls._smallest_repeating_step_period(raw_steps)
            branch_template = raw_steps[:period]
            branch_contracts = [
                dict(item)
                for item in comparison_branches.get("branches", []) or []
                if isinstance(item, Mapping)
            ]
            branch_anchors = [
                str(item.get("entity_key", "") or "")
                for item in branch_contracts[:2]
            ]
            if len(branch_anchors) != 2 or not all(branch_anchors):
                branch_anchors = [str(value) for value in anchors[:2]]
            for branch_index, anchor in enumerate(branch_anchors):
                branch_id = f"branch_{branch_index}"
                branch_ids.append(branch_id)
                root_variable = f"{branch_id}_anchor"
                initial_bindings[root_variable] = anchor
                current_variable = root_variable
                for local_index, step in enumerate(branch_template):
                    output_variable = f"{branch_id}_value_{local_index}"
                    edges.append(compile_edge(
                        step,
                        edge_index=len(edges),
                        input_variable=current_variable,
                        output_variable=output_variable,
                        branch_id=branch_id,
                    ))
                    current_variable = output_variable
            projection = {
                "operator": cls._comparison_operator(query),
                "branch_value_variables": [
                    f"{branch_id}_value_{len(branch_template) - 1}"
                    for branch_id in branch_ids
                ],
                "branch_anchor_variables": [
                    f"{branch_id}_anchor" for branch_id in branch_ids
                ],
                "answer_type": answer_type,
            }
        else:
            first_variable = str(raw_steps[0].get("from", "anchor") or "anchor") if raw_steps else "anchor"
            if anchors:
                initial_bindings[first_variable] = str(anchors[0])
            bound_variables = {first_variable}
            for step_index, step in enumerate(raw_steps):
                input_variable = str(step.get("from", "") or "")
                output_variable = str(step.get("to", "") or "")
                if not input_variable or not output_variable:
                    errors.append(f"missing_variable_at_step_{step_index}")
                    continue
                if input_variable not in bound_variables:
                    errors.append(f"unbound_input_{input_variable}_at_step_{step_index}")
                edges.append(compile_edge(
                    step,
                    edge_index=len(edges),
                    input_variable=input_variable,
                    output_variable=output_variable,
                    branch_id="main",
                ))
                bound_variables.add(output_variable)
            if raw_steps and str(raw_steps[-1].get("to", "") or "") != "answer":
                errors.append("schema_does_not_project_answer")

        for edge in edges:
            if not edge["relation"]:
                errors.append(f"missing_relation_at_edge_{edge['edge_index']}")
        if not initial_bindings:
            errors.append("no_grounded_anchor_binding")
        return {
            "version": "typed_plan_ir_v1",
            "plan_id": plan_id,
            "topology": topology,
            "branch_ids": branch_ids,
            "initial_bindings": initial_bindings,
            "edges": edges,
            "projection": projection,
            "answer_type": answer_type,
            "valid": not errors,
            "errors": errors,
            "lossless": True,
        }

    @classmethod
    def _typed_ir_semantic_score(
        cls,
        typed_ir: Mapping[str, Any],
        expected_relations: Sequence[str],
        terminal_profile: Mapping[str, Any],
        answer_type_compatibility: float,
        comparison: bool,
    ) -> Dict[str, float]:
        edges = list(typed_ir.get("edges", []) or [])
        branch_ids = list(typed_ir.get("branch_ids", []) or [])
        if branch_ids:
            first_branch = branch_ids[0]
            relation_path = [
                str(edge.get("relation", "") or "")
                for edge in edges
                if edge.get("branch_id") == first_branch
            ]
        else:
            relation_path = [str(edge.get("relation", "") or "") for edge in edges]
        relation_coverage = cls._path_predicate_coverage(
            expected_relations,
            relation_path,
        )
        terminal_aliases = [
            str(terminal_profile.get("canonical_relation", "") or ""),
            *(terminal_profile.get("aliases", []) or []),
        ]
        terminal_relation = relation_path[-1] if relation_path else ""
        terminal_match = max(
            (
                max(
                    cls._typed_relation_overlap(terminal_relation, alias)[0],
                    cls._typed_relation_overlap(alias, terminal_relation)[0],
                )
                for alias in terminal_aliases
                if alias
            ),
            default=(1.0 if not terminal_profile.get("resolved") else 0.0),
        )
        expected_topology = "dual_branch_comparison" if comparison else "linear_chain"
        topology_match = float(typed_ir.get("topology") == expected_topology)
        variable_validity = float(bool(typed_ir.get("valid", False)))
        score = (
            0.30 * relation_coverage
            + 0.25 * terminal_match
            + 0.20 * float(answer_type_compatibility)
            + 0.15 * topology_match
            + 0.10 * variable_validity
        )
        return {
            "score": round(score, 6),
            "relation_coverage": round(relation_coverage, 6),
            "terminal_relation_match": round(terminal_match, 6),
            "answer_type_compatibility": round(float(answer_type_compatibility), 6),
            "topology_match": round(topology_match, 6),
            "variable_validity": round(variable_validity, 6),
        }

    def _compile_relation_realizations(
        self,
        query_plan_log: Mapping[str, Any],
        relations: Sequence[str],
        selected_schema: Mapping[str, Any] = None,
    ) -> Dict[str, Any]:
        """Resolve conceptual ontology edges to corpus-observed predicates."""
        output: Dict[str, List[Dict[str, Any]]] = {str(relation): [] for relation in relations}
        plans = self.relation_realization_map.get("plans", {})
        if not self.enable_relation_realization or not isinstance(plans, Mapping):
            return {
                "enabled": self.enable_relation_realization,
                "available": bool(plans),
                "relations": output,
                "realized_relation_count": 0,
            }

        selected_schema = dict(selected_schema or {})
        selected_plan_id = str(selected_schema.get("plan_id", "") or "")
        runtime_plans: List[Mapping[str, Any]] = []
        seen_plan_ids = set()
        sidecar_plans = (
            dict(query_plan_log.get("query_condition", {}) or {}).get(
                "compiler_schema_plans", []
            )
            if self.enable_relation_keyed_catalog else []
        )
        for plan in list(sidecar_plans or []) + list(
            query_plan_log.get("matched_plans", []) or []
        ):
            if not isinstance(plan, Mapping):
                continue
            plan_id = str(plan.get("id", "") or "")
            if not plan_id or plan_id in seen_plan_ids:
                continue
            seen_plan_ids.add(plan_id)
            runtime_plans.append(plan)

        selected_plans = [
            plan for plan in runtime_plans
            if str(plan.get("id", "") or "") == selected_plan_id
        ]
        if selected_plan_id and not selected_plans:
            self._runtime_rejections["selected_schema_not_in_runtime_plans"] = 1
        if selected_plan_id:
            source_plans = list(selected_plans)
            if (
                self.enable_relation_keyed_catalog
                and self.relation_keyed_allow_matched_schema_fallback
            ):
                source_plans.extend(
                    plan for plan in runtime_plans
                    if str(plan.get("id", "") or "") != selected_plan_id
                )
        else:
            source_plans = list(runtime_plans)

        artifact_contract_ok = True
        certified_only = bool(self.relation_realization_map.get("enforce_certified_only", False))
        catalog_lookup_count = 0
        for matched_plan in source_plans:
            if not isinstance(matched_plan, Mapping):
                continue
            plan_id = str(
                matched_plan.get("realization_source_plan_id")
                or matched_plan.get("id")
                or ""
            )
            plan_entry = plans.get(plan_id, {}) if plan_id else {}
            if (
                certified_only
                and isinstance(plan_entry, Mapping)
                and not bool(plan_entry.get("certified_executable", False))
            ):
                self._runtime_rejections["uncertified_incomplete_physical_plan"] = (
                    self._runtime_rejections.get(
                        "uncertified_incomplete_physical_plan", 0
                    ) + 1
                )





            steps = plan_entry.get("steps", {}) if isinstance(plan_entry, Mapping) else {}
            runtime_schema_hash = str(matched_plan.get("schema_sha256", "") or "")
            artifact_schema_hash = str(
                plan_entry.get("schema_sha256", "") if isinstance(plan_entry, Mapping) else ""
            )
            if self.require_relation_artifact_contract and (
                not runtime_schema_hash
                or not artifact_schema_hash
                or runtime_schema_hash != artifact_schema_hash
            ):
                artifact_contract_ok = False
                self._runtime_rejections["relation_artifact_schema_mismatch"] = (
                    self._runtime_rejections.get("relation_artifact_schema_mismatch", 0) + 1
                )
                continue
            directions = list(
                dict(matched_plan.get("integrity_constraints", {}) or {}).get(
                    "relation_directions", []
                ) or []
            )
            path = list(matched_plan.get("path", []) or [])
            signatures = list(matched_plan.get("relation_signatures", []) or []) or [
                step.get("relation_signature", "") if isinstance(step, Mapping) else ""
                for step in path
            ]
            families = list(matched_plan.get("relation_families", []) or []) or [
                step.get("relation_family", "") if isinstance(step, Mapping) else ""
                for step in path
            ]
            for relation_idx, relation in enumerate(relations):
                if self.enable_relation_keyed_catalog:
                    step_entries = list(steps.items()) if isinstance(steps, Mapping) else []
                else:
                    step_entries = [(
                        str(relation_idx),
                        steps.get(str(relation_idx), {}) if isinstance(steps, Mapping) else {},
                    )]
                for step_key, step_entry in step_entries:
                    if not isinstance(step_entry, Mapping):
                        continue
                    try:
                        source_step = int(step_key)
                    except (TypeError, ValueError):
                        source_step = int(step_entry.get("step", relation_idx) or relation_idx)
                    catalog_lookup_count += 1
                    if certified_only:
                        expected = list(
                            step_entry.get("certification_relation_terms", []) or []
                        ) or [step_entry.get("expected_relation_signature", "")]
                    else:
                        expected = [
                            step_entry.get("expected_relation_signature", ""),
                            step_entry.get("expected_relation_family", ""),
                            *(step_entry.get("expected_relation_terms", []) or []),
                            *(step_entry.get("expected_signatures", []) or []),
                            *(step_entry.get("expected_families", []) or []),
                        ]
                    if 0 <= source_step < len(signatures):
                        expected.append(signatures[source_step])
                    if 0 <= source_step < len(families):
                        expected.append(families[source_step])
                    compatibility = max(
                        (
                            max(
                                self._typed_relation_overlap(relation, value)[0],
                                self._typed_relation_overlap(value, relation)[0],
                            )
                            for value in expected if value
                        ),
                        default=0.0,
                    )
                    if compatibility < 0.50:
                        continue
                    direction = str(
                        step_entry.get("expected_direction", "")
                        or (directions[source_step] if source_step < len(directions) else "")
                        or "either"
                    ).lower()
                    for realization in step_entry.get("realizations", []) or []:
                        if not isinstance(realization, Mapping):
                            continue
                        if certified_only and not bool(realization.get("certified", False)):
                            self._runtime_rejections["uncertified_relation_realization"] = (
                                self._runtime_rejections.get("uncertified_relation_realization", 0) + 1
                            )
                            continue
                        predicate = self._entity_key(
                            realization.get("normalized_relation")
                            or realization.get("relation")
                            or ""
                        )
                        if not predicate:
                            continue
                        output[str(relation)].append({
                            "predicate": predicate,
                            "count": int(realization.get("count", 0) or 0),
                            "score": float(realization.get("score", 0.0) or 0.0),
                            "historical_support": int(
                                realization.get("historical_support", 0) or 0
                            ),
                            "historical_concept_precision": float(
                                realization.get("historical_concept_precision", 0.0) or 0.0
                            ),
                            "historical_concept_precision_lcb": float(
                                realization.get("historical_concept_precision_lcb", 0.0) or 0.0
                            ),
                            "semantic_compatibility": float(
                                realization.get("semantic_compatibility", 0.0) or 0.0
                            ),
                            "direction_consistency": float(
                                realization.get("direction_consistency", 0.0) or 0.0
                            ),
                            "certified": bool(realization.get("certified", False)),
                            "source_plan_id": plan_id,
                            "source_step": source_step,
                            "source_scope": (
                                "selected_schema"
                                if selected_plan_id and plan_id == selected_plan_id
                                else "matched_schema_fallback"
                            ),
                            "expected_direction": str(
                                realization.get("observed_direction", "")
                                or direction
                            ).lower(),
                            "compatibility": round(float(compatibility), 6),
                        })

        active: Dict[str, List[str]] = {}
        predicate_directions: Dict[str, Dict[str, str]] = {}
        realized_count = 0
        fallback_realization_count = 0
        for relation, candidates in output.items():
            selected_candidates = [
                item for item in candidates
                if item.get("source_scope") == "selected_schema"
            ]
            if selected_candidates:
                candidates = selected_candidates
            else:
                fallback_realization_count += len(candidates)
            best_by_predicate: Dict[str, Dict[str, Any]] = {}
            for candidate in candidates:
                predicate = str(candidate["predicate"])
                previous = best_by_predicate.get(predicate)
                key = (
                    int(bool(candidate.get("certified", False))),
                    float(candidate.get("historical_concept_precision_lcb", 0.0) or 0.0),
                    float(candidate.get("semantic_compatibility", 0.0) or 0.0),
                    int(candidate.get("historical_support", 0) or 0),
                    float(candidate.get("historical_concept_precision", 0.0) or 0.0),
                    float(candidate["compatibility"]),
                    int(candidate["count"]),
                    float(candidate["score"]),
                )
                previous_key = (
                    int(bool(previous.get("certified", False))),
                    float(previous.get("historical_concept_precision_lcb", 0.0) or 0.0),
                    float(previous.get("semantic_compatibility", 0.0) or 0.0),
                    int(previous.get("historical_support", 0) or 0),
                    float(previous.get("historical_concept_precision", 0.0) or 0.0),
                    float(previous["compatibility"]),
                    int(previous["count"]),
                    float(previous["score"]),
                ) if previous else (-1, -1.0, -1.0, -1, -1.0, -1.0, -1, -1.0)
                if previous is None or key > previous_key:
                    best_by_predicate[predicate] = candidate
            historically_supported = [
                item for item in best_by_predicate.values()
                if int(item.get("historical_support", 0) or 0) > 0
            ]
            ranking_pool = historically_supported or list(best_by_predicate.values())
            selected = sorted(
                ranking_pool,
                key=lambda item: (
                    int(bool(item.get("certified", False))),
                    float(item.get("historical_concept_precision_lcb", 0.0) or 0.0),
                    float(item.get("semantic_compatibility", 0.0) or 0.0),
                    int(item.get("historical_support", 0) or 0),
                    float(item.get("historical_concept_precision", 0.0) or 0.0),
                    float(item["compatibility"]),
                    int(item["count"]),
                    float(item["score"]),
                ),
                reverse=True,
            )[:6 if certified_only else (3 if self.require_relation_artifact_contract else 12)]
            output[relation] = selected
            if selected:
                self._active_relation_directions[str(relation)] = str(
                    selected[0].get("expected_direction", "either") or "either"
                )
            predicate_directions[relation] = {
                str(item["predicate"]): str(
                    item.get("expected_direction", "either") or "either"
                )
                for item in selected
            }
            active[relation] = self._unique(
                [item["predicate"] for item in selected]
                if certified_only else [relation] + [item["predicate"] for item in selected],
                limit=13,
            )
            if certified_only and selected:
                self._strict_relation_predicates.add(str(relation))
            realized_count += len(selected)
        self._active_relation_realizations = active
        self._active_relation_predicate_directions = predicate_directions
        return {
            "enabled": True,
            "available": bool(plans),
            "selected_plan_id": selected_plan_id,
            "artifact_contract_ok": artifact_contract_ok,
            "certified_only": certified_only,
            "entity_binding_contract": str(
                self.relation_realization_map.get("entity_binding_contract", "") or ""
            ),
            "relation_keyed_catalog": self.enable_relation_keyed_catalog,
            "matched_schema_fallback_enabled": self.relation_keyed_allow_matched_schema_fallback,
            "catalog_lookup_count": catalog_lookup_count,
            "fallback_realization_count": fallback_realization_count,
            "unresolved_relations": [
                relation for relation, candidates in output.items() if not candidates
            ],
            "relations": output,
            "realized_relation_count": realized_count,
        }

    @classmethod
    def _surface_relations(cls, query: str, answer_type: str) -> List[str]:
        query_norm = cls._entity_key(query)
        non_event_birth = any(
            re.search(pattern, query_norm)
            for pattern in cls._NON_EVENT_BIRTH_CONTEXTS
        )
        positioned: List[Tuple[int, str]] = []
        answer_family = QueryPlanOntologyBindProbeJoin._answer_type_family(answer_type)
        for canonical, pattern in cls._TYPED_RELATION_PATTERNS:
            match = re.search(pattern, query_norm)
            if not match:
                continue
            if canonical == "born" and non_event_birth:
                continue
            relation = canonical
            if canonical == "born":
                relation = "born in" if answer_family == "location" else "born on"
            elif canonical == "died":
                relation = "died in" if answer_family == "location" else "died on"
            positioned.append((int(match.start()), relation))
        positioned.sort(key=lambda item: item[0])
        return cls._unique((relation for _, relation in positioned), limit=6)

    @classmethod
    def _parenthetical_role_qualifiers(cls, query: str) -> List[str]:
        qualifiers: List[str] = []
        for value in re.findall(r"\(([^()]*)\)", str(query or "")):
            normalized = cls._entity_key(value)
            tokens = set(cls._identity_tokens(normalized))
            if tokens and tokens.intersection(cls._NON_ENTITY_ROLE_TOKENS):
                qualifiers.append(normalized)
        return cls._unique(qualifiers, limit=8)

    @classmethod
    def _is_parenthetical_role_relation(
        cls,
        relation: Any,
        qualifiers: Sequence[str],
    ) -> bool:
        relation_tokens = set(cls._identity_tokens(relation))
        if not relation_tokens:
            return False
        for qualifier in qualifiers:
            qualifier_tokens = set(cls._identity_tokens(qualifier))
            if relation_tokens.issubset(qualifier_tokens):
                return True
            if max(
                cls._typed_relation_overlap(relation, qualifier)[0],
                cls._typed_relation_overlap(qualifier, relation)[0],
            ) >= 0.80:
                return True
        return False

    @classmethod
    def _semantic_unique_relations(
        cls,
        relations: Iterable[Any],
        limit: int = 6,
    ) -> List[str]:
        output: List[str] = []
        for relation in cls._unique(relations, limit=16):
            if any(
                max(
                    cls._typed_relation_overlap(relation, previous)[0],
                    cls._typed_relation_overlap(previous, relation)[0],
                ) >= 0.80
                for previous in output
            ):
                continue
            output.append(relation)
            if len(output) >= limit:
                break
        return output

    @classmethod
    def _typed_relation_overlap(cls, relation: Any, predicate: Any) -> Tuple[float, List[str]]:
        aliases = {
            "born": "birth",
            "birth": "birth",
            "died": "death",
            "death": "death",
            "graduated": "graduate",
            "graduating": "graduate",
            "educated": "education",
            "study": "education",
            "studies": "education",
            "studying": "education",
            "studied": "education",
            "attended": "education",
            "directed": "director",
            "direct": "director",
            "performed": "performer",
            "performer": "performer",
            "singer": "performer",
            "sang": "performer",
            "starred": "performer",
            "starring": "performer",
            "acted": "performer",
            "actor": "performer",
            "actress": "performer",
            "burial": "burial",
            "buried": "burial",
            "interred": "burial",
            "composed": "composer",
            "written": "author",
            "wrote": "author",
            "writer": "author",
            "issued": "issue",
            "founded": "found",
            "established": "found",
            "located": "location",
            "citizenship": "nationality",
            "married": "spouse",
            "wife": "spouse",
            "husband": "spouse",
        }

        def semantic_tokens(value: Any) -> List[str]:
            return [aliases.get(token, token) for token in cls._identity_tokens(value)]

        left = semantic_tokens(relation)
        right = semantic_tokens(predicate)
        if not left or not right:
            return 0.0, []
        right_set = set(right)
        hits = [token for token in left if token in right_set]
        return min(len(hits) / max(len(set(left)), 1), 1.0), hits

    def _relation_path_candidates(
        self,
        query_plan_log: Mapping[str, Any],
    ) -> List[Dict[str, Any]]:
        output: List[Dict[str, Any]] = []
        seen_plan_ids = set()
        condition = query_plan_log.get("query_condition", {}) or {}
        sidecar_plans = (
            list(condition.get("compiler_schema_plans", []) or [])
            if self.enable_execution_aware_contract and isinstance(condition, Mapping)
            else []
        )
        plan_sources = [
            (plan, "schema_linked_sidecar") for plan in sidecar_plans
        ] + [
            (plan, "matched_plan")
            for plan in query_plan_log.get("matched_plans", []) or []
        ]
        for plan, source in plan_sources:
            if not isinstance(plan, Mapping):
                continue
            plan_id = str(plan.get("id", "") or "")
            if plan_id and plan_id in seen_plan_ids:
                continue
            if plan_id:
                seen_plan_ids.add(plan_id)
            relations = []
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
            raw_relation_count = len([relation for relation in relations if relation])
            relations = self._unique(relations, limit=6)
            if relations:
                precision = plan.get("effective_precision")
                if precision is None:
                    precision = plan.get("precision", 0.0)
                output.append({
                    "relations": relations,
                    "raw_relation_count": raw_relation_count,
                    "source": source,
                    "plan_id": plan_id,
                    "plan_scale": float(plan.get("lift_weight_scale", 0.0) or 0.0),
                    "precision": float(precision or 0.0),
                    "confidence": float(plan.get("confidence", 0.0) or 0.0),
                    "answer_type": str(plan.get("answer_type", "") or ""),
                    "anchor_slots": list(plan.get("anchor_slots", []) or []),
                    "path": list(plan.get("path", []) or []),
                    "answer_constraints": dict(plan.get("answer_constraints", {}) or {}),
                    "integrity_constraints": dict(plan.get("integrity_constraints", {}) or {}),
                    "schema_sha256": str(plan.get("schema_sha256", "") or ""),
                    "schema_link_score": float(plan.get("schema_link_score", 0.0) or 0.0),
                })
        for path in query_plan_log.get("sample_paths", []) or []:
            if not isinstance(path, Mapping):
                continue
            relations = self._unique(path.get("relations", []) or [], limit=6)
            if relations:
                output.append({
                    "relations": relations,
                    "source": "sample_path",
                    "plan_id": str(path.get("plan_id", "") or ""),
                    "plan_scale": float(path.get("plan_lift_weight_scale", 0.0) or 0.0),
                })
        return output

    @classmethod
    def _answer_type_compatibility(cls, query_type: Any, schema_type: Any) -> float:
        query_family = QueryPlanOntologyBindProbeJoin._answer_type_family(query_type)
        schema_family = QueryPlanOntologyBindProbeJoin._answer_type_family(schema_type)
        generic = {"", "unknown", "entity", "string", "answer"}
        if query_family in generic:
            return 0.75
        if schema_family == query_family:
            return 1.0
        if schema_family in generic:
            return 0.35
        return 0.0

    @classmethod
    def _path_predicate_coverage(
        cls,
        expected_relations: Sequence[str],
        path_relations: Sequence[str],
    ) -> float:
        if not expected_relations or not path_relations:
            return 0.0
        covered = 0
        for expected in expected_relations:
            best = max(
                (
                    max(
                        cls._typed_relation_overlap(expected, relation)[0],
                        cls._typed_relation_overlap(relation, expected)[0],
                    )
                    for relation in path_relations
                ),
                default=0.0,
            )
            if best >= 0.60:
                covered += 1
        return covered / max(len(expected_relations), 1)

    def _compile_typed_contract(
        self,
        query: str,
        query_plan_log: Mapping[str, Any],
        plan: Mapping[str, Any],
    ) -> Dict[str, Any]:
        condition = query_plan_log.get("query_condition", {}) or {}
        if not isinstance(condition, Mapping):
            condition = {}
        schema_answer_type = str(plan.get("answer_type", "") or "")
        condition_answer_type = str(condition.get("answer_type", "") or "")
        question_answer_type = QueryPlanOntologyBindProbeJoin._question_answer_type(query, "")
        terminal_seed_type = question_answer_type
        if terminal_seed_type in {"", "unknown", "entity", "string", "answer"}:
            terminal_seed_type = condition_answer_type or schema_answer_type
        terminal_profile = QueryPlanOntologyBindProbeJoin._terminal_relation_profile(
            query,
            {"answer_type": terminal_seed_type},
        )
        expected_answer_types = list(terminal_profile.get("expected_answer_types", []) or [])
        if (
            self.enable_execution_aware_contract
            and terminal_profile.get("resolved")
            and expected_answer_types
        ):
            answer_type = str(expected_answer_types[0])
            answer_type_source = "query_terminal_relation"
        elif self.enable_execution_aware_contract and question_answer_type not in {
            "", "unknown", "entity", "string", "answer",
        }:
            answer_type = str(question_answer_type)
            answer_type_source = "query_question_form"
        elif self.enable_execution_aware_contract and condition_answer_type not in {
            "", "unknown", "entity", "string", "answer",
        }:
            answer_type = condition_answer_type
            answer_type_source = "query_condition"
        else:
            answer_type = schema_answer_type or condition_answer_type or question_answer_type
            answer_type_source = "matched_plan"
        raw_anchors = list(plan.get("anchors", []) or [])
        slot_binding: Dict[str, Any] = {"enabled": False}
        if self.enable_executable_ontology_compiler:
            slot_binding = self._bind_query_surface_slots(
                query,
                raw_anchors,
                enable_branch_compiler=True,
            )
            anchors = list(slot_binding.get("executable_anchors", []) or [])
            comparison_query = bool(slot_binding.get("comparison_query", False))
        else:
            anchors = self._canonical_anchors(query, raw_anchors)
            comparison_query = self._is_comparison(query, anchors)
        comparison_branches: Dict[str, Any] = {"enabled": False, "complete": True, "branches": []}
        if self.enable_executable_ontology_compiler and comparison_query:
            comparison_branches = dict(slot_binding.get("comparison_branches", {}) or {})
        elif self.enable_branch_complete_compiler and comparison_query:
            comparison_branches = self._compile_comparison_branches(query, anchors)
        if comparison_query:




            answer_type = "entity"
            answer_type_source = "comparison_branch_projection"
        surface_relations = self._surface_relations(query, answer_type)
        parenthetical_role_qualifiers = self._parenthetical_role_qualifiers(query)
        surface_relations = [
            relation for relation in surface_relations
            if not self._is_parenthetical_role_relation(
                relation,
                parenthetical_role_qualifiers,
            )
        ]
        query_norm = self._entity_key(query)
        non_event_birth = any(
            re.search(pattern, query_norm)
            for pattern in self._NON_EVENT_BIRTH_CONTEXTS
        )

        condition_relations: List[str] = []
        for value in (
            list(condition.get("relation_terms", []) or [])
            + list(condition.get("relation_signatures", []) or [])
        ):
            relation = QueryPlanOntologyBindProbeJoin._norm(value)
            if non_event_birth and relation in {"birth", "birth date", "birth place"}:
                continue
            relation_tokens = set(self._identity_tokens(relation))
            if relation_tokens and relation_tokens.issubset(self._ANSWER_DESCRIPTOR_TOKENS):
                continue
            if self._is_parenthetical_role_relation(
                relation,
                parenthetical_role_qualifiers,
            ):
                continue
            query_tokens = set(self._identity_tokens(query_norm))
            if relation_tokens and relation_tokens.intersection(query_tokens):
                condition_relations.append(relation)

        expected_relations = self._semantic_unique_relations(
            list(surface_relations) + condition_relations,
            limit=6,
        )
        if (
            terminal_profile.get("resolved")
            and terminal_profile.get("canonical_relation")
            and not non_event_birth
        ):
            terminal_relation = str(terminal_profile.get("canonical_relation", "") or "")
            if not any(
                max(
                    self._typed_relation_overlap(terminal_relation, relation)[0],
                    self._typed_relation_overlap(relation, terminal_relation)[0],
                ) >= 0.60
                for relation in expected_relations
            ):
                expected_relations.append(terminal_relation)
        expected_relations = self._semantic_unique_relations(expected_relations, limit=6)

        ranked_paths: List[Dict[str, Any]] = []
        for candidate in self._relation_path_candidates(query_plan_log):
            candidate = dict(candidate)
            candidate["relations"] = self._semantic_unique_relations([
                relation
                for relation in list(candidate.get("relations", []) or [])
                if not self._is_parenthetical_role_relation(
                    relation,
                    parenthetical_role_qualifiers,
                )
            ], limit=self.max_hops)
            if not candidate["relations"]:
                continue
            coverage = self._path_predicate_coverage(
                expected_relations,
                candidate.get("relations", []) or [],
            )
            candidate["query_predicate_coverage"] = round(float(coverage), 6)
            candidate["answer_type_compatibility"] = round(
                self._answer_type_compatibility(
                    answer_type,
                    candidate.get("answer_type", ""),
                ),
                6,
            )
            if self.enable_lossless_typed_plan_ir:
                typed_ir = self._compile_lossless_plan_ir(
                    query=query,
                    candidate=candidate,
                    anchors=anchors,
                    answer_type=answer_type,
                    comparison=comparison_query,
                    comparison_branches=comparison_branches,
                )
                typed_ir_score = self._typed_ir_semantic_score(
                    typed_ir=typed_ir,
                    expected_relations=expected_relations,
                    terminal_profile=terminal_profile,
                    answer_type_compatibility=float(
                        candidate.get("answer_type_compatibility", 0.0) or 0.0
                    ),
                    comparison=comparison_query,
                )
                candidate["typed_plan_ir"] = typed_ir
                candidate["typed_ir_semantic_score"] = typed_ir_score
            candidate_relations = list(candidate.get("relations", []) or [])
            candidate["structural_nonredundancy"] = round(
                len(self._semantic_unique_relations(candidate_relations, limit=6))
                / max(int(candidate.get("raw_relation_count", len(candidate_relations)) or 0), 1),
                6,
            )
            ranked_paths.append(candidate)

        ranked_paths.sort(
            key=lambda candidate: (
                float(
                    dict(candidate.get("typed_ir_semantic_score", {}) or {}).get(
                        "score", 0.0
                    )
                ) if self.enable_lossless_typed_plan_ir else 0.0,
                int(
                    bool(
                        dict(candidate.get("typed_plan_ir", {}) or {}).get(
                            "valid", False
                        )
                    )
                ) if self.enable_lossless_typed_plan_ir else 0,
                float(candidate.get("query_predicate_coverage", 0.0)),
                float(candidate.get("answer_type_compatibility", 0.0)),
                float(candidate.get("precision", 0.0)),
                float(candidate.get("schema_link_score", 0.0)),
                float(candidate.get("structural_nonredundancy", 0.0)),
                float(candidate.get("confidence", 0.0)),
                float(candidate.get("plan_scale", 0.0)),
            ),
            reverse=True,
        )
        deduplicated_paths: List[Dict[str, Any]] = []
        seen_relation_paths = set()
        for candidate in ranked_paths:
            if self.enable_lossless_typed_plan_ir:
                candidate_ir = dict(candidate.get("typed_plan_ir", {}) or {})
                relation_key = (
                    str(candidate_ir.get("topology", "") or ""),
                    tuple(
                        (
                            str(edge.get("input_variable", "") or ""),
                            self._entity_key(edge.get("relation", "")),
                            str(edge.get("direction", "either") or "either"),
                            str(edge.get("output_variable", "") or ""),
                            str(edge.get("branch_id", "main") or "main"),
                        )
                        for edge in candidate_ir.get("edges", []) or []
                        if isinstance(edge, Mapping)
                    ),
                )
            else:
                relation_key = tuple(
                    self._entity_key(relation)
                    for relation in candidate.get("relations", []) or []
                )
            if not relation_key or relation_key in seen_relation_paths:
                continue
            seen_relation_paths.add(relation_key)
            deduplicated_paths.append(candidate)
        ranked_paths = deduplicated_paths
        best_path: Dict[str, Any] = dict(ranked_paths[0]) if ranked_paths else {}

        ontology_relation_alternatives: List[Dict[str, Any]] = []
        if self.enable_query_relation_alternative_recovery:
            for candidate in ranked_paths:
                answer_compatibility = float(
                    candidate.get("answer_type_compatibility", 0.0) or 0.0
                )
                precision = float(candidate.get("precision", 0.0) or 0.0)
                predicate_coverage = float(
                    candidate.get("query_predicate_coverage", 0.0) or 0.0
                )




                if answer_compatibility <= 0.0:
                    continue
                if predicate_coverage <= 0.0 and precision < 0.50:
                    continue
                ontology_relation_alternatives.append(dict(candidate))
                if len(ontology_relation_alternatives) >= (
                    self.max_query_relation_alternatives + 1
                ):
                    break

        use_path = bool(
            best_path
            and float(best_path.get("query_predicate_coverage", 0.0))
            >= self.min_query_predicate_coverage
            and (
                not self.enable_execution_aware_contract
                or float(best_path.get("answer_type_compatibility", 0.0)) > 0.0
            )
            and (
                not self.enable_lossless_typed_plan_ir
                or bool(
                    dict(best_path.get("typed_plan_ir", {}) or {}).get(
                        "valid", False
                    )
                )
            )
        )
        relations = self._ordered_relation_contract(
            surface_relations=expected_relations,
            ontology_relations=(best_path.get("relations", []) or []) if use_path else [],
            terminal_profile=(
                terminal_profile
                if not non_event_birth
                else {"resolved": False, "canonical_relation": "", "aliases": []}
            ),
            ontology_order_is_authoritative=use_path,
        )
        if use_path and best_path.get("answer_type") and not self.enable_execution_aware_contract:
            answer_type = str(best_path.get("answer_type", "") or answer_type)
        predicate_coverage = (
            float(best_path.get("query_predicate_coverage", 0.0))
            if use_path else (1.0 if expected_relations else 0.0)
        )
        anchor_exact_count = sum(
            1 for anchor in anchors
            if f" {anchor} " in f" {query_norm} "
        )
        anchor_grounding = anchor_exact_count / max(len(anchors), 1)
        confidence = (
            0.45 * predicate_coverage
            + 0.35 * anchor_grounding
            + 0.20 * float(bool(terminal_profile.get("resolved") or len(relations) >= 2))
        )
        admissible = bool(
            anchors
            and relations
            and predicate_coverage >= self.min_query_predicate_coverage
            and anchor_grounding >= 1.0
            and (
                not comparison_query
                or not (self.enable_branch_complete_compiler or self.enable_executable_ontology_compiler)
                or comparison_branches.get("complete", False)
            )
            and (
                not self.enable_lossless_typed_plan_ir
                or bool(
                    dict(best_path.get("typed_plan_ir", {}) or {}).get(
                        "valid", False
                    )
                )
            )
        )
        answer_type_compatibility = (
            float(best_path.get("answer_type_compatibility", 0.0))
            if best_path else 0.0
        )
        execution_contract_valid = bool(
            not best_path
            or not self.enable_execution_aware_contract
            or answer_type_compatibility > 0.0
        )
        return {
            "enabled": True,
            "admissible": admissible,
            "anchors": anchors,
            "relations": self._semantic_unique_relations(relations, limit=6),
            "surface_relations": surface_relations,
            "condition_relations": self._unique(condition_relations, limit=6),
            "parenthetical_role_qualifiers": parenthetical_role_qualifiers,
            "relation_source": str(best_path.get("source", "query_surface") if use_path else "query_surface"),
            "source_plan_id": str(best_path.get("plan_id", "") if use_path else ""),
            "answer_type": answer_type,
            "schema_answer_type": str(best_path.get("answer_type", "") or schema_answer_type),
            "answer_type_source": answer_type_source,
            "answer_type_compatibility": round(answer_type_compatibility, 6),
            "execution_contract_valid": execution_contract_valid,
            "terminal_relation_profile": terminal_profile,
            "selected_schema": best_path if use_path else {},
            "logical_plan_ir": (
                dict(best_path.get("typed_plan_ir", {}) or {}) if use_path else {}
            ),
            "typed_ir_semantic_score": dict(
                best_path.get("typed_ir_semantic_score", {}) or {}
            ),
            "typed_ir_candidates": [
                {
                    "rank": rank,
                    "plan_id": str(candidate.get("plan_id", "") or ""),
                    "relations": list(candidate.get("relations", []) or []),
                    "semantic_score": dict(
                        candidate.get("typed_ir_semantic_score", {}) or {}
                    ),
                    "typed_plan_ir": dict(candidate.get("typed_plan_ir", {}) or {}),
                }
                for rank, candidate in enumerate(ranked_paths[:6], start=1)
            ] if self.enable_lossless_typed_plan_ir else [],
            "query_predicate_coverage": round(float(predicate_coverage), 6),
            "anchor_grounding": round(float(anchor_grounding), 6),
            "contract_confidence": round(float(confidence), 6),
            "comparison_query": comparison_query,
            "comparison_branches": comparison_branches,
            "query_surface_slot_binding": slot_binding,
            "non_event_birth_context": non_event_birth,
            "best_ontology_path": best_path,
            "ontology_relation_alternatives": ontology_relation_alternatives,
            "ontology_relation_alternative_count": len(ontology_relation_alternatives),
            "rejection_reason": "" if admissible else (
                "no_exact_query_anchor" if anchor_grounding < 1.0
                else "invalid_typed_plan_ir" if (
                    self.enable_lossless_typed_plan_ir
                    and not bool(
                        dict(best_path.get("typed_plan_ir", {}) or {}).get(
                            "valid", False
                        )
                    )
                )
                else "comparison_branch_incomplete" if (
                    comparison_query
                    and (self.enable_branch_complete_compiler or self.enable_executable_ontology_compiler)
                    and not comparison_branches.get("complete", False)
                )
                else "insufficient_query_predicate_coverage"
            ),
        }

    @classmethod
    def _is_comparison(cls, query: str, anchors: Sequence[str]) -> bool:
        query_norm = QueryPlanOntologyBindProbeJoin._norm(query)
        if len(anchors) < 2:
            return False
        branch_contract = cls._compile_comparison_branches(query, anchors)
        if not branch_contract.get("complete", False):
            return False
        strong_comparison = any(
            re.search(rf"\b{re.escape(cue)}\b", query_norm)
            for cue in cls._COMPARISON_CUES
        )
        interrogative = bool(re.search(r"\b(?:which|who|what)\b", query_norm))
        separator = str(branch_contract.get("separator", "") or "")
        explicit_choice = interrogative and separator == "or"
        coordinated_comparison = separator in {"and", "or"} and strong_comparison
        return explicit_choice or coordinated_comparison

    @staticmethod
    def _exact_endpoint_match(endpoint: str, binding: str) -> bool:
        endpoint_keys = set(QueryPlanFactorizedExecutor._entity_alias_keys(endpoint))
        binding_keys = set(QueryPlanFactorizedExecutor._entity_alias_keys(binding))
        return bool(endpoint_keys and binding_keys and endpoint_keys.intersection(binding_keys))

    def _opposite_endpoint(
        self,
        subject: str,
        obj: str,
        binding: str,
        direction: str = "either",
    ) -> Tuple[str, float]:
        helper = QueryPlanOntologyBindProbeJoin
        direction = str(direction or "either").lower()
        if self.require_exact_entity_keys:
            if direction != "reverse" and self._exact_endpoint_match(subject, binding):
                return obj, 1.0
            if direction != "forward" and self._exact_endpoint_match(obj, binding):
                return subject, 1.0
            return "", 0.0
        if direction != "reverse" and helper._phrase_hit(subject, binding, min_coverage=0.76):
            return obj, 1.0
        if direction != "forward" and helper._phrase_hit(obj, binding, min_coverage=0.76):
            return subject, 1.0
        subject_overlap, _ = helper._token_overlap(binding, subject)
        object_overlap, _ = helper._token_overlap(binding, obj)
        if direction == "forward":
            object_overlap = 0.0
        elif direction == "reverse":
            subject_overlap = 0.0
        if max(subject_overlap, object_overlap) < 0.66:
            return "", 0.0
        return (obj, subject_overlap) if subject_overlap >= object_overlap else (subject, object_overlap)

    def _score_entry(
        self,
        entry: Mapping[str, Any],
        binding: str,
        relation: str,
        answer_type: str,
        terminal_step: bool,
        direction_override: str = "",
    ) -> Dict[str, Any]:
        helper = QueryPlanOntologyBindProbeJoin
        subject = str(entry.get("subject", "") or "")
        predicate = str(entry.get("predicate", "") or "")
        obj = str(entry.get("object", "") or "")
        configured_direction = self._active_relation_directions.get(relation, "either")
        predicate_key = self._entity_key(predicate)
        predicate_direction = self._active_relation_predicate_directions.get(
            relation, {}
        ).get(predicate_key, configured_direction)
        expected_direction = str(direction_override or predicate_direction or "either")
        if self.enable_typed_semantic_compiler:
            relation_score = 0.0
            relation_hits: List[str] = []
            matched_relation = relation
            relation_surfaces = self._active_relation_realizations.get(relation, [relation])
            if relation in self._strict_relation_predicates:
                exact_matches = [
                    relation_surface for relation_surface in relation_surfaces
                    if self._entity_key(relation_surface) == predicate_key
                ]
                if exact_matches:
                    relation_score = 1.0
                    relation_hits = self._identity_tokens(predicate)
                    matched_relation = exact_matches[0]
            else:
                for relation_surface in relation_surfaces:
                    forward_score, forward_hits = self._typed_relation_overlap(relation_surface, predicate)
                    reverse_score, reverse_hits = self._typed_relation_overlap(predicate, relation_surface)
                    if reverse_score > forward_score:
                        score, hits = reverse_score, reverse_hits
                    else:
                        score, hits = forward_score, forward_hits
                    if score > relation_score:
                        relation_score = score
                        relation_hits = hits
                        matched_relation = relation_surface
        else:
            relation_score, relation_hits = helper._relation_overlap(relation, predicate)
            matched_relation = relation
        terminal_entity, entity_score = self._opposite_endpoint(
            subject,
            obj,
            binding,
            direction=expected_direction,
        )
        type_score = helper._terminal_type_score(terminal_entity, answer_type) if terminal_step else 1.0
        index_match_score = (
            float(entity_score)
            if self.require_exact_entity_keys
            else min(max(float(entry.get("entity_match_score", entity_score) or 0.0), 0.0), 1.0)
        )
        raw_score = (
            0.34 * index_match_score
            + 0.36 * relation_score
            + 0.20 * type_score
            + 0.10 * entity_score
        )
        return {
            "passage_idx": int(entry.get("passage_idx", -1)),
            "subject": subject,
            "predicate": predicate,
            "object": obj,
            "terminal_entity": terminal_entity,
            "entity_score": round(float(entity_score), 6),
            "relation_score": round(float(relation_score), 6),
            "relation_hits": relation_hits,
            "matched_relation_surface": matched_relation,
            "expected_direction": predicate_direction,
            "access_direction": expected_direction,
            "inverse_direction_fallback": bool(
                direction_override and expected_direction != configured_direction
            ),
            "warehouse_lookup_mode": str(entry.get("warehouse_lookup_mode", "exact_key") or "exact_key"),
            "warehouse_entity_key": str(entry.get("warehouse_entity_key", "") or ""),
            "type_score": round(float(type_score), 6),
            "exact_entity_key": bool(
                self._exact_endpoint_match(subject, binding)
                or self._exact_endpoint_match(obj, binding)
            ),
            "raw_score": round(float(raw_score), 6),
            "triple": [subject, predicate, obj],
        }

    def _indexed_access(
        self,
        binding: str,
        relation: str,
        answer_type: str,
        terminal_step: bool,
        lookup_entity_relations: Callable[[str, int], Sequence[Mapping[str, Any]]],
        direction_override: str = "",
    ) -> List[Dict[str, Any]]:
        rows = list(lookup_entity_relations(binding, self.max_index_rows) or [])
        scored: List[Dict[str, Any]] = []
        for row in rows:
            profile = self._score_entry(
                row,
                binding,
                relation,
                answer_type,
                terminal_step,
                direction_override=direction_override,
            )
            if profile["passage_idx"] < 0 or not profile["terminal_entity"]:
                continue
            if float(profile["relation_score"]) < self.min_relation_score:
                continue
            if terminal_step and float(profile["type_score"]) < self.min_type_score:
                continue
            scored.append(profile)
        scored.sort(
            key=lambda item: (
                float(item.get("raw_score", 0.0)),
                float(item.get("relation_score", 0.0)),
                float(item.get("type_score", 0.0)),
            ),
            reverse=True,
        )
        expected_direction = str(
            direction_override
            or self._active_relation_directions.get(relation, "either")
            or "either"
        )
        if (
            scored
            or not self.enable_inverse_direction_fallback
            or expected_direction not in {"forward", "reverse"}
        ):
            return scored

        inverse_direction = "reverse" if expected_direction == "forward" else "forward"
        for row in rows:
            profile = self._score_entry(
                row,
                binding,
                relation,
                answer_type,
                terminal_step,
                direction_override=inverse_direction,
            )
            if profile["passage_idx"] < 0 or not profile["terminal_entity"]:
                continue



            if not profile["exact_entity_key"]:
                continue
            if float(profile["relation_score"]) < self.min_relation_score:
                continue
            if terminal_step and float(profile["type_score"]) < self.min_type_score:
                continue
            profile["raw_score"] = round(
                float(profile["raw_score"]) * (1.0 - self.inverse_direction_penalty),
                6,
            )
            scored.append(profile)
        scored.sort(
            key=lambda item: (
                float(item.get("raw_score", 0.0)),
                float(item.get("relation_score", 0.0)),
                float(item.get("type_score", 0.0)),
            ),
            reverse=True,
        )
        return scored

    @classmethod
    def _valid_intermediate_join_key(cls, value: Any) -> bool:
        normalized = QueryPlanOntologyBindProbeJoin._norm(value)
        if not normalized or len(normalized) < 3 or len(normalized) > 96:
            return False
        if re.fullmatch(r"[\d\s\-/:,.]+", normalized):
            return False
        if re.fullmatch(r"(?:\d{1,2}\s+)?(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{3,4}", normalized):
            return False
        tokens = cls._identity_tokens(normalized)
        if not tokens or all(token.isdigit() for token in tokens):
            return False
        return True

    def _dense_access(
        self,
        query: str,
        binding: str,
        relation: str,
        answer_type: str,
        terminal_step: bool,
        dense_retrieve: Callable[[str], Tuple[Sequence[int], Sequence[float]]],
        get_openie_record: Callable[[int], Mapping[str, Any]],
        direction_override: str = "",
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        probe_query = " ".join(piece for piece in (query, binding, relation, answer_type) if piece)
        try:
            dense_ids, dense_scores = dense_retrieve(probe_query)
        except Exception as exc:
            return [], {"query": probe_query, "error": str(exc)[:200]}
        dense_ids = list(dense_ids[: self.dense_top_k])
        dense_scores = list(dense_scores[: self.dense_top_k])
        max_dense = max((float(score) for score in dense_scores), default=0.0)
        rows: List[Dict[str, Any]] = []
        for dense_rank, passage_idx_value in enumerate(dense_ids, start=1):
            passage_idx = int(passage_idx_value)
            record = get_openie_record(passage_idx) or {}
            for triple in record.get("extracted_triples", []) or []:
                if not isinstance(triple, (list, tuple)) or len(triple) != 3:
                    continue
                subject, predicate, obj = [str(value or "") for value in triple]
                profile = self._score_entry(
                    {
                        "passage_idx": passage_idx,
                        "subject": subject,
                        "predicate": predicate,
                        "object": obj,
                        "entity_match_score": 1.0,
                    },
                    binding,
                    relation,
                    answer_type,
                    terminal_step,
                    direction_override=direction_override,
                )
                if not profile["terminal_entity"]:
                    continue
                if float(profile["relation_score"]) < self.min_relation_score:
                    continue
                if terminal_step and float(profile["type_score"]) < self.min_type_score:
                    continue
                dense_raw = float(dense_scores[dense_rank - 1]) if dense_rank <= len(dense_scores) else 0.0
                dense_norm = dense_raw / max_dense if max_dense > 0.0 else 1.0 / dense_rank
                profile["dense_rank"] = dense_rank
                profile["dense_score"] = round(float(dense_norm), 6)
                profile["raw_score"] = round(
                    0.82 * float(profile["raw_score"]) + 0.18 * min(max(dense_norm, 0.0), 1.0),
                    6,
                )
                rows.append(profile)
        best_by_key: Dict[Tuple[int, str], Dict[str, Any]] = {}
        for row in rows:
            key = (int(row["passage_idx"]), QueryPlanOntologyBindProbeJoin._norm(row["terminal_entity"]))
            if key not in best_by_key or float(row["raw_score"]) > float(best_by_key[key]["raw_score"]):
                best_by_key[key] = row
        selected = sorted(
            best_by_key.values(),
            key=lambda item: float(item.get("raw_score", 0.0)),
            reverse=True,
        )
        return selected, {"query": probe_query, "dense_count": len(dense_ids), "matched_count": len(selected)}

    def _access(
        self,
        query: str,
        binding: str,
        relation: str,
        answer_type: str,
        terminal_step: bool,
        dense_retrieve: Callable[[str], Tuple[Sequence[int], Sequence[float]]],
        get_openie_record: Callable[[int], Mapping[str, Any]],
        lookup_entity_relations: Callable[[str, int], Sequence[Mapping[str, Any]]],
        probes: List[Dict[str, Any]],
        direction_override: str = "",
    ) -> List[Dict[str, Any]]:
        if self.physical_access_strategy == "dense_first":
            started_at = time.perf_counter()
            dense_rows, dense_log = self._dense_access(
                query,
                binding,
                relation,
                answer_type,
                terminal_step,
                dense_retrieve,
                get_openie_record,
                direction_override=direction_override,
            )
            dense_log["access_path"] = "dense_first"
            dense_log["binding"] = binding
            dense_log["relation"] = relation
            dense_log["latency_ms"] = round(
                (time.perf_counter() - started_at) * 1000.0,
                6,
            )
            probes.append(dense_log)
            return dense_rows
        started_at = time.perf_counter()
        indexed = self._indexed_access(
            binding,
            relation,
            answer_type,
            terminal_step,
            lookup_entity_relations,
            direction_override=direction_override,
        )
        probes.append({
            "access_path": "entity_relation_index",
            "binding": binding,
            "relation": relation,
            "matched_count": len(indexed),
            "inverse_direction_fallback_count": sum(
                1 for item in indexed if item.get("inverse_direction_fallback", False)
            ),
            "latency_ms": round(
                (time.perf_counter() - started_at) * 1000.0,
                6,
            ),
        })
        if (
            self.physical_access_strategy == "index_only"
            or indexed
            or len(probes) >= self.max_probes
        ):
            return indexed
        started_at = time.perf_counter()
        dense_rows, dense_log = self._dense_access(
            query,
            binding,
            relation,
            answer_type,
            terminal_step,
            dense_retrieve,
            get_openie_record,
            direction_override=direction_override,
        )
        dense_log["access_path"] = "dense_fallback"
        dense_log["binding"] = binding
        dense_log["relation"] = relation
        dense_log["latency_ms"] = round(
            (time.perf_counter() - started_at) * 1000.0,
            6,
        )
        probes.append(dense_log)
        return dense_rows

    def _entity_relation_index_scan(
        self,
        query: str,
        anchors: Sequence[str],
        relations: Sequence[str],
        answer_type: str,
        dense_retrieve: Callable[[str], Tuple[Sequence[int], Sequence[float]]],
        get_openie_record: Callable[[int], Mapping[str, Any]],
        lookup_entity_relations: Callable[[str, int], Sequence[Mapping[str, Any]]],
        probes: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        candidates: List[Dict[str, Any]] = []
        for anchor in anchors[:2]:
            for relation in relations[:3]:
                rows = self._access(
                    query,
                    anchor,
                    relation,
                    answer_type,
                    True,
                    dense_retrieve,
                    get_openie_record,
                    lookup_entity_relations,
                    probes,
                )
                for row in rows[: self.beam_width]:
                    score = float(row.get("raw_score", 0.0))
                    estimated_cost = 0.20 if "dense_rank" not in row else 1.0 + self.dense_top_k / 100.0
                    utility = score - self.cost_weight * estimated_cost
                    if utility < self.min_candidate_utility:
                        continue
                    candidates.append({
                        **row,
                        "operator": "entity_relation_index_scan",
                        "anchor_entity": anchor,
                        "relations": [relation],
                        "answer_type": answer_type,
                        "estimated_cost": round(estimated_cost, 6),
                        "utility": round(utility, 6),
                    })
                if len(probes) >= self.max_probes:
                    break
            if len(probes) >= self.max_probes:
                break
        return candidates, []

    def _provenance_plan_portfolio(
        self,
        query: str,
        anchors: Sequence[str],
        relations: Sequence[str],
        answer_type: str,
        primary_operator: str,
        dense_retrieve: Callable[[str], Tuple[Sequence[int], Sequence[float]]],
        get_openie_record: Callable[[int], Mapping[str, Any]],
        lookup_entity_relations: Callable[[str, int], Sequence[Mapping[str, Any]]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Execute independently valid plan fragments after a full join misses.

        Each fragment carries the ontology relation slot it proves. This is an
        outer-join fallback over physical access paths: one failed hop cannot
        erase evidence already validated for another hop or comparison branch.
        """
        relation_path = list(relations[: self.max_hops])
        portfolio_probes: List[Dict[str, Any]] = []
        candidates: List[Dict[str, Any]] = []
        for anchor in anchors[:2]:
            for slot_idx, relation in enumerate(relation_path):
                if len(portfolio_probes) >= self.max_probes:
                    break
                terminal_step = slot_idx == len(relation_path) - 1
                slot_answer_type = answer_type if terminal_step else "entity"
                rows = self._access(
                    query,
                    anchor,
                    relation,
                    slot_answer_type,
                    terminal_step,
                    dense_retrieve,
                    get_openie_record,
                    lookup_entity_relations,
                    portfolio_probes,
                )
                for row in rows[: self.beam_width]:
                    score = float(row.get("raw_score", 0.0) or 0.0)
                    estimated_cost = (
                        0.20 if "dense_rank" not in row
                        else 1.0 + self.dense_top_k / 100.0
                    )
                    utility = score - self.cost_weight * estimated_cost
                    if utility < self.min_candidate_utility:
                        continue
                    passage_idx = int(row.get("passage_idx", -1) or -1)
                    if passage_idx < 0:
                        continue
                    candidates.append({
                        **row,
                        "operator": "entity_relation_index_scan",
                        "anchor_entity": anchor,
                        "relations": [relation],
                        "answer_type": answer_type,
                        "estimated_cost": round(estimated_cost, 6),
                        "utility": round(utility, 6),
                        "provenance_plan_portfolio": True,
                        "portfolio_primary_operator": primary_operator,
                        "proof_relation_slot": slot_idx,
                        "proof_relation": relation,
                        "partial_proof_completeness": round(
                            1.0 / max(len(relation_path), 1),
                            6,
                        ),
                    })
            if len(portfolio_probes) >= self.max_probes:
                break
        return candidates, portfolio_probes

    @classmethod
    def _branch_qualifier_coverage(
        cls,
        branch: Mapping[str, Any],
        row: Mapping[str, Any],
        passage: str,
    ) -> float:
        qualifiers = list(branch.get("qualifiers", []) or [])
        if not qualifiers:
            return 1.0
        evidence = " ".join([
            str(passage or ""),
            str(row.get("subject", "") or ""),
            str(row.get("predicate", "") or ""),
            str(row.get("object", "") or ""),
        ])
        evidence_norm = cls._entity_key(evidence)
        hits = sum(
            1
            for qualifier in qualifiers
            if f" {cls._entity_key(qualifier)} " in f" {evidence_norm} "
        )
        return hits / max(len(qualifiers), 1)

    def _cascaded_bind_join(
        self,
        query: str,
        anchors: Sequence[str],
        relations: Sequence[str],
        answer_type: str,
        dense_retrieve: Callable[[str], Tuple[Sequence[int], Sequence[float]]],
        get_openie_record: Callable[[int], Mapping[str, Any]],
        lookup_entity_relations: Callable[[str, int], Sequence[Mapping[str, Any]]],
        probes: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        if len(relations) < 2:
            return [], []
        relation_path = list(relations[: self.max_hops])
        states: List[Dict[str, Any]] = [
            {
                "binding": anchor,
                "bindings": [anchor],
                "passage_indices": [],
                "triples": [],
                "step_scores": [],
                "cost": 0.0,
            }
            for anchor in anchors[:2]
        ]
        for step_idx, relation in enumerate(relation_path):
            next_states: List[Dict[str, Any]] = []
            terminal_step = step_idx == len(relation_path) - 1
            for state in states:
                if len(probes) >= self.max_probes:
                    break
                rows = self._access(
                    query,
                    str(state["binding"]),
                    relation,
                    answer_type,
                    terminal_step,
                    dense_retrieve,
                    get_openie_record,
                    lookup_entity_relations,
                    probes,
                )
                for row in rows[: self.beam_width]:
                    next_entity = str(row.get("terminal_entity", "") or "")
                    if not next_entity:
                        continue
                    if (
                        self.require_typed_join_keys
                        and not terminal_step
                        and not self._valid_intermediate_join_key(next_entity)
                    ):
                        self._runtime_rejections["invalid_intermediate_join_key"] = (
                            self._runtime_rejections.get("invalid_intermediate_join_key", 0) + 1
                        )
                        continue
                    next_key = QueryPlanOntologyBindProbeJoin._norm(next_entity)
                    if next_key in {
                        QueryPlanOntologyBindProbeJoin._norm(value)
                        for value in state.get("bindings", [])
                    }:
                        continue
                    next_states.append({
                        "binding": next_entity,
                        "bindings": list(state["bindings"]) + [next_entity],
                        "passage_indices": list(state["passage_indices"]) + [int(row["passage_idx"])],
                        "triples": list(state["triples"]) + [list(row["triple"])],
                        "step_scores": list(state["step_scores"]) + [float(row["raw_score"])],
                        "cost": float(state["cost"]) + (0.20 if "dense_rank" not in row else 1.0),
                        "terminal_profile": row,
                    })
            if not next_states:
                return [], []
            next_states.sort(
                key=lambda item: sum(item["step_scores"]) / max(len(item["step_scores"]), 1),
                reverse=True,
            )
            states = next_states[: self.beam_width]

        candidates: List[Dict[str, Any]] = []
        groups: List[Dict[str, Any]] = []
        for state_idx, state in enumerate(states):
            passage_indices = list(dict.fromkeys(int(value) for value in state["passage_indices"]))
            if not passage_indices:
                continue
            mean_score = sum(state["step_scores"]) / max(len(state["step_scores"]), 1)
            estimated_cost = float(state["cost"])
            completeness = min(len(state["step_scores"]) / max(len(relation_path), 1), 1.0)
            utility = 0.82 * mean_score + 0.18 * completeness - self.cost_weight * estimated_cost
            if utility < self.min_candidate_utility:
                continue
            group_id = f"cascaded_bind_join::{state_idx}"
            group = {
                "group_id": group_id,
                "operator": "cascaded_bind_join",
                "passage_indices": passage_indices,
                "bindings": list(state["bindings"]),
                "relations": relation_path,
                "triples": list(state["triples"]),
                "answer_type": answer_type,
                "chain_completeness": round(completeness, 6),
                "score": round(mean_score, 6),
                "estimated_cost": round(estimated_cost, 6),
                "utility": round(utility, 6),
                "typed_join_keys_valid": True,
            }
            groups.append(group)
            terminal_profile = dict(state.get("terminal_profile", {}) or {})
            candidates.append({
                **terminal_profile,
                "passage_idx": passage_indices[-1],
                "operator": "cascaded_bind_join",
                "group_id": group_id,
                "anchor_entity": state["bindings"][0],
                "bindings": list(state["bindings"]),
                "relations": relation_path,
                "answer_type": answer_type,
                "estimated_cost": round(estimated_cost, 6),
                "utility": round(utility, 6),
            })
        return candidates, groups

    def _execute_typed_ir_branch(
        self,
        query: str,
        edges: Sequence[Mapping[str, Any]],
        initial_bindings: Mapping[str, str],
        dense_retrieve: Callable[[str], Tuple[Sequence[int], Sequence[float]]],
        get_openie_record: Callable[[int], Mapping[str, Any]],
        lookup_entity_relations: Callable[[str, int], Sequence[Mapping[str, Any]]],
        probes: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        states: List[Dict[str, Any]] = [{
            "variable_bindings": dict(initial_bindings),
            "lineage": [],
            "step_scores": [],
            "cost": 0.0,
        }]
        for edge in edges:
            next_states: List[Dict[str, Any]] = []
            input_variable = str(edge.get("input_variable", "") or "")
            output_variable = str(edge.get("output_variable", "") or "")
            relation = str(edge.get("relation", "") or "")
            direction = str(edge.get("direction", "either") or "either")
            output_type = str(edge.get("output_type", "entity") or "entity")
            validate_output_type = output_type not in {"", "unknown", "string", "answer"}
            for state in states:
                bindings = dict(state.get("variable_bindings", {}) or {})
                input_binding = str(bindings.get(input_variable, "") or "")
                if not input_binding:
                    self._runtime_rejections["typed_ir_unbound_input"] = (
                        self._runtime_rejections.get("typed_ir_unbound_input", 0) + 1
                    )
                    continue
                rows = self._access(
                    query,
                    input_binding,
                    relation,
                    output_type,
                    validate_output_type,
                    dense_retrieve,
                    get_openie_record,
                    lookup_entity_relations,
                    probes,
                    direction_override=direction,
                )
                for row in rows[: self.beam_width]:
                    output_binding = str(row.get("terminal_entity", "") or "")
                    if not output_binding:
                        continue
                    existing = str(bindings.get(output_variable, "") or "")
                    if existing and not self._exact_endpoint_match(existing, output_binding):
                        self._runtime_rejections["typed_ir_variable_conflict"] = (
                            self._runtime_rejections.get("typed_ir_variable_conflict", 0) + 1
                        )
                        continue
                    next_bindings = dict(bindings)
                    next_bindings[output_variable] = output_binding
                    lineage_entry = {
                        **dict(row),
                        "edge_index": int(edge.get("edge_index", 0) or 0),
                        "branch_id": str(edge.get("branch_id", "main") or "main"),
                        "input_variable": input_variable,
                        "input_binding": input_binding,
                        "relation": relation,
                        "direction": direction,
                        "output_variable": output_variable,
                        "output_binding": output_binding,
                        "output_type": output_type,
                        "binding_unified": True,
                    }
                    next_states.append({
                        "variable_bindings": next_bindings,
                        "lineage": list(state.get("lineage", []) or []) + [lineage_entry],
                        "step_scores": list(state.get("step_scores", []) or []) + [
                            float(row.get("raw_score", 0.0) or 0.0)
                        ],
                        "cost": float(state.get("cost", 0.0) or 0.0)
                        + (0.20 if "dense_rank" not in row else 1.0),
                    })
            if not next_states:
                self._runtime_rejections[
                    f"typed_ir_no_binding_edge_{int(edge.get('edge_index', 0) or 0)}"
                ] = 1
                return []
            next_states.sort(
                key=lambda item: sum(item["step_scores"])
                / max(len(item["step_scores"]), 1),
                reverse=True,
            )
            states = next_states[: self.beam_width]
        return states

    def _typed_variable_unification_join(
        self,
        query: str,
        typed_ir: Mapping[str, Any],
        anchors: Sequence[str],
        answer_type: str,
        dense_retrieve: Callable[[str], Tuple[Sequence[int], Sequence[float]]],
        get_openie_record: Callable[[int], Mapping[str, Any]],
        lookup_entity_relations: Callable[[str, int], Sequence[Mapping[str, Any]]],
        probes: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        if not typed_ir.get("valid", False):
            self._runtime_rejections["invalid_typed_plan_ir"] = 1
            return [], []
        edges = [dict(edge) for edge in typed_ir.get("edges", []) or []]
        if not edges:
            return [], []
        branch_ids = list(typed_ir.get("branch_ids", []) or [])
        initial_bindings = dict(typed_ir.get("initial_bindings", {}) or {})
        branch_states: List[List[Dict[str, Any]]] = []
        if branch_ids:
            for branch_id in branch_ids:
                branch_edges = [
                    edge for edge in edges if edge.get("branch_id") == branch_id
                ]
                branch_root = str(branch_edges[0].get("input_variable", "") or "")
                branch_initial = {
                    branch_root: str(initial_bindings.get(branch_root, "") or "")
                }
                states = self._execute_typed_ir_branch(
                    query,
                    branch_edges,
                    branch_initial,
                    dense_retrieve,
                    get_openie_record,
                    lookup_entity_relations,
                    probes,
                )
                if not states:
                    self._runtime_rejections["typed_ir_incomplete_branch"] = (
                        self._runtime_rejections.get("typed_ir_incomplete_branch", 0) + 1
                    )
                    return [], []
                branch_states.append(states[: self.beam_width])
            combined_states: List[Dict[str, Any]] = []
            for combination in itertools.islice(
                itertools.product(*branch_states),
                self.beam_width,
            ):
                merged_bindings: Dict[str, str] = {}
                lineage: List[Dict[str, Any]] = []
                scores: List[float] = []
                cost = 0.0
                compatible = True
                for state in combination:
                    for variable, value in dict(
                        state.get("variable_bindings", {}) or {}
                    ).items():
                        previous = merged_bindings.get(variable)
                        if previous and not self._exact_endpoint_match(previous, value):
                            compatible = False
                            break
                        merged_bindings[variable] = value
                    if not compatible:
                        break
                    lineage.extend(list(state.get("lineage", []) or []))
                    scores.extend(list(state.get("step_scores", []) or []))
                    cost += float(state.get("cost", 0.0) or 0.0)
                if compatible:
                    combined_states.append({
                        "variable_bindings": merged_bindings,
                        "lineage": lineage,
                        "step_scores": scores,
                        "cost": cost,
                    })
        else:
            root_variable = str(edges[0].get("input_variable", "") or "")
            root_candidates = [
                str(value) for value in anchors if str(value or "").strip()
            ] or [str(initial_bindings.get(root_variable, "") or "")]
            combined_states = []
            for anchor in root_candidates[: self.beam_width]:
                combined_states.extend(self._execute_typed_ir_branch(
                    query,
                    edges,
                    {root_variable: anchor},
                    dense_retrieve,
                    get_openie_record,
                    lookup_entity_relations,
                    probes,
                ))
            combined_states.sort(
                key=lambda item: sum(item["step_scores"])
                / max(len(item["step_scores"]), 1),
                reverse=True,
            )
            combined_states = combined_states[: self.beam_width]

        candidates: List[Dict[str, Any]] = []
        groups: List[Dict[str, Any]] = []
        for state_index, state in enumerate(combined_states):
            lineage = list(state.get("lineage", []) or [])
            passage_indices = list(dict.fromkeys(
                int(item.get("passage_idx", -1))
                for item in lineage
                if int(item.get("passage_idx", -1)) >= 0
            ))
            if not passage_indices:
                continue
            scores = list(state.get("step_scores", []) or [])
            mean_score = sum(scores) / max(len(scores), 1)
            completeness = len(lineage) / max(len(edges), 1)
            cost = float(state.get("cost", 0.0) or 0.0)
            utility = 0.78 * mean_score + 0.22 * completeness - self.cost_weight * cost
            if utility < self.min_candidate_utility:
                continue
            group_id = f"typed_variable_unification_join::{state_index}"
            certificate = {
                "valid": True,
                "plan_id": str(typed_ir.get("plan_id", "") or ""),
                "topology": str(typed_ir.get("topology", "") or ""),
                "edge_count": len(edges),
                "realized_edge_count": len(lineage),
                "complete": len(lineage) == len(edges),
                "directions_preserved": all(
                    str(item.get("direction", "")) in {"forward", "reverse", "either"}
                    for item in lineage
                ),
                "variable_bindings": dict(state.get("variable_bindings", {}) or {}),
                "projection": dict(typed_ir.get("projection", {}) or {}),
                "lineage": lineage,
            }
            groups.append({
                "group_id": group_id,
                "operator": "typed_variable_unification_join",
                "passage_indices": passage_indices,
                "bindings": list(certificate["variable_bindings"].values()),
                "relations": [str(edge.get("relation", "") or "") for edge in edges],
                "answer_type": answer_type,
                "chain_completeness": round(completeness, 6),
                "score": round(mean_score, 6),
                "estimated_cost": round(cost, 6),
                "utility": round(utility, 6),
                "binding_certificate": certificate,
            })
            lineage_by_passage = {
                int(item.get("passage_idx", -1)): item for item in lineage
            }
            for passage_idx in passage_indices:
                profile = dict(lineage_by_passage.get(passage_idx, {}) or {})
                candidates.append({
                    **profile,
                    "passage_idx": passage_idx,
                    "operator": "typed_variable_unification_join",
                    "group_id": group_id,
                    "answer_type": answer_type,
                    "estimated_cost": round(cost, 6),
                    "utility": round(utility, 6),
                    "binding_certificate_valid": True,
                    "typed_plan_ir_version": str(typed_ir.get("version", "") or ""),
                })
        return candidates, groups

    def _dual_branch_comparison_join(
        self,
        query: str,
        anchors: Sequence[str],
        branches: Sequence[Mapping[str, Any]],
        relations: Sequence[str],
        answer_type: str,
        dense_retrieve: Callable[[str], Tuple[Sequence[int], Sequence[float]]],
        get_openie_record: Callable[[int], Mapping[str, Any]],
        lookup_entity_relations: Callable[[str, int], Sequence[Mapping[str, Any]]],
        probes: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        helper = QueryPlanOntologyBindProbeJoin
        plan = {"answer_type": answer_type}
        terminal_profile = helper._terminal_relation_profile(query, plan)
        query_norm = helper._norm(query)
        branch_answer_type = answer_type
        if re.search(r"\b(?:earlier|later|first|last|older|younger|before|after)\b", query_norm):
            branch_answer_type = "date"
        elif re.search(r"\b(?:more|less|higher|lower|many|population)\b", query_norm):
            branch_answer_type = "number"
        terminal_relations = list(terminal_profile.get("aliases", []) or [])
        relation_pool = self._unique(terminal_relations + list(relations), limit=4)
        branch_contracts = [dict(item) for item in branches if isinstance(item, Mapping)]
        if not branch_contracts:
            branch_contracts = [
                {"branch_idx": idx, "entity_key": anchor, "qualifiers": []}
                for idx, anchor in enumerate(anchors[:2])
            ]
        if len(branch_contracts) != 2 or not relation_pool:
            return [], []
        branch_rows: List[Dict[str, Any]] = []
        for branch_idx, branch in enumerate(branch_contracts):
            anchor = str(branch.get("entity_key", "") or "")
            if not anchor:
                return [], []
            best: Dict[str, Any] = {}
            for relation in relation_pool:
                rows = self._access(
                    query,
                    anchor,
                    relation,
                    branch_answer_type,
                    True,
                    dense_retrieve,
                    get_openie_record,
                    lookup_entity_relations,
                    probes,
                )
                for row in rows:
                    passage_idx = int(row.get("passage_idx", -1))
                    record = get_openie_record(passage_idx) or {}
                    qualifier_coverage = self._branch_qualifier_coverage(
                        branch,
                        row,
                        str(record.get("passage", "") or ""),
                    )
                    if self.require_branch_qualifier_evidence and qualifier_coverage < 1.0:
                        self._runtime_rejections["comparison_qualifier_mismatch"] = (
                            self._runtime_rejections.get("comparison_qualifier_mismatch", 0) + 1
                        )
                        continue
                    branch_score = (
                        0.86 * float(row.get("raw_score", 0.0))
                        + 0.14 * qualifier_coverage
                    )
                    if not best or branch_score > float(best.get("branch_score", 0.0)):
                        best = dict(row)
                        best["relation"] = relation
                        best["qualifier_coverage"] = round(float(qualifier_coverage), 6)
                        best["branch_score"] = round(float(branch_score), 6)
                if len(probes) >= self.max_probes:
                    break
            if not best:
                return [], []
            best["branch_idx"] = branch_idx
            best["anchor_entity"] = anchor
            best["branch_contract"] = dict(branch)
            branch_rows.append(best)

        passage_indices = list(dict.fromkeys(int(row["passage_idx"]) for row in branch_rows))
        if len(passage_indices) < 2:
            return [], []
        mean_score = sum(float(row["raw_score"]) for row in branch_rows) / len(branch_rows)
        branch_balance = 1.0 - abs(
            float(branch_rows[0]["raw_score"]) - float(branch_rows[1]["raw_score"])
        )
        qualifier_coverage = min(
            float(row.get("qualifier_coverage", 1.0) or 0.0)
            for row in branch_rows
        )
        estimated_cost = sum(0.20 if "dense_rank" not in row else 1.0 for row in branch_rows)
        utility = (
            0.68 * mean_score
            + 0.18 * max(branch_balance, 0.0)
            + 0.14 * qualifier_coverage
            - self.cost_weight * estimated_cost
        )
        if utility < self.min_candidate_utility:
            return [], []
        group_id = "dual_branch_comparison_join::0"
        group = {
            "group_id": group_id,
            "operator": "dual_branch_comparison_join",
            "passage_indices": passage_indices,
            "anchors": [str(item.get("entity_key", "") or "") for item in branch_contracts],
            "branch_contracts": branch_contracts,
            "relations": [str(row.get("relation", "") or "") for row in branch_rows],
            "branch_rows": branch_rows,
            "answer_type": answer_type,
            "branch_answer_type": branch_answer_type,
            "branch_balance": round(max(branch_balance, 0.0), 6),
            "branch_complete": True,
            "branch_confidence": 1.0,
            "qualifier_coverage": round(float(qualifier_coverage), 6),
            "score": round(mean_score, 6),
            "estimated_cost": round(estimated_cost, 6),
            "utility": round(utility, 6),
        }
        candidates = [
            {
                **row,
                "operator": "dual_branch_comparison_join",
                "group_id": group_id,
                "relations": group["relations"],
                "answer_type": answer_type,
                "estimated_cost": round(estimated_cost, 6),
                "utility": round(utility, 6),
            }
            for row in branch_rows
        ]
        return candidates, [group]

    @staticmethod
    def _deduplicate_candidates(candidates: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
        best_by_passage: Dict[int, Dict[str, Any]] = {}
        for candidate in candidates:
            passage_idx = int(candidate.get("passage_idx", -1))
            if passage_idx < 0:
                continue
            item = dict(candidate)
            previous = best_by_passage.get(passage_idx)
            if previous is None or float(item.get("utility", 0.0)) > float(previous.get("utility", 0.0)):
                best_by_passage[passage_idx] = item
        return sorted(
            best_by_passage.values(),
            key=lambda item: (
                float(item.get("utility", 0.0)),
                float(item.get("raw_score", 0.0)),
            ),
            reverse=True,
        )

    @classmethod
    def _certify_entity_bound_terminal_projection(
        cls,
        *,
        relations: Sequence[Any],
        relation_realization: Mapping[str, Any],
        candidates: Sequence[Mapping[str, Any]],
        groups: Sequence[Mapping[str, Any]],
        endpoint_grounded_source_schema_ids: Sequence[str],
    ) -> Dict[str, Any]:
        """Certify a missing terminal relation from an exact bound passage.

        This is deliberately narrower than relation-realization fallback.  It
        only closes one unresolved *terminal* slot after all preceding slots
        have certified physical predicates, and it requires the query-local
        compiler to provide a strict, provenance-validated chain ending in an
        exact entity-bound passage with contextual terminal evidence.
        """
        result: Dict[str, Any] = {
            "enabled": True,
            "certified": False,
            "reason": "prerequisites_not_met",
            "terminal_relation": "",
            "candidate_doc_ids": [],
            "source_schema_ids": [
                str(value) for value in endpoint_grounded_source_schema_ids
                if str(value or "").strip()
            ],
        }
        normalized_relations = [
            str(value) for value in relations if str(value or "").strip()
        ]
        unresolved = [
            str(value)
            for value in relation_realization.get("unresolved_relations", []) or []
            if str(value or "").strip()
        ]
        if not result["source_schema_ids"]:
            result["reason"] = "no_endpoint_grounded_schema"
            return result
        if not bool(relation_realization.get("certified_only", False)):
            result["reason"] = "relation_catalog_not_certified_only"
            return result
        if not normalized_relations or len(unresolved) != 1:
            result["reason"] = "terminal_projection_requires_one_unresolved_relation"
            return result
        terminal_relation = normalized_relations[-1]
        unresolved_relation = unresolved[0]
        relation_match = max(
            cls._typed_relation_overlap(terminal_relation, unresolved_relation)[0],
            cls._typed_relation_overlap(unresolved_relation, terminal_relation)[0],
        )
        if relation_match < 0.80:
            result["reason"] = "unresolved_relation_is_not_terminal"
            return result

        realized = dict(relation_realization.get("relations", {}) or {})
        for prefix_relation in normalized_relations[:-1]:
            if not list(realized.get(prefix_relation, []) or []):
                result["reason"] = "certified_relation_prefix_incomplete"
                return result

        strict_group_ids = {
            str(group.get("group_id", "") or "")
            for group in groups
            if isinstance(group, Mapping)
            and bool(group.get("strict_chain_certificate", False))
            and bool(group.get("provenance_validated", False))
            and bool(group.get("schema_to_evidence_grounding", False))
            and bool(group.get("exact_anchor", False))
            and float(group.get("chain_completeness", 0.0) or 0.0) >= 1.0
            and int(group.get("contextual_relation_count", 0) or 0) > 0
        }
        candidate_doc_ids = list(dict.fromkeys(
            int(candidate.get("passage_idx", -1))
            for candidate in candidates
            if isinstance(candidate, Mapping)
            and str(candidate.get("group_id", "") or "") in strict_group_ids
            and int(candidate.get("passage_idx", -1)) >= 0
            and bool(candidate.get("exact_entity_key", False))
            and bool(candidate.get("provenance_validated", False))
            and bool(candidate.get("candidate_window_path", False))
            and int(candidate.get("contextual_relation_count", 0) or 0) > 0
        ))
        if not candidate_doc_ids:
            result["reason"] = "no_strict_entity_bound_terminal_passage"
            return result

        result.update({
            "certified": True,
            "reason": "strict_entity_bound_terminal_projection",
            "terminal_relation": terminal_relation,
            "unresolved_relation": unresolved_relation,
            "candidate_doc_ids": candidate_doc_ids,
            "strict_group_ids": sorted(strict_group_ids),
            "operator": "entity_bound_passage_projection",
        })
        return result

    def run(
        self,
        query: str,
        dataset: str,
        base_doc_ids: Sequence[int],
        query_plan_log: Mapping[str, Any],
        dense_retrieve: Callable[[str], Tuple[Sequence[int], Sequence[float]]],
        get_openie_record: Callable[[int], Mapping[str, Any]],
        lookup_entity_relations: Callable[[str, int], Sequence[Mapping[str, Any]]],
        runtime_statistics: Mapping[str, Any] = None,
    ) -> FactorizedQueryExecutorResult:
        self._runtime_rejections = {}
        self._active_relation_realizations = {}
        self._active_relation_directions = {}
        self._active_relation_predicate_directions = {}
        self._strict_relation_predicates = set()
        compile_started_at = time.perf_counter()
        plan = QueryPlanOntologyBindProbeJoin._plan_features(query_plan_log)
        answer_type = str(plan.get("answer_type", "") or "")
        typed_contract: Dict[str, Any] = {"enabled": False, "admissible": True}
        if self.enable_typed_semantic_compiler:
            typed_contract = self._compile_typed_contract(
                query=query,
                query_plan_log=query_plan_log,
                plan=plan,
            )
            anchors = list(typed_contract.get("anchors", []) or [])
            relations = list(typed_contract.get("relations", []) or [])
            ordered_relations = list(relations)
            answer_type = str(typed_contract.get("answer_type", "") or answer_type)
        else:
            anchors = list(plan.get("anchors", []) or [])
            ordered_relations = self._ordered_relations(query_plan_log)
            relations = list(ordered_relations)
            if not relations:
                relations = list(plan.get("fallback_relations", []) or [])
        typed_plan_ir = dict(typed_contract.get("logical_plan_ir", {}) or {})
        typed_ir_relations = self._semantic_unique_relations(
            [
                str(edge.get("relation", "") or "")
                for edge in typed_plan_ir.get("edges", []) or []
                if isinstance(edge, Mapping)
            ],
            limit=self.max_hops,
        )
        if (
            self.require_typed_variable_unification
            and typed_plan_ir.get("valid", False)
            and typed_ir_relations
        ):
            relations = list(typed_ir_relations)
        terminal_profile = (
            dict(typed_contract.get("terminal_relation_profile", {}) or {})
            if self.enable_typed_semantic_compiler
            else QueryPlanOntologyBindProbeJoin._terminal_relation_profile(query, plan)
        )
        terminal_relation = str(terminal_profile.get("canonical_relation", "") or "")
        derived_terminal_hop = False
        if (
            not self.enable_typed_semantic_compiler
            and terminal_profile.get("resolved")
            and terminal_relation
        ):
            terminal_already_present = any(
                QueryPlanOntologyBindProbeJoin._relation_overlap(
                    terminal_relation,
                    relation,
                )[0] >= 0.60
                for relation in relations
            )
            if not terminal_already_present and relations:
                prefix_relations = list(relations if ordered_relations else relations[:1])
                relations = self._unique(prefix_relations + [terminal_relation], limit=4)
                derived_terminal_hop = len(relations) >= 2
        comparison = self._is_comparison(query, anchors)
        comparison_branch_contract = dict(
            typed_contract.get("comparison_branches", {}) or {}
        )
        if (
            (self.enable_branch_complete_compiler or self.enable_executable_ontology_compiler)
            and comparison
            and not comparison_branch_contract.get("enabled")
        ):
            comparison_branch_contract = self._compile_comparison_branches(query, anchors)
        comparison_branches = list(comparison_branch_contract.get("branches", []) or [])
        if (self.enable_branch_complete_compiler or self.enable_executable_ontology_compiler) and comparison:
            comparison = bool(comparison_branch_contract.get("complete", False))

        candidate_window_relations = list(relations)
        candidate_window_relation_source = "compiled_query_contract"
        ontology_relation_alternatives = list(
            typed_contract.get("ontology_relation_alternatives", []) or []
        )
        if self.enable_candidate_window_path_compiler:
            best_ontology_path = dict(typed_contract.get("best_ontology_path", {}) or {})
            best_ontology_relations = list(best_ontology_path.get("relations", []) or [])
            if not candidate_window_relations and best_ontology_relations:
                candidate_window_relations = self._semantic_unique_relations(
                    best_ontology_relations,
                    limit=self.max_hops,
                )
                candidate_window_relation_source = "certified_ontology_recovery"

        relation_realization = self._compile_relation_realizations(
            query_plan_log=query_plan_log,
            relations=(
                candidate_window_relations
                if (
                    self.enable_candidate_window_path_compiler
                    and not self.require_typed_variable_unification
                )
                else (typed_ir_relations or relations)
            ),
            selected_schema=typed_contract.get("selected_schema", {}),
        )
        if self.enable_executable_ontology_compiler:
            typed_contract["relation_realization"] = relation_realization
            if (
                relation_realization.get("certified_only", False)
                and relation_realization.get("unresolved_relations")
            ):
                typed_contract["admissible"] = False
                typed_contract["rejection_reason"] = "uncertified_physical_relation"
                self._runtime_rejections["uncertified_physical_relation"] = len(
                    relation_realization.get("unresolved_relations", []) or []
                )

        runtime_schema_plans: List[Mapping[str, Any]] = []
        runtime_schema_plans.extend(
            plan
            for plan in (
                dict(query_plan_log.get("query_condition", {}) or {}).get(
                    "compiler_schema_plans", []
                )
                or []
            )
            if isinstance(plan, Mapping)
        )
        runtime_schema_plans.extend(
            plan
            for plan in query_plan_log.get("matched_plans", []) or []
            if isinstance(plan, Mapping)
        )
        runtime_schema_by_id = {
            str(plan.get("id", "") or ""): plan
            for plan in runtime_schema_plans
            if plan.get("id")
        }
        selected_schema_id = str(
            dict(typed_contract.get("selected_schema", {}) or {}).get(
                "plan_id", ""
            )
            or ""
        )
        if selected_schema_id and selected_schema_id in runtime_schema_by_id:
            compiled_source_schema_ids = [selected_schema_id]
        else:
            compiled_source_schema_ids = list(dict.fromkeys(
                str(plan.get("id", "") or "")
                for plan in query_plan_log.get("matched_plans", []) or []
                if isinstance(plan, Mapping) and plan.get("id")
            ))




        endpoint_grounded_source_schema_ids = [
            selected_schema_id
            for _ in [0]
            if selected_schema_id
            and selected_schema_id in runtime_schema_by_id
            and dict(
                runtime_schema_by_id[selected_schema_id].get(
                    "endpoint_grounded_route_contract", {}
                )
                or {}
            ).get("enabled")
            is True
            and runtime_schema_by_id[selected_schema_id].get(
                "endpoint_grounded_route_contract_matched", False
            )
            is True
        ]

        logical_evidence_plan: Dict[str, Any] = {
            "enabled": self.enable_evidence_query_compiler,
            "admissible": True,
        }
        compiled_plan = None
        if self.enable_evidence_query_compiler:
            compiled_plan = OntologyEvidenceQueryCompiler().compile(
                query=query,
                dataset=dataset,
                anchors=anchors,
                relations=relations,
                answer_type=answer_type,
                comparison=comparison,
                typed_contract=typed_contract,
                source_schema_ids=compiled_source_schema_ids,
            )
            logical_evidence_plan = compiled_plan.to_dict()
        logical_compile_latency_ms = (
            time.perf_counter() - compile_started_at
        ) * 1000.0

        operator = ""
        execution_access_strategy = self.physical_access_strategy
        execution_max_index_rows = self.max_index_rows
        execution_dense_top_k = self.dense_top_k
        execution_beam_width = self.beam_width
        physical_plan_optimizer: Dict[str, Any] = {
            "enabled": self.enable_cost_based_physical_planner,
            "decision_reason": "disabled",
            "alternatives": [],
        }
        if self.enable_cost_based_physical_planner:
            optimizer_started_at = time.perf_counter()
            if compiled_plan is None:
                physical_plan_optimizer["decision_reason"] = "logical_compiler_disabled"
            elif not self.evidence_catalog_path:
                physical_plan_optimizer["decision_reason"] = "catalog_path_missing"
            else:
                catalog = EvidenceCatalog.from_path(self.evidence_catalog_path)
                optimizer = CostBasedEvidenceQueryOptimizer(
                    catalog=catalog,
                    quality_weight=self.physical_planner_quality_weight,
                    recall_weight=self.physical_planner_recall_weight,
                    proof_weight=self.physical_planner_proof_weight,
                    cost_weight=self.physical_planner_cost_weight,
                    risk_weight=self.physical_planner_risk_weight,
                    min_proof_coverage=self.physical_planner_min_proof_coverage,
                    max_estimated_cost=self.physical_planner_max_estimated_cost,
                    no_op_utility=self.physical_planner_no_op_utility,
                    selection_mode=self.physical_planner_selection_mode,
                    max_loss_probability=self.physical_planner_max_loss_probability,
                    confidence_delta=self.physical_planner_confidence_delta,
                    min_catalog_samples=self.physical_planner_min_catalog_samples,
                )
                physical_plan_optimizer = optimizer.optimize(
                    logical_plan=compiled_plan,
                    enabled_operators={
                        "entity_relation_index_scan": self.enable_entity_relation_index_scan,
                        "cascaded_bind_join": self.enable_cascaded_bind_join,
                        "dual_branch_comparison_join": self.enable_dual_branch_comparison_join,
                    },
                    runtime_statistics=runtime_statistics,
                    max_index_rows=self.max_index_rows,
                    dense_top_k=self.dense_top_k,
                    beam_width=self.beam_width,
                )
                selected_physical_plan = dict(
                    physical_plan_optimizer.get("selected", {}) or {}
                )
                operator = str(selected_physical_plan.get("operator", "") or "")
                if operator:
                    execution_access_strategy = str(
                        selected_physical_plan.get("access_strategy", "adaptive")
                        or "adaptive"
                    )
                    execution_max_index_rows = max(
                        int(selected_physical_plan.get("max_index_rows", self.max_index_rows)),
                        1,
                    )
                    execution_dense_top_k = max(
                        int(selected_physical_plan.get("dense_top_k", self.dense_top_k)),
                        1,
                    )
                    execution_beam_width = max(
                        int(selected_physical_plan.get("beam_width", self.beam_width)),
                        1,
                    )
            physical_plan_optimizer["optimization_latency_ms"] = round(
                (time.perf_counter() - optimizer_started_at) * 1000.0,
                6,
            )
        else:
            if comparison and self.enable_dual_branch_comparison_join:
                operator = "dual_branch_comparison_join"
            elif dataset in {"nq_rear", "popqa"} and self.enable_entity_relation_index_scan:
                operator = "entity_relation_index_scan"
            elif (
                len(relations) >= 2
                and (len(ordered_relations) >= 2 or dataset == "musique" or derived_terminal_hop)
                and self.enable_cascaded_bind_join
            ):
                operator = "cascaded_bind_join"
            elif self.enable_entity_relation_index_scan:
                operator = "entity_relation_index_scan"

        if (
            self.require_typed_variable_unification
            and typed_plan_ir.get("valid", False)
            and self.enable_cascaded_bind_join
        ):
            operator = "typed_variable_unification_join"

        log: Dict[str, Any] = {
            "enabled": True,
            "dataset": dataset,
            "selected_operator": operator,
            "anchors": anchors,
            "relations": relations,
            "ordered_relation_count": len(ordered_relations),
            "terminal_relation_profile": terminal_profile,
            "derived_terminal_hop": derived_terminal_hop,
            "answer_type": answer_type,
            "comparison_query": comparison,
            "executable_ontology_compiler_enabled": self.enable_executable_ontology_compiler,
            "whole_warehouse_probe_enabled": self.enable_whole_warehouse_probe,
            "branch_complete_compiler_enabled": self.enable_branch_complete_compiler,
            "require_branch_qualifier_evidence": self.require_branch_qualifier_evidence,
            "logical_evidence_plan": logical_evidence_plan,
            "schema_execution_provenance": {
                "schema_version": "endpoint_schema_execution_provenance_v1",
                "source_schema_ids": list(compiled_source_schema_ids),
                "selected_source_schema_id": selected_schema_id,
                "endpoint_grounded_source_schema_ids": list(
                    endpoint_grounded_source_schema_ids
                ),
                "logical_plan_id": str(
                    logical_evidence_plan.get("plan_id", "") or ""
                ),
                "relation_sequence": list(relations),
                "route_contract_matched": bool(
                    endpoint_grounded_source_schema_ids
                ),
                "logical_plan_compiled": bool(
                    logical_evidence_plan.get("plan_id")
                ),
                "logical_plan_admissible": bool(
                    logical_evidence_plan.get("admissible", False)
                ),
                "relation_realization_resolved": bool(
                    relations
                    and not (
                        relation_realization.get("unresolved_relations", []) or []
                    )
                ),
                "unresolved_relations": list(
                    relation_realization.get("unresolved_relations", []) or []
                ),
                "physical_operator_selected": bool(operator),
                "selected_operator": operator,
                "physical_execution_completed": False,
                "candidate_count": 0,
                "external_candidate_count": 0,
                "candidate_doc_ids": [],
                "external_candidate_doc_ids": [],
            },
            "logical_compile_latency_ms": round(logical_compile_latency_ms, 6),
            "physical_plan_optimizer": physical_plan_optimizer,
            "physical_access_strategy": execution_access_strategy,
            "comparison_branch_contract": comparison_branch_contract,
            "typed_semantic_compiler_enabled": self.enable_typed_semantic_compiler,
            "execution_aware_contract_enabled": self.enable_execution_aware_contract,
            "lossless_typed_plan_ir_enabled": self.enable_lossless_typed_plan_ir,
            "typed_variable_unification_required": self.require_typed_variable_unification,
            "certificate_aware_plan_portfolio_enabled": (
                self.enable_certificate_aware_plan_portfolio
            ),
            "certificate_aware_access_path_reoptimization_enabled": (
                self.enable_certificate_aware_access_path_reoptimization
            ),
            "logical_plan_ir": typed_plan_ir,
            "typed_contract": typed_contract,
            "relation_realization": relation_realization,
            "relation_keyed_catalog_enabled": self.enable_relation_keyed_catalog,
            "query_relation_alternative_recovery_enabled": self.enable_query_relation_alternative_recovery,
            "max_query_relation_alternatives": self.max_query_relation_alternatives,
            "ontology_relation_alternative_count": len(ontology_relation_alternatives),
            "provenance_plan_portfolio_enabled": self.enable_provenance_plan_portfolio,
            "candidate_window_path_compiler_enabled": self.enable_candidate_window_path_compiler,
            "candidate_window_path_entity_index_expansion_enabled": self.candidate_window_path_enable_entity_index_expansion,
            "candidate_window_path_bound_dense_expansion_enabled": self.candidate_window_path_enable_bound_dense_expansion,
            "candidate_window_path_relation_agnostic_bridge_enabled": self.candidate_window_path_enable_relation_agnostic_bridge,
            "candidate_window_relation_source": candidate_window_relation_source,
            "inverse_direction_fallback_enabled": self.enable_inverse_direction_fallback,
            "inverse_direction_penalty": self.inverse_direction_penalty,
            "require_exact_entity_keys": self.require_exact_entity_keys,
            "require_typed_join_keys": self.require_typed_join_keys,
            "entity_relation_index_scan_enabled": self.enable_entity_relation_index_scan,
            "cascaded_bind_join_enabled": self.enable_cascaded_bind_join,
            "dual_branch_comparison_join_enabled": self.enable_dual_branch_comparison_join,
            "probes": [],
            "candidates": [],
            "groups": [],
            "candidate_count": 0,
            "group_count": 0,
            "skip_reason": "",
        }
        if not query_plan_log or not query_plan_log.get("enabled", False):
            log["skip_reason"] = "no_query_plan"
            return FactorizedQueryExecutorResult(log=log)

        candidate_window_result = FactorizedQueryExecutorResult()
        if (
            self.enable_candidate_window_path_compiler
            and not self.require_typed_variable_unification
        ):
            compiler = QueryLocalEvidencePathCompiler(
                window_size=self.candidate_window_path_size,
                beam_width=self.candidate_window_path_beam_width,
                max_paths=self.candidate_window_path_max_paths,
                max_hops=self.max_hops,
                min_anchor_score=self.candidate_window_path_min_anchor_score,
                min_relation_score=self.candidate_window_path_min_relation_score,
                min_path_score=self.candidate_window_path_min_path_score,
                allow_contextual_terminal_edges=(
                    self.candidate_window_path_allow_contextual_terminal_edges
                ),
                enable_entity_index_expansion=(
                    self.candidate_window_path_enable_entity_index_expansion
                ),
                enable_bound_dense_expansion=(
                    self.candidate_window_path_enable_bound_dense_expansion
                ),
                entity_index_top_k=self.candidate_window_path_entity_index_top_k,
                bound_dense_top_k=self.candidate_window_path_bound_dense_top_k,
                max_access_probes=self.candidate_window_path_max_access_probes,
                max_external_passages=self.candidate_window_path_max_external_passages,
                enable_relation_agnostic_bridge=(
                    self.candidate_window_path_enable_relation_agnostic_bridge
                ),
                bridge_max_paths=self.candidate_window_path_bridge_max_paths,
                bridge_max_edges_per_state=(
                    self.candidate_window_path_bridge_max_edges_per_state
                ),
                bridge_min_path_score=(
                    self.candidate_window_path_bridge_min_path_score
                ),
            )
            local_result = compiler.run(
                query=query,
                base_doc_ids=base_doc_ids,
                anchors=anchors,
                relations=candidate_window_relations,
                answer_type=answer_type,
                comparison=bool(
                    comparison
                    or typed_contract.get("comparison_query", False)
                ),
                get_openie_record=get_openie_record,
                relation_realizations=self._active_relation_realizations,
                relation_directions={
                    relation: {
                        "default": self._active_relation_directions.get(relation, "either"),
                        "predicates": self._active_relation_predicate_directions.get(relation, {}),
                    }
                    for relation in candidate_window_relations
                },
                terminal_type_score=(
                    QueryPlanOntologyBindProbeJoin._terminal_type_score
                ),
                lookup_entity_relations=lookup_entity_relations,
                dense_retrieve=dense_retrieve,
            )
            candidate_window_result = FactorizedQueryExecutorResult(
                candidates=list(local_result.candidates),
                groups=list(local_result.groups),
                log=dict(local_result.log),
            )
            log["candidate_window_path_compiler"] = dict(local_result.log)

            alternative_attempts: List[Dict[str, Any]] = []
            primary_relation_key = tuple(
                self._entity_key(relation) for relation in candidate_window_relations
            )
            primary_active_realizations = dict(self._active_relation_realizations)
            primary_active_directions = dict(self._active_relation_directions)
            primary_active_predicate_directions = {
                relation: dict(directions)
                for relation, directions in self._active_relation_predicate_directions.items()
            }
            primary_strict_predicates = set(self._strict_relation_predicates)
            attempted_relation_keys = {primary_relation_key}
            attempted_alternatives = 0
            try:
                if self.enable_query_relation_alternative_recovery:
                    for alternative in ontology_relation_alternatives:
                        alternative_relations = self._semantic_unique_relations(
                            alternative.get("relations", []) or [],
                            limit=self.max_hops,
                        )
                        relation_key = tuple(
                            self._entity_key(relation)
                            for relation in alternative_relations
                        )
                        if (
                            not relation_key
                            or relation_key in attempted_relation_keys
                            or attempted_alternatives >= self.max_query_relation_alternatives
                        ):
                            continue
                        attempted_relation_keys.add(relation_key)
                        attempted_alternatives += 1
                        alternative_realization = self._compile_relation_realizations(
                            query_plan_log=query_plan_log,
                            relations=alternative_relations,
                            selected_schema=alternative,
                        )
                        attempt_log: Dict[str, Any] = {
                            "rank": attempted_alternatives,
                            "plan_id": str(alternative.get("plan_id", "") or ""),
                            "source": str(alternative.get("source", "") or ""),
                            "relations": list(alternative_relations),
                            "query_predicate_coverage": float(
                                alternative.get("query_predicate_coverage", 0.0) or 0.0
                            ),
                            "answer_type_compatibility": float(
                                alternative.get("answer_type_compatibility", 0.0) or 0.0
                            ),
                            "precision": float(alternative.get("precision", 0.0) or 0.0),
                            "relation_realization": alternative_realization,
                            "candidate_count": 0,
                            "group_count": 0,
                            "skip_reason": "",
                        }
                        if (
                            alternative_realization.get("certified_only", False)
                            and alternative_realization.get("unresolved_relations")
                        ):
                            attempt_log["skip_reason"] = "uncertified_alternative_relation"
                            alternative_attempts.append(attempt_log)
                            continue

                        alternative_result = compiler.run(
                            query=query,
                            base_doc_ids=base_doc_ids,
                            anchors=anchors,
                            relations=alternative_relations,
                            answer_type=answer_type,
                            comparison=bool(
                                comparison
                                or typed_contract.get("comparison_query", False)
                            ),
                            get_openie_record=get_openie_record,
                            relation_realizations=self._active_relation_realizations,
                            relation_directions={
                                relation: {
                                    "default": self._active_relation_directions.get(
                                        relation, "either"
                                    ),
                                    "predicates": self._active_relation_predicate_directions.get(
                                        relation, {}
                                    ),
                                }
                                for relation in alternative_relations
                            },
                            terminal_type_score=(
                                QueryPlanOntologyBindProbeJoin._terminal_type_score
                            ),
                            lookup_entity_relations=lookup_entity_relations,
                            dense_retrieve=dense_retrieve,
                        )
                        group_id_map: Dict[str, str] = {}
                        namespaced_groups: List[Dict[str, Any]] = []
                        for group in alternative_result.groups:
                            item = dict(group)
                            old_group_id = str(item.get("group_id", "") or "")
                            new_group_id = (
                                f"relation_alternative_{attempted_alternatives}::{old_group_id}"
                            )
                            group_id_map[old_group_id] = new_group_id
                            item["group_id"] = new_group_id
                            item["relation_alternative_rank"] = attempted_alternatives
                            item["relation_alternative_plan_id"] = attempt_log["plan_id"]
                            namespaced_groups.append(item)
                        namespaced_candidates: List[Dict[str, Any]] = []
                        for candidate in alternative_result.candidates:
                            item = dict(candidate)
                            old_group_id = str(item.get("group_id", "") or "")
                            if old_group_id in group_id_map:
                                item["group_id"] = group_id_map[old_group_id]
                            item["relation_alternative_rank"] = attempted_alternatives
                            item["relation_alternative_plan_id"] = attempt_log["plan_id"]
                            item["relation_alternative_recovery"] = True
                            namespaced_candidates.append(item)
                        candidate_window_result.candidates.extend(namespaced_candidates)
                        candidate_window_result.groups.extend(namespaced_groups)
                        attempt_log["candidate_count"] = len(namespaced_candidates)
                        attempt_log["group_count"] = len(namespaced_groups)
                        attempt_log["skip_reason"] = str(
                            alternative_result.log.get("skip_reason", "") or ""
                        )
                        attempt_log["compiler"] = dict(alternative_result.log)
                        alternative_attempts.append(attempt_log)
            finally:
                self._active_relation_realizations = primary_active_realizations
                self._active_relation_directions = primary_active_directions
                self._active_relation_predicate_directions = (
                    primary_active_predicate_directions
                )
                self._strict_relation_predicates = primary_strict_predicates

            successful_alternatives = sum(
                1 for attempt in alternative_attempts
                if int(attempt.get("candidate_count", 0) or 0) > 0
            )
            log["candidate_window_relation_alternatives"] = {
                "enabled": self.enable_query_relation_alternative_recovery,
                "available_count": len(ontology_relation_alternatives),
                "attempted_count": len(alternative_attempts),
                "successful_count": successful_alternatives,
                "candidate_count": sum(
                    int(attempt.get("candidate_count", 0) or 0)
                    for attempt in alternative_attempts
                ),
                "group_count": sum(
                    int(attempt.get("group_count", 0) or 0)
                    for attempt in alternative_attempts
                ),
                "attempts": alternative_attempts,
            }
            if successful_alternatives:
                log["candidate_window_relation_source"] = (
                    "compiled_query_contract_plus_ontology_alternatives"
                )

        terminal_projection = self._certify_entity_bound_terminal_projection(
            relations=candidate_window_relations,
            relation_realization=relation_realization,
            candidates=candidate_window_result.candidates,
            groups=candidate_window_result.groups,
            endpoint_grounded_source_schema_ids=(
                endpoint_grounded_source_schema_ids
            ),
        )
        log["terminal_projection_certification"] = terminal_projection
        if terminal_projection.get("certified", False):
            relation_realization = dict(relation_realization)
            relation_realization["unresolved_relations"] = []
            relation_realization["terminal_projection"] = dict(
                terminal_projection
            )
            typed_contract["relation_realization"] = relation_realization
            if typed_contract.get("rejection_reason") == "uncertified_physical_relation":
                typed_contract["admissible"] = True
                typed_contract["rejection_reason"] = ""
            if logical_evidence_plan.get("rejection_reason") == "uncertified_physical_relation":
                logical_evidence_plan["admissible"] = True
                logical_evidence_plan["rejection_reason"] = ""
            self._runtime_rejections.pop("uncertified_physical_relation", None)
            self._runtime_rejections["terminal_projection_recovered"] = len(
                terminal_projection.get("candidate_doc_ids", []) or []
            )
            log["typed_contract"] = typed_contract
            log["logical_evidence_plan"] = logical_evidence_plan
            log["relation_realization"] = relation_realization
            provenance = dict(log.get("schema_execution_provenance", {}) or {})
            provenance.update({
                "logical_plan_admissible": True,
                "relation_realization_resolved": True,
                "unresolved_relations": [],
                "terminal_projection_certified": True,
                "terminal_projection_operator": str(
                    terminal_projection.get("operator", "") or ""
                ),
                "terminal_projection_candidate_doc_ids": list(
                    terminal_projection.get("candidate_doc_ids", []) or []
                ),
            })
            log["schema_execution_provenance"] = provenance
        candidate_window_succeeded = bool(candidate_window_result.candidates)

        if (
            self.enable_evidence_query_compiler
            and not logical_evidence_plan.get("admissible", False)
            and not candidate_window_succeeded
        ):
            log["skip_reason"] = str(
                logical_evidence_plan.get("rejection_reason", "logical_plan_rejected")
                or "logical_plan_rejected"
            )
            return FactorizedQueryExecutorResult(log=log)
        if (
            self.enable_typed_semantic_compiler
            and not typed_contract.get("admissible", False)
            and not candidate_window_succeeded
        ):
            log["skip_reason"] = str(
                typed_contract.get("rejection_reason", "typed_contract_rejected")
                or "typed_contract_rejected"
            )
            return FactorizedQueryExecutorResult(log=log)
        if not anchors:
            log["skip_reason"] = "no_anchor_entities"
            return FactorizedQueryExecutorResult(log=log)
        if not relations and not candidate_window_succeeded:
            log["skip_reason"] = "no_relational_access_path"
            return FactorizedQueryExecutorResult(log=log)
        if not operator and not candidate_window_succeeded:
            log["skip_reason"] = (
                "cost_based_noop"
                if self.enable_cost_based_physical_planner
                else "physical_operator_disabled"
            )
            return FactorizedQueryExecutorResult(log=log)

        probes: List[Dict[str, Any]] = []
        original_physical_settings = (
            self.physical_access_strategy,
            self.max_index_rows,
            self.dense_top_k,
            self.beam_width,
        )
        self.physical_access_strategy = execution_access_strategy
        self.max_index_rows = execution_max_index_rows
        self.dense_top_k = execution_dense_top_k
        self.beam_width = execution_beam_width
        execution_started_at = time.perf_counter()
        try:
            if candidate_window_succeeded:
                operator = "candidate_window_provenance_join"
                log["selected_operator"] = operator
                candidates = list(candidate_window_result.candidates)
                groups = list(candidate_window_result.groups)
            elif operator == "dual_branch_comparison_join":
                candidates, groups = self._dual_branch_comparison_join(
                    query,
                    anchors,
                    comparison_branches,
                    relations,
                    answer_type,
                    dense_retrieve,
                    get_openie_record,
                    lookup_entity_relations,
                    probes,
                )
            elif operator == "typed_variable_unification_join":
                candidates, groups = self._typed_variable_unification_join(
                    query,
                    typed_plan_ir,
                    anchors,
                    answer_type,
                    dense_retrieve,
                    get_openie_record,
                    lookup_entity_relations,
                    probes,
                )
            elif operator == "cascaded_bind_join":
                candidates, groups = self._cascaded_bind_join(
                    query,
                    anchors,
                    relations,
                    answer_type,
                    dense_retrieve,
                    get_openie_record,
                    lookup_entity_relations,
                    probes,
                )
            else:
                candidates, groups = self._entity_relation_index_scan(
                    query,
                    anchors,
                    relations,
                    answer_type,
                    dense_retrieve,
                    get_openie_record,
                    lookup_entity_relations,
                    probes,
                )
        finally:
            (
                self.physical_access_strategy,
                self.max_index_rows,
                self.dense_top_k,
                self.beam_width,
            ) = original_physical_settings

        certificate_portfolio_log: Dict[str, Any] = {
            "enabled": self.enable_certificate_aware_plan_portfolio,
            "triggered": False,
            "primary_operator": operator,
            "primary_plan_id": str(typed_plan_ir.get("plan_id", "") or ""),
            "primary_candidate_count": len(candidates),
            "primary_certificate_complete": any(
                bool(dict(group.get("binding_certificate", {}) or {}).get("complete"))
                for group in groups
                if isinstance(group, Mapping)
            ),
            "alternative_typed_attempts": [],
            "access_path_reoptimization": {
                "enabled": self.enable_certificate_aware_access_path_reoptimization,
                "attempted": False,
                "source_access_path": self.physical_access_strategy,
                "target_access_path": "dense_first",
                "candidate_count": 0,
                "group_count": 0,
                "certificate_complete": False,
                "probe_count": 0,
                "decision_reason": "disabled",
            },
            "fallback_operator": "",
            "selected_operator": operator,
            "selected_plan_id": str(typed_plan_ir.get("plan_id", "") or ""),
            "executed_plan_count": 1,
            "decision_reason": "disabled",
        }
        if (
            self.enable_certificate_aware_plan_portfolio
            and operator == "typed_variable_unification_join"
        ):
            certificate_portfolio_log["triggered"] = True
            primary_complete = bool(
                certificate_portfolio_log["primary_certificate_complete"]
            )
            if (
                not primary_complete
                and self.enable_certificate_aware_access_path_reoptimization
                and self.physical_access_strategy != "dense_first"
            ):
                reoptimization_log = certificate_portfolio_log[
                    "access_path_reoptimization"
                ]
                reoptimization_log.update({
                    "attempted": True,
                    "decision_reason": "executed",
                })
                reoptimized_probes: List[Dict[str, Any]] = []
                original_access_strategy = self.physical_access_strategy
                try:
                    self.physical_access_strategy = "dense_first"
                    reoptimized_candidates, reoptimized_groups = (
                        self._typed_variable_unification_join(
                            query,
                            typed_plan_ir,
                            anchors,
                            answer_type,
                            dense_retrieve,
                            get_openie_record,
                            lookup_entity_relations,
                            reoptimized_probes,
                        )
                    )
                finally:
                    self.physical_access_strategy = original_access_strategy
                probes.extend(reoptimized_probes)
                reoptimized_complete = any(
                    bool(certificate.get("complete"))
                    and bool(certificate.get("lineage"))
                    and all(
                        bool(item.get("exact_entity_key", False))
                        for item in certificate.get("lineage", []) or []
                        if isinstance(item, Mapping)
                    )
                    for group in reoptimized_groups
                    if isinstance(group, Mapping)
                    for certificate in [
                        dict(group.get("binding_certificate", {}) or {})
                    ]
                )
                reoptimization_log.update({
                    "candidate_count": len(reoptimized_candidates),
                    "group_count": len(reoptimized_groups),
                    "certificate_complete": reoptimized_complete,
                    "probe_count": len(reoptimized_probes),
                    "decision_reason": (
                        "complete_typed_certificate"
                        if reoptimized_complete
                        else "no_complete_exact_typed_certificate"
                    ),
                })
                certificate_portfolio_log["executed_plan_count"] += 1
                if reoptimized_complete:
                    candidates = reoptimized_candidates
                    groups = reoptimized_groups
                    primary_complete = True
                    certificate_portfolio_log.update({
                        "selected_operator": "typed_variable_unification_join",
                        "selected_plan_id": str(
                            typed_plan_ir.get("plan_id", "") or ""
                        ),
                        "decision_reason": (
                            "primary_dense_reoptimization_certificate_complete"
                        ),
                    })
            if primary_complete:
                if certificate_portfolio_log["decision_reason"] == "disabled":
                    certificate_portfolio_log["decision_reason"] = (
                        "primary_typed_certificate_complete"
                    )
            else:
                primary_ir_key = (
                    str(typed_plan_ir.get("plan_id", "") or ""),
                    tuple(
                        (
                            str(edge.get("input_variable", "") or ""),
                            self._entity_key(edge.get("relation", "")),
                            str(edge.get("direction", "either") or "either"),
                            str(edge.get("output_variable", "") or ""),
                            str(edge.get("branch_id", "main") or "main"),
                        )
                        for edge in typed_plan_ir.get("edges", []) or []
                        if isinstance(edge, Mapping)
                    ),
                )
                saved_realizations = dict(self._active_relation_realizations)
                saved_directions = dict(self._active_relation_directions)
                saved_predicate_directions = {
                    relation: dict(values)
                    for relation, values in self._active_relation_predicate_directions.items()
                }
                saved_strict_predicates = set(self._strict_relation_predicates)
                selected_alternative_ir: Dict[str, Any] = {}
                try:
                    alternative_count = 0
                    for raw_alternative in typed_contract.get(
                        "typed_ir_candidates", []
                    ) or []:
                        if alternative_count >= self.max_query_relation_alternatives:
                            break
                        if not isinstance(raw_alternative, Mapping):
                            continue
                        alternative_ir = dict(
                            raw_alternative.get("typed_plan_ir", {}) or {}
                        )
                        if not alternative_ir.get("valid", False):
                            continue
                        alternative_key = (
                            str(alternative_ir.get("plan_id", "") or ""),
                            tuple(
                                (
                                    str(edge.get("input_variable", "") or ""),
                                    self._entity_key(edge.get("relation", "")),
                                    str(edge.get("direction", "either") or "either"),
                                    str(edge.get("output_variable", "") or ""),
                                    str(edge.get("branch_id", "main") or "main"),
                                )
                                for edge in alternative_ir.get("edges", []) or []
                                if isinstance(edge, Mapping)
                            ),
                        )
                        if alternative_key == primary_ir_key:
                            continue
                        semantic_score = float(
                            dict(raw_alternative.get("semantic_score", {}) or {}).get(
                                "score", 0.0
                            )
                            or 0.0
                        )
                        if semantic_score < 0.65:
                            continue
                        alternative_count += 1
                        alternative_schema = {
                            "plan_id": str(raw_alternative.get("plan_id", "") or ""),
                            "relations": list(raw_alternative.get("relations", []) or []),
                            "typed_plan_ir": alternative_ir,
                        }
                        realization = self._compile_relation_realizations(
                            query_plan_log=query_plan_log,
                            relations=list(raw_alternative.get("relations", []) or []),
                            selected_schema=alternative_schema,
                        )
                        alternative_probes: List[Dict[str, Any]] = []
                        alternative_candidates, alternative_groups = (
                            self._typed_variable_unification_join(
                                query,
                                alternative_ir,
                                anchors,
                                answer_type,
                                dense_retrieve,
                                get_openie_record,
                                lookup_entity_relations,
                                alternative_probes,
                            )
                        )
                        probes.extend(alternative_probes)
                        complete = any(
                            bool(
                                dict(group.get("binding_certificate", {}) or {}).get(
                                    "complete"
                                )
                            )
                            for group in alternative_groups
                            if isinstance(group, Mapping)
                        )
                        certificate_portfolio_log[
                            "alternative_typed_attempts"
                        ].append({
                            "rank": alternative_count,
                            "plan_id": str(
                                alternative_ir.get("plan_id", "") or ""
                            ),
                            "semantic_score": round(semantic_score, 6),
                            "candidate_count": len(alternative_candidates),
                            "group_count": len(alternative_groups),
                            "certificate_complete": complete,
                            "relation_realization_available": bool(
                                realization.get("available", False)
                            ),
                            "unresolved_relations": list(
                                realization.get("unresolved_relations", []) or []
                            ),
                        })
                        certificate_portfolio_log["executed_plan_count"] += 1
                        if complete:
                            candidates = alternative_candidates
                            groups = alternative_groups
                            selected_alternative_ir = alternative_ir
                            break
                finally:
                    self._active_relation_realizations = saved_realizations
                    self._active_relation_directions = saved_directions
                    self._active_relation_predicate_directions = (
                        saved_predicate_directions
                    )
                    self._strict_relation_predicates = saved_strict_predicates

                if selected_alternative_ir:
                    typed_plan_ir = selected_alternative_ir
                    log["logical_plan_ir"] = dict(selected_alternative_ir)
                    certificate_portfolio_log.update({
                        "selected_operator": "typed_variable_unification_join",
                        "selected_plan_id": str(
                            selected_alternative_ir.get("plan_id", "") or ""
                        ),
                        "decision_reason": "alternative_typed_certificate_complete",
                    })
                else:
                    fallback_operator = ""
                    if comparison and self.enable_dual_branch_comparison_join:
                        fallback_operator = "dual_branch_comparison_join"
                    elif len(relations) >= 2 and self.enable_cascaded_bind_join:
                        fallback_operator = "cascaded_bind_join"
                    elif self.enable_entity_relation_index_scan:
                        fallback_operator = "entity_relation_index_scan"
                    fallback_candidates: List[Dict[str, Any]] = []
                    fallback_groups: List[Dict[str, Any]] = []
                    fallback_probes: List[Dict[str, Any]] = []
                    if fallback_operator == "dual_branch_comparison_join":
                        fallback_candidates, fallback_groups = (
                            self._dual_branch_comparison_join(
                                query,
                                anchors,
                                comparison_branches,
                                relations,
                                answer_type,
                                dense_retrieve,
                                get_openie_record,
                                lookup_entity_relations,
                                fallback_probes,
                            )
                        )
                    elif fallback_operator == "cascaded_bind_join":
                        fallback_candidates, fallback_groups = self._cascaded_bind_join(
                            query,
                            anchors,
                            relations,
                            answer_type,
                            dense_retrieve,
                            get_openie_record,
                            lookup_entity_relations,
                            fallback_probes,
                        )
                    elif fallback_operator == "entity_relation_index_scan":
                        fallback_candidates, fallback_groups = (
                            self._entity_relation_index_scan(
                                query,
                                anchors,
                                relations,
                                answer_type,
                                dense_retrieve,
                                get_openie_record,
                                lookup_entity_relations,
                                fallback_probes,
                            )
                        )
                    probes.extend(fallback_probes)
                    if fallback_operator:
                        certificate_portfolio_log["executed_plan_count"] += 1
                    certificate_portfolio_log.update({
                        "fallback_operator": fallback_operator,
                        "fallback_candidate_count": len(fallback_candidates),
                        "fallback_group_count": len(fallback_groups),
                    })
                    if fallback_candidates:
                        candidates = fallback_candidates
                        groups = fallback_groups
                        operator = fallback_operator
                        log["primary_selected_operator"] = (
                            "typed_variable_unification_join"
                        )
                        log["selected_operator"] = operator
                        certificate_portfolio_log.update({
                            "selected_operator": operator,
                            "selected_plan_id": "",
                            "decision_reason": "typed_failed_fallback_succeeded",
                        })
                    else:
                        certificate_portfolio_log["decision_reason"] = (
                            "all_certificate_aware_plans_failed"
                        )
        log["certificate_aware_plan_portfolio"] = certificate_portfolio_log

        primary_operator = operator
        primary_candidate_count = len(candidates)
        portfolio_log: Dict[str, Any] = {
            "enabled": self.enable_provenance_plan_portfolio,
            "triggered": False,
            "primary_operator": primary_operator,
            "primary_candidate_count": primary_candidate_count,
            "candidate_count": 0,
            "covered_relation_slots": [],
            "probe_count": 0,
            "decision_reason": "disabled",
        }
        if self.enable_provenance_plan_portfolio:
            if candidates:
                portfolio_log["decision_reason"] = "primary_plan_succeeded"
            else:
                portfolio_candidates, portfolio_probes = self._provenance_plan_portfolio(
                    query=query,
                    anchors=anchors,
                    relations=relations,
                    answer_type=answer_type,
                    primary_operator=primary_operator,
                    dense_retrieve=dense_retrieve,
                    get_openie_record=get_openie_record,
                    lookup_entity_relations=lookup_entity_relations,
                )
                probes.extend(portfolio_probes)
                portfolio_log.update({
                    "triggered": True,
                    "candidate_count": len(portfolio_candidates),
                    "covered_relation_slots": sorted({
                        int(item.get("proof_relation_slot", -1))
                        for item in portfolio_candidates
                        if int(item.get("proof_relation_slot", -1)) >= 0
                    }),
                    "probe_count": len(portfolio_probes),
                    "decision_reason": (
                        "partial_proofs_materialized"
                        if portfolio_candidates else "no_validated_partial_proofs"
                    ),
                })
                if portfolio_candidates:
                    candidates = portfolio_candidates
                    groups = []
                    operator = "entity_relation_index_scan"
                    log["selected_operator"] = operator
                    log["primary_selected_operator"] = primary_operator
        log["provenance_plan_portfolio"] = portfolio_log

        execution_latency_ms = (time.perf_counter() - execution_started_at) * 1000.0

        candidates = self._deduplicate_candidates(candidates)[: self.max_candidates]
        baseline_rank = {int(doc_id): rank + 1 for rank, doc_id in enumerate(base_doc_ids)}
        for candidate in candidates:
            passage_idx = int(candidate["passage_idx"])
            candidate["original_rank"] = int(baseline_rank.get(passage_idx, 10**9))
            candidate["was_in_baseline"] = passage_idx in baseline_rank
            candidate["from_beyond_top5"] = int(candidate["original_rank"]) > 5
            candidate["factorized_query_executor"] = True
            candidate["typed_contract_confidence"] = float(
                typed_contract.get("contract_confidence", 0.0) or 0.0
            )
            candidate["query_predicate_coverage"] = float(
                typed_contract.get("query_predicate_coverage", 0.0) or 0.0
            )
            candidate["source_schema_ids"] = list(compiled_source_schema_ids)
            candidate["endpoint_grounded_source_schema_ids"] = list(
                endpoint_grounded_source_schema_ids
            )
            candidate["logical_plan_id"] = str(
                logical_evidence_plan.get("plan_id", "") or ""
            )
            candidate["physical_operator"] = operator
        log["probes"] = probes
        log["probe_count"] = len(probes)
        log["execution_latency_ms"] = round(execution_latency_ms, 6)
        log["probe_latency_ms"] = round(
            sum(float(item.get("latency_ms", 0.0) or 0.0) for item in probes),
            6,
        )
        log["candidates"] = candidates
        log["groups"] = groups
        log["candidate_count"] = len(candidates)
        log["candidate_from_beyond_top5_count"] = sum(
            1 for candidate in candidates if candidate.get("from_beyond_top5", False)
        )
        log["candidate_from_beyond_top5"] = bool(log["candidate_from_beyond_top5_count"])
        log["inverse_direction_fallback_candidate_count"] = sum(
            1 for candidate in candidates
            if candidate.get("inverse_direction_fallback", False)
        )
        selected_plan = dict(physical_plan_optimizer.get("selected", {}) or {})
        estimated_cardinality = float(
            selected_plan.get("estimated_cardinality", 0.0) or 0.0
        )
        actual_cardinality = float(len(candidates))
        log["estimated_candidate_cardinality"] = round(estimated_cardinality, 6)
        log["actual_candidate_cardinality"] = int(actual_cardinality)
        log["cardinality_absolute_error"] = round(
            abs(estimated_cardinality - actual_cardinality),
            6,
        )
        if estimated_cardinality == 0.0 and actual_cardinality == 0.0:
            cardinality_q_error = 1.0
        else:
            cardinality_q_error = max(
                (estimated_cardinality + 1.0) / (actual_cardinality + 1.0),
                (actual_cardinality + 1.0) / (estimated_cardinality + 1.0),
            )
        log["cardinality_q_error"] = round(cardinality_q_error, 6)
        log["group_count"] = len(groups)
        log["external_candidate_count"] = sum(1 for item in candidates if not item["was_in_baseline"])
        provenance = dict(log.get("schema_execution_provenance", {}) or {})
        provenance.update({
            "physical_operator_selected": bool(operator),
            "selected_operator": operator,
            "physical_execution_completed": bool(operator and candidates),
            "candidate_count": len(candidates),
            "external_candidate_count": int(log["external_candidate_count"]),
            "candidate_doc_ids": [
                int(item["passage_idx"]) for item in candidates
            ],
            "external_candidate_doc_ids": [
                int(item["passage_idx"])
                for item in candidates
                if not item["was_in_baseline"]
            ],
        })
        log["schema_execution_provenance"] = provenance
        log["runtime_rejections"] = dict(self._runtime_rejections)
        log["skip_reason"] = "candidate_only" if candidates else "no_validated_candidates"
        return FactorizedQueryExecutorResult(candidates=candidates, groups=groups, log=log)
