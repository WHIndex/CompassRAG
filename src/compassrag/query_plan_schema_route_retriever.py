import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from .query_ontology_prior import tokenize
from .query_plan_structural_candidate_retriever import QueryPlanOntologyStructuralCandidateRetriever
from .utils.misc_utils import text_processing


SCHEMA_ROUTE_POLICY_FEATURE_NAMES = [
    "confidence",
    "score",
    "dense_rank_inv",
    "dense_rank_log",
    "dense_rank_le_5",
    "dense_rank_le_20",
    "original_rank_inv",
    "original_rank_log",
    "original_rank_le_5",
    "original_rank_le_10",
    "original_rank_le_20",
    "was_in_baseline",
    "route_entity",
    "route_triple",
    "route_path",
    "route_dfs",
    "route_community",
    "source_schema",
    "source_sample_path",
    "source_matched_plan",
    "source_community",
    "matched_anchor",
    "matched_terminal",
    "answer_hit",
    "relation_hit_count",
    "relation_hit_count_log",
    "relation_path_len",
    "relation_path_len_log",
    "has_fact_indices",
    "fact_indices_log",
    "signature_strict",
    "signature_soft",
    "path_score",
    "plan_precision",
    "anchor_token_count_log",
    "anchor_concrete",
    "terminal_present",
    "terminal_token_count_log",
    "query_condition_multihop",
    "query_condition_simple",
    "answer_type_date",
    "answer_type_location",
    "answer_type_person",
    "answer_type_number",
    "answer_type_entity",
    "matched_plan_count_log",
    "strict_executable_path_count_log",
    "executable_path_count_log",
]


_POLICY_GENERIC_TERMS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "have", "had", "in", "is", "of", "on", "or", "the", "to",
    "was", "were", "with", "what", "when", "where", "which", "who",
    "whom", "whose", "why", "how",
}


def _policy_safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return default
        return number
    except Exception:
        return default


def _policy_safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def _policy_sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _policy_tokens(value: Any, max_tokens: int = 16) -> List[str]:
    output: List[str] = []
    seen = set()
    for token in tokenize(text_processing(str(value or "").replace("_", " "))):
        if token in _POLICY_GENERIC_TERMS or token in seen:
            continue
        output.append(token)
        seen.add(token)
        if len(output) >= max_tokens:
            break
    return output


def _schema_route_query_condition_features(query_plan_log: Mapping[str, Any]) -> Dict[str, float]:
    condition = query_plan_log.get("query_condition", {}) or {}
    if not isinstance(condition, Mapping):
        condition = {}
    answer_type = str(condition.get("answer_type", "") or "").lower()
    text = " ".join(
        str(condition.get(key, "") or "")
        for key in ("query_type", "intent", "answer_type", "family", "mode")
    ).lower()
    relation_families = " ".join(str(item) for item in condition.get("relation_families", []) or []).lower()
    relation_signatures = " ".join(str(item) for item in condition.get("relation_signatures", []) or []).lower()
    combined = f"{text} {relation_families} {relation_signatures}"
    return {
        "query_condition_multihop": 1.0 if any(token in combined for token in ("multi", "bridge", "hop", "composition", "comparison")) else 0.0,
        "query_condition_simple": 1.0 if any(token in combined for token in ("simple", "factoid", "single")) else 0.0,
        "answer_type_date": 1.0 if answer_type == "date" else 0.0,
        "answer_type_location": 1.0 if answer_type == "location" else 0.0,
        "answer_type_person": 1.0 if answer_type == "person" else 0.0,
        "answer_type_number": 1.0 if answer_type == "number" else 0.0,
        "answer_type_entity": 1.0 if answer_type not in {"date", "location", "person", "number"} else 0.0,
    }


def extract_schema_route_policy_features(
    query_plan_log: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> Dict[str, float]:
    contexts = candidate.get("contexts", []) or []
    contexts = [context for context in contexts if isinstance(context, Mapping)]
    context = contexts[0] if contexts else {}
    probe = candidate.get("probe", {}) or {}
    if not isinstance(probe, Mapping):
        probe = {}

    route = str(candidate.get("route") or probe.get("route") or context.get("route") or "").lower()
    source = str(context.get("schema_route_source") or probe.get("source") or "").lower()
    dense_rank = _policy_safe_int(candidate.get("dense_rank"), 10_000)
    if dense_rank <= 0:
        dense_rank = 10_000
    original_rank = _policy_safe_int(candidate.get("original_rank"), 10_000)
    if original_rank <= 0:
        original_rank = 10_000

    relation_path = probe.get("relation_path", []) or context.get("relations", []) or []
    if not isinstance(relation_path, list):
        relation_path = []
    fact_indices = context.get("fact_indices", []) or []
    if not isinstance(fact_indices, list):
        fact_indices = []
    signature_match_mode = str(context.get("signature_match_mode") or "").lower()
    relation_hits = candidate.get("relation_hits", []) or []
    if not isinstance(relation_hits, list):
        relation_hits = []

    anchor = probe.get("anchor") or context.get("anchor_entity") or ""
    terminal = probe.get("terminal") or context.get("end_entity") or ""
    anchor_tokens = _policy_tokens(anchor)
    terminal_tokens = _policy_tokens(terminal)

    features = {
        "confidence": _policy_safe_float(candidate.get("confidence"), 0.0),
        "score": _policy_safe_float(candidate.get("score"), 0.0),
        "dense_rank_inv": 1.0 / float(dense_rank),
        "dense_rank_log": math.log1p(float(dense_rank)),
        "dense_rank_le_5": 1.0 if dense_rank <= 5 else 0.0,
        "dense_rank_le_20": 1.0 if dense_rank <= 20 else 0.0,
        "original_rank_inv": 1.0 / float(original_rank),
        "original_rank_log": math.log1p(float(original_rank)),
        "original_rank_le_5": 1.0 if original_rank <= 5 else 0.0,
        "original_rank_le_10": 1.0 if original_rank <= 10 else 0.0,
        "original_rank_le_20": 1.0 if original_rank <= 20 else 0.0,
        "was_in_baseline": 1.0 if bool(candidate.get("was_in_baseline")) else 0.0,
        "route_entity": 1.0 if route == "entity" else 0.0,
        "route_triple": 1.0 if route == "triple" else 0.0,
        "route_path": 1.0 if route == "path" else 0.0,
        "route_dfs": 1.0 if route == "dfs" else 0.0,
        "route_community": 1.0 if route == "community" else 0.0,
        "source_schema": 1.0 if source == "schema" else 0.0,
        "source_sample_path": 1.0 if source == "sample_path" else 0.0,
        "source_matched_plan": 1.0 if source == "matched_plan" else 0.0,
        "source_community": 1.0 if source == "community" else 0.0,
        "matched_anchor": 1.0 if bool(candidate.get("matched_anchor")) else 0.0,
        "matched_terminal": 1.0 if bool(candidate.get("matched_terminal")) else 0.0,
        "answer_hit": 1.0 if bool(candidate.get("answer_hit")) else 0.0,
        "relation_hit_count": float(len(relation_hits)),
        "relation_hit_count_log": math.log1p(float(len(relation_hits))),
        "relation_path_len": float(len(relation_path)),
        "relation_path_len_log": math.log1p(float(len(relation_path))),
        "has_fact_indices": 1.0 if fact_indices else 0.0,
        "fact_indices_log": math.log1p(float(len(fact_indices))),
        "signature_strict": 1.0 if "strict" in signature_match_mode else 0.0,
        "signature_soft": 1.0 if "soft" in signature_match_mode else 0.0,
        "path_score": _policy_safe_float(context.get("path_score"), 0.0),
        "plan_precision": _policy_safe_float(context.get("plan_effective_precision"), 0.0),
        "anchor_token_count_log": math.log1p(float(len(anchor_tokens))),
        "anchor_concrete": 1.0 if len(anchor_tokens) >= 2 else 0.0,
        "terminal_present": 1.0 if bool(terminal_tokens) else 0.0,
        "terminal_token_count_log": math.log1p(float(len(terminal_tokens))),
        "matched_plan_count_log": math.log1p(_policy_safe_float(query_plan_log.get("matched_plan_count"), 0.0)),
        "strict_executable_path_count_log": math.log1p(_policy_safe_float(query_plan_log.get("strict_executable_path_count"), 0.0)),
        "executable_path_count_log": math.log1p(_policy_safe_float(query_plan_log.get("executable_path_count"), 0.0)),
    }
    features.update(_schema_route_query_condition_features(query_plan_log))
    return {name: float(features.get(name, 0.0)) for name in SCHEMA_ROUTE_POLICY_FEATURE_NAMES}


def schema_route_policy_feature_vector(
    features: Mapping[str, float],
    feature_names: List[str] = None,
) -> List[float]:
    names = feature_names or SCHEMA_ROUTE_POLICY_FEATURE_NAMES
    return [float(features.get(name, 0.0)) for name in names]


class QueryPlanSchemaRoutePolicy:
    def __init__(self, model_path: str = "") -> None:
        self.model_path = str(model_path or "")
        self.model = self._load_model(self.model_path)

    @staticmethod
    def _load_model(model_path: str) -> Dict[str, Any]:
        if not model_path or not os.path.exists(model_path):
            return {}
        with open(model_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}

    def is_ready(self) -> bool:
        return bool(self.model.get("feature_names")) and bool(self.model.get("weights"))

    def score_features(self, features: Mapping[str, float]) -> float:
        if not self.is_ready():
            return 0.0
        names = list(self.model.get("feature_names", SCHEMA_ROUTE_POLICY_FEATURE_NAMES) or SCHEMA_ROUTE_POLICY_FEATURE_NAMES)
        weights = [float(value) for value in self.model.get("weights", [])]
        means = [float(value) for value in self.model.get("means", [0.0] * len(names))]
        stds = [float(value) if float(value) != 0.0 else 1.0 for value in self.model.get("stds", [1.0] * len(names))]
        bias = float(self.model.get("bias", 0.0) or 0.0)
        values = schema_route_policy_feature_vector(features, names)
        total = bias
        for index, value in enumerate(values):
            if index >= len(weights):
                break
            total += weights[index] * ((value - means[index]) / stds[index])
        return _policy_sigmoid(total)

    def score(self, query_plan_log: Mapping[str, Any], candidate: Mapping[str, Any]) -> Dict[str, Any]:
        features = extract_schema_route_policy_features(query_plan_log, candidate)
        score = self.score_features(features)
        return {
            "score": round(float(score), 6),
            "model_path": self.model_path,
            "model_ready": self.is_ready(),
            "feature_snapshot": {
                "route_path": round(features.get("route_path", 0.0), 4),
                "route_dfs": round(features.get("route_dfs", 0.0), 4),
                "source_sample_path": round(features.get("source_sample_path", 0.0), 4),
                "matched_terminal": round(features.get("matched_terminal", 0.0), 4),
                "fact_indices_log": round(features.get("fact_indices_log", 0.0), 4),
                "original_rank_log": round(features.get("original_rank_log", 0.0), 4),
            },
        }


@dataclass
class QueryPlanSchemaRouteResult:
    sorted_doc_ids: np.ndarray
    sorted_doc_scores: np.ndarray
    log: Dict[str, Any] = field(default_factory=dict)


class QueryPlanSchemaRouteRetriever:
    """Schema-guided decomposition with route-specific dense retrieval.

        The retriever materializes schema routes as retrieval probes:
      - entity: anchor-centric lookup
      - triple: anchor + relation lookup
      - path/dfs: multi-relation path lookup
      - community: topic/schema lookup for weakly anchored queries

        Route candidates are appended and then selected by the evidence-set
        retriever, which preserves strong incumbent evidence while extending recall.
    """

    GENERIC_TERMS = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
        "has", "have", "had", "in", "is", "of", "on", "or", "the", "to",
        "was", "were", "with", "what", "when", "where", "which", "who",
        "whom", "whose", "why", "how",
    }

    ANSWER_HINTS = {
        "date": ["date", "year", "month", "day", "born", "died", "released"],
        "time": ["date", "year", "time", "when"],
        "year": ["date", "year", "when"],
        "number": ["number", "count", "total", "age", "rank", "population"],
        "quantity": ["number", "count", "total"],
        "location": ["place", "city", "country", "state", "county", "location"],
        "person": ["person", "name", "who"],
        "organization": ["organization", "company", "team", "group"],
        "work": ["work", "film", "book", "album", "song"],
        "creative work": ["work", "film", "book", "album", "song"],
    }

    def __init__(
        self,
        dense_top_k: int = 80,
        max_subqueries: int = 8,
        max_candidates: int = 6,
        min_confidence: float = 0.58,
        preserve_top_k: int = 2,
        synthetic_original_rank: int = 35,
        candidate_only: bool = True,
        append_score_margin: float = 1e-7,
        require_anchor_hit: bool = True,
        require_route_signal: bool = True,
        enable_community_route: bool = True,
        enable_dfs_route: bool = True,
        policy_model_path: str = "",
        policy_mode: str = "off",
        policy_min_score: float = 0.55,
        policy_reject_below: float = 0.0,
        policy_score_weight: float = 0.12,
    ) -> None:
        self.dense_top_k = max(int(dense_top_k), 1)
        self.max_subqueries = max(int(max_subqueries), 0)
        self.max_candidates = max(int(max_candidates), 0)
        self.min_confidence = min(max(float(min_confidence), 0.0), 1.0)
        self.preserve_top_k = max(int(preserve_top_k), 0)
        self.synthetic_original_rank = max(int(synthetic_original_rank), 1)
        self.candidate_only = bool(candidate_only)
        self.append_score_margin = max(float(append_score_margin), 0.0)
        self.require_anchor_hit = bool(require_anchor_hit)
        self.require_route_signal = bool(require_route_signal)
        self.enable_community_route = bool(enable_community_route)
        self.enable_dfs_route = bool(enable_dfs_route)
        self.policy_model_path = str(policy_model_path or "")
        self.policy_mode = str(policy_mode or "off").lower()
        if self.policy_mode not in {"off", "score", "filter"}:
            self.policy_mode = "off"
        self.policy_min_score = min(max(float(policy_min_score), 0.0), 1.0)
        self.policy_reject_below = min(max(float(policy_reject_below), 0.0), 1.0)
        self.policy_score_weight = max(float(policy_score_weight), 0.0)
        self.policy = QueryPlanSchemaRoutePolicy(self.policy_model_path)

    @staticmethod
    def _norm(value: Any) -> str:
        return text_processing(str(value or ""))

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _unique(values: Iterable[Any], max_items: int = 32) -> List[str]:
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
    def _tokens(cls, value: Any, max_tokens: int = 8) -> List[str]:
        output: List[str] = []
        seen = set()
        for token in tokenize(cls._norm(str(value or "").replace("_", " "))):
            if token in cls.GENERIC_TERMS or token in seen:
                continue
            output.append(token)
            seen.add(token)
            if len(output) >= max_tokens:
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
        return matched / max(len(phrase_tokens), 1) >= 0.75

    def _answer_hints(self, query_plan_log: Mapping[str, Any]) -> List[str]:
        condition = query_plan_log.get("query_condition", {}) or {}
        answer_type = self._norm(condition.get("answer_type", ""))
        hints: List[str] = []
        hints.extend(condition.get("answer_hints", []) or [])
        hints.extend(self.ANSWER_HINTS.get(answer_type, []))
        if answer_type:
            hints.append(answer_type)
        return self._unique(hints, max_items=8)

    def _anchors(self, query_plan_log: Mapping[str, Any]) -> List[str]:
        anchors: List[Any] = []
        anchors.extend(query_plan_log.get("anchor_entities", []) or [])
        for path in query_plan_log.get("sample_paths", []) or []:
            if isinstance(path, Mapping):
                anchors.append(path.get("anchor_entity", ""))
        for passage in query_plan_log.get("lifted_passages", []) or []:
            for context in passage.get("contexts", []) or []:
                if isinstance(context, Mapping):
                    anchors.append(context.get("anchor_entity", ""))
        return [
            item for item in self._unique(anchors, max_items=16)
            if not QueryPlanOntologyStructuralCandidateRetriever.is_generic_anchor(item)
        ]

    def _terminals(self, query_plan_log: Mapping[str, Any]) -> List[str]:
        terminals: List[Any] = []
        for path in query_plan_log.get("sample_paths", []) or []:
            if isinstance(path, Mapping):
                terminals.append(path.get("end_entity", ""))
                terminals.append(path.get("entity", ""))
        for passage in query_plan_log.get("lifted_passages", []) or []:
            for context in passage.get("contexts", []) or []:
                if isinstance(context, Mapping):
                    terminals.append(context.get("end_entity", ""))
        return [
            item for item in self._unique(terminals, max_items=16)
            if not QueryPlanOntologyStructuralCandidateRetriever.is_generic_anchor(item)
        ]

    def _relations(self, query_plan_log: Mapping[str, Any]) -> List[str]:
        values: List[Any] = []
        condition = query_plan_log.get("query_condition", {}) or {}
        values.extend(condition.get("relation_terms", []) or [])
        values.extend(condition.get("relation_signatures", []) or [])
        values.extend(condition.get("relation_families", []) or [])
        for plan in query_plan_log.get("matched_plans", []) or []:
            if isinstance(plan, Mapping):
                values.extend(plan.get("relation_signatures", []) or [])
                values.extend(plan.get("relation_families", []) or [])
                values.extend(plan.get("triggers", []) or [])
        for path in query_plan_log.get("sample_paths", []) or []:
            if isinstance(path, Mapping):
                values.extend(path.get("relations", []) or [])
        phrases = []
        for value in self._unique(values, max_items=48):
            tokens = self._tokens(value, max_tokens=5)
            if tokens:
                phrases.append(" ".join(tokens))
        return self._unique(phrases, max_items=18)

    def _plan_path_specs(self, query_plan_log: Mapping[str, Any]) -> List[Dict[str, Any]]:
        specs: List[Dict[str, Any]] = []
        for plan in query_plan_log.get("matched_plans", []) or []:
            if not isinstance(plan, Mapping):
                continue
            plan_id = str(plan.get("id") or "schema_route")
            precision = self._safe_float(plan.get("effective_precision", plan.get("precision", 0.0)), 0.0)
            relations = []
            for step in plan.get("path", []) or []:
                if not isinstance(step, Mapping):
                    continue
                relation = (
                    step.get("relation_signature")
                    or step.get("relation_family")
                    or " ".join(step.get("relation_aliases", []) or [])
                )
                relation_text = " ".join(self._tokens(relation, max_tokens=5))
                if relation_text:
                    relations.append(relation_text)
            if relations:
                specs.append({
                    "plan_id": plan_id,
                    "precision": precision,
                    "relations": self._unique(relations, max_items=6),
                    "answer_type": self._norm(plan.get("answer_type", "")),
                })
        return specs[:8]

    def _query_relation_token_set(self, query_plan_log: Mapping[str, Any]) -> set:
        condition = query_plan_log.get("query_condition", {}) or {}
        values: List[Any] = []
        values.extend(condition.get("relation_terms", []) or [])
        values.extend(condition.get("relation_signatures", []) or [])
        values.extend(condition.get("relation_families", []) or [])
        output = set()
        for value in values:
            output.update(self._tokens(value, max_tokens=8))
        return output

    def _sample_path_specs(self, query_plan_log: Mapping[str, Any]) -> List[Dict[str, Any]]:
        specs: List[Dict[str, Any]] = []
        query_relation_tokens = self._query_relation_token_set(query_plan_log)
        for path in query_plan_log.get("sample_paths", []) or []:
            if not isinstance(path, Mapping):
                continue
            anchor = self._norm(path.get("anchor_entity", ""))
            terminal = self._norm(path.get("end_entity", ""))
            if anchor and QueryPlanOntologyStructuralCandidateRetriever.is_generic_anchor(anchor):
                anchor = ""
            if terminal and QueryPlanOntologyStructuralCandidateRetriever.is_generic_anchor(terminal):
                terminal = ""
            relations = []
            for relation in path.get("relations", []) or []:
                relation_text = " ".join(self._tokens(relation, max_tokens=5))
                if relation_text:
                    relations.append(relation_text)
            relations = relations[:6]
            if not relations:
                continue
            path_relation_tokens = {
                token
                for relation in relations
                for token in self._tokens(relation, max_tokens=8)
            }
            if query_relation_tokens and not query_relation_tokens.intersection(path_relation_tokens):
                continue
            score = self._safe_float(path.get("score"), 0.0)
            precision = self._safe_float(
                path.get("plan_effective_precision", path.get("plan_lift_weight_scale", 0.0)),
                0.0,
            )
            specs.append({
                "plan_id": str(path.get("plan_id") or "sample_path"),
                "anchor": anchor,
                "terminal": terminal,
                "relations": relations,
                "fact_indices": list(path.get("fact_indices", []) or []),
                "signature_match_mode": str(path.get("signature_match_mode") or ""),
                "score": score,
                "precision": precision,
                "answer_type": self._norm(path.get("plan_answer_type", "")),
                "source": "sample_path",
            })
        specs.sort(
            key=lambda item: (
                len(item.get("relations", []) or []),
                self._safe_float(item.get("score"), 0.0),
                self._safe_float(item.get("precision"), 0.0),
            ),
            reverse=True,
        )
        deduped: List[Dict[str, Any]] = []
        seen = set()
        for spec in specs:
            key = (
                spec.get("anchor", ""),
                spec.get("terminal", ""),
                tuple(spec.get("relations", []) or []),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(spec)
            if len(deduped) >= 16:
                break
        return deduped

    def build_route_probes(self, query: str, query_plan_log: Mapping[str, Any]) -> List[Dict[str, Any]]:
        anchors = self._anchors(query_plan_log)
        terminals = self._terminals(query_plan_log)
        relations = self._relations(query_plan_log)
        answer_hints = self._answer_hints(query_plan_log)
        sample_path_specs = self._sample_path_specs(query_plan_log)
        plan_specs = self._plan_path_specs(query_plan_log)
        hint_text = " ".join(answer_hints[:2])

        probes: List[Dict[str, Any]] = []
        seen = set()

        def add_probe(route: str, text: str, anchor: str = "", relation: str = "", terminal: str = "",
                      plan_id: str = "schema_route", precision: float = 0.0, relation_path: Sequence[str] = (),
                      fact_indices: Sequence[int] = (), source: str = "schema", path_score: float = 0.0,
                      signature_match_mode: str = "") -> None:
            key = self._norm(f"{route} {text}")
            if not key or key in seen:
                return
            seen.add(key)
            probes.append({
                "route": route,
                "query": text,
                "anchor": anchor,
                "relation": relation,
                "terminal": terminal,
                "relation_path": list(relation_path or []),
                "answer_hints": answer_hints,
                "plan_id": plan_id,
                "plan_effective_precision": precision,
                "fact_indices": list(fact_indices or []),
                "source": source,
                "path_score": path_score,
                "signature_match_mode": signature_match_mode,
            })




        for spec in sample_path_specs[:8]:
            relation_path = list(spec.get("relations", []) or [])
            relation_chain = " ".join(relation_path[:4])
            anchor = str(spec.get("anchor", ""))
            terminal = str(spec.get("terminal", ""))
            route = "dfs" if self.enable_dfs_route and (
                len(relation_path) >= 2 or len(spec.get("fact_indices", []) or []) >= 2
            ) else "path"
            add_probe(
                route,
                " ".join(piece for piece in [anchor, relation_chain, terminal, hint_text] if piece),
                anchor=anchor,
                relation=relation_chain,
                terminal=terminal,
                plan_id=str(spec.get("plan_id") or "sample_path"),
                precision=self._safe_float(spec.get("precision"), 0.0),
                relation_path=relation_path,
                fact_indices=spec.get("fact_indices", []) or [],
                source="sample_path",
                path_score=self._safe_float(spec.get("score"), 0.0),
                signature_match_mode=str(spec.get("signature_match_mode") or ""),
            )
            if len(relation_path) == 1 and anchor:
                add_probe(
                    "triple",
                    " ".join(piece for piece in [anchor, relation_path[0], terminal, hint_text] if piece),
                    anchor=anchor,
                    relation=relation_path[0],
                    terminal=terminal,
                    plan_id=str(spec.get("plan_id") or "sample_path"),
                    precision=self._safe_float(spec.get("precision"), 0.0),
                    relation_path=relation_path,
                    fact_indices=spec.get("fact_indices", []) or [],
                    source="sample_path",
                    path_score=self._safe_float(spec.get("score"), 0.0),
                    signature_match_mode=str(spec.get("signature_match_mode") or ""),
                )
            if len(probes) >= max(self.max_subqueries // 2, 4):
                break

        for anchor in anchors[:4]:
            add_probe("entity", " ".join(piece for piece in [query, anchor] if piece), anchor=anchor)
            for relation in relations[:4]:
                add_probe(
                    "triple",
                    " ".join(piece for piece in [anchor, relation, hint_text] if piece),
                    anchor=anchor,
                    relation=relation,
                )
        for spec in plan_specs[:6]:
            relation_path = list(spec.get("relations", []) or [])
            if not relation_path:
                continue
            relation_chain = " ".join(relation_path[:4])
            for anchor in anchors[:4] or [""]:
                for terminal in terminals[:3] or [""]:
                    route = "dfs" if self.enable_dfs_route and len(relation_path) >= 2 else "path"
                    add_probe(
                        route,
                        " ".join(piece for piece in [anchor, relation_chain, terminal, hint_text] if piece),
                        anchor=anchor,
                        relation=relation_chain,
                        terminal=terminal,
                        plan_id=str(spec.get("plan_id") or "schema_route"),
                        precision=self._safe_float(spec.get("precision"), 0.0),
                        relation_path=relation_path,
                        source="matched_plan",
                    )
                    if len(probes) >= self.max_subqueries:
                        return probes[: self.max_subqueries]
        if self.enable_community_route:
            condition = query_plan_log.get("query_condition", {}) or {}
            no_anchor = not anchors
            broad_query = len(plan_specs) == 0 or self._safe_float(query_plan_log.get("matched_plan_count"), 0.0) <= 1
            if no_anchor or broad_query:
                add_probe(
                    "community",
                    " ".join(piece for piece in [query, " ".join(relations[:4]), hint_text] if piece),
                    relation=" ".join(relations[:4]),
                    terminal=self._norm(condition.get("answer_type", "")),
                    source="community",
                )
        if not probes:
            add_probe("entity", query)
        return probes[: self.max_subqueries]

    def _answer_type_hit(self, text_norm: str, hints: Sequence[str]) -> bool:
        tokens = set(tokenize(text_norm))
        hint_tokens = {token for hint in hints for token in self._tokens(hint, max_tokens=3)}
        if hint_tokens and tokens.intersection(hint_tokens):
            return True
        if any(hint in {"date", "year", "month", "day", "time"} for hint in hints):
            return any(token.isdigit() and len(token) == 4 for token in tokens)
        if any(hint in {"number", "count", "total", "age", "quantity"} for hint in hints):
            return any(token.isdigit() for token in tokens)
        return False

    def _score_candidate(
        self,
        passage_text: str,
        dense_score: float,
        max_dense_score: float,
        probe: Mapping[str, Any],
    ) -> Tuple[float, Dict[str, Any]]:
        text_norm = self._norm(passage_text)
        route = str(probe.get("route") or "entity")
        anchor_hit = self._contains_phrase(text_norm, probe.get("anchor", ""))
        terminal_hit = self._contains_phrase(text_norm, probe.get("terminal", ""))
        relation_terms = list(probe.get("relation_path", []) or [])
        if probe.get("relation"):
            relation_terms.append(probe.get("relation", ""))
        relation_tokens = []
        for relation in relation_terms:
            relation_tokens.extend(self._tokens(relation, max_tokens=5))
        relation_tokens = list(dict.fromkeys(relation_tokens))
        text_tokens = set(tokenize(text_norm))
        relation_hits = [token for token in relation_tokens if token in text_tokens]
        relation_score = min(len(relation_hits) / max(min(len(relation_tokens), 4), 1), 1.0)
        answer_hit = self._answer_type_hit(text_norm, probe.get("answer_hints", []) or [])
        dense_norm = 0.0 if max_dense_score <= 0.0 else float(dense_score) / max_dense_score
        dense_norm = min(max(dense_norm, 0.0), 1.0)
        precision = min(max(self._safe_float(probe.get("plan_effective_precision"), 0.0), 0.0), 1.0)
        path_score = min(max(self._safe_float(probe.get("path_score"), 0.0) / 3.0, 0.0), 1.0)
        relation_path_len = len(probe.get("relation_path", []) or [])

        route_bonus = {
            "entity": 0.05,
            "triple": 0.06,
            "path": 0.12,
            "dfs": 0.16,
            "community": 0.01,
        }.get(route, 0.06)
        confidence = (
            0.30 * dense_norm
            + 0.22 * (1.0 if anchor_hit else 0.0)
            + 0.22 * relation_score
            + 0.11 * (1.0 if terminal_hit else 0.0)
            + 0.05 * (1.0 if answer_hit else 0.0)
            + 0.04 * precision
            + 0.04 * path_score
            + 0.02 * min(relation_path_len / 2.0, 1.0)
            + route_bonus
        )
        detail = {
            "route": route,
            "anchor_hit": anchor_hit,
            "terminal_hit": terminal_hit,
            "relation_hits": relation_hits,
            "relation_score": relation_score,
            "answer_hit": answer_hit,
            "dense_norm": dense_norm,
            "plan_precision": precision,
            "path_score": path_score,
            "relation_path_len": relation_path_len,
            "confidence": min(max(confidence, 0.0), 1.0),
        }
        return detail["confidence"], detail

    def _route_accepts(self, probe: Mapping[str, Any], detail: Mapping[str, Any]) -> Tuple[bool, str]:
        route = str(probe.get("route") or "entity")
        has_anchor = bool(self._norm(probe.get("anchor", "")))
        if self.require_anchor_hit and has_anchor and route in {"entity", "triple", "path", "dfs"}:
            if not bool(detail.get("anchor_hit")):
                return False, "anchor_miss"
        route_signal = bool(
            detail.get("relation_hits")
            or detail.get("terminal_hit")
            or detail.get("answer_hit")
            or (route == "entity" and detail.get("anchor_hit"))
            or (route == "community" and detail.get("dense_norm", 0.0) >= 0.92)
        )
        if self.require_route_signal and not route_signal:
            return False, "route_signal_miss"
        relation_score = self._safe_float(detail.get("relation_score"), 0.0)
        dense_norm = self._safe_float(detail.get("dense_norm"), 0.0)
        relation_hit_count = len(detail.get("relation_hits", []) or [])
        if route == "community":
            if dense_norm < 0.97 or not (detail.get("answer_hit") or relation_score >= 0.50):
                return False, "weak_community_route"
        if route == "entity":
            if not detail.get("anchor_hit") or dense_norm < 0.72:
                return False, "weak_entity_route"
        if route == "triple":
            if relation_score < 0.50:
                return False, "weak_triple_relation"
            if not (detail.get("terminal_hit") or detail.get("answer_hit") or dense_norm >= 0.94):
                return False, "weak_triple_endpoint"
        if route in {"path", "dfs"}:
            if relation_score < 0.50:
                return False, "weak_path_relation"
            if not (detail.get("terminal_hit") or detail.get("answer_hit") or relation_hit_count >= 2):
                return False, "weak_path_endpoint"
        return True, "accepted"

    def retrieve(
        self,
        query: str,
        sorted_doc_ids: Sequence[int],
        sorted_doc_scores: Sequence[float],
        query_plan_log: Mapping[str, Any],
        dense_retrieve: Callable[[str], Tuple[np.ndarray, np.ndarray]],
        get_passage_text: Callable[[int], str],
    ) -> QueryPlanSchemaRouteResult:
        base_doc_ids = [int(doc_id) for doc_id in list(sorted_doc_ids)]
        base_scores = np.array(sorted_doc_scores, dtype=np.float32)
        log: Dict[str, Any] = {
            "enabled": True,
            "dense_top_k": self.dense_top_k,
            "max_subqueries": self.max_subqueries,
            "max_candidates": self.max_candidates,
            "min_confidence": self.min_confidence,
            "candidate_only": self.candidate_only,
            "route_probes": [],
            "added_candidates": [],
            "route_candidate_counts": {},
            "rejection_reasons": {},
            "policy": {
                "mode": self.policy_mode,
                "model_path": self.policy_model_path,
                "model_ready": self.policy.is_ready(),
                "min_score": self.policy_min_score,
                "reject_below": self.policy_reject_below,
                "score_weight": self.policy_score_weight,
                "scored": 0,
                "rejected": 0,
            },
        }
        if self.max_subqueries <= 0 or self.max_candidates <= 0:
            log["skip_reason"] = "disabled_by_zero_limit"
            return QueryPlanSchemaRouteResult(np.array(base_doc_ids, dtype=np.intp), base_scores, log)
        if not base_doc_ids:
            log["skip_reason"] = "empty_retrieval"
            return QueryPlanSchemaRouteResult(np.array(base_doc_ids, dtype=np.intp), base_scores, log)
        if not query_plan_log or not query_plan_log.get("enabled", False):
            log["skip_reason"] = "no_query_plan_log"
            return QueryPlanSchemaRouteResult(np.array(base_doc_ids, dtype=np.intp), base_scores, log)

        probes = self.build_route_probes(query, query_plan_log)
        log["route_probes"] = [
            {
                "route": probe.get("route", ""),
                "query": probe.get("query", ""),
                "anchor": probe.get("anchor", ""),
                "relation": probe.get("relation", ""),
                "terminal": probe.get("terminal", ""),
                "plan_id": probe.get("plan_id", ""),
                "relation_path": probe.get("relation_path", []),
                "source": probe.get("source", ""),
                "fact_indices": probe.get("fact_indices", []),
                "path_score": probe.get("path_score", 0.0),
                "signature_match_mode": probe.get("signature_match_mode", ""),
            }
            for probe in probes
        ]
        if not probes:
            log["skip_reason"] = "no_route_probes"
            return QueryPlanSchemaRouteResult(np.array(base_doc_ids, dtype=np.intp), base_scores, log)

        original_rank_by_doc = {int(doc_id): rank + 1 for rank, doc_id in enumerate(base_doc_ids)}
        preserved = set(base_doc_ids[: self.preserve_top_k])
        selected_by_passage: Dict[int, Dict[str, Any]] = {}
        rejection_reasons: Dict[str, int] = {}
        route_candidate_counts: Dict[str, int] = {}

        for probe in probes:
            route = str(probe.get("route") or "entity")
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
                accepted, reason = self._route_accepts(probe, detail)
                if not accepted:
                    rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
                    continue
                if confidence < self.min_confidence:
                    rejection_reasons["low_confidence"] = rejection_reasons.get("low_confidence", 0) + 1
                    continue
                route_candidate_counts[route] = route_candidate_counts.get(route, 0) + 1
                original_rank = min(
                    int(original_rank_by_doc.get(passage_idx, self.synthetic_original_rank)),
                    self.synthetic_original_rank,
                    max(self.preserve_top_k + dense_rank, 1),
                )
                candidate = {
                    "passage_idx": passage_idx,
                    "confidence": float(confidence),
                    "dense_rank": int(dense_rank),
                    "original_rank": int(original_rank),
                    "was_in_baseline": passage_idx in original_rank_by_doc,
                    "score": float(confidence + 0.01 / max(dense_rank, 1)),
                    "route": route,
                    "probe": {
                        "query": probe.get("query", ""),
                        "route": route,
                        "anchor": probe.get("anchor", ""),
                        "relation": probe.get("relation", ""),
                        "terminal": probe.get("terminal", ""),
                        "plan_id": probe.get("plan_id", ""),
                        "relation_path": probe.get("relation_path", []),
                    },
                    "matched_anchor": bool(detail["anchor_hit"]),
                    "matched_terminal": bool(detail["terminal_hit"]),
                    "relation_hits": detail["relation_hits"],
                    "answer_hit": bool(detail["answer_hit"]),
                    "contexts": [{
                        "schema_route_evidence": True,
                        "schema_route_source": probe.get("source", "schema"),
                        "route": route,
                        "support_key": f"schema_route::{route}::{probe.get('plan_id', '')}::{passage_idx}",
                        "plan_id": probe.get("plan_id", "schema_route"),
                        "anchor_entity": probe.get("anchor", ""),
                        "end_entity": probe.get("terminal", ""),
                        "relations": list(probe.get("relation_path", []) or ([probe.get("relation", "")] if probe.get("relation") else [])),
                        "fact_indices": list(probe.get("fact_indices", []) or []),
                        "fact_position": 0,
                        "signature_match_mode": probe.get("signature_match_mode") or f"schema_route_{route}",
                        "plan_effective_precision": probe.get("plan_effective_precision", 0.0),
                        "path_score": probe.get("path_score", 0.0),
                        "plan_lift_weight_scale": 0.92,
                        "score": float(confidence),
                    }],
                }
                if self.policy_mode != "off" and self.policy.is_ready():
                    policy_decision = self.policy.score(query_plan_log, candidate)
                    policy_score = self._safe_float(policy_decision.get("score"), 0.0)
                    candidate["schema_route_policy"] = policy_decision
                    for context in candidate.get("contexts", []) or []:
                        if isinstance(context, dict):
                            context["schema_route_policy_score"] = policy_score
                            context["schema_route_policy_mode"] = self.policy_mode
                    log["policy"]["scored"] += 1
                    if policy_score < self.policy_reject_below:
                        rejection_reasons["policy_reject_below"] = rejection_reasons.get("policy_reject_below", 0) + 1
                        log["policy"]["rejected"] += 1
                        continue
                    if self.policy_mode == "filter" and policy_score < self.policy_min_score:
                        rejection_reasons["policy_filter_below_min"] = rejection_reasons.get("policy_filter_below_min", 0) + 1
                        log["policy"]["rejected"] += 1
                        continue
                    adjusted_score = float(candidate.get("score", 0.0)) + self.policy_score_weight * policy_score
                    candidate["score"] = float(adjusted_score)
                    candidate["confidence"] = min(
                        1.0,
                        float(candidate.get("confidence", 0.0)) + self.policy_score_weight * max(policy_score - 0.5, 0.0) * 0.25,
                    )
                    for context in candidate.get("contexts", []) or []:
                        if isinstance(context, dict):
                            context["score"] = max(float(context.get("score", 0.0)), float(candidate["confidence"]))
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
        selected = candidates[: self.max_candidates]
        log["candidate_count"] = len(candidates)
        log["selected_count"] = len(selected)
        log["added_candidates"] = selected
        log["route_candidate_counts"] = dict(route_candidate_counts)
        log["rejection_reasons"] = dict(sorted(rejection_reasons.items(), key=lambda kv: kv[1], reverse=True)[:10])
        if not selected:
            log["skip_reason"] = "no_verified_route_candidates"
            return QueryPlanSchemaRouteResult(np.array(base_doc_ids, dtype=np.intp), base_scores, log)

        if self.candidate_only:
            score_by_doc = {
                int(doc_id): float(base_scores[idx]) if idx < len(base_scores) else 0.0
                for idx, doc_id in enumerate(base_doc_ids)
            }
            min_tail_score = float(base_scores[-1]) if len(base_scores) else 0.0
            appended = []
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
            return QueryPlanSchemaRouteResult(np.array(new_doc_ids, dtype=np.intp), new_scores, log)

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
        log["changed_top5"] = list(base_doc_ids[:5]) != final_ids[:5]
        log["changed_top2"] = list(base_doc_ids[:2]) != final_ids[:2]
        return QueryPlanSchemaRouteResult(np.array(final_ids, dtype=np.intp), final_scores, log)
