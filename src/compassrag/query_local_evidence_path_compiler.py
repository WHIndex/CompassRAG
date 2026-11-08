import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence, Tuple


@dataclass
class QueryLocalPathCompilerResult:
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    groups: List[Dict[str, Any]] = field(default_factory=list)
    log: Dict[str, Any] = field(default_factory=dict)


class QueryLocalEvidencePathCompiler:
    """Compile a logical ontology path over the query's candidate passages.

    Unlike a global predicate lookup, this operator materializes a bounded
    query-local graph and carries the passage provenance of every binding.
    A result is emitted only when all logical hops have a connected proof.
    """

    _TOKEN_ALIASES = {
        "acted": "performer",
        "actor": "performer",
        "actress": "performer",
        "author": "author",
        "birth": "birth",
        "born": "birth",
        "burial": "burial",
        "buried": "burial",
        "citizenship": "nationality",
        "compose": "composer",
        "composed": "composer",
        "composer": "composer",
        "countries": "nationality",
        "country": "nationality",
        "death": "death",
        "died": "death",
        "direct": "director",
        "directed": "director",
        "director": "director",
        "interred": "burial",
        "nation": "nationality",
        "national": "nationality",
        "nationality": "nationality",
        "perform": "performer",
        "performed": "performer",
        "performer": "performer",
        "sang": "performer",
        "singer": "performer",
        "star": "performer",
        "starred": "performer",
        "starring": "performer",
        "written": "author",
        "writer": "author",
        "wrote": "author",
    }
    _IDENTITY_STOPWORDS = {"a", "an", "of", "the"}
    _GENERIC_PREDICATES = {
        "are", "became", "has", "have", "is", "is a", "was", "was a", "were",
    }
    _DEMONYM_SUFFIXES = ("an", "ean", "ian", "ic", "ish", "ese", "i")
    _DEMONYM_EXCEPTIONS = {
        "american", "british", "dutch", "french", "greek", "swiss", "welsh",
    }

    def __init__(
        self,
        window_size: int = 200,
        beam_width: int = 12,
        max_paths: int = 8,
        max_hops: int = 4,
        min_anchor_score: float = 0.86,
        min_relation_score: float = 0.50,
        min_path_score: float = 0.58,
        allow_contextual_terminal_edges: bool = True,
        enable_entity_index_expansion: bool = False,
        enable_bound_dense_expansion: bool = False,
        entity_index_top_k: int = 64,
        bound_dense_top_k: int = 32,
        max_access_probes: int = 8,
        max_external_passages: int = 96,
        enable_relation_agnostic_bridge: bool = False,
        bridge_max_paths: int = 8,
        bridge_max_edges_per_state: int = 48,
        bridge_min_path_score: float = 0.55,
    ) -> None:
        self.window_size = max(int(window_size), 1)
        self.beam_width = max(int(beam_width), 1)
        self.max_paths = max(int(max_paths), 1)
        self.max_hops = min(max(int(max_hops), 1), 6)
        self.min_anchor_score = min(max(float(min_anchor_score), 0.0), 1.0)
        self.min_relation_score = min(max(float(min_relation_score), 0.0), 1.0)
        self.min_path_score = min(max(float(min_path_score), 0.0), 1.0)
        self.allow_contextual_terminal_edges = bool(allow_contextual_terminal_edges)
        self.enable_entity_index_expansion = bool(enable_entity_index_expansion)
        self.enable_bound_dense_expansion = bool(enable_bound_dense_expansion)
        self.entity_index_top_k = max(int(entity_index_top_k), 1)
        self.bound_dense_top_k = max(int(bound_dense_top_k), 1)
        self.max_access_probes = max(int(max_access_probes), 1)
        self.max_external_passages = max(int(max_external_passages), 0)
        self.enable_relation_agnostic_bridge = bool(enable_relation_agnostic_bridge)
        self.bridge_max_paths = max(int(bridge_max_paths), 1)
        self.bridge_max_edges_per_state = max(int(bridge_max_edges_per_state), 8)
        self.bridge_min_path_score = min(max(float(bridge_min_path_score), 0.0), 1.0)

    @staticmethod
    def _norm(value: Any) -> str:
        normalized = unicodedata.normalize("NFKD", str(value or ""))
        ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
        text = re.sub(r"[^a-z0-9]+", " ", ascii_text.lower())
        return " ".join(text.split())

    @classmethod
    def _identity_tokens(cls, value: Any) -> List[str]:
        return [
            token for token in cls._norm(value).split()
            if token not in cls._IDENTITY_STOPWORDS
        ]

    @classmethod
    def _entity_aliases(cls, value: Any) -> List[str]:
        raw = str(value or "")
        values = [raw, re.sub(r"\s*\([^)]*\)\s*", " ", raw)]
        output: List[str] = []
        for candidate in values:
            normalized = cls._norm(candidate)
            if normalized and normalized not in output:
                output.append(normalized)
            tokens = cls._identity_tokens(candidate)
            token_alias = " ".join(tokens)
            if token_alias and token_alias not in output:
                output.append(token_alias)
        return output

    @classmethod
    def _semantic_tokens(cls, value: Any) -> List[str]:
        return [
            cls._TOKEN_ALIASES.get(token, token)
            for token in cls._identity_tokens(value)
            if token not in {"by", "in", "on", "at", "from", "place"}
        ]

    @classmethod
    def _entity_match_score(cls, left: Any, right: Any) -> float:
        left_tokens = set(cls._identity_tokens(left))
        right_tokens = set(cls._identity_tokens(right))
        if not left_tokens or not right_tokens:
            return 0.0
        if left_tokens == right_tokens:
            return 1.0
        intersection = len(left_tokens.intersection(right_tokens))
        coverage = intersection / max(min(len(left_tokens), len(right_tokens)), 1)
        jaccard = intersection / max(len(left_tokens.union(right_tokens)), 1)
        return min(0.75 * coverage + 0.25 * jaccard, 1.0)

    @classmethod
    def _lexical_relation_score(cls, relation: Any, predicate: Any) -> float:
        relation_key = cls._norm(relation)
        predicate_key = cls._norm(predicate)
        if not relation_key or not predicate_key:
            return 0.0
        if relation_key == predicate_key:
            return 1.0
        left = set(cls._semantic_tokens(relation))
        right = set(cls._semantic_tokens(predicate))
        if not left or not right:
            return 0.0
        overlap = len(left.intersection(right))
        if not overlap:
            return 0.0
        coverage = overlap / max(len(left), 1)
        precision = overlap / max(len(right), 1)
        return min(0.70 * coverage + 0.30 * precision, 1.0)

    @classmethod
    def _looks_like_nationality(cls, value: Any) -> bool:
        normalized = cls._norm(value)
        tokens = normalized.split()
        if not tokens or len(tokens) > 5 or any(token.isdigit() for token in tokens):
            return False
        head = tokens[0]
        return head in cls._DEMONYM_EXCEPTIONS or head.endswith(cls._DEMONYM_SUFFIXES)

    def _contextual_terminal_score(
        self,
        relation: Any,
        predicate: Any,
        terminal_entity: Any,
    ) -> float:
        if not self.allow_contextual_terminal_edges:
            return 0.0
        predicate_key = self._norm(predicate)
        if predicate_key not in self._GENERIC_PREDICATES:
            return 0.0
        relation_tokens = set(self._semantic_tokens(relation))
        if "nationality" in relation_tokens and self._looks_like_nationality(terminal_entity):
            return 0.68
        if "occupation" in relation_tokens:
            terminal_tokens = self._identity_tokens(terminal_entity)
            if terminal_tokens and not any(token.isdigit() for token in terminal_tokens):
                return 0.62
        return 0.0

    @staticmethod
    def _deduplicate(values: Iterable[int]) -> List[int]:
        output: List[int] = []
        seen = set()
        for value in values:
            value = int(value)
            if value in seen:
                continue
            seen.add(value)
            output.append(value)
        return output

    @classmethod
    def _edge_key(
        cls,
        passage_idx: Any,
        subject: Any,
        predicate: Any,
        obj: Any,
    ) -> Tuple[int, str, str, str]:
        return (
            int(passage_idx),
            cls._norm(subject),
            cls._norm(predicate),
            cls._norm(obj),
        )

    def _append_edge(
        self,
        edges: List[Dict[str, Any]],
        endpoint_index: Dict[str, List[int]],
        edge_keys: Dict[Tuple[int, str, str, str], int],
        *,
        passage_idx: Any,
        candidate_rank: Any,
        subject: Any,
        predicate: Any,
        obj: Any,
        access_source: str,
        access_score: float = 0.0,
    ) -> bool:
        try:
            passage_idx = int(passage_idx)
            candidate_rank = max(int(candidate_rank), 1)
        except (TypeError, ValueError):
            return False
        subject = str(subject or "").strip()
        predicate = str(predicate or "").strip()
        obj = str(obj or "").strip()
        if passage_idx < 0 or not subject or not predicate or not obj:
            return False
        key = self._edge_key(passage_idx, subject, predicate, obj)
        if key in edge_keys:
            edge = edges[edge_keys[key]]
            sources = list(edge.get("access_sources", []) or [])
            if access_source and access_source not in sources:
                sources.append(access_source)
            edge["access_sources"] = sources
            edge["access_score"] = max(
                float(edge.get("access_score", 0.0) or 0.0),
                float(access_score or 0.0),
            )
            edge["candidate_rank"] = min(
                int(edge.get("candidate_rank", candidate_rank) or candidate_rank),
                candidate_rank,
            )
            return False
        edge_idx = len(edges)
        edge_keys[key] = edge_idx
        edges.append({
            "passage_idx": passage_idx,
            "candidate_rank": candidate_rank,
            "subject": subject,
            "predicate": predicate,
            "object": obj,
            "triple": [subject, predicate, obj],
            "access_sources": [access_source] if access_source else [],
            "access_score": min(max(float(access_score or 0.0), 0.0), 1.0),
        })
        for endpoint in (subject, obj):
            for alias in self._entity_aliases(endpoint):
                endpoint_index[alias].append(edge_idx)
        return True

    def _append_openie_record(
        self,
        edges: List[Dict[str, Any]],
        endpoint_index: Dict[str, List[int]],
        edge_keys: Dict[Tuple[int, str, str, str], int],
        *,
        passage_idx: int,
        candidate_rank: int,
        record: Mapping[str, Any],
        access_source: str,
        access_score: float = 0.0,
    ) -> int:
        added = 0
        for triple in record.get("extracted_triples", []) or []:
            if not isinstance(triple, (list, tuple)) or len(triple) != 3:
                continue
            added += int(self._append_edge(
                edges,
                endpoint_index,
                edge_keys,
                passage_idx=passage_idx,
                candidate_rank=candidate_rank,
                subject=triple[0],
                predicate=triple[1],
                obj=triple[2],
                access_source=access_source,
                access_score=access_score,
            ))
        return added

    def _build_local_graph(
        self,
        base_doc_ids: Sequence[int],
        get_openie_record: Callable[[int], Mapping[str, Any]],
    ) -> Tuple[
        List[Dict[str, Any]],
        Dict[str, List[int]],
        Dict[Tuple[int, str, str, str], int],
        Dict[int, int],
        Dict[str, int],
    ]:
        edges: List[Dict[str, Any]] = []
        endpoint_index: Dict[str, List[int]] = defaultdict(list)
        edge_keys: Dict[Tuple[int, str, str, str], int] = {}
        doc_ids = self._deduplicate(base_doc_ids)[: self.window_size]
        rank_by_doc = {
            int(passage_idx): rank
            for rank, passage_idx in enumerate(doc_ids, start=1)
        }
        for rank, passage_idx in enumerate(doc_ids, start=1):
            record = get_openie_record(passage_idx) or {}
            self._append_openie_record(
                edges,
                endpoint_index,
                edge_keys,
                passage_idx=int(passage_idx),
                candidate_rank=rank,
                record=record,
                access_source="semantic_candidate_window",
            )
        return edges, endpoint_index, edge_keys, rank_by_doc, {
            "window_passage_count": len(doc_ids),
            "window_triple_count": len(edges),
            "window_entity_key_count": len(endpoint_index),
        }

    def _expand_binding_access(
        self,
        *,
        query: str,
        binding: str,
        relation: str,
        edges: List[Dict[str, Any]],
        endpoint_index: Dict[str, List[int]],
        edge_keys: Dict[Tuple[int, str, str, str], int],
        rank_by_doc: Mapping[int, int],
        get_openie_record: Callable[[int], Mapping[str, Any]],
        relation_realizations: Mapping[str, Sequence[Any]],
        lookup_entity_relations: Callable[[str, int], Sequence[Mapping[str, Any]]] | None,
        dense_retrieve: Callable[[str], Tuple[Sequence[int], Sequence[float]]] | None,
        access_state: Dict[str, Any],
        rejections: Counter,
    ) -> None:
        expansion_key = (self._norm(binding), self._norm(relation))
        if not all(expansion_key) or expansion_key in access_state["expanded_bindings"]:
            return
        access_state["expanded_bindings"].add(expansion_key)

        def admit_passage(passage_idx: Any) -> Tuple[bool, int]:
            try:
                passage_idx = int(passage_idx)
            except (TypeError, ValueError):
                return False, self.window_size + self.max_external_passages + 1
            if passage_idx < 0:
                return False, self.window_size + self.max_external_passages + 1
            if passage_idx in rank_by_doc:
                return True, int(rank_by_doc[passage_idx])
            external_rank_by_doc = access_state["external_rank_by_doc"]
            if passage_idx not in external_rank_by_doc:
                if len(external_rank_by_doc) >= self.max_external_passages:
                    rejections["external_passage_budget_exhausted"] += 1
                    return False, self.window_size + self.max_external_passages + 1
                external_rank_by_doc[passage_idx] = (
                    self.window_size + len(external_rank_by_doc) + 1
                )
            return True, int(external_rank_by_doc[passage_idx])

        if self.enable_entity_index_expansion and lookup_entity_relations is not None:
            access_state["entity_index_probe_count"] += 1
            try:
                rows = list(
                    lookup_entity_relations(binding, self.entity_index_top_k) or []
                )[: self.entity_index_top_k]
            except Exception:
                rows = []
                rejections["entity_index_probe_failed"] += 1
            added = 0
            for row in rows:
                if not isinstance(row, Mapping):
                    continue
                allowed, candidate_rank = admit_passage(row.get("passage_idx"))
                if not allowed:
                    continue
                added += int(self._append_edge(
                    edges,
                    endpoint_index,
                    edge_keys,
                    passage_idx=row.get("passage_idx"),
                    candidate_rank=candidate_rank,
                    subject=row.get("subject"),
                    predicate=row.get("predicate"),
                    obj=row.get("object"),
                    access_source="entity_relation_index",
                    access_score=float(row.get("entity_match_score", 1.0) or 0.0),
                ))
            access_state["entity_index_edge_count"] += added
            if not added:
                rejections["entity_index_no_new_edge"] += 1

        if not self.enable_bound_dense_expansion or dense_retrieve is None:
            return
        if access_state["bound_dense_probe_count"] >= self.max_access_probes:
            rejections["bound_dense_probe_budget_exhausted"] += 1
            return
        surfaces = self._surfaces_for_relation(relation, relation_realizations)
        surface = str(surfaces[0] if surfaces else relation).strip()
        probe = " ".join(value for value in (binding, surface) if value).strip()
        if not probe:
            return
        access_state["bound_dense_probe_count"] += 1
        access_state["bound_dense_probes"].append({
            "binding": binding,
            "relation": relation,
            "query": probe,
        })
        try:
            dense_ids, dense_scores = dense_retrieve(probe)
        except Exception:
            rejections["bound_dense_probe_failed"] += 1
            return
        dense_ids = list(dense_ids)[: self.bound_dense_top_k] if dense_ids is not None else []
        dense_scores = list(dense_scores)[: len(dense_ids)] if dense_scores is not None else []
        added = 0
        for dense_rank, passage_idx in enumerate(dense_ids, start=1):
            allowed, candidate_rank = admit_passage(passage_idx)
            if not allowed:
                continue
            try:
                record = get_openie_record(int(passage_idx)) or {}
            except Exception:
                record = {}
            access_score = 1.0 / math.log2(float(dense_rank) + 1.0)
            if dense_rank - 1 < len(dense_scores):
                try:
                    access_score = max(access_score, float(dense_scores[dense_rank - 1]))
                except (TypeError, ValueError):
                    pass
            added += self._append_openie_record(
                edges,
                endpoint_index,
                edge_keys,
                passage_idx=int(passage_idx),
                candidate_rank=candidate_rank,
                record=record,
                access_source="bound_dense_openie",
                access_score=access_score,
            )
        access_state["bound_dense_edge_count"] += added
        if not added:
            rejections["bound_dense_no_new_openie_edge"] += 1

    def _resolve_anchor_bindings(
        self,
        anchor: str,
        edges: Sequence[Mapping[str, Any]],
        endpoint_index: Mapping[str, Sequence[int]],
    ) -> List[Dict[str, Any]]:
        exact_edge_ids = set()
        for alias in self._entity_aliases(anchor):
            exact_edge_ids.update(int(value) for value in endpoint_index.get(alias, []) or [])
        bindings: Dict[str, Dict[str, Any]] = {}
        for edge_idx in exact_edge_ids:
            edge = edges[edge_idx]
            for endpoint in (edge["subject"], edge["object"]):
                if set(self._entity_aliases(anchor)).intersection(self._entity_aliases(endpoint)):
                    bindings[self._norm(endpoint)] = {
                        "binding": endpoint,
                        "anchor_score": 1.0,
                        "exact_anchor": True,
                    }
        if bindings:
            return list(bindings.values())[: self.beam_width]

        anchor_tokens = self._identity_tokens(anchor)
        if not anchor_tokens:
            return []
        endpoints = {
            str(edge[field])
            for edge in edges
            for field in ("subject", "object")
        }
        ranked: List[Tuple[float, str]] = []
        for endpoint in endpoints:
            score = self._entity_match_score(anchor, endpoint)
            if score >= self.min_anchor_score:
                ranked.append((score, endpoint))
        ranked.sort(key=lambda item: (item[0], -len(item[1])), reverse=True)
        return [
            {"binding": endpoint, "anchor_score": round(score, 6), "exact_anchor": False}
            for score, endpoint in ranked[: self.beam_width]
        ]

    @classmethod
    def _direction_for_relation(
        cls,
        relation: str,
        relation_directions: Mapping[str, Any],
        predicate: str = "",
    ) -> str:
        relation_key = cls._norm(relation)
        for key, value in relation_directions.items():
            if cls._norm(key) == relation_key:
                if isinstance(value, Mapping):
                    predicate_key = cls._norm(predicate)
                    predicate_directions = dict(value.get("predicates", {}) or {})
                    direction = next(
                        (
                            str(candidate or "either").lower()
                            for candidate_predicate, candidate in predicate_directions.items()
                            if cls._norm(candidate_predicate) == predicate_key
                        ),
                        str(value.get("default", "either") or "either").lower(),
                    )
                else:
                    direction = str(value or "either").lower()
                return direction if direction in {"forward", "reverse", "either"} else "either"
        return "either"

    @classmethod
    def _predicate_role_compatible(cls, relation: Any, predicate: Any) -> bool:
        relation_tokens = set(cls._semantic_tokens(relation))
        predicate_tokens = set(cls._semantic_tokens(predicate))
        predicate_norm = cls._norm(predicate)
        if "father" in relation_tokens or "father" in cls._norm(relation).split():
            return bool({"father", "paternal"} & set(predicate_norm.split()))
        if "mother" in relation_tokens or "mother" in cls._norm(relation).split():
            return bool({"mother", "maternal"} & set(predicate_norm.split()))
        if "spouse" in relation_tokens or "spouse" in cls._norm(relation).split():
            return bool({"spouse", "wife", "husband", "married"} & set(predicate_norm.split()))
        return bool(predicate_tokens) or bool(predicate_norm)

    @classmethod
    def _surfaces_for_relation(
        cls,
        relation: str,
        relation_realizations: Mapping[str, Sequence[Any]],
    ) -> List[str]:
        relation_key = cls._norm(relation)
        surfaces = [relation]
        for key, values in relation_realizations.items():
            if cls._norm(key) != relation_key:
                continue
            surfaces.extend(str(value or "") for value in values or [])
        output: List[str] = []
        seen = set()
        for surface in surfaces:
            normalized = cls._norm(surface)
            if (
                normalized
                and normalized not in seen
                and cls._predicate_role_compatible(relation, surface)
            ):
                seen.add(normalized)
                output.append(surface)
        return output

    def _traverse(
        self,
        state: Mapping[str, Any],
        query: str,
        relation: str,
        terminal_step: bool,
        answer_type: str,
        edges: List[Dict[str, Any]],
        endpoint_index: Dict[str, List[int]],
        edge_keys: Dict[Tuple[int, str, str, str], int],
        rank_by_doc: Mapping[int, int],
        get_openie_record: Callable[[int], Mapping[str, Any]],
        relation_realizations: Mapping[str, Sequence[Any]],
        relation_directions: Mapping[str, Any],
        terminal_type_score: Callable[[str, str], float],
        lookup_entity_relations: Callable[[str, int], Sequence[Mapping[str, Any]]] | None,
        dense_retrieve: Callable[[str], Tuple[Sequence[int], Sequence[float]]] | None,
        access_state: Dict[str, Any],
        rejections: Counter,
    ) -> List[Dict[str, Any]]:
        binding = str(state.get("binding", "") or "")
        self._expand_binding_access(
            query=query,
            binding=binding,
            relation=relation,
            edges=edges,
            endpoint_index=endpoint_index,
            edge_keys=edge_keys,
            rank_by_doc=rank_by_doc,
            get_openie_record=get_openie_record,
            relation_realizations=relation_realizations,
            lookup_entity_relations=lookup_entity_relations,
            dense_retrieve=dense_retrieve,
            access_state=access_state,
            rejections=rejections,
        )
        edge_ids = set()
        for alias in self._entity_aliases(binding):
            edge_ids.update(int(value) for value in endpoint_index.get(alias, []) or [])
        if not edge_ids:
            rejections["binding_not_in_candidate_graph"] += 1
            return []
        surfaces = self._surfaces_for_relation(relation, relation_realizations)
        output: List[Dict[str, Any]] = []
        for edge_idx in edge_ids:
            edge = edges[edge_idx]
            if not self._predicate_role_compatible(relation, edge["predicate"]):
                rejections["predicate_role_mismatch"] += 1
                continue
            direction = self._direction_for_relation(
                relation,
                relation_directions,
                str(edge["predicate"]),
            )
            subject_match = bool(
                set(self._entity_aliases(binding)).intersection(
                    self._entity_aliases(edge["subject"])
                )
            )
            object_match = bool(
                set(self._entity_aliases(binding)).intersection(
                    self._entity_aliases(edge["object"])
                )
            )
            traversals: List[Tuple[str, str]] = []
            if subject_match and direction != "reverse":
                traversals.append((str(edge["object"]), "forward"))
            if object_match and direction != "forward":
                traversals.append((str(edge["subject"]), "reverse"))
            for terminal_entity, access_direction in traversals:
                relation_score = max(
                    (self._lexical_relation_score(surface, edge["predicate"]) for surface in surfaces),
                    default=0.0,
                )
                contextual_relation = False
                if terminal_step and relation_score < self.min_relation_score:
                    contextual = self._contextual_terminal_score(
                        relation,
                        edge["predicate"],
                        terminal_entity,
                    )
                    if contextual > relation_score:
                        relation_score = contextual
                        contextual_relation = True
                if relation_score < self.min_relation_score:
                    rejections["relation_mismatch"] += 1
                    continue
                type_score = (
                    min(max(float(terminal_type_score(terminal_entity, answer_type)), 0.0), 1.0)
                    if terminal_step else 1.0
                )
                if terminal_step and type_score <= 0.0:
                    rejections["terminal_type_mismatch"] += 1
                    continue
                rank_bonus = 1.0 / math.log2(float(edge["candidate_rank"]) + 1.0)
                step_score = (
                    0.64 * relation_score
                    + 0.16 * float(state.get("anchor_score", 1.0) or 0.0)
                    + 0.12 * type_score
                    + 0.08 * rank_bonus
                )
                previous_bindings = {self._norm(value) for value in state.get("bindings", []) or []}
                if self._norm(terminal_entity) in previous_bindings:
                    rejections["cyclic_binding"] += 1
                    continue
                output.append({
                    "binding": terminal_entity,
                    "bindings": list(state.get("bindings", []) or []) + [terminal_entity],
                    "passage_indices": list(state.get("passage_indices", []) or []) + [int(edge["passage_idx"])],
                    "triples": list(state.get("triples", []) or []) + [list(edge["triple"])],
                    "step_scores": list(state.get("step_scores", []) or []) + [float(step_score)],
                    "relation_scores": list(state.get("relation_scores", []) or []) + [float(relation_score)],
                    "type_scores": list(state.get("type_scores", []) or []) + [float(type_score)],
                    "candidate_ranks": list(state.get("candidate_ranks", []) or []) + [int(edge["candidate_rank"])],
                    "access_methods": list(state.get("access_methods", []) or []) + [
                        list(edge.get("access_sources", []) or [])
                    ],
                    "access_scores": list(state.get("access_scores", []) or []) + [
                        float(edge.get("access_score", 0.0) or 0.0)
                    ],
                    "access_directions": list(state.get("access_directions", []) or []) + [access_direction],
                    "expected_directions": list(state.get("expected_directions", []) or []) + [direction],
                    "contextual_relation_count": int(state.get("contextual_relation_count", 0) or 0) + int(contextual_relation),
                    "anchor_score": float(state.get("anchor_score", 1.0) or 0.0),
                    "exact_anchor": bool(state.get("exact_anchor", False)),
                })
        return output

    def _execute_anchor(
        self,
        query: str,
        anchor: str,
        relations: Sequence[str],
        answer_type: str,
        edges: List[Dict[str, Any]],
        endpoint_index: Dict[str, List[int]],
        edge_keys: Dict[Tuple[int, str, str, str], int],
        rank_by_doc: Mapping[int, int],
        get_openie_record: Callable[[int], Mapping[str, Any]],
        relation_realizations: Mapping[str, Sequence[Any]],
        relation_directions: Mapping[str, Any],
        terminal_type_score: Callable[[str, str], float],
        lookup_entity_relations: Callable[[str, int], Sequence[Mapping[str, Any]]] | None,
        dense_retrieve: Callable[[str], Tuple[Sequence[int], Sequence[float]]] | None,
        access_state: Dict[str, Any],
        rejections: Counter,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        anchor_bindings = self._resolve_anchor_bindings(anchor, edges, endpoint_index)
        states = [
            {
                **binding,
                "bindings": [anchor, binding["binding"]] if self._norm(anchor) != self._norm(binding["binding"]) else [anchor],
                "passage_indices": [],
                "triples": [],
                "step_scores": [],
                "relation_scores": [],
                "type_scores": [],
                "candidate_ranks": [],
                "access_methods": [],
                "access_scores": [],
                "access_directions": [],
                "expected_directions": [],
                "contextual_relation_count": 0,
            }
            for binding in anchor_bindings
        ]
        if not states:
            rejections["anchor_not_in_candidate_graph"] += 1
            return [], anchor_bindings
        for step_idx, relation in enumerate(relations[: self.max_hops]):
            next_states: List[Dict[str, Any]] = []
            terminal_step = step_idx == len(relations[: self.max_hops]) - 1
            for state in states:
                next_states.extend(self._traverse(
                    state=state,
                    query=query,
                    relation=relation,
                    terminal_step=terminal_step,
                    answer_type=answer_type,
                    edges=edges,
                    endpoint_index=endpoint_index,
                    edge_keys=edge_keys,
                    rank_by_doc=rank_by_doc,
                    get_openie_record=get_openie_record,
                    relation_realizations=relation_realizations,
                    relation_directions=relation_directions,
                    terminal_type_score=terminal_type_score,
                    lookup_entity_relations=lookup_entity_relations,
                    dense_retrieve=dense_retrieve,
                    access_state=access_state,
                    rejections=rejections,
                ))
            if not next_states:
                rejections[f"path_failed_at_hop_{step_idx}"] += 1
                return [], anchor_bindings
            best_by_binding: Dict[str, Dict[str, Any]] = {}
            for state in next_states:
                key = self._norm(state["binding"])
                score = sum(state["step_scores"]) / max(len(state["step_scores"]), 1)
                if key not in best_by_binding or score > sum(best_by_binding[key]["step_scores"]) / max(len(best_by_binding[key]["step_scores"]), 1):
                    best_by_binding[key] = state
            states = sorted(
                best_by_binding.values(),
                key=lambda item: sum(item["step_scores"]) / max(len(item["step_scores"]), 1),
                reverse=True,
            )[: self.beam_width]
        return states, anchor_bindings

    def _run_relation_agnostic_bridge(
        self,
        query: str,
        base_doc_ids: Sequence[int],
        anchors: Sequence[str],
        answer_type: str,
        get_openie_record: Callable[[int], Mapping[str, Any]],
        terminal_type_score: Callable[[str, str], float],
    ) -> QueryLocalPathCompilerResult:
        """Discover a grounded bridge when the ontology relation path is incomplete.

        This is deliberately a bounded, evidence-first fallback.  It never
        invents a predicate or treats a schema match as proof: every hop is an
        observed OpenIE edge, every next binding is the endpoint of the prior
        edge, and the emitted slate carries all passage provenance.  The
        relation-agnostic name means only that the logical relation label is
        unavailable; it does not mean the evidence graph is unconstrained.
        """
        log: Dict[str, Any] = {
            "enabled": True,
            "operator": "query_local_relation_agnostic_bridge",
            "query": str(query or ""),
            "anchors": list(anchors),
            "answer_type": str(answer_type or ""),
            "candidate_count": 0,
            "group_count": 0,
            "complete_path_count": 0,
            "skip_reason": "",
        }
        if not anchors:
            log["skip_reason"] = "no_anchor_entities"
            return QueryLocalPathCompilerResult(log=log)

        edges, endpoint_index, _edge_keys, rank_by_doc, graph_stats = self._build_local_graph(
            base_doc_ids,
            get_openie_record,
        )
        log.update(graph_stats)
        if not edges:
            log["skip_reason"] = "no_local_edges"
            return QueryLocalPathCompilerResult(log=log)

        def edge_ids_for(binding: str) -> List[int]:
            ids = set()
            for alias in self._entity_aliases(binding):
                ids.update(int(value) for value in endpoint_index.get(alias, []) or [])
            return sorted(
                ids,
                key=lambda idx: (
                    int(edges[idx].get("candidate_rank", self.window_size + 1)),
                    -float(edges[idx].get("access_score", 0.0) or 0.0),
                ),
            )[: self.bridge_max_edges_per_state]

        def query_relation_signal(predicate: str) -> float:
            query_tokens = set(self._semantic_tokens(query))
            predicate_tokens = set(self._semantic_tokens(predicate))
            if not query_tokens or not predicate_tokens:
                return 0.0
            return len(query_tokens.intersection(predicate_tokens)) / max(len(predicate_tokens), 1)

        states: List[Dict[str, Any]] = []
        seen_initial = set()
        for anchor in anchors[:2]:
            for binding in self._resolve_anchor_bindings(anchor, edges, endpoint_index):
                key = self._norm(binding.get("binding", ""))
                if not key or key in seen_initial:
                    continue
                seen_initial.add(key)
                states.append({
                    "binding": str(binding["binding"]),
                    "bindings": [str(anchor), str(binding["binding"])],
                    "passage_indices": [],
                    "triples": [],
                    "step_scores": [],
                    "relation_scores": [],
                    "type_scores": [],
                    "candidate_ranks": [],
                    "access_methods": [],
                    "access_directions": [],
                    "expected_directions": [],
                    "anchor_score": float(binding.get("anchor_score", 1.0) or 0.0),
                    "exact_anchor": bool(binding.get("exact_anchor", False)),
                })
        if not states:
            log["skip_reason"] = "anchor_not_in_candidate_graph"
            return QueryLocalPathCompilerResult(log=log)

        complete: List[Dict[str, Any]] = []
        max_hops = min(self.max_hops, 4)
        for hop in range(max_hops):
            next_states: List[Dict[str, Any]] = []
            for state in states:
                previous_bindings = {
                    self._norm(value) for value in state.get("bindings", []) or []
                }
                for edge_idx in edge_ids_for(str(state["binding"])):
                    edge = edges[edge_idx]
                    subject_match = bool(set(self._entity_aliases(state["binding"])).intersection(
                        self._entity_aliases(edge["subject"])
                    ))
                    object_match = bool(set(self._entity_aliases(state["binding"])).intersection(
                        self._entity_aliases(edge["object"])
                    ))
                    traversals = []
                    if subject_match:
                        traversals.append((str(edge["object"]), "forward"))
                    if object_match:
                        traversals.append((str(edge["subject"]), "reverse"))
                    for next_binding, direction in traversals:
                        next_key = self._norm(next_binding)
                        if not next_key or next_key in previous_bindings:
                            continue
                        passage_indices = list(state.get("passage_indices", [])) + [int(edge["passage_idx"])]
                        bindings = list(state.get("bindings", [])) + [next_binding]
                        relation_signal = query_relation_signal(str(edge["predicate"]))
                        rank_bonus = 1.0 / math.log2(float(edge["candidate_rank"]) + 1.0)
                        terminal_score = min(max(float(terminal_type_score(next_binding, answer_type)), 0.0), 1.0)
                        step_score = 0.46 * rank_bonus + 0.24 * relation_signal + 0.30 * terminal_score
                        child = {
                            "binding": next_binding,
                            "bindings": bindings,
                            "passage_indices": passage_indices,
                            "triples": list(state.get("triples", [])) + [list(edge["triple"])],
                            "step_scores": list(state.get("step_scores", [])) + [step_score],
                            "relation_scores": list(state.get("relation_scores", [])) + [relation_signal],
                            "type_scores": list(state.get("type_scores", [])) + [terminal_score],
                            "candidate_ranks": list(state.get("candidate_ranks", [])) + [int(edge["candidate_rank"])],
                            "access_methods": list(state.get("access_methods", [])) + [list(edge.get("access_sources", []))],
                            "access_directions": list(state.get("access_directions", [])) + [direction],
                            "expected_directions": list(state.get("expected_directions", [])) + ["unknown"],
                            "anchor_score": float(state.get("anchor_score", 1.0) or 0.0),
                            "exact_anchor": bool(state.get("exact_anchor", False)),
                        }
                        unique_passages = list(dict.fromkeys(passage_indices))
                        mean_score = sum(child["step_scores"]) / max(len(child["step_scores"]), 1)


                        if len(unique_passages) >= 2 and len(child["triples"]) >= 2 and terminal_score > 0.0:
                            child["bridge_score"] = mean_score
                            complete.append(child)
                        next_states.append(child)
            if not next_states:
                break
            next_states.sort(
                key=lambda item: (
                    sum(item["step_scores"]) / max(len(item["step_scores"]), 1),
                    -max(item["candidate_ranks"], default=self.window_size + 1),
                ),
                reverse=True,
            )
            states = next_states[: self.beam_width]

        complete.sort(key=lambda item: float(item.get("bridge_score", 0.0)), reverse=True)
        candidates: List[Dict[str, Any]] = []
        groups: List[Dict[str, Any]] = []
        seen_paths = set()
        for state_idx, state in enumerate(complete[: self.bridge_max_paths]):
            passage_indices = list(dict.fromkeys(int(value) for value in state["passage_indices"]))
            if len(passage_indices) < 2 or tuple(passage_indices) in seen_paths:
                continue
            seen_paths.add(tuple(passage_indices))
            score = float(state.get("bridge_score", 0.0) or 0.0)
            if score < self.bridge_min_path_score:
                continue
            group_id = f"query_local_relation_agnostic_bridge::{state_idx}"
            group = {
                "group_id": group_id,
                "operator": "query_local_relation_agnostic_bridge",
                "passage_indices": passage_indices,
                "terminal_passage_indices": passage_indices[-1:],
                "bindings": list(state["bindings"]),
                "relations": [str(triple[1]) for triple in state["triples"]],
                "triples": list(state["triples"]),
                "score": round(score, 6),
                "utility": round(score, 6),
                "estimated_cost": round(float(len(state["triples"])), 6),
                "chain_completeness": 1.0,
                "proof_confidence": round(score, 6),
                "proof_relation_coverage": 1.0,
                "proof_type_consistency": round(min(state["type_scores"]), 6),
                "proof_provenance_coverage": 1.0,
                "provenance_validated": True,
                "strict_chain_certificate": False,
                "bridge_chain_certificate": True,
                "query_local_bridge": True,
                "relation_contract_mode": "observed_bridge",
                "schema_to_evidence_grounding": True,
                "physical_access_methods": ["semantic_candidate_window"],
                "mixed_access_path": False,
                "exact_anchor": bool(state["exact_anchor"]),
                "candidate_ranks": list(state["candidate_ranks"]),
                "access_directions": list(state["access_directions"]),
            }
            groups.append(group)
            candidates.append({
                "passage_idx": int(passage_indices[-1]),
                "operator": "query_local_relation_agnostic_bridge",
                "group_id": group_id,
                "anchor_entity": list(anchors[:1]),
                "terminal_entity": str(state["binding"]),
                "relations": group["relations"],
                "answer_type": answer_type,
                "relation_score": round(min(state["relation_scores"]), 6),
                "type_score": round(min(state["type_scores"]), 6),
                "raw_score": round(score, 6),
                "utility": round(score, 6),
                "estimated_cost": float(len(state["triples"])),
                "exact_entity_key": bool(state["exact_anchor"]),
                "provenance_validated": True,
                "query_local_bridge": True,
                "bridge_chain_certificate": True,
                "contextual_relation_count": 0,
                "triple": list(state["triples"][-1]),
            })
        log.update({
            "candidate_count": len(candidates),
            "group_count": len(groups),
            "complete_path_count": len(complete),
            "bridge_hop_count": max((len(item.get("triples", [])) for item in complete), default=0),
            "skip_reason": "" if candidates else "no_relation_agnostic_bridge",
        })
        return QueryLocalPathCompilerResult(candidates=candidates, groups=groups, log=log)

    def run(
        self,
        query: str,
        base_doc_ids: Sequence[int],
        anchors: Sequence[str],
        relations: Sequence[str],
        answer_type: str,
        comparison: bool,
        get_openie_record: Callable[[int], Mapping[str, Any]],
        relation_realizations: Mapping[str, Sequence[Any]] = None,
        relation_directions: Mapping[str, Any] = None,
        terminal_type_score: Callable[[str, str], float] = None,
        lookup_entity_relations: Callable[[str, int], Sequence[Mapping[str, Any]]] | None = None,
        dense_retrieve: Callable[[str], Tuple[Sequence[int], Sequence[float]]] | None = None,
    ) -> QueryLocalPathCompilerResult:
        relation_path = [str(value or "") for value in relations[: self.max_hops] if str(value or "").strip()]
        log: Dict[str, Any] = {
            "enabled": True,
            "operator": "candidate_window_provenance_join",
            "query": str(query or ""),
            "anchors": list(anchors),
            "relations": relation_path,
            "comparison": bool(comparison),
            "candidate_count": 0,
            "group_count": 0,
            "skip_reason": "",
        }
        if not anchors:
            log["skip_reason"] = "no_anchor_entities"
            return QueryLocalPathCompilerResult(log=log)
        if not relation_path:
            if self.enable_relation_agnostic_bridge:
                return self._run_relation_agnostic_bridge(
                    query=str(query or ""),
                    base_doc_ids=base_doc_ids,
                    anchors=anchors,
                    answer_type=answer_type,
                    get_openie_record=get_openie_record,
                    terminal_type_score=(
                        terminal_type_score or (lambda _entity, _answer_type: 1.0)
                    ),
                )
            log["skip_reason"] = "no_relation_path"
            return QueryLocalPathCompilerResult(log=log)

        edges, endpoint_index, edge_keys, rank_by_doc, graph_stats = self._build_local_graph(
            base_doc_ids,
            get_openie_record,
        )
        log.update(graph_stats)
        relation_realizations = dict(relation_realizations or {})
        relation_directions = dict(relation_directions or {})
        terminal_type_score = terminal_type_score or (lambda _entity, _answer_type: 1.0)
        rejections: Counter = Counter()
        access_state: Dict[str, Any] = {
            "expanded_bindings": set(),
            "external_rank_by_doc": {},
            "entity_index_probe_count": 0,
            "entity_index_edge_count": 0,
            "bound_dense_probe_count": 0,
            "bound_dense_edge_count": 0,
            "bound_dense_probes": [],
        }
        for anchor in anchors[:2 if comparison else len(anchors)]:
            self._expand_binding_access(
                query=str(query or ""),
                binding=str(anchor or ""),
                relation=relation_path[0],
                edges=edges,
                endpoint_index=endpoint_index,
                edge_keys=edge_keys,
                rank_by_doc=rank_by_doc,
                get_openie_record=get_openie_record,
                relation_realizations=relation_realizations,
                lookup_entity_relations=lookup_entity_relations,
                dense_retrieve=dense_retrieve,
                access_state=access_state,
                rejections=rejections,
            )
        if not edges:
            log["skip_reason"] = "all_access_paths_have_no_openie_edges"
            log["rejection_reasons"] = dict(rejections)
            return QueryLocalPathCompilerResult(log=log)

        anchor_states: List[List[Dict[str, Any]]] = []
        anchor_profiles: List[Dict[str, Any]] = []
        branch_answer_type = "entity" if comparison or self._norm(answer_type) == "boolean" else answer_type
        for anchor in anchors[:2 if comparison else len(anchors)]:
            states, resolved = self._execute_anchor(
                query=str(query or ""),
                anchor=str(anchor),
                relations=relation_path,
                answer_type=branch_answer_type,
                edges=edges,
                endpoint_index=endpoint_index,
                edge_keys=edge_keys,
                rank_by_doc=rank_by_doc,
                get_openie_record=get_openie_record,
                relation_realizations=relation_realizations,
                relation_directions=relation_directions,
                terminal_type_score=terminal_type_score,
                lookup_entity_relations=lookup_entity_relations,
                dense_retrieve=dense_retrieve,
                access_state=access_state,
                rejections=rejections,
            )
            anchor_states.append(states)
            anchor_profiles.append({
                "anchor": str(anchor),
                "resolved_bindings": resolved,
                "complete_path_count": len(states),
            })
        log["anchor_profiles"] = anchor_profiles

        complete_states: List[Dict[str, Any]] = []
        if comparison:
            if len(anchor_states) != 2 or not all(anchor_states):
                rejections["comparison_branch_incomplete"] += 1
            else:
                for left in anchor_states[0][: self.beam_width]:
                    for right in anchor_states[1][: self.beam_width]:
                        complete_states.append({
                            "binding": [left["binding"], right["binding"]],
                            "bindings": [left["bindings"], right["bindings"]],
                            "passage_indices": list(left["passage_indices"]) + list(right["passage_indices"]),
                            "triples": list(left["triples"]) + list(right["triples"]),
                            "step_scores": list(left["step_scores"]) + list(right["step_scores"]),
                            "relation_scores": list(left["relation_scores"]) + list(right["relation_scores"]),
                            "type_scores": list(left["type_scores"]) + list(right["type_scores"]),
                            "candidate_ranks": list(left["candidate_ranks"]) + list(right["candidate_ranks"]),
                            "access_methods": list(left["access_methods"]) + list(right["access_methods"]),
                            "access_scores": list(left["access_scores"]) + list(right["access_scores"]),
                            "access_directions": [left["access_directions"], right["access_directions"]],
                            "expected_directions": [left["expected_directions"], right["expected_directions"]],
                            "contextual_relation_count": int(left["contextual_relation_count"]) + int(right["contextual_relation_count"]),
                            "anchor_score": min(float(left["anchor_score"]), float(right["anchor_score"])),
                            "exact_anchor": bool(left["exact_anchor"] and right["exact_anchor"]),
                            "branch_states": [left, right],
                        })
        else:
            complete_states = [state for states in anchor_states for state in states]

        complete_states.sort(
            key=lambda item: sum(item["step_scores"]) / max(len(item["step_scores"]), 1),
            reverse=True,
        )
        candidates: List[Dict[str, Any]] = []
        groups: List[Dict[str, Any]] = []
        for state_idx, state in enumerate(complete_states[: self.max_paths]):
            mean_score = sum(state["step_scores"]) / max(len(state["step_scores"]), 1)
            min_relation = min(state["relation_scores"], default=0.0)
            min_type = min(state["type_scores"], default=0.0)
            if mean_score < self.min_path_score:
                rejections["path_score_below_threshold"] += 1
                continue
            passage_indices = self._deduplicate(state["passage_indices"])
            if not passage_indices:
                continue
            proof_coverage = 1.0
            expected_steps = len(relation_path) * (2 if comparison else 1)
            observed_directions = [
                value
                for branch in state.get("access_directions", []) or []
                for value in (branch if isinstance(branch, list) else [branch])
            ]
            expected_directions = [
                value
                for branch in state.get("expected_directions", []) or []
                for value in (branch if isinstance(branch, list) else [branch])
            ]
            direction_consistent = bool(
                len(observed_directions) == expected_steps
                and all(
                    expected == "either" or expected == observed
                    for observed, expected in zip(observed_directions, expected_directions)
                )
            )
            strict_chain_certificate = bool(
                state.get("exact_anchor", False)
                and len(state.get("triples", []) or []) == expected_steps
                and len(state.get("access_methods", []) or []) == expected_steps
                and min_relation >= self.min_relation_score
                and min_type > 0.0
                and direction_consistent
            )
            if not strict_chain_certificate:
                rejections["strict_chain_certificate_failed"] += 1
                continue
            cost = len(state["triples"]) + 0.01 * sum(state["candidate_ranks"])
            utility = 0.72 * mean_score + 0.18 * proof_coverage + 0.10 * min_type - 0.005 * cost
            group_id = f"candidate_window_provenance_join::{state_idx}"
            physical_access_methods = [
                str(method)
                for methods in state.get("access_methods", []) or []
                for method in (methods if isinstance(methods, list) else [methods])
                if str(method or "").strip()
            ]
            group = {
                "group_id": group_id,
                "operator": "candidate_window_provenance_join",
                "passage_indices": passage_indices,
                "terminal_passage_indices": self._deduplicate([
                    branch["passage_indices"][-1]
                    for branch in state.get("branch_states", [])
                    if branch.get("passage_indices")
                ]) if comparison else [passage_indices[-1]],
                "bindings": state["bindings"],
                "relations": relation_path,
                "triples": state["triples"],
                "answer_type": answer_type,
                "score": round(mean_score, 6),
                "utility": round(utility, 6),
                "estimated_cost": round(cost, 6),
                "chain_completeness": 1.0,
                "proof_confidence": round(min(mean_score, min_relation, float(state["anchor_score"])), 6),
                "proof_relation_coverage": proof_coverage,
                "proof_type_consistency": round(min_type, 6),
                "proof_provenance_coverage": 1.0,
                "provenance_validated": True,
                "strict_chain_certificate": strict_chain_certificate,
                "strict_chain_certificate_details": {
                    "expected_step_count": expected_steps,
                    "observed_step_count": len(state.get("triples", []) or []),
                    "exact_anchor": bool(state.get("exact_anchor", False)),
                    "direction_consistent": direction_consistent,
                    "direction_constrained_count": sum(
                        int(value != "either") for value in expected_directions
                    ),
                    "minimum_relation_score": round(min_relation, 6),
                    "minimum_type_score": round(min_type, 6),
                },
                "candidate_window_path": True,
                "schema_to_evidence_grounding": True,
                "physical_access_methods": list(dict.fromkeys(physical_access_methods)),
                "mixed_access_path": len(set(physical_access_methods)) > 1,
                "exact_anchor": bool(state["exact_anchor"]),
                "contextual_relation_count": int(state["contextual_relation_count"]),
                "candidate_ranks": list(state["candidate_ranks"]),
                "access_directions": state["access_directions"],
            }
            groups.append(group)
            terminal_passages = group["terminal_passage_indices"] or [passage_indices[-1]]
            for passage_idx in terminal_passages:
                candidates.append({
                    "passage_idx": int(passage_idx),
                    "operator": "candidate_window_provenance_join",
                    "group_id": group_id,
                    "anchor_entity": list(anchors[:2] if comparison else anchors[:1]),
                    "terminal_entity": state["binding"],
                    "relations": relation_path,
                    "answer_type": answer_type,
                    "relation_score": round(min_relation, 6),
                    "type_score": round(min_type, 6),
                    "raw_score": round(mean_score, 6),
                    "utility": round(utility, 6),
                    "estimated_cost": round(cost, 6),
                    "exact_entity_key": bool(state["exact_anchor"]),
                    "provenance_validated": True,
                    "candidate_window_path": True,
                    "contextual_relation_count": int(state["contextual_relation_count"]),
                    "triple": list(state["triples"][-1]),
                })

        log.update({
            "candidate_count": len(candidates),
            "group_count": len(groups),
            "complete_path_count": len(complete_states),
            "contextual_relation_path_count": sum(
                1 for group in groups if int(group.get("contextual_relation_count", 0)) > 0
            ),
            "entity_index_expansion_enabled": self.enable_entity_index_expansion,
            "bound_dense_expansion_enabled": self.enable_bound_dense_expansion,
            "entity_index_probe_count": int(access_state["entity_index_probe_count"]),
            "entity_index_edge_count": int(access_state["entity_index_edge_count"]),
            "bound_dense_probe_count": int(access_state["bound_dense_probe_count"]),
            "bound_dense_edge_count": int(access_state["bound_dense_edge_count"]),
            "bound_dense_probes": list(access_state["bound_dense_probes"]),
            "external_passage_count": len(access_state["external_rank_by_doc"]),
            "external_passage_ids": list(access_state["external_rank_by_doc"]),
            "materialized_triple_count": len(edges),
            "rejection_reasons": dict(rejections),
            "skip_reason": "" if candidates else "no_complete_candidate_window_path",
        })
        if self.enable_relation_agnostic_bridge:
            bridge_result = self._run_relation_agnostic_bridge(
                query=str(query or ""),
                base_doc_ids=base_doc_ids,
                anchors=anchors,
                answer_type=answer_type,
                get_openie_record=get_openie_record,
                terminal_type_score=terminal_type_score,
            )
            log["relation_agnostic_bridge"] = dict(bridge_result.log)
            candidates.extend(bridge_result.candidates)
            groups.extend(bridge_result.groups)
            log["candidate_count"] = len(candidates)
            log["group_count"] = len(groups)
            log["bridge_candidate_count"] = len(bridge_result.candidates)
            log["bridge_group_count"] = len(bridge_result.groups)
            if bridge_result.candidates:
                log["skip_reason"] = ""
        return QueryLocalPathCompilerResult(candidates=candidates, groups=groups, log=log)
