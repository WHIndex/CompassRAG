"""Gold-free ontology features over the complete retrieval aperture.

The module scans the immutable retrieval window and emits
one compact, typed proof profile per passage.  Gold labels are intentionally
absent; offline counterfactual replay attaches them on training traces only.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence, Set


FULL_APERTURE_MODE = "proof_full_aperture_nested_topk"
FULL_APERTURE_SCHEMA = "ontology_full_aperture_proof_slate_v1"

_TOKEN = re.compile(r"[a-z0-9]+")
_DATE = re.compile(
    r"\b(?:1[0-9]{3}|20[0-9]{2}|"
    r"january|february|march|april|may|june|july|august|"
    r"september|october|november|december)\b",
    re.IGNORECASE,
)
_NUMBER = re.compile(r"\b\d+(?:[.,]\d+)?\b")

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do",
    "does", "for", "from", "how", "in", "is", "it", "of", "on", "or",
    "the", "to", "was", "were", "what", "when", "where", "which", "who",
    "whose", "with",
}
_ROLE_HEADS = {
    "actor", "author", "composer", "creator", "director", "founder",
    "leader", "member", "owner", "parent", "producer", "singer", "writer",
}
_GENERIC_ROLE_OBJECTS = {
    "book", "company", "film", "group", "movie", "novel", "organization",
    "person", "song", "team", "work",
}
_ANSWER_TYPE_TERMS = {
    "date": {"born", "date", "day", "died", "month", "year"},
    "location": {
        "area", "city", "country", "county", "district", "location", "place",
        "province", "region", "state", "town", "village",
    },
    "numeric": {"amount", "count", "number", "population", "total"},
    "organisation": {
        "association", "company", "corporation", "foundation", "group",
        "institute", "organization", "society", "team", "university",
    },
    "person": {
        "actor", "author", "composer", "director", "founder", "person",
        "producer", "writer",
    },
}


def is_full_aperture_mode(value: Any) -> bool:
    return str(value or "").strip().lower() == FULL_APERTURE_MODE


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        return default if math.isnan(number) or math.isinf(number) else number
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _ids(values: Iterable[Any]) -> List[int]:
    result: List[int] = []
    seen = set()
    for value in values:
        passage_idx = _safe_int(value)
        if passage_idx < 0 or passage_idx in seen:
            continue
        seen.add(passage_idx)
        result.append(passage_idx)
    return result


def _tokens(value: Any, *, drop_stopwords: bool = False) -> List[str]:
    values = _TOKEN.findall(str(value or "").lower())
    if drop_stopwords:
        values = [value for value in values if value not in _STOPWORDS]
    return values


def _coverage(needle: Iterable[str], haystack: Set[str]) -> float:
    required = set(needle)
    if not required or not haystack:
        return 0.0
    return len(required & haystack) / len(required)


def _phrase_coverage(values: Iterable[Any], passage_tokens: Set[str]) -> float:
    return max(
        (_coverage(_tokens(value, drop_stopwords=True), passage_tokens) for value in values),
        default=0.0,
    )


def _canonical_answer_type(value: Any) -> str:
    text = str(value or "").strip().lower().replace("_", " ")
    aliases = {
        "city": "location", "country": "location", "place": "location",
        "human": "person", "people": "person", "number": "numeric",
        "organization": "organisation", "org": "organisation", "time": "date",
        "year": "date",
    }
    return aliases.get(text, text)


def _is_role_pseudo_anchor(value: Any) -> bool:
    tokens = _tokens(value, drop_stopwords=False)
    if len(tokens) < 3 or "of" not in tokens:
        return False
    split = tokens.index("of")
    return bool(
        set(tokens[:split]) & _ROLE_HEADS
        and set(tokens[split + 1 :]) & _GENERIC_ROLE_OBJECTS
    )


def _relation_slots(query_plan_log: Mapping[str, Any]) -> List[List[str]]:
    factorized = query_plan_log.get("factorized_query_executor", {}) or {}
    typed = factorized.get("typed_contract", {}) or {}
    surface = [
        str(value).strip()
        for value in typed.get("surface_relations", []) or []
        if str(value).strip()
    ]
    logical = factorized.get("logical_evidence_plan", {}) or {}
    canonical = [
        str(value).strip()
        for value in logical.get("relations", []) or factorized.get("relations", []) or []
        if str(value).strip()
    ]
    physical = [
        [
            str(value).strip()
            for value in operator.get("physical_predicates", []) or []
            if str(value).strip()
        ]
        for operator in logical.get("operators", []) or []
        if isinstance(operator, Mapping)
        and str(operator.get("op", "")).strip().lower() == "traverse"
    ]
    traverses = [
        operator
        for operator in logical.get("operators", []) or []
        if isinstance(operator, Mapping)
        and str(operator.get("op", "")).strip().lower() == "traverse"
    ]
    if traverses:
        ordered_slots: List[List[str]] = []
        seen = set()
        for operator in sorted(
            traverses,
            key=lambda item: (
                int(item.get("hop", 0) or 0),
                int(item.get("branch", 0) or 0),
            ),
        ):
            relation = str(operator.get("relation", "") or "").strip()
            key = (int(operator.get("hop", 0) or 0), relation.lower())
            if not relation or key in seen:
                continue
            seen.add(key)
            values = [relation] + [
                str(value).strip()
                for value in operator.get("physical_predicates", []) or []
                if str(value).strip()
            ]
            ordered_slots.append(list(dict.fromkeys(value.lower() for value in values)))
        if ordered_slots:
            return ordered_slots
    size = max(len(surface), len(canonical), len(physical))
    slots: List[List[str]] = []
    for index in range(size):
        values: List[str] = []
        if index < len(surface):
            values.append(surface[index])
        if index < len(canonical):
            values.append(canonical[index])
        if index < len(physical):
            values.extend(physical[index])
        deduplicated = list(dict.fromkeys(value.lower() for value in values if value))
        if deduplicated:
            slots.append(deduplicated)
    return slots


def _anchors(query_plan_log: Mapping[str, Any]) -> List[str]:
    factorized = query_plan_log.get("factorized_query_executor", {}) or {}
    typed = factorized.get("typed_contract", {}) or {}
    logical = factorized.get("logical_evidence_plan", {}) or {}
    values = (
        factorized.get("anchors", [])
        or typed.get("anchors", [])
        or logical.get("anchors", [])
        or []
    )
    result = []
    for value in values:
        text = str(value or "").strip()
        informative = _tokens(text, drop_stopwords=True)
        if not text or not informative or _is_role_pseudo_anchor(text):
            continue
        result.append(text)
    return list(dict.fromkeys(result))


def _answer_type(query_plan_log: Mapping[str, Any]) -> str:
    factorized = query_plan_log.get("factorized_query_executor", {}) or {}
    typed = factorized.get("typed_contract", {}) or {}
    logical = factorized.get("logical_evidence_plan", {}) or {}
    return _canonical_answer_type(
        typed.get("answer_type")
        or logical.get("answer_type")
        or factorized.get("answer_type")
    )


def _query_structure_signature(
    query_plan_log: Mapping[str, Any],
    slots: Sequence[Sequence[str]],
    anchors: Sequence[str],
    answer_type: str,
) -> Dict[str, Any]:
    """Expose the gold-free logical signature consumed by physical plans."""
    factorized = query_plan_log.get("factorized_query_executor", {}) or {}
    logical = factorized.get("logical_evidence_plan", {}) or {}
    typed = factorized.get("typed_contract", {}) or {}
    comparison = bool(
        factorized.get("comparison_query", False)
        or typed.get("comparison_query", False)
    )
    relation_slots = [list(slot) for slot in slots if slot]
    if comparison:
        topology = "branch_comparison"
    elif len(relation_slots) >= 2:
        topology = "ordered_multihop_chain"
    elif relation_slots:
        topology = "single_relation"
    else:
        topology = "unresolved"
    source_schema_ids = [
        str(value)
        for value in logical.get("source_schema_ids", []) or []
        if str(value)
    ]
    if not source_schema_ids:
        source_schema_ids = [
            str(plan.get("id", "") or "")
            for plan in query_plan_log.get("matched_plans", []) or []
            if isinstance(plan, Mapping) and plan.get("id")
        ]
    return {
        "schema_version": "query_structure_signature_v1",
        "topology": topology,
        "anchors": list(anchors),
        "anchor_count": len(anchors),
        "relation_slots": relation_slots,
        "canonical_relations": [slot[0] for slot in relation_slots],
        "relation_slot_count": len(relation_slots),
        "answer_type": answer_type,
        "comparison": comparison,
        "source_schema_ids": list(dict.fromkeys(source_schema_ids)),
        "gold_free": True,
    }


def _answer_type_score(answer_type: str, text: str, passage_tokens: Set[str]) -> float:
    if not answer_type:
        return 0.0
    if answer_type == "date":
        return float(bool(_DATE.search(text)))
    if answer_type == "numeric":
        return float(bool(_NUMBER.search(text)))
    terms = _ANSWER_TYPE_TERMS.get(answer_type, set())
    return float(bool(terms & passage_tokens)) if terms else 0.0


def _fingerprint(text: str) -> str:
    normalized = " ".join(_tokens(text))
    if not normalized:
        return ""
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:24]


def _entity_key(value: Any) -> str:
    return " ".join(_tokens(value, drop_stopwords=True))


def _entity_match(left: str, right: str) -> bool:
    if not left or not right:
        return False
    if left == right:
        return True
    left_tokens = set(left.split())
    right_tokens = set(right.split())
    overlap = len(left_tokens & right_tokens)
    return bool(
        min(len(left_tokens), len(right_tokens)) >= 2
        and overlap / min(len(left_tokens), len(right_tokens)) >= 0.80
    )


def _terminal_binding_type_score(answer_type: str, path: Mapping[str, Any]) -> float:
    steps = path.get("steps", []) or []
    if not steps:
        return 0.0
    terminal = str(steps[-1].get("output_value", "") or "").strip()
    predicate = str(steps[-1].get("predicate", "") or "").lower()
    canonical = _canonical_answer_type(answer_type)
    if not terminal:
        return 0.0
    if canonical == "date":
        return float(bool(_DATE.search(terminal)))
    if canonical == "numeric":
        return float(bool(_NUMBER.search(terminal)))
    if canonical == "location":
        if _DATE.search(terminal) or re.fullmatch(r"[\d\W]+", terminal):
            return 0.0
        markers = (
            " in", " at", "from", "located", "location", "place",
            "city", "country", "state", "town", "village",
        )
        return float(any(marker in f" {predicate}" for marker in markers))


    return 1.0


def _compiled_candidate_window_paths(
    query_plan_log: Mapping[str, Any],
    candidate_by_id: Mapping[int, Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Import provenance paths emitted by the query-local compiler.

    The full-aperture materializer historically rebuilt paths from lexical
    slot matches and silently discarded the executor's already connected
    entity bindings.  Preserve those bindings as first-class physical paths
    so the downstream portfolio optimizes the evidence that was actually
    compiled for this query.
    """
    factorized = query_plan_log.get("factorized_query_executor", {}) or {}
    supported_operators = {
        "candidate_window_provenance_join",
        "proof_gap_bound_access_path",
    }
    selected_operator = str(
        factorized.get("selected_operator", "") or ""
    )
    if selected_operator not in supported_operators:
        return []

    result: List[Dict[str, Any]] = []
    seen = set()
    for group in factorized.get("groups", []) or []:
        if not isinstance(group, Mapping):
            continue
        group_operator = str(group.get("operator", "") or "")
        is_bridge = bool(
            group.get("query_local_bridge", False)
            or group.get("bridge_chain_certificate", False)
            or group_operator == "query_local_relation_agnostic_bridge"
        )
        if group_operator != selected_operator and not is_bridge:
            continue
        if not bool(group.get("provenance_validated", False)):
            continue
        if not bool(group.get("strict_chain_certificate", False)) and not bool(
            group.get("bridge_chain_certificate", False)
        ):
            continue
        if _safe_float(group.get("chain_completeness")) < 1.0:
            continue
        if _safe_float(group.get("proof_relation_coverage")) < 1.0:
            continue
        passage_ids = [
            value
            for value in dict.fromkeys(
                _safe_int(item)
                for item in group.get("passage_indices", []) or []
            )
            if value in candidate_by_id
        ]
        if not passage_ids:
            continue
        if is_bridge and len(passage_ids) < 2:
            continue
        terminal_ids = [
            value
            for value in dict.fromkeys(
                _safe_int(item)
                for item in group.get("terminal_passage_indices", []) or []
            )
            if value in candidate_by_id
        ]
        if not terminal_ids:
            terminal_ids = passage_ids[-1:]
        key = (tuple(passage_ids), tuple(terminal_ids))
        if key in seen:
            continue
        seen.add(key)
        triples = [
            list(value)
            for value in group.get("triples", []) or []
            if isinstance(value, (list, tuple)) and len(value) == 3
        ]
        if not triples:
            triples = [
                list(edge.get("triple", []) or [])
                for edge in group.get("proof_edges", []) or []
                if isinstance(edge, Mapping)
                and len(list(edge.get("triple", []) or [])) == 3
            ]
        result.append({
            "path_id": str(group.get("group_id", "") or ""),
            "passage_ids": passage_ids,
            "terminal_passage_ids": terminal_ids,
            "relations": [
                str(value or "")
                for value in group.get("relations", []) or []
            ],
            "triples": triples,
            "bindings": group.get("bindings", []),
            "physical_access_methods": list(
                group.get("physical_access_methods", []) or []
            ),
            "mixed_access_path": bool(group.get("mixed_access_path", False)),
            "schema_to_evidence_grounding": bool(
                group.get("schema_to_evidence_grounding", False)
            ),
            "min_relation_score": _safe_float(
                group.get("proof_confidence"),
                _safe_float(group.get("score")),
            ),
            "terminal_answer_type_score": _safe_float(
                group.get("proof_type_consistency")
            ),
            "proof_relation_coverage": _safe_float(
                group.get("proof_relation_coverage")
            ),
            "proof_provenance_coverage": _safe_float(
                group.get("proof_provenance_coverage")
            ),
            "contextual_relation_count": _safe_int(
                group.get("contextual_relation_count"), 0
            ),
            "exact_anchor": bool(group.get("exact_anchor", False)),
            "query_local_compiler_validated": True,
            "strict_chain_certificate": True,
            "query_local_bridge": is_bridge,
            "bridge_chain_certificate": bool(
                group.get("bridge_chain_certificate", False)
            ),
            "relation_contract_mode": str(
                group.get("relation_contract_mode", "") or ""
            ),
            "strict_chain_certificate_details": dict(
                group.get("strict_chain_certificate_details", {}) or {}
            ),
            "anchor_endpoint_bound": True,
            "binding_consistent": True,
        })
    return result


def _materialize_local_proof_paths(
    slots: Sequence[Sequence[str]],
    anchors: Sequence[str],
    rows_by_slot: Sequence[Sequence[Dict[str, Any]]],
    passage_anchor_coverage: Mapping[int, float],
    *,
    max_paths: int = 256,
    require_endpoint_anchor: bool = False,
    answer_type: str = "",
) -> List[Dict[str, Any]]:
    if not slots or not rows_by_slot or not rows_by_slot[0]:
        return []
    anchor_keys = [_entity_key(value) for value in anchors if _entity_key(value)]
    paths: List[Dict[str, Any]] = []
    for row in rows_by_slot[0]:
        orientations = []
        subject_anchor = any(
            _entity_match(row["subject_key"], value) for value in anchor_keys
        )
        object_anchor = any(
            _entity_match(row["object_key"], value) for value in anchor_keys
        )
        if subject_anchor:
            orientations.append((
                row["subject_key"], row["object_key"],
                row["subject"], row["object"], "forward",
            ))
        if object_anchor:
            orientations.append((
                row["object_key"], row["subject_key"],
                row["object"], row["subject"], "reverse",
            ))
        if (
            not orientations
            and not require_endpoint_anchor
            and passage_anchor_coverage.get(row["passage_idx"], 0.0) > 0.0
        ):
            orientations.extend([
                (
                    row["subject_key"], row["object_key"],
                    row["subject"], row["object"], "forward",
                ),
                (
                    row["object_key"], row["subject_key"],
                    row["object"], row["subject"], "reverse",
                ),
            ])
        for start_key, end_key, start_value, end_value, direction in orientations:
            paths.append({
                "start_key": start_key,
                "end_key": end_key,
                "start_value": start_value,
                "end_value": end_value,
                "rows": [row],
                "steps": [{
                    "slot_index": int(row["slot_index"]),
                    "passage_idx": int(row["passage_idx"]),
                    "subject": row["subject"],
                    "predicate": row["predicate"],
                    "object": row["object"],
                    "input_key": start_key,
                    "output_key": end_key,
                    "input_value": start_value,
                    "output_value": end_value,
                    "observed_direction": direction,
                }],
                "min_relation_score": row["relation_score"],
            })
    paths = sorted(
        paths,
        key=lambda value: (
            value["min_relation_score"],
            -len({row["passage_idx"] for row in value["rows"]}),
        ),
        reverse=True,
    )[:max_paths]

    for slot_index in range(1, len(slots)):
        extended: List[Dict[str, Any]] = []
        for path in paths:
            for row in rows_by_slot[slot_index]:
                orientations = []
                if _entity_match(path["end_key"], row["subject_key"]):
                    orientations.append((
                        row["object_key"], row["object"], "forward",
                    ))
                if _entity_match(path["end_key"], row["object_key"]):
                    orientations.append((
                        row["subject_key"], row["subject"], "reverse",
                    ))
                for end_key, end_value, direction in orientations:
                    if not end_key:
                        continue
                    extended.append({
                        "start_key": path["start_key"],
                        "end_key": end_key,
                        "start_value": path["start_value"],
                        "end_value": end_value,
                        "rows": [*path["rows"], row],
                        "steps": [*path["steps"], {
                            "slot_index": int(row["slot_index"]),
                            "passage_idx": int(row["passage_idx"]),
                            "subject": row["subject"],
                            "predicate": row["predicate"],
                            "object": row["object"],
                            "input_key": path["end_key"],
                            "output_key": end_key,
                            "input_value": path["end_value"],
                            "output_value": end_value,
                            "observed_direction": direction,
                        }],
                        "min_relation_score": min(
                            path["min_relation_score"], row["relation_score"]
                        ),
                    })
        paths = sorted(
            extended,
            key=lambda value: (
                value["min_relation_score"],
                len({row["passage_idx"] for row in value["rows"]}),
                -sum(row["original_rank"] for row in value["rows"]),
            ),
            reverse=True,
        )[:max_paths]
        if not paths:
            break
    if not paths or len(paths[0]["rows"]) != len(slots):
        return []
    for path in paths:
        path["terminal_answer_type_score"] = _terminal_binding_type_score(
            answer_type, path
        )
    if require_endpoint_anchor and _canonical_answer_type(answer_type) in {
        "date", "location", "numeric",
    }:
        paths = [
            path for path in paths
            if path["terminal_answer_type_score"] >= 1.0
        ]
    return sorted(
        paths,
        key=lambda value: (
            value.get("terminal_answer_type_score", 0.0),
            value["min_relation_score"],
            -sum(row["original_rank"] for row in value["rows"]),
        ),
        reverse=True,
    )


def materialize_full_aperture_proof_slate(
    query_plan_log: Dict[str, Any],
    get_passage_text: Callable[[int], str],
    *,
    get_openie_record: Callable[[int], Mapping[str, Any]] | None = None,
    max_window: int = 200,
) -> Dict[str, Any]:
    """Attach compact proof features for every passage in the rank window."""
    incumbent = query_plan_log.get("candidate_admission_incumbent", {}) or {}
    incumbent_window_ids = [
        _safe_int(value)
        for value in incumbent.get("candidate_window_doc_ids", []) or []
        if _safe_int(value) >= 0
    ][: max(5, int(max_window))]
    incumbent_window_scores = [
        _safe_float(value)
        for value in incumbent.get("candidate_window_doc_scores", []) or []
    ][: len(incumbent_window_ids)]
    score_by_id = {
        passage_idx: incumbent_window_scores[index]
        for index, passage_idx in enumerate(incumbent_window_ids)
        if index < len(incumbent_window_scores)
    }
    factorized = query_plan_log.get("factorized_query_executor", {}) or {}
    grounded_external_ids: List[int] = []
    incumbent_id_set = set(incumbent_window_ids)
    for group in factorized.get("groups", []) or []:
        if not isinstance(group, Mapping):
            continue
        if not bool(group.get("strict_chain_certificate", False)):
            continue
        if not bool(group.get("provenance_validated", False)):
            continue
        for value in group.get("passage_indices", []) or []:
            passage_idx = _safe_int(value)
            if (
                passage_idx >= 0
                and passage_idx not in incumbent_id_set
                and passage_idx not in grounded_external_ids
            ):
                grounded_external_ids.append(passage_idx)
    grounded_external_ids = grounded_external_ids[:96]
    window_ids = incumbent_window_ids + grounded_external_ids
    window_scores = [score_by_id.get(passage_idx, 0.0) for passage_idx in window_ids]
    slots = _relation_slots(query_plan_log)
    anchors = _anchors(query_plan_log)
    answer_type = _answer_type(query_plan_log)
    query_structure_signature = _query_structure_signature(
        query_plan_log, slots, anchors, answer_type
    )
    unified_access = query_plan_log.get("unified_access_contract", {}) or {}
    access_by_id = {
        _safe_int(item.get("passage_idx")): item
        for item in unified_access.get("provenance", []) or []
        if isinstance(item, Mapping) and _safe_int(item.get("passage_idx")) >= 0
    }
    query_tokens = set(_tokens(query_plan_log.get("query"), drop_stopwords=True))
    candidates: List[Dict[str, Any]] = []
    slot_hit_counts = [0 for _ in slots]
    structured_slot_hit_counts = [0 for _ in slots]
    rows_by_slot: List[List[Dict[str, Any]]] = [[] for _ in slots]

    for rank, passage_idx in enumerate(window_ids, start=1):
        try:
            text = str(get_passage_text(passage_idx) or "")
        except Exception:
            text = ""
        passage_tokens = set(_tokens(text, drop_stopwords=True))
        text_slot_coverages = [
            _phrase_coverage(slot, passage_tokens) for slot in slots
        ]
        try:
            openie_record = (
                get_openie_record(passage_idx) if get_openie_record is not None else {}
            ) or {}
        except Exception:
            openie_record = {}
        triples = [
            tuple(str(part or "") for part in triple)
            for triple in openie_record.get("extracted_triples", []) or []
            if isinstance(triple, (list, tuple)) and len(triple) == 3
        ]
        structured_slot_coverages: List[float] = []
        best_triple_by_slot: List[Sequence[str] | None] = []
        for slot in slots:
            scored = [
                (_phrase_coverage(slot, set(_tokens(triple[1], drop_stopwords=True))), triple)
                for triple in triples
            ]
            best_score, best_triple = max(
                scored, key=lambda value: value[0], default=(0.0, None)
            )
            structured_slot_coverages.append(best_score)
            best_triple_by_slot.append(best_triple)
        for slot_index, slot in enumerate(slots):
            threshold = (
                1.0
                if len(_tokens(slot[0], drop_stopwords=True)) <= 1
                else 0.67
            )
            for triple in triples:
                relation_score = _phrase_coverage(
                    slot, set(_tokens(triple[1], drop_stopwords=True))
                )
                if relation_score < threshold:
                    continue
                rows_by_slot[slot_index].append({
                    "passage_idx": passage_idx,
                    "original_rank": rank,
                    "subject": triple[0],
                    "predicate": triple[1],
                    "object": triple[2],
                    "subject_key": _entity_key(triple[0]),
                    "object_key": _entity_key(triple[2]),
                    "relation_score": relation_score,
                    "slot_index": slot_index,
                })
        slot_coverages = [
            max(text_score, structured_score)
            for text_score, structured_score in zip(
                text_slot_coverages, structured_slot_coverages
            )
        ]
        slot_hits = [
            coverage >= (1.0 if len(_tokens(slot[0], drop_stopwords=True)) <= 1 else 0.67)
            for coverage, slot in zip(slot_coverages, slots)
        ]
        for index, hit in enumerate(slot_hits):
            slot_hit_counts[index] += int(hit)
            structured_slot_hit_counts[index] += int(
                structured_slot_coverages[index]
                >= (1.0 if len(_tokens(slots[index][0], drop_stopwords=True)) <= 1 else 0.67)
            )
        covered_slots = sum(slot_hits)
        slot_coverage = covered_slots / max(len(slots), 1)
        text_anchor_coverage = _phrase_coverage(anchors, passage_tokens)
        structured_anchor_coverage = max(
            (
                _phrase_coverage(
                    anchors,
                    set(_tokens(f"{triple[0]} {triple[2]}", drop_stopwords=True)),
                )
                for triple in triples
            ),
            default=0.0,
        )
        anchor_coverage = max(text_anchor_coverage, structured_anchor_coverage)
        query_coverage = _coverage(query_tokens, passage_tokens)
        answer_type_score = max(
            _answer_type_score(answer_type, text, passage_tokens),
            max(
                (
                    _answer_type_score(
                        answer_type,
                        triple[2],
                        set(_tokens(triple[2], drop_stopwords=True)),
                    )
                    for triple in triples
                ),
                default=0.0,
            ),
        )
        terminal_slot_match = bool(slot_hits[-1]) if slot_hits else False
        complete_single_passage_proof = bool(
            slots and covered_slots == len(slots) and anchor_coverage > 0.0
        )
        score = window_scores[rank - 1] if rank - 1 < len(window_scores) else 0.0
        candidate = {
            "passage_idx": passage_idx,
            "original_rank": rank,
            "score": score,
            "confidence": slot_coverage,
            "content_fingerprint": _fingerprint(text),
            "content_available": bool(text),
            "full_aperture": True,
            "anchor_coverage": anchor_coverage,
            "text_anchor_coverage": text_anchor_coverage,
            "structured_anchor_coverage": structured_anchor_coverage,
            "query_coverage": query_coverage,
            "relation_slot_coverages": slot_coverages,
            "text_relation_slot_coverages": text_slot_coverages,
            "structured_relation_slot_coverages": structured_slot_coverages,
            "relation_slot_hits": [int(value) for value in slot_hits],
            "covered_relation_slot_count": covered_slots,
            "required_relation_slot_count": len(slots),
            "relation_coverage_score": slot_coverage,
            "query_relation_coverage": slot_coverage,
            "query_predicate_coverage": slot_coverage,
            "terminal_relation_match": terminal_slot_match,
            "answer_type": answer_type,
            "answer_type_compatible": answer_type_score > 0.0,
            "answer_type_score": answer_type_score,
            "matched_anchor": anchor_coverage > 0.0,
            "matched_terminal": terminal_slot_match,
            "has_proof_signal": covered_slots > 0,
            "role_consistent": terminal_slot_match or anchor_coverage > 0.0,
            "complete_provenance": complete_single_passage_proof,
            "single_passage_proof": complete_single_passage_proof,
            "exact_entity_key": anchor_coverage >= 0.67,
            "binding": complete_single_passage_proof,
            "typed_slot_binding": covered_slots > 0,
            "access_sources": list(
                access_by_id.get(passage_idx, {}).get("access_sources", []) or []
            ),
            "semantic_access_rank": access_by_id.get(passage_idx, {}).get(
                "semantic_rank"
            ),
            "physical_access_rank": access_by_id.get(passage_idx, {}).get(
                "physical_rank"
            ),
            "grounding_external_access": passage_idx in grounded_external_ids,
            "structured_triple_count_log": math.log1p(len(triples)),
            "binding_entities": list(dict.fromkeys(
                normalized
                for triple in triples
                for endpoint in (triple[0], triple[2])
                for normalized in [" ".join(_tokens(endpoint))]
                if normalized
            ))[:16],
            "contexts": [{
                "relations": [
                    slots[index][0]
                    for index, hit in enumerate(slot_hits)
                    if hit
                ],
                "relation_slot_hits": [int(value) for value in slot_hits],
                "anchor_entity": anchors[0] if anchors and anchor_coverage > 0.0 else "",
                "plan_effective_precision": slot_coverage,
                "score": score,
                "matched_triples": [
                    list(triple) for triple in best_triple_by_slot if triple
                ],
            }],
        }
        if complete_single_passage_proof:
            candidate["proof_certificate"] = {
                "schema": FULL_APERTURE_SCHEMA,
                "gold_free": True,
                "covered_relation_slots": len(slots),
            }
        candidates.append(candidate)

    candidate_by_id = {
        int(candidate["passage_idx"]): candidate for candidate in candidates
    }
    complete_paths = _materialize_local_proof_paths(
        slots,
        anchors,
        rows_by_slot,
        {
            int(candidate["passage_idx"]): float(candidate["anchor_coverage"])
            for candidate in candidates
        },
        answer_type=answer_type,
    )
    bound_paths = _materialize_local_proof_paths(
        slots,
        anchors,
        rows_by_slot,
        {
            int(candidate["passage_idx"]): float(candidate["anchor_coverage"])
            for candidate in candidates
        },
        require_endpoint_anchor=True,
        answer_type=answer_type,
    )
    compiled_paths = _compiled_candidate_window_paths(
        query_plan_log,
        candidate_by_id,
    )
    for path_index, path in enumerate(complete_paths):
        path_passage_ids = list(dict.fromkeys(
            int(row["passage_idx"]) for row in path["rows"]
        ))
        path_id = hashlib.sha256(
            repr([
                (row["passage_idx"], row["subject_key"], row["predicate"], row["object_key"])
                for row in path["rows"]
            ]).encode("utf-8")
        ).hexdigest()[:20]
        for row in path["rows"]:
            candidate = candidate_by_id.get(int(row["passage_idx"]))
            if not candidate:
                continue
            candidate.setdefault("local_proof_path_ids", []).append(path_id)
            candidate.setdefault("local_proof_path_positions", []).append(
                int(row["slot_index"])
            )
            candidate["local_proof_path_complete"] = True
            candidate["local_proof_path_passage_count"] = len(path_passage_ids)
            candidate["local_proof_path_min_relation_score"] = float(
                path["min_relation_score"]
            )
            candidate["proof_certificate"] = {
                "schema": "ontology_aperture_local_join_path_v1",
                "gold_free": True,
                "path_id": path_id,
                "path_passage_ids": path_passage_ids,
                "path_index": path_index,
            }
            candidate["complete_provenance"] = True
            candidate["binding"] = True
            candidate["has_proof_signal"] = True

    for path_index, path in enumerate(bound_paths):
        path_passage_ids = list(dict.fromkeys(
            int(row["passage_idx"]) for row in path["rows"]
        ))
        path_id = hashlib.sha256(
            repr([
                (
                    step["passage_idx"], step["input_key"], step["predicate"],
                    step["output_key"], step["observed_direction"],
                )
                for step in path["steps"]
            ]).encode("utf-8")
        ).hexdigest()[:20]
        for row in path["rows"]:
            candidate = candidate_by_id.get(int(row["passage_idx"]))
            if not candidate:
                continue
            candidate.setdefault("bound_proof_path_ids", []).append(path_id)
            candidate.setdefault("bound_proof_path_positions", []).append(
                int(row["slot_index"])
            )
            candidate["bound_proof_path_complete"] = True
            candidate["bound_proof_path_passage_count"] = len(path_passage_ids)
            candidate["bound_proof_path_min_relation_score"] = float(
                path["min_relation_score"]
            )
            candidate["bound_terminal_answer_type_score"] = float(
                path["terminal_answer_type_score"]
            )
            candidate["anchor_endpoint_bound"] = True
            candidate["binding"] = True
            candidate["has_proof_signal"] = True




    for path_index, path in enumerate(compiled_paths):
        path_id = str(path.get("path_id", "") or "") or hashlib.sha256(
            repr([
                path.get("passage_ids", []),
                path.get("relations", []),
                path.get("bindings", []),
            ]).encode("utf-8")
        ).hexdigest()[:20]
        terminal_ids = set(path.get("terminal_passage_ids", []) or [])
        passage_ids = _ids(path.get("passage_ids", []) or [])
        for position, passage_idx in enumerate(passage_ids):
            candidate = candidate_by_id.get(passage_idx)
            if not candidate:
                continue
            candidate.setdefault("local_proof_path_ids", []).append(path_id)
            candidate.setdefault("local_proof_path_positions", []).append(position)
            candidate.setdefault("bound_proof_path_ids", []).append(path_id)
            candidate.setdefault("bound_proof_path_positions", []).append(position)
            candidate["local_proof_path_complete"] = True
            candidate["local_proof_path_passage_count"] = len(passage_ids)
            candidate["local_proof_path_min_relation_score"] = float(
                path.get("min_relation_score", 0.0) or 0.0
            )
            candidate["bound_proof_path_complete"] = True
            candidate["bound_proof_path_passage_count"] = len(passage_ids)
            candidate["bound_proof_path_min_relation_score"] = float(
                path.get("min_relation_score", 0.0) or 0.0
            )
            candidate["bound_terminal_answer_type_score"] = float(
                path.get("terminal_answer_type_score", 0.0) or 0.0
            )
            candidate["anchor_endpoint_bound"] = True
            candidate["binding"] = True
            candidate["complete_provenance"] = True
            candidate["has_proof_signal"] = True
            candidate["query_local_compiler_validated"] = True
            candidate["candidate_window_terminal_member"] = passage_idx in terminal_ids
            candidate["relation_coverage_score"] = max(
                float(candidate.get("relation_coverage_score", 0.0) or 0.0),
                float(path.get("proof_relation_coverage", 0.0) or 0.0),
            )
            candidate["proof_certificate"] = {
                "schema": "query_local_candidate_window_provenance_path_v1",
                "gold_free": True,
                "path_id": path_id,
                "path_passage_ids": passage_ids,
                "terminal_passage_ids": sorted(terminal_ids),
                "path_index": path_index,
            }

    log = {
        "enabled": True,
        "schema_version": FULL_APERTURE_SCHEMA,
        "gold_free": True,
        "window_size": len(window_ids),
        "incumbent_window_size": len(incumbent_window_ids),
        "grounding_external_passage_count": len(grounded_external_ids),
        "grounding_external_passage_ids": grounded_external_ids,
        "candidate_count": len(candidates),
        "relation_slots": slots,
        "relation_slot_count": len(slots),
        "query_structure_signature": query_structure_signature,
        "unified_access_contract": {
            "enabled": bool(unified_access),
            "schema_version": unified_access.get("schema_version", ""),
            "semantic_count": _safe_int(unified_access.get("semantic_count"), 0),
            "physical_count": _safe_int(unified_access.get("physical_count"), 0),
            "window_count": _safe_int(unified_access.get("window_count"), 0),
        },
        "relation_slot_hit_counts": slot_hit_counts,
        "structured_relation_slot_hit_counts": structured_slot_hit_counts,
        "anchors": anchors,
        "answer_type": answer_type,
        "content_available_count": sum(
            int(candidate["content_available"]) for candidate in candidates
        ),
        "proof_signal_candidate_count": sum(
            int(candidate["has_proof_signal"]) for candidate in candidates
        ),
        "complete_single_passage_proof_count": sum(
            int(candidate["single_passage_proof"]) for candidate in candidates
        ),
        "local_proof_path_count": len(complete_paths) + len(compiled_paths),
        "local_proof_path_passage_count": sum(
            int(bool(candidate.get("local_proof_path_complete")))
            for candidate in candidates
        ),
        "local_proof_paths": [
            {
                "path_id": hashlib.sha256(
                    repr([
                        (row["passage_idx"], row["subject_key"], row["predicate"], row["object_key"])
                        for row in path["rows"]
                    ]).encode("utf-8")
                ).hexdigest()[:20],
                "passage_ids": list(dict.fromkeys(
                    int(row["passage_idx"]) for row in path["rows"]
                )),
                "relations": [row["predicate"] for row in path["rows"]],
                "min_relation_score": path["min_relation_score"],
            }
            for path in complete_paths[:32]
        ] + [dict(path) for path in compiled_paths[:32]],
        "bound_proof_path_count": len(bound_paths) + len(compiled_paths),
        "bound_proof_path_passage_count": sum(
            int(bool(candidate.get("bound_proof_path_complete")))
            for candidate in candidates
        ),
        "bound_proof_paths": [
            {
                "path_id": hashlib.sha256(
                    repr([
                        (
                            step["passage_idx"], step["input_key"],
                            step["predicate"], step["output_key"],
                            step["observed_direction"],
                        )
                        for step in path["steps"]
                    ]).encode("utf-8")
                ).hexdigest()[:20],
                "passage_ids": list(dict.fromkeys(
                    int(row["passage_idx"]) for row in path["rows"]
                )),
                "relations": [row["predicate"] for row in path["rows"]],
                "start_entity": path["start_value"],
                "terminal_entity": path["end_value"],
                "start_key": path["start_key"],
                "terminal_key": path["end_key"],
                "binding_keys": [
                    path["steps"][0]["input_key"],
                    *[step["output_key"] for step in path["steps"]],
                ],
                "steps": path["steps"],
                "anchor_endpoint_bound": True,
                "binding_consistent": True,
                "terminal_answer_type_score": path[
                    "terminal_answer_type_score"
                ],
                "min_relation_score": path["min_relation_score"],
            }
            for path in bound_paths[:32]
        ] + [dict(path) for path in compiled_paths[:32]],
        "candidate_window_compiled_path_count": len(compiled_paths),
        "candidates": candidates,
    }
    query_plan_log["full_aperture_proof_slate"] = log
    return log
