"""Rank-partitioned counterfactual evidence-set optimizer.

The compiler enumerates every non-incumbent top-2 subset of the immutable
top-5 slate.  These actions only reorder existing top-5 passages, so they can
improve R@2 while preserving top-5 membership and therefore R@5 exactly.
Offline counterfactual replay supplies labels; runtime features are gold-free.
"""

from __future__ import annotations

from itertools import combinations
import hashlib
import json
import math
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

try:
    from . import query_plan_content_complete_set_comparator as content_complete
except ImportError:
    import query_plan_content_complete_set_comparator as content_complete


MODEL_TYPE = "rank_partitioned_counterfactual_set_optimizer"
RANK_PARTITIONED_MODES = {
    "rank_partitioned_top2",
    "rank_partitioned_unified",
    "rank_partitioned_external_only",
}

V224_RANK_PARTITION_FEATURE_NAMES = [
    "rank_partition_action",
    "top5_membership_invariant",
    "rank_partition_zero_admission_cost",
    "selected_original_rank_min",
    "selected_original_rank_max",
    "selected_original_rank_mean",
    "selected_original_rank_inv_mean",
    "selected_original_rank_sum_inv",
    "selected_rank1_preserved",
    "selected_rank2_preserved",
    "selected_from_rank3_to5_count",
    "displaced_incumbent_count",
    "displaced_rank1_incumbent",
    "displaced_rank2_incumbent",
    "selected_retrieval_score_max",
    "selected_retrieval_score_min",
    "selected_retrieval_score_mean",
    "selected_retrieval_score_spread",
    "displaced_retrieval_score_max",
    "displaced_retrieval_score_mean",
    "selected_minus_displaced_retrieval_score",
    "selected_ontology_backed_fraction",
    "selected_ontology_source_count_log",
    "selected_ontology_source_agreement",
    "selected_proof_fraction",
    "selected_role_consistent_fraction",
    "selected_complete_provenance_fraction",
    "selected_relation_coverage_mean",
    "selected_answer_type_compatible_fraction",
    "rank_partition_action_pool_fraction",
    "rank_partition_action_pool_size_log",
]


def feature_names(base_module) -> List[str]:
    return [
        *content_complete.feature_names(base_module),
        *V224_RANK_PARTITION_FEATURE_NAMES,
    ]


def content_feature_names(base_module) -> Set[str]:
    return content_complete.content_feature_names(base_module)


def rank_partition_feature_names() -> Set[str]:
    return set(V224_RANK_PARTITION_FEATURE_NAMES)


def is_rank_partitioned_mode(candidate_mode: str) -> bool:
    return str(candidate_mode or "").strip().lower() in RANK_PARTITIONED_MODES


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


def _safe_bool(value: Any) -> float:
    if isinstance(value, str):
        return float(value.strip().lower() in {"1", "true", "yes", "on"})
    return float(bool(value))


def _mean(values: Iterable[float]) -> float:
    rows = list(values)
    return sum(rows) / len(rows) if rows else 0.0


def _max_candidate_value(
    refs: Sequence[Tuple[str, Mapping[str, Any]]], names: Sequence[str]
) -> float:
    return max(
        (
            _safe_float(candidate.get(name), 0.0)
            for _, candidate in refs
            for name in names
        ),
        default=0.0,
    )


def required_passage_ids(
    query_plan_log: Dict[str, Any],
    *,
    base_module,
    max_single_candidates: int = 24,
    max_pair_candidates: int = 12,
    candidate_mode: str = "rank_partitioned_top2",
    enforce_cq: bool = True,
) -> Set[int]:
    """Return passages needed to compare every compiled action by content."""
    mode = str(candidate_mode or "").strip().lower()
    incumbent = query_plan_log.get("candidate_admission_incumbent", {}) or {}
    result = {
        _safe_int(value)
        for value in incumbent.get("doc_ids", []) or []
        if _safe_int(value) >= 0
    }
    if mode in {"rank_partitioned_unified", "rank_partitioned_external_only"}:
        result.update(content_complete.required_passage_ids(
            query_plan_log,
            base_module=base_module,
            max_single_candidates=max_single_candidates,
            max_pair_candidates=max_pair_candidates,
            candidate_mode="content_complete_top2",
            enforce_cq=enforce_cq,
        ))
    return result


def _ontology_pair_features(
    query_plan_log: Dict[str, Any],
    selected_ids: Sequence[int],
    grouped_refs: Mapping[int, Sequence[Tuple[str, Mapping[str, Any]]]],
    *,
    base_module,
) -> Tuple[Dict[str, float], List[str]]:
    rows: List[Dict[str, float]] = []
    all_refs: List[Tuple[str, Mapping[str, Any]]] = []
    backed = 0
    for passage_idx in selected_ids:
        refs = list(grouped_refs.get(passage_idx, []) or [])
        if refs:
            backed += 1
            all_refs.extend(refs)
            rows.append(base_module.aggregate_passage_candidate_features(
                query_plan_log, passage_idx, refs, "top2"
            ))
    aggregated: Dict[str, float] = {}
    for name in getattr(base_module, "V209_FEATURE_NAMES", []):
        aggregated[name] = max(
            (_safe_float(row.get(name), 0.0) for row in rows), default=0.0
        )

    sources = {str(source).strip().lower() for source, _ in all_refs if str(source).strip()}
    selected_count = max(len(selected_ids), 1)
    features = {
        **aggregated,
        "selected_ontology_backed_fraction": backed / selected_count,
        "selected_ontology_source_count_log": math.log1p(len(sources)),
        "selected_ontology_source_agreement": min(len(sources) / 4.0, 1.0),
        "selected_proof_fraction": _mean(
            max(
                (
                    _safe_bool(candidate.get("has_proof_signal"))
                    or _safe_bool(candidate.get("proof_certificate"))
                    for _, candidate in grouped_refs.get(passage_idx, []) or []
                ),
                default=0.0,
            )
            for passage_idx in selected_ids
        ),
        "selected_role_consistent_fraction": _mean(
            _max_candidate_value(
                list(grouped_refs.get(passage_idx, []) or []),
                ("role_consistent",),
            )
            for passage_idx in selected_ids
        ),
        "selected_complete_provenance_fraction": _mean(
            _max_candidate_value(
                list(grouped_refs.get(passage_idx, []) or []),
                ("complete_provenance",),
            )
            for passage_idx in selected_ids
        ),
        "selected_relation_coverage_mean": _mean(
            _max_candidate_value(
                list(grouped_refs.get(passage_idx, []) or []),
                ("relation_coverage_score", "query_relation_coverage"),
            )
            for passage_idx in selected_ids
        ),
        "selected_answer_type_compatible_fraction": _mean(
            _max_candidate_value(
                list(grouped_refs.get(passage_idx, []) or []),
                ("answer_type_compatible", "answer_type_match"),
            )
            for passage_idx in selected_ids
        ),
    }
    return features, sorted(sources) or ["incumbent_top5"]


def _rank_partition_actions(
    query_plan_log: Dict[str, Any], *, base_module
) -> List[Dict[str, Any]]:
    incumbent = query_plan_log.get("candidate_admission_incumbent", {}) or {}
    incumbent_ids = [
        _safe_int(value)
        for value in incumbent.get("doc_ids", []) or []
        if _safe_int(value) >= 0
    ][:5]
    incumbent_scores = [
        _safe_float(value)
        for value in incumbent.get("doc_scores", []) or []
    ][:5]
    if len(incumbent_ids) < 5:
        return []

    context = query_plan_log.get("candidate_local_context", {}) or {}
    passage_context = context.get("passages", {}) or {}
    grouped_refs = base_module.group_query_plan_evidence_candidate_refs(query_plan_log)
    names = feature_names(base_module)
    score_by_id = {
        passage_idx: incumbent_scores[index] if index < len(incumbent_scores) else 0.0
        for index, passage_idx in enumerate(incumbent_ids)
    }
    rank_by_id = {passage_idx: index + 1 for index, passage_idx in enumerate(incumbent_ids)}
    state_hash = str(
        incumbent.get("state_hash") or base_module.final_state_hash(incumbent_ids)
    )
    state_version = str(
        incumbent.get("state_version") or base_module.V210_STATE_VERSION
    )
    result: List[Dict[str, Any]] = []

    for positions in combinations(range(5), 2):
        if positions == (0, 1):
            continue
        selected_ids = [incumbent_ids[position] for position in positions]
        displaced_ids = [
            passage_idx for passage_idx in incumbent_ids[:2]
            if passage_idx not in selected_ids
        ]
        preserved_ids = [
            passage_idx for passage_idx in incumbent_ids[:2]
            if passage_idx in selected_ids
        ]
        selected_ranks = [rank_by_id[value] for value in selected_ids]
        selected_scores = [score_by_id[value] for value in selected_ids]
        displaced_scores = [score_by_id[value] for value in displaced_ids]
        ontology, sources = _ontology_pair_features(
            query_plan_log,
            selected_ids,
            grouped_refs,
            base_module=base_module,
        )
        provisional = {
            "action_kind": "select_top2_subset",
            "passage_ids": selected_ids,
            "candidate_ids": selected_ids,
            "preserved_incumbent_ids": preserved_ids,
            "displaced_incumbent_ids": displaced_ids,
        }
        local, metadata = content_complete._content_features(
            provisional, passage_context, incumbent_ids
        )
        if not metadata["identity_complete"] or not metadata["semantic_complete"]:
            continue

        features = {name: 0.0 for name in names}
        features.update(ontology)
        features.update(local)
        features.update({
            "rank_partition_action": 1.0,
            "top5_membership_invariant": 1.0,
            "rank_partition_zero_admission_cost": 1.0,
            "selected_original_rank_min": float(min(selected_ranks)),
            "selected_original_rank_max": float(max(selected_ranks)),
            "selected_original_rank_mean": _mean(selected_ranks),
            "selected_original_rank_inv_mean": _mean(
                1.0 / rank for rank in selected_ranks
            ),
            "selected_original_rank_sum_inv": 1.0 / sum(selected_ranks),
            "selected_rank1_preserved": float(incumbent_ids[0] in selected_ids),
            "selected_rank2_preserved": float(incumbent_ids[1] in selected_ids),
            "selected_from_rank3_to5_count": float(
                sum(rank >= 3 for rank in selected_ranks)
            ),
            "displaced_incumbent_count": float(len(displaced_ids)),
            "displaced_rank1_incumbent": float(incumbent_ids[0] in displaced_ids),
            "displaced_rank2_incumbent": float(incumbent_ids[1] in displaced_ids),
            "selected_retrieval_score_max": max(selected_scores, default=0.0),
            "selected_retrieval_score_min": min(selected_scores, default=0.0),
            "selected_retrieval_score_mean": _mean(selected_scores),
            "selected_retrieval_score_spread": (
                max(selected_scores, default=0.0) - min(selected_scores, default=0.0)
            ),
            "displaced_retrieval_score_max": max(displaced_scores, default=0.0),
            "displaced_retrieval_score_mean": _mean(displaced_scores),
            "selected_minus_displaced_retrieval_score": (
                _mean(selected_scores) - _mean(displaced_scores)
            ),
        })
        content_state = [
            str((passage_context.get(str(value), {}) or {}).get("content_fingerprint") or "")
            for value in incumbent_ids
        ]
        content_state_fingerprint = hashlib.sha256(
            json.dumps(content_state, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:24]
        identity_payload = {
            "schema": "rank_partitioned_top2_v1",
            "query": str(context.get("query_fingerprint") or ""),
            "content_state": content_state_fingerprint,
            "selected": metadata["candidate_fingerprints"],
            "displaced": metadata["displaced_fingerprints"],
        }
        action = {
            **provisional,
            "final_ranks": selected_ranks,
            "preserved_incumbent_rank": (
                rank_by_id[preserved_ids[0]] if len(preserved_ids) == 1 else 0
            ),
            "sources": sources,
            "expected_state_hash": state_hash,
            "state_version": state_version,
            "features": {name: _safe_float(features.get(name), 0.0) for name in names},
            "content_schema_version": "rank_partitioned_top2_v1",
            "query_fingerprint": str(context.get("query_fingerprint") or ""),
            "candidate_fingerprints": metadata["candidate_fingerprints"],
            "displaced_fingerprints": metadata["displaced_fingerprints"],
            "content_state_fingerprint": content_state_fingerprint,
            "action_fingerprint": hashlib.sha256(
                json.dumps(identity_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()[:24],
            "content_identity_complete": True,
            "semantic_identity_complete": True,
            "rank_partition_action": True,
            "top5_membership_invariant": True,
            "v224_rank_partitioned_action": True,
        }
        action["source_signature"] = base_module.normalized_action_source_signature(sources)
        action["provenance_signature"] = base_module.exact_action_source_signature(sources)
        action["source_risk_group"] = base_module.source_risk_group_key(action)
        result.append(action)
    return result


def _external_actions(
    query_plan_log: Dict[str, Any],
    *,
    base_module,
    max_single_candidates: int,
    max_pair_candidates: int,
    enforce_cq: bool,
) -> List[Dict[str, Any]]:
    names = feature_names(base_module)
    result = []
    for action in content_complete.generate_content_complete_post_physical_actions(
        query_plan_log,
        base_module=base_module,
        max_single_candidates=max_single_candidates,
        max_pair_candidates=max_pair_candidates,
        candidate_mode="content_complete_top2",
        enforce_cq=enforce_cq,
    ):
        features = {
            name: _safe_float((action.get("features", {}) or {}).get(name), 0.0)
            for name in names
        }
        result.append({
            **action,
            "features": features,
            "rank_partition_action": False,
            "top5_membership_invariant": False,
            "v224_rank_partitioned_action": False,
        })
    return result


def _annotate_action_pool(actions: List[Dict[str, Any]]) -> None:
    if not actions:
        return
    rank_count = sum(bool(action.get("rank_partition_action")) for action in actions)
    fraction = rank_count / len(actions)
    size_log = math.log1p(rank_count)
    proxies = [
        _safe_float((action.get("features", {}) or {}).get("semantic_comparator_proxy"))
        for action in actions
    ]
    mean_proxy = _mean(proxies)
    ordering = sorted(range(len(actions)), key=lambda index: proxies[index], reverse=True)
    rank_by_index = {index: rank for rank, index in enumerate(ordering, 1)}
    for index, action in enumerate(actions):
        features = action["features"]
        features["rank_partition_action_pool_fraction"] = fraction
        features["rank_partition_action_pool_size_log"] = size_log
        rank = rank_by_index[index]
        features["semantic_comparator_margin_to_query_mean"] = proxies[index] - mean_proxy
        features["semantic_comparator_rank_inv"] = 1.0 / rank
        features["semantic_comparator_is_query_best"] = float(rank == 1)
        features["local_comparator_proxy"] = proxies[index]
        features["local_comparator_margin_to_group_mean"] = proxies[index] - mean_proxy
        features["local_comparator_rank_inv"] = 1.0 / rank
        features["local_comparator_is_group_best"] = float(rank == 1)


def generate_rank_partitioned_post_physical_actions(
    query_plan_log: Dict[str, Any],
    *,
    base_module,
    max_single_candidates: int = 24,
    max_pair_candidates: int = 12,
    candidate_mode: str = "rank_partitioned_top2",
    enforce_cq: bool = True,
) -> List[Dict[str, Any]]:
    """Compile lossless rank actions, optionally alongside external admissions."""
    mode = str(candidate_mode or "").strip().lower()
    if mode not in RANK_PARTITIONED_MODES:
        return []
    actions: List[Dict[str, Any]] = []
    if mode != "rank_partitioned_external_only":
        actions.extend(_rank_partition_actions(query_plan_log, base_module=base_module))
    if mode in {"rank_partitioned_unified", "rank_partitioned_external_only"}:
        actions.extend(_external_actions(
            query_plan_log,
            base_module=base_module,
            max_single_candidates=max_single_candidates,
            max_pair_candidates=max_pair_candidates,
            enforce_cq=enforce_cq,
        ))



    deduplicated: List[Dict[str, Any]] = []
    seen = set()
    for action in actions:
        key = (
            str(action.get("action_kind") or ""),
            tuple(sorted(_safe_int(value) for value in action.get("passage_ids", []) or [])),
        )
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(action)
    _annotate_action_pool(deduplicated)
    return deduplicated
