"""Content-complete counterfactual evidence-set comparison.

The comparator enriches post-physical slate actions with query-local content
and dense-semantic evidence.  It compares each candidate against the exact
incumbent passage it would displace and rejects actions whose content identity
cannot be materialized.  Gold labels are attached only by the offline builder.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set

try:
    from . import query_plan_end_to_end_slate_optimizer as end_to_end
except ImportError:
    import query_plan_end_to_end_slate_optimizer as end_to_end


MODEL_TYPE = "content_complete_counterfactual_set_comparator"
CONTENT_COMPLETE_MODES = {"content_complete_top2", "content_complete_unified"}

V223_SEMANTIC_FEATURE_NAMES = [
    "candidate_semantic_score_available_fraction",
    "displaced_semantic_score_available_fraction",
    "candidate_semantic_score_max",
    "candidate_semantic_score_mean",
    "displaced_semantic_score_max",
    "displaced_semantic_score_mean",
    "candidate_minus_displaced_semantic_score",
    "candidate_semantic_better_fraction",
    "candidate_pair_semantic_min",
    "candidate_pair_semantic_spread",
    "candidate_fingerprint_unique_fraction",
    "candidate_displaced_content_collision",
    "content_complete_action",
    "semantic_comparator_proxy",
    "semantic_comparator_margin_to_query_mean",
    "semantic_comparator_rank_inv",
    "semantic_comparator_is_query_best",
]


def feature_names(base_module) -> List[str]:
    return [
        *end_to_end.feature_names(base_module),
        *V223_SEMANTIC_FEATURE_NAMES,
    ]


def content_feature_names(base_module) -> Set[str]:
    return {
        *base_module.V213_CANDIDATE_LOCAL_FEATURE_NAMES,
        *V223_SEMANTIC_FEATURE_NAMES,
    }


def is_content_complete_mode(candidate_mode: str) -> bool:
    return str(candidate_mode or "").strip().lower() in CONTENT_COMPLETE_MODES


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


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(_safe_float(row.get(key), 0.0) for row in rows) / len(rows)


def _maximum(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    return max((_safe_float(row.get(key), 0.0) for row in rows), default=0.0)


def _set_overlap(left: Iterable[int], right: Iterable[int]) -> float:
    left_set = set(left or [])
    right_set = set(right or [])
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / max(len(left_set), 1)


def _set_jaccard(left: Iterable[int], right: Iterable[int]) -> float:
    left_set = set(left or [])
    right_set = set(right or [])
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / max(len(left_set | right_set), 1)


def _displaced_ids(action: Mapping[str, Any], incumbent_ids: Sequence[int]) -> List[int]:
    explicit = [
        _safe_int(value)
        for value in action.get("displaced_incumbent_ids", []) or []
        if _safe_int(value) >= 0
    ]
    if explicit:
        return explicit
    action_kind = str(action.get("action_kind") or "")
    if action_kind == "replace_top5_slot5":
        return list(incumbent_ids[4:5])
    if action_kind == "replace_top2_pair":
        return list(incumbent_ids[:2])
    preserved = {
        _safe_int(value)
        for value in action.get("preserved_incumbent_ids", []) or []
        if _safe_int(value) >= 0
    }
    return [value for value in incumbent_ids[:2] if value not in preserved]


def _base_actions(
    query_plan_log: Dict[str, Any],
    *,
    base_module,
    max_single_candidates: int,
    max_pair_candidates: int,
    candidate_mode: str,
    enforce_cq: bool,
) -> List[Dict[str, Any]]:
    cached = query_plan_log.get("_v223_content_complete_base_actions")
    if isinstance(cached, list):
        return cached
    actions = end_to_end.generate_end_to_end_counterfactual_actions(
        query_plan_log,
        base_module=base_module,
        max_single_candidates=max_single_candidates,
        max_pair_candidates=max_pair_candidates,
        candidate_mode="unified",
        enforce_cq=enforce_cq,
        zero_physical_features=False,
        zero_cq_features=False,
    )
    if str(candidate_mode).strip().lower() == "content_complete_top2":
        actions = [
            action for action in actions
            if str(action.get("action_kind") or "") == "select_top2_subset"
        ]
    query_plan_log["_v223_content_complete_base_actions"] = actions
    return actions


def required_passage_ids(
    query_plan_log: Dict[str, Any],
    *,
    base_module,
    max_single_candidates: int = 24,
    max_pair_candidates: int = 12,
    candidate_mode: str = "content_complete_top2",
    enforce_cq: bool = True,
) -> Set[int]:
    """Return the bounded set whose content must be late-materialized."""
    incumbent = query_plan_log.get("candidate_admission_incumbent", {}) or {}
    incumbent_ids = [
        _safe_int(value)
        for value in incumbent.get("doc_ids", []) or []
        if _safe_int(value) >= 0
    ]
    result = set(incumbent_ids)
    for action in _base_actions(
        query_plan_log,
        base_module=base_module,
        max_single_candidates=max_single_candidates,
        max_pair_candidates=max_pair_candidates,
        candidate_mode=candidate_mode,
        enforce_cq=enforce_cq,
    ):
        for key in ("passage_ids", "candidate_ids", "preserved_incumbent_ids"):
            result.update(
                _safe_int(value)
                for value in action.get(key, []) or []
                if _safe_int(value) >= 0
            )
        result.update(_displaced_ids(action, incumbent_ids))
    return result


def _content_features(
    action: Mapping[str, Any],
    passage_context: Mapping[str, Any],
    incumbent_ids: Sequence[int],
) -> tuple[Dict[str, float], Dict[str, Any]]:
    def record(passage_idx: int) -> Mapping[str, Any]:
        value = passage_context.get(str(passage_idx), {}) or {}
        return value if isinstance(value, Mapping) else {}

    candidate_ids = [
        _safe_int(value)
        for value in action.get("candidate_ids", []) or []
        if _safe_int(value) >= 0
    ]
    displaced_ids = _displaced_ids(action, incumbent_ids)
    candidate_rows = [record(value) for value in candidate_ids]
    displaced_rows = [record(value) for value in displaced_ids]
    candidate_available = [row for row in candidate_rows if row.get("content_available")]
    displaced_available = [row for row in displaced_rows if row.get("content_available")]
    candidate_semantic = [
        row for row in candidate_rows if row.get("semantic_score_available")
    ]
    displaced_semantic = [
        row for row in displaced_rows if row.get("semantic_score_available")
    ]

    candidate_query_max = _maximum(candidate_rows, "query_coverage")
    displaced_query_max = _maximum(displaced_rows, "query_coverage")
    candidate_title_max = _maximum(candidate_rows, "title_query_coverage")
    displaced_title_max = _maximum(displaced_rows, "title_query_coverage")
    query_hashes = next(
        (
            row.get("query_token_hashes", []) or []
            for row in [*candidate_rows, *displaced_rows]
            if row.get("query_token_hashes")
        ),
        [],
    )
    candidate_union = set()
    for row in candidate_rows:
        candidate_union.update(row.get("token_hashes", []) or [])
    pair_union_coverage = _set_overlap(query_hashes, candidate_union)
    similarities = [
        _set_jaccard(
            candidate.get("token_hashes", []) or [],
            displaced.get("token_hashes", []) or [],
        )
        for candidate in candidate_rows
        for displaced in displaced_rows
    ]
    max_similarity = max(similarities, default=0.0)
    candidate_semantic_values = [
        _safe_float(row.get("semantic_score")) for row in candidate_semantic
    ]
    displaced_semantic_values = [
        _safe_float(row.get("semantic_score")) for row in displaced_semantic
    ]
    candidate_semantic_max = max(candidate_semantic_values, default=0.0)
    displaced_semantic_max = max(displaced_semantic_values, default=0.0)
    candidate_fingerprints = [
        str(row.get("content_fingerprint") or "") for row in candidate_rows
    ]
    displaced_fingerprints = [
        str(row.get("content_fingerprint") or "") for row in displaced_rows
    ]
    nonempty_candidate_fingerprints = [
        value for value in candidate_fingerprints if value
    ]
    identity_complete = bool(candidate_rows and displaced_rows) and (
        len(candidate_available) == len(candidate_rows)
        and len(displaced_available) == len(displaced_rows)
    )
    semantic_complete = bool(candidate_rows and displaced_rows) and (
        len(candidate_semantic) == len(candidate_rows)
        and len(displaced_semantic) == len(displaced_rows)
    )
    semantic_margin = candidate_semantic_max - displaced_semantic_max
    lexical_margin = candidate_query_max - displaced_query_max
    semantic_proxy = (
        semantic_margin
        + 0.35 * lexical_margin
        + 0.20 * (candidate_title_max - displaced_title_max)
        + 0.15 * max(pair_union_coverage - candidate_query_max, 0.0)
        - 0.10 * max_similarity
    )
    features = {
        "candidate_content_available_fraction": len(candidate_available)
        / max(len(candidate_rows), 1),
        "displaced_content_available_fraction": len(displaced_available)
        / max(len(displaced_rows), 1),
        "candidate_query_coverage_max": candidate_query_max,
        "candidate_query_coverage_mean": _mean(candidate_rows, "query_coverage"),
        "candidate_title_query_coverage_max": candidate_title_max,
        "candidate_title_query_coverage_mean": _mean(
            candidate_rows, "title_query_coverage"
        ),
        "candidate_query_bigram_coverage_max": _maximum(
            candidate_rows, "query_bigram_coverage"
        ),
        "displaced_query_coverage_max": displaced_query_max,
        "displaced_title_query_coverage_max": displaced_title_max,
        "candidate_minus_displaced_query_coverage": lexical_margin,
        "candidate_minus_displaced_title_coverage": (
            candidate_title_max - displaced_title_max
        ),
        "candidate_displaced_token_jaccard_max": max_similarity,
        "candidate_displaced_token_novelty_min": 1.0 - max_similarity,
        "candidate_pair_query_union_coverage": pair_union_coverage,
        "candidate_pair_complementarity": max(
            pair_union_coverage - candidate_query_max, 0.0
        ),
        "candidate_text_length_log_mean": _mean(candidate_rows, "token_count_log"),
        "displaced_text_length_log_mean": _mean(
            displaced_rows, "token_count_log"
        ),
        "content_identity_complete": float(identity_complete),
        "candidate_semantic_score_available_fraction": len(candidate_semantic)
        / max(len(candidate_rows), 1),
        "displaced_semantic_score_available_fraction": len(displaced_semantic)
        / max(len(displaced_rows), 1),
        "candidate_semantic_score_max": candidate_semantic_max,
        "candidate_semantic_score_mean": (
            sum(candidate_semantic_values) / len(candidate_semantic_values)
            if candidate_semantic_values else 0.0
        ),
        "displaced_semantic_score_max": displaced_semantic_max,
        "displaced_semantic_score_mean": (
            sum(displaced_semantic_values) / len(displaced_semantic_values)
            if displaced_semantic_values else 0.0
        ),
        "candidate_minus_displaced_semantic_score": semantic_margin,
        "candidate_semantic_better_fraction": (
            sum(
                candidate > displaced
                for candidate in candidate_semantic_values
                for displaced in displaced_semantic_values
            )
            / max(len(candidate_semantic_values) * len(displaced_semantic_values), 1)
        ),
        "candidate_pair_semantic_min": min(candidate_semantic_values, default=0.0),
        "candidate_pair_semantic_spread": (
            max(candidate_semantic_values, default=0.0)
            - min(candidate_semantic_values, default=0.0)
        ),
        "candidate_fingerprint_unique_fraction": len(
            set(nonempty_candidate_fingerprints)
        ) / max(len(candidate_fingerprints), 1),
        "candidate_displaced_content_collision": float(
            bool(
                set(nonempty_candidate_fingerprints)
                & {value for value in displaced_fingerprints if value}
            )
        ),
        "content_complete_action": float(identity_complete and semantic_complete),
        "semantic_comparator_proxy": semantic_proxy,
        "semantic_comparator_margin_to_query_mean": 0.0,
        "semantic_comparator_rank_inv": 0.0,
        "semantic_comparator_is_query_best": 0.0,
    }
    metadata = {
        "candidate_ids": candidate_ids,
        "displaced_ids": displaced_ids,
        "candidate_fingerprints": candidate_fingerprints,
        "displaced_fingerprints": displaced_fingerprints,
        "identity_complete": identity_complete,
        "semantic_complete": semantic_complete,
        "semantic_proxy": semantic_proxy,
    }
    return features, metadata


def generate_content_complete_post_physical_actions(
    query_plan_log: Dict[str, Any],
    *,
    base_module,
    max_single_candidates: int = 24,
    max_pair_candidates: int = 12,
    candidate_mode: str = "content_complete_top2",
    enforce_cq: bool = True,
) -> List[Dict[str, Any]]:
    """Compile actions that compare content against the exact incumbent."""
    context = query_plan_log.get("candidate_local_context", {}) or {}
    passage_context = context.get("passages", {}) or {}
    incumbent = query_plan_log.get("candidate_admission_incumbent", {}) or {}
    incumbent_ids = [
        _safe_int(value)
        for value in incumbent.get("doc_ids", []) or []
        if _safe_int(value) >= 0
    ]
    names = feature_names(base_module)
    enriched: List[Dict[str, Any]] = []
    for action in _base_actions(
        query_plan_log,
        base_module=base_module,
        max_single_candidates=max_single_candidates,
        max_pair_candidates=max_pair_candidates,
        candidate_mode=candidate_mode,
        enforce_cq=enforce_cq,
    ):
        local, metadata = _content_features(action, passage_context, incumbent_ids)
        if not metadata["identity_complete"] or not metadata["semantic_complete"]:
            continue
        features = {
            name: _safe_float((action.get("features", {}) or {}).get(name), 0.0)
            for name in names
        }
        features.update(local)
        incumbent_fingerprints = [
            str((passage_context.get(str(value), {}) or {}).get("content_fingerprint") or "")
            for value in incumbent_ids[:5]
        ]
        content_state_fingerprint = hashlib.sha256(
            json.dumps(incumbent_fingerprints, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:24]
        identity_payload = {
            "schema": "content_complete_post_physical_v1",
            "query": str(context.get("query_fingerprint") or ""),
            "content_state": content_state_fingerprint,
            "kind": str(action.get("action_kind") or ""),
            "preserved_rank": _safe_int(action.get("preserved_incumbent_rank"), 0),
            "candidates": sorted(metadata["candidate_fingerprints"]),
            "displaced": sorted(metadata["displaced_fingerprints"]),
        }
        enriched.append({
            **action,
            "features": {name: _safe_float(features.get(name), 0.0) for name in names},
            "content_schema_version": "content_complete_post_physical_v1",
            "query_fingerprint": str(context.get("query_fingerprint") or ""),
            "candidate_fingerprints": metadata["candidate_fingerprints"],
            "displaced_incumbent_ids": metadata["displaced_ids"],
            "displaced_fingerprints": metadata["displaced_fingerprints"],
            "content_state_fingerprint": content_state_fingerprint,
            "action_fingerprint": hashlib.sha256(
                json.dumps(
                    identity_payload, sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
            ).hexdigest()[:24],
            "content_identity_complete": True,
            "semantic_identity_complete": True,
            "v223_content_complete_action": True,
        })

    if enriched:
        proxies = [
            _safe_float(action["features"].get("semantic_comparator_proxy"))
            for action in enriched
        ]
        mean_proxy = sum(proxies) / len(proxies)
        ordering = sorted(
            range(len(enriched)), key=lambda index: proxies[index], reverse=True
        )
        rank_by_index = {index: rank for rank, index in enumerate(ordering, 1)}
        for index, action in enumerate(enriched):
            features = action["features"]
            rank = rank_by_index[index]
            features["semantic_comparator_margin_to_query_mean"] = (
                proxies[index] - mean_proxy
            )
            features["semantic_comparator_rank_inv"] = 1.0 / rank
            features["semantic_comparator_is_query_best"] = float(rank == 1)

            features["local_comparator_proxy"] = proxies[index]
            features["local_comparator_margin_to_group_mean"] = (
                proxies[index] - mean_proxy
            )
            features["local_comparator_rank_inv"] = 1.0 / rank
            features["local_comparator_is_group_best"] = float(rank == 1)
    return enriched
