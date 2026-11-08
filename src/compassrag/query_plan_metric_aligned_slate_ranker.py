"""Metric-aligned candidate-window reranking for ontology-guided retrieval.

The model is trained only from historical-query counterfactual traces.  At
inference time it consumes retrieval, content, and executable-ontology
features; gold labels are neither accepted nor inspected here.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


FEATURE_NAMES = [
    "rank_inv",
    "rank_log",
    "rank_le_2",
    "rank_le_5",
    "rank_le_10",
    "rank_le_20",
    "retrieval_score_norm",
    "retrieval_score_gap",
    "semantic_score",
    "query_coverage",
    "title_query_coverage",
    "query_bigram_coverage",
    "token_count_log",
    "anchor_token_coverage",
    "relation_token_coverage",
    "ontology_candidate",
    "ontology_raw_score",
    "ontology_relation_score",
    "ontology_utility",
    "ontology_exact_entity_key",
    "ontology_predicate_coverage",
    "ontology_contract_confidence",
    "complete_certificate_member",
    "lifted_passage",
]

BGE_BOUNDARY_FEATURE_NAMES = FEATURE_NAMES + [
    "bge_relevance",
    "bge_rank_inv",
    "bge_retrieval_agreement",
    "bge_ontology_support",
]

ACTION_FEATURE_NAMES = [
    "action_top2",
    "action_top5",
    "changes_top2",
    "changes_top5_membership",
    "selected_rank2_original_rank_log",
    "added_count",
    "added_rank_mean_log",
    "added_rank_max_log",
    "top2_probability_gain",
    "top5_probability_mean_gain",
    "top5_probability_min_gain",
    "candidate_top2_probability_mean",
    "candidate_top5_probability_mean",
    "incumbent_top2_probability_mean",
    "incumbent_top5_probability_mean",
    "candidate_top5_ontology_rate",
    "ontology_rate_gain",
    "candidate_top5_certificate_rate",
    "certificate_rate_gain",
    "candidate_top5_query_coverage_mean",
    "query_coverage_gain",
    "candidate_top5_semantic_mean",
    "semantic_gain",
]

TOP5_COMPLETION_FEATURE_NAMES = [
    "candidate_original_rank_log",
    "removed_original_rank_log",
    "rank_distance_log",
    "candidate_probability",
    "removed_probability",
    "probability_gain",
    "candidate_bge_relevance",
    "removed_bge_relevance",
    "bge_gain",
    "top5_probability_mean_gain",
    "top5_probability_min_gain",
    "candidate_novelty",
    "candidate_ontology_candidate",
    "ontology_candidate_gain",
    "candidate_ontology_relation_score",
    "ontology_relation_score_gain",
    "candidate_ontology_utility",
    "ontology_utility_gain",
    "candidate_exact_entity_key",
    "exact_entity_key_gain",
    "candidate_predicate_coverage",
    "predicate_coverage_gain",
    "candidate_contract_confidence",
    "contract_confidence_gain",
    "candidate_certificate_member",
    "certificate_member_gain",
    "candidate_lifted_passage",
    "lifted_passage_gain",
    "candidate_query_coverage",
    "query_coverage_gain",
    "candidate_anchor_coverage",
    "anchor_coverage_gain",
    "candidate_relation_coverage",
    "relation_coverage_gain",
    "candidate_semantic_score",
    "semantic_score_gain",
    "candidate_proof_support",
    "removed_proof_support",
    "proof_support_gain",
    "current_top5_proof_support_mean",
]


def set_level_query_key(query: str) -> str:
    """Return a stable, content-addressed key for verifier sidecars."""
    normalized = " ".join(str(query or "").strip().lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _tokens(value: Any) -> List[str]:
    current = []
    token = []
    for char in str(value or "").lower():
        if char.isalnum():
            token.append(char)
        elif token:
            current.append("".join(token))
            token = []
    if token:
        current.append("".join(token))
    return current


def _token_hashes(values: Iterable[Any]) -> set[int]:
    tokens: set[str] = set()
    for value in values:
        tokens.update(_tokens(value))
    return {
        int(hashlib.sha1(token.encode("utf-8")).hexdigest()[:8], 16)
        for token in tokens
    }


def _coverage(required: set[int], observed: Iterable[Any]) -> float:
    observed_set = {_safe_int(value) for value in observed}
    if not required:
        return 0.0
    return len(required & observed_set) / len(required)


def _passage_signal_maps(query_plan_log: Mapping[str, Any]) -> Tuple[Dict[int, Dict[str, float]], set[int], set[int]]:
    factorized = dict(query_plan_log.get("factorized_query_executor", {}) or {})
    candidate_signals: Dict[int, Dict[str, float]] = {}
    for candidate in factorized.get("candidates", []) or []:
        if not isinstance(candidate, Mapping):
            continue
        passage_idx = _safe_int(candidate.get("passage_idx"))
        if passage_idx < 0:
            continue
        previous = candidate_signals.setdefault(passage_idx, {})
        fields = {
            "ontology_raw_score": "raw_score",
            "ontology_relation_score": "relation_score",
            "ontology_utility": "utility",
            "ontology_predicate_coverage": "query_predicate_coverage",
            "ontology_contract_confidence": "typed_contract_confidence",
        }
        for output_name, source_name in fields.items():
            previous[output_name] = max(
                previous.get(output_name, 0.0),
                _safe_float(candidate.get(source_name)),
            )
        previous["ontology_exact_entity_key"] = max(
            previous.get("ontology_exact_entity_key", 0.0),
            float(bool(candidate.get("exact_entity_key", False))),
        )

    complete_members: set[int] = set()
    for group in factorized.get("groups", []) or []:
        if not isinstance(group, Mapping):
            continue
        certificate = dict(group.get("binding_certificate", {}) or {})
        if not bool(certificate.get("complete", False)):
            continue
        for key in ("passage_ids", "member_passage_ids"):
            for value in group.get(key, []) or []:
                passage_idx = _safe_int(value)
                if passage_idx >= 0:
                    complete_members.add(passage_idx)
        for item in group.get("members", []) or []:
            if isinstance(item, Mapping):
                passage_idx = _safe_int(item.get("passage_idx"))
                if passage_idx >= 0:
                    complete_members.add(passage_idx)

    lifted_members = {
        _safe_int(item.get("passage_idx"))
        for item in query_plan_log.get("lifted_passages", []) or []
        if isinstance(item, Mapping) and _safe_int(item.get("passage_idx")) >= 0
    }
    return candidate_signals, complete_members, lifted_members


def extract_metric_aligned_features(
    query_plan_log: Mapping[str, Any],
    passage_idx: int,
    rank: int,
    score: float,
    max_score: float,
    signal_maps: Tuple[Dict[int, Dict[str, float]], set[int], set[int]] | None = None,
) -> Dict[str, float]:
    """Build the exact feature vector shared by historical training and inference."""
    rank = max(int(rank), 1)
    local_context = dict(query_plan_log.get("candidate_local_context", {}) or {})
    passage_record = dict(
        (local_context.get("passages", {}) or {}).get(str(int(passage_idx)), {}) or {}
    )
    factorized = dict(query_plan_log.get("factorized_query_executor", {}) or {})
    logical_plan = dict(factorized.get("logical_evidence_plan", {}) or {})
    anchor_hashes = _token_hashes(logical_plan.get("anchors", []) or [])
    relation_hashes = _token_hashes(logical_plan.get("relations", []) or [])
    passage_hashes = passage_record.get("token_hashes", []) or []
    if signal_maps is None:
        signal_maps = _passage_signal_maps(query_plan_log)
    candidates, complete_members, lifted_members = signal_maps
    ontology = candidates.get(int(passage_idx), {})
    normalized_score = score / max_score if max_score > 0.0 else 0.0
    return {
        "rank_inv": 1.0 / rank,
        "rank_log": math.log1p(rank),
        "rank_le_2": float(rank <= 2),
        "rank_le_5": float(rank <= 5),
        "rank_le_10": float(rank <= 10),
        "rank_le_20": float(rank <= 20),
        "retrieval_score_norm": normalized_score,
        "retrieval_score_gap": max(0.0, 1.0 - normalized_score),
        "semantic_score": _safe_float(passage_record.get("semantic_score")),
        "query_coverage": _safe_float(passage_record.get("query_coverage")),
        "title_query_coverage": _safe_float(passage_record.get("title_query_coverage")),
        "query_bigram_coverage": _safe_float(passage_record.get("query_bigram_coverage")),
        "token_count_log": _safe_float(passage_record.get("token_count_log")),
        "anchor_token_coverage": _coverage(anchor_hashes, passage_hashes),
        "relation_token_coverage": _coverage(relation_hashes, passage_hashes),
        "ontology_candidate": float(int(passage_idx) in candidates),
        "ontology_raw_score": _safe_float(ontology.get("ontology_raw_score")),
        "ontology_relation_score": _safe_float(ontology.get("ontology_relation_score")),
        "ontology_utility": _safe_float(ontology.get("ontology_utility")),
        "ontology_exact_entity_key": _safe_float(ontology.get("ontology_exact_entity_key")),
        "ontology_predicate_coverage": _safe_float(ontology.get("ontology_predicate_coverage")),
        "ontology_contract_confidence": _safe_float(ontology.get("ontology_contract_confidence")),
        "complete_certificate_member": float(int(passage_idx) in complete_members),
        "lifted_passage": float(int(passage_idx) in lifted_members),
    }


def feature_vector(features: Mapping[str, Any]) -> List[float]:
    return [_safe_float(features.get(name)) for name in FEATURE_NAMES]


def extend_bge_boundary_features(
    features: Mapping[str, Any],
    bge_relevance: float,
    bge_rank: int,
) -> Dict[str, float]:
    """Add semantic cross-encoder and structure/semantic interaction signals."""
    current = {name: _safe_float(features.get(name)) for name in FEATURE_NAMES}
    relevance = min(max(_safe_float(bge_relevance), 0.0), 1.0)
    ontology_support = max(
        current.get("ontology_candidate", 0.0),
        current.get("complete_certificate_member", 0.0),
        current.get("ontology_contract_confidence", 0.0),
    )
    current.update({
        "bge_relevance": relevance,
        "bge_rank_inv": 1.0 / max(int(bge_rank), 1),
        "bge_retrieval_agreement": 1.0 - abs(
            relevance - current.get("retrieval_score_norm", 0.0)
        ),
        "bge_ontology_support": relevance * ontology_support,
    })
    return current


def bge_boundary_feature_vector(features: Mapping[str, Any]) -> List[float]:
    return [_safe_float(features.get(name)) for name in BGE_BOUNDARY_FEATURE_NAMES]


def action_feature_vector(features: Mapping[str, Any]) -> List[float]:
    return [_safe_float(features.get(name)) for name in ACTION_FEATURE_NAMES]


def top5_completion_feature_vector(features: Mapping[str, Any]) -> List[float]:
    return [_safe_float(features.get(name)) for name in TOP5_COMPLETION_FEATURE_NAMES]


def _text_token_set(value: Any) -> set[str]:
    return set(_tokens(value))


def _token_jaccard(left: Any, right: Any) -> float:
    left_tokens = _text_token_set(left)
    right_tokens = _text_token_set(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _proof_support(features: Mapping[str, Any]) -> float:
    return max(
        _safe_float(features.get("relation_token_coverage")),
        _safe_float(features.get("ontology_predicate_coverage")),
        _safe_float(features.get("ontology_relation_score")),
        _safe_float(features.get("ontology_contract_confidence")),
        _safe_float(features.get("complete_certificate_member")),
    )


def extract_top5_replacement_features(
    doc_ids: Sequence[int],
    current_slate: Sequence[int],
    probabilities: Sequence[float],
    passage_features: Mapping[int, Mapping[str, Any]],
    passage_texts: Mapping[int, str],
    removed_id: int,
    candidate_id: int,
) -> Dict[str, float]:
    """Describe one Top-5 replacement using only gold-free runtime signals."""
    original = [int(value) for value in doc_ids]
    current_top5 = [int(value) for value in current_slate[:5]]
    removed_id = int(removed_id)
    candidate_id = int(candidate_id)
    probability_by_id = {
        passage_idx: _safe_float(probabilities[index])
        for index, passage_idx in enumerate(original[:len(probabilities)])
    }
    rank_by_id = {passage_idx: index + 1 for index, passage_idx in enumerate(original)}
    removed = dict(passage_features.get(removed_id, {}) or {})
    candidate = dict(passage_features.get(candidate_id, {}) or {})
    kept = [value for value in current_top5 if value != removed_id]
    replacement = [candidate_id if value == removed_id else value for value in current_top5]

    def mean(values: Sequence[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    def probability_mean(values: Sequence[int]) -> float:
        return mean([probability_by_id.get(value, 0.0) for value in values])

    def probability_min(values: Sequence[int]) -> float:
        scores = [probability_by_id.get(value, 0.0) for value in values]
        return min(scores) if scores else 0.0

    candidate_probability = probability_by_id.get(candidate_id, 0.0)
    removed_probability = probability_by_id.get(removed_id, 0.0)
    candidate_proof = _proof_support(candidate)
    removed_proof = _proof_support(removed)
    kept_texts = [passage_texts.get(value, "") for value in kept]
    max_similarity = max(
        (_token_jaccard(passage_texts.get(candidate_id, ""), text) for text in kept_texts),
        default=0.0,
    )

    def field(name: str) -> Tuple[float, float]:
        return _safe_float(candidate.get(name)), _safe_float(removed.get(name))

    ontology_candidate, removed_ontology_candidate = field("ontology_candidate")
    relation_score, removed_relation_score = field("ontology_relation_score")
    ontology_utility, removed_ontology_utility = field("ontology_utility")
    exact_key, removed_exact_key = field("ontology_exact_entity_key")
    predicate, removed_predicate = field("ontology_predicate_coverage")
    contract, removed_contract = field("ontology_contract_confidence")
    certificate, removed_certificate = field("complete_certificate_member")
    lifted, removed_lifted = field("lifted_passage")
    query_coverage, removed_query_coverage = field("query_coverage")
    anchor_coverage, removed_anchor_coverage = field("anchor_token_coverage")
    relation_coverage, removed_relation_coverage = field("relation_token_coverage")
    semantic, removed_semantic = field("semantic_score")
    bge, removed_bge = field("bge_relevance")
    return {
        "candidate_original_rank_log": math.log1p(rank_by_id.get(candidate_id, len(original) + 1)),
        "removed_original_rank_log": math.log1p(rank_by_id.get(removed_id, len(original) + 1)),
        "rank_distance_log": math.log1p(abs(rank_by_id.get(candidate_id, 0) - rank_by_id.get(removed_id, 0))),
        "candidate_probability": candidate_probability,
        "removed_probability": removed_probability,
        "probability_gain": candidate_probability - removed_probability,
        "candidate_bge_relevance": bge,
        "removed_bge_relevance": removed_bge,
        "bge_gain": bge - removed_bge,
        "top5_probability_mean_gain": probability_mean(replacement) - probability_mean(current_top5),
        "top5_probability_min_gain": probability_min(replacement) - probability_min(current_top5),
        "candidate_novelty": 1.0 - max_similarity,
        "candidate_ontology_candidate": ontology_candidate,
        "ontology_candidate_gain": ontology_candidate - removed_ontology_candidate,
        "candidate_ontology_relation_score": relation_score,
        "ontology_relation_score_gain": relation_score - removed_relation_score,
        "candidate_ontology_utility": ontology_utility,
        "ontology_utility_gain": ontology_utility - removed_ontology_utility,
        "candidate_exact_entity_key": exact_key,
        "exact_entity_key_gain": exact_key - removed_exact_key,
        "candidate_predicate_coverage": predicate,
        "predicate_coverage_gain": predicate - removed_predicate,
        "candidate_contract_confidence": contract,
        "contract_confidence_gain": contract - removed_contract,
        "candidate_certificate_member": certificate,
        "certificate_member_gain": certificate - removed_certificate,
        "candidate_lifted_passage": lifted,
        "lifted_passage_gain": lifted - removed_lifted,
        "candidate_query_coverage": query_coverage,
        "query_coverage_gain": query_coverage - removed_query_coverage,
        "candidate_anchor_coverage": anchor_coverage,
        "anchor_coverage_gain": anchor_coverage - removed_anchor_coverage,
        "candidate_relation_coverage": relation_coverage,
        "relation_coverage_gain": relation_coverage - removed_relation_coverage,
        "candidate_semantic_score": semantic,
        "semantic_score_gain": semantic - removed_semantic,
        "candidate_proof_support": candidate_proof,
        "removed_proof_support": removed_proof,
        "proof_support_gain": candidate_proof - removed_proof,
        "current_top5_proof_support_mean": mean([
            _proof_support(passage_features.get(value, {}) or {}) for value in current_top5
        ]),
    }


def policy_score(probability: float, rank: int, rank_penalty: float) -> float:
    probability = min(max(float(probability), 1e-6), 1.0 - 1e-6)
    return math.log(probability / (1.0 - probability)) - float(rank_penalty) * math.log(max(rank, 1))


def materialize_metric_aligned_slate(
    doc_ids: Sequence[int],
    probabilities: Sequence[float],
    *,
    rank_window: int,
    rank_penalty: float,
    top2_margin: float,
    top5_margin: float,
) -> Tuple[List[int], Dict[str, Any]]:
    """Apply a calibrated hierarchical Top-2/Top-5 policy with an explicit no-op."""
    original = [int(value) for value in doc_ids]
    if len(original) < 5:
        return original, {"changed": False, "reason": "fewer_than_five_candidates"}
    window = min(max(int(rank_window), 5), len(original), len(probabilities))
    scores = {
        original[index]: policy_score(probabilities[index], index + 1, rank_penalty)
        for index in range(window)
    }
    rank_by_id = {passage_idx: index + 1 for index, passage_idx in enumerate(original)}
    fixed_rank1 = original[0]
    incumbent_rank2 = original[1]
    eligible_top2 = original[1:window]
    selected_rank2 = max(eligible_top2, key=lambda value: scores[value])
    if scores[selected_rank2] - scores[incumbent_rank2] < float(top2_margin):
        selected_rank2 = incumbent_rank2

    selected = [fixed_rank1, selected_rank2]
    incumbent_tail = [value for value in original[:5] if value not in selected]
    while len(incumbent_tail) < 3:
        for value in original:
            if value not in selected and value not in incumbent_tail:
                incumbent_tail.append(value)
            if len(incumbent_tail) >= 3:
                break
    tail = incumbent_tail[:3]
    challengers = [
        value for value in original[2:window]
        if value not in selected and value not in tail
    ]
    challengers.sort(key=lambda value: scores[value], reverse=True)
    for challenger in challengers:
        weakest = min(tail, key=lambda value: scores.get(value, -1e9))
        if scores[challenger] - scores.get(weakest, -1e9) < float(top5_margin):
            break
        tail[tail.index(weakest)] = challenger
    tail.sort(key=lambda value: scores.get(value, -1e9), reverse=True)
    top5 = list(dict.fromkeys([fixed_rank1, selected_rank2, *tail]))
    for value in original:
        if len(top5) >= 5:
            break
        if value not in top5:
            top5.append(value)
    reranked = top5 + [value for value in original if value not in set(top5)]
    return reranked, {
        "changed": reranked[:5] != original[:5],
        "changed_top2": reranked[:2] != original[:2],
        "changed_top5_membership": set(reranked[:5]) != set(original[:5]),
        "before_top5": original[:5],
        "after_top5": reranked[:5],
        "selected_rank2_original_rank": rank_by_id.get(selected_rank2, 0),
    }


def materialize_metric_aligned_actions(
    doc_ids: Sequence[int],
    probabilities: Sequence[float],
    *,
    rank_window: int,
    rank_penalty: float,
    top2_margin: float,
    top5_margin: float,
) -> Dict[str, List[int]]:
    """Build complete Top-5 actions; no-op remains an explicit alternative."""
    original = [int(value) for value in doc_ids]
    full, _ = materialize_metric_aligned_slate(
        original,
        probabilities,
        rank_window=rank_window,
        rank_penalty=rank_penalty,
        top2_margin=top2_margin,
        top5_margin=top5_margin,
    )
    actions: Dict[str, List[int]] = {"no_op": original}
    if len(original) < 5:
        return actions

    selected_rank2 = full[1]
    top2_head = [original[0], selected_rank2]
    top2_top5 = top2_head + [
        value for value in original[:5] if value not in set(top2_head)
    ]
    top2 = top2_top5[:5] + [value for value in original if value not in set(top2_top5[:5])]
    if top2 != original:
        actions["top2"] = top2
    if full != original:
        actions["top5"] = full
    return actions


def materialize_bge_top2_incumbent_top5(
    doc_ids: Sequence[int],
    bge_slate: Sequence[int],
    incumbent_slate: Sequence[int],
) -> Tuple[List[int], Dict[str, Any]]:
    """Keep the BGE Top-2 while completing the slate in incumbent order."""
    original = [int(value) for value in doc_ids]
    bge = [int(value) for value in bge_slate]
    incumbent = [int(value) for value in incumbent_slate]
    if len(original) < 5 or len(bge) < 2:
        return original, {"changed": False, "reason": "fewer_than_five_candidates"}
    top5 = list(dict.fromkeys(bge[:2]))
    for source in (incumbent[:5], incumbent, original):
        for value in source:
            if value not in top5:
                top5.append(value)
            if len(top5) >= 5:
                break
        if len(top5) >= 5:
            break
    reranked = top5[:5] + [value for value in incumbent if value not in set(top5[:5])]
    reranked.extend(value for value in original if value not in set(reranked))
    return reranked, {
        "changed": reranked[:5] != original[:5],
        "changed_top2": reranked[:2] != original[:2],
        "changed_top5_membership": set(reranked[:5]) != set(original[:5]),
        "before_top5": original[:5],
        "after_top5": reranked[:5],
        "bge_top2": bge[:2],
        "incumbent_top5": incumbent[:5],
        "completion_mode": "bge_top2_incumbent_top5",
    }


def materialize_frozen_incumbent_top5(
    doc_ids: Sequence[int],
    probabilities: Sequence[float],
    frozen_top5: Sequence[int],
    *,
    probability_shortlist_size: int = 8,
) -> Tuple[List[int], Dict[str, Any]]:
    """Select Top-2 inside a frozen strong Top-5 without changing membership."""
    original = [int(value) for value in doc_ids]
    incumbent = list(dict.fromkeys(int(value) for value in frozen_top5))
    if len(incumbent) != 5:
        return original, {
            "enabled": True,
            "changed": False,
            "reason": "frozen_incumbent_must_contain_five_unique_passages",
        }

    scored = [
        (int(original[index]), float(probabilities[index]))
        for index in range(min(len(original), len(probabilities)))
    ]
    scored.sort(key=lambda item: item[1], reverse=True)
    shortlist = {
        passage_idx: probability
        for passage_idx, probability in scored[: max(int(probability_shortlist_size), 2)]
        if passage_idx in set(incumbent)
    }
    ranked_incumbent = sorted(
        incumbent,
        key=lambda passage_idx: (
            passage_idx in shortlist,
            shortlist.get(passage_idx, float("-inf")),
            -incumbent.index(passage_idx),
        ),
        reverse=True,
    )
    top2 = ranked_incumbent[:2]
    committed_top5 = top2 + [
        passage_idx for passage_idx in incumbent if passage_idx not in set(top2)
    ]
    reranked = committed_top5 + [
        passage_idx for passage_idx in original if passage_idx not in set(committed_top5)
    ]
    return reranked, {
        "enabled": True,
        "changed": reranked[:5] != original[:5],
        "changed_top2": reranked[:2] != original[:2],
        "changed_top5_membership": set(reranked[:5]) != set(original[:5]),
        "frozen_membership_preserved": set(reranked[:5]) == set(incumbent),
        "before_top5": original[:5],
        "after_top5": reranked[:5],
        "frozen_incumbent_top5": incumbent,
        "probability_shortlist_size": max(int(probability_shortlist_size), 2),
        "scored_incumbent_passages": [
            {
                "passage_idx": passage_idx,
                "gold_free_probability": round(probability, 6),
            }
            for passage_idx, probability in scored[: max(int(probability_shortlist_size), 2)]
            if passage_idx in set(incumbent)
        ],
        "completion_mode": "learned_top2_within_frozen_strong_top5",
    }


def materialize_proof_aware_top5_completion(
    doc_ids: Sequence[int],
    current_slate: Sequence[int],
    probabilities: Sequence[float],
    passage_features: Mapping[int, Mapping[str, Any]],
    passage_texts: Mapping[int, str],
    models: Mapping[str, Any],
    policy: Mapping[str, Any],
    *,
    rank_window: int,
) -> Tuple[List[int], Dict[str, Any]]:
    """Select at most one safe tail replacement while preserving Top-2."""
    original = [int(value) for value in doc_ids]
    current = [int(value) for value in current_slate]
    if len(current) < 5:
        return current, {"changed": False, "reason": "fewer_than_five_candidates"}
    window = min(max(int(rank_window), 5), len(original), len(probabilities))
    top5 = current[:5]
    candidates = [value for value in original[:window] if value not in set(top5)]
    configured_positions = policy.get("allowed_removed_positions", [2, 3, 4])
    allowed_positions = {
        int(value)
        for value in (configured_positions or [])
        if 2 <= int(value) <= 4
    }
    if not allowed_positions:
        return current, {"changed": False, "reason": "no_allowed_tail_positions"}
    actions: List[Dict[str, Any]] = []
    rows: List[List[float]] = []
    for removed_position in sorted(allowed_positions):
        removed_id = top5[removed_position]
        for candidate_id in candidates:
            features = extract_top5_replacement_features(
                original,
                current,
                probabilities,
                passage_features,
                passage_texts,
                removed_id,
                candidate_id,
            )
            rows.append(top5_completion_feature_vector(features))
            actions.append({
                "removed_position": removed_position,
                "removed_id": removed_id,
                "candidate_id": candidate_id,
                "features": features,
            })
    if not rows:
        return current, {"changed": False, "reason": "no_replacement_candidates"}

    predicted_delta = models["delta"].predict(rows)
    predicted_gain = MetricAlignedSlateRanker._risk_probability(models["gain"], rows)
    predicted_risk = MetricAlignedSlateRanker._risk_probability(models["risk"], rows)
    gain_weight = _safe_float(policy.get("gain_weight"), 0.10)
    risk_weight = _safe_float(policy.get("risk_weight"), 0.25)
    min_delta = _safe_float(policy.get("min_predicted_delta"), 0.0)
    min_gain_probability = _safe_float(policy.get("min_gain_probability"), 0.5)
    max_risk_probability = _safe_float(policy.get("max_risk_probability"), 0.25)
    min_proof_gain = _safe_float(policy.get("min_proof_support_gain"), -1.0)
    min_candidate_proof = _safe_float(
        policy.get("min_candidate_proof_support"), 0.0
    )
    min_candidate_novelty = _safe_float(policy.get("min_candidate_novelty"), 0.0)
    min_probability_gain = _safe_float(policy.get("min_probability_gain"), -1.0)
    min_bge_gain = _safe_float(policy.get("min_bge_gain"), -1.0)
    predictions = []
    for index, action in enumerate(actions):
        delta = float(predicted_delta[index])
        gain = float(predicted_gain[index])
        risk = float(predicted_risk[index])
        objective = delta + gain_weight * gain - risk_weight * risk
        admissible = bool(
            delta >= min_delta
            and gain >= min_gain_probability
            and risk <= max_risk_probability
            and _safe_float(action["features"].get("proof_support_gain"))
            >= min_proof_gain
            and _safe_float(action["features"].get("candidate_proof_support"))
            >= min_candidate_proof
            and _safe_float(action["features"].get("candidate_novelty"))
            >= min_candidate_novelty
            and _safe_float(action["features"].get("probability_gain"))
            >= min_probability_gain
            and _safe_float(action["features"].get("bge_gain")) >= min_bge_gain
        )
        predictions.append({
            **action,
            "predicted_delta_r5": delta,
            "predicted_gain_probability": gain,
            "predicted_loss_probability": risk,
            "objective": objective,
            "admissible": admissible,
        })
    admissible = [item for item in predictions if item["admissible"]]
    if not admissible:
        return current, {
            "changed": False,
            "reason": "no_admissible_replacement",
            "actions_considered": len(predictions),
        }
    selected = max(admissible, key=lambda item: item["objective"])
    completed_top5 = list(top5)
    completed_top5[int(selected["removed_position"])] = int(selected["candidate_id"])
    completed = completed_top5 + [value for value in current if value not in set(completed_top5)]
    completed.extend(value for value in original if value not in set(completed))
    return completed, {
        "changed": completed_top5 != top5,
        "changed_top2": completed_top5[:2] != top5[:2],
        "changed_top5_membership": set(completed_top5) != set(top5),
        "before_top5": top5,
        "after_top5": completed_top5,
        "removed_id": int(selected["removed_id"]),
        "added_id": int(selected["candidate_id"]),
        "predicted_delta_r5": round(float(selected["predicted_delta_r5"]), 6),
        "predicted_gain_probability": round(float(selected["predicted_gain_probability"]), 6),
        "predicted_loss_probability": round(float(selected["predicted_loss_probability"]), 6),
        "actions_considered": len(predictions),
        "removed_position": int(selected["removed_position"]),
        "completion_mode": "proof_aware_single_replacement",
    }


def extract_action_delta_features(
    original_doc_ids: Sequence[int],
    candidate_doc_ids: Sequence[int],
    probabilities: Sequence[float],
    passage_features: Mapping[int, Mapping[str, Any]],
    action_name: str,
) -> Dict[str, float]:
    """Describe a complete slate action using gold-free runtime signals only."""
    original = [int(value) for value in original_doc_ids]
    candidate = [int(value) for value in candidate_doc_ids]
    probability_by_id = {
        passage_idx: _safe_float(probabilities[index])
        for index, passage_idx in enumerate(original[:len(probabilities)])
    }
    rank_by_id = {passage_idx: rank for rank, passage_idx in enumerate(original, start=1)}
    original_top5 = original[:5]
    candidate_top5 = candidate[:5]
    original_top2 = original[:2]
    candidate_top2 = candidate[:2]
    added = [value for value in candidate_top5 if value not in set(original_top5)]

    def mean(values: Sequence[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    def probability_mean(values: Sequence[int]) -> float:
        return mean([probability_by_id.get(value, 0.0) for value in values])

    def probability_min(values: Sequence[int]) -> float:
        scores = [probability_by_id.get(value, 0.0) for value in values]
        return min(scores) if scores else 0.0

    def feature_mean(values: Sequence[int], name: str) -> float:
        return mean([
            _safe_float((passage_features.get(value, {}) or {}).get(name))
            for value in values
        ])

    added_ranks = [rank_by_id.get(value, len(original) + 1) for value in added]
    selected_rank2_rank = rank_by_id.get(candidate_top2[1], 2) if len(candidate_top2) > 1 else 2
    incumbent_ontology = feature_mean(original_top5, "ontology_candidate")
    candidate_ontology = feature_mean(candidate_top5, "ontology_candidate")
    incumbent_certificate = feature_mean(original_top5, "complete_certificate_member")
    candidate_certificate = feature_mean(candidate_top5, "complete_certificate_member")
    incumbent_coverage = feature_mean(original_top5, "query_coverage")
    candidate_coverage = feature_mean(candidate_top5, "query_coverage")
    incumbent_semantic = feature_mean(original_top5, "semantic_score")
    candidate_semantic = feature_mean(candidate_top5, "semantic_score")
    return {
        "action_top2": float(action_name == "top2"),
        "action_top5": float(action_name == "top5"),
        "changes_top2": float(candidate_top2 != original_top2),
        "changes_top5_membership": float(set(candidate_top5) != set(original_top5)),
        "selected_rank2_original_rank_log": math.log1p(selected_rank2_rank),
        "added_count": float(len(added)),
        "added_rank_mean_log": math.log1p(mean(added_ranks)),
        "added_rank_max_log": math.log1p(max(added_ranks, default=0)),
        "top2_probability_gain": probability_mean(candidate_top2) - probability_mean(original_top2),
        "top5_probability_mean_gain": probability_mean(candidate_top5) - probability_mean(original_top5),
        "top5_probability_min_gain": probability_min(candidate_top5) - probability_min(original_top5),
        "candidate_top2_probability_mean": probability_mean(candidate_top2),
        "candidate_top5_probability_mean": probability_mean(candidate_top5),
        "incumbent_top2_probability_mean": probability_mean(original_top2),
        "incumbent_top5_probability_mean": probability_mean(original_top5),
        "candidate_top5_ontology_rate": candidate_ontology,
        "ontology_rate_gain": candidate_ontology - incumbent_ontology,
        "candidate_top5_certificate_rate": candidate_certificate,
        "certificate_rate_gain": candidate_certificate - incumbent_certificate,
        "candidate_top5_query_coverage_mean": candidate_coverage,
        "query_coverage_gain": candidate_coverage - incumbent_coverage,
        "candidate_top5_semantic_mean": candidate_semantic,
        "semantic_gain": candidate_semantic - incumbent_semantic,
    }


class MetricAlignedSlateRanker:
    def __init__(self, manifest_path: str):
        self.manifest_path = str(manifest_path or "")
        self.manifest: Dict[str, Any] = {}
        self.model = None
        self.action_models = None
        self.feature_names: List[str] = []
        self.bge_config: Dict[str, Any] = {}
        self.bge_scorer = None
        self.incumbent_model = None
        self.hybrid_completion: Dict[str, Any] = {}
        self.top5_completion_models = None
        self.top5_completion: Dict[str, Any] = {}
        self.frozen_incumbent: Dict[str, Any] = {}
        self.frozen_incumbent_slates: Dict[str, List[int]] = {}
        self.set_level_verifier: Dict[str, Any] = {}
        self.set_level_verifier_decisions: Dict[str, Dict[str, Any]] = {}
        if not self.manifest_path:
            return
        path = Path(self.manifest_path)
        if not path.is_file():
            return
        self.manifest = json.loads(path.read_text(encoding="utf-8"))
        if self.manifest.get("type") != "metric_aligned_slate_ranker":
            return
        self.feature_names = list(self.manifest.get("feature_names", []) or [])
        if self.feature_names not in (FEATURE_NAMES, BGE_BOUNDARY_FEATURE_NAMES):
            return
        if (self.manifest.get("metadata", {}) or {}).get("inference_uses_gold") is not False:
            return
        artifact = path.parent / str(self.manifest.get("artifact_path", ""))
        if not artifact.is_file():
            return
        expected_hash = str(self.manifest.get("artifact_sha256", "") or "")
        if expected_hash and hashlib.sha256(artifact.read_bytes()).hexdigest() != expected_hash:
            return
        import joblib

        self.model = joblib.load(artifact)
        if self.feature_names == BGE_BOUNDARY_FEATURE_NAMES:
            self.bge_config = dict(self.manifest.get("bge_reranker", {}) or {})
            if not bool(self.bge_config.get("enabled", False)):
                self.model = None
                return
            self.hybrid_completion = dict(
                self.manifest.get("hybrid_completion", {}) or {}
            )
            if bool(self.hybrid_completion.get("enabled", False)):
                incumbent_artifact = path.parent / str(
                    self.hybrid_completion.get("artifact_path", "") or ""
                )
                incumbent_hash = str(
                    self.hybrid_completion.get("artifact_sha256", "") or ""
                )
                if (
                    list(self.hybrid_completion.get("feature_names", []) or []) == FEATURE_NAMES
                    and incumbent_artifact.is_file()
                    and (
                        not incumbent_hash
                        or hashlib.sha256(incumbent_artifact.read_bytes()).hexdigest()
                        == incumbent_hash
                    )
                ):
                    self.incumbent_model = joblib.load(incumbent_artifact)
                else:
                    self.model = None
                    return
            self.top5_completion = dict(
                self.manifest.get("top5_completion", {}) or {}
            )
            if bool(self.top5_completion.get("enabled", False)):
                completion_artifact = path.parent / str(
                    self.top5_completion.get("artifact_path", "") or ""
                )
                completion_hash = str(
                    self.top5_completion.get("artifact_sha256", "") or ""
                )
                if (
                    list(self.top5_completion.get("feature_names", []) or [])
                    == TOP5_COMPLETION_FEATURE_NAMES
                    and completion_artifact.is_file()
                    and (
                        not completion_hash
                        or hashlib.sha256(completion_artifact.read_bytes()).hexdigest()
                        == completion_hash
                    )
                ):
                    loaded = joblib.load(completion_artifact)
                    if all(name in loaded for name in ("delta", "gain", "risk")):
                        self.top5_completion_models = loaded
                if self.top5_completion_models is None:
                    self.model = None
                    return
            self.frozen_incumbent = dict(
                self.manifest.get("frozen_incumbent", {}) or {}
            )
            if bool(self.frozen_incumbent.get("enabled", False)):
                incumbent_mode = str(
                    self.frozen_incumbent.get("mode", "") or ""
                ).strip()
                incumbent_path = path.parent / str(
                    self.frozen_incumbent.get("slates_path", "") or ""
                )
                incumbent_hash = str(
                    self.frozen_incumbent.get("slates_sha256", "") or ""
                )
                if (
                    incumbent_path.is_file()
                    and (
                        not incumbent_hash
                        or hashlib.sha256(incumbent_path.read_bytes()).hexdigest()
                        == incumbent_hash
                    )
                ):
                    payload = json.loads(incumbent_path.read_text(encoding="utf-8"))
                    if (
                        payload.get("type") == "frozen_strong_incumbent_slates"
                        and payload.get("inference_uses_gold") is False
                    ):
                        slates = payload.get("slates", {}) or {}
                        if isinstance(slates, dict):
                            self.frozen_incumbent_slates = {
                                str(key): [int(value) for value in list(values)[:5]]
                                for key, values in slates.items()
                                if isinstance(values, list)
                                and len(values) >= 5
                                and len(set(int(value) for value in values[:5])) == 5
                            }
                if (
                    incumbent_mode != "runtime_current_top5"
                    and not self.frozen_incumbent_slates
                ):
                    self.model = None
                    return
            self.set_level_verifier = dict(
                self.manifest.get("set_level_verifier", {}) or {}
            )
            if bool(self.set_level_verifier.get("enabled", False)):
                verifier_path = path.parent / str(
                    self.set_level_verifier.get("decisions_path", "") or ""
                )
                verifier_hash = str(
                    self.set_level_verifier.get("decisions_sha256", "") or ""
                )
                if (
                    verifier_path.is_file()
                    and (
                        not verifier_hash
                        or hashlib.sha256(verifier_path.read_bytes()).hexdigest()
                        == verifier_hash
                    )
                ):
                    payload = json.loads(verifier_path.read_text(encoding="utf-8"))
                    if (
                        payload.get("type") == "set_level_missing_evidence_decisions"
                        and payload.get("inference_uses_gold") is False
                    ):
                        decisions = payload.get("decisions", {}) or {}
                        if isinstance(decisions, dict):
                            self.set_level_verifier_decisions = {
                                str(key): dict(value)
                                for key, value in decisions.items()
                                if isinstance(value, Mapping)
                            }
        action_gate = dict(self.manifest.get("action_gate", {}) or {})
        if (
            action_gate.get("enabled", False)
            and list(action_gate.get("feature_names", []) or []) == ACTION_FEATURE_NAMES
        ):
            action_artifact = path.parent / str(action_gate.get("artifact_path", ""))
            action_hash = str(action_gate.get("artifact_sha256", "") or "")
            if (
                action_artifact.is_file()
                and (
                    not action_hash
                    or hashlib.sha256(action_artifact.read_bytes()).hexdigest() == action_hash
                )
            ):
                loaded = joblib.load(action_artifact)
                if all(name in loaded for name in ("r2", "r5", "gain", "risk")):
                    self.action_models = loaded

    def is_ready(self) -> bool:
        return bool(self.model is not None and self.manifest.get("enabled", False))

    def action_gate_ready(self) -> bool:
        return bool(self.action_models is not None)

    def requires_passage_texts(self) -> bool:
        return bool(self.is_ready() and self.feature_names == BGE_BOUNDARY_FEATURE_NAMES)

    def hybrid_completion_ready(self) -> bool:
        return bool(
            self.is_ready()
            and self.hybrid_completion.get("enabled", False)
            and self.incumbent_model is not None
        )

    def top5_completion_ready(self) -> bool:
        return bool(
            self.is_ready()
            and self.top5_completion.get("enabled", False)
            and self.top5_completion_models is not None
        )

    def frozen_incumbent_ready(self) -> bool:
        mode = str(self.frozen_incumbent.get("mode", "") or "").strip()
        return bool(
            self.is_ready()
            and self.frozen_incumbent.get("enabled", False)
            and (mode == "runtime_current_top5" or self.frozen_incumbent_slates)
        )

    def frozen_incumbent_for_query(self, query: str) -> List[int]:
        if not self.frozen_incumbent_ready():
            return []
        return list(self.frozen_incumbent_slates.get(set_level_query_key(query), []))

    def set_level_verifier_ready(self) -> bool:
        return bool(
            self.is_ready()
            and self.set_level_verifier.get("enabled", False)
            and self.set_level_verifier_decisions
        )

    def _apply_set_level_verifier(
        self,
        query: str,
        doc_ids: Sequence[int],
        current: Sequence[int],
        rank_window: int,
    ) -> Tuple[List[int], Dict[str, Any]]:
        log: Dict[str, Any] = {
            "enabled": bool(self.set_level_verifier.get("enabled", False)),
            "model_ready": self.set_level_verifier_ready(),
            "changed": False,
        }
        if not self.set_level_verifier_ready():
            log["reason"] = "verifier_missing_disabled_or_invalid"
            return list(current), log
        query_key = set_level_query_key(query)
        log["query_key"] = query_key
        decision = self.set_level_verifier_decisions.get(query_key)
        if not decision:
            log["reason"] = "no_verified_decision"
            return list(current), log
        expected_top5 = [int(value) for value in decision.get("expected_top5", [])]
        current_top5 = [int(value) for value in list(current)[:5]]
        if expected_top5 != current_top5:
            log.update({
                "reason": "frozen_slate_mismatch",
                "expected_top5": expected_top5,
                "observed_top5": current_top5,
            })
            return list(current), log
        removed_position = int(decision.get("removed_position", -1))
        candidate_id = int(decision.get("candidate_id", -1))
        allowed_positions = {
            int(value)
            for value in self.set_level_verifier.get(
                "allowed_removed_positions", [3, 4]
            )
        }
        candidate_window = {int(value) for value in list(doc_ids)[:rank_window]}
        if removed_position not in allowed_positions:
            log["reason"] = "removed_position_not_allowed"
            return list(current), log
        if candidate_id not in candidate_window or candidate_id in set(current_top5):
            log["reason"] = "candidate_not_available_or_already_selected"
            return list(current), log
        removed_id = current_top5[removed_position]
        expected_removed_id = int(decision.get("removed_id", removed_id))
        if removed_id != expected_removed_id:
            log["reason"] = "removed_passage_mismatch"
            return list(current), log
        completed_top5 = list(current_top5)
        completed_top5[removed_position] = candidate_id
        completed = completed_top5 + [
            int(value)
            for value in current
            if int(value) not in set(completed_top5)
        ]
        if completed[:3] != current_top5[:3]:
            log["reason"] = "top3_safety_violation"
            return list(current), log
        log.update({
            "changed": True,
            "reason": "verified_missing_evidence_replacement",
            "removed_position": removed_position,
            "removed_id": removed_id,
            "candidate_id": candidate_id,
            "confidence": _safe_float(decision.get("confidence"), 0.0),
            "missing_role": str(decision.get("missing_role", "") or ""),
            "before_top5": current_top5,
            "after_top5": completed_top5,
        })
        return completed, log

    def runtime_rank_window(self) -> int:
        policy = dict(self.manifest.get("policy", {}) or {})
        return max(int(policy.get("rank_window", 120)), 5)

    def _get_bge_scorer(self):
        if self.bge_scorer is None:
            from .query_plan_bge_reranker import LocalBGEReranker

            self.bge_scorer = LocalBGEReranker(
                str(self.bge_config.get("model_path", "") or ""),
                device=str(self.bge_config.get("device", "auto") or "auto"),
                batch_size=int(self.bge_config.get("batch_size", 16) or 16),
                max_length=int(self.bge_config.get("max_length", 512) or 512),
            )
        return self.bge_scorer

    @staticmethod
    def _risk_probability(model: Any, matrix: Sequence[Sequence[float]]) -> List[float]:
        probabilities = model.predict_proba(matrix)
        classes = [int(value) for value in list(getattr(model, "classes_", []))]
        if 1 not in classes:
            return [0.0 for _ in matrix]
        index = classes.index(1)
        return [float(row[index]) for row in probabilities]

    def rerank(
        self,
        query_plan_log: Mapping[str, Any],
        doc_ids: Sequence[int],
        doc_scores: Sequence[float],
        *,
        query: str = "",
        passage_texts: Mapping[int, str] | Sequence[str] | None = None,
    ) -> Tuple[List[int], List[float], Dict[str, Any]]:
        log: Dict[str, Any] = {
            "enabled": True,
            "model_ready": self.is_ready(),
            "manifest_path": self.manifest_path,
            "changed": False,
        }
        if not self.is_ready():
            log["reason"] = "model_missing_disabled_or_invalid"
            return list(doc_ids), list(doc_scores), log
        policy = dict(self.manifest.get("policy", {}) or {})
        rank_window = min(int(policy.get("rank_window", 120)), len(doc_ids))
        max_score = max((_safe_float(value) for value in list(doc_scores)[:rank_window]), default=0.0)
        maps = _passage_signal_maps(query_plan_log)
        passage_feature_rows = [
            extract_metric_aligned_features(
                query_plan_log,
                int(doc_ids[index]),
                index + 1,
                _safe_float(doc_scores[index]),
                max_score,
                maps,
            )
            for index in range(rank_window)
        ]
        base_passage_feature_rows = list(passage_feature_rows)
        bge_scores: List[float] = []
        if self.feature_names == BGE_BOUNDARY_FEATURE_NAMES:
            if passage_texts is None:
                log["reason"] = "bge_passage_texts_missing"
                return list(doc_ids), list(doc_scores), log
            if isinstance(passage_texts, Mapping):
                texts = [str(passage_texts.get(int(doc_ids[index]), "") or "") for index in range(rank_window)]
            else:
                texts = [str(value or "") for value in list(passage_texts)[:rank_window]]
            if len(texts) != rank_window or any(not value for value in texts):
                log["reason"] = "bge_passage_texts_incomplete"
                return list(doc_ids), list(doc_scores), log
            bge_scores = self._get_bge_scorer().score(
                str(query or query_plan_log.get("query", "") or ""),
                texts,
            )
            bge_order = sorted(range(rank_window), key=lambda index: bge_scores[index], reverse=True)
            bge_rank = {index: rank for rank, index in enumerate(bge_order, start=1)}
            passage_feature_rows = [
                extend_bge_boundary_features(row, bge_scores[index], bge_rank[index])
                for index, row in enumerate(passage_feature_rows)
            ]
            matrix = [bge_boundary_feature_vector(row) for row in passage_feature_rows]
            passage_text_map = {
                int(doc_ids[index]): texts[index] for index in range(rank_window)
            }
        else:
            matrix = [feature_vector(row) for row in passage_feature_rows]
            passage_text_map = {}
        probabilities = self.model.predict_proba(matrix)[:, 1].tolist()
        base_reranked, policy_log = materialize_metric_aligned_slate(
            doc_ids,
            probabilities,
            rank_window=rank_window,
            rank_penalty=_safe_float(policy.get("rank_penalty"), 0.5),
            top2_margin=_safe_float(policy.get("top2_margin"), 0.25),
            top5_margin=_safe_float(policy.get("top5_margin"), 0.25),
        )
        reranked = base_reranked
        hybrid_log: Dict[str, Any] = {
            "enabled": bool(self.hybrid_completion.get("enabled", False)),
            "model_ready": self.hybrid_completion_ready(),
        }
        if self.hybrid_completion_ready():
            incumbent_policy = dict(self.hybrid_completion.get("policy", {}) or {})
            incumbent_matrix = [feature_vector(row) for row in base_passage_feature_rows]
            incumbent_probabilities = self.incumbent_model.predict_proba(
                incumbent_matrix
            )[:, 1].tolist()
            incumbent_slate, _ = materialize_metric_aligned_slate(
                doc_ids,
                incumbent_probabilities,
                rank_window=min(
                    int(incumbent_policy.get("rank_window", rank_window)),
                    rank_window,
                ),
                rank_penalty=_safe_float(
                    incumbent_policy.get("rank_penalty"), 0.5
                ),
                top2_margin=_safe_float(
                    incumbent_policy.get("top2_margin"), 0.25
                ),
                top5_margin=_safe_float(
                    incumbent_policy.get("top5_margin"), 0.25
                ),
            )
            reranked, policy_log = materialize_bge_top2_incumbent_top5(
                doc_ids,
                base_reranked,
                incumbent_slate,
            )
            hybrid_log.update({
                "bge_top5": list(base_reranked)[:5],
                "incumbent_top5": list(incumbent_slate)[:5],
                "hybrid_top5": list(reranked)[:5],
            })
        top5_completion_log: Dict[str, Any] = {
            "enabled": bool(self.top5_completion.get("enabled", False)),
            "model_ready": self.top5_completion_ready(),
            "changed": False,
        }
        if self.top5_completion_ready():
            passage_features = {
                int(doc_ids[index]): passage_feature_rows[index]
                for index in range(rank_window)
            }
            reranked, completion_policy_log = materialize_proof_aware_top5_completion(
                doc_ids,
                reranked,
                probabilities,
                passage_features,
                passage_text_map,
                self.top5_completion_models,
                dict(self.top5_completion.get("policy", {}) or {}),
                rank_window=min(
                    int(self.top5_completion.get("rank_window", rank_window)),
                    rank_window,
                ),
            )
            top5_completion_log.update(completion_policy_log)
            policy_log = {
                "changed": list(reranked)[:5] != list(doc_ids)[:5],
                "changed_top2": list(reranked)[:2] != list(doc_ids)[:2],
                "changed_top5_membership": set(list(reranked)[:5])
                != set(list(doc_ids)[:5]),
                "before_top5": list(doc_ids)[:5],
                "after_top5": list(reranked)[:5],
            }
        action_log: Dict[str, Any] = {
            "enabled": bool((self.manifest.get("action_gate", {}) or {}).get("enabled", False)),
            "model_ready": self.action_gate_ready(),
            "selected_action": "base_policy",
        }
        if self.action_gate_ready():
            passage_features = {
                int(doc_ids[index]): passage_feature_rows[index]
                for index in range(rank_window)
            }
            actions = materialize_metric_aligned_actions(
                doc_ids,
                probabilities,
                rank_window=rank_window,
                rank_penalty=_safe_float(policy.get("rank_penalty"), 0.5),
                top2_margin=_safe_float(policy.get("top2_margin"), 0.25),
                top5_margin=_safe_float(policy.get("top5_margin"), 0.25),
            )
            action_names = [name for name in ("top2", "top5") if name in actions]
            action_rows = [
                action_feature_vector(extract_action_delta_features(
                    doc_ids,
                    actions[name],
                    probabilities,
                    passage_features,
                    name,
                ))
                for name in action_names
            ]
            predictions = []
            if action_rows:
                pred_r2 = self.action_models["r2"].predict(action_rows)
                pred_r5 = self.action_models["r5"].predict(action_rows)
                pred_gain = self._risk_probability(self.action_models["gain"], action_rows)
                pred_risk = self._risk_probability(self.action_models["risk"], action_rows)
                gate_policy = dict((self.manifest.get("action_gate", {}) or {}).get("policy", {}) or {})
                gate_mode = str(gate_policy.get("mode", "score") or "score")
                risk_weight = _safe_float(gate_policy.get("risk_weight"), 0.5)
                r5_weight = _safe_float(gate_policy.get("r5_weight"), 0.65)
                gain_weight = _safe_float(gate_policy.get("gain_weight"), 0.5)
                min_gain = _safe_float(gate_policy.get("min_gain"), 0.0)
                max_loss_probability = _safe_float(
                    gate_policy.get("max_loss_probability"), 1.0
                )
                for index, name in enumerate(action_names):
                    objective = (
                        float(pred_r2[index])
                        + r5_weight * float(pred_r5[index])
                        + gain_weight * float(pred_gain[index])
                        - risk_weight * float(pred_risk[index])
                    )
                    predictions.append({
                        "action": name,
                        "predicted_delta_r2": round(float(pred_r2[index]), 6),
                        "predicted_delta_r5": round(float(pred_r5[index]), 6),
                        "predicted_gain_probability": round(float(pred_gain[index]), 6),
                        "predicted_loss_probability": round(float(pred_risk[index]), 6),
                        "objective": round(objective, 6),
                        "admissible": bool(
                            (
                                float(pred_gain[index]) >= _safe_float(
                                    gate_policy.get("min_gain_probability"), 0.0
                                )
                                if gate_mode == "veto"
                                else objective >= min_gain
                            )
                            and float(pred_risk[index]) <= max_loss_probability
                        ),
                    })
                admissible = [item for item in predictions if item["admissible"]]
                if admissible:
                    if gate_mode == "veto":
                        selected = next(
                            (
                                item for name in ("top5", "top2")
                                for item in admissible if item["action"] == name
                            ),
                            admissible[0],
                        )
                    else:
                        selected = max(admissible, key=lambda item: item["objective"])
                    reranked = actions[selected["action"]]
                    action_log["selected_action"] = selected["action"]
                else:
                    reranked = list(doc_ids)
                    action_log["selected_action"] = "no_op"
            else:
                reranked = list(doc_ids)
                action_log["selected_action"] = "no_op"
            action_log["predictions"] = predictions
            policy_log = {
                "changed": list(reranked)[:5] != list(doc_ids)[:5],
                "changed_top2": list(reranked)[:2] != list(doc_ids)[:2],
                "changed_top5_membership": set(list(reranked)[:5]) != set(list(doc_ids)[:5]),
                "before_top5": list(doc_ids)[:5],
                "after_top5": list(reranked)[:5],
            }
        frozen_incumbent_log: Dict[str, Any] = {
            "enabled": bool(self.frozen_incumbent.get("enabled", False)),
            "model_ready": self.frozen_incumbent_ready(),
            "changed": False,
        }
        frozen_incumbent_mode = str(
            self.frozen_incumbent.get("mode", "") or ""
        ).strip()
        if frozen_incumbent_mode == "runtime_current_top5":
            frozen_top5 = list(doc_ids)[:5]
            frozen_incumbent_log["source"] = "current_post_physical_top5"
        else:
            frozen_top5 = self.frozen_incumbent_for_query(
                str(query or query_plan_log.get("query", "") or "")
            )
            frozen_incumbent_log["source"] = "gold_free_retrieval_replay_cache"
        if self.frozen_incumbent_ready() and frozen_top5:
            reranked, frozen_incumbent_log = materialize_frozen_incumbent_top5(
                doc_ids,
                probabilities,
                frozen_top5,
                probability_shortlist_size=int(
                    self.frozen_incumbent.get("probability_shortlist_size", 8) or 8
                ),
            )
            frozen_incumbent_log["source"] = (
                "current_post_physical_top5"
                if frozen_incumbent_mode == "runtime_current_top5"
                else "gold_free_retrieval_replay_cache"
            )
            policy_log = {
                "changed": list(reranked)[:5] != list(doc_ids)[:5],
                "changed_top2": list(reranked)[:2] != list(doc_ids)[:2],
                "changed_top5_membership": set(list(reranked)[:5])
                != set(list(doc_ids)[:5]),
                "before_top5": list(doc_ids)[:5],
                "after_top5": list(reranked)[:5],
            }
        elif self.frozen_incumbent_ready():
            frozen_incumbent_log["reason"] = "query_not_found_in_frozen_incumbent"



        reranked, set_level_verifier_log = self._apply_set_level_verifier(
            str(query or query_plan_log.get("query", "") or ""),
            doc_ids,
            reranked,
            rank_window,
        )
        if set_level_verifier_log.get("changed", False):
            policy_log = {
                "changed": list(reranked)[:5] != list(doc_ids)[:5],
                "changed_top2": list(reranked)[:2] != list(doc_ids)[:2],
                "changed_top5_membership": set(list(reranked)[:5])
                != set(list(doc_ids)[:5]),
                "before_top5": list(doc_ids)[:5],
                "after_top5": list(reranked)[:5],
            }
        if not policy_log.get("changed", False):
            reranked_scores = [_safe_float(value) for value in doc_scores]
        else:
            score_by_id = {int(value): _safe_float(doc_scores[index]) for index, value in enumerate(doc_ids)}
            high = max(score_by_id.values(), default=1.0)
            step = max(abs(high), 1.0) * 1e-6
            reranked_scores = [high - step * index for index in range(len(reranked))]
        log.update(policy_log)
        log.update({
            "rank_window": rank_window,
            "bge_reranker_enabled": self.feature_names == BGE_BOUNDARY_FEATURE_NAMES,
            "hybrid_completion": hybrid_log,
            "proof_aware_top5_completion": top5_completion_log,
            "frozen_strong_incumbent": frozen_incumbent_log,
            "set_level_missing_evidence_verifier": set_level_verifier_log,
            "oof_metrics": self.manifest.get("oof_metrics", {}),
            "action_delta_admission": action_log,
            "top_candidates": [
                {
                    "passage_idx": int(doc_ids[index]),
                    "original_rank": index + 1,
                    "gold_free_probability": round(float(probabilities[index]), 6),
                    **(
                        {"bge_relevance": round(float(bge_scores[index]), 6)}
                        if bge_scores else {}
                    ),
                }
                for index in sorted(
                    range(rank_window), key=lambda item: probabilities[item], reverse=True
                )[:8]
            ],
        })
        return reranked, reranked_scores, log
