"""Query-grouped nonlinear delta admission.

The offline trainer exports compact decision-tree ensembles as JSON. This
module deliberately has no sklearn dependency: runtime scoring is a small
tree traversal over gold-free action and query-group features, followed by an
incumbent-relative, one-sided safety admission rule.
"""

from __future__ import annotations

import math
import struct
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple


ACTION_KIND_FEATURES = (
    "v220_action_select_top2_subset",
    "v220_action_replace_top2_pair",
    "v220_action_replace_top5_slot5",
)

QUERY_SUMMARY_BASE_FEATURES = (
    "candidate_score",
    "candidate_confidence",
    "local_comparator_proxy",
    "local_comparator_margin_to_group_mean",
    "local_comparator_rank_inv",
    "original_rank_inv",
    "candidate_query_coverage_max",
    "candidate_minus_displaced_query_coverage",
    "candidate_minus_displaced_title_coverage",
    "candidate_pair_query_union_coverage",
    "candidate_pair_complementarity",
    "candidate_source_agreement",
    "unified_source_agreement",
    "candidate_plan_precision_max",
    "candidate_branch_score_max",
    "candidate_utility_max",
    "candidate_proof_fraction",
    "candidate_role_consistent_fraction",
    "candidate_answer_type_compatible_fraction",
    "pair_complete_proof",
    "displaced_incumbent_score",
    "action_displacement_risk_proxy",
    "physical_estimated_recall_delta",
    "physical_estimated_loss_probability",
    "physical_objective",
    "cq_contract_confidence",
    "cq_predicate_coverage",
    "cq_typed_contract_confidence",
    "cq_typed_predicate_coverage",
    "candidate_minus_displaced_semantic_score",
    "semantic_comparator_proxy",
    "semantic_comparator_rank_inv",
    "rank_partition_action",
    "top5_membership_invariant",
    "selected_original_rank_mean",
    "selected_original_rank_max",
    "selected_rank1_preserved",
    "selected_rank2_preserved",
    "selected_from_rank3_to5_count",
    "selected_minus_displaced_retrieval_score",
    "selected_ontology_backed_fraction",
    "selected_ontology_source_agreement",
    "selected_proof_fraction",
    "selected_role_consistent_fraction",
    "selected_complete_provenance_fraction",
    "selected_real_proof_fraction",
    "selected_derived_proof_fraction",
    "selected_proof_certificate_strength_mean",
    "selected_proof_obligation_coverage_mean",
    "selected_proof_obligation_coverage_min",
    "selected_branch_coverage_mean",
    "selected_plan_identity_fraction",
    "selected_exact_entity_key_fraction",
    "selected_binding_fraction",
    "selected_minus_displaced_proof_coverage",
    "proof_catalog_coverage_fraction",
    "proof_catalog_passage_count_log",
    "selected_full_aperture_fraction",
    "selected_aperture_anchor_coverage_mean",
    "selected_aperture_query_coverage_mean",
    "selected_aperture_answer_type_score_mean",
    "selected_aperture_terminal_relation_fraction",
    "selected_aperture_relation_slot_coverage_mean",
    "displaced_aperture_relation_slot_coverage_mean",
    "selected_minus_displaced_aperture_slot_coverage",
    "selected_aperture_structured_anchor_coverage_mean",
    "selected_aperture_structured_triple_count_log_mean",
    "result_slate_aperture_relation_slot_coverage",
    "result_slate_aperture_binding_join_count_log",
    "result_slate_aperture_connected_passage_fraction",
    "selected_aperture_local_proof_path_fraction",
    "selected_aperture_local_path_score_mean",
    "result_slate_aperture_shared_path_count_log",
    "nested_stage_top5_membership",
    "nested_stage_top2_ordering",
    "top2_membership_invariant",
    *ACTION_KIND_FEATURES,
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        return default if math.isnan(number) or math.isinf(number) else number
    except (TypeError, ValueError):
        return default


def action_features(action: Mapping[str, Any]) -> Dict[str, float]:
    """Return raw features plus stable action-kind indicators."""
    features = {
        str(name): _safe_float(value)
        for name, value in (action.get("features", {}) or {}).items()
    }
    kind = str(action.get("action_kind") or "")
    features.update({
        "v220_action_select_top2_subset": float(kind == "select_top2_subset"),
        "v220_action_replace_top2_pair": float(kind == "replace_top2_pair"),
        "v220_action_replace_top5_slot5": float(kind == "replace_top5_slot5"),
    })
    return features


def proof_access_path(features: Mapping[str, float]) -> str:
    """Map a semijoin action to its calibrated physical access path."""
    if _safe_float(features.get("semijoin_strong_executable")) >= 0.5:
        return "strong_executable"
    if _safe_float(features.get("semijoin_fallback_eligible")) >= 0.5:
        return "calibrated_fallback"
    return "unclassified"


def query_summary_features(
    feature_rows: Sequence[Mapping[str, float]],
) -> Dict[str, float]:
    """Build permutation-invariant features for the query-level no-op gate."""
    result: Dict[str, float] = {
        "v220_action_count_log": math.log1p(len(feature_rows)),
    }
    for name in QUERY_SUMMARY_BASE_FEATURES:
        values = sorted(
            (_safe_float(row.get(name), 0.0) for row in feature_rows),
            reverse=True,
        )
        if not values:
            values = [0.0]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        result[f"v220_query_max__{name}"] = values[0]
        result[f"v220_query_mean__{name}"] = mean
        result[f"v220_query_std__{name}"] = math.sqrt(max(variance, 0.0))
        result[f"v220_query_range__{name}"] = values[0] - values[-1]
        result[f"v220_query_top_gap__{name}"] = (
            values[0] - values[1] if len(values) > 1 else values[0]
        )
    return result


def query_summary_feature_names() -> List[str]:
    names = ["v220_action_count_log"]
    for name in QUERY_SUMMARY_BASE_FEATURES:
        for prefix in ("max", "mean", "std", "range", "top_gap"):
            names.append(f"v220_query_{prefix}__{name}")
    return names


def _tree_value(tree: Mapping[str, Any], values: Sequence[float]) -> float:
    children_left = tree.get("children_left", []) or []
    children_right = tree.get("children_right", []) or []
    features = tree.get("feature", []) or []
    thresholds = tree.get("threshold", []) or []
    outputs = tree.get("value", []) or []
    node = 0
    while node < len(children_left) and int(children_left[node]) >= 0:
        feature_index = int(features[node])
        value = values[feature_index] if 0 <= feature_index < len(values) else 0.0


        value = struct.unpack("!f", struct.pack("!f", float(value)))[0]
        node = (
            int(children_left[node])
            if value <= _safe_float(thresholds[node])
            else int(children_right[node])
        )
    return _safe_float(outputs[node] if 0 <= node < len(outputs) else 0.0)


def predict_ensemble(
    head: Mapping[str, Any], features: Mapping[str, float]
) -> Tuple[float, float]:
    names = list(head.get("feature_names", []) or [])
    values = [_safe_float(features.get(name), 0.0) for name in names]
    predictions = [
        _tree_value(tree, values) for tree in head.get("trees", []) or []
        if isinstance(tree, Mapping)
    ]
    if not predictions:
        return _safe_float(head.get("fallback_value"), 0.0), 0.0
    mean = sum(predictions) / len(predictions)
    variance = sum((value - mean) ** 2 for value in predictions) / len(predictions)
    return mean, math.sqrt(max(variance, 0.0))


def _prediction_bundle(
    expert: Mapping[str, Any], features: Mapping[str, float]
) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for name in ("winner", "gain", "loss", "delta_r2", "delta_r5"):
        mean, std = predict_ensemble(expert.get(f"{name}_head", {}) or {}, features)
        if name in {"winner", "gain", "loss"}:
            mean = min(max(mean, 0.0), 1.0)
        result[name] = mean
        result[f"{name}_std"] = std
    return result


def route_feature_regime(
    model: Mapping[str, Any], features: Mapping[str, float]
) -> Dict[str, Any]:
    """Route by standardized query features without consulting dataset identity."""
    router = model.get("feature_router", {}) or {}
    if router.get("type") == "ontology_signature_opportunity_tree":
        names = [str(value) for value in router.get("feature_names", []) or []]
        children_left = router.get("children_left", []) or []
        children_right = router.get("children_right", []) or []
        split_features = router.get("feature", []) or []
        thresholds = router.get("threshold", []) or []
        probabilities = router.get("opportunity_probability", []) or []
        leaf_regimes = router.get("leaf_regimes", {}) or {}
        if not names or not children_left or not isinstance(leaf_regimes, Mapping):
            return {
                "regime_key": "__global__",
                "distance": 0.0,
                "available": False,
                "router_type": "ontology_signature_opportunity_tree",
            }
        values = [_safe_float(features.get(name), 0.0) for name in names]
        node = 0
        path: List[Dict[str, Any]] = []
        while 0 <= node < len(children_left) and int(children_left[node]) >= 0:
            feature_index = int(
                split_features[node] if node < len(split_features) else -1
            )
            threshold = _safe_float(
                thresholds[node] if node < len(thresholds) else 0.0
            )
            value = values[feature_index] if 0 <= feature_index < len(values) else 0.0
            value = struct.unpack("!f", struct.pack("!f", float(value)))[0]
            go_left = value <= threshold
            path.append({
                "feature": names[feature_index]
                if 0 <= feature_index < len(names) else "",
                "value": value,
                "threshold": threshold,
                "branch": "left" if go_left else "right",
            })
            node = int(children_left[node] if go_left else children_right[node])
        regime_key = str(leaf_regimes.get(str(node)) or "__global__")
        return {
            "regime_key": regime_key,
            "distance": 0.0,
            "available": regime_key != "__global__",
            "router_type": "ontology_signature_opportunity_tree",
            "signature_leaf": int(node),
            "signature_path": path,
            "signature_opportunity_prior": _safe_float(
                probabilities[node] if 0 <= node < len(probabilities) else 0.0
            ),
        }
    names = [str(value) for value in router.get("feature_names", []) or []]
    means = router.get("means", []) or []
    scales = router.get("scales", []) or []
    centroids = router.get("centroids", {}) or {}
    if not names or not isinstance(centroids, Mapping) or not centroids:
        return {
            "regime_key": "__global__",
            "distance": 0.0,
            "available": False,
        }

    standardized = []
    for index, name in enumerate(names):
        mean = _safe_float(means[index] if index < len(means) else 0.0)
        scale = max(
            abs(_safe_float(scales[index] if index < len(scales) else 1.0)),
            1e-6,
        )
        standardized.append((_safe_float(features.get(name)) - mean) / scale)

    best_key = "__global__"
    best_distance = math.inf
    for raw_key, raw_centroid in centroids.items():
        if not isinstance(raw_centroid, Sequence):
            continue
        centroid = [
            _safe_float(raw_centroid[index] if index < len(raw_centroid) else 0.0)
            for index in range(len(standardized))
        ]
        distance = math.sqrt(
            sum(
                (value - center) ** 2
                for value, center in zip(standardized, centroid)
            ) / max(len(standardized), 1)
        )
        if distance < best_distance:
            best_key = str(raw_key)
            best_distance = distance
    return {
        "regime_key": best_key,
        "distance": best_distance if math.isfinite(best_distance) else 0.0,
        "available": best_key != "__global__",
    }


def _action_score(prediction: Mapping[str, float], policy: Mapping[str, Any]) -> float:
    score = (
        _safe_float(policy.get("winner_weight"), 1.0) * prediction["winner"]
        + _safe_float(policy.get("gain_weight"), 0.0) * prediction["gain"]
        + _safe_float(policy.get("delta_r2_weight"), 0.0) * prediction["delta_r2"]
        + _safe_float(policy.get("delta_r5_weight"), 0.0) * prediction["delta_r5"]
        - _safe_float(policy.get("loss_weight"), 0.0) * prediction["loss"]
    )
    target_metric = str(policy.get("target_metric") or "delta_r2")
    if target_metric not in {"delta_r2", "delta_r5"}:
        target_metric = "delta_r2"
    uncertainty = (
        prediction["winner_std"]
        + 0.5 * prediction["loss_std"]
        + 0.25 * prediction[f"{target_metric}_std"]
    )
    return score - _safe_float(policy.get("uncertainty_penalty"), 0.0) * uncertainty


def incumbent_safety_profile(
    action: Mapping[str, Any],
    prediction: Mapping[str, float],
    profile: Mapping[str, Any],
) -> Dict[str, Any]:
    """Evaluate the learned action against the immutable final incumbent."""
    conformal = profile.get("conformal_safety", {}) or {}
    conformal_enabled = bool(conformal.get("enabled", False))
    enabled = bool(profile.get("incumbent_safe_admission", False) or conformal_enabled)
    beta = max(_safe_float(profile.get("confidence_beta"), 0.0), 0.0)
    action_kind = str(action.get("action_kind") or "")
    target = "top5" if action_kind == "replace_top5_slot5" else "top2"
    metric = "delta_r5" if target == "top5" else "delta_r2"
    bounds = conformal.get("bounds", {}) or {}
    action_bounds = (
        bounds.get(action_kind)
        or bounds.get(target)
        or bounds.get("__default__")
        or {}
    )
    if conformal_enabled and action_bounds:
        if bool(action_bounds.get("normalized", False)):
            delta_margin = max(
                _safe_float(action_bounds.get("delta_multiplier"), 0.0), 0.0
            ) * max(
                prediction[f"{metric}_std"],
                _safe_float(action_bounds.get("delta_scale_floor"), 0.02),
            )
            loss_margin = max(
                _safe_float(action_bounds.get("loss_multiplier"), 0.0), 0.0
            ) * max(
                prediction["loss_std"],
                _safe_float(action_bounds.get("loss_scale_floor"), 0.02),
            )
        else:
            delta_margin = max(
                _safe_float(action_bounds.get("delta_margin"), 0.0), 0.0
            )
            loss_margin = max(
                _safe_float(action_bounds.get("loss_margin"), 0.0), 0.0
            )
        metric_lcb = prediction[metric] - delta_margin
        loss_ucb = min(max(
            prediction["loss"] + loss_margin,
            0.0,
        ), 1.0)
        safety_method = "split_conformal"
    else:
        metric_lcb = prediction[metric] - beta * prediction[f"{metric}_std"]
        loss_ucb = min(
            max(prediction["loss"] + beta * prediction["loss_std"], 0.0),
            1.0,
        )
        safety_method = "ensemble_dispersion"
    min_lcb = _safe_float(
        profile.get(
            "min_top5_delta_lcb" if target == "top5" else "min_top2_delta_lcb"
        ),
        -math.inf,
    )
    max_loss_ucb = _safe_float(profile.get("max_loss_ucb"), math.inf)
    passed = bool(not enabled or (metric_lcb >= min_lcb and loss_ucb <= max_loss_ucb))
    return {
        "enabled": enabled,
        "passed": passed,
        "target": target,
        "metric": metric,
        "method": safety_method,
        "conformal_alpha": _safe_float(conformal.get("alpha"), 0.0),
        "confidence_beta": beta,
        "metric_lcb": metric_lcb,
        "min_metric_lcb": min_lcb,
        "loss_ucb": loss_ucb,
        "max_loss_ucb": max_loss_ucb,
    }


def proof_displacement_dominance_profile(
    action: Mapping[str, Any],
) -> Dict[str, Any]:
    """Protect an incumbent proof witness from a structurally weaker replacement."""
    contract = action.get("proof_contract", {}) or {}
    selected = [
        row for row in contract.get("selected", []) or []
        if isinstance(row, Mapping)
    ]
    displaced = [
        row for row in contract.get("displaced", []) or []
        if isinstance(row, Mapping)
    ]

    def any_flag(rows: Sequence[Mapping[str, Any]], name: str) -> bool:
        return any(bool(row.get(name, False)) for row in rows)

    def maximum(rows: Sequence[Mapping[str, Any]], name: str) -> float:
        return max((_safe_float(row.get(name)) for row in rows), default=0.0)

    selected_real_proof = any_flag(selected, "real_proof")
    displaced_real_proof = any_flag(displaced, "real_proof")
    selected_complete_provenance = any_flag(selected, "complete_provenance")
    displaced_complete_provenance = any_flag(displaced, "complete_provenance")
    selected_coverage = maximum(selected, "obligation_coverage")
    displaced_coverage = maximum(displaced, "obligation_coverage")
    selected_strength = maximum(selected, "certificate_strength")
    displaced_strength = maximum(displaced, "certificate_strength")

    violations: List[str] = []
    if not selected or not displaced:
        violations.append("missing_proof_contract_rows")
    if displaced_real_proof and not selected_real_proof:
        violations.append("real_proof_witness_lost")
    if displaced_complete_provenance and not selected_complete_provenance:
        violations.append("complete_provenance_witness_lost")
    if selected_coverage + 1e-12 < displaced_coverage:
        violations.append("proof_obligation_coverage_regressed")
    if selected_strength + 1e-12 < displaced_strength:
        violations.append("proof_certificate_strength_regressed")

    return {
        "enabled": True,
        "passed": not violations,
        "reason": violations[0] if violations else "proof_dominance_preserved",
        "violations": violations,
        "selected_real_proof": selected_real_proof,
        "displaced_real_proof": displaced_real_proof,
        "selected_complete_provenance": selected_complete_provenance,
        "displaced_complete_provenance": displaced_complete_provenance,
        "selected_obligation_coverage_max": selected_coverage,
        "displaced_obligation_coverage_max": displaced_coverage,
        "selected_certificate_strength_max": selected_strength,
        "displaced_certificate_strength_max": displaced_strength,
    }


def rank_feature_routed_residual_opportunity(
    ranker,
    query_plan_log: Dict[str, Any],
    *,
    action_generator: Callable[..., List[Dict[str, Any]]],
    safe_int: Callable[[Any, int], int],
    candidate_compiler: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Two-stage, feature-routed admission with an explicit no-op decision."""
    incumbent = query_plan_log.get("candidate_admission_incumbent", {}) or {}
    metadata = ranker._metadata()
    audit_dataset = ranker._dataset_name(query_plan_log)
    log: Dict[str, Any] = {
        "enabled": True,
        "model_path": ranker.model_path,
        "model_ready": ranker.is_ready(),
        "mode": "feature_routed_residual_opportunity_cbo",
        "candidate_compiler": candidate_compiler,
        "audit_dataset": audit_dataset,
        "dataset_name_used_for_routing": False,
        "scored_count": 0,
        "promoted_count": 0,
        "committed_top2_count": 0,
        "committed_top5_count": 0,
        "committed_pair_count": 0,
        "no_op": True,
        "top_actions": [],
        "state_hash": incumbent.get("state_hash", ""),
        "state_version": incumbent.get("state_version", ""),
    }
    if bool(incumbent.get("defer_decision", False)):
        log["skip_reason"] = "deferred_until_final_state"
        return query_plan_log, log
    if not ranker.is_ready():
        log["skip_reason"] = "model_not_ready"
        return query_plan_log, log

    actions = action_generator(
        query_plan_log,
        max_single_candidates=max(
            1, safe_int(metadata.get("max_single_candidates"), 24)
        ),
        max_pair_candidates=max(
            2, safe_int(metadata.get("max_pair_candidates"), 12)
        ),
    )
    if not actions:
        log["skip_reason"] = "no_counterfactual_actions"
        return query_plan_log, log

    rows = [action_features(action) for action in actions]
    query_features = query_summary_features(rows)
    route = route_feature_regime(ranker.model, query_features)
    routed_regime_key = str(route.get("regime_key") or "__global__")
    regime_key = routed_regime_key
    profiles = ranker.model.get("regimes", {}) or {}
    experts = ranker.model.get("experts", {}) or {}
    profile = profiles.get(regime_key) or {}
    expert = experts.get(regime_key) or {}
    if not profile or not expert:
        regime_key = "__global__"
        profile = profiles.get(regime_key) or {}
        expert = experts.get(regime_key) or {}
    log.update({
        "feature_regime": regime_key,
        "feature_regime_distance": round(_safe_float(route.get("distance")), 6),
        "feature_router_available": bool(route.get("available", False)),
        "policy_source": "challenger",
    })
    if not profile or not expert:
        log["skip_reason"] = "feature_regime_model_missing"
        return query_plan_log, log
    incumbent_fallbacks = ranker.model.get("incumbent_fallbacks", {}) or {}
    if (
        not bool(profile.get("enabled", False))
        and bool(metadata.get("incumbent_policy_fallback_enabled", False))
    ):
        fallback = incumbent_fallbacks.get(regime_key, {}) or {}
        fallback_profile = fallback.get("profile", {}) or {}
        fallback_expert_key = str(fallback.get("expert_key") or regime_key)
        fallback_expert = experts.get(fallback_expert_key, {}) or {}
        if (
            bool(fallback.get("enabled", False))
            and bool(fallback_profile.get("enabled", False))
            and fallback_expert
        ):
            log["incumbent_regime_fallback_applied"] = True
            log["incumbent_regime_fallback_reason"] = str(
                fallback.get("challenger_disabled_reason")
                or profile.get("disabled_reason")
                or "challenger_not_admitted"
            )
            log["incumbent_regime_fallback_expert"] = fallback_expert_key
            log["policy_source"] = "incumbent"
            profile = fallback_profile
            expert = fallback_expert
    allow_signature_fallback = bool(metadata.get(
        "soft_signature_fallback",
        metadata.get("ontology_signature_constrained_opportunity", False),
    ))
    holdout_certified_fallback = bool(metadata.get(
        "holdout_certified_signature_fallback", False
    ))
    certified_fallback_profile = (
        profile.get("certified_global_fallback_profile", {}) or {}
    )
    if (
        not bool(profile.get("enabled", False))
        and allow_signature_fallback
        and regime_key != "__global__"
        and bool((profiles.get("__global__", {}) or {}).get("enabled", False))
        and (
            not holdout_certified_fallback
            or (
                bool(profile.get("fallback_holdout_certified", False))
                and bool(certified_fallback_profile.get("enabled", False))
            )
        )
    ):
        log["signature_regime_fallback_reason"] = str(
            profile.get("disabled_reason", "signature_regime_not_admitted")
        )
        regime_key = "__global__"
        profile = (
            certified_fallback_profile
            if holdout_certified_fallback
            else profiles.get(regime_key) or {}
        )
        expert = experts.get(regime_key) or {}
        log["feature_regime"] = regime_key
        log["signature_regime_fallback_applied"] = True
        log["policy_source"] = "global_fallback"
    if not bool(profile.get("enabled", False)):
        log["skip_reason"] = str(
            profile.get("disabled_reason", "feature_regime_not_admitted")
        )
        return query_plan_log, log

    query_opportunity, query_uncertainty = predict_ensemble(
        expert.get("query_gate_head", {}) or {}, query_features
    )
    query_opportunity = min(max(query_opportunity, 0.0), 1.0)
    target_metric = str(profile.get("target_metric") or metadata.get("target_metric") or "r2")
    target_metric = "r5" if target_metric.lower() in {"r5", "delta_r5"} else "r2"
    metric_name = f"delta_{target_metric}"
    access_path_profiles = profile.get("access_path_profiles", {}) or {}
    proof_tiered = bool(
        metadata.get("proof_tiered_access_paths", False)
        and access_path_profiles
    )
    proof_dominance_guard = bool(
        metadata.get("proof_dominance_displacement_guard", False)
        and target_metric == "r5"
    )
    min_opportunity = _safe_float(
        profile.get("min_opportunity_probability"), math.inf
    )
    min_action_lcb = _safe_float(profile.get("min_action_delta_lcb"), math.inf)

    scored: List[Dict[str, Any]] = []
    for action, features in zip(actions, rows):
        prediction = _prediction_bundle(expert, features)
        access_path = proof_access_path(features) if proof_tiered else "all"
        action_profile = (
            access_path_profiles.get(access_path, {}) or {}
            if proof_tiered else profile
        )
        path_enabled = bool(action_profile.get("enabled", False))
        safety = (
            incumbent_safety_profile(action, prediction, action_profile)
            if path_enabled else {
                "passed": False,
                "reason": "uncertified_proof_access_path",
                "metric_lcb": -math.inf,
            }
        )
        metric_lcb = _safe_float(safety.get("metric_lcb"), prediction[metric_name])
        path_min_opportunity = _safe_float(
            action_profile.get("min_opportunity_probability"), math.inf
        )
        path_min_action_lcb = _safe_float(
            action_profile.get("min_action_delta_lcb"), math.inf
        )
        path_opportunity_pass = bool(
            path_enabled and query_opportunity > path_min_opportunity
        )
        path_action_pass = bool(
            path_enabled and metric_lcb > path_min_action_lcb
        )
        dominance = (
            proof_displacement_dominance_profile(action)
            if proof_dominance_guard else {
                "enabled": False,
                "passed": True,
                "reason": "proof_dominance_guard_disabled",
                "violations": [],
            }
        )
        path_admitted = bool(
            path_opportunity_pass
            and path_action_pass
            and safety.get("passed", True)
        )
        scored.append({
            **action,
            "v233_prediction": prediction,
            "v233_metric_delta_lcb": metric_lcb,
            "v233_incumbent_safety": safety,
            "v242_access_path": access_path,
            "v242_access_path_enabled": path_enabled,
            "v242_access_path_opportunity_pass": path_opportunity_pass,
            "v242_access_path_action_pass": path_action_pass,
            "v242_access_path_admitted": path_admitted,
            "v249_proof_dominance": dominance,
            "v249_proof_dominance_pass": bool(
                dominance.get("passed", True)
            ),
            "v242_min_opportunity_probability": path_min_opportunity,
            "v242_min_action_delta_lcb": path_min_action_lcb,
        })
    scored.sort(
        key=lambda action: (
            _safe_float(action.get("v233_metric_delta_lcb"), -math.inf),
            _safe_float(
                (action.get("v233_prediction", {}) or {}).get(metric_name),
                -math.inf,
            ),
            _safe_float((action.get("features", {}) or {}).get("candidate_score")),
        ),
        reverse=True,
    )
    admissible = [
        action for action in scored
        if (
            bool(action.get("v242_access_path_admitted", False))
            if proof_tiered else
            bool((action.get("v233_incumbent_safety", {}) or {}).get("passed", True))
        )
    ]
    selected = admissible[0] if admissible else {}
    selected_dominance_pass = bool(
        selected.get("v249_proof_dominance_pass", True)
    )
    if proof_tiered:
        min_opportunity = _safe_float(
            selected.get("v242_min_opportunity_probability"), math.inf
        )
        min_action_lcb = _safe_float(
            selected.get("v242_min_action_delta_lcb"), math.inf
        )
        opportunity_pass = bool(
            selected.get("v242_access_path_opportunity_pass", False)
        )
        safety_pass = bool(selected)
        action_pass = bool(selected.get("v242_access_path_action_pass", False))
        approved = bool(selected and selected_dominance_pass)
    else:
        opportunity_pass = query_opportunity > min_opportunity
        safety_pass = bool(selected)
        action_pass = bool(
            selected
            and _safe_float(selected.get("v233_metric_delta_lcb"), -math.inf)
            > min_action_lcb
        )
        approved = bool(
            opportunity_pass
            and safety_pass
            and action_pass
            and selected_dominance_pass
        )
    log.update({
        "scored_count": len(scored),
        "target_metric": target_metric,
        "opportunity_probability": round(query_opportunity, 6),
        "opportunity_uncertainty": round(query_uncertainty, 6),
        "min_opportunity_probability": min_opportunity,
        "min_action_delta_lcb": min_action_lcb,
        "opportunity_gate_pass": opportunity_pass,
        "incumbent_safety_pass": safety_pass,
        "action_delta_gate_pass": action_pass,
        "incumbent_safety_pass_count": len(admissible),
        "incumbent_safety_veto_count": len(scored) - len(admissible),
        "proof_dominance_displacement_guard": proof_dominance_guard,
        "proof_dominance_veto_count": sum(
            not bool(action.get("v249_proof_dominance_pass", True))
            for action in scored
        ),
        "selected_proof_dominance_pass": selected_dominance_pass,
        "selected_proof_dominance_veto": bool(
            selected and not selected_dominance_pass
        ),
        "proof_tiered_access_paths": proof_tiered,
        "enabled_access_paths": sorted(
            key for key, value in access_path_profiles.items()
            if value.get("enabled", False)
        ),
        "selected_access_path": str(
            selected.get("v242_access_path") or "no_op"
        ),
        "routed_feature_regime": routed_regime_key,
        "opportunity_router_type": str(route.get("router_type") or "centroid"),
        "ontology_signature_leaf": route.get("signature_leaf"),
        "ontology_signature_path": route.get("signature_path", []),
        "ontology_signature_opportunity_prior": round(
            _safe_float(route.get("signature_opportunity_prior")), 6
        ),
    })
    if approved:
        action_kind = str(selected.get("action_kind") or "")
        log["selected_action"] = {
            key: value
            for key, value in selected.items()
            if key not in {"features", "v233_prediction"}
        }
        log["selected_action"]["prediction"] = selected["v233_prediction"]
        log["promoted_count"] = 1
        log["no_op"] = False
        log["committed_top5_count"] = int(action_kind == "replace_top5_slot5")
        log["committed_top2_count"] = int(action_kind != "replace_top5_slot5")
        log["committed_pair_count"] = int(action_kind == "replace_top2_pair")
    else:
        log["skip_reason"] = (
            "proof_dominance_displacement_veto"
            if proof_dominance_guard
            and selected
            and not selected_dominance_pass
            else "no_holdout_certified_proof_access_path"
            if proof_tiered and not admissible
            else "no_residual_evidence_opportunity"
            if not opportunity_pass
            else "incumbent_safety_veto"
            if not safety_pass
            else "nonpositive_metric_delta_lcb"
        )

    for action in scored[:10]:
        prediction = action.get("v233_prediction", {}) or {}
        log["top_actions"].append({
            "action_kind": action.get("action_kind"),
            "passage_ids": action.get("passage_ids", []),
            "candidate_ids": action.get("candidate_ids", []),
            "action_fingerprint": action.get("action_fingerprint", ""),
            "predicted_metric_delta": round(
                _safe_float(prediction.get(metric_name)), 6
            ),
            "predicted_loss_probability": round(
                _safe_float(prediction.get("loss")), 6
            ),
            "metric_delta_lcb": round(
                _safe_float(action.get("v233_metric_delta_lcb")), 6
            ),
            "proof_access_path": action.get("v242_access_path", "all"),
            "proof_access_path_enabled": bool(
                action.get("v242_access_path_enabled", not proof_tiered)
            ),
            "proof_access_path_admitted": bool(
                action.get("v242_access_path_admitted", False)
            ),
            "proof_dominance_pass": bool(
                action.get("v249_proof_dominance_pass", True)
            ),
            "proof_dominance_reason": str(
                (action.get("v249_proof_dominance", {}) or {}).get(
                    "reason", ""
                )
            ),
            "approved": bool(approved and action is selected),
        })
    return query_plan_log, log


def rank_query_grouped_nonlinear_delta(
    ranker,
    query_plan_log: Dict[str, Any],
    *,
    action_generator: Callable[..., List[Dict[str, Any]]],
    safe_int: Callable[[Any, int], int],
    candidate_compiler: str = "end_to_end_counterfactual_evidence_relation",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if bool(ranker._metadata().get("feature_routed_residual_opportunity", False)):
        return rank_feature_routed_residual_opportunity(
            ranker,
            query_plan_log,
            action_generator=action_generator,
            safe_int=safe_int,
            candidate_compiler=candidate_compiler,
        )
    dataset = ranker._dataset_name(query_plan_log)
    incumbent = query_plan_log.get("candidate_admission_incumbent", {}) or {}
    profile = (ranker.model.get("datasets", {}) or {}).get(dataset, {}) or {}
    metadata = ranker._metadata()
    log: Dict[str, Any] = {
        "enabled": True,
        "model_path": ranker.model_path,
        "model_ready": ranker.is_ready(),
        "mode": "query_grouped_nonlinear_delta_admission",
        "candidate_compiler": candidate_compiler,
        "dataset": dataset,
        "scored_count": 0,
        "promoted_count": 0,
        "committed_top2_count": 0,
        "committed_top5_count": 0,
        "committed_pair_count": 0,
        "no_op": True,
        "top_actions": [],
        "state_hash": incumbent.get("state_hash", ""),
        "state_version": incumbent.get("state_version", ""),
    }
    if bool(incumbent.get("defer_decision", False)):
        log["skip_reason"] = "deferred_until_final_state"
        log["deferred"] = True
        return query_plan_log, log
    dataset_enabled, reason = ranker._dataset_is_enabled(dataset)
    if not dataset_enabled:
        log["skip_reason"] = reason
        return query_plan_log, log
    if not bool(profile.get("enabled", False)):
        log["skip_reason"] = str(
            profile.get("disabled_reason", "dataset_grouped_oof_admission_disabled")
        )
        return query_plan_log, log

    actions = action_generator(
        query_plan_log,
        max_single_candidates=max(1, safe_int(metadata.get("max_single_candidates"), 24)),
        max_pair_candidates=max(2, safe_int(metadata.get("max_pair_candidates"), 12)),
    )
    if not actions:
        log["skip_reason"] = "no_end_to_end_actions"
        return query_plan_log, log

    experts = ranker.model.get("experts", {}) or {}
    requested_expert = str(profile.get("expert_key") or dataset)
    expert_key = requested_expert if requested_expert in experts else dataset
    if expert_key not in experts:
        expert_key = "__global__"
    expert = experts.get(expert_key) or {}
    if not expert:
        log["skip_reason"] = "nonlinear_expert_missing"
        return query_plan_log, log

    rows = [action_features(action) for action in actions]
    query_features = query_summary_features(rows)
    query_gain, query_uncertainty = predict_ensemble(
        expert.get("query_gate_head", {}) or {}, query_features
    )
    query_gain = min(max(query_gain, 0.0), 1.0)
    min_query_gain = _safe_float(profile.get("min_query_gain_probability"), -math.inf)
    policy = profile.get("policy", {}) or {}
    min_action_score = _safe_float(profile.get("min_action_score"), math.inf)

    scored: List[Dict[str, Any]] = []
    for action, features in zip(actions, rows):
        prediction = _prediction_bundle(expert, features)
        score = _action_score(prediction, policy)
        safety = incumbent_safety_profile(action, prediction, profile)
        scored.append({
            **action,
            "v220_prediction": prediction,
            "v220_action_score": score,
            "v221_incumbent_safety": safety,
        })
    scored.sort(
        key=lambda action: (
            _safe_float(action.get("v220_action_score"), -math.inf),
            _safe_float(
                (action.get("features", {}) or {}).get("candidate_score"),
                -math.inf,
            ),
        ),
        reverse=True,
    )
    admissible = [
        action
        for action in scored
        if bool((action.get("v221_incumbent_safety", {}) or {}).get("passed", True))
    ]
    selected = admissible[0] if admissible else {}
    gate_pass = query_gain > min_query_gain
    safety_pass = bool(selected)
    action_pass = bool(
        selected
        and _safe_float(selected.get("v220_action_score"), -math.inf) > min_action_score
    )
    approved = gate_pass and safety_pass and action_pass
    log.update({
        "scored_count": len(scored),
        "rank_partition_action_count": sum(
            bool(action.get("rank_partition_action")) for action in scored
        ),
        "top5_invariant_action_count": sum(
            bool(action.get("top5_membership_invariant")) for action in scored
        ),
        "expert_key": expert_key,
        "incumbent_safe_admission": bool(
            profile.get("incumbent_safe_admission", False)
        ),
        "incumbent_safety_pass_count": len(admissible),
        "incumbent_safety_veto_count": len(scored) - len(admissible),
        "query_gain_probability": round(query_gain, 6),
        "query_gate_uncertainty": round(query_uncertainty, 6),
        "min_query_gain_probability": min_query_gain,
        "min_action_score": min_action_score,
        "query_gate_pass": gate_pass,
        "incumbent_safety_pass": safety_pass,
        "action_gate_pass": action_pass,
    })
    if approved:
        action_kind = str(selected.get("action_kind") or "")
        log["selected_action"] = {
            key: value
            for key, value in selected.items()
            if key not in {"features", "v220_prediction"}
        }
        log["selected_action"]["prediction"] = selected["v220_prediction"]
        log["promoted_count"] = 1
        log["no_op"] = False
        if action_kind == "replace_top5_slot5":
            log["committed_top5_count"] = 1
        else:
            log["committed_top2_count"] = 1
        if action_kind == "replace_top2_pair":
            log["committed_pair_count"] = 1
    else:
        log["skip_reason"] = (
            "query_no_op_gate_rejected" if not gate_pass
            else "incumbent_safety_veto" if not safety_pass
            else "action_value_gate_rejected"
        )

    for action in scored[:10]:
        prediction = action.get("v220_prediction", {}) or {}
        log["top_actions"].append({
            "action_kind": action.get("action_kind"),
            "passage_ids": action.get("passage_ids", []),
            "candidate_ids": action.get("candidate_ids", []),
            "sources": action.get("sources", []),
            "source_signature": action.get("source_signature", "unknown"),
            "action_fingerprint": action.get("action_fingerprint", ""),
            "rank_partition_action": bool(action.get("rank_partition_action")),
            "top5_membership_invariant": bool(
                action.get("top5_membership_invariant")
            ),
            "nonlinear_action_score": round(
                _safe_float(action.get("v220_action_score")), 6
            ),
            "predicted_winner_probability": round(
                _safe_float(prediction.get("winner")), 6
            ),
            "predicted_gain_probability": round(
                _safe_float(prediction.get("gain")), 6
            ),
            "predicted_loss_probability": round(
                _safe_float(prediction.get("loss")), 6
            ),
            "predicted_delta_r2": round(
                _safe_float(prediction.get("delta_r2")), 6
            ),
            "predicted_delta_r5": round(
                _safe_float(prediction.get("delta_r5")), 6
            ),
            "incumbent_safety": {
                key: round(value, 6) if isinstance(value, float) else value
                for key, value in (
                    action.get("v221_incumbent_safety", {}) or {}
                ).items()
            },
            "approved": bool(approved and action is selected),
        })
    return query_plan_log, log
