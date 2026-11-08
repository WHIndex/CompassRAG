"""Learned, metric-aligned admission for factorized retrieval actions."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence


FEATURE_NAMES = [
    "target_top2",
    "target_top5",
    "operator_entity_scan",
    "operator_cascaded_join",
    "operator_comparison_join",
    "operator_evidence_warehouse",
    "action_top2_single",
    "action_top2_group",
    "action_top5_single",
    "action_top5_chain",
    "dataset_2wiki",
    "dataset_musique",
    "dataset_nq",
    "dataset_popqa",
    "utility",
    "semantic_score",
    "contract_confidence",
    "predicate_coverage",
    "anchor_grounding",
    "branch_complete",
    "branch_confidence",
    "qualifier_coverage",
    "relation_score_min",
    "relation_score_mean",
    "type_score_min",
    "type_score_mean",
    "branch_balance",
    "chain_completeness",
    "source_in_top2",
    "selected_doc_count",
    "external_doc_count",
    "missing_target_fraction",
    "displaced_slots",
    "best_original_rank_log_inv",
    "worst_original_rank_log_inv",
    "mean_probe_rows",
    "max_probe_rows",
    "dense_probe_fraction",
    "warehouse_text_score",
    "warehouse_path_score",
    "warehouse_confidence",
    "warehouse_coverage_gain",
    "candidate_choice_rank_inv",
    "candidate_choice_fraction",
    "query_document_overlap",
    "relation_term_coverage",
    "answer_type_match",
    "selected_score_mean",
    "displaced_score_mean",
    "selected_displaced_score_margin",
]



V185_FEATURE_NAMES = ["action_no_op", *FEATURE_NAMES]

HIERARCHICAL_GATE_FEATURE_NAMES = [
    *V185_FEATURE_NAMES,
    "group_candidate_count_log",
    "group_rank_score",
    "group_rank_margin",
    "group_utility_max",
    "group_utility_mean",
    "group_semantic_max",
    "group_semantic_mean",
    "group_warehouse_fraction",
    "group_external_fraction",
    "group_top2_group_fraction",
    "group_chain_fraction",
]

V186_SET_FEATURE_NAMES = [
    "action_evidence_set",
    "set_compiler_score",
    "set_slot_coverage",
    "set_branch_coverage",
    "set_provenance_diversity",
    "set_redundancy",
    "set_execution_cost",
    "set_slot_gain",
    "set_branch_gain",
    "set_member_count",
    "set_external_count",
]
V186_FEATURE_NAMES = [*V185_FEATURE_NAMES, *V186_SET_FEATURE_NAMES]
V186_HIERARCHICAL_GATE_FEATURE_NAMES = [
    *HIERARCHICAL_GATE_FEATURE_NAMES,
    *V186_SET_FEATURE_NAMES,
    "group_set_fraction",
    "group_set_score_max",
    "group_set_cost_mean",
    "group_set_slot_coverage_max",
    "group_set_branch_coverage_max",
]

V187_MISSING_SLOT_FEATURE_NAMES = [
    "action_missing_slot_recovery",
    "missing_slot_new_to_action_space",
    "missing_slot_terminal_only_projection",
    "missing_slot_probe_cost",
    "missing_slot_relation_count",
]
V187_FEATURE_NAMES = [*V185_FEATURE_NAMES, *V187_MISSING_SLOT_FEATURE_NAMES]
V187_HIERARCHICAL_GATE_FEATURE_NAMES = [
    *HIERARCHICAL_GATE_FEATURE_NAMES,
    *V187_MISSING_SLOT_FEATURE_NAMES,
    "group_missing_slot_fraction",
    "group_new_action_space_fraction",
    "group_missing_slot_cost_mean",
]

V188_PROOF_FEATURE_NAMES = [
    "action_proof_carrying_join",
    "proof_valid",
    "proof_branch_complete",
    "proof_relation_coverage",
    "proof_type_consistency",
    "proof_provenance_coverage",
    "proof_exact_binding_coverage",
    "proof_return_projection_valid",
    "proof_confidence",
    "proof_member_count",
]
V188_FEATURE_NAMES = [*V187_FEATURE_NAMES, *V188_PROOF_FEATURE_NAMES]
V188_HIERARCHICAL_GATE_FEATURE_NAMES = [
    *V187_HIERARCHICAL_GATE_FEATURE_NAMES,
    *V188_PROOF_FEATURE_NAMES,
    "group_proof_fraction",
    "group_valid_proof_fraction",
    "group_proof_confidence_max",
    "group_proof_member_count_mean",
]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: Any, low: float = 0.0, high: float = 1.0) -> float:
    return min(max(_safe_float(value), low), high)


def _rank_log_inv(rank: Any) -> float:
    try:
        rank_value = max(int(rank), 2)
    except (TypeError, ValueError):
        rank_value = 10**9
    return 1.0 / max(math.log2(rank_value), 1.0)


def factorized_action_features(profile: Mapping[str, Any]) -> Dict[str, float]:
    """Convert an action profile into the stable feature contract."""
    dataset = str(profile.get("dataset", "") or "").strip().lower()
    operator = str(profile.get("operator", "") or "").strip().lower()
    target = str(profile.get("target", "") or "").strip().lower()
    action = str(profile.get("action", "") or "").strip().lower()
    relation_scores = [
        _clamp(value) for value in profile.get("relation_scores", []) or []
    ]
    type_scores = [_clamp(value) for value in profile.get("type_scores", []) or []]
    original_ranks = list(profile.get("original_ranks", []) or [])
    probe_rows = [
        max(_safe_float(item.get("matched_count"), 0.0), 0.0)
        for item in profile.get("probes", []) or []
        if isinstance(item, Mapping)
    ]
    dense_probes = sum(
        1
        for item in profile.get("probes", []) or []
        if isinstance(item, Mapping) and item.get("access_path") == "dense_fallback"
    )
    probe_count = len(list(profile.get("probes", []) or []))
    selected_count = max(int(_safe_float(profile.get("selected_doc_count"), 0.0)), 0)
    return {
        "target_top2": 1.0 if target == "top2" else 0.0,
        "target_top5": 1.0 if target != "top2" else 0.0,
        "operator_entity_scan": 1.0 if operator == "entity_relation_index_scan" else 0.0,
        "operator_cascaded_join": 1.0 if operator == "cascaded_bind_join" else 0.0,
        "operator_comparison_join": 1.0 if operator == "dual_branch_comparison_join" else 0.0,
        "operator_evidence_warehouse": 1.0 if operator == "evidence_warehouse_scan" else 0.0,
        "action_top2_single": 1.0 if target == "top2" and "group" not in action else 0.0,
        "action_top2_group": 1.0 if target == "top2" and "group" in action else 0.0,
        "action_top5_single": 1.0 if target == "top5" and "chain" not in action else 0.0,
        "action_top5_chain": 1.0 if target == "top5" and "chain" in action else 0.0,
        "dataset_2wiki": 1.0 if dataset == "2wikimultihopqa" else 0.0,
        "dataset_musique": 1.0 if dataset == "musique" else 0.0,
        "dataset_nq": 1.0 if dataset == "nq_rear" else 0.0,
        "dataset_popqa": 1.0 if dataset == "popqa" else 0.0,
        "utility": _clamp(profile.get("utility")),
        "semantic_score": _clamp(profile.get("semantic_score")),
        "contract_confidence": _clamp(profile.get("contract_confidence")),
        "predicate_coverage": _clamp(profile.get("predicate_coverage")),
        "anchor_grounding": _clamp(profile.get("anchor_grounding")),
        "branch_complete": 1.0 if profile.get("branch_complete") else 0.0,
        "branch_confidence": _clamp(profile.get("branch_confidence")),
        "qualifier_coverage": _clamp(profile.get("qualifier_coverage"), 0.0, 1.0),
        "relation_score_min": min(relation_scores, default=0.0),
        "relation_score_mean": sum(relation_scores) / max(len(relation_scores), 1),
        "type_score_min": min(type_scores, default=0.0),
        "type_score_mean": sum(type_scores) / max(len(type_scores), 1),
        "branch_balance": _clamp(profile.get("branch_balance")),
        "chain_completeness": _clamp(profile.get("chain_completeness")),
        "source_in_top2": 1.0 if profile.get("source_in_top2") else 0.0,
        "selected_doc_count": min(selected_count / 3.0, 1.0),
        "external_doc_count": min(
            max(int(_safe_float(profile.get("external_doc_count"), 0.0)), 0) / 3.0,
            1.0,
        ),
        "missing_target_fraction": _clamp(profile.get("missing_target_fraction")),
        "displaced_slots": min(
            max(int(_safe_float(profile.get("displaced_slots"), 0.0)), 0) / 3.0,
            1.0,
        ),
        "best_original_rank_log_inv": max(
            (_rank_log_inv(rank) for rank in original_ranks),
            default=0.0,
        ),
        "worst_original_rank_log_inv": min(
            (_rank_log_inv(rank) for rank in original_ranks),
            default=0.0,
        ),
        "mean_probe_rows": min(sum(probe_rows) / max(len(probe_rows), 1) / 64.0, 1.0),
        "max_probe_rows": min(max(probe_rows, default=0.0) / 64.0, 1.0),
        "dense_probe_fraction": dense_probes / max(probe_count, 1),
        "warehouse_text_score": _clamp(profile.get("warehouse_text_score")),
        "warehouse_path_score": _clamp(profile.get("warehouse_path_score")),
        "warehouse_confidence": _clamp(profile.get("warehouse_confidence")),
        "warehouse_coverage_gain": _clamp(
            _safe_float(profile.get("warehouse_coverage_gain"), 0.0) / 5.0
        ),
        "candidate_choice_rank_inv": 1.0 / max(
            _safe_float(profile.get("candidate_choice_rank"), 1.0),
            1.0,
        ),
        "candidate_choice_fraction": _clamp(profile.get("candidate_choice_fraction")),
        "query_document_overlap": _clamp(profile.get("query_document_overlap")),
        "relation_term_coverage": _clamp(profile.get("relation_term_coverage")),
        "answer_type_match": _clamp(profile.get("answer_type_match")),
        "selected_score_mean": _clamp(profile.get("selected_score_mean")),
        "displaced_score_mean": _clamp(profile.get("displaced_score_mean")),
        "selected_displaced_score_margin": _clamp(
            profile.get("selected_displaced_score_margin")
        ),
    }


def factorized_action_features_v185(profile: Mapping[str, Any]) -> Dict[str, float]:
    """Return the explicit-no-op feature contract used by hierarchical models."""
    features = factorized_action_features(profile)
    action = str(profile.get("action", "") or "").strip().lower()
    is_no_op = action == "no_op"
    features["action_no_op"] = 1.0 if is_no_op else 0.0
    if not is_no_op:
        return {name: float(features.get(name, 0.0)) for name in V185_FEATURE_NAMES}





    keep = {
        "dataset_2wiki",
        "dataset_musique",
        "dataset_nq",
        "dataset_popqa",
    }
    for name in FEATURE_NAMES:
        if name not in keep:
            features[name] = 0.0
    return {name: float(features.get(name, 0.0)) for name in V185_FEATURE_NAMES}


def factorized_action_features_v186(profile: Mapping[str, Any]) -> Dict[str, float]:
    """Add ontology-set coverage and physical-cost features to the base contract."""
    features = factorized_action_features_v185(profile)
    action = str(profile.get("action", "") or "").strip().lower()
    member_count = max(int(_safe_float(profile.get("set_member_count"))), 0)
    external_count = max(int(_safe_float(profile.get("set_external_count"))), 0)
    features.update({
        "action_evidence_set": 1.0 if "evidence_set" in action else 0.0,
        "set_compiler_score": _clamp(profile.get("set_compiler_score")),
        "set_slot_coverage": _clamp(profile.get("set_slot_coverage")),
        "set_branch_coverage": _clamp(profile.get("set_branch_coverage")),
        "set_provenance_diversity": _clamp(profile.get("set_provenance_diversity")),
        "set_redundancy": _clamp(profile.get("set_redundancy")),
        "set_execution_cost": _clamp(profile.get("set_execution_cost")),
        "set_slot_gain": min(max(_safe_float(profile.get("set_slot_gain")), 0.0) / 3.0, 1.0),
        "set_branch_gain": min(max(_safe_float(profile.get("set_branch_gain")), 0.0) / 3.0, 1.0),
        "set_member_count": min(member_count / 3.0, 1.0),
        "set_external_count": min(external_count / 3.0, 1.0),
    })
    return {name: float(features.get(name, 0.0)) for name in V186_FEATURE_NAMES}


def factorized_action_features_v187(profile: Mapping[str, Any]) -> Dict[str, float]:
    """Add candidate-space expansion and dependent-join cost features."""
    features = factorized_action_features_v185(profile)
    features.update({
        "action_missing_slot_recovery": 1.0 if profile.get("missing_slot_recovery") else 0.0,
        "missing_slot_new_to_action_space": (
            1.0 if profile.get("missing_slot_new_to_action_space") else 0.0
        ),
        "missing_slot_terminal_only_projection": (
            1.0 if profile.get("missing_slot_terminal_only_projection") else 0.0
        ),
        "missing_slot_probe_cost": min(
            max(_safe_float(profile.get("missing_slot_probe_cost")), 0.0) / 8.0,
            1.0,
        ),
        "missing_slot_relation_count": min(
            max(_safe_float(profile.get("missing_slot_relation_count")), 0.0) / 4.0,
            1.0,
        ),
    })
    return {name: float(features.get(name, 0.0)) for name in V187_FEATURE_NAMES}


def factorized_action_features_v188(profile: Mapping[str, Any]) -> Dict[str, float]:
    """Represent typed proof validity separately from missing-slot expansion."""
    features = factorized_action_features_v187(profile)
    features.update({
        "action_proof_carrying_join": 1.0 if profile.get("proof_carrying_join") else 0.0,
        "proof_valid": 1.0 if profile.get("proof_valid") else 0.0,
        "proof_branch_complete": 1.0 if profile.get("proof_branch_complete") else 0.0,
        "proof_relation_coverage": _clamp(profile.get("proof_relation_coverage")),
        "proof_type_consistency": _clamp(profile.get("proof_type_consistency")),
        "proof_provenance_coverage": _clamp(profile.get("proof_provenance_coverage")),
        "proof_exact_binding_coverage": _clamp(profile.get("proof_exact_binding_coverage")),
        "proof_return_projection_valid": (
            1.0 if profile.get("proof_return_projection_valid") else 0.0
        ),
        "proof_confidence": _clamp(profile.get("proof_confidence")),
        "proof_member_count": min(
            max(_safe_float(profile.get("proof_member_count")), 0.0) / 5.0,
            1.0,
        ),
    })
    return {name: float(features.get(name, 0.0)) for name in V188_FEATURE_NAMES}


def hierarchical_gate_features(
    profile: Mapping[str, Any],
    candidate_profiles: Sequence[Mapping[str, Any]],
    rank_scores: Sequence[float],
    selected_index: int,
) -> Dict[str, float]:
    """Build a query-action gate row after the target-local ranker picks a winner."""
    features = factorized_action_features_v185(profile)
    candidate_profiles = list(candidate_profiles)
    scores = [float(value) for value in rank_scores]
    count = max(len(candidate_profiles), 1)
    selected_score = scores[selected_index] if 0 <= selected_index < len(scores) else 0.0
    runner_up = max(
        (score for idx, score in enumerate(scores) if idx != selected_index),
        default=selected_score,
    )
    utilities = [_clamp(item.get("utility")) for item in candidate_profiles]
    semantics = [_clamp(item.get("semantic_score")) for item in candidate_profiles]
    external = [
        min(max(_safe_float(item.get("external_doc_count")), 0.0) / 3.0, 1.0)
        for item in candidate_profiles
    ]
    actions = [
        str(item.get("action", "") or "").strip().lower()
        for item in candidate_profiles
    ]
    operators = [
        str(item.get("operator", "") or "").strip().lower()
        for item in candidate_profiles
    ]
    features.update({
        "group_candidate_count_log": min(math.log1p(count) / math.log(25.0), 1.0),
        "group_rank_score": selected_score,
        "group_rank_margin": selected_score - runner_up,
        "group_utility_max": max(utilities, default=0.0),
        "group_utility_mean": sum(utilities) / count,
        "group_semantic_max": max(semantics, default=0.0),
        "group_semantic_mean": sum(semantics) / count,
        "group_warehouse_fraction": sum(
            operator == "evidence_warehouse_scan" for operator in operators
        ) / count,
        "group_external_fraction": sum(external) / count,
        "group_top2_group_fraction": sum("top2" in action and "group" in action for action in actions) / count,
        "group_chain_fraction": sum("chain" in action for action in actions) / count,
    })
    return {
        name: float(features.get(name, 0.0))
        for name in HIERARCHICAL_GATE_FEATURE_NAMES
    }


def hierarchical_gate_features_v186(
    profile: Mapping[str, Any],
    candidate_profiles: Sequence[Mapping[str, Any]],
    rank_scores: Sequence[float],
    selected_index: int,
) -> Dict[str, float]:
    """Build a target-local risk gate with set-space opportunity context."""
    features = hierarchical_gate_features(
        profile=profile,
        candidate_profiles=candidate_profiles,
        rank_scores=rank_scores,
        selected_index=selected_index,
    )
    selected_features = factorized_action_features_v186(profile)
    for name in V186_SET_FEATURE_NAMES:
        features[name] = selected_features[name]
    candidates = list(candidate_profiles)
    set_candidates = [
        item for item in candidates
        if "evidence_set" in str(item.get("action", "") or "").strip().lower()
    ]
    count = max(len(candidates), 1)
    features.update({
        "group_set_fraction": len(set_candidates) / count,
        "group_set_score_max": max(
            (_clamp(item.get("set_compiler_score")) for item in set_candidates),
            default=0.0,
        ),
        "group_set_cost_mean": sum(
            _clamp(item.get("set_execution_cost")) for item in set_candidates
        ) / max(len(set_candidates), 1),
        "group_set_slot_coverage_max": max(
            (_clamp(item.get("set_slot_coverage")) for item in set_candidates),
            default=0.0,
        ),
        "group_set_branch_coverage_max": max(
            (_clamp(item.get("set_branch_coverage")) for item in set_candidates),
            default=0.0,
        ),
    })
    return {
        name: float(features.get(name, 0.0))
        for name in V186_HIERARCHICAL_GATE_FEATURE_NAMES
    }


def hierarchical_gate_features_v187(
    profile: Mapping[str, Any],
    candidate_profiles: Sequence[Mapping[str, Any]],
    rank_scores: Sequence[float],
    selected_index: int,
) -> Dict[str, float]:
    """Build a query-level gate with missing-slot opportunity and probe cost."""
    features = hierarchical_gate_features(
        profile=profile,
        candidate_profiles=candidate_profiles,
        rank_scores=rank_scores,
        selected_index=selected_index,
    )
    selected_features = factorized_action_features_v187(profile)
    for name in V187_MISSING_SLOT_FEATURE_NAMES:
        features[name] = selected_features[name]
    candidates = list(candidate_profiles)
    missing = [item for item in candidates if item.get("missing_slot_recovery")]
    count = max(len(candidates), 1)
    features.update({
        "group_missing_slot_fraction": len(missing) / count,
        "group_new_action_space_fraction": sum(
            bool(item.get("missing_slot_new_to_action_space")) for item in missing
        ) / count,
        "group_missing_slot_cost_mean": sum(
            min(max(_safe_float(item.get("missing_slot_probe_cost")), 0.0) / 8.0, 1.0)
            for item in missing
        ) / max(len(missing), 1),
    })
    return {
        name: float(features.get(name, 0.0))
        for name in V187_HIERARCHICAL_GATE_FEATURE_NAMES
    }


def hierarchical_gate_features_v188(
    profile: Mapping[str, Any],
    candidate_profiles: Sequence[Mapping[str, Any]],
    rank_scores: Sequence[float],
    selected_index: int,
) -> Dict[str, float]:
    """Gate typed proof actions using validity, coverage, and set capacity."""
    features = hierarchical_gate_features_v187(
        profile=profile,
        candidate_profiles=candidate_profiles,
        rank_scores=rank_scores,
        selected_index=selected_index,
    )
    selected_features = factorized_action_features_v188(profile)
    for name in V188_PROOF_FEATURE_NAMES:
        features[name] = selected_features[name]
    candidates = list(candidate_profiles)
    proof_actions = [item for item in candidates if item.get("proof_carrying_join")]
    count = max(len(candidates), 1)
    features.update({
        "group_proof_fraction": len(proof_actions) / count,
        "group_valid_proof_fraction": sum(
            bool(item.get("proof_valid")) for item in proof_actions
        ) / count,
        "group_proof_confidence_max": max(
            (_clamp(item.get("proof_confidence")) for item in proof_actions),
            default=0.0,
        ),
        "group_proof_member_count_mean": sum(
            min(max(_safe_float(item.get("proof_member_count")), 0.0) / 5.0, 1.0)
            for item in proof_actions
        ) / max(len(proof_actions), 1),
    })
    return {
        name: float(features.get(name, 0.0))
        for name in V188_HIERARCHICAL_GATE_FEATURE_NAMES
    }


class FactorizedDeltaAdmissionModel:
    """Load logistic gain/loss heads and make a conservative Pareto decision."""

    def __init__(self, model_path: str = "") -> None:
        self.model_path = str(model_path or "")
        self.model: Dict[str, Any] = {}
        self.error = ""
        if not self.model_path:
            self.error = "model_path_empty"
            return
        path = Path(self.model_path).expanduser()
        if not path.is_file():
            self.error = "model_not_found"
            return
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if payload.get("type") not in {
                "factorized_delta_admission_model",
                "factorized_action_replay_model",
                "factorized_listwise_action_cbo_model",
                "factorized_hierarchical_action_cbo_model",
                "factorized_risk_calibrated_set_cbo_model",
                "factorized_missing_slot_hierarchical_cbo_model",
                "factorized_proof_join_hierarchical_cbo_model",
            }:
                self.error = "model_type_mismatch"
                return
            self.model = payload
        except (OSError, ValueError, TypeError) as exc:
            self.error = f"model_load_error:{type(exc).__name__}"

    @property
    def ready(self) -> bool:
        return bool(self.model) and not self.error

    @property
    def listwise(self) -> bool:
        return self.model.get("type") in {
            "factorized_listwise_action_cbo_model",
            "factorized_hierarchical_action_cbo_model",
            "factorized_risk_calibrated_set_cbo_model",
            "factorized_missing_slot_hierarchical_cbo_model",
            "factorized_proof_join_hierarchical_cbo_model",
        }

    @property
    def hierarchical(self) -> bool:
        return self.model.get("type") in {
            "factorized_hierarchical_action_cbo_model",
            "factorized_risk_calibrated_set_cbo_model",
            "factorized_missing_slot_hierarchical_cbo_model",
            "factorized_proof_join_hierarchical_cbo_model",
        }

    @property
    def evidence_set_model(self) -> bool:
        return self.model.get("type") == "factorized_risk_calibrated_set_cbo_model"

    @property
    def missing_slot_model(self) -> bool:
        return self.model.get("type") == "factorized_missing_slot_hierarchical_cbo_model"

    @property
    def proof_join_model(self) -> bool:
        return self.model.get("type") == "factorized_proof_join_hierarchical_cbo_model"

    @staticmethod
    def _sigmoid(value: float) -> float:
        value = min(max(float(value), -40.0), 40.0)
        return 1.0 / (1.0 + math.exp(-value))

    @classmethod
    def _score_head(cls, head: Mapping[str, Any], features: Mapping[str, float]) -> float:
        names = list(head.get("feature_names", []) or FEATURE_NAMES)
        weights = list(head.get("weights", []) or [])
        means = list(head.get("means", []) or [])
        stds = list(head.get("stds", []) or [])
        if len(weights) != len(names):
            return 0.0
        total = _safe_float(head.get("bias"), 0.0)
        for idx, name in enumerate(names):
            value = _safe_float(features.get(name), 0.0)
            mean = _safe_float(means[idx], 0.0) if idx < len(means) else 0.0
            std = max(abs(_safe_float(stds[idx], 1.0)), 1e-6) if idx < len(stds) else 1.0
            total += _safe_float(weights[idx], 0.0) * ((value - mean) / std)
        return cls._sigmoid(total)

    @staticmethod
    def _score_regression_head(
        head: Mapping[str, Any],
        features: Mapping[str, float],
    ) -> float:
        names = list(head.get("feature_names", []) or FEATURE_NAMES)
        weights = list(head.get("weights", []) or [])
        means = list(head.get("means", []) or [])
        stds = list(head.get("stds", []) or [])
        if len(weights) != len(names):
            return 0.0
        total = _safe_float(head.get("bias"), 0.0)
        for idx, name in enumerate(names):
            value = _safe_float(features.get(name), 0.0)
            mean = _safe_float(means[idx], 0.0) if idx < len(means) else 0.0
            std = max(abs(_safe_float(stds[idx], 1.0)), 1e-6) if idx < len(stds) else 1.0
            total += _safe_float(weights[idx], 0.0) * ((value - mean) / std)
        return min(max(float(total), -1.0), 1.0)

    def _section(self, dataset: str, operator: str, target: str) -> Mapping[str, Any]:
        sections = self.model.get("sections", {}) or {}
        keys = (
            f"{dataset}|{operator}|{target}",
            f"{operator}|{target}",
            target,
            "all",
        )
        for key in keys:
            section = sections.get(key)
            if isinstance(section, Mapping) and section.get("enabled", True):
                return section
        return {}

    def _dataset_listwise_section(self, dataset: str) -> Mapping[str, Any]:
        """Return an exact, validated dataset policy; never cross domain boundaries."""
        if not self.listwise:
            return {}
        dataset = str(dataset or "").strip().lower()
        supported = {
            str(value or "").strip().lower()
            for value in (self.model.get("metadata", {}) or {}).get(
                "supported_datasets", []
            ) or []
            if str(value or "").strip()
        }
        if dataset not in supported:
            return {}
        section = (self.model.get("dataset_sections", {}) or {}).get(dataset, {})
        if not isinstance(section, Mapping) or not section.get("enabled", False):
            return {}
        return section

    @staticmethod
    def _score_linear_head(
        head: Mapping[str, Any],
        features: Mapping[str, float],
    ) -> float:
        names = list(head.get("feature_names", []) or FEATURE_NAMES)
        weights = list(head.get("weights", []) or [])
        means = list(head.get("means", []) or [])
        stds = list(head.get("stds", []) or [])
        if len(weights) != len(names):
            return 0.0
        total = _safe_float(head.get("bias"), 0.0)
        for idx, name in enumerate(names):
            value = _safe_float(features.get(name), 0.0)
            mean = _safe_float(means[idx], 0.0) if idx < len(means) else 0.0
            std = max(abs(_safe_float(stds[idx], 1.0)), 1e-6) if idx < len(stds) else 1.0
            total += _safe_float(weights[idx], 0.0) * ((value - mean) / std)
        return float(total)

    def decide_group(
        self,
        profiles: Sequence[Mapping[str, Any]],
        fail_open: bool = False,
    ) -> list[Dict[str, Any]]:
        """Score one query's actions jointly against its explicit no-op action."""
        feature_fn = (
            factorized_action_features_v186
            if self.evidence_set_model else
            factorized_action_features_v188
            if self.proof_join_model else
            factorized_action_features_v187
            if self.missing_slot_model else
            factorized_action_features_v185
            if self.hierarchical else factorized_action_features
        )
        feature_rows = [feature_fn(profile) for profile in profiles]
        if not profiles:
            return []
        dataset = str(
            next(
                (
                    profile.get("dataset", "")
                    for profile in profiles
                    if str(profile.get("action", "") or "") != "no_op"
                ),
                profiles[0].get("dataset", ""),
            )
            or ""
        ).strip().lower()
        if not self.ready:
            return [
                {
                    "approved": bool(fail_open),
                    "reason": (
                        "model_unavailable_fail_open"
                        if fail_open else self.error or "model_unavailable"
                    ),
                    "model_ready": False,
                    "features": features,
                }
                for features in feature_rows
            ]
        section = self._dataset_listwise_section(dataset)
        if not section:
            decisions = []
            for profile, features in zip(profiles, feature_rows):
                is_no_op = str(profile.get("action", "") or "") == "no_op"
                decisions.append({
                    "approved": bool(is_no_op or fail_open),
                    "reason": (
                        "explicit_no_op"
                        if is_no_op else
                        "unsupported_domain_fail_open"
                        if fail_open else "unsupported_domain_abstain"
                    ),
                    "model_ready": True,
                    "dataset_supported": False,
                    "dataset": dataset,
                    "features": features,
                })
            return decisions

        if self.hierarchical:
            return self._decide_hierarchical_group(
                profiles=profiles,
                feature_rows=feature_rows,
                dataset=dataset,
                section=section,
            )

        rank_head = section.get("rank_head", {}) or {}
        scores = [self._score_linear_head(rank_head, features) for features in feature_rows]
        no_op_index = next(
            (
                idx for idx, profile in enumerate(profiles)
                if str(profile.get("action", "") or "") == "no_op"
            ),
            -1,
        )
        no_op_score = scores[no_op_index] if no_op_index >= 0 else 0.0
        min_margin = _safe_float(section.get("min_score_margin"), 0.0)
        decisions = []
        for profile, features, score in zip(profiles, feature_rows, scores):
            action = str(profile.get("action", "") or "")
            margin = float(score - no_op_score)
            is_no_op = action == "no_op"
            approved = bool(is_no_op or margin >= min_margin)
            decisions.append({
                "approved": approved,
                "reason": (
                    "explicit_no_op"
                    if is_no_op else
                    "grouped_oof_listwise_gain"
                    if approved else "listwise_score_below_noop"
                ),
                "model_ready": True,
                "dataset_supported": True,
                "dataset": dataset,
                "section": dataset,
                "listwise_score": round(float(score), 6),
                "no_op_score": round(float(no_op_score), 6),
                "score_margin": round(float(margin), 6),

                "expected_delta": round(float(margin), 6),
                "expected_delta_source": "query_group_listwise_margin",
                "min_score_margin": round(float(min_margin), 6),
                "loss_probability": 0.0,
                "grouped_oof_validation": dict(section.get("grouped_oof_validation", {}) or {}),
                "features": features,
            })
        return decisions

    def _decide_hierarchical_group(
        self,
        profiles: Sequence[Mapping[str, Any]],
        feature_rows: Sequence[Mapping[str, float]],
        dataset: str,
        section: Mapping[str, Any],
    ) -> list[Dict[str, Any]]:
        """Rank actions within each target, then decide whether either winner beats no-op."""
        decisions: list[Dict[str, Any]] = []
        target_sections = section.get("target_sections", {}) or {}
        candidate_indices_by_target: Dict[str, list[int]] = {}
        for idx, profile in enumerate(profiles):
            action = str(profile.get("action", "") or "").strip().lower()
            if action == "no_op":
                continue
            if not bool(profile.get("hard_admissible", True)):
                continue
            target = str(profile.get("target", "") or "").strip().lower()
            candidate_indices_by_target.setdefault(target, []).append(idx)

        selected_by_index: Dict[int, Dict[str, Any]] = {}
        for target, indices in candidate_indices_by_target.items():
            target_section = target_sections.get(target, {}) or {}
            if not target_section.get("enabled", False):
                continue
            rank_head = target_section.get("rank_head", {}) or {}
            target_profiles = [profiles[idx] for idx in indices]
            rank_scores = [
                self._score_linear_head(rank_head, feature_rows[idx])
                for idx in indices
            ]
            winner_local_idx = max(
                range(len(indices)),
                key=lambda idx: rank_scores[idx],
            )
            winner_idx = indices[winner_local_idx]
            gate_feature_fn = (
                hierarchical_gate_features_v186
                if self.evidence_set_model else hierarchical_gate_features
            )
            if self.missing_slot_model:
                gate_feature_fn = hierarchical_gate_features_v187
            if self.proof_join_model:
                gate_feature_fn = hierarchical_gate_features_v188
            gate_features = gate_feature_fn(
                profile=profiles[winner_idx],
                candidate_profiles=target_profiles,
                rank_scores=rank_scores,
                selected_index=winner_local_idx,
            )
            gain_probability = self._score_head(
                target_section.get("gain_head", {}) or {}, gate_features
            )
            loss_probability = self._score_head(
                target_section.get("loss_head", {}) or {}, gate_features
            )
            mean_gain = max(_safe_float(target_section.get("mean_gain")), 0.0)
            mean_loss = max(_safe_float(target_section.get("mean_loss_magnitude")), 0.0)
            expected_delta = gain_probability * mean_gain - loss_probability * mean_loss
            min_expected_delta = _safe_float(
                target_section.get("min_expected_delta"), 0.0
            )
            max_loss_probability = _safe_float(
                target_section.get("max_loss_probability"), 1.0
            )
            approved = bool(
                expected_delta >= min_expected_delta
                and loss_probability <= max_loss_probability
            )
            selected_by_index[winner_idx] = {
                "approved": approved,
                "reason": (
                    "hierarchical_opportunity_gain"
                    if approved else
                    "hierarchical_predicted_loss"
                    if loss_probability > max_loss_probability else
                    "hierarchical_no_op_preferred"
                ),
                "model_ready": True,
                "dataset_supported": True,
                "dataset": dataset,
                "section": f"{dataset}|{target}",
                "target": target,
                "listwise_score": round(float(rank_scores[winner_local_idx]), 6),
                "no_op_score": 0.0,
                "score_margin": round(float(expected_delta - min_expected_delta), 6),
                "gain_probability": round(float(gain_probability), 6),
                "loss_probability": round(float(loss_probability), 6),
                "mean_gain": round(float(mean_gain), 6),
                "mean_loss_magnitude": round(float(mean_loss), 6),
                "expected_delta": round(float(expected_delta), 6),
                "expected_delta_source": "hierarchical_gain_loss_gate",
                "min_expected_delta": round(float(min_expected_delta), 6),
                "max_loss_probability": round(float(max_loss_probability), 6),
                "rank_candidate_count": len(indices),
                "gate_features": gate_features,
                "grouped_oof_validation": dict(
                    target_section.get("grouped_oof_validation", {}) or {}
                ),
                "features": dict(feature_rows[winner_idx]),
            }

        for idx, (profile, features) in enumerate(zip(profiles, feature_rows)):
            action = str(profile.get("action", "") or "").strip().lower()
            if action == "no_op":
                decisions.append({
                    "approved": True,
                    "reason": "explicit_no_op",
                    "model_ready": True,
                    "dataset_supported": True,
                    "dataset": dataset,
                    "expected_delta": 0.0,
                    "loss_probability": 0.0,
                    "features": dict(features),
                })
                continue
            if idx in selected_by_index:
                decisions.append(selected_by_index[idx])
                continue
            target = str(profile.get("target", "") or "").strip().lower()
            reason = (
                "hierarchical_rank_loser"
                if (target_sections.get(target, {}) or {}).get("enabled", False)
                else "hierarchical_target_unsupported"
            )
            decisions.append({
                "approved": False,
                "reason": reason,
                "model_ready": True,
                "dataset_supported": True,
                "dataset": dataset,
                "target": target,
                "expected_delta": 0.0,
                "loss_probability": 0.0,
                "features": dict(features),
            })
        return decisions

    def decide(
        self,
        profile: Mapping[str, Any],
        min_expected_delta: float = 0.0,
        max_loss_probability: float = 0.20,
        fail_open: bool = False,
    ) -> Dict[str, Any]:
        features = factorized_action_features(profile)
        if self.listwise:
            decisions = self.decide_group([profile], fail_open=fail_open)
            return decisions[0] if decisions else {
                "approved": False,
                "reason": "empty_action_group",
                "model_ready": self.ready,
                "features": features,
            }
        if not self.ready:
            return {
                "approved": bool(fail_open),
                "reason": "model_unavailable_fail_open" if fail_open else self.error or "model_unavailable",
                "model_ready": False,
                "features": features,
            }
        dataset = str(profile.get("dataset", "") or "").strip().lower()
        operator = str(profile.get("operator", "") or "").strip().lower()
        target = str(profile.get("target", "top5") or "top5").strip().lower()
        section = self._section(dataset, operator, target)
        if not section:
            return {
                "approved": bool(fail_open),
                "reason": "no_compatible_section_fail_open" if fail_open else "no_compatible_section",
                "model_ready": True,
                "features": features,
            }
        gain_probability = self._score_head(section.get("gain_head", {}) or {}, features)
        loss_probability = self._score_head(section.get("loss_head", {}) or {}, features)
        gain_scale = max(_safe_float(section.get("mean_gain"), 0.0), 0.0)
        loss_scale = max(_safe_float(section.get("mean_loss_magnitude"), 0.0), 0.0)
        delta_head = section.get("delta_head", {}) or {}
        if delta_head:
            expected_delta = self._score_regression_head(delta_head, features)
            expected_delta_source = "direct_counterfactual_regression"
        else:
            expected_delta = gain_probability * gain_scale - loss_probability * loss_scale
            expected_delta_source = "gain_loss_probability_product"
        approved = bool(
            loss_probability <= float(max_loss_probability)
            and expected_delta >= float(min_expected_delta)
        )
        return {
            "approved": approved,
            "reason": "learned_pareto_gain" if approved else (
                "predicted_displacement_loss"
                if loss_probability > float(max_loss_probability)
                else "insufficient_expected_delta"
            ),
            "model_ready": True,
            "section": str(section.get("name", "")),
            "gain_probability": round(float(gain_probability), 6),
            "loss_probability": round(float(loss_probability), 6),
            "mean_gain": round(float(gain_scale), 6),
            "mean_loss_magnitude": round(float(loss_scale), 6),
            "expected_delta": round(float(expected_delta), 6),
            "expected_delta_source": expected_delta_source,
            "min_expected_delta": round(float(min_expected_delta), 6),
            "max_loss_probability": round(float(max_loss_probability), 6),
            "features": features,
        }
