"""Incumbent-preserving residual re-optimization for evidence actions."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

try:
    from .query_plan_delta_admission import (
        V186_SET_FEATURE_NAMES,
        V188_FEATURE_NAMES,
        factorized_action_features_v186,
        factorized_action_features_v188,
    )
except ImportError:
    _FEATURE_PATH = Path(__file__).with_name("query_plan_delta_admission.py")
    _FEATURE_SPEC = importlib.util.spec_from_file_location(
        "query_plan_incumbent_delta_features", _FEATURE_PATH
    )
    if _FEATURE_SPEC is None or _FEATURE_SPEC.loader is None:
        raise RuntimeError(f"Cannot load feature contract from {_FEATURE_PATH}")
    _FEATURE_MODULE = importlib.util.module_from_spec(_FEATURE_SPEC)
    _FEATURE_SPEC.loader.exec_module(_FEATURE_MODULE)
    V186_SET_FEATURE_NAMES = list(_FEATURE_MODULE.V186_SET_FEATURE_NAMES)
    V188_FEATURE_NAMES = list(_FEATURE_MODULE.V188_FEATURE_NAMES)
    factorized_action_features_v186 = _FEATURE_MODULE.factorized_action_features_v186
    factorized_action_features_v188 = _FEATURE_MODULE.factorized_action_features_v188


V193_BASE_FEATURE_NAMES = [
    *V188_FEATURE_NAMES,
    *[
        name for name in V186_SET_FEATURE_NAMES
        if name not in V188_FEATURE_NAMES
    ],
]

V193_PAIR_CONTEXT_FEATURE_NAMES = [
    "incumbent_is_no_op",
    "incumbent_is_protected",
    "candidate_is_protected",
    "same_target",
    "candidate_changes_top2",
    "candidate_changes_top5",
    "candidate_count_log",
    "target_candidate_count_log",
    "candidate_utility_rank_inv",
    "target_utility_rank_inv",
]

V193_RELATIVE_FEATURE_NAMES = [
    f"relative_{name}" for name in V193_BASE_FEATURE_NAMES
]
V193_FEATURE_NAMES = [
    *[f"candidate_{name}" for name in V193_BASE_FEATURE_NAMES],
    *[f"incumbent_{name}" for name in V193_BASE_FEATURE_NAMES],
    *V193_RELATIVE_FEATURE_NAMES,
    *V193_PAIR_CONTEXT_FEATURE_NAMES,
]


V194_OPPORTUNITY_SIGNAL_FEATURES = [
    "candidate_utility",
    "relative_utility",
    "candidate_semantic_score",
    "relative_semantic_score",
    "candidate_contract_confidence",
    "relative_contract_confidence",
    "candidate_predicate_coverage",
    "relative_predicate_coverage",
    "candidate_anchor_grounding",
    "relative_anchor_grounding",
    "candidate_branch_confidence",
    "relative_branch_confidence",
    "candidate_chain_completeness",
    "relative_chain_completeness",
    "candidate_missing_target_fraction",
    "relative_missing_target_fraction",
    "candidate_warehouse_text_score",
    "relative_warehouse_text_score",
    "candidate_warehouse_path_score",
    "relative_warehouse_path_score",
    "candidate_warehouse_confidence",
    "relative_warehouse_confidence",
    "candidate_query_document_overlap",
    "relative_query_document_overlap",
    "candidate_relation_term_coverage",
    "relative_relation_term_coverage",
    "candidate_answer_type_match",
    "relative_answer_type_match",
    "candidate_selected_displaced_score_margin",
    "relative_selected_displaced_score_margin",
    "candidate_set_compiler_score",
    "relative_set_compiler_score",
    "candidate_set_slot_coverage",
    "relative_set_slot_coverage",
    "candidate_set_branch_coverage",
    "relative_set_branch_coverage",
    "candidate_set_provenance_diversity",
    "relative_set_provenance_diversity",
    "candidate_set_redundancy",
    "relative_set_redundancy",
    "candidate_set_execution_cost",
    "relative_set_execution_cost",
]

V194_OPPORTUNITY_FEATURE_NAMES = [
    "candidate_count_log",
    "incumbent_is_no_op",
    "incumbent_is_protected",
    *[
        f"{aggregate}_{name}"
        for name in V194_OPPORTUNITY_SIGNAL_FEATURES
        for aggregate in ("max", "mean", "min")
    ],
]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp01(value: Any) -> float:
    return min(max(_safe_float(value), 0.0), 1.0)


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _as_string_set(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        raw_values = value.replace(";", ",").split(",")
    elif isinstance(value, (list, tuple, set)):
        raw_values = value
    else:
        raw_values = [value]
    return {
        str(item).strip().lower()
        for item in raw_values
        if str(item).strip()
    }


def is_protected_incumbent(profile: Mapping[str, Any]) -> bool:
    """Return whether an incumbent carries a complete proof/chain contract."""
    action = str(profile.get("action", "") or "").strip().lower()
    operator = str(profile.get("operator", "") or "").strip().lower()
    return bool(
        action in {"chain_complete_top5", "proof_set_top5"}
        or action.startswith("proof_")
        or operator == "proof_carrying_dependent_join"
        or (
            profile.get("proof_valid", False)
            and profile.get("proof_branch_complete", False)
        )
    )


def incumbent_action_family(profile: Mapping[str, Any]) -> str:
    """Map physical actions to stable expert families."""
    action = str(profile.get("action", "") or "").strip().lower()
    operator = str(profile.get("operator", "") or "").strip().lower()
    if action == "no_op":
        return "no_op"
    if "warehouse" in action or operator == "evidence_warehouse_scan":
        return "warehouse"
    if "evidence_set" in action or operator == "ontology_evidence_set_compiler":
        return "evidence_set"
    if action.startswith("proof_") or "proof" in operator:
        return "proof"
    if action.startswith("candidate_window_chain"):
        return "chain"
    if action.startswith("atomic_"):
        return "atomic"
    if action.startswith("simple_entity_relation") or operator == "entity_relation_index_scan":
        return "entity_relation"
    if "missing_slot" in action or "missing_slot" in operator:
        return "missing_slot"
    if "comparison" in action or "comparison" in operator:
        return "comparison"
    if "chain" in action or "join" in operator:
        return "chain"
    return "other"


def build_v194_opportunity_features(
    pair_feature_rows: Sequence[Mapping[str, Any]],
    candidate_indices: Sequence[int],
) -> Dict[str, float]:
    """Aggregate a query/family action group into a gold-free opportunity row."""
    indices = [
        int(idx) for idx in candidate_indices
        if 0 <= int(idx) < len(pair_feature_rows)
    ]
    if not indices:
        return {name: 0.0 for name in V194_OPPORTUNITY_FEATURE_NAMES}
    first = pair_feature_rows[indices[0]]
    features: Dict[str, float] = {
        "candidate_count_log": min(
            math.log1p(len(indices)) / math.log(81.0), 1.0
        ),
        "incumbent_is_no_op": float(first.get("incumbent_is_no_op", 0.0)),
        "incumbent_is_protected": float(
            first.get("incumbent_is_protected", 0.0)
        ),
    }
    for name in V194_OPPORTUNITY_SIGNAL_FEATURES:
        values = [
            float(pair_feature_rows[idx].get(name, 0.0)) for idx in indices
        ]
        features[f"max_{name}"] = max(values)
        features[f"mean_{name}"] = sum(values) / max(len(values), 1)
        features[f"min_{name}"] = min(values)
    return {
        name: float(features.get(name, 0.0))
        for name in V194_OPPORTUNITY_FEATURE_NAMES
    }


def factorized_action_features_v193(
    profile: Mapping[str, Any],
) -> Dict[str, float]:
    """Combine proof/missing-slot features with evidence-set physical features."""
    features = factorized_action_features_v188(profile)
    set_features = factorized_action_features_v186(profile)
    for name in V186_SET_FEATURE_NAMES:
        features[name] = float(set_features.get(name, 0.0))
    return {
        name: float(features.get(name, 0.0))
        for name in V193_BASE_FEATURE_NAMES
    }


def build_incumbent_pair_feature_rows(
    profiles: Sequence[Mapping[str, Any]],
    incumbent_index: int,
) -> List[Dict[str, float]]:
    """Build candidate features relative to the selected incumbent plan."""
    rows = [dict(profile or {}) for profile in profiles]
    if not rows:
        return []
    if incumbent_index < 0 or incumbent_index >= len(rows):
        raise ValueError("incumbent_index is outside the action group")

    base_rows = [factorized_action_features_v193(row) for row in rows]
    incumbent = rows[incumbent_index]
    incumbent_features = base_rows[incumbent_index]
    incumbent_target = str(incumbent.get("target", "") or "").strip().lower()
    candidate_indices = [
        idx for idx, row in enumerate(rows)
        if idx != incumbent_index
        and str(row.get("action", "") or "").strip().lower() != "no_op"
    ]
    candidate_count = max(len(candidate_indices), 1)

    output: List[Dict[str, float]] = []
    for idx, (row, candidate_features) in enumerate(zip(rows, base_rows)):
        target = str(row.get("target", "") or "").strip().lower()
        target_indices = [
            candidate_idx for candidate_idx in candidate_indices
            if str(rows[candidate_idx].get("target", "") or "").strip().lower()
            == target
        ]
        utility = candidate_features.get("utility", 0.0)
        utility_rank = 1 + sum(
            base_rows[candidate_idx].get("utility", 0.0) > utility
            for candidate_idx in candidate_indices
        )
        target_utility_rank = 1 + sum(
            base_rows[candidate_idx].get("utility", 0.0) > utility
            for candidate_idx in target_indices
        )
        features: Dict[str, float] = {}
        for name in V193_BASE_FEATURE_NAMES:
            candidate_value = float(candidate_features.get(name, 0.0))
            incumbent_value = float(incumbent_features.get(name, 0.0))
            features[f"candidate_{name}"] = candidate_value
            features[f"incumbent_{name}"] = incumbent_value
            features[f"relative_{name}"] = candidate_value - incumbent_value
        features.update({
            "incumbent_is_no_op": 1.0 if str(
                incumbent.get("action", "") or ""
            ).strip().lower() == "no_op" else 0.0,
            "incumbent_is_protected": (
                1.0 if is_protected_incumbent(incumbent) else 0.0
            ),
            "candidate_is_protected": (
                1.0 if is_protected_incumbent(row) else 0.0
            ),
            "same_target": 1.0 if target == incumbent_target else 0.0,
            "candidate_changes_top2": 1.0 if row.get("changes_top2") else 0.0,
            "candidate_changes_top5": 1.0 if row.get("changes_top5") else 0.0,
            "candidate_count_log": min(
                math.log1p(len(candidate_indices)) / math.log(81.0), 1.0
            ),
            "target_candidate_count_log": min(
                math.log1p(len(target_indices)) / math.log(81.0), 1.0
            ),
            "candidate_utility_rank_inv": 1.0 / max(utility_rank, 1),
            "target_utility_rank_inv": 1.0 / max(target_utility_rank, 1),
        })
        output.append({
            name: float(features.get(name, 0.0))
            for name in V193_FEATURE_NAMES
        })
    return output


class IncumbentResidualReoptimizer:
    """Admit only a risk-bounded action that improves on an incumbent plan."""

    def __init__(self, model_path: str = "") -> None:
        self.model_path = str(model_path or "")
        self.ready = False
        self.error = ""
        self.bundle: Dict[str, Any] = {}
        if not self.model_path:
            self.error = "model_path_missing"
            return
        try:
            import joblib

            path = Path(self.model_path).expanduser()
            if path.suffix.lower() == ".json":
                manifest = json.loads(path.read_text(encoding="utf-8"))
                artifact_path = Path(str(manifest.get("artifact_path", "") or ""))
                if not artifact_path.is_absolute():
                    artifact_path = path.parent / artifact_path
                path = artifact_path
            self.bundle = dict(joblib.load(path) or {})
            if self.bundle.get("type") not in {
                "incumbent_relative_residual_optimizer",
                "hierarchical_family_pareto_optimizer",
                "r2_first_expert_slate_optimizer",
                "runtime_shadow_slate_verifier_optimizer",
                "state_conditioned_counterfactual_set_ranker",
                "query_grouped_counterfactual_slate_optimizer",
            }:
                raise ValueError("unsupported incumbent reoptimizer model type")
            if list(self.bundle.get("feature_names", []) or []) != V193_FEATURE_NAMES:
                raise ValueError("incumbent reoptimizer feature contract mismatch")
            if self.hierarchical_family and list(
                self.bundle.get("opportunity_feature_names", []) or []
            ) != V194_OPPORTUNITY_FEATURE_NAMES:
                raise ValueError("opportunity feature contract mismatch")
            if self.query_grouped_slate_optimizer and list(
                self.bundle.get("opportunity_feature_names", []) or []
            ) != V194_OPPORTUNITY_FEATURE_NAMES:
                raise ValueError("query-grouped opportunity feature contract mismatch")
            if (
                self.hierarchical_family
                and self.bundle.get("top2_local_feature_names")
                and list(self.bundle.get("top2_local_feature_names", []) or [])
                != V193_FEATURE_NAMES
            ):
                raise ValueError("top2 local feature contract mismatch")
            self.ready = True
        except Exception as exc:
            self.error = f"model_load_failed:{type(exc).__name__}:{exc}"
            self.bundle = {}

    @property
    def supported_datasets(self) -> List[str]:
        return sorted(set(
            (self.bundle.get("dataset_models", {}) or {}).keys()
        ) | set(
            (self.bundle.get("top2_local_sections", {}) or {}).keys()
        ))

    @property
    def hierarchical_family(self) -> bool:
        return self.bundle.get("type") in {
            "hierarchical_family_pareto_optimizer",
            "r2_first_expert_slate_optimizer",
            "runtime_shadow_slate_verifier_optimizer",
        }

    @property
    def r2_first_expert_slate_optimizer(self) -> bool:
        return self.bundle.get("type") == "r2_first_expert_slate_optimizer"

    @property
    def runtime_shadow_slate_verifier_optimizer(self) -> bool:
        return self.bundle.get("type") == "runtime_shadow_slate_verifier_optimizer"

    @property
    def state_conditioned_set_ranker(self) -> bool:
        return (
            self.bundle.get("type")
            == "state_conditioned_counterfactual_set_ranker"
        )

    @property
    def query_grouped_slate_optimizer(self) -> bool:
        return (
            self.bundle.get("type")
            == "query_grouped_counterfactual_slate_optimizer"
        )

    def _runtime_shadow_slate_decision(
        self,
        profile: Mapping[str, Any],
        incumbent_profile: Mapping[str, Any],
        decision: Mapping[str, Any],
        features: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Gold-free runtime verifier for risky top-2 slate mutations."""
        metadata = self.bundle.get("metadata", {}) or {}
        enabled = bool(
            self.runtime_shadow_slate_verifier_optimizer
            or metadata.get("runtime_shadow_slate_verifier", False)
        )
        target = str(profile.get("target", "") or "").strip().lower()
        changes_top2 = bool(profile.get("changes_top2", target == "top2"))
        base = {
            "runtime_shadow_slate_verifier": enabled,
            "shadow_slate_applicable": bool(
                enabled and target == "top2" and changes_top2
            ),
            "shadow_slate_approved": True,
            "shadow_slate_reason": "shadow_not_applicable",
            "shadow_slate_support_score": 0.0,
            "shadow_slate_risk_score": 0.0,
            "shadow_slate_margin": 0.0,
        }
        if not base["shadow_slate_applicable"]:
            return base

        def threshold(name: str, default: float) -> float:
            return _safe_float(metadata.get(name), default)

        action = str(profile.get("action", "") or "").strip().lower()
        family = incumbent_action_family(profile)
        semantic = _clamp01(profile.get("semantic_score"))
        utility = _clamp01(profile.get("utility"))
        contract = _clamp01(profile.get("contract_confidence"))
        predicate = _clamp01(profile.get("predicate_coverage"))
        anchor = _clamp01(profile.get("anchor_grounding"))
        branch = _clamp01(profile.get("branch_confidence"))
        qualifier = _clamp01(profile.get("qualifier_coverage"))
        chain = _clamp01(profile.get("chain_completeness"))
        query_overlap = _clamp01(profile.get("query_document_overlap"))
        relation = _clamp01(profile.get("relation_term_coverage"))
        answer_type = _clamp01(profile.get("answer_type_match"))
        warehouse_text = _clamp01(profile.get("warehouse_text_score"))
        warehouse_path = _clamp01(profile.get("warehouse_path_score"))
        warehouse_confidence = _clamp01(profile.get("warehouse_confidence"))
        set_score = _clamp01(profile.get("set_compiler_score"))
        set_slot = _clamp01(profile.get("set_slot_coverage"))
        set_branch = _clamp01(profile.get("set_branch_coverage"))
        proof_confidence = _clamp01(profile.get("proof_confidence"))
        proof_relation = _clamp01(profile.get("proof_relation_coverage"))
        proof_type = _clamp01(profile.get("proof_type_consistency"))
        proof_binding = _clamp01(profile.get("proof_exact_binding_coverage"))
        proof_valid = 1.0 if profile.get("proof_valid", False) else 0.0
        source_in_top2 = 1.0 if profile.get("source_in_top2", False) else 0.0
        p2_gain = _clamp01(decision.get("r2_gain_probability"))
        p2_loss = _clamp01(decision.get("r2_loss_probability"))
        p5_loss = _clamp01(decision.get("r5_loss_probability"))
        predicted_margin = max(p2_gain - p2_loss, 0.0)
        selected_count = max(
            int(_safe_float(profile.get("selected_doc_count"), 0.0)), 0
        )
        external_count = max(
            int(_safe_float(profile.get("external_doc_count"), 0.0)), 0
        )
        displaced_slots = max(
            int(_safe_float(profile.get("displaced_slots"), 0.0)), 0
        )
        missing_fraction = _clamp01(profile.get("missing_target_fraction"))
        selected_margin = _clamp01(
            profile.get("selected_displaced_score_margin")
        )

        ranks_value = profile.get("original_ranks", [])
        ranks: List[int] = []
        if isinstance(ranks_value, (list, tuple)):
            for rank_value in ranks_value:
                rank = int(_safe_float(rank_value, 0.0))
                if rank > 0:
                    ranks.append(rank)
        best_original_rank = min(ranks) if ranks else 10 ** 6
        best_rank_inv = _clamp01(features.get("candidate_best_original_rank_log_inv"))
        if best_rank_inv <= 0.0:
            best_rank_inv = _clamp01(features.get("best_original_rank_log_inv"))
        rank_risk = 1.0 - best_rank_inv if best_rank_inv > 0.0 else 1.0

        proof_support = max(
            proof_confidence,
            proof_relation,
            proof_type,
            proof_binding,
            proof_valid,
        )
        warehouse_support = max(warehouse_text, warehouse_path, warehouse_confidence)
        structural_support = max(
            contract,
            predicate,
            anchor,
            branch,
            qualifier,
            chain,
            set_score,
            set_slot,
            set_branch,
            proof_support,
        )
        lexical_support = max(
            query_overlap,
            relation,
            answer_type,
            warehouse_text,
            proof_relation,
        )
        certified = bool(
            proof_valid >= 1.0
            or proof_support >= 0.80
            or (
                family in {"chain", "proof", "evidence_set"}
                and structural_support >= 0.72
                and lexical_support >= 0.12
            )
        )
        support_score = min(
            1.0,
            0.13 * semantic
            + 0.10 * utility
            + 0.16 * query_overlap
            + 0.12 * relation
            + 0.09 * answer_type
            + 0.08 * contract
            + 0.07 * predicate
            + 0.06 * anchor
            + 0.05 * branch
            + 0.05 * chain
            + 0.05 * warehouse_support
            + 0.05 * proof_support
            + 0.04 * max(set_score, set_slot, set_branch)
            + 0.04 * source_in_top2
            + 0.06 * predicted_margin
        )
        risk_score = min(
            1.0,
            0.22 * min(displaced_slots / 2.0, 1.0)
            + 0.18 * min(external_count / 2.0, 1.0)
            + 0.17 * rank_risk
            + 0.14 * (1.0 - selected_margin)
            + 0.13 * missing_fraction
            + 0.10 * max(p2_loss, p5_loss)
            + 0.06 * (1.0 - max(lexical_support, structural_support))
        )
        if source_in_top2:
            risk_score = max(risk_score - 0.08, 0.0)
        if certified:
            risk_score = max(risk_score - 0.10, 0.0)
            support_score = min(support_score + 0.08, 1.0)
        if action == "top2_group" or selected_count >= 2:
            support_score = min(support_score + 0.04, 1.0)
            risk_score = max(risk_score - 0.04, 0.0)

        min_support = threshold("shadow_min_support_score", 0.34)
        max_risk = threshold("shadow_max_risk_score", 0.68)
        min_margin = threshold("shadow_min_support_risk_margin", -0.18)
        single_min_support = threshold("shadow_single_min_support_score", 0.40)
        single_max_rank = int(threshold("shadow_single_max_original_rank", 80.0))
        min_text_relation = threshold("shadow_min_text_or_relation_score", 0.025)
        margin = support_score - risk_score
        reason = "shadow_top2_approved"
        approved = True

        weak_model = bool(predicted_margin < threshold("shadow_weak_model_margin", 0.12))
        if semantic <= threshold("shadow_min_semantic_score", 0.025) and not certified:
            approved = False
            reason = "shadow_top2_low_semantic_support"
        elif (
            max(query_overlap, relation, warehouse_text, proof_relation)
            < min_text_relation
            and not certified
            and not source_in_top2
        ):
            approved = False
            reason = "shadow_top2_low_query_relation_support"
        elif answer_type <= 0.0 and support_score < min_support + 0.08 and weak_model:
            approved = False
            reason = "shadow_top2_answer_type_conflict"
        elif (
            selected_count <= 1
            and best_original_rank > single_max_rank
            and support_score < single_min_support
            and not certified
        ):
            approved = False
            reason = "shadow_top2_single_deep_weak_support"
        elif risk_score > max_risk and margin < min_margin + 0.05:
            approved = False
            reason = "shadow_top2_displacement_risk"
        elif support_score < min_support:
            approved = False
            reason = "shadow_top2_support_below_floor"
        elif margin < min_margin:
            approved = False
            reason = "shadow_top2_negative_support_risk_margin"

        base.update({
            "shadow_slate_approved": bool(approved),
            "shadow_slate_reason": reason,
            "shadow_slate_support_score": round(float(support_score), 8),
            "shadow_slate_risk_score": round(float(risk_score), 8),
            "shadow_slate_margin": round(float(margin), 8),
            "shadow_slate_family": family,
            "shadow_slate_best_original_rank": int(best_original_rank),
            "shadow_slate_lexical_support": round(float(lexical_support), 8),
            "shadow_slate_structural_support": round(float(structural_support), 8),
            "shadow_slate_certified": bool(certified),
            "shadow_slate_source_in_top2": bool(source_in_top2),
        })
        return base

    def _runtime_top2_deployment_guard_decision(
        self,
        profile: Mapping[str, Any],
        incumbent_profile: Mapping[str, Any],
        decision: Mapping[str, Any],
        features: Mapping[str, Any],
        section: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Deployment-time guard for top-2 mutations.

        Cross-validation policies are trained on counterfactual labels, but a
        positive replay score can still displace strong rank-2 incumbents at
        runtime. This guard is
        gold-free: it admits a top-2 mutation only when the learned delta,
        support evidence, and displacement risk agree.
        """
        metadata = self.bundle.get("metadata", {}) or {}
        enabled = bool(
            section.get("runtime_top2_deployment_guard", False)
            or metadata.get("runtime_top2_deployment_guard", False)
        )
        target = str(profile.get("target", "") or "").strip().lower()
        changes_top2 = bool(profile.get("changes_top2", target == "top2"))
        base = {
            "top2_deployment_guard": enabled,
            "top2_deployment_guard_applicable": bool(
                enabled and target == "top2" and changes_top2
            ),
            "top2_deployment_guard_approved": True,
            "top2_deployment_guard_reason": "top2_deployment_guard_not_applicable",
            "top2_deployment_guard_support_score": 0.0,
            "top2_deployment_guard_risk_score": 0.0,
            "top2_deployment_guard_margin": 0.0,
            "factoid_top2_aperture_guard": False,
            "factoid_top2_aperture_applicable": False,
            "factoid_top2_aperture_approved": True,
            "factoid_top2_aperture_reason": "factoid_top2_aperture_not_applicable",
            "operator_mutability_contract": False,
            "operator_mutability_contract_applicable": False,
            "operator_mutability_contract_approved": True,
            "operator_mutability_contract_reason": (
                "operator_mutability_contract_not_applicable"
            ),
        }
        if not base["top2_deployment_guard_applicable"]:
            return base

        def threshold(name: str, default: float) -> float:
            section_value = section.get(name)
            if section_value is not None and section_value != "":
                return _safe_float(section_value, default)
            return _safe_float(metadata.get(name), default)

        def mode_value(name: str, default: str) -> str:
            value = str(section.get(name, "") or metadata.get(name, "") or default)
            return value.strip().lower()

        level = mode_value("top2_deployment_guard_level", "standard")
        generalized = bool(
            section.get("generalized_top2_local", False)
            or decision.get("generalized_top2_local", False)
        )
        generalized_mode = mode_value(
            "generalized_top2_deployment_mode", "strict"
        )
        dataset = str(
            profile.get("dataset", "")
            or incumbent_profile.get("dataset", "")
            or ""
        ).strip().lower()
        family = incumbent_action_family(profile)
        action_name = str(profile.get("action", "") or "").strip().lower()
        p2_gain = _clamp01(decision.get("r2_gain_probability"))
        p2_loss = _clamp01(decision.get("r2_loss_probability"))
        p5_loss = _clamp01(decision.get("r5_loss_probability"))
        d2 = _safe_float(decision.get("expected_delta_r2"), 0.0)
        predicted_margin = p2_gain - p2_loss

        semantic = _clamp01(profile.get("semantic_score"))
        utility = _clamp01(profile.get("utility"))
        contract = _clamp01(profile.get("contract_confidence"))
        predicate = _clamp01(profile.get("predicate_coverage"))
        anchor = _clamp01(profile.get("anchor_grounding"))
        branch = _clamp01(profile.get("branch_confidence"))
        qualifier = _clamp01(profile.get("qualifier_coverage"))
        chain = _clamp01(profile.get("chain_completeness"))
        query_overlap = _clamp01(profile.get("query_document_overlap"))
        relation = _clamp01(profile.get("relation_term_coverage"))
        answer_type = _clamp01(profile.get("answer_type_match"))
        warehouse_text = _clamp01(profile.get("warehouse_text_score"))
        warehouse_path = _clamp01(profile.get("warehouse_path_score"))
        warehouse_confidence = _clamp01(profile.get("warehouse_confidence"))
        set_score = _clamp01(profile.get("set_compiler_score"))
        set_slot = _clamp01(profile.get("set_slot_coverage"))
        set_branch = _clamp01(profile.get("set_branch_coverage"))
        proof_confidence = _clamp01(profile.get("proof_confidence"))
        proof_relation = _clamp01(profile.get("proof_relation_coverage"))
        proof_type = _clamp01(profile.get("proof_type_consistency"))
        proof_binding = _clamp01(profile.get("proof_exact_binding_coverage"))
        proof_valid = 1.0 if profile.get("proof_valid", False) else 0.0
        source_in_top2 = 1.0 if profile.get("source_in_top2", False) else 0.0
        selected_margin = _clamp01(
            profile.get("selected_displaced_score_margin")
        )
        selected_count = max(
            int(_safe_float(profile.get("selected_doc_count"), 0.0)), 0
        )
        external_count = max(
            int(_safe_float(profile.get("external_doc_count"), 0.0)), 0
        )
        displaced_slots = max(
            int(_safe_float(profile.get("displaced_slots"), 0.0)), 0
        )
        missing_fraction = _clamp01(profile.get("missing_target_fraction"))

        ranks_value = profile.get("original_ranks", [])
        ranks: List[int] = []
        if isinstance(ranks_value, (list, tuple)):
            for rank_value in ranks_value:
                rank = int(_safe_float(rank_value, 0.0))
                if rank > 0:
                    ranks.append(rank)
        best_original_rank = min(ranks) if ranks else 10 ** 6
        best_rank_inv = _clamp01(features.get("candidate_best_original_rank_log_inv"))
        if best_rank_inv <= 0.0:
            best_rank_inv = _clamp01(features.get("best_original_rank_log_inv"))
        rank_risk = 1.0 - best_rank_inv if best_rank_inv > 0.0 else 1.0

        direct_text_support = max(
            query_overlap,
            relation,
            answer_type,
            warehouse_text,
            proof_relation,
        )
        structural_support = max(
            contract,
            predicate,
            anchor,
            branch,
            qualifier,
            chain,
            set_score,
            set_slot,
            set_branch,
            warehouse_path,
            warehouse_confidence,
            proof_confidence,
            proof_type,
            proof_binding,
            proof_valid,
        )
        certified = bool(
            proof_valid >= 1.0
            or proof_confidence >= 0.86
            or (
                family in {"chain", "proof", "evidence_set"}
                and structural_support >= 0.78
                and direct_text_support >= 0.10
            )
        )
        support_score = min(
            1.0,
            0.18 * semantic
            + 0.11 * utility
            + 0.24 * direct_text_support
            + 0.15 * structural_support
            + 0.10 * selected_margin
            + 0.08 * source_in_top2
            + 0.10 * max(predicted_margin, 0.0)
            + 0.04 * (1.0 if certified else 0.0),
        )
        risk_score = min(
            1.0,
            0.24 * p2_loss
            + 0.10 * p5_loss
            + 0.18 * rank_risk
            + 0.15 * (1.0 - selected_margin)
            + 0.13 * min(external_count / 2.0, 1.0)
            + 0.10 * min(displaced_slots / 2.0, 1.0)
            + 0.06 * missing_fraction
            + 0.04 * (0.0 if certified else 1.0),
        )
        if selected_count >= 2:
            support_score = min(support_score + 0.04, 1.0)
            risk_score = max(risk_score - 0.03, 0.0)
        if source_in_top2:
            risk_score = max(risk_score - 0.06, 0.0)
        if certified:
            risk_score = max(risk_score - 0.08, 0.0)

        if level == "strict":
            default_min_gain = 0.70
            default_max_loss = 0.09
            default_min_margin = 0.50
            default_min_d2 = 0.08
            default_min_support = 0.48
            default_min_direct = 0.04
            default_max_rank = 35.0
        else:
            default_min_gain = 0.62
            default_max_loss = 0.14
            default_min_margin = 0.30
            default_min_d2 = 0.03
            default_min_support = 0.40
            default_min_direct = 0.025
            default_max_rank = 80.0
        if generalized:
            default_min_gain = max(default_min_gain, 0.84)
            default_max_loss = min(default_max_loss, 0.05)
            default_min_margin = max(default_min_margin, 0.72)
            default_min_d2 = max(default_min_d2, 0.14)
            default_min_support = max(default_min_support, 0.62)
            default_min_direct = max(default_min_direct, 0.08)
            default_max_rank = min(default_max_rank, 12.0)

        min_gain = threshold("top2_deployment_min_gain_probability", default_min_gain)
        max_loss = threshold("top2_deployment_max_loss_probability", default_max_loss)
        min_margin = threshold("top2_deployment_min_gain_loss_margin", default_min_margin)
        min_d2 = threshold("top2_deployment_min_expected_r2", default_min_d2)
        min_support = threshold("top2_deployment_min_support_score", default_min_support)
        min_direct = threshold(
            "top2_deployment_min_direct_support_score", default_min_direct
        )
        max_rank = threshold("top2_deployment_max_original_rank", default_max_rank)
        support_risk_margin = support_score - risk_score
        factoid_guard_enabled = _as_bool(
            section.get(
                "runtime_factoid_top2_aperture_guard",
                metadata.get("runtime_factoid_top2_aperture_guard", False),
            ),
            False,
        )
        factoid_datasets = (
            _as_string_set(section.get("factoid_top2_aperture_guard_datasets"))
            or _as_string_set(metadata.get("factoid_top2_aperture_guard_datasets"))
            or {"nq_rear", "popqa"}
        )
        factoid_families = (
            _as_string_set(section.get("factoid_top2_aperture_families"))
            or _as_string_set(metadata.get("factoid_top2_aperture_families"))
            or {"atomic", "entity_relation", "missing_slot", "warehouse"}
        )
        factoid_requires_proof = _as_bool(
            section.get(
                "factoid_generalized_top2_requires_proof",
                metadata.get("factoid_generalized_top2_requires_proof", True),
            ),
            True,
        )
        factoid_min_direct = threshold(
            "factoid_top2_min_direct_support_score", 0.12
        )
        factoid_max_rank = threshold("factoid_top2_max_original_rank", 5.0)
        factoid_applicable = bool(
            factoid_guard_enabled and dataset in factoid_datasets
        )
        factoid_approved = True
        factoid_reason = (
            "factoid_top2_aperture_approved"
            if factoid_applicable
            else "factoid_top2_aperture_not_applicable"
        )
        if factoid_applicable and family in factoid_families:
            if generalized and factoid_requires_proof and not certified:
                factoid_approved = False
                factoid_reason = "factoid_top2_generalized_requires_proof"
            elif (
                direct_text_support < factoid_min_direct
                and source_in_top2 <= 0.0
                and not certified
            ):
                factoid_approved = False
                factoid_reason = "factoid_top2_direct_signal_below_floor"
            elif (
                best_original_rank > factoid_max_rank
                and source_in_top2 <= 0.0
                and not certified
            ):
                factoid_approved = False
                factoid_reason = "factoid_top2_candidate_outside_aperture"
            elif (
                selected_count <= 1
                and external_count > 0
                and source_in_top2 <= 0.0
                and not certified
            ):
                factoid_approved = False
                factoid_reason = "factoid_top2_external_singleton_without_anchor"

        operator_guard_enabled = _as_bool(
            section.get(
                "runtime_operator_mutability_contract",
                metadata.get("runtime_operator_mutability_contract", False),
            ),
            False,
        )
        operator_datasets = (
            _as_string_set(section.get("operator_mutability_contract_datasets"))
            or _as_string_set(metadata.get("operator_mutability_contract_datasets"))
            or {"2wikimultihopqa"}
        )
        operator_families = (
            _as_string_set(section.get("operator_mutability_contract_families"))
            or _as_string_set(metadata.get("operator_mutability_contract_families"))
            or {"warehouse", "entity_relation"}
        )
        operator_actions = (
            _as_string_set(section.get("operator_mutability_contract_actions"))
            or _as_string_set(metadata.get("operator_mutability_contract_actions"))
            or {"warehouse_candidate_top2", "simple_entity_relation_top2"}
        )
        operator_targets = (
            _as_string_set(section.get("operator_mutability_contract_targets"))
            or _as_string_set(metadata.get("operator_mutability_contract_targets"))
            or {"top2"}
        )
        operator_require_proof_families = (
            _as_string_set(section.get("operator_mutability_require_proof_families"))
            or _as_string_set(metadata.get("operator_mutability_require_proof_families"))
            or {"warehouse", "entity_relation"}
        )
        operator_min_direct = threshold(
            "operator_mutability_min_direct_support_score", 0.10
        )
        operator_min_structural = threshold(
            "operator_mutability_min_structural_support_score", 0.78
        )
        operator_min_chain = threshold(
            "operator_mutability_min_chain_completeness", 0.62
        )
        operator_max_rank = threshold(
            "operator_mutability_max_original_rank", 5.0
        )
        operator_allow_source_top2 = _as_bool(
            section.get(
                "operator_mutability_allow_source_top2",
                metadata.get("operator_mutability_allow_source_top2", True),
            ),
            True,
        )
        operator_allow_certified = _as_bool(
            section.get(
                "operator_mutability_allow_certified",
                metadata.get("operator_mutability_allow_certified", True),
            ),
            True,
        )
        operator_target_match = not operator_targets or target in operator_targets
        operator_family_match = not operator_families or family in operator_families
        operator_action_match = not operator_actions or action_name in operator_actions
        operator_applicable = bool(
            operator_guard_enabled
            and dataset in operator_datasets
            and operator_target_match
            and (operator_family_match or operator_action_match)
        )
        operator_has_source_anchor = bool(
            operator_allow_source_top2 and source_in_top2 > 0.0
        )
        operator_has_proof_certificate = bool(
            operator_allow_certified
            and (
                certified
                or proof_valid >= 1.0
                or proof_confidence >= 0.86
            )
        )
        operator_has_chain_certificate = bool(
            chain >= operator_min_chain
            and structural_support >= operator_min_structural
            and direct_text_support >= operator_min_direct
        )
        operator_has_certificate = bool(
            operator_has_source_anchor
            or operator_has_proof_certificate
            or operator_has_chain_certificate
        )
        operator_requires_proof = bool(
            family in operator_require_proof_families
            or action_name in operator_actions
        )
        operator_approved = True
        operator_reason = (
            "operator_mutability_contract_approved"
            if operator_applicable
            else "operator_mutability_contract_not_applicable"
        )
        if operator_applicable:
            if operator_requires_proof and not operator_has_certificate:
                operator_approved = False
                operator_reason = "operator_mutability_requires_certificate"
            elif (
                direct_text_support < operator_min_direct
                and not operator_has_certificate
            ):
                operator_approved = False
                operator_reason = "operator_mutability_direct_signal_below_floor"
            elif (
                structural_support < operator_min_structural
                and not operator_has_certificate
            ):
                operator_approved = False
                operator_reason = (
                    "operator_mutability_structural_signal_below_floor"
                )
            elif (
                best_original_rank > operator_max_rank
                and not operator_has_certificate
            ):
                operator_approved = False
                operator_reason = "operator_mutability_candidate_outside_aperture"

        approved = True
        reason = "top2_deployment_guard_approved"
        if level == "none":
            reason = "top2_deployment_guard_disabled_by_level"
        elif factoid_applicable and not factoid_approved:
            approved = False
            reason = factoid_reason
        elif operator_applicable and not operator_approved:
            approved = False
            reason = operator_reason
        elif generalized and generalized_mode == "shadow_only":
            approved = False
            reason = "top2_deployment_generalized_shadow_only"
        elif generalized and generalized_mode == "off":
            approved = False
            reason = "top2_deployment_generalized_disabled"
        elif p2_gain < min_gain:
            approved = False
            reason = "top2_deployment_gain_probability_below_floor"
        elif p2_loss > max_loss:
            approved = False
            reason = "top2_deployment_loss_probability_above_floor"
        elif predicted_margin < min_margin:
            approved = False
            reason = "top2_deployment_gain_loss_margin_below_floor"
        elif d2 < min_d2:
            approved = False
            reason = "top2_deployment_expected_r2_below_floor"
        elif support_score < min_support and not certified:
            approved = False
            reason = "top2_deployment_support_below_floor"
        elif direct_text_support < min_direct and not certified:
            approved = False
            reason = "top2_deployment_direct_signal_below_floor"
        elif (
            best_original_rank > max_rank
            and source_in_top2 <= 0.0
            and not certified
        ):
            approved = False
            reason = "top2_deployment_deep_candidate_without_anchor"
        elif support_risk_margin < -0.02 and not certified:
            approved = False
            reason = "top2_deployment_support_risk_margin_below_floor"

        base.update({
            "top2_deployment_guard_approved": bool(approved),
            "top2_deployment_guard_reason": reason,
            "top2_deployment_guard_level": level,
            "top2_deployment_generalized": bool(generalized),
            "top2_deployment_generalized_mode": generalized_mode,
            "top2_deployment_guard_family": family,
            "top2_deployment_guard_support_score": round(float(support_score), 8),
            "top2_deployment_guard_risk_score": round(float(risk_score), 8),
            "top2_deployment_guard_margin": round(
                float(support_risk_margin), 8
            ),
            "factoid_top2_aperture_guard": bool(factoid_guard_enabled),
            "factoid_top2_aperture_applicable": bool(factoid_applicable),
            "factoid_top2_aperture_approved": bool(factoid_approved),
            "factoid_top2_aperture_reason": factoid_reason,
            "factoid_top2_aperture_dataset": dataset,
            "factoid_top2_aperture_family": family,
            "factoid_top2_aperture_requires_proof": bool(
                factoid_requires_proof
            ),
            "factoid_top2_min_direct_support_score": round(
                float(factoid_min_direct), 8
            ),
            "factoid_top2_max_original_rank": round(float(factoid_max_rank), 8),
            "operator_mutability_contract": bool(operator_guard_enabled),
            "operator_mutability_contract_applicable": bool(operator_applicable),
            "operator_mutability_contract_approved": bool(operator_approved),
            "operator_mutability_contract_reason": operator_reason,
            "operator_mutability_contract_dataset": dataset,
            "operator_mutability_contract_family": family,
            "operator_mutability_contract_action": action_name,
            "operator_mutability_requires_proof": bool(operator_requires_proof),
            "operator_mutability_has_certificate": bool(operator_has_certificate),
            "operator_mutability_has_source_anchor": bool(operator_has_source_anchor),
            "operator_mutability_has_proof_certificate": bool(
                operator_has_proof_certificate
            ),
            "operator_mutability_has_chain_certificate": bool(
                operator_has_chain_certificate
            ),
            "operator_mutability_min_direct_support_score": round(
                float(operator_min_direct), 8
            ),
            "operator_mutability_min_structural_support_score": round(
                float(operator_min_structural), 8
            ),
            "operator_mutability_min_chain_completeness": round(
                float(operator_min_chain), 8
            ),
            "operator_mutability_max_original_rank": round(
                float(operator_max_rank), 8
            ),
            "top2_deployment_direct_support_score": round(
                float(direct_text_support), 8
            ),
            "top2_deployment_structural_support_score": round(
                float(structural_support), 8
            ),
            "top2_deployment_best_original_rank": int(best_original_rank),
            "top2_deployment_certified": bool(certified),
            "top2_deployment_source_in_top2": bool(source_in_top2),
        })
        return base

    @staticmethod
    def _class_probability(
        classifier: Any,
        probabilities: np.ndarray,
        label: int,
    ) -> np.ndarray:
        positions = {
            int(value): idx for idx, value in enumerate(classifier.classes_)
        }
        if label not in positions:
            return np.zeros(probabilities.shape[0], dtype=float)
        return probabilities[:, positions[label]]

    def decide_group(
        self,
        profiles: Sequence[Mapping[str, Any]],
        incumbent_index: int,
        hard_admissible: Optional[Sequence[bool]] = None,
        min_expected_delta: Optional[float] = None,
        max_loss_probability: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        rows = [dict(profile or {}) for profile in profiles]
        if not rows:
            return []
        if incumbent_index < 0 or incumbent_index >= len(rows):
            raise ValueError("incumbent_index is outside the action group")
        admissible = list(hard_admissible) if hard_admissible is not None else [
            bool(row.get("hard_admissible", True)) for row in rows
        ]
        if len(admissible) != len(rows):
            raise ValueError("hard_admissible must align with profiles")

        if self.state_conditioned_set_ranker:
            return self._decide_state_conditioned_set_group(
                rows,
                incumbent_index=incumbent_index,
                hard_admissible=admissible,
                min_expected_delta=min_expected_delta,
                max_loss_probability=max_loss_probability,
            )

        if self.query_grouped_slate_optimizer:
            return self._decide_query_grouped_slate(
                rows,
                incumbent_index=incumbent_index,
                hard_admissible=admissible,
                min_expected_delta=min_expected_delta,
                max_loss_probability=max_loss_probability,
            )

        if self.hierarchical_family:
            return self._decide_hierarchical_family_group(
                rows,
                incumbent_index=incumbent_index,
                hard_admissible=admissible,
                min_expected_delta=min_expected_delta,
                max_loss_probability=max_loss_probability,
            )

        dataset = str(rows[incumbent_index].get("dataset", "") or "").strip().lower()
        dataset_models = (self.bundle.get("dataset_models", {}) or {}).get(dataset, {})
        features = build_incumbent_pair_feature_rows(rows, incumbent_index)
        decisions: List[Dict[str, Any]] = []
        incumbent_protected = is_protected_incumbent(rows[incumbent_index])
        for idx, row in enumerate(rows):
            action = str(row.get("action", "") or "").strip().lower()
            decisions.append({
                "approved": idx == incumbent_index,
                "reason": (
                    "incumbent_preserved" if idx == incumbent_index
                    else "residual_candidate_not_selected"
                ),
                "model_ready": self.ready,
                "dataset_supported": bool(dataset_models),
                "incumbent_relative": True,
                "expected_delta": 0.0,
                "loss_probability": 0.0,
                "features": features[idx],
                "action": action,
            })

        if not self.ready or not dataset_models:
            reason = "model_not_ready" if not self.ready else "unsupported_domain_abstain"
            for idx, decision in enumerate(decisions):
                if idx != incumbent_index:
                    decision["reason"] = reason
            return decisions

        eligible_by_target: Dict[str, List[int]] = {}
        for idx, row in enumerate(rows):
            if idx == incumbent_index:
                continue
            action = str(row.get("action", "") or "").strip().lower()
            target = str(row.get("target", "") or "").strip().lower()
            if action == "no_op":
                decisions[idx]["reason"] = "raw_no_op_cannot_replace_incumbent"
                continue
            if not admissible[idx]:
                decisions[idx]["reason"] = "hard_admissibility_rejected"
                continue
            if incumbent_protected:
                decisions[idx]["reason"] = "protected_incumbent"
                continue
            if target == "top5" and row.get("changes_top2", False):
                decisions[idx]["reason"] = "top5_candidate_changes_top2"
                continue
            if target not in dataset_models:
                decisions[idx]["reason"] = "target_model_unavailable"
                continue
            eligible_by_target.setdefault(target, []).append(idx)

        approved_candidates: List[int] = []
        for target, indices in eligible_by_target.items():
            section = dataset_models[target]
            matrix = np.asarray([
                [features[idx][name] for name in V193_FEATURE_NAMES]
                for idx in indices
            ], dtype=float)
            classifier = section["classifier"]
            probabilities = classifier.predict_proba(matrix)
            gain_probability = self._class_probability(classifier, probabilities, 2)
            loss_probability = self._class_probability(classifier, probabilities, 0)
            mean_gain = float(section.get("mean_gain", 0.0) or 0.0)
            mean_loss = float(section.get("mean_loss_magnitude", 0.0) or 0.0)
            expected = gain_probability * mean_gain - loss_probability * mean_loss
            threshold = float(section.get("min_expected_delta", 0.0) or 0.0)
            risk_limit = float(section.get("max_loss_probability", 0.20) or 0.20)
            if min_expected_delta is not None and float(min_expected_delta) >= 0.0:
                threshold = float(min_expected_delta)
            if (
                max_loss_probability is not None
                and float(max_loss_probability) >= 0.0
            ):
                risk_limit = float(max_loss_probability)
            for local_idx, idx in enumerate(indices):
                decision = decisions[idx]
                decision.update({
                    "target": target,
                    "expected_delta": round(float(expected[local_idx]), 8),
                    "gain_probability": round(float(gain_probability[local_idx]), 8),
                    "loss_probability": round(float(loss_probability[local_idx]), 8),
                    "min_expected_delta": round(threshold, 8),
                    "max_loss_probability": round(risk_limit, 8),
                })
                if float(expected[local_idx]) < threshold:
                    decision["reason"] = "relative_delta_below_threshold"
                elif float(loss_probability[local_idx]) > risk_limit:
                    decision["reason"] = "relative_loss_risk_above_threshold"
                else:
                    decision["reason"] = "residual_candidate_eligible"
                    approved_candidates.append(idx)

        if not approved_candidates:
            return decisions
        winner = max(
            approved_candidates,
            key=lambda idx: (
                float(decisions[idx].get("expected_delta", -1.0) or -1.0),
                -float(decisions[idx].get("loss_probability", 1.0) or 1.0),
                float(rows[idx].get("semantic_score", 0.0) or 0.0),
            ),
        )
        decisions[winner]["approved"] = True
        decisions[winner]["reason"] = "incumbent_relative_winner"
        decisions[winner]["winner"] = True
        return decisions

    def _decide_query_grouped_slate(
        self,
        rows: Sequence[Mapping[str, Any]],
        incumbent_index: int,
        hard_admissible: Sequence[bool],
        min_expected_delta: Optional[float],
        max_loss_probability: Optional[float],
    ) -> List[Dict[str, Any]]:
        """Select at most one replacement from the full query action slate."""
        profiles = [dict(row or {}) for row in rows]
        dataset = str(
            profiles[incumbent_index].get("dataset", "") or ""
        ).strip().lower()
        section = (
            (self.bundle.get("dataset_models", {}) or {}).get(dataset, {})
            or {}
        )
        pair_features = build_incumbent_pair_feature_rows(
            profiles, incumbent_index
        )
        incumbent_protected = is_protected_incumbent(
            profiles[incumbent_index]
        )
        decisions: List[Dict[str, Any]] = []
        for idx, profile in enumerate(profiles):
            decisions.append({
                "approved": idx == incumbent_index,
                "reason": (
                    "incumbent_preserved" if idx == incumbent_index
                    else "query_grouped_slate_candidate_not_selected"
                ),
                "model_ready": self.ready,
                "dataset_supported": bool(section),
                "incumbent_relative": True,
                "query_grouped_slate_optimizer": True,
                "family": incumbent_action_family(profile),
                "expected_delta": 0.0,
                "winner_probability": 0.0,
                "opportunity_probability": 0.0,
                "opportunity_gate_passed": False,
                "features": pair_features[idx],
            })

        if not self.ready or not section:
            reason = (
                "model_not_ready" if not self.ready
                else "unsupported_domain_abstain"
            )
            for idx, decision in enumerate(decisions):
                if idx != incumbent_index:
                    decision["reason"] = reason
            return decisions

        eligible: List[int] = []
        for idx, profile in enumerate(profiles):
            if idx == incumbent_index:
                continue
            action = str(profile.get("action", "") or "").strip().lower()
            target = str(profile.get("target", "") or "").strip().lower()
            if action == "no_op":
                decisions[idx]["reason"] = "raw_no_op_cannot_replace_incumbent"
                continue
            if not hard_admissible[idx]:
                decisions[idx]["reason"] = "hard_admissibility_rejected"
                continue
            if incumbent_protected:
                decisions[idx]["reason"] = "protected_incumbent"
                continue
            if target not in {"top2", "top5"}:
                decisions[idx]["reason"] = "unsupported_target"
                continue
            if target == "top5" and profile.get("changes_top2", False):
                decisions[idx]["reason"] = "top5_candidate_changes_top2"
                continue
            eligible.append(idx)

        if not eligible:
            return decisions

        opportunity_probability = 1.0
        opportunity_classifier = section.get("opportunity_classifier")
        if opportunity_classifier is not None:
            opportunity_features = build_v194_opportunity_features(
                pair_features, eligible
            )
            opportunity_matrix = np.asarray([[
                opportunity_features[name]
                for name in V194_OPPORTUNITY_FEATURE_NAMES
            ]], dtype=float)
            opportunity_probabilities = (
                opportunity_classifier.predict_proba(opportunity_matrix)
            )
            opportunity_probability = float(self._class_probability(
                opportunity_classifier,
                opportunity_probabilities,
                1,
            )[0])
        opportunity_threshold = float(
            section.get("opportunity_threshold", 0.0) or 0.0
        )
        opportunity_gate_passed = (
            opportunity_probability >= opportunity_threshold
        )
        for idx in eligible:
            decisions[idx].update({
                "opportunity_probability": round(opportunity_probability, 8),
                "opportunity_threshold": round(opportunity_threshold, 8),
                "opportunity_gate_passed": bool(opportunity_gate_passed),
            })
            if not opportunity_gate_passed:
                decisions[idx]["reason"] = "query_opportunity_gate_rejected"
        if not opportunity_gate_passed:
            return decisions

        matrix = np.asarray([[
            pair_features[idx][name] for name in V193_FEATURE_NAMES
        ] for idx in eligible], dtype=float)
        winner_classifier = section.get("winner_classifier")
        if winner_classifier is None:
            for idx in eligible:
                decisions[idx]["reason"] = "winner_classifier_missing"
            return decisions
        winner_probabilities = winner_classifier.predict_proba(matrix)
        winner_probability = self._class_probability(
            winner_classifier, winner_probabilities, 1
        )
        threshold = float(
            section.get("winner_probability_threshold", 0.0) or 0.0
        )
        if min_expected_delta is not None and float(min_expected_delta) >= 0.0:
            threshold = max(threshold, min(float(min_expected_delta), 1.0))
        risk_limit = float(
            section.get("max_loss_probability_proxy", 1.0) or 1.0
        )
        if (
            max_loss_probability is not None
            and float(max_loss_probability) >= 0.0
        ):
            risk_limit = min(risk_limit, float(max_loss_probability))

        approved_candidates: List[int] = []
        for local_idx, idx in enumerate(eligible):
            probability = float(winner_probability[local_idx])


            loss_proxy = 1.0 - probability
            decision = decisions[idx]
            decision.update({
                "winner_probability": round(probability, 8),
                "expected_delta": round(probability, 8),
                "winner_probability_threshold": round(threshold, 8),
                "loss_probability": round(loss_proxy, 8),
                "max_loss_probability_proxy": round(risk_limit, 8),
                "target": str(profiles[idx].get("target", "") or "").lower(),
            })
            if probability < threshold:
                decision["reason"] = "winner_probability_below_threshold"
            elif loss_proxy > risk_limit:
                decision["reason"] = "winner_risk_proxy_above_threshold"
            else:
                decision["reason"] = "query_grouped_slate_eligible"
                approved_candidates.append(idx)

        if not approved_candidates:
            return decisions
        winner = max(
            approved_candidates,
            key=lambda idx: (
                float(decisions[idx].get("winner_probability", 0.0) or 0.0),
                1.0 if str(profiles[idx].get("target", "") or "").lower()
                == "top2" else 0.0,
                float(profiles[idx].get("semantic_score", 0.0) or 0.0),
            ),
        )
        decisions[winner]["approved"] = True
        decisions[winner]["winner"] = True
        decisions[winner]["reason"] = "query_grouped_slate_winner"
        return decisions

    def _decide_state_conditioned_set_group(
        self,
        rows: Sequence[Mapping[str, Any]],
        incumbent_index: int,
        hard_admissible: Sequence[bool],
        min_expected_delta: Optional[float],
        max_loss_probability: Optional[float],
    ) -> List[Dict[str, Any]]:
        """Rank complete counterfactual replacement sets against one incumbent."""
        profiles = [dict(row or {}) for row in rows]
        dataset = str(
            profiles[incumbent_index].get("dataset", "") or ""
        ).strip().lower()
        section = (
            (self.bundle.get("dataset_models", {}) or {}).get(dataset, {})
            or {}
        )
        features = build_incumbent_pair_feature_rows(
            profiles, incumbent_index
        )
        incumbent_protected = is_protected_incumbent(
            profiles[incumbent_index]
        )
        decisions: List[Dict[str, Any]] = []
        for idx, profile in enumerate(profiles):
            decisions.append({
                "approved": idx == incumbent_index,
                "reason": (
                    "incumbent_preserved" if idx == incumbent_index
                    else "state_conditioned_candidate_not_selected"
                ),
                "model_ready": self.ready,
                "dataset_supported": bool(section),
                "incumbent_relative": True,
                "state_conditioned_set_ranker": True,
                "family": incumbent_action_family(profile),
                "expected_delta": 0.0,
                "expected_delta_r2": 0.0,
                "expected_delta_r5": 0.0,
                "loss_probability": 0.0,
                "r2_loss_probability": 0.0,
                "r5_loss_probability": 0.0,
                "features": features[idx],
            })

        if not self.ready or not section:
            reason = (
                "model_not_ready" if not self.ready
                else "unsupported_domain_abstain"
            )
            for idx, decision in enumerate(decisions):
                if idx != incumbent_index:
                    decision["reason"] = reason
            return decisions

        eligible: List[int] = []
        for idx, profile in enumerate(profiles):
            if idx == incumbent_index:
                continue
            action = str(profile.get("action", "") or "").strip().lower()
            target = str(profile.get("target", "") or "").strip().lower()
            if action == "no_op":
                decisions[idx]["reason"] = "raw_no_op_cannot_replace_incumbent"
                continue
            if not hard_admissible[idx]:
                decisions[idx]["reason"] = "hard_admissibility_rejected"
                continue
            if incumbent_protected:
                decisions[idx]["reason"] = "protected_incumbent"
                continue
            if target not in {"top2", "top5"}:
                decisions[idx]["reason"] = "unsupported_target"
                continue
            if target == "top5" and profile.get("changes_top2", False):
                decisions[idx]["reason"] = "top5_candidate_changes_top2"
                continue
            eligible.append(idx)

        if not eligible:
            return decisions

        matrix = np.asarray([[
            features[idx][name] for name in V193_FEATURE_NAMES
        ] for idx in eligible], dtype=float)
        r2_classifier = section["r2_classifier"]
        r5_classifier = section["r5_classifier"]
        r2_probabilities = r2_classifier.predict_proba(matrix)
        r5_probabilities = r5_classifier.predict_proba(matrix)
        p2_gain = self._class_probability(
            r2_classifier, r2_probabilities, 2
        )
        p2_loss = self._class_probability(
            r2_classifier, r2_probabilities, 0
        )
        p5_gain = self._class_probability(
            r5_classifier, r5_probabilities, 2
        )
        p5_loss = self._class_probability(
            r5_classifier, r5_probabilities, 0
        )
        expected_r2 = np.asarray(
            section["r2_regressor"].predict(matrix), dtype=float
        )
        expected_r5 = np.asarray(
            section["r5_regressor"].predict(matrix), dtype=float
        )
        r2_weight = float(section.get("score_r2_weight", 2.0) or 2.0)
        r5_weight = float(section.get("score_r5_weight", 1.0) or 1.0)
        risk_penalty = float(section.get("risk_penalty_weight", 0.0) or 0.0)
        scores = (
            r2_weight * expected_r2
            + r5_weight * expected_r5
            - risk_penalty * (p2_loss + p5_loss)
        )
        threshold = float(section.get("score_threshold", 0.0) or 0.0)
        r2_risk_limit = float(
            section.get("max_r2_loss_probability", 0.20) or 0.20
        )
        r5_risk_limit = float(
            section.get("max_r5_loss_probability", 0.25) or 0.25
        )
        if min_expected_delta is not None and float(min_expected_delta) >= 0.0:
            threshold = float(min_expected_delta)
        if (
            max_loss_probability is not None
            and float(max_loss_probability) >= 0.0
        ):
            r2_risk_limit = float(max_loss_probability)
            r5_risk_limit = float(max_loss_probability)

        approved_candidates: List[int] = []
        for local_idx, idx in enumerate(eligible):
            target = str(profiles[idx].get("target", "") or "").lower()
            d2 = float(expected_r2[local_idx])
            d5 = float(expected_r5[local_idx])
            score = float(scores[local_idx])
            loss2 = float(p2_loss[local_idx])
            loss5 = float(p5_loss[local_idx])
            min_d2 = float(section.get(
                f"min_expected_r2_{target}", 0.0
            ) or 0.0)
            min_d5 = float(section.get(
                f"min_expected_r5_{target}", 0.0
            ) or 0.0)
            decision = decisions[idx]
            decision.update({
                "target": target,
                "expected_delta": round(score, 8),
                "expected_delta_r2": round(d2, 8),
                "expected_delta_r5": round(d5, 8),
                "r2_gain_probability": round(float(p2_gain[local_idx]), 8),
                "r2_loss_probability": round(loss2, 8),
                "r5_gain_probability": round(float(p5_gain[local_idx]), 8),
                "r5_loss_probability": round(loss5, 8),
                "loss_probability": round(max(loss2, loss5), 8),
                "score_threshold": round(threshold, 8),
                "max_r2_loss_probability": round(r2_risk_limit, 8),
                "max_r5_loss_probability": round(r5_risk_limit, 8),
                "min_expected_r2": round(min_d2, 8),
                "min_expected_r5": round(min_d5, 8),
            })
            if score < threshold:
                decision["reason"] = "set_score_below_noop_threshold"
            elif d2 < min_d2:
                decision["reason"] = "expected_r2_below_pareto_floor"
            elif d5 < min_d5:
                decision["reason"] = "expected_r5_below_pareto_floor"
            elif loss2 > r2_risk_limit:
                decision["reason"] = "r2_loss_risk_above_threshold"
            elif loss5 > r5_risk_limit:
                decision["reason"] = "r5_loss_risk_above_threshold"
            else:
                decision["reason"] = "state_conditioned_set_eligible"
                approved_candidates.append(idx)

        if not approved_candidates:
            return decisions
        winner = max(
            approved_candidates,
            key=lambda idx: (
                float(decisions[idx].get("expected_delta", -1.0)),
                -float(decisions[idx].get("r2_loss_probability", 1.0)),
                -float(decisions[idx].get("r5_loss_probability", 1.0)),
                float(profiles[idx].get("semantic_score", 0.0) or 0.0),
            ),
        )
        decisions[winner]["approved"] = True
        decisions[winner]["winner"] = True
        decisions[winner]["reason"] = "state_conditioned_set_winner"
        return decisions

    def _decide_hierarchical_family_group(
        self,
        rows: Sequence[Mapping[str, Any]],
        incumbent_index: int,
        hard_admissible: Sequence[bool],
        min_expected_delta: Optional[float],
        max_loss_probability: Optional[float],
    ) -> List[Dict[str, Any]]:
        """Run opportunity gating, family experts, and Pareto admission."""
        profiles = [dict(row or {}) for row in rows]
        metadata = self.bundle.get("metadata", {}) or {}
        r2_first_mode = bool(
            self.r2_first_expert_slate_optimizer
            or self.runtime_shadow_slate_verifier_optimizer
            or metadata.get("r2_first_expert_slate", False)
        )
        shadow_verifier_enabled = bool(
            self.runtime_shadow_slate_verifier_optimizer
            or metadata.get("runtime_shadow_slate_verifier", False)
        )
        dataset = str(
            profiles[incumbent_index].get("dataset", "") or ""
        ).strip().lower()
        dataset_models = (self.bundle.get("dataset_models", {}) or {}).get(
            dataset, {}
        )
        top2_local_section = (
            (self.bundle.get("top2_local_sections", {}) or {}).get(dataset, {})
            or {}
        )
        dataset_supported = bool(dataset_models or top2_local_section)
        pair_features = build_incumbent_pair_feature_rows(
            profiles, incumbent_index
        )
        incumbent_protected = is_protected_incumbent(
            profiles[incumbent_index]
        )
        decisions: List[Dict[str, Any]] = []
        for idx, profile in enumerate(profiles):
            decisions.append({
                "approved": idx == incumbent_index,
                "reason": (
                    "incumbent_preserved" if idx == incumbent_index
                    else "hierarchical_candidate_not_selected"
                ),
                "model_ready": self.ready,
                "dataset_supported": dataset_supported,
                "incumbent_relative": True,
                "hierarchical_family_optimizer": True,
                "r2_first_expert_slate_optimizer": bool(r2_first_mode),
                "runtime_shadow_slate_verifier": bool(
                    shadow_verifier_enabled
                ),
                "shadow_slate_applicable": False,
                "shadow_slate_approved": True,
                "shadow_slate_reason": "shadow_not_applicable",
                "shadow_slate_support_score": 0.0,
                "shadow_slate_risk_score": 0.0,
                "shadow_slate_margin": 0.0,
                "top2_deployment_guard": False,
                "top2_deployment_guard_applicable": False,
                "top2_deployment_guard_approved": True,
                "top2_deployment_guard_reason": (
                    "top2_deployment_guard_not_applicable"
                ),
                "top2_deployment_guard_support_score": 0.0,
                "top2_deployment_guard_risk_score": 0.0,
                "top2_deployment_guard_margin": 0.0,
                "top2_local_discriminator": False,
                "top2_local_approved": False,
                "family": incumbent_action_family(profile),
                "expected_delta": 0.0,
                "expected_delta_r2": 0.0,
                "expected_delta_r5": 0.0,
                "loss_probability": 0.0,
                "r2_loss_probability": 0.0,
                "r5_loss_probability": 0.0,
                "opportunity_probability": 0.0,
                "opportunity_gate_passed": False,
                "pareto_approved": False,
                "features": pair_features[idx],
            })

        if not self.ready or not dataset_supported:
            reason = (
                "model_not_ready" if not self.ready
                else "unsupported_domain_abstain"
            )
            for idx, decision in enumerate(decisions):
                if idx != incumbent_index:
                    decision["reason"] = reason
            return decisions

        local_indices: List[int] = []
        if top2_local_section:
            local_families = {
                str(value).strip().lower()
                for value in top2_local_section.get("families", []) or []
                if str(value).strip()
            }
            for idx, profile in enumerate(profiles):
                if idx == incumbent_index:
                    continue
                action = str(profile.get("action", "") or "").strip().lower()
                target = str(profile.get("target", "") or "").strip().lower()
                family = incumbent_action_family(profile)
                if action == "no_op" or target != "top2":
                    continue
                if local_families and family not in local_families:
                    continue
                if not hard_admissible[idx]:
                    decisions[idx]["reason"] = "hard_admissibility_rejected"
                    continue
                if incumbent_protected:
                    decisions[idx]["reason"] = "protected_incumbent"
                    continue
                local_indices.append(idx)
            if local_indices:
                opportunity_probability = 1.0
                opportunity_threshold = float(
                    top2_local_section.get("opportunity_threshold", 0.0) or 0.0
                )
                if top2_local_section.get("opportunity_classifier") is not None:
                    opportunity_features = build_v194_opportunity_features(
                        pair_features, local_indices
                    )
                    opportunity_matrix = np.asarray([[
                        opportunity_features[name]
                        for name in V194_OPPORTUNITY_FEATURE_NAMES
                    ]], dtype=float)
                    opportunity_classifier = top2_local_section[
                        "opportunity_classifier"
                    ]
                    opportunity_probabilities = (
                        opportunity_classifier.predict_proba(opportunity_matrix)
                    )
                    opportunity_probability = float(self._class_probability(
                        opportunity_classifier,
                        opportunity_probabilities,
                        1,
                    )[0])
                opportunity_passed = (
                    opportunity_probability >= opportunity_threshold
                )
                for idx in local_indices:
                    decisions[idx].update({
                        "top2_local_discriminator": True,
                        "top2_local_opportunity_probability": round(
                            opportunity_probability, 8
                        ),
                        "top2_local_opportunity_threshold": round(
                            opportunity_threshold, 8
                        ),
                    })
                local_approved: List[int] = []
                if not opportunity_passed:
                    for idx in local_indices:
                        decisions[idx]["reason"] = (
                            "top2_local_opportunity_rejected"
                        )
                else:
                    matrix = np.asarray([[
                        pair_features[idx][name] for name in V193_FEATURE_NAMES
                    ] for idx in local_indices], dtype=float)
                    r2_classifier = top2_local_section["r2_classifier"]
                    r5_classifier = top2_local_section["r5_classifier"]
                    r2_probabilities = r2_classifier.predict_proba(matrix)
                    r5_probabilities = r5_classifier.predict_proba(matrix)
                    r2_gain_probability = self._class_probability(
                        r2_classifier, r2_probabilities, 2
                    )
                    r2_loss_probability = self._class_probability(
                        r2_classifier, r2_probabilities, 0
                    )
                    r5_gain_probability = self._class_probability(
                        r5_classifier, r5_probabilities, 2
                    )
                    r5_loss_probability = self._class_probability(
                        r5_classifier, r5_probabilities, 0
                    )
                    expected_r2 = (
                        r2_gain_probability
                        * float(top2_local_section.get("mean_r2_gain", 0.0))
                        - r2_loss_probability
                        * float(top2_local_section.get(
                            "mean_r2_loss_magnitude", 0.0
                        ))
                    )
                    expected_r5 = (
                        r5_gain_probability
                        * float(top2_local_section.get("mean_r5_gain", 0.0))
                        - r5_loss_probability
                        * float(top2_local_section.get(
                            "mean_r5_loss_magnitude", 0.0
                        ))
                    )
                    min_primary_delta = float(
                        top2_local_section.get("min_primary_delta", 0.0) or 0.0
                    )
                    min_secondary_delta = float(
                        top2_local_section.get("min_secondary_delta", 0.0) or 0.0
                    )
                    risk_limit = float(
                        top2_local_section.get("max_loss_probability", 0.20)
                        or 0.20
                    )
                    runtime_verified_top2 = bool(
                        top2_local_section.get(
                            "runtime_verified_top2_admission", False
                        )
                    )
                    runtime_min_r2_gain_probability = float(
                        top2_local_section.get(
                            "runtime_min_r2_gain_probability", 0.0
                        ) or 0.0
                    )
                    runtime_min_r2_gain_loss_margin = float(
                        top2_local_section.get(
                            "runtime_min_r2_gain_loss_margin", 0.0
                        ) or 0.0
                    )
                    runtime_max_loss_probability = float(
                        top2_local_section.get(
                            "runtime_max_loss_probability", risk_limit
                        ) or risk_limit
                    )
                    if runtime_verified_top2:
                        risk_limit = min(risk_limit, runtime_max_loss_probability)
                        min_secondary_delta = max(min_secondary_delta, 0.0)
                    if (
                        min_expected_delta is not None
                        and float(min_expected_delta) >= 0.0
                    ):
                        min_primary_delta = float(min_expected_delta)
                    if (
                        max_loss_probability is not None
                        and float(max_loss_probability) >= 0.0
                    ):
                        risk_limit = float(max_loss_probability)
                    for local_idx, idx in enumerate(local_indices):
                        d2 = float(expected_r2[local_idx])
                        d5 = float(expected_r5[local_idx])
                        p2_gain = float(r2_gain_probability[local_idx])
                        p2_loss = float(r2_loss_probability[local_idx])
                        p5_gain = float(r5_gain_probability[local_idx])
                        p5_loss = float(r5_loss_probability[local_idx])
                        r2_gain_loss_margin = p2_gain - p2_loss
                        if runtime_verified_top2:
                            score = (
                                4.0 * d2
                                + 0.25 * d5
                                + 0.5 * r2_gain_loss_margin
                                - 1.5 * p2_loss
                                - 0.5 * p5_loss
                            )
                        else:
                            score = 2.5 * d2 + 0.5 * d5
                        decision = decisions[idx]
                        decision.update({
                            "target": "top2",
                            "family": incumbent_action_family(profiles[idx]),
                            "runtime_verified_top2_admission": (
                                runtime_verified_top2
                            ),
                            "generalized_top2_local": bool(
                                top2_local_section.get(
                                    "generalized_top2_local", False
                                )
                            ),
                            "expected_delta": round(score, 8),
                            "expected_delta_r2": round(d2, 8),
                            "expected_delta_r5": round(d5, 8),
                            "loss_probability": round(max(p2_loss, p5_loss), 8),
                            "r2_gain_probability": round(p2_gain, 8),
                            "r2_loss_probability": round(p2_loss, 8),
                            "r5_gain_probability": round(p5_gain, 8),
                            "r5_loss_probability": round(p5_loss, 8),
                            "r2_gain_loss_margin": round(
                                r2_gain_loss_margin, 8
                            ),
                            "runtime_min_r2_gain_probability": round(
                                runtime_min_r2_gain_probability, 8
                            ),
                            "runtime_min_r2_gain_loss_margin": round(
                                runtime_min_r2_gain_loss_margin, 8
                            ),
                            "min_primary_delta": round(min_primary_delta, 8),
                            "min_secondary_delta": round(min_secondary_delta, 8),
                            "max_loss_probability": round(risk_limit, 8),
                        })
                        if d2 < min_primary_delta:
                            decision["reason"] = (
                                "top2_local_primary_delta_below_threshold"
                            )
                        elif d5 < min_secondary_delta:
                            decision["reason"] = (
                                "top2_local_secondary_delta_below_floor"
                            )
                        elif p2_loss > risk_limit:
                            decision["reason"] = (
                                "top2_local_r2_loss_risk_above_threshold"
                            )
                        elif p5_loss > risk_limit:
                            decision["reason"] = (
                                "top2_local_r5_loss_risk_above_threshold"
                            )
                        elif (
                            runtime_verified_top2
                            and p2_gain < runtime_min_r2_gain_probability
                        ):
                            decision["reason"] = (
                                "top2_local_r2_gain_probability_below_verified_floor"
                            )
                        elif (
                            runtime_verified_top2
                            and r2_gain_loss_margin
                            < runtime_min_r2_gain_loss_margin
                        ):
                            decision["reason"] = (
                                "top2_local_gain_loss_margin_below_verified_floor"
                            )
                        else:
                            deployment_decision = (
                                self._runtime_top2_deployment_guard_decision(
                                    profiles[idx],
                                    profiles[incumbent_index],
                                    decision,
                                    pair_features[idx],
                                    top2_local_section,
                                )
                            )
                            decision.update(deployment_decision)
                            if not deployment_decision.get(
                                "top2_deployment_guard_approved", True
                            ):
                                decision["reason"] = str(
                                    deployment_decision.get(
                                        "top2_deployment_guard_reason",
                                        "top2_deployment_guard_rejected",
                                    )
                                )
                                continue
                            shadow_decision = self._runtime_shadow_slate_decision(
                                profiles[idx],
                                profiles[incumbent_index],
                                decision,
                                pair_features[idx],
                            )
                            decision.update(shadow_decision)
                            if not shadow_decision.get(
                                "shadow_slate_approved", True
                            ):
                                decision["reason"] = str(
                                    shadow_decision.get(
                                        "shadow_slate_reason",
                                        "shadow_top2_rejected",
                                    )
                                )
                            else:
                                decision["reason"] = (
                                    "top2_local_pairwise_eligible"
                                )
                                decision["pareto_approved"] = True
                                decision["top2_local_approved"] = True
                                local_approved.append(idx)
                if local_approved:
                    winner = max(
                        local_approved,
                        key=lambda idx: (
                            float(decisions[idx].get("expected_delta", -1.0)),
                            float(decisions[idx].get(
                                "top2_local_opportunity_probability", 0.0
                            )),
                            -float(decisions[idx].get(
                                "loss_probability", 1.0
                            )),
                            float(profiles[idx].get("semantic_score", 0.0) or 0.0),
                        ),
                    )
                    decisions[winner]["approved"] = True
                    decisions[winner]["winner"] = True
                    decisions[winner]["reason"] = "top2_local_pairwise_winner"
                    return decisions

        grouped: Dict[tuple[str, str], List[int]] = {}
        for idx, profile in enumerate(profiles):
            if idx == incumbent_index:
                continue
            if decisions[idx].get("top2_local_discriminator", False):



                continue
            action = str(profile.get("action", "") or "").strip().lower()
            target = str(profile.get("target", "") or "").strip().lower()
            family = incumbent_action_family(profile)
            if action == "no_op":
                decisions[idx]["reason"] = "raw_no_op_cannot_replace_incumbent"
                continue
            if not hard_admissible[idx]:
                decisions[idx]["reason"] = "hard_admissibility_rejected"
                continue
            if incumbent_protected:
                decisions[idx]["reason"] = "protected_incumbent"
                continue
            if target == "top5" and profile.get("changes_top2", False):
                decisions[idx]["reason"] = "top5_candidate_changes_top2"
                continue
            section = (
                (dataset_models.get(target, {}) or {}).get(family, {}) or {}
            )
            if not section:
                decisions[idx]["reason"] = "family_model_unavailable"
                continue
            grouped.setdefault((target, family), []).append(idx)

        approved_candidates: List[int] = []
        for (target, family), indices in grouped.items():
            section = dataset_models[target][family]
            opportunity_features = build_v194_opportunity_features(
                pair_features, indices
            )
            opportunity_matrix = np.asarray([[
                opportunity_features[name]
                for name in V194_OPPORTUNITY_FEATURE_NAMES
            ]], dtype=float)
            opportunity_classifier = section["opportunity_classifier"]
            opportunity_probabilities = opportunity_classifier.predict_proba(
                opportunity_matrix
            )
            opportunity_probability = float(self._class_probability(
                opportunity_classifier, opportunity_probabilities, 1
            )[0])
            opportunity_threshold = float(
                section.get("opportunity_threshold", 0.50) or 0.50
            )
            opportunity_passed = opportunity_probability >= opportunity_threshold
            for idx in indices:
                decisions[idx].update({
                    "target": target,
                    "family": family,
                    "opportunity_probability": round(
                        opportunity_probability, 8
                    ),
                    "opportunity_threshold": round(opportunity_threshold, 8),
                    "opportunity_gate_passed": bool(opportunity_passed),
                })
            if not opportunity_passed:
                for idx in indices:
                    decisions[idx]["reason"] = "opportunity_gate_rejected"
                continue

            matrix = np.asarray([[
                pair_features[idx][name] for name in V193_FEATURE_NAMES
            ] for idx in indices], dtype=float)
            r2_classifier = section["r2_classifier"]
            r5_classifier = section["r5_classifier"]
            r2_probabilities = r2_classifier.predict_proba(matrix)
            r5_probabilities = r5_classifier.predict_proba(matrix)
            r2_gain_probability = self._class_probability(
                r2_classifier, r2_probabilities, 2
            )
            r2_loss_probability = self._class_probability(
                r2_classifier, r2_probabilities, 0
            )
            r5_gain_probability = self._class_probability(
                r5_classifier, r5_probabilities, 2
            )
            r5_loss_probability = self._class_probability(
                r5_classifier, r5_probabilities, 0
            )
            expected_r2 = (
                r2_gain_probability * float(section.get("mean_r2_gain", 0.0))
                - r2_loss_probability
                * float(section.get("mean_r2_loss_magnitude", 0.0))
            )
            expected_r5 = (
                r5_gain_probability * float(section.get("mean_r5_gain", 0.0))
                - r5_loss_probability
                * float(section.get("mean_r5_loss_magnitude", 0.0))
            )
            min_primary_delta = float(
                section.get("min_primary_delta", 0.0) or 0.0
            )
            min_secondary_delta = float(
                section.get("min_secondary_delta", 0.0) or 0.0
            )
            risk_limit = float(
                section.get("max_loss_probability", 0.20) or 0.20
            )
            runtime_verified_top2 = bool(
                target == "top2"
                and section.get("runtime_verified_top2_admission", False)
            )
            runtime_min_r2_gain_probability = float(
                section.get("runtime_min_r2_gain_probability", 0.0) or 0.0
            )
            runtime_min_r2_gain_loss_margin = float(
                section.get("runtime_min_r2_gain_loss_margin", 0.0) or 0.0
            )
            runtime_max_loss_probability = float(
                section.get("runtime_max_loss_probability", risk_limit)
                or risk_limit
            )
            if runtime_verified_top2:
                risk_limit = min(risk_limit, runtime_max_loss_probability)
                min_secondary_delta = max(min_secondary_delta, 0.0)
            if min_expected_delta is not None and float(min_expected_delta) >= 0.0:
                min_primary_delta = float(min_expected_delta)
            if (
                max_loss_probability is not None
                and float(max_loss_probability) >= 0.0
            ):
                risk_limit = float(max_loss_probability)
            for local_idx, idx in enumerate(indices):
                d2 = float(expected_r2[local_idx])
                d5 = float(expected_r5[local_idx])
                p2_gain = float(r2_gain_probability[local_idx])
                p2_loss = float(r2_loss_probability[local_idx])
                p5_gain = float(r5_gain_probability[local_idx])
                p5_loss = float(r5_loss_probability[local_idx])
                r2_gain_loss_margin = p2_gain - p2_loss
                primary = d2 if target == "top2" else d5
                secondary = d5 if target == "top2" else d2
                primary_loss = p2_loss if target == "top2" else p5_loss
                secondary_loss = p5_loss if target == "top2" else p2_loss
                if runtime_verified_top2:
                    pareto_score = (
                        4.0 * d2
                        + 0.25 * d5
                        + 0.5 * r2_gain_loss_margin
                        - 1.5 * p2_loss
                        - 0.5 * p5_loss
                    )
                else:
                    pareto_score = (
                        2.0 * d2 + d5 if target == "top2"
                        else d5 + 2.0 * min(d2, 0.0)
                    )
                decision = decisions[idx]
                decision.update({
                    "expected_delta": round(pareto_score, 8),
                    "expected_delta_r2": round(d2, 8),
                    "expected_delta_r5": round(d5, 8),
                    "loss_probability": round(max(p2_loss, p5_loss), 8),
                    "runtime_verified_top2_admission": runtime_verified_top2,
                    "r2_gain_probability": round(p2_gain, 8),
                    "r2_loss_probability": round(p2_loss, 8),
                    "r5_gain_probability": round(p5_gain, 8),
                    "r5_loss_probability": round(p5_loss, 8),
                    "r2_gain_loss_margin": round(r2_gain_loss_margin, 8),
                    "runtime_min_r2_gain_probability": round(
                        runtime_min_r2_gain_probability, 8
                    ),
                    "runtime_min_r2_gain_loss_margin": round(
                        runtime_min_r2_gain_loss_margin, 8
                    ),
                    "min_primary_delta": round(min_primary_delta, 8),
                    "min_secondary_delta": round(min_secondary_delta, 8),
                    "max_loss_probability": round(risk_limit, 8),
                })
                if primary < min_primary_delta:
                    decision["reason"] = "primary_delta_below_threshold"
                elif secondary < min_secondary_delta:
                    decision["reason"] = "secondary_delta_below_pareto_floor"
                elif primary_loss > risk_limit:
                    decision["reason"] = "primary_loss_risk_above_threshold"
                elif secondary_loss > risk_limit:
                    decision["reason"] = "secondary_loss_risk_above_threshold"
                elif (
                    runtime_verified_top2
                    and p2_gain < runtime_min_r2_gain_probability
                ):
                    decision["reason"] = (
                        "verified_top2_r2_gain_probability_below_floor"
                    )
                elif (
                    runtime_verified_top2
                    and r2_gain_loss_margin < runtime_min_r2_gain_loss_margin
                ):
                    decision["reason"] = (
                        "verified_top2_gain_loss_margin_below_floor"
                    )
                else:
                    deployment_decision = (
                        self._runtime_top2_deployment_guard_decision(
                            profiles[idx],
                            profiles[incumbent_index],
                            decision,
                            pair_features[idx],
                            section,
                        )
                    )
                    decision.update(deployment_decision)
                    if not deployment_decision.get(
                        "top2_deployment_guard_approved", True
                    ):
                        decision["reason"] = str(
                            deployment_decision.get(
                                "top2_deployment_guard_reason",
                                "top2_deployment_guard_rejected",
                            )
                        )
                        continue
                    shadow_decision = self._runtime_shadow_slate_decision(
                        profiles[idx],
                        profiles[incumbent_index],
                        decision,
                        pair_features[idx],
                    )
                    decision.update(shadow_decision)
                    if not shadow_decision.get("shadow_slate_approved", True):
                        decision["reason"] = str(
                            shadow_decision.get(
                                "shadow_slate_reason",
                                "shadow_top2_rejected",
                            )
                        )
                    else:
                        decision["reason"] = "pareto_candidate_eligible"
                        decision["pareto_approved"] = True
                        approved_candidates.append(idx)

        if not approved_candidates:
            return decisions
        if r2_first_mode:
            top2_candidates = [
                idx for idx in approved_candidates
                if str(profiles[idx].get("target", "") or "").lower()
                == "top2"
            ]
            if top2_candidates:
                approved_candidates = top2_candidates
        winner = max(
            approved_candidates,
            key=lambda idx: (
                float(decisions[idx].get("expected_delta", -1.0)),
                float(decisions[idx].get("opportunity_probability", 0.0)),
                -float(decisions[idx].get("loss_probability", 1.0)),
                float(profiles[idx].get("semantic_score", 0.0) or 0.0),
            ),
        )
        decisions[winner]["approved"] = True
        decisions[winner]["winner"] = True
        decisions[winner]["reason"] = "hierarchical_pareto_winner"
        return decisions
