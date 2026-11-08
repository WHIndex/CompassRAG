"""Learned query-level optimizer for competing Top-2/Top-5 evidence sets."""

from __future__ import annotations

import json
import importlib.util
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

try:
    from .query_plan_delta_admission import (
        V186_FEATURE_NAMES,
        factorized_action_features_v186,
    )
except ImportError:
    _FEATURE_PATH = Path(__file__).with_name("query_plan_delta_admission.py")
    _FEATURE_SPEC = importlib.util.spec_from_file_location(
        "query_plan_set_level_delta_features", _FEATURE_PATH
    )
    if _FEATURE_SPEC is None or _FEATURE_SPEC.loader is None:
        raise RuntimeError(f"Cannot load feature contract from {_FEATURE_PATH}")
    _FEATURE_MODULE = importlib.util.module_from_spec(_FEATURE_SPEC)
    _FEATURE_SPEC.loader.exec_module(_FEATURE_MODULE)
    V186_FEATURE_NAMES = list(_FEATURE_MODULE.V186_FEATURE_NAMES)
    factorized_action_features_v186 = _FEATURE_MODULE.factorized_action_features_v186


SET_LEVEL_CONTEXT_FEATURE_NAMES = [
    "set_context_candidate_count_log",
    "set_context_target_count_log",
    "set_context_top2_fraction",
    "set_context_evidence_set_fraction",
    "set_context_target_evidence_set_fraction",
    "set_context_utility_max",
    "set_context_utility_mean",
    "set_context_utility_relative_to_max",
    "set_context_utility_rank_inv",
    "set_context_target_utility_rank_inv",
    "set_context_semantic_max",
    "set_context_semantic_mean",
    "set_context_semantic_relative_to_max",
    "set_context_margin_max",
    "set_context_margin_relative_to_max",
]
SET_LEVEL_FEATURE_NAMES = [*V186_FEATURE_NAMES, *SET_LEVEL_CONTEXT_FEATURE_NAMES]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: Any, low: float = 0.0, high: float = 1.0) -> float:
    return min(max(_safe_float(value), low), high)


def build_set_level_feature_rows(
    profiles: Sequence[Mapping[str, Any]],
    precomputed_features: Optional[Sequence[Mapping[str, Any]]] = None,
) -> List[Dict[str, float]]:
    """Build action features with query-local opportunity-set statistics."""
    rows = [dict(profile or {}) for profile in profiles]
    if precomputed_features is not None and len(precomputed_features) != len(rows):
        raise ValueError("precomputed_features must align with profiles")
    base_rows = [
        {
            name: _safe_float(features.get(name), 0.0)
            for name in V186_FEATURE_NAMES
        }
        for features in precomputed_features
    ] if precomputed_features is not None else [
        factorized_action_features_v186(profile) for profile in rows
    ]

    candidate_indices = [
        idx
        for idx, profile in enumerate(rows)
        if str(profile.get("action", "") or "").strip().lower() != "no_op"
    ]
    candidate_count = max(len(candidate_indices), 1)
    utilities = [base_rows[idx]["utility"] for idx in candidate_indices]
    semantics = [base_rows[idx]["semantic_score"] for idx in candidate_indices]
    margins = [
        base_rows[idx]["selected_displaced_score_margin"]
        for idx in candidate_indices
    ]
    utility_max = max(utilities, default=0.0)
    semantic_max = max(semantics, default=0.0)
    margin_max = max(margins, default=0.0)
    utility_mean = sum(utilities) / candidate_count
    semantic_mean = sum(semantics) / candidate_count
    top2_fraction = sum(
        base_rows[idx]["target_top2"] > 0.5 for idx in candidate_indices
    ) / candidate_count
    set_fraction = sum(
        base_rows[idx]["action_evidence_set"] > 0.5 for idx in candidate_indices
    ) / candidate_count

    output: List[Dict[str, float]] = []
    for idx, (profile, base) in enumerate(zip(rows, base_rows)):
        action = str(profile.get("action", "") or "").strip().lower()
        target = str(profile.get("target", "") or "").strip().lower()
        target_indices = [
            candidate_idx
            for candidate_idx in candidate_indices
            if str(rows[candidate_idx].get("target", "") or "").strip().lower()
            == target
        ]
        target_count = max(len(target_indices), 1)
        target_set_fraction = sum(
            base_rows[candidate_idx]["action_evidence_set"] > 0.5
            for candidate_idx in target_indices
        ) / target_count
        utility = base["utility"]
        semantic = base["semantic_score"]
        margin = base["selected_displaced_score_margin"]
        utility_rank = 1 + sum(
            base_rows[candidate_idx]["utility"] > utility
            for candidate_idx in candidate_indices
        )
        target_utility_rank = 1 + sum(
            base_rows[candidate_idx]["utility"] > utility
            for candidate_idx in target_indices
        )
        context = {
            "set_context_candidate_count_log": min(
                math.log1p(len(candidate_indices)) / math.log(65.0), 1.0
            ),
            "set_context_target_count_log": min(
                math.log1p(len(target_indices)) / math.log(65.0), 1.0
            ),
            "set_context_top2_fraction": top2_fraction,
            "set_context_evidence_set_fraction": set_fraction,
            "set_context_target_evidence_set_fraction": target_set_fraction,
            "set_context_utility_max": utility_max,
            "set_context_utility_mean": utility_mean,
            "set_context_utility_relative_to_max": _clamp(
                0.5 + 0.5 * (utility - utility_max)
            ),
            "set_context_utility_rank_inv": 1.0 / max(utility_rank, 1),
            "set_context_target_utility_rank_inv": (
                1.0 / max(target_utility_rank, 1)
            ),
            "set_context_semantic_max": semantic_max,
            "set_context_semantic_mean": semantic_mean,
            "set_context_semantic_relative_to_max": _clamp(
                0.5 + 0.5 * (semantic - semantic_max)
            ),
            "set_context_margin_max": margin_max,
            "set_context_margin_relative_to_max": _clamp(
                0.5 + 0.5 * (margin - margin_max)
            ),
        }
        if action == "no_op":
            context = {name: 0.0 for name in SET_LEVEL_CONTEXT_FEATURE_NAMES}
        output.append({
            name: float({**base, **context}.get(name, 0.0))
            for name in SET_LEVEL_FEATURE_NAMES
        })
    return output


class SetLevelEvidenceOptimizer:
    """Select one evidence-set mutation by predicted metric gain and regret risk."""

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
            if self.bundle.get("type") != "query_level_set_utility_optimizer":
                raise ValueError("unsupported set-level model type")
            if list(self.bundle.get("feature_names", []) or []) != SET_LEVEL_FEATURE_NAMES:
                raise ValueError("set-level feature contract mismatch")
            self.ready = True
        except Exception as exc:
            self.error = f"model_load_failed:{type(exc).__name__}:{exc}"
            self.bundle = {}

    @property
    def supported_datasets(self) -> List[str]:
        return sorted((self.bundle.get("dataset_models", {}) or {}).keys())

    def decide_group(
        self,
        profiles: Sequence[Mapping[str, Any]],
        hard_admissible: Optional[Sequence[bool]] = None,
        min_expected_utility: Optional[float] = None,
        max_loss_probability: Optional[float] = None,
        fail_open: bool = False,
    ) -> List[Dict[str, Any]]:
        rows = [dict(profile or {}) for profile in profiles]
        admissible = list(hard_admissible) if hard_admissible is not None else [
            bool(row.get("hard_admissible", True)) for row in rows
        ]
        if len(admissible) != len(rows):
            raise ValueError("hard_admissible must align with profiles")
        if not rows:
            return []
        dataset = str(rows[0].get("dataset", "") or "").strip().lower()
        dataset_model = (self.bundle.get("dataset_models", {}) or {}).get(dataset)
        if not self.ready or not dataset_model:
            reason = "model_not_ready" if not self.ready else "unsupported_domain_abstain"
            return [
                {
                    "approved": str(row.get("action", "") or "").lower() == "no_op",
                    "reason": "explicit_no_op" if str(
                        row.get("action", "") or ""
                    ).lower() == "no_op" else reason,
                    "model_ready": self.ready,
                    "dataset_supported": bool(dataset_model),
                    "expected_delta": 0.0,
                    "expected_utility": 0.0,
                    "loss_probability": 0.0,
                    "set_level_optimizer": True,
                }
                for row in rows
            ]

        feature_rows = build_set_level_feature_rows(rows)
        matrix = np.asarray([
            [feature_row[name] for name in SET_LEVEL_FEATURE_NAMES]
            for feature_row in feature_rows
        ], dtype=float)
        classifier = dataset_model["classifier"]
        probabilities = classifier.predict_proba(matrix)
        class_index = {
            int(label): idx for idx, label in enumerate(classifier.classes_)
        }
        loss_probability = probabilities[:, class_index.get(0, 0)]
        gain_probability = probabilities[:, class_index.get(2, probabilities.shape[1] - 1)]
        mean_gain = float(dataset_model.get("mean_gain", 0.0) or 0.0)
        mean_loss = float(dataset_model.get("mean_loss_magnitude", 0.0) or 0.0)
        expected = gain_probability * mean_gain - loss_probability * mean_loss
        threshold = float(dataset_model.get("min_expected_utility", 0.0) or 0.0)
        risk_limit = float(dataset_model.get("max_loss_probability", 0.25) or 0.25)
        if min_expected_utility is not None and float(min_expected_utility) >= 0.0:
            threshold = float(min_expected_utility)
        if max_loss_probability is not None and float(max_loss_probability) >= 0.0:
            risk_limit = float(max_loss_probability)

        decisions: List[Dict[str, Any]] = []
        eligible_indices: List[int] = []
        for idx, row in enumerate(rows):
            action = str(row.get("action", "") or "").strip().lower()
            decision = {
                "approved": action == "no_op",
                "reason": "explicit_no_op" if action == "no_op" else "set_utility_loser",
                "model_ready": True,
                "dataset_supported": True,
                "expected_delta": round(float(expected[idx]), 8),
                "expected_utility": round(float(expected[idx]), 8),
                "gain_probability": round(float(gain_probability[idx]), 8),
                "loss_probability": round(float(loss_probability[idx]), 8),
                "min_expected_utility": round(threshold, 8),
                "max_loss_probability": round(risk_limit, 8),
                "set_level_optimizer": True,
            }
            if action != "no_op" and admissible[idx]:
                eligible_indices.append(idx)
            elif action != "no_op":
                decision["reason"] = "hard_admissibility_rejected"
            decisions.append(decision)

        if not eligible_indices:
            return decisions
        winner = max(
            eligible_indices,
            key=lambda idx: (
                float(expected[idx]),
                -float(loss_probability[idx]),
                float(gain_probability[idx]),
            ),
        )
        winner_decision = decisions[winner]
        if float(expected[winner]) < threshold:
            winner_decision["reason"] = "expected_utility_below_threshold"
        elif float(loss_probability[winner]) > risk_limit:
            winner_decision["reason"] = "loss_risk_above_threshold"
        else:
            winner_decision["approved"] = True
            winner_decision["reason"] = "query_level_set_winner"
        winner_decision["winner"] = True
        if fail_open and not self.ready:
            winner_decision["approved"] = True
            winner_decision["reason"] = "model_failure_fail_open"
        return decisions
