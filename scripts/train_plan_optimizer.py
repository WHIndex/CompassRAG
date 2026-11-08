#!/usr/bin/env python3
"""Train a complete-evidence CBO with entity-disjoint out-of-fold calibration."""

from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[1]
for value in (str(ROOT), str(ROOT / "scripts")):
    if value not in sys.path:
        sys.path.insert(0, value)

import plan_optimizer_core as base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--workload-dir", required=True)
    parser.add_argument("--output-model", required=True)
    parser.add_argument("--action-table-dir", default="")
    parser.add_argument(
        "--checkpoint-dir",
        default="",
        help="Directory for atomic per-fold OOF prediction checkpoints.",
    )
    parser.add_argument(
        "--resume-checkpoints",
        action="store_true",
        help="Reuse compatible completed-fold checkpoints.",
    )
    parser.add_argument(
        "--trusted-stage1-checkpoint-fingerprint",
        default="",
        help=(
            "Explicit fingerprint of Stage-1 OOF predictions produced before "
            "a calibration-only code change. Exact held-out action keys are "
            "still validated before reuse; Stage-2 checkpoints are never "
            "reused through this override."
        ),
    )
    parser.add_argument("--datasets", default="2wikimultihopqa,hotpotqa,musique")
    parser.add_argument("--allowed-plans", default=",".join(base.ALTERNATIVE_PLANS))
    parser.add_argument("--trees", type=int, default=480)
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--min-leaf", type=int, default=10)
    parser.add_argument("--max-features", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=277)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument(
        "--selection-mode",
        choices=(
            "pareto_lcb",
            "risk_budgeted_lexicographic",
            "safe_gain_listwise",
            "absolute_quality_regret",
            "query_relative_slate_regret",
        ),
        default="safe_gain_listwise",
    )
    parser.add_argument("--pareto-safe-gain-label", action="store_true", default=True)
    parser.add_argument("--balance-outcome-targets", action="store_true", default=True)
    parser.add_argument("--query-balanced-sample-weights", action="store_true")
    parser.add_argument("--absolute-quality-regret-targets", action="store_true")
    parser.add_argument("--query-relative-slate-regret-targets", action="store_true")
    parser.add_argument("--max-calibration-loss-rate", type=float, default=0.08)
    parser.add_argument("--max-holdout-loss-rate", type=float, default=0.10)
    parser.add_argument("--max-top5-calibration-loss-rate", type=float, default=0.10)
    parser.add_argument("--max-top5-holdout-loss-rate", type=float, default=0.12)
    parser.add_argument("--max-admission-loss-probability", type=float, default=-1.0)
    parser.add_argument(
        "--max-top5-admission-loss-probability", type=float, default=-1.0
    )
    parser.add_argument("--max-characteristic-loss-rate", type=float, default=0.15)
    parser.add_argument("--min-delta-r2-mean", type=float, default=0.0)
    parser.add_argument("--min-delta-r5-mean", type=float, default=0.0)
    parser.add_argument("--min-calibration-selected", type=int, default=6)
    parser.add_argument("--min-holdout-selected", type=int, default=2)
    parser.add_argument("--min-characteristic-selected", type=int, default=0)
    parser.add_argument("--min-domain-selected", type=int, default=0)
    parser.add_argument("--min-domain-utility-per-action", type=float, default=0.0)
    parser.add_argument("--robust-calibration-objective", action="store_true")
    parser.add_argument(
        "--outcome-model-scope",
        choices=(
            "global", "objective_class", "objective_family", "plan", "structural"
        ),
        default="global",
        help=(
            "Estimator sharing scope. objective_class trains separate, "
            "plan-comparable Top-2 ordering and Top-5 completion experts; "
            "objective_family additionally isolates each physical-plan family."
        ),
    )
    parser.add_argument(
        "--topology-partitioned-plan-routing",
        action="store_true",
        help=(
            "Train and deploy each physical-plan family only on the compiled "
            "query topology for which its safety contract was certified."
        ),
    )
    parser.add_argument(
        "--topology-plan-routing-profile",
        choices=(
            base.TOPOLOGY_ROUTING_STRICT_V280,
            base.TOPOLOGY_ROUTING_RECALL_SAFE_V281,
            base.TOPOLOGY_ROUTING_BOUNDED_V282,
            base.TOPOLOGY_ROUTING_BOUNDED_V283,
            base.TOPOLOGY_ROUTING_WAREHOUSE_V284,
            base.TOPOLOGY_ROUTING_COMBINED_V285,
            base.TOPOLOGY_ROUTING_HIERARCHICAL_V286,
            base.TOPOLOGY_ROUTING_OBJECTIVE_FAMILY_V287,
            base.TOPOLOGY_ROUTING_MULTI_ACCESS_V289,
            base.TOPOLOGY_ROUTING_STRUCTURAL_RECOVERY_V293,
        ),
        default=base.TOPOLOGY_ROUTING_STRICT_V280,
        help=(
            "Dataset-agnostic mapping from compiled query topology to "
            "eligible physical-plan families."
        ),
    )
    parser.add_argument(
        "--incumbent-contract",
        choices=("current_pipeline_ranking", "immutable_semantic_fact_ranking"),
        default="current_pipeline_ranking",
        help="Ranking endpoint against which every counterfactual is labeled.",
    )
    parser.add_argument(
        "--hierarchical-objective-execution",
        action="store_true",
        help=(
            "Select a Top-2 ordering first, freeze that prefix, and calibrate "
            "Top-5 completion as residual gain against the selected prefix."
        ),
    )
    parser.add_argument(
        "--query-grouped-hard-negative-top5",
        action="store_true",
        help=(
            "Keep every positive residual Top-5 action and only the most "
            "plausible negatives from each query and physical-plan family."
        ),
    )
    parser.add_argument(
        "--stage2-slate-dedup",
        action="store_true",
        help=(
            "Deduplicate identical ordered Top-5 slates across physical-plan "
            "families before Stage-2 training, calibration, and deployment."
        ),
    )
    parser.add_argument(
        "--stage2-proof-connected-primary",
        action="store_true",
        help=(
            "Use proof-connected Top-5 slates as the primary Stage-2 lane and "
            "admit an open-path slate only by an OOF-calibrated regret margin."
        ),
    )
    parser.add_argument(
        "--top5-hard-negatives-per-query",
        type=int,
        default=12,
        help="Maximum retained non-positive Stage-2 actions per query/family.",
    )
    parser.add_argument(
        "--grouped_oof-score-alignment",
        action="store_true",
        help=(
            "Align each final objective-family expert back to its paired OOF "
            "score scale before applying OOF-certified admission thresholds."
        ),
    )
    parser.add_argument(
        "--structure-conditioned-outcome-calibration",
        action="store_true",
        help=(
            "Calibrate OOF predictions to observed historical outcomes within "
            "each rank objective, plan family, and query-contract regime."
        ),
    )

    parser.set_defaults(
        rematerialize_plan_catalog=True,
        enable_gain_probability_gate=True,
    )
    args = parser.parse_args()
    args.global_outcome_model = args.outcome_model_scope == "global"
    args.objective_class_outcome_model = (
        args.outcome_model_scope == "objective_class"
    )
    args.objective_family_outcome_model = (
        args.outcome_model_scope == "objective_family"
    )
    args.structural_expert_model = args.outcome_model_scope == "structural"
    return args


CHECKPOINT_SCHEMA = "grouped_oof_complete_evidence_plan_fold_v1"


def row_checkpoint_key(row: Mapping[str, Any]) -> str:
    return json.dumps(
        [
            str(row.get("query_key", "")),
            str(row.get("plan_name", "")),
            str(row.get("plan_instance", "")),
        ],
        ensure_ascii=True,
        separators=(",", ":"),
    )


def checkpoint_fingerprint(
    args: argparse.Namespace,
    contract_path: Path,
    datasets: Sequence[str],
    selected_plans: set[str],
    fold_count: int,
) -> str:
    trace_root = Path(args.trace_dir)
    trace_files = []
    for dataset in datasets:
        paths = [trace_root / f"{dataset}.jsonl"]
        if not paths[0].is_file():
            paths = sorted(trace_root.rglob(f"{dataset}.jsonl"))
        for path in paths:
            if path.is_file():
                stat = path.stat()
                trace_files.append({
                    "path": str(path.resolve()),
                    "size": int(stat.st_size),
                    "mtime_ns": int(stat.st_mtime_ns),
                })
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "contract_sha256": hashlib.sha256(contract_path.read_bytes()).hexdigest(),
        "trainer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "base_trainer_sha256": hashlib.sha256(
            Path(base.__file__).read_bytes()
        ).hexdigest(),
        "trace_files": trace_files,
        "datasets": list(datasets),
        "allowed_plans": sorted(selected_plans),
        "fold_count": int(fold_count),
        "trees": int(args.trees),
        "max_depth": int(args.max_depth),
        "min_leaf": int(args.min_leaf),
        "max_features": float(args.max_features),
        "seed": int(args.seed),
        "alpha": float(args.alpha),
        "selection_mode": str(args.selection_mode),
        "absolute_quality_regret_targets": bool(
            args.absolute_quality_regret_targets
        ),
        "query_relative_slate_regret_targets": bool(
            args.query_relative_slate_regret_targets
        ),
        "outcome_model_scope": str(args.outcome_model_scope),
        "pareto_safe_gain_label": bool(args.pareto_safe_gain_label),
        "balance_outcome_targets": bool(args.balance_outcome_targets),
        "query_balanced_sample_weights": bool(
            args.query_balanced_sample_weights
        ),
        "topology_partitioned_plan_routing": bool(
            args.topology_partitioned_plan_routing
        ),
        "grouped_oof_score_alignment": bool(args.grouped_oof_score_alignment),
        "structure_conditioned_outcome_calibration": bool(
            args.structure_conditioned_outcome_calibration
        ),
        "topology_plan_routing_profile": str(args.topology_plan_routing_profile),
        "incumbent_contract": str(args.incumbent_contract),
        "hierarchical_objective_execution": bool(
            args.hierarchical_objective_execution
        ),
        "query_grouped_hard_negative_top5": bool(
            args.query_grouped_hard_negative_top5
        ),
        "top5_hard_negatives_per_query": int(
            args.top5_hard_negatives_per_query
        ),
        "stage2_slate_dedup": bool(args.stage2_slate_dedup),
        "stage2_proof_connected_primary": bool(
            args.stage2_proof_connected_primary
        ),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_fold_checkpoint(
    path: Path,
    *,
    fingerprint: str,
    fold: int,
    predicted: Sequence[Mapping[str, Any]],
    report: Mapping[str, Any],
) -> None:
    payload = {
        "schema_version": CHECKPOINT_SCHEMA,
        "fingerprint": fingerprint,
        "fold": int(fold),
        "report": dict(report),
        "predictions": [
            {
                "action_key": row_checkpoint_key(row),
                "prediction": dict(row.get("prediction", {}) or {}),
            }
            for row in predicted
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
    temporary.replace(path)


def load_fold_checkpoint(
    path: Path,
    *,
    fingerprint: str,
    fold: int,
    heldout_rows: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema_version") != CHECKPOINT_SCHEMA:
        raise ValueError(f"incompatible cross-validation checkpoint schema: {path}")
    if payload.get("fingerprint") != fingerprint:
        raise ValueError(f"cross-validation checkpoint fingerprint mismatch: {path}")
    if int(payload.get("fold", -1)) != fold:
        raise ValueError(f"cross-validation checkpoint fold mismatch: {path}")
    predictions = {
        str(item.get("action_key", "")): item.get("prediction", {}) or {}
        for item in payload.get("predictions", []) or []
        if isinstance(item, Mapping)
    }
    expected_keys = [row_checkpoint_key(row) for row in heldout_rows]
    if len(predictions) != len(expected_keys) or set(predictions) != set(expected_keys):
        raise ValueError(f"cross-validation checkpoint action set mismatch: {path}")
    restored = []
    for row, action_key in zip(heldout_rows, expected_keys):
        value = dict(row)
        value["prediction"] = dict(predictions[action_key])
        value["grouped_oof_fold"] = fold
        restored.append(value)
    return restored, dict(payload.get("report", {}) or {})


def update_lcbs(rows: Sequence[Dict[str, Any]], residuals: Mapping[str, float]) -> None:
    for row in rows:
        prediction = row.get("prediction", {}) or {}
        r2_residual = float(residuals.get("delta_r2", 0.0))
        r5_residual = float(residuals.get("delta_r5", 0.0))
        prediction["delta_r2_lcb"] = float(prediction.get("delta_r2", 0.0)) + r2_residual
        prediction["delta_r5_lcb"] = float(prediction.get("delta_r5", 0.0)) + r5_residual
        prediction["utility_lcb"] = (
            2.0 * prediction["delta_r2_lcb"] + prediction["delta_r5_lcb"]
        )
        prediction["utility_mean"] = (
            2.0 * float(prediction.get("delta_r2", 0.0))
            + float(prediction.get("delta_r5", 0.0))
        )
        prediction["safety_margin"] = float(
            prediction.get("gain_probability", 0.0)
        ) - float(prediction.get("loss_probability", 0.0))
        if "quality_r2" in prediction and "quality_r5" in prediction:
            prediction["quality_r2_lcb"] = float(
                prediction.get("quality_r2", 0.0)
            ) + float(residuals.get("quality_r2", 0.0))
            prediction["quality_r5_lcb"] = float(
                prediction.get("quality_r5", 0.0)
            ) + float(residuals.get("quality_r5", 0.0))
        if "regret_r2" in prediction and "regret_r5" in prediction:
            prediction["regret_r2_ucb"] = max(
                0.0,
                float(prediction.get("regret_r2", 0.0))
                + float(residuals.get("regret_r2", 0.0)),
            )
            prediction["regret_r5_ucb"] = max(
                0.0,
                float(prediction.get("regret_r5", 0.0))
                + float(residuals.get("regret_r5", 0.0)),
            )
        row["prediction"] = prediction


def residual_target_names(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    return [
        target
        for target in (
            "delta_r2", "delta_r5", "quality_r2", "quality_r5",
            "regret_r2", "regret_r5",
        )
        if rows and all(
            target in row and target in (row.get("prediction", {}) or {})
            for row in rows
        )
    ]


def calibrated_residual(
    rows: Sequence[Mapping[str, Any]], target: str, alpha: float
) -> float:
    residuals = [
        float(row[target]) - float((row.get("prediction", {}) or {})[target])
        for row in rows
    ]
    quantile = 1.0 - alpha if target.startswith("regret_") else alpha
    return float(np.quantile(residuals, quantile))


def _objective_query_relative_regret_score(row: Mapping[str, Any]) -> float:
    prediction = row.get("prediction", {}) or {}
    if base.plan_objective_class(row) == "top5_completion":
        return float(prediction.get("regret_r5_ucb", 1.0))
    return (
        2.0 * float(prediction.get("regret_r2_ucb", 1.0))
        + float(prediction.get("regret_r5_ucb", 1.0))
    )


def _objective_query_relative_key(row: Mapping[str, Any]) -> Tuple[float, ...]:
    if base.plan_objective_class(row) != "top5_completion":
        return base.query_relative_slate_key(row)
    prediction = row.get("prediction", {}) or {}
    features = row.get("features", {}) or {}
    proof_coherence = sum(float(features.get(name, 0.0) or 0.0) for name in (
        "plan_completion_path_connected",
        "plan_introduced_complete_proof_fraction",
        "plan_introduced_relation_coverage_mean",
        "residual_candidate_complete_provenance",
        "residual_candidate_terminal_relation_match",
    ))
    return (
        -float(prediction.get("regret_r5_ucb", 1.0)),
        proof_coherence,
        -float(features.get("plan_top5_displacement_count", 0.0) or 0.0),
        -float(features.get("estimated_cost_log", 0.0) or 0.0),
        float(str(row.get("plan_name", "")) == base.PLAN_SEMANTIC),
    )


def _is_query_relative_comparator(row: Mapping[str, Any]) -> bool:
    return str(row.get("plan_name", "")) in {
        base.PLAN_SEMANTIC,
        base.PLAN_SEMANTIC_TOP2_NOOP,
    }


def _is_stage2_connected_primary(row: Mapping[str, Any]) -> bool:
    return bool(
        str(row.get("plan_name", "")) == base.PLAN_SEMANTIC
        or float((row.get("features", {}) or {}).get(
            "plan_completion_path_connected", 0.0
        ) or 0.0) > 0.0
    )


def choose_precomputed(
    rows: Sequence[Dict[str, Any]],
    *,
    utility_threshold: float,
    max_loss_probability: float,
    selection_mode: str,
    min_delta_r2_mean: float,
    min_delta_r5_mean: float,
    min_gain_probability: float,
    allowed_characteristics: set[str] | None = None,
    proof_connected_primary: bool = False,
) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if allowed_characteristics is not None and (
            base.query_characteristic(row) not in allowed_characteristics
        ):
            continue
        prediction = row.get("prediction", {}) or {}
        objective_class = base.plan_objective_class(row)
        if selection_mode == "query_relative_slate_regret":
            eligible = bool(
                "regret_r2_ucb" in prediction and "regret_r5_ucb" in prediction
            )
        elif selection_mode == "absolute_quality_regret":
            metric = "r5" if objective_class == "top5_completion" else "r2"
            eligible = bool(
                float(prediction.get(f"quality_{metric}_lcb", -1.0))
                > utility_threshold
                and float(prediction.get(f"regret_{metric}_probability", 1.0))
                <= max_loss_probability
            )
        elif selection_mode == "safe_gain_listwise":
            eligible = bool(
                float(prediction.get("gain_probability", 0.0)) >= min_gain_probability
                and float(prediction.get("loss_probability", 1.0)) <= max_loss_probability
                and float(prediction.get("safety_margin", 0.0)) > utility_threshold
            )
        elif selection_mode == "risk_budgeted_lexicographic":
            eligible = bool(
                float(prediction.get("delta_r2", 0.0)) >= min_delta_r2_mean
                and float(prediction.get("delta_r5", 0.0)) >= min_delta_r5_mean
                and float(prediction.get("gain_probability", 0.0)) >= min_gain_probability
                and float(prediction.get("utility_mean", 0.0)) > utility_threshold
                and float(prediction.get("loss_probability", 1.0)) <= max_loss_probability
            )
        else:
            eligible = bool(
                float(prediction.get("delta_r2_lcb", -1.0)) >= 0.0
                and float(prediction.get("delta_r5_lcb", -1.0)) >= 0.0
                and float(prediction.get("gain_probability", 0.0)) >= min_gain_probability
                and float(prediction.get("utility_lcb", 0.0)) > utility_threshold
                and float(prediction.get("loss_probability", 1.0)) <= max_loss_probability
            )
        if eligible:
            grouped[row["query_key"]].append(row)
    if selection_mode == "query_relative_slate_regret":
        selected = []
        for candidates in grouped.values():
            objective_class = base.plan_objective_class(candidates[0])
            if proof_connected_primary and objective_class == "top5_completion":
                primary = [
                    row for row in candidates
                    if _is_stage2_connected_primary(row)
                ]
                rescue = [
                    row for row in candidates
                    if not _is_stage2_connected_primary(row)
                ]
                best = max(
                    primary or candidates,
                    key=_objective_query_relative_key,
                )
                if rescue:
                    rescue_best = max(rescue, key=_objective_query_relative_key)
                    rescue_advantage = (
                        _objective_query_relative_regret_score(best)
                        - _objective_query_relative_regret_score(rescue_best)
                    )
                    if rescue_advantage + 1e-12 >= utility_threshold:
                        best = rescue_best
                selected.append(best)
                continue
            best = max(candidates, key=_objective_query_relative_key)
            semantic = next((
                row for row in candidates
                if _is_query_relative_comparator(row)
            ), None)
            if semantic is not None and not _is_query_relative_comparator(best):
                advantage = (
                    _objective_query_relative_regret_score(semantic)
                    - _objective_query_relative_regret_score(best)
                )
                if advantage + 1e-12 < utility_threshold:
                    best = semantic
            selected.append(best)
        return selected
    if selection_mode == "absolute_quality_regret":
        objective_class = (
            base.plan_objective_class(next(iter(rows)))
            if rows else "top2_ordering"
        )
        metric = "r5" if objective_class == "top5_completion" else "r2"
        other_metric = "r2" if metric == "r5" else "r5"
        key = lambda row: (
            row["prediction"][f"quality_{metric}_lcb"],
            row["prediction"][f"optimal_{metric}_probability"],
            -row["prediction"][f"regret_{metric}_probability"],
            row["prediction"][f"quality_{other_metric}_lcb"],
        )
    elif selection_mode == "safe_gain_listwise":
        key = lambda row: (
            row["prediction"]["safety_margin"],
            row["prediction"]["gain_probability"],
            -row["prediction"]["loss_probability"],
            row["prediction"]["delta_r2"],
            row["prediction"]["delta_r5"],
        )
    elif selection_mode == "risk_budgeted_lexicographic":
        objective_class = (
            base.plan_objective_class(next(iter(rows))) if rows else "top2_ordering"
        )
        if objective_class == "top5_completion":
            key = lambda row: (
                row["prediction"]["delta_r5"],
                row["prediction"]["utility_mean"],
                row["prediction"]["gain_probability"],
                -row["prediction"]["loss_probability"],
                row["prediction"]["delta_r2"],
            )
        else:
            key = lambda row: (
                row["prediction"]["delta_r2"],
                row["prediction"]["delta_r5"],
                row["prediction"]["utility_mean"],
                row["prediction"]["gain_probability"],
                -row["prediction"]["loss_probability"],
            )
    else:
        objective_class = (
            base.plan_objective_class(next(iter(rows))) if rows else "top2_ordering"
        )
        if objective_class == "top5_completion":
            key = lambda row: (
                row["prediction"]["delta_r5_lcb"],
                row["prediction"]["utility_lcb"],
                row["prediction"]["delta_r2_lcb"],
            )
        else:
            key = lambda row: (
                row["prediction"]["utility_lcb"],
                row["prediction"]["delta_r2_lcb"],
                row["prediction"]["delta_r5_lcb"],
            )
    return [max(candidates, key=key) for candidates in grouped.values()]


def calibrate_oof_policy(
    rows: Sequence[Dict[str, Any]], args: argparse.Namespace
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    query_count = len({row["query_key"] for row in rows})
    proof_connected_primary = bool(
        getattr(args, "stage2_proof_connected_primary", False)
        and rows
        and all(
            base.plan_objective_class(row) == "top5_completion"
            for row in rows
        )
    )
    if args.selection_mode == "query_relative_slate_regret":
        by_query: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_query[str(row.get("query_key", ""))].append(row)
        utilities = []
        for candidates in by_query.values():
            objective_class = base.plan_objective_class(candidates[0])
            if proof_connected_primary and objective_class == "top5_completion":
                primary = [
                    row for row in candidates
                    if _is_stage2_connected_primary(row)
                ]
                rescue = [
                    row for row in candidates
                    if not _is_stage2_connected_primary(row)
                ]
                if not primary or not rescue:
                    continue
                primary_best = max(primary, key=_objective_query_relative_key)
                rescue_best = max(rescue, key=_objective_query_relative_key)
                utilities.append(
                    _objective_query_relative_regret_score(primary_best)
                    - _objective_query_relative_regret_score(rescue_best)
                )
            else:
                semantic = next((
                    row for row in candidates
                    if _is_query_relative_comparator(row)
                ), None)
                alternatives = [
                    row for row in candidates
                    if not _is_query_relative_comparator(row)
                ]
                if semantic is None or not alternatives:
                    continue
                best = max(alternatives, key=_objective_query_relative_key)
                utilities.append(
                    _objective_query_relative_regret_score(semantic)
                    - _objective_query_relative_regret_score(best)
                )
    elif args.selection_mode == "absolute_quality_regret":
        utilities = []
        for row in rows:
            objective_class = base.plan_objective_class(row)
            metric = "r5" if objective_class == "top5_completion" else "r2"
            utilities.append(float(row["prediction"][f"quality_{metric}_lcb"]))
    elif args.selection_mode == "safe_gain_listwise":
        utilities = [float(row["prediction"]["safety_margin"]) for row in rows]
    elif args.selection_mode == "risk_budgeted_lexicographic":
        utilities = [float(row["prediction"]["utility_mean"]) for row in rows]
    else:
        utilities = [float(row["prediction"]["utility_lcb"]) for row in rows]
    thresholds = sorted({
        0.0,
        max(0.0, max(utilities)) if utilities else 0.0,
        *(
            [max(0.0, max(utilities)) + 1e-9]
            if utilities and proof_connected_primary
            else []
        ),
        *(
            max(0.0, float(value))
            for value in np.quantile(utilities, np.linspace(0.0, 0.97, 30))
        ),
    }) if utilities else [0.0]
    gain_values = [float(row["prediction"].get("gain_probability", 0.0)) for row in rows]
    gain_thresholds = sorted({
        0.0,
        *(
            min(1.0, max(0.0, float(value)))
            for value in np.quantile(gain_values, np.linspace(0.25, 0.98, 16))
        ),
    }) if gain_values and args.selection_mode not in {
        "absolute_quality_regret", "query_relative_slate_regret"
    } else [0.0]
    loss_caps = (
        [1.0]
        if args.selection_mode == "query_relative_slate_regret"
        else base.admission_loss_probability_caps(args)
    )
    feasible = []
    for loss_cap in loss_caps:
        for gain_threshold in gain_thresholds:
            for threshold in thresholds:
                selected = choose_precomputed(
                    rows,
                    utility_threshold=threshold,
                    max_loss_probability=loss_cap,
                    selection_mode=args.selection_mode,
                    min_delta_r2_mean=args.min_delta_r2_mean,
                    min_delta_r5_mean=args.min_delta_r5_mean,
                    min_gain_probability=gain_threshold,
                    proof_connected_primary=proof_connected_primary,
                )
                summary = base.report(selected, query_count)
                fold_safe = True
                if args.selection_mode == "query_relative_slate_regret":
                    for fold in sorted({
                        int(row.get("grouped_oof_fold", -1)) for row in rows
                    }):
                        fold_rows = [
                            row for row in selected
                            if int(row.get("grouped_oof_fold", -1)) == fold
                        ]
                        fold_report = base.report(
                            fold_rows,
                            len({row.get("query_key") for row in fold_rows}),
                        )
                        fold_safe = fold_safe and bool(
                            fold_report["selected_loss_rate"]
                            <= args.max_holdout_loss_rate
                            and fold_report["delta_r2_sum"] >= -1e-12
                            and fold_report["delta_r5_sum"] >= -1e-12
                        )
                if (
                    summary["selected_count"] >= args.min_calibration_selected
                    and summary["selected_loss_rate"] <= args.max_calibration_loss_rate
                    and summary["delta_r2_sum"] >= -1e-12
                    and summary["delta_r5_sum"] >= -1e-12
                    and (summary["delta_r2_sum"] > 1e-12 or summary["delta_r5_sum"] > 1e-12)
                    and base.characteristic_safe(summary, args)
                    and base.dataset_safe(summary, args, args.max_calibration_loss_rate)
                    and fold_safe
                ):
                    utility = 2.0 * summary["delta_r2_sum"] + summary["delta_r5_sum"]
                    feasible.append((
                        utility,
                        -summary["selected_count"],
                        -summary["selected_loss_rate"],
                        gain_threshold,
                        threshold,
                        loss_cap,
                        summary,
                    ))
    if not feasible:
        return {
            "selection_mode": args.selection_mode,
            "calibration_feasible": False,
            "min_absolute_quality_lcb": float("inf"),
            "min_regret_advantage": float("inf"),
            "proof_connected_primary_admission": bool(
                proof_connected_primary
            ),
            "min_unconnected_regret_advantage": float("inf"),
            "min_utility_lcb": float("inf"),
            "min_utility_mean": float("inf"),
            "min_safety_margin": float("inf"),
            "max_loss_probability": 0.0,
            "min_delta_r2_mean": args.min_delta_r2_mean,
            "min_delta_r5_mean": args.min_delta_r5_mean,
            "min_gain_probability": float("inf"),
            "allowed_query_characteristics": [],
        }, base.report([], query_count)
    _, _, _, gain_threshold, threshold, loss_cap, summary = max(feasible)
    characteristics = sorted({base.query_characteristic(row) for row in rows})
    return {
        "selection_mode": args.selection_mode,
        "calibration_feasible": True,
        "min_absolute_quality_lcb": (
            float(threshold)
            if args.selection_mode == "absolute_quality_regret"
            else 0.0
        ),
        "min_regret_advantage": (
            0.0
            if args.selection_mode == "query_relative_slate_regret"
            and proof_connected_primary
            else float(threshold)
            if args.selection_mode == "query_relative_slate_regret"
            else 0.0
        ),
        "proof_connected_primary_admission": bool(proof_connected_primary),
        "min_unconnected_regret_advantage": (
            float(threshold) if proof_connected_primary else 0.0
        ),
        "min_utility_lcb": float(threshold) if args.selection_mode == "pareto_lcb" else 0.0,
        "min_utility_mean": float(threshold) if args.selection_mode == "risk_budgeted_lexicographic" else 0.0,
        "min_safety_margin": float(threshold) if args.selection_mode == "safe_gain_listwise" else 0.0,
        "max_loss_probability": float(loss_cap),
        "max_empirical_calibration_loss_rate": float(
            args.max_calibration_loss_rate
        ),
        "admission_loss_probability_search_max": float(
            getattr(
                args,
                "max_admission_loss_probability",
                args.max_calibration_loss_rate,
            )
        ),
        "min_delta_r2_lcb": 0.0,
        "min_delta_r5_lcb": 0.0,
        "min_delta_r2_mean": args.min_delta_r2_mean,
        "min_delta_r5_mean": args.min_delta_r5_mean,
        "min_gain_probability": float(gain_threshold),
        "allowed_query_characteristics": characteristics,
    }, summary


def selected_with_policies(
    rows: Sequence[Dict[str, Any]], policies: Mapping[str, Mapping[str, Any]]
) -> List[Dict[str, Any]]:
    eligible: List[Dict[str, Any]] = []
    for objective_class, policy in policies.items():
        if not policy.get("calibration_feasible", False):
            continue
        class_rows = [row for row in rows if base.plan_objective_class(row) == objective_class]
        threshold_key = (
            "min_unconnected_regret_advantage"
            if bool(policy.get("proof_connected_primary_admission", False))
            and policy["selection_mode"] == "query_relative_slate_regret"
            else "min_regret_advantage"
            if policy["selection_mode"] == "query_relative_slate_regret"
            else "min_absolute_quality_lcb"
            if policy["selection_mode"] == "absolute_quality_regret"
            else
            "min_safety_margin"
            if policy["selection_mode"] == "safe_gain_listwise"
            else "min_utility_mean"
            if policy["selection_mode"] == "risk_budgeted_lexicographic"
            else "min_utility_lcb"
        )
        allowed_characteristics = set(
            policy.get("allowed_query_characteristics", []) or []
        )
        eligible.extend(choose_precomputed(
            class_rows,
            utility_threshold=float(policy.get(threshold_key, 0.0)),
            max_loss_probability=float(policy.get("max_loss_probability", 0.0)),
            selection_mode=str(policy["selection_mode"]),
            min_delta_r2_mean=float(policy.get("min_delta_r2_mean", 0.0)),
            min_delta_r5_mean=float(policy.get("min_delta_r5_mean", 0.0)),
            min_gain_probability=float(policy.get("min_gain_probability", 0.0)),
            allowed_characteristics=(allowed_characteristics or None),
            proof_connected_primary=bool(
                policy.get("proof_connected_primary_admission", False)
            ),
        ))
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        grouped[row["query_key"]].append(row)
    return [
        max(
            candidates,
            key=lambda row: _objective_selection_key(
                row,
                str(
                    policies.get(base.plan_objective_class(row), {}).get(
                        "selection_mode", "pareto_lcb"
                    )
                ),
                base.plan_objective_class(row),
            ),
        )
        for candidates in grouped.values()
    ]


def selected_for_policy(
    rows: Sequence[Dict[str, Any]], policy: Mapping[str, Any]
) -> List[Dict[str, Any]]:
    if not bool(policy.get("calibration_feasible", False)):
        return []
    mode = str(policy.get("selection_mode", "pareto_lcb"))
    threshold_key = (
        "min_unconnected_regret_advantage"
        if bool(policy.get("proof_connected_primary_admission", False))
        and mode == "query_relative_slate_regret"
        else "min_regret_advantage"
        if mode == "query_relative_slate_regret"
        else "min_absolute_quality_lcb"
        if mode == "absolute_quality_regret"
        else
        "min_safety_margin"
        if mode == "safe_gain_listwise"
        else "min_utility_mean"
        if mode == "risk_budgeted_lexicographic"
        else "min_utility_lcb"
    )
    allowed_characteristics = set(
        policy.get("allowed_query_characteristics", []) or []
    )
    return choose_precomputed(
        rows,
        utility_threshold=float(policy.get(threshold_key, 0.0)),
        max_loss_probability=float(policy.get("max_loss_probability", 0.0)),
        selection_mode=mode,
        min_delta_r2_mean=float(policy.get("min_delta_r2_mean", 0.0)),
        min_delta_r5_mean=float(policy.get("min_delta_r5_mean", 0.0)),
        min_gain_probability=float(policy.get("min_gain_probability", 0.0)),
        allowed_characteristics=(allowed_characteristics or None),
        proof_connected_primary=bool(
            policy.get("proof_connected_primary_admission", False)
        ),
    )


def _objective_selection_key(
    row: Mapping[str, Any], selection_mode: str, objective_class: str
) -> Tuple[float, ...]:
    prediction = row.get("prediction", {}) or {}
    if selection_mode == "query_relative_slate_regret":
        return _objective_query_relative_key(row)
    if selection_mode == "absolute_quality_regret":
        metric = "r5" if objective_class == "top5_completion" else "r2"
        other_metric = "r2" if metric == "r5" else "r5"
        return (
            float(prediction.get(f"quality_{metric}_lcb", -1.0)),
            float(prediction.get(f"optimal_{metric}_probability", 0.0)),
            -float(prediction.get(f"regret_{metric}_probability", 1.0)),
            float(prediction.get(f"quality_{other_metric}_lcb", -1.0)),
        )
    if selection_mode == "safe_gain_listwise":
        return (
            float(prediction.get("safety_margin", 0.0)),
            float(prediction.get("gain_probability", 0.0)),
            -float(prediction.get("loss_probability", 0.0)),
            float(prediction.get("delta_r2", 0.0)),
            float(prediction.get("delta_r5", 0.0)),
        )
    if selection_mode == "risk_budgeted_lexicographic":
        if objective_class == "top5_completion":
            return (
                float(prediction.get("delta_r5", 0.0)),
                float(prediction.get("utility_mean", 0.0)),
                float(prediction.get("gain_probability", 0.0)),
                -float(prediction.get("loss_probability", 0.0)),
                float(prediction.get("delta_r2", 0.0)),
            )
        return (
            float(prediction.get("delta_r2", 0.0)),
            float(prediction.get("delta_r5", 0.0)),
            float(prediction.get("utility_mean", 0.0)),
            float(prediction.get("gain_probability", 0.0)),
            -float(prediction.get("loss_probability", 0.0)),
        )
    if objective_class == "top5_completion":
        return (
            float(prediction.get("delta_r5_lcb", 0.0)),
            float(prediction.get("utility_lcb", 0.0)),
            float(prediction.get("delta_r2_lcb", 0.0)),
        )
    return (
        float(prediction.get("utility_lcb", 0.0)),
        float(prediction.get("delta_r2_lcb", 0.0)),
        float(prediction.get("delta_r5_lcb", 0.0)),
    )


def selected_for_objective_policy(
    rows: Sequence[Dict[str, Any]],
    policy: Mapping[str, Any],
    objective_class: str,
) -> List[Dict[str, Any]]:
    """Apply family-specific gates, then compare admitted plans per query."""
    family_policies = policy.get("plan_family_policies", {}) or {}
    if not family_policies:
        return selected_for_policy(rows, policy)
    admitted: List[Dict[str, Any]] = []
    comparator_rows = [
        row for row in rows if _is_query_relative_comparator(row)
    ]
    for plan_name, family_policy in family_policies.items():
        family_rows = [
            row for row in rows if str(row.get("plan_name", "")) == plan_name
        ]
        regime_policies = family_policy.get("structural_regime_policies", {}) or {}
        if regime_policies:
            for regime, regime_policy in regime_policies.items():
                regime_rows = [
                    row for row in family_rows
                    if base.query_contract_regime(row.get("features", {}) or {})
                    == regime
                ]
                query_keys = {row["query_key"] for row in regime_rows}
                competition_rows = [
                    *regime_rows,
                    *[
                        row for row in comparator_rows
                        if row["query_key"] in query_keys
                        and base.query_contract_regime(
                            row.get("features", {}) or {}
                        ) == regime
                    ],
                ]
                admitted.extend(
                    selected_for_policy(competition_rows, regime_policy)
                )
        else:
            query_keys = {row["query_key"] for row in family_rows}
            competition_rows = [
                *family_rows,
                *[
                    row for row in comparator_rows
                    if row["query_key"] in query_keys
                ],
            ]
            admitted.extend(selected_for_policy(competition_rows, family_policy))
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in admitted:
        grouped[str(row.get("query_key", ""))].append(row)
    selection_mode = str(policy.get("selection_mode", "pareto_lcb"))
    return [
        max(
            candidates,
            key=lambda row: _objective_selection_key(
                row, selection_mode, objective_class
            ),
        )
        for candidates in grouped.values()
    ]


def objective_family_residuals(
    rows: Sequence[Dict[str, Any]], alpha: float
) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("plan_name", ""))].append(row)
    result: Dict[str, Dict[str, float]] = {}
    for plan_name, family_rows in grouped.items():
        targets = [
            target
            for target in (
                "delta_r2", "delta_r5", "quality_r2", "quality_r5",
                "regret_r2", "regret_r5",
            )
            if all(
                target in row and target in (row.get("prediction", {}) or {})
                for row in family_rows
            )
        ]
        result[plan_name] = {
            target: calibrated_residual(family_rows, target, alpha)
            for target in targets
        }
    return result


def update_objective_family_lcbs(
    rows: Sequence[Dict[str, Any]],
    residuals: Mapping[str, Mapping[str, float]],
) -> None:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("plan_name", ""))].append(row)
    for plan_name, family_rows in grouped.items():
        update_lcbs(family_rows, residuals.get(plan_name, {}))


def calibrate_objective_family_policy(
    rows: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
    objective_class: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    family_policies: Dict[str, Dict[str, Any]] = {}
    family_reports: Dict[str, Dict[str, Any]] = {}
    for plan_name in sorted({str(row.get("plan_name", "")) for row in rows}):
        family_rows = [
            row for row in rows if str(row.get("plan_name", "")) == plan_name
        ]
        if plan_name in {base.PLAN_SEMANTIC, base.PLAN_SEMANTIC_TOP2_NOOP}:




            family_policy = {
                "selection_mode": args.selection_mode,
                "calibration_feasible": True,
                "min_absolute_quality_lcb": 0.0,
                "min_regret_advantage": 0.0,
                "min_utility_lcb": 0.0,
                "min_utility_mean": 0.0,
                "min_safety_margin": 0.0,
                "max_loss_probability": 1.0,
                "min_delta_r2_lcb": 0.0,
                "min_delta_r5_lcb": 0.0,
                "min_delta_r2_mean": 0.0,
                "min_delta_r5_mean": 0.0,
                "min_gain_probability": 0.0,
                "allowed_query_characteristics": sorted({
                    base.query_characteristic(row) for row in family_rows
                }),
                "zero_modification_comparator": True,
            }
            family_report = base.report(
                family_rows,
                len({row["query_key"] for row in family_rows}),
            )
        else:
            comparator_rows = [
                row for row in rows if _is_query_relative_comparator(row)
            ]
            regime_policies: Dict[str, Dict[str, Any]] = {}
            regime_reports: Dict[str, Dict[str, Any]] = {}
            for regime in sorted({
                base.query_contract_regime(row.get("features", {}) or {})
                for row in family_rows
            }):
                regime_rows = [
                    row for row in family_rows
                    if base.query_contract_regime(row.get("features", {}) or {})
                    == regime
                ]
                query_keys = {row["query_key"] for row in regime_rows}
                competition_rows = [
                    *regime_rows,
                    *[
                        row for row in comparator_rows
                        if row["query_key"] in query_keys
                        and base.query_contract_regime(
                            row.get("features", {}) or {}
                        ) == regime
                    ],
                ]
                regime_policy, regime_report = calibrate_oof_policy(
                    competition_rows, args
                )
                regime_policy["query_contract_regime"] = regime
                regime_policies[regime] = regime_policy
                regime_reports[regime] = regime_report
            family_policy = {
                "selection_mode": args.selection_mode,
                "calibration_feasible": any(
                    bool(value.get("calibration_feasible", False))
                    for value in regime_policies.values()
                ),
                "structural_regime_policies": regime_policies,
                "contract_conditioned_admission": True,
            }
            family_selected: List[Dict[str, Any]] = []
            for regime, regime_policy in regime_policies.items():
                regime_rows = [
                    row for row in family_rows
                    if base.query_contract_regime(row.get("features", {}) or {})
                    == regime
                ]
                query_keys = {row["query_key"] for row in regime_rows}
                competition_rows = [
                    *regime_rows,
                    *[
                        row for row in comparator_rows
                        if row["query_key"] in query_keys
                        and base.query_contract_regime(
                            row.get("features", {}) or {}
                        ) == regime
                    ],
                ]
                family_selected.extend(
                    selected_for_policy(competition_rows, regime_policy)
                )
            family_report = {
                **base.report(
                    family_selected,
                    len({row["query_key"] for row in family_rows}),
                ),
                "structural_regimes": regime_reports,
            }
        family_policy.update({
            "objective_class": objective_class,
            "plan_family": plan_name,
        })
        family_policies[plan_name] = family_policy
        family_reports[plan_name] = family_report
    policy = {
        "selection_mode": args.selection_mode,
        "calibration_feasible": any(
            value.get("calibration_feasible", False)
            for value in family_policies.values()
        ),
        "objective_class": objective_class,
        "plan_family_policies": family_policies,
    }
    selected = selected_for_objective_policy(rows, policy, objective_class)
    report = {
        **base.report(selected, len({row["query_key"] for row in rows})),
        "plan_families": family_reports,
    }
    return policy, report


SCORE_ALIGNMENT_FIELDS = (
    "delta_r2",
    "delta_r5",
    "loss_probability",
    "gain_probability",
    "delta_r2_lcb",
    "delta_r5_lcb",
    "utility_mean",
    "utility_lcb",
    "safety_margin",
    "quality_r2",
    "quality_r5",
    "quality_r2_lcb",
    "quality_r5_lcb",
    "optimal_r2_probability",
    "optimal_r5_probability",
    "regret_r2_probability",
    "regret_r5_probability",
    "regret_r2",
    "regret_r5",
    "regret_r2_ucb",
    "regret_r5_ucb",
)

OUTCOME_CALIBRATION_FIELDS = (
    "delta_r2",
    "delta_r5",
    "gain_probability",
    "loss_probability",
    "quality_r2",
    "quality_r5",
    "optimal_r2_probability",
    "optimal_r5_probability",
    "regret_r2",
    "regret_r5",
)


def _outcome_calibration_key(
    objective_class: str, plan_name: str, regime: str
) -> str:
    return "::".join((objective_class, plan_name or "*", regime or "*"))


def _fit_oof_outcome_group(
    rows: Sequence[Mapping[str, Any]], alpha: float
) -> Dict[str, Any]:
    """Train monotone score-to-outcome maps and rank-specific residual bands."""
    fields: Dict[str, Any] = {}
    for field in OUTCOME_CALIBRATION_FIELDS:
        pairs = [
            (
                float((row.get("prediction", {}) or {})[field]),
                float(row[field]),
            )
            for row in rows
            if field in row
            and field in (row.get("prediction", {}) or {})
        ]
        if len(pairs) < 20:
            continue
        source, target = zip(*pairs)
        mapping = _fit_paired_isotonic_alignment(source, target)
        if not mapping:
            continue
        aligned = np.interp(
            np.asarray(source, dtype=float),
            np.asarray(mapping["source_knots"], dtype=float),
            np.asarray(mapping["target_knots"], dtype=float),
        )
        residuals = np.asarray(target, dtype=float) - aligned
        fields[field] = {
            **mapping,
            "residual_lower": float(np.quantile(residuals, alpha)),
            "residual_upper": float(np.quantile(residuals, 1.0 - alpha)),
            "sample_count": len(pairs),
        }
    return {
        "sample_count": len(rows),
        "fields": fields,
    }


def fit_structure_conditioned_outcome_calibration(
    rows: Sequence[Dict[str, Any]], alpha: float
) -> Dict[str, Any]:
    """Estimate reliability at class, family, and structural-regime levels."""
    groups: Dict[str, Dict[str, Any]] = {}
    by_objective: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_family: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    by_regime: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        objective_class = base.plan_objective_class(row)
        plan_name = str(row.get("plan_name", ""))
        regime = base.query_contract_regime(row.get("features", {}) or {})
        by_objective[objective_class].append(row)
        by_family[(objective_class, plan_name)].append(row)
        by_regime[(objective_class, plan_name, regime)].append(row)

    for objective_class, group_rows in by_objective.items():
        payload = _fit_oof_outcome_group(group_rows, alpha)
        if payload["fields"]:
            groups[_outcome_calibration_key(objective_class, "*", "*")] = payload
    for (objective_class, plan_name), group_rows in by_family.items():
        payload = _fit_oof_outcome_group(group_rows, alpha)
        if payload["fields"]:
            groups[_outcome_calibration_key(objective_class, plan_name, "*")] = payload
    for (objective_class, plan_name, regime), group_rows in by_regime.items():
        if len(group_rows) < 40:
            continue
        payload = _fit_oof_outcome_group(group_rows, alpha)
        if payload["fields"]:
            groups[
                _outcome_calibration_key(objective_class, plan_name, regime)
            ] = payload
    return {
        "enabled": True,
        "method": "rank_family_regime_oof_isotonic_v1",
        "alpha": float(alpha),
        "groups": groups,
    }


def apply_structure_conditioned_outcome_calibration(
    rows: Sequence[Dict[str, Any]], calibration: Mapping[str, Any]
) -> None:
    """Apply the most specific available OOF reliability map in-place."""
    groups = calibration.get("groups", {}) or {}
    for row in rows:
        prediction = dict(row.get("prediction", {}) or {})
        objective_class = base.plan_objective_class(row)
        plan_name = str(row.get("plan_name", ""))
        regime = base.query_contract_regime(row.get("features", {}) or {})
        group = next((
            groups.get(key)
            for key in (
                _outcome_calibration_key(objective_class, plan_name, regime),
                _outcome_calibration_key(objective_class, plan_name, "*"),
                _outcome_calibration_key(objective_class, "*", "*"),
            )
            if isinstance(groups.get(key), Mapping)
        ), {}) or {}
        for field, mapping in (group.get("fields", {}) or {}).items():
            if field not in prediction:
                continue
            source = np.asarray(mapping.get("source_knots", []) or [], dtype=float)
            target = np.asarray(mapping.get("target_knots", []) or [], dtype=float)
            if len(source) < 2 or len(source) != len(target):
                continue
            value = float(np.interp(float(prediction[field]), source, target))
            if field.endswith("probability"):
                value = min(1.0, max(0.0, value))
            prediction[field] = value
        if "optimal_r2_probability" in prediction:
            prediction["regret_r2_probability"] = (
                1.0 - float(prediction["optimal_r2_probability"])
            )
        if "optimal_r5_probability" in prediction:
            prediction["regret_r5_probability"] = (
                1.0 - float(prediction["optimal_r5_probability"])
            )
        prediction["utility_mean"] = (
            2.0 * float(prediction.get("delta_r2", 0.0))
            + float(prediction.get("delta_r5", 0.0))
        )
        prediction["safety_margin"] = (
            float(prediction.get("gain_probability", 0.0))
            - float(prediction.get("loss_probability", 0.0))
        )
        prediction["outcome_calibration_group"] = next((
            key
            for key in (
                _outcome_calibration_key(objective_class, plan_name, regime),
                _outcome_calibration_key(objective_class, plan_name, "*"),
                _outcome_calibration_key(objective_class, "*", "*"),
            )
            if isinstance(groups.get(key), Mapping)
        ), "")
        prediction["outcome_calibration_applied"] = bool(
            prediction["outcome_calibration_group"]
        )
        row["prediction"] = prediction


def _fit_paired_isotonic_alignment(
    source: Sequence[float], target: Sequence[float]
) -> Dict[str, Any]:
    """Train a compact monotone map from final-training scores to OOF scores."""
    source_array = np.asarray(source, dtype=float)
    target_array = np.asarray(target, dtype=float)
    finite = np.isfinite(source_array) & np.isfinite(target_array)
    source_array = source_array[finite]
    target_array = target_array[finite]
    if len(source_array) < 20 or len(np.unique(source_array)) < 2:
        return {}
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(source_array, target_array)


    source_knots = np.unique(np.quantile(
        source_array, np.linspace(0.0, 1.0, 33)
    ))
    if len(source_knots) < 2:
        return {}
    target_knots = calibrator.predict(source_knots)
    aligned = calibrator.predict(source_array)
    mae_before = float(np.mean(np.abs(source_array - target_array)))
    mae_after = float(np.mean(np.abs(aligned - target_array)))
    return {
        "source_knots": source_knots.astype(float).tolist(),
        "target_knots": target_knots.astype(float).tolist(),
        "mae_before": mae_before,
        "mae_after": mae_after,
    }


def fit_objective_family_score_alignments(
    oof_rows: Sequence[Dict[str, Any]],
    final_rows: Sequence[Dict[str, Any]],
    objective_class: str,
) -> Dict[str, Dict[str, Any]]:
    """Pair final-training and OOF scores for the same historical ranking action."""
    oof_by_key = {
        row_checkpoint_key(row): row
        for row in oof_rows
        if base.plan_objective_class(row) == objective_class
        and isinstance(row.get("prediction"), Mapping)
    }
    paired_by_family: Dict[str, List[Tuple[Dict[str, Any], Dict[str, Any]]]] = (
        defaultdict(list)
    )
    for row in final_rows:
        if base.plan_objective_class(row) != objective_class:
            continue
        oof_row = oof_by_key.get(row_checkpoint_key(row))
        if oof_row is None or not isinstance(row.get("prediction"), Mapping):
            continue
        paired_by_family[str(row.get("plan_name", ""))].append((row, oof_row))

    alignments: Dict[str, Dict[str, Any]] = {}
    for plan_name, pairs in sorted(paired_by_family.items()):
        fields: Dict[str, Any] = {}
        for field in SCORE_ALIGNMENT_FIELDS:
            mapping = _fit_paired_isotonic_alignment(
                [float(final["prediction"].get(field, 0.0)) for final, _ in pairs],
                [float(oof["prediction"].get(field, 0.0)) for _, oof in pairs],
            )
            if mapping:
                fields[field] = mapping
        if fields:
            alignments[plan_name] = {
                "method": "paired_oof_isotonic_v1",
                "objective_class": objective_class,
                "plan_family": plan_name,
                "pair_count": len(pairs),
                "fields": fields,
            }
    return alignments


def _stage2_hardness(row: Mapping[str, Any]) -> Tuple[float, ...]:
    features = row.get("features", {}) or {}
    proof_score = sum(float(features.get(name, 0.0) or 0.0) for name in (
        "residual_candidate_path_connected",
        "residual_candidate_complete_provenance",
        "residual_candidate_binding",
        "residual_candidate_typed_slot_binding",
        "residual_candidate_terminal_relation_match",
        "residual_candidate_exact_entity_key",
        "residual_candidate_role_consistent",
    ))
    coverage = sum(float(features.get(name, 0.0) or 0.0) for name in (
        "residual_candidate_query_coverage",
        "residual_candidate_anchor_coverage",
        "residual_candidate_relation_coverage",
        "residual_candidate_answer_type_score",
    ))
    original_rank = float(
        features.get("residual_candidate_original_rank_log", 99.0) or 99.0
    )
    return (
        proof_score,
        coverage,
        float(features.get("residual_candidate_confidence", 0.0) or 0.0),
        float(features.get("residual_candidate_score", 0.0) or 0.0),
        -original_rank,
        str(row.get("plan_instance", "")),
    )


def deduplicate_stage2_slates(
    rows: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Collapse cross-family aliases of the same ordered residual Top-5 slate."""
    grouped: Dict[Tuple[str, Tuple[int, ...]], List[Dict[str, Any]]] = defaultdict(list)
    passthrough: List[Dict[str, Any]] = []
    for row in rows:
        slate = tuple(_unique_ids(row.get("doc_ids", []) or [])[:5])
        if not slate:
            passthrough.append(row)
            continue
        grouped[(str(row.get("query_key", "")), slate)].append(row)

    selected_ids = {id(row) for row in passthrough}
    duplicate_groups = 0
    for group_rows in grouped.values():
        representative = max(
            group_rows,
            key=lambda row: (
                float(str(row.get("plan_name", "")) == base.PLAN_SEMANTIC),
                *_stage2_hardness(row),
            ),
        )
        selected_ids.add(id(representative))
        duplicate_groups += int(len(group_rows) > 1)
    deduplicated = [
        row for row in rows if id(row) in selected_ids
    ]
    return deduplicated, {
        "candidate_count_before": len(rows),
        "candidate_count_after": len(deduplicated),
        "duplicate_slate_group_count": duplicate_groups,
        "dropped_alias_count": len(rows) - len(deduplicated),
    }


def sample_stage2_training_rows(
    rows: Sequence[Dict[str, Any]],
    hard_negatives_per_query: int,
    *,
    group_across_families: bool = False,
) -> List[Dict[str, Any]]:
    """Retain sparse outcomes plus a bounded listwise neighborhood per query."""
    limit = max(1, int(hard_negatives_per_query))
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        group_family = "" if group_across_families else str(
            row.get("plan_name", "")
        )
        grouped[(str(row.get("query_key", "")), group_family)].append(row)
    sampled: List[Dict[str, Any]] = []
    for group_rows in grouped.values():
        informative = [
            row for row in group_rows
            if float(row.get("gain_probability", 0.0)) > 0.5
            or float(row.get("loss_probability", 0.0)) > 0.5
        ]
        oracle_neutral = [
            row for row in group_rows
            if row not in informative
            and float(row.get("optimal_r5_probability", 0.0)) > 0.5
        ]
        neutral = [
            row for row in group_rows
            if row not in informative and row not in oracle_neutral
        ]
        sampled.extend(informative)
        sampled.extend(
            sorted(oracle_neutral, key=_stage2_hardness, reverse=True)[:2]
        )
        sampled.extend(sorted(neutral, key=_stage2_hardness, reverse=True)[:limit])
    return sampled


def model_key_objective_family(model_key: str) -> Tuple[str, str]:
    if model_key.startswith(base.OBJECTIVE_FAMILY_EXPERT_PREFIX):
        value = model_key[len(base.OBJECTIVE_FAMILY_EXPERT_PREFIX):]
        objective_class, _, plan_name = value.partition("::")
        return objective_class, plan_name
    if model_key.startswith(base.OBJECTIVE_CLASS_EXPERT_PREFIX):
        return model_key[len(base.OBJECTIVE_CLASS_EXPERT_PREFIX):], ""
    return "", ""


def _unique_ids(values: Sequence[Any]) -> List[int]:
    result: List[int] = []
    seen: set[int] = set()
    for value in values:
        doc_id = int(value)
        if doc_id < 0 or doc_id in seen:
            continue
        seen.add(doc_id)
        result.append(doc_id)
    return result


def build_stage2_residual_rows(
    rows: Sequence[Dict[str, Any]],
    stage1_selected: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Relabel Top-5 actions against the OOF-selected, frozen Top-2 prefix."""
    stage1_by_query = {row["query_key"]: row for row in stage1_selected}
    residual_rows: List[Dict[str, Any]] = []
    for row in rows:
        if base.plan_objective_class(row) != "top5_completion":
            continue
        semantic_ids = _unique_ids(row.get("semantic_doc_ids", []) or [])[:5]
        completion_ids = _unique_ids(row.get("doc_ids", []) or [])[:5]
        gold_ids = set(_unique_ids(row.get("gold_doc_ids", []) or []))
        if len(semantic_ids) < 2 or len(completion_ids) < 2 or not gold_ids:
            continue
        stage1 = stage1_by_query.get(str(row.get("query_key", "")))
        if stage1 is None:
            stage1_ids = list(semantic_ids)
            stage1_name = base.PLAN_SEMANTIC
        else:
            stage1_ids = _unique_ids([
                *(stage1.get("doc_ids", []) or [])[:2],
                *semantic_ids,
            ])[:5]
            stage1_name = str(stage1.get("plan_instance", ""))
        composed_ids = base.compose_hierarchical_topk(
            stage1_ids,
            completion_ids,
            semantic_ids,
            top_k=5,
        )
        denominator = max(len(gold_ids), 1)
        baseline_r2 = len(set(stage1_ids[:2]) & gold_ids) / denominator
        baseline_r5 = len(set(stage1_ids[:5]) & gold_ids) / denominator
        candidate_r2 = len(set(composed_ids[:2]) & gold_ids) / denominator
        candidate_r5 = len(set(composed_ids[:5]) & gold_ids) / denominator
        delta_r2 = candidate_r2 - baseline_r2
        delta_r5 = candidate_r5 - baseline_r5
        value = dict(row)
        value.update({
            "doc_ids": composed_ids,
            "stage1_doc_ids": stage1_ids,
            "stage1_plan_instance": stage1_name,
            "delta_r2": delta_r2,
            "delta_r5": delta_r5,
            "baseline_quality_r2": baseline_r2,
            "baseline_quality_r5": baseline_r5,
            "quality_r2": candidate_r2,
            "quality_r5": candidate_r5,
            "gain_probability": float(delta_r2 > 1e-12 or delta_r5 > 1e-12),
            "loss_probability": float(delta_r2 < -1e-12 or delta_r5 < -1e-12),
        })
        residual_rows.append(value)
    base.annotate_query_oracle_targets(residual_rows)
    return residual_rows


def combine_hierarchical_selections(
    stage1_selected: Sequence[Dict[str, Any]],
    stage2_selected: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Create one final-slate outcome row per query for OOF certification."""
    first = {row["query_key"]: row for row in stage1_selected}
    second = {row["query_key"]: row for row in stage2_selected}
    combined: List[Dict[str, Any]] = []
    for query_key in sorted(set(first) | set(second)):
        stage1 = first.get(query_key)
        stage2 = second.get(query_key)
        template = dict(stage2 or stage1 or {})
        delta_r2 = float((stage1 or {}).get("delta_r2", 0.0)) + float(
            (stage2 or {}).get("delta_r2", 0.0)
        )
        delta_r5 = float((stage1 or {}).get("delta_r5", 0.0)) + float(
            (stage2 or {}).get("delta_r5", 0.0)
        )
        template.update({
            "query_key": query_key,
            "plan_name": str(
                (stage2 or stage1 or {}).get("plan_name", base.PLAN_SEMANTIC)
            ),
            "delta_r2": delta_r2,
            "delta_r5": delta_r5,
            "gain_probability": float(delta_r2 > 1e-12 or delta_r5 > 1e-12),
            "loss_probability": float(delta_r2 < -1e-12 or delta_r5 < -1e-12),
            "stage1_plan_instance": str(
                (stage1 or {}).get("plan_instance", base.PLAN_SEMANTIC)
            ),
            "stage2_plan_instance": str((stage2 or {}).get("plan_instance", "")),
        })
        combined.append(template)
    return combined


def grouped_oof_stage2_predictions(
    rows: Sequence[Dict[str, Any]],
    fold_by_query: Mapping[str, int],
    fold_count: int,
    args: argparse.Namespace,
    checkpoint_dir: Path,
    fingerprint: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[int]]:
    """Train leakage-free residual Top-5 experts after Stage-1 is frozen."""
    predicted_rows: List[Dict[str, Any]] = []
    reports: List[Dict[str, Any]] = []
    resumed: List[int] = []
    stage_fingerprint = hashlib.sha256(
        f"{fingerprint}:hierarchical-stage2".encode()
    ).hexdigest()
    for fold in range(fold_count):
        heldout = [row for row in rows if fold_by_query[row["query_key"]] == fold]
        train = [row for row in rows if fold_by_query[row["query_key"]] != fold]
        if not train or not heldout:
            raise RuntimeError(f"empty Stage-2 train/heldout rows for fold {fold}")
        training_rows = (
            sample_stage2_training_rows(
                train,
                args.top5_hard_negatives_per_query,
                group_across_families=bool(args.stage2_slate_dedup),
            )
            if args.query_grouped_hard_negative_top5
            else train
        )
        checkpoint_path = checkpoint_dir / f"stage2_fold_{fold}.json.gz"
        if args.resume_checkpoints and checkpoint_path.is_file():
            restored, report = load_fold_checkpoint(
                checkpoint_path,
                fingerprint=stage_fingerprint,
                fold=fold,
                heldout_rows=heldout,
            )
            predicted_rows.extend(restored)
            reports.append(report)
            resumed.append(fold)
            print(
                f"[grouped_oof-cbo] Stage-2 fold={fold + 1}/{fold_count} "
                f"restored checkpoint actions={len(restored)}",
                flush=True,
            )
            continue
        print(
            f"[grouped_oof-cbo] Stage-2 fold={fold + 1}/{fold_count} "
            f"train_actions={len(train)} fit_actions={len(training_rows)} "
            f"heldout_actions={len(heldout)}",
            flush=True,
        )
        fold_args = argparse.Namespace(**vars(args))
        fold_args.seed = args.seed + 10_000 + fold * 101
        fold_args.skip_model_export = True
        fold_args.skip_residual_calibration = True
        payloads, models = base.train_models(training_rows, training_rows, fold_args)
        for payload in payloads.values():
            payload["delta_r2_residual_lcb"] = 0.0
            payload["delta_r5_residual_lcb"] = 0.0
            payload["quality_r2_residual_lcb"] = 0.0
            payload["quality_r5_residual_lcb"] = 0.0
        predicted = base.attach_predictions(heldout, payloads, models)
        for row in predicted:
            row["grouped_oof_fold"] = fold
        report = {
            "fold": fold,
            "stage": "residual_top5_completion",
            "train_query_count": len({row["query_key"] for row in train}),
            "heldout_query_count": len({row["query_key"] for row in heldout}),
            "train_action_count": len(train),
            "fit_action_count": len(training_rows),
            "heldout_action_count": len(heldout),
        }
        write_fold_checkpoint(
            checkpoint_path,
            fingerprint=stage_fingerprint,
            fold=fold,
            predicted=predicted,
            report=report,
        )
        predicted_rows.extend(predicted)
        reports.append(report)
        del payloads, models, predicted
        gc.collect()
    return predicted_rows, reports, resumed


def main() -> None:
    args = parse_args()
    if (
        args.selection_mode == "absolute_quality_regret"
        and not args.absolute_quality_regret_targets
    ):
        raise ValueError(
            "absolute_quality_regret selection requires "
            "--absolute-quality-regret-targets"
        )
    if (
        args.selection_mode == "query_relative_slate_regret"
        and not args.query_relative_slate_regret_targets
    ):
        raise ValueError(
            "query_relative_slate_regret selection requires "
            "--query-relative-slate-regret-targets"
        )
    if (
        args.hierarchical_objective_execution
        and args.outcome_model_scope not in {"objective_class", "objective_family"}
    ):
        raise ValueError(
            "hierarchical objective execution requires "
            "--outcome-model-scope objective_class or objective_family"
        )
    if (
        args.structure_conditioned_outcome_calibration
        and not args.grouped_oof_score_alignment
    ):
        raise ValueError(
            "structure-conditioned outcome calibration requires "
            "--grouped_oof-score-alignment so final-training scores are mapped to the "
            "calibrated OOF outcome scale"
        )
    datasets = [value.strip() for value in args.datasets.split(",") if value.strip()]
    selected_plans = set(base.allowed_plans(args))
    contract_path = Path(args.workload_dir) / "workload_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    fold_count = int(contract.get("fold_count", 0) or 0)
    if contract.get("type") != "historical_query_grouped_oof_contract" or fold_count < 3:
        raise ValueError("workload is not a valid historical-query cross-validation contract")

    replay_gold = base.historical_gold_ids(args.workload_dir, "q_h", datasets)
    query_keys: set[str] = set()
    print(
        "[grouped_oof-cbo] loading traces; allowed_plans="
        + ",".join(sorted(selected_plans)),
        flush=True,
    )
    rows = [
        row for row in base.rows_from_traces(
            args.trace_dir,
            datasets,
            query_keys,
            replay_gold,
            pareto_safe_gain_label=bool(args.pareto_safe_gain_label),
            allowed_plan_names=selected_plans,
            query_relative_slate_regret=bool(
                args.query_relative_slate_regret_targets
            ),
        )
    ]
    topology_plan_families: Dict[str, List[str]] = {}
    if args.topology_partitioned_plan_routing:
        topology_plan_families = base.topology_partitioned_plan_families(
            args.topology_plan_routing_profile
        )
        before_count = len(rows)
        rows = [
            row for row in rows
            if str(row.get("plan_name", "")) == base.PLAN_SEMANTIC
            or base.plan_allowed_for_topology(
                str(row.get("plan_name", "")),
                row.get("features", {}) or {},
                topology_plan_families,
            )
        ]
        print(
            f"[grouped_oof-cbo] topology routing retained={len(rows)}/"
            f"{before_count} actions",
            flush=True,
        )
    if (
        args.hierarchical_objective_execution
        and args.query_relative_slate_regret_targets
    ):


        top2_rows = [
            row for row in rows
            if base.plan_objective_class(row) == "top2_ordering"
        ]
        base.annotate_query_oracle_targets(top2_rows)
    if not rows:
        raise RuntimeError("no valid historical replay actions")
    print(
        f"[grouped_oof-cbo] loaded actions={len(rows)} "
        f"historical_queries={len(query_keys)} folds={fold_count}",
        flush=True,
    )
    fold_by_query: Dict[str, int] = {}
    for dataset in datasets:
        assignments = (
            (contract.get("datasets", {}) or {}).get(dataset, {}) or {}
        ).get("query_fold", []) or []
        for query_index, fold in enumerate(assignments):
            fold_by_query[f"{dataset}:{query_index}"] = int(fold)
    missing = sorted({row["query_key"] for row in rows} - set(fold_by_query))
    if missing:
        raise ValueError(f"trace queries missing cross-validation assignments: {missing[:8]}")

    checkpoint_dir = (
        Path(args.checkpoint_dir)
        if args.checkpoint_dir
        else Path(args.output_model).parent / "grouped_oof_checkpoints"
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = checkpoint_fingerprint(
        args, contract_path, datasets, selected_plans, fold_count
    )
    oof_rows: List[Dict[str, Any]] = []
    fold_model_reports = []
    resumed_folds: List[int] = []
    for fold in range(fold_count):
        heldout_rows = [row for row in rows if fold_by_query[row["query_key"]] == fold]
        checkpoint_path = checkpoint_dir / f"fold_{fold}.json.gz"
        if args.resume_checkpoints and checkpoint_path.is_file():
            predicted, report = load_fold_checkpoint(
                checkpoint_path,
                fingerprint=(
                    args.trusted_stage1_checkpoint_fingerprint or fingerprint
                ),
                fold=fold,
                heldout_rows=heldout_rows,
            )
            oof_rows.extend(predicted)
            fold_model_reports.append(report)
            resumed_folds.append(fold)
            print(
                f"[grouped_oof-cbo] fold={fold + 1}/{fold_count} "
                f"restored checkpoint actions={len(predicted)}",
                flush=True,
            )
            del predicted, heldout_rows
            gc.collect()
            continue
        train_rows = [row for row in rows if fold_by_query[row["query_key"]] != fold]
        if not train_rows or not heldout_rows:
            raise RuntimeError(f"empty train/heldout rows for fold {fold}")
        print(
            f"[grouped_oof-cbo] fold={fold + 1}/{fold_count} "
            f"train_actions={len(train_rows)} heldout_actions={len(heldout_rows)}",
            flush=True,
        )
        fold_args = argparse.Namespace(**vars(args))
        fold_args.seed = args.seed + fold * 101


        fold_args.skip_model_export = True
        fold_args.skip_residual_calibration = True
        plan_payloads, models = base.train_models(train_rows, train_rows, fold_args)
        for payload in plan_payloads.values():
            payload["delta_r2_residual_lcb"] = 0.0
            payload["delta_r5_residual_lcb"] = 0.0
            payload["quality_r2_residual_lcb"] = 0.0
            payload["quality_r5_residual_lcb"] = 0.0
        predicted = base.attach_predictions(heldout_rows, plan_payloads, models)
        for row in predicted:
            row["grouped_oof_fold"] = fold
        report = {
            "fold": fold,
            "train_query_count": len({row["query_key"] for row in train_rows}),
            "heldout_query_count": len({row["query_key"] for row in heldout_rows}),
            "train_action_count": len(train_rows),
            "heldout_action_count": len(heldout_rows),
        }
        write_fold_checkpoint(
            checkpoint_path,
            fingerprint=fingerprint,
            fold=fold,
            predicted=predicted,
            report=report,
        )
        oof_rows.extend(predicted)
        fold_model_reports.append(report)
        print(
            f"[grouped_oof-cbo] fold={fold + 1}/{fold_count} complete "
            f"checkpoint={checkpoint_path}",
            flush=True,
        )
        del train_rows, heldout_rows, plan_payloads, models, predicted
        gc.collect()

    class_policies: Dict[str, Dict[str, Any]] = {}
    class_reports: Dict[str, Dict[str, Any]] = {}
    global_policy: Dict[str, Any] = {}
    stage2_rows: List[Dict[str, Any]] = []
    stage2_dedup_report: Dict[str, int] = {
        "candidate_count_before": 0,
        "candidate_count_after": 0,
        "duplicate_slate_group_count": 0,
        "dropped_alias_count": 0,
    }
    stage2_oof_rows: List[Dict[str, Any]] = []
    stage2_resumed_folds: List[int] = []
    outcome_calibration: Dict[str, Any] = {
        "enabled": bool(args.structure_conditioned_outcome_calibration),
        "method": (
            "rank_family_regime_oof_isotonic_v1"
            if args.structure_conditioned_outcome_calibration else "disabled"
        ),
        "alpha": float(args.alpha),
        "groups": {},
    }
    if args.hierarchical_objective_execution:
        top2_oof_rows = [
            row for row in oof_rows
            if base.plan_objective_class(row) == "top2_ordering"
        ]
        if not top2_oof_rows:
            raise RuntimeError("hierarchical execution has no Top-2 OOF actions")
        if args.structure_conditioned_outcome_calibration:
            top2_calibration = fit_structure_conditioned_outcome_calibration(
                top2_oof_rows, args.alpha
            )
            outcome_calibration["groups"].update(
                top2_calibration.get("groups", {}) or {}
            )
            apply_structure_conditioned_outcome_calibration(
                top2_oof_rows, top2_calibration
            )
        if args.outcome_model_scope == "objective_family":
            top2_residuals = objective_family_residuals(
                top2_oof_rows, args.alpha
            )
            update_objective_family_lcbs(top2_oof_rows, top2_residuals)
        else:
            top2_residuals = {
                target: calibrated_residual(top2_oof_rows, target, args.alpha)
                for target in residual_target_names(top2_oof_rows)
            }
            update_lcbs(top2_oof_rows, top2_residuals)
        top2_args = base.objective_class_args(args, "top2_ordering")
        if args.outcome_model_scope == "objective_family":
            top2_policy, top2_report = calibrate_objective_family_policy(
                top2_oof_rows, top2_args, "top2_ordering"
            )
        else:
            top2_policy, top2_report = calibrate_oof_policy(
                top2_oof_rows, top2_args
            )
        top2_policy.update({
            "objective_class": "top2_ordering",
            "risk_contract": "strict_top2_and_top5_non_regression",
        })
        class_policies["top2_ordering"] = top2_policy
        class_reports["top2_ordering"] = top2_report
        stage1_selected = selected_for_objective_policy(
            top2_oof_rows, top2_policy, "top2_ordering"
        )
        stage2_rows = build_stage2_residual_rows(rows, stage1_selected)
        if not stage2_rows:
            raise RuntimeError("hierarchical execution has no residual Top-5 actions")
        if args.stage2_slate_dedup:
            stage2_rows, stage2_dedup_report = deduplicate_stage2_slates(
                stage2_rows
            )
            print(
                "[grouped_oof-cbo] Stage-2 slate dedup "
                f"before={stage2_dedup_report['candidate_count_before']} "
                f"after={stage2_dedup_report['candidate_count_after']} "
                f"duplicate_groups={stage2_dedup_report['duplicate_slate_group_count']}",
                flush=True,
            )
        else:
            stage2_dedup_report.update({
                "candidate_count_before": len(stage2_rows),
                "candidate_count_after": len(stage2_rows),
            })
        stage2_oof_rows, stage2_reports, stage2_resumed_folds = (
            grouped_oof_stage2_predictions(
                stage2_rows,
                fold_by_query,
                fold_count,
                args,
                checkpoint_dir,
                fingerprint,
            )
        )
        fold_model_reports.extend(stage2_reports)
        if args.structure_conditioned_outcome_calibration:
            top5_calibration = fit_structure_conditioned_outcome_calibration(
                stage2_oof_rows, args.alpha
            )
            outcome_calibration["groups"].update(
                top5_calibration.get("groups", {}) or {}
            )
            apply_structure_conditioned_outcome_calibration(
                stage2_oof_rows, top5_calibration
            )
        if args.outcome_model_scope == "objective_family":
            top5_residuals = objective_family_residuals(
                stage2_oof_rows, args.alpha
            )
            update_objective_family_lcbs(stage2_oof_rows, top5_residuals)
        else:
            top5_residuals = {
                target: calibrated_residual(stage2_oof_rows, target, args.alpha)
                for target in residual_target_names(stage2_oof_rows)
            }
            update_lcbs(stage2_oof_rows, top5_residuals)
        top5_args = base.objective_class_args(args, "top5_completion")
        if args.outcome_model_scope == "objective_family":
            top5_policy, top5_report = calibrate_objective_family_policy(
                stage2_oof_rows, top5_args, "top5_completion"
            )
        else:
            top5_policy, top5_report = calibrate_oof_policy(
                stage2_oof_rows, top5_args
            )
        top5_policy.update({
            "objective_class": "top5_completion",
            "risk_contract": "frozen_top2_residual_r5_completion",
        })
        class_policies["top5_completion"] = top5_policy
        class_reports["top5_completion"] = top5_report
        stage2_selected = selected_for_objective_policy(
            stage2_oof_rows, top5_policy, "top5_completion"
        )
        selected = combine_hierarchical_selections(
            stage1_selected, stage2_selected
        )
        oof_rows = [*top2_oof_rows, *stage2_oof_rows]
        residuals: Dict[str, Any] = {
            "top2_ordering": top2_residuals,
            "top5_completion": top5_residuals,
        }
    else:
        if args.structure_conditioned_outcome_calibration:
            outcome_calibration = fit_structure_conditioned_outcome_calibration(
                oof_rows, args.alpha
            )
            apply_structure_conditioned_outcome_calibration(
                oof_rows, outcome_calibration
            )
        residuals = {
            target: calibrated_residual(oof_rows, target, args.alpha)
            for target in residual_target_names(oof_rows)
        }
        update_lcbs(oof_rows, residuals)
        if args.selection_mode == "query_relative_slate_regret":
            global_policy, summary = calibrate_oof_policy(oof_rows, args)
            global_policy["risk_contract"] = (
                "query_local_complete_slate_regret_with_semantic_competitor"
            )
            class_reports["complete_slate"] = summary
            selected = selected_for_policy(oof_rows, global_policy)
        else:
            for objective_class in ("top2_ordering", "top5_completion"):
                class_args = base.objective_class_args(args, objective_class)
                class_rows = [
                    row for row in oof_rows
                    if base.plan_objective_class(row) == objective_class
                ]
                objective_policy, summary = calibrate_oof_policy(
                    class_rows, class_args
                )
                objective_policy.update({
                    "objective_class": objective_class,
                    "risk_contract": (
                        "top2_membership_invariant_r5_completion"
                        if objective_class == "top5_completion"
                        else "strict_top2_and_top5_non_regression"
                    ),
                })
                class_policies[objective_class] = objective_policy
                class_reports[objective_class] = summary
            selected = selected_with_policies(oof_rows, class_policies)
    pooled_report = base.report(selected, len(query_keys))
    fold_certificates: Dict[str, Dict[str, Any]] = {}
    folds_safe = True
    for fold in range(fold_count):
        fold_queries = {
            key for key, assigned_fold in fold_by_query.items() if assigned_fold == fold
        }
        fold_selected = [row for row in selected if row["query_key"] in fold_queries]
        summary = base.report(fold_selected, len(fold_queries))
        fold_safe = bool(
            summary["selected_count"] == 0
            or (
                summary["selected_loss_rate"] <= args.max_holdout_loss_rate
                and summary["delta_r2_sum"] >= -1e-12
                and summary["delta_r5_sum"] >= -1e-12
            )
        )
        fold_certificates[str(fold)] = {"passed": fold_safe, **summary}
        folds_safe = folds_safe and fold_safe

    passed = bool(
        (
            global_policy.get("calibration_feasible", False)
            or any(
                value.get("calibration_feasible", False)
                for value in class_policies.values()
            )
        )
        and pooled_report["selected_count"] >= args.min_calibration_selected
        and pooled_report["selected_loss_rate"] <= args.max_holdout_loss_rate
        and pooled_report["delta_r2_sum"] >= -1e-12
        and pooled_report["delta_r5_sum"] >= -1e-12
        and (pooled_report["delta_r2_sum"] > 1e-12 or pooled_report["delta_r5_sum"] > 1e-12)
        and folds_safe
    )
    print(
        f"[grouped_oof-cbo] oof_selected={pooled_report['selected_count']} "
        f"loss_rate={pooled_report['selected_loss_rate']:.6f} "
        f"delta_r2={pooled_report['delta_r2_per_query']:.6f} "
        f"delta_r5={pooled_report['delta_r5_per_query']:.6f} "
        f"certificate_passed={passed}",
        flush=True,
    )

    print("[grouped_oof-cbo] training final deployment model", flush=True)
    final_payloads, final_trainingted = base.train_models(rows, rows, args)
    if args.hierarchical_objective_execution:
        stage2_fit_rows = (
            sample_stage2_training_rows(
                stage2_rows,
                args.top5_hard_negatives_per_query,
                group_across_families=bool(args.stage2_slate_dedup),
            )
            if args.query_grouped_hard_negative_top5
            else stage2_rows
        )
        stage2_payloads, stage2_fitted = base.train_models(
            stage2_fit_rows, stage2_fit_rows, args
        )
        top5_keys = {
            key for key in stage2_payloads
            if model_key_objective_family(key)[0] == "top5_completion"
        }
        if not top5_keys:
            raise RuntimeError(
                "hierarchical deployment requires at least one Top-5 expert"
            )
        for model_key in list(final_payloads):
            if model_key_objective_family(model_key)[0] == "top5_completion":
                final_payloads.pop(model_key, None)
                final_trainingted.pop(model_key, None)
        for top5_key in top5_keys:
            final_payloads[top5_key] = stage2_payloads[top5_key]
            final_trainingted[top5_key] = stage2_fitted[top5_key]
        top2_keys = {
            key for key in final_payloads
            if model_key_objective_family(key)[0] == "top2_ordering"
        }
        if not top2_keys:
            raise RuntimeError(
                "hierarchical deployment requires at least one Top-2 expert"
            )
        for model_key, payload_value in final_payloads.items():
            objective_class, plan_name = model_key_objective_family(model_key)
            if objective_class not in residuals:
                continue
            class_residuals = residuals[objective_class]
            if args.outcome_model_scope == "objective_family":
                class_residuals = class_residuals.get(plan_name, {})
            if not class_residuals:
                continue
            payload_value["delta_r2_residual_lcb"] = class_residuals["delta_r2"]
            payload_value["delta_r5_residual_lcb"] = class_residuals["delta_r5"]
            if "quality_r2" in class_residuals:
                payload_value["quality_r2_residual_lcb"] = class_residuals[
                    "quality_r2"
                ]
            if "quality_r5" in class_residuals:
                payload_value["quality_r5_residual_lcb"] = class_residuals[
                    "quality_r5"
                ]
            if "regret_r2" in class_residuals:
                payload_value["regret_r2_residual_ucb"] = class_residuals[
                    "regret_r2"
                ]
            if "regret_r5" in class_residuals:
                payload_value["regret_r5_residual_ucb"] = class_residuals[
                    "regret_r5"
                ]
            payload_value["residual_source"] = (
                "entity_disjoint_oof_frozen_top2_residual_predictions"
                if objective_class == "top5_completion"
                else "entity_disjoint_out_of_fold_predictions"
            )
        del stage2_fitted
    else:
        for payload_value in final_payloads.values():
            payload_value["delta_r2_residual_lcb"] = residuals["delta_r2"]
            payload_value["delta_r5_residual_lcb"] = residuals["delta_r5"]
            if "quality_r2" in residuals:
                payload_value["quality_r2_residual_lcb"] = residuals["quality_r2"]
            if "quality_r5" in residuals:
                payload_value["quality_r5_residual_lcb"] = residuals["quality_r5"]
            if "regret_r2" in residuals:
                payload_value["regret_r2_residual_ucb"] = residuals["regret_r2"]
            if "regret_r5" in residuals:
                payload_value["regret_r5_residual_ucb"] = residuals["regret_r5"]
            payload_value["residual_source"] = (
                "entity_disjoint_out_of_fold_predictions"
            )
    score_alignments: Dict[str, Dict[str, Any]] = {}
    full_rows = [
        *[
            row for row in rows
            if base.plan_objective_class(row) == "top2_ordering"
        ],
        *(stage2_rows if args.hierarchical_objective_execution else [
            row for row in rows
            if base.plan_objective_class(row) == "top5_completion"
        ]),
    ]
    if args.grouped_oof_score_alignment or args.action_table_dir:
        full_predicted = base.attach_predictions(
            full_rows, final_payloads, final_trainingted
        )
        if args.grouped_oof_score_alignment:
            for objective_class in ("top2_ordering", "top5_completion"):
                score_alignments[objective_class] = (
                    fit_objective_family_score_alignments(
                        oof_rows, full_predicted, objective_class
                    )
                )
            alignment_pairs = sum(
                int(value.get("pair_count", 0))
                for families in score_alignments.values()
                for value in families.values()
            )
            print(
                f"[grouped_oof-cbo] paired OOF score alignment "
                f"families={sum(len(value) for value in score_alignments.values())} "
                f"pairs={alignment_pairs}",
                flush=True,
            )
            if args.structure_conditioned_outcome_calibration:
                print(
                    "[grouped_oof-cbo] structure-conditioned OOF outcome "
                    f"calibration groups={len(outcome_calibration['groups'])}",
                    flush=True,
                )
        if args.action_table_dir:
            base.write_action_table(args.action_table_dir, "grouped_oof_oof", oof_rows)
            base.write_action_table(
                args.action_table_dir, "full_historical_fit", full_predicted
            )
        del full_predicted
    del full_rows
    del final_trainingted
    gc.collect()
    policy = {
        **global_policy,
        "selection_mode": args.selection_mode,
        "calibration_feasible": bool(
            global_policy.get("calibration_feasible", False)
            or any(
                value.get("calibration_feasible", False)
                for value in class_policies.values()
            )
        ),
        "objective_class_policies": class_policies,
        "hierarchical_objective_execution": bool(
            args.hierarchical_objective_execution
        ),
        "allowed_plans": sorted(selected_plans),
        "topology_plan_families": topology_plan_families,
        "topology_plan_routing_profile": args.topology_plan_routing_profile,
        "query_grouped_shortlist_per_family": max(int(os.environ.get(
            "QUERY_PLAN_ONTOLOGY_COMPLETE_PLAN_SHORTLIST_PER_FAMILY", "0"
        )), 0),
        "stage2_slate_dedup": bool(args.stage2_slate_dedup),
        "stage2_proof_connected_primary": bool(
            args.stage2_proof_connected_primary
        ),
    }
    payload = {
        "type": base.MODEL_TYPE,
        "version": (
            315
            if args.structure_conditioned_outcome_calibration
            else 295
            if args.stage2_proof_connected_primary
            else 294
            if args.stage2_slate_dedup
            and args.outcome_model_scope == "objective_class"
            else 293
            if args.topology_plan_routing_profile
            == base.TOPOLOGY_ROUTING_STRUCTURAL_RECOVERY_V293
            else 292
            if args.selection_mode == "query_relative_slate_regret"
            else 291
            if args.selection_mode == "absolute_quality_regret"
            else 289
            if args.topology_plan_routing_profile
            == base.TOPOLOGY_ROUTING_MULTI_ACCESS_V289
            else 288
            if args.grouped_oof_score_alignment
            else 287
            if args.topology_plan_routing_profile
            == base.TOPOLOGY_ROUTING_OBJECTIVE_FAMILY_V287
            else 286
            if args.topology_plan_routing_profile
            == base.TOPOLOGY_ROUTING_HIERARCHICAL_V286
            else 285
            if args.topology_plan_routing_profile
            == base.TOPOLOGY_ROUTING_COMBINED_V285
            else 284
            if args.topology_plan_routing_profile
            == base.TOPOLOGY_ROUTING_WAREHOUSE_V284
            else 283
            if args.topology_plan_routing_profile
            == base.TOPOLOGY_ROUTING_BOUNDED_V283
            else 282
            if args.topology_plan_routing_profile
            == base.TOPOLOGY_ROUTING_BOUNDED_V282
            else 281
            if args.topology_plan_routing_profile
            == base.TOPOLOGY_ROUTING_RECALL_SAFE_V281
            else 280
            if args.topology_partitioned_plan_routing
            else 277
        ),
        "feature_names": list(base.FEATURE_NAMES),
        "dataset_identity_used_for_routing": False,
        "runtime_uses_gold": False,
        "calibration_validation_merged": True,
        "split_protocol": {
            "q_h": "all leakage-free historical queries",
            "grouped_oof": f"{fold_count}-fold support-entity-disjoint out-of-fold calibration",
            "q_test": "final evaluation only",
            "q_cal": "removed",
            "q_val": "removed",
        },
        "training_label_source": "historical_counterfactual_complete_slate_replay",
        "training_label_semantics": "official_exact_document_recall",
        "baseline_endpoint": "final_ranking_incumbent",
        "incumbent_contract": args.incumbent_contract,
        "physical_plan_catalog_rematerialized": True,
        "training_targets": list(base.training_targets(args)),
        "target_semantics": (
            "query_relative_complete_slate_regret"
            if args.query_relative_slate_regret_targets
            else "absolute_plan_quality_and_query_local_oracle_regret"
            if args.absolute_quality_regret_targets
            else "incumbent_relative_recall_delta"
        ),
        "online_incumbent_relative_gate": bool(
            args.selection_mode not in {
                "absolute_quality_regret", "query_relative_slate_regret"
            }
        ),
        "risk_label_semantics": (
            "graded_recall_regret_against_best_query_local_complete_slate"
            if args.query_relative_slate_regret_targets
            else "probability_of_regret_against_best_query_local_feasible_slate"
            if args.absolute_quality_regret_targets
            else "probability_of_recall_loss_against_incumbent"
        ),
        "gain_label_semantics": "pareto_safe_recall_gain",
        "outcome_class_balancing": bool(args.balance_outcome_targets),
        "query_balanced_sample_weights": bool(
            args.query_balanced_sample_weights
        ),
        "outcome_model_scope": (
            "rank_partitioned_objective_family_experts"
            if args.outcome_model_scope == "objective_family"
            else "rank_partitioned_objective_class_experts"
            if args.outcome_model_scope == "objective_class"
            else "query_contract_structural_experts"
            if args.outcome_model_scope == "structural"
            else "global_plan_comparable"
            if args.outcome_model_scope == "global"
            else "per_plan_family"
        ),
        "objective": "grouped_oof_calibrated_top2_ordering_and_top5_completion",
        "stage2_training_protocol": (
            "query_grouped_cross_family_deduplicated_listwise_actions"
            if args.stage2_slate_dedup
            else "query_grouped_sparse_outcomes_plus_bounded_oracle_and_hard_neutral_actions"
            if args.query_grouped_hard_negative_top5
            else "all_residual_actions"
        ),
        "execution_protocol": (
            "freeze_oof_selected_top2_then_optimize_residual_top5"
            if args.hierarchical_objective_execution
            else "single_complete_slate_selection"
        ),
        "allowed_plans": sorted(selected_plans),
        "semantic_fact_access_role": (
            "first_class_zero_modification_physical_plan"
            if args.query_relative_slate_regret_targets
            else "in_plan_scan_operator" if base.PLAN_SIGNATURE_GROUNDED in selected_plans
            else "independent_incumbent"
        ),
        "final_slate_owner": "complete_evidence_plan_optimizer",
        "query_structure_signature_enabled": bool(
            base.PLAN_SIGNATURE_GROUNDED in selected_plans
        ),
        "characteristic_routing": (
            "compiled_query_topology_then_rank_objective"
            if args.topology_partitioned_plan_routing
            else "dataset_agnostic_query_contract_features"
        ),
        "topology_plan_families": topology_plan_families,
        "topology_plan_routing_profile": args.topology_plan_routing_profile,
        "plans": final_payloads,
        "policy": policy,
        "grouped_oof_score_alignment": {
            "enabled": bool(args.grouped_oof_score_alignment),
            "method": (
                "paired_final_to_outcome_calibrated_oof_isotonic_v2"
                if args.structure_conditioned_outcome_calibration
                else "paired_oof_isotonic_v1"
                if args.grouped_oof_score_alignment else "disabled"
            ),
            "alignments": score_alignments,
        },
        "grouped_oof_outcome_calibration": outcome_calibration,
        "calibration_report": pooled_report,
        "objective_class_calibration_reports": class_reports,
        "holdout_certificate": {
            "passed": passed,
            "certificate_type": "entity_disjoint_kfold_out_of_fold",
            "max_loss_rate": args.max_holdout_loss_rate,
            "fold_certificates": fold_certificates,
            **pooled_report,
        },
        "grouped_oof": {
            "fold_count": fold_count,
            "fold_models": fold_model_reports,
            "checkpoint_schema": CHECKPOINT_SCHEMA,
            "checkpoint_resumed_folds": resumed_folds,
            "stage2_checkpoint_resumed_folds": stage2_resumed_folds,
            "oof_residual_quantiles": residuals,
            "oof_action_count": len(oof_rows),
            "historical_query_count": len(query_keys),
            "stage2_full_action_count": len(stage2_rows),
            "stage2_slate_dedup": stage2_dedup_report,
            "stage2_fit_action_count": len(
                sample_stage2_training_rows(
                    stage2_rows,
                    args.top5_hard_negatives_per_query,
                    group_across_families=bool(args.stage2_slate_dedup),
                )
                if args.query_grouped_hard_negative_top5
                else stage2_rows
            ),
            "top5_hard_negatives_per_query": int(
                args.top5_hard_negatives_per_query
            ),
        },
        "split_counts": {
            "historical_queries": len(query_keys),
            "historical_rows": len(rows),
            "oof_rows": len(oof_rows),
        },
    }
    output = Path(args.output_model)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "model": str(output),
        "grouped_oof_certificate_passed": passed,
        "oof_report": pooled_report,
        "fold_certificates": fold_certificates,
    }, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
