#!/usr/bin/env python3
"""Train and holdout-certify a gold-free selector over complete evidence slates."""

from __future__ import annotations

import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from compassrag.query_plan_complete_evidence_plan_optimizer import (
    ALTERNATIVE_PLANS,
    FEATURE_NAMES,
    OBJECTIVE_CLASS_EXPERT_PREFIX,
    OBJECTIVE_FAMILY_EXPERT_PREFIX,
    STRUCTURAL_EXPERT_PREFIX,
    TOPOLOGY_ROUTING_BOUNDED_V282,
    TOPOLOGY_ROUTING_BOUNDED_V283,
    TOPOLOGY_ROUTING_WAREHOUSE_V284,
    TOPOLOGY_ROUTING_COMBINED_V285,
    TOPOLOGY_ROUTING_HIERARCHICAL_V286,
    TOPOLOGY_ROUTING_MULTI_ACCESS_V289,
    TOPOLOGY_ROUTING_OBJECTIVE_FAMILY_V287,
    TOPOLOGY_ROUTING_RECALL_SAFE_V281,
    TOPOLOGY_ROUTING_STRUCTURAL_RECOVERY_V293,
    TOPOLOGY_ROUTING_STRICT_V280,
    MODEL_TYPE,
    PLAN_SEMANTIC,
    PLAN_SEMANTIC_TOP2_NOOP,
    PLAN_SIGNATURE_GROUNDED,
    compose_hierarchical_topk,
    materialize_complete_plan_portfolio,
    objective_family_model_key,
    plan_allowed_for_topology,
    query_contract_regime,
    query_plan_topology,
    topology_partitioned_plan_families,
)


TARGETS = (
    "delta_r2",
    "delta_r5",
    "gain_probability",
    "loss_probability",
)
ABSOLUTE_QUALITY_TARGETS = (
    "quality_r2",
    "quality_r5",
    "optimal_r2_probability",
    "optimal_r5_probability",
)
QUERY_RELATIVE_REGRET_TARGETS = (
    "regret_r2",
    "regret_r5",
)
GLOBAL_MODEL_KEY = "__global_plan_outcome__"


def training_targets(args: argparse.Namespace) -> Tuple[str, ...]:
    if bool(getattr(args, "query_relative_slate_regret_targets", False)):
        return (*TARGETS, *ABSOLUTE_QUALITY_TARGETS, *QUERY_RELATIVE_REGRET_TARGETS)
    if bool(getattr(args, "absolute_quality_regret_targets", False)):
        return (*TARGETS, *ABSOLUTE_QUALITY_TARGETS)
    return TARGETS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--induction-trace-dir", required=True)
    parser.add_argument("--calibration-trace-dir", required=True)
    parser.add_argument("--holdout-trace-dir", required=True)
    parser.add_argument("--output-model", required=True)
    parser.add_argument(
        "--action-table-dir",
        default="",
        help="Optional directory for compact, reusable replay-action tables.",
    )
    parser.add_argument("--datasets", default="2wikimultihopqa,hotpotqa,musique")
    parser.add_argument(
        "--rematerialize-plan-catalog",
        action="store_true",
        help=(
            "Re-enumerate physical plans from frozen traces and recompute exact "
            "counterfactual labels instead of consuming the trace's old catalog."
        ),
    )
    parser.add_argument(
        "--workload-dir",
        default="",
        help="Historical workload containing q_ind/q_cal/q_val and corpora.",
    )
    parser.add_argument(
        "--allowed-plans",
        default=",".join(ALTERNATIVE_PLANS),
        help="Comma-separated physical plans eligible for training and runtime selection.",
    )
    parser.add_argument("--trees", type=int, default=360)
    parser.add_argument("--max-depth", type=int, default=9)
    parser.add_argument("--min-leaf", type=int, default=8)
    parser.add_argument("--max-features", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=252)
    parser.add_argument("--jobs", type=int, default=4)
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
        default="pareto_lcb",
        help=(
            "pareto_lcb preserves the default per-metric lower-bound gates; "
            "risk_budgeted_lexicographic calibrates an aggregate loss budget "
            "and ranks admissible actions by expected R@2, then expected R@5; "
            "safe_gain_listwise classifies Pareto-safe gains and selects the "
            "best admitted plan within each query's physical-plan slate."
        ),
    )
    parser.add_argument(
        "--absolute-quality-regret-targets",
        action="store_true",
        help=(
            "Train absolute Recall@2/Recall@5 quality and query-local oracle "
            "optimality heads. This removes incumbent-relative delta gates "
            "from online admission while retaining OOF safety certification."
        ),
    )
    parser.add_argument(
        "--query-relative-slate-regret-targets",
        action="store_true",
        help=(
            "Train query-local Recall@2/Recall@5 regret heads over complete "
            "Top-k slates, including semantic fact retrieval as a learned "
            "zero-modification physical plan."
        ),
    )
    parser.add_argument(
        "--pareto-safe-gain-label",
        action="store_true",
        help=(
            "Label a gain only when at least one recall metric improves and "
            "neither R@2 nor R@5 regresses."
        ),
    )
    parser.add_argument(
        "--balance-outcome-targets",
        action="store_true",
        help=(
            "Use inverse-frequency sample weights for the sparse gain/loss "
            "heads. Thresholds remain calibrated on the untouched Q_cal split."
        ),
    )
    parser.add_argument(
        "--query-balanced-sample-weights",
        action="store_true",
        help=(
            "Normalize training weight per historical query so queries that "
            "materialize many plan instances cannot dominate the outcome model."
        ),
    )
    parser.add_argument(
        "--global-outcome-model",
        action="store_true",
        help=(
            "Train one shared outcome model across all physical plans. Plan-family "
            "one-hot features keep the alternatives identifiable while making "
            "their listwise scores directly comparable."
        ),
    )
    parser.add_argument(
        "--structural-expert-model",
        action="store_true",
        help=(
            "Train one plan-comparable outcome expert per dataset-agnostic query "
            "contract regime (comparison, nested bridge, two-hop, or long chain)."
        ),
    )
    parser.add_argument("--max-calibration-loss-rate", type=float, default=0.08)
    parser.add_argument("--max-holdout-loss-rate", type=float, default=0.10)
    parser.add_argument("--max-top5-calibration-loss-rate", type=float, default=0.08)
    parser.add_argument("--max-top5-holdout-loss-rate", type=float, default=0.10)
    parser.add_argument(
        "--max-admission-loss-probability",
        type=float,
        default=-1.0,
        help=(
            "Largest predicted loss-probability threshold considered during "
            "OOF calibration. A negative value preserves the default behavior "
            "of tying this search bound to the empirical loss-rate budget."
        ),
    )
    parser.add_argument(
        "--max-top5-admission-loss-probability",
        type=float,
        default=-1.0,
        help=(
            "Top-5-specific predicted-risk search bound. The selected policy "
            "must still satisfy the empirical calibration and holdout loss "
            "budgets."
        ),
    )
    parser.add_argument(
        "--min-delta-r2-mean",
        type=float,
        default=0.0,
        help="Minimum predicted mean change in exact document Recall@2.",
    )
    parser.add_argument(
        "--min-delta-r5-mean",
        type=float,
        default=0.0,
        help="Minimum predicted mean change in exact document Recall@5.",
    )
    parser.add_argument("--min-calibration-selected", type=int, default=8)
    parser.add_argument("--min-holdout-selected", type=int, default=3)
    parser.add_argument("--min-characteristic-selected", type=int, default=3)
    parser.add_argument("--max-characteristic-loss-rate", type=float, default=0.15)
    parser.add_argument(
        "--min-domain-selected",
        type=int,
        default=0,
        help=(
            "Offline-only robustness constraint: require at least this many "
            "selected calibration/holdout actions in every historical domain."
        ),
    )
    parser.add_argument(
        "--min-domain-utility-per-action",
        type=float,
        default=0.0,
        help=(
            "Minimum offline calibration utility margin, 2*delta_R2+delta_R5, "
            "per selected action in every historical domain."
        ),
    )
    parser.add_argument(
        "--robust-calibration-objective",
        action="store_true",
        help=(
            "Prefer the policy with the strongest worst-domain utility before "
            "maximizing aggregate utility. Dataset identity remains absent "
            "from runtime features and routing."
        ),
    )
    return parser.parse_args()


def allowed_plans(args: argparse.Namespace) -> Tuple[str, ...]:
    configured = str(getattr(args, "allowed_plans", "") or "")
    values = tuple(
        value.strip() for value in configured.split(",") if value.strip()
    )
    valid_plans = set(ALTERNATIVE_PLANS)
    if bool(getattr(args, "query_relative_slate_regret_targets", False)):
        valid_plans.add(PLAN_SEMANTIC)
    invalid = sorted(set(values) - valid_plans)
    if invalid:
        raise ValueError(f"unknown physical plans: {invalid}")
    return values or tuple(ALTERNATIVE_PLANS)


def iter_records(root: str, datasets: Sequence[str]) -> Iterable[Tuple[str, Dict[str, Any]]]:
    directory = Path(root)
    for dataset in datasets:
        paths = [directory / f"{dataset}.jsonl"]
        if not paths[0].is_file():
            paths = list(directory.rglob(f"{dataset}.jsonl"))
        for path in paths:
            if not path.is_file():
                continue
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    value = json.loads(line)
                    if isinstance(value, dict):
                        yield dataset, value


def historical_gold_ids(
    workload_dir: str,
    split: str,
    datasets: Sequence[str],
) -> Dict[str, Dict[int, set[int]]]:
    root = Path(workload_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"missing historical workload: {root}")
    result: Dict[str, Dict[int, set[int]]] = {}
    for dataset in datasets:
        corpus_path = root / "corpora" / f"{dataset}_corpus.json"
        query_path = root / split / f"{dataset}.json"
        corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
        queries = json.loads(query_path.read_text(encoding="utf-8"))
        title_to_ids: Dict[str, List[int]] = defaultdict(list)
        for passage_idx, passage in enumerate(corpus):
            if isinstance(passage, Mapping):
                title_to_ids[str(passage.get("title", "") or "")].append(
                    passage_idx
                )
        dataset_gold: Dict[int, set[int]] = {}
        for query_idx, query in enumerate(queries):
            titles = set()
            if isinstance(query, Mapping):
                titles.update(
                    str(value[0])
                    for value in query.get("supporting_facts", []) or []
                    if isinstance(value, (list, tuple)) and value
                )
                titles.update(
                    str(value.get("title", "") or "")
                    for value in query.get("paragraphs", []) or []
                    if isinstance(value, Mapping) and bool(value.get("is_supporting"))
                )
                titles.update(
                    str(value)
                    for value in query.get("gold_titles", []) or []
                )
            gold_ids = {
                passage_idx
                for title in titles
                for passage_idx in title_to_ids.get(title, [])
            }
            dataset_gold[query_idx] = gold_ids
        result[dataset] = dataset_gold
    return result


def rows_from_traces(
    root: str,
    datasets: Sequence[str],
    query_keys: set[str] | None = None,
    rematerialized_gold: Mapping[str, Mapping[int, set[int]]] | None = None,
    pareto_safe_gain_label: bool = False,
    allowed_plan_names: set[str] | None = None,
    query_relative_slate_regret: bool = False,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for dataset, record in iter_records(root, datasets):
        query_key = f"{dataset}:{int(record.get('query_idx', -1))}"
        if query_keys is not None:
            query_keys.add(query_key)
        portfolio = record.get("complete_evidence_plan_optimizer", {}) or {}
        labels = record.get("complete_evidence_plan_training", {}) or {}
        if rematerialized_gold is not None:
            incumbent = record.get("final_ranking_incumbent", {}) or {}
            incumbent_ids = list(
                incumbent.get("candidate_window_doc_ids", [])
                or incumbent.get("doc_ids", [])
            )
            incumbent_scores = list(
                incumbent.get("candidate_window_doc_scores", [])
                or incumbent.get("doc_scores", [])
            )
            portfolio = materialize_complete_plan_portfolio(
                record,
                incumbent_ids,
                incumbent_scores,
                top_k=5,
                allowed_plan_families=allowed_plan_names,
                query_relative_semantic_competitor=query_relative_slate_regret,
            )
            gold_ids = set(
                (rematerialized_gold.get(dataset, {}) or {}).get(
                    int(record.get("query_idx", -1)), set()
                )
            )
            denominator = max(len(gold_ids), 1)
            labels = {
                "labels_valid": bool(gold_ids),
                "label_semantics": "official_exact_document_recall",
                "baseline_endpoint": "final_ranking_incumbent",
                "plans": {},
            }
            for plan_instance, spec in (portfolio.get("plans", {}) or {}).items():
                doc_ids = [int(value) for value in spec.get("doc_ids", []) or []]
                labels["plans"][plan_instance] = {
                    "doc_ids": doc_ids[:5],
                    "recall_at2": len(set(doc_ids[:2]) & gold_ids) / denominator,
                    "recall_at5": len(set(doc_ids[:5]) & gold_ids) / denominator,
                }
        if not labels.get("labels_valid", False):
            continue
        if (
            labels.get("label_semantics")
            != "official_exact_document_recall"
            or labels.get("baseline_endpoint")
            != "final_ranking_incumbent"
            or not portfolio.get("endpoint_contract_matched", False)
        ):
            continue
        plan_labels = labels.get("plans", {}) or {}
        plan_specs = portfolio.get("plans", {}) or {}
        semantic = plan_labels.get(PLAN_SEMANTIC, {}) or {}
        if not semantic:
            continue
        final_incumbent = record.get("final_ranking_incumbent", {}) or {}
        final_top5 = [
            int(value)
            for value in final_incumbent.get("doc_ids", []) or []
        ][:5]
        semantic_top5 = [
            int(value) for value in semantic.get("doc_ids", []) or []
        ][:5]
        if not final_top5 or semantic_top5 != final_top5:
            continue
        baseline_r2 = float(semantic.get("recall_at2", 0.0) or 0.0)
        baseline_r5 = float(semantic.get("recall_at5", 0.0) or 0.0)
        for plan_instance, spec in plan_specs.items():
            if not isinstance(spec, Mapping):
                continue
            plan_name = str(spec.get("plan_family") or plan_instance)
            if plan_name not in ALTERNATIVE_PLANS and not (
                query_relative_slate_regret and plan_name == PLAN_SEMANTIC
            ):
                continue





            if allowed_plan_names is not None and plan_name not in allowed_plan_names:
                continue
            outcome = plan_labels.get(plan_instance, {}) or {}
            if not spec.get("valid", False) or not outcome:
                continue
            quality_r2 = float(outcome.get("recall_at2", 0.0) or 0.0)
            quality_r5 = float(outcome.get("recall_at5", 0.0) or 0.0)
            delta_r2 = quality_r2 - baseline_r2
            delta_r5 = quality_r5 - baseline_r5
            features = spec.get("features", {}) or {}
            has_gain = delta_r2 > 1e-12 or delta_r5 > 1e-12
            has_loss = delta_r2 < -1e-12 or delta_r5 < -1e-12
            rows.append({
                "query_key": query_key,
                "dataset": dataset,
                "plan_name": plan_name,
                "plan_instance": str(plan_instance),
                "features": {
                    name: float(features.get(name, 0.0) or 0.0)
                    for name in FEATURE_NAMES
                },
                "doc_ids": [
                    int(value) for value in outcome.get("doc_ids", []) or []
                ][:5],
                "semantic_doc_ids": list(semantic_top5),
                "gold_doc_ids": sorted(gold_ids) if rematerialized_gold is not None else [],
                "baseline_quality_r2": baseline_r2,
                "baseline_quality_r5": baseline_r5,
                "quality_r2": quality_r2,
                "quality_r5": quality_r5,
                "delta_r2": delta_r2,
                "delta_r5": delta_r5,
                "gain_probability": float(
                    has_gain and (not pareto_safe_gain_label or not has_loss)
                ),
                "loss_probability": float(has_loss),
            })
    annotate_query_oracle_targets(
        rows, complete_slate_competition=query_relative_slate_regret
    )
    return rows


def annotate_query_oracle_targets(
    rows: Sequence[Dict[str, Any]], *, complete_slate_competition: bool = False
) -> None:
    """Attach absolute quality and regret labels within each executable slate.

    The semantic retrieval result participates only as another feasible plan
    when defining the oracle quality. It is not used as the origin of the
    learned target. Grouping by rank objective prevents Top-2 ordering plans
    and membership-preserving Top-5 completion plans from supervising each
    other under incompatible execution contracts.
    """
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        objective_class = (
            "complete_slate"
            if complete_slate_competition
            else plan_objective_class(row)
        )
        grouped[(str(row.get("query_key", "")), objective_class)].append(row)
    for group_rows in grouped.values():
        oracle_r2 = max(
            [float(row.get("baseline_quality_r2", 0.0)) for row in group_rows]
            + [float(row.get("quality_r2", 0.0)) for row in group_rows]
        )
        oracle_r5 = max(
            [float(row.get("baseline_quality_r5", 0.0)) for row in group_rows]
            + [float(row.get("quality_r5", 0.0)) for row in group_rows]
        )
        for row in group_rows:
            quality_r2 = float(row.get("quality_r2", 0.0))
            quality_r5 = float(row.get("quality_r5", 0.0))
            regret_r2 = max(oracle_r2 - quality_r2, 0.0)
            regret_r5 = max(oracle_r5 - quality_r5, 0.0)
            row.update({
                "oracle_quality_r2": oracle_r2,
                "oracle_quality_r5": oracle_r5,
                "regret_r2": regret_r2,
                "regret_r5": regret_r5,
                "optimal_r2_probability": float(regret_r2 <= 1e-12),
                "optimal_r5_probability": float(regret_r5 <= 1e-12),
            })


def matrix(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    return np.asarray([
        [float((row.get("features", {}) or {}).get(name, 0.0)) for name in FEATURE_NAMES]
        for row in rows
    ], dtype=np.float64)


def export_forest(model: RandomForestRegressor) -> Dict[str, Any]:
    trees = []
    for estimator in model.estimators_:
        tree = estimator.tree_
        trees.append({
            "children_left": tree.children_left.astype(int).tolist(),
            "children_right": tree.children_right.astype(int).tolist(),
            "feature": tree.feature.astype(int).tolist(),
            "threshold": tree.threshold.astype(float).tolist(),
            "value": tree.value[:, 0, 0].astype(float).tolist(),
        })
    return {
        "type": "json_random_forest_regressor",
        "feature_names": list(FEATURE_NAMES),
        "trees": trees,
        "fallback_value": float(np.mean([
            estimator.tree_.value[0, 0, 0] for estimator in model.estimators_
        ])),
    }


def train_models(
    induction: Sequence[Dict[str, Any]],
    calibration: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, RandomForestRegressor]]]:
    export_model_payload = not bool(getattr(args, "skip_model_export", False))
    calibrate_residuals = not bool(
        getattr(args, "skip_residual_calibration", False)
    )
    payload: Dict[str, Any] = {}
    models: Dict[str, Dict[str, RandomForestRegressor]] = {}
    if bool(getattr(args, "structural_expert_model", False)):
        regimes = sorted({
            query_contract_regime(row.get("features", {}) or {})
            for row in [*induction, *calibration]
        })
        partitions = [
            (
                STRUCTURAL_EXPERT_PREFIX + regime,
                [
                    row for row in induction
                    if query_contract_regime(row.get("features", {}) or {}) == regime
                ],
                [
                    row for row in calibration
                    if query_contract_regime(row.get("features", {}) or {}) == regime
                ],
            )
            for regime in regimes
        ]
    elif bool(getattr(args, "objective_family_outcome_model", False)):
        objective_families = sorted({
            (plan_objective_class(row), str(row.get("plan_name", "")))
            for row in [*induction, *calibration]
            if str(row.get("plan_name", ""))
        })
        partitions = [
            (
                objective_family_model_key(objective_class, plan_name),
                [
                    row for row in induction
                    if plan_objective_class(row) == objective_class
                    and str(row.get("plan_name", "")) == plan_name
                ],
                [
                    row for row in calibration
                    if plan_objective_class(row) == objective_class
                    and str(row.get("plan_name", "")) == plan_name
                ],
            )
            for objective_class, plan_name in objective_families
        ]
    elif bool(getattr(args, "objective_class_outcome_model", False)):
        partitions = [
            (
                OBJECTIVE_CLASS_EXPERT_PREFIX + objective_class,
                [
                    row for row in induction
                    if plan_objective_class(row) == objective_class
                ],
                [
                    row for row in calibration
                    if plan_objective_class(row) == objective_class
                ],
            )
            for objective_class in ("top2_ordering", "top5_completion")
        ]
    elif bool(getattr(args, "global_outcome_model", False)):
        partitions = [(GLOBAL_MODEL_KEY, list(induction), list(calibration))]
    else:
        partitions = [
            (
                plan_name,
                [row for row in induction if row["plan_name"] == plan_name],
                [row for row in calibration if row["plan_name"] == plan_name],
            )
            for plan_name in allowed_plans(args)
        ]
    for offset, (plan_name, train_rows, cal_rows) in enumerate(partitions):
        if not train_rows or not cal_rows:
            continue
        train_x = matrix(train_rows)
        cal_x = matrix(cal_rows) if calibrate_residuals else None
        plan_payload: Dict[str, Any] = {
            "induction_count": len(train_rows),
            "calibration_count": len(cal_rows),
            "covered_plan_families": sorted({
                str(row.get("plan_name", "")) for row in train_rows
            }),
        }
        models[plan_name] = {}
        active_targets = training_targets(args)
        for target_index, target in enumerate(active_targets):
            train_y = np.asarray([
                float(
                    row.get(
                        target,
                        float(
                            target == "gain_probability"
                            and (
                                float(row.get("delta_r2", 0.0)) > 1e-12
                                or float(row.get("delta_r5", 0.0)) > 1e-12
                            )
                        ),
                    )
                )
                for row in train_rows
            ])
            model = RandomForestRegressor(
                n_estimators=args.trees,
                max_depth=args.max_depth,
                min_samples_leaf=args.min_leaf,
                max_features=args.max_features,
                bootstrap=True,
                random_state=args.seed + 17 * offset + target_index,
                n_jobs=args.jobs,
            )
            sample_weight = None
            if bool(getattr(args, "query_balanced_sample_weights", False)):
                query_counts: Dict[str, int] = defaultdict(int)
                for row in train_rows:
                    query_counts[str(row.get("query_key", ""))] += 1
                sample_weight = np.asarray([
                    1.0 / max(query_counts[str(row.get("query_key", ""))], 1)
                    for row in train_rows
                ], dtype=np.float64)
                sample_weight /= max(float(np.mean(sample_weight)), 1e-12)
                plan_payload["query_balanced_sample_weights"] = True
            if (
                bool(getattr(args, "balance_outcome_targets", False))
                and target in {
                    "gain_probability",
                    "loss_probability",
                    "optimal_r2_probability",
                    "optimal_r5_probability",
                }
            ):
                positive = train_y > 0.5
                positive_count = int(np.sum(positive))
                negative_count = int(len(train_y) - positive_count)
                if positive_count and negative_count:
                    class_weight = np.where(
                        positive,
                        len(train_y) / (2.0 * positive_count),
                        len(train_y) / (2.0 * negative_count),
                    )
                    sample_weight = (
                        class_weight
                        if sample_weight is None
                        else sample_weight * class_weight
                    )
                    plan_payload[f"{target}_class_balance"] = {
                        "positive_count": positive_count,
                        "negative_count": negative_count,
                    }
            model.fit(train_x, train_y, sample_weight=sample_weight)
            models[plan_name][target] = model
            if export_model_payload:
                plan_payload[f"{target}_head"] = export_forest(model)
            if calibrate_residuals and target in {
                "delta_r2", "delta_r5", "quality_r2", "quality_r5",
                "regret_r2", "regret_r5",
            }:
                cal_y = np.asarray([float(row[target]) for row in cal_rows])
                assert cal_x is not None
                residual = cal_y - model.predict(cal_x)
                suffix = "residual_ucb" if target.startswith("regret_") else "residual_lcb"
                quantile = 1.0 - args.alpha if target.startswith("regret_") else args.alpha
                plan_payload[f"{target}_{suffix}"] = float(
                    np.quantile(residual, quantile)
                )
        payload[plan_name] = plan_payload
    if not payload:
        raise RuntimeError("no physical plan has rows in both induction and calibration")
    return payload, models


def predict_row(
    row: Mapping[str, Any],
    plan_payload: Mapping[str, Any],
    models: Mapping[str, RandomForestRegressor],
) -> Dict[str, float]:
    values = matrix([row])
    r2 = float(models["delta_r2"].predict(values)[0])
    r5 = float(models["delta_r5"].predict(values)[0])
    loss = float(models["loss_probability"].predict(values)[0])
    gain_model = models.get("gain_probability")
    gain = (
        float(gain_model.predict(values)[0])
        if gain_model is not None
        else float(r2 > 1e-12 or r5 > 1e-12)
    )
    r2_lcb = r2 + float(plan_payload.get("delta_r2_residual_lcb", 0.0))
    r5_lcb = r5 + float(plan_payload.get("delta_r5_residual_lcb", 0.0))
    result = {
        "delta_r2": r2,
        "delta_r5": r5,
        "loss_probability": loss,
        "gain_probability": gain,
        "delta_r2_lcb": r2_lcb,
        "delta_r5_lcb": r5_lcb,
        "utility_mean": 2.0 * r2 + r5,
        "utility_lcb": 2.0 * r2_lcb + r5_lcb,
        "safety_margin": gain - loss,
    }
    if "quality_r2" in models and "quality_r5" in models:
        quality_r2 = float(models["quality_r2"].predict(values)[0])
        quality_r5 = float(models["quality_r5"].predict(values)[0])
        optimal_r2 = float(models["optimal_r2_probability"].predict(values)[0])
        optimal_r5 = float(models["optimal_r5_probability"].predict(values)[0])
        result.update({
            "quality_r2": quality_r2,
            "quality_r5": quality_r5,
            "quality_r2_lcb": quality_r2 + float(
                plan_payload.get("quality_r2_residual_lcb", 0.0)
            ),
            "quality_r5_lcb": quality_r5 + float(
                plan_payload.get("quality_r5_residual_lcb", 0.0)
            ),
            "optimal_r2_probability": optimal_r2,
            "optimal_r5_probability": optimal_r5,
            "regret_r2_probability": 1.0 - optimal_r2,
            "regret_r5_probability": 1.0 - optimal_r5,
        })
    if "regret_r2" in models and "regret_r5" in models:
        regret_r2 = max(0.0, float(models["regret_r2"].predict(values)[0]))
        regret_r5 = max(0.0, float(models["regret_r5"].predict(values)[0]))
        result.update({
            "regret_r2": regret_r2,
            "regret_r5": regret_r5,
            "regret_r2_ucb": max(0.0, regret_r2 + float(
                plan_payload.get("regret_r2_residual_ucb", 0.0)
            )),
            "regret_r5_ucb": max(0.0, regret_r5 + float(
                plan_payload.get("regret_r5_residual_ucb", 0.0)
            )),
        })
    return result


def outcome_model_key(
    row: Mapping[str, Any],
    plan_payloads: Mapping[str, Any],
    models: Mapping[str, Any],
) -> str:
    objective_family_key = objective_family_model_key(
        plan_objective_class(row), str(row.get("plan_name", ""))
    )
    if objective_family_key in plan_payloads and objective_family_key in models:
        return objective_family_key
    structural_key = STRUCTURAL_EXPERT_PREFIX + query_contract_regime(
        row.get("features", {}) or {}
    )
    if structural_key in plan_payloads and structural_key in models:
        return structural_key
    if GLOBAL_MODEL_KEY in plan_payloads and GLOBAL_MODEL_KEY in models:
        return GLOBAL_MODEL_KEY
    objective_key = OBJECTIVE_CLASS_EXPERT_PREFIX + plan_objective_class(row)
    if objective_key in plan_payloads and objective_key in models:
        return objective_key
    return str(row.get("plan_name", ""))


def attach_predictions(
    rows: Sequence[Dict[str, Any]],
    plan_payloads: Mapping[str, Any],
    models: Mapping[str, Mapping[str, RandomForestRegressor]],
) -> List[Dict[str, Any]]:
    """Predict each plan partition once instead of once per policy trial."""
    predicted = [dict(row) for row in rows]
    grouped: Dict[str, List[int]] = defaultdict(list)
    for index, row in enumerate(predicted):
        if isinstance(row.get("prediction"), Mapping):
            continue
        model_key = outcome_model_key(row, plan_payloads, models)
        if model_key in plan_payloads and model_key in models:
            grouped[model_key].append(index)
    for model_key, indices in grouped.items():
        plan_rows = [predicted[index] for index in indices]
        values = matrix(plan_rows)
        models = models[model_key]
        delta_r2 = models["delta_r2"].predict(values)
        delta_r5 = models["delta_r5"].predict(values)
        loss = models["loss_probability"].predict(values)
        gain_model = models.get("gain_probability")
        gain = (
            gain_model.predict(values)
            if gain_model is not None
            else np.asarray([
                float(value2 > 1e-12 or value5 > 1e-12)
                for value2, value5 in zip(delta_r2, delta_r5)
            ])
        )
        plan_payload = plan_payloads[model_key]
        r2_residual = float(plan_payload.get("delta_r2_residual_lcb", 0.0))
        r5_residual = float(plan_payload.get("delta_r5_residual_lcb", 0.0))
        has_absolute_quality = (
            "quality_r2" in models and "quality_r5" in models
        )
        quality_r2 = models["quality_r2"].predict(values) if has_absolute_quality else None
        quality_r5 = models["quality_r5"].predict(values) if has_absolute_quality else None
        optimal_r2 = (
            models["optimal_r2_probability"].predict(values)
            if has_absolute_quality else None
        )
        optimal_r5 = (
            models["optimal_r5_probability"].predict(values)
            if has_absolute_quality else None
        )
        has_query_regret = "regret_r2" in models and "regret_r5" in models
        regret_r2 = models["regret_r2"].predict(values) if has_query_regret else None
        regret_r5 = models["regret_r5"].predict(values) if has_query_regret else None
        for offset, index in enumerate(indices):
            r2 = float(delta_r2[offset])
            r5 = float(delta_r5[offset])
            r2_lcb = r2 + r2_residual
            r5_lcb = r5 + r5_residual
            prediction = {
                "delta_r2": r2,
                "delta_r5": r5,
                "loss_probability": float(loss[offset]),
                "gain_probability": float(gain[offset]),
                "delta_r2_lcb": r2_lcb,
                "delta_r5_lcb": r5_lcb,
                "utility_mean": 2.0 * r2 + r5,
                "utility_lcb": 2.0 * r2_lcb + r5_lcb,
                "safety_margin": float(gain[offset] - loss[offset]),
            }
            if has_absolute_quality:
                q2 = float(quality_r2[offset])
                q5 = float(quality_r5[offset])
                opt2 = float(optimal_r2[offset])
                opt5 = float(optimal_r5[offset])
                prediction.update({
                    "quality_r2": q2,
                    "quality_r5": q5,
                    "quality_r2_lcb": q2 + float(
                        plan_payload.get("quality_r2_residual_lcb", 0.0)
                    ),
                    "quality_r5_lcb": q5 + float(
                        plan_payload.get("quality_r5_residual_lcb", 0.0)
                    ),
                    "optimal_r2_probability": opt2,
                    "optimal_r5_probability": opt5,
                    "regret_r2_probability": 1.0 - opt2,
                    "regret_r5_probability": 1.0 - opt5,
                })
            if has_query_regret:
                pred_regret_r2 = max(0.0, float(regret_r2[offset]))
                pred_regret_r5 = max(0.0, float(regret_r5[offset]))
                prediction.update({
                    "regret_r2": pred_regret_r2,
                    "regret_r5": pred_regret_r5,
                    "regret_r2_ucb": max(0.0, pred_regret_r2 + float(
                        plan_payload.get("regret_r2_residual_ucb", 0.0)
                    )),
                    "regret_r5_ucb": max(0.0, pred_regret_r5 + float(
                        plan_payload.get("regret_r5_residual_ucb", 0.0)
                    )),
                })
            predicted[index]["prediction"] = prediction
    return predicted


def query_relative_regret_score(row: Mapping[str, Any]) -> float:
    prediction = row.get("prediction", {}) or {}
    return (
        2.0 * float(prediction.get("regret_r2_ucb", 1.0))
        + float(prediction.get("regret_r5_ucb", 1.0))
    )


def query_relative_slate_key(row: Mapping[str, Any]) -> Tuple[float, ...]:
    prediction = row.get("prediction", {}) or {}
    features = row.get("features", {}) or {}
    proof_coherence = sum(float(features.get(name, 0.0) or 0.0) for name in (
        "plan_completion_path_connected",
        "plan_introduced_complete_proof_fraction",
        "plan_introduced_relation_coverage_mean",
        "residual_candidate_complete_provenance",
        "residual_candidate_terminal_relation_match",
    ))
    displacement = (
        2.0 * float(features.get("plan_top2_displacement_count", 0.0) or 0.0)
        + float(features.get("plan_top5_displacement_count", 0.0) or 0.0)
    )
    return (
        -float(prediction.get("regret_r2_ucb", 1.0)),
        -float(prediction.get("regret_r5_ucb", 1.0)),
        proof_coherence,
        -displacement,
        -float(features.get("estimated_cost_log", 0.0) or 0.0),
        float(str(row.get("plan_name", "")) in {
            PLAN_SEMANTIC, PLAN_SEMANTIC_TOP2_NOOP,
        }),
    )


def choose_rows(
    rows: Sequence[Dict[str, Any]],
    plan_payloads: Mapping[str, Any],
    models: Mapping[str, Mapping[str, RandomForestRegressor]],
    *,
    utility_threshold: float,
    max_loss_probability: float,
    selection_mode: str = "pareto_lcb",
    min_delta_r2_mean: float = 0.0,
    min_delta_r5_mean: float = 0.0,
    min_gain_probability: float = 0.0,
    allowed_characteristics: set[str] | None = None,
) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if (
            allowed_characteristics is not None
            and query_characteristic(row) not in allowed_characteristics
        ):
            continue
        model_key = outcome_model_key(row, plan_payloads, models)
        if model_key not in plan_payloads or model_key not in models:
            continue
        prediction = row.get("prediction")
        if not isinstance(prediction, Mapping):
            prediction = predict_row(
                row, plan_payloads[model_key], models[model_key]
            )
        objective_class = plan_objective_class(row)
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
                prediction["gain_probability"] >= min_gain_probability
                and prediction["loss_probability"] <= max_loss_probability
                and prediction["safety_margin"] > utility_threshold
            )
        elif selection_mode == "risk_budgeted_lexicographic":
            eligible = bool(
                prediction["delta_r2"] >= min_delta_r2_mean
                and prediction["delta_r5"] >= min_delta_r5_mean
                and prediction["gain_probability"] >= min_gain_probability
                and prediction["utility_mean"] > utility_threshold
                and prediction["loss_probability"] <= max_loss_probability
            )
        else:
            eligible = bool(
                prediction["delta_r2_lcb"] >= 0.0
                and prediction["delta_r5_lcb"] >= 0.0
                and prediction["gain_probability"] >= min_gain_probability
                and prediction["utility_lcb"] > utility_threshold
                and prediction["loss_probability"] <= max_loss_probability
            )
        if eligible:
            grouped[row["query_key"]].append({**row, "prediction": prediction})
    selected = []
    for candidates in grouped.values():
        if selection_mode == "query_relative_slate_regret":
            best = max(candidates, key=query_relative_slate_key)
            semantic = next(
                (
                    row for row in candidates
                    if str(row.get("plan_name", "")) == PLAN_SEMANTIC
                ),
                None,
            )
            if semantic is not None and str(best.get("plan_name", "")) != PLAN_SEMANTIC:
                advantage = (
                    query_relative_regret_score(semantic)
                    - query_relative_regret_score(best)
                )
                if advantage + 1e-12 < utility_threshold:
                    best = semantic
            selected.append(best)
            continue
        if selection_mode == "absolute_quality_regret":
            objective_class = plan_objective_class(candidates[0])
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
            objective_class = plan_objective_class(candidates[0])
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
            objective_class = plan_objective_class(candidates[0])
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
        selected.append(max(candidates, key=key))
    return selected


def choose_rows_by_objective_class(
    rows: Sequence[Dict[str, Any]],
    plan_payloads: Mapping[str, Any],
    models: Mapping[str, Mapping[str, RandomForestRegressor]],
    class_policies: Mapping[str, Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Apply separately calibrated gates, then choose one complete slate.

    A single utility threshold is structurally biased toward Top-2-changing
    actions because their utility contains a doubled R@2 term.  Calibrate the
    membership-safe Top-5 completion class independently, then compare only
    actions that passed their own class contract.
    """
    eligible: List[Dict[str, Any]] = []
    for objective_class, policy in class_policies.items():
        if not bool(policy.get("calibration_feasible", False)):
            continue
        regime_policies = policy.get("structural_regime_policies", {}) or {}
        routed_policies = (
            list(regime_policies.items())
            if regime_policies
            else [("", policy)]
        )
        for regime, routed_policy in routed_policies:
            if not bool(routed_policy.get("calibration_feasible", False)):
                continue
            class_rows = [
                row for row in rows
                if plan_objective_class(row) == objective_class
                and (
                    not regime
                    or query_contract_regime(row.get("features", {}) or {})
                    == regime
                )
            ]
            policy_mode = str(
                routed_policy.get("selection_mode", "pareto_lcb")
            )
            threshold_key = (
                "min_absolute_quality_lcb"
                if policy_mode == "absolute_quality_regret"
                else
                "min_safety_margin"
                if policy_mode == "safe_gain_listwise"
                else "min_utility_mean"
                if policy_mode == "risk_budgeted_lexicographic"
                else "min_utility_lcb"
            )
            eligible.extend(choose_rows(
                class_rows,
                plan_payloads,
                models,
                utility_threshold=float(
                    routed_policy.get(threshold_key, 0.0)
                ),
                max_loss_probability=float(
                    routed_policy.get("max_loss_probability", 0.0)
                ),
                selection_mode=policy_mode,
                min_delta_r2_mean=float(
                    routed_policy.get("min_delta_r2_mean", 0.0)
                ),
                min_delta_r5_mean=float(
                    routed_policy.get("min_delta_r5_mean", 0.0)
                ),
                min_gain_probability=float(
                    routed_policy.get("min_gain_probability", 0.0)
                ),
                allowed_characteristics=set(
                    routed_policy.get("allowed_query_characteristics", []) or []
                ),
            ))

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        grouped[row["query_key"]].append(row)
    selected = []
    for candidates in grouped.values():
        class_policy = (
            class_policies.get(plan_objective_class(candidates[0]), {}) or {}
        )
        regime_policies = class_policy.get(
            "structural_regime_policies", {}
        ) or {}
        routed_policy = regime_policies.get(
            query_contract_regime(candidates[0].get("features", {}) or {}),
            class_policy,
        )
        mode = str(routed_policy.get("selection_mode", "pareto_lcb"))
        if mode == "absolute_quality_regret":
            objective_class = plan_objective_class(candidates[0])
            metric = "r5" if objective_class == "top5_completion" else "r2"
            other_metric = "r2" if metric == "r5" else "r5"
            key = lambda row: (
                row["prediction"].get(f"quality_{metric}_lcb", -1.0),
                row["prediction"].get(f"optimal_{metric}_probability", 0.0),
                -row["prediction"].get(f"regret_{metric}_probability", 1.0),
                row["prediction"].get(f"quality_{other_metric}_lcb", -1.0),
            )
        elif mode == "safe_gain_listwise":
            key = lambda row: (
                row["prediction"]["safety_margin"],
                row["prediction"]["gain_probability"],
                -row["prediction"]["loss_probability"],
                row["prediction"]["delta_r2"],
                row["prediction"]["delta_r5"],
            )
        elif mode == "risk_budgeted_lexicographic":
            key = lambda row: (
                row["prediction"]["delta_r2"],
                row["prediction"]["delta_r5"],
                row["prediction"]["utility_mean"],
                -row["prediction"]["loss_probability"],
            )
        else:
            key = lambda row: (
                row["prediction"]["utility_lcb"],
                row["prediction"]["delta_r2_lcb"],
                row["prediction"]["delta_r5_lcb"],
                -row["prediction"]["loss_probability"],
            )
        selected.append(max(candidates, key=key))
    return selected


def query_characteristic(row: Mapping[str, Any]) -> str:
    features = row.get("features", {}) or {}
    strict_paths = float(features.get("strict_executable_path_count_log", 0.0) or 0.0)
    relation_slots = float(features.get("relation_slot_count", 0.0) or 0.0)
    if strict_paths <= 0.0:
        return "no_strict_path"
    if relation_slots <= 1.0:
        return "single_relation"
    if relation_slots <= 2.0:
        return "two_relation"
    return "branch_or_long_chain"


def plan_objective_class(row: Mapping[str, Any]) -> str:
    """Separate membership-safe Top-5 completion from Top-2-changing plans."""
    features = row.get("features", {}) or {}
    if float(features.get("plan_objective_top2_noop", 0.0) or 0.0) >= 0.5:
        return "top2_ordering"
    if float(features.get("plan_top2_membership_invariant", 0.0) or 0.0) >= 0.5:
        return "top5_completion"
    return "top2_ordering"


def objective_class_args(
    args: argparse.Namespace, objective_class: str
) -> argparse.Namespace:
    """Return the risk contract used to calibrate one rank partition."""
    class_args = copy.copy(args)
    if objective_class == "top5_completion":
        class_args.enable_gain_probability_gate = True
        class_args.max_calibration_loss_rate = float(
            args.max_top5_calibration_loss_rate
        )
        class_args.max_holdout_loss_rate = float(args.max_top5_holdout_loss_rate)
        class_args.max_characteristic_loss_rate = max(
            float(args.max_characteristic_loss_rate),
            float(args.max_top5_calibration_loss_rate),
        )
        configured_cap = float(
            getattr(args, "max_top5_admission_loss_probability", -1.0)
        )
        class_args.max_admission_loss_probability = (
            configured_cap
            if configured_cap >= 0.0
            else float(args.max_top5_calibration_loss_rate)
        )
    else:
        class_args.enable_gain_probability_gate = bool(
            args.selection_mode == "safe_gain_listwise"
        )
        configured_cap = float(
            getattr(args, "max_admission_loss_probability", -1.0)
        )
        class_args.max_admission_loss_probability = (
            configured_cap
            if configured_cap >= 0.0
            else float(args.max_calibration_loss_rate)
        )
    return class_args


def admission_loss_probability_caps(
    args: argparse.Namespace,
) -> Tuple[float, ...]:
    """Candidate model-score cutoffs under an empirical risk constraint.

    A classifier score and an observed policy loss rate are different
    quantities. The former is a threshold calibrated on OOF predictions; the
    latter is the safety budget that every selected policy must satisfy. Older
    versions incorrectly forced both to use the same numeric ceiling.
    """
    search_max = float(
        getattr(
            args,
            "max_admission_loss_probability",
            args.max_calibration_loss_rate,
        )
    )
    search_max = min(max(search_max, 0.0), 1.0)
    candidates = {
        cap
        for cap in (
            0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.35, 0.50,
            0.65, 0.75, 0.90, 0.95,
        )
        if cap <= search_max + 1e-12
    }
    candidates.add(search_max)
    return tuple(sorted(candidates))


def _basic_report(
    selected: Sequence[Mapping[str, Any]], query_count: int
) -> Dict[str, Any]:
    r2 = [float(row["delta_r2"]) for row in selected]
    r5 = [float(row["delta_r5"]) for row in selected]
    losses = [
        value2 < -1e-12 or value5 < -1e-12
        for value2, value5 in zip(r2, r5)
    ]
    return {
        "query_count": query_count,
        "selected_count": len(selected),
        "selection_rate": len(selected) / max(query_count, 1),
        "delta_r2_sum": sum(r2),
        "delta_r5_sum": sum(r5),
        "delta_r2_per_query": sum(r2) / max(query_count, 1),
        "delta_r5_per_query": sum(r5) / max(query_count, 1),
        "selected_loss_count": sum(losses),
        "selected_loss_rate": sum(losses) / max(len(selected), 1),
        "selected_plan_counts": {
            plan: sum(row["plan_name"] == plan for row in selected)
            for plan in ALTERNATIVE_PLANS
        },
    }


def report(selected: Sequence[Mapping[str, Any]], query_count: int) -> Dict[str, Any]:
    result = _basic_report(selected, query_count)
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in selected:
        grouped[query_characteristic(row)].append(row)
    result["characteristic_reports"] = {
        name: _basic_report(rows, len(rows))
        for name, rows in sorted(grouped.items())
    }
    grouped_datasets: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in selected:
        grouped_datasets[str(row.get("dataset", "unknown"))].append(row)
    result["dataset_reports"] = {
        name: _basic_report(rows, len(rows))
        for name, rows in sorted(grouped_datasets.items())
    }
    return result


def characteristic_safe(summary: Mapping[str, Any], args: argparse.Namespace) -> bool:
    min_selected = int(getattr(args, "min_characteristic_selected", 3))
    max_loss = float(getattr(args, "max_characteristic_loss_rate", 0.15))
    for group in (summary.get("characteristic_reports", {}) or {}).values():
        if int(group.get("selected_count", 0)) < min_selected:
            continue
        if (
            float(group.get("selected_loss_rate", 0.0)) > max_loss
            or float(group.get("delta_r2_sum", 0.0)) < -1e-12
            or float(group.get("delta_r5_sum", 0.0)) < -1e-12
        ):
            return False
    return True


def dataset_safe(
    summary: Mapping[str, Any],
    args: argparse.Namespace,
    max_loss_rate: float | None = None,
) -> bool:




    max_loss = (
        float(max_loss_rate)
        if max_loss_rate is not None
        else float(getattr(args, "max_calibration_loss_rate", 0.08))
    )
    reports = summary.get("dataset_reports", {}) or {}
    min_selected = max(0, int(getattr(args, "min_domain_selected", 0)))
    expected_datasets = {
        value.strip()
        for value in str(getattr(args, "datasets", "") or "").split(",")
        if value.strip()
    }
    if min_selected:
        for dataset in expected_datasets:
            group = reports.get(dataset, {}) or {}
            if int(group.get("selected_count", 0)) < min_selected:
                return False
            if (
                2.0 * float(group.get("delta_r2_sum", 0.0))
                + float(group.get("delta_r5_sum", 0.0))
                <= 1e-12
            ):
                return False
            utility_per_action = (
                2.0 * float(group.get("delta_r2_sum", 0.0))
                + float(group.get("delta_r5_sum", 0.0))
            ) / max(int(group.get("selected_count", 0)), 1)
            if utility_per_action + 1e-12 < float(
                getattr(args, "min_domain_utility_per_action", 0.0)
            ):
                return False
    for group in reports.values():
        if (
            float(group.get("selected_loss_rate", 0.0)) > max_loss
            or float(group.get("delta_r2_sum", 0.0)) < -1e-12
            or float(group.get("delta_r5_sum", 0.0)) < -1e-12
        ):
            return False
    return True


def worst_dataset_utility_per_action(
    summary: Mapping[str, Any], expected_datasets: Iterable[str] = ()
) -> float:
    utilities = []
    reports = summary.get("dataset_reports", {}) or {}
    for dataset in expected_datasets:
        if dataset not in reports:
            return float("-inf")
    for group in reports.values():
        selected_count = int(group.get("selected_count", 0))
        if selected_count <= 0:
            continue
        utility = (
            2.0 * float(group.get("delta_r2_sum", 0.0))
            + float(group.get("delta_r5_sum", 0.0))
        ) / selected_count
        utilities.append(utility)
    return min(utilities) if utilities else float("-inf")


def calibrate_policy(
    rows: Sequence[Dict[str, Any]],
    plan_payloads: Mapping[str, Any],
    models: Mapping[str, Mapping[str, RandomForestRegressor]],
    args: argparse.Namespace,
    query_count: int | None = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    predicted_rows = attach_predictions(rows, plan_payloads, models)
    if query_count is None:
        query_count = len({row["query_key"] for row in predicted_rows})
    predicted_utilities = []
    predicted_gains = []
    for row in predicted_rows:
        model_key = outcome_model_key(row, plan_payloads, models)
        if model_key not in plan_payloads or model_key not in models:
            continue
        prediction = row.get("prediction", {}) or {}
        if args.selection_mode == "absolute_quality_regret":
            objective_class = plan_objective_class(row)
            metric = "r5" if objective_class == "top5_completion" else "r2"
            predicted_utilities.append(prediction[f"quality_{metric}_lcb"])
        elif args.selection_mode == "safe_gain_listwise":
            predicted_utilities.append(prediction["safety_margin"])
            predicted_gains.append(prediction["gain_probability"])
        elif args.selection_mode == "risk_budgeted_lexicographic":
            if (
                prediction["delta_r2"] >= args.min_delta_r2_mean
                and prediction["delta_r5"] >= args.min_delta_r5_mean
            ):
                predicted_utilities.append(prediction["utility_mean"])
                predicted_gains.append(prediction["gain_probability"])
        elif prediction["delta_r2_lcb"] >= 0.0 and prediction["delta_r5_lcb"] >= 0.0:
            predicted_utilities.append(prediction["utility_lcb"])
            predicted_gains.append(prediction["gain_probability"])
    thresholds = [0.0]
    if predicted_utilities:
        thresholds.extend(
            float(value)
            for value in np.quantile(
                predicted_utilities, np.linspace(0.0, 0.95, 24)
            )
        )


    thresholds = sorted({max(0.0, float(value)) for value in thresholds})
    gain_thresholds = [0.0]
    if predicted_gains and bool(
        getattr(args, "enable_gain_probability_gate", False)
        and args.selection_mode != "absolute_quality_regret"
    ):
        gain_thresholds.extend(
            float(value)
            for value in np.quantile(
                predicted_gains, np.linspace(0.25, 0.98, 14)
            )
        )
    gain_thresholds = sorted({
        min(1.0, max(0.0, float(value))) for value in gain_thresholds
    })
    loss_caps = admission_loss_probability_caps(args)
    feasible = []
    for loss_cap in loss_caps:
        for gain_threshold in gain_thresholds:
            for threshold in thresholds:
                selected = choose_rows(
                    predicted_rows,
                    plan_payloads,
                    models,
                    utility_threshold=threshold,
                    max_loss_probability=loss_cap,
                    selection_mode=args.selection_mode,
                    min_delta_r2_mean=args.min_delta_r2_mean,
                    min_delta_r5_mean=args.min_delta_r5_mean,
                    min_gain_probability=gain_threshold,
                )
                summary = report(selected, query_count)
                if (
                    summary["selected_count"] >= args.min_calibration_selected
                    and summary["selected_loss_rate"] <= args.max_calibration_loss_rate
                    and summary["delta_r2_sum"] >= -1e-12
                    and summary["delta_r5_sum"] >= -1e-12
                    and (summary["delta_r2_sum"] > 1e-12 or summary["delta_r5_sum"] > 1e-12)
                    and characteristic_safe(summary, args)
                    and dataset_safe(
                        summary,
                        args,
                        max_loss_rate=args.max_calibration_loss_rate,
                    )
                ):
                    aggregate_utility = (
                        2.0 * summary["delta_r2_sum"]
                        + summary["delta_r5_sum"]
                    )
                    expected_datasets = (
                        {
                            value.strip()
                            for value in str(args.datasets or "").split(",")
                            if value.strip()
                        }
                        if getattr(args, "robust_calibration_objective", False)
                        else set()
                    )
                    worst_utility = worst_dataset_utility_per_action(
                        summary, expected_datasets
                    )
                    primary = (
                        worst_utility
                        if getattr(args, "robust_calibration_objective", False)
                        else aggregate_utility
                    )
                    secondary = (
                        aggregate_utility
                        if getattr(args, "robust_calibration_objective", False)
                        else worst_utility
                    )
                    feasible.append((
                        primary,
                        secondary,
                        summary["selected_count"],
                        -summary["selected_loss_rate"],
                        gain_threshold,
                        threshold,
                        loss_cap,
                        summary,
                    ))
    if not feasible:
        policy = {
            "selection_mode": args.selection_mode,
            "min_absolute_quality_lcb": float("inf"),
            "min_utility_lcb": float("inf"),
            "min_utility_mean": float("inf"),
            "min_safety_margin": float("inf"),
            "max_loss_probability": 0.0,
            "min_delta_r2_lcb": 0.0,
            "min_delta_r5_lcb": 0.0,
            "min_delta_r2_mean": args.min_delta_r2_mean,
            "min_delta_r5_mean": args.min_delta_r5_mean,
            "min_gain_probability": float("inf"),
            "calibration_feasible": False,
        }
        return policy, report([], query_count)
    _, _, _, _, gain_threshold, threshold, loss_cap, summary = max(feasible)
    allowed_characteristics = sorted(
        name
        for name, group in (summary.get("characteristic_reports", {}) or {}).items()
        if int(group.get("selected_count", 0)) >= args.min_characteristic_selected
        and float(group.get("selected_loss_rate", 0.0))
        <= args.max_characteristic_loss_rate
        and float(group.get("delta_r2_sum", 0.0)) >= -1e-12
        and float(group.get("delta_r5_sum", 0.0)) >= -1e-12
    )
    policy = {
        "selection_mode": args.selection_mode,
        "min_absolute_quality_lcb": (
            float(threshold)
            if args.selection_mode == "absolute_quality_regret"
            else 0.0
        ),
        "min_utility_lcb": (
            float(threshold) if args.selection_mode == "pareto_lcb" else 0.0
        ),
        "min_utility_mean": (
            float(threshold)
            if args.selection_mode == "risk_budgeted_lexicographic"
            else 0.0
        ),
        "min_safety_margin": (
            float(threshold)
            if args.selection_mode == "safe_gain_listwise"
            else 0.0
        ),
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
        "calibration_feasible": True,
        "allowed_query_characteristics": allowed_characteristics,
    }
    return policy, summary


def write_action_table(
    output_dir: str,
    split: str,
    rows: Sequence[Dict[str, Any]],
) -> None:
    if not output_dir:
        return
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{split}.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps({
                "query_key": row.get("query_key"),
                "dataset": row.get("dataset"),
                "plan_name": row.get("plan_name"),
                "plan_instance": row.get("plan_instance"),
                "features": row.get("features", {}),
                "delta_r2": row.get("delta_r2", 0.0),
                "delta_r5": row.get("delta_r5", 0.0),
                "gain_probability": row.get("gain_probability", 0.0),
                "loss_probability": row.get("loss_probability", 0.0),
                "prediction": row.get("prediction", {}),
            }, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    if (
        args.selection_mode == "query_relative_slate_regret"
        and not args.query_relative_slate_regret_targets
    ):
        raise ValueError(
            "query_relative_slate_regret selection requires "
            "--query-relative-slate-regret-targets"
        )
    selected_plans = set(allowed_plans(args))
    datasets = [value.strip() for value in args.datasets.split(",") if value.strip()]
    replay_gold = {}
    if args.rematerialize_plan_catalog:
        if not args.workload_dir:
            raise ValueError("--workload-dir is required for plan rematerialization")
        print("[cbo] loading historical exact-evidence contracts", flush=True)
        replay_gold = {
            "induction": historical_gold_ids(
                args.workload_dir, "q_ind", datasets
            ),
            "calibration": historical_gold_ids(
                args.workload_dir, "q_cal", datasets
            ),
            "holdout": historical_gold_ids(
                args.workload_dir, "q_val", datasets
            ),
        }
    print("[cbo] loading induction traces", flush=True)
    induction_query_keys: set[str] = set()
    calibration_query_keys: set[str] = set()
    holdout_query_keys: set[str] = set()
    induction = [
        row for row in rows_from_traces(
            args.induction_trace_dir,
            datasets,
            induction_query_keys,
            replay_gold.get("induction"),
            pareto_safe_gain_label=bool(args.pareto_safe_gain_label),
            allowed_plan_names=selected_plans,
            query_relative_slate_regret=bool(
                args.query_relative_slate_regret_targets
            ),
        )
    ]
    print(f"[cbo] induction rows={len(induction)}", flush=True)
    print("[cbo] loading calibration traces", flush=True)
    calibration = [
        row for row in rows_from_traces(
            args.calibration_trace_dir,
            datasets,
            calibration_query_keys,
            replay_gold.get("calibration"),
            pareto_safe_gain_label=bool(args.pareto_safe_gain_label),
            allowed_plan_names=selected_plans,
            query_relative_slate_regret=bool(
                args.query_relative_slate_regret_targets
            ),
        )
    ]
    print(f"[cbo] calibration rows={len(calibration)}", flush=True)
    print("[cbo] loading holdout traces", flush=True)
    holdout = [
        row for row in rows_from_traces(
            args.holdout_trace_dir,
            datasets,
            holdout_query_keys,
            replay_gold.get("holdout"),
            pareto_safe_gain_label=bool(args.pareto_safe_gain_label),
            allowed_plan_names=selected_plans,
            query_relative_slate_regret=bool(
                args.query_relative_slate_regret_targets
            ),
        )
    ]
    if not induction or not calibration or not holdout:
        raise RuntimeError(
            f"empty portfolio split: induction={len(induction)} "
            f"calibration={len(calibration)} holdout={len(holdout)}"
        )
    print(f"[cbo] holdout rows={len(holdout)}", flush=True)
    print("[cbo] training plan-level models", flush=True)
    plan_payloads, models = train_models(induction, calibration, args)
    print("[cbo] materializing batched replay predictions", flush=True)
    induction_predicted = attach_predictions(induction, plan_payloads, models)
    calibration_predicted = attach_predictions(calibration, plan_payloads, models)
    holdout_predicted = attach_predictions(holdout, plan_payloads, models)
    write_action_table(args.action_table_dir, "induction", induction_predicted)
    write_action_table(args.action_table_dir, "calibration", calibration_predicted)
    write_action_table(args.action_table_dir, "holdout", holdout_predicted)
    print("[cbo] calibrating rank-partitioned selection policies", flush=True)
    class_policies: Dict[str, Dict[str, Any]] = {}
    class_calibration_reports: Dict[str, Dict[str, Any]] = {}
    for objective_class in ("top2_ordering", "top5_completion"):
        class_args = objective_class_args(args, objective_class)
        class_rows = [
            row for row in calibration_predicted
            if plan_objective_class(row) == objective_class
        ]
        if bool(getattr(args, "structural_expert_model", False)):
            regime_policies: Dict[str, Dict[str, Any]] = {}
            regime_reports: Dict[str, Dict[str, Any]] = {}
            regimes = sorted({
                query_contract_regime(row.get("features", {}) or {})
                for row in class_rows
            })
            for regime in regimes:
                regime_rows = [
                    row for row in class_rows
                    if query_contract_regime(row.get("features", {}) or {})
                    == regime
                ]
                regime_query_count = len({
                    row["query_key"] for row in regime_rows
                })
                regime_policy, regime_report = calibrate_policy(
                    regime_rows,
                    plan_payloads,
                    models,
                    class_args,
                    query_count=regime_query_count,
                )
                regime_policy["query_contract_regime"] = regime
                regime_policies[regime] = regime_policy
                regime_reports[regime] = regime_report
            class_policy = {
                "selection_mode": args.selection_mode,
                "calibration_feasible": any(
                    bool(value.get("calibration_feasible", False))
                    for value in regime_policies.values()
                ),
                "structural_regime_policies": regime_policies,
            }
            class_report = report(
                choose_rows_by_objective_class(
                    class_rows,
                    plan_payloads,
                    models,
                    {objective_class: class_policy},
                ),
                len(calibration_query_keys),
            )
            class_report["structural_regime_reports"] = regime_reports
        else:
            class_policy, class_report = calibrate_policy(
                class_rows,
                plan_payloads,
                models,
                class_args,
                query_count=len(calibration_query_keys),
            )
        class_policy["objective_class"] = objective_class
        class_policy["risk_contract"] = (
            "top2_membership_invariant_r5_completion"
            if objective_class == "top5_completion"
            else "strict_top2_and_top5_non_regression"
        )
        class_policies[objective_class] = class_policy
        class_calibration_reports[objective_class] = class_report

    calibration_selected = choose_rows_by_objective_class(
        calibration_predicted,
        plan_payloads,
        models,
        class_policies,
    )
    calibration_report = report(
        calibration_selected, len(calibration_query_keys)
    )
    policy = {
        "selection_mode": args.selection_mode,
        "calibration_feasible": any(
            bool(value.get("calibration_feasible", False))
            for value in class_policies.values()
        ),
        "objective_class_policies": class_policies,
        "allowed_plans": sorted(selected_plans),
    }
    holdout_queries = len(holdout_query_keys)
    holdout_selected = choose_rows_by_objective_class(
        holdout_predicted,
        plan_payloads,
        models,
        class_policies,
    ) if policy["calibration_feasible"] else []
    holdout_report = report(holdout_selected, holdout_queries)
    class_holdout_certificates: Dict[str, Dict[str, Any]] = {}
    class_certificates_passed = True
    for objective_class, class_policy in class_policies.items():
        class_args = objective_class_args(args, objective_class)
        class_rows = [
            row for row in holdout_predicted
            if plan_objective_class(row) == objective_class
        ]
        class_selected = choose_rows_by_objective_class(
            class_rows,
            plan_payloads,
            models,
            {objective_class: class_policy},
        )
        class_report = report(class_selected, holdout_queries)
        class_feasible = bool(class_policy.get("calibration_feasible", False))
        regime_certificates: Dict[str, Dict[str, Any]] = {}
        regime_certificates_passed = True
        for regime, regime_policy in (
            class_policy.get("structural_regime_policies", {}) or {}
        ).items():
            regime_feasible = bool(
                regime_policy.get("calibration_feasible", False)
            )
            regime_rows = [
                row for row in class_rows
                if query_contract_regime(row.get("features", {}) or {})
                == regime
            ]
            regime_query_count = len({
                row["query_key"] for row in regime_rows
            })
            routed_class_policy = {
                **class_policy,
                "structural_regime_policies": {regime: regime_policy},
            }
            regime_selected = choose_rows_by_objective_class(
                regime_rows,
                plan_payloads,
                models,
                {objective_class: routed_class_policy},
            )
            regime_report = report(regime_selected, regime_query_count)
            regime_passed = bool(
                not regime_feasible
                or (
                    regime_report["selected_count"]
                    >= args.min_holdout_selected
                    and regime_report["selected_loss_rate"]
                    <= class_args.max_holdout_loss_rate
                    and regime_report["delta_r2_sum"] >= -1e-12
                    and regime_report["delta_r5_sum"] >= -1e-12
                    and characteristic_safe(regime_report, class_args)
                    and dataset_safe(
                        regime_report,
                        class_args,
                        max_loss_rate=class_args.max_holdout_loss_rate,
                    )
                    and (
                        regime_report["delta_r2_sum"] > 1e-12
                        or regime_report["delta_r5_sum"] > 1e-12
                    )
                )
            )
            regime_certificates[regime] = {
                "passed": regime_passed,
                "calibration_feasible": regime_feasible,
                **regime_report,
            }
            regime_certificates_passed = (
                regime_certificates_passed and regime_passed
            )
        class_passed = bool(
            not class_feasible
            or (
                regime_certificates_passed
                and class_report["selected_count"] >= args.min_holdout_selected
                and class_report["selected_loss_rate"]
                <= class_args.max_holdout_loss_rate
                and class_report["delta_r2_sum"] >= -1e-12
                and class_report["delta_r5_sum"] >= -1e-12
                and characteristic_safe(class_report, class_args)
                and dataset_safe(
                    class_report,
                    class_args,
                    max_loss_rate=class_args.max_holdout_loss_rate,
                )
                and (
                    class_report["delta_r2_sum"] > 1e-12
                    or class_report["delta_r5_sum"] > 1e-12
                )
            )
        )
        class_holdout_certificates[objective_class] = {
            "passed": class_passed,
            "calibration_feasible": class_feasible,
            "max_loss_rate": class_args.max_holdout_loss_rate,
            "structural_regime_certificates": regime_certificates,
            **class_report,
        }
        class_certificates_passed = class_certificates_passed and class_passed
    aggregate_holdout_loss_rate = max(
        float(args.max_holdout_loss_rate),
        float(args.max_top5_holdout_loss_rate),
    )
    passed = bool(
        policy["calibration_feasible"]
        and class_certificates_passed
        and holdout_report["selected_count"] >= args.min_holdout_selected
        and holdout_report["selected_loss_rate"] <= aggregate_holdout_loss_rate
        and holdout_report["delta_r2_sum"] >= -1e-12
        and holdout_report["delta_r5_sum"] >= -1e-12
        and characteristic_safe(holdout_report, args)
        and dataset_safe(
            holdout_report,
            args,
            max_loss_rate=aggregate_holdout_loss_rate,
        )
        and (
            holdout_report["delta_r2_sum"] > 1e-12
            or holdout_report["delta_r5_sum"] > 1e-12
        )
    )
    payload = {
        "type": MODEL_TYPE,
        "version": 274 if args.selection_mode == "safe_gain_listwise" else 271,
        "feature_names": list(FEATURE_NAMES),
        "dataset_identity_used_for_routing": False,
        "runtime_uses_gold": False,
        "calibration_validation_merged": False,
        "split_protocol": {
            "q_ind": "fit_plan_outcome_estimators",
            "q_cal": "calibrate_policy_thresholds_and_risk_budget",
            "q_val": "frozen_policy_holdout_certificate_only",
        },
        "training_label_source": "historical_counterfactual_unified_physical_plan_replay",
        "physical_plan_catalog_rematerialized": bool(
            args.rematerialize_plan_catalog
        ),
        "training_label_semantics": "official_exact_document_recall",
        "baseline_endpoint": "final_ranking_incumbent",
        "training_targets": list(training_targets(args)),
        "target_semantics": (
            "absolute_plan_quality_and_query_local_oracle_regret"
            if bool(getattr(args, "absolute_quality_regret_targets", False))
            else "incumbent_relative_recall_delta"
        ),
        "gain_label_semantics": (
            "pareto_safe_recall_gain"
            if args.pareto_safe_gain_label
            else "any_recall_gain"
        ),
        "outcome_class_balancing": bool(args.balance_outcome_targets),
        "outcome_model_scope": (
            "query_contract_structural_experts"
            if args.structural_expert_model
            else "global_plan_comparable"
            if args.global_outcome_model
            else "per_plan_family"
        ),
        "robust_calibration_objective": bool(
            getattr(args, "robust_calibration_objective", False)
        ),
        "objective": (
            "hierarchical_top2_ordering_and_candidate_level_top5_completion"
        ),
        "allowed_plans": sorted(selected_plans),
        "semantic_fact_access_role": (
            "in_plan_scan_operator"
            if PLAN_SIGNATURE_GROUNDED in selected_plans
            else "independent_incumbent"
        ),
        "final_slate_owner": (
            "ontology_guided_evidence_executor"
            if PLAN_SIGNATURE_GROUNDED in selected_plans
            else "complete_evidence_plan_optimizer"
        ),
        "query_structure_signature_enabled": bool(
            PLAN_SIGNATURE_GROUNDED in selected_plans
        ),
        "characteristic_routing": (
            "dataset_agnostic_query_contract_experts"
            if args.structural_expert_model
            else "relation_shape_only"
        ),
        "rank_partition_contract": {
            "top2_ordering": "top5_membership_invariant",
            "top5_completion": "top2_order_invariant",
            "nested_topk": "joint_top2_ordering_and_top5_completion",
            "completion_join": "proof_path_connected_to_incumbent_top5",
            "residual_top5": (
                "incumbent_top4_invariant_candidate_reservoir_completion"
            ),
        },
        "plans": plan_payloads,
        "policy": policy,
        "calibration_report": calibration_report,
        "objective_class_calibration_reports": class_calibration_reports,
        "holdout_certificate": {
            "passed": passed,
            "max_loss_rate": aggregate_holdout_loss_rate,
            "objective_class_certificates": class_holdout_certificates,
            **holdout_report,
        },
        "split_counts": {
            "induction_queries": len(induction_query_keys),
            "calibration_queries": len(calibration_query_keys),
            "holdout_queries": len(holdout_query_keys),
            "induction_rows": len(induction),
            "calibration_rows": len(calibration),
            "holdout_rows": len(holdout),
        },
    }
    output = Path(args.output_model)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "model": str(output),
        "holdout_passed": passed,
        "calibration": calibration_report,
        "holdout": holdout_report,
    }, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
