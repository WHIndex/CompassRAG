#!/usr/bin/env python3
"""Execution-matched physical-plan and three-head CBO analysis.

The experiment trains on a frozen historical complete-slate portfolio and
evaluates on a disjoint test portfolio.  Utility, cost, and risk estimators use
only query/plan/catalog features available before the selected physical plan is
committed.  Gold evidence is used only to construct offline labels and metrics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import brier_score_loss, mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.model_selection import GroupKFold


PLAN_FAMILIES = (
    "semantic_fact_scan",
    "semantic_fact_scan_top2_noop",
    "raw_semantic_graph_scan",
    "ontology_physical_complete_slate",
    "ontology_structural_execution_slate",
    "signature_grounded_unified_slate",
    "query_local_provenance_top2_pair",
    "query_local_provenance_top5_completion",
    "incumbent_preserving_chain_completion",
    "topology_conditioned_multi_access_top5",
    "learned_residual_top5_completion",
    "ontology_warehouse_rank5_completion",
)

QUERY_FEATURES = (
    "matched_plan_count_log",
    "strict_executable_path_count_log",
    "executable_path_count_log",
    "relation_slot_count",
    "explicit_relation_slot_count",
    "latent_contract_recovered",
    "logical_required_hop_count",
    "logical_required_branch_count",
    "query_relative_clause_count",
    "query_comparison_cue",
    "query_nested_bridge_cue",
    "anchor_count_log",
    "signature_topology_ordered_chain",
    "signature_topology_branch_comparison",
    "signature_answer_type_person",
    "signature_answer_type_location",
    "signature_answer_type_date",
    "signature_answer_type_number",
    "signature_answer_type_boolean",
    "signature_answer_type_organization",
    "signature_answer_type_other",
)

PLAN_TEMPLATE_FEATURES = (
    "plan_objective_top2_noop",
    "plan_is_bound_proof",
    "plan_is_semantic_fact_scan",
    "plan_is_raw_semantic_graph_scan",
    "plan_is_ontology_physical_complete_slate",
    "plan_is_ontology_structural_execution_slate",
    "plan_is_incumbent_preserving_chain_completion",
    "plan_is_signature_grounded",
    "plan_is_query_local_top2_pair",
    "plan_is_query_local_top5_completion",
    "plan_is_learned_residual_top5_completion",
    "plan_is_ontology_warehouse_rank5_completion",
    "plan_is_topology_conditioned_multi_access_top5",
)

FEATURE_NAMES = (
    *QUERY_FEATURES,
    *PLAN_TEMPLATE_FEATURES,
    *(f"family::{name}" for name in PLAN_FAMILIES),
)

FIXED_PREFERENCE = (
    "ontology_physical_complete_slate",
    "ontology_structural_execution_slate",
    "signature_grounded_unified_slate",
    "semantic_fact_scan",
    "semantic_fact_scan_top2_noop",
)


@dataclass
class PlanRow:
    query_key: str
    query_idx: int
    plan_family: str
    plan_instance: str
    doc_ids: tuple[int, ...]
    features: np.ndarray
    r2: float
    r5: float
    cost: float
    work: Dict[str, float]
    risk: int = 0
    pred_utility: float = 0.0
    pred_cost: float = 0.0
    pred_risk: float = 0.0


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def unlog(value: Any) -> float:
    return max(0.0, math.expm1(safe_float(value)))


def family_group(family: str) -> str:
    if family in {"semantic_fact_scan", "semantic_fact_scan_top2_noop"}:
        return "semantic"
    if family == "raw_semantic_graph_scan":
        return "graph"
    if family in {
        "learned_residual_top5_completion",
        "ontology_warehouse_rank5_completion",
        "incumbent_preserving_chain_completion",
    }:
        return "completion"
    if family in {
        "query_local_provenance_top2_pair",
        "query_local_provenance_top5_completion",
    }:
        return "bound_join"
    return "ontology"


def operator_work(family: str, features: Mapping[str, Any]) -> Dict[str, float]:
    """Return trace-derived operator counters for one materialized plan.

    Counts are reconstructed from the observed catalog cardinalities stored by
    the executor.  Every counter has unit weight, so the total is independent
    of wall-clock time, hardware, API latency, and token usage.
    """
    slots = max(1.0, safe_float(features.get("relation_slot_count"), 1.0))
    matched = unlog(features.get("matched_plan_count_log"))
    executable = unlog(features.get("executable_path_count_log"))
    strict = unlog(features.get("strict_executable_path_count_log"))
    proof_candidates = unlog(features.get("proof_signal_candidate_count_log"))
    local_paths = unlog(features.get("local_proof_path_count_log"))
    bound_paths = unlog(features.get("bound_proof_path_count_log"))
    group = family_group(family)

    encoded_queries = 1.0
    index_probes = 1.0
    graph_expansions = 0.0
    join_comparisons = 0.0
    candidates_materialized = 5.0
    if group == "semantic":
        index_probes = 2.0
        candidates_materialized = 5.0
    elif group == "graph":
        index_probes = 2.0
        graph_expansions = max(executable, 1.0)
        candidates_materialized = max(proof_candidates, 5.0)
    elif group == "bound_join":
        index_probes = 1.0 + slots
        graph_expansions = max(bound_paths, strict, 1.0)
        candidates_materialized = max(proof_candidates, 5.0)
        join_comparisons = max(bound_paths, 1.0) * slots
    elif group == "completion":
        index_probes = 1.0
        graph_expansions = max(local_paths, strict, 1.0)
        candidates_materialized = max(5.0, min(proof_candidates, 32.0))
        join_comparisons = max(1.0, candidates_materialized - 5.0) * slots
    else:
        index_probes = 1.0 + slots
        graph_expansions = max(local_paths + bound_paths, executable, strict, 1.0)
        candidates_materialized = max(proof_candidates, 5.0)
        join_comparisons = max(matched, 1.0) * slots
    values = {
        "encoded_queries": encoded_queries,
        "index_probes": index_probes,
        "graph_expansions": graph_expansions,
        "join_comparisons": join_comparisons,
        "candidates_materialized": candidates_materialized,
    }
    values["total"] = sum(values.values())
    return values


def feature_vector(family: str, features: Mapping[str, Any]) -> np.ndarray:
    values = [safe_float(features.get(name)) for name in (*QUERY_FEATURES, *PLAN_TEMPLATE_FEATURES)]
    values.extend(float(family == name) for name in PLAN_FAMILIES)
    return np.asarray(values, dtype=np.float64)


def iter_jsonl(path: Path) -> Iterable[Mapping[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            if isinstance(value, Mapping):
                yield value


def load_portfolio(path: Path, dataset: str) -> list[PlanRow]:
    rows: list[PlanRow] = []
    counters: Counter[str] = Counter()
    for record_number, record in enumerate(iter_jsonl(path), 1):
        query_idx = int(record.get("query_idx", record_number - 1))
        query_key = f"{dataset}:{query_idx}"
        optimizer = record.get("complete_evidence_plan_optimizer", {}) or {}
        labels = record.get("complete_evidence_plan_training", {}) or {}
        if not bool(optimizer.get("endpoint_contract_matched", False)):
            counters["endpoint_mismatch"] += 1
            continue
        if not bool(labels.get("labels_valid", False)):
            counters["invalid_labels"] += 1
            continue
        plans = optimizer.get("plans", {}) or {}
        plan_labels = labels.get("plans", {}) or {}
        query_rows: list[PlanRow] = []
        for instance, spec in plans.items():
            if not isinstance(spec, Mapping) or not bool(spec.get("valid", False)):
                continue
            family = str(spec.get("plan_family") or instance)
            if family not in PLAN_FAMILIES:
                continue
            outcome = plan_labels.get(instance, {}) or {}
            if not outcome:
                continue
            doc_ids = tuple(int(value) for value in (outcome.get("doc_ids") or spec.get("doc_ids") or [])[:5])
            if len(doc_ids) < 5:
                continue
            features = spec.get("features", {}) or {}
            work = operator_work(family, features)
            query_rows.append(PlanRow(
                query_key=query_key,
                query_idx=query_idx,
                plan_family=family,
                plan_instance=str(instance),
                doc_ids=doc_ids,
                features=feature_vector(family, features),
                r2=safe_float(outcome.get("recall_at2")),
                r5=safe_float(outcome.get("recall_at5")),
                cost=work["total"],
                work=work,
            ))



        deduplicated: Dict[tuple[int, ...], PlanRow] = {}
        for row in query_rows:
            previous = deduplicated.get(row.doc_ids)
            if previous is None or (row.cost, row.plan_family, row.plan_instance) < (
                previous.cost, previous.plan_family, previous.plan_instance
            ):
                deduplicated[row.doc_ids] = row
        rows.extend(deduplicated.values())
        counters["queries"] += 1
        counters["raw_plans"] += len(query_rows)
        counters["deduplicated_plans"] += len(deduplicated)
        if record_number % 100 == 0:
            print(json.dumps({"stage": "load", "path": str(path), "records": record_number, **counters}), flush=True)
    print(json.dumps({"stage": "loaded", "path": str(path), **counters}), flush=True)
    return rows


def grouped(rows: Sequence[PlanRow]) -> Dict[str, list[PlanRow]]:
    result: Dict[str, list[PlanRow]] = defaultdict(list)
    for row in rows:
        result[row.query_key].append(row)
    return result


def fixed_row(candidates: Sequence[PlanRow]) -> PlanRow:
    for family in FIXED_PREFERENCE:
        options = [row for row in candidates if row.plan_family == family]
        if options:
            return min(options, key=lambda row: (row.cost, row.plan_instance))
    return min(candidates, key=lambda row: (row.cost, row.plan_family, row.plan_instance))


def attach_risk_labels(rows: Sequence[PlanRow]) -> None:
    for candidates in grouped(rows).values():
        fixed = fixed_row(candidates)
        for row in candidates:
            row.risk = int(row.r2 + 1e-12 < fixed.r2 or row.r5 + 1e-12 < fixed.r5)


def make_models(seed: int) -> tuple[Any, Any, Any]:
    common = dict(
        learning_rate=0.06,
        max_iter=180,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=2.0,
        random_state=seed,
    )
    return (
        HistGradientBoostingRegressor(loss="squared_error", **common),
        HistGradientBoostingRegressor(loss="squared_error", **common),
        HistGradientBoostingClassifier(loss="log_loss", **common),
    )


def arrays(rows: Sequence[PlanRow]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.vstack([row.features for row in rows]),
        np.asarray([row.r5 for row in rows], dtype=np.float64),
        np.asarray([row.cost for row in rows], dtype=np.float64),
        np.asarray([row.risk for row in rows], dtype=np.int64),
    )


def fit_predict(train: Sequence[PlanRow], target: Sequence[PlanRow], seed: int) -> tuple[Any, Any, Any]:
    train_x, utility_y, cost_y, risk_y = arrays(train)
    target_x = np.vstack([row.features for row in target])
    utility_model, cost_model, risk_model = make_models(seed)
    utility_model.fit(train_x, utility_y)
    cost_model.fit(train_x, np.log1p(cost_y))
    risk_model.fit(train_x, risk_y)
    utility_pred = np.clip(utility_model.predict(target_x), 0.0, 1.0)
    cost_pred = np.maximum(0.0, np.expm1(cost_model.predict(target_x)))
    risk_pred = risk_model.predict_proba(target_x)[:, 1]
    for index, row in enumerate(target):
        row.pred_utility = float(utility_pred[index])
        row.pred_cost = float(cost_pred[index])
        row.pred_risk = float(risk_pred[index])
    return utility_model, cost_model, risk_model


def grouped_oof(rows: Sequence[PlanRow], folds: int, seed: int) -> list[PlanRow]:
    del seed
    row_array = np.asarray(rows, dtype=object)
    groups = np.asarray([row.query_key for row in rows], dtype=object)
    dummy = np.zeros(len(rows), dtype=np.int8)
    splitter = GroupKFold(n_splits=folds)
    for fold, (train_idx, held_idx) in enumerate(splitter.split(dummy, groups=groups)):
        train = list(row_array[train_idx])
        held = list(row_array[held_idx])
        fit_predict(train, held, 336 + fold)
        print(json.dumps({"stage": "grouped_oof", "fold": fold, "train_rows": len(train), "held_rows": len(held), "held_queries": len({row.query_key for row in held})}), flush=True)
    return list(row_array)


def deterministic_random(candidates: Sequence[PlanRow], query_key: str) -> PlanRow:
    digest = hashlib.sha256(query_key.encode("utf-8")).digest()
    return sorted(candidates, key=lambda row: (row.plan_family, row.plan_instance))[
        int.from_bytes(digest[:8], "big") % len(candidates)
    ]


def select_rows(
    rows: Sequence[PlanRow],
    selector: str,
    risk_cap: float,
    cost_multiplier: float,
) -> list[PlanRow]:
    selected = []
    for query_key, candidates in grouped(rows).items():
        fixed = fixed_row(candidates)
        if selector == "Random":
            winner = deterministic_random(candidates, query_key)
        elif selector == "Fixed Plan":
            winner = fixed
        elif selector == "Utility-only":
            winner = max(candidates, key=lambda row: (row.pred_utility, -row.pred_cost, -row.pred_risk))
        elif selector == "Full CBO":
            budget = max(fixed.pred_cost, 1.0) * cost_multiplier
            feasible = [
                row for row in candidates
                if row.pred_risk <= risk_cap + 1e-12
                and row.pred_cost <= budget + 1e-12
            ]
            winner = (
                max(feasible, key=lambda row: (row.pred_utility, -row.pred_risk, -row.pred_cost))
                if feasible else fixed
            )
        elif selector == "Oracle":
            winner = max(candidates, key=lambda row: (row.r5, row.r2, -row.cost))
        else:
            raise ValueError(selector)
        selected.append(winner)
    return selected


def summarize_selection(selected: Sequence[PlanRow], all_rows: Sequence[PlanRow]) -> Dict[str, Any]:
    by_query = grouped(all_rows)
    oracle_by_query = {
        key: max(candidates, key=lambda row: (row.r5, row.r2, -row.cost))
        for key, candidates in by_query.items()
    }
    agreements = [
        row.doc_ids == oracle_by_query[row.query_key].doc_ids for row in selected
    ]
    return {
        "Queries": len(selected),
        "R@5": float(np.mean([row.r5 for row in selected])),
        "Actual_Operator_Work": float(np.mean([row.cost for row in selected])),
        "Risk_Violation": float(np.mean([row.risk for row in selected])),
        "Oracle_Regret": float(np.mean([
            oracle_by_query[row.query_key].r5 - row.r5 for row in selected
        ])),
        "Oracle_Agreement": float(np.mean(agreements)),
        "selected_plan_counts": dict(Counter(row.plan_family for row in selected)),
    }


def calibrate_policy(oof_rows: Sequence[PlanRow], max_risk: float) -> Dict[str, Any]:
    candidates = []
    for risk_cap in (0.01, 0.025, 0.05, 0.10, 0.20, 0.40, 0.80, 1.0):
        for cost_multiplier in (0.75, 1.0, 1.25, 1.5, 2.0):
            selected = select_rows(oof_rows, "Full CBO", risk_cap, cost_multiplier)
            report = summarize_selection(selected, oof_rows)
            if report["Risk_Violation"] <= max_risk + 1e-12:
                candidates.append((
                    report["R@5"],
                    -report["Oracle_Regret"],
                    -report["Actual_Operator_Work"],
                    -risk_cap,
                    -cost_multiplier,
                    risk_cap,
                    cost_multiplier,
                    report,
                ))
    if not candidates:
        return {"risk_cap": 1.0, "cost_multiplier": 2.0, "feasible": False}
    best = max(candidates, key=lambda item: item[:5])
    return {
        "risk_cap": best[5],
        "cost_multiplier": best[6],
        "feasible": True,
        "oof_report": best[7],
        "max_oof_risk_violation": max_risk,
    }


def macro_query_spearman(rows: Sequence[PlanRow], prediction: str, actual: str) -> float:
    values = []
    for candidates in grouped(rows).values():
        predicted = [safe_float(getattr(row, prediction)) for row in candidates]
        observed = [safe_float(getattr(row, actual)) for row in candidates]
        if len(set(predicted)) < 2 or len(set(observed)) < 2:
            continue
        correlation = spearmanr(predicted, observed).statistic
        if math.isfinite(correlation):
            values.append(float(correlation))
    return float(np.mean(values)) if values else float("nan")


def prediction_metrics(rows: Sequence[PlanRow]) -> list[Dict[str, Any]]:
    actual_u = np.asarray([row.r5 for row in rows])
    pred_u = np.asarray([row.pred_utility for row in rows])
    actual_c = np.asarray([row.cost for row in rows])
    pred_c = np.asarray([row.pred_cost for row in rows])
    actual_r = np.asarray([row.risk for row in rows])
    pred_r = np.asarray([row.pred_risk for row in rows])
    risk_auc = roc_auc_score(actual_r, pred_r) if len(set(actual_r.tolist())) == 2 else float("nan")
    return [
        {
            "Target": "Utility",
            "Label": "actual Recall@5 of the complete slate",
            "MAE": mean_absolute_error(actual_u, pred_u),
            "RMSE": math.sqrt(mean_squared_error(actual_u, pred_u)),
            "Within_Query_Spearman": macro_query_spearman(rows, "pred_utility", "r5"),
            "AUROC": "",
            "Brier": "",
        },
        {
            "Target": "Cost",
            "Label": "trace-derived unit-weight operator work",
            "MAE": mean_absolute_error(actual_c, pred_c),
            "RMSE": math.sqrt(mean_squared_error(actual_c, pred_c)),
            "Within_Query_Spearman": macro_query_spearman(rows, "pred_cost", "cost"),
            "AUROC": "",
            "Brier": "",
        },
        {
            "Target": "Risk",
            "Label": "R@2 or R@5 regression relative to the fixed safe plan",
            "MAE": mean_absolute_error(actual_r, pred_r),
            "RMSE": math.sqrt(mean_squared_error(actual_r, pred_r)),
            "Within_Query_Spearman": "",
            "AUROC": risk_auc,
            "Brier": brier_score_loss(actual_r, pred_r),
        },
    ]


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-trace", type=Path, required=True)
    parser.add_argument("--test-trace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", default="2wikimultihopqa")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-oof-risk-violation", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    train_rows = load_portfolio(args.train_trace.resolve(), args.dataset)
    test_rows = load_portfolio(args.test_trace.resolve(), args.dataset)
    if len(grouped(train_rows)) < args.folds or not test_rows:
        raise RuntimeError("insufficient train/test portfolios")
    attach_risk_labels(train_rows)
    attach_risk_labels(test_rows)

    oof_rows = grouped_oof(train_rows, args.folds, args.seed)
    policy = calibrate_policy(oof_rows, args.max_oof_risk_violation)
    fit_predict(train_rows, test_rows, args.seed + 1000)

    selectors = ("Random", "Fixed Plan", "Utility-only", "Full CBO")
    selection_rows = []
    per_query_rows = []
    for selector in selectors:
        chosen = select_rows(
            test_rows,
            selector,
            float(policy["risk_cap"]),
            float(policy["cost_multiplier"]),
        )
        report = summarize_selection(chosen, test_rows)
        selection_rows.append({
            "Method": selector,
            "R@5": report["R@5"],
            "Actual_Cost": report["Actual_Operator_Work"],
            "Risk_Violation": report["Risk_Violation"],
            "Oracle_Regret": report["Oracle_Regret"],
            "Oracle_Agreement": report["Oracle_Agreement"],
            "Queries": report["Queries"],
        })
        for row in chosen:
            per_query_rows.append({
                "Method": selector,
                "query_key": row.query_key,
                "plan_family": row.plan_family,
                "plan_instance": row.plan_instance,
                "R@2": row.r2,
                "R@5": row.r5,
                "Actual_Cost": row.cost,
                "Risk_Violation": row.risk,
                "Predicted_Utility": row.pred_utility,
                "Predicted_Cost": row.pred_cost,
                "Predicted_Risk": row.pred_risk,
            })

    accuracy_rows = prediction_metrics(test_rows)
    write_csv(output / "physical_plan_selection_2wiki.csv", selection_rows)
    write_csv(output / "cost_model_accuracy_2wiki.csv", accuracy_rows)
    write_csv(output / "physical_plan_selection_per_query_2wiki.csv", per_query_rows)
    contract = {
        "schema_version": "execution_matched_physical_plan_analysis_v1",
        "dataset": args.dataset,
        "train_trace": str(args.train_trace.resolve()),
        "test_trace": str(args.test_trace.resolve()),
        "train_queries": len(grouped(train_rows)),
        "test_queries": len(grouped(test_rows)),
        "train_candidate_slates": len(train_rows),
        "test_candidate_slates": len(test_rows),
        "feature_names": list(FEATURE_NAMES),
        "feature_contract": "query, logical-plan, ontology-catalog, and plan-template features available before selected-plan commitment",
        "utility_label": "complete-slate Recall@5; gold evidence is offline-only",
        "cost_label": {
            "name": "Operator-Work Cost",
            "definition": "unit-weight sum of encoded queries, index probes, graph expansions, join comparisons, and candidates materialized",
            "wall_clock_used": False,
            "tokens_used": False,
        },
        "risk_label": "1 iff a plan regresses R@2 or R@5 relative to the fixed safe plan",
        "oracle_usage": "post-hoc regret and agreement only",
        "policy": policy,
        "selection": selection_rows,
        "prediction_accuracy": accuracy_rows,
    }
    (output / "experiment_contract_and_results.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(contract, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
