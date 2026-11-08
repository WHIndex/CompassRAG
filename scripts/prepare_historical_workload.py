#!/usr/bin/env python3
"""Build a leakage-free historical workload and entity-disjoint cross-validation folds."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from workload import (
    DATASET_LIMITS,
    DATASET_SCAN_LIMITS,
    DEFAULT_DATASETS,
    ROOT,
    anchor_entities,
    choose_source,
    corpus_rows,
    exact_gold_coverage,
    load_rows,
    load_eval_identity,
    normalize_row,
    normalize_text,
    priority,
    question,
    select_rows,
    sha256_file,
    split_stratum,
    stable_id,
    write_json,
)


def select_nested_historical_rows(
    dataset: str,
    source: Path,
    limit: int,
    seed: str,
    eval_ids: set[str],
    eval_questions: set[str],
) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Select a deterministic nested subset from an already frozen Q_H."""
    eligible: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()
    overlap = duplicates = 0
    source_rows = list(load_rows(source))
    for row in source_rows:
        identity = stable_id(row) or normalize_text(question(row))
        if not identity or identity in seen_ids:
            duplicates += 1
            continue
        seen_ids.add(identity)
        if identity in eval_ids or normalize_text(question(row)) in eval_questions:
            overlap += 1
            continue
        eligible.append(normalize_row(dataset, row))
    eligible.sort(key=lambda row: priority(seed, dataset, row))
    return eligible[:limit], {
        "scanned_rows": len(source_rows),
        "eval_overlap_excluded": overlap,
        "duplicate_rows_excluded": duplicates,
        "canonical_history_rows": len(eligible),
    }


def assign_entity_disjoint_folds(
    rows: Sequence[Dict[str, Any]],
    dataset: str,
    fold_count: int,
    seed: str,
) -> List[int]:
    """Greedily balance strata while keeping shared support entities together."""
    parent = list(range(len(rows)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    entity_owner: Dict[str, int] = {}
    for index, row in enumerate(rows):
        for entity in anchor_entities(dataset, row):
            union(index, entity_owner.setdefault(entity, index))

    components: Dict[int, List[int]] = defaultdict(list)
    for index in range(len(rows)):
        components[find(index)].append(index)

    total_strata = Counter(split_stratum(dataset, row) for row in rows)
    target_size = len(rows) / max(fold_count, 1)
    target_strata = {
        label: count / max(fold_count, 1) for label, count in total_strata.items()
    }
    fold_sizes = [0] * fold_count
    fold_strata = [Counter() for _ in range(fold_count)]
    assignments = [-1] * len(rows)
    ordered = sorted(
        components.values(),
        key=lambda indices: (
            -len(indices),
            min(total_strata[split_stratum(dataset, rows[index])] for index in indices),
            priority(seed, dataset, rows[indices[0]]),
        ),
    )
    for indices in ordered:
        labels = Counter(split_stratum(dataset, rows[index]) for index in indices)

        def score(fold: int) -> tuple[float, int]:
            size_need = (target_size - fold_sizes[fold]) / max(target_size, 1.0)
            stratum_need = sum(
                count * (target_strata[label] - fold_strata[fold][label])
                / max(target_strata[label], 1.0)
                for label, count in labels.items()
            ) / max(sum(labels.values()), 1)
            digest = int.from_bytes(
                hashlib.sha256(
                    f"{seed}|{dataset}|{fold}|{indices[0]}".encode("utf-8")
                ).digest(),
                "big",
            )
            return (size_need + stratum_need) / 2.0, -digest

        destination = max(range(fold_count), key=score)
        for index in indices:
            assignments[index] = destination
        fold_sizes[destination] += len(indices)
        fold_strata[destination].update(labels)
    if any(value < 0 for value in assignments):
        raise RuntimeError(f"incomplete cross-validation assignment for {dataset}")
    return assignments


def fold_overlap_report(
    rows: Sequence[Mapping[str, Any]],
    assignments: Sequence[int],
    fold_count: int,
    dataset: str,
) -> Dict[str, Any]:
    entities = []
    for fold in range(fold_count):
        selected = [
            anchor_entities(dataset, row)
            for index, row in enumerate(rows)
            if assignments[index] == fold
        ]
        entities.append(set().union(*selected) if selected else set())
    return {
        f"fold_{left}_fold_{right}": len(entities[left] & entities[right])
        for left in range(fold_count)
        for right in range(left + 1, fold_count)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--deployment-dir", default="reproduce/query_eval_full_1000")
    parser.add_argument(
        "--fixed-corpus-dir",
        default="",
        help="Optional corpus directory held constant across historical-query budgets.",
    )
    parser.add_argument(
        "--fixed-historical-query-dir",
        default="",
        help=(
            "Optional canonical Q_H directory. Budgeted workloads are nested "
            "deterministic subsets of these frozen historical queries."
        ),
    )
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--max-history-rows", type=int, default=3000)
    parser.add_argument("--max-scan-rows", type=int, default=30000)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", default="historical-grouped-oof")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    if args.folds < 3:
        raise SystemExit("--folds must be at least 3")

    output = Path(args.output_dir).expanduser().resolve()
    deployment = (ROOT / args.deployment_dir).resolve()
    fixed_corpus_dir = (
        Path(args.fixed_corpus_dir).expanduser().resolve()
        if args.fixed_corpus_dir else None
    )
    fixed_historical_query_dir = (
        Path(args.fixed_historical_query_dir).expanduser().resolve()
        if args.fixed_historical_query_dir else None
    )
    datasets = [value.strip() for value in args.datasets.split(",") if value.strip()]
    manifest: Dict[str, Any] = {
        "type": "historical_query_grouped_oof_contract",
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": "all_historical_queries_for_ontology_plus_entity_disjoint_kfold_oof_calibration",
        "historical_query_role": "offline_ontology_induction_and_grouped_oof_cost_model_calibration_only",
        "test_query_role": "final_evaluation_only_never_read_by_induction_or_calibration",
        "fold_count": args.folds,
        "seed": args.seed,
        "datasets": {},
    }
    all_valid = True
    for dataset in datasets:
        source = (
            fixed_historical_query_dir / f"{dataset}.json"
            if fixed_historical_query_dir is not None
            else choose_source(dataset)
        )
        if not source.is_file():
            raise FileNotFoundError(
                f"canonical historical queries are missing for {dataset}: {source}"
            )
        eval_ids, eval_questions, eval_path = load_eval_identity(dataset, deployment)
        limit = min(args.max_history_rows, DATASET_LIMITS.get(dataset, args.max_history_rows))
        scan_limit = min(args.max_scan_rows, DATASET_SCAN_LIMITS.get(dataset, args.max_scan_rows))
        if fixed_historical_query_dir is not None:
            rows, selection = select_nested_historical_rows(
                dataset, source, limit, args.seed, eval_ids, eval_questions
            )
        else:
            rows, selection = select_rows(
                dataset,
                source,
                limit,
                args.seed,
                eval_ids,
                eval_questions,
                scan_limit,
            )
        rows = sorted(rows, key=lambda row: priority(args.seed, dataset, row))
        assignments = assign_entity_disjoint_folds(
            rows, dataset, args.folds, args.seed
        )
        q_h_path = output / "q_h" / f"{dataset}.json"
        write_json(q_h_path, rows)
        for fold in range(args.folds):
            heldout = [row for index, row in enumerate(rows) if assignments[index] == fold]
            train = [row for index, row in enumerate(rows) if assignments[index] != fold]
            write_json(output / "folds" / f"fold_{fold}" / "train" / f"{dataset}.json", train)
            write_json(output / "folds" / f"fold_{fold}" / "heldout" / f"{dataset}.json", heldout)

        if fixed_corpus_dir is not None:
            fixed_corpus_path = fixed_corpus_dir / f"{dataset}_corpus.json"
            if not fixed_corpus_path.is_file():
                raise FileNotFoundError(
                    f"fixed historical corpus is missing for {dataset}: {fixed_corpus_path}"
                )
            corpus = json.loads(fixed_corpus_path.read_text(encoding="utf-8"))
            if not isinstance(corpus, list):
                raise ValueError(f"fixed corpus is not a row list: {fixed_corpus_path}")
        else:
            fixed_corpus_path = None
            corpus = corpus_rows(dataset, rows)
        corpus_path = output / "corpora" / f"{dataset}_corpus.json"
        write_json(corpus_path, corpus)
        coverage = exact_gold_coverage(dataset, rows, corpus)
        identities = [stable_id(row) for row in rows]
        overlap = sum(identity in eval_ids for identity in identities if identity)
        question_overlap = selection["eval_overlap_excluded"]
        valid = bool(rows and overlap == 0 and coverage["gold_document_exact_coverage"] >= 0.999999)
        all_valid = all_valid and valid
        manifest["datasets"][dataset] = {
            "source": str(source),
            "source_sha256": sha256_file(source),
            "historical_query_source_protocol": (
                "nested_subset_of_canonical_q_h"
                if fixed_historical_query_dir is not None
                else "deterministic_selection_from_raw_training_data"
            ),
            "fixed_historical_query_source": str(source) if fixed_historical_query_dir else "",
            "deployment_query_file": str(eval_path) if eval_path else "",
            "historical_query_file": str(q_h_path),
            "historical_query_sha256": sha256_file(q_h_path),
            "corpus_file": str(corpus_path),
            "corpus_sha256": sha256_file(corpus_path),
            "evidence_corpus_protocol": (
                "fixed_external_corpus" if fixed_corpus_path is not None
                else "historical_query_support_union"
            ),
            "fixed_corpus_source": str(fixed_corpus_path or ""),
            "query_count": len(rows),
            "fold_sizes": {
                str(fold): sum(value == fold for value in assignments)
                for fold in range(args.folds)
            },
            "query_fold": assignments,
            "support_entity_fold_overlap": fold_overlap_report(
                rows, assignments, args.folds, dataset
            ),
            "test_identity_overlap": overlap,
            "excluded_test_overlap": question_overlap,
            "selection": selection,
            "coverage": coverage,
            "valid": valid,
        }
        print(
            f"[{dataset}] q_h={len(rows)} folds="
            f"{manifest['datasets'][dataset]['fold_sizes']} corpus={len(corpus)} valid={valid}",
            flush=True,
        )

    manifest["valid"] = all_valid
    write_json(output / "workload_contract.json", manifest)
    if args.strict and not all_valid:
        raise SystemExit("historical cross-validation workload contract failed")


if __name__ == "__main__":
    main()
