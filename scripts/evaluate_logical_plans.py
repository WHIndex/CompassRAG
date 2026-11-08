#!/usr/bin/env python3
"""Evaluate the semantic fidelity of frozen logical evidence plans."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
import sys

SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from compassrag.remote import RemoteEndpoint
DEFAULT_OUTPUT = ROOT / "outputs" / "logical_plan_quality"
METRICS = (
    "entity_role_correct",
    "relation_dependency_correct",
    "variable_binding_correct",
    "structural_constraint_preservation",
)

SYSTEM_PROMPT = """You are an independent evaluator of logical evidence plans for multi-hop question answering. Evaluate only whether the plan faithfully represents the semantics and structure of the natural-language question. Do not evaluate retrieval success, answer correctness, writing style, or exact relation wording. Treat synonymous relation names as equivalent.

Score these four criteria independently as 0 or 1:
1. entity_role_correct: anchors, intermediate entity roles, projected answer role, and answer type agree with the question.
2. relation_dependency_correct: required relations are present and ordered or branched according to the question's dependency structure.
3. variable_binding_correct: intermediate variables, joins, and operator inputs/outputs bind the same entities consistently; do not require an unnecessary join when none is semantically needed.
4. structural_constraint_preservation: hop count, chain/branch/comparison/intersection shape, directionality, and explicit structural constraints preserve the question.

Return one JSON object only, with integer fields for the four criteria, a confidence number from 0 to 1, and a concise rationale. Be conservative when a plan omits a relation or connects the correct relations through the wrong intermediate variable."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trace",
        action="append",
        required=True,
        metavar="DATASET=PATH",
        help="Dataset name and logical-plan trace. Repeat for each dataset.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--judge-model", default=os.environ.get("LOGICAL_PLAN_JUDGE_MODEL", "reader-model"))
    parser.add_argument("--concurrency", type=int, default=int(os.environ.get("LOGICAL_PLAN_JUDGE_CONCURRENCY", "8")))
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument("--manual-sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def compact_plan(trace: dict[str, Any]) -> dict[str, Any]:
    executor = trace.get("factorized_query_executor", {}) or {}
    plan = executor.get("logical_evidence_plan", {}) or {}
    plan_ir = executor.get("logical_plan_ir", {}) or {}
    if not plan_ir:
        plan_ir = executor.get("typed_contract", {}).get("typed_plan_ir", {}) or {}
    return {
        "plan_id": plan.get("plan_id", plan_ir.get("plan_id", "")),
        "query_shape": plan.get("query_shape", plan_ir.get("topology", "")),
        "anchors": plan.get("anchors", (plan_ir.get("initial_bindings", {}) or {})),
        "relations": plan.get("relations", []),
        "answer_type": plan.get("answer_type", plan_ir.get("answer_type", "")),
        "required_hop_count": plan.get("required_hop_count", len(plan_ir.get("edges", []) or [])),
        "required_branch_count": plan.get("required_branch_count", len(plan_ir.get("branch_ids", []) or []) or 1),
        "operators": plan.get("operators", []),
        "typed_edges": plan_ir.get("edges", []),
        "projection": plan_ir.get("projection", {}),
        "proof_obligations": plan.get("proof_obligations", []),
        "admissible": plan.get("admissible"),
    }


def trace_paths(values: list[str]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for value in values:
        dataset, separator, path = value.partition("=")
        if not separator or not dataset.strip() or not path.strip():
            raise ValueError(f"invalid --trace value: {value!r}")
        paths[dataset.strip()] = Path(path).expanduser().resolve()
    return paths


def source_rows(paths: dict[str, Path], limit: int) -> list[dict[str, Any]]:
    output = []
    for dataset, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"logical-plan trace is missing: {path}")
        traces = jsonl(path)
        if limit:
            traces = traces[:limit]
        for offset, trace in enumerate(traces):
            plan = compact_plan(trace)
            if not plan.get("plan_id") and not plan.get("operators") and not plan.get("typed_edges"):
                raise RuntimeError(f"logical plan missing for {dataset} row {offset}")
            output.append({
                "dataset": dataset,
                "query_idx": int(trace.get("query_idx", offset)),
                "question": str(trace.get("query", "")),
                "logical_plan": plan,
                "source_trace": str(path),
            })
    return output


def parse_judgment(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    match = re.search(r"\{.*\}", cleaned, flags=re.S)
    if not match:
        raise ValueError("judge did not return a JSON object")
    value = json.loads(match.group(0))
    result: dict[str, Any] = {}
    for metric in METRICS:
        raw = value.get(metric)
        if isinstance(raw, bool):
            raw = int(raw)
        if isinstance(raw, str):
            raw = 1 if raw.strip().lower() in {"1", "true", "yes", "correct", "pass"} else 0
        raw = int(raw)
        if raw not in (0, 1):
            raise ValueError(f"invalid binary score for {metric}: {raw}")
        result[metric] = raw
    result["confidence"] = min(1.0, max(0.0, float(value.get("confidence", 0.5))))
    result["rationale"] = str(value.get("rationale", "")).strip()[:1200]
    return result


_thread_local = threading.local()


def client() -> RemoteEndpoint:
    value = getattr(_thread_local, "client", None)
    if value is None:
        value = RemoteEndpoint(
            api_key=os.environ["MODEL_API_KEY"],
            base_url=os.environ.get("MODEL_BASE_URL") or os.environ.get("LLM_BASE_URL"),
            timeout=float(os.environ.get("LOGICAL_PLAN_JUDGE_TIMEOUT_SECONDS", "180")),
        )
        _thread_local.client = value
    return value


def judge(row: dict[str, Any], model: str, max_retries: int) -> dict[str, Any]:
    prompt = json.dumps(
        {"question": row["question"], "logical_evidence_plan": row["logical_plan"]},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    last_error = ""
    for attempt in range(max_retries):
        try:
            response = client().chat(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=512,
            )
            choice = response["choices"][0]
            raw = choice.get("message", {}).get("content", "") or ""
            parsed = parse_judgment(raw)
            usage = response.get("usage", {}) or {}
            return {
                **row,
                **parsed,
                "status": "ok",
                "judge_model": model,
                "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
            }
        except Exception as error:
            last_error = repr(error)
            time.sleep(min(30.0, 1.5 * (2**attempt)))
    return {**row, "status": "failed", "judge_model": model, "error": last_error}


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, Any]], datasets: list[str]) -> list[dict[str, Any]]:
    summary = []
    for dataset in datasets:
        selected = [row for row in rows if row.get("dataset") == dataset and row.get("status") == "ok"]
        expected = sum(row.get("dataset") == dataset for row in rows)
        item: dict[str, Any] = {
            "dataset": dataset,
            "num_plans": expected,
            "num_judged": len(selected),
            "completion_rate": len(selected) / expected if expected else 0.0,
        }
        for metric in METRICS:
            item[metric] = sum(int(row[metric]) for row in selected) / len(selected) if selected else ""
        item["all_four_correct"] = (
            sum(all(int(row[metric]) for metric in METRICS) for row in selected) / len(selected)
            if selected else ""
        )
        summary.append(item)
    macro: dict[str, Any] = {
        "dataset": "Macro Average",
        "num_plans": sum(int(row["num_plans"]) for row in summary),
        "num_judged": sum(int(row["num_judged"]) for row in summary),
        "completion_rate": sum(float(row["completion_rate"]) for row in summary) / len(summary),
    }
    for metric in (*METRICS, "all_four_correct"):
        values = [float(row[metric]) for row in summary if row[metric] != ""]
        macro[metric] = sum(values) / len(values) if values else ""
    return [*summary, macro]


def manual_sample(rows: list[dict[str, Any]], datasets: list[str], sample_size: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    output = []
    for dataset in datasets:
        selected = [row for row in rows if row.get("dataset") == dataset and row.get("status") == "ok"]
        failures = [row for row in selected if not all(int(row[m]) for m in METRICS)]
        passes = [row for row in selected if all(int(row[m]) for m in METRICS)]
        rng.shuffle(failures)
        rng.shuffle(passes)
        failure_target = min(len(failures), sample_size // 2)
        sample = failures[:failure_target] + passes[: sample_size - failure_target]
        if len(sample) < sample_size:
            used = {(row["dataset"], row["query_idx"]) for row in sample}
            remainder = [row for row in selected if (row["dataset"], row["query_idx"]) not in used]
            sample.extend(remainder[: sample_size - len(sample)])
        for row in sample:
            output.append({
                **row,
                "human_entity_role_correct": "",
                "human_relation_dependency_correct": "",
                "human_variable_binding_correct": "",
                "human_structural_constraint_preservation": "",
                "human_note": "",
            })
    return output


def main() -> None:
    args = parse_args()
    if not os.environ.get("MODEL_API_KEY"):
        raise SystemExit("MODEL_API_KEY is not set")
    paths = trace_paths(args.trace)
    datasets = list(paths)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "logical_plan_judgments.jsonl"
    sources = source_rows(paths, args.limit)
    latest = {
        (str(row.get("dataset")), int(row.get("query_idx", -1))): row
        for row in jsonl(checkpoint)
    }
    completed = {key for key, row in latest.items() if row.get("status") == "ok"}
    pending = [row for row in sources if (row["dataset"], row["query_idx"]) not in completed]
    print(f"source_plans={len(sources)} completed={len(completed)} pending={len(pending)}", flush=True)

    lock = threading.Lock()
    with checkpoint.open("a", encoding="utf-8") as handle:
        with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
            futures = {
                pool.submit(judge, row, args.judge_model, args.max_retries): row
                for row in pending
            }
            for index, future in enumerate(as_completed(futures), 1):
                result = future.result()
                with lock:
                    handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                    handle.flush()
                print(
                    f"[{index}/{len(pending)}] {result['dataset']}:{result['query_idx']} {result['status']}",
                    flush=True,
                )

    latest = {
        (str(row.get("dataset")), int(row.get("query_idx", -1))): row
        for row in jsonl(checkpoint)
    }
    final_rows = [latest[(row["dataset"], row["query_idx"])] for row in sources if (row["dataset"], row["query_idx"]) in latest]
    ok_rows = [row for row in final_rows if row.get("status") == "ok"]
    row_fields = [
        "dataset", "query_idx", "question", *METRICS, "confidence", "rationale",
        "status", "judge_model", "prompt_tokens", "completion_tokens", "source_trace", "logical_plan",
    ]
    write_csv(args.output_dir / "logical_plan_quality_per_query.csv", ok_rows, row_fields)
    summary_rows = summarize(final_rows, datasets)
    summary_fields = ["dataset", "num_plans", "num_judged", "completion_rate", *METRICS, "all_four_correct"]
    write_csv(args.output_dir / "logical_plan_quality_summary.csv", summary_rows, summary_fields)
    audit_rows = manual_sample(ok_rows, datasets, args.manual_sample_size, args.seed)
    audit_fields = [
        "dataset", "query_idx", "question", *METRICS, "confidence", "rationale",
        "human_entity_role_correct", "human_relation_dependency_correct",
        "human_variable_binding_correct", "human_structural_constraint_preservation", "human_note",
        "judge_model", "logical_plan",
    ]
    write_csv(args.output_dir / "human_review_sample.csv", audit_rows, audit_fields)
    config = {
        "protocol": "logical-plan-semantic-quality-v1",
        "judge_model": args.judge_model,
        "concurrency": args.concurrency,
        "seed": args.seed,
        "manual_sample_size_per_dataset": args.manual_sample_size,
        "trace_paths": {key: str(value) for key, value in paths.items()},
        "rubric": list(METRICS),
        "uses_gold_answers_or_evidence": False,
    }
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"summary={args.output_dir / 'logical_plan_quality_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
