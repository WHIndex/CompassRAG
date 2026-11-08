#!/usr/bin/env python3
"""Freeze an ontology induced from historical queries without reading Q_test."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
COMPASSRAG_SRC = ROOT / "src" / "compassrag"
for value in (str(ROOT / "scripts"), str(COMPASSRAG_SRC)):
    if value not in sys.path:
        sys.path.insert(0, value)

from query_plan_ontology import build_hierarchical_routing_model
from validate_ontology import sanitize_plan


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def empirical_support(plan: Mapping[str, Any]) -> dict[str, Any]:
    contrastive = dict(plan.get("contrastive_induction", {}) or {})
    quality = dict(
        plan.get("construction_quality", {})
        or plan.get("multiagent_quality", {})
        or {}
    )
    grounding = dict(plan.get("symbolic_grounding", {}) or {})
    negative_types = list(contrastive.get("hard_negative_types", []) or [])
    source_clusters = list(contrastive.get("cluster_ids", []) or [])
    examples = [
        item for item in plan.get("examples", []) or [] if isinstance(item, Mapping)
    ]
    source_query_ids = [
        str(item.get("id", "") or "") for item in examples if item.get("id")
    ]
    hard_negative_ids = sorted({
        str(negative.get("doc_id", "") or "")
        for example in examples
        for negative in (
            (example.get("contrastive_cq", {}) or {}).get("hard_negatives", []) or []
        )
        if isinstance(negative, Mapping) and negative.get("doc_id")
    })
    return {
        "source_split": "q_ind",
        "positive_support": int(plan.get("support", 0) or 0),
        "grounded_support": int(grounding.get("supported_example_count", 0) or 0),
        "positive_coverage": float(grounding.get("path_coverage", 0.0) or 0.0),
        "hard_negative_types": negative_types,
        "hard_negative_type_count": len(negative_types),
        "hard_negative_count": len(hard_negative_ids),
        "hard_negative_test_count": int(
            quality.get("hard_negative_test_count", 0) or 0
        ),
        "hard_negative_rejection_rate": quality.get(
            "hard_negative_rejection_rate"
        ),
        "execution_eligible": bool(plan.get("execution_eligible", False)),
        "critic_risk": str(quality.get("risk_level", "unknown") or "unknown"),
        "source_clusters": source_clusters,
        "source_query_count": int(plan.get("support", 0) or 0),
        "repair_iterations": int(plan.get("repair_iterations", 0) or 0),
        "provenance": {
            "source_query_ids": source_query_ids,
            "source_query_id_hashes": [
                hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
                for value in source_query_ids
            ],
            "source_cq_ids": source_query_ids,
            "hard_negative_ids": hard_negative_ids,
            "atomic_proposal": dict(plan.get("atomic_proposal", {}) or {}),
            "aggregator_decision": dict(
                plan.get("hierarchical_aggregation", {}) or {}
            ),
            "verifier_result": dict(plan.get("contrastive_verification", {}) or {}),
            "critic_warnings": list(plan.get("critic_warnings", []) or []),
            "repairs": list(plan.get("repair_operations", []) or []),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    history = parser.add_mutually_exclusive_group(required=True)
    history.add_argument("--q-ind")
    history.add_argument("--q-h")
    parser.add_argument("--contrastive-cq", default="")
    parser.add_argument(
        "--runtime-eligible-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Freeze only plans that passed multi-agent execution eligibility checks.",
    )
    parser.add_argument(
        "--min-runtime-support",
        type=int,
        default=0,
        help="Minimum historical-query support required by a frozen runtime plan.",
    )
    parser.add_argument(
        "--rejected-output",
        default="",
        help="Optional JSON audit file for schemas excluded by the runtime quality gate.",
    )
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    history_path = Path(args.q_h or args.q_ind).expanduser().resolve()
    source_split = "q_h" if args.q_h else "q_ind"
    cq_path = Path(args.contrastive_cq).expanduser().resolve() if args.contrastive_cq else None
    for path in (input_path, history_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if cq_path is not None and not cq_path.is_file():
        raise FileNotFoundError(cq_path)

    source = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(source, dict):
        raise ValueError("induced ontology must be a JSON object")
    if cq_path is not None:
        cq_payload = json.loads(cq_path.read_text(encoding="utf-8"))
        if cq_payload.get("source_split") != source_split:
            raise ValueError(
                f"contrastive CQ split {cq_payload.get('source_split')} does not match {source_split}"
            )
    else:
        cq_payload = {}

    frozen_plans = []
    sanitation = []
    rejected_plans = []
    for raw_plan in source.get("plans", []) or []:
        if not isinstance(raw_plan, Mapping):
            continue
        rejection_reasons = []
        if args.runtime_eligible_only and not bool(
            raw_plan.get("execution_eligible", False)
        ):
            rejection_reasons.append("not_execution_eligible")
        support_count = int(raw_plan.get("support", 0) or 0)
        if support_count < max(args.min_runtime_support, 0):
            rejection_reasons.append("insufficient_historical_support")
        if rejection_reasons:
            rejected_plans.append({
                "schema_id": str(raw_plan.get("id", "") or ""),
                "support": support_count,
                "execution_eligible": bool(
                    raw_plan.get("execution_eligible", False)
                ),
                "reasons": rejection_reasons,
            })
            continue
        plan, stats = sanitize_plan(raw_plan)
        support = empirical_support(raw_plan)
        support["source_split"] = source_split
        plan["construction_time_statistics"] = support
        plan["structure_source"] = f"{source_split}_only"
        plan["schema_definition_sha256"] = stable_hash({
            "id": plan.get("id"),
            "intent": plan.get("intent"),
            "anchor_slots": plan.get("anchor_slots", []),
            "path": plan.get("path", []),
            "answer_type": plan.get("answer_type"),
            "answer_constraints": plan.get("answer_constraints", {}),
            "integrity_constraints": plan.get("integrity_constraints", {}),
        })
        frozen_plans.append(plan)
        sanitation.append({"schema_id": plan.get("id", ""), **stats})
    if not frozen_plans:
        raise ValueError(f"{source_split} produced no executable schema to freeze")

    dataset = str(source.get("dataset", "") or "")
    frozen = deepcopy(source)
    frozen.update({
        "version": max(int(source.get("version", 1) or 1), 4),
        "kind": "frozen_contrastive_evidence_ontology",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "builder": "scripts/freeze_ontology.py",
        "structure_frozen": True,
        "structure_source_split": source_split,
        "historical_query_sha256": sha256_file(history_path),
        "contrastive_cq_sha256": sha256_file(cq_path) if cq_path else "",
        "q_cal_read": False,
        "q_val_read": False,
        "q_test_read": False,
        "plans": frozen_plans,
        "routing_model": build_hierarchical_routing_model(frozen_plans, dataset),
        "freeze_diagnostics": {
            "source_schema_count": len(source.get("plans", []) or []),
            "schema_count": len(frozen_plans),
            "rejected_schema_count": len(rejected_plans),
            "runtime_eligible_only": bool(args.runtime_eligible_only),
            "min_runtime_support": max(args.min_runtime_support, 0),
            "runtime_examples_removed": True,
            "sanitation": sanitation,
            "cq_record_count": len(cq_payload.get("records", []) or []),
        },
    })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(frozen, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_path)
    if args.rejected_output:
        rejected_output = Path(args.rejected_output).expanduser().resolve()
        rejected_output.parent.mkdir(parents=True, exist_ok=True)
        rejected_output.write_text(
            json.dumps({
                "source_ontology": str(input_path),
                "runtime_eligible_only": bool(args.runtime_eligible_only),
                "min_runtime_support": max(args.min_runtime_support, 0),
                "rejected_schema_count": len(rejected_plans),
                "schemas": rejected_plans,
            }, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(f"frozen_ontology={output_path}")
    print(f"frozen_ontology_sha256={sha256_file(output_path)}")
    print(f"schema_count={len(frozen_plans)}")
    print(f"rejected_schema_count={len(rejected_plans)}")


if __name__ == "__main__":
    main()
