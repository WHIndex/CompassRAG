"""Atomic execution of a certified ontology-derived evidence plan.

The ontology plan is a first-class physical plan.  When its route, proof, and
deployment certificate pass, its complete top-k is sent to the reader.  The
semantic/fact ranking is retained only as an atomic fallback; passages from the
two plans are never blended in the reader window.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from .query_plan_complete_evidence_plan_optimizer import (
    PLAN_ONTOLOGY,
    materialize_complete_plan_portfolio,
)


POLICY_TYPE = "certified_ontology_direct_policy"
SCHEMA_VERSION = "certified_ontology_direct_execution_v1"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _ids(values: Iterable[Any]) -> List[int]:
    result: List[int] = []
    seen = set()
    for value in values:
        passage_idx = _safe_int(value, -1)
        if passage_idx < 0 or passage_idx in seen:
            continue
        seen.add(passage_idx)
        result.append(passage_idx)
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _matched_schema_ids(query_plan_log: Mapping[str, Any]) -> List[str]:
    result: List[str] = []
    for item in query_plan_log.get("matched_plans", []) or []:
        if not isinstance(item, Mapping):
            continue
        plan_id = str(item.get("id", "") or "")
        if plan_id and plan_id not in result:
            result.append(plan_id)
    if result:
        return result
    condition = query_plan_log.get("query_condition", {}) or {}
    for item in condition.get("compiler_schema_plans", []) or []:
        if not isinstance(item, Mapping):
            continue
        plan_id = str(item.get("id", "") or "")
        if plan_id and plan_id not in result:
            result.append(plan_id)
    return result


def _monotone_scores(count: int, source_scores: Sequence[float]) -> List[float]:
    values = [_safe_float(value) for value in source_scores]
    ceiling = max(values, default=1.0)
    floor = min(values, default=0.0)
    step = max(abs(ceiling - floor), 1.0) / max(count, 1)
    return [ceiling - rank * step for rank in range(count)]


class CertifiedOntologyDirectExecutor:
    """Commit an ontology-only evidence slate or fall back atomically."""

    def __init__(
        self,
        *,
        policy_path: str = "",
        ontology_path: str = "",
        top_k: int = 5,
        require_policy: bool = True,
        min_matched_plans: int = 1,
        min_strict_paths: int = 1,
        min_relation_coverage: float = 0.35,
        min_answer_type_fraction: float = 0.20,
        require_complete_proof: bool = True,
    ) -> None:
        self.policy_path = str(policy_path or "")
        self.ontology_path = str(ontology_path or "")
        self.top_k = max(int(top_k), 2)
        self.require_policy = bool(require_policy)
        self.min_matched_plans = max(int(min_matched_plans), 1)
        self.min_strict_paths = max(int(min_strict_paths), 0)
        self.min_relation_coverage = max(float(min_relation_coverage), 0.0)
        self.min_answer_type_fraction = max(float(min_answer_type_fraction), 0.0)
        self.require_complete_proof = bool(require_complete_proof)
        self.policy: Dict[str, Any] = {}
        if self.policy_path:
            path = Path(self.policy_path).expanduser().resolve()
            if path.is_file():
                value = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(value, dict):
                    raise ValueError("ontology direct policy must be a JSON object")
                if value.get("type") != POLICY_TYPE:
                    raise ValueError(
                        f"incompatible ontology direct policy type: {value.get('type')!r}"
                    )
                self.policy = value
        self._validate_artifact_contract()

    def _validate_artifact_contract(self) -> None:
        if not self.policy:
            if self.require_policy:
                raise ValueError("certified ontology direct execution requires a policy")
            return
        expected = str(self.policy.get("ontology_sha256", "") or "")
        if expected:
            ontology = Path(self.ontology_path).expanduser().resolve()
            if not ontology.is_file():
                raise ValueError("cannot validate ontology hash: ontology file is missing")
            observed = _sha256(ontology)
            if observed != expected:
                raise ValueError(
                    "ontology direct policy was calibrated for a different frozen ontology "
                    f"(policy={expected[:12]} runtime={observed[:12]})"
                )

    def _policy_gate(self, matched_ids: Sequence[str]) -> Tuple[bool, str]:
        if not self.policy:
            return (not self.require_policy), (
                "policy_optional" if not self.require_policy else "policy_missing"
            )
        if not bool((self.policy.get("certification", {}) or {}).get("passed", False)):
            return False, "q_val_policy_not_certified"
        mask = self.policy.get("deployment_mask", {}) or {}
        certified = set(mask.get("certified_schema_ids", []) or [])
        rejected = set(mask.get("rejected_schema_ids", []) or [])
        if not certified:
            return False, "no_q_val_certified_schemas"
        if rejected.intersection(matched_ids) and not certified.intersection(matched_ids):
            return False, "matched_schema_rejected_by_q_val"
        if certified and not certified.intersection(matched_ids):
            return False, "no_q_val_certified_schema_matched"
        return True, "q_val_certified"

    def execute(
        self,
        current_doc_ids: Sequence[int],
        current_doc_scores: Sequence[float],
        query_plan_log: Mapping[str, Any],
        *,
        shadow_only: bool = False,
    ) -> Tuple[List[int], List[float], Dict[str, Any]]:
        portfolio = materialize_complete_plan_portfolio(
            query_plan_log,
            current_doc_ids,
            current_doc_scores,
            top_k=self.top_k,
        )
        ontology_plan = (portfolio.get("plans", {}) or {}).get(PLAN_ONTOLOGY, {}) or {}
        ontology_ids = _ids(ontology_plan.get("doc_ids", []) or [])
        proof_slate = query_plan_log.get("full_aperture_proof_slate", {}) or {}
        candidates = [
            item for item in proof_slate.get("candidates", []) or []
            if isinstance(item, Mapping) and bool(item.get("has_proof_signal"))
        ]
        relation_coverages = [
            _safe_float(item.get("relation_coverage_score")) for item in candidates
        ]
        answer_compatible = sum(
            int(bool(item.get("answer_type_compatible"))) for item in candidates
        )
        mean_relation_coverage = (
            sum(relation_coverages) / len(relation_coverages)
            if relation_coverages else 0.0
        )
        answer_type_fraction = answer_compatible / max(len(candidates), 1)
        matched_ids = _matched_schema_ids(query_plan_log)
        policy_passed, policy_reason = self._policy_gate(matched_ids)

        checks = {
            "query_plan_enabled": bool(query_plan_log.get("enabled", False)),
            "matched_schema": len(matched_ids) >= self.min_matched_plans,
            "strict_executable_path": _safe_int(
                query_plan_log.get("strict_executable_path_count"), 0
            ) >= self.min_strict_paths,
            "complete_ontology_topk": bool(ontology_plan.get("valid"))
            and len(ontology_ids) >= self.top_k,
            "complete_proof": (
                _safe_int(proof_slate.get("local_proof_path_count"), 0) > 0
                or _safe_int(proof_slate.get("complete_single_passage_proof_count"), 0) > 0
            ) if self.require_complete_proof else True,
            "relation_coverage": mean_relation_coverage >= self.min_relation_coverage,
            "answer_type_compatibility": answer_type_fraction >= self.min_answer_type_fraction,
            "deployment_policy": policy_passed,
        }
        failed = [name for name, passed in checks.items() if not passed]
        certified = not failed
        committed = bool(certified and not shadow_only)
        baseline_ids = _ids(current_doc_ids)
        baseline_scores = [_safe_float(value) for value in current_doc_scores]
        if committed:


            output_ids = _ids([
                *ontology_ids,
                *[
                    item.get("passage_idx")
                    for item in candidates
                    if isinstance(item, Mapping)
                ],
            ])
            output_scores = _monotone_scores(len(output_ids), baseline_scores)
        else:
            output_ids = baseline_ids
            output_scores = baseline_scores[: len(output_ids)]

        log = {
            **portfolio,
            "component": "certified_ontology_direct_executor",
            "schema_version": SCHEMA_VERSION,
            "enabled": True,
            "gold_free": True,
            "policy_path": self.policy_path,
            "policy_loaded": bool(self.policy),
            "policy_reason": policy_reason,
            "matched_schema_ids": matched_ids,
            "checks": checks,
            "failed_checks": failed,
            "certification_status": "CERTIFIED" if certified else "REJECTED",
            "mean_relation_coverage": mean_relation_coverage,
            "answer_type_compatible_fraction": answer_type_fraction,
            "selection_contract": "ontology_topk_or_atomic_baseline_fallback",
            "reader_context_contract": "ontology_only_when_committed",
            "baseline_used_as_supplement": False,
            "shadow_only": bool(shadow_only),
            "selected_plan": PLAN_ONTOLOGY if committed else "baseline_fallback",
            "fallback_reason": "" if committed else ",".join(failed) or "shadow_only",
            "atomic_execution": {
                "executed": committed,
                "action_kind": "replace_reader_slate_with_ontology_plan",
                "before_topk": baseline_ids[: self.top_k],
                "after_topk": output_ids[: self.top_k],
            },
        }
        return output_ids, output_scores, log
