"""Bounded runtime re-optimization for ontology-guided evidence plans.

The controller is deliberately not a free-form language-model loop.  Each
agent has a typed contract: the verifier reports structural proof gaps, the
planner proposes only materialized physical plans, and the critic either
commits one complete slate or rolls the transaction back.  This makes the
runtime behavior reproducible, gold-free, and suitable for ablation.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from .query_plan_complete_evidence_plan_optimizer import (
    CompleteEvidencePlanPortfolioOptimizer,
    PLAN_BRANCH_COMPLETE_TOP2,
    PLAN_CHAIN_COMPLETION,
    PLAN_LEARNED_RESIDUAL_TOP5,
    PLAN_QUERY_LOCAL_TOP2_PAIR,
    PLAN_QUERY_LOCAL_TOP5,
    PLAN_RANK_PARTITIONED_NESTED,
    PLAN_RANK_PARTITIONED_TOP2,
    PLAN_RANK_PARTITIONED_TOP5,
    PLAN_SIGNATURE_GROUNDED,
    materialize_complete_plan_portfolio,
    recover_latent_query_contract,
)


SCHEMA_VERSION = "latent_contract_delta_critic_agent_v2"

TOP2_REPAIR_FAMILIES = {
    PLAN_BRANCH_COMPLETE_TOP2,
    PLAN_QUERY_LOCAL_TOP2_PAIR,
    PLAN_RANK_PARTITIONED_TOP2,
    PLAN_RANK_PARTITIONED_NESTED,
    PLAN_SIGNATURE_GROUNDED,
}
TOP5_REPAIR_FAMILIES = {
    PLAN_CHAIN_COMPLETION,
    PLAN_LEARNED_RESIDUAL_TOP5,
    PLAN_QUERY_LOCAL_TOP5,
    PLAN_RANK_PARTITIONED_TOP5,
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        return default if math.isnan(number) or math.isinf(number) else number
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
        try:
            passage_idx = int(value)
        except (TypeError, ValueError):
            continue
        if passage_idx < 0 or passage_idx in seen:
            continue
        seen.add(passage_idx)
        result.append(passage_idx)
    return result


def _candidate_lookup(query_plan_log: Mapping[str, Any]) -> Dict[int, Dict[str, Any]]:
    proof_slate = query_plan_log.get("full_aperture_proof_slate", {}) or {}
    result: Dict[int, Dict[str, Any]] = {}
    for value in proof_slate.get("candidates", []) or []:
        if not isinstance(value, Mapping):
            continue
        passage_idx = _safe_int(value.get("passage_idx"), -1)
        if passage_idx >= 0:
            result[passage_idx] = dict(value)
    return result


def _monotone_scores(
    selected_ids: Sequence[int],
    source_ids: Sequence[int],
    source_scores: Sequence[float],
) -> List[float]:
    score_by_id = {
        int(passage_idx): _safe_float(score)
        for passage_idx, score in zip(source_ids, source_scores)
    }
    observed = list(score_by_id.values())
    ceiling = max(observed, default=1.0)
    floor = min(observed, default=0.0)
    step = max(abs(ceiling - floor), 1.0) / max(len(selected_ids), 1)
    return [
        score_by_id.get(int(passage_idx), ceiling - rank * step)
        for rank, passage_idx in enumerate(selected_ids)
    ]


class EvidenceVerifierAgent:
    """Diagnose proof, binding, and terminal-type gaps in the current slate."""

    @staticmethod
    def profile(
        doc_ids: Sequence[int],
        query_plan_log: Mapping[str, Any],
    ) -> Dict[str, Any]:
        ids = _ids(doc_ids)[:5]
        lookup = _candidate_lookup(query_plan_log)
        proof_slate = query_plan_log.get("full_aperture_proof_slate", {}) or {}
        signature = proof_slate.get("query_structure_signature", {}) or {}
        latent_contract = recover_latent_query_contract(
            query_plan_log, proof_slate
        )

        def fraction(limit: int, predicate: Any) -> float:
            values = ids[:limit]
            if not values:
                return 0.0
            return sum(
                int(predicate(lookup.get(passage_idx, {})))
                for passage_idx in values
            ) / len(values)

        def mean(limit: int, key: str) -> float:
            values = ids[:limit]
            if not values:
                return 0.0
            return sum(
                _safe_float(lookup.get(passage_idx, {}).get(key))
                for passage_idx in values
            ) / len(values)

        relation_slot_count = _safe_int(
            latent_contract.get("effective_relation_slot_count"), 0
        )
        explicit_relation_slot_count = _safe_int(
            latent_contract.get("explicit_relation_slot_count"), 0
        )
        top2_complete = fraction(
            2,
            lambda value: bool(
                value.get("complete_provenance")
                or value.get("local_proof_path_complete")
            ),
        )
        top5_complete = fraction(
            5,
            lambda value: bool(
                value.get("complete_provenance")
                or value.get("local_proof_path_complete")
            ),
        )
        top2_bound = fraction(
            2, lambda value: bool(value.get("bound_proof_path_complete"))
        )
        top5_bound = fraction(
            5, lambda value: bool(value.get("bound_proof_path_complete"))
        )
        top5_terminal = fraction(
            5,
            lambda value: bool(value.get("terminal_relation_match"))
            and (
                not str(signature.get("answer_type", "") or "")
                or bool(value.get("answer_type_compatible"))
                or _safe_float(value.get("bound_terminal_answer_type_score")) > 0.0
            ),
        )
        relation_top2 = mean(2, "relation_coverage_score")
        relation_top5 = mean(5, "relation_coverage_score")
        query_local_path_count = sum(
            bool(value.get("query_local_compiler_validated"))
            for value in lookup.values()
        )
        diagnostics: List[str] = []
        if relation_slot_count < 2:
            diagnostics.append("not_a_multihop_contract")
        else:
            if top2_complete < 0.5 or top2_bound < 0.5:
                diagnostics.append("top2_missing_bound_proof")
            if relation_top2 < 0.5:
                diagnostics.append("top2_relation_gap")
            if top5_complete < 0.4 or top5_bound < 0.4:
                diagnostics.append("top5_proof_incomplete")
            if relation_top5 < 0.5:
                diagnostics.append("top5_relation_gap")
            if top5_terminal <= 0.0:
                diagnostics.append("terminal_type_or_relation_unverified")
            if query_local_path_count <= 0:
                diagnostics.append("no_query_local_bound_path")
        return {
            "top5": ids,
            "top2_complete_proof_fraction": round(top2_complete, 6),
            "top5_complete_proof_fraction": round(top5_complete, 6),
            "top2_bound_proof_fraction": round(top2_bound, 6),
            "top5_bound_proof_fraction": round(top5_bound, 6),
            "top2_relation_coverage": round(relation_top2, 6),
            "top5_relation_coverage": round(relation_top5, 6),
            "top5_terminal_verified_fraction": round(top5_terminal, 6),
            "query_local_path_member_count": query_local_path_count,
            "relation_slot_count": relation_slot_count,
            "explicit_relation_slot_count": explicit_relation_slot_count,
            "latent_contract_recovered": bool(
                latent_contract.get("latent_contract_recovered")
            ),
            "latent_contract": latent_contract,
            "topology": str(signature.get("topology", "") or ""),
            "diagnostics": diagnostics,
        }


class EvidenceRepairPlannerAgent:
    """Map verifier failures to legal complete-plan repair transactions."""

    def __init__(
        self,
        *,
        enable_top2: bool,
        enable_top5: bool,
        min_relation_gain: float,
        max_top5_displacements: int,
        require_query_local_proof: bool,
    ) -> None:
        self.enable_top2 = bool(enable_top2)
        self.enable_top5 = bool(enable_top5)
        self.min_relation_gain = max(float(min_relation_gain), 0.0)
        self.max_top5_displacements = max(int(max_top5_displacements), 0)
        self.require_query_local_proof = bool(require_query_local_proof)

    def propose(
        self,
        portfolio: Mapping[str, Any],
        profile: Mapping[str, Any],
        current_ids: Sequence[int],
        used_instances: Sequence[str],
        predictions: Mapping[str, Mapping[str, Any]] | None = None,
        require_calibrated_delta: bool = False,
    ) -> List[Dict[str, Any]]:
        current_top2 = _ids(current_ids)[:2]
        current_top5 = _ids(current_ids)[:5]
        used = set(used_instances)
        diagnostics = set(profile.get("diagnostics", []) or [])
        predictions = predictions or {}
        proposals: List[Dict[str, Any]] = []
        for plan_instance, raw_plan in (portfolio.get("plans", {}) or {}).items():
            if plan_instance in used or not isinstance(raw_plan, Mapping):
                continue
            if not bool(raw_plan.get("valid", False)):
                continue
            plan_ids = _ids(raw_plan.get("doc_ids", []))[:5]
            if len(plan_ids) < 5 or plan_ids == current_top5:
                continue
            family = str(raw_plan.get("plan_family", "") or "")
            features = dict(raw_plan.get("features", {}) or {})
            prediction = dict(predictions.get(str(plan_instance), {}) or {})
            if require_calibrated_delta and not bool(
                prediction.get("eligible", False)
            ):
                continue
            changes_top2 = plan_ids[:2] != current_top2
            objective = "top2_repair" if changes_top2 else "top5_completion"
            if changes_top2:
                if not self.enable_top2 or family not in TOP2_REPAIR_FAMILIES:
                    continue
                if not ({"top2_missing_bound_proof", "top2_relation_gap"} & diagnostics):
                    continue
                if _safe_float(features.get("plan_top5_displacement_count")) > self.max_top5_displacements:
                    continue
                rank_partitioned = family in {
                    PLAN_RANK_PARTITIONED_TOP2,
                    PLAN_RANK_PARTITIONED_NESTED,
                }
                if rank_partitioned:
                    if not require_calibrated_delta:
                        continue
                    if _safe_float(prediction.get("delta_r2")) < 0.0:
                        continue
                    if _safe_float(prediction.get("delta_r5")) < 0.0:
                        continue
                    if (
                        _safe_float(features.get("plan_complete_proof_top2_fraction")) < 0.5
                        and _safe_float(features.get("plan_relation_coverage_top2"))
                        <= _safe_float(profile.get("top2_relation_coverage"))
                    ):
                        continue
                else:
                    if _safe_float(features.get("plan_introduced_top2_proof_fraction")) < 1.0:
                        continue
                    if _safe_float(features.get("plan_complete_proof_top2_fraction")) < 1.0:
                        continue
                    if _safe_float(features.get("plan_signature_bound_top2_fraction")) < 0.5:
                        continue
                relation_gain = (
                    _safe_float(features.get("plan_relation_coverage_top2"))
                    - _safe_float(profile.get("top2_relation_coverage"))
                )
                if not rank_partitioned and relation_gain < self.min_relation_gain:
                    continue
            else:
                if not self.enable_top5 or family not in TOP5_REPAIR_FAMILIES:
                    continue
                if not (
                    {"top5_proof_incomplete", "top5_relation_gap", "terminal_type_or_relation_unverified"}
                    & diagnostics
                ):
                    continue
                if plan_ids[:2] != current_top2:
                    continue
                if _safe_float(features.get("plan_top5_displacement_count")) > 1.0:
                    continue
                if _safe_float(features.get("plan_introduced_top5_proof_fraction")) < 1.0:
                    continue
                if _safe_float(features.get("plan_introduced_complete_proof_fraction")) < 1.0:
                    continue
                relation_gain = (
                    _safe_float(features.get("plan_relation_coverage_top5"))
                    - _safe_float(profile.get("top5_relation_coverage"))
                )
                if relation_gain < self.min_relation_gain:
                    continue
                if family == PLAN_LEARNED_RESIDUAL_TOP5:
                    strict_residual = bool(
                        _safe_float(features.get("residual_candidate_complete_provenance")) >= 1.0
                        and _safe_float(features.get("residual_candidate_binding")) >= 1.0
                        and _safe_float(features.get("residual_candidate_typed_slot_binding")) >= 1.0
                        and _safe_float(features.get("residual_candidate_terminal_relation_match")) >= 1.0
                        and _safe_float(features.get("residual_candidate_role_consistent")) >= 1.0
                        and (
                            _safe_float(features.get("residual_candidate_exact_entity_key")) >= 1.0
                            or _safe_float(features.get("residual_candidate_path_connected")) >= 1.0
                        )
                    )
                    if not strict_residual:
                        continue
            if self.require_query_local_proof and family not in {
                PLAN_QUERY_LOCAL_TOP2_PAIR,
                PLAN_QUERY_LOCAL_TOP5,
                PLAN_SIGNATURE_GROUNDED,
                PLAN_BRANCH_COMPLETE_TOP2,
                PLAN_CHAIN_COMPLETION,
                PLAN_LEARNED_RESIDUAL_TOP5,
                PLAN_RANK_PARTITIONED_TOP2,
                PLAN_RANK_PARTITIONED_NESTED,
            }:
                continue
            proposals.append({
                "plan_instance": str(plan_instance),
                "plan_family": family,
                "objective": objective,
                "doc_ids": plan_ids,
                "features": features,
                "prediction": prediction,
                "relation_gain": round(relation_gain, 6),
            })
        return proposals

    @staticmethod
    def priority(proposal: Mapping[str, Any]) -> Tuple[Any, ...]:
        features = proposal.get("features", {}) or {}
        objective = str(proposal.get("objective", "") or "")
        prediction = proposal.get("prediction", {}) or {}
        if objective == "top2_repair":
            return (
                2,
                _safe_float(prediction.get("safety_margin")),
                _safe_float(prediction.get("gain_probability")),
                _safe_float(prediction.get("delta_r2")),
                _safe_float(prediction.get("delta_r5")),
                _safe_float(features.get("plan_signature_bound_top2_fraction")),
                _safe_float(features.get("plan_complete_proof_top2_fraction")),
                _safe_float(features.get("plan_bound_proof_top2_fraction")),
                _safe_float(features.get("plan_relation_coverage_top2")),
                _safe_float(features.get("plan_bound_terminal_type_top2")),
                _safe_float(features.get("plan_semantic_top5_retained_fraction")),
                -_safe_float(features.get("estimated_cost_log")),
            )
        return (
            1,
            _safe_float(prediction.get("safety_margin")),
            _safe_float(prediction.get("gain_probability")),
            _safe_float(prediction.get("delta_r5")),
            _safe_float(features.get("residual_candidate_path_connected")),
            _safe_float(features.get("residual_candidate_exact_entity_key")),
            _safe_float(features.get("plan_complete_proof_top5_fraction")),
            _safe_float(features.get("plan_bound_proof_top5_fraction")),
            _safe_float(features.get("plan_relation_coverage_top5")),
            _safe_float(features.get("plan_bound_terminal_type_top5")),
            -_safe_float(features.get("estimated_cost_log")),
        )


class TransactionalEvidenceCriticAgent:
    """Validate a proposed slate and return an auditable commit decision."""

    def __init__(
        self,
        *,
        max_estimated_cost: float,
        require_calibrated_delta: bool = False,
    ) -> None:
        self.max_estimated_cost = max(float(max_estimated_cost), 1.0)
        self.require_calibrated_delta = bool(require_calibrated_delta)

    def review(
        self,
        proposal: Mapping[str, Any],
        before: Mapping[str, Any],
        after: Mapping[str, Any],
        current_ids: Sequence[int],
    ) -> Dict[str, Any]:
        objective = str(proposal.get("objective", "") or "")
        features = proposal.get("features", {}) or {}
        prediction = proposal.get("prediction", {}) or {}
        proposed_ids = _ids(proposal.get("doc_ids", []))[:5]
        current_top2 = _ids(current_ids)[:2]
        checks: Dict[str, bool] = {
            "complete_top5": len(proposed_ids) == 5 and len(set(proposed_ids)) == 5,
            "cost_budget": math.expm1(
                _safe_float(features.get("estimated_cost_log"))
            ) <= self.max_estimated_cost,
            "multihop_contract": _safe_int(after.get("relation_slot_count"), 0) >= 2,
        }
        if self.require_calibrated_delta:
            checks.update({
                "calibrated_delta_admitted": bool(
                    prediction.get("eligible", False)
                ),
                "predicted_r2_non_regression": _safe_float(
                    prediction.get("delta_r2"), -1.0
                ) >= 0.0,
                "predicted_r5_non_regression": _safe_float(
                    prediction.get("delta_r5"), -1.0
                ) >= 0.0,
            })
        if objective == "top2_repair":
            before_quality = (
                _safe_float(before.get("top2_complete_proof_fraction")),
                _safe_float(before.get("top2_bound_proof_fraction")),
                _safe_float(before.get("top2_relation_coverage")),
            )
            after_quality = (
                _safe_float(after.get("top2_complete_proof_fraction")),
                _safe_float(after.get("top2_bound_proof_fraction")),
                _safe_float(after.get("top2_relation_coverage")),
            )
            rank_partitioned = str(proposal.get("plan_family", "") or "") in {
                PLAN_RANK_PARTITIONED_TOP2,
                PLAN_RANK_PARTITIONED_NESTED,
            }
            checks.update({
                "top2_changed": proposed_ids[:2] != current_top2,
                "bounded_top5_displacement": len(
                    set(proposed_ids) - set(_ids(current_ids)[:5])
                ) <= 2,
                "top2_proof_supported": (
                    _safe_float(after.get("top2_complete_proof_fraction")) >= 1.0
                    if not rank_partitioned
                    else _safe_float(after.get("top2_complete_proof_fraction")) >= 0.5
                ),
                "query_bound_or_calibrated": (
                    _safe_float(features.get("plan_signature_bound_top2_fraction")) >= 0.5
                    or (
                        rank_partitioned
                        and bool(prediction.get("eligible", False))
                    )
                ),
                "structural_or_calibrated_progress": (
                    after_quality > before_quality
                    or (
                        rank_partitioned
                        and _safe_float(prediction.get("gain_probability")) > 0.0
                    )
                ),
            })
        else:
            before_quality = (
                _safe_float(before.get("top5_complete_proof_fraction")),
                _safe_float(before.get("top5_bound_proof_fraction")),
                _safe_float(before.get("top5_relation_coverage")),
                _safe_float(before.get("top5_terminal_verified_fraction")),
            )
            after_quality = (
                _safe_float(after.get("top5_complete_proof_fraction")),
                _safe_float(after.get("top5_bound_proof_fraction")),
                _safe_float(after.get("top5_relation_coverage")),
                _safe_float(after.get("top5_terminal_verified_fraction")),
            )
            checks.update({
                "top2_order_preserved": proposed_ids[:2] == current_top2,
                "single_top5_displacement": len(
                    set(proposed_ids) - set(_ids(current_ids)[:5])
                ) <= 1,
                "strict_structural_progress": after_quality > before_quality,
            })
        failed = [name for name, passed in checks.items() if not passed]
        return {
            "approved": not failed,
            "checks": checks,
            "failed_checks": failed,
            "decision": "commit" if not failed else "rollback",
        }


class AgenticEvidenceRepairOptimizer:
    """Run a bounded verify-plan-execute-critic loop over evidence slates."""

    def __init__(
        self,
        *,
        top_k: int = 5,
        max_iterations: int = 2,
        enable_top2: bool = True,
        enable_top5: bool = True,
        min_relation_gain: float = 0.10,
        max_top5_displacements: int = 2,
        max_estimated_cost: float = 384.0,
        require_query_local_proof: bool = True,
        action_model_path: str = "",
        require_calibrated_delta: bool = False,
        skip_after_upstream_commit: bool = True,
    ) -> None:
        self.top_k = max(int(top_k), 5)
        self.max_iterations = min(max(int(max_iterations), 1), 4)
        self.action_model_path = str(action_model_path or "")
        self.require_calibrated_delta = bool(require_calibrated_delta)
        self.skip_after_upstream_commit = bool(skip_after_upstream_commit)
        self.action_model = (
            CompleteEvidencePlanPortfolioOptimizer(
                self.action_model_path, top_k=self.top_k
            )
            if self.action_model_path
            else None
        )
        self.verifier = EvidenceVerifierAgent()
        self.planner = EvidenceRepairPlannerAgent(
            enable_top2=enable_top2,
            enable_top5=enable_top5,
            min_relation_gain=min_relation_gain,
            max_top5_displacements=max_top5_displacements,
            require_query_local_proof=require_query_local_proof,
        )
        self.critic = TransactionalEvidenceCriticAgent(
            max_estimated_cost=max_estimated_cost,
            require_calibrated_delta=self.require_calibrated_delta,
        )

    @staticmethod
    def _local_log(
        query_plan_log: Mapping[str, Any],
        doc_ids: Sequence[int],
        doc_scores: Sequence[float],
    ) -> Dict[str, Any]:
        local = dict(query_plan_log)
        local["final_ranking_incumbent"] = {
            "doc_ids": _ids(doc_ids)[:5],
            "doc_scores": [_safe_float(value) for value in doc_scores[:5]],
            "candidate_window_doc_ids": _ids(doc_ids),
            "candidate_window_doc_scores": [
                _safe_float(value) for value in doc_scores[: len(doc_ids)]
            ],
        }
        return local

    def optimize(
        self,
        current_doc_ids: Sequence[int],
        current_doc_scores: Sequence[float],
        query_plan_log: Mapping[str, Any],
        *,
        shadow_only: bool = False,
    ) -> Tuple[List[int], List[float], Dict[str, Any]]:
        original_ids = _ids(current_doc_ids)
        original_scores = [_safe_float(value) for value in current_doc_scores]
        working_ids = list(original_ids)
        working_scores = list(original_scores)
        used_instances: List[str] = []
        iterations: List[Dict[str, Any]] = []
        committed_count = 0

        upstream_atomic = (
            query_plan_log.get("complete_evidence_plan_optimizer", {}) or {}
        ).get("atomic_execution", {}) or {}
        if self.skip_after_upstream_commit and bool(
            upstream_atomic.get("executed", False)
        ):
            profile = self.verifier.profile(
                original_ids,
                self._local_log(query_plan_log, original_ids, original_scores),
            )
            return original_ids, original_scores, {
                "enabled": True,
                "schema_version": SCHEMA_VERSION,
                "gold_free": True,
                "bounded": True,
                "skip_reason": "upstream_cbo_transaction_already_committed",
                "before_top5": original_ids[:5],
                "after_top5": original_ids[:5],
                "iterations": [],
                "committed_count": 0,
                "final_verifier": profile,
                "atomic_execution": {
                    "executed": False,
                    "action_kind": "latent_contract_delta_critic_agent",
                    "transaction_count": 0,
                    "before_top5": original_ids[:5],
                    "after_top5": original_ids[:5],
                },
            }

        for iteration_index in range(self.max_iterations):
            local_log = self._local_log(query_plan_log, working_ids, working_scores)
            predictions: Mapping[str, Mapping[str, Any]] = {}
            action_model_ready = bool(
                self.action_model is not None and self.action_model.is_ready()
            )
            if action_model_ready:
                _, _, portfolio = self.action_model.optimize(
                    working_ids,
                    working_scores,
                    local_log,
                    shadow_only=True,
                )
                predictions = portfolio.get("predictions", {}) or {}
            else:
                portfolio = materialize_complete_plan_portfolio(
                    local_log,
                    working_ids,
                    working_scores,
                    top_k=self.top_k,
                )
            before = self.verifier.profile(working_ids, local_log)
            proposals = self.planner.propose(
                portfolio,
                before,
                working_ids,
                used_instances,
                predictions=predictions,
                require_calibrated_delta=self.require_calibrated_delta,
            )
            proposals.sort(key=self.planner.priority, reverse=True)
            iteration_log: Dict[str, Any] = {
                "iteration": iteration_index + 1,
                "verifier": before,
                "proposal_count": len(proposals),
                "action_model_ready": action_model_ready,
                "calibrated_prediction_count": len(predictions),
                "proposal_families": list(dict.fromkeys(
                    str(value.get("plan_family", "") or "")
                    for value in proposals[:12]
                )),
                "decision": "stop",
                "reason": "no_contract_safe_repair",
            }
            approved: Dict[str, Any] | None = None
            approved_review: Dict[str, Any] | None = None
            approved_after: Dict[str, Any] | None = None
            reviews: List[Dict[str, Any]] = []
            for proposal in proposals:
                after = self.verifier.profile(proposal["doc_ids"], local_log)
                review = self.critic.review(
                    proposal, before, after, working_ids
                )
                reviews.append({
                    "plan_instance": proposal["plan_instance"],
                    "plan_family": proposal["plan_family"],
                    "objective": proposal["objective"],
                    "relation_gain": proposal["relation_gain"],
                    "prediction": proposal.get("prediction", {}),
                    **review,
                })
                if review["approved"]:
                    approved = proposal
                    approved_review = review
                    approved_after = after
                    break
            iteration_log["critic_reviews"] = reviews[:12]
            if approved is None:
                iterations.append(iteration_log)
                break

            used_instances.append(str(approved["plan_instance"]))
            iteration_log.update({
                "decision": "shadow_commit" if shadow_only else "commit",
                "reason": "all_transactional_checks_passed",
                "selected_plan": approved["plan_family"],
                "selected_plan_instance": approved["plan_instance"],
                "objective": approved["objective"],
                "before_top5": working_ids[:5],
                "after_top5": approved["doc_ids"][:5],
                "after_verifier": approved_after,
                "critic": approved_review,
            })
            iterations.append(iteration_log)
            if shadow_only:
                break
            working_ids = _ids([
                *approved["doc_ids"],
                *working_ids,
                *((query_plan_log.get("semantic_plan_incumbent", {}) or {}).get(
                    "candidate_window_doc_ids", []
                )),
            ])
            working_scores = _monotone_scores(
                working_ids, original_ids, original_scores
            )
            committed_count += 1

        final_ids = original_ids if shadow_only else working_ids
        final_scores = original_scores if shadow_only else working_scores
        final_profile = self.verifier.profile(
            final_ids,
            self._local_log(query_plan_log, final_ids, final_scores),
        )
        log = {
            "enabled": True,
            "schema_version": SCHEMA_VERSION,
            "gold_free": True,
            "bounded": True,
            "max_iterations": self.max_iterations,
            "shadow_only": bool(shadow_only),
            "action_model_path": self.action_model_path,
            "action_model_ready": bool(
                self.action_model is not None and self.action_model.is_ready()
            ),
            "require_calibrated_delta": self.require_calibrated_delta,
            "agents": [
                "query_contract_recovery",
                "evidence_verifier",
                "repair_planner",
                "calibrated_delta_critic",
                "transactional_critic",
                "physical_plan_executor",
            ],
            "before_top5": original_ids[:5],
            "after_top5": final_ids[:5],
            "iterations": iterations,
            "committed_count": committed_count,
            "final_verifier": final_profile,
            "atomic_execution": {
                "executed": committed_count > 0,
                "action_kind": "bounded_agentic_evidence_repair",
                "transaction_count": committed_count,
                "before_top5": original_ids[:5],
                "after_top5": final_ids[:5],
            },
        }
        return final_ids, final_scores, log
