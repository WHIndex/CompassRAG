"""Composable post-physical counterfactual evidence optimizer.

This optimizer observes the immutable slate produced by all
base retrieval and physical-planning operators, ranks counterfactual slate
actions, and atomically commits at most one action.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

from . import query_plan_supervised_evidence_ranker as base_ranker
from .query_plan_end_to_end_slate_optimizer import (
    generate_end_to_end_counterfactual_actions,
    group_unified_candidate_refs,
)
from .query_plan_nonlinear_delta_admission import (
    rank_query_grouped_nonlinear_delta,
)
from .query_plan_content_complete_set_comparator import (
    MODEL_TYPE as CONTENT_COMPLETE_MODEL_TYPE,
    generate_content_complete_post_physical_actions,
    is_content_complete_mode,
    required_passage_ids as content_complete_required_passage_ids,
)
from .query_plan_rank_partitioned_set_optimizer import (
    MODEL_TYPE as RANK_PARTITIONED_MODEL_TYPE,
    generate_rank_partitioned_post_physical_actions,
    is_rank_partitioned_mode,
    required_passage_ids as rank_partitioned_required_passage_ids,
)
from .query_plan_proof_carrying_nested_topk_optimizer import (
    FEATURE_ROUTED_STAGE_MODEL_TYPE,
    ONTOLOGY_SIGNATURE_STAGE_MODEL_TYPE,
    MODEL_TYPE as PROOF_NESTED_MODEL_TYPE,
    STAGE_MODEL_TYPE as PROOF_NESTED_STAGE_MODEL_TYPE,
    generate_proof_carrying_stage_actions,
    is_nested_topk_mode,
    required_passage_ids as proof_nested_required_passage_ids,
    stages_for_mode as proof_nested_stages_for_mode,
)
from .query_plan_supervised_evidence_ranker import (
    QueryPlanSupervisedEvidenceRanker,
    apply_passage_admission_action_to_slate,
    build_final_state_snapshot,
)
from .query_plan_full_aperture_proof_slate import is_full_aperture_mode
from .query_plan_executable_proof_semijoin import (
    generate_executable_semijoin_stage_actions,
    is_executable_semijoin_mode,
    required_passage_ids as executable_semijoin_required_passage_ids,
    stages_for_mode as executable_semijoin_stages_for_mode,
)


MODEL_TYPES = {
    "incumbent_safe_nonlinear_delta_optimizer",
    "composable_post_physical_delta_optimizer",
    CONTENT_COMPLETE_MODEL_TYPE,
    RANK_PARTITIONED_MODEL_TYPE,
    PROOF_NESTED_STAGE_MODEL_TYPE,
    FEATURE_ROUTED_STAGE_MODEL_TYPE,
    ONTOLOGY_SIGNATURE_STAGE_MODEL_TYPE,
}


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


class _PostPhysicalModelAdapter(QueryPlanSupervisedEvidenceRanker):
    """Reuse feature helpers without replacing the process-wide ranker class."""

    def is_ready(self) -> bool:
        if self.model.get("type") in {
            FEATURE_ROUTED_STAGE_MODEL_TYPE,
            ONTOLOGY_SIGNATURE_STAGE_MODEL_TYPE,
        }:
            return bool(
                self.model.get("feature_names")
                and self.model.get("experts")
                and self.model.get("regimes")
                and self.model.get("feature_router")
            )
        if self.model.get("type") in MODEL_TYPES:
            return bool(
                self.model.get("feature_names")
                and self.model.get("experts")
                and self.model.get("datasets")
            )
        return False


class QueryPlanPostPhysicalDeltaOptimizer:
    """Rank residual evidence slates relative to the final base incumbent."""

    def __init__(
        self,
        model_path: str,
        *,
        candidate_mode: str = "unified",
        enforce_cq: bool = True,
        max_candidates: int = 24,
    ) -> None:
        self.model_path = str(model_path or "")
        self.candidate_mode = str(candidate_mode or "unified").strip().lower()
        self.enforce_cq = bool(enforce_cq)
        self.max_candidates = max(1, int(max_candidates))
        self.ranker = _PostPhysicalModelAdapter(
            model_path=self.model_path,
            max_candidates=self.max_candidates,
        )
        self._nested_stage_rankers: Dict[str, _PostPhysicalModelAdapter] = {}
        if self.uses_nested_topk_actions() and self.ranker.model.get("type") == PROOF_NESTED_MODEL_TYPE:
            bundle_dir = Path(self.model_path).expanduser().resolve().parent
            for stage, spec in (self.ranker.model.get("stages", {}) or {}).items():
                if not isinstance(spec, dict):
                    continue
                stage_path = Path(str(spec.get("model_path") or "")).expanduser()
                if not stage_path.is_absolute():
                    stage_path = (bundle_dir / stage_path).resolve()
                self._nested_stage_rankers[str(stage)] = _PostPhysicalModelAdapter(
                    model_path=str(stage_path),
                    max_candidates=self.max_candidates,
                )

    def is_ready(self) -> bool:
        if self.uses_nested_topk_actions():
            required = self._nested_stages()
            return bool(required) and all(
                stage in self._nested_stage_rankers
                and self._nested_stage_rankers[stage].is_ready()
                for stage in required
            )
        return self.ranker.is_ready()

    def requires_content_complete_context(self) -> bool:
        return (
            is_content_complete_mode(self.candidate_mode)
            or is_rank_partitioned_mode(self.candidate_mode)
            or is_nested_topk_mode(self.candidate_mode)
            or is_executable_semijoin_mode(self.candidate_mode)
        )

    def uses_rank_partitioned_actions(self) -> bool:
        return is_rank_partitioned_mode(self.candidate_mode)

    def uses_nested_topk_actions(self) -> bool:
        return (
            is_nested_topk_mode(self.candidate_mode)
            or self.uses_executable_semijoin_actions()
        )

    def uses_executable_semijoin_actions(self) -> bool:
        return is_executable_semijoin_mode(self.candidate_mode)

    def _nested_stages(self) -> Tuple[str, ...]:
        if self.uses_executable_semijoin_actions():
            return executable_semijoin_stages_for_mode(self.candidate_mode)
        return proof_nested_stages_for_mode(self.candidate_mode)

    def uses_full_aperture(self) -> bool:
        return (
            is_full_aperture_mode(self.candidate_mode)
            or self.uses_executable_semijoin_actions()
        )

    def candidate_window(self) -> int:
        return 200 if self.uses_full_aperture() else 120

    def _metadata(self) -> Dict[str, Any]:
        if self.uses_nested_topk_actions():
            value = self.ranker.model.get("metadata", {}) or {}
            metadata = dict(value) if isinstance(value, dict) else {}
            metadata.setdefault("max_single_candidates", self.max_candidates)
            metadata.setdefault(
                "max_pair_candidates", max(12, self.max_candidates // 2)
            )
            metadata.setdefault("max_fallback_per_signal", 20)
            metadata.setdefault(
                "max_fallback_per_signal_by_stage",
                {"top5": 20, "top2": 6},
            )
            return metadata
        return self.ranker._metadata() if self.is_ready() else {}

    def _fallback_budget(self, stage: str) -> int:
        metadata = self._metadata()
        by_stage = metadata.get("max_fallback_per_signal_by_stage", {}) or {}
        if isinstance(by_stage, dict) and stage in by_stage:
            return max(0, _safe_int(by_stage.get(stage), 0))
        return max(
            0, _safe_int(metadata.get("max_fallback_per_signal"), 20)
        )

    def required_passage_ids(
        self, query_plan_log: Dict[str, Any]
    ) -> List[int]:
        if not self.requires_content_complete_context():
            return []
        metadata = self._metadata()
        required = (
            executable_semijoin_required_passage_ids
            if self.uses_executable_semijoin_actions()
            else proof_nested_required_passage_ids
            if self.uses_nested_topk_actions()
            else rank_partitioned_required_passage_ids
            if self.uses_rank_partitioned_actions()
            else content_complete_required_passage_ids
        )
        return sorted(required(
            query_plan_log,
            base_module=base_ranker,
            max_single_candidates=max(
                1, _safe_int(metadata.get("max_single_candidates"), 24)
            ),
            max_pair_candidates=max(
                2, _safe_int(metadata.get("max_pair_candidates"), 12)
            ),
            candidate_mode=self.candidate_mode,
            enforce_cq=self.enforce_cq,
        ))

    def _actions(self, query_plan_log: Dict[str, Any], **kwargs) -> List[Dict[str, Any]]:
        if self.uses_nested_topk_actions():
            stage = str(kwargs.pop("stage", "") or "")
            if self.uses_executable_semijoin_actions():
                return generate_executable_semijoin_stage_actions(
                    query_plan_log,
                    base_module=base_ranker,
                    stage=stage,
                    candidate_mode=self.candidate_mode,
                    enforce_cq=self.enforce_cq,
                    max_fallback_per_signal=self._fallback_budget(stage),
                    **kwargs,
                )
            return generate_proof_carrying_stage_actions(
                query_plan_log,
                base_module=base_ranker,
                stage=stage,
                enforce_cq=self.enforce_cq,
                **kwargs,
            )
        if self.uses_rank_partitioned_actions():
            return generate_rank_partitioned_post_physical_actions(
                query_plan_log,
                base_module=base_ranker,
                candidate_mode=self.candidate_mode,
                enforce_cq=self.enforce_cq,
                **kwargs,
            )
        if self.requires_content_complete_context():
            return generate_content_complete_post_physical_actions(
                query_plan_log,
                base_module=base_ranker,
                candidate_mode=self.candidate_mode,
                enforce_cq=self.enforce_cq,
                **kwargs,
            )
        return generate_end_to_end_counterfactual_actions(
            query_plan_log,
            base_module=base_ranker,
            candidate_mode=self.candidate_mode,
            enforce_cq=self.enforce_cq,
            zero_physical_features=False,
            zero_cq_features=False,
            **kwargs,
        )

    def _optimize_nested_topk(
        self,
        doc_ids: List[int],
        doc_scores: List[float],
        query_plan_log: Dict[str, Any],
        *,
        shadow_only: bool,
        common: Dict[str, Any],
    ) -> Tuple[List[int], List[float], Dict[str, Any]]:
        original_ids = list(doc_ids)
        original_scores = list(doc_scores)
        stages = self._nested_stages()
        metadata = self._metadata()
        stage_logs: Dict[str, Any] = {}

        if shadow_only:
            compiled_action_kinds: Counter = Counter()
            compiled_action_count = 0
            for stage in stages:
                actions = self._actions(
                    query_plan_log,
                    stage=stage,
                    max_single_candidates=max(
                        1, _safe_int(metadata.get("max_single_candidates"), 24)
                    ),
                    max_pair_candidates=max(
                        2, _safe_int(metadata.get("max_pair_candidates"), 12)
                    ),
                )
                stage_logs[stage] = {
                    "mode": "proof_carrying_nested_topk_shadow",
                    "nested_stage": stage,
                    "compiled_action_count": len(actions),
                    "compiled_action_kinds": dict(Counter(
                        str(action.get("action_kind") or "unknown")
                        for action in actions
                    )),
                    "no_op": True,
                    "skip_reason": "shadow_collection",
                    "atomic_execution": {"executed": False, "reason": "shadow_only"},
                }
                compiled_action_count += len(actions)
                compiled_action_kinds.update(
                    str(action.get("action_kind") or "unknown")
                    for action in actions
                )
            return doc_ids, doc_scores, {
                **common,
                "mode": "proof_carrying_nested_topk_optimizer",
                "shadow_only": True,
                "stages": stage_logs,
                "compiled_action_count": compiled_action_count,
                "compiled_action_kinds": dict(sorted(compiled_action_kinds.items())),
                "scored_count": 0,
                "promoted_count": 0,
                "no_op": True,
                "skip_reason": "shadow_collection",
                "atomic_execution": {"executed": False, "reason": "shadow_only"},
            }

        if not self.is_ready():
            return doc_ids, doc_scores, {
                **common,
                "mode": "proof_carrying_nested_topk_optimizer",
                "shadow_only": False,
                "stages": {},
                "scored_count": 0,
                "promoted_count": 0,
                "no_op": True,
                "skip_reason": "model_not_ready",
                "atomic_execution": {"executed": False, "reason": "no_op"},
            }

        for stage in stages:
            query_plan_log["candidate_admission_incumbent"] = build_final_state_snapshot(
                doc_ids,
                doc_scores,
                candidate_window=self.candidate_window(),
                defer_decision=False,
            )
            stage_ranker = self._nested_stage_rankers[stage]

            def generate(record, **kwargs):
                if self.uses_executable_semijoin_actions():
                    return generate_executable_semijoin_stage_actions(
                        record,
                        base_module=base_ranker,
                        stage=stage,
                        enforce_cq=self.enforce_cq,
                        max_fallback_per_signal=self._fallback_budget(stage),
                        **kwargs,
                    )
                return generate_proof_carrying_stage_actions(
                    record,
                    base_module=base_ranker,
                    stage=stage,
                    enforce_cq=self.enforce_cq,
                    **kwargs,
                )

            _, stage_log = rank_query_grouped_nonlinear_delta(
                stage_ranker,
                query_plan_log,
                action_generator=generate,
                safe_int=_safe_int,
                candidate_compiler=(
                    f"executable_proof_semijoin_{stage}"
                    if self.uses_executable_semijoin_actions()
                    else f"proof_carrying_nested_{stage}"
                ),
            )
            stage_log["nested_stage"] = stage
            stage_log["stage_model_path"] = stage_ranker.model_path
            doc_ids, doc_scores, stage_log = apply_passage_admission_action_to_slate(
                list(doc_ids), list(doc_scores), stage_log
            )
            stage_logs[stage] = stage_log

        executed = any(
            bool((stage_log.get("atomic_execution", {}) or {}).get("executed", False))
            for stage_log in stage_logs.values()
        )
        final_top5 = list(doc_ids[:5])
        selected_actions = [
            stage_log.get("selected_action", {})
            for stage_log in stage_logs.values()
            if bool((stage_log.get("atomic_execution", {}) or {}).get("executed", False))
        ]
        log = {
            **common,
            "mode": "proof_carrying_nested_topk_optimizer",
            "shadow_only": False,
            "stages": stage_logs,
            "selected_actions": selected_actions,
            "scored_count": sum(
                int(stage_log.get("scored_count", 0) or 0)
                for stage_log in stage_logs.values()
            ),
            "promoted_count": len(selected_actions),
            "committed_top2_count": int(bool(
                (stage_logs.get("top2", {}).get("atomic_execution", {}) or {}).get(
                    "executed", False
                )
            )),
            "committed_top5_count": int(bool(
                (stage_logs.get("top5", {}).get("atomic_execution", {}) or {}).get(
                    "executed", False
                )
            )),
            "committed_pair_count": sum(
                int(stage_log.get("committed_pair_count", 0) or 0)
                for stage_log in stage_logs.values()
            ),
            "no_op": not executed,
            "top2_membership_preserved_by_top5_stage": (
                original_ids[:2]
                == (stage_logs.get("top5", {}).get("atomic_execution", {}) or {}).get(
                    "after_top5", original_ids[:5]
                )[:2]
            ),
            "top5_membership_preserved_by_top2_stage": (
                set(final_top5)
                == set((stage_logs.get("top2", {}).get("atomic_execution", {}) or {}).get(
                    "before_top5", final_top5
                ))
            ),
            "atomic_execution": {
                "executed": executed,
                "action_kind": "proof_carrying_nested_topk",
                "target": "nested_topk" if len(stages) > 1 else stages[0],
                "before_top5": original_ids[:5],
                "after_top5": final_top5,
                "changed_top2": original_ids[:2] != final_top5[:2],
                "changed_top5": original_ids[:5] != final_top5,
                "stage_execution": {
                    stage: stage_log.get("atomic_execution", {})
                    for stage, stage_log in stage_logs.items()
                },
            },
        }
        if not executed:
            if list(doc_ids) != original_ids or list(doc_scores) != original_scores:
                raise RuntimeError(
                    "nested top-k no-op violated snapshot identity: ranking changed without a committed action"
                )
            log["skip_reason"] = "all_nested_stages_no_op"
            log["noop_identity_validated"] = True
        return doc_ids, doc_scores, log

    def optimize(
        self,
        doc_ids: List[int],
        doc_scores: List[float],
        query_plan_log: Dict[str, Any],
        *,
        shadow_only: bool = False,
    ) -> Tuple[List[int], List[float], Dict[str, Any]]:
        incumbent = build_final_state_snapshot(
            doc_ids,
            doc_scores,
            candidate_window=self.candidate_window(),
            defer_decision=False,
        )


        query_plan_log["candidate_admission_incumbent"] = incumbent

        grouped_candidates = group_unified_candidate_refs(
            query_plan_log,
            base_module=base_ranker,
            candidate_mode=self.candidate_mode,
        )
        common = {
            "enabled": True,
            "component": "post_physical_delta_optimizer",
            "model_path": self.model_path,
            "model_ready": self.is_ready(),
            "candidate_mode": self.candidate_mode,
            "cq_constraints_enforced": self.enforce_cq,
            "unified_candidate_count": len(grouped_candidates),
            "incumbent_state_contract": "post_base_and_physical_operators",
            "incumbent_state_hash": incumbent.get("state_hash", ""),
            "incumbent_state_version": incumbent.get("state_version", ""),
            "atomic_state_hash_validation": True,
            "upstream_ranker_preserved": True,
            "content_complete": self.requires_content_complete_context(),
            "rank_partitioned": self.uses_rank_partitioned_actions(),
            "proof_carrying_nested_topk": self.uses_nested_topk_actions(),
            "executable_proof_semijoin": self.uses_executable_semijoin_actions(),
            "full_aperture_proof_slate": self.uses_full_aperture(),
            "candidate_window": self.candidate_window(),
            "top5_membership_invariant": self.uses_rank_partitioned_actions()
            and self.candidate_mode == "rank_partitioned_top2",
        }

        if self.uses_nested_topk_actions():
            return self._optimize_nested_topk(
                doc_ids,
                doc_scores,
                query_plan_log,
                shadow_only=shadow_only,
                common=common,
            )

        if shadow_only:
            metadata = self._metadata()
            actions = self._actions(
                query_plan_log,
                max_single_candidates=max(
                    1, _safe_int(metadata.get("max_single_candidates"), 24)
                ),
                max_pair_candidates=max(
                    2, _safe_int(metadata.get("max_pair_candidates"), 12)
                ),
            )
            action_kinds = Counter(
                str(action.get("action_kind") or "unknown") for action in actions
            )
            return doc_ids, doc_scores, {
                **common,
                "mode": "post_physical_shadow_collection",
                "shadow_only": True,
                "scored_count": 0,
                "compiled_action_count": len(actions),
                "compiled_action_kinds": dict(sorted(action_kinds.items())),
                "promoted_count": 0,
                "no_op": True,
                "skip_reason": "shadow_collection",
                "atomic_execution": {"executed": False, "reason": "shadow_only"},
            }

        if not self.is_ready():
            return doc_ids, doc_scores, {
                **common,
                "mode": "composable_post_physical_delta_optimizer",
                "scored_count": 0,
                "promoted_count": 0,
                "no_op": True,
                "skip_reason": "model_not_ready",
                "atomic_execution": {"executed": False, "reason": "no_op"},
            }

        candidate_compiler = (
            "rank_partitioned_post_physical_top2"
            if self.uses_rank_partitioned_actions()
            else "content_complete_post_physical_top2"
            if self.requires_content_complete_context()
            else "post_physical_counterfactual_evidence_relation"
        )
        _, log = rank_query_grouped_nonlinear_delta(
            self.ranker,
            query_plan_log,
            action_generator=self._actions,
            safe_int=_safe_int,
            candidate_compiler=candidate_compiler,
        )
        log.update(common)
        log["mode"] = (
            "rank_partitioned_counterfactual_set_optimizer"
            if self.uses_rank_partitioned_actions()
            else "content_complete_counterfactual_set_comparator"
            if self.requires_content_complete_context()
            else "composable_post_physical_delta_optimizer"
        )
        log["shadow_only"] = False
        updated_ids, updated_scores, log = apply_passage_admission_action_to_slate(
            list(doc_ids),
            list(doc_scores),
            log,
        )
        return updated_ids, updated_scores, log
