from dataclasses import dataclass, field
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .query_plan_state_policy import QueryPlanStateRetrieverPolicy


@dataclass
class StateRetrieverDecision:
    passage_idx: int
    original_rank: int
    final_rank: int
    selected: bool
    reason: str
    score: float
    baseline_score: float
    evidence_score: float
    coverage_gain: float
    redundancy: float
    added_roles: int
    added_anchors: int
    added_plans: int
    added_relations: int
    added_terminals: int
    state_policy: Dict[str, Any] = field(default_factory=dict)


class QueryPlanStateGraphRetriever:
    """Re-rank passages by query-plan state coverage.

    Earlier query-plan modules mostly produced weak score boosts or QA hints.
    This module treats the query-plan ontology as an executable retrieval state:
    top-k evidence should cover complementary slots in the inferred reasoning
    path instead of repeating the same high-scoring passage type.
    """

    def __init__(
        self,
        top_k: int = 5,
        candidate_top_k: int = 50,
        preserve_top_k: int = 1,
        max_original_rank: int = 50,
        baseline_weight: float = 0.45,
        coverage_weight: float = 1.00,
        evidence_weight: float = 0.65,
        redundancy_penalty: float = 0.25,
        min_evidence_score: float = 0.0,
        require_new_slot: bool = False,
        max_ontology_passages: int = 1,
        require_policy_approval: bool = True,
        policy_fallback_min_confidence: float = 0.85,
        policy_fallback_max_original_rank: int = 5,
        require_structural_slot: bool = True,
        state_policy_path: str = "",
        state_policy_mode: str = "off",
    ):
        self.top_k = max(int(top_k), 1)
        self.candidate_top_k = max(int(candidate_top_k), self.top_k)
        self.preserve_top_k = max(int(preserve_top_k), 0)
        self.max_original_rank = int(max_original_rank)
        self.baseline_weight = float(baseline_weight)
        self.coverage_weight = float(coverage_weight)
        self.evidence_weight = float(evidence_weight)
        self.redundancy_penalty = float(redundancy_penalty)
        self.min_evidence_score = max(float(min_evidence_score), 0.0)
        self.require_new_slot = bool(require_new_slot)
        self.max_ontology_passages = max(int(max_ontology_passages), 0)
        self.require_policy_approval = bool(require_policy_approval)
        self.policy_fallback_min_confidence = min(max(float(policy_fallback_min_confidence), 0.0), 1.0)
        self.policy_fallback_max_original_rank = int(policy_fallback_max_original_rank)
        self.require_structural_slot = bool(require_structural_slot)
        self.state_policy_path = str(state_policy_path or "")
        self.state_policy_mode = str(state_policy_mode or "off").strip().lower()
        if self.state_policy_mode not in {"off", "gate", "override"}:
            self.state_policy_mode = "off"
        self.state_policy: Optional[QueryPlanStateRetrieverPolicy] = None
        if self.state_policy_mode != "off" and QueryPlanStateRetrieverPolicy.exists(self.state_policy_path):
            self.state_policy = QueryPlanStateRetrieverPolicy.load(self.state_policy_path)

    @staticmethod
    def _norm(value: Any) -> str:
        return " ".join(str(value or "").strip().lower().split())

    @staticmethod
    def _safe_int(value: Any, default: int = 10**9) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _signature(self, contexts: Iterable[Mapping[str, Any]]) -> Dict[str, set]:
        signature = {
            "roles": set(),
            "anchors": set(),
            "plans": set(),
            "relations": set(),
            "facts": set(),
            "terminals": set(),
            "verified_paths": set(),
        }
        for context in contexts or []:
            plan_id = self._norm(context.get("plan_id"))
            anchor = self._norm(context.get("anchor_entity"))
            terminal = self._norm(context.get("end_entity"))
            if plan_id:
                signature["plans"].add(plan_id)
            if anchor:
                signature["anchors"].add(anchor)
            if terminal:
                signature["terminals"].add(terminal)
            fact_position = context.get("fact_position")
            if plan_id and fact_position is not None:
                signature["roles"].add((plan_id, str(fact_position)))
            for fact_idx in context.get("fact_indices", []) or []:
                signature["facts"].add(str(fact_idx))
            rel_tuple = tuple(
                self._norm(relation)
                for relation in (context.get("relations", []) or [])
                if self._norm(relation)
            )
            if rel_tuple:
                signature["relations"].add(rel_tuple)
            if context.get("answer_path_verified"):
                signature["verified_paths"].add((
                    plan_id,
                    terminal,
                    str(context.get("fact_position", "")),
                ))
        return signature

    @staticmethod
    def _merge_signature(target: Dict[str, set], source: Mapping[str, set]) -> None:
        for key in target:
            target[key].update(source.get(key, set()) or set())

    def _coverage_gain(self, signature: Mapping[str, set], covered: Mapping[str, set]) -> Tuple[float, Dict[str, int]]:
        added_roles = len(signature.get("roles", set()) - covered.get("roles", set()))
        added_anchors = len(signature.get("anchors", set()) - covered.get("anchors", set()))
        added_plans = len(signature.get("plans", set()) - covered.get("plans", set()))
        added_relations = len(signature.get("relations", set()) - covered.get("relations", set()))
        added_facts = len(signature.get("facts", set()) - covered.get("facts", set()))
        added_terminals = len(signature.get("terminals", set()) - covered.get("terminals", set()))
        added_verified = len(signature.get("verified_paths", set()) - covered.get("verified_paths", set()))
        gain = (
            1.15 * added_roles
            + 0.85 * added_relations
            + 0.65 * added_terminals
            + 0.45 * added_anchors
            + 0.35 * added_plans
            + 0.15 * min(added_facts, 3)
            + 0.45 * added_verified
        )
        return float(gain), {
            "roles": added_roles,
            "anchors": added_anchors,
            "plans": added_plans,
            "relations": added_relations,
            "facts": added_facts,
            "terminals": added_terminals,
            "verified_paths": added_verified,
        }

    def _redundancy(self, signature: Mapping[str, set], covered: Mapping[str, set]) -> float:
        overlap = 0.0
        for key, weight in (
            ("roles", 0.90),
            ("relations", 0.65),
            ("terminals", 0.45),
            ("anchors", 0.25),
            ("plans", 0.20),
        ):
            overlap += weight * len(signature.get(key, set()) & covered.get(key, set()))
        return float(overlap)

    def _evidence_score(self, passage: Mapping[str, Any], max_raw_score: float) -> float:
        contexts = passage.get("contexts", []) or []
        if not contexts:
            return 0.0
        max_raw_score = max(float(max_raw_score or 0.0), 1e-8)
        raw_score = self._safe_float(passage.get("raw_lift_score"), 0.0)
        raw = min(max(raw_score / max_raw_score, 0.0), 1.0)
        decision = passage.get("reranker_decision", {}) or {}
        confidence = self._safe_float(decision.get("confidence"), 0.0)
        action = str(decision.get("action", "") or "")
        strict = any(
            str(context.get("signature_match_mode", "") or "").strip().lower() == "strict"
            for context in contexts
        )
        verified = any(bool(context.get("answer_path_verified")) for context in contexts)
        max_plan_scale = max(
            (self._safe_float(context.get("plan_lift_weight_scale"), 1.0) for context in contexts),
            default=1.0,
        )
        path_len = max(
            (
                max(
                    len(context.get("fact_indices", []) or []),
                    len(context.get("relations", []) or []),
                    1,
                )
                for context in contexts
            ),
            default=1,
        )
        score = (
            0.45 * raw
            + 0.25 * confidence
            + 0.15 * min(max_plan_scale, 1.0)
            + 0.10 * min(len(contexts) / 2.0, 1.0)
        )
        if action == "promote":
            score += 0.12
        if strict:
            score += 0.12
        if verified:
            score += 0.10
        if path_len >= 2:
            score += 0.06
        return float(score)

    def _baseline_score(self, original_rank: int, score_by_doc: Mapping[int, float], doc_id: int) -> float:
        if original_rank <= 0:
            return 0.0
        rank_score = 1.0 / np.sqrt(float(original_rank))
        raw_score = max(float(score_by_doc.get(int(doc_id), 0.0)), 0.0)
        return float(0.65 * rank_score + 0.35 * raw_score)

    def _policy_allows_ontology_selection(
        self,
        passage: Mapping[str, Any],
        original_rank: int,
    ) -> Tuple[bool, str]:
        if not self.require_policy_approval:
            return True, "policy_gate_disabled"

        decision = passage.get("reranker_decision", {}) or {}
        policy = decision.get("policy", {}) or {}
        if policy:
            if policy.get("approved"):
                return True, "policy_approved"
            return False, f"policy_rejected:{policy.get('reason', 'unknown')}"

        confidence = self._safe_float(decision.get("confidence"), 0.0)
        rank_allowed = (
            self.policy_fallback_max_original_rank <= 0
            or original_rank <= self.policy_fallback_max_original_rank
        )
        if confidence >= self.policy_fallback_min_confidence and rank_allowed:
            return True, "high_confidence_rank_fallback"
        return False, "missing_policy_low_confidence_or_rank"

    def _state_policy_decision(
        self,
        query: str,
        query_condition: Mapping[str, Any],
        query_type: str,
        passage: Mapping[str, Any],
        original_rank: int,
        baseline_score: float,
        evidence_score: float,
        coverage_gain: float,
        redundancy: float,
        added: Mapping[str, int],
    ) -> Dict[str, Any]:
        if self.state_policy is None or self.state_policy_mode == "off":
            return {
                "enabled": False,
                "mode": self.state_policy_mode,
                "approved": None,
                "reason": "state_policy_disabled",
            }
        decision = passage.get("reranker_decision", {}) or {}
        features = self.state_policy.extract_features(
            query=query,
            query_condition=query_condition,
            query_type=query_type,
            original_rank=original_rank,
            confidence=self._safe_float(decision.get("confidence"), 0.0),
            evidence_score=evidence_score,
            baseline_score=baseline_score,
            coverage_gain=coverage_gain,
            redundancy=redundancy,
            added=added,
            contexts=passage.get("contexts", []) or [],
            passage_policy_decision=(decision.get("policy", {}) or {}),
        )
        result = self.state_policy.evaluate(features)
        result.update({
            "enabled": True,
            "mode": self.state_policy_mode,
            "features": features,
        })
        return result

    def rerank(
        self,
        sorted_doc_ids: Sequence[int],
        sorted_doc_scores: Sequence[float],
        query_plan_log: Mapping[str, Any],
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        query = str(query_plan_log.get("query", "") or "")
        query_condition = query_plan_log.get("query_condition", {}) or {}
        passage_reranker_log = query_plan_log.get("passage_reranker", {}) or {}
        query_type = str(passage_reranker_log.get("query_type", "") or "unknown")
        base_doc_ids = [int(doc_id) for doc_id in list(sorted_doc_ids)]
        base_scores = np.array(sorted_doc_scores, dtype=np.float32)
        score_by_doc = {
            int(doc_id): float(base_scores[idx]) if idx < len(base_scores) else 0.0
            for idx, doc_id in enumerate(base_doc_ids)
        }
        original_rank_by_doc = {doc_id: rank + 1 for rank, doc_id in enumerate(base_doc_ids)}
        log: Dict[str, Any] = {
            "enabled": True,
            "top_k": self.top_k,
            "candidate_top_k": self.candidate_top_k,
            "preserve_top_k": self.preserve_top_k,
            "max_original_rank": self.max_original_rank,
            "baseline_weight": self.baseline_weight,
            "coverage_weight": self.coverage_weight,
            "evidence_weight": self.evidence_weight,
            "redundancy_penalty": self.redundancy_penalty,
            "min_evidence_score": self.min_evidence_score,
            "require_new_slot": self.require_new_slot,
            "max_ontology_passages": self.max_ontology_passages,
            "require_policy_approval": self.require_policy_approval,
            "policy_fallback_min_confidence": self.policy_fallback_min_confidence,
            "policy_fallback_max_original_rank": self.policy_fallback_max_original_rank,
            "require_structural_slot": self.require_structural_slot,
            "state_policy_mode": self.state_policy_mode,
            "state_policy_path": self.state_policy_path,
            "state_policy_loaded": self.state_policy is not None,
            "state_policy_rules": len(self.state_policy.rules) if self.state_policy is not None else 0,
            "changed_top_k": False,
            "selected_ontology_count": 0,
            "gate_rejection_count": 0,
            "gate_rejection_reasons": {},
            "selected_passages": [],
            "candidate_decisions": [],
            "skip_reason": "",
        }
        if not base_doc_ids:
            log["skip_reason"] = "empty_retrieval"
            return np.array([], dtype=np.intp), base_scores, log

        lifted = list(query_plan_log.get("lifted_passages", []) or [])
        if not lifted:
            log["skip_reason"] = "no_lifted_passages"
            return np.array(base_doc_ids, dtype=np.intp), base_scores, log

        passage_details: Dict[int, Mapping[str, Any]] = {}
        max_raw_score = max((self._safe_float(item.get("raw_lift_score"), 0.0) for item in lifted), default=1.0)
        for passage in lifted:
            passage_idx = self._safe_int(passage.get("passage_idx"), default=-1)
            if passage_idx < 0:
                continue
            old = passage_details.get(passage_idx)
            if old is None or len(passage.get("contexts", []) or []) > len(old.get("contexts", []) or []):
                passage_details[passage_idx] = passage

        if not passage_details:
            log["skip_reason"] = "no_passage_details"
            return np.array(base_doc_ids, dtype=np.intp), base_scores, log

        candidate_ids = []
        for doc_id in base_doc_ids[: self.candidate_top_k]:
            candidate_ids.append(int(doc_id))
        for passage_idx, passage in passage_details.items():
            original_rank = self._safe_int(passage.get("original_rank"))
            if self.max_original_rank > 0 and original_rank > self.max_original_rank:
                continue
            if passage_idx not in candidate_ids:
                candidate_ids.append(int(passage_idx))

        selected = base_doc_ids[: min(self.preserve_top_k, len(base_doc_ids), self.top_k)]
        selected_set = set(selected)
        covered = {
            "roles": set(),
            "anchors": set(),
            "plans": set(),
            "relations": set(),
            "facts": set(),
            "terminals": set(),
            "verified_paths": set(),
        }
        for doc_id in selected:
            passage = passage_details.get(int(doc_id))
            if passage:
                self._merge_signature(covered, self._signature(passage.get("contexts", []) or []))

        decisions: List[StateRetrieverDecision] = []
        ontology_selected = 0
        gate_rejection_reasons = defaultdict(int)
        gate_rejected_passages = set()

        def reject_once(passage_idx: int, reason: str) -> None:
            key = (int(passage_idx), str(reason))
            if key in gate_rejected_passages:
                return
            gate_rejected_passages.add(key)
            gate_rejection_reasons[str(reason)] += 1

        while len(selected) < min(self.top_k, len(base_doc_ids)):
            best = None
            for passage_idx in candidate_ids:
                if passage_idx in selected_set:
                    continue
                original_rank = int(original_rank_by_doc.get(int(passage_idx), self._safe_int(
                    passage_details.get(int(passage_idx), {}).get("original_rank") if passage_details.get(int(passage_idx)) else None
                )))
                if self.max_original_rank > 0 and original_rank > self.max_original_rank and passage_idx not in base_doc_ids[: self.top_k]:
                    continue
                passage = passage_details.get(int(passage_idx), {})
                contexts = passage.get("contexts", []) or []
                signature = self._signature(contexts)
                coverage_gain, added = self._coverage_gain(signature, covered)
                redundancy = self._redundancy(signature, covered)
                evidence_score = self._evidence_score(passage, max_raw_score=max_raw_score)
                baseline_score = self._baseline_score(original_rank, score_by_doc, int(passage_idx))
                use_ontology_signal = False
                state_policy_decision = {
                    "enabled": False,
                    "mode": self.state_policy_mode,
                    "approved": None,
                    "reason": "not_evaluated",
                }
                if contexts and ontology_selected < self.max_ontology_passages:
                    structural_gain = (
                        int(added.get("roles", 0))
                        + int(added.get("relations", 0))
                        + int(added.get("plans", 0))
                        + int(added.get("verified_paths", 0))
                    )
                    policy_allowed, policy_reason = self._policy_allows_ontology_selection(
                        passage,
                        original_rank=original_rank,
                    )
                    state_policy_decision = self._state_policy_decision(
                        query=query,
                        query_condition=query_condition,
                        query_type=query_type,
                        passage=passage,
                        original_rank=original_rank,
                        baseline_score=baseline_score,
                        evidence_score=evidence_score,
                        coverage_gain=coverage_gain,
                        redundancy=redundancy,
                        added=added,
                    )
                    if self.state_policy is not None:
                        if self.state_policy_mode == "override":
                            if bool(state_policy_decision.get("approved")):
                                policy_allowed = True
                                policy_reason = "state_policy_override_approved"
                            else:
                                policy_allowed = False
                                policy_reason = f"state_policy_rejected:{state_policy_decision.get('reason', 'unknown')}"
                        elif self.state_policy_mode == "gate" and not bool(state_policy_decision.get("approved")):
                            policy_allowed = False
                            policy_reason = f"state_policy_rejected:{state_policy_decision.get('reason', 'unknown')}"
                    if not policy_allowed:
                        reject_once(passage_idx, policy_reason)
                    elif evidence_score < self.min_evidence_score:
                        reject_once(passage_idx, "low_evidence_score")
                    elif self.require_new_slot and coverage_gain <= 0.0:
                        reject_once(passage_idx, "no_new_slot")
                    elif self.require_structural_slot and structural_gain <= 0:
                        reject_once(passage_idx, "terminal_or_anchor_only")
                    else:
                        use_ontology_signal = True
                elif contexts and ontology_selected >= self.max_ontology_passages:
                    reject_once(passage_idx, "max_ontology_passages_reached")

                if use_ontology_signal:
                    score = (
                        self.baseline_weight * baseline_score
                        + self.coverage_weight * coverage_gain
                        + self.evidence_weight * evidence_score
                        - self.redundancy_penalty * redundancy
                    )
                else:
                    score = self.baseline_weight * baseline_score
                candidate_key = (
                    score,
                    coverage_gain if use_ontology_signal else 0.0,
                    evidence_score if use_ontology_signal else 0.0,
                    -original_rank,
                    int(passage_idx),
                )
                candidate_payload = (
                    score,
                    coverage_gain if use_ontology_signal else 0.0,
                    evidence_score if use_ontology_signal else 0.0,
                    -original_rank,
                    int(passage_idx),
                    original_rank,
                    baseline_score,
                    redundancy if use_ontology_signal else 0.0,
                    added,
                    use_ontology_signal,
                    signature,
                    state_policy_decision,
                )
                if best is None or candidate_key > best[0]:
                    best = (candidate_key, candidate_payload)
            if best is None:
                break
            (
                score,
                coverage_gain,
                evidence_score,
                _negative_rank,
                passage_idx,
                original_rank,
                baseline_score,
                redundancy,
                added,
                use_ontology_signal,
                signature,
                state_policy_decision,
            ) = best[1]
            selected.append(int(passage_idx))
            selected_set.add(int(passage_idx))
            self._merge_signature(covered, signature)
            if use_ontology_signal:
                ontology_selected += 1
            decisions.append(StateRetrieverDecision(
                passage_idx=int(passage_idx),
                original_rank=int(original_rank),
                final_rank=len(selected),
                selected=True,
                reason="selected_state_coverage" if use_ontology_signal else "selected_baseline_fill",
                score=round(float(score), 6),
                baseline_score=round(float(baseline_score), 6),
                evidence_score=round(float(evidence_score), 6),
                coverage_gain=round(float(coverage_gain), 6),
                redundancy=round(float(redundancy), 6),
                added_roles=int(added.get("roles", 0)),
                added_anchors=int(added.get("anchors", 0)),
                added_plans=int(added.get("plans", 0)),
                added_relations=int(added.get("relations", 0)),
                added_terminals=int(added.get("terminals", 0)),
                state_policy=state_policy_decision if use_ontology_signal else {},
            ))

        if len(selected) <= self.preserve_top_k:
            log["skip_reason"] = "no_state_selection"
            return np.array(base_doc_ids, dtype=np.intp), base_scores, log

        for doc_id in base_doc_ids:
            if int(doc_id) in selected_set:
                continue
            selected.append(int(doc_id))
            selected_set.add(int(doc_id))
        final_doc_ids = np.array(selected, dtype=np.intp)
        final_scores = np.array([score_by_doc.get(int(doc_id), 0.0) for doc_id in selected], dtype=np.float32)
        before_top = base_doc_ids[: self.top_k]
        after_top = selected[: self.top_k]
        selected_ontology_count = sum(
            1 for decision in decisions
            if decision.selected and decision.reason == "selected_state_coverage"
        )
        total_added_slots = sum(
            decision.added_roles
            + decision.added_anchors
            + decision.added_plans
            + decision.added_relations
            + decision.added_terminals
            for decision in decisions
            if decision.selected
        )
        log.update({
            "changed_top_k": before_top != after_top,
            "selected_ontology_count": int(selected_ontology_count),
            "total_added_slots": int(total_added_slots),
            "gate_rejection_count": int(sum(gate_rejection_reasons.values())),
            "gate_rejection_reasons": dict(
                sorted(gate_rejection_reasons.items(), key=lambda item: item[1], reverse=True)[:8]
            ),
            "covered_roles": len(covered["roles"]),
            "covered_anchors": len(covered["anchors"]),
            "covered_plans": len(covered["plans"]),
            "covered_relations": len(covered["relations"]),
            "covered_terminals": len(covered["terminals"]),
            "top_before": before_top,
            "top_after": after_top,
            "selected_passages": [decision.__dict__ for decision in decisions if decision.selected],
            "candidate_decisions": [decision.__dict__ for decision in decisions[:12]],
        })
        return final_doc_ids, final_scores, log
