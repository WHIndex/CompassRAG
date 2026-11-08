"""Counterfactual top-k action materialization for retrieval optimization."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Sequence


def _unique_doc_ids(values: Iterable[Any], limit: int = 3) -> List[int]:
    output: List[int] = []
    seen = set()
    for value in values or []:
        try:
            doc_id = int(value)
        except (TypeError, ValueError):
            continue
        if doc_id < 0 or doc_id in seen:
            continue
        seen.add(doc_id)
        output.append(doc_id)
        if len(output) >= max(int(limit), 1):
            break
    return output


def _bounded_float(value: Any) -> float:
    try:
        return min(max(float(value or 0.0), 0.0), 1.0)
    except (TypeError, ValueError):
        return 0.0


def materialize_counterfactual_ranking(
    base_doc_ids: Sequence[int],
    selected_doc_ids: Sequence[int],
    rewrite_mode: str,
) -> Dict[str, Any]:
    """Materialize one independent action without mutating the control ranking."""
    current = [int(value) for value in base_doc_ids]
    selected = _unique_doc_ids(selected_doc_ids, limit=3)
    mode = str(rewrite_mode or "").strip().lower()
    if len(current) < 5:
        return {"valid": False, "reason": "ranking_too_short", "ranking": current}
    if mode == "no_op":
        rewritten = list(current)
    elif mode == "top2_single":
        if len(selected) != 1 or selected[0] not in current:
            return {"valid": False, "reason": "invalid_top2_single", "ranking": current}
        protected = current[0]
        if selected[0] == protected:
            return {"valid": False, "reason": "already_rank1", "ranking": current}
        rewritten = [protected, selected[0]] + [
            doc_id for doc_id in current if doc_id not in {protected, selected[0]}
        ]
    elif mode == "top2_group":
        if len(selected) != 2 or any(doc_id not in current for doc_id in selected):
            return {"valid": False, "reason": "invalid_top2_group", "ranking": current}
        rewritten = selected + [doc_id for doc_id in current if doc_id not in set(selected)]
    elif mode == "top5_single":
        if len(selected) != 1 or selected[0] not in current:
            return {"valid": False, "reason": "invalid_top5_single", "ranking": current}
        selected_idx = current.index(selected[0])
        if selected_idx < 5:
            return {"valid": False, "reason": "already_in_top5", "ranking": current}
        rewritten = list(current)
        rewritten[4], rewritten[selected_idx] = rewritten[selected_idx], rewritten[4]
    elif mode == "top5_chain":
        if not selected or any(doc_id not in current for doc_id in selected):
            return {"valid": False, "reason": "invalid_top5_chain", "ranking": current}
        protected = current[:2]
        selected = [doc_id for doc_id in selected if doc_id not in set(protected)][:3]
        if not selected:
            return {"valid": False, "reason": "chain_inside_protected_prefix", "ranking": current}
        rewritten = protected + selected + [
            doc_id
            for doc_id in current
            if doc_id not in set(protected) and doc_id not in set(selected)
        ]
    else:
        return {"valid": False, "reason": "unknown_rewrite_mode", "ranking": current}

    before_top2 = current[:2]
    before_top5 = current[:5]
    after_top2 = rewritten[:2]
    after_top5 = rewritten[:5]
    displaced = [doc_id for doc_id in before_top5 if doc_id not in set(after_top5)]
    return {
        "valid": True,
        "reason": "",
        "ranking": rewritten,
        "before_top2": before_top2,
        "after_top2": after_top2,
        "before_top5": before_top5,
        "after_top5": after_top5,
        "changed_top2": before_top2 != after_top2,
        "changed_top5": set(before_top5) != set(after_top5),
        "top5_order_changed": before_top5 != after_top5,
        "displaced_passage_indices": displaced,
    }


def proof_certificate_profile(
    group: Mapping[str, Any],
    min_confidence: float = 0.95,
) -> Dict[str, Any]:
    """Validate the logical proof before it authorizes a physical top-k delta."""
    reasons: List[str] = []
    confidence = _bounded_float(group.get("proof_confidence"))
    coverage_fields = (
        "proof_relation_coverage",
        "proof_type_consistency",
        "proof_provenance_coverage",
        "proof_exact_binding_coverage",
    )
    coverages = {
        field: _bounded_float(group.get(field))
        for field in coverage_fields
    }
    identity_required = bool(group.get("proof_identity_required", False))
    identity_coverages = {
        "proof_branch_identity_coverage": _bounded_float(
            group.get("proof_branch_identity_coverage")
        ),
        "proof_branch_qualifier_coverage": _bounded_float(
            group.get("proof_branch_qualifier_coverage")
        ),
    }
    if not bool(group.get("proof_valid", False)):
        reasons.append("proof_invalid")
    if not bool(group.get("branch_complete", False)):
        reasons.append("branch_incomplete")
    for field, value in coverages.items():
        if value < 1.0 - 1e-9:
            reasons.append(f"{field}_incomplete")
    if identity_required:
        for field, value in identity_coverages.items():
            if value < 1.0 - 1e-9:
                reasons.append(f"{field}_incomplete")
    if not bool(group.get("proof_return_projection_valid", False)):
        reasons.append("return_projection_invalid")
    if confidence < float(min_confidence):
        reasons.append("proof_confidence_below_threshold")
    return {
        "admissible": not reasons,
        "rejection_reasons": reasons,
        "proof_confidence": confidence,
        "proof_identity_required": identity_required,
        **coverages,
        **identity_coverages,
    }


def candidate_window_chain_certificate_profile(
    group: Mapping[str, Any],
    *,
    min_confidence: float = 0.55,
) -> Dict[str, Any]:
    """Validate a query-local provenance path before it may change Top-5.

    Candidate-window paths use a lighter certificate schema than the proof
    join executor.  Keep the validation explicit here so a complete path is
    not silently discarded merely because it came from a different physical
    operator, while incomplete or directionally inconsistent paths remain
    shadow-only.
    """
    details = group.get("strict_chain_certificate_details", {}) or {}
    confidence = _bounded_float(group.get("proof_confidence"))
    coverages = {
        "proof_relation_coverage": _bounded_float(
            group.get("proof_relation_coverage")
        ),
        "proof_provenance_coverage": _bounded_float(
            group.get("proof_provenance_coverage")
        ),
    }
    type_consistency = _bounded_float(group.get("proof_type_consistency"))
    reasons: List[str] = []
    if not bool(group.get("strict_chain_certificate", False)):
        reasons.append("strict_chain_certificate_missing")
    if not bool(group.get("provenance_validated", False)):
        reasons.append("provenance_not_validated")
    if _bounded_float(group.get("chain_completeness")) < 1.0 - 1e-9:
        reasons.append("chain_incomplete")
    if not bool(group.get("exact_anchor", details.get("exact_anchor", False))):
        reasons.append("anchor_not_exact")
    if not bool(details.get("direction_consistent", False)):
        reasons.append("direction_inconsistent")
    for field, value in coverages.items():
        if value < 1.0 - 1e-9:
            reasons.append(f"{field}_incomplete")
    if type_consistency <= 0.0:
        reasons.append("answer_type_inconsistent")
    if confidence < float(min_confidence):
        reasons.append("proof_confidence_below_threshold")
    return {
        "admissible": not reasons,
        "rejection_reasons": reasons,
        "proof_confidence": confidence,
        "proof_type_consistency": type_consistency,
        **coverages,
    }


def build_candidate_window_chain_action_specs(
    group: Mapping[str, Any],
    candidate_profiles: Sequence[Mapping[str, Any]],
    base_doc_ids: Sequence[int],
    rank_by_doc: Mapping[int, int],
    *,
    min_confidence: float = 0.55,
    max_top5_insertions: int = 3,
    enable_top5: bool = True,
) -> List[Dict[str, Any]]:
    """Materialize the minimal Top-5 delta for a certified local path.

    The semantic Top-2 prefix is immutable.  Only path members absent from the
    incumbent Top-5 are inserted, and the action is emitted only when the full
    provenance path still fits in the resulting Top-5.  This turns the local
    path compiler into a real physical action without introducing a direct,
    uncalibrated commit.
    """
    certificate = candidate_window_chain_certificate_profile(
        group,
        min_confidence=min_confidence,
    )
    if not enable_top5 or not certificate["admissible"]:
        return []

    current = [int(value) for value in base_doc_ids]
    before_top5 = set(current[:5])
    available = {int(doc_id) for doc_id in rank_by_doc}
    proof_docs = [
        doc_id
        for doc_id in _unique_doc_ids(
            group.get("passage_indices", []) or [],
            limit=max(len(group.get("passage_indices", []) or []), 1),
        )
        if doc_id in available
    ]
    residual = sorted(
        (doc_id for doc_id in proof_docs if doc_id not in before_top5),
        key=lambda doc_id: (int(rank_by_doc.get(doc_id, 10**9)), doc_id),
    )
    if not proof_docs or not residual or len(residual) > max(int(max_top5_insertions), 1):
        return []

    materialized = materialize_counterfactual_ranking(
        current,
        residual,
        "top5_chain",
    )
    if not materialized.get("valid", False) or not set(proof_docs).issubset(
        set(materialized.get("after_top5", []) or [])
    ):
        return []

    profile_by_doc: Dict[int, Dict[str, Any]] = {}
    for item in candidate_profiles:
        try:
            profile_by_doc[int(item.get("passage_idx"))] = dict(item)
        except (AttributeError, TypeError, ValueError):
            continue
    return [{
        "action_name": "candidate_window_chain_top5",
        "rewrite_mode": "top5_chain",
        "selected_passage_indices": residual,
        "selected_profiles": [
            profile_by_doc[doc_id] for doc_id in residual if doc_id in profile_by_doc
        ],
        "profile": {
            "proof_valid": True,
            "branch_complete": True,
            "proof_exact_binding_coverage": 1.0,
            "proof_return_projection_valid": True,
            "proof_member_count": len(proof_docs),
            "proof_certificate_admissible": True,
            "proof_residual_commit": True,
            "proof_residual_doc_count": len(residual),
            "proof_residual_selected_count": len(residual),
            "proof_residual_remaining_count": 0,
            "proof_preserved_top5_count": len(proof_docs) - len(residual),
            "proof_residual_target": "top5",
            "candidate_window_chain_certificate": certificate,
        },
    }]


def build_proof_residual_action_specs(
    group: Mapping[str, Any],
    candidate_profiles: Sequence[Mapping[str, Any]],
    base_doc_ids: Sequence[int],
    rank_by_doc: Mapping[int, int],
    *,
    min_confidence: float = 0.95,
    max_top5_insertions: int = 1,
    enable_top5: bool = True,
    enable_top2: bool = True,
) -> List[Dict[str, Any]]:
    """Compile a minimal physical delta from a complete logical proof.

    Existing top-k proof members act as a materialized prefix.  Only missing
    members are inserted, so repeated evidence cannot consume displacement
    budget.  A top-2 comparison projection is considered only when no Top-5
    residual exists for the same proof group.
    """
    certificate = proof_certificate_profile(group, min_confidence=min_confidence)
    if not certificate["admissible"]:
        return []

    current = [int(value) for value in base_doc_ids]
    before_top2 = set(current[:2])
    before_top5 = set(current[:5])
    available = set(_unique_doc_ids(rank_by_doc.keys(), limit=max(len(rank_by_doc), 1)))

    def available_doc_ids(values: Iterable[Any]) -> List[int]:
        raw_values = list(values or [])
        normalized = _unique_doc_ids(raw_values, limit=max(len(raw_values), 1))
        return [doc_id for doc_id in normalized if doc_id in available]

    proof_docs = available_doc_ids(group.get("passage_indices", []) or [])
    terminal_docs = available_doc_ids(group.get("terminal_passage_indices", []) or [])
    if not proof_docs:
        return []

    profile_by_doc: Dict[int, Dict[str, Any]] = {}
    for item in candidate_profiles:
        try:
            passage_idx = int(item.get("passage_idx"))
        except (AttributeError, TypeError, ValueError):
            continue
        profile_by_doc[passage_idx] = dict(item)
    residual_top5 = sorted(
        (doc_id for doc_id in proof_docs if doc_id not in before_top5),
        key=lambda doc_id: (int(rank_by_doc.get(doc_id, 10**9)), doc_id),
    )
    preserved_top5 = [doc_id for doc_id in proof_docs if doc_id in before_top5]
    base_profile = {
        "proof_certificate_admissible": True,
        "proof_residual_commit": True,
        "proof_residual_doc_count": len(residual_top5),
        "proof_preserved_top5_count": len(preserved_top5),
        "proof_total_doc_count": len(proof_docs),
        "proof_certificate": certificate,
    }
    actions: List[Dict[str, Any]] = []
    max_insertions = max(int(max_top5_insertions), 1)
    if residual_top5:
        if not enable_top5 or len(residual_top5) > max_insertions:
            return actions
        selected = residual_top5
        materialized = materialize_counterfactual_ranking(
            current,
            selected,
            "top5_chain",
        )
        physical_projection_complete = bool(
            materialized.get("valid", False)
            and set(proof_docs).issubset(
                set(materialized.get("after_top5", []) or [])
            )
        )
        if not physical_projection_complete:
            return actions
        actions.append({
            "action_name": "proof_residual_top5",
            "rewrite_mode": "top5_chain",
            "selected_passage_indices": selected,
            "selected_profiles": [
                profile_by_doc[doc_id] for doc_id in selected if doc_id in profile_by_doc
            ],
            "profile": {
                **base_profile,
                "proof_residual_selected_count": len(selected),
                "proof_residual_remaining_count": 0,
                "proof_residual_target": "top5",
                "proof_physical_projection_complete": True,
            },
        })
        return actions

    logical_contract = group.get("logical_contract", {}) or {}
    return_contract = logical_contract.get("return_contract", {}) or {}
    comparison_projection = bool(
        logical_contract.get("comparison_query", False)
        and str(return_contract.get("projection", "") or "") == "branch_anchor"
    )
    if (
        enable_top2
        and comparison_projection
        and len(terminal_docs) == 2
        and set(terminal_docs) != before_top2
    ):
        actions.append({
            "action_name": "proof_certified_terminal_pair_top2",
            "rewrite_mode": "top2_group",
            "selected_passage_indices": terminal_docs,
            "selected_profiles": [
                profile_by_doc[doc_id] for doc_id in terminal_docs if doc_id in profile_by_doc
            ],
            "profile": {
                **base_profile,
                "proof_residual_selected_count": len(
                    [doc_id for doc_id in terminal_docs if doc_id not in before_top2]
                ),
                "proof_residual_remaining_count": 0,
                "proof_residual_target": "top2",
            },
        })
    return actions


def select_post_optimizer_proof_residual(
    proof_sources: Sequence[Mapping[str, Any]],
    base_doc_ids: Sequence[int],
    rank_by_doc: Mapping[int, int],
    *,
    min_confidence: float = 0.95,
    max_top5_insertions: int = 2,
    enable_top5: bool = True,
    enable_top2: bool = False,
) -> Dict[str, Any]:
    """Recompile and select one proof delta after the primary optimizer.

    The primary action may change both the protected Top-2 prefix and which
    proof members are already resident in Top-5.  Residual actions therefore
    have to be compiled against the committed primary ranking, not replayed
    from a stale baseline.  Only one complete proof group is admitted so
    independent groups cannot consume each other's displacement budget.
    """
    current = [int(value) for value in base_doc_ids]
    candidates: List[Dict[str, Any]] = []
    rejection_counts: Dict[str, int] = {
        "no_compilable_delta": 0,
        "invalid_materialization": 0,
        "top2_prefix_changed": 0,
        "incomplete_physical_projection": 0,
    }
    for source_index, source in enumerate(proof_sources or []):
        group = dict(source.get("group", {}) or {})
        profiles = list(source.get("candidate_profiles", []) or [])
        specs = build_proof_residual_action_specs(
            group=group,
            candidate_profiles=profiles,
            base_doc_ids=current,
            rank_by_doc=rank_by_doc,
            min_confidence=min_confidence,
            max_top5_insertions=max_top5_insertions,
            enable_top5=enable_top5,
            enable_top2=enable_top2,
        )
        if not specs:
            rejection_counts["no_compilable_delta"] += 1
            continue
        proof_docs = set(_unique_doc_ids(
            group.get("passage_indices", []) or [],
            limit=max(len(group.get("passage_indices", []) or []), 1),
        ))
        for spec in specs:
            materialized = materialize_counterfactual_ranking(
                current,
                spec.get("selected_passage_indices", []) or [],
                str(spec.get("rewrite_mode", "") or ""),
            )
            if not materialized.get("valid", False):
                rejection_counts["invalid_materialization"] += 1
                continue
            target = str((spec.get("profile", {}) or {}).get(
                "proof_residual_target", ""
            ) or "")
            if target == "top5" and materialized.get("after_top2") != current[:2]:
                rejection_counts["top2_prefix_changed"] += 1
                continue
            if target == "top5" and not proof_docs.issubset(
                set(materialized.get("after_top5", []) or [])
            ):
                rejection_counts["incomplete_physical_projection"] += 1
                continue
            certificate = (spec.get("profile", {}) or {}).get(
                "proof_certificate", {}
            ) or {}
            candidates.append({
                "source_index": source_index,
                "group": group,
                "spec": dict(spec),
                "materialized": materialized,
                "selection_key": (
                    1 if target == "top5" else 0,
                    _bounded_float(certificate.get("proof_confidence")),
                    int((spec.get("profile", {}) or {}).get(
                        "proof_residual_selected_count", 0
                    ) or 0),
                    _bounded_float(group.get("utility")),
                    -sum(
                        int(rank_by_doc.get(int(doc_id), 10**9))
                        for doc_id in spec.get("selected_passage_indices", []) or []
                    ),
                ),
            })

    selected = max(candidates, key=lambda item: item["selection_key"]) if candidates else {}
    return {
        "enabled": True,
        "source_count": len(proof_sources or []),
        "candidate_count": len(candidates),
        "rejection_counts": rejection_counts,
        "selected": bool(selected),
        "selected_source_index": int(selected.get("source_index", -1)),
        "selected_group": dict(selected.get("group", {}) or {}),
        "selected_spec": dict(selected.get("spec", {}) or {}),
        "materialized": dict(selected.get("materialized", {}) or {}),
    }
