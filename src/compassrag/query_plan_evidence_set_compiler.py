"""Ontology-guided compilation of candidate passages into bounded set actions."""

from __future__ import annotations

from itertools import combinations
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: Any, low: float = 0.0, high: float = 1.0) -> float:
    return min(max(_safe_float(value), low), high)


def _strings(values: Iterable[Any]) -> Set[str]:
    return {
        str(value).strip().lower()
        for value in values or []
        if str(value or "").strip()
    }


def _candidate_view(item: Mapping[str, Any], rank_by_doc: Mapping[int, int]) -> Dict[str, Any]:
    try:
        passage_idx = int(item.get("passage_idx", -1))
    except (TypeError, ValueError):
        passage_idx = -1
    contexts = [
        context for context in item.get("contexts", []) or []
        if isinstance(context, Mapping)
    ]
    slot_keys = _strings(item.get("decomposition_slots", []) or [])
    relation_keys: Set[str] = set()
    branch_keys: Set[str] = set()
    provenance_keys: Set[str] = set()
    for context in contexts:
        slot_keys.update(_strings(context.get("slot_ids", []) or []))
        relation_keys.update(_strings(context.get("relations", []) or []))
        plan_id = str(context.get("plan_id", "") or "").strip().lower()
        support_key = str(context.get("support_key", "") or "").strip().lower()
        anchor = str(context.get("anchor_entity", "") or "").strip().lower()
        terminal = str(context.get("end_entity", "") or "").strip().lower()
        if plan_id:
            provenance_keys.add(f"plan:{plan_id}")
        if support_key:
            provenance_keys.add(f"support:{support_key.split('::', 1)[0]}")
        if anchor:
            branch_keys.add(f"anchor:{anchor}")
        if terminal:
            branch_keys.add(f"terminal:{terminal}")
    if not slot_keys:
        slot_keys.update(f"relation:{value}" for value in relation_keys)
    source = str(item.get("source", "") or "").strip().lower()
    if source:
        provenance_keys.add(f"source:{source}")
    text_score = _clamp((item.get("text_verifier", {}) or {}).get("score"))
    path_score = _clamp((item.get("path_aware", {}) or {}).get("score"))
    confidence = _clamp(item.get("confidence"))
    raw_score = max(_safe_float(item.get("score")), 0.0)
    raw_score_norm = raw_score / (1.0 + raw_score)
    quality = (
        0.30 * text_score
        + 0.30 * path_score
        + 0.25 * confidence
        + 0.15 * raw_score_norm
    )
    return {
        "passage_idx": passage_idx,
        "rank": int(rank_by_doc.get(passage_idx, 10**9)),
        "quality": _clamp(quality),
        "confidence": confidence,
        "text_score": text_score,
        "path_score": path_score,
        "slot_keys": slot_keys,
        "branch_keys": branch_keys,
        "provenance_keys": provenance_keys,
        "relation_keys": relation_keys,
        "profile": dict(item),
    }


def _pairwise_redundancy(members: Sequence[Mapping[str, Any]]) -> float:
    similarities: List[float] = []
    for left, right in combinations(members, 2):
        left_keys = set(left.get("slot_keys", set())) | set(left.get("relation_keys", set()))
        right_keys = set(right.get("slot_keys", set())) | set(right.get("relation_keys", set()))
        union = left_keys | right_keys
        similarities.append(len(left_keys & right_keys) / max(len(union), 1))
    return sum(similarities) / max(len(similarities), 1)


def _set_profile(
    members: Sequence[Mapping[str, Any]],
    all_slots: Set[str],
    all_branches: Set[str],
    probe_rows: float,
) -> Dict[str, Any]:
    slots = set().union(*(set(item.get("slot_keys", set())) for item in members))
    branches = set().union(*(set(item.get("branch_keys", set())) for item in members))
    provenance = set().union(*(set(item.get("provenance_keys", set())) for item in members))
    provenance_signatures = {
        tuple(sorted(set(item.get("provenance_keys", set()))))
        for item in members
    }
    member_slot_max = max((len(item.get("slot_keys", set())) for item in members), default=0)
    member_branch_max = max((len(item.get("branch_keys", set())) for item in members), default=0)
    slot_gain = max(len(slots) - member_slot_max, 0)
    branch_gain = max(len(branches) - member_branch_max, 0)
    redundancy = _pairwise_redundancy(members)
    quality_mean = sum(_safe_float(item.get("quality")) for item in members) / max(len(members), 1)
    slot_coverage = len(slots) / max(len(all_slots), 1) if all_slots else 0.0
    branch_coverage = len(branches) / max(len(all_branches), 1) if all_branches else 0.0
    provenance_diversity = len(provenance_signatures) / max(len(members), 1)
    external_count = sum(int(item.get("rank", 10**9)) > 5 for item in members)
    rank_cost = sum(
        min(max(int(item.get("rank", 10**9)) - 5, 0) / 95.0, 1.0)
        for item in members
    ) / max(len(members), 1)
    normalized_probe_cost = min(max(probe_rows, 0.0) / 128.0, 1.0)
    execution_cost = _clamp(
        0.45 * rank_cost
        + 0.35 * normalized_probe_cost
        + 0.20 * (len(members) / 3.0)
    )
    score = (
        0.42 * quality_mean
        + 0.20 * slot_coverage
        + 0.14 * branch_coverage
        + 0.12 * provenance_diversity
        + 0.07 * min(slot_gain / 2.0, 1.0)
        + 0.05 * min(branch_gain / 2.0, 1.0)
        - 0.12 * redundancy
        - 0.08 * execution_cost
    )
    return {
        "compiler_score": round(_clamp(score), 6),
        "slot_coverage": round(_clamp(slot_coverage), 6),
        "branch_coverage": round(_clamp(branch_coverage), 6),
        "provenance_diversity": round(_clamp(provenance_diversity), 6),
        "redundancy": round(_clamp(redundancy), 6),
        "execution_cost": round(_clamp(execution_cost), 6),
        "slot_gain": int(slot_gain),
        "branch_gain": int(branch_gain),
        "member_count": len(members),
        "external_count": int(external_count),
        "slot_keys": sorted(slots),
        "branch_keys": sorted(branches),
        "provenance_keys": sorted(provenance),
        "quality_mean": round(_clamp(quality_mean), 6),
        "probe_rows": int(max(probe_rows, 0.0)),
    }


def compile_ontology_evidence_sets(
    warehouse_samples: Sequence[Mapping[str, Any]],
    base_doc_ids: Sequence[int],
    probes: Sequence[Mapping[str, Any]] = (),
    max_candidates: int = 10,
    beam_width: int = 16,
    max_top2_actions: int = 6,
    max_top5_actions: int = 10,
    require_complementarity: bool = True,
) -> Dict[str, Any]:
    """Compile bounded top-2/top-5 set actions from ontology-bearing candidates."""
    rank_by_doc = {int(doc_id): rank + 1 for rank, doc_id in enumerate(base_doc_ids)}
    views = []
    seen = set()
    for item in warehouse_samples or []:
        if not isinstance(item, Mapping):
            continue
        view = _candidate_view(item, rank_by_doc)
        passage_idx = int(view["passage_idx"])
        if passage_idx < 0 or passage_idx in seen or passage_idx not in rank_by_doc:
            continue
        if int(view["rank"]) <= 2:
            continue
        seen.add(passage_idx)
        views.append(view)
    views.sort(
        key=lambda item: (
            float(item["quality"]),
            len(item["slot_keys"]),
            len(item["branch_keys"]),
            -int(item["rank"]),
        ),
        reverse=True,
    )
    views = views[: max(int(max_candidates), 2)]
    all_slots = set().union(*(item["slot_keys"] for item in views)) if views else set()
    all_branches = set().union(*(item["branch_keys"] for item in views)) if views else set()
    probe_rows = sum(
        max(_safe_float(probe.get("matched_count")), 0.0)
        for probe in probes or []
        if isinstance(probe, Mapping)
    )



    beam: List[Tuple[int, ...]] = [tuple()]
    states: Dict[Tuple[int, ...], Dict[str, Any]] = {}
    for _ in range(3):
        expanded = set(beam)
        for state in beam:
            start = state[-1] + 1 if state else 0
            for idx in range(start, len(views)):
                expanded.add((*state, idx))
        scored = []
        for state in expanded:
            if not state:
                continue
            members = [views[idx] for idx in state]
            profile = _set_profile(members, all_slots, all_branches, probe_rows)
            states[state] = profile
            scored.append((float(profile["compiler_score"]), state))
        beam = [state for _, state in sorted(scored, reverse=True)[: max(int(beam_width), 1)]]

    actions: List[Dict[str, Any]] = []
    for state, profile in states.items():
        members = [views[idx] for idx in state]
        member_count = len(members)
        if member_count < 2:
            continue
        complementary = bool(
            int(profile["slot_gain"]) > 0
            or int(profile["branch_gain"]) > 0
            or float(profile["provenance_diversity"]) >= 1.0
        )
        if require_complementarity and not complementary:
            continue
        passage_indices = [int(item["passage_idx"]) for item in members]
        common = {
            "selected_passage_indices": passage_indices,
            "member_profiles": [dict(item["profile"]) for item in members],
            "profile": profile,
            "utility": float(profile["compiler_score"]),
            "semantic_score": _clamp(
                0.45 * float(profile["quality_mean"])
                + 0.30 * float(profile["slot_coverage"])
                + 0.25 * float(profile["branch_coverage"])
            ),
            "hard_admissible": complementary,
        }
        if member_count == 2:
            actions.append({
                **common,
                "action_name": "ontology_evidence_set_top2",
                "rewrite_mode": "top2_group",
                "target": "top2",
            })
        if any(int(item["rank"]) > 5 for item in members):
            actions.append({
                **common,
                "action_name": "ontology_evidence_set_top5",
                "rewrite_mode": "top5_chain",
                "target": "top5",
            })

    top2 = sorted(
        (item for item in actions if item["target"] == "top2"),
        key=lambda item: float(item["utility"]),
        reverse=True,
    )[: max(int(max_top2_actions), 0)]
    top5 = sorted(
        (item for item in actions if item["target"] == "top5"),
        key=lambda item: float(item["utility"]),
        reverse=True,
    )[: max(int(max_top5_actions), 0)]
    return {
        "actions": [*top2, *top5],
        "candidate_count": len(views),
        "compiled_state_count": len(states),
        "top2_action_count": len(top2),
        "top5_action_count": len(top5),
        "beam_width": max(int(beam_width), 1),
        "max_candidates": max(int(max_candidates), 2),
        "all_slot_count": len(all_slots),
        "all_branch_count": len(all_branches),
        "probe_rows": int(probe_rows),
    }
