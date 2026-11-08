#!/usr/bin/env python3
"""Validate, sanitize, and repair executable evidence schemas on held-out workloads."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
COMPASSRAG_SRC = ROOT / "src" / "compassrag"
if str(COMPASSRAG_SRC) not in sys.path:
    sys.path.insert(0, str(COMPASSRAG_SRC))

from query_plan_ontology import (
    QueryPlanOntologyMatcher,
    _relation_family,
    build_hierarchical_routing_model,
    get_answers,
    get_evidence_triples,
    get_question,
    infer_answer_type,
    load_json_or_jsonl,
    normalize_text,
    question_intent,
    relation_signature,
    routing_parameters_for_scores,
    tokenize,
)


TYPE_TERMS = {
    "date": {"date", "day", "month", "year", "century", "time"},
    "location": {"location", "place", "city", "country", "county", "state", "region"},
    "person": {"person", "people", "human", "individual"},
    "work": {"work", "film", "movie", "book", "song", "album", "publication"},
    "nationality": {"nationality", "citizenship", "country"},
    "entity": {"entity", "value", "name"},
    "boolean": {"boolean", "yes", "no", "true", "false"},
    "quantity": {"quantity", "count", "number", "dose", "rate", "percentage"},
    "gene": {"gene", "genetic", "mutation", "variant", "allele"},
    "drug": {"drug", "medication", "compound", "treatment", "therapy"},
    "protein": {"protein", "enzyme", "receptor", "biomarker"},
    "disease": {"disease", "disorder", "syndrome", "condition"},
    "biomedical_entity": {"biomedical", "entity", "concept"},
    "list": {"list", "entities", "items", "values"},
    "definition": {"definition", "description", "meaning", "concept"},
}
GENERIC_REJECT_TERMS = {"unknown", "not available", "n/a", "none", "null"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def plan_relations(
    plan: Mapping[str, Any], dataset: str = ""
) -> Tuple[set[str], set[str]]:
    signatures: set[str] = set()
    families: set[str] = set()
    for step in plan.get("path", []) or []:
        if not isinstance(step, Mapping):
            continue
        raw = str(step.get("relation_signature") or step.get("relation_family") or "")
        signature = relation_signature(raw, dataset=dataset)
        family = _relation_family(str(step.get("relation_family") or raw))
        if signature:
            signatures.add(signature)
        if family:
            families.add(family)
        for candidate in step.get("relation_signature_alternatives", []) or []:
            candidate_signature = relation_signature(candidate, dataset=dataset)
            if candidate_signature:
                signatures.add(candidate_signature)
        for candidate in step.get("relation_family_alternatives", []) or []:
            candidate_family = _relation_family(str(candidate))
            if candidate_family:
                families.add(candidate_family)
    return signatures, families


def row_profile(row: Mapping[str, Any], dataset: str = "") -> Dict[str, Any]:
    query = get_question(dict(row))
    answers = get_answers(dict(row))
    triples = get_evidence_triples(dict(row))
    signatures = {relation_signature(item[1], dataset=dataset) for item in triples}
    signatures.discard("")
    families = {_relation_family(item[1]) for item in triples}
    families.discard("")
    route_answer_type = infer_answer_type(query, [])
    target_answer_type = infer_answer_type(query, answers)
    route_intent = question_intent(query, route_answer_type, [])
    target_intent = question_intent(query, target_answer_type, families)
    return {
        "query": query,
        "query_tokens": set(tokenize(query)),


        "route_answer_type": route_answer_type,
        "route_intent": route_intent,
        "target_answer_type": target_answer_type,
        "target_intent": target_intent,
        "relation_signatures": signatures,
        "relation_families": families,
        "hop_count": len(triples),
    }


def plan_profile(plan: Mapping[str, Any], dataset: str = "") -> Dict[str, Any]:
    signatures, families = plan_relations(plan, dataset=dataset)
    trigger_tokens = {
        token
        for trigger in plan.get("triggers", []) or []
        for token in tokenize(str(trigger))
    }
    path = [item for item in plan.get("path", []) or [] if isinstance(item, Mapping)]
    intent = str(plan.get("intent", "") or "")
    return {
        "trigger_tokens": trigger_tokens,
        "answer_type": str(plan.get("answer_type", "entity") or "entity"),
        "intent": intent,
        "comparison": intent == "comparison" or "comparison" in intent,
        "relation_signatures": signatures,
        "relation_families": families,
        "hop_count": len(path),
    }


def route_score(profile: Mapping[str, Any], workload: Mapping[str, Any]) -> float:
    if (
        str(profile.get("intent", "")).startswith("biomedical_")
        and profile["answer_type"] not in {
            workload["route_answer_type"],
            "entity",
            "biomedical_entity",
        }
    ):
        return 0.0
    trigger_tokens = set(profile["trigger_tokens"])
    query_tokens = set(workload["query_tokens"])
    overlap = len(trigger_tokens & query_tokens)
    if not overlap:
        return 0.0
    trigger_score = overlap / max(min(len(trigger_tokens), len(query_tokens)), 1)
    answer_score = (
        1.0 if profile["answer_type"] == workload["route_answer_type"] else 0.0
    )
    relation_hints = set(profile["relation_signatures"]) | set(profile["relation_families"])
    relation_tokens = {token for value in relation_hints for token in tokenize(value)}
    relation_score = len(relation_tokens & query_tokens) / max(len(relation_tokens), 1)
    return 0.62 * trigger_score + 0.25 * answer_score + 0.13 * relation_score


def structurally_compatible(profile: Mapping[str, Any], workload: Mapping[str, Any]) -> bool:
    plan_answer = str(profile["answer_type"])
    row_answer = str(workload["target_answer_type"])
    answer_ok = plan_answer == row_answer or plan_answer in {"entity", "biomedical_entity"}
    if not answer_ok:
        return False
    plan_comparison = bool(profile["comparison"])
    row_comparison = workload["target_intent"] == "comparison"
    if plan_comparison != row_comparison:
        return False
    plan_signatures = set(profile["relation_signatures"])
    row_signatures = set(workload["relation_signatures"])
    plan_families = set(profile["relation_families"])
    row_families = set(workload["relation_families"])
    if row_signatures or row_families:
        return bool(plan_signatures & row_signatures or plan_families & row_families)
    return len(set(profile["trigger_tokens"]) & set(workload["query_tokens"])) >= 2


def empty_stats() -> MutableMapping[str, float]:
    return defaultdict(float)


def evaluate(
    plans: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    top_k: int,
    min_route_score: float,
    routing_model: Mapping[str, Any] | None = None,
    min_plan_confidence: float = 0.0,
    deployment_schema_ids: set[str] | None = None,
) -> Dict[str, Any]:
    dataset = str((routing_model or {}).get("dataset", "") or "")
    profiles = {
        str(plan.get("id")): plan_profile(plan, dataset=dataset) for plan in plans
    }
    matcher = None
    if routing_model:
        matcher = QueryPlanOntologyMatcher({
            "dataset": routing_model.get("dataset", ""),
            "plans": list(plans),
            "routing_model": dict(routing_model),
        })
    per_plan: Dict[str, MutableMapping[str, float]] = {
        str(plan.get("id")): empty_stats() for plan in plans
    }
    query_covered = queries_with_compatible = 0
    routed_pairs = true_pairs = answer_type_violations = branch_violations = 0
    route_class_counts: Counter = Counter()
    for row in rows:
        workload = row_profile(row, dataset=dataset)
        compatible_ids = {
            plan_id
            for plan_id, profile in profiles.items()
            if structurally_compatible(profile, workload)
        }
        queries_with_compatible += int(bool(compatible_ids))
        if matcher is not None:
            scored = [
                (score, str(plan.get("id", "")))
                for score, plan in matcher.rank(
                    workload["query"], min_confidence=min_plan_confidence
                )
            ]
        else:
            scored = sorted(
                (
                    (route_score(profile, workload), plan_id)
                    for plan_id, profile in profiles.items()
                ),
                reverse=True,
            )
        policy = dict((routing_model or {}).get("policy", {}) or {})
        adaptive = str(policy.get("routing_mode", "")) == "score_margin_adaptive"
        route_limit, route_threshold, route_class, _ = routing_parameters_for_scores(
            policy,
            [float(score) for score, _ in scored],
            requested_top_k=top_k,
            min_route_score=None if adaptive else min_route_score,
        )
        route_class_counts[route_class] += 1
        routed_ids = {
            plan_id
            for score, plan_id in scored[:route_limit]
            if score >= route_threshold
        }
        if deployment_schema_ids is not None:
            routed_ids &= deployment_schema_ids
        query_covered += int(bool(routed_ids & compatible_ids))
        for plan_id in compatible_ids:
            per_plan[plan_id]["opportunities"] += 1
            if plan_id not in routed_ids:
                per_plan[plan_id]["fn"] += 1
        for plan_id in routed_ids:
            stats = per_plan[plan_id]
            profile = profiles[plan_id]
            stats["routed"] += 1
            routed_pairs += 1
            if plan_id in compatible_ids:
                stats["tp"] += 1
                true_pairs += 1
            else:
                stats["fp"] += 1
            if profile["answer_type"] not in {
                workload["target_answer_type"],
                "entity",
            }:
                stats["answer_type_violations"] += 1
                answer_type_violations += 1
            if bool(profile["comparison"]) != (
                workload["target_intent"] == "comparison"
            ):
                stats["branch_violations"] += 1
                branch_violations += 1
    results: Dict[str, Dict[str, Any]] = {}
    for plan_id, raw in per_plan.items():
        routed = int(raw["routed"])
        tp = int(raw["tp"])
        fp = int(raw["fp"])
        opportunities = int(raw["opportunities"])
        precision = tp / max(routed, 1)
        recall = tp / max(opportunities, 1)
        posterior_precision = (tp + 1.0) / (routed + 2.0)
        results[plan_id] = {
            "routed": routed,
            "opportunities": opportunities,
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": int(raw["fn"]),
            "precision": precision,
            "recall": recall,
            "posterior_precision": posterior_precision,
            "hard_negative_false_positive_rate": fp / max(routed, 1),
            "answer_type_violations": int(raw["answer_type_violations"]),
            "branch_violations": int(raw["branch_violations"]),
        }
    return {
        "queries": len(rows),
        "queries_with_compatible_schema": queries_with_compatible,
        "covered_queries": query_covered,
        "coverage": query_covered / max(len(rows), 1),
        "conditional_coverage": query_covered / max(queries_with_compatible, 1),
        "routed_pairs": routed_pairs,
        "true_positive_pairs": true_pairs,
        "routing_precision": true_pairs / max(routed_pairs, 1),
        "answer_type_violation_rate": answer_type_violations / max(routed_pairs, 1),
        "branch_violation_rate": branch_violations / max(routed_pairs, 1),
        "route_class_counts": dict(route_class_counts),
        "plans": results,
    }


def calibrate_routing_policy(
    plans: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    routing_model: Mapping[str, Any],
    max_top_m: int,
    target_precision: float,
    min_plan_confidence: float = 0.0,
    max_thresholds: int = 64,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Choose parent/leaf Top-M and a score threshold on calibration only."""
    parent_limit = max(
        min(len(routing_model.get("parent_vectors", {})), 3),
        1,
    )
    score_model = deepcopy(dict(routing_model))
    score_model["policy"] = {
        **dict(score_model.get("policy", {}) or {}),
        "parent_top_m": parent_limit,
    }
    matcher = QueryPlanOntologyMatcher({
        "dataset": routing_model.get("dataset", ""),
        "plans": list(plans),
        "routing_model": score_model,
    })
    scores = sorted({
        round(float(score), 8)
        for row in rows
        for score, _ in matcher.rank(
            get_question(dict(row)), min_confidence=min_plan_confidence
        )
    })
    threshold_limit = max(int(max_thresholds), 2)
    if len(scores) > threshold_limit:
        scores = [
            scores[round(index * (len(scores) - 1) / (threshold_limit - 1))]
            for index in range(threshold_limit)
        ]
    thresholds = sorted({0.0, *scores})
    dataset = str(routing_model.get("dataset", "") or "")
    profiles = {
        str(plan.get("id")): plan_profile(plan, dataset=dataset) for plan in plans
    }
    workloads = [row_profile(row, dataset=dataset) for row in rows]
    compatible_by_row = [
        {
            plan_id
            for plan_id, profile in profiles.items()
            if structurally_compatible(profile, workload)
        }
        for workload in workloads
    ]
    ranked_by_parent: Dict[int, List[List[Tuple[float, str]]]] = {}

    def aggregate_summary(
        parent_top_m: int,
        leaf_top_m: int,
        threshold: float,
    ) -> Dict[str, Any]:
        if parent_top_m not in ranked_by_parent:
            cached_model = deepcopy(dict(routing_model))
            cached_model["policy"] = {
                **dict(cached_model.get("policy", {}) or {}),
                "parent_top_m": parent_top_m,
            }
            cached_matcher = QueryPlanOntologyMatcher({
                "dataset": dataset,
                "plans": list(plans),
                "routing_model": cached_model,
            })
            ranked_by_parent[parent_top_m] = [
                [
                    (float(score), str(plan.get("id", "")))
                    for score, plan in cached_matcher.rank(
                        workload["query"], min_confidence=min_plan_confidence
                    )
                ]
                for workload in workloads
            ]

        query_covered = routed_pairs = true_pairs = 0
        answer_type_violations = branch_violations = 0
        for workload, compatible_ids, ranked in zip(
            workloads,
            compatible_by_row,
            ranked_by_parent[parent_top_m],
        ):
            routed_ids = {
                plan_id
                for score, plan_id in ranked[:leaf_top_m]
                if score >= threshold
            }
            query_covered += int(bool(routed_ids & compatible_ids))
            routed_pairs += len(routed_ids)
            true_pairs += len(routed_ids & compatible_ids)
            for plan_id in routed_ids:
                profile = profiles[plan_id]
                if profile["answer_type"] not in {
                    workload["target_answer_type"],
                    "entity",
                }:
                    answer_type_violations += 1
                if bool(profile["comparison"]) != (
                    workload["target_intent"] == "comparison"
                ):
                    branch_violations += 1
        queries_with_compatible = sum(bool(value) for value in compatible_by_row)
        return {
            "queries": len(rows),
            "queries_with_compatible_schema": queries_with_compatible,
            "covered_queries": query_covered,
            "coverage": query_covered / max(len(rows), 1),
            "conditional_coverage": query_covered / max(queries_with_compatible, 1),
            "routed_pairs": routed_pairs,
            "true_positive_pairs": true_pairs,
            "routing_precision": true_pairs / max(routed_pairs, 1),
            "answer_type_violation_rate": answer_type_violations / max(routed_pairs, 1),
            "branch_violation_rate": branch_violations / max(routed_pairs, 1),
        }

    trials = []
    for parent_top_m in range(1, parent_limit + 1):
        for leaf_top_m in range(1, max(min(max_top_m, len(plans)), 1) + 1):
            for threshold in thresholds:
                summary = aggregate_summary(
                    parent_top_m,
                    leaf_top_m,
                    threshold,
                )
                precision = float(summary["routing_precision"])
                coverage = float(summary["coverage"])
                feasible = precision >= target_precision
                f1 = 2.0 * precision * coverage / max(precision + coverage, 1e-12)
                trials.append({
                    "key": (
                        feasible,
                        coverage if feasible else f1,
                        precision,
                        -int(summary["routed_pairs"]),
                        -leaf_top_m,
                        -parent_top_m,
                        -threshold,
                    ),
                    "feasible": feasible,
                    "parent_top_m": parent_top_m,
                    "leaf_top_m": leaf_top_m,
                    "threshold": threshold,
                    "summary": summary,
                })
    if trials:
        best = max(trials, key=lambda item: item["key"])
        best_model = deepcopy(dict(routing_model))
        best_model["policy"] = {
            **dict(best_model.get("policy", {}) or {}),
            "parent_top_m": int(best["parent_top_m"]),
        }
        best["summary"] = evaluate(
            plans,
            rows,
            top_k=int(best["leaf_top_m"]),
            min_route_score=float(best["threshold"]),
            routing_model=best_model,
            min_plan_confidence=min_plan_confidence,
        )
    else:
        best = {
            "feasible": False,
            "parent_top_m": 1,
            "leaf_top_m": 1,
            "threshold": 0.0,
            "summary": evaluate(
                plans, rows, 1, 0.0, routing_model, min_plan_confidence
            ),
        }
    policy = {
        "parent_top_m": int(best["parent_top_m"]),
        "leaf_top_m": int(best["leaf_top_m"]),
        "min_score": round(float(best["threshold"]), 8),
        "target_precision": float(target_precision),
        "target_precision_satisfied": bool(best["feasible"]),
        "source": "calibration_grid_search",
        "objective": "maximize_coverage_subject_to_routing_precision",
        "calibration_queries": len(rows),
    }
    diagnostics = {
        key: value for key, value in best["summary"].items() if key != "plans"
    }
    diagnostics["candidate_policy_count"] = len(trials)
    return policy, diagnostics


def calibrate_margin_adaptive_routing_policy(
    plans: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    routing_model: Mapping[str, Any],
    max_top_m: int,
    target_precision: float,
    min_plan_confidence: float = 0.0,
    max_thresholds: int = 32,
    max_margin_cutoffs: int = 6,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Learn query-local access budgets from routing uncertainty.

    A large top-1/top-2 score margin denotes a decisive query and receives a
    narrow leaf budget. Ambiguous queries receive a wider budget. All cutoffs,
    budgets, and score thresholds are selected on calibration rows only.
    """
    parent_limit = max(
        min(len(routing_model.get("parent_vectors", {})), 3),
        1,
    )
    leaf_limit = max(min(int(max_top_m), len(plans)), 1)
    dataset = str(routing_model.get("dataset", "") or "")
    profiles = {
        str(plan.get("id")): plan_profile(plan, dataset=dataset) for plan in plans
    }
    workloads = [row_profile(row, dataset=dataset) for row in rows]
    compatible_by_row = [
        {
            plan_id
            for plan_id, profile in profiles.items()
            if structurally_compatible(profile, workload)
        }
        for workload in workloads
    ]
    ranked_by_parent: Dict[int, List[List[Tuple[float, str]]]] = {}

    def ranked_rows(parent_top_m: int) -> List[List[Tuple[float, str]]]:
        if parent_top_m not in ranked_by_parent:
            candidate_model = deepcopy(dict(routing_model))
            candidate_model["policy"] = {
                **dict(candidate_model.get("policy", {}) or {}),
                "parent_top_m": parent_top_m,
            }
            matcher = QueryPlanOntologyMatcher({
                "dataset": dataset,
                "plans": list(plans),
                "routing_model": candidate_model,
            })
            ranked_by_parent[parent_top_m] = [
                [
                    (float(score), str(plan.get("id", "")))
                    for score, plan in matcher.rank(
                        workload["query"], min_confidence=min_plan_confidence
                    )
                ]
                for workload in workloads
            ]
        return ranked_by_parent[parent_top_m]

    reference_rankings = ranked_rows(parent_limit)
    raw_scores = sorted({
        round(score, 8)
        for ranked in reference_rankings
        for score, _ in ranked
    })
    threshold_limit = max(int(max_thresholds), 2)
    if len(raw_scores) > threshold_limit:
        raw_scores = [
            raw_scores[round(index * (len(raw_scores) - 1) / (threshold_limit - 1))]
            for index in range(threshold_limit)
        ]
    thresholds = sorted({0.0, *raw_scores})

    raw_margins = sorted({
        round(ranked[0][0] - ranked[1][0], 8)
        for ranked in reference_rankings
        if len(ranked) >= 2
    })
    margin_limit = max(int(max_margin_cutoffs), 2)
    if len(raw_margins) > margin_limit:
        raw_margins = [
            raw_margins[round(index * (len(raw_margins) - 1) / (margin_limit - 1))]
            for index in range(margin_limit)
        ]
    upper_margin = (max(raw_margins) + 1e-8) if raw_margins else 1.0
    margin_cutoffs = sorted({0.0, upper_margin, *raw_margins})

    def aggregate_summary(
        parent_top_m: int,
        decisive_top_m: int,
        ambiguous_top_m: int,
        margin_cutoff: float,
        threshold: float,
    ) -> Dict[str, Any]:
        query_covered = routed_pairs = true_pairs = 0
        answer_type_violations = branch_violations = 0
        route_class_counts: Counter = Counter()
        for workload, compatible_ids, ranked in zip(
            workloads,
            compatible_by_row,
            ranked_rows(parent_top_m),
        ):
            margin = ranked[0][0] - ranked[1][0] if len(ranked) >= 2 else 1.0
            route_class = "decisive" if margin >= margin_cutoff else "ambiguous"
            route_class_counts[route_class] += 1
            route_limit = (
                decisive_top_m if route_class == "decisive" else ambiguous_top_m
            )
            routed_ids = {
                plan_id
                for score, plan_id in ranked[:route_limit]
                if score >= threshold
            }
            query_covered += int(bool(routed_ids & compatible_ids))
            routed_pairs += len(routed_ids)
            true_pairs += len(routed_ids & compatible_ids)
            for plan_id in routed_ids:
                profile = profiles[plan_id]
                if profile["answer_type"] not in {
                    workload["target_answer_type"],
                    "entity",
                }:
                    answer_type_violations += 1
                if bool(profile["comparison"]) != (
                    workload["target_intent"] == "comparison"
                ):
                    branch_violations += 1
        queries_with_compatible = sum(bool(value) for value in compatible_by_row)
        return {
            "queries": len(rows),
            "queries_with_compatible_schema": queries_with_compatible,
            "covered_queries": query_covered,
            "coverage": query_covered / max(len(rows), 1),
            "conditional_coverage": query_covered / max(queries_with_compatible, 1),
            "routed_pairs": routed_pairs,
            "true_positive_pairs": true_pairs,
            "routing_precision": true_pairs / max(routed_pairs, 1),
            "answer_type_violation_rate": answer_type_violations / max(routed_pairs, 1),
            "branch_violation_rate": branch_violations / max(routed_pairs, 1),
            "route_class_counts": dict(route_class_counts),
        }

    trials = []
    for parent_top_m in range(1, parent_limit + 1):
        for decisive_top_m in range(1, leaf_limit + 1):
            for ambiguous_top_m in range(decisive_top_m, leaf_limit + 1):
                for margin_cutoff in margin_cutoffs:
                    for threshold in thresholds:
                        summary = aggregate_summary(
                            parent_top_m,
                            decisive_top_m,
                            ambiguous_top_m,
                            margin_cutoff,
                            threshold,
                        )
                        precision = float(summary["routing_precision"])
                        coverage = float(summary["coverage"])
                        feasible = precision >= target_precision
                        f1 = 2.0 * precision * coverage / max(
                            precision + coverage, 1e-12
                        )
                        trials.append({
                            "key": (
                                feasible,
                                coverage if feasible else f1,
                                precision,
                                -int(summary["routed_pairs"]),
                                -ambiguous_top_m,
                                -decisive_top_m,
                                -parent_top_m,
                                -threshold,
                            ),
                            "feasible": feasible,
                            "parent_top_m": parent_top_m,
                            "decisive_top_m": decisive_top_m,
                            "ambiguous_top_m": ambiguous_top_m,
                            "margin_cutoff": margin_cutoff,
                            "threshold": threshold,
                            "summary": summary,
                        })

    if trials:
        best = max(trials, key=lambda item: item["key"])
    else:
        best = {
            "feasible": False,
            "parent_top_m": 1,
            "decisive_top_m": 1,
            "ambiguous_top_m": 1,
            "margin_cutoff": 0.0,
            "threshold": 0.0,
            "summary": {},
        }
    policy = {
        "parent_top_m": int(best["parent_top_m"]),
        "leaf_top_m": int(best["ambiguous_top_m"]),
        "min_score": round(float(best["threshold"]), 8),
        "routing_mode": "score_margin_adaptive",
        "margin_cutoff": round(float(best["margin_cutoff"]), 8),
        "class_policies": {
            "decisive": {
                "leaf_top_m": int(best["decisive_top_m"]),
                "min_score": round(float(best["threshold"]), 8),
            },
            "ambiguous": {
                "leaf_top_m": int(best["ambiguous_top_m"]),
                "min_score": round(float(best["threshold"]), 8),
            },
        },
        "target_precision": float(target_precision),
        "target_precision_satisfied": bool(best["feasible"]),
        "source": "calibration_margin_adaptive_grid_search",
        "objective": "maximize_coverage_subject_to_precision_and_access_budget",
        "calibration_queries": len(rows),
        "inference_features": ["top1_top2_score_margin"],
        "dataset_name_feature": False,
    }
    final_model = deepcopy(dict(routing_model))
    final_model["policy"] = policy
    summary = evaluate(
        plans,
        rows,
        top_k=int(policy["leaf_top_m"]),
        min_route_score=float(policy["min_score"]),
        routing_model=final_model,
        min_plan_confidence=min_plan_confidence,
    )
    diagnostics = {key: value for key, value in summary.items() if key != "plans"}
    diagnostics["candidate_policy_count"] = len(trials)
    diagnostics["margin_candidate_count"] = len(margin_cutoffs)
    return policy, diagnostics


def sanitized_type_terms(answer_type: str) -> List[str]:
    return sorted(TYPE_TERMS.get(answer_type, TYPE_TERMS["entity"]))


def sanitize_plan(plan: Mapping[str, Any]) -> Tuple[Dict[str, Any], Dict[str, int]]:
    sanitized = deepcopy(dict(plan))
    answer_type = str(sanitized.get("answer_type", "entity") or "entity").lower()
    old_constraints = dict(sanitized.get("answer_constraints", {}) or {})
    old_allowed = list(old_constraints.get("allowed_terms", []) or [])
    old_rules = list(old_constraints.get("object_type_rules", []) or [])
    allowed = sanitized_type_terms(answer_type)
    rules = [
        str(rule).strip()
        for rule in old_rules
        if str(rule).strip()
        and len(str(rule)) <= 180
        and not any(char.isdigit() for char in str(rule))
    ]
    default_rule = f"answer_type={answer_type}"
    if default_rule not in rules:
        rules.insert(0, default_rule)
    sanitized["answer_constraints"] = {
        "allowed_terms": allowed,
        "reject_terms": sorted(GENERIC_REJECT_TERMS),
        "object_type_rules": rules[:8],
    }
    removed_step_terms = 0
    path = []
    for step in sanitized.get("path", []) or []:
        if not isinstance(step, Mapping):
            continue
        updated = deepcopy(dict(step))
        old_step_allowed = list(updated.get("terminal_object_allow_terms", []) or [])
        removed_step_terms += max(len(old_step_allowed) - len(allowed), 0)
        if updated.get("to") == "answer" or old_step_allowed:
            updated["terminal_object_allow_terms"] = allowed
            updated["terminal_object_reject_terms"] = sorted(GENERIC_REJECT_TERMS)
        path.append(updated)
    sanitized["path"] = path
    example_ids = [
        str(item.get("id", ""))
        for item in sanitized.get("examples", []) or []
        if isinstance(item, Mapping) and item.get("id")
    ]
    sanitized["training_provenance"] = {
        "example_count": len(sanitized.get("examples", []) or []),
        "example_id_hashes": [
            hashlib.sha256(value.encode("utf-8")).hexdigest()[:16] for value in example_ids
        ],
        "runtime_uses_examples": False,
    }
    sanitized.pop("examples", None)
    removed_supervision_fields = 0
    for key in list(sanitized):
        if key.endswith("_supervision"):
            sanitized.pop(key, None)
            removed_supervision_fields += 1
    intent = str(sanitized.get("intent", "") or "")
    sanitized["integrity_constraints"] = {
        "domain_type": str(
            ((sanitized.get("anchor_slots", []) or [{}])[0] or {}).get("type", "entity")
        ),
        "range_type": answer_type,
        "min_hops": len(path),
        "max_hops": len(path),
        "branch_cardinality": 2 if "comparison" in intent else 1,
        "relation_directions": [str(item.get("direction", "forward")) for item in path],
        "acyclic": True,
    }
    return sanitized, {
        "removed_answer_allowed_terms": len(old_allowed),
        "removed_instance_rules": max(len(old_rules) - len(rules), 0),
        "removed_step_allowed_terms": removed_step_terms,
        "removed_examples": len(example_ids),
        "removed_supervision_fields": removed_supervision_fields,
    }


def validation_score(plan: Mapping[str, Any]) -> float:
    cq = dict(plan.get("competency_validation", {}).get("cq_holdout", {}) or {})
    calibration = dict(plan.get("competency_validation", {}).get("calibration", {}) or {})



    evidence = calibration if calibration else cq
    return (
        0.60 * float(evidence.get("posterior_precision", 0.0) or 0.0)
        + 0.30 * float(evidence.get("recall", 0.0) or 0.0)
        + 0.10 * min(float(plan.get("support", 0.0) or 0.0) / 20.0, 1.0)
    )


def detect_conflicts(plans: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    conflicts = []
    for left_index, left in enumerate(plans):
        left_profile = plan_profile(left)
        for right in plans[left_index + 1:]:
            right_profile = plan_profile(right)
            union = left_profile["trigger_tokens"] | right_profile["trigger_tokens"]
            overlap = left_profile["trigger_tokens"] & right_profile["trigger_tokens"]
            trigger_jaccard = len(overlap) / max(len(union), 1)
            if trigger_jaccard < 0.75:
                continue
            answer_conflict = left_profile["answer_type"] != right_profile["answer_type"]
            relation_conflict = not bool(
                left_profile["relation_signatures"] & right_profile["relation_signatures"]
                or left_profile["relation_families"] & right_profile["relation_families"]
            )
            if not answer_conflict and not relation_conflict:
                continue
            winner, loser = sorted(
                (left, right), key=validation_score, reverse=True
            )
            conflicts.append({
                "left_plan_id": left.get("id"),
                "right_plan_id": right.get("id"),
                "trigger_jaccard": round(trigger_jaccard, 6),
                "answer_type_conflict": answer_conflict,
                "relation_path_conflict": relation_conflict,
                "winner_plan_id": winner.get("id"),
                "loser_plan_id": loser.get("id"),
                "winner_score": round(validation_score(winner), 6),
                "loser_score": round(validation_score(loser), 6),
                "score_gap": round(
                    validation_score(winner) - validation_score(loser), 6
                ),
            })
    return conflicts


def status_for_plan(
    plan: Mapping[str, Any],
    min_precision: float,
    min_true_positives: int,
    retain_structural_rare: bool = False,
) -> Tuple[str, str]:
    validation = dict(plan.get("competency_validation", {}) or {})
    calibration = dict(validation.get("calibration", {}) or {})
    if (
        int(calibration.get("true_positive", 0) or 0) >= min_true_positives
        and float(calibration.get("posterior_precision", 0.0) or 0.0) >= min_precision
    ):
        return "validated", "calibration_pass"
    if (
        int(calibration.get("true_positive", 0) or 0) >= 1
        and int(calibration.get("false_positive", 0) or 0) == 0
    ):
        return "validated_sparse", "calibration_sparse_zero_false_positive"
    if int(calibration.get("opportunities", 0) or 0) >= min_true_positives:
        return "routing_candidate", "calibration_structural_support_global_router_gates"
    if (
        retain_structural_rare
        and int(plan.get("support", 0) or 0) >= 1
        and len(plan.get("path", []) or []) >= 3
    ):
        return "structural_rare", "induction_grounded_long_chain_no_calibration_opportunity"
    return "rejected", "competency_precision_or_support_failed"


def write_outputs(
    output_dir: Path,
    ontology: Mapping[str, Any],
    report_rows: Sequence[Mapping[str, Any]],
    diagnostics: Mapping[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ontology_path = output_dir / "query_plan_ontology.json"
    temporary = ontology_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(ontology, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(ontology_path)
    (output_dir / "competency_diagnostics.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    columns = list(report_rows[0]) if report_rows else ["plan_id", "status"]
    with (output_dir / "competency_plan_report.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(report_rows)
    lines = [
        "# Workload Competency Validation",
        "",
        f"- input plans: {diagnostics['input_plan_count']}",
        f"- deployed plans: {diagnostics['deployed_plan_count']}",
        f"- validated plans: {diagnostics['status_counts'].get('validated', 0)}",
        f"- provisional plans: {diagnostics['status_counts'].get('provisional', 0)}",
        f"- CQ coverage: {diagnostics['cq_summary']['coverage']:.6f}",
        f"- CQ routing precision: {diagnostics['cq_summary']['routing_precision']:.6f}",
        f"- global conflicts: {len(diagnostics['global_conflicts'])}",
    ]
    (output_dir / "competency_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--calibration-path", required=True)
    parser.add_argument("--cq-path", required=True)
    parser.add_argument("--workload-contract", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--min-route-score", type=float, default=0.20)
    parser.add_argument("--min-cq-precision", type=float, default=0.55)
    parser.add_argument("--runtime-min-confidence", type=float, default=0.72)
    parser.add_argument(
        "--routing-target-precision",
        type=float,
        default=None,
        help="Calibration precision constraint; defaults to --min-cq-precision for compatibility.",
    )
    parser.add_argument("--min-true-positives", type=int, default=2)
    parser.add_argument("--min-deployed-plans", type=int, default=0)
    parser.add_argument("--conflict-reject-margin", type=float, default=0.10)
    parser.add_argument(
        "--routing-grid-points",
        type=int,
        default=64,
        help="Maximum score thresholds considered during calibration.",
    )
    parser.add_argument(
        "--adaptive-routing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Learn uncertainty-conditioned leaf budgets from calibration queries.",
    )
    parser.add_argument(
        "--routing-margin-grid-points",
        type=int,
        default=6,
        help="Maximum top-1/top-2 margin cutoffs considered during calibration.",
    )
    parser.add_argument(
        "--retain-structural-rare-plans",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Admit induction-grounded 3+ hop schemas lacking calibration opportunities.",
    )
    parser.add_argument(
        "--deployment-policy",
        choices=("validated", "all_sanitized"),
        default="validated",
        help="all_sanitized is reserved for the strict w/o competency-gate ablation.",
    )
    args = parser.parse_args()

    input_path = Path(args.input_path).expanduser().resolve()
    calibration_path = Path(args.calibration_path).expanduser().resolve()
    cq_path = Path(args.cq_path).expanduser().resolve()
    contract_path = Path(args.workload_contract).expanduser().resolve()
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("type") != "workload_competency_split_contract":
        raise SystemExit("invalid workload competency contract")
    if contract.get("split") != "train_only" or contract.get("contract_ok") is not True:
        raise SystemExit("workload competency contract is not a valid train-only split")
    if contract.get("inference_uses_gold") is not False:
        raise SystemExit("workload contract violates gold-free inference")

    ontology = json.loads(input_path.read_text(encoding="utf-8"))
    input_plans = [item for item in ontology.get("plans", []) or [] if isinstance(item, Mapping)]
    sanitized_plans = []
    sanitation_totals = Counter()
    for plan in input_plans:
        sanitized, stats = sanitize_plan(plan)
        sanitized_plans.append(sanitized)
        sanitation_totals.update(stats)
    calibration_rows = load_json_or_jsonl(calibration_path)
    cq_rows = load_json_or_jsonl(cq_path)
    routing_target_precision = (
        float(args.routing_target_precision)
        if args.routing_target_precision is not None
        else float(args.min_cq_precision)
    )
    routing_model = build_hierarchical_routing_model(
        sanitized_plans,
        dataset=args.dataset,
    )
    calibrator = (
        calibrate_margin_adaptive_routing_policy
        if args.adaptive_routing
        else calibrate_routing_policy
    )
    calibration_kwargs = {
        "max_top_m": args.top_k,
        "target_precision": routing_target_precision,
        "min_plan_confidence": args.runtime_min_confidence,
        "max_thresholds": args.routing_grid_points,
    }
    if args.adaptive_routing:
        calibration_kwargs["max_margin_cutoffs"] = args.routing_margin_grid_points
    routing_policy, routing_calibration = calibrator(
        sanitized_plans,
        calibration_rows,
        routing_model,
        **calibration_kwargs,
    )
    routing_model["policy"] = routing_policy
    calibration_summary = evaluate(
        sanitized_plans,
        calibration_rows,
        routing_policy["leaf_top_m"],
        routing_policy["min_score"],
        routing_model,
        args.runtime_min_confidence,
    )
    cq_summary = evaluate(
        sanitized_plans,
        cq_rows,
        routing_policy["leaf_top_m"],
        routing_policy["min_score"],
        routing_model,
        args.runtime_min_confidence,
    )

    for plan in sanitized_plans:
        plan_id = str(plan.get("id"))
        plan["competency_validation"] = {
            "calibration": calibration_summary["plans"].get(plan_id, {}),
            "cq_holdout": cq_summary["plans"].get(plan_id, {}),
            "inference_uses_gold": False,
        }
        status, reason = status_for_plan(
            plan,
            routing_target_precision,
            args.min_true_positives,
            retain_structural_rare=args.retain_structural_rare_plans,
        )
        if float(plan.get("confidence", 0.0) or 0.0) < args.runtime_min_confidence:
            status, reason = "rejected", "below_runtime_confidence_constraint"
        plan["competency_status"] = status
        plan["competency_decision_reason"] = reason
        plan["competency_score"] = round(validation_score(plan), 6)

    conflicts = detect_conflicts(sanitized_plans)
    conflict_losers = {
        str(item["loser_plan_id"])
        for item in conflicts
        if float(item.get("score_gap", 0.0) or 0.0) >= args.conflict_reject_margin
    }
    for plan in sanitized_plans:
        plan_id = str(plan.get("id"))
        if args.deployment_policy == "all_sanitized":
            plan["competency_status"] = "sanitized_ablation"
            plan["competency_decision_reason"] = "competency_gate_disabled"
        elif plan_id in conflict_losers:
            plan["competency_status"] = "rejected"
            plan["competency_decision_reason"] = "global_schema_conflict_loser"

    deployed = [plan for plan in sanitized_plans if plan.get("competency_status") != "rejected"]
    minimum = min(max(args.min_deployed_plans, 0), len(sanitized_plans))
    if len(deployed) < minimum:
        rejected = sorted(
            (
                plan
                for plan in sanitized_plans
                if plan not in deployed
                and str(plan.get("id")) not in conflict_losers
                and float(plan.get("confidence", 0.0) or 0.0) >= args.runtime_min_confidence
            ),
            key=validation_score,
            reverse=True,
        )
        for plan in rejected[:minimum - len(deployed)]:
            plan["competency_status"] = "provisional"
            plan["competency_decision_reason"] = "coverage_floor_provisional"
            deployed.append(plan)
    deployed.sort(
        key=lambda item: (validation_score(item), float(item.get("support", 0) or 0)),
        reverse=True,
    )
    rejected_plans = [plan for plan in sanitized_plans if plan not in deployed]
    status_counts = Counter(str(plan.get("competency_status")) for plan in sanitized_plans)
    admission_calibration_summary = calibration_summary
    admission_cq_summary = cq_summary
    admission_routing_policy = routing_policy
    routing_model = build_hierarchical_routing_model(deployed, dataset=args.dataset)
    routing_policy, routing_calibration = calibrator(
        deployed,
        calibration_rows,
        routing_model,
        **calibration_kwargs,
    )
    routing_model["policy"] = routing_policy
    calibration_summary = evaluate(
        deployed,
        calibration_rows,
        routing_policy["leaf_top_m"],
        routing_policy["min_score"],
        routing_model,
        args.runtime_min_confidence,
    )
    cq_summary = evaluate(
        deployed,
        cq_rows,
        routing_policy["leaf_top_m"],
        routing_policy["min_score"],
        routing_model,
        args.runtime_min_confidence,
    )
    diagnostics = {
        "type": "workload_competency_validation",
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "input_path": str(input_path),
        "input_sha256": sha256_file(input_path),
        "calibration_path": str(calibration_path),
        "calibration_sha256": sha256_file(calibration_path),
        "cq_path": str(cq_path),
        "cq_sha256": sha256_file(cq_path),
        "workload_contract": str(contract_path),
        "workload_contract_sha256": sha256_file(contract_path),
        "input_plan_count": len(input_plans),
        "deployed_plan_count": len(deployed),
        "rejected_plan_count": len(rejected_plans),
        "status_counts": dict(status_counts),
        "sanitation": dict(sanitation_totals),
        "calibration_summary": {key: value for key, value in calibration_summary.items() if key != "plans"},
        "cq_summary": {key: value for key, value in cq_summary.items() if key != "plans"},
        "admission_calibration_summary": {
            key: value for key, value in admission_calibration_summary.items() if key != "plans"
        },
        "admission_cq_diagnostic": {
            key: value for key, value in admission_cq_summary.items() if key != "plans"
        },
        "admission_routing_policy": admission_routing_policy,
        "routing_policy": routing_policy,
        "routing_calibration": routing_calibration,
        "global_conflicts": conflicts,
        "repaired_global_conflict_count": len(conflict_losers),
        "deployment_policy": args.deployment_policy,
        "runtime_min_confidence": args.runtime_min_confidence,
        "adaptive_routing": bool(args.adaptive_routing),
        "retain_structural_rare_plans": bool(args.retain_structural_rare_plans),
        "dataset_name_routing_feature": False,
        "deployment_decision_split": "calibration",
        "cq_holdout_role": "report_only",
        "training_uses_gold": True,
        "inference_uses_gold": False,
    }
    retained_parents = {
        str(value.get("parent_id", ""))
        for value in routing_model["plan_vectors"].values()
    }
    for plan in sanitized_plans:
        plan.pop("routing_training", None)
    runtime_deployed = deepcopy(deployed)
    runtime_rejected = deepcopy(rejected_plans)
    for plan in [*runtime_deployed, *runtime_rejected]:
        validation = dict(plan.get("competency_validation", {}) or {})
        validation.pop("cq_holdout", None)
        validation["deployment_decision_split"] = "calibration"
        plan["competency_validation"] = validation
    runtime_diagnostics = deepcopy(diagnostics)
    for key in (
        "cq_path",
        "cq_sha256",
        "cq_summary",
        "admission_cq_diagnostic",
    ):
        runtime_diagnostics.pop(key, None)
    runtime_diagnostics["cq_holdout_role"] = "external_report_not_embedded"
    output_ontology = deepcopy(ontology)
    output_ontology.update({
        "version": max(int(ontology.get("version", 1) or 1), 3),
        "kind": "workload_competency_validated_executable_evidence_ontology",
        "builder": "scripts/validate_ontology.py",
        "plans": runtime_deployed,
        "rejected_plans": runtime_rejected,
        "routing_model": routing_model,
        "schema_hierarchy": {
            "root": {
                "id": "workload_root",
                "children": sorted(retained_parents),
            },
            "parents": [
                {
                    "id": parent_id,
                    "children": sorted(
                        plan_id
                        for plan_id, value in routing_model["plan_vectors"].items()
                        if value.get("parent_id") == parent_id
                    ),
                }
                for parent_id in sorted(retained_parents)
            ],
        },
        "workload_competency_validation": runtime_diagnostics,
    })
    report_rows = []
    for plan in sanitized_plans:
        validation = plan.get("competency_validation", {})
        calibration = validation.get("calibration", {})
        cq = validation.get("cq_holdout", {})
        report_rows.append({
            "plan_id": plan.get("id"),
            "status": plan.get("competency_status"),
            "decision_reason": plan.get("competency_decision_reason"),
            "score": plan.get("competency_score"),
            "support": plan.get("support", 0),
            "calibration_tp": calibration.get("true_positive", 0),
            "calibration_fp": calibration.get("false_positive", 0),
            "calibration_precision": calibration.get("precision", 0.0),
            "cq_tp": cq.get("true_positive", 0),
            "cq_fp": cq.get("false_positive", 0),
            "cq_precision": cq.get("precision", 0.0),
            "cq_recall": cq.get("recall", 0.0),
        })
    output_dir = Path(args.output_root).expanduser().resolve() / args.dataset
    write_outputs(output_dir, output_ontology, report_rows, diagnostics)
    print(
        f"dataset={args.dataset} input={len(input_plans)} deployed={len(deployed)} "
        f"validated={status_counts.get('validated', 0)} cq_coverage={diagnostics['cq_summary']['coverage']:.6f}"
    )


if __name__ == "__main__":
    main()
