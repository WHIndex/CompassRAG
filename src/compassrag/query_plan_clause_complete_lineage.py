"""Query-characteristic contracts for clause-complete evidence planning.

The helpers in this module are deliberately gold-free and independent of
dataset identity.  They estimate how many relational clauses an incoming
query asks the system to satisfy, compare that requirement with a compiled
logical or physical path, and expose stable lineage tokens for evidence-set
replacement checks.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set


_CLAUSE_CUE = re.compile(
    r"\b(?:who|whom|whose|which|that|where|when|after|before|while|despite)\b",
    re.IGNORECASE,
)
_TOKEN = re.compile(r"[a-z0-9]+")


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


def _norm(value: Any) -> str:
    return " ".join(_TOKEN.findall(str(value or "").lower()))


def _semantic_unique(values: Iterable[Any]) -> List[str]:
    result: List[str] = []
    seen: Set[str] = set()
    for value in values:
        text = str(value or "").strip()
        key = _norm(text)
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result


def _logical_contract(
    query_plan_log: Mapping[str, Any],
    factorized_log: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    if isinstance(factorized_log, Mapping):
        value = factorized_log.get("logical_evidence_plan", {}) or {}
        if isinstance(value, Mapping):
            return value
    factorized = query_plan_log.get("factorized_query_executor", {}) or {}
    value = factorized.get("logical_evidence_plan", {}) or {}
    return value if isinstance(value, Mapping) else {}


def _answer_type(
    query_plan_log: Mapping[str, Any],
    factorized_log: Mapping[str, Any] | None,
) -> str:
    factorized = factorized_log if isinstance(factorized_log, Mapping) else (
        query_plan_log.get("factorized_query_executor", {}) or {}
    )
    typed = factorized.get("typed_contract", {}) or {}
    condition = query_plan_log.get("query_condition", {}) or {}
    return str(
        factorized.get("answer_type")
        or typed.get("answer_type")
        or condition.get("answer_type")
        or "entity"
    ).strip().lower()


def _compatible_schema_hops(
    query_plan_log: Mapping[str, Any],
    answer_type: str,
) -> int:
    condition = query_plan_log.get("query_condition", {}) or {}
    best = 0
    for plan in condition.get("compiler_schema_plans", []) or []:
        if not isinstance(plan, Mapping):
            continue
        plan_answer_type = str(plan.get("answer_type", "") or "").strip().lower()
        if (
            plan_answer_type
            and answer_type not in {"", "entity"}
            and plan_answer_type not in {"entity", answer_type}
        ):
            continue
        link_score = _safe_float(
            plan.get("schema_link_score", plan.get("effective_precision", 0.0))
        )
        relation_coverage = _safe_float(plan.get("schema_relation_coverage", 1.0), 1.0)
        if link_score < 0.35 or relation_coverage < 0.35:
            continue
        integrity = plan.get("integrity_constraints", {}) or {}
        path_hops = len(plan.get("path", []) or [])
        best = max(best, path_hops, _safe_int(integrity.get("min_hops"), 0))
    return best


def query_clause_profile(
    query: str,
    query_plan_log: Mapping[str, Any],
    factorized_log: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Estimate relational demand without labels or dataset-specific rules."""
    logical = _logical_contract(query_plan_log, factorized_log)
    condition = query_plan_log.get("query_condition", {}) or {}
    logical_relations = _semantic_unique(logical.get("relations", []) or [])
    condition_signatures = _semantic_unique(
        condition.get("relation_signatures", []) or []
    )
    condition_terms = _semantic_unique(condition.get("relation_terms", []) or [])
    query_text = str(query or "")
    clause_matches = list(_CLAUSE_CUE.finditer(query_text))
    embedded_clause_count = sum(
        len(_TOKEN.findall(query_text[:match.start()])) > 1
        for match in clause_matches
    )
    surface_lower_bound = 1 + embedded_clause_count
    logical_hops = max(
        len(logical_relations),
        _safe_int(logical.get("required_hop_count"), 0),
    )
    branch_count = max(_safe_int(logical.get("required_branch_count"), 0), 1)
    schema_hops = _compatible_schema_hops(
        query_plan_log,
        _answer_type(query_plan_log, factorized_log),
    )
    relation_signal_count = (
        len(condition_signatures)
        if condition_signatures
        else len(condition_terms)
    )
    required_clause_count = min(
        max(
            1,
            surface_lower_bound,
            logical_hops,
            schema_hops,
            relation_signal_count,
            branch_count,
        ),
        6,
    )
    logical_relation_count = len(logical_relations)
    logical_coverage = min(
        logical_relation_count / max(required_clause_count, 1), 1.0
    )
    logical_admissible = bool(logical.get("admissible", True))
    route_reason = str(
        (
            factorized_log
            if isinstance(factorized_log, Mapping)
            else query_plan_log.get("factorized_query_executor", {}) or {}
        ).get("route_reason", "")
        or ""
    )
    explicit_missing_relation = route_reason in {
        "insufficient_query_predicate_coverage",
        "no_relational_access_path",
        "logical_plan_missing_relation",
    }
    logical_complete = bool(
        logical_admissible
        and logical_relation_count >= required_clause_count
        and not explicit_missing_relation
    )
    return {
        "schema_version": "query_clause_profile_v1",
        "gold_free": True,
        "dataset_identity_used": False,
        "required_clause_count": required_clause_count,
        "surface_clause_lower_bound": surface_lower_bound,
        "embedded_clause_count": embedded_clause_count,
        "logical_relation_count": logical_relation_count,
        "logical_required_hops": logical_hops,
        "compatible_schema_hops": schema_hops,
        "condition_relation_count": relation_signal_count,
        "required_branch_count": branch_count,
        "logical_clause_coverage": logical_coverage,
        "logical_clause_complete": logical_complete,
        "long_chain": required_clause_count >= 3,
        "set_level_evidence_required": bool(
            required_clause_count >= 3 or branch_count > 1
        ),
        "explicit_missing_relation": explicit_missing_relation,
    }


def path_clause_profile(
    relations: Sequence[Any],
    query_profile: Mapping[str, Any],
) -> Dict[str, Any]:
    relation_count = len(_semantic_unique(relations))
    required = max(_safe_int(query_profile.get("required_clause_count"), 1), 1)
    coverage = min(relation_count / required, 1.0)
    return {
        "path_relation_count": relation_count,
        "query_required_clause_count": required,
        "query_clause_coverage": coverage,
        "query_clause_complete": relation_count >= required,
    }


def row_lineage_tokens(row: Mapping[str, Any]) -> Set[str]:
    """Return semantic obligations witnessed by one passage catalog row."""
    tokens: Set[str] = set()
    for obligation in row.get("satisfied_obligations", []) or []:
        normalized = _norm(obligation).replace(" ", "_")
        if normalized:
            tokens.add(f"obligation:{normalized}")
    for index, hit in enumerate(row.get("aperture_relation_slot_hits", []) or []):
        if bool(hit):
            tokens.add(f"relation_slot:{index}")
    for branch in row.get("branch_ids", []) or []:
        normalized = _norm(branch)
        if normalized:
            tokens.add(f"branch:{normalized}")
    if bool(row.get("aperture_terminal_relation")):
        tokens.add("terminal_relation")
    if bool(row.get("answer_type_compatible")):
        tokens.add("answer_type")
    return tokens


def set_lineage_tokens(
    passage_ids: Sequence[int],
    catalog: Mapping[int, Mapping[str, Any]],
) -> Set[str]:
    return set().union(*(
        row_lineage_tokens(catalog.get(int(passage_id), {}) or {})
        for passage_id in passage_ids
    )) if passage_ids else set()


def lineage_replacement_profile(
    before_ids: Sequence[int],
    after_ids: Sequence[int],
    catalog: Mapping[int, Mapping[str, Any]],
) -> Dict[str, Any]:
    before = set_lineage_tokens(before_ids, catalog)
    after = set_lineage_tokens(after_ids, catalog)
    lost = before - after
    gained = after - before
    return {
        "before_tokens": sorted(before),
        "after_tokens": sorted(after),
        "lost_tokens": sorted(lost),
        "gained_tokens": sorted(gained),
        "before_coverage_count": len(before),
        "after_coverage_count": len(after),
        "coverage_gain": len(gained),
        "unique_obligation_loss": len(lost),
        "witness_preserved": not lost,
    }
