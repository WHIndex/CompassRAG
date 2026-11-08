import json
import re
import threading
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .sentence_evidence_utils import infer_answer_target_type


TOKEN_RE = re.compile(r"[a-z0-9]+")
DATE_SPAN_RE = re.compile(
    r"\b(?:\d{1,2}\s+)?(?:january|february|march|april|may|june|july|august|"
    r"september|october|november|december)\s+\d{1,2},?\s+\d{3,4}\b|"
    r"\b\d{1,2}\s+(?:january|february|march|april|may|june|july|august|"
    r"september|october|november|december)\s+\d{3,4}\b|"
    r"\b\d{3,4}\b",
    re.IGNORECASE,
)
NUMBER_VALUE_RE = re.compile(r"\b\d+(?:,\d{3})*(?:\.\d+)?%?\b")
DECADE_VALUE_RE = re.compile(r"\b(?:1[0-9]{3}|20[0-9]{2}|21[0-9]{2})s\b", re.IGNORECASE)
CONTENT_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "was", "were", "are", "from",
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
    "did", "does", "has", "have", "had", "his", "her", "its", "their", "they",
    "them", "into", "about", "after", "before", "between", "during", "also",
    "than", "then", "there", "these", "those", "can", "could", "would",
    "should", "not", "all", "one", "two", "first", "last", "name", "called",
    "known",
}
TITLE_CASE_RE = re.compile(r"\b[A-Z][A-Za-z0-9'’.-]+(?:\s+(?:of|the|and|de|la|van|von|[A-Z][A-Za-z0-9'’.-]+)){0,6}\b")
QUOTED_SPAN_RE = re.compile(r"[\"'“”‘’]([^\"'“”‘’]{2,100})[\"'“”‘’]")
ENTITY_ANCHOR_STOPWORDS = CONTENT_STOPWORDS | {
    "what", "which", "where", "when", "who", "whom", "whose", "name",
    "city", "country", "state", "county", "place", "date", "year",
    "born", "birth", "birthplace", "died", "death", "director", "writer",
    "author", "composer", "producer", "creator", "founder", "designer",
    "father", "mother", "grandfather", "grandmother", "paternal", "maternal",
    "spouse", "wife", "husband", "son", "daughter", "children", "parent",
}
FAMILY_OR_ROLE_CHAIN_RE = re.compile(
    r"\b(?:paternal|maternal|grandfather|grandmother|father|mother|parent|"
    r"spouse|wife|husband|child|children|son|daughter|uncle|aunt|sibling|"
    r"director|writer|author|composer|producer|creator|founder|designer|"
    r"employer|alma mater|birthplace|place of birth|place of death|cause of death)\s+of\b|"
    r"(?:'s|’s)\s+(?:paternal|maternal|grandfather|grandmother|father|mother|parent|"
    r"spouse|wife|husband|child|children|son|daughter|uncle|aunt|sibling|"
    r"director|writer|author|composer|producer|creator|founder|designer|"
    r"employer|birthplace|cause of death)\b|"
    r"\bof\s+(?:the\s+)?(?:director|writer|author|composer|producer|creator|founder|designer|"
    r"father|mother|spouse|wife|husband|performer)\b",
    re.IGNORECASE,
)
RELATIVE_CLAUSE_RE = re.compile(
    r"\b(?:that|which|who|where)\s+(?:was|were|is|are|did|does|has|have|had|"
    r"located|based|founded|created|released|published|completed|built|born|died|directed|written|composed)\b",
    re.IGNORECASE,
)
COMPARISON_RE = re.compile(
    r"\b(?:both|same|compare|which of|are .+ and .+ from the same|"
    r"what do .+ and .+ have in common)\b",
    re.IGNORECASE,
)
SIMPLE_ATTRIBUTE_RE = re.compile(
    r"^(?:what|who|where|when|in what city|in which city)\s+(?:is|was|did)\s+"
    r".{1,120}?\s+(?:occupation|profession|nationality|born|die|died|birthplace|"
    r"date of birth|date of death|place of birth|place of death|located|headquartered)\??$",
    re.IGNORECASE,
)
LONG_ANSWER_RE = re.compile(
    r"^(?:why|how did|how does|how do|what happened|what happens|what causes|"
    r"what caused|describe|explain)\b",
    re.IGNORECASE,
)
RELATION_CUE_PATTERNS = {
    "birth_place": [
        re.compile(r"\b(?:was\s+)?born\s+(?:in|at|near)\b", re.IGNORECASE),
        re.compile(r"\b(?:birthplace|place\s+of\s+birth)\b", re.IGNORECASE),
    ],
    "birth_date": [
        re.compile(r"\b(?:was\s+)?born\s+(?:on|in)\b", re.IGNORECASE),
        re.compile(r"\b(?:date\s+of\s+birth|birth\s+date)\b", re.IGNORECASE),
    ],
    "death_place": [
        re.compile(r"\b(?:died|death)\s+(?:in|at|near)\b", re.IGNORECASE),
        re.compile(r"\b(?:place\s+of\s+death|deathplace)\b", re.IGNORECASE),
    ],
    "death_date": [
        re.compile(r"\b(?:died|death)\s+(?:on|in)\b", re.IGNORECASE),
        re.compile(r"\b(?:date\s+of\s+death|death\s+date)\b", re.IGNORECASE),
    ],
    "family_person": [
        re.compile(
            r"\b(?:father|mother|grandfather|grandmother|paternal|maternal|"
            r"spouse|wife|husband|son|daughter|children|parent)\b",
            re.IGNORECASE,
        ),
    ],
    "creator_person": [
        re.compile(
            r"\b(?:director|author|writer|composer|producer|creator|founder|designer)\b",
            re.IGNORECASE,
        ),
    ],
    "location": [
        re.compile(r"\b(?:located|headquartered|capital|county|city|country|state|region)\b", re.IGNORECASE),
    ],
    "release_date": [
        re.compile(r"\b(?:released|published|premiered|aired|broadcast|issued)\s+(?:on|in)\b", re.IGNORECASE),
        re.compile(r"\b(?:release|publication|premiere|air)\s+(?:date|year)\b", re.IGNORECASE),
    ],
    "founding_date": [
        re.compile(r"\b(?:founded|established|formed|created|opened)\s+(?:on|in)\b", re.IGNORECASE),
        re.compile(r"\b(?:founding|establishment|formation)\s+(?:date|year)\b", re.IGNORECASE),
    ],
}


def unpack_llm_result(result):
    response = result[0] if isinstance(result, tuple) else result
    metadata = result[1] if isinstance(result, tuple) and len(result) > 1 else {}
    if isinstance(result, tuple) and len(result) > 2:
        metadata = dict(metadata or {})
        metadata["cache_hit"] = result[2]
    return response, metadata


def _doc_title(doc: str) -> str:
    if not doc:
        return ""
    return str(doc).split("\n", 1)[0].strip()


def _doc_snippet(doc: str, max_chars: int = 360) -> str:
    if not doc:
        return ""
    parts = str(doc).split("\n", 1)
    body = parts[1] if len(parts) > 1 else parts[0]
    body = re.sub(r"\s+", " ", body).strip()
    return body[:max_chars]


def _normalize_query(text: Any) -> str:
    return " ".join(TOKEN_RE.findall(str(text or "").lower()))


def build_planned_retrieval_evidence_signature(
    doc_ids: List[int],
    memory_docs: Optional[List[str]] = None,
) -> Tuple[Tuple[int, ...], Tuple[str, ...]]:
    """Return a stable signature for the evidence visible to the next round."""

    ordered_doc_ids = []
    seen_doc_ids = set()
    for raw_doc_id in doc_ids or []:
        try:
            doc_id = int(raw_doc_id)
        except (TypeError, ValueError):
            continue
        if doc_id in seen_doc_ids:
            continue
        seen_doc_ids.add(doc_id)
        ordered_doc_ids.append(doc_id)

    ordered_memory_docs = []
    seen_memory_docs = set()
    for memory_doc in memory_docs or []:
        normalized = _normalize_query(str(memory_doc or "")[:1200])
        if not normalized or normalized in seen_memory_docs:
            continue
        seen_memory_docs.add(normalized)
        ordered_memory_docs.append(normalized)
    return tuple(ordered_doc_ids), tuple(ordered_memory_docs)


def build_planned_retrieval_decision_frontier(
    merge_info: Optional[Dict[str, Any]],
    *,
    materialized_doc_ids: Optional[List[int]] = None,
    late_residual_doc_ids: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Summarize candidates that can change the reader-visible evidence pool.

    Raw retriever top-k sets are intentionally excluded. Candidate churn that was
    rejected by admission guards is not query progress. The frontier contains only
    candidates selected by a physical evidence operator and actually materialized,
    plus memory tuples selected for append-only materialization.
    """

    merge_info = merge_info if isinstance(merge_info, dict) else {}
    materialized_ids = set()
    for raw_doc_id in materialized_doc_ids or []:
        try:
            materialized_ids.add(int(raw_doc_id))
        except (TypeError, ValueError):
            continue

    signature_items = set()
    action_counts: Counter = Counter()
    candidate_scores: List[float] = []

    def add_candidate(
        action: str,
        value: Any,
        *,
        require_materialized: bool = True,
        score: Any = None,
    ) -> None:
        if isinstance(value, dict):
            raw_doc_id = value.get("doc_id")
            if score is None:
                score = (
                    value.get("budgeted_evidence_subgraph_score")
                    if value.get("budgeted_evidence_subgraph_score") is not None
                    else value.get("calibrated_score", value.get("plan_score"))
                )
        else:
            raw_doc_id = value
        try:
            doc_id = int(raw_doc_id)
        except (TypeError, ValueError):
            return
        if require_materialized and materialized_ids and doc_id not in materialized_ids:
            return
        signature_items.add(f"{action}:{doc_id}")
        action_counts[action] += 1
        try:
            candidate_scores.append(float(score))
        except (TypeError, ValueError):
            pass

    for item in merge_info.get("promotions", []) or []:
        add_candidate("promote", item)
    for item in merge_info.get("expansions", []) or []:
        add_candidate("expand", item)
    for doc_id in late_residual_doc_ids or []:
        add_candidate("residual_append", doc_id)
    for doc_id in merge_info.get("promotion_utility_veto_soft_append_doc_ids", []) or []:
        add_candidate("utility_append", doc_id)

    object_memory = merge_info.get("object_memory_append_only") or {}
    for item in object_memory.get("top_candidates", []) or []:
        if item.get("accepted"):


            add_candidate("object_memory", item, require_materialized=False)

    subgraph = merge_info.get("budgeted_evidence_subgraph") or {}
    for doc_id in subgraph.get("selected_doc_ids", []) or []:
        add_candidate("subgraph", doc_id)

    top2 = merge_info.get("top2_override") or {}
    pairwise = top2.get("pairwise_utility_search") or {}
    selected_pairwise = pairwise.get("selected_doc_id")
    if selected_pairwise is not None:
        add_candidate("pairwise", selected_pairwise)
    for field, action in (
        ("bridge_slot_append_doc_ids", "bridge_slot_append"),
        ("schema_deficit_soft_append_doc_ids", "schema_deficit_append"),
    ):
        for doc_id in pairwise.get(field, []) or []:
            add_candidate(action, doc_id)

    signature = tuple(sorted(signature_items))
    return {
        "signature": signature,
        "signature_size": len(signature),
        "signature_preview": list(signature[:12]),
        "actionable_candidate_count": len(signature),
        "action_counts": dict(sorted(action_counts.items())),
        "best_candidate_utility": (
            round(max(candidate_scores), 6) if candidate_scores else 0.0
        ),
        "raw_candidate_count": int(merge_info.get("raw_candidate_count", 0) or 0),
        "guarded_rejected_count": int(
            merge_info.get("guarded_rejected_count", 0) or 0
        ),
        "ignored_rejected_candidate_count": int(
            merge_info.get("guarded_rejected_count", 0) or 0
        ),
    }


def planned_retrieval_deficit_signature(verdict: Dict[str, Any]) -> Tuple[str, ...]:
    """Canonicalize verifier deficits while ignoring query-surface paraphrases."""

    missing_slots = verdict.get("missing_slots") or []
    if not isinstance(missing_slots, list):
        missing_slots = [missing_slots]
    normalized_slots = sorted(
        {
            _normalize_query(item)
            for item in missing_slots
            if _normalize_query(item)
        }
    )
    if normalized_slots:
        return tuple(f"slot:{item}" for item in normalized_slots)

    values = []
    for key in ("missing", "uncertain_anchors"):
        raw_value = verdict.get(key)
        raw_items = raw_value if isinstance(raw_value, list) else [raw_value]
        for item in raw_items:
            normalized = _normalize_query(item)
            if normalized:
                values.append(f"{key}:{normalized}")
    return tuple(sorted(set(values)))


class PlannedRetrievalCostController:
    """Choose CONTINUE or STOP for the expensive planned-retrieval macro operator.

    The cost values are normalized execution units, not currency.  A stop is only
    admissible after the verifier deficit and reader-visible evidence have reached
    a repeated fixed point, which keeps the controller conservative under noisy
    LLM judgments.
    """

    def __init__(
        self,
        enabled: bool = False,
        min_rounds: int = 2,
        fixed_point_patience: int = 2,
        resolution_prior: float = 0.12,
        no_progress_decay: float = 0.50,
        planner_cost: float = 0.04,
        verifier_cost: float = 0.04,
        retriever_cost: float = 0.015,
        min_continue_net_benefit: float = 0.0,
        candidate_frontier_mode: str = "raw_candidates",
        shadow_only: bool = False,
    ):
        self.enabled = bool(enabled)
        self.min_rounds = max(1, int(min_rounds))
        self.fixed_point_patience = max(1, int(fixed_point_patience))
        self.resolution_prior = max(0.0, float(resolution_prior))
        self.no_progress_decay = _clamp(no_progress_decay)
        self.planner_cost = max(0.0, float(planner_cost))
        self.verifier_cost = max(0.0, float(verifier_cost))
        self.retriever_cost = max(0.0, float(retriever_cost))
        self.min_continue_net_benefit = float(min_continue_net_benefit)
        frontier_mode = str(candidate_frontier_mode or "raw_candidates").strip().lower()
        if frontier_mode not in {"raw_candidates", "decision_relevant"}:
            frontier_mode = "raw_candidates"
        self.candidate_frontier_mode = frontier_mode
        self.shadow_only = bool(shadow_only)
        self.previous_deficit_signature: Optional[Tuple[str, ...]] = None
        self.previous_candidate_signature: Optional[Tuple[Any, ...]] = None
        self.no_progress_streak = 0
        self.evaluated_count = 0
        self.no_progress_round_count = 0
        self.stop_count = 0
        self.would_stop_count = 0
        self.first_would_stop_round: Optional[int] = None
        self.estimated_avoided_rounds = 0
        self.estimated_avoided_api_calls = 0
        self.estimated_avoided_retriever_calls = 0
        self.estimated_shadow_avoided_rounds = 0
        self.estimated_shadow_avoided_api_calls = 0
        self.estimated_shadow_avoided_retriever_calls = 0
        self.estimated_observed_cost = 0.0

    def observe(
        self,
        *,
        round_number: int,
        max_rounds: int,
        verdict: Dict[str, Any],
        before_evidence_signature: Tuple[Tuple[int, ...], Tuple[str, ...]],
        after_evidence_signature: Tuple[Tuple[int, ...], Tuple[str, ...]],
        applied: bool,
        query_count: int,
        candidate_signature: Optional[Tuple[Any, ...]] = None,
        actionable_candidate_count: int = 0,
        best_candidate_utility: float = 0.0,
    ) -> Dict[str, Any]:
        round_number = max(1, int(round_number))
        max_rounds = max(round_number, int(max_rounds))
        query_count = max(0, int(query_count))
        deficit_signature = planned_retrieval_deficit_signature(verdict or {})
        deficit_changed = (
            self.previous_deficit_signature is not None
            and deficit_signature != self.previous_deficit_signature
        )
        candidate_signature = tuple(candidate_signature or ())
        if self.candidate_frontier_mode == "decision_relevant":



            retrieval_changed = bool(
                self.previous_candidate_signature is not None
                and candidate_signature
                and candidate_signature != self.previous_candidate_signature
            )
        else:
            retrieval_changed = (
                self.previous_candidate_signature is not None
                and candidate_signature != self.previous_candidate_signature
            )
        evidence_changed = before_evidence_signature != after_evidence_signature
        progress = bool(evidence_changed or deficit_changed or retrieval_changed)
        if progress:
            self.no_progress_streak = 0
        else:
            self.no_progress_streak += 1
            self.no_progress_round_count += 1

        expected_resolution = self.resolution_prior
        if self.no_progress_streak:
            expected_resolution *= self.no_progress_decay ** self.no_progress_streak
        estimated_next_round_cost = (
            self.planner_cost
            + self.verifier_cost
            + self.retriever_cost * max(1, query_count)
        )
        net_benefit = expected_resolution - estimated_next_round_cost
        fixed_point = self.no_progress_streak >= self.fixed_point_patience
        eligible = round_number >= self.min_rounds and round_number < max_rounds
        would_stop = bool(
            self.enabled
            and eligible
            and fixed_point
            and net_benefit <= self.min_continue_net_benefit
        )
        should_stop = bool(would_stop and not self.shadow_only)
        remaining_rounds = max(0, max_rounds - round_number)
        avoided_api_calls = 0
        avoided_retriever_calls = 0
        first_would_stop = bool(would_stop and self.first_would_stop_round is None)
        if first_would_stop:
            self.first_would_stop_round = round_number
            self.would_stop_count += 1
            self.estimated_shadow_avoided_rounds += remaining_rounds
            self.estimated_shadow_avoided_api_calls += remaining_rounds * 2
            self.estimated_shadow_avoided_retriever_calls += (
                remaining_rounds * max(1, query_count)
            )
        if should_stop:
            self.stop_count += 1
            self.estimated_avoided_rounds += remaining_rounds
            avoided_api_calls = remaining_rounds * 2
            avoided_retriever_calls = remaining_rounds * max(1, query_count)
            self.estimated_avoided_api_calls += avoided_api_calls
            self.estimated_avoided_retriever_calls += avoided_retriever_calls

        self.evaluated_count += 1
        self.estimated_observed_cost += estimated_next_round_cost
        self.previous_deficit_signature = deficit_signature
        self.previous_candidate_signature = candidate_signature
        return {
            "enabled": self.enabled,
            "shadow_only": self.shadow_only,
            "candidate_frontier_mode": self.candidate_frontier_mode,
            "operator": "PLANNED_RETRIEVAL_ROUND",
            "decision": "STOP" if should_stop else "CONTINUE",
            "policy_decision": "STOP" if would_stop else "CONTINUE",
            "would_stop": would_stop,
            "first_would_stop": first_would_stop,
            "reason": (
                "fixed_point_non_positive_net_benefit"
                if should_stop
                else "shadow_fixed_point_non_positive_net_benefit"
                if would_stop
                else "continue_until_safe_fixed_point"
            ),
            "round": round_number,
            "evidence_changed": evidence_changed,
            "deficit_changed": deficit_changed,
            "retrieval_changed": retrieval_changed,
            "operator_reported_applied": bool(applied),
            "progress": progress,
            "no_progress_streak": self.no_progress_streak,
            "fixed_point": fixed_point,
            "eligible": eligible,
            "deficit_signature": list(deficit_signature),
            "candidate_signature_size": len(candidate_signature),
            "candidate_signature_preview": list(candidate_signature[:12]),
            "actionable_candidate_count": max(0, int(actionable_candidate_count)),
            "best_candidate_utility": round(
                max(0.0, float(best_candidate_utility or 0.0)), 6
            ),
            "expected_resolution_utility": round(expected_resolution, 6),
            "estimated_next_round_cost": round(estimated_next_round_cost, 6),
            "net_benefit": round(net_benefit, 6),
            "min_continue_net_benefit": self.min_continue_net_benefit,
            "estimated_avoided_rounds": remaining_rounds if should_stop else 0,
            "estimated_avoided_api_calls": avoided_api_calls,
            "estimated_avoided_retriever_calls": avoided_retriever_calls,
        }

    def summary(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "shadow_only": self.shadow_only,
            "candidate_frontier_mode": self.candidate_frontier_mode,
            "operator": "PLANNED_RETRIEVAL_ROUND",
            "min_rounds": self.min_rounds,
            "fixed_point_patience": self.fixed_point_patience,
            "resolution_prior": self.resolution_prior,
            "no_progress_decay": self.no_progress_decay,
            "planner_cost": self.planner_cost,
            "verifier_cost": self.verifier_cost,
            "retriever_cost": self.retriever_cost,
            "min_continue_net_benefit": self.min_continue_net_benefit,
            "evaluated_count": self.evaluated_count,
            "no_progress_round_count": self.no_progress_round_count,
            "stop_count": self.stop_count,
            "would_stop_count": self.would_stop_count,
            "first_would_stop_round": self.first_would_stop_round,
            "estimated_avoided_rounds": self.estimated_avoided_rounds,
            "estimated_avoided_api_calls": self.estimated_avoided_api_calls,
            "estimated_avoided_retriever_calls": self.estimated_avoided_retriever_calls,
            "estimated_shadow_avoided_rounds": self.estimated_shadow_avoided_rounds,
            "estimated_shadow_avoided_api_calls": self.estimated_shadow_avoided_api_calls,
            "estimated_shadow_avoided_retriever_calls": (
                self.estimated_shadow_avoided_retriever_calls
            ),
            "estimated_observed_cost": round(self.estimated_observed_cost, 6),
        }


def _content_token_set(text: Any) -> set:
    return {
        token
        for token in TOKEN_RE.findall(str(text or "").lower())
        if len(token) > 2 and token not in CONTENT_STOPWORDS
    }


def _query_anchor_overlap(question: Any, planned_query: Any) -> float:
    question_tokens = _content_token_set(question)
    planned_tokens = _content_token_set(planned_query)
    if not question_tokens or not planned_tokens:
        return 0.0
    return len(question_tokens & planned_tokens) / max(1, min(len(question_tokens), len(planned_tokens)))


def _clamp(value: Any, low: float = 0.0, high: float = 1.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = low
    if not np.isfinite(numeric):
        numeric = low
    return min(high, max(low, numeric))


def _item_query_text(item: Optional[Dict[str, Any]]) -> str:
    if not item:
        return ""
    queries = item.get("queries") or []
    if not queries and item.get("query"):
        queries = [item.get("query")]
    return " ".join(str(query) for query in queries if query)


def _rank_prior(original_rank: Any) -> float:
    if original_rank is None:
        return 0.0
    try:
        rank = int(original_rank)
    except (TypeError, ValueError):
        return 0.0
    if rank < 0:
        return 0.0
    return 1.0 / float(rank + 1)


def _schema_verified_utility_prior(
    question: Any,
    candidate_doc: Any,
    candidate_item: Dict[str, Any],
    candidate_features: Dict[str, Any],
    intervention_gate: Dict[str, Any],
    enabled: bool,
    max_bonus: float,
    min_schema_score: float,
    min_schema_idf: float,
    min_overlap: float,
    max_original_rank: int,
    numeric_requires_value: bool,
    block_long_context: bool,
    rescue_enabled: bool,
    min_rescue_bonus: float,
) -> Dict[str, Any]:
    info = {
        "enabled": bool(enabled),
        "accepted": False,
        "rescue_allowed": False,
        "rescued": False,
        "bonus": 0.0,
        "reason": "disabled",
        "reject_reasons": [],
    }
    if not enabled:
        return info

    schema_item = candidate_item.get("schema_memory") or {}
    sources = set(candidate_item.get("candidate_sources", []) or [])
    source = str(candidate_item.get("candidate_source") or "")
    if source:
        sources.add(source)
    if "query_schema_memory" not in sources or not schema_item:
        info["reason"] = "no_query_schema_memory_signal"
        return info

    if block_long_context:
        question_type = str(intervention_gate.get("question_type", "") or "")
        long_context_state = (
            bool(intervention_gate.get("long_context_shape", False))
            or bool(intervention_gate.get("long_answer_signal", False))
            or question_type in {"long_answer_or_narrative", "long_context_evidence", "long_context_factoid"}
            or str(intervention_gate.get("reason", "")) in {
                "long_context_shape_freezes_top2",
                "long_answer_long_evidence_freezes_top2",
                "long_evidence_low_title_structure_freezes_top2",
                "very_long_evidence_freezes_top2",
            }
        )
        if long_context_state:
            info["reason"] = "long_context_or_narrative_blocked"
            info["reject_reasons"].append("long_context_or_narrative_blocked")
            return info

    try:
        original_rank = int(candidate_item.get("original_rank"))
    except (TypeError, ValueError):
        original_rank = 999999
    if original_rank < 2 or original_rank > max_original_rank:
        info["reason"] = "outside_schema_prior_rank_window"
        info["reject_reasons"].append("outside_schema_prior_rank_window")
        return info

    schema_score = float(schema_item.get("score", 0.0) or 0.0)
    schema_idf = float(schema_item.get("schema_idf", 0.0) or 0.0)
    schema_overlap = max(
        float(schema_item.get("query_overlap", 0.0) or 0.0),
        float(schema_item.get("context_overlap", 0.0) or 0.0),
    )
    if schema_score < min_schema_score:
        info["reason"] = "low_schema_score"
        info["reject_reasons"].append("low_schema_score")
        return info
    if schema_idf < min_schema_idf:
        info["reason"] = "low_schema_specificity"
        info["reject_reasons"].append("low_schema_specificity")
        return info
    if schema_overlap < min_overlap:
        info["reason"] = "weak_schema_query_overlap"
        info["reject_reasons"].append("weak_schema_query_overlap")
        return info

    target_type = infer_answer_target_type(str(question or ""))
    candidate_doc_text = str(candidate_doc or "")
    if numeric_requires_value:
        if target_type == "date" and not DATE_SPAN_RE.search(candidate_doc_text):
            info["reason"] = "date_question_without_date_value"
            info["reject_reasons"].append("date_question_without_date_value")
            return info
        if target_type == "number" and not NUMBER_VALUE_RE.search(candidate_doc_text):
            info["reason"] = "number_question_without_numeric_value"
            info["reject_reasons"].append("number_question_without_numeric_value")
            return info

    specificity = _clamp((schema_idf - min_schema_idf) / max(1.0 - min_schema_idf, 1e-6))
    overlap_signal = _clamp(schema_overlap)
    score_signal = _clamp(schema_score)
    support_signal = _clamp(float(candidate_features.get("query_support", 0.0) or 0.0))
    source_signal = 1.0 if {"planned_retrieval", "original_retrieval"} <= sources else 0.72
    rank_signal = _rank_prior(original_rank)
    raw_bonus = max_bonus * (
        0.30 * specificity
        + 0.24 * overlap_signal
        + 0.22 * score_signal
        + 0.14 * support_signal
        + 0.06 * source_signal
        + 0.04 * rank_signal
    )
    bonus = _clamp(raw_bonus, 0.0, max_bonus)
    info.update({
        "accepted": bonus > 0.0,
        "rescue_allowed": bool(rescue_enabled and bonus >= min_rescue_bonus),
        "bonus": round(float(bonus), 6),
        "reason": "schema_verified_prior_applied" if bonus > 0.0 else "zero_bonus",
        "target_type": target_type,
        "schema_score": round(schema_score, 6),
        "schema_idf": round(schema_idf, 6),
        "schema_overlap": round(schema_overlap, 6),
        "original_rank": original_rank,
        "source_signal": round(source_signal, 6),
        "query_support": round(support_signal, 6),
    })
    return info


PAIRWISE_UTILITY_SOURCE_SET = {
    "planned_and_original",
    "planned_retrieval",
    "original_retrieval",
    "query_schema_memory",
}

SCHEMA_VERIFIED_PRIOR_RESCUABLE_REJECTIONS = {
    "pair_verifier_low_margin",
    "pair_verifier_weak_candidate_query_support",
    "pairwise_utility_low_pair_score",
}


def _normalize_pairwise_utility_sources(
    value: Any,
    default: str = "planned_and_original",
) -> List[str]:
    source_text = str(value or default).strip().lower()
    if source_text in {"*", "all", "any"}:
        return sorted(PAIRWISE_UTILITY_SOURCE_SET)
    raw_sources = {
        item.strip()
        for item in source_text.replace(";", ",").split(",")
        if item.strip()
    }
    sources = {
        "query_schema_memory" if item == "schema_memory" else item
        for item in raw_sources
    }
    sources = sources & PAIRWISE_UTILITY_SOURCE_SET
    if not sources:
        return _normalize_pairwise_utility_sources(default, "planned_and_original")
    return sorted(sources)


def _query_schema_memory_intervention_gate(
    candidate_item: Dict[str, Any],
    intervention_gate: Dict[str, Any],
    policy: str = "risk_aware",
    min_top2_score: float = 0.66,
    min_top2_schema_idf: float = 0.32,
    min_top2_overlap: float = 0.10,
    max_top2_schema_frequency: int = 80,
    max_top2_original_rank: int = 24,
    min_top2_bridge_strength: int = 3,
) -> Dict[str, Any]:
    """Decide whether a query-schema-memory candidate may change rank 2.

    Query schema memory is useful for finding bridge-like passages, but pure
    schema neighbors can also be high-fanout distractors. This gate keeps
    planned-retrieval-supported candidates eligible while downgrading risky
    schema-only candidates to append-only evidence.
    """
    mode = str(policy or "risk_aware").strip().lower()
    if mode in {"false", "none", "disabled"}:
        mode = "off"
    if mode not in {"off", "risk_aware", "append_only"}:
        mode = "risk_aware"

    sources = {
        str(source)
        for source in (candidate_item.get("candidate_sources", []) or [])
        if source
    }
    candidate_source = str(candidate_item.get("candidate_source") or "")
    if candidate_source:
        sources.add(candidate_source)
    schema_item = candidate_item.get("schema_memory") or {}
    has_schema = "query_schema_memory" in sources or bool(schema_item)

    info = {
        "enabled": mode != "off",
        "policy": mode,
        "action": "allow_top2",
        "reason": "no_query_schema_signal",
        "reject_reasons": [],
        "sources": sorted(sources),
        "schema_score": 0.0,
        "schema_idf": 0.0,
        "schema_frequency": 0,
        "schema_overlap": 0.0,
        "original_rank": candidate_item.get("original_rank"),
        "bridge_strength": int(intervention_gate.get("bridge_strength", 0) or 0),
        "thresholds": {
            "min_top2_score": round(float(min_top2_score), 6),
            "min_top2_schema_idf": round(float(min_top2_schema_idf), 6),
            "min_top2_overlap": round(float(min_top2_overlap), 6),
            "max_top2_schema_frequency": int(max_top2_schema_frequency),
            "max_top2_original_rank": int(max_top2_original_rank),
            "min_top2_bridge_strength": int(min_top2_bridge_strength),
        },
    }
    if not has_schema or mode == "off":
        info["reason"] = "gate_disabled" if mode == "off" else "no_query_schema_signal"
        return info

    if "planned_retrieval" in sources or "planned_and_original" in sources:
        info["reason"] = "schema_candidate_has_planned_support"
        return info

    schema_score = float(schema_item.get("score", candidate_item.get("normalized_score", 0.0)) or 0.0)
    schema_idf = float(schema_item.get("schema_idf", 0.0) or 0.0)
    schema_frequency = int(schema_item.get("schema_frequency", 0) or 0)
    schema_overlap = max(
        float(schema_item.get("query_overlap", candidate_item.get("query_overlap", 0.0)) or 0.0),
        float(schema_item.get("context_overlap", 0.0) or 0.0),
        float(candidate_item.get("query_overlap", 0.0) or 0.0),
    )
    try:
        original_rank = int(candidate_item.get("original_rank"))
    except (TypeError, ValueError):
        original_rank = 999999

    bridge_strength = int(intervention_gate.get("bridge_strength", 0) or 0)
    question_type = str(intervention_gate.get("question_type", "unknown") or "unknown")
    long_context_state = (
        bool(intervention_gate.get("long_context_shape", False))
        or bool(intervention_gate.get("long_answer_signal", False))
        or str(intervention_gate.get("rank_intervention", "")) == "top5_only"
    )

    info.update({
        "schema_score": round(float(schema_score), 6),
        "schema_idf": round(float(schema_idf), 6),
        "schema_frequency": int(schema_frequency),
        "schema_overlap": round(float(schema_overlap), 6),
        "original_rank": original_rank,
        "bridge_strength": bridge_strength,
        "question_type": question_type,
        "long_context_state": bool(long_context_state),
    })

    if mode == "append_only":
        info.update({
            "action": "append_only",
            "reason": "policy_append_only",
            "reject_reasons": ["query_schema_policy_append_only"],
        })
        return info

    reject_reasons = []
    if question_type != "bridge_or_compositional":
        reject_reasons.append("query_schema_not_bridge_question")
    if bridge_strength < int(min_top2_bridge_strength):
        reject_reasons.append("query_schema_weak_bridge_state")
    if long_context_state:
        reject_reasons.append("query_schema_long_context_risk")
    if schema_score < float(min_top2_score):
        reject_reasons.append("query_schema_low_score")
    if schema_idf < float(min_top2_schema_idf):
        reject_reasons.append("query_schema_low_idf")
    if schema_overlap < float(min_top2_overlap):
        reject_reasons.append("query_schema_low_overlap")
    if max_top2_schema_frequency > 0 and schema_frequency > int(max_top2_schema_frequency):
        reject_reasons.append("query_schema_high_frequency")
    if original_rank < 2 or original_rank >= int(max_top2_original_rank):
        reject_reasons.append("query_schema_outside_rank_window")

    if reject_reasons:
        info.update({
            "action": "append_only",
            "reason": "schema_only_candidate_downgraded_to_append",
            "reject_reasons": reject_reasons,
        })
        return info

    info.update({
        "action": "allow_top2",
        "reason": "schema_only_candidate_passed_risk_gate",
    })
    return info


def _schema_deficit_promotion_gate(
    question: Any,
    top1_doc: Any,
    old_rank2_doc: Any,
    candidate_doc: Any,
    candidate_item: Dict[str, Any],
    pair_verifier_info: Optional[Dict[str, Any]],
    query_schema_gate_info: Optional[Dict[str, Any]],
    bridge_slot_info: Optional[Dict[str, Any]],
    intervention_gate: Optional[Dict[str, Any]],
    enabled: bool = False,
    min_score: float = 0.62,
    min_gain: float = 0.10,
    min_schema_score: float = 0.68,
    min_schema_idf: float = 0.30,
    max_original_rank: int = 60,
    require_bridge_constraint: bool = True,
    min_bridge_score: float = 0.54,
    min_bridge_margin: float = 0.08,
    min_bridge_missing_gain: float = 0.10,
    min_bridge_anchor_overlap: float = 0.24,
) -> Dict[str, Any]:
    """Promote schema-memory candidates only when they fill an evidence deficit.

    This is a second-stage safety valve for query-schema memory. The first gate
    conservatively downgrades schema-only candidates to append-only. Here, a
    schema-only candidate can recover top-2 eligibility when it has specific
    relation/schema evidence and the current rank-2 passage leaves a measurable
    missing slot.
    """
    source_set = {
        str(source)
        for source in (candidate_item.get("candidate_sources", []) or [])
        if source
    }
    candidate_source = str(candidate_item.get("candidate_source") or "")
    if candidate_source:
        source_set.add(candidate_source)
    schema_item = candidate_item.get("schema_memory") or {}
    gate_action = str((query_schema_gate_info or {}).get("action", "") or "")
    schema_frequency = int(
        schema_item.get(
            "schema_frequency",
            schema_item.get("frequency", 999999),
        ) or 999999
    )

    info: Dict[str, Any] = {
        "enabled": bool(enabled),
        "accepted": False,
        "reason": "disabled" if not enabled else "not_evaluated",
        "reject_reasons": [],
        "score": 0.0,
        "gain": 0.0,
        "sources": sorted(source_set),
        "gate_action": gate_action,
    }
    if not enabled:
        return info
    if "query_schema_memory" not in source_set and not schema_item:
        info.update({
            "reason": "no_query_schema_signal",
            "reject_reasons": ["schema_deficit_no_query_schema_signal"],
        })
        return info
    if "planned_retrieval" in source_set:
        info.update({
            "reason": "planned_supported_candidate_uses_existing_gate",
            "reject_reasons": ["schema_deficit_planned_supported"],
        })
        return info
    if gate_action not in {"append_only", "reject"}:
        info.update({
            "reason": "already_top2_eligible",
            "reject_reasons": ["schema_deficit_already_top2_eligible"],
        })
        return info

    try:
        original_rank = int(candidate_item.get("original_rank"))
    except (TypeError, ValueError):
        original_rank = 999999
    schema_score = float(schema_item.get("score", candidate_item.get("normalized_score", 0.0)) or 0.0)
    schema_idf = float(schema_item.get("schema_idf", 0.0) or 0.0)
    schema_overlap = max(
        float(schema_item.get("query_overlap", candidate_item.get("query_overlap", 0.0)) or 0.0),
        float(schema_item.get("context_overlap", 0.0) or 0.0),
        float(candidate_item.get("query_overlap", 0.0) or 0.0),
    )
    bridge_strength = int((intervention_gate or {}).get("bridge_strength", 0) or 0)
    question_type = str((intervention_gate or {}).get("question_type", "unknown") or "unknown")
    long_context_state = (
        bool((intervention_gate or {}).get("long_context_shape", False))
        or bool((intervention_gate or {}).get("long_answer_signal", False))
        or str((intervention_gate or {}).get("rank_intervention", "")) == "top5_only"
    )

    pair_base = (pair_verifier_info or {}).get("base") or {}
    pair_candidate = (pair_verifier_info or {}).get("candidate") or {}
    pair_margin = float((pair_verifier_info or {}).get("pair_margin", 0.0) or 0.0)
    candidate_query_support = float(pair_candidate.get("query_support", 0.0) or 0.0)
    candidate_complementarity = float(pair_candidate.get("complementarity", 0.0) or 0.0)
    candidate_pair_coverage = float(pair_candidate.get("pair_coverage", 0.0) or 0.0)
    base_query_support = float(pair_base.get("query_support", 0.0) or 0.0)
    base_complementarity = float(pair_base.get("complementarity", 0.0) or 0.0)
    base_pair_coverage = float(pair_base.get("pair_coverage", 0.0) or 0.0)

    bridge_slot_info = bridge_slot_info or {}
    bridge_score = float(bridge_slot_info.get("score", 0.0) or 0.0)
    bridge_margin = float(bridge_slot_info.get("margin", 0.0) or 0.0)
    deficit_gate = bridge_slot_info.get("deficit_gate") or {}
    old_pair_deficit = bool(deficit_gate.get("old_pair_deficit", False))
    missing_gain = float(deficit_gate.get("missing_support_gain", 0.0) or 0.0)
    bridge_candidate = bridge_slot_info.get("candidate") or {}
    old_bridge = bridge_slot_info.get("old_rank2") or {}
    candidate_relation_score = float(bridge_candidate.get("relation_score", 0.0) or 0.0)
    candidate_entity_score = float(bridge_candidate.get("entity_score", 0.0) or 0.0)
    candidate_answer_type_score = float(bridge_candidate.get("answer_type_score", 0.0) or 0.0)
    candidate_anchor_overlap = float(bridge_candidate.get("anchor_overlap", 0.0) or 0.0)
    candidate_missing_support = float(bridge_candidate.get("missing_slot_support", 0.0) or 0.0)
    old_missing_support = float(old_bridge.get("missing_slot_support", 0.0) or 0.0)

    lexical_deficit = (
        base_pair_coverage < 0.55
        or base_query_support < 0.45
        or base_complementarity < 0.08
    )
    structural_gain = max(0.0, candidate_query_support - base_query_support)
    coverage_gain = max(0.0, candidate_pair_coverage - base_pair_coverage)
    complementarity_gain = max(0.0, candidate_complementarity - base_complementarity)
    answer_relation_signal = max(
        candidate_relation_score,
        candidate_entity_score,
        candidate_answer_type_score,
        candidate_anchor_overlap,
    )
    schema_specificity = 0.5 * _clamp(schema_score) + 0.5 * _clamp(schema_idf)
    gain = (
        0.35 * max(0.0, bridge_margin)
        + 0.25 * max(0.0, missing_gain)
        + 0.20 * structural_gain
        + 0.12 * coverage_gain
        + 0.08 * complementarity_gain
    )
    score = (
        0.24 * _clamp(schema_score)
        + 0.16 * _clamp(schema_idf)
        + 0.16 * _clamp(bridge_score)
        + 0.12 * _clamp(candidate_query_support)
        + 0.10 * _clamp(candidate_complementarity)
        + 0.08 * _clamp(answer_relation_signal)
        + 0.08 * _clamp(max(pair_margin, 0.0))
        + 0.06 * _rank_prior(original_rank)
    )
    schema_supported_deficit = (
        lexical_deficit
        and schema_specificity >= 0.48
        and (
            missing_gain >= (0.5 * min_gain)
            or bridge_margin > 0.0
            or candidate_complementarity >= max(0.08, base_complementarity + 0.04)
        )
    )
    clear_deficit = old_pair_deficit or schema_supported_deficit
    bridge_slot_replace_passed = bool(bridge_slot_info.get("accepted", False))
    bridge_score_passed = bridge_score >= float(min_bridge_score)
    bridge_margin_passed = bridge_margin >= float(min_bridge_margin)
    bridge_missing_gain_passed = (
        missing_gain >= float(min_bridge_missing_gain)
        or (
            candidate_missing_support >= 0.45
            and candidate_missing_support >= old_missing_support + 0.08
        )
    )
    bridge_anchor_passed = (
        candidate_anchor_overlap >= float(min_bridge_anchor_overlap)
        or candidate_entity_score >= 0.28
    )
    bridge_relation_passed = (
        candidate_relation_score >= 0.20
        or candidate_entity_score >= 0.30
    )
    bridge_constraint_passed = (
        bridge_slot_replace_passed
        or (
            old_pair_deficit
            and bridge_score_passed
            and bridge_margin_passed
            and bridge_missing_gain_passed
            and bridge_anchor_passed
            and bridge_relation_passed
        )
    )
    rare_schema_signal = (
        schema_idf >= max(min_schema_idf + 0.10, 0.40)
        or 0 < schema_frequency <= 20
    )
    calibrated_min_score = min_score
    calibrated_min_schema_score = min_schema_score
    calibrated_min_gain = min_gain
    if question_type == "bridge_or_compositional" and clear_deficit:
        if old_pair_deficit:
            calibrated_min_score -= 0.16
            calibrated_min_schema_score -= 0.12
            calibrated_min_gain -= 0.04
        else:
            calibrated_min_score -= 0.12
            calibrated_min_schema_score -= 0.10
            calibrated_min_gain -= 0.03
        if rare_schema_signal and schema_overlap >= 0.06:
            calibrated_min_score -= 0.03
            calibrated_min_schema_score -= 0.04
        if answer_relation_signal >= 0.45 or candidate_query_support >= 0.42:
            calibrated_min_score -= 0.03
    calibrated_min_score = max(0.44, float(calibrated_min_score))
    calibrated_min_schema_score = max(0.50, float(calibrated_min_schema_score))
    calibrated_min_gain = max(0.03, float(calibrated_min_gain))
    score += 0.05 * _clamp(schema_specificity) if clear_deficit else 0.0
    score += 0.03 if rare_schema_signal and schema_overlap >= 0.06 else 0.0

    reject_reasons: List[str] = []
    if question_type != "bridge_or_compositional":
        reject_reasons.append("schema_deficit_not_bridge_question")
    if long_context_state and bridge_strength < 4:
        reject_reasons.append("schema_deficit_long_context_risk")
    if schema_score < calibrated_min_schema_score:
        reject_reasons.append("schema_deficit_low_schema_score")
    if schema_idf < min_schema_idf:
        reject_reasons.append("schema_deficit_low_schema_idf")
    if original_rank < 2 or original_rank >= int(max_original_rank):
        reject_reasons.append("schema_deficit_outside_rank_window")
    if score < calibrated_min_score:
        reject_reasons.append("schema_deficit_low_score")
    if gain < calibrated_min_gain and not clear_deficit:
        reject_reasons.append("schema_deficit_low_gain_without_clear_deficit")
    if not clear_deficit:
        reject_reasons.append("schema_deficit_no_current_evidence_deficit")
    if require_bridge_constraint and not bridge_constraint_passed:
        reject_reasons.append("schema_deficit_bridge_constraint_failed")
        if not old_pair_deficit:
            reject_reasons.append("schema_deficit_no_bridge_old_pair_deficit")
        if not bridge_score_passed:
            reject_reasons.append("schema_deficit_low_bridge_score")
        if not bridge_margin_passed:
            reject_reasons.append("schema_deficit_low_bridge_margin")
        if not bridge_missing_gain_passed:
            reject_reasons.append("schema_deficit_low_bridge_missing_gain")
        if not bridge_anchor_passed:
            reject_reasons.append("schema_deficit_low_bridge_anchor")
        if not bridge_relation_passed:
            reject_reasons.append("schema_deficit_low_bridge_relation")
    if candidate_query_support < 0.32 and answer_relation_signal < 0.35:
        reject_reasons.append("schema_deficit_weak_candidate_support")
    if schema_overlap < 0.06 and candidate_query_support < 0.40:
        reject_reasons.append("schema_deficit_weak_query_overlap")

    accepted = not reject_reasons
    info.update({
        "accepted": accepted,
        "reason": "schema_deficit_promoted" if accepted else "schema_deficit_rejected",
        "reject_reasons": reject_reasons,
        "score": round(float(score), 6),
        "gain": round(float(gain), 6),
        "schema_score": round(float(schema_score), 6),
        "schema_idf": round(float(schema_idf), 6),
        "schema_frequency": schema_frequency,
        "schema_overlap": round(float(schema_overlap), 6),
        "original_rank": original_rank,
        "bridge_strength": bridge_strength,
        "question_type": question_type,
        "long_context_state": bool(long_context_state),
        "old_pair_deficit": bool(old_pair_deficit),
        "schema_supported_deficit": bool(schema_supported_deficit),
        "clear_deficit": bool(clear_deficit),
        "schema_specificity": round(float(schema_specificity), 6),
        "lexical_deficit": bool(lexical_deficit),
        "missing_gain": round(float(missing_gain), 6),
        "bridge_score": round(float(bridge_score), 6),
        "bridge_margin": round(float(bridge_margin), 6),
        "bridge_slot_replace_passed": bool(bridge_slot_replace_passed),
        "bridge_constraint_required": bool(require_bridge_constraint),
        "bridge_constraint_passed": bool(bridge_constraint_passed),
        "bridge_score_passed": bool(bridge_score_passed),
        "bridge_margin_passed": bool(bridge_margin_passed),
        "bridge_missing_gain_passed": bool(bridge_missing_gain_passed),
        "bridge_anchor_passed": bool(bridge_anchor_passed),
        "bridge_relation_passed": bool(bridge_relation_passed),
        "pair_margin": round(float(pair_margin), 6),
        "candidate_query_support": round(float(candidate_query_support), 6),
        "base_query_support": round(float(base_query_support), 6),
        "candidate_pair_coverage": round(float(candidate_pair_coverage), 6),
        "base_pair_coverage": round(float(base_pair_coverage), 6),
        "candidate_complementarity": round(float(candidate_complementarity), 6),
        "base_complementarity": round(float(base_complementarity), 6),
        "candidate_missing_slot_support": round(float(candidate_missing_support), 6),
        "old_missing_slot_support": round(float(old_missing_support), 6),
        "answer_relation_signal": round(float(answer_relation_signal), 6),
        "thresholds": {
            "min_score": round(float(min_score), 6),
            "min_gain": round(float(min_gain), 6),
            "min_schema_score": round(float(min_schema_score), 6),
            "min_schema_idf": round(float(min_schema_idf), 6),
            "calibrated_min_score": round(float(calibrated_min_score), 6),
            "calibrated_min_gain": round(float(calibrated_min_gain), 6),
            "calibrated_min_schema_score": round(float(calibrated_min_schema_score), 6),
            "max_original_rank": int(max_original_rank),
            "require_bridge_constraint": bool(require_bridge_constraint),
            "min_bridge_score": round(float(min_bridge_score), 6),
            "min_bridge_margin": round(float(min_bridge_margin), 6),
            "min_bridge_missing_gain": round(float(min_bridge_missing_gain), 6),
            "min_bridge_anchor_overlap": round(float(min_bridge_anchor_overlap), 6),
        },
    })
    return info


def _schema_deficit_soft_append_gate(
    candidate_item: Dict[str, Any],
    schema_deficit_info: Optional[Dict[str, Any]],
    intervention_gate: Optional[Dict[str, Any]],
    enabled: bool = True,
    min_score: float = 0.48,
    min_schema_score: float = 0.58,
    min_schema_idf: float = 0.24,
    min_query_support: float = 0.34,
    max_original_rank: int = 80,
) -> Dict[str, Any]:
    """Allow strong schema-deficit candidates to append evidence without rank edits."""
    info: Dict[str, Any] = {
        "enabled": bool(enabled),
        "accepted": False,
        "reason": "disabled" if not enabled else "not_evaluated",
        "reject_reasons": [],
    }
    if not enabled:
        return info
    schema_deficit_info = schema_deficit_info or {}
    if schema_deficit_info.get("accepted"):
        info.update({
            "reason": "already_promoted",
            "reject_reasons": ["schema_soft_append_already_promoted"],
        })
        return info

    source_set = {
        str(source)
        for source in (candidate_item.get("candidate_sources", []) or [])
        if source
    }
    candidate_source = str(candidate_item.get("candidate_source") or "")
    if candidate_source:
        source_set.add(candidate_source)
    schema_item = candidate_item.get("schema_memory") or {}
    if "query_schema_memory" not in source_set and not schema_item:
        info.update({
            "reason": "no_query_schema_signal",
            "reject_reasons": ["schema_soft_append_no_query_schema_signal"],
        })
        return info

    try:
        original_rank = int(candidate_item.get("original_rank"))
    except (TypeError, ValueError):
        original_rank = int(schema_deficit_info.get("original_rank", 999999) or 999999)

    question_type = str(
        schema_deficit_info.get(
            "question_type",
            (intervention_gate or {}).get("question_type", "unknown"),
        ) or "unknown"
    )
    bridge_strength = int(
        schema_deficit_info.get(
            "bridge_strength",
            (intervention_gate or {}).get("bridge_strength", 0),
        ) or 0
    )
    long_context_state = bool(
        schema_deficit_info.get(
            "long_context_state",
            bool((intervention_gate or {}).get("long_context_shape", False))
            or bool((intervention_gate or {}).get("long_answer_signal", False)),
        )
    )
    score = float(schema_deficit_info.get("score", 0.0) or 0.0)
    schema_score = float(schema_deficit_info.get("schema_score", schema_item.get("score", 0.0)) or 0.0)
    schema_idf = float(schema_deficit_info.get("schema_idf", schema_item.get("schema_idf", 0.0)) or 0.0)
    schema_overlap = float(
        schema_deficit_info.get(
            "schema_overlap",
            max(
                float(schema_item.get("query_overlap", candidate_item.get("query_overlap", 0.0)) or 0.0),
                float(schema_item.get("context_overlap", 0.0) or 0.0),
                float(candidate_item.get("query_overlap", 0.0) or 0.0),
            ),
        )
        or 0.0
    )
    candidate_query_support = float(schema_deficit_info.get("candidate_query_support", 0.0) or 0.0)
    candidate_pair_coverage = float(schema_deficit_info.get("candidate_pair_coverage", 0.0) or 0.0)
    base_pair_coverage = float(schema_deficit_info.get("base_pair_coverage", 0.0) or 0.0)
    candidate_complementarity = float(schema_deficit_info.get("candidate_complementarity", 0.0) or 0.0)
    candidate_missing_support = float(schema_deficit_info.get("candidate_missing_slot_support", 0.0) or 0.0)
    missing_gain = float(schema_deficit_info.get("missing_gain", 0.0) or 0.0)
    answer_relation_signal = float(schema_deficit_info.get("answer_relation_signal", 0.0) or 0.0)
    bridge_constraint_passed = bool(schema_deficit_info.get("bridge_constraint_passed", False))

    coverage_signal = (
        candidate_query_support >= float(min_query_support)
        or schema_overlap >= 0.12
        or candidate_complementarity >= 0.12
        or candidate_missing_support >= 0.40
        or candidate_pair_coverage >= base_pair_coverage + 0.10
        or answer_relation_signal >= 0.45
    )
    schema_signal = (
        schema_score >= float(min_schema_score)
        and schema_idf >= float(min_schema_idf)
        and score >= float(min_score)
    )

    reject_reasons: List[str] = []
    if question_type != "bridge_or_compositional":
        reject_reasons.append("schema_soft_append_not_bridge_question")
    if long_context_state and bridge_strength < 4:
        reject_reasons.append("schema_soft_append_long_context_risk")
    if original_rank < 2 or original_rank >= int(max_original_rank):
        reject_reasons.append("schema_soft_append_outside_rank_window")
    if not schema_signal:
        reject_reasons.append("schema_soft_append_weak_schema_signal")
    if not coverage_signal:
        reject_reasons.append("schema_soft_append_weak_coverage_signal")
    if (
        "schema_deficit_no_query_schema_signal"
        in set(schema_deficit_info.get("reject_reasons", []) or [])
    ):
        reject_reasons.append("schema_soft_append_no_query_schema_signal")

    accepted = not reject_reasons
    info.update({
        "accepted": accepted,
        "reason": "schema_deficit_soft_appended" if accepted else "schema_deficit_soft_append_rejected",
        "reject_reasons": reject_reasons,
        "score": round(float(score), 6),
        "schema_score": round(float(schema_score), 6),
        "schema_idf": round(float(schema_idf), 6),
        "schema_overlap": round(float(schema_overlap), 6),
        "original_rank": original_rank,
        "question_type": question_type,
        "bridge_strength": bridge_strength,
        "long_context_state": bool(long_context_state),
        "candidate_query_support": round(float(candidate_query_support), 6),
        "candidate_pair_coverage": round(float(candidate_pair_coverage), 6),
        "base_pair_coverage": round(float(base_pair_coverage), 6),
        "candidate_complementarity": round(float(candidate_complementarity), 6),
        "candidate_missing_slot_support": round(float(candidate_missing_support), 6),
        "missing_gain": round(float(missing_gain), 6),
        "answer_relation_signal": round(float(answer_relation_signal), 6),
        "bridge_constraint_passed": bool(bridge_constraint_passed),
        "schema_signal": bool(schema_signal),
        "coverage_signal": bool(coverage_signal),
        "thresholds": {
            "min_score": round(float(min_score), 6),
            "min_schema_score": round(float(min_schema_score), 6),
            "min_schema_idf": round(float(min_schema_idf), 6),
            "min_query_support": round(float(min_query_support), 6),
            "max_original_rank": int(max_original_rank),
        },
    })
    return info


def _schema_reliability_gate(
    question: Any,
    candidate_doc: Any,
    candidate_item: Dict[str, Any],
    schema_deficit_info: Optional[Dict[str, Any]],
    soft_append_info: Optional[Dict[str, Any]],
    intervention_gate: Optional[Dict[str, Any]],
    enabled: bool = True,
    min_score: float = 0.62,
    rescue_min_score: float = 0.76,
    min_anchor_overlap: float = 0.08,
    max_schema_frequency: int = 80,
    numeric_requires_value: bool = True,
) -> Dict[str, Any]:
    """Calibrate schema-memory candidates before they are appended as QA evidence.

    This gate is intentionally rank-stable: it never promotes a candidate into
    the retrieval top-k. It only decides whether a query-schema memory edge is
    reliable enough to be appended to the reader as extra evidence.
    """
    soft_append_info = soft_append_info or {}
    schema_deficit_info = schema_deficit_info or {}
    intervention_gate = intervention_gate or {}
    info: Dict[str, Any] = {
        "enabled": bool(enabled),
        "accepted": bool(soft_append_info.get("accepted", False)) if not enabled else False,
        "rescued": False,
        "action": "disabled" if not enabled else "not_evaluated",
        "reason": "disabled" if not enabled else "not_evaluated",
        "reject_reasons": [],
    }
    if not enabled:
        return info

    source_set = {
        str(source)
        for source in (candidate_item.get("candidate_sources", []) or [])
        if source
    }
    candidate_source = str(candidate_item.get("candidate_source") or "")
    if candidate_source:
        source_set.add(candidate_source)
    schema_item = candidate_item.get("schema_memory") or {}
    has_schema_signal = "query_schema_memory" in source_set or bool(schema_item)

    question_type = str(
        schema_deficit_info.get(
            "question_type",
            intervention_gate.get("question_type", "unknown"),
        )
        or "unknown"
    )
    bridge_strength = int(
        schema_deficit_info.get(
            "bridge_strength",
            intervention_gate.get("bridge_strength", 0),
        )
        or 0
    )
    long_context_state = bool(
        schema_deficit_info.get(
            "long_context_state",
            bool(intervention_gate.get("long_context_shape", False))
            or bool(intervention_gate.get("long_answer_signal", False)),
        )
    )

    schema_score = float(
        schema_deficit_info.get("schema_score", schema_item.get("score", 0.0)) or 0.0
    )
    schema_idf = float(
        schema_deficit_info.get("schema_idf", schema_item.get("schema_idf", 0.0)) or 0.0
    )
    schema_support = float(schema_item.get("schema_support", 0.0) or 0.0)
    try:
        schema_frequency = int(schema_item.get("schema_frequency", 0) or 0)
    except (TypeError, ValueError):
        schema_frequency = 0
    query_overlap = max(
        float(candidate_item.get("query_overlap", 0.0) or 0.0),
        float(schema_item.get("query_overlap", 0.0) or 0.0),
        float(schema_deficit_info.get("schema_overlap", 0.0) or 0.0),
    )
    context_overlap = float(schema_item.get("context_overlap", 0.0) or 0.0)
    candidate_query_support = float(
        schema_deficit_info.get("candidate_query_support", 0.0) or 0.0
    )
    missing_gain = float(schema_deficit_info.get("missing_gain", 0.0) or 0.0)
    candidate_missing_support = float(
        schema_deficit_info.get("candidate_missing_slot_support", 0.0) or 0.0
    )
    answer_relation_signal = float(
        schema_deficit_info.get("answer_relation_signal", 0.0) or 0.0
    )
    bridge_constraint_passed = bool(schema_deficit_info.get("bridge_constraint_passed", False))

    relation_type = _infer_relation_guard_type(question)
    relation_signal = _relation_cue_score(candidate_doc, relation_type)
    answer_target_type = infer_answer_target_type(question)
    answer_type_signal = _answer_type_signal_score(candidate_doc, answer_target_type)

    frequency_prior = 1.0
    if max_schema_frequency > 0 and schema_frequency > 0:
        frequency_prior = _clamp(1.0 - (schema_frequency / max(float(max_schema_frequency), 1.0)) * 0.60)
    anchor_signal = max(query_overlap, context_overlap)
    reliability_score = (
        0.24 * _clamp(schema_score)
        + 0.18 * _clamp(schema_idf)
        + 0.10 * _clamp(schema_support)
        + 0.14 * _clamp(anchor_signal)
        + 0.12 * _clamp(candidate_query_support)
        + 0.10 * _clamp(max(missing_gain, candidate_missing_support))
        + 0.05 * _clamp(max(relation_signal, answer_relation_signal))
        + 0.03 * _clamp(answer_type_signal)
        + 0.04 * _clamp(frequency_prior)
    )
    if bridge_constraint_passed:
        reliability_score += 0.04
    reliability_score = _clamp(reliability_score)

    reject_reasons: List[str] = []
    if not has_schema_signal:
        reject_reasons.append("schema_reliability_no_schema_signal")
    if max_schema_frequency > 0 and schema_frequency > max_schema_frequency:
        reject_reasons.append("schema_reliability_high_frequency_schema")
    if long_context_state and bridge_strength < 4:
        reject_reasons.append("schema_reliability_long_context_risk")
    if (
        anchor_signal < min_anchor_overlap
        and max(relation_signal, answer_relation_signal) < 0.20
        and candidate_query_support < 0.35
    ):
        reject_reasons.append("schema_reliability_weak_anchor")
    if (
        numeric_requires_value
        and answer_target_type in {"date", "number"}
        and answer_type_signal <= 0.0
    ):
        reject_reasons.append("schema_reliability_missing_answer_type_value")

    soft_accepted = bool(soft_append_info.get("accepted", False))
    hard_rejected = bool(reject_reasons)
    accepted = soft_accepted and not hard_rejected and reliability_score >= float(min_score)
    rescued = (
        not soft_accepted
        and not hard_rejected
        and question_type == "bridge_or_compositional"
        and reliability_score >= float(rescue_min_score)
        and (
            bridge_constraint_passed
            or missing_gain >= 0.10
            or candidate_missing_support >= 0.45
            or candidate_query_support >= 0.42
        )
    )
    if not accepted and not rescued and reliability_score < float(min_score):
        reject_reasons.append("schema_reliability_low_score")

    if accepted:
        action = "pass_soft_append"
        reason = "schema_reliability_passed"
    elif rescued:
        action = "rescue_append"
        reason = "schema_reliability_rescued"
    else:
        action = "reject_append"
        reason = "schema_reliability_rejected"

    info.update({
        "accepted": bool(accepted or rescued),
        "rescued": bool(rescued),
        "action": action,
        "reason": reason,
        "reject_reasons": reject_reasons,
        "score": round(float(reliability_score), 6),
        "schema_score": round(float(schema_score), 6),
        "schema_idf": round(float(schema_idf), 6),
        "schema_support": round(float(schema_support), 6),
        "schema_frequency": int(schema_frequency),
        "frequency_prior": round(float(frequency_prior), 6),
        "anchor_signal": round(float(anchor_signal), 6),
        "candidate_query_support": round(float(candidate_query_support), 6),
        "missing_gain": round(float(missing_gain), 6),
        "candidate_missing_slot_support": round(float(candidate_missing_support), 6),
        "relation_type": relation_type,
        "relation_signal": round(float(relation_signal), 6),
        "answer_relation_signal": round(float(answer_relation_signal), 6),
        "answer_target_type": answer_target_type,
        "answer_type_signal": round(float(answer_type_signal), 6),
        "bridge_constraint_passed": bool(bridge_constraint_passed),
        "question_type": question_type,
        "bridge_strength": bridge_strength,
        "thresholds": {
            "min_score": round(float(min_score), 6),
            "rescue_min_score": round(float(rescue_min_score), 6),
            "min_anchor_overlap": round(float(min_anchor_overlap), 6),
            "max_schema_frequency": int(max_schema_frequency),
            "numeric_requires_value": bool(numeric_requires_value),
        },
    })
    return info


def _counterfactual_pair_features(
    question: Any,
    top1_doc: Any,
    second_doc: Any,
    evidence_item: Optional[Dict[str, Any]],
    trusted_rank2_prior: bool = False,
) -> Dict[str, float]:
    question_tokens = _content_token_set(question)
    top1_tokens = _content_token_set(top1_doc)
    second_tokens = _content_token_set(second_doc)
    pair_tokens = top1_tokens | second_tokens
    query_tokens = _content_token_set(_item_query_text(evidence_item))

    if question_tokens:
        pair_coverage = len(question_tokens & pair_tokens) / len(question_tokens)
        top1_coverage = len(question_tokens & top1_tokens) / len(question_tokens)
        second_coverage = len(question_tokens & second_tokens) / len(question_tokens)
        complementarity = len((question_tokens - top1_tokens) & second_tokens) / len(question_tokens)
    else:
        pair_coverage = 0.0
        top1_coverage = 0.0
        second_coverage = 0.0
        complementarity = 0.0

    if query_tokens:
        query_support = len(query_tokens & second_tokens) / len(query_tokens)
    else:
        query_support = second_coverage

    if top1_tokens and second_tokens:
        redundancy = len(top1_tokens & second_tokens) / max(1, len(top1_tokens | second_tokens))
    else:
        redundancy = 0.0

    plan_quality = 0.0
    query_overlap = 0.0
    query_agreement = 0.0
    rank_prior = 0.0
    if evidence_item:
        plan_quality = _clamp(float(evidence_item.get("plan_score", 0.0)) / 1.20)
        query_overlap = _clamp(evidence_item.get("query_overlap", 0.0))
        query_agreement = _clamp(float(evidence_item.get("query_count", 0)) / 3.0)
        rank_prior = _rank_prior(evidence_item.get("original_rank"))

    if trusted_rank2_prior:


        plan_quality = max(plan_quality, 0.62)
        query_agreement = max(query_agreement, 0.34)
        rank_prior = max(rank_prior, 0.50)

    score = (
        0.34 * pair_coverage
        + 0.22 * complementarity
        + 0.16 * query_support
        + 0.14 * plan_quality
        + 0.08 * query_agreement
        + 0.06 * rank_prior
        + 0.04 * query_overlap
        - 0.06 * redundancy
    )
    return {
        "score": _clamp(score, -1.0, 2.0),
        "pair_coverage": _clamp(pair_coverage),
        "top1_coverage": _clamp(top1_coverage),
        "second_coverage": _clamp(second_coverage),
        "complementarity": _clamp(complementarity),
        "query_support": _clamp(query_support),
        "plan_quality": _clamp(plan_quality),
        "query_agreement": _clamp(query_agreement),
        "rank_prior": _clamp(rank_prior),
        "query_overlap": _clamp(query_overlap),
        "redundancy": _clamp(redundancy),
    }


def _counterfactual_top2_pair_verifier(
    question: Any,
    top1_doc: Any,
    old_rank2_doc: Any,
    candidate_doc: Any,
    candidate_item: Dict[str, Any],
    old_rank2_item: Optional[Dict[str, Any]],
    min_margin: float,
    max_coverage_loss: float,
    min_candidate_query_support: float,
) -> Dict[str, Any]:
    info = {
        "enabled": True,
        "accepted": False,
        "reason": "not_evaluated",
        "reject_reasons": [],
    }
    if not top1_doc or not old_rank2_doc or not candidate_doc:
        info.update({
            "reason": "missing_pair_text",
            "reject_reasons": ["pair_verifier_missing_text"],
        })
        return info

    base = _counterfactual_pair_features(
        question=question,
        top1_doc=top1_doc,
        second_doc=old_rank2_doc,
        evidence_item=old_rank2_item,
        trusted_rank2_prior=True,
    )
    candidate = _counterfactual_pair_features(
        question=question,
        top1_doc=top1_doc,
        second_doc=candidate_doc,
        evidence_item=candidate_item,
        trusted_rank2_prior=False,
    )
    margin = candidate["score"] - base["score"]
    coverage_loss = base["pair_coverage"] - candidate["pair_coverage"]
    complementarity_loss = base["complementarity"] - candidate["complementarity"]

    reject_reasons = []
    if margin < min_margin:
        reject_reasons.append("pair_verifier_low_margin")
    if coverage_loss > max_coverage_loss:
        reject_reasons.append("pair_verifier_coverage_loss")
    if complementarity_loss > max(0.02, max_coverage_loss):
        reject_reasons.append("pair_verifier_complementarity_loss")
    if candidate["query_support"] < min_candidate_query_support:
        reject_reasons.append("pair_verifier_weak_candidate_query_support")

    accepted = not reject_reasons
    info.update({
        "accepted": accepted,
        "reason": "counterfactual_pair_margin_passed" if accepted else "counterfactual_pair_rejected",
        "reject_reasons": reject_reasons,
        "base_pair_score": round(base["score"], 6),
        "candidate_pair_score": round(candidate["score"], 6),
        "pair_margin": round(margin, 6),
        "coverage_loss": round(coverage_loss, 6),
        "complementarity_loss": round(complementarity_loss, 6),
        "base": {k: round(v, 6) for k, v in base.items()},
        "candidate": {k: round(v, 6) for k, v in candidate.items()},
        "min_margin": round(float(min_margin), 6),
        "max_coverage_loss": round(float(max_coverage_loss), 6),
        "min_candidate_query_support": round(float(min_candidate_query_support), 6),
    })
    return info


def _split_sentences(text: Any, max_sentences: int = 40) -> List[str]:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    if not cleaned:
        return []
    pieces = re.split(r"(?<=[.!?])\s+|;\s+", cleaned)
    return [piece.strip()[:520] for piece in pieces if piece.strip()][:max_sentences]


def _infer_relation_guard_type(question: Any) -> str:
    q = re.sub(r"\s+", " ", str(question or "").lower()).strip()
    if re.search(r"\bwhere\b", q) and re.search(r"\bborn\b|\bbirthplace\b|\bplace of birth\b", q):
        return "birth_place"
    if re.search(r"\bwhen\b|\bwhat date\b|\bwhat year\b|\bdate\b", q) and re.search(r"\bborn\b|\bbirth\b", q):
        return "birth_date"
    if re.search(r"\bwhere\b", q) and re.search(r"\bdied\b|\bdeath\b|\bplace of death\b", q):
        return "death_place"
    if re.search(r"\bwhen\b|\bwhat date\b|\bwhat year\b|\bdate\b", q) and re.search(r"\bdied\b|\bdeath\b", q):
        return "death_date"
    if re.search(
        r"\b(?:father|mother|grandfather|grandmother|paternal|maternal|spouse|wife|husband|son|daughter)\b",
        q,
    ):
        return "family_person"
    if re.search(r"\b(?:director|author|writer|composer|producer|creator|founder|designer)\b", q):
        return "creator_person"
    if re.search(r"\b(?:where|located|capital|which city|which country|which state|which county)\b", q):
        return "location"
    if re.search(r"\b(?:when|what year|what date|date|year)\b", q) and re.search(
        r"\b(?:released|published|premiered|aired|broadcast|issued)\b", q,
    ):
        return "release_date"
    if re.search(r"\b(?:when|what year|what date|date|year)\b", q) and re.search(
        r"\b(?:founded|established|formed|created|opened)\b", q,
    ):
        return "founding_date"
    return "generic"


def _relation_cue_score(text: Any, relation_type: str) -> float:
    if not text or relation_type == "generic":
        return 0.0
    patterns = RELATION_CUE_PATTERNS.get(relation_type) or []
    if not patterns:
        return 0.0
    sentences = _split_sentences(text)
    if not sentences:
        return 0.0
    cue_hits = 0
    object_hits = 0
    for sentence in sentences:
        if any(pattern.search(sentence) for pattern in patterns):
            cue_hits += 1
            if DATE_SPAN_RE.search(sentence) or TITLE_CASE_RE.search(sentence):
                object_hits += 1
    return _clamp(0.40 * min(cue_hits, 3) / 3.0 + 0.60 * min(object_hits, 3) / 3.0)


def _normalize_object_span(text: Any) -> str:
    tokens = [
        token
        for token in TOKEN_RE.findall(str(text or "").lower())
        if len(token) > 1 and token not in CONTENT_STOPWORDS
    ]
    return " ".join(tokens[:8])


def _normalize_anchor_span(text: Any) -> Tuple[str, ...]:
    tokens = [
        token
        for token in TOKEN_RE.findall(str(text or "").lower())
        if len(token) > 1 and token not in ENTITY_ANCHOR_STOPWORDS
    ]
    return tuple(tokens[:8])


def _extract_entity_anchor_sets(*texts: Any, limit: int = 12) -> List[Tuple[str, ...]]:
    anchors: List[Tuple[str, ...]] = []
    seen = set()
    for text in texts:
        text = str(text or "")
        if not text:
            continue
        spans = list(QUOTED_SPAN_RE.findall(text)) + list(TITLE_CASE_RE.findall(text))
        for span in spans:
            normalized = _normalize_anchor_span(span)
            if not normalized:
                continue
            if len(normalized) == 1 and len(normalized[0]) <= 3:
                continue
            if normalized in seen:
                continue
            seen.add(normalized)
            anchors.append(normalized)
            if len(anchors) >= limit:
                return anchors
    return anchors


def _format_anchor(anchor: Tuple[str, ...]) -> str:
    return " ".join(anchor)


def _sentence_anchor_overlap(sentence: Any, anchors: List[Tuple[str, ...]]) -> float:
    if not anchors:
        return 0.0
    sentence_tokens = _content_token_set(sentence)
    if not sentence_tokens:
        return 0.0
    best = 0.0
    for anchor in anchors:
        anchor_tokens = set(anchor)
        if not anchor_tokens:
            continue
        best = max(best, len(anchor_tokens & sentence_tokens) / len(anchor_tokens))
    return _clamp(best)


def _object_matches_anchor(normalized_object: str, anchors: List[Tuple[str, ...]]) -> bool:
    object_tokens = set(normalized_object.split())
    if not object_tokens:
        return False
    for anchor in anchors:
        anchor_tokens = set(anchor)
        if not anchor_tokens:
            continue
        overlap = len(object_tokens & anchor_tokens) / max(1, min(len(object_tokens), len(anchor_tokens)))
        if overlap >= 0.80:
            return True
    return False


def _extract_relation_objects_from_sentences(
    sentences: List[str],
    relation_type: str,
    limit: int = 8,
    anchors: Optional[List[Tuple[str, ...]]] = None,
) -> List[str]:
    objects: List[str] = []
    anchors = anchors or []
    for sentence in sentences[:12]:
        if relation_type in {"birth_date", "death_date"}:
            spans = DATE_SPAN_RE.findall(sentence)
        else:
            spans = TITLE_CASE_RE.findall(sentence)
        for span in spans:
            normalized = _normalize_object_span(span)
            if not normalized or normalized in objects:
                continue
            if normalized in {"he", "she", "it", "him", "her", "his", "hers", "its"}:
                continue
            if _object_matches_anchor(normalized, anchors):
                continue
            objects.append(normalized)
            if len(objects) >= limit:
                return objects
    return objects


def _extract_relation_objects(text: Any, relation_type: str, limit: int = 8) -> List[str]:
    if not text or relation_type == "generic":
        return []
    patterns = RELATION_CUE_PATTERNS.get(relation_type) or []
    sentences = _split_sentences(text)
    if patterns:
        sentences = [
            sentence for sentence in sentences
            if any(pattern.search(sentence) for pattern in patterns)
        ]
    return _extract_relation_objects_from_sentences(sentences, relation_type, limit=limit)


def _entity_conditioned_relation_evidence(
    text: Any,
    relation_type: str,
    anchors: List[Tuple[str, ...]],
    anchor_min_overlap: float,
) -> Dict[str, Any]:
    info = {
        "score": 0.0,
        "objects": [],
        "relation_sentence_count": 0,
        "anchored_relation_sentence_count": 0,
        "best_anchor_overlap": 0.0,
        "reason": "not_applicable",
    }
    if not text or relation_type == "generic" or not anchors:
        return info

    patterns = RELATION_CUE_PATTERNS.get(relation_type) or []
    if not patterns:
        return info

    anchor_min_overlap = _clamp(anchor_min_overlap)
    relation_sentences = []
    anchored_sentences = []
    best_anchor_overlap = 0.0
    for sentence in _split_sentences(text, max_sentences=60):
        if not any(pattern.search(sentence) for pattern in patterns):
            continue
        relation_sentences.append(sentence)
        anchor_overlap = _sentence_anchor_overlap(sentence, anchors)
        best_anchor_overlap = max(best_anchor_overlap, anchor_overlap)
        if anchor_overlap >= anchor_min_overlap:
            anchored_sentences.append(sentence)

    objects = _extract_relation_objects_from_sentences(
        anchored_sentences,
        relation_type,
        limit=8,
        anchors=anchors,
    )
    score = (
        0.45 * min(len(anchored_sentences), 3) / 3.0
        + 0.35 * min(len(objects), 3) / 3.0
        + 0.20 * min(best_anchor_overlap, 1.0)
    )
    reason = "anchored_relation_found" if anchored_sentences else "no_anchored_relation_sentence"
    info.update({
        "score": _clamp(score, 0.0, 2.0),
        "objects": objects,
        "relation_sentence_count": len(relation_sentences),
        "anchored_relation_sentence_count": len(anchored_sentences),
        "best_anchor_overlap": round(best_anchor_overlap, 6),
        "reason": reason,
    })
    return info


def _relation_object_consistency_guard(
    question: Any,
    planned_query: Any,
    old_rank2_doc: Any,
    candidate_doc: Any,
    pair_verifier_info: Optional[Dict[str, Any]],
    min_relation_score: float,
    min_relation_margin: float,
    entity_conditioned: bool = False,
    entity_anchor_min_overlap: float = 0.50,
) -> Dict[str, Any]:
    relation_type = _infer_relation_guard_type(question)
    info = {
        "enabled": True,
        "accepted": True,
        "reason": "not_applicable",
        "reject_reasons": [],
        "relation_type": relation_type,
        "entity_conditioned": bool(entity_conditioned),
    }
    if relation_type == "generic" or not old_rank2_doc or not candidate_doc:
        return info

    old_score = _relation_cue_score(old_rank2_doc, relation_type)
    candidate_score = _relation_cue_score(candidate_doc, relation_type)
    old_objects = _extract_relation_objects(old_rank2_doc, relation_type)
    candidate_objects = _extract_relation_objects(candidate_doc, relation_type)
    effective_old_score = old_score
    effective_candidate_score = candidate_score
    effective_old_objects = old_objects
    effective_candidate_objects = candidate_objects
    guard_prefix = "relation_object"
    entity_info = None
    if entity_conditioned:
        anchors = _extract_entity_anchor_sets(question, planned_query)
        old_entity = _entity_conditioned_relation_evidence(
            old_rank2_doc,
            relation_type,
            anchors,
            anchor_min_overlap=entity_anchor_min_overlap,
        )
        candidate_entity = _entity_conditioned_relation_evidence(
            candidate_doc,
            relation_type,
            anchors,
            anchor_min_overlap=entity_anchor_min_overlap,
        )
        entity_info = {
            "anchor_min_overlap": round(float(entity_anchor_min_overlap), 6),
            "anchors": [_format_anchor(anchor) for anchor in anchors[:8]],
            "old": {
                **old_entity,
                "score": round(float(old_entity.get("score", 0.0)), 6),
            },
            "candidate": {
                **candidate_entity,
                "score": round(float(candidate_entity.get("score", 0.0)), 6),
            },
        }
        use_entity_conditioned = bool(anchors) and (
            float(old_entity.get("score", 0.0) or 0.0) >= min_relation_score
            or float(candidate_entity.get("score", 0.0) or 0.0) >= min_relation_score
            or bool(old_entity.get("objects"))
            or bool(candidate_entity.get("objects"))
        )
        if use_entity_conditioned:
            effective_old_score = float(old_entity.get("score", 0.0) or 0.0)
            effective_candidate_score = float(candidate_entity.get("score", 0.0) or 0.0)
            effective_old_objects = list(old_entity.get("objects") or [])
            effective_candidate_objects = list(candidate_entity.get("objects") or [])
            guard_prefix = "entity_conditioned_relation_object"

    old_object_set = set(old_objects)
    candidate_object_set = set(candidate_objects)
    effective_old_object_set = set(effective_old_objects)
    effective_candidate_object_set = set(effective_candidate_objects)
    pair_base = (pair_verifier_info or {}).get("base") or {}
    pair_candidate = (pair_verifier_info or {}).get("candidate") or {}
    old_query_support = float(pair_base.get("query_support", 0.0) or 0.0)
    candidate_query_support = float(pair_candidate.get("query_support", 0.0) or 0.0)

    reject_reasons = []
    if (
        effective_old_score >= min_relation_score
        and effective_candidate_score + min_relation_margin < effective_old_score
        and old_query_support >= 0.45
    ):
        reject_reasons.append(f"{guard_prefix}_weaker_than_original")
    if (
        effective_old_object_set
        and effective_candidate_object_set
        and effective_old_object_set.isdisjoint(effective_candidate_object_set)
        and effective_old_score >= min_relation_score
        and effective_candidate_score <= effective_old_score + min_relation_margin
        and old_query_support >= 0.45
        and candidate_query_support <= old_query_support + 0.20
    ):
        reject_reasons.append(f"{guard_prefix}_conflict_with_original")

    accepted = not reject_reasons
    info.update({
        "accepted": accepted,
        "reason": (
            f"{guard_prefix}_consistent"
            if accepted else f"{guard_prefix}_rejected"
        ),
        "reject_reasons": reject_reasons,
        "old_relation_score": round(old_score, 6),
        "candidate_relation_score": round(candidate_score, 6),
        "effective_old_relation_score": round(effective_old_score, 6),
        "effective_candidate_relation_score": round(effective_candidate_score, 6),
        "old_objects": old_objects[:6],
        "candidate_objects": candidate_objects[:6],
        "effective_old_objects": effective_old_objects[:6],
        "effective_candidate_objects": effective_candidate_objects[:6],
        "old_query_support": round(old_query_support, 6),
        "candidate_query_support": round(candidate_query_support, 6),
        "min_relation_score": round(float(min_relation_score), 6),
        "min_relation_margin": round(float(min_relation_margin), 6),
        "entity_conditioned_info": entity_info,
    })
    return info


def _normalize_answer_type_value(span: Any, target_type: str) -> str:
    value = re.sub(r"\s+", " ", str(span or "").strip().lower())
    value = value.replace(",", "")
    value = value.strip(" .,:;()[]{}")
    if target_type in {"date", "number"}:
        value = value.replace(" percent", "%")
    return value


def _extract_answer_type_values(
    text: Any,
    target_type: str,
    anchors: Optional[List[Tuple[str, ...]]] = None,
    limit: int = 12,
) -> List[str]:
    target_type = str(target_type or "factoid")
    anchors = anchors or []
    spans: List[str] = []
    text = str(text or "")
    if not text:
        return []
    if target_type == "date":
        spans.extend(DATE_SPAN_RE.findall(text))
        spans.extend(DECADE_VALUE_RE.findall(text))
    elif target_type == "number":
        spans.extend(NUMBER_VALUE_RE.findall(text))
    elif target_type in {"person", "location", "organization", "work", "factoid"}:
        spans.extend(TITLE_CASE_RE.findall(text))
    else:
        return []

    values: List[str] = []
    for span in spans:
        normalized = _normalize_answer_type_value(span, target_type)
        if not normalized or normalized in values:
            continue
        if target_type not in {"date", "number"} and _object_matches_anchor(normalized, anchors):
            continue
        if normalized in {"he", "she", "it", "him", "her", "his", "hers", "its"}:
            continue
        values.append(normalized)
        if len(values) >= limit:
            break
    return values


def _answer_type_value_profile(
    question: Any,
    planned_query: Any,
    doc: Any,
    target_type: str,
) -> Dict[str, Any]:
    relation_type = _infer_relation_guard_type(f"{question} {planned_query}")
    anchors = _extract_entity_anchor_sets(question, planned_query, limit=12)
    query_tokens = _content_token_set(planned_query) or _content_token_set(question)
    question_values = set(_extract_answer_type_values(question, target_type, anchors=[]))

    best_score = 0.0
    best_sentence = ""
    best_values: List[str] = []
    all_values: List[str] = []
    for sentence in _split_sentences(doc, max_sentences=80):
        values = [
            value for value in _extract_answer_type_values(sentence, target_type, anchors=anchors)
            if value not in question_values
        ]
        if not values:
            continue
        for value in values:
            if value not in all_values:
                all_values.append(value)

        sentence_tokens = _content_token_set(sentence)
        query_overlap = (
            len(query_tokens & sentence_tokens) / len(query_tokens)
            if query_tokens and sentence_tokens else 0.0
        )
        anchor_overlap = _sentence_anchor_overlap(sentence, anchors)
        patterns = RELATION_CUE_PATTERNS.get(relation_type) or []
        relation_hit = (
            1.0
            if patterns and any(pattern.search(sentence) for pattern in patterns)
            else 0.0
        )
        if relation_type == "generic":
            relation_hit = 0.20
        value_signal = min(1.0, len(values) / 2.0)
        score = (
            0.36 * value_signal
            + 0.24 * query_overlap
            + 0.20 * anchor_overlap
            + 0.14 * relation_hit
            + 0.06 * min(1.0, len(sentence_tokens) / 24.0 if sentence_tokens else 0.0)
        )
        if score > best_score:
            best_score = score
            best_sentence = sentence[:260]
            best_values = values[:6]

    return {
        "target_type": target_type,
        "relation_type": relation_type,
        "score": round(float(_clamp(best_score, 0.0, 2.0)), 6),
        "values": all_values[:12],
        "best_values": best_values,
        "best_sentence": best_sentence,
    }


def _answer_type_conflict_guard(
    question: Any,
    planned_query: Any,
    old_rank2_doc: Any,
    candidate_doc: Any,
    pair_verifier_info: Optional[Dict[str, Any]],
    min_old_signal: float = 0.55,
    min_candidate_signal: float = 0.35,
    min_conflict_margin: float = 0.08,
    numeric_only: bool = True,
) -> Dict[str, Any]:
    target_type = infer_answer_target_type(str(question or ""))
    info = {
        "enabled": True,
        "accepted": True,
        "reason": "not_applicable",
        "reject_reasons": [],
        "target_type": target_type,
    }
    supported_types = {"date", "number"} if numeric_only else {
        "date", "number", "person", "location", "organization", "work",
    }
    if target_type not in supported_types or not old_rank2_doc or not candidate_doc:
        return info

    min_old_signal = max(0.0, float(min_old_signal))
    min_candidate_signal = max(0.0, float(min_candidate_signal))
    min_conflict_margin = max(0.0, float(min_conflict_margin))
    old_profile = _answer_type_value_profile(question, "", old_rank2_doc, target_type)
    candidate_profile = _answer_type_value_profile(question, planned_query, candidate_doc, target_type)
    old_score = float(old_profile.get("score", 0.0) or 0.0)
    candidate_score = float(candidate_profile.get("score", 0.0) or 0.0)
    old_values = set(old_profile.get("values") or [])
    candidate_values = set(candidate_profile.get("values") or [])
    pair_base = (pair_verifier_info or {}).get("base") or {}
    pair_candidate = (pair_verifier_info or {}).get("candidate") or {}
    old_query_support = float(pair_base.get("query_support", 0.0) or 0.0)
    candidate_query_support = float(pair_candidate.get("query_support", 0.0) or 0.0)

    reject_reasons: List[str] = []
    if (
        old_score >= min_old_signal
        and candidate_score < min_candidate_signal
        and candidate_query_support <= old_query_support + 0.10
    ):
        reject_reasons.append("answer_type_candidate_missing_or_weak_value")

    conflict = bool(old_values and candidate_values and old_values.isdisjoint(candidate_values))
    old_not_weaker = (
        old_score >= min_old_signal
        and candidate_score <= old_score + min_conflict_margin
        and candidate_query_support <= old_query_support + 0.05
    )
    if conflict and old_not_weaker:
        reject_reasons.append("answer_type_value_conflict_with_original")

    if (
        target_type in {"date", "number"}
        and len(candidate_values) > 3
        and candidate_score < old_score + 0.15
        and old_score >= min_old_signal
    ):
        reject_reasons.append("answer_type_candidate_value_ambiguous")

    accepted = not reject_reasons
    info.update({
        "accepted": accepted,
        "reason": "answer_type_consistent" if accepted else "answer_type_conflict_rejected",
        "reject_reasons": reject_reasons,
        "numeric_only": bool(numeric_only),
        "min_old_signal": round(float(min_old_signal), 6),
        "min_candidate_signal": round(float(min_candidate_signal), 6),
        "min_conflict_margin": round(float(min_conflict_margin), 6),
        "old": old_profile,
        "candidate": candidate_profile,
        "old_query_support": round(old_query_support, 6),
        "candidate_query_support": round(candidate_query_support, 6),
    })
    return info


def _support_set_best_sentence_score(
    question_tokens: set,
    doc: Any,
    target_type: str,
    anchors: List[Tuple[str, ...]],
) -> float:
    best = 0.0
    for sentence in _split_sentences(doc, max_sentences=60):
        sentence_tokens = _content_token_set(sentence)
        if not sentence_tokens:
            continue
        query_overlap = (
            len(question_tokens & sentence_tokens) / max(1, len(question_tokens))
            if question_tokens else 0.0
        )
        answer_signal = _answer_type_signal_score(sentence, target_type)
        anchor_signal = _sentence_anchor_overlap(sentence, anchors)
        early_signal = 1.0 / (1.0 + min(12, len(sentence)))
        score = (
            0.46 * query_overlap
            + 0.24 * answer_signal
            + 0.22 * anchor_signal
            + 0.08 * early_signal
        )
        best = max(best, score)
    return _clamp(best)


def _support_set_jaccard(left: set, right: set) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / max(1, len(left | right))


def _support_set_anchor_hits(text: Any, anchors: List[Tuple[str, ...]]) -> set:
    hits = set()
    if not anchors:
        return hits
    text_tokens = _content_token_set(text)
    if not text_tokens:
        return hits
    for anchor in anchors:
        anchor_tokens = set(anchor)
        if not anchor_tokens:
            continue
        overlap = len(anchor_tokens & text_tokens) / max(1, len(anchor_tokens))
        if overlap >= 0.50:
            hits.add(_format_anchor(anchor))
    return hits


def _support_set_utility_rerank(
    question: Any,
    base_ids: List[int],
    base_scores: List[float],
    current_top: List[int],
    ranked_candidates: List[Dict[str, Any]],
    schema_memory_candidates: List[Dict[str, Any]],
    utility_candidates: List[Dict[str, Any]],
    doc_lookup: Optional[Callable[[int], str]],
    target_k: int = 5,
    protect_top_k: int = 2,
    max_replacements: int = 2,
    max_candidates: int = 96,
    max_original_rank: int = 200,
    min_candidate_score: float = 0.52,
    redundancy_penalty: float = 0.18,
    preserve_original_topk: bool = True,
    preserve_min_gain: float = 0.03,
    preserve_max_coverage_loss: float = 0.02,
    require_missing_slot_gate: bool = True,
    missing_slot_min_score: float = 0.50,
    missing_slot_min_improvement: float = 0.06,
    missing_slot_min_anchor_overlap: float = 0.34,
    missing_slot_require_relation: bool = True,
    missing_slot_min_missing_gain: float = 0.08,
) -> Tuple[List[int], Dict[str, Any], List[Dict[str, Any]]]:
    """Select a top-k support set with coverage/novelty-aware greedy utility.

    Unlike rank-2 replacement, this optimizes the whole evidence set. It keeps a
    protected prefix, then greedily selects passages that add question-token
    coverage, entity/schema support, answer-type signal, and bridge-slot cues
    while penalizing redundancy and conflicting numeric/date values.
    """
    target_k = max(1, int(target_k))
    protect_top_k = max(0, min(int(protect_top_k), target_k))
    max_replacements = max(0, int(max_replacements))
    max_candidates = max(target_k, int(max_candidates))
    max_original_rank = max(target_k, int(max_original_rank))
    min_candidate_score = max(0.0, float(min_candidate_score))
    redundancy_penalty = max(0.0, float(redundancy_penalty))
    preserve_original_topk = bool(preserve_original_topk)
    preserve_min_gain = max(0.0, float(preserve_min_gain))
    preserve_max_coverage_loss = max(0.0, float(preserve_max_coverage_loss))
    require_missing_slot_gate = bool(require_missing_slot_gate)
    missing_slot_min_score = max(0.0, float(missing_slot_min_score))
    missing_slot_min_improvement = max(0.0, float(missing_slot_min_improvement))
    missing_slot_min_anchor_overlap = _clamp(missing_slot_min_anchor_overlap)
    missing_slot_require_relation = bool(missing_slot_require_relation)
    missing_slot_min_missing_gain = max(0.0, float(missing_slot_min_missing_gain))
    info = {
        "enabled": True,
        "applied": False,
        "reason": "not_evaluated",
        "target_k": target_k,
        "protect_top_k": protect_top_k,
        "max_replacements": max_replacements,
        "max_candidates": max_candidates,
        "max_original_rank": max_original_rank,
        "min_candidate_score": round(float(min_candidate_score), 6),
        "redundancy_penalty": round(float(redundancy_penalty), 6),
        "coverage_preservation_policy": {
            "preserve_original_topk": preserve_original_topk,
            "min_gain": round(float(preserve_min_gain), 6),
            "max_coverage_loss": round(float(preserve_max_coverage_loss), 6),
        },
        "require_missing_slot_gate": require_missing_slot_gate,
        "missing_slot_gate_policy": {
            "min_score": round(float(missing_slot_min_score), 6),
            "min_improvement": round(float(missing_slot_min_improvement), 6),
            "min_anchor_overlap": round(float(missing_slot_min_anchor_overlap), 6),
            "require_relation": missing_slot_require_relation,
            "min_missing_gain": round(float(missing_slot_min_missing_gain), 6),
        },
        "candidate_count": 0,
        "selected_doc_ids": [],
        "replaced_doc_ids": [],
        "replacement_count": 0,
        "promoted_to_top2": 0,
        "promoted_to_top5": 0,
        "missing_slot_gate_rejected_count": 0,
        "missing_slot_gate_reject_reason_counts": {},
        "coverage_preservation_rejected_count": 0,
        "coverage_preservation_reject_reason_counts": {},
        "selected": [],
        "top_candidates": [],
    }
    if doc_lookup is None:
        info["reason"] = "missing_doc_lookup"
        return current_top[:target_k], info, []
    if len(base_ids) < target_k or not current_top:
        info["reason"] = "not_enough_ranked_docs"
        return current_top[:target_k], info, []

    old_top = [int(doc_id) for doc_id in base_ids[:target_k]]
    old_top_set = set(old_top)
    current_top = [int(doc_id) for doc_id in current_top[:target_k]]
    base_rank = {int(doc_id): rank for rank, doc_id in enumerate(base_ids)}
    base_score = {int(doc_id): float(base_scores[rank]) for rank, doc_id in enumerate(base_ids)}
    question_tokens = _content_token_set(question)
    target_type = infer_answer_target_type(str(question or ""))
    relation_type = _infer_relation_guard_type(question)
    anchors = _extract_entity_anchor_sets(question, limit=14)
    top1_doc = str(doc_lookup(int(base_ids[0])) or "") if len(base_ids) >= 1 else ""
    old_rank2_doc = str(doc_lookup(int(base_ids[1])) or "") if len(base_ids) >= 2 else ""

    pool: Dict[int, Dict[str, Any]] = {}

    def add_candidate(doc_id_raw: Any, source: str, item: Optional[Dict[str, Any]] = None):
        try:
            doc_id = int(doc_id_raw)
        except (TypeError, ValueError):
            return
        item = dict(item or {})
        existing = pool.get(doc_id)
        if existing is None:
            original_rank = item.get("original_rank", base_rank.get(doc_id))
            try:
                original_rank = int(original_rank) if original_rank is not None else None
            except (TypeError, ValueError):
                original_rank = None
            existing = {
                "doc_id": doc_id,
                "query": item.get("query", ""),
                "queries": list(item.get("queries", []) or []),
                "query_count": int(item.get("query_count", 0) or 0),
                "plan_rank": item.get("plan_rank", 999999),
                "plan_score": float(item.get("plan_score", 0.0) or 0.0),
                "retrieval_score": float(item.get("retrieval_score", base_score.get(doc_id, 0.0)) or 0.0),
                "normalized_score": float(item.get("normalized_score", 0.0) or 0.0),
                "calibrated_score": float(item.get("calibrated_score", 0.0) or 0.0),
                "original_rank": original_rank,
                "query_overlap": float(item.get("query_overlap", 0.0) or 0.0),
                "candidate_sources": [],
                "schema_memory": item.get("schema_memory"),
                "top2_pair_verifier": item.get("top2_pair_verifier"),
                "bridge_slot_rescue": item.get("bridge_slot_rescue"),
                "relation_object_guard": item.get("relation_object_guard"),
                "answer_type_conflict_guard": item.get("answer_type_conflict_guard"),
                "pairwise_utility_score": item.get("pairwise_utility_score"),
                "pairwise_utility_margin": item.get("pairwise_utility_margin"),
            }
            pool[doc_id] = existing
        sources = existing.setdefault("candidate_sources", [])
        if source and source not in sources:
            sources.append(source)
        for candidate_source in item.get("candidate_sources", []) or []:
            if candidate_source and candidate_source not in sources:
                sources.append(candidate_source)
        if item.get("plan_score") is not None:
            existing["plan_score"] = max(
                float(existing.get("plan_score", 0.0) or 0.0),
                float(item.get("plan_score", 0.0) or 0.0),
            )
        if item.get("query_overlap") is not None:
            existing["query_overlap"] = max(
                float(existing.get("query_overlap", 0.0) or 0.0),
                float(item.get("query_overlap", 0.0) or 0.0),
            )
        if item.get("schema_memory") and not existing.get("schema_memory"):
            existing["schema_memory"] = item.get("schema_memory")
        for key in (
            "top2_pair_verifier",
            "bridge_slot_rescue",
            "relation_object_guard",
            "answer_type_conflict_guard",
            "pairwise_utility_score",
            "pairwise_utility_margin",
        ):
            if item.get(key) is not None:
                existing[key] = item.get(key)
        if item.get("query") and not existing.get("query"):
            existing["query"] = item.get("query")
        if item.get("queries"):
            for query in item.get("queries", []) or []:
                if query not in existing.setdefault("queries", []):
                    existing["queries"].append(query)
        existing["query_count"] = max(
            int(existing.get("query_count", 0) or 0),
            int(item.get("query_count", 0) or 0),
        )

    for doc_id in current_top:
        add_candidate(doc_id, "current_top")
    for rank, doc_id in enumerate(base_ids[:min(max_original_rank, len(base_ids))]):
        add_candidate(doc_id, "original_retrieval", {
            "doc_id": int(doc_id),
            "original_rank": rank,
            "plan_rank": 999999,
            "plan_score": 0.50 + 0.24 * _rank_prior(rank),
            "retrieval_score": base_score.get(int(doc_id), 0.0),
        })
    for item in ranked_candidates[:max_candidates]:
        add_candidate(item.get("doc_id"), "planned_retrieval", item)
    for item in utility_candidates[:max_candidates]:
        add_candidate(item.get("doc_id"), "pairwise_utility", item)
    for schema_rank, schema_item_raw in enumerate(schema_memory_candidates[:max_candidates]):
        schema_item = dict(schema_item_raw or {})
        try:
            schema_doc_id = int(schema_item.get("doc_id"))
        except (TypeError, ValueError):
            continue
        schema_score = float(schema_item.get("score", 0.0) or 0.0)
        add_candidate(schema_doc_id, "query_schema_memory", {
            "doc_id": schema_doc_id,
            "query": "query schema memory",
            "queries": ["query schema memory"],
            "query_count": 1,
            "plan_rank": schema_rank,
            "plan_score": 0.42 + 0.58 * _clamp(schema_score),
            "retrieval_score": base_score.get(schema_doc_id, 0.0),
            "normalized_score": _clamp(schema_score),
            "calibrated_score": _clamp(schema_score),
            "original_rank": base_rank.get(schema_doc_id, schema_item.get("original_rank")),
            "query_overlap": max(
                float(schema_item.get("query_overlap", 0.0) or 0.0),
                float(schema_item.get("context_overlap", 0.0) or 0.0),
            ),
            "schema_memory": schema_item,
        })

    profiled: List[Dict[str, Any]] = []
    for item in pool.values():
        doc_id = int(item["doc_id"])
        doc = str(doc_lookup(doc_id) or "")
        if not doc:
            continue
        title = _doc_title(doc)
        doc_tokens = _content_token_set(doc[:5000])
        title_tokens = _content_token_set(title)
        query_overlap = (
            len(question_tokens & doc_tokens) / max(1, len(question_tokens))
            if question_tokens else 0.0
        )
        title_anchor = (
            len(question_tokens & title_tokens) / max(1, min(len(question_tokens), len(title_tokens)))
            if question_tokens and title_tokens else 0.0
        )
        sentence_score = _support_set_best_sentence_score(question_tokens, doc, target_type, anchors)
        relation_score = _relation_cue_score(doc, relation_type)
        anchor_hits = _support_set_anchor_hits(doc, anchors)
        anchor_score = min(1.0, len(anchor_hits) / max(1, min(3, len(anchors)))) if anchors else 0.0
        answer_profile = _answer_type_value_profile(
            question,
            _item_query_text(item),
            doc,
            target_type,
        )
        answer_score = max(
            _answer_type_signal_score(doc[:3000], target_type),
            float(answer_profile.get("score", 0.0) or 0.0),
        )
        schema_item = item.get("schema_memory") or {}
        schema_score = float(schema_item.get("score", 0.0) or 0.0)
        schema_idf = float(schema_item.get("schema_idf", 0.0) or 0.0)
        schema_signal = 0.60 * _clamp(schema_score) + 0.40 * _clamp(schema_idf)
        pairwise_signal = _clamp(item.get("pairwise_utility_score", 0.0))
        bridge_signal = _clamp((item.get("bridge_slot_rescue") or {}).get("score", 0.0))
        rank_prior = _rank_prior(item.get("original_rank"))
        plan_quality = _clamp(float(item.get("plan_score", 0.0) or 0.0) / 1.20)
        source_set = set(item.get("candidate_sources", []) or [])
        source_signal = 0.0
        if "planned_retrieval" in source_set:
            source_signal += 0.35
        if "query_schema_memory" in source_set:
            source_signal += 0.25
        if "pairwise_utility" in source_set:
            source_signal += 0.20
        if "current_top" in source_set:
            source_signal += 0.20
        source_signal = _clamp(source_signal)
        support_set_bridge_slot = None
        missing_slot_gate_evaluated = bool(require_missing_slot_gate and doc_id not in old_top_set)
        schema_gate_ok = (
            "query_schema_memory" in source_set
            and schema_score >= 0.72
            and schema_idf >= 0.42
            and max(float(item.get("query_overlap", 0.0) or 0.0), query_overlap) >= 0.08
        )
        if missing_slot_gate_evaluated:
            support_set_bridge_slot = _bridge_slot_coverage_score(
                question=question,
                top1_doc=top1_doc,
                old_rank2_doc=old_rank2_doc,
                candidate_doc=doc,
                candidate_item=item,
                old_rank2_item=None,
                min_score=missing_slot_min_score,
                min_improvement=missing_slot_min_improvement,
                min_anchor_overlap=missing_slot_min_anchor_overlap,
                require_relation=missing_slot_require_relation,
                require_old_deficit=False,
                deficit_max_old_score=0.50,
                min_missing_gain=missing_slot_min_missing_gain,
                append_on_no_deficit=True,
            )
            bridge_signal = max(
                bridge_signal,
                _clamp(float(support_set_bridge_slot.get("score", 0.0) or 0.0)),
            )
        bridge_slot_gate_ok = bool((support_set_bridge_slot or {}).get("append_accepted", False))
        missing_slot_gate_passed = (
            not missing_slot_gate_evaluated
            or bridge_slot_gate_ok
            or schema_gate_ok
        )
        base_utility = (
            0.18 * query_overlap
            + 0.13 * title_anchor
            + 0.15 * sentence_score
            + 0.11 * relation_score
            + 0.10 * answer_score
            + 0.10 * plan_quality
            + 0.08 * schema_signal
            + 0.06 * pairwise_signal
            + 0.05 * bridge_signal
            + 0.06 * rank_prior
            + 0.04 * source_signal
        )
        profiled.append({
            **item,
            "title": title,
            "tokens": doc_tokens,
            "question_tokens": question_tokens & doc_tokens,
            "anchor_hits": anchor_hits,
            "answer_values": set(answer_profile.get("values") or []),
            "query_overlap": max(float(item.get("query_overlap", 0.0) or 0.0), query_overlap),
            "title_anchor": title_anchor,
            "sentence_score": sentence_score,
            "relation_score": relation_score,
            "anchor_score": anchor_score,
            "answer_score": answer_score,
            "schema_signal": schema_signal,
            "pairwise_signal": pairwise_signal,
            "bridge_signal": bridge_signal,
            "support_set_bridge_slot": support_set_bridge_slot,
            "missing_slot_gate_evaluated": missing_slot_gate_evaluated,
            "missing_slot_gate_passed": missing_slot_gate_passed,
            "missing_slot_schema_gate_ok": schema_gate_ok,
            "rank_prior": rank_prior,
            "plan_quality": plan_quality,
            "source_signal": source_signal,
            "base_utility": base_utility,
            "candidate_sources": sorted(source_set),
        })

    if len(profiled) <= target_k:
        info["reason"] = "not_enough_profiled_candidates"
        info["candidate_count"] = len(profiled)
        return current_top[:target_k], info, []

    by_id = {int(item["doc_id"]): item for item in profiled}
    protected_ids = [doc_id for doc_id in current_top[:protect_top_k] if doc_id in by_id]
    selected: List[Dict[str, Any]] = [by_id[doc_id] for doc_id in protected_ids]
    selected_ids = {int(item["doc_id"]) for item in selected}
    covered_question_tokens = set()
    covered_anchor_hits = set()
    selected_answer_values = set()
    for item in selected:
        covered_question_tokens.update(item.get("question_tokens", set()))
        covered_anchor_hits.update(item.get("anchor_hits", set()))
        selected_answer_values.update(item.get("answer_values", set()))

    original_replacements = sum(1 for doc_id in selected_ids if doc_id not in old_top_set)
    debug_candidates = []

    def coverage_preservation_gate(item: Dict[str, Any]) -> Dict[str, Any]:
        doc_id = int(item["doc_id"])
        result: Dict[str, Any] = {
            "evaluated": False,
            "passed": True,
            "reason": "not_applicable",
            "coverage_loss": 0.0,
            "anchor_loss": 0.0,
            "preserve_margin": 0.0,
            "old_competitor_doc_id": None,
            "old_competitor_title": "",
        }
        if not preserve_original_topk or doc_id in old_top_set:
            item["coverage_preservation"] = result
            return result
        unselected_old_items = [
            by_id[int(old_id)]
            for old_id in old_top
            if int(old_id) not in selected_ids and int(old_id) in by_id
        ]
        if not unselected_old_items:
            item["coverage_preservation"] = result
            return result

        old_competitor = min(
            unselected_old_items,
            key=lambda old_item: (
                float(old_item.get("base_utility", 0.0) or 0.0),
                int(base_rank.get(int(old_item["doc_id"]), 999999)),
            ),
        )
        candidate_question_tokens = set(item.get("question_tokens", set()))
        old_question_tokens = set(old_competitor.get("question_tokens", set()))
        candidate_question_union = covered_question_tokens | candidate_question_tokens
        old_question_union = covered_question_tokens | old_question_tokens
        question_denom = max(1, len(question_tokens))
        coverage_gain_vs_old = (
            len(candidate_question_union) - len(old_question_union)
        ) / question_denom
        coverage_loss = max(0.0, -coverage_gain_vs_old)

        candidate_anchor_hits = set(item.get("anchor_hits", set()))
        old_anchor_hits = set(old_competitor.get("anchor_hits", set()))
        if anchors:
            anchor_denom = max(1, min(3, len(anchors)))
            candidate_anchor_union = covered_anchor_hits | candidate_anchor_hits
            old_anchor_union = covered_anchor_hits | old_anchor_hits
            anchor_gain_vs_old = (
                len(candidate_anchor_union) - len(old_anchor_union)
            ) / anchor_denom
            anchor_loss = max(0.0, -anchor_gain_vs_old)
        else:
            anchor_gain_vs_old = 0.0
            anchor_loss = 0.0

        utility_margin = (
            float(item.get("base_utility", 0.0) or 0.0)
            - float(old_competitor.get("base_utility", 0.0) or 0.0)
        )
        bridge_gain = (
            float(item.get("bridge_signal", 0.0) or 0.0)
            - float(old_competitor.get("bridge_signal", 0.0) or 0.0)
        )
        schema_gain = (
            float(item.get("schema_signal", 0.0) or 0.0)
            - float(old_competitor.get("schema_signal", 0.0) or 0.0)
        )
        answer_gain = (
            float(item.get("answer_score", 0.0) or 0.0)
            - float(old_competitor.get("answer_score", 0.0) or 0.0)
        )
        preserve_margin = (
            utility_margin
            + 0.26 * coverage_gain_vs_old
            + 0.14 * anchor_gain_vs_old
            + 0.08 * bridge_gain
            + 0.05 * schema_gain
            + 0.04 * answer_gain
            - 0.20 * coverage_loss
            - 0.12 * anchor_loss
        )
        reject_reasons = []
        if coverage_loss > preserve_max_coverage_loss and preserve_margin < preserve_min_gain:
            reject_reasons.append("coverage_loss_without_gain")
        if preserve_margin < -preserve_min_gain:
            reject_reasons.append("negative_preserve_margin")
        result = {
            "evaluated": True,
            "passed": not reject_reasons,
            "reason": "passed" if not reject_reasons else reject_reasons[0],
            "reject_reasons": reject_reasons,
            "coverage_loss": round(float(coverage_loss), 6),
            "anchor_loss": round(float(anchor_loss), 6),
            "coverage_gain_vs_old": round(float(coverage_gain_vs_old), 6),
            "anchor_gain_vs_old": round(float(anchor_gain_vs_old), 6),
            "utility_margin": round(float(utility_margin), 6),
            "bridge_gain": round(float(bridge_gain), 6),
            "schema_gain": round(float(schema_gain), 6),
            "answer_gain": round(float(answer_gain), 6),
            "preserve_margin": round(float(preserve_margin), 6),
            "old_competitor_doc_id": int(old_competitor["doc_id"]),
            "old_competitor_title": old_competitor.get("title", ""),
        }
        item["coverage_preservation"] = result
        return result

    def candidate_incremental_score(item: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        doc_id = int(item["doc_id"])
        outside_old_top = doc_id not in old_top_set
        if outside_old_top and original_replacements >= max_replacements:
            return -1e9, {"blocked": 1.0}
        if outside_old_top and require_missing_slot_gate and not item.get("missing_slot_gate_passed", False):
            bridge_slot_gate = item.get("support_set_bridge_slot") or {}
            return -1e9, {
                "missing_slot_gate_rejected": 1.0,
                "bridge_slot_score": float(bridge_slot_gate.get("score", 0.0) or 0.0),
                "schema_gate_ok": 1.0 if item.get("missing_slot_schema_gate_ok") else 0.0,
            }
        if outside_old_top:
            coverage_gate = coverage_preservation_gate(item)
            if not coverage_gate.get("passed", True):
                return -1e9, {
                    "coverage_preservation_rejected": 1.0,
                    "coverage_loss": float(coverage_gate.get("coverage_loss", 0.0) or 0.0),
                    "anchor_loss": float(coverage_gate.get("anchor_loss", 0.0) or 0.0),
                    "preserve_margin": float(coverage_gate.get("preserve_margin", 0.0) or 0.0),
                    "old_competitor_doc_id": float(
                        coverage_gate.get("old_competitor_doc_id") or -1
                    ),
                }
        if (
            outside_old_top
            and float(item.get("base_utility", 0.0) or 0.0) < min_candidate_score
            and "current_top" not in set(item.get("candidate_sources", []) or [])
        ):
            return -1e9, {"below_min_candidate_score": 1.0}
        candidate_question_tokens = set(item.get("question_tokens", set()))
        coverage_gain = (
            len(candidate_question_tokens - covered_question_tokens) / max(1, len(question_tokens))
            if question_tokens else 0.0
        )
        candidate_anchor_hits = set(item.get("anchor_hits", set()))
        anchor_gain = (
            len(candidate_anchor_hits - covered_anchor_hits) / max(1, min(3, len(anchors)))
            if anchors else 0.0
        )
        answer_values = set(item.get("answer_values", set()))
        answer_gain = float(item.get("answer_score", 0.0) or 0.0)
        conflict_penalty = 0.0
        if selected_answer_values and answer_values:
            if selected_answer_values.isdisjoint(answer_values) and target_type in {"date", "number"}:
                conflict_penalty = 0.12
            else:
                answer_gain *= 0.35
        redundancy = 0.0
        for selected_item in selected:
            redundancy = max(
                redundancy,
                _support_set_jaccard(
                    set(item.get("tokens", set())),
                    set(selected_item.get("tokens", set())),
                ),
            )
        incremental = (
            float(item.get("base_utility", 0.0) or 0.0)
            + 0.28 * coverage_gain
            + 0.16 * anchor_gain
            + 0.08 * answer_gain
            + 0.04 * float(item.get("bridge_signal", 0.0) or 0.0)
            - redundancy_penalty * redundancy
            - conflict_penalty
        )
        parts = {
            "coverage_gain": coverage_gain,
            "anchor_gain": anchor_gain,
            "answer_gain": answer_gain,
            "redundancy": redundancy,
            "conflict_penalty": conflict_penalty,
            "base_utility": float(item.get("base_utility", 0.0) or 0.0),
        }
        return incremental, parts

    selectable = sorted(
        profiled,
        key=lambda item: (
            -float(item.get("base_utility", 0.0) or 0.0),
            item.get("original_rank") if item.get("original_rank") is not None else 999999,
        ),
    )[:max_candidates]

    while len(selected) < target_k:
        best_item = None
        best_score = -1e9
        best_parts: Dict[str, float] = {}
        for item in selectable:
            doc_id = int(item["doc_id"])
            if doc_id in selected_ids:
                continue
            score, parts = candidate_incremental_score(item)
            if score > best_score:
                best_item = item
                best_score = score
                best_parts = parts
        if best_item is None or best_score <= -1e8:
            for doc_id in current_top:
                if doc_id in selected_ids or doc_id not in by_id:
                    continue
                best_item = by_id[doc_id]
                best_score, best_parts = candidate_incremental_score(best_item)
                break
        if best_item is None:
            break
        selected.append(best_item)
        selected_ids.add(int(best_item["doc_id"]))
        covered_question_tokens.update(best_item.get("question_tokens", set()))
        covered_anchor_hits.update(best_item.get("anchor_hits", set()))
        selected_answer_values.update(best_item.get("answer_values", set()))
        if int(best_item["doc_id"]) not in old_top_set:
            original_replacements += 1
        best_item["support_set_incremental_score"] = best_score
        best_item["support_set_score_parts"] = best_parts

    for doc_id in current_top:
        if len(selected) >= target_k:
            break
        if doc_id in selected_ids or doc_id not in by_id:
            continue
        selected.append(by_id[doc_id])
        selected_ids.add(doc_id)

    selected_doc_ids = [int(item["doc_id"]) for item in selected[:target_k]]
    if len(selected_doc_ids) < target_k:
        info["reason"] = "could_not_fill_topk"
        info["candidate_count"] = len(profiled)
        return current_top[:target_k], info, []

    old_top_set = set(old_top)
    new_top_set = set(selected_doc_ids)
    replaced_doc_ids = [doc_id for doc_id in old_top if doc_id not in new_top_set]
    promoted_items = [
        item for item in selected[:target_k]
        if int(item["doc_id"]) not in old_top_set
    ]
    missing_slot_gate_reject_counts = Counter()
    missing_slot_gate_rejected_count = 0
    coverage_preservation_reject_counts = Counter()
    coverage_preservation_rejected_count = 0
    for item in profiled:
        if not item.get("missing_slot_gate_evaluated"):
            continue
        if item.get("missing_slot_gate_passed"):
            continue
        missing_slot_gate_rejected_count += 1
        bridge_slot_gate = item.get("support_set_bridge_slot") or {}
        missing_slot_gate_reject_counts.update(
            bridge_slot_gate.get("append_reject_reasons")
            or bridge_slot_gate.get("reject_reasons")
            or ["missing_slot_gate_rejected"]
        )
    for item in profiled:
        preservation = item.get("coverage_preservation") or {}
        if not preservation.get("evaluated"):
            continue
        if preservation.get("passed", True):
            continue
        coverage_preservation_rejected_count += 1
        coverage_preservation_reject_counts.update(
            preservation.get("reject_reasons")
            or [preservation.get("reason") or "coverage_preservation_rejected"]
        )
    for item in selectable[:min(12, len(selectable))]:
        score, parts = candidate_incremental_score(item)
        bridge_slot_gate = item.get("support_set_bridge_slot") or {}
        debug_candidates.append({
            "doc_id": int(item["doc_id"]),
            "title": item.get("title", ""),
            "sources": item.get("candidate_sources", []),
            "original_rank": item.get("original_rank"),
            "base_utility": round(float(item.get("base_utility", 0.0) or 0.0), 6),
            "incremental_score": round(float(score), 6) if score > -1e8 else None,
            "query_overlap": round(float(item.get("query_overlap", 0.0) or 0.0), 6),
            "sentence_score": round(float(item.get("sentence_score", 0.0) or 0.0), 6),
            "relation_score": round(float(item.get("relation_score", 0.0) or 0.0), 6),
            "answer_score": round(float(item.get("answer_score", 0.0) or 0.0), 6),
            "schema_signal": round(float(item.get("schema_signal", 0.0) or 0.0), 6),
            "missing_slot_gate_evaluated": bool(item.get("missing_slot_gate_evaluated", False)),
            "missing_slot_gate_passed": bool(item.get("missing_slot_gate_passed", False)),
            "missing_slot_schema_gate_ok": bool(item.get("missing_slot_schema_gate_ok", False)),
            "bridge_slot_score": round(float(bridge_slot_gate.get("score", 0.0) or 0.0), 6),
            "bridge_slot_reason": bridge_slot_gate.get("reason"),
            "coverage_preservation": item.get("coverage_preservation"),
            "parts": {k: round(float(v), 6) for k, v in parts.items()},
            "selected": int(item["doc_id"]) in new_top_set,
        })

    applied = selected_doc_ids != current_top[:target_k]
    info.update({
        "applied": bool(applied),
        "reason": "support_set_selected_new_topk" if applied else "current_topk_already_best",
        "candidate_count": len(profiled),
        "selected_doc_ids": selected_doc_ids,
        "old_top_doc_ids": old_top,
        "input_top_doc_ids": current_top[:target_k],
        "replaced_doc_ids": replaced_doc_ids,
        "replacement_count": len([doc_id for doc_id in selected_doc_ids if doc_id not in old_top_set]),
        "promoted_to_top2": len(set(selected_doc_ids[:min(2, len(selected_doc_ids))]) - set(old_top[:2])),
        "promoted_to_top5": len(set(selected_doc_ids[:min(5, len(selected_doc_ids))]) - old_top_set),
        "missing_slot_gate_rejected_count": int(missing_slot_gate_rejected_count),
        "missing_slot_gate_reject_reason_counts": dict(missing_slot_gate_reject_counts),
        "coverage_preservation_rejected_count": int(coverage_preservation_rejected_count),
        "coverage_preservation_reject_reason_counts": dict(coverage_preservation_reject_counts),
        "selected": [
            {
                "doc_id": int(item["doc_id"]),
                "title": item.get("title", ""),
                "sources": item.get("candidate_sources", []),
                "original_rank": item.get("original_rank"),
                "plan_rank": item.get("plan_rank"),
                "plan_score": round(float(item.get("plan_score", 0.0) or 0.0), 6),
                "base_utility": round(float(item.get("base_utility", 0.0) or 0.0), 6),
                "incremental_score": round(float(item.get("support_set_incremental_score", 0.0) or 0.0), 6),
                "query_overlap": round(float(item.get("query_overlap", 0.0) or 0.0), 6),
                "sentence_score": round(float(item.get("sentence_score", 0.0) or 0.0), 6),
                "relation_score": round(float(item.get("relation_score", 0.0) or 0.0), 6),
                "answer_score": round(float(item.get("answer_score", 0.0) or 0.0), 6),
                "schema_signal": round(float(item.get("schema_signal", 0.0) or 0.0), 6),
                "missing_slot_gate_evaluated": bool(item.get("missing_slot_gate_evaluated", False)),
                "missing_slot_gate_passed": bool(item.get("missing_slot_gate_passed", False)),
                "missing_slot_schema_gate_ok": bool(item.get("missing_slot_schema_gate_ok", False)),
                "bridge_slot_score": round(
                    float((item.get("support_set_bridge_slot") or {}).get("score", 0.0) or 0.0),
                    6,
                ),
                "bridge_slot_reason": (item.get("support_set_bridge_slot") or {}).get("reason"),
                "coverage_preservation": item.get("coverage_preservation"),
                "score_parts": {
                    k: round(float(v), 6)
                    for k, v in (item.get("support_set_score_parts") or {}).items()
                },
            }
            for item in selected[:target_k]
        ],
        "top_candidates": debug_candidates,
    })
    return selected_doc_ids, info, promoted_items


def _budgeted_evidence_subgraph_select(
    question: Any,
    base_ids: List[int],
    target_k: int,
    candidate_groups: List[Tuple[str, List[Dict[str, Any]]]],
    ranked_candidates: List[Dict[str, Any]],
    doc_lookup: Optional[Callable[[int], str]],
    max_append: int = 1,
    max_candidates: int = 48,
    min_utility: float = 0.48,
    redundancy_penalty: float = 0.16,
    require_bridge_or_gap: bool = True,
    intervention_gate: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Select append-only reader evidence with a budgeted subgraph objective.

    This keeps retrieval ranking stable. It only decides which extra passages
    should be appended to the reader by combining provenance, schema/relation
    signals, missing-slot support, answer-type compatibility, and marginal
    coverage under a small evidence budget.
    """
    max_append = max(0, int(max_append))
    max_candidates = max(1, int(max_candidates))
    min_utility = max(0.0, float(min_utility))
    redundancy_penalty = max(0.0, float(redundancy_penalty))
    require_bridge_or_gap = bool(require_bridge_or_gap)
    intervention_gate = dict(intervention_gate or {})
    info: Dict[str, Any] = {
        "enabled": True,
        "applied": False,
        "reason": "not_evaluated",
        "max_append": max_append,
        "max_candidates": max_candidates,
        "min_utility": round(float(min_utility), 6),
        "redundancy_penalty": round(float(redundancy_penalty), 6),
        "require_bridge_or_gap": require_bridge_or_gap,
        "candidate_count": 0,
        "eligible_count": 0,
        "selected_count": 0,
        "selected_doc_ids": [],
        "reject_reason_counts": {},
        "source_counts": {},
        "selected": [],
        "top_candidates": [],
    }
    if max_append <= 0:
        info["reason"] = "zero_append_budget"
        return [], info
    if doc_lookup is None:
        info["reason"] = "missing_doc_lookup"
        return [], info
    if not base_ids:
        info["reason"] = "missing_base_docs"
        return [], info

    base_visible = {int(doc_id) for doc_id in base_ids[:max(1, int(target_k))]}
    question_tokens = _content_token_set(question)
    target_type = infer_answer_target_type(str(question or ""))
    relation_type = _infer_relation_guard_type(question)
    anchors = _extract_entity_anchor_sets(question, limit=14)

    pool: Dict[int, Dict[str, Any]] = {}

    def add_item(source: str, raw_item: Dict[str, Any]) -> None:
        try:
            doc_id = int(raw_item.get("doc_id"))
        except (TypeError, ValueError):
            return
        if doc_id in base_visible:
            return
        item = pool.get(doc_id)
        if item is None:
            item = dict(raw_item)
            item["doc_id"] = doc_id
            if source == "query_schema_memory" and not item.get("schema_memory"):
                item["schema_memory"] = dict(raw_item)
            default_query = item.get("query") or source.replace("_", " ")
            item.setdefault("query", default_query)
            if not item.get("queries"):
                item["queries"] = [default_query] if default_query else []
            item.setdefault("query_count", 1 if default_query else 0)
            item.setdefault("plan_rank", int(item.get("rank", 999999) or 999999))
            item.setdefault("plan_score", float(item.get("score", 0.0) or 0.0))
            item.setdefault("retrieval_score", float(item.get("score", 0.0) or 0.0))
            item.setdefault("normalized_score", float(item.get("score", 0.0) or 0.0))
            item.setdefault("calibrated_score", float(item.get("score", 0.0) or 0.0))
            item.setdefault("candidate_source", source)
            item["candidate_sources"] = []
            pool[doc_id] = item
        else:
            if float(raw_item.get("plan_score", 0.0) or 0.0) > float(item.get("plan_score", 0.0) or 0.0):
                for key, value in raw_item.items():
                    if key != "candidate_sources":
                        item[key] = value
        sources = item.setdefault("candidate_sources", [])
        if source and source not in sources:
            sources.append(source)
        for candidate_source in raw_item.get("candidate_sources", []) or []:
            if candidate_source and candidate_source not in sources:
                sources.append(candidate_source)

    for source, items in candidate_groups:
        for item in items or []:
            add_item(source, item)

    if len(pool) < max_candidates:
        for item in ranked_candidates:
            if len(pool) >= max_candidates:
                break
            add_item("planned_retrieval", item)

    if not pool:
        info["reason"] = "no_append_candidates"
        return [], info

    visible_tokens = set()
    visible_anchor_hits = set()
    for doc_id in base_ids[:max(1, int(target_k))]:
        try:
            doc = str(doc_lookup(int(doc_id)) or "")
        except Exception:
            doc = ""
        visible_tokens.update(_content_token_set(doc[:5000]) & question_tokens)
        visible_anchor_hits.update(_support_set_anchor_hits(doc, anchors))

    profiled: List[Dict[str, Any]] = []
    reject_counts = Counter()
    source_counts = Counter()
    for item in pool.values():
        doc_id = int(item["doc_id"])
        doc = str(item.get("object_memory_doc") or doc_lookup(doc_id) or "")
        if not doc:
            reject_counts.update(["empty_doc"])
            continue
        source_set = set(item.get("candidate_sources", []) or [])
        source_counts.update(source_set)
        title = _doc_title(doc)
        doc_tokens = _content_token_set(doc[:5000])
        title_tokens = _content_token_set(title)
        candidate_question_tokens = doc_tokens & question_tokens
        question_overlap = (
            len(candidate_question_tokens) / max(1, len(question_tokens))
            if question_tokens else 0.0
        )
        query_overlap = max(
            question_overlap,
            float(item.get("query_overlap", 0.0) or 0.0),
        )
        title_anchor = (
            len(question_tokens & title_tokens) / max(1, min(len(question_tokens), len(title_tokens)))
            if question_tokens and title_tokens else 0.0
        )
        sentence_score = _support_set_best_sentence_score(
            question_tokens,
            doc,
            target_type,
            anchors,
        )
        relation_score = _relation_cue_score(doc, relation_type)
        anchor_hits = _support_set_anchor_hits(doc, anchors)
        anchor_score = (
            min(1.0, len(anchor_hits) / max(1, min(3, len(anchors))))
            if anchors else 0.0
        )
        answer_profile = _answer_type_value_profile(
            question,
            _item_query_text(item),
            doc,
            target_type,
        )
        answer_score = max(
            _answer_type_signal_score(doc[:3000], target_type),
            float(answer_profile.get("score", 0.0) or 0.0),
        )
        bridge_info = item.get("bridge_slot_rescue") or {}
        schema_info = item.get("schema_deficit_soft_append") or {}
        schema_rel = item.get("schema_reliability_gate") or {}
        schema_memory = item.get("schema_memory") or {}
        schema_score = max(
            float(schema_memory.get("score", 0.0) or 0.0),
            float(schema_info.get("schema_score", 0.0) or 0.0),
        )
        schema_idf = max(
            float(schema_memory.get("schema_idf", 0.0) or 0.0),
            float(schema_info.get("schema_idf", 0.0) or 0.0),
        )
        schema_reliability = float(schema_rel.get("score", 0.0) or 0.0)
        schema_signal = (
            0.45 * _clamp(schema_score)
            + 0.30 * _clamp(schema_idf)
            + 0.25 * _clamp(schema_reliability)
        )
        bridge_score = float(bridge_info.get("score", 0.0) or 0.0)
        bridge_margin = max(0.0, float(bridge_info.get("margin", 0.0) or 0.0))
        deficit_gate = bridge_info.get("deficit_gate") or {}
        missing_gain = max(
            float(deficit_gate.get("missing_support_gain", 0.0) or 0.0),
            float(schema_info.get("gain", 0.0) or 0.0),
        )
        plan_quality = _clamp(float(item.get("plan_score", 0.0) or 0.0) / 1.20)
        rank_prior = _rank_prior(item.get("original_rank"))
        provenance = 0.0
        if "object_memory" in source_set:
            provenance += 0.24
        if "bridge_slot" in source_set:
            provenance += 0.22
        if "schema_deficit" in source_set:
            provenance += 0.20
        if "planned_retrieval" in source_set:
            provenance += 0.12
        provenance = _clamp(provenance)
        bridge_or_gap = (
            bridge_score >= 0.42
            or bridge_margin >= 0.04
            or missing_gain >= 0.06
            or schema_signal >= 0.48
            or (relation_score >= 0.18 and sentence_score >= 0.24)
            or (anchor_score >= 0.34 and sentence_score >= 0.22)
            or (query_overlap >= 0.14 and answer_score >= 0.20)
            or "object_memory" in source_set
            or "bridge_slot" in source_set
            or "schema_deficit" in source_set
            or "query_schema_memory" in source_set
        )
        if require_bridge_or_gap and not bridge_or_gap:
            reject_counts.update(["no_bridge_or_deficit_signal"])
            continue

        base_utility = (
            0.16 * query_overlap
            + 0.11 * title_anchor
            + 0.15 * sentence_score
            + 0.12 * relation_score
            + 0.10 * answer_score
            + 0.11 * schema_signal
            + 0.10 * _clamp(bridge_score)
            + 0.08 * _clamp(missing_gain)
            + 0.06 * plan_quality
            + 0.05 * rank_prior
            + 0.06 * provenance
        )
        if bool(intervention_gate.get("long_context_shape", False)):
            base_utility -= 0.04 * (1.0 - max(sentence_score, answer_score))

        profiled.append({
            **item,
            "doc_id": doc_id,
            "doc_text": doc,
            "title": title,
            "tokens": doc_tokens,
            "question_tokens": candidate_question_tokens,
            "anchor_hits": set(anchor_hits),
            "answer_values": set(answer_profile.get("values") or []),
            "query_overlap": query_overlap,
            "title_anchor": title_anchor,
            "sentence_score": sentence_score,
            "relation_score": relation_score,
            "anchor_score": anchor_score,
            "answer_score": answer_score,
            "schema_signal": schema_signal,
            "bridge_signal": _clamp(bridge_score),
            "missing_gain": missing_gain,
            "plan_quality": plan_quality,
            "rank_prior": rank_prior,
            "provenance": provenance,
            "base_utility": base_utility,
            "candidate_sources": sorted(source_set),
        })

    profiled.sort(
        key=lambda item: (
            -float(item.get("base_utility", 0.0) or 0.0),
            item.get("original_rank") if item.get("original_rank") is not None else 999999,
        )
    )
    profiled = profiled[:max_candidates]
    info["candidate_count"] = len(profiled)
    info["source_counts"] = dict(source_counts)
    if not profiled:
        info["reason"] = "no_profiled_candidates"
        info["reject_reason_counts"] = dict(reject_counts)
        return [], info

    selected: List[Dict[str, Any]] = []
    selected_ids = set()
    covered_question_tokens = set(visible_tokens)
    covered_anchor_hits = set(visible_anchor_hits)
    selected_answer_values = set()

    def incremental_score(item: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        candidate_question_tokens = set(item.get("question_tokens", set()))
        candidate_anchor_hits = set(item.get("anchor_hits", set()))
        new_question = candidate_question_tokens - covered_question_tokens
        new_anchor = candidate_anchor_hits - covered_anchor_hits
        question_gain = len(new_question) / max(1, len(question_tokens))
        anchor_gain = (
            len(new_anchor) / max(1, min(3, len(anchors)))
            if anchors else 0.0
        )
        max_redundancy = 0.0
        candidate_tokens = set(item.get("tokens", set()))
        for other in selected:
            other_tokens = set(other.get("tokens", set()))
            if not candidate_tokens or not other_tokens:
                continue
            max_redundancy = max(
                max_redundancy,
                len(candidate_tokens & other_tokens) / max(1, min(len(candidate_tokens), len(other_tokens))),
            )
        if visible_tokens and candidate_question_tokens:
            max_redundancy = max(
                max_redundancy,
                len(candidate_question_tokens & visible_tokens) / max(1, len(candidate_question_tokens)),
            )
        conflict_penalty = 0.0
        answer_values = set(item.get("answer_values", set()))
        if target_type in {"date", "number"} and selected_answer_values and answer_values:
            if answer_values.isdisjoint(selected_answer_values):
                conflict_penalty = 0.10
        score = (
            float(item.get("base_utility", 0.0) or 0.0)
            + 0.18 * question_gain
            + 0.12 * anchor_gain
            - redundancy_penalty * max_redundancy
            - conflict_penalty
        )
        return score, {
            "question_gain": question_gain,
            "anchor_gain": anchor_gain,
            "redundancy": max_redundancy,
            "conflict_penalty": conflict_penalty,
        }

    debug = []
    while len(selected) < max_append:
        best_item = None
        best_score = -1e9
        best_parts: Dict[str, float] = {}
        for item in profiled:
            doc_id = int(item["doc_id"])
            if doc_id in selected_ids:
                continue
            score, parts = incremental_score(item)
            if score > best_score:
                best_item = item
                best_score = score
                best_parts = parts
        if best_item is None:
            break
        best_item["budgeted_subgraph_score"] = float(best_score)
        best_item["budgeted_subgraph_parts"] = best_parts
        accepted = best_score >= min_utility
        if len(debug) < 12:
            debug.append({
                "doc_id": int(best_item["doc_id"]),
                "title": best_item.get("title", ""),
                "sources": best_item.get("candidate_sources", []),
                "score": round(float(best_score), 6),
                "base_utility": round(float(best_item.get("base_utility", 0.0) or 0.0), 6),
                "question_gain": round(float(best_parts.get("question_gain", 0.0)), 6),
                "anchor_gain": round(float(best_parts.get("anchor_gain", 0.0)), 6),
                "redundancy": round(float(best_parts.get("redundancy", 0.0)), 6),
                "query_overlap": round(float(best_item.get("query_overlap", 0.0) or 0.0), 6),
                "schema_signal": round(float(best_item.get("schema_signal", 0.0) or 0.0), 6),
                "bridge_signal": round(float(best_item.get("bridge_signal", 0.0) or 0.0), 6),
                "missing_gain": round(float(best_item.get("missing_gain", 0.0) or 0.0), 6),
                "accepted": bool(accepted),
            })
        if not accepted:
            reject_counts.update(["below_min_utility"])
            break
        best_item["budgeted_subgraph_source"] = ",".join(
            str(source) for source in best_item.get("candidate_sources", []) if source
        )
        selected.append(best_item)
        selected_ids.add(int(best_item["doc_id"]))
        covered_question_tokens.update(best_item.get("question_tokens", set()))
        covered_anchor_hits.update(best_item.get("anchor_hits", set()))
        selected_answer_values.update(best_item.get("answer_values", set()))

    info.update({
        "applied": bool(selected),
        "reason": "selected_budgeted_evidence_subgraph" if selected else "no_candidate_above_min_utility",
        "eligible_count": len(profiled),
        "selected_count": len(selected),
        "selected_doc_ids": [int(item["doc_id"]) for item in selected],
        "reject_reason_counts": dict(reject_counts),
        "selected": [
            {
                "doc_id": int(item["doc_id"]),
                "title": item.get("title", ""),
                "sources": item.get("candidate_sources", []),
                "score": round(float(item.get("budgeted_subgraph_score", 0.0)), 6),
                "base_utility": round(float(item.get("base_utility", 0.0) or 0.0), 6),
                "parts": {
                    key: round(float(value), 6)
                    for key, value in (item.get("budgeted_subgraph_parts") or {}).items()
                },
            }
            for item in selected
        ],
        "top_candidates": debug,
    })
    return selected, info


def _answer_type_signal_score(text: Any, target_type: str) -> float:
    text = str(text or "")
    target_type = str(target_type or "factoid")
    if not text:
        return 0.0
    lowered = text.lower()
    if target_type in {"date", "number"}:
        return 1.0 if DATE_SPAN_RE.search(text) else 0.0
    if target_type == "occupation":
        return 1.0 if re.search(
            r"\b(?:actor|actress|singer|writer|author|director|producer|politician|"
            r"footballer|player|coach|composer|scientist|professor|artist|journalist|"
            r"lawyer|physician|businessperson|entrepreneur|profession|occupation)\b",
            lowered,
        ) else 0.0
    if target_type in {"person", "location", "organization", "work"}:
        return min(1.0, len(TITLE_CASE_RE.findall(text)) / 3.0)
    return 0.0


def _bridge_slot_coverage_score(
    question: Any,
    top1_doc: Any,
    old_rank2_doc: Any,
    candidate_doc: Any,
    candidate_item: Dict[str, Any],
    old_rank2_item: Optional[Dict[str, Any]],
    min_score: float,
    min_improvement: float,
    min_anchor_overlap: float,
    require_relation: bool = True,
    require_old_deficit: bool = True,
    deficit_max_old_score: float = 0.50,
    min_missing_gain: float = 0.15,
    append_on_no_deficit: bool = True,
) -> Dict[str, Any]:
    """Score whether a rejected pairwise candidate fills the missing bridge slot.

    This rescue path is intentionally more structural than the pair verifier: it
    looks for entity anchors, relation-specific evidence, and answer-type cues in
    the candidate passage. It can recover candidates with weak lexical overlap
    when they complete a relation slot that the current top1+rank2 pair lacks.
    """
    planned_query = _item_query_text(candidate_item)
    relation_type = _infer_relation_guard_type(f"{question} {planned_query}")
    target_type = infer_answer_target_type(str(question or ""))
    info = {
        "enabled": True,
        "accepted": False,
        "reason": "not_evaluated",
        "reject_reasons": [],
        "relation_type": relation_type,
        "answer_target_type": target_type,
        "append_accepted": False,
        "append_reject_reasons": [],
        "replace_requires_old_deficit": bool(require_old_deficit),
    }
    if not top1_doc or not old_rank2_doc or not candidate_doc:
        info.update({
            "reason": "missing_pair_text",
            "reject_reasons": ["bridge_slot_missing_text"],
        })
        return info
    if require_relation and relation_type == "generic":
        info.update({
            "reason": "generic_relation_not_rescued",
            "reject_reasons": ["bridge_slot_generic_relation"],
        })
        return info

    anchors = _extract_entity_anchor_sets(question, planned_query, top1_doc, limit=16)
    anchor_min = _clamp(min_anchor_overlap)
    candidate_anchor_overlap = _sentence_anchor_overlap(candidate_doc, anchors)
    old_anchor_overlap = _sentence_anchor_overlap(old_rank2_doc, anchors)
    top1_anchor_overlap = _sentence_anchor_overlap(top1_doc, anchors)

    candidate_relation_score = _relation_cue_score(candidate_doc, relation_type)
    old_relation_score = _relation_cue_score(old_rank2_doc, relation_type)
    candidate_entity = _entity_conditioned_relation_evidence(
        candidate_doc,
        relation_type,
        anchors,
        anchor_min_overlap=max(0.20, min(anchor_min, 0.80)),
    )
    old_entity = _entity_conditioned_relation_evidence(
        old_rank2_doc,
        relation_type,
        anchors,
        anchor_min_overlap=max(0.20, min(anchor_min, 0.80)),
    )
    candidate_objects = list(candidate_entity.get("objects") or [])
    if not candidate_objects:
        candidate_objects = _extract_relation_objects(candidate_doc, relation_type)
    old_objects = list(old_entity.get("objects") or [])
    if not old_objects:
        old_objects = _extract_relation_objects(old_rank2_doc, relation_type)

    query_tokens = _content_token_set(planned_query)
    question_tokens = _content_token_set(question)
    candidate_tokens = _content_token_set(candidate_doc)
    top1_tokens = _content_token_set(top1_doc)
    old_tokens = _content_token_set(old_rank2_doc)
    slot_tokens = query_tokens or question_tokens
    missing_from_top1 = slot_tokens - top1_tokens
    if slot_tokens:
        candidate_query_support = len(slot_tokens & candidate_tokens) / len(slot_tokens)
        old_query_support = len(slot_tokens & old_tokens) / len(slot_tokens)
    else:
        candidate_query_support = 0.0
        old_query_support = 0.0
    if missing_from_top1:
        candidate_missing_support = len(missing_from_top1 & candidate_tokens) / len(missing_from_top1)
        old_missing_support = len(missing_from_top1 & old_tokens) / len(missing_from_top1)
    else:
        candidate_missing_support = candidate_query_support
        old_missing_support = old_query_support

    candidate_answer_type_score = _answer_type_signal_score(candidate_doc, target_type)
    old_answer_type_score = _answer_type_signal_score(old_rank2_doc, target_type)
    candidate_entity_score = float(candidate_entity.get("score", 0.0) or 0.0)
    old_entity_score = float(old_entity.get("score", 0.0) or 0.0)
    candidate_object_score = min(1.0, len(candidate_objects) / 2.0)
    old_object_score = min(1.0, len(old_objects) / 2.0)
    missing_support_gain = candidate_missing_support - old_missing_support

    if top1_tokens and candidate_tokens:
        redundancy = len(top1_tokens & candidate_tokens) / max(1, len(top1_tokens | candidate_tokens))
    else:
        redundancy = 0.0
    plan_quality = _clamp(float(candidate_item.get("plan_score", 0.0) or 0.0) / 1.20)
    rank_prior = _rank_prior(candidate_item.get("original_rank"))

    candidate_score = (
        0.24 * candidate_entity_score
        + 0.18 * candidate_relation_score
        + 0.17 * candidate_missing_support
        + 0.14 * candidate_query_support
        + 0.12 * candidate_anchor_overlap
        + 0.07 * candidate_answer_type_score
        + 0.05 * candidate_object_score
        + 0.05 * plan_quality
        + 0.03 * rank_prior
        - 0.05 * redundancy
    )
    old_score = (
        0.24 * old_entity_score
        + 0.18 * old_relation_score
        + 0.17 * old_missing_support
        + 0.14 * old_query_support
        + 0.12 * old_anchor_overlap
        + 0.07 * old_answer_type_score
        + 0.05 * old_object_score
    )
    margin = candidate_score - old_score

    deficit_max_old_score = max(0.0, float(deficit_max_old_score))
    min_missing_gain = max(0.0, float(min_missing_gain))
    old_slot_sufficient = (
        old_score >= deficit_max_old_score
        or (
            old_missing_support >= 0.55
            and (old_anchor_overlap >= anchor_min or old_entity_score >= 0.24)
        )
        or (
            old_query_support >= 0.58
            and (old_relation_score >= 0.25 or old_entity_score >= 0.24)
        )
        or (
            old_answer_type_score >= 0.80
            and old_missing_support >= 0.40
            and (old_anchor_overlap >= anchor_min or old_relation_score >= 0.25)
        )
    )
    old_pair_deficit = not old_slot_sufficient

    structural_reject_reasons = []
    if candidate_score < min_score:
        structural_reject_reasons.append("bridge_slot_low_score")
    if margin < min_improvement:
        structural_reject_reasons.append("bridge_slot_low_improvement")
    if anchors and candidate_anchor_overlap < anchor_min and candidate_entity_score < min_score:
        structural_reject_reasons.append("bridge_slot_weak_anchor_evidence")
    if (
        missing_support_gain < min_missing_gain
        and candidate_entity_score < old_entity_score + 0.10
        and candidate_relation_score < old_relation_score + 0.10
    ):
        structural_reject_reasons.append("bridge_slot_low_missing_gain")
    if (
        require_relation
        and relation_type != "generic"
        and candidate_relation_score < 0.20
        and candidate_entity_score < 0.20
    ):
        structural_reject_reasons.append("bridge_slot_missing_relation_cue")
    if candidate_objects and old_objects and set(candidate_objects).isdisjoint(set(old_objects)):


        if candidate_entity_score < old_entity_score + max(0.10, min_improvement):
            structural_reject_reasons.append("bridge_slot_object_conflict")

    reject_reasons = list(structural_reject_reasons)
    if require_old_deficit and not old_pair_deficit:
        reject_reasons.append("bridge_slot_old_top2_sufficient")

    append_reject_reasons = list(structural_reject_reasons)
    if not append_on_no_deficit:
        append_reject_reasons.append("bridge_slot_append_disabled")
    accepted = not reject_reasons
    append_accepted = not append_reject_reasons
    info.update({
        "accepted": accepted,
        "append_accepted": append_accepted,
        "reason": (
            "bridge_slot_replace_passed"
            if accepted else
            "bridge_slot_append_only"
            if append_accepted else
            "bridge_slot_coverage_rejected"
        ),
        "reject_reasons": reject_reasons,
        "append_reject_reasons": append_reject_reasons,
        "score": round(float(candidate_score), 6),
        "old_score": round(float(old_score), 6),
        "margin": round(float(margin), 6),
        "min_score": round(float(min_score), 6),
        "min_improvement": round(float(min_improvement), 6),
        "min_anchor_overlap": round(float(anchor_min), 6),
        "deficit_gate": {
            "require_old_deficit": bool(require_old_deficit),
            "old_pair_deficit": bool(old_pair_deficit),
            "old_slot_sufficient": bool(old_slot_sufficient),
            "deficit_max_old_score": round(float(deficit_max_old_score), 6),
            "min_missing_gain": round(float(min_missing_gain), 6),
            "missing_support_gain": round(float(missing_support_gain), 6),
            "append_on_no_deficit": bool(append_on_no_deficit),
        },
        "anchors": [_format_anchor(anchor) for anchor in anchors[:8]],
        "candidate": {
            "anchor_overlap": round(float(candidate_anchor_overlap), 6),
            "relation_score": round(float(candidate_relation_score), 6),
            "entity_score": round(float(candidate_entity_score), 6),
            "query_support": round(float(candidate_query_support), 6),
            "missing_slot_support": round(float(candidate_missing_support), 6),
            "answer_type_score": round(float(candidate_answer_type_score), 6),
            "objects": candidate_objects[:6],
        },
        "old_rank2": {
            "anchor_overlap": round(float(old_anchor_overlap), 6),
            "relation_score": round(float(old_relation_score), 6),
            "entity_score": round(float(old_entity_score), 6),
            "query_support": round(float(old_query_support), 6),
            "missing_slot_support": round(float(old_missing_support), 6),
            "answer_type_score": round(float(old_answer_type_score), 6),
            "objects": old_objects[:6],
        },
        "top1_anchor_overlap": round(float(top1_anchor_overlap), 6),
        "redundancy": round(float(redundancy), 6),
        "plan_quality": round(float(plan_quality), 6),
        "rank_prior": round(float(rank_prior), 6),
    })
    return info


def _object_consistent_evidence_memory_gate(
    question: Any,
    planned_query: Any,
    top1_doc: Any,
    old_rank2_doc: Any,
    candidate_doc: Any,
    bridge_slot_info: Optional[Dict[str, Any]],
    min_score: float = 0.42,
    max_sentences: int = 2,
    max_objects: int = 1,
    max_chars: int = 720,
) -> Dict[str, Any]:
    """Build a compact evidence memory only when its answer object is coherent.

    Bridge-slot retrieval is useful when it discovers a missing hop, but adding
    the whole passage can inject several plausible final answers. This gate keeps
    only the sentence(s) that carry the relation/object signal and rejects memory
    candidates that introduce conflicting answer objects.
    """
    relation_type = str((bridge_slot_info or {}).get("relation_type") or "").strip()
    if not relation_type:
        relation_type = _infer_relation_guard_type(f"{question} {planned_query}")
    target_type = str((bridge_slot_info or {}).get("answer_target_type") or "").strip()
    if not target_type:
        target_type = infer_answer_target_type(str(question or ""))

    info = {
        "enabled": True,
        "accepted": False,
        "reason": "not_evaluated",
        "reject_reasons": [],
        "relation_type": relation_type,
        "answer_target_type": target_type,
        "memory_doc": "",
        "memory_title": "",
        "selected_sentences": [],
        "objects": [],
        "conflicting_object_count": 0,
    }
    if not candidate_doc:
        info.update({
            "reason": "missing_candidate_text",
            "reject_reasons": ["object_memory_missing_candidate_text"],
        })
        return info

    min_score = max(0.0, float(min_score))
    max_sentences = max(1, int(max_sentences))
    max_objects = max(0, int(max_objects))
    max_chars = max(120, int(max_chars))
    anchors = _extract_entity_anchor_sets(question, planned_query, limit=12)
    patterns = RELATION_CUE_PATTERNS.get(relation_type) or []
    query_tokens = _content_token_set(planned_query) or _content_token_set(question)
    candidate_parts = str(candidate_doc or "").split("\n", 1)
    candidate_body = candidate_parts[1] if len(candidate_parts) > 1 else candidate_parts[0]

    def sentence_query_overlap(sentence: Any) -> float:
        if not query_tokens:
            return 0.0
        sentence_tokens = _content_token_set(sentence)
        if not sentence_tokens:
            return 0.0
        return _clamp(len(query_tokens & sentence_tokens) / len(query_tokens))

    def extract_sentence_objects(sentence: str) -> List[str]:
        if relation_type != "generic":
            relation_objects = _extract_relation_objects_from_sentences(
                [sentence],
                relation_type,
                limit=8,
                anchors=anchors,
            )
            if relation_objects:
                return relation_objects
        if target_type in {"date", "number"}:
            spans = DATE_SPAN_RE.findall(sentence)
        elif target_type in {"person", "location", "organization", "work", "factoid"}:
            spans = TITLE_CASE_RE.findall(sentence)
        else:
            spans = []
        objects = []
        for span in spans:
            normalized = _normalize_object_span(span)
            if not normalized or normalized in objects:
                continue
            if normalized in {"he", "she", "it", "him", "her", "his", "hers", "its"}:
                continue
            if _object_matches_anchor(normalized, anchors):
                continue
            objects.append(normalized)
            if len(objects) >= 8:
                break
        return objects

    scored_sentences = []
    for sentence in _split_sentences(candidate_body, max_sentences=80):
        if len(sentence) < 12:
            continue
        relation_hit = 0.0
        if patterns:
            relation_hit = 1.0 if any(pattern.search(sentence) for pattern in patterns) else 0.0
        elif relation_type == "generic":
            relation_hit = 0.25
        anchor_overlap = _sentence_anchor_overlap(sentence, anchors)
        query_overlap = sentence_query_overlap(sentence)
        answer_type_score = _answer_type_signal_score(sentence, target_type)
        objects = extract_sentence_objects(sentence)
        object_signal = min(1.0, len(objects) / max(1.0, float(max_objects or 1)))
        score = (
            0.28 * relation_hit
            + 0.22 * anchor_overlap
            + 0.20 * query_overlap
            + 0.18 * answer_type_score
            + 0.12 * object_signal
        )
        if relation_type != "generic" and not relation_hit and answer_type_score < 0.5:
            score *= 0.60
        scored_sentences.append({
            "sentence": sentence,
            "score": _clamp(score, 0.0, 2.0),
            "relation_hit": relation_hit,
            "anchor_overlap": anchor_overlap,
            "query_overlap": query_overlap,
            "answer_type_score": answer_type_score,
            "objects": objects,
        })

    scored_sentences.sort(
        key=lambda item: (
            -float(item.get("score", 0.0)),
            -float(item.get("relation_hit", 0.0)),
            -float(item.get("anchor_overlap", 0.0)),
            -float(item.get("query_overlap", 0.0)),
        )
    )
    selected = []
    seen_sentences = set()
    char_budget = max_chars
    for item in scored_sentences:
        if float(item.get("score", 0.0)) < min_score:
            continue
        sentence = str(item.get("sentence", "")).strip()
        key = _normalize_query(sentence)
        if not key or key in seen_sentences:
            continue
        if selected and len(sentence) + sum(len(s["sentence"]) for s in selected) > char_budget:
            continue
        selected.append(item)
        seen_sentences.add(key)
        if len(selected) >= max_sentences:
            break

    if not selected:
        info.update({
            "reason": "no_high_signal_sentence",
            "reject_reasons": ["object_memory_no_high_signal_sentence"],
            "best_sentence_score": (
                round(float(scored_sentences[0]["score"]), 6)
                if scored_sentences else 0.0
            ),
            "min_score": round(float(min_score), 6),
        })
        return info

    objects = []
    for item in selected:
        for obj in item.get("objects", []) or []:
            if obj not in objects:
                objects.append(obj)
    if target_type in {"date", "number", "person", "location", "organization", "work", "factoid"}:
        conflicting_count = max(0, len(objects) - max_objects)
    else:
        conflicting_count = 0

    bridge_candidate = (bridge_slot_info or {}).get("candidate") or {}
    bridge_old = (bridge_slot_info or {}).get("old_rank2") or {}
    old_objects = set(bridge_old.get("objects") or [])
    object_set = set(objects)
    old_score = float(bridge_old.get("entity_score", 0.0) or 0.0) + float(
        bridge_old.get("relation_score", 0.0) or 0.0
    )
    candidate_score = float(bridge_candidate.get("entity_score", 0.0) or 0.0) + float(
        bridge_candidate.get("relation_score", 0.0) or 0.0
    )
    if old_objects and object_set and old_objects.isdisjoint(object_set) and old_score >= 0.60:
        if candidate_score < old_score + 0.20:
            conflicting_count = max(conflicting_count, 1)

    reject_reasons = []
    if max_objects > 0 and conflicting_count > 0:
        reject_reasons.append("object_memory_conflicting_answer_objects")
    if relation_type != "generic" and not any(float(item.get("relation_hit", 0.0)) > 0 for item in selected):
        reject_reasons.append("object_memory_missing_relation_sentence")
    if anchors and max(float(item.get("anchor_overlap", 0.0)) for item in selected) < 0.20:
        reject_reasons.append("object_memory_weak_anchor_sentence")
    if target_type in {"date", "number", "person", "location", "organization", "work"} and not objects:
        reject_reasons.append("object_memory_missing_answer_object")

    if reject_reasons:
        info.update({
            "reason": "object_memory_rejected",
            "reject_reasons": reject_reasons,
            "selected_sentences": [
                {
                    "sentence": item["sentence"][:240],
                    "score": round(float(item["score"]), 6),
                    "objects": item.get("objects", [])[:6],
                }
                for item in selected[:max_sentences]
            ],
            "objects": objects[:8],
            "conflicting_object_count": conflicting_count,
            "min_score": round(float(min_score), 6),
        })
        return info

    source_title = _doc_title(candidate_doc)
    memory_title = f"Object-consistent evidence memory: {source_title}" if source_title else "Object-consistent evidence memory"
    body_sentences = [item["sentence"].strip() for item in selected]
    memory_doc = f"{memory_title}\n" + " ".join(body_sentences)
    if len(memory_doc) > max_chars:
        memory_doc = memory_doc[:max_chars].rsplit(" ", 1)[0].strip()

    info.update({
        "accepted": True,
        "reason": "object_memory_accepted",
        "memory_doc": memory_doc,
        "memory_title": memory_title,
        "selected_sentences": [
            {
                "sentence": item["sentence"][:240],
                "score": round(float(item["score"]), 6),
                "relation_hit": round(float(item["relation_hit"]), 6),
                "anchor_overlap": round(float(item["anchor_overlap"]), 6),
                "query_overlap": round(float(item["query_overlap"]), 6),
                "answer_type_score": round(float(item["answer_type_score"]), 6),
                "objects": item.get("objects", [])[:6],
            }
            for item in selected
        ],
        "objects": objects[:8],
        "conflicting_object_count": conflicting_count,
        "sentence_count": len(selected),
        "max_objects": max_objects,
        "min_score": round(float(min_score), 6),
        "max_chars": max_chars,
    })
    return info


def _clean_query(text: Any) -> str:
    query = re.sub(r"\s+", " ", str(text or "")).strip(" \t\n\r\"'")
    query = re.sub(r"^(?:query|search query|subquery)\s*\d*\s*:\s*", "", query, flags=re.IGNORECASE)
    return query[:240]


def _extract_json_object(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    text = text.strip()
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {}
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def parse_planned_queries(text: str, max_queries: int = 3) -> List[str]:
    data = _extract_json_object(text)
    candidates = []
    if isinstance(data.get("queries"), list):
        candidates.extend(data["queries"])
    if isinstance(data.get("search_queries"), list):
        candidates.extend(data["search_queries"])
    if isinstance(data.get("subqueries"), list):
        candidates.extend(data["subqueries"])

    if not candidates:
        for line in str(text or "").splitlines():
            cleaned = _clean_query(re.sub(r"^[\-\*\d\.\)\s]+", "", line))
            if cleaned and "{" not in cleaned and "}" not in cleaned:
                candidates.append(cleaned)

    output = []
    seen = set()
    for item in candidates:
        if isinstance(item, dict):
            item = item.get("query") or item.get("search_query") or item.get("text") or ""
        cleaned = _clean_query(item)
        norm = _normalize_query(cleaned)
        if not cleaned or len(norm.split()) < 2 or norm in seen:
            continue
        seen.add(norm)
        output.append(cleaned)
        if len(output) >= max_queries:
            break
    return output


def build_planned_retrieval_prompt(
    question: str,
    docs: List[str],
    max_titles: int = 5,
    max_snippets: int = 3,
    missing_instruction: str = "",
) -> List[Dict[str, str]]:
    titles = []
    snippets = []
    for doc in docs[:max_titles]:
        title = _doc_title(doc)
        if title:
            titles.append(title)
        if len(snippets) < max_snippets:
            snippet = _doc_snippet(doc)
            if snippet:
                snippets.append(f"{title}: {snippet}")

    user_lines = [
        "Original question:",
        question,
        "",
        "Current top evidence titles:",
        "; ".join(titles) if titles else "None",
    ]
    if snippets:
        user_lines.extend(["", "Current evidence snippets:"])
        user_lines.extend(snippets)
    if missing_instruction:
        user_lines.extend(["", "Verifier says the evidence is still missing:"])
        user_lines.append(missing_instruction)
    user_lines.extend([
        "",
        "Return retrieval-only search queries that could find the missing bridge/support passages.",
        "Do not answer the question.",
        "Treat current evidence as tentative; do not trust an unrelated title just because it is currently top-ranked.",
        "Prefer queries that bind one original question entity to one missing relation or bridge slot.",
        "Avoid drifting to a different entity, city, country, date, race, work, or organization unless the evidence explicitly supports that bridge.",
        'JSON schema: {"queries": ["query 1", "query 2"], "rationale": "short retrieval plan"}',
    ])

    return [
        {
            "role": "system",
            "content": (
                "You are a skeptical retrieval query planner for multi-hop QA. "
                "Generate short, specific search queries for missing supporting evidence, "
                "while avoiding distractor evidence from the current retrieval results. "
                "Never provide the final answer."
            ),
        },
        {"role": "user", "content": "\n".join(user_lines)},
    ]


def build_evidence_verification_prompt(
    question: str,
    docs: List[str],
    max_titles: int = 5,
    max_snippets: int = 5,
) -> List[Dict[str, str]]:
    titles = []
    snippets = []
    for doc in docs[:max_titles]:
        title = _doc_title(doc)
        if title:
            titles.append(title)
        if len(snippets) < max_snippets:
            snippet = _doc_snippet(doc)
            if snippet:
                snippets.append(f"{title}: {snippet}")

    user_lines = [
        "Question:",
        question,
        "",
        "Current evidence titles:",
        "; ".join(titles) if titles else "None",
    ]
    if snippets:
        user_lines.extend(["", "Current evidence snippets:"])
        user_lines.extend(snippets)
    user_lines.extend([
        "",
        "Decide whether this evidence is sufficient to answer the question.",
        "Mark sufficient=true only if the evidence directly supports every required hop and the final answer type.",
        "If a top-ranked title appears unrelated or leads to a different entity/path, list it as a rejected anchor.",
        "If insufficient, describe the missing bridge/support slots and give retrieval-only search queries.",
        'JSON schema: {"sufficient": true/false, "missing": "short reason", "missing_slots": ["slot"], "uncertain_anchors": ["anchor"], "rejected_anchors": ["anchor"], "queries": ["query 1", "query 2"]}',
    ])

    return [
        {
            "role": "system",
            "content": (
                "You are a skeptical evidence verifier for multi-hop retrieval. "
                "You do not answer the question. You only judge whether the retrieved evidence "
                "is enough, identify missing or misleading evidence, and propose missing retrieval targets."
            ),
        },
        {"role": "user", "content": "\n".join(user_lines)},
    ]


def parse_evidence_verdict(text: str, max_queries: int = 3) -> Dict[str, Any]:
    data = _extract_json_object(text)
    sufficient_raw = data.get("sufficient", False)
    if isinstance(sufficient_raw, str):
        sufficient = sufficient_raw.strip().lower() in {"true", "yes", "sufficient", "1"}
    else:
        sufficient = bool(sufficient_raw)

    missing = data.get("missing") or data.get("reason") or ""
    missing_slots = data.get("missing_slots") if isinstance(data.get("missing_slots"), list) else []
    uncertain_anchors = data.get("uncertain_anchors") if isinstance(data.get("uncertain_anchors"), list) else []
    rejected_anchors = data.get("rejected_anchors") if isinstance(data.get("rejected_anchors"), list) else []
    queries = []
    if isinstance(data.get("queries"), list):
        queries.extend(data["queries"])
    if isinstance(data.get("search_queries"), list):
        queries.extend(data["search_queries"])
    parsed_queries = []
    seen = set()
    for query in queries:
        if isinstance(query, dict):
            query = query.get("query") or query.get("search_query") or query.get("text") or ""
        cleaned = _clean_query(query)
        norm = _normalize_query(cleaned)
        if not cleaned or len(norm.split()) < 2 or norm in seen:
            continue
        seen.add(norm)
        parsed_queries.append(cleaned)
        if len(parsed_queries) >= max_queries:
            break

    return {
        "sufficient": sufficient,
        "missing": str(missing)[:300],
        "missing_slots": [str(item)[:160] for item in missing_slots[:6]],
        "uncertain_anchors": [str(item)[:120] for item in uncertain_anchors[:6]],
        "rejected_anchors": [str(item)[:120] for item in rejected_anchors[:6]],
        "queries": parsed_queries,
        "raw_response_preview": str(text or "")[:500],
    }


def build_residual_deficit_queries(
    question: str,
    verdict: Optional[Dict[str, Any]] = None,
    docs: Optional[List[str]] = None,
    max_queries: int = 2,
) -> List[str]:
    """Build deterministic fallback queries for unresolved evidence slots.

    The LLM planner can repeat itself after several closed-loop rounds. This
    residual planner uses the verifier's missing-slot state and stable question
    anchors to create one or two new retrieval-only queries without editing the
    retrieval ranking directly.
    """
    verdict = verdict or {}
    docs = docs or []
    max_queries = max(0, int(max_queries))
    if max_queries <= 0:
        return []

    question_text = re.sub(r"\s+", " ", str(question or "")).strip()
    if not question_text:
        return []

    rejected_anchor_text = " ".join(str(item) for item in verdict.get("rejected_anchors", []) or [])
    rejected_tokens = set(TOKEN_RE.findall(rejected_anchor_text.lower()))
    question_entities = []
    seen_entities = set()
    for match in TITLE_CASE_RE.findall(question_text):
        entity = re.sub(r"\s+", " ", match).strip(" ,.;:?")
        entity_norm = entity.lower()
        if (
            len(entity) < 2
            or entity_norm in seen_entities
            or entity_norm in {"what", "where", "when", "which", "who", "how"}
        ):
            continue
        entity_tokens = set(TOKEN_RE.findall(entity_norm))
        if entity_tokens and entity_tokens <= rejected_tokens:
            continue
        seen_entities.add(entity_norm)
        question_entities.append(entity)
        if len(question_entities) >= 3:
            break

    if not question_entities:
        content_tokens = [
            token
            for token in TOKEN_RE.findall(question_text.lower())
            if len(token) > 3 and token not in CONTENT_STOPWORDS
        ]
        if content_tokens:
            question_entities.append(" ".join(content_tokens[:4]))

    slot_texts = []
    for key in ("missing_slots", "missing", "uncertain_anchors"):
        values = verdict.get(key)
        if isinstance(values, list):
            slot_texts.extend(str(value) for value in values if value)
        elif values:
            slot_texts.append(str(values))
    if not slot_texts:
        verifier_queries = verdict.get("queries", []) or []
        out = []
        seen = set()
        if isinstance(verifier_queries, list):
            for query in verifier_queries:
                cleaned = _clean_query(str(query or ""))
                norm = _normalize_query(cleaned)
                if not cleaned or len(norm.split()) < 2 or norm in seen:
                    continue
                seen.add(norm)
                out.append(cleaned)
                if len(out) >= max_queries:
                    break
        return out

    doc_titles = []
    for doc in docs[:5]:
        title = _doc_title(doc)
        if title:
            doc_titles.append(title)

    candidates = []
    primary_anchor = question_entities[0] if question_entities else ""
    secondary_anchor = question_entities[1] if len(question_entities) > 1 else ""
    for slot in slot_texts[:4]:
        slot_clean = re.sub(r"\s+", " ", str(slot or "")).strip(" ,.;:")
        if not slot_clean:
            continue
        if primary_anchor:
            candidates.append(f"{primary_anchor} {slot_clean}")
        if primary_anchor and secondary_anchor:
            candidates.append(f"{primary_anchor} {secondary_anchor} {slot_clean}")
        candidates.append(f"{question_text} {slot_clean}")

    for title in doc_titles[:2]:
        if primary_anchor and title.lower() != primary_anchor.lower():
            candidates.append(f"{primary_anchor} {title} missing supporting evidence")

    out = []
    seen = set()
    for candidate in candidates:
        cleaned = _clean_query(candidate)
        norm = _normalize_query(cleaned)
        if not cleaned or len(norm.split()) < 2 or norm in seen:
            continue
        seen.add(norm)
        out.append(cleaned)
        if len(out) >= max_queries:
            break
    return out


def _titlecase_entity_count(text: str) -> int:
    entities = []
    for match in TITLE_CASE_RE.findall(str(text or "")):
        cleaned = match.strip()
        if len(cleaned) < 2:
            continue
        if cleaned.lower() in {"what", "where", "when", "which", "who", "how"}:
            continue
        entities.append(cleaned)
    return len(entities)


def _doc_title_empty_ratio(docs: List[str], max_docs: int = 5) -> float:
    if not docs:
        return 0.0
    checked_docs = docs[:max(1, max_docs)]
    empty_count = 0
    for doc in checked_docs:
        if not _doc_title(doc):
            empty_count += 1
    return empty_count / max(1, len(checked_docs))


def route_planned_retrieval(
    question: str,
    docs: List[str],
    bridge_cues: bool = False,
    route_mode: str = "adaptive",
    base_max_rounds: int = 5,
    base_max_queries: int = 3,
    base_max_promotions: int = 2,
    base_protect_top_k: int = 1,
    base_plan_top_k: int = 20,
    base_min_score: float = 0.18,
) -> Dict[str, Any]:
    """Choose a retrieval budget without using dataset names or labels.

    The router is deliberately deterministic: it treats closed-loop retrieval
    as a scarce intervention for questions that look like they need missing
    bridge evidence, and it protects single-hop or long-answer questions from
    broad query expansion.
    """
    mode = str(route_mode or "off").strip().lower()
    base_max_rounds = max(0, int(base_max_rounds))
    base_max_queries = max(1, int(base_max_queries))
    base_max_promotions = max(0, int(base_max_promotions))
    base_protect_top_k = max(0, int(base_protect_top_k))
    base_plan_top_k = max(1, int(base_plan_top_k))
    base_min_score = float(base_min_score)
    answer_target_type = infer_answer_target_type(question)
    question_text = re.sub(r"\s+", " ", str(question or "")).strip()
    question_lower = question_text.lower()
    question_tokens = TOKEN_RE.findall(question_lower)
    entity_count = _titlecase_entity_count(question_text)
    family_or_role_chain = bool(FAMILY_OR_ROLE_CHAIN_RE.search(question_text))
    relative_clause = bool(RELATIVE_CLAUSE_RE.search(question_text))
    comparison = bool(COMPARISON_RE.search(question_text))
    simple_attribute = bool(SIMPLE_ATTRIBUTE_RE.search(question_text))
    long_answer = bool(LONG_ANSWER_RE.search(question_text))
    possessive_count = len(re.findall(r"(?:'s|’s)\b", question_text))
    avg_doc_chars = 0.0
    if docs:
        avg_doc_chars = sum(len(str(doc or "")) for doc in docs[:5]) / max(1, min(5, len(docs)))
    empty_title_ratio = _doc_title_empty_ratio(docs)
    long_context_shape = (empty_title_ratio >= 0.60 and avg_doc_chars >= 700) or avg_doc_chars >= 2400

    bridge_strength = 0
    if bridge_cues:
        bridge_strength += 2
    if family_or_role_chain:
        bridge_strength += 2
    if comparison:
        bridge_strength += 2
    if relative_clause and entity_count >= 1:
        bridge_strength += 1
    if possessive_count >= 2:
        bridge_strength += 2
    elif possessive_count == 1 and not simple_attribute:
        bridge_strength += 1
    if entity_count >= 2 and re.search(r"\b(?:of|from|by|in|for|with|before|after|located|based)\b", question_lower):
        bridge_strength += 1

    route = {
        "role": "router",
        "enabled": mode != "off",
        "route_mode": mode,
        "question_type": "off",
        "profile": "legacy",
        "reason": "routing_disabled",
        "allow_retrieval": True,
        "budget_policy": "adaptive_query_budget",
        "selector_policy": "adaptive_query_threshold",
        "budget_action": "closed_loop_until_sufficient",
        "bridge_cues": bool(bridge_cues),
        "bridge_strength": bridge_strength,
        "answer_target_type": answer_target_type,
        "entity_count": entity_count,
        "question_token_count": len(question_tokens),
        "avg_top_doc_chars": round(avg_doc_chars, 2),
        "empty_title_ratio": round(empty_title_ratio, 3),
        "long_context_shape": bool(long_context_shape),
        "signals": {
            "family_or_role_chain": family_or_role_chain,
            "relative_clause": relative_clause,
            "comparison": comparison,
            "simple_attribute": simple_attribute,
            "long_answer": long_answer,
            "possessive_count": possessive_count,
        },
        "effective_max_rounds": base_max_rounds,
        "effective_max_queries": base_max_queries,
        "effective_max_promotions": base_max_promotions,
        "effective_protect_top_k": base_protect_top_k,
        "effective_plan_top_k": base_plan_top_k,
        "effective_min_score": base_min_score,
    }
    if mode == "off":
        return route

    if long_context_shape:
        route.update({
            "question_type": "long_context_evidence",
            "profile": "long_context_evidence_only",
            "reason": "long_context_evidence_shape",
            "allow_retrieval": False,
            "budget_action": "early_stop_no_retrieval",
            "effective_max_rounds": 0,
            "effective_max_queries": 0,
            "effective_max_promotions": 0,
            "effective_protect_top_k": max(base_protect_top_k, 5),
            "effective_min_score": max(base_min_score, 0.40),
        })
    elif bridge_strength >= 2:
        route.update({
            "question_type": "bridge_or_compositional",
            "profile": "full_closed_loop",
            "reason": "bridge_or_composition_signals",
            "budget_action": "closed_loop_until_sufficient",
        })
    elif simple_attribute:
        route.update({
            "question_type": "single_hop_attribute",
            "profile": "verifier_only",
            "reason": "single_hop_attribute_protected",
            "allow_retrieval": False,
            "budget_action": "early_stop_no_retrieval",
            "effective_max_rounds": 0,
            "effective_max_promotions": 0,
            "effective_protect_top_k": max(base_protect_top_k, 2),
        })
    elif long_answer:
        route.update({
            "question_type": "long_answer_or_narrative",
            "profile": "verifier_only",
            "reason": "long_answer_expansion_risk",
            "allow_retrieval": False,
            "budget_action": "early_stop_no_retrieval",
            "effective_max_rounds": 0,
            "effective_max_promotions": 0,
            "effective_protect_top_k": max(base_protect_top_k, 2),
        })
    elif answer_target_type in {"date", "number"}:
        route.update({
            "question_type": "numeric_or_temporal",
            "profile": "conservative_closed_loop",
            "reason": "numeric_temporal_precision_risk",
            "budget_action": "closed_loop_until_sufficient",
            "effective_max_rounds": min(base_max_rounds, 1),
            "effective_max_queries": min(base_max_queries, 2),
            "effective_max_promotions": min(base_max_promotions, 1),
            "effective_protect_top_k": max(base_protect_top_k, 2),
            "effective_min_score": max(base_min_score, 0.30),
        })
    elif avg_doc_chars >= 2800 and bridge_strength == 0:
        route.update({
            "question_type": "long_context_factoid",
            "profile": "verifier_only",
            "reason": "long_context_without_bridge_signals",
            "allow_retrieval": False,
            "budget_action": "early_stop_no_retrieval",
            "effective_max_rounds": 0,
            "effective_max_promotions": 0,
            "effective_protect_top_k": max(base_protect_top_k, 2),
        })
    else:
        route.update({
            "question_type": "generic_factoid",
            "profile": "conservative_closed_loop",
            "reason": "generic_low_bridge_uncertainty",
            "budget_action": "closed_loop_until_sufficient",
            "effective_max_rounds": min(base_max_rounds, 2),
            "effective_max_queries": min(base_max_queries, 2),
            "effective_max_promotions": min(base_max_promotions, 1),
            "effective_protect_top_k": max(base_protect_top_k, 2),
            "effective_min_score": max(base_min_score, 0.28),
        })
    return route


class QuestionRouterAgent:
    role = "router"

    def __init__(
        self,
        route_mode: str = "adaptive",
        base_max_rounds: int = 5,
        base_max_queries: int = 3,
        base_max_promotions: int = 2,
        base_protect_top_k: int = 1,
        base_plan_top_k: int = 20,
        base_min_score: float = 0.18,
        qa_top_k: int = 5,
    ):
        self.route_mode = route_mode
        self.base_max_rounds = base_max_rounds
        self.base_max_queries = base_max_queries
        self.base_max_promotions = base_max_promotions
        self.base_protect_top_k = base_protect_top_k
        self.base_plan_top_k = base_plan_top_k
        self.base_min_score = base_min_score
        self.qa_top_k = qa_top_k

    def run(self, question: str, docs: List[str], bridge_cues: bool = False) -> Dict[str, Any]:
        return route_planned_retrieval(
            question=question,
            docs=docs,
            bridge_cues=bridge_cues,
            route_mode=self.route_mode,
            base_max_rounds=self.base_max_rounds,
            base_max_queries=self.base_max_queries,
            base_max_promotions=self.base_max_promotions,
            base_protect_top_k=self.base_protect_top_k,
            base_plan_top_k=self.base_plan_top_k,
            base_min_score=self.base_min_score,
        )


class PlannerAgent:
    role = "planner"

    def __init__(self, llm_model, max_queries: int = 3, max_snippets: int = 3, max_tokens: int = 160):
        self.llm_model = llm_model
        self.max_queries = max_queries
        self.max_snippets = max_snippets
        self.max_tokens = max_tokens

    def run(self, question: str, docs: List[str], missing_instruction: str = "") -> Dict[str, Any]:
        messages = build_planned_retrieval_prompt(
            question=question,
            docs=docs,
            max_titles=len(docs),
            max_snippets=self.max_snippets,
            missing_instruction=missing_instruction,
        )
        try:
            result = self.llm_model.infer(messages, max_completion_tokens=self.max_tokens)
        except TypeError:
            result = self.llm_model.infer(messages)
        response, metadata = unpack_llm_result(result)
        queries = parse_planned_queries(response, max_queries=self.max_queries)
        return {
            "role": self.role,
            "queries": queries,
            "raw_response_preview": str(response)[:500],
            "metadata": metadata,
        }


class RetrieverAgent:
    role = "retriever"

    def __init__(self, retrieval_system):
        self.retrieval_system = retrieval_system
        if not hasattr(retrieval_system, "_query_embedding_lock"):
            retrieval_system._query_embedding_lock = threading.Lock()
        self.embedding_lock = retrieval_system._query_embedding_lock

    def prepare_queries(self, queries: List[str]) -> None:
        system = self.retrieval_system
        clean_queries = [str(query).strip() for query in queries if str(query).strip()]
        if not clean_queries:
            return

        with self.embedding_lock:
            system.get_query_embeddings(clean_queries)

    def run(self, query: str, ensure_embeddings: bool = True) -> Dict[str, Any]:
        system = self.retrieval_system
        if ensure_embeddings:
            self.prepare_queries([query])
        fact_scores = system.get_fact_scores(query)
        top_k_fact_indices, top_k_facts, rerank_log = system.rerank_facts(query, fact_scores)
        candidate_fact_count = len((rerank_log or {}).get("facts_before_rerank", []))
        if len(top_k_facts) == 0:
            doc_ids, doc_scores = system.dense_passage_retrieval(query)
            mode = "dense"
        else:
            doc_ids, doc_scores = system.graph_search_with_fact_entities(
                query=query,
                link_top_k=system.global_config.linking_top_k,
                query_fact_scores=fact_scores,
                top_k_facts=top_k_facts,
                top_k_fact_indices=top_k_fact_indices,
                passage_node_weight=system.global_config.passage_node_weight,
            )
            mode = "graph"
        return {
            "role": self.role,
            "query": query,
            "mode": mode,
            "doc_ids": doc_ids,
            "doc_scores": doc_scores,
            "top_fact_count": len(top_k_facts),
            "candidate_fact_count": candidate_fact_count,
            "fallback_to_dense": mode == "dense",
        }


class VerifierAgent:
    role = "verifier"

    def __init__(self, llm_model, max_queries: int = 3, max_snippets: int = 3, max_tokens: int = 160):
        self.llm_model = llm_model
        self.max_queries = max_queries
        self.max_snippets = max_snippets
        self.max_tokens = max_tokens

    def run(self, question: str, docs: List[str]) -> Dict[str, Any]:
        messages = build_evidence_verification_prompt(
            question=question,
            docs=docs,
            max_titles=len(docs),
            max_snippets=self.max_snippets,
        )
        try:
            try:
                result = self.llm_model.infer(messages, max_completion_tokens=self.max_tokens)
            except TypeError:
                result = self.llm_model.infer(messages)
        except Exception as exc:
            message = str(exc)
            if "content_filter" not in message and "BadRequest" not in message and "400" not in message:
                raise
            return {
                "sufficient": False,
                "missing": "verifier_skipped_due_to_provider_content_filter",
                "missing_slots": [],
                "uncertain_anchors": [],
                "rejected_anchors": [],
                "queries": [],
                "raw_response_preview": message[:500],
                "role": self.role,
                "metadata": {"error": "content_filter_fallback"},
                "fallback": "content_filter",
            }
        response, metadata = unpack_llm_result(result)
        verdict = parse_evidence_verdict(response, max_queries=self.max_queries)
        verdict["role"] = self.role
        verdict["metadata"] = metadata
        return verdict


def route_risk_aware_merge_policy(
    route_info: Dict[str, Any],
    requested_merge_mode: str,
    protect_top_k: int,
    max_promotions: int,
    qa_top_k: int = 5,
) -> Dict[str, Any]:
    requested = str(requested_merge_mode or "append_only").strip().lower()
    question_type = str(route_info.get("question_type", "unknown") or "unknown")
    profile = str(route_info.get("profile", "unknown") or "unknown")
    bridge_strength = int(route_info.get("bridge_strength", 0) or 0)

    policy = {
        "requested_merge_mode": requested,
        "effective_merge_mode": requested,
        "intervention_risk": "manual",
        "reason": "manual_merge_mode",
        "protect_top_k": max(0, int(protect_top_k)),
        "max_promotions": max(0, int(max_promotions)),
        "can_change_top2": False,
    }
    bridge_ready = (
        question_type == "bridge_or_compositional"
        and profile == "full_closed_loop"
        and bridge_strength >= 2
    )
    if requested == "guarded_promote" and bridge_ready:
        policy.update({
            "intervention_risk": "medium",
            "reason": "manual_guarded_promote_bridge_state",
            "protect_top_k": min(max(2, int(protect_top_k)), max(1, int(qa_top_k))),
            "max_promotions": max(1, int(max_promotions)),
        })
        return policy
    if requested not in {"risk_aware", "risk_aware_guarded"}:
        return policy

    policy.update({
        "intervention_risk": "high",
        "effective_merge_mode": "append_only",
        "reason": f"risk_aware_default_append_only_for_{question_type}",
        "protect_top_k": min(max(2, int(protect_top_k)), max(1, int(qa_top_k))),
        "max_promotions": max(1, int(max_promotions)),
    })

    if bridge_ready:
        policy.update({
            "intervention_risk": "medium",
            "effective_merge_mode": "guarded_promote",
            "reason": "bridge_compositional_allows_guarded_top5_promotion",
            "protect_top_k": min(max(2, int(protect_top_k)), max(1, int(qa_top_k))),
            "max_promotions": max(1, int(max_promotions)),
        })
    elif question_type in {"single_hop_attribute", "long_answer_or_narrative", "long_context_evidence", "long_context_factoid"}:
        policy.update({
            "intervention_risk": "very_high",
            "reason": f"{question_type}_uses_append_only_to_avoid_rank_noise",
        })
    elif question_type == "numeric_or_temporal":
        policy.update({
            "intervention_risk": "high",
            "reason": "numeric_temporal_uses_append_only_for_precision",
        })
    elif question_type == "generic_factoid":
        policy.update({
            "intervention_risk": "high",
            "reason": "generic_factoid_uses_append_only_until_top2_policy_exists",
        })
    return policy


def route_context_state_intervention_gate(
    route_info: Dict[str, Any],
    merge_policy: Dict[str, Any],
    gate_mode: str = "off",
    min_bridge_strength: int = 2,
    long_context_avg_doc_chars: float = 1200.0,
    long_context_empty_title_ratio: float = 0.50,
    long_context_min_bridge_strength: int = 4,
) -> Dict[str, Any]:
    """Decide whether planned retrieval may alter the top-2 evidence slots.

    The gate is query-level and dataset-agnostic. It keeps the broader top-5
    expansion path available while freezing rank-2 when the current evidence
    state looks long-context or answer-generation oriented.
    """
    mode = str(gate_mode or "off").strip().lower()
    if mode in {"false", "none", "disabled"}:
        mode = "off"
    if mode not in {"off", "context_state"}:
        mode = "off"

    question_type = str(route_info.get("question_type", "unknown") or "unknown")
    profile = str(route_info.get("profile", "unknown") or "unknown")
    signals = route_info.get("signals") or {}
    try:
        bridge_strength = int(route_info.get("bridge_strength", 0) or 0)
    except (TypeError, ValueError):
        bridge_strength = 0
    try:
        avg_doc_chars = float(route_info.get("avg_top_doc_chars", 0.0) or 0.0)
    except (TypeError, ValueError):
        avg_doc_chars = 0.0
    try:
        empty_title_ratio = float(route_info.get("empty_title_ratio", 0.0) or 0.0)
    except (TypeError, ValueError):
        empty_title_ratio = 0.0

    effective_merge_mode = str(merge_policy.get("effective_merge_mode", "") or "").strip().lower()
    intervention_risk = str(merge_policy.get("intervention_risk", "manual") or "manual")
    base_top2_allowed = (
        effective_merge_mode == "guarded_promote"
        and question_type == "bridge_or_compositional"
        and profile == "full_closed_loop"
        and intervention_risk == "medium"
    )
    top5_allowed = effective_merge_mode in {"guarded_promote", "replace_topk"}

    gate = {
        "mode": mode,
        "enabled": mode != "off",
        "top2_allowed": bool(base_top2_allowed),
        "top5_allowed": bool(top5_allowed),
        "rank_intervention": "top2" if base_top2_allowed else "top5_or_append_only",
        "reason": "gate_disabled" if mode == "off" else "base_policy_allows_top2",
        "question_type": question_type,
        "profile": profile,
        "intervention_risk": intervention_risk,
        "bridge_strength": bridge_strength,
        "avg_top_doc_chars": round(avg_doc_chars, 2),
        "empty_title_ratio": round(empty_title_ratio, 3),
        "long_context_shape": bool(route_info.get("long_context_shape", False)),
        "long_answer_signal": bool(signals.get("long_answer", False)),
        "thresholds": {
            "min_bridge_strength": max(0, int(min_bridge_strength)),
            "long_context_avg_doc_chars": float(long_context_avg_doc_chars),
            "long_context_empty_title_ratio": _clamp(long_context_empty_title_ratio),
            "long_context_min_bridge_strength": max(0, int(long_context_min_bridge_strength)),
        },
    }
    if mode == "off":
        return gate
    if not base_top2_allowed:
        gate.update({
            "top2_allowed": False,
            "rank_intervention": "top5_or_append_only",
            "reason": "base_policy_disallows_top2",
        })
        return gate

    min_bridge_strength = max(0, int(min_bridge_strength))
    long_context_min_bridge_strength = max(0, int(long_context_min_bridge_strength))
    long_context_empty_title_ratio = _clamp(long_context_empty_title_ratio)
    long_context_avg_doc_chars = max(0.0, float(long_context_avg_doc_chars))

    if bool(route_info.get("long_context_shape", False)):
        gate.update({
            "top2_allowed": False,
            "rank_intervention": "top5_only",
            "reason": "long_context_shape_freezes_top2",
        })
    elif (
        bool(signals.get("long_answer", False))
        and avg_doc_chars >= long_context_avg_doc_chars
        and bridge_strength < long_context_min_bridge_strength
    ):
        gate.update({
            "top2_allowed": False,
            "rank_intervention": "top5_only",
            "reason": "long_answer_long_evidence_freezes_top2",
        })
    elif bridge_strength < min_bridge_strength:
        gate.update({
            "top2_allowed": False,
            "rank_intervention": "top5_only",
            "reason": "weak_bridge_state_freezes_top2",
        })
    elif (
        avg_doc_chars >= long_context_avg_doc_chars
        and empty_title_ratio >= long_context_empty_title_ratio
        and bridge_strength < long_context_min_bridge_strength
    ):
        gate.update({
            "top2_allowed": False,
            "rank_intervention": "top5_only",
            "reason": "long_evidence_low_title_structure_freezes_top2",
        })
    elif (
        avg_doc_chars >= 1.6 * long_context_avg_doc_chars
        and bridge_strength < long_context_min_bridge_strength
    ):
        gate.update({
            "top2_allowed": False,
            "rank_intervention": "top5_only",
            "reason": "very_long_evidence_freezes_top2",
        })
    else:
        gate.update({
            "top2_allowed": True,
            "rank_intervention": "top2",
            "reason": "bridge_state_allows_top2",
        })
    return gate


def route_pairwise_utility_source_policy(
    route_info: Dict[str, Any],
    intervention_gate: Dict[str, Any],
    configured_allowed_sources: str = "planned_and_original",
    source_policy: str = "fixed",
    adaptive_min_bridge_strength: int = 3,
    adaptive_max_avg_doc_chars: float = 900.0,
    adaptive_allow_long_answer: bool = False,
) -> Dict[str, Any]:
    """Select which candidate sources may compete in pairwise rank-2 search.

    The conservative default only lets candidates supported by both planned
    retrieval and the original retrieval list change rank 2. Adaptive mode
    selectively recovers pure original-retrieval candidates when the query state
    looks like a compact, high-confidence bridge question.
    """
    mode = str(source_policy or "fixed").strip().lower()
    if mode in {"off", "none", "disabled"}:
        mode = "fixed"
    if mode not in {"fixed", "adaptive", "selective"}:
        mode = "fixed"

    base_sources = _normalize_pairwise_utility_sources(
        configured_allowed_sources,
        default="planned_and_original",
    )
    signals = route_info.get("signals") or {}
    question_type = str(route_info.get("question_type", "unknown") or "unknown")
    profile = str(route_info.get("profile", "unknown") or "unknown")
    try:
        bridge_strength = int(route_info.get("bridge_strength", 0) or 0)
    except (TypeError, ValueError):
        bridge_strength = 0
    try:
        avg_doc_chars = float(route_info.get("avg_top_doc_chars", 0.0) or 0.0)
    except (TypeError, ValueError):
        avg_doc_chars = 0.0

    adaptive_min_bridge_strength = max(0, int(adaptive_min_bridge_strength))
    adaptive_max_avg_doc_chars = max(0.0, float(adaptive_max_avg_doc_chars))
    long_answer_signal = bool(signals.get("long_answer", False))
    long_context_state = (
        bool(route_info.get("long_context_shape", False))
        or question_type in {"long_answer_or_narrative", "long_context_evidence", "long_context_factoid"}
        or (long_answer_signal and not bool(adaptive_allow_long_answer))
    )

    policy = {
        "mode": mode,
        "configured_allowed_sources": base_sources,
        "effective_allowed_sources": ",".join(base_sources),
        "effective_allowed_source_list": base_sources,
        "reason": "fixed_config",
        "bridge_strength": bridge_strength,
        "avg_top_doc_chars": round(avg_doc_chars, 2),
        "long_context_state": bool(long_context_state),
        "long_answer_signal": bool(long_answer_signal),
        "thresholds": {
            "adaptive_min_bridge_strength": adaptive_min_bridge_strength,
            "adaptive_max_avg_doc_chars": adaptive_max_avg_doc_chars,
            "adaptive_allow_long_answer": bool(adaptive_allow_long_answer),
        },
    }
    if mode == "fixed":
        return policy
    if not bool(intervention_gate.get("top2_allowed", False)):
        policy["reason"] = "top2_gate_disallows_loose_recovery"
        return policy
    if str(intervention_gate.get("reason", "")) != "bridge_state_allows_top2":
        policy["reason"] = "intervention_gate_not_bridge_state"
        return policy
    if question_type != "bridge_or_compositional" or profile != "full_closed_loop":
        policy["reason"] = "not_compact_bridge_closed_loop"
        return policy
    if bridge_strength < adaptive_min_bridge_strength:
        policy["reason"] = "bridge_strength_below_loose_recovery_threshold"
        return policy
    if long_context_state:
        policy["reason"] = "long_context_or_long_answer_keeps_strict_sources"
        return policy
    if adaptive_max_avg_doc_chars > 0.0 and avg_doc_chars > adaptive_max_avg_doc_chars:
        policy["reason"] = "evidence_too_long_for_original_retrieval_recovery"
        return policy

    recovered_sources = sorted(set(base_sources) | {"original_retrieval"})
    policy.update({
        "effective_allowed_sources": ",".join(recovered_sources),
        "effective_allowed_source_list": recovered_sources,
        "reason": "selective_original_retrieval_recovery_allowed",
    })
    return policy


class MergerAgent:
    role = "merger"

    def __init__(
        self,
        target_k: int = 5,
        protect_top_k: int = 1,
        max_promotions: int = 2,
        plan_top_k: int = 20,
        min_plan_score: float = 0.18,
        merge_mode: str = "guarded_promote",
        question: str = "",
        guarded_min_score: float = 0.82,
        guarded_max_original_rank: int = 80,
        guarded_max_plan_rank: int = 8,
        guarded_min_query_overlap: float = 0.08,
        top2_override_enabled: bool = False,
        top2_override_min_score: float = 0.96,
        top2_override_max_original_rank: int = 30,
        top2_override_max_plan_rank: int = 2,
        top2_override_min_query_overlap: float = 0.18,
        top2_override_min_query_hits: int = 1,
        top2_pair_verifier_enabled: bool = False,
        top2_pair_verifier_mode: str = "hard",
        top2_pair_verifier_score_weight: float = 0.12,
        top2_pair_verifier_min_margin: float = 0.04,
        top2_pair_verifier_max_coverage_loss: float = 0.02,
        top2_pair_verifier_min_candidate_query_support: float = 0.45,
        relation_object_guard_enabled: bool = False,
        relation_object_guard_min_score: float = 0.55,
        relation_object_guard_min_margin: float = 0.08,
        entity_conditioned_relation_guard_enabled: bool = False,
        entity_relation_anchor_min_overlap: float = 0.50,
        intervention_gate: Optional[Dict[str, Any]] = None,
        pairwise_utility_search_enabled: bool = False,
        pairwise_utility_max_original_rank: int = 30,
        pairwise_utility_max_candidates: int = 32,
        pairwise_utility_min_margin: float = 0.12,
        pairwise_utility_min_pair_score: float = 0.65,
        pairwise_utility_min_candidate_support: float = 0.45,
        pairwise_utility_allowed_sources: str = "planned_and_original",
        pairwise_utility_original_fallback_only: bool = True,
        query_schema_memory_intervention_policy: str = "risk_aware",
        query_schema_memory_top2_min_score: float = 0.66,
        query_schema_memory_top2_min_schema_idf: float = 0.32,
        query_schema_memory_top2_min_overlap: float = 0.10,
        query_schema_memory_top2_max_schema_frequency: int = 80,
        query_schema_memory_top2_max_original_rank: int = 24,
        query_schema_memory_top2_min_bridge_strength: int = 3,
        schema_deficit_promotion_enabled: bool = False,
        schema_deficit_min_score: float = 0.62,
        schema_deficit_min_gain: float = 0.10,
        schema_deficit_min_schema_score: float = 0.68,
        schema_deficit_min_schema_idf: float = 0.30,
        schema_deficit_max_original_rank: int = 60,
        schema_deficit_require_bridge_constraint: bool = True,
        schema_deficit_min_bridge_score: float = 0.54,
        schema_deficit_min_bridge_margin: float = 0.08,
        schema_deficit_min_bridge_missing_gain: float = 0.10,
        schema_deficit_min_bridge_anchor_overlap: float = 0.24,
        schema_deficit_soft_append_enabled: bool = True,
        schema_deficit_soft_append_min_score: float = 0.48,
        schema_deficit_soft_append_min_schema_score: float = 0.58,
        schema_deficit_soft_append_min_schema_idf: float = 0.24,
        schema_deficit_soft_append_min_query_support: float = 0.34,
        schema_deficit_soft_append_max_original_rank: int = 80,
        schema_reliability_gate_enabled: bool = False,
        schema_reliability_min_score: float = 0.62,
        schema_reliability_rescue_min_score: float = 0.76,
        schema_reliability_min_anchor_overlap: float = 0.08,
        schema_reliability_max_schema_frequency: int = 80,
        schema_reliability_numeric_requires_value: bool = True,
        schema_verified_prior_enabled: bool = False,
        schema_verified_prior_bonus: float = 0.08,
        schema_verified_prior_min_schema_score: float = 0.72,
        schema_verified_prior_min_schema_idf: float = 0.58,
        schema_verified_prior_min_overlap: float = 0.18,
        schema_verified_prior_max_original_rank: int = 20,
        schema_verified_prior_numeric_requires_value: bool = True,
        schema_verified_prior_block_long_context: bool = True,
        schema_verified_prior_rescue_enabled: bool = False,
        schema_verified_prior_min_rescue_bonus: float = 0.045,
        answer_type_conflict_guard_enabled: bool = False,
        answer_type_conflict_guard_min_old_signal: float = 0.55,
        answer_type_conflict_guard_min_candidate_signal: float = 0.35,
        answer_type_conflict_guard_min_conflict_margin: float = 0.08,
        answer_type_conflict_guard_numeric_only: bool = True,
        support_set_rerank_enabled: bool = False,
        support_set_rerank_target_k: int = 5,
        support_set_rerank_protect_top_k: int = 2,
        support_set_rerank_max_replacements: int = 2,
        support_set_rerank_max_candidates: int = 96,
        support_set_rerank_max_original_rank: int = 200,
        support_set_rerank_min_candidate_score: float = 0.52,
        support_set_rerank_redundancy_penalty: float = 0.18,
        support_set_rerank_preserve_original_topk: bool = True,
        support_set_rerank_preserve_min_gain: float = 0.03,
        support_set_rerank_preserve_max_coverage_loss: float = 0.02,
        bridge_slot_rescue_enabled: bool = False,
        bridge_slot_rescue_min_score: float = 0.58,
        bridge_slot_rescue_min_improvement: float = 0.08,
        bridge_slot_rescue_min_anchor_overlap: float = 0.34,
        bridge_slot_rescue_require_relation: bool = True,
        bridge_slot_rescue_require_old_deficit: bool = True,
        bridge_slot_rescue_deficit_max_old_score: float = 0.50,
        bridge_slot_rescue_min_missing_gain: float = 0.15,
        bridge_slot_rescue_append_on_no_deficit: bool = True,
        object_memory_enabled: bool = False,
        object_memory_min_score: float = 0.42,
        object_memory_max_sentences: int = 2,
        object_memory_max_objects: int = 1,
        object_memory_max_chars: int = 720,
        object_memory_append_only: bool = True,
        object_memory_append_max_items: int = 1,
        object_memory_append_candidate_top_k: int = 12,
        object_memory_append_min_plan_score: float = 0.50,
        budgeted_evidence_subgraph_enabled: bool = False,
        budgeted_evidence_subgraph_max_append: int = 1,
        budgeted_evidence_subgraph_max_candidates: int = 48,
        budgeted_evidence_subgraph_min_utility: float = 0.48,
        budgeted_evidence_subgraph_redundancy_penalty: float = 0.16,
        budgeted_evidence_subgraph_require_bridge_or_gap: bool = True,
        evidence_frontier_enabled: bool = False,
        promotion_utility_veto_enabled: bool = False,
        promotion_utility_veto_min_utility: float = 0.18,
        promotion_utility_veto_require_bridge_or_gap: bool = True,
        promotion_utility_veto_apply_to_guarded: bool = False,
        promotion_utility_veto_high_risk_only: bool = True,
        promotion_utility_veto_append_on_reject: bool = True,
        doc_lookup: Optional[Callable[[int], str]] = None,
    ):
        self.target_k = target_k
        self.protect_top_k = protect_top_k
        self.max_promotions = max_promotions
        self.plan_top_k = plan_top_k
        self.min_plan_score = min_plan_score
        self.merge_mode = merge_mode
        self.question = question
        self.guarded_min_score = guarded_min_score
        self.guarded_max_original_rank = guarded_max_original_rank
        self.guarded_max_plan_rank = guarded_max_plan_rank
        self.guarded_min_query_overlap = guarded_min_query_overlap
        self.top2_override_enabled = top2_override_enabled
        self.top2_override_min_score = top2_override_min_score
        self.top2_override_max_original_rank = top2_override_max_original_rank
        self.top2_override_max_plan_rank = top2_override_max_plan_rank
        self.top2_override_min_query_overlap = top2_override_min_query_overlap
        self.top2_override_min_query_hits = top2_override_min_query_hits
        self.top2_pair_verifier_enabled = top2_pair_verifier_enabled
        self.top2_pair_verifier_mode = top2_pair_verifier_mode
        self.top2_pair_verifier_score_weight = top2_pair_verifier_score_weight
        self.top2_pair_verifier_min_margin = top2_pair_verifier_min_margin
        self.top2_pair_verifier_max_coverage_loss = top2_pair_verifier_max_coverage_loss
        self.top2_pair_verifier_min_candidate_query_support = top2_pair_verifier_min_candidate_query_support
        self.relation_object_guard_enabled = relation_object_guard_enabled
        self.relation_object_guard_min_score = relation_object_guard_min_score
        self.relation_object_guard_min_margin = relation_object_guard_min_margin
        self.entity_conditioned_relation_guard_enabled = entity_conditioned_relation_guard_enabled
        self.entity_relation_anchor_min_overlap = entity_relation_anchor_min_overlap
        self.intervention_gate = intervention_gate or {}
        self.pairwise_utility_search_enabled = pairwise_utility_search_enabled
        self.pairwise_utility_max_original_rank = pairwise_utility_max_original_rank
        self.pairwise_utility_max_candidates = pairwise_utility_max_candidates
        self.pairwise_utility_min_margin = pairwise_utility_min_margin
        self.pairwise_utility_min_pair_score = pairwise_utility_min_pair_score
        self.pairwise_utility_min_candidate_support = pairwise_utility_min_candidate_support
        self.pairwise_utility_allowed_sources = pairwise_utility_allowed_sources
        self.pairwise_utility_original_fallback_only = pairwise_utility_original_fallback_only
        self.query_schema_memory_intervention_policy = query_schema_memory_intervention_policy
        self.query_schema_memory_top2_min_score = query_schema_memory_top2_min_score
        self.query_schema_memory_top2_min_schema_idf = query_schema_memory_top2_min_schema_idf
        self.query_schema_memory_top2_min_overlap = query_schema_memory_top2_min_overlap
        self.query_schema_memory_top2_max_schema_frequency = query_schema_memory_top2_max_schema_frequency
        self.query_schema_memory_top2_max_original_rank = query_schema_memory_top2_max_original_rank
        self.query_schema_memory_top2_min_bridge_strength = query_schema_memory_top2_min_bridge_strength
        self.schema_deficit_promotion_enabled = schema_deficit_promotion_enabled
        self.schema_deficit_min_score = schema_deficit_min_score
        self.schema_deficit_min_gain = schema_deficit_min_gain
        self.schema_deficit_min_schema_score = schema_deficit_min_schema_score
        self.schema_deficit_min_schema_idf = schema_deficit_min_schema_idf
        self.schema_deficit_max_original_rank = schema_deficit_max_original_rank
        self.schema_deficit_require_bridge_constraint = schema_deficit_require_bridge_constraint
        self.schema_deficit_min_bridge_score = schema_deficit_min_bridge_score
        self.schema_deficit_min_bridge_margin = schema_deficit_min_bridge_margin
        self.schema_deficit_min_bridge_missing_gain = schema_deficit_min_bridge_missing_gain
        self.schema_deficit_min_bridge_anchor_overlap = schema_deficit_min_bridge_anchor_overlap
        self.schema_deficit_soft_append_enabled = schema_deficit_soft_append_enabled
        self.schema_deficit_soft_append_min_score = schema_deficit_soft_append_min_score
        self.schema_deficit_soft_append_min_schema_score = schema_deficit_soft_append_min_schema_score
        self.schema_deficit_soft_append_min_schema_idf = schema_deficit_soft_append_min_schema_idf
        self.schema_deficit_soft_append_min_query_support = schema_deficit_soft_append_min_query_support
        self.schema_deficit_soft_append_max_original_rank = schema_deficit_soft_append_max_original_rank
        self.schema_reliability_gate_enabled = schema_reliability_gate_enabled
        self.schema_reliability_min_score = schema_reliability_min_score
        self.schema_reliability_rescue_min_score = schema_reliability_rescue_min_score
        self.schema_reliability_min_anchor_overlap = schema_reliability_min_anchor_overlap
        self.schema_reliability_max_schema_frequency = schema_reliability_max_schema_frequency
        self.schema_reliability_numeric_requires_value = schema_reliability_numeric_requires_value
        self.schema_verified_prior_enabled = schema_verified_prior_enabled
        self.schema_verified_prior_bonus = schema_verified_prior_bonus
        self.schema_verified_prior_min_schema_score = schema_verified_prior_min_schema_score
        self.schema_verified_prior_min_schema_idf = schema_verified_prior_min_schema_idf
        self.schema_verified_prior_min_overlap = schema_verified_prior_min_overlap
        self.schema_verified_prior_max_original_rank = schema_verified_prior_max_original_rank
        self.schema_verified_prior_numeric_requires_value = schema_verified_prior_numeric_requires_value
        self.schema_verified_prior_block_long_context = schema_verified_prior_block_long_context
        self.schema_verified_prior_rescue_enabled = schema_verified_prior_rescue_enabled
        self.schema_verified_prior_min_rescue_bonus = schema_verified_prior_min_rescue_bonus
        self.answer_type_conflict_guard_enabled = answer_type_conflict_guard_enabled
        self.answer_type_conflict_guard_min_old_signal = answer_type_conflict_guard_min_old_signal
        self.answer_type_conflict_guard_min_candidate_signal = answer_type_conflict_guard_min_candidate_signal
        self.answer_type_conflict_guard_min_conflict_margin = answer_type_conflict_guard_min_conflict_margin
        self.answer_type_conflict_guard_numeric_only = answer_type_conflict_guard_numeric_only
        self.support_set_rerank_enabled = support_set_rerank_enabled
        self.support_set_rerank_target_k = support_set_rerank_target_k
        self.support_set_rerank_protect_top_k = support_set_rerank_protect_top_k
        self.support_set_rerank_max_replacements = support_set_rerank_max_replacements
        self.support_set_rerank_max_candidates = support_set_rerank_max_candidates
        self.support_set_rerank_max_original_rank = support_set_rerank_max_original_rank
        self.support_set_rerank_min_candidate_score = support_set_rerank_min_candidate_score
        self.support_set_rerank_redundancy_penalty = support_set_rerank_redundancy_penalty
        self.support_set_rerank_preserve_original_topk = support_set_rerank_preserve_original_topk
        self.support_set_rerank_preserve_min_gain = support_set_rerank_preserve_min_gain
        self.support_set_rerank_preserve_max_coverage_loss = (
            support_set_rerank_preserve_max_coverage_loss
        )
        self.bridge_slot_rescue_enabled = bridge_slot_rescue_enabled
        self.bridge_slot_rescue_min_score = bridge_slot_rescue_min_score
        self.bridge_slot_rescue_min_improvement = bridge_slot_rescue_min_improvement
        self.bridge_slot_rescue_min_anchor_overlap = bridge_slot_rescue_min_anchor_overlap
        self.bridge_slot_rescue_require_relation = bridge_slot_rescue_require_relation
        self.bridge_slot_rescue_require_old_deficit = bridge_slot_rescue_require_old_deficit
        self.bridge_slot_rescue_deficit_max_old_score = bridge_slot_rescue_deficit_max_old_score
        self.bridge_slot_rescue_min_missing_gain = bridge_slot_rescue_min_missing_gain
        self.bridge_slot_rescue_append_on_no_deficit = bridge_slot_rescue_append_on_no_deficit
        self.object_memory_enabled = object_memory_enabled
        self.object_memory_min_score = object_memory_min_score
        self.object_memory_max_sentences = object_memory_max_sentences
        self.object_memory_max_objects = object_memory_max_objects
        self.object_memory_max_chars = object_memory_max_chars
        self.object_memory_append_only = object_memory_append_only
        self.object_memory_append_max_items = object_memory_append_max_items
        self.object_memory_append_candidate_top_k = object_memory_append_candidate_top_k
        self.object_memory_append_min_plan_score = object_memory_append_min_plan_score
        self.budgeted_evidence_subgraph_enabled = budgeted_evidence_subgraph_enabled
        self.budgeted_evidence_subgraph_max_append = budgeted_evidence_subgraph_max_append
        self.budgeted_evidence_subgraph_max_candidates = budgeted_evidence_subgraph_max_candidates
        self.budgeted_evidence_subgraph_min_utility = budgeted_evidence_subgraph_min_utility
        self.budgeted_evidence_subgraph_redundancy_penalty = (
            budgeted_evidence_subgraph_redundancy_penalty
        )
        self.budgeted_evidence_subgraph_require_bridge_or_gap = (
            budgeted_evidence_subgraph_require_bridge_or_gap
        )
        self.evidence_frontier_enabled = evidence_frontier_enabled
        self.promotion_utility_veto_enabled = promotion_utility_veto_enabled
        self.promotion_utility_veto_min_utility = promotion_utility_veto_min_utility
        self.promotion_utility_veto_require_bridge_or_gap = (
            promotion_utility_veto_require_bridge_or_gap
        )
        self.promotion_utility_veto_apply_to_guarded = promotion_utility_veto_apply_to_guarded
        self.promotion_utility_veto_high_risk_only = promotion_utility_veto_high_risk_only
        self.promotion_utility_veto_append_on_reject = promotion_utility_veto_append_on_reject
        self.doc_lookup = doc_lookup

    def run(
        self,
        base_doc_ids: np.ndarray,
        base_doc_scores: np.ndarray,
        planned_results: List[Tuple[str, np.ndarray, np.ndarray]],
        schema_memory_candidates: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        doc_ids, doc_scores, info = merge_planned_retrieval_results(
            base_doc_ids=base_doc_ids,
            base_doc_scores=base_doc_scores,
            planned_results=planned_results,
            schema_memory_candidates=schema_memory_candidates,
            target_k=self.target_k,
            protect_top_k=self.protect_top_k,
            max_promotions=self.max_promotions,
            plan_top_k=self.plan_top_k,
            min_plan_score=self.min_plan_score,
            merge_mode=self.merge_mode,
            question=self.question,
            guarded_min_score=self.guarded_min_score,
            guarded_max_original_rank=self.guarded_max_original_rank,
            guarded_max_plan_rank=self.guarded_max_plan_rank,
            guarded_min_query_overlap=self.guarded_min_query_overlap,
            top2_override_enabled=self.top2_override_enabled,
            top2_override_min_score=self.top2_override_min_score,
            top2_override_max_original_rank=self.top2_override_max_original_rank,
            top2_override_max_plan_rank=self.top2_override_max_plan_rank,
            top2_override_min_query_overlap=self.top2_override_min_query_overlap,
            top2_override_min_query_hits=self.top2_override_min_query_hits,
            top2_pair_verifier_enabled=self.top2_pair_verifier_enabled,
            top2_pair_verifier_mode=self.top2_pair_verifier_mode,
            top2_pair_verifier_score_weight=self.top2_pair_verifier_score_weight,
            top2_pair_verifier_min_margin=self.top2_pair_verifier_min_margin,
            top2_pair_verifier_max_coverage_loss=self.top2_pair_verifier_max_coverage_loss,
            top2_pair_verifier_min_candidate_query_support=self.top2_pair_verifier_min_candidate_query_support,
            relation_object_guard_enabled=self.relation_object_guard_enabled,
            relation_object_guard_min_score=self.relation_object_guard_min_score,
            relation_object_guard_min_margin=self.relation_object_guard_min_margin,
            entity_conditioned_relation_guard_enabled=self.entity_conditioned_relation_guard_enabled,
            entity_relation_anchor_min_overlap=self.entity_relation_anchor_min_overlap,
            intervention_gate=self.intervention_gate,
            pairwise_utility_search_enabled=self.pairwise_utility_search_enabled,
            pairwise_utility_max_original_rank=self.pairwise_utility_max_original_rank,
            pairwise_utility_max_candidates=self.pairwise_utility_max_candidates,
            pairwise_utility_min_margin=self.pairwise_utility_min_margin,
            pairwise_utility_min_pair_score=self.pairwise_utility_min_pair_score,
            pairwise_utility_min_candidate_support=self.pairwise_utility_min_candidate_support,
            pairwise_utility_allowed_sources=self.pairwise_utility_allowed_sources,
            pairwise_utility_original_fallback_only=self.pairwise_utility_original_fallback_only,
            query_schema_memory_intervention_policy=self.query_schema_memory_intervention_policy,
            query_schema_memory_top2_min_score=self.query_schema_memory_top2_min_score,
            query_schema_memory_top2_min_schema_idf=self.query_schema_memory_top2_min_schema_idf,
            query_schema_memory_top2_min_overlap=self.query_schema_memory_top2_min_overlap,
            query_schema_memory_top2_max_schema_frequency=self.query_schema_memory_top2_max_schema_frequency,
            query_schema_memory_top2_max_original_rank=self.query_schema_memory_top2_max_original_rank,
            query_schema_memory_top2_min_bridge_strength=self.query_schema_memory_top2_min_bridge_strength,
            schema_deficit_promotion_enabled=self.schema_deficit_promotion_enabled,
            schema_deficit_min_score=self.schema_deficit_min_score,
            schema_deficit_min_gain=self.schema_deficit_min_gain,
            schema_deficit_min_schema_score=self.schema_deficit_min_schema_score,
            schema_deficit_min_schema_idf=self.schema_deficit_min_schema_idf,
            schema_deficit_max_original_rank=self.schema_deficit_max_original_rank,
            schema_deficit_require_bridge_constraint=self.schema_deficit_require_bridge_constraint,
            schema_deficit_min_bridge_score=self.schema_deficit_min_bridge_score,
            schema_deficit_min_bridge_margin=self.schema_deficit_min_bridge_margin,
            schema_deficit_min_bridge_missing_gain=self.schema_deficit_min_bridge_missing_gain,
            schema_deficit_min_bridge_anchor_overlap=self.schema_deficit_min_bridge_anchor_overlap,
            schema_deficit_soft_append_enabled=self.schema_deficit_soft_append_enabled,
            schema_deficit_soft_append_min_score=self.schema_deficit_soft_append_min_score,
            schema_deficit_soft_append_min_schema_score=self.schema_deficit_soft_append_min_schema_score,
            schema_deficit_soft_append_min_schema_idf=self.schema_deficit_soft_append_min_schema_idf,
            schema_deficit_soft_append_min_query_support=self.schema_deficit_soft_append_min_query_support,
            schema_deficit_soft_append_max_original_rank=self.schema_deficit_soft_append_max_original_rank,
            schema_reliability_gate_enabled=self.schema_reliability_gate_enabled,
            schema_reliability_min_score=self.schema_reliability_min_score,
            schema_reliability_rescue_min_score=self.schema_reliability_rescue_min_score,
            schema_reliability_min_anchor_overlap=self.schema_reliability_min_anchor_overlap,
            schema_reliability_max_schema_frequency=self.schema_reliability_max_schema_frequency,
            schema_reliability_numeric_requires_value=self.schema_reliability_numeric_requires_value,
            schema_verified_prior_enabled=self.schema_verified_prior_enabled,
            schema_verified_prior_bonus=self.schema_verified_prior_bonus,
            schema_verified_prior_min_schema_score=self.schema_verified_prior_min_schema_score,
            schema_verified_prior_min_schema_idf=self.schema_verified_prior_min_schema_idf,
            schema_verified_prior_min_overlap=self.schema_verified_prior_min_overlap,
            schema_verified_prior_max_original_rank=self.schema_verified_prior_max_original_rank,
            schema_verified_prior_numeric_requires_value=self.schema_verified_prior_numeric_requires_value,
            schema_verified_prior_block_long_context=self.schema_verified_prior_block_long_context,
            schema_verified_prior_rescue_enabled=self.schema_verified_prior_rescue_enabled,
            schema_verified_prior_min_rescue_bonus=self.schema_verified_prior_min_rescue_bonus,
            answer_type_conflict_guard_enabled=self.answer_type_conflict_guard_enabled,
            answer_type_conflict_guard_min_old_signal=self.answer_type_conflict_guard_min_old_signal,
            answer_type_conflict_guard_min_candidate_signal=self.answer_type_conflict_guard_min_candidate_signal,
            answer_type_conflict_guard_min_conflict_margin=self.answer_type_conflict_guard_min_conflict_margin,
            answer_type_conflict_guard_numeric_only=self.answer_type_conflict_guard_numeric_only,
            support_set_rerank_enabled=self.support_set_rerank_enabled,
            support_set_rerank_target_k=self.support_set_rerank_target_k,
            support_set_rerank_protect_top_k=self.support_set_rerank_protect_top_k,
            support_set_rerank_max_replacements=self.support_set_rerank_max_replacements,
            support_set_rerank_max_candidates=self.support_set_rerank_max_candidates,
            support_set_rerank_max_original_rank=self.support_set_rerank_max_original_rank,
            support_set_rerank_min_candidate_score=self.support_set_rerank_min_candidate_score,
            support_set_rerank_redundancy_penalty=self.support_set_rerank_redundancy_penalty,
            support_set_rerank_preserve_original_topk=(
                self.support_set_rerank_preserve_original_topk
            ),
            support_set_rerank_preserve_min_gain=self.support_set_rerank_preserve_min_gain,
            support_set_rerank_preserve_max_coverage_loss=(
                self.support_set_rerank_preserve_max_coverage_loss
            ),
            bridge_slot_rescue_enabled=self.bridge_slot_rescue_enabled,
            bridge_slot_rescue_min_score=self.bridge_slot_rescue_min_score,
            bridge_slot_rescue_min_improvement=self.bridge_slot_rescue_min_improvement,
            bridge_slot_rescue_min_anchor_overlap=self.bridge_slot_rescue_min_anchor_overlap,
            bridge_slot_rescue_require_relation=self.bridge_slot_rescue_require_relation,
            bridge_slot_rescue_require_old_deficit=self.bridge_slot_rescue_require_old_deficit,
            bridge_slot_rescue_deficit_max_old_score=self.bridge_slot_rescue_deficit_max_old_score,
            bridge_slot_rescue_min_missing_gain=self.bridge_slot_rescue_min_missing_gain,
            bridge_slot_rescue_append_on_no_deficit=self.bridge_slot_rescue_append_on_no_deficit,
            object_memory_enabled=self.object_memory_enabled,
            object_memory_min_score=self.object_memory_min_score,
            object_memory_max_sentences=self.object_memory_max_sentences,
            object_memory_max_objects=self.object_memory_max_objects,
            object_memory_max_chars=self.object_memory_max_chars,
            object_memory_append_only=self.object_memory_append_only,
            object_memory_append_max_items=self.object_memory_append_max_items,
            object_memory_append_candidate_top_k=self.object_memory_append_candidate_top_k,
            object_memory_append_min_plan_score=self.object_memory_append_min_plan_score,
            budgeted_evidence_subgraph_enabled=self.budgeted_evidence_subgraph_enabled,
            budgeted_evidence_subgraph_max_append=self.budgeted_evidence_subgraph_max_append,
            budgeted_evidence_subgraph_max_candidates=self.budgeted_evidence_subgraph_max_candidates,
            budgeted_evidence_subgraph_min_utility=self.budgeted_evidence_subgraph_min_utility,
            budgeted_evidence_subgraph_redundancy_penalty=(
                self.budgeted_evidence_subgraph_redundancy_penalty
            ),
            budgeted_evidence_subgraph_require_bridge_or_gap=(
                self.budgeted_evidence_subgraph_require_bridge_or_gap
            ),
            evidence_frontier_enabled=self.evidence_frontier_enabled,
            promotion_utility_veto_enabled=self.promotion_utility_veto_enabled,
            promotion_utility_veto_min_utility=self.promotion_utility_veto_min_utility,
            promotion_utility_veto_require_bridge_or_gap=(
                self.promotion_utility_veto_require_bridge_or_gap
            ),
            promotion_utility_veto_apply_to_guarded=(
                self.promotion_utility_veto_apply_to_guarded
            ),
            promotion_utility_veto_high_risk_only=(
                self.promotion_utility_veto_high_risk_only
            ),
            promotion_utility_veto_append_on_reject=(
                self.promotion_utility_veto_append_on_reject
            ),
            doc_lookup=self.doc_lookup,
        )
        info["role"] = self.role
        return doc_ids, doc_scores, info


def merge_planned_retrieval_results(
    base_doc_ids: np.ndarray,
    base_doc_scores: np.ndarray,
    planned_results: List[Tuple[str, np.ndarray, np.ndarray]],
    schema_memory_candidates: Optional[List[Dict[str, Any]]] = None,
    target_k: int = 5,
    protect_top_k: int = 2,
    max_promotions: int = 2,
    plan_top_k: int = 20,
    min_plan_score: float = 0.18,
    merge_mode: str = "guarded_promote",
    question: str = "",
    guarded_min_score: float = 0.82,
    guarded_max_original_rank: int = 80,
    guarded_max_plan_rank: int = 8,
    guarded_min_query_overlap: float = 0.08,
    top2_override_enabled: bool = False,
    top2_override_min_score: float = 0.96,
    top2_override_max_original_rank: int = 30,
    top2_override_max_plan_rank: int = 2,
    top2_override_min_query_overlap: float = 0.18,
    top2_override_min_query_hits: int = 1,
    top2_pair_verifier_enabled: bool = False,
    top2_pair_verifier_mode: str = "hard",
    top2_pair_verifier_score_weight: float = 0.12,
    top2_pair_verifier_min_margin: float = 0.04,
    top2_pair_verifier_max_coverage_loss: float = 0.02,
    top2_pair_verifier_min_candidate_query_support: float = 0.45,
    relation_object_guard_enabled: bool = False,
    relation_object_guard_min_score: float = 0.55,
    relation_object_guard_min_margin: float = 0.08,
    entity_conditioned_relation_guard_enabled: bool = False,
    entity_relation_anchor_min_overlap: float = 0.50,
    intervention_gate: Optional[Dict[str, Any]] = None,
    pairwise_utility_search_enabled: bool = False,
    pairwise_utility_max_original_rank: int = 30,
    pairwise_utility_max_candidates: int = 32,
    pairwise_utility_min_margin: float = 0.12,
    pairwise_utility_min_pair_score: float = 0.65,
    pairwise_utility_min_candidate_support: float = 0.45,
    pairwise_utility_allowed_sources: str = "planned_and_original",
    pairwise_utility_original_fallback_only: bool = True,
    query_schema_memory_intervention_policy: str = "risk_aware",
    query_schema_memory_top2_min_score: float = 0.66,
    query_schema_memory_top2_min_schema_idf: float = 0.32,
    query_schema_memory_top2_min_overlap: float = 0.10,
    query_schema_memory_top2_max_schema_frequency: int = 80,
    query_schema_memory_top2_max_original_rank: int = 24,
    query_schema_memory_top2_min_bridge_strength: int = 3,
    schema_deficit_promotion_enabled: bool = False,
    schema_deficit_min_score: float = 0.62,
    schema_deficit_min_gain: float = 0.10,
    schema_deficit_min_schema_score: float = 0.68,
    schema_deficit_min_schema_idf: float = 0.30,
    schema_deficit_max_original_rank: int = 60,
    schema_deficit_require_bridge_constraint: bool = True,
    schema_deficit_min_bridge_score: float = 0.54,
    schema_deficit_min_bridge_margin: float = 0.08,
    schema_deficit_min_bridge_missing_gain: float = 0.10,
    schema_deficit_min_bridge_anchor_overlap: float = 0.24,
    schema_deficit_soft_append_enabled: bool = True,
    schema_deficit_soft_append_min_score: float = 0.48,
    schema_deficit_soft_append_min_schema_score: float = 0.58,
    schema_deficit_soft_append_min_schema_idf: float = 0.24,
    schema_deficit_soft_append_min_query_support: float = 0.34,
    schema_deficit_soft_append_max_original_rank: int = 80,
    schema_reliability_gate_enabled: bool = False,
    schema_reliability_min_score: float = 0.62,
    schema_reliability_rescue_min_score: float = 0.76,
    schema_reliability_min_anchor_overlap: float = 0.08,
    schema_reliability_max_schema_frequency: int = 80,
    schema_reliability_numeric_requires_value: bool = True,
    schema_verified_prior_enabled: bool = False,
    schema_verified_prior_bonus: float = 0.08,
    schema_verified_prior_min_schema_score: float = 0.72,
    schema_verified_prior_min_schema_idf: float = 0.58,
    schema_verified_prior_min_overlap: float = 0.18,
    schema_verified_prior_max_original_rank: int = 20,
    schema_verified_prior_numeric_requires_value: bool = True,
    schema_verified_prior_block_long_context: bool = True,
    schema_verified_prior_rescue_enabled: bool = False,
    schema_verified_prior_min_rescue_bonus: float = 0.045,
    answer_type_conflict_guard_enabled: bool = False,
    answer_type_conflict_guard_min_old_signal: float = 0.55,
    answer_type_conflict_guard_min_candidate_signal: float = 0.35,
    answer_type_conflict_guard_min_conflict_margin: float = 0.08,
    answer_type_conflict_guard_numeric_only: bool = True,
    support_set_rerank_enabled: bool = False,
    support_set_rerank_target_k: int = 5,
    support_set_rerank_protect_top_k: int = 2,
    support_set_rerank_max_replacements: int = 2,
    support_set_rerank_max_candidates: int = 96,
    support_set_rerank_max_original_rank: int = 200,
    support_set_rerank_min_candidate_score: float = 0.52,
    support_set_rerank_redundancy_penalty: float = 0.18,
    support_set_rerank_preserve_original_topk: bool = True,
    support_set_rerank_preserve_min_gain: float = 0.03,
    support_set_rerank_preserve_max_coverage_loss: float = 0.02,
    bridge_slot_rescue_enabled: bool = False,
    bridge_slot_rescue_min_score: float = 0.58,
    bridge_slot_rescue_min_improvement: float = 0.08,
    bridge_slot_rescue_min_anchor_overlap: float = 0.34,
    bridge_slot_rescue_require_relation: bool = True,
    bridge_slot_rescue_require_old_deficit: bool = True,
    bridge_slot_rescue_deficit_max_old_score: float = 0.50,
    bridge_slot_rescue_min_missing_gain: float = 0.15,
    bridge_slot_rescue_append_on_no_deficit: bool = True,
    object_memory_enabled: bool = False,
    object_memory_min_score: float = 0.42,
    object_memory_max_sentences: int = 2,
    object_memory_max_objects: int = 1,
    object_memory_max_chars: int = 720,
    object_memory_append_only: bool = True,
    object_memory_append_max_items: int = 1,
    object_memory_append_candidate_top_k: int = 12,
    object_memory_append_min_plan_score: float = 0.50,
    budgeted_evidence_subgraph_enabled: bool = False,
    budgeted_evidence_subgraph_max_append: int = 1,
    budgeted_evidence_subgraph_max_candidates: int = 48,
    budgeted_evidence_subgraph_min_utility: float = 0.48,
    budgeted_evidence_subgraph_redundancy_penalty: float = 0.16,
    budgeted_evidence_subgraph_require_bridge_or_gap: bool = True,
    evidence_frontier_enabled: bool = False,
    promotion_utility_veto_enabled: bool = False,
    promotion_utility_veto_min_utility: float = 0.18,
    promotion_utility_veto_require_bridge_or_gap: bool = True,
    promotion_utility_veto_apply_to_guarded: bool = False,
    promotion_utility_veto_high_risk_only: bool = True,
    promotion_utility_veto_append_on_reject: bool = True,
    doc_lookup: Optional[Callable[[int], str]] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    target_k = max(1, int(target_k))
    protect_top_k = min(max(0, int(protect_top_k)), target_k, len(base_doc_ids))
    max_promotions = max(0, int(max_promotions))
    plan_top_k = max(1, int(plan_top_k))
    merge_mode = str(merge_mode or "append_only").strip().lower()
    guarded_min_score = float(guarded_min_score)
    guarded_max_original_rank = max(1, int(guarded_max_original_rank))
    guarded_max_plan_rank = max(1, int(guarded_max_plan_rank))
    guarded_min_query_overlap = max(0.0, float(guarded_min_query_overlap))
    top2_override_enabled = bool(top2_override_enabled)
    top2_override_min_score = float(top2_override_min_score)
    top2_override_max_original_rank = max(2, int(top2_override_max_original_rank))
    top2_override_max_plan_rank = max(1, int(top2_override_max_plan_rank))
    top2_override_min_query_overlap = max(0.0, float(top2_override_min_query_overlap))
    top2_override_min_query_hits = max(1, int(top2_override_min_query_hits))
    top2_pair_verifier_enabled = bool(top2_pair_verifier_enabled)
    top2_pair_verifier_mode = str(top2_pair_verifier_mode or "hard").strip().lower()
    if top2_pair_verifier_mode not in {"hard", "soft"}:
        top2_pair_verifier_mode = "hard"
    top2_pair_verifier_score_weight = max(0.0, float(top2_pair_verifier_score_weight))
    top2_pair_verifier_min_margin = max(0.0, float(top2_pair_verifier_min_margin))
    top2_pair_verifier_max_coverage_loss = max(0.0, float(top2_pair_verifier_max_coverage_loss))
    top2_pair_verifier_min_candidate_query_support = max(
        0.0,
        float(top2_pair_verifier_min_candidate_query_support),
    )
    relation_object_guard_enabled = bool(relation_object_guard_enabled)
    relation_object_guard_min_score = max(0.0, float(relation_object_guard_min_score))
    relation_object_guard_min_margin = max(0.0, float(relation_object_guard_min_margin))
    entity_conditioned_relation_guard_enabled = bool(entity_conditioned_relation_guard_enabled)
    entity_relation_anchor_min_overlap = _clamp(entity_relation_anchor_min_overlap, 0.0, 1.0)
    intervention_gate = dict(intervention_gate or {})
    pairwise_utility_search_enabled = bool(pairwise_utility_search_enabled)
    pairwise_utility_max_original_rank = max(2, int(pairwise_utility_max_original_rank))
    pairwise_utility_max_candidates = max(1, int(pairwise_utility_max_candidates))
    pairwise_utility_min_margin = max(0.0, float(pairwise_utility_min_margin))
    pairwise_utility_min_pair_score = max(0.0, float(pairwise_utility_min_pair_score))
    pairwise_utility_min_candidate_support = max(
        0.0,
        float(pairwise_utility_min_candidate_support),
    )
    pairwise_utility_allowed_sources_normalized = _normalize_pairwise_utility_sources(
        pairwise_utility_allowed_sources,
        default="planned_and_original",
    )
    pairwise_utility_allowed_source_set = set(pairwise_utility_allowed_sources_normalized)
    pairwise_utility_original_fallback_only = bool(pairwise_utility_original_fallback_only)
    query_schema_memory_intervention_policy = str(
        query_schema_memory_intervention_policy or "risk_aware"
    ).strip().lower()
    if query_schema_memory_intervention_policy not in {"off", "risk_aware", "append_only"}:
        query_schema_memory_intervention_policy = "risk_aware"
    query_schema_memory_top2_min_score = max(0.0, float(query_schema_memory_top2_min_score))
    query_schema_memory_top2_min_schema_idf = _clamp(query_schema_memory_top2_min_schema_idf)
    query_schema_memory_top2_min_overlap = _clamp(query_schema_memory_top2_min_overlap)
    query_schema_memory_top2_max_schema_frequency = max(0, int(query_schema_memory_top2_max_schema_frequency))
    query_schema_memory_top2_max_original_rank = max(2, int(query_schema_memory_top2_max_original_rank))
    query_schema_memory_top2_min_bridge_strength = max(0, int(query_schema_memory_top2_min_bridge_strength))
    schema_deficit_promotion_enabled = bool(schema_deficit_promotion_enabled)
    schema_deficit_min_score = max(0.0, float(schema_deficit_min_score))
    schema_deficit_min_gain = max(0.0, float(schema_deficit_min_gain))
    schema_deficit_min_schema_score = max(0.0, float(schema_deficit_min_schema_score))
    schema_deficit_min_schema_idf = max(0.0, float(schema_deficit_min_schema_idf))
    schema_deficit_max_original_rank = max(2, int(schema_deficit_max_original_rank))
    schema_deficit_require_bridge_constraint = bool(schema_deficit_require_bridge_constraint)
    schema_deficit_min_bridge_score = max(0.0, float(schema_deficit_min_bridge_score))
    schema_deficit_min_bridge_margin = max(0.0, float(schema_deficit_min_bridge_margin))
    schema_deficit_min_bridge_missing_gain = max(0.0, float(schema_deficit_min_bridge_missing_gain))
    schema_deficit_min_bridge_anchor_overlap = _clamp(schema_deficit_min_bridge_anchor_overlap)
    schema_deficit_soft_append_enabled = bool(schema_deficit_soft_append_enabled)
    schema_deficit_soft_append_min_score = max(0.0, float(schema_deficit_soft_append_min_score))
    schema_deficit_soft_append_min_schema_score = max(0.0, float(schema_deficit_soft_append_min_schema_score))
    schema_deficit_soft_append_min_schema_idf = max(0.0, float(schema_deficit_soft_append_min_schema_idf))
    schema_deficit_soft_append_min_query_support = _clamp(schema_deficit_soft_append_min_query_support)
    schema_deficit_soft_append_max_original_rank = max(2, int(schema_deficit_soft_append_max_original_rank))
    schema_reliability_gate_enabled = bool(schema_reliability_gate_enabled)
    schema_reliability_min_score = _clamp(schema_reliability_min_score)
    schema_reliability_rescue_min_score = _clamp(schema_reliability_rescue_min_score)
    schema_reliability_min_anchor_overlap = _clamp(schema_reliability_min_anchor_overlap)
    schema_reliability_max_schema_frequency = max(0, int(schema_reliability_max_schema_frequency))
    schema_reliability_numeric_requires_value = bool(schema_reliability_numeric_requires_value)
    schema_verified_prior_enabled = bool(schema_verified_prior_enabled)
    schema_verified_prior_bonus = max(0.0, float(schema_verified_prior_bonus))
    schema_verified_prior_min_schema_score = max(0.0, float(schema_verified_prior_min_schema_score))
    schema_verified_prior_min_schema_idf = _clamp(schema_verified_prior_min_schema_idf)
    schema_verified_prior_min_overlap = _clamp(schema_verified_prior_min_overlap)
    schema_verified_prior_max_original_rank = max(2, int(schema_verified_prior_max_original_rank))
    schema_verified_prior_numeric_requires_value = bool(schema_verified_prior_numeric_requires_value)
    schema_verified_prior_block_long_context = bool(schema_verified_prior_block_long_context)
    schema_verified_prior_rescue_enabled = bool(schema_verified_prior_rescue_enabled)
    schema_verified_prior_min_rescue_bonus = max(0.0, float(schema_verified_prior_min_rescue_bonus))
    answer_type_conflict_guard_enabled = bool(answer_type_conflict_guard_enabled)
    answer_type_conflict_guard_min_old_signal = _clamp(
        answer_type_conflict_guard_min_old_signal,
        0.0,
        2.0,
    )
    answer_type_conflict_guard_min_candidate_signal = _clamp(
        answer_type_conflict_guard_min_candidate_signal,
        0.0,
        2.0,
    )
    answer_type_conflict_guard_min_conflict_margin = max(
        0.0,
        float(answer_type_conflict_guard_min_conflict_margin),
    )
    answer_type_conflict_guard_numeric_only = bool(answer_type_conflict_guard_numeric_only)
    support_set_rerank_enabled = bool(support_set_rerank_enabled)
    support_set_rerank_target_k = max(1, int(support_set_rerank_target_k))
    support_set_rerank_protect_top_k = max(0, int(support_set_rerank_protect_top_k))
    support_set_rerank_max_replacements = max(0, int(support_set_rerank_max_replacements))
    support_set_rerank_max_candidates = max(
        support_set_rerank_target_k,
        int(support_set_rerank_max_candidates),
    )
    support_set_rerank_max_original_rank = max(
        support_set_rerank_target_k,
        int(support_set_rerank_max_original_rank),
    )
    support_set_rerank_min_candidate_score = max(
        0.0,
        float(support_set_rerank_min_candidate_score),
    )
    support_set_rerank_redundancy_penalty = max(
        0.0,
        float(support_set_rerank_redundancy_penalty),
    )
    support_set_rerank_preserve_original_topk = bool(
        support_set_rerank_preserve_original_topk
    )
    support_set_rerank_preserve_min_gain = max(
        0.0,
        float(support_set_rerank_preserve_min_gain),
    )
    support_set_rerank_preserve_max_coverage_loss = max(
        0.0,
        float(support_set_rerank_preserve_max_coverage_loss),
    )
    bridge_slot_rescue_enabled = bool(bridge_slot_rescue_enabled)
    bridge_slot_rescue_min_score = max(0.0, float(bridge_slot_rescue_min_score))
    bridge_slot_rescue_min_improvement = max(0.0, float(bridge_slot_rescue_min_improvement))
    bridge_slot_rescue_min_anchor_overlap = _clamp(bridge_slot_rescue_min_anchor_overlap)
    bridge_slot_rescue_require_relation = bool(bridge_slot_rescue_require_relation)
    bridge_slot_rescue_require_old_deficit = bool(bridge_slot_rescue_require_old_deficit)
    bridge_slot_rescue_deficit_max_old_score = max(
        0.0,
        float(bridge_slot_rescue_deficit_max_old_score),
    )
    bridge_slot_rescue_min_missing_gain = max(0.0, float(bridge_slot_rescue_min_missing_gain))
    bridge_slot_rescue_append_on_no_deficit = bool(bridge_slot_rescue_append_on_no_deficit)
    object_memory_enabled = bool(object_memory_enabled)
    object_memory_min_score = max(0.0, float(object_memory_min_score))
    object_memory_max_sentences = max(1, int(object_memory_max_sentences))
    object_memory_max_objects = max(0, int(object_memory_max_objects))
    object_memory_max_chars = max(120, int(object_memory_max_chars))
    object_memory_append_only = bool(object_memory_append_only)
    object_memory_append_max_items = max(0, int(object_memory_append_max_items))
    object_memory_append_candidate_top_k = max(1, int(object_memory_append_candidate_top_k))
    object_memory_append_min_plan_score = max(0.0, float(object_memory_append_min_plan_score))
    budgeted_evidence_subgraph_enabled = bool(budgeted_evidence_subgraph_enabled)
    budgeted_evidence_subgraph_max_append = max(0, int(budgeted_evidence_subgraph_max_append))
    budgeted_evidence_subgraph_max_candidates = max(1, int(budgeted_evidence_subgraph_max_candidates))
    budgeted_evidence_subgraph_min_utility = max(0.0, float(budgeted_evidence_subgraph_min_utility))
    budgeted_evidence_subgraph_redundancy_penalty = max(
        0.0,
        float(budgeted_evidence_subgraph_redundancy_penalty),
    )
    budgeted_evidence_subgraph_require_bridge_or_gap = bool(
        budgeted_evidence_subgraph_require_bridge_or_gap
    )
    evidence_frontier_enabled = bool(evidence_frontier_enabled)
    promotion_utility_veto_enabled = bool(promotion_utility_veto_enabled)
    promotion_utility_veto_min_utility = max(0.0, float(promotion_utility_veto_min_utility))
    promotion_utility_veto_require_bridge_or_gap = bool(
        promotion_utility_veto_require_bridge_or_gap
    )
    promotion_utility_veto_apply_to_guarded = bool(promotion_utility_veto_apply_to_guarded)
    promotion_utility_veto_high_risk_only = bool(promotion_utility_veto_high_risk_only)
    promotion_utility_veto_append_on_reject = bool(promotion_utility_veto_append_on_reject)
    evidence_frontier_active = (
        evidence_frontier_enabled
        or budgeted_evidence_subgraph_enabled
        or (
            merge_mode == "append_only"
            and (
                bridge_slot_rescue_enabled
                or schema_deficit_soft_append_enabled
                or object_memory_append_only
            )
        )
    )
    utility_frontier_active = pairwise_utility_search_enabled or evidence_frontier_active
    if merge_mode not in {"append_only", "replace_topk", "guarded_promote"}:
        merge_mode = "append_only"

    base_ids = [int(item) for item in base_doc_ids.tolist()]
    base_scores = [float(item) for item in base_doc_scores.tolist()]
    base_rank = {doc_id: rank for rank, doc_id in enumerate(base_ids)}
    base_score = {doc_id: base_scores[rank] for rank, doc_id in enumerate(base_ids)}
    schema_memory_candidates = list(schema_memory_candidates or [])
    old_top = set(base_ids[:target_k])
    protected = base_ids[:protect_top_k]
    old_rank2_doc_id = base_ids[1] if len(base_ids) > 1 else None
    doc_text_cache: Dict[int, str] = {}

    def get_doc_text(doc_id: int) -> str:
        if doc_lookup is None:
            return ""
        doc_id = int(doc_id)
        if doc_id in doc_text_cache:
            return doc_text_cache[doc_id]
        try:
            value = doc_lookup(doc_id)
        except Exception:
            value = ""
        text = str(value or "")
        doc_text_cache[doc_id] = text
        return text

    candidates: Dict[int, Dict[str, Any]] = {}
    old_rank2_planned_evidence: Optional[Dict[str, Any]] = None
    raw_candidate_count = 0
    duplicate_candidate_hits = 0
    existing_top_candidate_count = 0
    low_score_candidate_count = 0
    for query, doc_ids, doc_scores in planned_results:
        if doc_ids is None or doc_scores is None:
            continue
        top_doc_ids = doc_ids[:plan_top_k].tolist()
        top_scores = np.asarray(doc_scores[:len(top_doc_ids)], dtype=float)
        if len(top_doc_ids) == 0:
            continue
        finite_scores = top_scores[np.isfinite(top_scores)]
        if len(finite_scores) > 0:
            min_score = float(np.min(finite_scores))
            max_score = float(np.max(finite_scores))
        else:
            min_score = 0.0
            max_score = 0.0

        for rank, doc_id_raw in enumerate(top_doc_ids):
            doc_id = int(doc_id_raw)
            raw_candidate_count += 1
            score = float(doc_scores[rank]) if rank < len(doc_scores) else 0.0
            query_overlap = _query_anchor_overlap(question, query)
            rank_bonus = 1.0 / (rank + 1)
            if np.isfinite(score) and max_score > min_score:
                normalized_score = (score - min_score) / (max_score - min_score)
            else:
                normalized_score = rank_bonus
            calibrated_score = 0.70 * normalized_score + 0.30 * rank_bonus
            if calibrated_score < min_plan_score:
                low_score_candidate_count += 1
                continue
            original_rank = base_rank.get(doc_id)
            original_bonus = 0.0
            if original_rank is not None:
                original_bonus = 0.20 / (original_rank + 1)
            plan_score = calibrated_score + original_bonus
            if doc_id in old_top:
                existing_top_candidate_count += 1
                if old_rank2_doc_id is not None and doc_id == old_rank2_doc_id:
                    if old_rank2_planned_evidence is None:
                        old_rank2_planned_evidence = {
                            "doc_id": doc_id,
                            "query": query,
                            "queries": [query],
                            "query_count": 1,
                            "plan_rank": rank,
                            "plan_score": plan_score,
                            "retrieval_score": score,
                            "normalized_score": normalized_score,
                            "calibrated_score": calibrated_score,
                            "original_rank": original_rank,
                            "query_overlap": query_overlap,
                        }
                    else:
                        old_rank2_planned_evidence["query_count"] = int(
                            old_rank2_planned_evidence.get("query_count", 1)
                        ) + 1
                        if query not in old_rank2_planned_evidence.get("queries", []):
                            old_rank2_planned_evidence.setdefault("queries", []).append(query)
                        if plan_score > old_rank2_planned_evidence.get("plan_score", 0.0):
                            old_rank2_planned_evidence.update({
                                "query": query,
                                "plan_rank": rank,
                                "plan_score": plan_score,
                                "retrieval_score": score,
                                "normalized_score": normalized_score,
                                "calibrated_score": calibrated_score,
                                "query_overlap": query_overlap,
                            })
                if not top2_override_enabled or doc_id in base_ids[:min(2, len(base_ids))]:
                    continue
            existing = candidates.get(doc_id)
            if existing is not None:
                duplicate_candidate_hits += 1
                existing["query_count"] = int(existing.get("query_count", 1)) + 1
                if query not in existing.get("queries", []):
                    existing.setdefault("queries", []).append(query)
            if existing is None or plan_score > existing["plan_score"]:
                previous_queries = list(existing.get("queries", [])) if existing is not None else []
                if query not in previous_queries:
                    previous_queries.append(query)
                candidates[doc_id] = {
                    "doc_id": doc_id,
                    "query": query,
                    "queries": previous_queries[:6],
                    "query_count": int(existing.get("query_count", 1)) if existing is not None else 1,
                    "plan_rank": rank,
                    "plan_score": plan_score,
                    "retrieval_score": score,
                    "normalized_score": normalized_score,
                    "calibrated_score": calibrated_score,
                    "original_rank": original_rank,
                    "query_overlap": query_overlap,
                }

    ranked_candidates = sorted(
        candidates.values(),
        key=lambda item: (-item["plan_score"], item["plan_rank"], item["original_rank"] if item["original_rank"] is not None else 999999),
    )

    effective_protect_top_k = protect_top_k
    new_top = list(base_ids[:target_k])
    seen = set(new_top)
    promoted = []
    expansions = []
    guard_rejections = []
    guard_rejection_counts = Counter()
    guarded_considered_count = 0
    top2_override_rejections = []
    top2_override_rejection_counts = Counter()
    top2_override_item = None
    bridge_slot_append_items = []
    bridge_slot_append_seen = set()
    schema_deficit_soft_append_items = []
    schema_deficit_soft_append_seen = set()
    promotion_utility_soft_append_items = []
    promotion_utility_soft_append_seen = set()
    promotion_utility_soft_append_count = 0
    object_memory_append_items = []
    object_memory_append_seen = set()
    object_memory_append_reject_counts = Counter()
    object_memory_append_debug = []
    object_memory_append_candidate_count = 0
    object_memory_append_rejected_count = 0
    object_memory_append_minimal_sentence_count = 0
    object_memory_append_info = {
        "enabled": bool(
            object_memory_enabled
            and object_memory_append_only
            and object_memory_append_max_items > 0
        ),
        "reason": (
            "disabled"
            if not (
                object_memory_enabled
                and object_memory_append_only
                and object_memory_append_max_items > 0
            )
            else "not_evaluated"
        ),
        "candidate_count": 0,
        "accepted_count": 0,
        "rejected_count": 0,
        "minimal_sentence_count": 0,
        "accepted_doc_ids": [],
        "reject_reason_counts": {},
        "policy": {
            "max_items": object_memory_append_max_items,
            "candidate_top_k": object_memory_append_candidate_top_k,
            "min_plan_score": object_memory_append_min_plan_score,
            "min_memory_score": object_memory_min_score,
            "max_sentences": object_memory_max_sentences,
            "max_objects": object_memory_max_objects,
            "max_chars": object_memory_max_chars,
        },
        "top_candidates": [],
    }
    support_set_utility_candidates: List[Dict[str, Any]] = []
    support_set_rerank_info = {
        "enabled": support_set_rerank_enabled,
        "applied": False,
        "reason": "disabled" if not support_set_rerank_enabled else "not_evaluated",
        "policy": {
            "target_k": support_set_rerank_target_k,
            "protect_top_k": support_set_rerank_protect_top_k,
            "max_replacements": support_set_rerank_max_replacements,
            "max_candidates": support_set_rerank_max_candidates,
            "max_original_rank": support_set_rerank_max_original_rank,
            "min_candidate_score": support_set_rerank_min_candidate_score,
            "redundancy_penalty": support_set_rerank_redundancy_penalty,
            "preserve_original_topk": support_set_rerank_preserve_original_topk,
            "preserve_min_gain": support_set_rerank_preserve_min_gain,
            "preserve_max_coverage_loss": support_set_rerank_preserve_max_coverage_loss,
        },
    }
    top2_override_info = {
        "enabled": top2_override_enabled,
        "applied": False,
        "reason": (
            "disabled_by_intervention_gate"
            if intervention_gate.get("enabled") and not intervention_gate.get("top2_allowed", True)
            else "disabled"
        ),
        "intervention_gate": intervention_gate,
        "policy": {
            "min_score": top2_override_min_score,
            "max_original_rank": top2_override_max_original_rank,
            "max_plan_rank": top2_override_max_plan_rank,
            "min_query_overlap": top2_override_min_query_overlap,
            "min_query_hits": top2_override_min_query_hits,
            "pair_verifier_enabled": top2_pair_verifier_enabled,
            "pair_verifier_mode": top2_pair_verifier_mode,
            "pair_verifier_score_weight": top2_pair_verifier_score_weight,
            "pair_verifier_min_margin": top2_pair_verifier_min_margin,
            "pair_verifier_max_coverage_loss": top2_pair_verifier_max_coverage_loss,
            "pair_verifier_min_candidate_query_support": top2_pair_verifier_min_candidate_query_support,
            "relation_object_guard_enabled": relation_object_guard_enabled,
            "relation_object_guard_min_score": relation_object_guard_min_score,
            "relation_object_guard_min_margin": relation_object_guard_min_margin,
            "entity_conditioned_relation_guard_enabled": entity_conditioned_relation_guard_enabled,
            "entity_relation_anchor_min_overlap": entity_relation_anchor_min_overlap,
            "pairwise_utility_search_enabled": pairwise_utility_search_enabled,
            "pairwise_utility_max_original_rank": pairwise_utility_max_original_rank,
            "pairwise_utility_max_candidates": pairwise_utility_max_candidates,
            "pairwise_utility_min_margin": pairwise_utility_min_margin,
            "pairwise_utility_min_pair_score": pairwise_utility_min_pair_score,
            "pairwise_utility_min_candidate_support": pairwise_utility_min_candidate_support,
            "pairwise_utility_allowed_sources": pairwise_utility_allowed_sources_normalized,
            "pairwise_utility_original_fallback_only": pairwise_utility_original_fallback_only,
            "schema_deficit_promotion_enabled": schema_deficit_promotion_enabled,
            "schema_deficit_min_score": schema_deficit_min_score,
            "schema_deficit_min_gain": schema_deficit_min_gain,
            "schema_deficit_min_schema_score": schema_deficit_min_schema_score,
            "schema_deficit_min_schema_idf": schema_deficit_min_schema_idf,
            "schema_deficit_max_original_rank": schema_deficit_max_original_rank,
            "schema_deficit_require_bridge_constraint": schema_deficit_require_bridge_constraint,
            "schema_deficit_min_bridge_score": schema_deficit_min_bridge_score,
            "schema_deficit_min_bridge_margin": schema_deficit_min_bridge_margin,
            "schema_deficit_min_bridge_missing_gain": schema_deficit_min_bridge_missing_gain,
            "schema_deficit_min_bridge_anchor_overlap": schema_deficit_min_bridge_anchor_overlap,
            "schema_deficit_soft_append_enabled": schema_deficit_soft_append_enabled,
            "schema_deficit_soft_append_min_score": schema_deficit_soft_append_min_score,
            "schema_deficit_soft_append_min_schema_score": schema_deficit_soft_append_min_schema_score,
            "schema_deficit_soft_append_min_schema_idf": schema_deficit_soft_append_min_schema_idf,
            "schema_deficit_soft_append_min_query_support": schema_deficit_soft_append_min_query_support,
            "schema_deficit_soft_append_max_original_rank": schema_deficit_soft_append_max_original_rank,
            "schema_reliability_gate_enabled": schema_reliability_gate_enabled,
            "schema_reliability_min_score": schema_reliability_min_score,
            "schema_reliability_rescue_min_score": schema_reliability_rescue_min_score,
            "schema_reliability_min_anchor_overlap": schema_reliability_min_anchor_overlap,
            "schema_reliability_max_schema_frequency": schema_reliability_max_schema_frequency,
            "schema_reliability_numeric_requires_value": schema_reliability_numeric_requires_value,
            "schema_verified_prior_enabled": schema_verified_prior_enabled,
            "schema_verified_prior_bonus": schema_verified_prior_bonus,
            "schema_verified_prior_min_schema_score": schema_verified_prior_min_schema_score,
            "schema_verified_prior_min_schema_idf": schema_verified_prior_min_schema_idf,
            "schema_verified_prior_min_overlap": schema_verified_prior_min_overlap,
            "schema_verified_prior_max_original_rank": schema_verified_prior_max_original_rank,
            "schema_verified_prior_numeric_requires_value": schema_verified_prior_numeric_requires_value,
            "schema_verified_prior_block_long_context": schema_verified_prior_block_long_context,
            "schema_verified_prior_rescue_enabled": schema_verified_prior_rescue_enabled,
            "schema_verified_prior_min_rescue_bonus": schema_verified_prior_min_rescue_bonus,
            "answer_type_conflict_guard_enabled": answer_type_conflict_guard_enabled,
            "answer_type_conflict_guard_min_old_signal": answer_type_conflict_guard_min_old_signal,
            "answer_type_conflict_guard_min_candidate_signal": answer_type_conflict_guard_min_candidate_signal,
            "answer_type_conflict_guard_min_conflict_margin": answer_type_conflict_guard_min_conflict_margin,
            "answer_type_conflict_guard_numeric_only": answer_type_conflict_guard_numeric_only,
            "bridge_slot_rescue_enabled": bridge_slot_rescue_enabled,
            "bridge_slot_rescue_min_score": bridge_slot_rescue_min_score,
            "bridge_slot_rescue_min_improvement": bridge_slot_rescue_min_improvement,
            "bridge_slot_rescue_min_anchor_overlap": bridge_slot_rescue_min_anchor_overlap,
            "bridge_slot_rescue_require_relation": bridge_slot_rescue_require_relation,
            "bridge_slot_rescue_require_old_deficit": bridge_slot_rescue_require_old_deficit,
            "bridge_slot_rescue_deficit_max_old_score": bridge_slot_rescue_deficit_max_old_score,
            "bridge_slot_rescue_min_missing_gain": bridge_slot_rescue_min_missing_gain,
            "bridge_slot_rescue_append_on_no_deficit": bridge_slot_rescue_append_on_no_deficit,
            "object_memory_enabled": object_memory_enabled,
            "object_memory_min_score": object_memory_min_score,
            "object_memory_max_sentences": object_memory_max_sentences,
            "object_memory_max_objects": object_memory_max_objects,
            "object_memory_max_chars": object_memory_max_chars,
            "evidence_frontier_enabled": evidence_frontier_enabled,
            "evidence_frontier_active": evidence_frontier_active,
            "promotion_utility_veto_enabled": promotion_utility_veto_enabled,
            "promotion_utility_veto_min_utility": promotion_utility_veto_min_utility,
            "promotion_utility_veto_require_bridge_or_gap": (
                promotion_utility_veto_require_bridge_or_gap
            ),
            "promotion_utility_veto_apply_to_guarded": (
                promotion_utility_veto_apply_to_guarded
            ),
            "promotion_utility_veto_high_risk_only": (
                promotion_utility_veto_high_risk_only
            ),
            "promotion_utility_veto_append_on_reject": (
                promotion_utility_veto_append_on_reject
            ),
        },
        "candidate_count": len(ranked_candidates),
        "old_rank2_planned_evidence": (
            {
                "doc_id": old_rank2_planned_evidence.get("doc_id"),
                "query_count": int(old_rank2_planned_evidence.get("query_count", 1)),
                "plan_score": round(float(old_rank2_planned_evidence.get("plan_score", 0.0)), 6),
                "plan_rank": old_rank2_planned_evidence.get("plan_rank"),
                "query_overlap": round(float(old_rank2_planned_evidence.get("query_overlap", 0.0)), 6),
                "queries": old_rank2_planned_evidence.get("queries", [])[:6],
            }
            if old_rank2_planned_evidence is not None else None
        ),
        "rejected_count": 0,
        "reject_reason_counts": {},
        "rejections": [],
        "eligible_count": 0,
        "eligible_candidates": [],
        "pairwise_utility_search": {
            "enabled": utility_frontier_active,
            "mode": (
                "decoupled_evidence_frontier"
                if evidence_frontier_active and not pairwise_utility_search_enabled
                else "pairwise_utility"
            ),
            "applied": False,
            "reason": "disabled" if not utility_frontier_active else "not_evaluated",
            "candidate_count": 0,
            "accepted_count": 0,
            "selected_doc_id": None,
            "selected_source": None,
            "min_margin": pairwise_utility_min_margin,
            "min_pair_score": pairwise_utility_min_pair_score,
            "min_candidate_support": pairwise_utility_min_candidate_support,
            "allowed_sources": pairwise_utility_allowed_sources_normalized,
            "original_fallback_only": pairwise_utility_original_fallback_only,
            "query_schema_intervention_policy": query_schema_memory_intervention_policy,
            "query_schema_intervention_thresholds": {
                "min_score": round(float(query_schema_memory_top2_min_score), 6),
                "min_schema_idf": round(float(query_schema_memory_top2_min_schema_idf), 6),
                "min_overlap": round(float(query_schema_memory_top2_min_overlap), 6),
                "max_schema_frequency": int(query_schema_memory_top2_max_schema_frequency),
                "max_original_rank": int(query_schema_memory_top2_max_original_rank),
                "min_bridge_strength": int(query_schema_memory_top2_min_bridge_strength),
            },
            "query_schema_intervention_allow_top2_count": 0,
            "query_schema_intervention_append_only_count": 0,
            "query_schema_intervention_reject_count": 0,
            "query_schema_intervention_reason_counts": {},
            "bridge_slot_rescue_enabled": bridge_slot_rescue_enabled,
            "bridge_slot_rescue_min_score": bridge_slot_rescue_min_score,
            "bridge_slot_rescue_min_improvement": bridge_slot_rescue_min_improvement,
            "bridge_slot_rescue_min_anchor_overlap": bridge_slot_rescue_min_anchor_overlap,
            "bridge_slot_rescue_require_relation": bridge_slot_rescue_require_relation,
            "bridge_slot_rescue_require_old_deficit": bridge_slot_rescue_require_old_deficit,
            "bridge_slot_rescue_deficit_max_old_score": bridge_slot_rescue_deficit_max_old_score,
            "bridge_slot_rescue_min_missing_gain": bridge_slot_rescue_min_missing_gain,
            "bridge_slot_rescue_append_on_no_deficit": bridge_slot_rescue_append_on_no_deficit,
            "bridge_slot_rescue_accepted_count": 0,
            "bridge_slot_rescue_rejected_count": 0,
            "bridge_slot_append_accepted_count": 0,
            "bridge_slot_append_deferred_visible_count": 0,
            "bridge_slot_append_doc_ids": [],
            "bridge_slot_rescue_reject_reason_counts": {},
            "schema_verified_prior_applied_count": 0,
            "schema_verified_prior_rescue_count": 0,
            "schema_verified_prior_reject_reason_counts": {},
            "object_memory_accepted_count": 0,
            "object_memory_rejected_count": 0,
            "object_memory_minimal_sentence_count": 0,
            "object_memory_reject_reason_counts": {},
            "top_candidates": [],
        },
    }
    promotion_utility_veto_evaluated_count = 0
    promotion_utility_veto_rejected_count = 0
    promotion_utility_veto_reason_counts = Counter()
    promotion_utility_veto_debug: List[Dict[str, Any]] = []

    def promotion_utility_candidate_sources(item: Dict[str, Any]) -> set:
        sources = set()
        raw_sources = item.get("candidate_sources", [])
        if isinstance(raw_sources, str):
            raw_sources = [part.strip() for part in raw_sources.split(",")]
        for source in raw_sources or []:
            if source:
                sources.add(str(source))
        source = item.get("candidate_source")
        if source:
            sources.add(str(source))
        if item.get("schema_memory"):
            sources.add("query_schema_memory")
        if item.get("budgeted_evidence_subgraph_selected"):
            sources.add("budgeted_evidence_subgraph")
        if item.get("dual_channel_projection") or item.get("shadow_candidate"):
            sources.add("shadow")
        return sources

    def is_promotion_utility_high_risk(item: Dict[str, Any]) -> bool:
        sources = promotion_utility_candidate_sources(item)
        planned_sources = {"planned_retrieval", "planned_and_original"}
        high_risk_sources = {
            "query_schema_memory",
            "schema_memory",
            "budgeted_evidence_subgraph",
            "shadow",
            "shadow_retrieval",
            "dense_fallback",
            "decoupled_frontier",
        }
        if sources & high_risk_sources and not (sources & planned_sources):
            return True
        if item.get("query_schema_intervention_gate") and not (sources & planned_sources):
            return True
        return False

    def evaluate_promotion_utility_veto(
        item: Dict[str, Any],
        context: str,
    ) -> Dict[str, Any]:
        nonlocal promotion_utility_veto_evaluated_count
        nonlocal promotion_utility_veto_rejected_count
        candidate_sources = sorted(promotion_utility_candidate_sources(item))
        high_risk = is_promotion_utility_high_risk(item)
        info: Dict[str, Any] = {
            "enabled": promotion_utility_veto_enabled,
            "accepted": True,
            "reason": "disabled" if not promotion_utility_veto_enabled else "not_evaluated",
            "context": context,
            "doc_id": item.get("doc_id"),
            "candidate_sources": candidate_sources,
            "high_risk": bool(high_risk),
            "high_risk_only": bool(promotion_utility_veto_high_risk_only),
            "reject_reasons": [],
        }
        if not promotion_utility_veto_enabled:
            return info
        if promotion_utility_veto_high_risk_only and not high_risk:
            info.update({
                "accepted": True,
                "reason": "skipped_low_risk_candidate_no_veto",
            })
            if len(promotion_utility_veto_debug) < 12:
                promotion_utility_veto_debug.append(info)
            return info
        if doc_lookup is None:
            info.update({
                "accepted": True,
                "reason": "missing_doc_lookup_no_veto",
            })
            return info

        promotion_utility_veto_evaluated_count += 1
        selected_items, selector_info = _budgeted_evidence_subgraph_select(
            question=question,
            base_ids=base_ids,
            target_k=2,
            candidate_groups=[("promotion_candidate", [dict(item)])],
            ranked_candidates=[],
            doc_lookup=get_doc_text,
            max_append=1,
            max_candidates=1,
            min_utility=promotion_utility_veto_min_utility,
            redundancy_penalty=budgeted_evidence_subgraph_redundancy_penalty,
            require_bridge_or_gap=promotion_utility_veto_require_bridge_or_gap,
            intervention_gate=intervention_gate,
        )
        top_candidate = None
        if selector_info.get("top_candidates"):
            top_candidate = selector_info["top_candidates"][0]
        accepted = bool(
            selected_items
            and int(selected_items[0].get("doc_id")) == int(item.get("doc_id"))
        )
        reason = (
            "promotion_utility_passed"
            if accepted
            else f"promotion_utility_veto_{selector_info.get('reason', 'rejected')}"
        )
        reject_reasons = [] if accepted else [reason]
        if not accepted:
            promotion_utility_veto_rejected_count += 1
            promotion_utility_veto_reason_counts.update(reject_reasons)

        info.update({
            "accepted": accepted,
            "reason": reason,
            "reject_reasons": reject_reasons,
            "selector_reason": selector_info.get("reason"),
            "min_utility": round(float(promotion_utility_veto_min_utility), 6),
            "require_bridge_or_gap": bool(promotion_utility_veto_require_bridge_or_gap),
            "score": (
                top_candidate.get("score")
                if isinstance(top_candidate, dict) else None
            ),
            "base_utility": (
                top_candidate.get("base_utility")
                if isinstance(top_candidate, dict) else None
            ),
            "query_overlap": (
                top_candidate.get("query_overlap")
                if isinstance(top_candidate, dict) else None
            ),
            "schema_signal": (
                top_candidate.get("schema_signal")
                if isinstance(top_candidate, dict) else None
            ),
            "bridge_signal": (
                top_candidate.get("bridge_signal")
                if isinstance(top_candidate, dict) else None
            ),
            "missing_gain": (
                top_candidate.get("missing_gain")
                if isinstance(top_candidate, dict) else None
            ),
            "source_counts": selector_info.get("source_counts", {}),
        })
        if len(promotion_utility_veto_debug) < 12:
            promotion_utility_veto_debug.append(info)
        return info

    def record_promotion_utility_soft_append(
        item: Dict[str, Any],
        veto_info: Optional[Dict[str, Any]],
        context: str,
    ) -> None:
        nonlocal promotion_utility_soft_append_count
        if not promotion_utility_veto_append_on_reject:
            return
        try:
            doc_id = int(item.get("doc_id"))
        except (TypeError, ValueError):
            return
        if doc_id in promotion_utility_soft_append_seen:
            return
        if doc_id in base_ids[:target_k]:
            return
        append_item = dict(item)
        append_item["promotion_utility_soft_append"] = True
        append_item["promotion_utility_soft_append_context"] = context
        append_item["promotion_utility_veto"] = veto_info
        promotion_utility_soft_append_items.append(append_item)
        promotion_utility_soft_append_seen.add(doc_id)
        promotion_utility_soft_append_count += 1

    if top2_override_enabled or evidence_frontier_active:
        if target_k < 2 or len(base_ids) < 2:
            top2_override_info["reason"] = "not_enough_ranked_docs"
        elif merge_mode != "guarded_promote" and not evidence_frontier_active:
            top2_override_info["reason"] = f"merge_mode_{merge_mode}_not_eligible"
        else:
            top2_override_info["reason"] = "no_candidate_accepted"
            top2_eligible_items = []
            for item in ranked_candidates:
                original_rank = item.get("original_rank")
                reject_reasons = []
                if item["doc_id"] == base_ids[0]:
                    reject_reasons.append("would_replace_rank1")
                if item["doc_id"] in base_ids[:2]:
                    reject_reasons.append("already_in_top2")
                if item["plan_score"] < top2_override_min_score:
                    reject_reasons.append("low_top2_score")
                if item["plan_rank"] >= top2_override_max_plan_rank:
                    reject_reasons.append("low_planned_rank")
                if original_rank is None or original_rank < 2 or original_rank >= top2_override_max_original_rank:
                    reject_reasons.append("outside_original_rank_window")
                if item.get("query_overlap", 0.0) < top2_override_min_query_overlap:
                    reject_reasons.append("weak_query_anchor_overlap")
                if int(item.get("query_count", 1)) < top2_override_min_query_hits:
                    reject_reasons.append("insufficient_query_agreement")

                pair_verifier_info = None
                promotion_utility_veto_info = None
                if not reject_reasons and (top2_pair_verifier_enabled or relation_object_guard_enabled):
                    pair_verifier_info = _counterfactual_top2_pair_verifier(
                        question=question,
                        top1_doc=get_doc_text(base_ids[0]),
                        old_rank2_doc=get_doc_text(base_ids[1]),
                        candidate_doc=get_doc_text(item["doc_id"]),
                        candidate_item=item,
                        old_rank2_item=old_rank2_planned_evidence,
                        min_margin=top2_pair_verifier_min_margin,
                        max_coverage_loss=top2_pair_verifier_max_coverage_loss,
                        min_candidate_query_support=top2_pair_verifier_min_candidate_query_support,
                    )
                    item["top2_pair_verifier"] = pair_verifier_info
                    if top2_pair_verifier_enabled and top2_pair_verifier_mode == "hard" and not pair_verifier_info.get("accepted", False):
                        reject_reasons.extend(pair_verifier_info.get("reject_reasons", []) or [
                            "pair_verifier_rejected"
                        ])

                relation_object_guard_info = None
                if not reject_reasons and relation_object_guard_enabled:
                    relation_object_guard_info = _relation_object_consistency_guard(
                        question=question,
                        planned_query=_item_query_text(item),
                        old_rank2_doc=get_doc_text(base_ids[1]),
                        candidate_doc=get_doc_text(item["doc_id"]),
                        pair_verifier_info=pair_verifier_info,
                        min_relation_score=relation_object_guard_min_score,
                        min_relation_margin=relation_object_guard_min_margin,
                        entity_conditioned=entity_conditioned_relation_guard_enabled,
                        entity_anchor_min_overlap=entity_relation_anchor_min_overlap,
                    )
                    item["relation_object_guard"] = relation_object_guard_info
                    if not relation_object_guard_info.get("accepted", True):
                        reject_reasons.extend(relation_object_guard_info.get("reject_reasons", []) or [
                            "relation_object_guard_rejected"
                        ])

                if not reject_reasons and promotion_utility_veto_enabled:
                    promotion_utility_veto_info = evaluate_promotion_utility_veto(
                        item,
                        context="top2_override",
                    )
                    item["promotion_utility_veto"] = promotion_utility_veto_info
                    if not promotion_utility_veto_info.get("accepted", True):
                        record_promotion_utility_soft_append(
                            item,
                            promotion_utility_veto_info,
                            context="top2_override",
                        )
                        reject_reasons.extend(
                            promotion_utility_veto_info.get("reject_reasons", [])
                            or ["promotion_utility_veto_rejected"]
                        )

                if reject_reasons:
                    top2_override_rejection_counts.update(reject_reasons)
                    if len(top2_override_rejections) < 12:
                        top2_override_rejections.append({
                            "doc_id": item["doc_id"],
                            "query": item["query"],
                            "plan_rank": item["plan_rank"],
                            "original_rank": item["original_rank"],
                            "plan_score": round(item["plan_score"], 6),
                            "query_overlap": round(item.get("query_overlap", 0.0), 6),
                            "query_count": int(item.get("query_count", 1)),
                            "reasons": reject_reasons,
                            "pair_verifier": pair_verifier_info,
                            "relation_object_guard": relation_object_guard_info,
                            "promotion_utility_veto": promotion_utility_veto_info,
                        })
                    continue

                top2_selection_score = float(item.get("plan_score", 0.0))
                if top2_pair_verifier_enabled and top2_pair_verifier_mode == "soft" and pair_verifier_info:
                    pair_margin = float(pair_verifier_info.get("pair_margin", 0.0) or 0.0)
                    coverage_loss = max(0.0, float(pair_verifier_info.get("coverage_loss", 0.0) or 0.0))
                    candidate_features = pair_verifier_info.get("candidate") or {}
                    query_support = float(candidate_features.get("query_support", 0.0) or 0.0)
                    query_support_deficit = max(
                        0.0,
                        top2_pair_verifier_min_candidate_query_support - query_support,
                    )
                    top2_selection_score += top2_pair_verifier_score_weight * pair_margin
                    top2_selection_score -= 0.50 * top2_pair_verifier_score_weight * coverage_loss
                    top2_selection_score -= 0.35 * top2_pair_verifier_score_weight * query_support_deficit
                item["top2_selection_score"] = top2_selection_score
                top2_eligible_items.append(item)

            if utility_frontier_active:
                utility_pool: Dict[int, Dict[str, Any]] = {}
                for item in ranked_candidates:
                    original_rank = item.get("original_rank")
                    if original_rank is None or original_rank < 2 or original_rank >= pairwise_utility_max_original_rank:
                        continue
                    candidate = dict(item)
                    candidate["candidate_source"] = "planned_retrieval"
                    candidate["candidate_sources"] = ["planned_retrieval"]
                    utility_pool[int(candidate["doc_id"])] = candidate

                schema_memory_candidate_count = 0
                schema_memory_duplicate_count = 0
                for schema_rank, schema_item_raw in enumerate(schema_memory_candidates):
                    schema_item = dict(schema_item_raw or {})
                    try:
                        doc_id = int(schema_item.get("doc_id"))
                    except (TypeError, ValueError):
                        continue
                    original_rank = base_rank.get(doc_id, schema_item.get("original_rank"))
                    try:
                        original_rank = int(original_rank)
                    except (TypeError, ValueError):
                        original_rank = None
                    if original_rank is None or original_rank < 2 or original_rank >= pairwise_utility_max_original_rank:
                        continue
                    schema_score = float(schema_item.get("score", 0.0) or 0.0)
                    if schema_score <= 0.0:
                        continue
                    schema_memory_candidate_count += 1
                    query_overlap = max(
                        float(schema_item.get("query_overlap", 0.0) or 0.0),
                        float(schema_item.get("context_overlap", 0.0) or 0.0),
                    )
                    plan_score = 0.42 + 0.58 * _clamp(schema_score)
                    existing = utility_pool.get(doc_id)
                    if existing is not None:
                        schema_memory_duplicate_count += 1
                        existing.setdefault("candidate_sources", [existing.get("candidate_source")])
                        if "query_schema_memory" not in existing["candidate_sources"]:
                            existing["candidate_sources"].append("query_schema_memory")
                        existing["schema_memory"] = schema_item
                        existing["plan_score"] = max(float(existing.get("plan_score", 0.0)), plan_score)
                        existing["query_overlap"] = max(float(existing.get("query_overlap", 0.0)), query_overlap)
                        existing["query_count"] = max(int(existing.get("query_count", 0) or 0), 1)
                        if not existing.get("query"):
                            existing["query"] = "query schema memory"
                        if not existing.get("queries"):
                            existing["queries"] = ["query schema memory"]
                        if existing.get("candidate_source") == "original_retrieval":
                            existing["candidate_source"] = "query_schema_memory"
                        continue
                    utility_pool[doc_id] = {
                        "doc_id": doc_id,
                        "query": "query schema memory",
                        "queries": ["query schema memory"],
                        "query_count": 1,
                        "plan_rank": int(schema_rank),
                        "plan_score": plan_score,
                        "retrieval_score": base_score.get(doc_id, 0.0),
                        "normalized_score": _clamp(schema_score),
                        "calibrated_score": _clamp(schema_score),
                        "original_rank": original_rank,
                        "query_overlap": query_overlap,
                        "candidate_source": "query_schema_memory",
                        "candidate_sources": ["query_schema_memory"],
                        "schema_memory": schema_item,
                    }

                original_limit = min(pairwise_utility_max_original_rank, len(base_ids))
                for original_rank, doc_id in enumerate(base_ids[:original_limit]):
                    if original_rank < 2:
                        continue
                    if doc_id in utility_pool:
                        utility_pool[doc_id].setdefault("candidate_sources", [utility_pool[doc_id].get("candidate_source")])
                        if "original_retrieval" not in utility_pool[doc_id]["candidate_sources"]:
                            utility_pool[doc_id]["candidate_sources"].append("original_retrieval")
                        if "planned_retrieval" in utility_pool[doc_id]["candidate_sources"]:
                            utility_pool[doc_id]["candidate_source"] = "planned_and_original"
                        continue
                    retrieval_score = base_score.get(doc_id, 0.0)
                    rank_prior = _rank_prior(original_rank)
                    utility_pool[doc_id] = {
                        "doc_id": doc_id,
                        "query": "",
                        "queries": [],
                        "query_count": 0,
                        "plan_rank": 999999,
                        "plan_score": 0.50 + 0.24 * rank_prior,
                        "retrieval_score": retrieval_score,
                        "normalized_score": 0.0,
                        "calibrated_score": 0.0,
                        "original_rank": original_rank,
                        "query_overlap": 0.0,
                        "candidate_source": "original_retrieval",
                        "candidate_sources": ["original_retrieval"],
                    }

                query_schema_intervention_counts = Counter()
                query_schema_intervention_reason_counts = Counter()
                for item in utility_pool.values():
                    gate_info = _query_schema_memory_intervention_gate(
                        item,
                        intervention_gate=intervention_gate,
                        policy=query_schema_memory_intervention_policy,
                        min_top2_score=query_schema_memory_top2_min_score,
                        min_top2_schema_idf=query_schema_memory_top2_min_schema_idf,
                        min_top2_overlap=query_schema_memory_top2_min_overlap,
                        max_top2_schema_frequency=query_schema_memory_top2_max_schema_frequency,
                        max_top2_original_rank=query_schema_memory_top2_max_original_rank,
                        min_top2_bridge_strength=query_schema_memory_top2_min_bridge_strength,
                    )
                    item["query_schema_intervention_gate"] = gate_info
                    if "query_schema_memory" in set(item.get("candidate_sources", []) or []):
                        action = str(gate_info.get("action", "allow_top2") or "allow_top2")
                        query_schema_intervention_counts[action] += 1
                        query_schema_intervention_reason_counts[
                            str(gate_info.get("reason", "unknown") or "unknown")
                        ] += 1

                unfiltered_utility_candidate_count = len(utility_pool)
                source_filtered_count = 0
                for doc_id in list(utility_pool.keys()):
                    source = str(utility_pool[doc_id].get("candidate_source") or "")
                    candidate_sources = {
                        str(item)
                        for item in utility_pool[doc_id].get("candidate_sources", []) or []
                        if item
                    }
                    candidate_sources.add(source)
                    if not (candidate_sources & pairwise_utility_allowed_source_set):
                        source_filtered_count += 1
                        del utility_pool[doc_id]

                utility_candidates = sorted(
                    utility_pool.values(),
                    key=lambda item: (
                        0 if "planned_retrieval" in set(item.get("candidate_sources", []) or []) else
                        1 if "query_schema_memory" in set(item.get("candidate_sources", []) or []) else
                        2,
                        -float(item.get("plan_score", 0.0)),
                        item.get("original_rank") if item.get("original_rank") is not None else 999999,
                    ),
                )[:pairwise_utility_max_candidates]
                support_set_utility_candidates = utility_candidates
                utility_eligible_items = []
                utility_debug = []
                bridge_slot_rescue_reject_counts = Counter()
                bridge_slot_rescue_accepted_count = 0
                bridge_slot_rescue_rejected_count = 0
                bridge_slot_append_accepted_count = 0
                bridge_slot_append_deferred_visible_count = 0
                object_memory_accepted_count = 0
                object_memory_rejected_count = 0
                object_memory_minimal_sentence_count = 0
                object_memory_reject_counts = Counter()
                schema_verified_prior_applied_count = 0
                schema_verified_prior_rescue_count = 0
                schema_verified_prior_reject_counts = Counter()
                schema_verified_prior_bonus_total = 0.0
                schema_deficit_promotion_accepted_count = 0
                schema_deficit_promotion_rejected_count = 0
                schema_deficit_promotion_reason_counts = Counter()
                schema_deficit_promotion_reject_counts = Counter()
                schema_deficit_candidate_debug = []
                schema_deficit_soft_append_accepted_count = 0
                schema_deficit_soft_append_rejected_count = 0
                schema_deficit_soft_append_reason_counts = Counter()
                schema_deficit_soft_append_reject_counts = Counter()
                schema_reliability_evaluated_count = 0
                schema_reliability_passed_count = 0
                schema_reliability_rescued_count = 0
                schema_reliability_rejected_count = 0
                schema_reliability_reason_counts = Counter()
                schema_reliability_reject_counts = Counter()
                answer_type_conflict_guard_evaluated_count = 0
                answer_type_conflict_guard_rejected_count = 0
                answer_type_conflict_guard_reject_counts = Counter()
                rescuable_pairwise_rejections = {
                    "pair_verifier_low_margin",
                    "pair_verifier_coverage_loss",
                    "pair_verifier_complementarity_loss",
                    "pair_verifier_weak_candidate_query_support",
                    "pairwise_utility_low_pair_score",
                    "query_schema_risk_gate_append_only",
                    "query_schema_risk_gate_rejected",
                }
                for item in utility_candidates:
                    if item["doc_id"] in base_ids[:2]:
                        continue
                    query_schema_gate_info = item.get("query_schema_intervention_gate") or {}
                    query_schema_gate_action = str(
                        query_schema_gate_info.get("action", "allow_top2") or "allow_top2"
                    )
                    query_schema_gate_append_only = query_schema_gate_action == "append_only"
                    pair_verifier_info = _counterfactual_top2_pair_verifier(
                        question=question,
                        top1_doc=get_doc_text(base_ids[0]),
                        old_rank2_doc=get_doc_text(base_ids[1]),
                        candidate_doc=get_doc_text(item["doc_id"]),
                        candidate_item=item,
                        old_rank2_item=old_rank2_planned_evidence,
                        min_margin=pairwise_utility_min_margin,
                        max_coverage_loss=top2_pair_verifier_max_coverage_loss,
                        min_candidate_query_support=pairwise_utility_min_candidate_support,
                    )
                    reject_reasons = list(pair_verifier_info.get("reject_reasons", []) or [])
                    candidate_pair_score = float(pair_verifier_info.get("candidate_pair_score", 0.0) or 0.0)
                    pair_margin = float(pair_verifier_info.get("pair_margin", 0.0) or 0.0)
                    candidate_features = pair_verifier_info.get("candidate") or {}
                    if candidate_pair_score < pairwise_utility_min_pair_score:
                        reject_reasons.append("pairwise_utility_low_pair_score")
                    if query_schema_gate_action == "append_only":
                        reject_reasons.append("query_schema_risk_gate_append_only")
                    elif query_schema_gate_action == "reject":
                        reject_reasons.append("query_schema_risk_gate_rejected")

                    bridge_slot_rescue_info = None
                    bridge_slot_rescued = False
                    object_memory_info = None
                    original_reject_reasons = list(reject_reasons)
                    if bridge_slot_rescue_enabled and reject_reasons:
                        bridge_slot_rescue_info = _bridge_slot_coverage_score(
                            question=question,
                            top1_doc=get_doc_text(base_ids[0]),
                            old_rank2_doc=get_doc_text(base_ids[1]),
                            candidate_doc=get_doc_text(item["doc_id"]),
                            candidate_item=item,
                            old_rank2_item=old_rank2_planned_evidence,
                            min_score=bridge_slot_rescue_min_score,
                            min_improvement=bridge_slot_rescue_min_improvement,
                            min_anchor_overlap=bridge_slot_rescue_min_anchor_overlap,
                            require_relation=bridge_slot_rescue_require_relation,
                            require_old_deficit=bridge_slot_rescue_require_old_deficit,
                            deficit_max_old_score=bridge_slot_rescue_deficit_max_old_score,
                            min_missing_gain=bridge_slot_rescue_min_missing_gain,
                            append_on_no_deficit=bridge_slot_rescue_append_on_no_deficit,
                        )
                        unsupported_rejections = [
                            reason for reason in reject_reasons
                            if reason not in rescuable_pairwise_rejections
                        ]
                        needs_object_memory = (
                            object_memory_enabled
                            and (
                                bridge_slot_rescue_info.get("accepted")
                                or bridge_slot_rescue_info.get("append_accepted")
                            )
                            and not unsupported_rejections
                        )
                        if needs_object_memory:
                            object_memory_info = _object_consistent_evidence_memory_gate(
                                question=question,
                                planned_query=_item_query_text(item),
                                top1_doc=get_doc_text(base_ids[0]),
                                old_rank2_doc=get_doc_text(base_ids[1]),
                                candidate_doc=get_doc_text(item["doc_id"]),
                                bridge_slot_info=bridge_slot_rescue_info,
                                min_score=object_memory_min_score,
                                max_sentences=object_memory_max_sentences,
                                max_objects=object_memory_max_objects,
                                max_chars=object_memory_max_chars,
                            )
                            item["object_consistent_memory"] = object_memory_info
                            if object_memory_info.get("accepted"):
                                object_memory_accepted_count += 1
                                object_memory_minimal_sentence_count += int(
                                    object_memory_info.get("sentence_count", 0) or 0
                                )
                                item["object_memory_doc"] = object_memory_info.get("memory_doc", "")
                            else:
                                object_memory_rejected_count += 1
                                object_memory_reject_counts.update(
                                    object_memory_info.get("reject_reasons", []) or [
                                        "object_memory_rejected"
                                    ]
                                )

                        object_memory_passed = (
                            not object_memory_enabled
                            or not (
                                bridge_slot_rescue_info.get("accepted")
                                or bridge_slot_rescue_info.get("append_accepted")
                            )
                            or bool((object_memory_info or {}).get("accepted", False))
                        )
                        if (
                            bridge_slot_rescue_info.get("accepted")
                            and not query_schema_gate_append_only
                            and not unsupported_rejections
                            and object_memory_passed
                        ):
                            reject_reasons = []
                            bridge_slot_rescued = True
                            bridge_slot_rescue_accepted_count += 1
                        else:
                            bridge_slot_rescue_rejected_count += 1
                            bridge_slot_rescue_reject_counts.update(
                                bridge_slot_rescue_info.get("reject_reasons", []) or [
                                    "bridge_slot_rescue_rejected"
                                ]
                            )
                            bridge_slot_rescue_reject_counts.update(unsupported_rejections)
                            if object_memory_info and not object_memory_info.get("accepted"):
                                bridge_slot_rescue_reject_counts.update(
                                    object_memory_info.get("reject_reasons", []) or [
                                        "object_memory_rejected"
                                    ]
                                )
                            if (
                                bridge_slot_rescue_info.get("append_accepted")
                                and not unsupported_rejections
                                and object_memory_passed
                            ):
                                if item["doc_id"] in base_ids[:target_k]:
                                    bridge_slot_append_deferred_visible_count += 1
                                elif item["doc_id"] not in bridge_slot_append_seen:
                                    item["bridge_slot_append_only"] = True
                                    bridge_slot_append_items.append(item)
                                    bridge_slot_append_seen.add(item["doc_id"])
                                    bridge_slot_append_accepted_count += 1

                    schema_deficit_info = None
                    candidate_source_set = {
                        str(source)
                        for source in (item.get("candidate_sources", []) or [])
                        if source
                    }
                    if item.get("candidate_source"):
                        candidate_source_set.add(str(item.get("candidate_source")))
                    has_query_schema_signal = (
                        "query_schema_memory" in candidate_source_set
                        or bool(item.get("schema_memory"))
                    )
                    if schema_deficit_promotion_enabled and reject_reasons and has_query_schema_signal:
                        if bridge_slot_rescue_info is None:
                            bridge_slot_rescue_info = _bridge_slot_coverage_score(
                                question=question,
                                top1_doc=get_doc_text(base_ids[0]),
                                old_rank2_doc=get_doc_text(base_ids[1]),
                                candidate_doc=get_doc_text(item["doc_id"]),
                                candidate_item=item,
                                old_rank2_item=old_rank2_planned_evidence,
                                min_score=bridge_slot_rescue_min_score,
                                min_improvement=bridge_slot_rescue_min_improvement,
                                min_anchor_overlap=bridge_slot_rescue_min_anchor_overlap,
                                require_relation=bridge_slot_rescue_require_relation,
                                require_old_deficit=bridge_slot_rescue_require_old_deficit,
                                deficit_max_old_score=bridge_slot_rescue_deficit_max_old_score,
                                min_missing_gain=bridge_slot_rescue_min_missing_gain,
                                append_on_no_deficit=bridge_slot_rescue_append_on_no_deficit,
                            )
                        schema_deficit_info = _schema_deficit_promotion_gate(
                            question=question,
                            top1_doc=get_doc_text(base_ids[0]),
                            old_rank2_doc=get_doc_text(base_ids[1]),
                            candidate_doc=get_doc_text(item["doc_id"]),
                            candidate_item=item,
                            pair_verifier_info=pair_verifier_info,
                            query_schema_gate_info=query_schema_gate_info,
                            bridge_slot_info=bridge_slot_rescue_info,
                            intervention_gate=intervention_gate,
                            enabled=schema_deficit_promotion_enabled,
                            min_score=schema_deficit_min_score,
                            min_gain=schema_deficit_min_gain,
                            min_schema_score=schema_deficit_min_schema_score,
                            min_schema_idf=schema_deficit_min_schema_idf,
                            max_original_rank=schema_deficit_max_original_rank,
                            require_bridge_constraint=schema_deficit_require_bridge_constraint,
                            min_bridge_score=schema_deficit_min_bridge_score,
                            min_bridge_margin=schema_deficit_min_bridge_margin,
                            min_bridge_missing_gain=schema_deficit_min_bridge_missing_gain,
                            min_bridge_anchor_overlap=schema_deficit_min_bridge_anchor_overlap,
                        )
                        schema_deficit_promotion_reason_counts.update([
                            schema_deficit_info.get("reason", "schema_deficit_unknown")
                        ])
                        schema_deficit_rescuable = {
                            "query_schema_risk_gate_append_only",
                            "query_schema_risk_gate_rejected",
                            "pair_verifier_low_margin",
                            "pair_verifier_coverage_loss",
                            "pair_verifier_complementarity_loss",
                            "pair_verifier_weak_candidate_query_support",
                            "pairwise_utility_low_pair_score",
                        }
                        if schema_deficit_info.get("accepted"):
                            rescued_reasons = [
                                reason for reason in reject_reasons
                                if reason in schema_deficit_rescuable
                            ]
                            remaining_reasons = [
                                reason for reason in reject_reasons
                                if reason not in schema_deficit_rescuable
                            ]
                            schema_deficit_info["rescued_reject_reasons"] = rescued_reasons[:6]
                            schema_deficit_info["remaining_reject_reasons"] = remaining_reasons[:6]
                            schema_deficit_info["effective_accepted"] = not remaining_reasons
                            reject_reasons = remaining_reasons
                            if schema_deficit_info["effective_accepted"]:
                                schema_deficit_promotion_accepted_count += 1
                                query_schema_gate_action = "allow_top2"
                                query_schema_gate_append_only = False
                                query_schema_gate_info = dict(query_schema_gate_info or {})
                                query_schema_gate_info.update({
                                    "action": "allow_top2",
                                    "reason": "schema_deficit_promoted",
                                    "schema_deficit_promotion": schema_deficit_info,
                                })
                                item["query_schema_intervention_gate"] = query_schema_gate_info
                            else:
                                schema_deficit_promotion_rejected_count += 1
                                schema_deficit_promotion_reject_counts.update(remaining_reasons)
                        else:
                            schema_deficit_info["effective_accepted"] = False
                            schema_deficit_promotion_rejected_count += 1
                            schema_deficit_promotion_reject_counts.update(
                                schema_deficit_info.get("reject_reasons", []) or [
                                    schema_deficit_info.get("reason", "schema_deficit_rejected")
                                ]
                            )

                        schema_soft_append_info = _schema_deficit_soft_append_gate(
                            candidate_item=item,
                            schema_deficit_info=schema_deficit_info,
                            intervention_gate=intervention_gate,
                            enabled=schema_deficit_soft_append_enabled,
                            min_score=schema_deficit_soft_append_min_score,
                            min_schema_score=schema_deficit_soft_append_min_schema_score,
                            min_schema_idf=schema_deficit_soft_append_min_schema_idf,
                            min_query_support=schema_deficit_soft_append_min_query_support,
                            max_original_rank=schema_deficit_soft_append_max_original_rank,
                        )
                        item["schema_deficit_soft_append"] = schema_soft_append_info
                        schema_deficit_soft_append_reason_counts.update([
                            schema_soft_append_info.get("reason", "schema_soft_append_unknown")
                        ])
                        schema_reliability_info = _schema_reliability_gate(
                            question=question,
                            candidate_doc=get_doc_text(item["doc_id"]),
                            candidate_item=item,
                            schema_deficit_info=schema_deficit_info,
                            soft_append_info=schema_soft_append_info,
                            intervention_gate=intervention_gate,
                            enabled=schema_reliability_gate_enabled,
                            min_score=schema_reliability_min_score,
                            rescue_min_score=schema_reliability_rescue_min_score,
                            min_anchor_overlap=schema_reliability_min_anchor_overlap,
                            max_schema_frequency=schema_reliability_max_schema_frequency,
                            numeric_requires_value=schema_reliability_numeric_requires_value,
                        )
                        item["schema_reliability_gate"] = schema_reliability_info
                        schema_soft_append_info["schema_reliability_gate"] = schema_reliability_info
                        if schema_reliability_gate_enabled:
                            schema_reliability_evaluated_count += 1
                            schema_reliability_reason_counts.update([
                                schema_reliability_info.get("reason", "schema_reliability_unknown")
                            ])
                            if schema_reliability_info.get("rescued"):
                                schema_reliability_rescued_count += 1
                                schema_soft_append_info.update({
                                    "accepted": True,
                                    "reason": "schema_reliability_rescued",
                                    "rescued_by_schema_reliability": True,
                                })
                            elif schema_reliability_info.get("accepted"):
                                schema_reliability_passed_count += 1
                            else:
                                schema_reliability_rejected_count += 1
                                schema_reliability_reject_counts.update(
                                    schema_reliability_info.get("reject_reasons", []) or [
                                        schema_reliability_info.get(
                                            "reason",
                                            "schema_reliability_rejected",
                                        )
                                    ]
                                )
                                schema_soft_append_info.update({
                                    "accepted": False,
                                    "reason": "schema_reliability_rejected",
                                    "reject_reasons": (
                                        schema_soft_append_info.get("reject_reasons", []) or []
                                    ) + (
                                        schema_reliability_info.get("reject_reasons", []) or [
                                            "schema_reliability_rejected"
                                        ]
                                    ),
                                })
                        if schema_soft_append_info.get("accepted"):
                            if item["doc_id"] not in base_ids[:target_k] and item["doc_id"] not in schema_deficit_soft_append_seen:
                                item["schema_deficit_append_only"] = True
                                schema_deficit_soft_append_items.append(item)
                                schema_deficit_soft_append_seen.add(item["doc_id"])
                                schema_deficit_soft_append_accepted_count += 1
                            else:
                                schema_deficit_soft_append_rejected_count += 1
                                schema_deficit_soft_append_reject_counts.update([
                                    "schema_soft_append_already_visible_or_seen"
                                ])
                        else:
                            schema_deficit_soft_append_rejected_count += 1
                            schema_deficit_soft_append_reject_counts.update(
                                schema_soft_append_info.get("reject_reasons", []) or [
                                    schema_soft_append_info.get("reason", "schema_soft_append_rejected")
                                ]
                            )

                    schema_prior_info = _schema_verified_utility_prior(
                        question=question,
                        candidate_doc=get_doc_text(item["doc_id"]),
                        candidate_item=item,
                        candidate_features=candidate_features,
                        intervention_gate=intervention_gate,
                        enabled=schema_verified_prior_enabled,
                        max_bonus=schema_verified_prior_bonus,
                        min_schema_score=schema_verified_prior_min_schema_score,
                        min_schema_idf=schema_verified_prior_min_schema_idf,
                        min_overlap=schema_verified_prior_min_overlap,
                        max_original_rank=schema_verified_prior_max_original_rank,
                        numeric_requires_value=schema_verified_prior_numeric_requires_value,
                        block_long_context=schema_verified_prior_block_long_context,
                        rescue_enabled=schema_verified_prior_rescue_enabled,
                        min_rescue_bonus=schema_verified_prior_min_rescue_bonus,
                    )
                    if schema_prior_info.get("accepted"):
                        schema_verified_prior_applied_count += 1
                        schema_verified_prior_bonus_total += float(schema_prior_info.get("bonus", 0.0) or 0.0)
                        non_rescuable_schema_rejections = [
                            reason for reason in reject_reasons
                            if reason not in SCHEMA_VERIFIED_PRIOR_RESCUABLE_REJECTIONS
                        ]
                        if (
                            reject_reasons
                            and schema_prior_info.get("rescue_allowed")
                            and not non_rescuable_schema_rejections
                        ):
                            schema_prior_info["rescued"] = True
                            schema_prior_info["rescued_reject_reasons"] = reject_reasons[:6]
                            reject_reasons = []
                            schema_verified_prior_rescue_count += 1
                    else:
                        schema_verified_prior_reject_counts.update(
                            schema_prior_info.get("reject_reasons", []) or [
                                schema_prior_info.get("reason", "schema_prior_rejected")
                            ]
                        )

                    relation_object_guard_info = None
                    if not reject_reasons and relation_object_guard_enabled:
                        relation_object_guard_info = _relation_object_consistency_guard(
                            question=question,
                            planned_query=_item_query_text(item),
                            old_rank2_doc=get_doc_text(base_ids[1]),
                            candidate_doc=get_doc_text(item["doc_id"]),
                            pair_verifier_info=pair_verifier_info,
                            min_relation_score=relation_object_guard_min_score,
                            min_relation_margin=relation_object_guard_min_margin,
                            entity_conditioned=entity_conditioned_relation_guard_enabled,
                            entity_anchor_min_overlap=entity_relation_anchor_min_overlap,
                        )
                        if not relation_object_guard_info.get("accepted", True):
                            reject_reasons.extend(relation_object_guard_info.get("reject_reasons", []) or [
                                "relation_object_guard_rejected"
                            ])

                    answer_type_conflict_guard_info = None
                    if not reject_reasons and answer_type_conflict_guard_enabled:
                        answer_type_conflict_guard_info = _answer_type_conflict_guard(
                            question=question,
                            planned_query=_item_query_text(item),
                            old_rank2_doc=get_doc_text(base_ids[1]),
                            candidate_doc=get_doc_text(item["doc_id"]),
                            pair_verifier_info=pair_verifier_info,
                            min_old_signal=answer_type_conflict_guard_min_old_signal,
                            min_candidate_signal=answer_type_conflict_guard_min_candidate_signal,
                            min_conflict_margin=answer_type_conflict_guard_min_conflict_margin,
                            numeric_only=answer_type_conflict_guard_numeric_only,
                        )
                        answer_type_conflict_guard_evaluated_count += 1
                        if not answer_type_conflict_guard_info.get("accepted", True):
                            guard_reasons = (
                                answer_type_conflict_guard_info.get("reject_reasons", [])
                                or ["answer_type_conflict_guard_rejected"]
                            )
                            reject_reasons.extend(guard_reasons)
                            answer_type_conflict_guard_rejected_count += 1
                            answer_type_conflict_guard_reject_counts.update(guard_reasons)

                    promotion_utility_veto_info = None
                    if not reject_reasons and promotion_utility_veto_enabled:
                        promotion_utility_veto_info = evaluate_promotion_utility_veto(
                            item,
                            context="pairwise_utility_top2",
                        )
                        if not promotion_utility_veto_info.get("accepted", True):
                            record_promotion_utility_soft_append(
                                item,
                                promotion_utility_veto_info,
                                context="pairwise_utility_top2",
                            )
                            reject_reasons.extend(
                                promotion_utility_veto_info.get("reject_reasons", [])
                                or ["promotion_utility_veto_rejected"]
                            )

                    rank_prior = _rank_prior(item.get("original_rank"))
                    utility_score = (
                        candidate_pair_score
                        + 0.40 * pair_margin
                        + 0.08 * rank_prior
                        + 0.04 * _clamp(float(item.get("plan_score", 0.0)) / 1.20)
                        + 0.04 * float(candidate_features.get("query_agreement", 0.0) or 0.0)
                        + float(schema_prior_info.get("bonus", 0.0) or 0.0)
                    )
                    if bridge_slot_rescued and bridge_slot_rescue_info:
                        utility_score = max(
                            utility_score,
                            float(bridge_slot_rescue_info.get("score", 0.0) or 0.0)
                            + 0.30 * float(bridge_slot_rescue_info.get("margin", 0.0) or 0.0)
                            + 0.04 * rank_prior,
                        )
                    item["pairwise_utility_search"] = True
                    item["top2_pair_verifier"] = pair_verifier_info
                    item["relation_object_guard"] = relation_object_guard_info
                    item["bridge_slot_rescue"] = bridge_slot_rescue_info
                    item["bridge_slot_rescued"] = bridge_slot_rescued
                    item["object_consistent_memory"] = object_memory_info
                    item["schema_deficit_promotion"] = schema_deficit_info
                    item["schema_verified_prior"] = schema_prior_info
                    item["query_schema_intervention_gate"] = query_schema_gate_info
                    item["answer_type_conflict_guard"] = answer_type_conflict_guard_info
                    item["promotion_utility_veto"] = promotion_utility_veto_info
                    item["top2_selection_score"] = utility_score
                    item["pairwise_utility_score"] = utility_score
                    item["pairwise_utility_margin"] = pair_margin
                    item["pairwise_utility_pair_score"] = candidate_pair_score

                    summary = {
                        "doc_id": item["doc_id"],
                        "source": item.get("candidate_source"),
                        "sources": item.get("candidate_sources", []),
                        "original_rank": item.get("original_rank"),
                        "plan_rank": item.get("plan_rank"),
                        "plan_score": round(float(item.get("plan_score", 0.0)), 6),
                        "pair_score": round(candidate_pair_score, 6),
                        "pair_margin": round(pair_margin, 6),
                        "utility_score": round(utility_score, 6),
                        "query_support": round(float(candidate_features.get("query_support", 0.0) or 0.0), 6),
                        "schema_prior_bonus": round(float(schema_prior_info.get("bonus", 0.0) or 0.0), 6),
                        "schema_prior_reason": schema_prior_info.get("reason"),
                        "schema_prior_rescued": bool(schema_prior_info.get("rescued", False)),
                        "schema_deficit_reason": (
                            (schema_deficit_info or {}).get("reason")
                        ),
                        "schema_deficit_score": (
                            round(float((schema_deficit_info or {}).get("score", 0.0)), 6)
                            if schema_deficit_info else None
                        ),
                        "schema_deficit_gain": (
                            round(float((schema_deficit_info or {}).get("gain", 0.0)), 6)
                            if schema_deficit_info else None
                        ),
                        "schema_deficit_schema_score": (
                            round(float((schema_deficit_info or {}).get("schema_score", 0.0)), 6)
                            if schema_deficit_info else None
                        ),
                        "schema_deficit_schema_idf": (
                            round(float((schema_deficit_info or {}).get("schema_idf", 0.0)), 6)
                            if schema_deficit_info else None
                        ),
                        "schema_deficit_schema_frequency": (
                            (schema_deficit_info or {}).get("schema_frequency")
                            if schema_deficit_info else None
                        ),
                        "schema_deficit_schema_specificity": (
                            round(float((schema_deficit_info or {}).get("schema_specificity", 0.0)), 6)
                            if schema_deficit_info else None
                        ),
                        "schema_deficit_clear_deficit": (
                            bool((schema_deficit_info or {}).get("clear_deficit", False))
                            if schema_deficit_info else None
                        ),
                        "schema_deficit_schema_supported_deficit": (
                            bool((schema_deficit_info or {}).get("schema_supported_deficit", False))
                            if schema_deficit_info else None
                        ),
                        "schema_deficit_calibrated_min_score": (
                            round(
                                float(
                                    ((schema_deficit_info or {}).get("thresholds") or {}).get(
                                        "calibrated_min_score",
                                        0.0,
                                    )
                                ),
                                6,
                            )
                            if schema_deficit_info else None
                        ),
                        "schema_deficit_calibrated_min_schema_score": (
                            round(
                                float(
                                    ((schema_deficit_info or {}).get("thresholds") or {}).get(
                                        "calibrated_min_schema_score",
                                        0.0,
                                    )
                                ),
                                6,
                            )
                            if schema_deficit_info else None
                        ),
                        "schema_deficit_effective_accepted": (
                            (schema_deficit_info or {}).get("effective_accepted")
                            if schema_deficit_info else None
                        ),
                        "schema_deficit_reject_reasons": (
                            (schema_deficit_info or {}).get("reject_reasons", []) or []
                        )[:6],
                        "schema_reliability_reason": (
                            (item.get("schema_reliability_gate") or {}).get("reason")
                        ),
                        "schema_reliability_score": (
                            round(float((item.get("schema_reliability_gate") or {}).get("score", 0.0)), 6)
                            if item.get("schema_reliability_gate") else None
                        ),
                        "schema_reliability_rescued": (
                            bool((item.get("schema_reliability_gate") or {}).get("rescued", False))
                            if item.get("schema_reliability_gate") else None
                        ),
                        "schema_reliability_reject_reasons": (
                            (item.get("schema_reliability_gate") or {}).get("reject_reasons", []) or []
                        )[:6],
                        "query_schema_gate_action": query_schema_gate_action,
                        "query_schema_gate_reason": query_schema_gate_info.get("reason"),
                        "query_schema_gate_reject_reasons": (
                            query_schema_gate_info.get("reject_reasons", []) or []
                        )[:6],
                        "accepted": not reject_reasons,
                        "reject_reasons": reject_reasons[:6],
                        "original_reject_reasons": original_reject_reasons[:6],
                        "bridge_slot_rescued": bridge_slot_rescued,
                        "bridge_slot_append_only": bool(item.get("bridge_slot_append_only", False)),
                        "bridge_slot_score": (
                            round(float(bridge_slot_rescue_info.get("score", 0.0)), 6)
                            if bridge_slot_rescue_info else None
                        ),
                        "bridge_slot_margin": (
                            round(float(bridge_slot_rescue_info.get("margin", 0.0)), 6)
                            if bridge_slot_rescue_info else None
                        ),
                        "bridge_slot_reason": (
                            bridge_slot_rescue_info.get("reason") if bridge_slot_rescue_info else None
                        ),
                        "bridge_slot_deficit_gate": (
                            bridge_slot_rescue_info.get("deficit_gate") if bridge_slot_rescue_info else None
                        ),
                        "bridge_slot_reject_reasons": (
                            (bridge_slot_rescue_info.get("reject_reasons", []) or [])[:6]
                            if bridge_slot_rescue_info else []
                        ),
                        "relation_guard_reason": (
                            (relation_object_guard_info or {}).get("reason")
                        ),
                        "answer_type_guard_reason": (
                            (answer_type_conflict_guard_info or {}).get("reason")
                        ),
                        "promotion_utility_veto_reason": (
                            (promotion_utility_veto_info or {}).get("reason")
                        ),
                        "promotion_utility_veto_score": (
                            (promotion_utility_veto_info or {}).get("score")
                        ),
                        "answer_type_guard_target_type": (
                            (answer_type_conflict_guard_info or {}).get("target_type")
                        ),
                        "answer_type_guard_old_values": (
                            (answer_type_conflict_guard_info or {}).get("old", {}).get("values", [])
                        ),
                        "answer_type_guard_candidate_values": (
                            (answer_type_conflict_guard_info or {}).get("candidate", {}).get("values", [])
                        ),
                        "object_memory_reason": (
                            (object_memory_info or {}).get("reason")
                        ),
                        "object_memory_objects": (
                            (object_memory_info or {}).get("objects", [])[:6]
                        ),
                    }
                    if len(utility_debug) < 10:
                        utility_debug.append(summary)
                    if has_query_schema_signal and len(schema_deficit_candidate_debug) < 12:
                        schema_deficit_candidate_debug.append(summary)
                    if reject_reasons:
                        continue
                    utility_eligible_items.append(item)

                selected_utility_pool = utility_eligible_items
                original_fallback_deferred_count = 0
                if pairwise_utility_original_fallback_only:
                    dual_source_items = [
                        item for item in utility_eligible_items
                        if item.get("candidate_source") != "original_retrieval"
                    ]
                    original_source_items = [
                        item for item in utility_eligible_items
                        if item.get("candidate_source") == "original_retrieval"
                    ]
                    if dual_source_items:
                        selected_utility_pool = dual_source_items
                        original_fallback_deferred_count = len(original_source_items)

                if selected_utility_pool:
                    selected_utility_pool.sort(
                        key=lambda item: (
                            -float(item.get("pairwise_utility_score", 0.0)),
                            -float(item.get("pairwise_utility_margin", 0.0)),
                            item.get("original_rank") if item.get("original_rank") is not None else 999999,
                        )
                    )
                    top2_eligible_items = selected_utility_pool
                    top2_override_info["pairwise_utility_search"].update({
                        "applied": True,
                        "reason": "selected_pairwise_utility_candidate",
                        "candidate_count": len(utility_candidates),
                        "accepted_count": len(utility_eligible_items),
                        "selection_pool_count": len(selected_utility_pool),
                        "original_fallback_deferred_count": original_fallback_deferred_count,
                        "bridge_slot_rescue_accepted_count": bridge_slot_rescue_accepted_count,
                        "bridge_slot_rescue_rejected_count": bridge_slot_rescue_rejected_count,
                        "bridge_slot_append_accepted_count": bridge_slot_append_accepted_count,
                        "bridge_slot_append_deferred_visible_count": bridge_slot_append_deferred_visible_count,
                        "bridge_slot_append_doc_ids": [
                            int(candidate["doc_id"]) for candidate in bridge_slot_append_items[:8]
                        ],
                        "bridge_slot_rescue_reject_reason_counts": dict(bridge_slot_rescue_reject_counts),
                        "object_memory_accepted_count": object_memory_accepted_count,
                        "object_memory_rejected_count": object_memory_rejected_count,
                        "object_memory_minimal_sentence_count": object_memory_minimal_sentence_count,
                        "object_memory_reject_reason_counts": dict(object_memory_reject_counts),
                        "unfiltered_candidate_count": unfiltered_utility_candidate_count,
                        "source_filtered_count": source_filtered_count,
                        "query_schema_memory_candidate_count": schema_memory_candidate_count,
                        "query_schema_memory_duplicate_count": schema_memory_duplicate_count,
                        "query_schema_intervention_allow_top2_count": int(
                            query_schema_intervention_counts.get("allow_top2", 0)
                        ),
                        "query_schema_intervention_append_only_count": int(
                            query_schema_intervention_counts.get("append_only", 0)
                        ),
                        "query_schema_intervention_reject_count": int(
                            query_schema_intervention_counts.get("reject", 0)
                        ),
                        "query_schema_intervention_reason_counts": dict(
                            query_schema_intervention_reason_counts
                        ),
                        "schema_verified_prior_applied_count": schema_verified_prior_applied_count,
                        "schema_verified_prior_rescue_count": schema_verified_prior_rescue_count,
                        "schema_verified_prior_bonus_total": round(schema_verified_prior_bonus_total, 6),
                        "schema_verified_prior_reject_reason_counts": dict(schema_verified_prior_reject_counts),
                        "schema_deficit_promotion_accepted_count": schema_deficit_promotion_accepted_count,
                        "schema_deficit_promotion_rejected_count": schema_deficit_promotion_rejected_count,
                        "schema_deficit_promotion_reason_counts": dict(
                            schema_deficit_promotion_reason_counts
                        ),
                        "schema_deficit_promotion_reject_reason_counts": dict(
                            schema_deficit_promotion_reject_counts
                        ),
                        "schema_deficit_soft_append_accepted_count": schema_deficit_soft_append_accepted_count,
                        "schema_deficit_soft_append_rejected_count": schema_deficit_soft_append_rejected_count,
                        "schema_deficit_soft_append_doc_ids": [
                            int(candidate["doc_id"]) for candidate in schema_deficit_soft_append_items[:8]
                        ],
                        "schema_deficit_soft_append_reason_counts": dict(
                            schema_deficit_soft_append_reason_counts
                        ),
                        "schema_deficit_soft_append_reject_reason_counts": dict(
                            schema_deficit_soft_append_reject_counts
                        ),
                        "schema_reliability_evaluated_count": schema_reliability_evaluated_count,
                        "schema_reliability_passed_count": schema_reliability_passed_count,
                        "schema_reliability_rescued_count": schema_reliability_rescued_count,
                        "schema_reliability_rejected_count": schema_reliability_rejected_count,
                        "schema_reliability_reason_counts": dict(schema_reliability_reason_counts),
                        "schema_reliability_reject_reason_counts": dict(
                            schema_reliability_reject_counts
                        ),
                        "schema_deficit_candidate_debug": schema_deficit_candidate_debug,
                        "answer_type_conflict_guard_evaluated_count": answer_type_conflict_guard_evaluated_count,
                        "answer_type_conflict_guard_rejected_count": answer_type_conflict_guard_rejected_count,
                        "answer_type_conflict_guard_reject_reason_counts": dict(answer_type_conflict_guard_reject_counts),
                        "selected_doc_id": selected_utility_pool[0]["doc_id"],
                        "selected_source": selected_utility_pool[0].get("candidate_source"),
                        "selected_sources": selected_utility_pool[0].get("candidate_sources", []),
                        "top_candidates": utility_debug,
                    })
                else:
                    top2_override_info["pairwise_utility_search"].update({
                        "applied": False,
                        "reason": "no_pairwise_utility_candidate_accepted",
                        "candidate_count": len(utility_candidates),
                        "accepted_count": 0,
                        "selection_pool_count": 0,
                        "original_fallback_deferred_count": 0,
                        "bridge_slot_rescue_accepted_count": bridge_slot_rescue_accepted_count,
                        "bridge_slot_rescue_rejected_count": bridge_slot_rescue_rejected_count,
                        "bridge_slot_append_accepted_count": bridge_slot_append_accepted_count,
                        "bridge_slot_append_deferred_visible_count": bridge_slot_append_deferred_visible_count,
                        "bridge_slot_append_doc_ids": [
                            int(candidate["doc_id"]) for candidate in bridge_slot_append_items[:8]
                        ],
                        "bridge_slot_rescue_reject_reason_counts": dict(bridge_slot_rescue_reject_counts),
                        "object_memory_accepted_count": object_memory_accepted_count,
                        "object_memory_rejected_count": object_memory_rejected_count,
                        "object_memory_minimal_sentence_count": object_memory_minimal_sentence_count,
                        "object_memory_reject_reason_counts": dict(object_memory_reject_counts),
                        "unfiltered_candidate_count": unfiltered_utility_candidate_count,
                        "source_filtered_count": source_filtered_count,
                        "query_schema_memory_candidate_count": schema_memory_candidate_count,
                        "query_schema_memory_duplicate_count": schema_memory_duplicate_count,
                        "query_schema_intervention_allow_top2_count": int(
                            query_schema_intervention_counts.get("allow_top2", 0)
                        ),
                        "query_schema_intervention_append_only_count": int(
                            query_schema_intervention_counts.get("append_only", 0)
                        ),
                        "query_schema_intervention_reject_count": int(
                            query_schema_intervention_counts.get("reject", 0)
                        ),
                        "query_schema_intervention_reason_counts": dict(
                            query_schema_intervention_reason_counts
                        ),
                        "schema_verified_prior_applied_count": schema_verified_prior_applied_count,
                        "schema_verified_prior_rescue_count": schema_verified_prior_rescue_count,
                        "schema_verified_prior_bonus_total": round(schema_verified_prior_bonus_total, 6),
                        "schema_verified_prior_reject_reason_counts": dict(schema_verified_prior_reject_counts),
                        "schema_deficit_promotion_accepted_count": schema_deficit_promotion_accepted_count,
                        "schema_deficit_promotion_rejected_count": schema_deficit_promotion_rejected_count,
                        "schema_deficit_promotion_reason_counts": dict(
                            schema_deficit_promotion_reason_counts
                        ),
                        "schema_deficit_promotion_reject_reason_counts": dict(
                            schema_deficit_promotion_reject_counts
                        ),
                        "schema_deficit_soft_append_accepted_count": schema_deficit_soft_append_accepted_count,
                        "schema_deficit_soft_append_rejected_count": schema_deficit_soft_append_rejected_count,
                        "schema_deficit_soft_append_doc_ids": [
                            int(candidate["doc_id"]) for candidate in schema_deficit_soft_append_items[:8]
                        ],
                        "schema_deficit_soft_append_reason_counts": dict(
                            schema_deficit_soft_append_reason_counts
                        ),
                        "schema_deficit_soft_append_reject_reason_counts": dict(
                            schema_deficit_soft_append_reject_counts
                        ),
                        "schema_reliability_evaluated_count": schema_reliability_evaluated_count,
                        "schema_reliability_passed_count": schema_reliability_passed_count,
                        "schema_reliability_rescued_count": schema_reliability_rescued_count,
                        "schema_reliability_rejected_count": schema_reliability_rejected_count,
                        "schema_reliability_reason_counts": dict(schema_reliability_reason_counts),
                        "schema_reliability_reject_reason_counts": dict(
                            schema_reliability_reject_counts
                        ),
                        "schema_deficit_candidate_debug": schema_deficit_candidate_debug,
                        "answer_type_conflict_guard_evaluated_count": answer_type_conflict_guard_evaluated_count,
                        "answer_type_conflict_guard_rejected_count": answer_type_conflict_guard_rejected_count,
                        "answer_type_conflict_guard_reject_reason_counts": dict(answer_type_conflict_guard_reject_counts),
                        "top_candidates": utility_debug,
                    })

            if top2_override_enabled and top2_eligible_items:
                top2_eligible_items.sort(
                    key=lambda item: (
                        -float(item.get("top2_selection_score", item.get("plan_score", 0.0))),
                        -float(item.get("plan_score", 0.0)),
                        item.get("plan_rank", 999999),
                        item.get("original_rank") if item.get("original_rank") is not None else 999999,
                    )
                )
                top2_override_item = top2_eligible_items[0]
                item = top2_override_item
                top2_override_info.update({
                    "applied": True,
                    "reason": (
                        "accepted_bridge_slot_rank2_evidence"
                        if item.get("bridge_slot_rescued")
                        else
                        "accepted_pairwise_utility_rank2_evidence"
                        if item.get("pairwise_utility_search")
                        else
                        "accepted_soft_pair_reranked_rank2_evidence"
                        if top2_pair_verifier_enabled and top2_pair_verifier_mode == "soft"
                        else "accepted_high_confidence_rank2_evidence"
                    ),
                    "doc_id": item["doc_id"],
                    "old_rank2_doc_id": base_ids[1],
                    "original_rank": item["original_rank"],
                    "plan_rank": item["plan_rank"],
                    "plan_score": round(item["plan_score"], 6),
                    "top2_selection_score": round(float(item.get("top2_selection_score", item["plan_score"])), 6),
                    "query": item["query"],
                    "queries": item.get("queries", [])[:6],
                    "query_overlap": round(item.get("query_overlap", 0.0), 6),
                    "query_count": int(item.get("query_count", 1)),
                    "candidate_source": item.get("candidate_source"),
                    "pairwise_utility_score": (
                        round(float(item.get("pairwise_utility_score")), 6)
                        if item.get("pairwise_utility_score") is not None else None
                    ),
                    "pairwise_utility_margin": (
                        round(float(item.get("pairwise_utility_margin")), 6)
                        if item.get("pairwise_utility_margin") is not None else None
                    ),
                    "pair_verifier": item.get("top2_pair_verifier"),
                    "relation_object_guard": item.get("relation_object_guard"),
                    "promotion_utility_veto": item.get("promotion_utility_veto"),
                    "bridge_slot_rescue": item.get("bridge_slot_rescue"),
                })
                top2_override_info["eligible_count"] = len(top2_eligible_items)
                top2_override_info["eligible_candidates"] = [
                    {
                        "doc_id": candidate["doc_id"],
                        "plan_score": round(float(candidate.get("plan_score", 0.0)), 6),
                        "top2_selection_score": round(
                            float(candidate.get("top2_selection_score", candidate.get("plan_score", 0.0))),
                            6,
                        ),
                        "plan_rank": candidate.get("plan_rank"),
                        "original_rank": candidate.get("original_rank"),
                        "query_overlap": round(float(candidate.get("query_overlap", 0.0)), 6),
                        "query_count": int(candidate.get("query_count", 1)),
                        "candidate_source": candidate.get("candidate_source"),
                        "pairwise_utility_score": (
                            round(float(candidate.get("pairwise_utility_score")), 6)
                            if candidate.get("pairwise_utility_score") is not None else None
                        ),
                        "pair_margin": (
                            (candidate.get("top2_pair_verifier") or {}).get("pair_margin")
                        ),
                        "bridge_slot_rescued": bool(candidate.get("bridge_slot_rescued", False)),
                        "bridge_slot_score": (
                            (candidate.get("bridge_slot_rescue") or {}).get("score")
                        ),
                        "bridge_slot_reason": (
                            (candidate.get("bridge_slot_rescue") or {}).get("reason")
                        ),
                        "relation_guard_reason": (
                            (candidate.get("relation_object_guard") or {}).get("reason")
                        ),
                    }
                    for candidate in top2_eligible_items[:6]
                ]
    if object_memory_enabled and object_memory_append_only and object_memory_append_max_items > 0:
        if len(base_ids) < 2:
            object_memory_append_info["reason"] = "not_enough_ranked_docs"
        elif doc_lookup is None:
            object_memory_append_info["reason"] = "missing_doc_lookup"
        else:
            top1_doc = get_doc_text(base_ids[0])
            old_rank2_doc = get_doc_text(base_ids[1])
            if not top1_doc or not old_rank2_doc:
                object_memory_append_info["reason"] = "missing_pair_text"
            else:
                object_memory_append_info["reason"] = "no_candidate_accepted"
                append_candidates = sorted(
                    ranked_candidates,
                    key=lambda candidate: (
                        -float(candidate.get("plan_score", 0.0) or 0.0),
                        candidate.get("plan_rank", 999999),
                        candidate.get("original_rank")
                        if candidate.get("original_rank") is not None else 999999,
                    ),
                )[:object_memory_append_candidate_top_k]
                append_min_improvement = max(0.0, bridge_slot_rescue_min_improvement * 0.50)
                append_min_missing_gain = max(0.0, bridge_slot_rescue_min_missing_gain * 0.50)
                for item in append_candidates:
                    if len(object_memory_append_items) >= object_memory_append_max_items:
                        break
                    object_memory_append_candidate_count += 1
                    reject_reasons = []
                    doc_id = int(item["doc_id"])
                    if doc_id in base_ids[:target_k]:
                        reject_reasons.append("object_memory_append_visible_in_topk")
                    if doc_id in object_memory_append_seen or doc_id in bridge_slot_append_seen:
                        reject_reasons.append("object_memory_append_duplicate_doc")
                    if float(item.get("plan_score", 0.0) or 0.0) < object_memory_append_min_plan_score:
                        reject_reasons.append("object_memory_append_low_plan_score")

                    bridge_slot_info = None
                    object_memory_info = None
                    if not reject_reasons:
                        bridge_slot_info = _bridge_slot_coverage_score(
                            question=question,
                            top1_doc=top1_doc,
                            old_rank2_doc=old_rank2_doc,
                            candidate_doc=get_doc_text(doc_id),
                            candidate_item=item,
                            old_rank2_item=old_rank2_planned_evidence,
                            min_score=bridge_slot_rescue_min_score,
                            min_improvement=append_min_improvement,
                            min_anchor_overlap=bridge_slot_rescue_min_anchor_overlap,
                            require_relation=bridge_slot_rescue_require_relation,
                            require_old_deficit=False,
                            deficit_max_old_score=bridge_slot_rescue_deficit_max_old_score,
                            min_missing_gain=append_min_missing_gain,
                            append_on_no_deficit=True,
                        )
                        if not bridge_slot_info.get("append_accepted", False):
                            reject_reasons.extend(
                                bridge_slot_info.get("append_reject_reasons", [])
                                or bridge_slot_info.get("reject_reasons", [])
                                or ["object_memory_append_bridge_slot_rejected"]
                            )

                    if not reject_reasons:
                        object_memory_info = _object_consistent_evidence_memory_gate(
                            question=question,
                            planned_query=_item_query_text(item),
                            top1_doc=top1_doc,
                            old_rank2_doc=old_rank2_doc,
                            candidate_doc=get_doc_text(doc_id),
                            bridge_slot_info=bridge_slot_info,
                            min_score=object_memory_min_score,
                            max_sentences=object_memory_max_sentences,
                            max_objects=object_memory_max_objects,
                            max_chars=object_memory_max_chars,
                        )
                        if not object_memory_info.get("accepted", False):
                            reject_reasons.extend(
                                object_memory_info.get("reject_reasons", [])
                                or ["object_memory_append_gate_rejected"]
                            )

                    memory_doc = (
                        str((object_memory_info or {}).get("memory_doc") or "").strip()
                        if object_memory_info else ""
                    )
                    if not reject_reasons and not memory_doc:
                        reject_reasons.append("object_memory_append_empty_memory_doc")

                    if reject_reasons:
                        object_memory_append_rejected_count += 1
                        object_memory_append_reject_counts.update(reject_reasons)
                    else:
                        memory_key = _normalize_query(memory_doc)
                        if not memory_key or memory_key in object_memory_append_seen:
                            object_memory_append_rejected_count += 1
                            object_memory_append_reject_counts.update([
                                "object_memory_append_duplicate_memory"
                            ])
                            reject_reasons.append("object_memory_append_duplicate_memory")
                        else:
                            memory_item = dict(item)
                            memory_item["bridge_slot_append_only"] = True
                            memory_item["object_memory_append_only"] = True
                            memory_item["bridge_slot_rescue"] = bridge_slot_info
                            memory_item["object_consistent_memory"] = object_memory_info
                            memory_item["object_memory_doc"] = memory_doc
                            memory_item["top2_selection_score"] = float(
                                item.get("plan_score", 0.0) or 0.0
                            )
                            object_memory_append_items.append(memory_item)
                            object_memory_append_seen.add(doc_id)
                            object_memory_append_seen.add(memory_key)
                            object_memory_append_minimal_sentence_count += int(
                                object_memory_info.get("sentence_count", 0) or 0
                            )

                    if len(object_memory_append_debug) < 10:
                        object_memory_append_debug.append({
                            "doc_id": doc_id,
                            "plan_rank": item.get("plan_rank"),
                            "original_rank": item.get("original_rank"),
                            "plan_score": round(float(item.get("plan_score", 0.0) or 0.0), 6),
                            "query_overlap": round(float(item.get("query_overlap", 0.0) or 0.0), 6),
                            "accepted": not reject_reasons,
                            "reject_reasons": reject_reasons[:6],
                            "bridge_slot_reason": (
                                (bridge_slot_info or {}).get("reason")
                                if bridge_slot_info else None
                            ),
                            "bridge_slot_score": (
                                round(float((bridge_slot_info or {}).get("score", 0.0) or 0.0), 6)
                                if bridge_slot_info else None
                            ),
                            "object_memory_reason": (
                                (object_memory_info or {}).get("reason")
                                if object_memory_info else None
                            ),
                            "object_memory_objects": (
                                (object_memory_info or {}).get("objects", [])[:6]
                                if object_memory_info else []
                            ),
                        })

                if object_memory_append_items:
                    object_memory_append_info["reason"] = "accepted_append_only_memory"
                elif object_memory_append_candidate_count == 0:
                    object_memory_append_info["reason"] = "no_candidates_considered"

    object_memory_append_info.update({
        "candidate_count": object_memory_append_candidate_count,
        "accepted_count": len(object_memory_append_items),
        "rejected_count": object_memory_append_rejected_count,
        "minimal_sentence_count": object_memory_append_minimal_sentence_count,
        "accepted_doc_ids": [int(item["doc_id"]) for item in object_memory_append_items[:8]],
        "reject_reason_counts": dict(object_memory_append_reject_counts),
        "top_candidates": object_memory_append_debug,
    })
    top2_override_info["rejected_count"] = sum(top2_override_rejection_counts.values())
    top2_override_info["reject_reason_counts"] = dict(top2_override_rejection_counts)
    top2_override_info["rejections"] = top2_override_rejections
    top2_override_info["promotion_utility_veto"] = {
        "enabled": promotion_utility_veto_enabled,
        "evaluated_count": promotion_utility_veto_evaluated_count,
        "rejected_count": promotion_utility_veto_rejected_count,
        "reason_counts": dict(promotion_utility_veto_reason_counts),
        "debug": promotion_utility_veto_debug,
        "min_utility": round(float(promotion_utility_veto_min_utility), 6),
        "require_bridge_or_gap": promotion_utility_veto_require_bridge_or_gap,
        "apply_to_guarded": promotion_utility_veto_apply_to_guarded,
        "high_risk_only": promotion_utility_veto_high_risk_only,
        "append_on_reject": promotion_utility_veto_append_on_reject,
        "soft_append_count": promotion_utility_soft_append_count,
        "soft_append_doc_ids": [
            int(item["doc_id"]) for item in promotion_utility_soft_append_items[:8]
        ],
    }
    budgeted_evidence_subgraph_info = {
        "enabled": bool(budgeted_evidence_subgraph_enabled),
        "applied": False,
        "reason": "disabled" if not budgeted_evidence_subgraph_enabled else "not_evaluated",
        "max_append": budgeted_evidence_subgraph_max_append,
        "max_candidates": budgeted_evidence_subgraph_max_candidates,
        "min_utility": round(float(budgeted_evidence_subgraph_min_utility), 6),
        "redundancy_penalty": round(float(budgeted_evidence_subgraph_redundancy_penalty), 6),
        "require_bridge_or_gap": budgeted_evidence_subgraph_require_bridge_or_gap,
        "selected_doc_ids": [],
    }

    def append_rank_stable_expansions(current_seen: set) -> None:
        nonlocal budgeted_evidence_subgraph_info
        remaining_budget = max(0, max_promotions - len(expansions))
        if remaining_budget <= 0:
            if budgeted_evidence_subgraph_enabled:
                budgeted_evidence_subgraph_info["reason"] = "no_remaining_append_budget"
            return
        if budgeted_evidence_subgraph_enabled:
            selector_budget = min(
                remaining_budget,
                budgeted_evidence_subgraph_max_append if budgeted_evidence_subgraph_max_append > 0 else remaining_budget,
            )
            selected_items, selector_info = _budgeted_evidence_subgraph_select(
                question=question,
                base_ids=base_ids,
                target_k=target_k,
                candidate_groups=[
                    ("object_memory", object_memory_append_items),
                    ("bridge_slot", bridge_slot_append_items),
                    ("schema_deficit", schema_deficit_soft_append_items),
                    ("promotion_utility_soft_append", promotion_utility_soft_append_items),
                    ("query_schema_memory", schema_memory_candidates),
                ],
                ranked_candidates=ranked_candidates,
                doc_lookup=get_doc_text,
                max_append=selector_budget,
                max_candidates=budgeted_evidence_subgraph_max_candidates,
                min_utility=budgeted_evidence_subgraph_min_utility,
                redundancy_penalty=budgeted_evidence_subgraph_redundancy_penalty,
                require_bridge_or_gap=budgeted_evidence_subgraph_require_bridge_or_gap,
                intervention_gate=intervention_gate,
            )
            budgeted_evidence_subgraph_info = selector_info
            for item in selected_items:
                doc_id = int(item["doc_id"])
                if doc_id in current_seen:
                    continue
                if len(expansions) >= max_promotions:
                    break
                item["budgeted_evidence_subgraph_selected"] = True
                expansions.append(item)
                current_seen.add(doc_id)
            return

        for item in sorted(
            object_memory_append_items,
            key=lambda candidate: (
                -float((candidate.get("bridge_slot_rescue") or {}).get("score", 0.0) or 0.0),
                -float(candidate.get("plan_score", 0.0) or 0.0),
                candidate.get("original_rank") if candidate.get("original_rank") is not None else 999999,
            ),
        ):
            doc_id = item["doc_id"]
            if doc_id in current_seen:
                continue
            if len(expansions) >= max_promotions:
                break
            expansions.append(item)
            current_seen.add(doc_id)

        for item in sorted(
            bridge_slot_append_items,
            key=lambda candidate: (
                -float((candidate.get("bridge_slot_rescue") or {}).get("score", 0.0) or 0.0),
                -float((candidate.get("bridge_slot_rescue") or {}).get("margin", 0.0) or 0.0),
                candidate.get("original_rank") if candidate.get("original_rank") is not None else 999999,
            ),
        ):
            doc_id = item["doc_id"]
            if doc_id in current_seen:
                continue
            if len(expansions) >= max_promotions:
                break
            expansions.append(item)
            current_seen.add(doc_id)

        for item in sorted(
            schema_deficit_soft_append_items,
            key=lambda candidate: (
                -float((candidate.get("schema_deficit_soft_append") or {}).get("score", 0.0) or 0.0),
                -float((candidate.get("schema_deficit_soft_append") or {}).get("candidate_query_support", 0.0) or 0.0),
                candidate.get("original_rank") if candidate.get("original_rank") is not None else 999999,
            ),
        ):
            doc_id = item["doc_id"]
            if doc_id in current_seen:
                continue
            if len(expansions) >= max_promotions:
                break
            expansions.append(item)
            current_seen.add(doc_id)

        for item in sorted(
            promotion_utility_soft_append_items,
            key=lambda candidate: (
                -float((candidate.get("promotion_utility_veto") or {}).get("score", 0.0) or 0.0),
                -float(candidate.get("plan_score", 0.0) or 0.0),
                candidate.get("original_rank") if candidate.get("original_rank") is not None else 999999,
            ),
        ):
            doc_id = item["doc_id"]
            if doc_id in current_seen:
                continue
            if len(expansions) >= max_promotions:
                break
            expansions.append(item)
            current_seen.add(doc_id)

        for item in ranked_candidates:
            doc_id = item["doc_id"]
            if doc_id in current_seen:
                continue
            if len(expansions) >= max_promotions:
                break
            expansions.append(item)
            current_seen.add(doc_id)

    if merge_mode == "replace_topk":
        new_top = list(protected)
        seen = set(new_top)
        for item in ranked_candidates:
            if len(new_top) >= target_k:
                break
            doc_id = item["doc_id"]
            if doc_id in seen:
                continue
            if len(promoted) >= max_promotions:
                continue
            new_top.append(doc_id)
            seen.add(doc_id)
            promoted.append(item)

        for doc_id in base_ids[:target_k]:
            if len(new_top) >= target_k:
                break
            if doc_id in seen:
                continue
            new_top.append(doc_id)
            seen.add(doc_id)

        for item in ranked_candidates:
            if len(new_top) >= target_k:
                break
            doc_id = item["doc_id"]
            if doc_id in seen:
                continue
            new_top.append(doc_id)
            seen.add(doc_id)
    elif merge_mode == "guarded_promote":
        effective_protect_top_k = min(max(protect_top_k, 2), target_k, len(base_ids))
        if top2_override_item is not None:
            new_top = [base_ids[0], top2_override_item["doc_id"]]
            promoted.append(top2_override_item)
        else:
            new_top = list(base_ids[:effective_protect_top_k])
        seen = set(new_top)

        for item in ranked_candidates:
            if len(promoted) >= max_promotions:
                break
            guarded_considered_count += 1
            doc_id = item["doc_id"]
            reject_reasons = []
            if doc_id in seen:
                reject_reasons.append("duplicate_or_protected")
            if item["plan_score"] < guarded_min_score:
                reject_reasons.append("low_guarded_score")
            if item["plan_rank"] >= guarded_max_plan_rank:
                reject_reasons.append("low_planned_rank")
            if item["original_rank"] is None or item["original_rank"] >= guarded_max_original_rank:
                reject_reasons.append("outside_original_rank_window")
            if item.get("query_overlap", 0.0) < guarded_min_query_overlap:
                reject_reasons.append("weak_query_anchor_overlap")

            promotion_utility_veto_info = None
            if (
                not reject_reasons
                and promotion_utility_veto_enabled
                and promotion_utility_veto_apply_to_guarded
            ):
                promotion_utility_veto_info = evaluate_promotion_utility_veto(
                    item,
                    context="guarded_promote_top5",
                )
                item["promotion_utility_veto"] = promotion_utility_veto_info
                if not promotion_utility_veto_info.get("accepted", True):
                    record_promotion_utility_soft_append(
                        item,
                        promotion_utility_veto_info,
                        context="guarded_promote_top5",
                    )
                    reject_reasons.extend(
                        promotion_utility_veto_info.get("reject_reasons", [])
                        or ["promotion_utility_veto_rejected"]
                    )

            if reject_reasons:
                guard_rejection_counts.update(reject_reasons)
                if len(guard_rejections) < 12:
                    guard_rejections.append({
                        "doc_id": doc_id,
                        "query": item["query"],
                        "plan_rank": item["plan_rank"],
                        "original_rank": item["original_rank"],
                        "plan_score": round(item["plan_score"], 6),
                        "query_overlap": round(item.get("query_overlap", 0.0), 6),
                        "reasons": reject_reasons,
                        "promotion_utility_veto": promotion_utility_veto_info,
                    })
                continue

            new_top.append(doc_id)
            seen.add(doc_id)
            promoted.append(item)

        for doc_id in base_ids[:target_k]:
            if len(new_top) >= target_k:
                break
            if doc_id in seen:
                continue
            new_top.append(doc_id)
            seen.add(doc_id)

        append_rank_stable_expansions(seen)
    else:


        append_rank_stable_expansions(seen)

    if support_set_rerank_enabled:
        if merge_mode != "guarded_promote":
            support_set_rerank_info["reason"] = f"merge_mode_{merge_mode}_not_supported"
        elif intervention_gate.get("enabled") and not intervention_gate.get("top5_allowed", True):
            support_set_rerank_info["reason"] = "disabled_by_intervention_gate"
        elif doc_lookup is None:
            support_set_rerank_info["reason"] = "missing_doc_lookup"
        else:
            rerank_target_k = min(target_k, support_set_rerank_target_k)
            support_top, support_info, support_promoted = _support_set_utility_rerank(
                question=question,
                base_ids=base_ids,
                base_scores=base_scores,
                current_top=new_top[:rerank_target_k],
                ranked_candidates=ranked_candidates,
                schema_memory_candidates=schema_memory_candidates,
                utility_candidates=support_set_utility_candidates,
                doc_lookup=get_doc_text,
                target_k=rerank_target_k,
                protect_top_k=support_set_rerank_protect_top_k,
                max_replacements=support_set_rerank_max_replacements,
                max_candidates=support_set_rerank_max_candidates,
                max_original_rank=support_set_rerank_max_original_rank,
                min_candidate_score=support_set_rerank_min_candidate_score,
                redundancy_penalty=support_set_rerank_redundancy_penalty,
                preserve_original_topk=support_set_rerank_preserve_original_topk,
                preserve_min_gain=support_set_rerank_preserve_min_gain,
                preserve_max_coverage_loss=(
                    support_set_rerank_preserve_max_coverage_loss
                ),
                require_missing_slot_gate=True,
                missing_slot_min_score=min(0.52, bridge_slot_rescue_min_score),
                missing_slot_min_improvement=max(0.06, bridge_slot_rescue_min_improvement * 0.50),
                missing_slot_min_anchor_overlap=bridge_slot_rescue_min_anchor_overlap,
                missing_slot_require_relation=bridge_slot_rescue_require_relation,
                missing_slot_min_missing_gain=max(0.08, bridge_slot_rescue_min_missing_gain * 0.50),
            )
            support_set_rerank_info.update(support_info)
            if support_info.get("applied"):
                support_seen = set(int(doc_id) for doc_id in support_top)
                new_top = [int(doc_id) for doc_id in support_top]
                for doc_id in base_ids[:target_k]:
                    if len(new_top) >= target_k:
                        break
                    if doc_id in support_seen:
                        continue
                    new_top.append(int(doc_id))
                    support_seen.add(int(doc_id))
                promoted = list(support_promoted)
                seen = set(new_top)
                expansions = [
                    item for item in expansions
                    if int(item["doc_id"]) not in seen
                ]
                for item in expansions:
                    seen.add(int(item["doc_id"]))

    expansion_ids = [item["doc_id"] for item in expansions]
    remaining = [doc_id for doc_id in base_ids if doc_id not in seen]
    new_ids = new_top + expansion_ids + remaining
    score_by_id = {item["doc_id"]: item["plan_score"] for item in ranked_candidates}
    new_scores = [score_by_id.get(doc_id, base_score.get(doc_id, 0.0)) for doc_id in new_ids]

    old_top2 = set(base_ids[:min(2, len(base_ids))])
    old_top5 = set(base_ids[:min(5, len(base_ids))])
    new_top2 = set(new_ids[:min(2, len(new_ids))])
    new_top5 = set(new_ids[:min(5, len(new_ids))])
    info = {
        "raw_candidate_count": raw_candidate_count,
        "candidate_count": len(ranked_candidates),
        "unique_novel_candidate_count": len(candidates),
        "duplicate_candidate_hits": duplicate_candidate_hits,
        "existing_top_candidate_count": existing_top_candidate_count,
        "low_score_candidate_count": low_score_candidate_count,
        "promoted_count": len(promoted),
        "promoted_to_top2": len(new_top2 - old_top2),
        "promoted_to_top5": len(new_top5 - old_top5),
        "merge_mode": merge_mode,
        "expansion_count": len(expansions),
        "expansion_doc_ids": expansion_ids,
        "expansion_memory_docs": [
            {
                "doc_id": int(item["doc_id"]),
                "memory_doc": item.get("object_memory_doc", ""),
                "memory_title": (item.get("object_consistent_memory") or {}).get("memory_title", ""),
                "objects": (item.get("object_consistent_memory") or {}).get("objects", [])[:6],
                "sentence_count": (item.get("object_consistent_memory") or {}).get("sentence_count", 0),
                "reason": (item.get("object_consistent_memory") or {}).get("reason"),
                "append_only": bool(item.get("object_memory_append_only", False)),
            }
            for item in expansions
            if item.get("object_memory_doc")
        ],
        "object_memory_append_only": object_memory_append_info,
        "budgeted_evidence_subgraph": budgeted_evidence_subgraph_info,
        "budgeted_evidence_subgraph_applied": (
            1 if budgeted_evidence_subgraph_info.get("applied") else 0
        ),
        "budgeted_evidence_subgraph_reason": budgeted_evidence_subgraph_info.get("reason"),
        "budgeted_evidence_subgraph_selected_count": int(
            budgeted_evidence_subgraph_info.get("selected_count", 0) or 0
        ),
        "budgeted_evidence_subgraph_selected_doc_ids": (
            budgeted_evidence_subgraph_info.get("selected_doc_ids") or []
        ),
        "support_set_rerank": support_set_rerank_info,
        "support_set_rerank_applied": 1 if support_set_rerank_info.get("applied") else 0,
        "support_set_rerank_reason": support_set_rerank_info.get("reason"),
        "support_set_rerank_promoted_to_top2": int(
            support_set_rerank_info.get("promoted_to_top2", 0) or 0
        ),
        "support_set_rerank_promoted_to_top5": int(
            support_set_rerank_info.get("promoted_to_top5", 0) or 0
        ),
        "protect_top_k": effective_protect_top_k,
        "max_promotions": max_promotions,
        "guard_policy": {
            "guarded_min_score": guarded_min_score,
            "guarded_max_original_rank": guarded_max_original_rank,
            "guarded_max_plan_rank": guarded_max_plan_rank,
            "guarded_min_query_overlap": guarded_min_query_overlap,
        },
        "guarded_candidate_count": len(ranked_candidates),
        "guarded_considered_count": guarded_considered_count,
        "guarded_accepted_count": len(promoted) if merge_mode == "guarded_promote" else 0,
        "guarded_rejected_count": sum(guard_rejection_counts.values()),
        "guarded_not_considered_count": (
            max(0, len(ranked_candidates) - guarded_considered_count)
            if merge_mode == "guarded_promote" else 0
        ),
        "guard_rejection_counts": dict(guard_rejection_counts),
        "guard_rejections": guard_rejections,
        "top2_override_enabled": top2_override_enabled,
        "top2_override_applied": 1 if top2_override_info.get("applied") else 0,
        "top2_override_reason": top2_override_info.get("reason"),
        "top2_override_rejected_count": top2_override_info.get("rejected_count", 0),
        "top2_override_reject_reason_counts": top2_override_info.get("reject_reason_counts", {}),
        "promotion_utility_veto_evaluated_count": promotion_utility_veto_evaluated_count,
        "promotion_utility_veto_rejected_count": promotion_utility_veto_rejected_count,
        "promotion_utility_veto_reason_counts": dict(promotion_utility_veto_reason_counts),
        "promotion_utility_veto_soft_append_count": promotion_utility_soft_append_count,
        "promotion_utility_veto_soft_append_doc_ids": [
            int(item["doc_id"]) for item in promotion_utility_soft_append_items[:8]
        ],
        "top2_override": top2_override_info,
        "intervention_gate": intervention_gate,
        "promotions": [
            {
                "doc_id": item["doc_id"],
                "query": item["query"],
                "queries": item.get("queries", [])[:6],
                "query_count": int(item.get("query_count", 1)),
                "plan_rank": item["plan_rank"],
                "original_rank": item["original_rank"],
                "plan_score": round(item["plan_score"], 6),
                "retrieval_score": round(item["retrieval_score"], 6),
                "normalized_score": round(item["normalized_score"], 6),
                "calibrated_score": round(item["calibrated_score"], 6),
                "query_overlap": round(item.get("query_overlap", 0.0), 6),
                "promotion_utility_veto": item.get("promotion_utility_veto"),
            }
            for item in promoted[:8]
        ],
        "expansions": [
            {
                "doc_id": item["doc_id"],
                "query": item["query"],
                "queries": item.get("queries", [])[:6],
                "query_count": int(item.get("query_count", 1)),
                "plan_rank": item["plan_rank"],
                "original_rank": item["original_rank"],
                "plan_score": round(item["plan_score"], 6),
                "retrieval_score": round(item["retrieval_score"], 6),
                "normalized_score": round(item["normalized_score"], 6),
                "calibrated_score": round(item["calibrated_score"], 6),
                "query_overlap": round(item.get("query_overlap", 0.0), 6),
                "bridge_slot_append_only": bool(item.get("bridge_slot_append_only", False)),
                "object_memory_append_only": bool(item.get("object_memory_append_only", False)),
                "schema_deficit_append_only": bool(item.get("schema_deficit_append_only", False)),
                "promotion_utility_soft_append": bool(
                    item.get("promotion_utility_soft_append", False)
                ),
                "promotion_utility_soft_append_context": item.get(
                    "promotion_utility_soft_append_context"
                ),
                "promotion_utility_veto": item.get("promotion_utility_veto"),
                "budgeted_evidence_subgraph_selected": bool(
                    item.get("budgeted_evidence_subgraph_selected", False)
                ),
                "budgeted_evidence_subgraph_score": (
                    round(float(item.get("budgeted_subgraph_score", 0.0) or 0.0), 6)
                    if item.get("budgeted_subgraph_score") is not None else None
                ),
                "budgeted_evidence_subgraph_source": item.get("budgeted_subgraph_source"),
                "bridge_slot_rescue": item.get("bridge_slot_rescue"),
                "schema_deficit_soft_append": item.get("schema_deficit_soft_append"),
                "schema_reliability_gate": item.get("schema_reliability_gate"),
                "object_consistent_memory": item.get("object_consistent_memory"),
            }
            for item in expansions[:8]
        ],
    }
    return np.array(new_ids, dtype=int), np.array(new_scores, dtype=float), info
