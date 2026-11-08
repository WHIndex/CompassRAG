"""Self-contained five-role residual retrieval over the local evidence index."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from .utils.planned_retrieval_utils import (
    MergerAgent,
    PlannerAgent,
    QuestionRouterAgent,
    RetrieverAgent,
    VerifierAgent,
    route_risk_aware_merge_policy,
)


_TOKEN_RE = re.compile(r"[a-z0-9]+")
_ONTOLOGY_HINT_KEYS = {
    "missing",
    "missing_relation",
    "missing_relations",
    "missing_slot",
    "missing_slots",
    "required_relation",
    "required_relations",
    "relation_chain",
    "relation_family",
    "relation_signature",
    "unresolved_relation",
    "unresolved_relations",
    "unresolved_slot",
    "unresolved_slots",
}


@dataclass(frozen=True)
class AgenticResidualRetrievalConfig:
    max_rounds: int = 3
    max_queries: int = 3
    max_snippets: int = 3
    max_tokens: int = 160
    protect_top_k: int = 1
    max_promotions: int = 2
    plan_top_k: int = 20
    min_plan_score: float = 0.18
    merge_mode: str = "guarded_promote"
    guarded_min_score: float = 0.82
    guarded_max_original_rank: int = 80
    guarded_max_plan_rank: int = 8
    guarded_min_query_overlap: float = 0.08
    route_mode: str = "adaptive"
    fusion_depth: int = 5
    rrf_k: float = 60.0
    base_weight: float = 1.0
    residual_weight: float = 1.0
    shadow_only: bool = False


def _normalize_query(text: Any) -> str:
    return " ".join(_TOKEN_RE.findall(str(text or "").lower()))


def _doc_text(retrieval_system: Any, doc_id: int) -> str:
    return retrieval_system.chunk_embedding_store.get_row(
        retrieval_system.passage_node_keys[int(doc_id)]
    )["content"]


def _safe_hint(value: Any) -> str:
    if isinstance(value, str):
        return re.sub(r"\s+", " ", value).strip()[:180]
    if isinstance(value, (int, float, bool)):
        return str(value)
    return ""


def collect_ontology_gap_hints(
    query_plan_log: Mapping[str, Any] | None,
    max_hints: int = 8,
) -> List[str]:
    """Extract only non-label structural gaps from the online plan trace."""
    hints: List[str] = []
    seen = set()

    def add(value: Any) -> None:
        text = _safe_hint(value)
        norm = _normalize_query(text)
        if not text or not norm or norm in seen:
            return
        seen.add(norm)
        hints.append(text)

    def visit(value: Any, parent_key: str = "", depth: int = 0) -> None:
        if len(hints) >= max_hints or depth > 5:
            return
        key = str(parent_key or "").lower()
        if "gold" in key or "reference_answer" in key:
            return
        if isinstance(value, Mapping):
            for child_key, child_value in value.items():
                child_key_norm = str(child_key or "").lower()
                if "gold" in child_key_norm or "reference_answer" in child_key_norm:
                    continue
                if child_key_norm in _ONTOLOGY_HINT_KEYS:
                    if isinstance(child_value, (list, tuple, set)):
                        for item in child_value:
                            if isinstance(item, Mapping):
                                visit(item, child_key_norm, depth + 1)
                            else:
                                add(item)
                    else:
                        add(child_value)
                elif isinstance(child_value, (Mapping, list, tuple)):
                    visit(child_value, child_key_norm, depth + 1)
                if len(hints) >= max_hints:
                    return
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item, key, depth + 1)
                if len(hints) >= max_hints:
                    return

    visit(query_plan_log or {})
    return hints[:max_hints]


def _missing_instruction(verdict: Mapping[str, Any], ontology_hints: Sequence[str]) -> str:
    parts: List[str] = []
    missing = str(verdict.get("missing", "") or "").strip()
    if missing:
        parts.append(f"missing: {missing}")
    for key, label in (
        ("missing_slots", "missing slots"),
        ("uncertain_anchors", "uncertain anchors"),
        ("rejected_anchors", "avoid distractors"),
        ("queries", "verifier suggested queries"),
    ):
        values = verdict.get(key, []) or []
        if isinstance(values, (list, tuple)) and values:
            parts.append(f"{label}: " + "; ".join(str(value) for value in values[:6]))
    if ontology_hints:
        parts.append("ontology-required evidence: " + "; ".join(ontology_hints[:8]))
    return " | ".join(parts)


def _deduplicate_queries(candidates: Iterable[Any], seen: set, limit: int) -> List[str]:
    output: List[str] = []
    for candidate in candidates:
        query = re.sub(r"\s+", " ", str(candidate or "")).strip()
        norm = _normalize_query(query)
        if not query or len(norm.split()) < 2 or norm in seen:
            continue
        seen.add(norm)
        output.append(query[:240])
        if len(output) >= max(1, int(limit)):
            break
    return output


def reciprocal_rank_fusion(
    base_doc_ids: Sequence[int],
    base_doc_scores: Sequence[float],
    residual_doc_ids: Sequence[int],
    residual_doc_scores: Sequence[float],
    *,
    depth: int = 5,
    rrf_k: float = 60.0,
    base_weight: float = 1.0,
    residual_weight: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Fuse two complete slates and retain a deterministic full ranking."""
    base_ids = [int(value) for value in list(base_doc_ids)]
    residual_ids = [int(value) for value in list(residual_doc_ids)]
    base_scores = [float(value) for value in list(base_doc_scores)]
    residual_scores = [float(value) for value in list(residual_doc_scores)]
    depth = max(1, int(depth))
    rrf_k = max(1.0, float(rrf_k))

    fused_scores: Dict[int, float] = {}
    base_rank = {doc_id: rank for rank, doc_id in enumerate(base_ids)}
    residual_rank = {doc_id: rank for rank, doc_id in enumerate(residual_ids)}
    for rank, doc_id in enumerate(base_ids[:depth], start=1):
        fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + float(base_weight) / (rrf_k + rank)
    for rank, doc_id in enumerate(residual_ids[:depth], start=1):
        fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + float(residual_weight) / (rrf_k + rank)

    prefix = sorted(
        fused_scores,
        key=lambda doc_id: (
            -fused_scores[doc_id],
            base_rank.get(doc_id, 10**9),
            residual_rank.get(doc_id, 10**9),
            doc_id,
        ),
    )
    merged_ids: List[int] = []
    seen = set()
    for doc_id in prefix + base_ids + residual_ids:
        if doc_id in seen:
            continue
        seen.add(doc_id)
        merged_ids.append(doc_id)

    floor = min(fused_scores.values(), default=1.0 / (rrf_k + depth))
    merged_scores: List[float] = []
    for rank, doc_id in enumerate(merged_ids):
        if doc_id in fused_scores:
            merged_scores.append(float(fused_scores[doc_id]))
            continue
        base_value = base_scores[base_rank[doc_id]] if doc_id in base_rank else 0.0
        residual_value = (
            residual_scores[residual_rank[doc_id]] if doc_id in residual_rank else 0.0
        )
        tie_break = max(base_value, residual_value, 0.0)
        merged_scores.append(float(floor * 0.5 / (rank + 1) + tie_break * 1e-12))

    info = {
        "mode": "equal_weight_reciprocal_rank_fusion",
        "depth": depth,
        "rrf_k": rrf_k,
        "base_weight": float(base_weight),
        "residual_weight": float(residual_weight),
        "base_top_ids": base_ids[:depth],
        "residual_top_ids": residual_ids[:depth],
        "fused_top_ids": merged_ids[:depth],
        "changed_top2": merged_ids[:2] != base_ids[:2],
        "changed_top5": merged_ids[:5] != base_ids[:5],
    }
    return np.asarray(merged_ids, dtype=np.int64), np.asarray(merged_scores, dtype=float), info


class AgenticResidualRetriever:
    """Verifier-planner-retriever-merger loop copied into the current system."""

    def __init__(self, retrieval_system: Any, config: AgenticResidualRetrievalConfig):
        self.system = retrieval_system
        self.config = config

    def run(
        self,
        question: str,
        base_doc_ids: Sequence[int],
        base_doc_scores: Sequence[float],
        query_plan_log: Mapping[str, Any] | None = None,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        config = self.config
        original_ids = np.asarray(base_doc_ids, dtype=np.int64)
        original_scores = np.asarray(base_doc_scores, dtype=float)
        residual_ids = np.asarray(base_doc_ids, dtype=np.int64)
        residual_scores = np.asarray(base_doc_scores, dtype=float)
        ontology_hints = collect_ontology_gap_hints(query_plan_log)
        initial_docs = [
            _doc_text(self.system, doc_id)
            for doc_id in residual_ids[: self.system.global_config.qa_top_k]
        ]

        router = QuestionRouterAgent(
            route_mode=config.route_mode,
            base_max_rounds=config.max_rounds,
            base_max_queries=config.max_queries,
            base_max_promotions=config.max_promotions,
            base_protect_top_k=config.protect_top_k,
            base_plan_top_k=config.plan_top_k,
            base_min_score=config.min_plan_score,
            qa_top_k=self.system.global_config.qa_top_k,
        )
        route = router.run(question, initial_docs, bridge_cues=bool(ontology_hints))
        trace: Dict[str, Any] = {
            "enabled": True,
            "implementation": "native_port_no_external_runtime_dependency",
            "roles": ["router", "verifier", "planner", "retriever", "merger"],
            "router": route,
            "ontology_gap_hints": ontology_hints,
            "rounds": [],
            "applied": False,
            "shadow_only": bool(config.shadow_only),
            "reason": "router_skipped",
        }
        if not route.get("allow_retrieval", True):
            return original_ids, original_scores, trace

        max_rounds = max(1, int(route.get("effective_max_rounds", config.max_rounds)))
        max_queries = max(1, int(route.get("effective_max_queries", config.max_queries)))
        protect_top_k = max(0, int(route.get("effective_protect_top_k", config.protect_top_k)))
        max_promotions = max(0, int(route.get("effective_max_promotions", config.max_promotions)))
        plan_top_k = max(1, int(route.get("effective_plan_top_k", config.plan_top_k)))
        min_plan_score = float(route.get("effective_min_score", config.min_plan_score))
        merge_policy = route_risk_aware_merge_policy(
            route_info=route,
            requested_merge_mode=config.merge_mode,
            protect_top_k=protect_top_k,
            max_promotions=max_promotions,
            qa_top_k=self.system.global_config.qa_top_k,
        )
        planner = PlannerAgent(
            self.system.llm_model,
            max_queries=max_queries,
            max_snippets=config.max_snippets,
            max_tokens=config.max_tokens,
        )
        verifier = VerifierAgent(
            self.system.llm_model,
            max_queries=max_queries,
            max_snippets=config.max_snippets,
            max_tokens=config.max_tokens,
        )
        retriever = RetrieverAgent(self.system)
        merger = MergerAgent(
            target_k=self.system.global_config.qa_top_k,
            protect_top_k=int(merge_policy.get("protect_top_k", protect_top_k)),
            max_promotions=int(merge_policy.get("max_promotions", max_promotions)),
            plan_top_k=plan_top_k,
            min_plan_score=min_plan_score,
            merge_mode=str(merge_policy.get("effective_merge_mode", config.merge_mode)),
            question=question,
            guarded_min_score=config.guarded_min_score,
            guarded_max_original_rank=config.guarded_max_original_rank,
            guarded_max_plan_rank=config.guarded_max_plan_rank,
            guarded_min_query_overlap=config.guarded_min_query_overlap,
            doc_lookup=lambda doc_id: _doc_text(self.system, doc_id),
        )

        seen_queries = set()
        for round_index in range(max_rounds):
            docs = [
                _doc_text(self.system, doc_id)
                for doc_id in residual_ids[: self.system.global_config.qa_top_k]
            ]
            round_trace: Dict[str, Any] = {"round": round_index + 1}
            try:
                verdict = verifier.run(question, docs)
                round_trace["verifier"] = verdict
                if verdict.get("sufficient"):
                    trace["reason"] = "initially_sufficient" if round_index == 0 else "verified_sufficient"
                    trace["rounds"].append(round_trace)
                    break

                plan = planner.run(
                    question,
                    docs,
                    missing_instruction=_missing_instruction(verdict, ontology_hints),
                )
                queries = _deduplicate_queries(
                    list(plan.get("queries", []) or []) + list(verdict.get("queries", []) or []),
                    seen_queries,
                    max_queries,
                )
                round_trace["planner"] = plan
                round_trace["queries"] = queries
                if not queries:
                    trace["reason"] = "no_new_residual_queries"
                    trace["rounds"].append(round_trace)
                    break

                retriever.prepare_queries(queries)
                planned_results = []
                retrieval_trace = []
                for residual_query in queries:
                    result = retriever.run(residual_query, ensure_embeddings=False)
                    planned_results.append(
                        (residual_query, result["doc_ids"], result["doc_scores"])
                    )
                    retrieval_trace.append({
                        "query": residual_query,
                        "mode": result.get("mode"),
                        "top_fact_count": int(result.get("top_fact_count", 0)),
                        "candidate_fact_count": int(result.get("candidate_fact_count", 0)),
                        "top_doc_ids": [int(value) for value in result["doc_ids"][:5]],
                    })
                residual_ids, residual_scores, merge_trace = merger.run(
                    base_doc_ids=residual_ids,
                    base_doc_scores=residual_scores,
                    planned_results=planned_results,
                )
                round_trace["retrieval"] = retrieval_trace
                round_trace["merge"] = merge_trace
                changed = residual_ids[:5].tolist() != original_ids[:5].tolist()
                trace["applied"] = bool(trace["applied"] or changed)
                trace["reason"] = "residual_slate_materialized" if changed else "no_new_evidence_candidates"
                trace["rounds"].append(round_trace)
            except Exception as exc:
                round_trace["error"] = f"{type(exc).__name__}: {exc}"[:500]
                trace["rounds"].append(round_trace)
                trace["reason"] = "agentic_round_failed_closed"
                break
        else:
            trace["reason"] = "max_rounds_reached"

        fused_ids, fused_scores, fusion_trace = reciprocal_rank_fusion(
            original_ids,
            original_scores,
            residual_ids,
            residual_scores,
            depth=config.fusion_depth,
            rrf_k=config.rrf_k,
            base_weight=config.base_weight,
            residual_weight=config.residual_weight,
        )
        trace["fusion"] = fusion_trace
        trace["residual_top_ids"] = [int(value) for value in residual_ids[:5]]
        trace["fused_top_ids"] = [int(value) for value in fused_ids[:5]]
        trace["applied"] = bool(fusion_trace.get("changed_top5", False))
        if config.shadow_only:
            trace["reason"] = "shadow_only"
            return original_ids, original_scores, trace
        return fused_ids, fused_scores, trace


def config_from_global(global_config: Any) -> AgenticResidualRetrievalConfig:
    prefix = "query_plan_ontology_agentic_residual_"
    return AgenticResidualRetrievalConfig(
        max_rounds=int(getattr(global_config, prefix + "max_rounds", 3)),
        max_queries=int(getattr(global_config, prefix + "max_queries", 3)),
        max_snippets=int(getattr(global_config, prefix + "max_snippets", 3)),
        max_tokens=int(getattr(global_config, prefix + "max_tokens", 160)),
        protect_top_k=int(getattr(global_config, prefix + "protect_top_k", 1)),
        max_promotions=int(getattr(global_config, prefix + "max_promotions", 2)),
        plan_top_k=int(getattr(global_config, prefix + "plan_top_k", 20)),
        min_plan_score=float(getattr(global_config, prefix + "min_plan_score", 0.18)),
        merge_mode=str(getattr(global_config, prefix + "merge_mode", "guarded_promote")),
        guarded_min_score=float(getattr(global_config, prefix + "guarded_min_score", 0.82)),
        guarded_max_original_rank=int(
            getattr(global_config, prefix + "guarded_max_original_rank", 80)
        ),
        guarded_max_plan_rank=int(
            getattr(global_config, prefix + "guarded_max_plan_rank", 8)
        ),
        guarded_min_query_overlap=float(
            getattr(global_config, prefix + "guarded_min_query_overlap", 0.08)
        ),
        route_mode=str(getattr(global_config, prefix + "route_mode", "adaptive")),
        fusion_depth=int(getattr(global_config, prefix + "fusion_depth", 5)),
        rrf_k=float(getattr(global_config, prefix + "rrf_k", 60.0)),
        base_weight=float(getattr(global_config, prefix + "base_weight", 1.0)),
        residual_weight=float(getattr(global_config, prefix + "residual_weight", 1.0)),
        shadow_only=bool(
            getattr(global_config, "query_plan_ontology_agentic_residual_shadow_only", False)
        ),
    )
