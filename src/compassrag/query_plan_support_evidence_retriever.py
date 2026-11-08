from dataclasses import dataclass, field
import json
import os
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from .query_ontology_prior import tokenize
from .query_plan_structural_candidate_retriever import QueryPlanOntologyStructuralCandidateRetriever
from .utils.misc_utils import text_processing


@dataclass
class QueryPlanSupportEvidenceResult:
    sorted_doc_ids: np.ndarray
    sorted_doc_scores: np.ndarray
    log: Dict[str, Any] = field(default_factory=dict)


class QueryPlanSupportEvidenceRetriever:
    """Generate ontology-guided dense support candidates.

    Subgraph retrieval scans the fact graph and then maps facts to passages.
    That can produce a huge number of relation-only candidates. This retriever
    takes the opposite route: use the query-plan ontology to form focused
    support-document probes, retrieve passages with the dense index, and verify
    that a passage mentions the concrete query anchor plus relation/answer-type
    evidence before exposing it to the evidence-set selector.
    """

    GENERIC_TERMS = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
        "has", "have", "had", "in", "is", "of", "on", "or", "the", "to",
        "was", "were", "with",
    }

    ANSWER_HINTS = {
        "date": ["date", "year", "month", "day", "born", "died", "death", "released"],
        "number": ["number", "count", "total", "age", "rank", "population"],
        "location": ["place", "city", "country", "state", "county", "location"],
        "person": ["person", "name", "who"],
        "organization": ["organization", "company", "team", "group"],
    }

    def __init__(
        self,
        dense_top_k: int = 30,
        max_probes: int = 10,
        max_added_passages: int = 4,
        min_confidence: float = 0.70,
        preserve_top_k: int = 2,
        synthetic_original_rank: int = 40,
        require_anchor_hit: bool = True,
        require_relation_or_answer_hit: bool = True,
        candidate_only: bool = True,
        append_score_margin: float = 1e-7,
        probe_policy_path: str = "",
        probe_policy_mode: str = "off",
    ):
        self.dense_top_k = max(int(dense_top_k), 1)
        self.max_probes = max(int(max_probes), 0)
        self.max_added_passages = max(int(max_added_passages), 0)
        self.min_confidence = min(max(float(min_confidence), 0.0), 1.0)
        self.preserve_top_k = max(int(preserve_top_k), 0)
        self.synthetic_original_rank = max(int(synthetic_original_rank), 1)
        self.require_anchor_hit = bool(require_anchor_hit)
        self.require_relation_or_answer_hit = bool(require_relation_or_answer_hit)
        self.candidate_only = bool(candidate_only)
        self.append_score_margin = max(float(append_score_margin), 0.0)
        self.probe_policy_path = str(probe_policy_path or "")
        self.probe_policy_mode = str(probe_policy_mode or "off").lower()
        self.probe_policy = self._load_probe_policy(self.probe_policy_path)

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _norm(value: Any) -> str:
        return text_processing(str(value or ""))

    @staticmethod
    def _policy_relation_key(value: Any) -> str:
        tokens = [
            token for token in tokenize(text_processing(str(value or "").replace("_", " ")))
            if token not in QueryPlanSupportEvidenceRetriever.GENERIC_TERMS
        ]
        return " ".join(tokens[:5])

    @staticmethod
    def _policy_key(plan_id: str, kind: str, relation: str) -> str:
        return f"{plan_id}\t{kind}\t{relation or '*'}"

    @staticmethod
    def _load_probe_policy(path: str) -> Dict[str, Any]:
        if not path or not os.path.exists(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as policy_file:
                policy = json.load(policy_file)
            return policy if isinstance(policy, dict) else {}
        except Exception:
            return {}

    def _probe_policy_entry(self, probe: Mapping[str, Any]) -> Dict[str, Any]:
        entries = self.probe_policy.get("entries", {}) if isinstance(self.probe_policy, Mapping) else {}
        if not isinstance(entries, Mapping):
            return {}
        plan_id = str(probe.get("plan_id") or "unknown_plan")
        kind = str(probe.get("kind") or "unknown")
        relation = self._policy_relation_key(probe.get("relation") or "")
        keys = [
            self._policy_key(plan_id, kind, relation),
            self._policy_key(plan_id, kind, "*"),
            self._policy_key("*", kind, relation),
        ]
        for key in keys:
            entry = entries.get(key)
            if isinstance(entry, Mapping):
                return dict(entry)
        return {}

    def _probe_allowed_by_policy(self, probe: Mapping[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        if self.probe_policy_mode in {"", "off", "none"} or not self.probe_policy:
            return True, {"decision": "policy_disabled"}
        entry = self._probe_policy_entry(probe)
        if not entry:
            return True, {"decision": "policy_unknown"}
        decision = str(entry.get("decision") or "unknown").lower()
        if self.probe_policy_mode in {"filter", "strict"} and decision == "reject":
            return False, entry
        return True, entry

    @classmethod
    def _non_generic_tokens(cls, value: Any, max_tokens: int = 5) -> List[str]:
        output = []
        seen = set()
        for token in tokenize(cls._norm(str(value or "").replace("_", " "))):
            if token in cls.GENERIC_TERMS or token in seen:
                continue
            output.append(token)
            seen.add(token)
            if len(output) >= max_tokens:
                break
        return output

    @classmethod
    def _contains_phrase(cls, text_norm: str, phrase: Any) -> bool:
        phrase_norm = cls._norm(phrase)
        if not phrase_norm:
            return False
        if f" {phrase_norm} " in f" {text_norm} ":
            return True
        phrase_tokens = cls._non_generic_tokens(phrase_norm, max_tokens=8)
        if not phrase_tokens:
            return False
        text_tokens = set(tokenize(text_norm))
        matched = sum(1 for token in phrase_tokens if token in text_tokens)
        return matched / max(len(phrase_tokens), 1) >= 0.75

    @staticmethod
    def _unique(values: Iterable[Any], max_items: int = 32) -> List[str]:
        output = []
        seen = set()
        for value in values or []:
            text = text_processing(str(value or ""))
            if not text or text in seen:
                continue
            output.append(text)
            seen.add(text)
            if len(output) >= max_items:
                break
        return output

    def _concrete_anchors(self, query_plan_log: Mapping[str, Any]) -> List[str]:
        anchors = []
        anchors.extend(query_plan_log.get("anchor_entities", []) or [])
        for path in query_plan_log.get("sample_paths", []) or []:
            if isinstance(path, Mapping):
                anchors.append(path.get("anchor_entity", ""))
        for passage in query_plan_log.get("lifted_passages", []) or []:
            for context in passage.get("contexts", []) or []:
                if isinstance(context, Mapping):
                    anchors.append(context.get("anchor_entity", ""))
        cleaned = []
        for anchor in self._unique(anchors, max_items=24):
            if not QueryPlanOntologyStructuralCandidateRetriever.is_generic_anchor(anchor):
                cleaned.append(anchor)
        return cleaned

    def _relation_phrases(self, query_plan_log: Mapping[str, Any]) -> List[str]:
        values = []
        query_condition = query_plan_log.get("query_condition", {}) or {}
        values.extend(query_condition.get("relation_terms", []) or [])
        values.extend(query_condition.get("relation_signatures", []) or [])
        values.extend(query_condition.get("relation_families", []) or [])
        for plan in query_plan_log.get("matched_plans", []) or []:
            if isinstance(plan, Mapping):
                values.extend(plan.get("relation_signatures", []) or [])
                values.extend(plan.get("relation_families", []) or [])
                values.extend(plan.get("triggers", []) or [])
        for path in query_plan_log.get("sample_paths", []) or []:
            if isinstance(path, Mapping):
                values.extend(path.get("relations", []) or [])
        phrases = []
        for value in self._unique(values, max_items=32):
            tokens = self._non_generic_tokens(value, max_tokens=5)
            if tokens:
                phrases.append(" ".join(tokens))
        return self._unique(phrases, max_items=16)

    def _terminal_terms(self, query_plan_log: Mapping[str, Any]) -> List[str]:
        terminals = []
        for path in query_plan_log.get("sample_paths", []) or []:
            if isinstance(path, Mapping):
                terminals.append(path.get("end_entity", ""))
                terminals.append(path.get("entity", ""))
        for passage in query_plan_log.get("lifted_passages", []) or []:
            for context in passage.get("contexts", []) or []:
                if isinstance(context, Mapping):
                    terminals.append(context.get("end_entity", ""))
        return [
            item for item in self._unique(terminals, max_items=24)
            if not QueryPlanOntologyStructuralCandidateRetriever.is_generic_anchor(item)
        ]

    def _answer_hints(self, query_plan_log: Mapping[str, Any]) -> List[str]:
        query_condition = query_plan_log.get("query_condition", {}) or {}
        answer_type = self._norm(query_condition.get("answer_type", ""))
        hints = []
        hints.extend(query_condition.get("answer_hints", []) or [])
        hints.extend(self.ANSWER_HINTS.get(answer_type, []))
        if answer_type:
            hints.append(answer_type)
        return self._unique(hints, max_items=8)

    def _build_probes(self, query: str, query_plan_log: Mapping[str, Any]) -> List[Dict[str, Any]]:
        anchors = self._concrete_anchors(query_plan_log)
        relations = self._relation_phrases(query_plan_log)
        terminals = self._terminal_terms(query_plan_log)
        answer_hints = self._answer_hints(query_plan_log)
        matched_plans = [
            plan for plan in query_plan_log.get("matched_plans", []) or []
            if isinstance(plan, Mapping)
        ]
        plan_specs = []
        for plan in matched_plans[:8]:
            plan_id = str(plan.get("id") or "support_evidence")
            precision = self._safe_float(
                plan.get("effective_precision", plan.get("precision", 0.0)),
                0.0,
            )
            for step_idx, step in enumerate(plan.get("path", []) or []):
                if not isinstance(step, Mapping):
                    continue
                values = []
                values.append(step.get("relation_signature", ""))
                values.append(step.get("relation_family", ""))
                values.extend(step.get("relation_aliases", []) or [])
                for value in values[:4]:
                    phrase = " ".join(self._non_generic_tokens(value, max_tokens=5))
                    if phrase:
                        plan_specs.append({
                            "plan_id": plan_id,
                            "plan_effective_precision": precision,
                            "step": step_idx,
                            "relation": phrase,
                            "relation_signature": step.get("relation_signature", ""),
                        })
        if not plan_specs:
            default_plan_id = str(matched_plans[0].get("id", "support_evidence")) if matched_plans else "support_evidence"
            default_precision = max(
                (self._safe_float(plan.get("effective_precision"), 0.0) for plan in matched_plans),
                default=0.0,
            )
            for relation in (relations[:6] or [""]):
                plan_specs.append({
                    "plan_id": default_plan_id,
                    "plan_effective_precision": default_precision,
                    "step": -1,
                    "relation": relation,
                    "relation_signature": relation,
                })
        probes = []
        seen = set()

        def add_probe(
            text: str,
            anchor: str = "",
            relation: str = "",
            terminal: str = "",
            kind: str = "support",
            plan_id: str = "support_evidence",
            plan_precision: float = 0.0,
            step: int = -1,
            relation_signature: str = "",
        ):
            key = self._norm(text)
            if not key or key in seen:
                return
            seen.add(key)
            probes.append({
                "query": text,
                "anchor": anchor,
                "relation": relation,
                "terminal": terminal,
                "answer_hints": answer_hints,
                "plan_id": plan_id,
                "plan_effective_precision": plan_precision,
                "step": step,
                "relation_signature": relation_signature,
                "kind": kind,
            })

        hint_text = " ".join(answer_hints[:2])
        for anchor in anchors[:6]:
            for spec in plan_specs[:12]:
                relation = spec.get("relation", "")
                pieces = [anchor, relation, hint_text]
                add_probe(
                    " ".join(piece for piece in pieces if piece),
                    anchor,
                    relation,
                    "",
                    "anchor_relation",
                    str(spec.get("plan_id", "support_evidence")),
                    self._safe_float(spec.get("plan_effective_precision"), 0.0),
                    int(spec.get("step", -1)),
                    str(spec.get("relation_signature", "")),
                )
                add_probe(
                    " ".join(piece for piece in [query, anchor, relation] if piece),
                    anchor,
                    relation,
                    "",
                    "query_anchor_relation",
                    str(spec.get("plan_id", "support_evidence")),
                    self._safe_float(spec.get("plan_effective_precision"), 0.0),
                    int(spec.get("step", -1)),
                    str(spec.get("relation_signature", "")),
                )
            for terminal in terminals[:4]:
                spec = plan_specs[0] if plan_specs else {}
                add_probe(
                    " ".join(piece for piece in [anchor, terminal, hint_text] if piece),
                    anchor,
                    "",
                    terminal,
                    "anchor_terminal",
                    str(spec.get("plan_id", "support_evidence")),
                    self._safe_float(spec.get("plan_effective_precision"), 0.0),
                    int(spec.get("step", -1)),
                    str(spec.get("relation_signature", "")),
                )

        for terminal in terminals[:6]:
            spec = plan_specs[0] if plan_specs else {}
            relation = str(spec.get("relation", "")) if spec else (relations[0] if relations else "")
            add_probe(
                " ".join(piece for piece in [terminal, relation, hint_text] if piece),
                "",
                relation,
                terminal,
                "terminal_relation",
                str(spec.get("plan_id", "support_evidence")),
                self._safe_float(spec.get("plan_effective_precision"), 0.0),
                int(spec.get("step", -1)),
                str(spec.get("relation_signature", "")),
            )

        if not probes and (relations or answer_hints):
            spec = plan_specs[0] if plan_specs else {}
            relation = " ".join(relations[:2])
            add_probe(
                " ".join(piece for piece in [query, relation, hint_text] if piece),
                "",
                relation,
                "",
                "query_relation",
                str(spec.get("plan_id", "support_evidence")),
                self._safe_float(spec.get("plan_effective_precision"), 0.0),
                int(spec.get("step", -1)),
                str(spec.get("relation_signature", "")),
            )

        return probes[: self.max_probes]

    def _answer_type_hit(self, text_norm: str, hints: Sequence[str]) -> bool:
        tokens = set(tokenize(text_norm))
        hint_tokens = {token for hint in hints for token in self._non_generic_tokens(hint, max_tokens=3)}
        if hint_tokens and tokens.intersection(hint_tokens):
            return True

        if any(hint in {"date", "year", "month", "day"} for hint in hints):
            if any(token.isdigit() and len(token) == 4 for token in tokens):
                return True
        if any(hint in {"number", "count", "total", "age"} for hint in hints):
            if any(token.isdigit() for token in tokens):
                return True
        return False

    def _score_candidate(
        self,
        passage_text: str,
        dense_score: float,
        max_dense_score: float,
        probe: Mapping[str, Any],
    ) -> Tuple[float, Dict[str, Any]]:
        text_norm = self._norm(passage_text)
        anchor_hit = self._contains_phrase(text_norm, probe.get("anchor", ""))
        terminal_hit = self._contains_phrase(text_norm, probe.get("terminal", ""))
        relation_tokens = self._non_generic_tokens(probe.get("relation", ""), max_tokens=5)
        text_tokens = set(tokenize(text_norm))
        relation_hits = [token for token in relation_tokens if token in text_tokens]
        relation_score = min(len(relation_hits) / max(min(len(relation_tokens), 3), 1), 1.0)
        answer_hit = self._answer_type_hit(text_norm, probe.get("answer_hints", []) or [])
        dense_norm = 0.0 if max_dense_score <= 0.0 else float(dense_score) / max_dense_score
        dense_norm = min(max(dense_norm, 0.0), 1.0)
        precision = min(max(self._safe_float(probe.get("plan_effective_precision"), 0.0), 0.0), 1.0)

        anchor_score = 1.0 if anchor_hit else 0.0
        terminal_score = 1.0 if terminal_hit else 0.0
        answer_score = 1.0 if answer_hit else 0.0
        confidence = (
            0.32 * dense_norm
            + 0.26 * anchor_score
            + 0.16 * relation_score
            + 0.12 * terminal_score
            + 0.08 * answer_score
            + 0.06 * precision
        )
        detail = {
            "anchor_hit": anchor_hit,
            "terminal_hit": terminal_hit,
            "relation_hits": relation_hits,
            "answer_hit": answer_hit,
            "dense_norm": dense_norm,
            "plan_precision": precision,
            "confidence": min(max(confidence, 0.0), 1.0),
        }
        return detail["confidence"], detail

    def retrieve(
        self,
        query: str,
        sorted_doc_ids: Sequence[int],
        sorted_doc_scores: Sequence[float],
        query_plan_log: Mapping[str, Any],
        dense_retrieve: Callable[[str], Tuple[np.ndarray, np.ndarray]],
        get_passage_text: Callable[[int], str],
    ) -> QueryPlanSupportEvidenceResult:
        base_doc_ids = [int(doc_id) for doc_id in list(sorted_doc_ids)]
        base_scores = np.array(sorted_doc_scores, dtype=np.float32)
        log: Dict[str, Any] = {
            "enabled": True,
            "dense_top_k": self.dense_top_k,
            "max_probes": self.max_probes,
            "max_added_passages": self.max_added_passages,
            "min_confidence": self.min_confidence,
            "synthetic_original_rank": self.synthetic_original_rank,
            "candidate_only": self.candidate_only,
            "probes": [],
            "added_candidates": [],
            "rejection_reasons": {},
        }
        if self.max_added_passages <= 0 or self.max_probes <= 0:
            log["skip_reason"] = "disabled_by_zero_limit"
            return QueryPlanSupportEvidenceResult(np.array(base_doc_ids, dtype=np.intp), base_scores, log)
        if not base_doc_ids:
            log["skip_reason"] = "empty_retrieval"
            return QueryPlanSupportEvidenceResult(np.array(base_doc_ids, dtype=np.intp), base_scores, log)
        if not query_plan_log or not query_plan_log.get("enabled", False):
            log["skip_reason"] = "no_query_plan_log"
            return QueryPlanSupportEvidenceResult(np.array(base_doc_ids, dtype=np.intp), base_scores, log)

        raw_probes = self._build_probes(query, query_plan_log)
        probes = []
        policy_rejected = 0
        for probe in raw_probes:
            allowed, policy_entry = self._probe_allowed_by_policy(probe)
            probe["policy"] = {
                "decision": str(policy_entry.get("decision", "allow")),
                "precision": policy_entry.get("precision"),
                "target_label": policy_entry.get("target_label"),
                "target_hits": policy_entry.get("target_hits"),
                "target_precision": policy_entry.get("target_precision"),
                "candidates": policy_entry.get("candidates"),
                "gold": policy_entry.get("gold"),
                "gold_from_gt2": policy_entry.get("gold_from_gt2"),
                "gold_from_gt5": policy_entry.get("gold_from_gt5"),
                "recall_boost": policy_entry.get("recall_boost"),
                "recall_min_confidence": policy_entry.get("recall_min_confidence"),
            }
            if not allowed:
                policy_rejected += 1
                continue
            probes.append(probe)
        log["probes"] = [
            {
                "query": probe.get("query", ""),
                "anchor": probe.get("anchor", ""),
                "relation": probe.get("relation", ""),
                "terminal": probe.get("terminal", ""),
                "kind": probe.get("kind", ""),
                "plan_id": probe.get("plan_id", ""),
                "step": probe.get("step", -1),
                "relation_signature": probe.get("relation_signature", ""),
                "policy": probe.get("policy", {}),
            }
            for probe in probes
        ]
        log["raw_probe_count"] = len(raw_probes)
        log["policy_rejected_probe_count"] = policy_rejected
        log["probe_policy_mode"] = self.probe_policy_mode
        log["probe_policy_loaded"] = bool(self.probe_policy)
        if not probes:
            log["skip_reason"] = "no_support_probes_after_policy" if raw_probes else "no_support_probes"
            return QueryPlanSupportEvidenceResult(np.array(base_doc_ids, dtype=np.intp), base_scores, log)

        original_rank_by_doc = {int(doc_id): rank + 1 for rank, doc_id in enumerate(base_doc_ids)}
        preserved = set(base_doc_ids[: self.preserve_top_k])
        selected_by_passage: Dict[int, Dict[str, Any]] = {}
        rejection_reasons: Dict[str, int] = {}

        for probe in probes:
            dense_doc_ids, dense_scores = dense_retrieve(str(probe.get("query", "")))
            dense_doc_ids = list(dense_doc_ids[: self.dense_top_k])
            dense_scores = list(dense_scores[: self.dense_top_k])
            max_dense_score = max((self._safe_float(score) for score in dense_scores), default=0.0)
            for dense_rank, passage_idx in enumerate(dense_doc_ids, start=1):
                passage_idx = int(passage_idx)
                if passage_idx in preserved:
                    rejection_reasons["preserved_top"] = rejection_reasons.get("preserved_top", 0) + 1
                    continue
                passage_text = get_passage_text(passage_idx)
                confidence, detail = self._score_candidate(
                    passage_text=passage_text,
                    dense_score=self._safe_float(dense_scores[dense_rank - 1] if dense_rank - 1 < len(dense_scores) else 0.0),
                    max_dense_score=max_dense_score,
                    probe=probe,
                )
                policy = probe.get("policy", {}) if isinstance(probe.get("policy", {}), Mapping) else {}
                policy_decision = str(policy.get("decision", "")).lower()
                recall_policy = policy_decision == "recall_allow"
                weak_policy = policy_decision == "weak_allow"
                if recall_policy:
                    recall_boost = max(1.0, self._safe_float(policy.get("recall_boost"), 1.12))
                    confidence = min(1.0, confidence * recall_boost + 0.03)
                    detail["policy_recall_boost"] = recall_boost
                    detail["confidence_after_policy"] = confidence
                elif weak_policy:
                    confidence = min(1.0, confidence * 1.02)
                    detail["policy_recall_boost"] = 1.02
                    detail["confidence_after_policy"] = confidence
                if self.require_anchor_hit and probe.get("anchor") and not detail["anchor_hit"]:
                    rejection_reasons["anchor_miss"] = rejection_reasons.get("anchor_miss", 0) + 1
                    continue
                relation_or_answer = bool(detail["relation_hits"] or detail["answer_hit"] or detail["terminal_hit"])
                if self.require_relation_or_answer_hit and not relation_or_answer:
                    if recall_policy and detail["anchor_hit"] and self._safe_float(detail.get("dense_norm"), 0.0) >= 0.86:
                        detail["policy_relation_relaxed"] = True
                    else:
                        rejection_reasons["relation_answer_miss"] = rejection_reasons.get("relation_answer_miss", 0) + 1
                        continue
                effective_min_confidence = self.min_confidence
                if recall_policy:
                    effective_min_confidence = min(
                        effective_min_confidence,
                        self._safe_float(policy.get("recall_min_confidence"), 0.52),
                    )
                elif weak_policy:
                    effective_min_confidence = min(effective_min_confidence, 0.58)
                if confidence < effective_min_confidence:
                    rejection_reasons["low_confidence"] = rejection_reasons.get("low_confidence", 0) + 1
                    continue
                original_rank = int(original_rank_by_doc.get(passage_idx, self.synthetic_original_rank))
                candidate = {
                    "passage_idx": passage_idx,
                    "confidence": float(confidence),
                    "dense_rank": int(dense_rank),
                    "original_rank": original_rank,
                    "was_in_baseline": passage_idx in original_rank_by_doc,
                    "score": float(confidence + 0.01 / max(dense_rank, 1)),
                    "probe": {
                        "query": probe.get("query", ""),
                        "anchor": probe.get("anchor", ""),
                        "relation": probe.get("relation", ""),
                        "terminal": probe.get("terminal", ""),
                        "kind": probe.get("kind", ""),
                        "plan_id": probe.get("plan_id", ""),
                        "step": probe.get("step", -1),
                        "relation_signature": probe.get("relation_signature", ""),
                        "policy": probe.get("policy", {}),
                    },
                    "matched_anchor": bool(detail["anchor_hit"]),
                    "matched_terminal": bool(detail["terminal_hit"]),
                    "relation_hits": detail["relation_hits"],
                    "answer_hit": bool(detail["answer_hit"]),
                    "probe_policy_decision": policy_decision,
                    "effective_min_confidence": float(effective_min_confidence),
                    "contexts": [{
                        "support_evidence": True,
                        "support_key": f"support_evidence::{probe.get('kind', '')}::{passage_idx}",
                        "plan_id": probe.get("plan_id", "support_evidence"),
                        "anchor_entity": probe.get("anchor", ""),
                        "end_entity": probe.get("terminal", ""),
                        "relations": [probe.get("relation", "")] if probe.get("relation") else [],
                        "fact_indices": [],
                        "fact_position": 0,
                        "signature_match_mode": "support_dense",
                        "plan_effective_precision": probe.get("plan_effective_precision", 0.0),
                        "plan_lift_weight_scale": 0.90,
                        "score": float(confidence),
                    }],
                }
                existing = selected_by_passage.get(passage_idx)
                if existing is None or candidate["score"] > existing["score"]:
                    selected_by_passage[passage_idx] = candidate

        candidates = sorted(
            selected_by_passage.values(),
            key=lambda item: (
                float(item.get("score", 0.0)),
                -int(item.get("dense_rank", 10**9)),
                -int(item.get("original_rank", 10**9)),
            ),
            reverse=True,
        )
        selected = candidates[: self.max_added_passages]
        log["candidate_count"] = len(candidates)
        log["selected_count"] = len(selected)
        log["added_candidates"] = selected
        log["rejection_reasons"] = dict(sorted(rejection_reasons.items(), key=lambda kv: kv[1], reverse=True)[:8])
        if not selected:
            log["skip_reason"] = "no_verified_support_candidates"
            return QueryPlanSupportEvidenceResult(np.array(base_doc_ids, dtype=np.intp), base_scores, log)
        if self.candidate_only:
            appended = [int(item["passage_idx"]) for item in selected if int(item["passage_idx"]) not in set(base_doc_ids)]
            score_by_doc = {
                int(doc_id): float(base_scores[idx]) if idx < len(base_scores) else 0.0
                for idx, doc_id in enumerate(base_doc_ids)
            }
            min_tail_score = float(base_scores[-1]) if len(base_scores) else 0.0
            for rank, passage_idx in enumerate(appended, start=1):
                score_by_doc[passage_idx] = min_tail_score - self.append_score_margin * rank
            new_doc_ids = base_doc_ids + [idx for idx in appended if idx not in set(base_doc_ids)]
            new_scores = np.array([score_by_doc[int(idx)] for idx in new_doc_ids], dtype=np.float32)
            log["appended_passage_count"] = len(appended)
            log["skip_reason"] = "candidate_only"
            return QueryPlanSupportEvidenceResult(np.array(new_doc_ids, dtype=np.intp), new_scores, log)

        selected_ids = {int(item["passage_idx"]) for item in selected}
        preserved_ids = [int(doc_id) for doc_id in base_doc_ids[: self.preserve_top_k]]
        rest = [int(doc_id) for doc_id in base_doc_ids if int(doc_id) not in selected_ids and int(doc_id) not in preserved_ids]
        reordered = preserved_ids + [int(item["passage_idx"]) for item in selected] + rest
        seen = set()
        final_ids = []
        for doc_id in reordered:
            if doc_id in seen:
                continue
            final_ids.append(doc_id)
            seen.add(doc_id)
        high = float(np.max(base_scores)) if base_scores.size else 1.0
        step = max(abs(high), 1.0) * self.append_score_margin
        final_scores = np.array([high - step * i for i in range(len(final_ids))], dtype=np.float32)
        final_rank = {int(doc_id): rank + 1 for rank, doc_id in enumerate(final_ids)}
        for item in selected:
            item["final_rank"] = int(final_rank.get(int(item["passage_idx"]), 0))
            item["rank_delta"] = int(item.get("original_rank", 0)) - int(item.get("final_rank", 0))
        log["changed_top5"] = list(base_doc_ids[:5]) != final_ids[:5]
        log["changed_top2"] = list(base_doc_ids[:2]) != final_ids[:2]
        return QueryPlanSupportEvidenceResult(np.array(final_ids, dtype=np.intp), final_scores, log)
