import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple


@dataclass
class QueryPlanLLMEvidenceVerifierResult:
    query_plan_log: Dict[str, Any]
    log: Dict[str, Any] = field(default_factory=dict)


class QueryPlanLLMEvidenceVerifier:
    """Use the LLM as a semantic judge for ontology-derived evidence candidates.

    The rule-based ontology retrievers often find plausible support passages,
    but their confidence scores do not separate gold/non-gold evidence well.
    This verifier is intentionally small and late-stage: it only judges a short
    list of ontology candidates and writes the decision back as candidate
    confidence for the evidence-set retriever.
    """

    def __init__(
        self,
        max_candidates: int = 8,
        min_original_rank: int = 3,
        max_original_rank: int = 30,
        min_approval_score: float = 0.70,
        approved_confidence: float = 0.98,
        rejected_confidence: float = 0.10,
        max_passage_chars: int = 850,
        max_completion_tokens: int = 700,
    ):
        self.max_candidates = max(int(max_candidates), 0)
        self.min_original_rank = max(int(min_original_rank), 1)
        self.max_original_rank = max(int(max_original_rank), 0)
        self.min_approval_score = min(max(float(min_approval_score), 0.0), 1.0)
        self.approved_confidence = min(max(float(approved_confidence), 0.0), 1.0)
        self.rejected_confidence = min(max(float(rejected_confidence), 0.0), 1.0)
        self.max_passage_chars = max(int(max_passage_chars), 120)
        self.max_completion_tokens = max(int(max_completion_tokens), 128)

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

    @staticmethod
    def _shorten(value: Any, max_chars: int) -> str:
        text = re.sub(r"\s+", " ", str(value or "").replace("\r", " ")).strip()
        if len(text) > max_chars:
            return text[:max_chars].rstrip() + "..."
        return text

    @staticmethod
    def _extract_json_object(text: Any) -> Dict[str, Any]:
        if isinstance(text, dict):
            return text
        value = str(text or "").strip()
        if not value:
            return {}
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            pass
        start = value.find("{")
        end = value.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(value[start:end + 1])
                return parsed if isinstance(parsed, dict) else {}
            except Exception:
                return {}
        return {}

    def _candidate_window(self, query_plan_log: Mapping[str, Any]) -> List[Dict[str, Any]]:
        support_log = query_plan_log.get("support_evidence_retriever", {}) or {}
        candidates = []
        for item in support_log.get("added_candidates", []) or []:
            if not isinstance(item, Mapping):
                continue
            original_rank = self._safe_int(item.get("original_rank"))
            if original_rank < self.min_original_rank:
                continue
            if self.max_original_rank > 0 and original_rank > self.max_original_rank:
                continue
            candidate = dict(item)
            candidate["_llm_verifier_source"] = "support_evidence_retriever"
            candidates.append(candidate)
        candidates.sort(
            key=lambda item: (
                self._safe_float(item.get("confidence"), self._safe_float(item.get("score"), 0.0)),
                -self._safe_int(item.get("original_rank")),
            ),
            reverse=True,
        )
        return candidates[: self.max_candidates]

    def _payload(
        self,
        query: str,
        query_plan_log: Mapping[str, Any],
        candidates: Sequence[Mapping[str, Any]],
        get_passage_text: Callable[[int], str],
    ) -> Dict[str, Any]:
        condition = query_plan_log.get("query_condition", {}) or {}
        matched_plans = []
        for plan in query_plan_log.get("matched_plans", []) or []:
            if not isinstance(plan, Mapping):
                continue
            matched_plans.append({
                "id": plan.get("id", ""),
                "intent": plan.get("intent", ""),
                "answer_type": plan.get("answer_type", ""),
                "relation_families": plan.get("relation_families", []) or [],
                "relation_signatures": plan.get("relation_signatures", []) or [],
            })
            if len(matched_plans) >= 4:
                break
        payload_candidates = []
        for item in candidates:
            passage_idx = self._safe_int(item.get("passage_idx"), -1)
            probe = item.get("probe", {}) or {}
            payload_candidates.append({
                "passage_idx": passage_idx,
                "original_rank": self._safe_int(item.get("original_rank")),
                "rule_confidence": round(self._safe_float(item.get("confidence")), 4),
                "probe": {
                    "anchor": probe.get("anchor", ""),
                    "relation": probe.get("relation", ""),
                    "terminal": probe.get("terminal", ""),
                    "kind": probe.get("kind", ""),
                },
                "matched": {
                    "anchor": bool(item.get("matched_anchor")),
                    "terminal": bool(item.get("matched_terminal")),
                    "relation_hits": item.get("relation_hits", []) or [],
                    "answer_hit": bool(item.get("answer_hit")),
                },
                "passage": self._shorten(get_passage_text(passage_idx), self.max_passage_chars),
            })
        return {
            "query": query,
            "query_plan": {
                "answer_type": condition.get("answer_type", ""),
                "answer_hints": condition.get("answer_hints", []) or [],
                "relation_terms": condition.get("relation_terms", []) or [],
                "relation_families": condition.get("relation_families", []) or [],
                "anchor_entities": query_plan_log.get("anchor_entities", []) or [],
                "matched_plans": matched_plans,
            },
            "candidates": payload_candidates,
        }

    def verify(
        self,
        query: str,
        query_plan_log: Mapping[str, Any],
        llm_infer: Callable[..., Tuple[str, Dict[str, Any]]],
        get_passage_text: Callable[[int], str],
    ) -> QueryPlanLLMEvidenceVerifierResult:
        updated_log = dict(query_plan_log)
        log: Dict[str, Any] = {
            "enabled": True,
            "max_candidates": self.max_candidates,
            "min_original_rank": self.min_original_rank,
            "max_original_rank": self.max_original_rank,
            "min_approval_score": self.min_approval_score,
            "considered_count": 0,
            "approved_count": 0,
            "rejected_count": 0,
            "decisions": [],
        }
        if self.max_candidates <= 0:
            log["skip_reason"] = "disabled_by_zero_limit"
            return QueryPlanLLMEvidenceVerifierResult(updated_log, log)
        if not query_plan_log or not query_plan_log.get("enabled", False):
            log["skip_reason"] = "no_query_plan_log"
            return QueryPlanLLMEvidenceVerifierResult(updated_log, log)
        support_log = dict(query_plan_log.get("support_evidence_retriever", {}) or {})
        support_items = [
            dict(item) for item in (support_log.get("added_candidates", []) or [])
            if isinstance(item, Mapping)
        ]
        if not support_items:
            log["skip_reason"] = "no_support_candidates"
            return QueryPlanLLMEvidenceVerifierResult(updated_log, log)

        candidates = self._candidate_window(query_plan_log)
        log["considered_count"] = len(candidates)
        if not candidates:
            log["skip_reason"] = "no_candidates_in_rank_window"
            return QueryPlanLLMEvidenceVerifierResult(updated_log, log)

        payload = self._payload(query, query_plan_log, candidates, get_passage_text)
        system_prompt = (
            "You are an ontology-guided evidence verifier for multi-hop retrieval. "
            "Given a question, its query-plan ontology, and candidate passages, select only passages "
            "that are likely useful supporting evidence for answering the question. "
            "A passage must connect the concrete query anchor to the required relation, terminal entity, "
            "or answer type. Reject passages that merely mention the topic but do not help the reasoning path. "
            "Return only JSON: {\"approved\":[{\"passage_idx\":int,\"score\":0-1,\"reason\":\"short\"}],"
            "\"rejected\":[{\"passage_idx\":int,\"reason\":\"short\"}]}."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    "Input JSON:\n"
                    + json.dumps(payload, ensure_ascii=False)
                    + "\nReturn only one valid JSON object."
                ),
            },
        ]
        try:
            response = llm_infer(
                messages=messages,
                max_completion_tokens=self.max_completion_tokens,
                temperature=0,
            )
            response_text = response[0] if isinstance(response, tuple) else response
            parsed = self._extract_json_object(response_text)
        except Exception as exc:
            log["skip_reason"] = "llm_error"
            log["error"] = str(exc)
            return QueryPlanLLMEvidenceVerifierResult(updated_log, log)

        approved = {}
        for item in parsed.get("approved", []) or []:
            if not isinstance(item, Mapping):
                continue
            passage_idx = self._safe_int(item.get("passage_idx"), -1)
            score = self._safe_float(item.get("score"), 0.0)
            if passage_idx >= 0 and score >= self.min_approval_score:
                approved[passage_idx] = {
                    "score": min(max(score, 0.0), 1.0),
                    "reason": str(item.get("reason", ""))[:240],
                }
        rejected_reasons = {}
        for item in parsed.get("rejected", []) or []:
            if isinstance(item, Mapping):
                passage_idx = self._safe_int(item.get("passage_idx"), -1)
                if passage_idx >= 0:
                    rejected_reasons[passage_idx] = str(item.get("reason", ""))[:240]

        considered_ids = {self._safe_int(item.get("passage_idx"), -1) for item in candidates}
        new_items = []
        for item in support_items:
            passage_idx = self._safe_int(item.get("passage_idx"), -1)
            if passage_idx not in considered_ids:
                new_items.append(item)
                continue
            decision = approved.get(passage_idx)
            if decision:
                llm_confidence = self.approved_confidence * decision["score"]
                item["confidence"] = max(self._safe_float(item.get("confidence")), llm_confidence)
                item["score"] = max(self._safe_float(item.get("score")), llm_confidence)
                item["llm_evidence_verifier"] = {
                    "approved": True,
                    "score": decision["score"],
                    "reason": decision["reason"],
                }
                log["approved_count"] += 1
                log["decisions"].append({
                    "passage_idx": passage_idx,
                    "approved": True,
                    "score": round(decision["score"], 4),
                    "reason": decision["reason"],
                    "original_rank": self._safe_int(item.get("original_rank")),
                })
            else:
                item["confidence"] = min(self._safe_float(item.get("confidence")), self.rejected_confidence)
                item["score"] = min(self._safe_float(item.get("score")), self.rejected_confidence)
                item["llm_evidence_verifier"] = {
                    "approved": False,
                    "reason": rejected_reasons.get(passage_idx, "not_approved"),
                }
                log["rejected_count"] += 1
                if len(log["decisions"]) < 16:
                    log["decisions"].append({
                        "passage_idx": passage_idx,
                        "approved": False,
                        "reason": rejected_reasons.get(passage_idx, "not_approved"),
                        "original_rank": self._safe_int(item.get("original_rank")),
                    })
            new_items.append(item)

        support_log["added_candidates"] = new_items
        support_log["llm_evidence_verifier_applied"] = True
        updated_log["support_evidence_retriever"] = support_log
        if not parsed:
            log["skip_reason"] = "parse_failed"
        elif log["approved_count"] <= 0:
            log["skip_reason"] = "no_approved_candidates"
        return QueryPlanLLMEvidenceVerifierResult(updated_log, log)
