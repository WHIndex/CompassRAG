import re
from typing import Any, Dict, List, Optional

from .query_plan_adapter import QueryPlanOntologyAdapter
from .utils.misc_utils import text_processing


class QueryPlanPassagePromotionVerifier:
    """Conservative verifier for ontology passage promotion.

    A passage can be ontology-related but still not be a gold supporting
    passage. Promotion is only worthwhile when the path looks like direct
    support for the answer-bearing passage, not merely a related entity.
    """

    BLOCKED_QUERY_TYPES = {
        "category_or_alias",
        "direct_slot_factoid",
        "extractive_factoid",
        "comparative",
        "discourse_or_long_answer",
        "non_extractive_answer",
        "explanatory_or_measure",
        "long_compositional_factoid",
    }

    GENERIC_ANCHORS = {
        "first",
        "second",
        "third",
        "north",
        "south",
        "east",
        "west",
        "region",
        "country",
        "city",
        "place",
        "person",
        "people",
        "film",
        "movie",
        "album",
        "book",
        "novel",
        "song",
        "mountain",
        "mountain chain",
    }

    MONTH_RE = re.compile(
        r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b"
    )

    def __init__(
        self,
        min_confidence: float = 0.85,
        max_original_rank: int = 5,
    ):
        self.min_confidence = min(max(float(min_confidence), 0.0), 1.0)
        self.max_original_rank = max(int(max_original_rank), 0)
        self.adapter = QueryPlanOntologyAdapter()

    @staticmethod
    def _path_len(context: Dict[str, Any]) -> int:
        return max(
            len(context.get("fact_indices", []) or []),
            len(context.get("relations", []) or []),
            1,
        )

    @staticmethod
    def _is_date_like(text: str) -> bool:
        text = text_processing(text)
        return bool(re.search(r"\b\d{3,4}\b", text) or QueryPlanPassagePromotionVerifier.MONTH_RE.search(text))

    @staticmethod
    def _is_numeric_like(text: str) -> bool:
        return bool(re.search(r"\d", text_processing(text)))

    def infer_answer_form(self, query: str, query_condition: Optional[Dict[str, Any]]) -> str:
        query_text = text_processing(query)
        answer_type = self.adapter.answer_type(query_condition)
        if re.search(r"\b(what year|which year|what date|which date|what month|which month|when)\b", query_text):
            return "date"
        if re.search(r"\b(how many|how much|number of|what number)\b", query_text):
            return "number"
        if re.search(r"\b(where|what city|which city|what country|which country|what state|which state|what county|which county)\b", query_text):
            return "location"
        if query_text.startswith(("who ", "whom ", "whose ")):
            return "person"
        if answer_type in {"date", "number", "location", "person"}:
            return answer_type
        return answer_type or "entity"

    def terminal_compatible(
        self,
        query: str,
        query_condition: Optional[Dict[str, Any]],
        contexts: List[Dict[str, Any]],
    ) -> bool:
        answer_form = self.infer_answer_form(query, query_condition)
        terminals = [str(context.get("end_entity", "") or "").strip() for context in contexts]
        terminals = [item for item in terminals if item]
        if not terminals:
            return False

        if answer_form == "date":
            return any(self._is_date_like(item) for item in terminals)
        if answer_form == "number":
            return any(self._is_numeric_like(item) for item in terminals)
        if answer_form == "location":
            for item in terminals:
                text = text_processing(item)
                if self._is_date_like(text) or re.fullmatch(r"\d+(?:\.\d+)?", text):
                    continue
                return True
            return False
        if answer_form == "person":
            for item in terminals:
                text = text_processing(item)
                if self._is_date_like(text) or re.fullmatch(r"\d+(?:\.\d+)?", text):
                    continue
                return True
            return False
        return True

    def verify(
        self,
        query: str,
        query_condition: Optional[Dict[str, Any]],
        query_type: str,
        original_rank: int,
        confidence: float,
        contexts: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        reasons: List[str] = []
        penalties: List[str] = []

        if confidence < self.min_confidence:
            penalties.append("confidence_below_verifier_threshold")
        if self.max_original_rank > 0 and original_rank > self.max_original_rank:
            penalties.append("original_rank_after_verifier_window")
        if query_type in self.BLOCKED_QUERY_TYPES:
            penalties.append(f"blocked_query_type_{query_type}")
        if not contexts:
            penalties.append("no_contexts")

        if any(str(context.get("plan_id", "")).startswith("simple_fallback::") for context in contexts):
            penalties.append("simple_fallback_not_promotable")

        max_path_len = max((self._path_len(context) for context in contexts), default=0)
        if max_path_len < 2:
            penalties.append("single_hop_path_not_promotable")
        else:
            reasons.append("multi_hop_context")

        modes = {
            str(context.get("signature_match_mode", "") or "").strip().lower()
            for context in contexts
        }
        modes.discard("")
        if "strict" not in modes:
            penalties.append("no_strict_context")
        else:
            reasons.append("strict_context")

        anchors = {
            text_processing(str(context.get("anchor_entity", "") or ""))
            for context in contexts
        }
        anchors.discard("")
        if anchors and anchors <= self.GENERIC_ANCHORS:
            penalties.append("generic_anchor_only")
        elif anchors:
            reasons.append("concrete_anchor")

        if self.terminal_compatible(query, query_condition, contexts):
            reasons.append("terminal_answer_form_compatible")
        else:
            penalties.append("terminal_answer_form_incompatible")

        verified = not penalties
        return {
            "verified": bool(verified),
            "min_confidence": self.min_confidence,
            "max_original_rank": self.max_original_rank,
            "reasons": reasons[:8],
            "penalties": penalties[:8],
        }
