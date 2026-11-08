import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .query_ontology_prior import tokenize
from .utils.misc_utils import text_processing


@dataclass(frozen=True)
class QueryPlanAdapterDecision:
    query_type: str
    skip_ppr: bool = False
    ppr_skip_reason: str = ""
    disable_qa_hint: bool = False
    qa_hint_skip_reason: str = ""


class QueryPlanOntologyAdapter:
    """Central policy for using query-plan ontology evidence.

    The ontology can help when it behaves like a structural retrieval plan, but
    it hurts when it becomes an answer-form prior or a broad fan-out seed. This
    adapter keeps those decisions in one place so CompassRAG's retrieval/QA path
    can stay mostly unchanged.
    """

    NON_EXTRACTIVE_ANSWER_TYPES = {
        "description",
        "explanation",
        "reason",
        "long_answer",
    }

    EXTRACTIVE_ANSWER_TYPES = {
        "person",
        "location",
        "date",
        "number",
        "organization",
        "entity",
    }

    CATEGORY_TERMS = {
        " occupation ",
        " profession ",
        " job ",
        " genre ",
        " type ",
        " kind ",
        " category ",
        " class ",
        " classification ",
        " nationality ",
        " language ",
        " political party ",
        " position ",
        " positions ",
        " role ",
        " roles ",
    }

    DIRECT_SLOT_TERMS = {
        " occupation ",
        " profession ",
        " job ",
        " nationality ",
        " citizenship ",
        " city ",
        " birthplace ",
        " born ",
        " died ",
        " death ",
        " headquarters ",
        " educated ",
        " education ",
        " spouse ",
        " wife ",
        " husband ",
        " genre ",
        " type ",
        " language ",
        " political party ",
        " position ",
        " role ",
    }

    DIRECT_SIGNATURES = {
        "occupation",
        "nationality",
        "birth_place",
        "birth_date",
        "death_place",
        "death_date",
        "headquarters_location",
        "education",
        "family_spouse",
    }

    COMPARATIVE_TERMS = {
        " first ",
        " earlier ",
        " later ",
        " older ",
        " younger ",
        " lowest ",
        " highest ",
        " most ",
        " least ",
        " longer ",
        " shorter ",
        " larger ",
        " smaller ",
        " before ",
        " after ",
        " same ",
        " both ",
        " either ",
        " oldest ",
        " youngest ",
        " newest ",
        " latest ",
        " earliest ",
        " longest ",
        " shortest ",
        " largest ",
        " smallest ",
        " greatest ",
        " fewest ",
        " best ",
        " born first ",
        " born later ",
        " born earlier ",
        " released first ",
        " released earlier ",
        " was released first ",
        " was released earlier ",
        " which film ",
        " which person ",
        " which album ",
        " which came first ",
        " died later ",
        " died earlier ",
        " born last ",
        " released later ",
        " has the director who ",
        " has the composer who ",
        " has the author who ",
    }

    BRIDGE_PATTERNS = [
        r"\bperson who\b",
        r"\bgroup who\b",
        r"\bactor who\b",
        r"\bactress who\b",
        r"\bartist who\b",
        r"\bperformer who\b",
        r"\bplayer who\b",
        r"\bgrandmother of\b",
        r"\bgrandfather of\b",
        r"\bpaternal grandmother\b",
        r"\bpaternal grandfather\b",
        r"\bmaternal grandmother\b",
        r"\bmaternal grandfather\b",
        r"\bplace of burial\b",
        r"\bplace of death\b",
        r"\bplace of birth\b",
        r"\bcity containing\b",
        r"\bcounty containing\b",
        r"\bcountry containing\b",
        r"\bparticipant of\b",
        r"\bmember of\b.*\bwhere\b",
        r"\bof the\s+(?:father|mother|spouse|wife|husband|grandfather|grandmother|director|composer|author|performer|player|participant)\s+of\b",
        r"\bs father\b",
        r"\bs mother\b",
        r"\bs wife\b",
        r"\bs husband\b",
        r"\bs spouse\b",
        r"\bs grandmother\b",
        r"\bs grandfather\b",
        r"\bs author\b",
        r"\bs director\b",
        r"\bs composer\b",
        r"\bplayed\b",
        r"\bperformed\b",
        r"\bcontaining\b",
        r"\bcompared to\b",
    ]

    DISCOURSE_PATTERNS = [
        r"\bwhy\b",
        r"\bhow does\b",
        r"\bhow do\b",
        r"\bhow did\b",
        r"\bwhat caused\b",
        r"\bwhat causes\b",
        r"\bwhat happens\b",
        r"\bwhat happened\b",
        r"\bwhat does\b",
        r"\bwhat did\b",
        r"\bwhen does\b",
        r"\bwhen did\b.*\brealize\b",
        r"\breason\b",
        r"\bin the story\b",
    ]

    HARMFUL_PPR_PATTERNS = [
        r"\bwhy\b",
        r"\bbecause\b",
        r"\bhow far\b",
        r"\bhow long\b",
        r"\bhow did\b",
        r"\bhow does\b",
        r"\bhow do\b",
        r"\bimportant (?:work )?because\b",
        r"\breason\b",
        r"\bexplain\b",
    ]

    NESTED_PATTERNS = [
        r"\bwhere\b.*\bwhere\b",
        r"\bwho\b.*\bwho\b",
        r"\bwhat\b.*\bwhere\b",
        r"\bwhere\b.*\bwhat\b",
        r"\bin the (?:city|county|state|country|region|place) where\b",
        r"\bfrom the (?:city|county|state|country|region|place) where\b",
        r"\bsource of (?:the )?body of water\b",
        r"\bbody of water where\b",
        r"\b(?:where|when|what|which|who)\b.*\b(?:performer|composer|author|director|producer|player|participant|member) of\b",
        r"\b(?:where|when|what|which|who)\b.*\b(?:father|mother|spouse|wife|husband|grandfather|grandmother) of\b",
        r"\b(?:performer|composer|author|director|producer|player|participant|member) of\b.*\b(?:where|when|what|which|who)\b",
        r"\b(?:father|mother|spouse|wife|husband|grandfather|grandmother) of\b.*\b(?:where|when|what|which|who)\b",
        r"\b(?:country|county|city|region|state) containing\b",
        r"\b(?:located|born|died|headquartered) in\b.*\b(?:where|which|what)\b",
        r"\bfinal resting place\b",
        r"\bimmediately north of\b",
        r"\bcompared to\b",
    ]

    COMPOSITION_TERMS = {
        "where",
        "which",
        "whose",
        "that",
        "between",
        "despite",
        "compared",
        "containing",
        "headquartered",
        "originated",
        "signed",
        "located",
        "born",
        "died",
        "member",
        "participant",
        "performer",
        "composer",
        "author",
        "director",
        "producer",
        "father",
        "mother",
        "spouse",
        "wife",
        "husband",
    }

    def _query_text(self, query: str) -> str:
        return f" {text_processing(query)} "

    def answer_type(self, query_condition: Optional[Dict[str, Any]]) -> str:
        query_condition = query_condition or {}
        return text_processing(str(query_condition.get("answer_type", "") or ""))

    def is_discourse_or_long_answer_query(self, query: str) -> bool:
        query_text = self._query_text(query)
        return any(re.search(pattern, query_text) for pattern in self.DISCOURSE_PATTERNS)

    def is_category_or_alias_query(self, query: str) -> bool:
        query_text = self._query_text(query)
        return any(term in query_text for term in self.CATEGORY_TERMS)

    def is_comparative_query(self, query: str) -> bool:
        query_text = self._query_text(query)
        return any(term in query_text for term in self.COMPARATIVE_TERMS)

    def is_bridge_pattern_query(
        self,
        query: str,
        query_condition: Optional[Dict[str, Any]] = None,
    ) -> bool:
        query_text = self._query_text(query)
        query_condition = query_condition or {}
        relation_signatures = {
            text_processing(str(item))
            for item in (query_condition.get("relation_signatures", []) or [])
        }
        relation_families = {
            text_processing(str(item))
            for item in (query_condition.get("relation_families", []) or [])
        }
        relation_terms = {
            text_processing(str(item))
            for item in (query_condition.get("relation_terms", []) or [])
        }
        direct_slot = self.is_direct_slot_query(query, query_condition)
        if not direct_slot and relation_families & {"family"}:
            return True
        bridge_relation_tokens = {
            "father",
            "mother",
            "spouse",
            "wife",
            "husband",
            "grandfather",
            "grandmother",
            "parent",
            "child",
            "son",
            "daughter",
            "sibling",
            "performer",
            "participant",
        }
        if not direct_slot and any(
            any(token in value for token in bridge_relation_tokens)
            for value in relation_signatures | relation_terms
        ):
            return True
        return any(re.search(pattern, query_text) for pattern in self.BRIDGE_PATTERNS)

    def is_direct_slot_query(
        self,
        query: str,
        query_condition: Optional[Dict[str, Any]] = None,
    ) -> bool:
        query_text = self._query_text(query)
        query_condition = query_condition or {}
        relation_signatures = set(query_condition.get("relation_signatures", []) or [])
        if any(term in query_text for term in self.DIRECT_SLOT_TERMS):
            return True
        return bool(relation_signatures & self.DIRECT_SIGNATURES)

    def is_direct_relation_lookup(self, query: str) -> bool:
        query_text = self._query_text(query)
        direct_patterns = [
            r"\bwho\s+(?:is|was|are|were)?\s*(?:the\s+)?(?:father|mother|spouse|wife|husband|grandfather|grandmother)\s+of\b",
            r"\bwho\s+(?:is|was|are|were)?\s*(?:the\s+)?(?:author|director|composer|performer|player)\s+of\b",
            r"\bwho\s+(?:directed|wrote|authored|composed|performed|played|won)\b",
            r"\bwhat\s+(?:is|was|are|were)?\s*(?:the\s+)?(?:occupation|profession|job|nationality|genre|type|language)\s+of\b",
            r"\bwhere\s+(?:is|was|are|were)?\s*(?:the\s+)?(?:birthplace|birth place|headquarters)\s+of\b",
            r"\bwhen\s+(?:was|were|did)\b",
        ]
        return any(re.search(pattern, query_text) for pattern in direct_patterns)

    def is_nested_or_complex_factoid_query(self, query: str) -> bool:
        query_text = self._query_text(query)
        return any(re.search(pattern, query_text) for pattern in self.NESTED_PATTERNS)

    def is_long_compositional_query(self, query: str) -> bool:
        query_text = text_processing(query)
        query_tokens = tokenize(query_text)
        if len(query_tokens) < 18:
            return False
        composition_count = sum(1 for token in query_tokens if token in self.COMPOSITION_TERMS)
        if len(query_tokens) >= 28:
            return True
        return composition_count >= 2

    def is_extractable_factoid_query(
        self,
        query: str,
        query_condition: Optional[Dict[str, Any]] = None,
    ) -> bool:
        query_text = text_processing(query)
        answer_type = self.answer_type(query_condition)
        if query_text.startswith((
            "who ",
            "where ",
            "when ",
            "in what year ",
            "in which year ",
            "what year ",
            "which year ",
            "what date ",
            "which date ",
            "what city ",
            "which city ",
            "what country ",
            "which country ",
            "how many ",
            "how much ",
        )):
            return True
        return answer_type in self.EXTRACTIVE_ANSWER_TYPES

    def classify_query(
        self,
        query: str,
        query_condition: Optional[Dict[str, Any]] = None,
    ) -> str:
        answer_type = self.answer_type(query_condition)
        if answer_type in self.NON_EXTRACTIVE_ANSWER_TYPES:
            return "non_extractive_answer"
        if self.is_discourse_or_long_answer_query(query):
            return "discourse_or_long_answer"
        if any(re.search(pattern, self._query_text(query)) for pattern in self.HARMFUL_PPR_PATTERNS):
            return "explanatory_or_measure"
        if self.is_category_or_alias_query(query):
            return "category_or_alias"
        if self.is_comparative_query(query):
            return "comparative"
        direct_relation_lookup = self.is_direct_relation_lookup(query)
        if self.is_nested_or_complex_factoid_query(query) and not direct_relation_lookup:
            return "nested_bridge_factoid"
        if self.is_long_compositional_query(query) and not direct_relation_lookup:
            return "long_compositional_factoid"
        if self.is_direct_slot_query(query, query_condition):
            return "direct_slot_factoid"
        if self.is_extractable_factoid_query(query, query_condition):
            return "extractive_factoid"
        return "other"

    def should_skip_ppr(
        self,
        query: str,
        query_condition: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, str]:
        query_type = self.classify_query(query, query_condition)
        if query_type in {
            "non_extractive_answer",
            "discourse_or_long_answer",
            "explanatory_or_measure",
        }:
            return True, query_type



        if query_type == "category_or_alias":
            return True, query_type
        return False, ""

    def should_disable_qa_format_hint(
        self,
        query: str,
        query_condition: Optional[Dict[str, Any]] = None,
        has_graph_evidence: bool = False,
        evidence_style: Optional[str] = None,
    ) -> Tuple[bool, str]:
        query_type = self.classify_query(query, query_condition)
        if query_type in {
            "category_or_alias",
            "non_extractive_answer",
            "discourse_or_long_answer",
            "explanatory_or_measure",
            "comparative",
            "nested_bridge_factoid",
            "long_compositional_factoid",
        }:
            return True, query_type
        if self.is_bridge_pattern_query(query, query_condition):
            return True, "bridge_pattern"
        if has_graph_evidence and evidence_style and evidence_style != "disabled":


            return True, "graph_evidence_only"
        return False, ""

    def resolve_qa_evidence_style(
        self,
        query: str,
        query_condition: Optional[Dict[str, Any]] = None,
        is_simple: bool = False,
        is_direct_slot: bool = False,
        simple_applied: bool = False,
        executable_path_count: int = 0,
        lifted_passage_count: int = 0,
    ) -> str:
        query_type = self.classify_query(query, query_condition)
        answer_type = self.answer_type(query_condition)
        query_text = text_processing(query)

        if query_type in {
            "comparative",
            "non_extractive_answer",
            "discourse_or_long_answer",
            "explanatory_or_measure",
            "long_compositional_factoid",
        }:
            return "disabled"

        if simple_applied or is_direct_slot or query_type == "category_or_alias":
            return "facts_only"

        if not is_simple and not is_direct_slot:
            if executable_path_count >= 2 and lifted_passage_count > 0:
                if answer_type == "person" or query_text.startswith("who "):
                    return "bridge_cues"
            return "disabled"

        if executable_path_count >= 2 and lifted_passage_count > 0:
            return "structured_chain"
        return "disabled"

    def decide(
        self,
        query: str,
        query_condition: Optional[Dict[str, Any]] = None,
        has_graph_evidence: bool = False,
        evidence_style: Optional[str] = None,
    ) -> QueryPlanAdapterDecision:
        query_type = self.classify_query(query, query_condition)
        skip_ppr, ppr_reason = self.should_skip_ppr(query, query_condition)
        disable_hint, hint_reason = self.should_disable_qa_format_hint(
            query=query,
            query_condition=query_condition,
            has_graph_evidence=has_graph_evidence,
            evidence_style=evidence_style,
        )
        return QueryPlanAdapterDecision(
            query_type=query_type,
            skip_ppr=skip_ppr,
            ppr_skip_reason=ppr_reason,
            disable_qa_hint=disable_hint,
            qa_hint_skip_reason=hint_reason,
        )
