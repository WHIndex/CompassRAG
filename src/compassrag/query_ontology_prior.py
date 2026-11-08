import ast
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, List, Tuple

import numpy as np

try:
    from .utils.logging_utils import get_logger
    from .utils.misc_utils import min_max_normalize, text_processing
except ImportError:
    import logging

    def get_logger(name):
        return logging.getLogger(name)

    def text_processing(text):
        return re.sub(r"\s+", " ", str(text).lower()).strip()

    def min_max_normalize(values):
        arr = np.asarray(values, dtype=np.float32)
        if arr.size == 0:
            return arr
        min_value = float(np.min(arr))
        max_value = float(np.max(arr))
        if max_value <= min_value:
            return np.zeros_like(arr)
        return (arr - min_value) / (max_value - min_value)


logger = get_logger(__name__)

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "had",
    "in", "is", "it", "its", "of", "on", "or", "that", "the", "to", "was", "were",
    "when", "where", "which", "who", "whom", "whose", "with", "what", "did", "does",
    "do", "both", "same", "many", "how", "her", "his", "him", "their", "them",
}

INTENT_TERMS = {
    "birth": {"born", "birth", "date"},
    "family": {"father", "mother", "son", "daughter", "spouse", "married", "wife", "husband"},
    "creator": {"author", "writer", "wrote", "director", "directed", "creator", "composer"},
    "location": {"city", "country", "located", "place", "capital", "where"},
    "work": {"film", "movie", "book", "album", "song", "novel", "series"},
    "organization": {"team", "company", "school", "university", "club", "party"},
    "event": {"award", "season", "championship", "competition", "tournament"},
    "time": {"year", "date", "when", "before", "after"},
    "occupation": {"occupation", "profession", "job"},
    "biomedical_treatment": {"treat", "treatment", "therapy", "prevent", "prophylaxis"},
    "biomedical_diagnosis": {"diagnose", "diagnosis", "detect", "screen", "biomarker"},
    "biomedical_mechanism": {"mechanism", "pathway", "regulate", "activate", "inhibit", "express"},
    "biomedical_association": {"associate", "association", "correlate", "risk", "cause"},
    "biomedical_genetics": {"gene", "genetic", "mutation", "variant", "allele", "protein"},
    "biomedical_definition": {"define", "definition", "describe", "abbreviation"},
    "biomedical_composition": {"compose", "composition", "contain", "include", "component"},
    "biomedical_localization": {"locate", "localization", "secrete"},
    "biomedical_inheritance": {"inherit", "autosomal", "dominant", "recessive"},
}

HIGH_SIGNAL_RELATION_TERMS = {
    "acted", "active", "adapted", "affiliated", "alma", "attended",
    "authored", "became", "belongs", "bordered",
    "born", "capital", "composed", "created", "died", "directed", "employed",
    "educated", "education", "expelled", "father", "filmed", "founded",
    "graduated", "headquarters",
    "headquartered", "invented", "located", "married", "member", "mother",
    "nationality", "occupation", "owned", "parent", "performed", "place",
    "profession", "citizenship", "county", "performed", "played", "printed",
    "produced", "published", "record", "region", "released", "served",
    "signed", "soloist", "son", "starred", "starring", "studied",
    "university", "wife", "won", "worked", "wrote", "written",
    "activate", "activation", "adverse", "allele", "associate", "associated",
    "association", "biomarker", "bind", "binding", "cause", "causes",
    "causation", "classify", "classification", "correlate", "correlation",
    "diagnose", "diagnosis", "dosage", "drug", "enzyme", "express",
    "expression", "gene", "genetic", "inhibit", "inhibition", "interact",
    "interaction", "mechanism", "mutation", "pathway", "prevent",
    "prevention", "protein", "receptor", "regulate", "regulation", "risk",
    "screen", "screening", "symptom", "therapy", "toxicity", "treat",
    "treatment", "variant",
    "abbreviation", "autosomal", "catalyze", "component", "compose",
    "composition", "contain", "definition", "describe", "dominant", "effective",
    "efficacy", "inherit", "inheritance", "localization", "recessive", "secrete",
}

LOW_QUALITY_RELATION_PHRASES = {
    "appears", "appears in", "asks", "associated with", "can be", "contains",
    "describes", "does", "features", "full name", "had", "has", "have",
    "include", "included", "includes", "interacts with", "is", "is a",
    "is an", "is associated with", "is described as", "is in",
    "is mentioned by", "is speaking to", "is sitting with", "is the date of",
    "label", "lost", "made", "mentions", "mentioned", "saw", "says",
    "speaks to", "took", "was", "was a", "was an", "were",
}

LOW_INFORMATION_RELATION_PREFIXES = (
    "is described",
    "is mentioned",
    "is speaking",
    "is sitting",
)

TYPE_HINT_TERMS = {
    "actor", "answer", "artist", "author", "birth", "book", "capital", "character",
    "city", "company", "country", "creator", "date", "director", "event", "family",
    "film", "location", "locations", "movie", "organization", "organizations",
    "parent", "people", "person", "persons", "place", "places", "player",
    "relation", "school", "song", "spouse", "team", "time", "title", "type",
    "university", "work", "works", "writer", "year",
    "biomarker", "biomedical", "boolean", "disease", "dosage", "drug",
    "enzyme", "gene", "mutation", "protein", "quantity", "receptor",
    "syndrome", "therapy", "treatment", "variant",
}

RELATION_FAMILY_RULES = (
    ("biomedical_adverse_effect", {"adverse", "toxicity", "contraindication"}),
    ("biomedical_dosage", {"dose", "dosage", "administered", "administration"}),
    ("biomedical_diagnosis", {"diagnosis", "diagnose", "detect", "screen", "biomarker"}),
    ("biomedical_treatment", {"treatment", "treat", "therapy", "prevent", "prophylaxis"}),
    ("biomedical_causation", {"causation", "cause", "induce", "etiology"}),
    ("biomedical_genetics", {"genetic", "gene", "mutation", "variant", "allele"}),
    ("biomedical_regulation", {"regulation", "regulate", "activate", "inhibit", "expression", "express"}),
    ("biomedical_interaction", {"interaction", "interact", "bind", "receptor", "target"}),
    ("biomedical_association", {"association", "associate", "correlate", "risk"}),
    ("biomedical_manifestation", {"manifestation", "manifest", "symptom", "phenotype"}),
    ("biomedical_epidemiology", {"epidemiology", "incidence", "prevalence", "mortality", "survival"}),
    ("biomedical_mechanism", {"mechanism", "pathway", "mediate", "function"}),
    ("biomedical_classification", {"classification", "classify", "subtype"}),
    ("biomedical_definition", {"definition", "define", "describe", "abbreviation"}),
    ("biomedical_composition", {"composition", "compose", "contain", "include", "component"}),
    ("biomedical_localization", {"localization", "localize", "located", "secrete"}),
    ("biomedical_inheritance", {"inheritance", "inherit", "autosomal", "dominant", "recessive"}),
    ("biomedical_comparison", {"comparison", "compare", "difference"}),
    ("biomedical_attribute", {"biomedical"}),
    ("birth", {"birth", "born", "birthplace"}),
    ("death", {"death", "died", "dies"}),
    ("nationality", {"nationality", "citizenship", "citizen"}),
    ("occupation", {"occupation", "profession", "job"}),
    ("location", {"located", "location", "headquartered", "headquarters", "capital", "bordered", "border", "situated", "county", "province", "region", "state", "country", "city", "part"}),
    ("creator", {"author", "authored", "writer", "written", "wrote", "director", "directed", "composer", "composed", "created", "creator"}),
    ("performer", {"actor", "acted", "starring", "starred", "played", "performed", "soloist"}),
    ("production", {"produced", "producer", "published", "publication", "printed", "released", "release"}),
    ("family", {"father", "mother", "parent", "son", "daughter", "wife", "husband", "spouse", "married"}),
    ("membership", {"member", "played", "served", "worked", "employed", "graduated", "studied", "educated", "education", "attended", "alma", "school", "university", "signed", "label", "affiliated"}),
    ("ownership", {"owned", "owner", "founded", "founder", "belongs"}),
    ("award", {"won", "award", "championship", "competition", "tournament"}),
    ("adaptation", {"adapted", "based"}),
    ("alias", {"known", "called", "named"}),
    ("expulsion", {"expelled"}),
)

QUERY_INTENT_TO_RELATION_FAMILIES = {
    "birth": {"birth", "location"},
    "family": {"family"},
    "creator": {"creator", "performer", "production"},
    "location": {"location", "birth", "death", "nationality"},
    "work": {"creator", "performer", "production", "adaptation"},
    "organization": {"membership", "ownership", "location"},
    "event": {"award"},
    "time": {"birth", "death", "production"},
    "occupation": {"occupation"},
    "biomedical_treatment": {"biomedical_treatment", "biomedical_adverse_effect", "biomedical_dosage"},
    "biomedical_diagnosis": {"biomedical_diagnosis", "biomedical_manifestation"},
    "biomedical_mechanism": {"biomedical_mechanism", "biomedical_regulation", "biomedical_interaction"},
    "biomedical_association": {"biomedical_association", "biomedical_causation", "biomedical_epidemiology"},
    "biomedical_genetics": {"biomedical_genetics", "biomedical_regulation", "biomedical_association"},
    "biomedical_definition": {"biomedical_definition", "biomedical_classification"},
    "biomedical_composition": {"biomedical_composition"},
    "biomedical_localization": {"biomedical_localization"},
    "biomedical_inheritance": {"biomedical_inheritance", "biomedical_genetics"},
}

RELATION_AUXILIARY_TOKENS = {
    "also", "best", "big", "can", "charter", "formerly", "known", "originally",
}

SCHEMA_TRIGGER_TERMS = (
    set().union(*INTENT_TERMS.values())
    | HIGH_SIGNAL_RELATION_TERMS
    | TYPE_HINT_TERMS
    | {
        "answer", "award", "birthplace", "bridge", "competition", "entity",
        "event", "family", "genre", "nationality", "occupation", "publication",
        "relationship", "release", "schema", "workplace",
    }
)


def normalize_text(value: Any) -> str:
    return text_processing(value).strip()


def tokenize(value: Any) -> List[str]:
    return [
        token
        for token in normalize_text(value).split()
        if len(token) > 2 and token not in STOPWORDS
    ]


def extract_json_object(text: Any) -> Dict[str, Any]:
    if isinstance(text, dict):
        return text
    if not isinstance(text, str):
        return {}

    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()

    for candidate in (cleaned, cleaned.replace("\n", " ")):
        try:
            parsed = json.loads(candidate)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            pass
        try:
            parsed = ast.literal_eval(candidate)
            return parsed if isinstance(parsed, dict) else {}
        except (SyntaxError, ValueError):
            pass

    decoder = json.JSONDecoder()
    for start in [match.start() for match in re.finditer(r"\{", cleaned)]:
        try:
            parsed, _ = decoder.raw_decode(cleaned[start:])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    return {}


def _clean_string_list(values: Any, max_items: int = 20) -> List[str]:
    if values is None:
        return []
    if not isinstance(values, list):
        values = [values]
    cleaned = []
    seen = set()
    for value in values:
        text = normalize_text(value)
        if not text or text in seen:
            continue
        cleaned.append(text)
        seen.add(text)
        if len(cleaned) >= max_items:
            break
    return cleaned


def _relation_aliases(relation: str) -> List[str]:
    relation = normalize_text(relation)
    aliases = [relation]
    aliases.extend(token for token in tokenize(relation) if token in HIGH_SIGNAL_RELATION_TERMS)
    return _clean_string_list(aliases, max_items=8)


def _is_low_quality_relation(relation: Any) -> bool:
    relation = normalize_text(relation)
    if not relation:
        return True
    if relation in LOW_QUALITY_RELATION_PHRASES:
        return True
    if any(relation.startswith(prefix) for prefix in LOW_INFORMATION_RELATION_PREFIXES):
        return True
    relation_tokens = set(tokenize(relation))
    if not relation_tokens:
        return True
    if len(relation_tokens) == 1 and not (relation_tokens & HIGH_SIGNAL_RELATION_TERMS):
        return True
    return False


def _clean_relation_aliases(relation: Any, aliases: Any = None, max_items: int = 12) -> List[str]:
    candidates = []
    if aliases:
        candidates.extend(aliases if isinstance(aliases, list) else [aliases])
    candidates.extend(_relation_aliases(relation))

    cleaned = []
    seen = set()
    for alias in candidates:
        alias = normalize_text(alias)
        if not alias or alias in seen:
            continue
        if _is_low_quality_relation(alias) and not (set(tokenize(alias)) & HIGH_SIGNAL_RELATION_TERMS):
            continue
        cleaned.append(alias)
        seen.add(alias)
        if len(cleaned) >= max_items:
            break
    return cleaned


def _stem_relation_token(token: str) -> str:
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    for suffix in ("ing", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) > len(suffix) + 3:
            return token[:-len(suffix)]
    return token


def _relation_family(relation: Any) -> str:
    tokens = [
        _stem_relation_token(token)
        for token in tokenize(relation)
        if token not in RELATION_AUXILIARY_TOKENS
    ]
    token_set = set(tokens)
    if not token_set:
        return ""

    for family, family_terms in RELATION_FAMILY_RULES:
        family_stems = {_stem_relation_token(term) for term in family_terms}
        if token_set & family_stems:
            return family

    return " ".join(tokens[:3])


def _relation_families(relations: Iterable[Any]) -> set[str]:
    families = set()
    for relation in relations:
        family = _relation_family(relation)
        if family:
            families.add(family)
    return families


def _relation_family_trigger_tokens(template: Dict[str, Any]) -> set[str]:
    tokens = set()
    for family in template.get("relation_families", []):
        tokens.update(tokenize(family))
    for relation in template.get("required_relations", []):
        family = _relation_family(relation)
        if family:
            tokens.update(tokenize(family))
        tokens.update(tokenize(relation))
    return tokens


def _query_intent_families(query: Any) -> set[str]:
    query_tokens = set(tokenize(query))
    families = set()
    for intent, terms in INTENT_TERMS.items():
        if query_tokens & terms:
            families.update(QUERY_INTENT_TO_RELATION_FAMILIES.get(intent, set()))



    if query_tokens & {"nationality", "citizenship"}:
        families.add("nationality")
    if query_tokens & {"director", "author", "writer", "composer"}:
        families.add("creator")
    if query_tokens & {"actor", "performer", "singer"}:
        families.add("performer")
    return families


def _relation_family_aligned_with_query(relation: Any, query: Any) -> bool:
    relation_family = _relation_family(relation)
    if not relation_family:
        return False
    query_families = _query_intent_families(query)
    if relation_family in query_families:
        return True
    relation_tokens = set(tokenize(relation))
    query_tokens = set(tokenize(query))
    return bool(query_tokens & relation_tokens & HIGH_SIGNAL_RELATION_TERMS)


def _schema_triggers_for_relation(relation: Any, source_queries: Iterable[str], max_items: int = 16) -> List[str]:
    relation_family = _relation_family(relation)
    relation_tokens = set(tokenize(relation))
    trigger_terms = set(tokenize(relation_family)) | (relation_tokens & SCHEMA_TRIGGER_TERMS)
    for query in source_queries:
        if not _relation_family_aligned_with_query(relation, query):
            continue
        query_tokens = set(tokenize(query))
        for token in query_tokens:
            if token in SCHEMA_TRIGGER_TERMS and (
                token in relation_tokens
                or token in tokenize(relation_family)
                or relation_family in _query_intent_families(query)
            ):
                trigger_terms.add(token)
    return _clean_schema_trigger_terms(sorted(trigger_terms), max_items=max_items)


def _aligned_trigger_terms_for_relation(relation: Any, triggers: Any, max_items: int = 16) -> List[str]:
    relation_family = _relation_family(relation)
    relation_tokens = set(tokenize(relation))
    family_tokens = set(tokenize(relation_family))
    raw_terms = _clean_schema_trigger_terms(
        _clean_string_list(triggers, max_items=max_items * 4) + [relation, relation_family],
        max_items=max_items * 4,
    )
    aligned = []
    seen = set()
    for token in raw_terms:
        token_families = _query_intent_families(token)
        if (
            token in relation_tokens
            or token in family_tokens
            or (relation_family and relation_family in token_families)
        ):
            if token not in seen:
                aligned.append(token)
                seen.add(token)
        if len(aligned) >= max_items:
            break
    return aligned


def _clean_trigger_terms(values: Any, max_items: int = 16) -> List[str]:
    terms = []
    seen = set()
    for value in _clean_string_list(values, max_items=max_items * 2):
        for token in tokenize(value):
            if token in seen:
                continue
            terms.append(token)
            seen.add(token)
            if len(terms) >= max_items:
                return terms
    return terms


def _clean_schema_trigger_terms(values: Any, max_items: int = 16) -> List[str]:
    """Keep reusable schema words, not historical entity/name fragments."""
    terms = []
    seen = set()
    for token in _clean_trigger_terms(values, max_items=max_items * 3):
        if token not in SCHEMA_TRIGGER_TERMS:
            continue
        if token in seen:
            continue
        terms.append(token)
        seen.add(token)
        if len(terms) >= max_items:
            break
    return terms


def _infer_type_hints(query_terms: Iterable[str], relation_counts: Counter) -> List[Dict[str, Any]]:
    terms = set(query_terms)
    relation_tokens = set()
    for relation in relation_counts:
        relation_tokens.update(tokenize(relation))

    candidates = []
    def add(term: str, aliases: List[str], support: float):
        candidates.append({
            "term": term,
            "aliases": _clean_string_list([term] + aliases, max_items=8),
            "support": support,
            "kind": "type_hint",
        })

    if terms & {"who", "father", "mother", "son", "daughter", "spouse", "wife", "husband"} or relation_tokens & {"father", "mother", "son", "wife", "married"}:
        add("person", ["parent", "family", "spouse"], 3.0)
    if terms & {"where", "city", "country", "place", "capital"} or relation_tokens & {"located", "born", "headquartered"}:
        add("location", ["place", "city", "country", "capital"], 3.0)
    if terms & {"when", "date", "year"} or relation_tokens & {"born", "died", "released", "published"}:
        add("date", ["time", "year"], 2.0)
    if terms & {"film", "movie", "book", "album", "song", "novel", "series"} or relation_tokens & {"directed", "written", "authored", "starring"}:
        add("work", ["film", "book", "song", "album"], 2.0)
    if terms & {"company", "school", "university", "team", "club"} or relation_tokens & {"founded", "headquartered", "played"}:
        add("organization", ["company", "school", "team", "university"], 2.0)
    if terms & {"award", "championship", "competition", "tournament"} or relation_tokens & {"won"}:
        add("event", ["award", "competition", "championship"], 2.0)

    seen = set()
    deduped = []
    for item in candidates:
        if item["term"] in seen:
            continue
        seen.add(item["term"])
        deduped.append(item)
    return deduped[:10]


def _is_reusable_entity_hint(term: Any) -> bool:
    tokens = set(tokenize(term))
    if not tokens:
        return False
    if tokens <= TYPE_HINT_TERMS:
        return True
    return False


def _clean_entity_terms(values: Any, max_items: int = 16) -> List[str]:
    cleaned = []
    seen = set()
    for value in _clean_string_list(values, max_items=max_items * 2):
        if not _is_reusable_entity_hint(value):
            continue
        if value in seen:
            continue
        cleaned.append(value)
        seen.add(value)
        if len(cleaned) >= max_items:
            break
    return cleaned


def _clean_supporting_facts(values: Any, max_items: int = 8) -> List[Dict[str, Any]]:
    if values is None:
        return []
    if not isinstance(values, list):
        values = [values]

    facts = []
    seen = set()
    for value in values:
        fact = value.get("fact") if isinstance(value, dict) else value
        if not isinstance(fact, (list, tuple)) or len(fact) != 3:
            continue
        triple = tuple(normalize_text(part) for part in fact)
        if len(triple) != 3 or not all(triple):
            continue
        if _is_low_quality_relation(triple[1]) or triple in seen:
            continue
        seen.add(triple)
        facts.append({"fact": list(triple)})
        if len(facts) >= max_items:
            break
    return facts


def _extract_relation_values_from_steps(steps: Any) -> List[str]:
    values = []
    if not isinstance(steps, list):
        return values
    for step in steps:
        if isinstance(step, dict):
            values.extend(_clean_string_list(step.get("relations", []) or step.get("relation_aliases", []), max_items=8))
            relation = step.get("relation")
            if relation:
                values.append(relation)
        elif isinstance(step, str):
            values.append(step)
    return values


def _clean_required_relations(template: Dict[str, Any], max_items: int = 12) -> List[str]:
    raw_values = []
    raw_values.extend(_clean_string_list(template.get("required_relations", []), max_items=max_items))
    raw_values.extend(_clean_string_list(template.get("relation_aliases", []), max_items=max_items))
    raw_values.extend(_extract_relation_values_from_steps(template.get("relation_steps", [])))

    relations = []
    seen = set()
    for value in raw_values:
        for alias in _clean_relation_aliases(value, max_items=4):
            if alias in seen:
                continue
            relations.append(alias)
            seen.add(alias)
            if len(relations) >= max_items:
                return relations
    return relations


def _clean_relation_steps(template: Dict[str, Any], required_relations: List[str]) -> List[Dict[str, Any]]:
    cleaned_steps = []
    raw_steps = template.get("relation_steps", [])
    if isinstance(raw_steps, list):
        for idx, step in enumerate(raw_steps):
            if not isinstance(step, dict):
                step = {"relations": [step]}
            relations = []
            for value in _clean_string_list(step.get("relations", []) or step.get("relation_aliases", []) or step.get("relation"), max_items=8):
                relations.extend(_clean_relation_aliases(value, max_items=4))
            relations = _clean_string_list(relations, max_items=8)
            if not relations:
                continue
            cleaned_steps.append({
                "role": normalize_text(step.get("role")) or ("anchor" if idx == 0 else "bridge"),
                "relations": relations,
            })
    if cleaned_steps:
        return cleaned_steps[:4]
    return [{"role": "schema", "relations": required_relations[:8]}] if required_relations else []


def _clean_bridge_template(template: Any, default_name: str = "bridge_template") -> Dict[str, Any] | None:
    if not isinstance(template, dict):
        return None

    required_relations = _clean_required_relations(template, max_items=12)
    triggers = _clean_schema_trigger_terms(
        template.get("triggers", [])
        or template.get("trigger_terms", [])
        or template.get("query_triggers", [])
        or template.get("query_pattern", ""),
        max_items=16,
    )
    answer_type_candidates = _clean_entity_terms(
        [
            template.get("answer_type", ""),
            template.get("target_type", ""),
            template.get("object_type", ""),
        ],
        max_items=4,
    )
    entity_terms = _clean_entity_terms(
        answer_type_candidates
        + _clean_string_list(template.get("entity_terms", []) or template.get("answer_types", []), max_items=16),
        max_items=16,
    )
    supporting_facts = _clean_supporting_facts(template.get("supporting_facts", []), max_items=8)

    if not triggers:
        triggers = _clean_schema_trigger_terms(required_relations + answer_type_candidates + entity_terms, max_items=16)

    distinct_relations = {normalize_text(relation) for relation in required_relations}
    distinct_relation_families = _relation_families(required_relations)
    support_relations = {
        normalize_text(item["fact"][1])
        for item in supporting_facts
        if isinstance(item, dict) and isinstance(item.get("fact"), list) and len(item["fact"]) == 3
    }
    support_relation_families = _relation_families(support_relations)

    if len(distinct_relations) < 2 or len(support_relations) < 2:
        return None
    if len(distinct_relation_families) < 2 or len(support_relation_families) < 2:
        return None
    if not required_relations or not triggers or not entity_terms or not supporting_facts:
        return None

    query_pattern = str(template.get("query_pattern", "") or template.get("pattern", "") or " ".join(triggers[:8]))
    confidence = min(max(float(template.get("confidence", 0.5) or 0.5), 0.0), 1.0)
    support = float(template.get("support", len(supporting_facts) or 1.0) or 1.0)

    source_type = normalize_text(template.get("source_type")) if template.get("source_type") else ""
    intermediate_type = normalize_text(template.get("intermediate_type")) if template.get("intermediate_type") else ""
    target_type = normalize_text(template.get("target_type")) if template.get("target_type") else ""

    return {
        "name": normalize_text(template.get("name")) or default_name,
        "query_pattern": query_pattern[:240],
        "answer_type": answer_type_candidates[0] if answer_type_candidates else entity_terms[0],
        "source_type": source_type[:80],
        "intermediate_type": intermediate_type[:80],
        "target_type": target_type[:80],
        "required_relations": required_relations,
        "relation_aliases": required_relations,
        "relation_families": sorted(distinct_relation_families),
        "relation_steps": _clean_relation_steps(template, required_relations),
        "triggers": triggers,
        "entity_terms": entity_terms,
        "supporting_facts": supporting_facts,
        "support": support,
        "candidate_support": float(template.get("candidate_support", 0.0) or 0.0),
        "path_precision": min(max(float(template.get("path_precision", 0.0) or 0.0), 0.0), 1.0),
        "confidence": confidence,
        "source": normalize_text(template.get("source")) if template.get("source") else "",
    }


def _relations_containing(relation_counts: Counter, terms: set[str], max_items: int = 4) -> List[str]:
    matches = []
    for relation, _ in relation_counts.most_common():
        relation_tokens = set(tokenize(relation))
        if relation_tokens & terms and not _is_low_quality_relation(relation):
            matches.append(relation)
            if len(matches) >= max_items:
                break
    return matches


def _query_signature(query: str) -> str:
    query_tokens = set(tokenize(query))
    matched_groups = [
        group for group, terms in INTENT_TERMS.items()
        if query_tokens & terms
    ]
    if matched_groups:
        return "+".join(sorted(matched_groups[:2]))
    fallback_tokens = sorted(query_tokens)[:2]
    return "+".join(fallback_tokens) if fallback_tokens else "misc"


class QueryOntologyPriorBuilder:
    """Builds reusable query ontology priors from historical queries and retrieved graph evidence."""

    def __init__(
        self,
        max_clusters: int = 12,
        max_queries_per_cluster: int = 30,
        max_facts_per_cluster: int = 80,
        use_llm_agents: bool = True,
        min_relation_support: int = 2,
        max_agent_rounds: int = 2,
    ):
        self.max_clusters = max(max_clusters, 1)
        self.max_queries_per_cluster = max(max_queries_per_cluster, 1)
        self.max_facts_per_cluster = max(max_facts_per_cluster, 1)
        self.use_llm_agents = use_llm_agents
        self.min_relation_support = max(min_relation_support, 1)
        self.max_agent_rounds = max(max_agent_rounds, 1)

    def build(
        self,
        query_contexts: List[Dict[str, Any]],
        llm_call: Callable[[str, str, Dict[str, Any]], Dict[str, Any]] | None = None,
        dataset_name: str | None = None,
    ) -> Dict[str, Any]:
        clusters = self._cluster_contexts(query_contexts)
        cluster_priors = []

        for cluster_id, contexts in enumerate(clusters):
            cluster_payload = self._build_cluster_payload(cluster_id, contexts)
            deterministic_prior = self._build_deterministic_cluster_prior(cluster_payload)
            llm_prior = {}
            if self.use_llm_agents and llm_call is not None:
                llm_prior = self._run_cluster_agents(cluster_payload, deterministic_prior, llm_call)
            cluster_priors.append(self._merge_cluster_prior(cluster_payload, deterministic_prior, llm_prior))

        return self._synthesize_prior(dataset_name=dataset_name, cluster_priors=cluster_priors)

    def _cluster_contexts(self, query_contexts: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for context in query_contexts:
            groups[_query_signature(context.get("query", ""))].append(context)

        sorted_groups = sorted(groups.items(), key=lambda item: len(item[1]), reverse=True)
        clusters = [contexts[:self.max_queries_per_cluster] for _, contexts in sorted_groups[:self.max_clusters]]
        overflow = []
        for _, contexts in sorted_groups[self.max_clusters:]:
            overflow.extend(contexts[:self.max_queries_per_cluster])
        if overflow:
            clusters[-1].extend(overflow[:self.max_queries_per_cluster])
        return clusters

    def _build_cluster_payload(self, cluster_id: int, contexts: List[Dict[str, Any]]) -> Dict[str, Any]:
        fact_counter = Counter()
        fact_scores = {}
        fact_examples = defaultdict(set)
        passage_snippets = []

        for context in contexts:
            for fact in context.get("facts", []):
                triple = tuple(normalize_text(part) for part in fact.get("fact", []))
                if len(triple) != 3 or not all(triple):
                    continue
                if _is_low_quality_relation(triple[1]):
                    continue
                fact_weight = max(float(fact.get("supervision_weight", 1.0) or 1.0), 1.0)
                fact_counter[triple] += fact_weight
                fact_scores[triple] = max(float(fact.get("score", 0.0)), fact_scores.get(triple, 0.0))
                if context.get("query"):
                    fact_examples[triple].add(context.get("query", ""))
            for passage in context.get("passages", [])[:1]:
                if len(passage_snippets) < 8:
                    passage_snippets.append(passage)

        facts = []
        for triple, support in fact_counter.most_common(self.max_facts_per_cluster):
            source_queries = sorted(fact_examples.get(triple, set()))
            facts.append({
                "fact": list(triple),
                "support": support,
                "query_support": len(source_queries),
                "score": round(fact_scores.get(triple, 0.0), 6),
                "source_query": source_queries[0] if source_queries else "",
                "source_queries": source_queries[:5],
            })

        return {
            "cluster_id": cluster_id,
            "queries": [context.get("query", "") for context in contexts],
            "facts": facts,
            "passages": passage_snippets,
        }

    def _build_deterministic_cluster_prior(self, cluster_payload: Dict[str, Any]) -> Dict[str, Any]:
        query_terms = Counter()
        relation_counts = Counter()
        relation_query_support = defaultdict(set)
        entity_counts = Counter()

        for query in cluster_payload.get("queries", []):
            query_terms.update(tokenize(query))

        for item in cluster_payload.get("facts", []):
            subject, relation, obj = item["fact"]
            support = int(item.get("support", 1))
            relation_counts[relation] += support
            source_queries = item.get("source_queries", []) or [item.get("source_query", "")]
            for source_query in source_queries:
                if source_query:
                    relation_query_support[relation].add(source_query)
            entity_counts[subject] += support
            entity_counts[obj] += support

        top_query_terms = [term for term, _ in query_terms.most_common(16)]
        relation_hints = []
        for relation, support in relation_counts.most_common(16):
            source_queries = relation_query_support.get(relation, set())
            aligned_source_queries = [
                query for query in source_queries
                if _relation_family_aligned_with_query(relation, query)
            ]
            query_support = len(aligned_source_queries)
            if query_support < self.min_relation_support or _is_low_quality_relation(relation):
                continue
            aliases = _clean_relation_aliases(relation)
            if not aliases:
                continue
            query_triggers = _schema_triggers_for_relation(relation, aligned_source_queries)
            if not query_triggers:
                continue
            relation_hints.append({
                "relation": relation,
                "aliases": aliases,
                "relation_family": _relation_family(relation),
                "query_triggers": query_triggers,
                "support": support,
                "query_support": query_support,
                "raw_query_support": len(source_queries),
                "confidence": min(1.0, 0.35 + query_support / 10.0),
            })
            if len(relation_hints) >= 12:
                break

        entity_hints = _infer_type_hints(top_query_terms, relation_counts)
        for entity, support in entity_counts.most_common(16):
            if len(entity_hints) >= 10:
                break
            if not _is_reusable_entity_hint(entity):
                continue
            entity_hints.append({
                "term": entity,
                "aliases": _clean_string_list([entity] + tokenize(entity), max_items=8),
                "support": support,
                "kind": "type_hint",
            })

        bridge_rules = []
        for idx, hint in enumerate(relation_hints[:6]):
            entity_terms = [item["term"] for item in entity_hints[:10]]
            if not entity_terms:
                entity_terms = top_query_terms[:8]
            bridge_rules.append({
                "name": f"cluster_{cluster_payload['cluster_id']}_bridge_{idx}",
                "description": f"Use relation '{hint['relation']}' when the query matches this cluster intent.",
                "triggers": hint.get("query_triggers", []),
                "relation_aliases": hint["aliases"],
                "entity_terms": entity_terms,
                "support": hint["support"],
                "confidence": hint["confidence"],
            })

        bridge_templates = self._build_deterministic_bridge_templates(
            cluster_payload=cluster_payload,
            relation_counts=relation_counts,
            relation_query_support=relation_query_support,
            relation_hints=relation_hints,
            entity_hints=entity_hints,
            top_query_terms=top_query_terms,
        )

        return {
            "entity_hints": entity_hints,
            "relation_hints": relation_hints,
            "bridge_rules": bridge_rules,
            "bridge_templates": bridge_templates,
        }

    def _build_deterministic_bridge_templates(
        self,
        cluster_payload: Dict[str, Any],
        relation_counts: Counter,
        relation_query_support: Dict[str, set[str]],
        relation_hints: List[Dict[str, Any]],
        entity_hints: List[Dict[str, Any]],
        top_query_terms: List[str],
    ) -> List[Dict[str, Any]]:
        templates = []
        facts = cluster_payload.get("facts", [])
        entity_terms = [item["term"] for item in entity_hints[:8]]
        if not entity_terms:
            entity_terms = ["person", "location", "work"]

        def item_source_queries(item: Dict[str, Any]) -> set[str]:
            values = item.get("source_queries", []) or [item.get("source_query", "")]
            return {query for query in values if query}

        def add_template(
            name: str,
            pattern: str,
            answer_type: str,
            relations: List[str],
            triggers: List[str],
            supporting_facts: List[Dict[str, Any]],
            support: float,
            confidence: float,
        ):
            relations = _clean_string_list(relations, max_items=8)
            relation_families = _relation_families(relations)
            if len(set(relations)) < 2 or len(relation_families) < 2:
                return
            template = _clean_bridge_template({
                "name": f"cluster_{cluster_payload['cluster_id']}_{name}",
                "query_pattern": pattern,
                "answer_type": answer_type,
                "required_relations": relations,
                "relation_families": sorted(relation_families),
                "relation_steps": [
                    {"role": "anchor", "relations": relations[: max(1, len(relations) // 2)]},
                    {"role": "answer", "relations": relations[max(1, len(relations) // 2):] or relations},
                ],
                "triggers": triggers or top_query_terms,
                "entity_terms": entity_terms + [answer_type],
                "supporting_facts": supporting_facts,
                "support": support,
                "confidence": confidence,
            })
            if template:
                templates.append(template)

        entity_to_items = defaultdict(list)
        for item in facts:
            fact = item.get("fact", [])
            if len(fact) != 3:
                continue
            subject, relation, obj = fact
            if len(relation_query_support.get(relation, set())) < self.min_relation_support:
                continue
            entity_to_items[subject].append(item)
            entity_to_items[obj].append(item)

        pair_stats: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for bridge_entity, incident_items in entity_to_items.items():
            relation_to_items = defaultdict(list)
            for item in incident_items:
                relation_to_items[item["fact"][1]].append(item)
            relations = sorted(relation_to_items)
            if len(relations) < 2:
                continue
            for idx, relation_a in enumerate(relations):
                for relation_b in relations[idx + 1:]:
                    family_a = _relation_family(relation_a)
                    family_b = _relation_family(relation_b)
                    if not family_a or not family_b or family_a == family_b:
                        continue
                    key = tuple(sorted((family_a, family_b)))
                    stat = pair_stats.setdefault(key, {
                        "relation_examples": defaultdict(Counter),
                        "bridge_entities": set(),
                        "source_queries": set(),
                        "supporting_facts": [],
                    })
                    stat["relation_examples"][family_a][relation_a] += len(relation_to_items[relation_a])
                    stat["relation_examples"][family_b][relation_b] += len(relation_to_items[relation_b])
                    stat["bridge_entities"].add(bridge_entity)
                    for item in relation_to_items[relation_a][:2] + relation_to_items[relation_b][:2]:
                        stat["source_queries"].update(item_source_queries(item))
                        if len(stat["supporting_facts"]) < 8:
                            stat["supporting_facts"].append({"fact": item["fact"]})

        ranked_pairs = sorted(
            pair_stats.items(),
            key=lambda item: (
                len(item[1]["source_queries"]),
                len(item[1]["bridge_entities"]),
                sum(
                    count
                    for examples in item[1]["relation_examples"].values()
                    for count in examples.values()
                ),
            ),
            reverse=True,
        )

        for pair_idx, ((family_a, family_b), stat) in enumerate(ranked_pairs):
            query_support = len(stat["source_queries"])
            bridge_entity_support = len(stat["bridge_entities"])
            if query_support < self.min_relation_support and bridge_entity_support < 2:
                continue
            relation_a = stat["relation_examples"][family_a].most_common(1)[0][0]
            relation_b = stat["relation_examples"][family_b].most_common(1)[0][0]
            relation_terms = tokenize(relation_a) + tokenize(relation_b) + tokenize(family_a) + tokenize(family_b)
            relation_term_set = set(relation_terms)
            schema_query_terms = [term for term in top_query_terms if term in relation_term_set]
            triggers = _clean_schema_trigger_terms(relation_terms + schema_query_terms, max_items=12)
            add_template(
                name=f"observed_relation_pair_{pair_idx}",
                pattern=(
                    "query can be answered by bridging two evidence facts through a shared entity "
                    f"using relation families '{family_a}' and '{family_b}'"
                ),
                answer_type=entity_terms[0] if entity_terms else "answer",
                relations=[relation_a, relation_b],
                triggers=triggers,
                supporting_facts=stat["supporting_facts"],
                support=float(query_support or bridge_entity_support),
                confidence=min(0.9, 0.45 + 0.08 * query_support + 0.04 * bridge_entity_support),
            )
            if len(templates) >= 8:
                break

        return templates[:8]

    def _run_cluster_agents(
        self,
        cluster_payload: Dict[str, Any],
        deterministic_prior: Dict[str, Any],
        llm_call: Callable[[str, str, Dict[str, Any]], Dict[str, Any]],
    ) -> Dict[str, Any]:
        base_payload = {
            "queries": cluster_payload.get("queries", []),
            "candidate_facts": cluster_payload.get("facts", []),
            "deterministic_prior": deterministic_prior,
        }

        entity_state = llm_call(
            "entity_agent",
            (
                "You are an Entity Agent building reusable graph ontology priors from historical QA queries. "
                "Extract only reusable answer/entity TYPES and query trigger terms. Do not return concrete names "
                "from candidate facts unless they are general categories. Prefer terms such as person, location, "
                "date, organization, work, event, creator, parent, spouse. Use only supplied facts and queries. "
                "Return JSON with keys: entity_hints, answer_types, query_triggers."
            ),
            base_payload,
        )
        relation_state = llm_call(
            "relation_agent",
            (
                "You are a Relation Agent building reusable graph ontology priors. "
                "Find relation schemas and aliases that repeatedly help answer this cluster of historical queries. "
                "Reject broad relations such as is, is a, has, includes, features, mentions, says, asks, interacts with. "
                "Keep only specific bridge relations such as born in, directed by, father of, located in, won, authored by. "
                "Use only supplied candidate facts. Return JSON with key relation_hints."
            ),
            {**base_payload, "entity_agent": entity_state},
        )

        bridge_payload = {**base_payload, "entity_agent": entity_state, "relation_agent": relation_state}
        bridge_state = llm_call(
            "bridge_agent_round_1",
            (
                "You are a Bridge Agent in a Saturn-style multi-agent ontology builder for graph QA. "
                "Create reusable bridge_templates from observed relation pairs in candidate_facts. Each template must describe a multi-hop "
                "schema with keys: name, query_pattern, answer_type, source_type, intermediate_type, "
                "target_type, required_relations, relation_steps, triggers, entity_terms, supporting_facts, confidence. "
                "Use only supplied candidate_facts as supporting_facts. Prefer relation-pair schemas where two facts share "
                "an intermediate entity and both relations are specific. Reject broad topic schemas such as generic "
                "creator_birth_place/entity_location/event_or_award unless the exact relation pair is observed in supporting_facts. "
                "Reject broad relations such as is/has/includes/features. "
                "Return JSON with keys: bridge_templates, bridge_rules."
            ),
            bridge_payload,
        )

        debate_state = {}
        for round_idx in range(1, self.max_agent_rounds + 1):
            critic_state = llm_call(
                f"critic_agent_round_{round_idx}",
                (
                    "You are a Critic Agent reviewing proposed graph ontology bridge_templates. "
                    "For each template, check evidence support, relation specificity, reusable query pattern, "
                    "and whether it can select query-anchored bridge facts instead of broad topical facts. "
                    "Reject templates with concrete answer/entity names, generic relations, missing supporting_facts, "
                    "unobserved relation pairs, high-fanout broad schemas, "
                    "or no clear source/intermediate/target type. Return JSON with keys: rejected, revision_notes, "
                    "approved_template_names."
                ),
                {**bridge_payload, "bridge_agent": bridge_state, "previous_debate": debate_state},
            )
            debate_state = critic_state
            if round_idx >= self.max_agent_rounds:
                break
            bridge_state = llm_call(
                f"bridge_agent_revision_round_{round_idx}",
                (
                    "You are the Bridge Agent revising templates after critic feedback. "
                    "Rewrite only templates that can be supported by supplied candidate_facts. "
                    "Each bridge_template must include supporting_facts and precise required_relations that form an observed relation pair. "
                    "Remove generic or one-off templates. Return JSON with keys: bridge_templates, bridge_rules."
                ),
                {
                    **bridge_payload,
                    "bridge_agent_previous": bridge_state,
                    "critic_feedback": critic_state,
                },
            )

        verifier_state = llm_call(
            "final_verifier_agent",
            (
                "You are the Final Verifier Agent. Keep only reusable, evidence-backed ontology priors. "
                "Return JSON with keys: entity_hints, relation_hints, bridge_rules, bridge_templates. "
                "For bridge_templates, require: query_pattern, answer_type, required_relations, triggers, "
                "supporting_facts, confidence. Reject broad relations is/is a/was/has/includes/features/mentions/says. "
                "Reject generic template families unless their exact relation pair appears in supporting_facts. "
                "Reject concrete entity names as entity_hints. Prefer schema-level types such as person, location, "
                "date, organization, work, event."
            ),
            {
                **base_payload,
                "entity_agent": entity_state,
                "relation_agent": relation_state,
                "bridge_agent": bridge_state,
                "critic_agent": debate_state,
            },
        )

        return verifier_state or {
            "entity_hints": entity_state.get("entity_hints", []),
            "relation_hints": relation_state.get("relation_hints", []),
            "bridge_rules": bridge_state.get("bridge_rules", []),
            "bridge_templates": bridge_state.get("bridge_templates", []),
        }

    def _merge_cluster_prior(
        self,
        cluster_payload: Dict[str, Any],
        deterministic_prior: Dict[str, Any],
        llm_prior: Dict[str, Any],
    ) -> Dict[str, Any]:
        merged = {
            "cluster_id": cluster_payload["cluster_id"],
            "queries": cluster_payload.get("queries", [])[:5],
            "entity_hints": [],
            "relation_hints": [],
            "bridge_rules": [],
            "bridge_templates": [],
        }

        merged["entity_hints"] = self._merge_entity_hints(
            deterministic_prior.get("entity_hints", []),
            llm_prior.get("entity_hints", []),
            llm_prior.get("answer_types", []),
        )
        merged["relation_hints"] = self._merge_relation_hints(
            deterministic_prior.get("relation_hints", []),
            llm_prior.get("relation_hints", []),
        )
        merged["bridge_rules"] = self._merge_bridge_rules(
            deterministic_prior.get("bridge_rules", []),
            llm_prior.get("bridge_rules", []),
        )
        merged["bridge_templates"] = self._merge_bridge_templates(
            deterministic_prior.get("bridge_templates", []),
            llm_prior.get("bridge_templates", []),
        )
        return merged

    @staticmethod
    def _merge_entity_hints(*hint_groups: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        counter = Counter()
        alias_map = defaultdict(set)
        for hints in hint_groups:
            for hint in hints or []:
                if not isinstance(hint, dict):
                    hint = {"term": hint}
                term = normalize_text(
                    hint.get("term") or hint.get("entity") or hint.get("name")
                    or hint.get("type") or hint.get("answer_type")
                )
                if not term or not _is_reusable_entity_hint(term):
                    continue
                counter[term] += float(hint.get("support", 1.0) or 1.0)
                alias_map[term].update(_clean_string_list(hint.get("aliases", []), max_items=12))
                alias_map[term].add(term)
        return [
            {
                "term": term,
                "aliases": _clean_entity_terms(sorted(alias_map[term])[:12], max_items=12),
                "support": support,
                "kind": "type_hint",
            }
            for term, support in counter.most_common(20)
        ]

    @staticmethod
    def _merge_relation_hints(*hint_groups: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        counter = Counter()
        alias_map = defaultdict(set)
        trigger_map = defaultdict(set)
        family_map = {}
        for hints in hint_groups:
            for hint in hints or []:
                if not isinstance(hint, dict):
                    hint = {"relation": hint}
                relation = normalize_text(hint.get("relation") or hint.get("name"))
                if not relation or _is_low_quality_relation(relation):
                    continue
                counter[relation] += float(hint.get("support", 1.0) or 1.0)
                alias_map[relation].update(_clean_relation_aliases(relation, hint.get("aliases", []), max_items=12))
                relation_family = normalize_text(hint.get("relation_family"))
                if relation_family in {"", "none", "null", "misc", "unknown"}:
                    relation_family = _relation_family(relation)
                family_map[relation] = relation_family
                trigger_map[relation].update(
                    _aligned_trigger_terms_for_relation(
                        relation,
                        hint.get("query_triggers", []) or hint.get("triggers", []),
                        max_items=16,
                    )
                )
        return [
            {
                "relation": relation,
                "aliases": sorted(alias_map[relation])[:12],
                "query_triggers": sorted(trigger_map[relation])[:16],
                "relation_family": family_map.get(relation) or _relation_family(relation),
                "support": support,
                "confidence": min(1.0, 0.35 + support / 20.0),
            }
            for relation, support in counter.most_common(24)
            if alias_map[relation]
        ]

    @staticmethod
    def _merge_bridge_rules(*rule_groups: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        rules = []
        seen = set()
        for group in rule_groups:
            for rule in group or []:
                if not isinstance(rule, dict):
                    continue
                raw_aliases = rule.get("relation_aliases", []) or rule.get("relations", [])
                relation_aliases = []
                for alias in _clean_string_list(raw_aliases, max_items=12):
                    relation_aliases.extend(_clean_relation_aliases(alias, max_items=4))
                relation_aliases = _clean_string_list(relation_aliases, max_items=12)
                triggers = _clean_trigger_terms(rule.get("triggers", []), max_items=16)
                entity_terms = _clean_entity_terms(rule.get("entity_terms", []) or rule.get("answer_types", []), max_items=16)
                if not relation_aliases or not triggers or not entity_terms:
                    continue
                key = (tuple(relation_aliases[:4]), tuple(triggers[:4]))
                if key in seen:
                    continue
                seen.add(key)
                rules.append({
                    "name": normalize_text(rule.get("name")) or f"bridge_rule_{len(rules)}",
                    "description": str(rule.get("description", ""))[:300],
                    "triggers": triggers,
                    "relation_aliases": relation_aliases,
                    "entity_terms": entity_terms,
                    "support": float(rule.get("support", 1.0) or 1.0),
                    "confidence": min(max(float(rule.get("confidence", 0.5) or 0.5), 0.0), 1.0),
                })
                if len(rules) >= 48:
                    return rules
        return rules

    @staticmethod
    def _merge_bridge_templates(*template_groups: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        templates = []
        seen = set()
        for group in template_groups:
            for template in group or []:
                cleaned = _clean_bridge_template(template, default_name=f"bridge_template_{len(templates)}")
                if not cleaned:
                    continue
                key = (
                    tuple(cleaned.get("triggers", [])[:5]),
                    tuple(cleaned.get("required_relations", [])[:6]),
                    cleaned.get("answer_type", ""),
                )
                if key in seen:
                    continue
                seen.add(key)
                templates.append(cleaned)
                if len(templates) >= 64:
                    return templates
        return templates

    @staticmethod
    def _synthesize_prior(dataset_name: str | None, cluster_priors: List[Dict[str, Any]]) -> Dict[str, Any]:
        entity_hints = QueryOntologyPriorBuilder._merge_entity_hints(
            *(cluster.get("entity_hints", []) for cluster in cluster_priors)
        )
        relation_hints = QueryOntologyPriorBuilder._merge_relation_hints(
            *(cluster.get("relation_hints", []) for cluster in cluster_priors)
        )
        bridge_rules = QueryOntologyPriorBuilder._merge_bridge_rules(
            *(cluster.get("bridge_rules", []) for cluster in cluster_priors)
        )
        bridge_templates = QueryOntologyPriorBuilder._merge_bridge_templates(
            *(cluster.get("bridge_templates", []) for cluster in cluster_priors)
        )
        return {
            "version": 2,
            "kind": "historical_query_ontology_prior",
            "dataset": dataset_name,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "entity_hints": entity_hints,
            "relation_hints": relation_hints,
            "bridge_rules": bridge_rules,
            "bridge_templates": bridge_templates,
            "clusters": cluster_priors,
        }

    @staticmethod
    def save(path: str, prior: Dict[str, Any]):
        with open(path, "w") as f:
            json.dump(prior, f, indent=2, ensure_ascii=False)


class QueryOntologyPriorMatcher:
    """Lightweight online matcher for historical query ontology priors."""

    def __init__(self, prior: Dict[str, Any]):
        self.prior = prior or {}
        self.relation_hints = self._sanitize_relation_hints(self.prior.get("relation_hints", []))
        self.entity_hints = self._sanitize_entity_hints(self.prior.get("entity_hints", []))
        self.bridge_rules = self._sanitize_bridge_rules(self.prior.get("bridge_rules", []))
        self.bridge_templates = self._sanitize_bridge_templates(self.prior.get("bridge_templates", []))

    @classmethod
    def load(cls, path: str) -> "QueryOntologyPriorMatcher":
        with open(path, "r") as f:
            return cls(json.load(f))

    def match(self, query: str, top_k: int = 8) -> Dict[str, Any]:
        query_tokens = set(tokenize(query))
        query_families = _query_intent_families(query)
        relation_matches = []
        bridge_matches = []
        entity_matches = []
        template_matches = []

        for hint in self.relation_hints:
            aliases = set(token for alias in hint.get("aliases", []) for token in tokenize(alias))
            triggers = set(token for trigger in hint.get("query_triggers", []) for token in tokenize(trigger))
            relation_family = normalize_text(hint.get("relation_family")) or _relation_family(hint.get("relation", ""))
            family_tokens = set(tokenize(relation_family))
            if not aliases:
                continue
            family_aligned = bool(relation_family and relation_family in query_families)
            relation_score = self._overlap_score(query_tokens, aliases | family_tokens)
            if not family_aligned and relation_score <= 0:
                continue
            trigger_score = self._overlap_score(query_tokens, triggers) if family_aligned else 0.0
            score = max(relation_score, trigger_score)
            if score > 0:
                relation_matches.append((score * float(hint.get("confidence", 0.5)), hint))

        for rule in self.bridge_rules:
            triggers = set(token for trigger in rule.get("triggers", []) for token in tokenize(trigger))
            relation_aliases = set(token for alias in rule.get("relation_aliases", []) for token in tokenize(alias))
            if not triggers or not relation_aliases:
                continue
            relation_families = _relation_families(rule.get("relation_aliases", []))
            family_tokens = set(token for family in relation_families for token in tokenize(family))
            family_aligned = bool(query_families & relation_families)
            relation_score = self._overlap_score(query_tokens, relation_aliases | family_tokens)
            if not family_aligned and relation_score <= 0:
                continue
            trigger_score = self._overlap_score(query_tokens, triggers) if family_aligned else 0.0
            score = max(relation_score, trigger_score)
            if score > 0:
                bridge_matches.append((score * float(rule.get("confidence", 0.5)), rule, relation_aliases))

        for hint in self.entity_hints:
            aliases = set(token for alias in hint.get("aliases", []) for token in tokenize(alias))
            score = self._overlap_score(query_tokens, aliases)
            if score > 0:
                entity_matches.append((score, hint))

        for template in self.bridge_templates:
            trigger_tokens = set(token for trigger in template.get("triggers", []) for token in tokenize(trigger))
            pattern_tokens = set(tokenize(template.get("query_pattern", "")))
            relation_tokens = set(
                token for relation in template.get("required_relations", []) for token in tokenize(relation)
            )
            family_tokens = _relation_family_trigger_tokens(template)
            relation_score = self._overlap_score(query_tokens, relation_tokens | family_tokens)
            if relation_score <= 0:
                continue
            trigger_score = self._overlap_score(query_tokens, trigger_tokens | pattern_tokens)
            score = max(0.75 * relation_score, 0.50 * trigger_score + 0.50 * relation_score)
            if score > 0:
                template_matches.append((score * float(template.get("confidence", 0.5)), template))

        relation_matches.sort(reverse=True, key=lambda item: item[0])
        bridge_matches.sort(reverse=True, key=lambda item: item[0])
        entity_matches.sort(reverse=True, key=lambda item: item[0])
        template_matches.sort(reverse=True, key=lambda item: item[0])

        return {
            "relations": [hint for _, hint in relation_matches[:top_k]],
            "bridges": [rule for _, rule, _ in bridge_matches[:top_k]],
            "entities": [hint for _, hint in entity_matches[:top_k]],
            "templates": [template for _, template in template_matches[:top_k]],
        }

    @staticmethod
    def _overlap_score(query_tokens: set[str], hint_tokens: set[str]) -> float:
        if not query_tokens or not hint_tokens:
            return 0.0
        overlap = query_tokens & hint_tokens
        if not overlap:
            return 0.0
        return len(overlap) / max(min(len(query_tokens), len(hint_tokens)), 1)

    @staticmethod
    def _sanitize_relation_hints(hints: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        sanitized = []
        for hint in hints or []:
            if not isinstance(hint, dict):
                hint = {"relation": hint}
            relation = normalize_text(hint.get("relation") or hint.get("name"))
            if not relation or _is_low_quality_relation(relation):
                continue
            aliases = _clean_relation_aliases(relation, hint.get("aliases", []), max_items=12)
            if not aliases:
                continue
            relation_family = normalize_text(hint.get("relation_family"))
            if relation_family in {"", "none", "null", "misc", "unknown"}:
                relation_family = _relation_family(relation)
            query_triggers = _aligned_trigger_terms_for_relation(
                relation,
                hint.get("query_triggers", []) or hint.get("triggers", []) or aliases,
                max_items=16,
            )
            sanitized.append({
                **hint,
                "relation": relation,
                "aliases": aliases,
                "relation_family": relation_family,
                "query_triggers": query_triggers,
            })
        return sanitized

    @staticmethod
    def _sanitize_entity_hints(hints: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        sanitized = []
        for hint in hints or []:
            if not isinstance(hint, dict):
                hint = {"term": hint}
            term = normalize_text(
                hint.get("term") or hint.get("entity") or hint.get("name")
                or hint.get("type") or hint.get("answer_type")
            )
            if not term or not _is_reusable_entity_hint(term):
                continue
            aliases = _clean_entity_terms([term] + _clean_string_list(hint.get("aliases", []), max_items=12), max_items=12)
            if not aliases:
                continue
            sanitized.append({**hint, "term": term, "aliases": aliases, "kind": "type_hint"})
        return sanitized

    @staticmethod
    def _sanitize_bridge_rules(rules: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        sanitized = []
        seen = set()
        for rule in rules or []:
            if not isinstance(rule, dict):
                continue
            relation_aliases = []
            for alias in _clean_string_list(rule.get("relation_aliases", []) or rule.get("relations", []), max_items=12):
                relation_aliases.extend(_clean_relation_aliases(alias, max_items=4))
            relation_aliases = _clean_string_list(relation_aliases, max_items=12)
            triggers = _clean_trigger_terms(rule.get("triggers", []), max_items=16)
            entity_terms = _clean_entity_terms(rule.get("entity_terms", []) or rule.get("answer_types", []), max_items=16)
            if not relation_aliases or not triggers or not entity_terms:
                continue
            key = (tuple(relation_aliases[:4]), tuple(triggers[:4]), tuple(entity_terms[:4]))
            if key in seen:
                continue
            seen.add(key)
            sanitized.append({
                **rule,
                "relation_aliases": relation_aliases,
                "triggers": triggers,
                "entity_terms": entity_terms,
                "confidence": min(max(float(rule.get("confidence", 0.5) or 0.5), 0.0), 1.0),
            })
        return sanitized

    @staticmethod
    def _sanitize_bridge_templates(templates: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        sanitized = []
        seen = set()
        for template in templates or []:
            cleaned = _clean_bridge_template(template, default_name=f"bridge_template_{len(sanitized)}")
            if not cleaned:
                continue
            key = (
                tuple(cleaned.get("triggers", [])[:5]),
                tuple(cleaned.get("required_relations", [])[:6]),
                cleaned.get("answer_type", ""),
            )
            if key in seen:
                continue
            seen.add(key)
            sanitized.append(cleaned)
        return sanitized

    def boost_fact_scores(
        self,
        query: str,
        query_fact_scores: np.ndarray,
        fact_triples: List[Tuple[str, str, str] | None],
        boost_weight: float = 0.12,
        top_k: int = 8,
        max_boost_facts: int = 80,
        schema_mode: str = "gated",
        template_candidate_top_k: int = 500,
        template_max_boost_facts: int = 20,
        hint_candidate_top_k: int = 500,
        require_query_anchor: bool = True,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        schema_mode = schema_mode if schema_mode in {"gated", "hints_only", "templates_only", "path_gated"} else "gated"
        log = {
            "enabled": True,
            "schema_mode": schema_mode,
            "matched_relation_count": 0,
            "matched_bridge_count": 0,
            "matched_template_count": 0,
            "boosted_fact_count": 0,
            "hint_candidate_top_k": hint_candidate_top_k,
            "require_query_anchor": require_query_anchor,
        }
        if len(query_fact_scores) == 0:
            return query_fact_scores, log

        matches = self.match(query, top_k=top_k)
        log["matched_relation_count"] = len(matches["relations"])
        log["matched_bridge_count"] = len(matches["bridges"])
        log["matched_entity_count"] = len(matches["entities"])
        log["matched_template_count"] = len(matches.get("templates", []))
        log["matched_templates"] = [
            {
                "name": template.get("name", ""),
                "required_relations": template.get("required_relations", []),
                "relation_families": template.get("relation_families", []),
                "answer_type": template.get("answer_type", ""),
                "confidence": template.get("confidence", 0.5),
            }
            for template in matches.get("templates", [])
        ]
        if not matches["relations"] and not matches["bridges"] and not matches["entities"] and not matches.get("templates"):
            return query_fact_scores, log

        if boost_weight <= 0.001:
            log["matching_only"] = True
            return query_fact_scores, log

        effective_schema_mode = schema_mode
        if schema_mode == "path_gated" and not matches.get("templates") and matches["relations"]:
            effective_schema_mode = "hints_only"
            log["schema_mode_fallback"] = "hints_only_no_templates"

        query_tokens = set(tokenize(query))
        relation_hint_tokens = self._collect_relation_tokens(matches)
        bridge_trigger_tokens = self._collect_bridge_trigger_tokens(matches)
        prior_entity_tokens = self._collect_entity_tokens(matches)

        boost = np.zeros_like(query_fact_scores, dtype=np.float32)
        if effective_schema_mode == "path_gated":
            boost = self._compute_path_gated_template_boost(
                query=query,
                query_fact_scores=query_fact_scores,
                templates=matches.get("templates", []),
                fact_triples=fact_triples,
                candidate_top_k=template_candidate_top_k,
                max_template_boost_facts=template_max_boost_facts,
            )
        elif effective_schema_mode != "hints_only":
            template_boost = self._compute_template_boost(
                query=query,
                query_fact_scores=query_fact_scores,
                templates=matches.get("templates", []),
                fact_triples=fact_triples,
                candidate_top_k=template_candidate_top_k,
                max_template_boost_facts=template_max_boost_facts,
            )
            if len(template_boost) == len(boost):
                boost = np.maximum(boost, template_boost)

        limit = min(len(query_fact_scores), len(fact_triples))
        if hint_candidate_top_k > 0 and limit > hint_candidate_top_k:
            hint_candidate_indices = np.argpartition(query_fact_scores[:limit], -hint_candidate_top_k)[-hint_candidate_top_k:]
        else:
            hint_candidate_indices = np.arange(limit)

        if effective_schema_mode not in {"templates_only", "path_gated"}:
            for fact_idx in hint_candidate_indices:
                fact_idx = int(fact_idx)
                triple = fact_triples[fact_idx]
                if triple is None:
                    continue
                subject, relation, obj = triple
                relation_tokens = set(tokenize(relation))
                entity_tokens = set(tokenize(f"{subject} {obj}"))

                relation_hit = self._overlap_score(relation_tokens, relation_hint_tokens)
                query_entity_hit = self._overlap_score(query_tokens, entity_tokens)
                prior_entity_hit = self._overlap_score(prior_entity_tokens, entity_tokens)
                bridge_hit = self._overlap_score(query_tokens, bridge_trigger_tokens)
                anchor_hit = query_entity_hit if require_query_anchor else max(query_entity_hit, prior_entity_hit)

                fact_boost = 0.0
                if bridge_hit > 0 and relation_hit > 0 and anchor_hit > 0:
                    fact_boost += 0.60 * bridge_hit + 0.80 * relation_hit + 0.35 * anchor_hit
                elif relation_hit > 0 and anchor_hit > 0:
                    fact_boost += 1.00 * relation_hit + 0.50 * anchor_hit

                if fact_boost > 0:
                    boost[fact_idx] = max(float(boost[fact_idx]), fact_boost * 0.7)

        nonzero = np.flatnonzero(boost)
        if len(nonzero) == 0:
            return query_fact_scores, log

        if max_boost_facts > 0 and len(nonzero) > max_boost_facts:
            ranking = sorted(
                nonzero.tolist(),
                key=lambda idx: (float(boost[idx]), float(query_fact_scores[idx])),
                reverse=True,
            )
            keep = np.array(ranking[:max_boost_facts], dtype=np.intp)
            filtered_boost = np.zeros_like(boost)
            filtered_boost[keep] = boost[keep]
            boost = filtered_boost
            nonzero = keep

        boost = min_max_normalize(boost)
        combined_scores = (1.0 - boost_weight) * query_fact_scores + boost_weight * boost
        log["boosted_fact_count"] = int(len(nonzero))
        log["boosted_fact_indices"] = [int(idx) for idx in nonzero.tolist()]
        return min_max_normalize(combined_scores), log

    def _compute_path_gated_template_boost(
        self,
        query: str,
        query_fact_scores: np.ndarray,
        templates: List[Dict[str, Any]],
        fact_triples: List[Tuple[str, str, str] | None],
        candidate_top_k: int = 500,
        max_template_boost_facts: int = 20,
    ) -> np.ndarray:
        """Boost only relation-matched facts that form a query-anchored local path.

        This mode is deliberately conservative.  A template is allowed to boost
        facts only when one top candidate fact touches concrete query tokens and
        another top candidate shares an entity with it under a compatible but
        non-identical relation family.  The goal is to prevent broad historical
        schemas from lifting unrelated facts with the right relation words.
        """
        boost = np.zeros(len(query_fact_scores), dtype=np.float32)
        limit = min(len(query_fact_scores), len(fact_triples))
        if not templates or limit == 0:
            return boost

        if candidate_top_k > 0 and limit > candidate_top_k:
            candidate_indices = np.argpartition(query_fact_scores[:limit], -candidate_top_k)[-candidate_top_k:]
        else:
            candidate_indices = np.arange(limit)

        query_tokens = set(tokenize(query))
        for template in templates:
            required_relations = template.get("required_relations", []) + template.get("relation_aliases", [])
            relation_tokens = set(token for relation in required_relations for token in tokenize(relation))
            relation_family_tokens = _relation_family_trigger_tokens(template)
            template_families = set(template.get("relation_families", [])) or _relation_families(required_relations)
            template_families = {normalize_text(family) for family in template_families if normalize_text(family)}

            if len(template_families) < 2 or not relation_tokens:
                continue

            trigger_tokens = set(token for trigger in template.get("triggers", []) for token in tokenize(trigger))
            pattern_tokens = set(tokenize(template.get("query_pattern", "")))
            trigger_hit = self._overlap_score(query_tokens, trigger_tokens | pattern_tokens | relation_family_tokens)
            relation_trigger_hit = self._overlap_score(query_tokens, relation_tokens | relation_family_tokens)
            if max(trigger_hit, relation_trigger_hit) <= 0:
                continue

            candidate_infos = []
            entity_to_candidate_positions = defaultdict(list)
            for fact_idx in candidate_indices:
                fact_idx = int(fact_idx)
                triple = fact_triples[fact_idx]
                if triple is None:
                    continue
                subject, relation, obj = triple
                relation_family = normalize_text(_relation_family(relation))
                fact_relation_tokens = set(tokenize(relation))
                if relation_family:
                    fact_relation_tokens.update(tokenize(relation_family))
                relation_hit = self._overlap_score(
                    fact_relation_tokens,
                    relation_tokens | relation_family_tokens | set(token for family in template_families for token in tokenize(family)),
                )
                if relation_hit <= 0 and relation_family not in template_families:
                    continue

                entities = [normalize_text(subject), normalize_text(obj)]
                entity_tokens = set(tokenize(f"{subject} {obj}"))
                query_anchor_hit = self._overlap_score(query_tokens, entity_tokens)
                info = {
                    "idx": fact_idx,
                    "family": relation_family or normalize_text(relation),
                    "relation_hit": max(relation_hit, 0.4 if relation_family in template_families else 0.0),
                    "query_anchor_hit": query_anchor_hit,
                    "entities": entities,
                }
                candidate_infos.append(info)
                pos = len(candidate_infos) - 1
                for entity in entities:
                    if entity:
                        entity_to_candidate_positions[entity].append(pos)

            if not candidate_infos:
                continue

            path_scores = defaultdict(float)
            for anchor_pos, anchor in enumerate(candidate_infos):
                if anchor["query_anchor_hit"] <= 0:
                    continue
                for entity in anchor["entities"]:
                    for neighbor_pos in entity_to_candidate_positions.get(entity, []):
                        if neighbor_pos == anchor_pos:
                            continue
                        neighbor = candidate_infos[neighbor_pos]
                        if neighbor["family"] == anchor["family"]:
                            continue
                        path_families = {anchor["family"], neighbor["family"]}
                        if len(path_families & template_families) < 2:
                            continue
                        path_score = (
                            0.60 * max(trigger_hit, relation_trigger_hit)
                            + 0.65 * anchor["relation_hit"]
                            + 0.65 * neighbor["relation_hit"]
                            + 0.80 * anchor["query_anchor_hit"]
                        )
                        path_scores[anchor["idx"]] = max(path_scores[anchor["idx"]], path_score)
                        path_scores[neighbor["idx"]] = max(path_scores[neighbor["idx"]], path_score * 0.95)

            for fact_idx, score in path_scores.items():
                boost[int(fact_idx)] = max(float(boost[int(fact_idx)]), float(score))

        nonzero = np.flatnonzero(boost)
        if max_template_boost_facts > 0 and len(nonzero) > max_template_boost_facts:
            ranking = sorted(
                nonzero.tolist(),
                key=lambda idx: (float(boost[idx]), float(query_fact_scores[idx])),
                reverse=True,
            )
            keep = np.array(ranking[:max_template_boost_facts], dtype=np.intp)
            filtered = np.zeros_like(boost)
            filtered[keep] = boost[keep]
            boost = filtered

        return boost

    def _compute_template_boost(
        self,
        query: str,
        query_fact_scores: np.ndarray,
        templates: List[Dict[str, Any]],
        fact_triples: List[Tuple[str, str, str] | None],
        candidate_top_k: int = 500,
        max_template_boost_facts: int = 20,
    ) -> np.ndarray:
        boost = np.zeros(len(query_fact_scores), dtype=np.float32)
        limit = min(len(query_fact_scores), len(fact_triples))
        if not templates or limit == 0:
            return boost

        if candidate_top_k > 0 and limit > candidate_top_k:
            candidate_indices = np.argpartition(query_fact_scores[:limit], -candidate_top_k)[-candidate_top_k:]
        else:
            candidate_indices = np.arange(limit)

        query_tokens = set(tokenize(query))
        for template in templates:
            trigger_tokens = set(token for trigger in template.get("triggers", []) for token in tokenize(trigger))
            pattern_tokens = set(tokenize(template.get("query_pattern", "")))
            template_trigger_hit = self._overlap_score(query_tokens, trigger_tokens | pattern_tokens)

            relation_tokens = set(
                token for relation in template.get("required_relations", []) for token in tokenize(relation)
            )
            relation_family_tokens = _relation_family_trigger_tokens(template)
            if not relation_tokens:
                continue
            relation_trigger_hit = self._overlap_score(query_tokens, relation_tokens | relation_family_tokens)
            if relation_trigger_hit <= 0:
                continue
            template_trigger_hit = max(template_trigger_hit, relation_trigger_hit)

            candidate_relation_families = set()
            for fact_idx in candidate_indices:
                triple = fact_triples[int(fact_idx)]
                if triple is None:
                    continue
                _, relation, _ = triple
                relation_family = _relation_family(relation)
                fact_relation_tokens = set(tokenize(relation))
                if relation_family:
                    fact_relation_tokens.update(tokenize(relation_family))
                if self._overlap_score(fact_relation_tokens, relation_tokens | relation_family_tokens) > 0:
                    candidate_relation_families.add(relation_family or relation)
                    if len(candidate_relation_families) >= 2:
                        break
            if len(candidate_relation_families) < 2:
                continue

            seed_entity_tokens = set()
            for fact_idx in candidate_indices:
                triple = fact_triples[int(fact_idx)]
                if triple is None:
                    continue
                subject, relation, obj = triple
                fact_relation_tokens = set(tokenize(relation))
                relation_family = _relation_family(relation)
                if relation_family:
                    fact_relation_tokens.update(tokenize(relation_family))
                relation_hit = self._overlap_score(fact_relation_tokens, relation_tokens | relation_family_tokens)
                if relation_hit <= 0:
                    continue
                entity_tokens = set(tokenize(f"{subject} {obj}"))
                query_entity_hit = self._overlap_score(query_tokens, entity_tokens)
                if query_entity_hit <= 0:
                    continue
                seed_entity_tokens.update(entity_tokens)
                boost[fact_idx] = max(
                    float(boost[fact_idx]),
                    0.75 * template_trigger_hit + 0.85 * relation_hit + 0.45 * query_entity_hit,
                )

            if not seed_entity_tokens:
                continue

            for fact_idx in candidate_indices:
                triple = fact_triples[int(fact_idx)]
                if triple is None:
                    continue
                subject, relation, obj = triple
                fact_relation_tokens = set(tokenize(relation))
                relation_family = _relation_family(relation)
                if relation_family:
                    fact_relation_tokens.update(tokenize(relation_family))
                relation_hit = self._overlap_score(fact_relation_tokens, relation_tokens | relation_family_tokens)
                if relation_hit <= 0:
                    continue
                entity_tokens = set(tokenize(f"{subject} {obj}"))
                seed_hit = self._overlap_score(seed_entity_tokens, entity_tokens)
                if seed_hit <= 0:
                    continue
                boost[fact_idx] = max(
                    float(boost[fact_idx]),
                    0.45 * template_trigger_hit + 1.00 * relation_hit + 0.90 * seed_hit,
                )

        nonzero = np.flatnonzero(boost)
        if max_template_boost_facts > 0 and len(nonzero) > max_template_boost_facts:
            ranking = sorted(
                nonzero.tolist(),
                key=lambda idx: (float(boost[idx]), float(query_fact_scores[idx])),
                reverse=True,
            )
            keep = np.array(ranking[:max_template_boost_facts], dtype=np.intp)
            filtered = np.zeros_like(boost)
            filtered[keep] = boost[keep]
            boost = filtered

        return boost

    @staticmethod
    def _collect_relation_tokens(matches: Dict[str, Any]) -> set[str]:
        tokens = set()
        for hint in matches.get("relations", []):
            for alias in hint.get("aliases", []) + [hint.get("relation", "")]:
                tokens.update(tokenize(alias))
        for rule in matches.get("bridges", []):
            for alias in rule.get("relation_aliases", []):
                tokens.update(tokenize(alias))
        for template in matches.get("templates", []):
            for alias in template.get("required_relations", []) + template.get("relation_aliases", []):
                tokens.update(tokenize(alias))
        return tokens

    @staticmethod
    def _collect_bridge_trigger_tokens(matches: Dict[str, Any]) -> set[str]:
        tokens = set()
        for rule in matches.get("bridges", []):
            for trigger in rule.get("triggers", []):
                tokens.update(tokenize(trigger))
        for template in matches.get("templates", []):
            for trigger in template.get("triggers", []):
                tokens.update(tokenize(trigger))
        return tokens

    @staticmethod
    def _collect_entity_tokens(matches: Dict[str, Any]) -> set[str]:
        tokens = set()
        for hint in matches.get("entities", []):
            for alias in hint.get("aliases", []) + [hint.get("term", "")]:
                tokens.update(tokenize(alias))
        for rule in matches.get("bridges", []):
            for term in rule.get("entity_terms", []):
                tokens.update(tokenize(term))
        for template in matches.get("templates", []):
            for term in template.get("entity_terms", []):
                tokens.update(tokenize(term))
        return tokens
