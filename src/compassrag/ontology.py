import json
import os
import re
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Tuple

from .utils.misc_utils import text_processing
from .utils.logging_utils import get_logger


logger = get_logger(__name__)

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "had",
    "in", "is", "it", "its", "of", "on", "or", "that", "the", "to", "was", "were",
    "when", "where", "which", "who", "whom", "whose", "with",
    "her", "his", "him", "their", "them",
}
GENERIC_ENTITY_TERMS = {
    "actor", "album", "area", "artist", "author", "book", "capital", "character",
    "city", "company", "country", "director", "event", "film", "game", "group",
    "location", "movie", "novel", "organization", "person", "place", "player",
    "river", "school", "season", "series", "song", "state", "team", "university",
    "writer", "after", "arena", "birth", "date", "father", "mother", "title",
    "time", "work", "best", "chart", "charts", "daughter", "final", "friend",
    "friends", "house", "label", "production", "son", "soundtrack",
}
WEAK_ENTITY_MATCH_TERMS = GENERIC_ENTITY_TERMS | {
    "american", "award", "championship", "college", "english", "football", "high",
    "league", "national", "north", "record", "senior", "states", "united", "war",
    "world", "academy",
}
RELATION_TRIGGER_TERMS = {
    "authored", "became", "born", "capital", "city", "created", "died", "directed",
    "director", "father", "founded", "headquartered", "located", "married", "member",
    "mother", "owned", "part", "performed", "played", "published", "released",
    "served", "son", "starred", "stars", "won", "wrote", "written",
}
LOW_QUALITY_RELATION_PHRASES = {
    "asks", "contains", "describes", "features", "full name", "had", "has", "have",
    "include", "includes", "is associated with", "is the date of", "label", "lost",
    "made", "mentions", "saw", "says", "took",
}


def _clean_term(value: Any) -> str:
    return text_processing(value).strip()


def _tokenize(value: str) -> List[str]:
    return [token for token in _clean_term(value).split() if len(token) > 2 and token not in STOPWORDS]


def _contains_clean_phrase(text: str, phrase: str) -> bool:
    return f" {phrase} " in f" {text} "


def _is_year_or_date(value: str) -> bool:
    value = value.strip().lower()
    if re.search(r"\b(1[5-9]\d{2}|20\d{2})\b", value):
        return True
    return bool(re.search(r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b", value))


def _is_number(value: str) -> bool:
    return bool(re.fullmatch(r"[\d,]+(\.\d+)?(%| percent| km| m| kg| million| billion| thousand)?", value.strip().lower()))


def _has_named_entity_signal(entity_tokens: set[str]) -> bool:
    return bool(entity_tokens - WEAK_ENTITY_MATCH_TERMS)


def _is_low_quality_query_entity(entity: str, entity_tokens: set[str]) -> bool:
    if not entity_tokens:
        return True
    if _is_year_or_date(entity) or _is_number(entity):
        return True
    if not _has_named_entity_signal(entity_tokens):
        return True
    return len(entity_tokens) == 1 and next(iter(entity_tokens)) in WEAK_ENTITY_MATCH_TERMS


def _is_low_quality_query_relation(relation: str) -> bool:
    relation_clean = _clean_term(relation)
    relation_tokens = set(_tokenize(relation_clean))
    if not relation_tokens:
        return True
    if relation_clean in LOW_QUALITY_RELATION_PHRASES:
        return True
    if len(relation_tokens) == 1 and next(iter(relation_tokens)) not in RELATION_TRIGGER_TERMS:
        return True
    return False


def _is_confident_fuzzy_entity_match(entity_tokens: set[str], query_tokens: set[str]) -> bool:
    overlap = entity_tokens & query_tokens
    if not overlap or overlap <= WEAK_ENTITY_MATCH_TERMS:
        return False

    overlap_count = len(overlap)
    coverage = overlap_count / len(entity_tokens)
    strong_overlap = overlap - WEAK_ENTITY_MATCH_TERMS

    if len(entity_tokens) <= 2:
        return overlap_count == len(entity_tokens) and bool(strong_overlap)
    if overlap_count >= 3 and coverage >= 0.5 and bool(strong_overlap):
        return True
    return coverage >= 0.75 and len(strong_overlap) >= 2


class EntityTypingAgent:
    """Infers coarse graph concepts from entity surfaces and relation context."""

    PERSON_HINTS = {
        "actor", "actress", "author", "composer", "director", "father", "mother", "person",
        "player", "politician", "president", "singer", "son", "wife", "writer",
    }
    LOCATION_HINTS = {
        "area", "capital", "city", "country", "county", "district", "located", "location",
        "province", "region", "river", "state", "town", "village",
    }
    WORK_HINTS = {
        "album", "book", "directed", "film", "movie", "novel", "published", "released",
        "song", "written",
    }
    ORG_HINTS = {
        "club", "company", "founded", "league", "member", "organization", "owned",
        "party", "school", "team", "university",
    }
    EVENT_HINTS = {
        "award", "championship", "competition", "event", "game", "held", "season",
        "tournament", "won",
    }

    def infer(self, entity: str, relations: Iterable[str]) -> List[str]:
        entity_text = _clean_term(entity)
        rel_text = " ".join(_clean_term(r) for r in relations)
        evidence = f"{entity_text} {rel_text}"
        types = []

        if _is_year_or_date(entity_text):
            types.append("Time")
        if _is_number(entity_text):
            types.append("Quantity")
        if any(hint in evidence for hint in self.PERSON_HINTS):
            types.append("Person")
        if any(hint in evidence for hint in self.LOCATION_HINTS):
            types.append("Location")
        if any(hint in evidence for hint in self.WORK_HINTS):
            types.append("CreativeWork")
        if any(hint in evidence for hint in self.ORG_HINTS):
            types.append("Organization")
        if any(hint in evidence for hint in self.EVENT_HINTS):
            types.append("Event")

        return types or ["Entity"]


class RelationSchemaAgent:
    """Aggregates relation phrases into reusable graph relation schemas."""

    def build(
        self,
        triples: List[Tuple[str, str, str]],
        entity_types: Dict[str, Counter],
        max_relation_schemas: int,
    ) -> List[Dict[str, Any]]:
        relation_to_examples: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        relation_counts = Counter()

        for subject, relation, obj in triples:
            relation_key = _clean_term(relation)
            if not relation_key or _is_low_quality_query_relation(relation_key):
                continue
            relation_counts[relation_key] += 1
            if len(relation_to_examples[relation_key]) < 5:
                relation_to_examples[relation_key].append((subject, obj))

        schemas = []
        for relation, support in relation_counts.most_common(max_relation_schemas):
            examples = relation_to_examples[relation]
            subject_types = Counter()
            object_types = Counter()
            for subject, obj in examples:
                subject_types.update(entity_types.get(subject, Counter({"Entity": 1})))
                object_types.update(entity_types.get(obj, Counter({"Entity": 1})))

            relation_tokens = _tokenize(relation)
            schemas.append({
                "relation": relation,
                "aliases": relation_tokens[:6],
                "support": support,
                "subject_types": [t for t, _ in subject_types.most_common(3)],
                "object_types": [t for t, _ in object_types.most_common(3)],
                "examples": [
                    {"subject": subject, "object": obj}
                    for subject, obj in examples[:3]
                ],
            })
        return schemas


class ConceptGroupingAgent:
    """Creates Saturn-style primitive concepts from graph entities and relation schemas."""

    CONCEPT_DESCRIPTIONS = {
        "Person": "People and biographical entities.",
        "Location": "Places, geopolitical entities, and physical locations.",
        "CreativeWork": "Films, books, songs, albums, and other works.",
        "Organization": "Teams, companies, schools, parties, and institutions.",
        "Event": "Awards, seasons, games, and historical events.",
        "Time": "Dates, years, and temporal expressions.",
        "Quantity": "Numbers, counts, rankings, and measurements.",
        "Entity": "General entities not assigned to a more specific concept.",
    }

    def build(
        self,
        entity_types: Dict[str, Counter],
        entity_relations: Dict[str, Counter],
        relation_schemas: List[Dict[str, Any]],
        max_entities_per_concept: int = 80,
    ) -> List[Dict[str, Any]]:
        concept_entities: Dict[str, Counter] = defaultdict(Counter)
        concept_relations: Dict[str, Counter] = defaultdict(Counter)

        for entity, type_counts in entity_types.items():
            for concept, count in type_counts.items():
                concept_entities[concept][entity] += count
                concept_relations[concept].update(entity_relations.get(entity, Counter()))

        relation_terms_by_concept: Dict[str, Counter] = defaultdict(Counter)
        for schema in relation_schemas:
            for concept in schema.get("subject_types", []) + schema.get("object_types", []):
                relation_terms_by_concept[concept].update(schema.get("aliases", []))

        concepts = []
        for concept, entities in sorted(concept_entities.items()):
            terms = Counter()
            terms.update(relation_terms_by_concept.get(concept, Counter()))
            for entity, _ in entities.most_common(30):
                terms.update(_tokenize(entity))

            concepts.append({
                "name": concept,
                "description": self.CONCEPT_DESCRIPTIONS.get(concept, f"{concept} graph concept."),
                "terms": [term for term, _ in terms.most_common(40)],
                "relations": [relation for relation, _ in concept_relations[concept].most_common(40)],
                "entities": [entity for entity, _ in entities.most_common(max_entities_per_concept)],
            })
        return concepts


class OntologySynthesisAgent:
    """Synthesizes agent outputs into a compact, reusable graph ontology."""

    def build(
        self,
        triples: List[Tuple[str, str, str]],
        concepts: List[Dict[str, Any]],
        relation_schemas: List[Dict[str, Any]],
        entity_types: Dict[str, Counter],
        entity_relations: Dict[str, Counter],
        entity_neighbors: Dict[str, Counter],
        max_entity_terms: int,
    ) -> Dict[str, Any]:
        entity_profiles = []
        entity_support = Counter()
        for subject, _, obj in triples:
            entity_support[subject] += 1
            entity_support[obj] += 1

        for entity, support in entity_support.most_common(max_entity_terms):
            entity_profiles.append({
                "entity": entity,
                "types": [name for name, _ in entity_types.get(entity, Counter({"Entity": 1})).most_common(3)],
                "relations": [name for name, _ in entity_relations.get(entity, Counter()).most_common(8)],
                "neighbors": [name for name, _ in entity_neighbors.get(entity, Counter()).most_common(8)],
                "support": support,
            })

        return {
            "version": 1,
            "builder": "saturn_inspired_graph_multi_agent",
            "stats": {
                "num_triples": len(triples),
                "num_concepts": len(concepts),
                "num_relation_schemas": len(relation_schemas),
                "num_entity_profiles": len(entity_profiles),
            },
            "concepts": concepts,
            "relation_schemas": relation_schemas,
            "entity_profiles": entity_profiles,
        }


class GraphOntologyBuilder:
    """Builds a lightweight graph ontology from CompassRAG OpenIE triples."""

    def __init__(self, max_entity_terms: int = 5000, max_relation_schemas: int = 300):
        self.max_entity_terms = max_entity_terms
        self.max_relation_schemas = max_relation_schemas
        self.entity_typing_agent = EntityTypingAgent()
        self.relation_schema_agent = RelationSchemaAgent()
        self.concept_grouping_agent = ConceptGroupingAgent()
        self.synthesis_agent = OntologySynthesisAgent()

    def build(self, openie_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        triples = self._collect_triples(openie_rows)
        entity_relations: Dict[str, Counter] = defaultdict(Counter)
        entity_neighbors: Dict[str, Counter] = defaultdict(Counter)

        for subject, relation, obj in triples:
            entity_relations[subject][_clean_term(relation)] += 1
            entity_relations[obj][_clean_term(relation)] += 1
            entity_neighbors[subject][obj] += 1
            entity_neighbors[obj][subject] += 1

        entity_types: Dict[str, Counter] = defaultdict(Counter)
        for entity, relations in entity_relations.items():
            for inferred_type in self.entity_typing_agent.infer(entity, relations.keys()):
                entity_types[entity][inferred_type] += 1

        relation_schemas = self.relation_schema_agent.build(
            triples=triples,
            entity_types=entity_types,
            max_relation_schemas=self.max_relation_schemas,
        )
        concepts = self.concept_grouping_agent.build(
            entity_types=entity_types,
            entity_relations=entity_relations,
            relation_schemas=relation_schemas,
        )
        ontology = self.synthesis_agent.build(
            triples=triples,
            concepts=concepts,
            relation_schemas=relation_schemas,
            entity_types=entity_types,
            entity_relations=entity_relations,
            entity_neighbors=entity_neighbors,
            max_entity_terms=self.max_entity_terms,
        )
        logger.info(
            "Built graph ontology with %d concepts, %d relation schemas, %d entity profiles",
            ontology["stats"]["num_concepts"],
            ontology["stats"]["num_relation_schemas"],
            ontology["stats"]["num_entity_profiles"],
        )
        return ontology

    def _collect_triples(self, openie_rows: List[Dict[str, Any]]) -> List[Tuple[str, str, str]]:
        triples = []
        seen = set()
        for row in openie_rows:
            for triple in row.get("extracted_triples", []):
                if len(triple) != 3:
                    continue
                subject, relation, obj = [_clean_term(value) for value in triple]
                if not subject or not relation or not obj:
                    continue
                key = (subject, relation, obj)
                if key in seen:
                    continue
                seen.add(key)
                triples.append(key)
        return triples

    @staticmethod
    def load(path: str) -> Dict[str, Any]:
        with open(path) as f:
            return json.load(f)

    @staticmethod
    def save(path: str, ontology: Dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(ontology, f, indent=2, sort_keys=True)


class GraphOntologyQueryExpander:
    """Generates Saturn-style ontology-aware query text for embedding retrieval."""

    def __init__(
        self,
        ontology: Dict[str, Any],
        query_top_k: int = 8,
        max_query_terms: int = 24,
        max_entities_per_query: int | None = None,
        max_relations_per_query: int | None = None,
        include_concepts: bool = True,
        include_related_terms: bool = True,
        min_concept_score: int = 2,
        require_entity_match: bool = False,
    ):
        self.ontology = ontology
        self.query_top_k = query_top_k
        self.max_query_terms = max_query_terms
        self.max_entities_per_query = max_entities_per_query or query_top_k
        self.max_relations_per_query = max_relations_per_query or query_top_k
        self.include_concepts = include_concepts
        self.include_related_terms = include_related_terms
        self.min_concept_score = min_concept_score
        self.require_entity_match = require_entity_match
        self.entity_profiles = ontology.get("entity_profiles", [])
        self.concepts = ontology.get("concepts", [])
        self.relation_schemas = ontology.get("relation_schemas", [])

    def augment_query(self, query: str) -> str:
        hints = self.analyze_query(query)
        if self.require_entity_match and not hints["entities"]:
            return query

        parts = []
        if self.include_concepts and hints["concepts"]:
            parts.append("concepts: " + ", ".join(hints["concepts"]))
        if hints["entities"]:
            parts.append("entities: " + ", ".join(hints["entities"]))
        if hints["relations"]:
            parts.append("relations: " + ", ".join(hints["relations"]))
        if self.include_related_terms and hints["related_terms"]:
            parts.append("related terms: " + ", ".join(hints["related_terms"]))

        if not parts:
            return query
        return f"{query}\nOntology hints: " + "; ".join(parts)

    def analyze_query(self, query: str) -> Dict[str, List[str]]:
        query_clean = _clean_term(query)
        query_tokens = set(_tokenize(query))

        matched_entities = self._match_entities(query_clean, query_tokens)
        matched_relations = self._match_relations(query_clean, query_tokens)
        matched_concepts = self._match_concepts(query_clean, query_tokens, matched_entities, matched_relations)
        related_terms = []
        if self.include_related_terms:
            related_terms = self._collect_related_terms(matched_entities, matched_relations, matched_concepts)

        return {
            "entities": matched_entities[:self.max_entities_per_query],
            "relations": matched_relations[:self.max_relations_per_query],
            "concepts": matched_concepts[:self.query_top_k] if self.include_concepts else [],
            "related_terms": related_terms[:self.max_query_terms],
        }

    def _match_entities(self, query_clean: str, query_tokens: set[str]) -> List[str]:
        scored = []
        for profile in self.entity_profiles:
            entity = profile.get("entity", "")
            if not entity:
                continue
            entity_tokens = set(_tokenize(entity))
            if _is_low_quality_query_entity(entity, entity_tokens):
                continue
            if len(entity_tokens) == 1:
                token = next(iter(entity_tokens))
                if token in GENERIC_ENTITY_TERMS or len(token) < 4:
                    continue
                if _contains_clean_phrase(query_clean, entity):
                    scored.append((profile.get("support", 0), entity))
                continue

            if entity and len(entity) > 3 and _contains_clean_phrase(query_clean, entity):
                scored.append((len(entity_tokens) + profile.get("support", 0), entity))
            elif _is_confident_fuzzy_entity_match(entity_tokens, query_tokens):
                overlap_count = len(entity_tokens & query_tokens)
                coverage = overlap_count / len(entity_tokens)
                scored.append((coverage + overlap_count, entity))
        return [entity for _, entity in sorted(scored, reverse=True)]

    def _match_relations(self, query_clean: str, query_tokens: set[str]) -> List[str]:
        scored = []
        for schema in self.relation_schemas:
            relation = schema.get("relation", "")
            if _is_low_quality_query_relation(relation):
                continue
            aliases = set(schema.get("aliases", []))
            relation_tokens = set(_tokenize(relation))
            if relation and relation in query_clean:
                scored.append((schema.get("support", 0), relation))
            else:
                candidate_tokens = aliases | relation_tokens
                matched_tokens = candidate_tokens & query_tokens
                if len(matched_tokens) >= 2:
                    scored.append((len(matched_tokens), relation))
                elif len(matched_tokens) == 1 and next(iter(matched_tokens)) in RELATION_TRIGGER_TERMS:
                    scored.append((1, relation))
        return [relation for _, relation in sorted(scored, reverse=True)]

    def _match_concepts(
        self,
        query_clean: str,
        query_tokens: set[str],
        matched_entities: List[str],
        matched_relations: List[str],
    ) -> List[str]:
        concept_scores = Counter()
        matched_entity_set = set(matched_entities)
        matched_relation_set = set(matched_relations)

        for concept in self.concepts:
            name = concept.get("name", "")
            terms = set(concept.get("terms", []))
            entities = set(concept.get("entities", []))
            relations = set(concept.get("relations", []))

            if _clean_term(name) in query_clean:
                concept_scores[name] += 3
            concept_scores[name] += len(terms & query_tokens)
            concept_scores[name] += 2 * len(entities & matched_entity_set)
            concept_scores[name] += len(relations & matched_relation_set)

        concepts = [name for name, score in concept_scores.most_common(self.query_top_k) if score >= self.min_concept_score]
        if "Entity" in concepts and len(concepts) > 1:
            concepts = [concept for concept in concepts if concept != "Entity"]
        return concepts

    def _collect_related_terms(
        self,
        matched_entities: List[str],
        matched_relations: List[str],
        matched_concepts: List[str],
    ) -> List[str]:
        terms = Counter()
        matched_entity_set = set(matched_entities)
        matched_relation_set = set(matched_relations)
        matched_concept_set = set(matched_concepts)

        for profile in self.entity_profiles:
            if profile.get("entity") in matched_entity_set:
                terms.update(profile.get("neighbors", []))
                terms.update(profile.get("relations", []))

        for schema in self.relation_schemas:
            if schema.get("relation") in matched_relation_set:
                terms.update(schema.get("subject_types", []))
                terms.update(schema.get("object_types", []))
                for example in schema.get("examples", []):
                    terms.update([example.get("subject", ""), example.get("object", "")])

        for concept in self.concepts:
            if concept.get("name") in matched_concept_set:
                terms.update(concept.get("terms", [])[:12])
                terms.update(concept.get("relations", [])[:8])

        return [
            term
            for term, _ in terms.most_common()
            if term and _tokenize(term) and term not in matched_entity_set
        ]
