import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple


_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does",
    "for", "from", "had", "has", "have", "he", "her", "his", "in", "is",
    "it", "its", "of", "on", "or", "she", "the", "their", "to", "was",
    "were", "what", "when", "where", "which", "who", "whom", "whose", "why",
    "with",
}

_RELATION_ALIASES = {
    "birth": {"birth", "born", "birthplace", "birth_place", "place_of_birth"},
    "death": {"death", "died", "dead", "deathplace", "death_date", "date_of_death"},
    "location": {"location", "located", "place", "part", "county", "city", "country", "region"},
    "occupation": {"occupation", "profession", "job", "career", "work"},
    "creator": {"creator", "created", "director", "directed", "writer", "written", "author"},
    "membership": {"member", "membership", "belong", "belongs", "part"},
    "family": {"father", "mother", "parent", "child", "spouse", "sibling", "wife", "husband"},
    "time": {"time", "date", "year", "when", "period"},
    "comparison": {"larger", "smaller", "higher", "lower", "older", "younger", "more", "less"},
}


def _norm(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().replace("_", " ").split())


def _tokens(value: Any) -> Set[str]:
    text = _norm(value)
    if not text:
        return set()
    raw = re.findall(r"[a-z0-9]+", text)
    result: Set[str] = set()
    for token in raw:
        if token in _STOPWORDS or len(token) <= 1:
            continue
        result.add(token)
        if token.endswith("s") and len(token) > 3:
            result.add(token[:-1])
        if token.endswith("ed") and len(token) > 4:
            result.add(token[:-2])
        if token.endswith("ing") and len(token) > 5:
            result.add(token[:-3])
    return result


def _flatten_terms(values: Iterable[Any]) -> List[str]:
    terms: List[str] = []
    for value in values or []:
        if isinstance(value, (list, tuple, set)):
            terms.extend(_flatten_terms(value))
            continue
        term = _norm(value)
        if term:
            terms.append(term)
    return terms


def _expanded_relation_tokens(terms: Iterable[Any]) -> Set[str]:
    tokens: Set[str] = set()
    normalized_terms = _flatten_terms(terms)
    for term in normalized_terms:
        term_tokens = _tokens(term)
        tokens.update(term_tokens)
        compact = term.replace(" ", "_")
        tokens.add(compact)
        for family, aliases in _RELATION_ALIASES.items():
            if family in term_tokens or family in term or term in aliases:
                for alias in aliases:
                    tokens.update(_tokens(alias))
                    tokens.add(alias.replace(" ", "_"))
    return tokens


def _answer_type_hit(answer_type: str, text: str) -> bool:
    answer_type = _norm(answer_type)
    text_norm = _norm(text)
    if not answer_type or not text_norm:
        return False
    if answer_type in {"date", "time", "year"}:
        return bool(re.search(r"\b(1[0-9]{3}|20[0-9]{2}|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", text_norm))
    if answer_type in {"number", "quantity", "count"}:
        return bool(re.search(r"\b[0-9]+(?:\.[0-9]+)?\b", text_norm))
    if answer_type in {"person", "location", "organization", "work", "creative work", "entity"}:
        return len(_tokens(text_norm)) > 0
    return answer_type in text_norm


@dataclass
class SchemaRoutePlan:
    anchor_terms: Set[str] = field(default_factory=set)
    query_terms: Set[str] = field(default_factory=set)
    relation_terms: Set[str] = field(default_factory=set)
    answer_type: str = ""
    matched_plan_ids: Set[str] = field(default_factory=set)
    relation_families: Set[str] = field(default_factory=set)
    subqueries: List[Dict[str, Any]] = field(default_factory=list)


class SchemaGuidedRouteAdapter:
    """Deterministic Youtu-style schema route adapter.

    Youtu-GraphRAG uses schema-aware decomposition and multiple retrieval
    routes. In this codebase we already have query-plan schema matches and
    ontology candidate contexts, so this adapter turns them into lightweight
    route coverage signals without adding online LLM calls.
    """

    def __init__(
        self,
        mode: str = "off",
        score_weight: float = 0.18,
        confidence_boost: float = 0.04,
        min_score: float = 0.30,
        reject_below: float = 0.0,
        max_subqueries: int = 6,
    ) -> None:
        self.mode = str(mode or "off").strip().lower()
        if self.mode not in {"off", "score", "filter", "selector"}:
            self.mode = "off"
        self.score_weight = max(float(score_weight), 0.0)
        self.confidence_boost = max(float(confidence_boost), 0.0)
        self.min_score = min(max(float(min_score), 0.0), 1.0)
        self.reject_below = min(max(float(reject_below), 0.0), 1.0)
        self.max_subqueries = max(int(max_subqueries), 1)

    def is_enabled(self) -> bool:
        return self.mode != "off"

    def build_plan(self, query_plan_log: Mapping[str, Any]) -> SchemaRoutePlan:
        condition = query_plan_log.get("query_condition", {}) or {}
        if not isinstance(condition, Mapping):
            condition = {}
        anchors = _flatten_terms(query_plan_log.get("anchor_entities", []) or [])
        query_terms = set()
        for key in ("query_tokens", "answer_hints", "relation_terms"):
            query_terms.update(_tokens(" ".join(_flatten_terms(condition.get(key, []) or []))))
        relation_families = set(_flatten_terms(condition.get("relation_families", []) or []))
        relation_signatures = _flatten_terms(condition.get("relation_signatures", []) or [])
        relation_terms = _expanded_relation_tokens(
            list(relation_families)
            + relation_signatures
            + _flatten_terms(condition.get("relation_terms", []) or [])
        )
        plan_ids: Set[str] = set()
        subqueries: List[Dict[str, Any]] = []
        for plan in query_plan_log.get("matched_plans", []) or []:
            if not isinstance(plan, Mapping):
                continue
            plan_id = _norm(plan.get("id"))
            if plan_id:
                plan_ids.add(plan_id)
            plan_relations = _flatten_terms(plan.get("relation_signatures", []) or []) + _flatten_terms(
                plan.get("relation_families", []) or []
            )
            plan_terms = sorted(_expanded_relation_tokens(plan_relations))
            if len(subqueries) < self.max_subqueries:
                subqueries.append({
                    "kind": "path" if len(plan_terms) > 1 else "triple",
                    "plan_id": plan_id,
                    "relations": plan_terms[:8],
                    "answer_type": _norm(plan.get("answer_type") or condition.get("answer_type")),
                })
        for anchor in anchors[: self.max_subqueries]:
            subqueries.append({
                "kind": "entity",
                "anchor": anchor,
                "relations": sorted(relation_terms)[:8],
                "answer_type": _norm(condition.get("answer_type")),
            })
            if len(subqueries) >= self.max_subqueries:
                break
        if not subqueries:
            subqueries.append({
                "kind": "community",
                "relations": sorted(relation_terms)[:8],
                "answer_type": _norm(condition.get("answer_type")),
            })
        return SchemaRoutePlan(
            anchor_terms=set(anchors),
            query_terms=query_terms,
            relation_terms=relation_terms,
            answer_type=_norm(condition.get("answer_type")),
            matched_plan_ids=plan_ids,
            relation_families=relation_families,
            subqueries=subqueries[: self.max_subqueries],
        )

    def _context_texts(self, contexts: Sequence[Mapping[str, Any]]) -> Tuple[str, str, str]:
        anchors: List[str] = []
        terminals: List[str] = []
        relations: List[str] = []
        for context in contexts or []:
            anchors.append(_norm(context.get("anchor_entity")))
            anchors.append(_norm(context.get("anchor")))
            anchors.append(_norm(context.get("subject")))
            terminals.append(_norm(context.get("end_entity")))
            terminals.append(_norm(context.get("terminal")))
            terminals.append(_norm(context.get("object")))
            relations.extend(_flatten_terms(context.get("relations", []) or []))
            relations.append(_norm(context.get("relation")))
            relations.extend(_flatten_terms(context.get("relation_path", []) or []))
        return " ".join(anchors), " ".join(terminals), " ".join(relations)

    def score_candidate(
        self,
        query_plan_log: Mapping[str, Any],
        candidate: Mapping[str, Any],
    ) -> Dict[str, Any]:
        plan = self.build_plan(query_plan_log)
        contexts = [
            context for context in (candidate.get("contexts", []) or [])
            if isinstance(context, Mapping)
        ]
        anchor_text, terminal_text, relation_text = self._context_texts(contexts)
        anchor_tokens = _tokens(anchor_text)
        terminal_tokens = _tokens(terminal_text)
        relation_tokens = _expanded_relation_tokens([relation_text])
        context_plan_ids = {
            _norm(context.get("plan_id"))
            for context in contexts
            if _norm(context.get("plan_id"))
        }
        entity_hit = bool(
            (plan.anchor_terms and any(_norm(anchor) and _norm(anchor) in anchor_text for anchor in plan.anchor_terms))
            or (plan.query_terms & (anchor_tokens | terminal_tokens))
        )
        relation_hit_count = len(plan.relation_terms & relation_tokens)
        relation_hit = relation_hit_count > 0
        plan_hit = bool(plan.matched_plan_ids & context_plan_ids)
        multihop_context = any(
            len(context.get("fact_indices", []) or []) >= 2
            or len(context.get("relations", []) or []) >= 2
            for context in contexts
        )
        triple_hit = bool(entity_hit and relation_hit)
        path_hit = bool((plan_hit or multihop_context) and (relation_hit or triple_hit))
        answer_hit = _answer_type_hit(plan.answer_type, terminal_text)
        community_hit = bool(not plan.anchor_terms and (relation_hit or answer_hit))
        covered_kinds = {
            kind for kind, hit in {
                "entity": entity_hit,
                "triple": triple_hit,
                "path": path_hit,
                "community": community_hit,
                "answer": answer_hit,
            }.items()
            if hit
        }
        subquery_hits = 0
        for subquery in plan.subqueries:
            kind = str(subquery.get("kind", "") or "")
            rels = set(subquery.get("relations", []) or [])
            rel_hit = bool(rels & relation_tokens)
            anchor = _norm(subquery.get("anchor"))
            sub_anchor_hit = bool(anchor and anchor in anchor_text)
            if kind == "entity" and (sub_anchor_hit or entity_hit):
                subquery_hits += 1
            elif kind == "triple" and ((sub_anchor_hit or entity_hit) and rel_hit):
                subquery_hits += 1
            elif kind == "path" and (plan_hit or (rel_hit and (multihop_context or entity_hit))):
                subquery_hits += 1
            elif kind == "community" and (rel_hit or community_hit):
                subquery_hits += 1
        route_score = 0.0
        route_score += 0.18 if entity_hit else 0.0
        route_score += 0.28 if triple_hit else 0.0
        route_score += 0.24 if path_hit else 0.0
        route_score += 0.10 if answer_hit else 0.0
        route_score += 0.06 if community_hit else 0.0
        route_score += min(relation_hit_count, 3) * 0.04
        route_score += min(subquery_hits, 3) * 0.06
        route_score = min(max(route_score, 0.0), 1.0)
        decision = "keep"
        if route_score >= self.min_score:
            decision = "promote"
        if self.mode in {"filter", "selector"} and route_score < self.reject_below:
            decision = "reject"
        return {
            "enabled": self.is_enabled(),
            "mode": self.mode,
            "score": round(route_score, 6),
            "decision": decision,
            "min_score": self.min_score,
            "reject_below": self.reject_below,
            "score_weight": self.score_weight,
            "confidence_boost": self.confidence_boost,
            "hits": {
                "entity": entity_hit,
                "triple": triple_hit,
                "path": path_hit,
                "community": community_hit,
                "answer": answer_hit,
                "plan": plan_hit,
            },
            "relation_hit_count": relation_hit_count,
            "subquery_hits": subquery_hits,
            "subquery_count": len(plan.subqueries),
            "covered_kinds": sorted(covered_kinds),
            "subqueries": plan.subqueries[:3],
        }
