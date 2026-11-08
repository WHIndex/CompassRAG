import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

from .query_ontology_prior import tokenize
from .query_plan_ontology import relation_signature
from .query_plan_structural_candidate_retriever import QueryPlanOntologyStructuralCandidateRetriever
from .utils.misc_utils import text_processing


@dataclass
class QueryPlanBindProbeJoinResult:
    added_candidates: List[Dict[str, Any]] = field(default_factory=list)
    log: Dict[str, Any] = field(default_factory=dict)


class QueryPlanOntologyBindProbeJoin:
    """Execute a bound second-hop access path from first-hop OpenIE evidence.

    Static ontology probes lose the join variable between reasoning steps. This
    operator first binds that variable from an actual triple in a high-ranked
    passage, then issues a dense probe containing the bound entity and the
    unresolved ontology relation. Candidates remain candidate-only so the CBO
    can compare their marginal benefit and displacement risk downstream.
    """

    _STOPWORDS = {
        "a", "an", "and", "are", "as", "at", "be", "been", "by", "did",
        "do", "does", "for", "from", "had", "has", "have", "how", "in",
        "is", "it", "of", "on", "or", "the", "to", "was", "were", "what",
        "when", "where", "which", "who", "whom", "why", "with",
    }

    _MONTH_NAMES = {
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december",
    }




    _TERMINAL_RELATION_SPECS = (
        {
            "key": "nationality",
            "cues": ("nationality", "citizenship", "national"),
            "aliases": ("nationality", "citizenship", "country of citizenship", "is a citizen of"),
            "answer_types": ("location", "country", "nationality"),
        },
        {
            "key": "occupation",
            "cues": ("occupation", "profession", "job"),
            "aliases": ("occupation", "profession", "worked as", "was a"),
            "answer_types": ("occupation", "profession", "string"),
        },
        {
            "key": "spouse",
            "cues": ("spouse", "wife", "husband", "married"),
            "aliases": ("spouse", "wife", "husband", "married to"),
            "answer_types": ("person", "entity"),
        },
        {
            "key": "mother",
            "cues": ("mother",),
            "aliases": ("mother", "parent"),
            "answer_types": ("person", "entity"),
        },
        {
            "key": "father",
            "cues": ("father",),
            "aliases": ("father", "parent"),
            "answer_types": ("person", "entity"),
        },
        {
            "key": "capital",
            "cues": ("capital",),
            "aliases": ("capital", "capital city", "has capital"),
            "answer_types": ("location", "city", "entity"),
        },
        {
            "key": "population",
            "cues": ("population", "inhabitants"),
            "aliases": ("population", "inhabitants", "population total"),
            "answer_types": ("number", "count", "quantity"),
        },
        {
            "key": "language",
            "cues": ("language", "languages"),
            "aliases": ("language", "official language", "spoken language"),
            "answer_types": ("language", "string", "entity"),
        },
        {
            "key": "genre",
            "cues": ("genre",),
            "aliases": ("genre", "type", "category"),
            "answer_types": ("genre", "string", "entity"),
        },
        {
            "key": "director",
            "cues": ("director", "directed"),
            "aliases": ("director", "directed by", "film director"),
            "answer_types": ("person", "entity"),
        },
        {
            "key": "author",
            "cues": ("author", "writer", "written", "wrote"),
            "aliases": ("author", "written by", "writer", "wrote"),
            "answer_types": ("person", "entity"),
        },
    )

    def __init__(
        self,
        anchor_top_k: int = 8,
        max_bindings: int = 4,
        max_probes: int = 8,
        dense_top_k: int = 64,
        max_candidates: int = 8,
        min_binding_score: float = 0.52,
        min_candidate_score: float = 0.62,
        min_relation_overlap: float = 0.20,
        synthetic_original_rank: int = 60,
        require_relation_or_answer: bool = True,
        enable_relation_role_constraints: bool = False,
        require_role_consistent_candidate: bool = False,
        min_terminal_relation_overlap: float = 0.40,
        min_terminal_type_score: float = 0.60,
        max_role_candidate_dense_rank: int = 0,
    ) -> None:
        self.anchor_top_k = max(int(anchor_top_k), 1)
        self.max_bindings = max(int(max_bindings), 1)
        self.max_probes = max(int(max_probes), 1)
        self.dense_top_k = max(int(dense_top_k), 1)
        self.max_candidates = max(int(max_candidates), 1)
        self.min_binding_score = min(max(float(min_binding_score), 0.0), 1.0)
        self.min_candidate_score = min(max(float(min_candidate_score), 0.0), 1.0)
        self.min_relation_overlap = min(max(float(min_relation_overlap), 0.0), 1.0)
        self.synthetic_original_rank = max(int(synthetic_original_rank), 1)
        self.require_relation_or_answer = bool(require_relation_or_answer)
        self.enable_relation_role_constraints = bool(enable_relation_role_constraints)
        self.require_role_consistent_candidate = bool(require_role_consistent_candidate)
        self.min_terminal_relation_overlap = min(max(float(min_terminal_relation_overlap), 0.0), 1.0)
        self.min_terminal_type_score = min(max(float(min_terminal_type_score), 0.0), 1.0)
        self.max_role_candidate_dense_rank = max(int(max_role_candidate_dense_rank), 0)

    @staticmethod
    def _norm(value: Any) -> str:
        return text_processing(str(value or "").replace("_", " "))

    @classmethod
    def _content_tokens(cls, value: Any, limit: int = 12) -> List[str]:
        output: List[str] = []
        seen = set()
        for token in tokenize(cls._norm(value)):
            if not token or token in cls._STOPWORDS or token in seen:
                continue
            output.append(token)
            seen.add(token)
            if len(output) >= limit:
                break
        return output

    @classmethod
    def _unique(cls, values: Iterable[Any], limit: int) -> List[str]:
        output: List[str] = []
        seen = set()
        for value in values or []:
            normalized = cls._norm(value)
            if not normalized or normalized in seen:
                continue
            output.append(normalized)
            seen.add(normalized)
            if len(output) >= limit:
                break
        return output

    @classmethod
    def _phrase_hit(cls, text: Any, phrase: Any, min_coverage: float = 0.72) -> bool:
        text_norm = cls._norm(text)
        phrase_norm = cls._norm(phrase)
        if not text_norm or not phrase_norm:
            return False
        if f" {phrase_norm} " in f" {text_norm} ":
            return True
        phrase_tokens = cls._content_tokens(phrase_norm, limit=10)
        if not phrase_tokens:
            return False
        text_tokens = set(tokenize(text_norm))
        matched = sum(1 for token in phrase_tokens if token in text_tokens)
        return matched / max(len(phrase_tokens), 1) >= min_coverage

    @classmethod
    def _token_overlap(cls, left: Any, right: Any) -> Tuple[float, List[str]]:
        left_tokens = cls._content_tokens(left, limit=12)
        right_tokens = set(cls._content_tokens(right, limit=24))
        if not left_tokens or not right_tokens:
            return 0.0, []
        hits = [token for token in left_tokens if token in right_tokens]
        return min(len(hits) / max(min(len(left_tokens), 4), 1), 1.0), hits

    @classmethod
    def _relation_overlap(cls, relation: Any, text: Any) -> Tuple[float, List[str]]:
        relation_text = cls._norm(relation)
        signature_text = relation_signature(relation_text).replace("_", " ")
        relation_terms = cls._unique([relation_text, signature_text], limit=2)
        best_score = 0.0
        best_hits: List[str] = []
        for relation_term in relation_terms:
            score, hits = cls._token_overlap(relation_term, text)
            if score > best_score:
                best_score = score
                best_hits = hits
        return best_score, best_hits

    @classmethod
    def _answer_hit(cls, text: Any, answer_hints: Sequence[str]) -> bool:
        text_norm = cls._norm(text)
        text_tokens = set(tokenize(text_norm))
        hints = cls._unique(answer_hints, limit=8)
        hint_tokens = {token for hint in hints for token in cls._content_tokens(hint, limit=3)}
        if hint_tokens.intersection(text_tokens):
            return True
        if any(hint in {"date", "time", "year", "when"} for hint in hints):
            return any(token.isdigit() and len(token) == 4 for token in text_tokens)
        if any(hint in {"number", "count", "quantity", "total"} for hint in hints):
            return any(token.isdigit() for token in text_tokens)
        return False

    @classmethod
    def _answer_type_family(cls, value: Any) -> str:
        value_norm = cls._norm(value)
        tokens = set(tokenize(value_norm))
        if tokens & {"date", "time", "year", "when", "day", "month"}:
            return "date"
        if tokens & {"number", "count", "quantity", "total", "amount", "population"}:
            return "number"
        if tokens & {"where", "location", "place", "country", "city", "state", "nationality"}:
            return "location"
        if tokens & {"who", "person", "people", "human"}:
            return "person"
        if tokens & {"boolean", "yes", "no"}:
            return "boolean"
        return value_norm or "unknown"

    @classmethod
    def _question_answer_type(cls, query: str, configured_type: Any) -> str:
        configured_family = cls._answer_type_family(configured_type)
        if configured_family not in {"unknown", "entity", "string", "answer"}:
            return configured_family
        query_norm = cls._norm(query)
        if query_norm.startswith("where "):
            return "location"
        if query_norm.startswith("when "):
            return "date"
        if query_norm.startswith("who ") or query_norm.startswith("whom "):
            return "person"
        if query_norm.startswith("how many ") or query_norm.startswith("how much "):
            return "number"
        padded_query = f" {query_norm} "
        if any(cue in padded_query for cue in (" date ", " year ", " birthday ")):
            return "date"
        if any(
            cue in padded_query
            for cue in (" place ", " location ", " country ", " city ", " nationality ")
        ):
            return "location"
        return configured_family

    @classmethod
    def _cue_position(cls, query_norm: str, cues: Sequence[str]) -> int:
        positions = []
        padded_query = f" {query_norm} "
        for cue in cues:
            cue_norm = cls._norm(cue)
            position = padded_query.find(f" {cue_norm} ")
            if position >= 0:
                positions.append(position)
        return min(positions, default=-1)

    @classmethod
    def _terminal_relation_profile(cls, query: str, plan: Mapping[str, Any]) -> Dict[str, Any]:
        query_norm = cls._norm(query)
        answer_type = cls._question_answer_type(query, plan.get("answer_type", ""))
        if "how old" in query_norm or " age " in f" {query_norm} ":
            return {
                "resolved": False,
                "source": "query_surface",
                "reason": "derived_answer_requires_arithmetic",
                "answer_type": answer_type,
                "canonical_relation": "",
                "aliases": [],
            }

        location_intent = answer_type == "location" or any(
            cue in f" {query_norm} " for cue in (" where ", " place ", " location ", " country ", " city ")
        )
        date_intent = answer_type == "date" or any(
            cue in f" {query_norm} " for cue in (" when ", " date ", " year ")
        )
        event_specs: List[Dict[str, Any]] = []
        for event_key, event_cues, date_aliases, place_aliases in (
            (
                "death",
                ("die", "died", "death", "deceased"),
                ("died", "died on", "date of death", "death date", "death"),
                ("died in", "place of death", "death place", "death"),
            ),
            (
                "birth",
                ("born", "birth"),
                ("born", "born on", "date of birth", "birth date"),
                ("born in", "place of birth", "birth place", "born"),
            ),
            (
                "release",
                ("released", "release", "published", "publication", "premiered", "premiere", "aired"),
                ("release date", "released on", "publication date", "published on", "premiered", "aired"),
                (),
            ),
            (
                "founding",
                ("founded", "established", "formed", "created"),
                ("founded", "established", "formation date", "inception"),
                (),
            ),
        ):
            position = cls._cue_position(query_norm, event_cues)
            if position < 0:
                continue
            if location_intent and place_aliases:
                event_specs.append({
                    "key": f"{event_key}_place",
                    "canonical_relation": place_aliases[0],
                    "aliases": list(place_aliases),
                    "answer_types": ["location"],
                    "position": position,
                    "intent_match": True,
                })
            elif date_intent and date_aliases:
                event_specs.append({
                    "key": f"{event_key}_date",
                    "canonical_relation": date_aliases[0],
                    "aliases": list(date_aliases),
                    "answer_types": ["date"],
                    "position": position,
                    "intent_match": True,
                })

        candidates: List[Dict[str, Any]] = list(event_specs)
        country_from = re.search(
            r"\b(?:which|what)\s+country\b.*\bfrom\b|\bfrom\b.*\b(?:which|what)\s+country\b",
            query_norm,
        )
        if country_from:
            candidates.append({
                "key": "country_of_citizenship",
                "canonical_relation": "country of citizenship",
                "aliases": [
                    "country of citizenship", "citizenship", "nationality", "country",
                ],
                "answer_types": ["location", "country", "nationality"],
                "position": int(country_from.start()),
                "intent_match": True,
            })
        for spec in cls._TERMINAL_RELATION_SPECS:
            position = cls._cue_position(query_norm, spec["cues"])
            if position < 0:
                continue
            candidates.append({
                "key": spec["key"],
                "canonical_relation": spec["aliases"][0],
                "aliases": list(spec["aliases"]),
                "answer_types": list(spec["answer_types"]),
                "position": position,
                "intent_match": False,
            })

        if not candidates:
            return {
                "resolved": False,
                "source": "query_surface",
                "reason": "no_typed_terminal_relation",
                "answer_type": answer_type,
                "canonical_relation": "",
                "aliases": [],
            }

        query_length = max(len(query_norm), 1)
        for candidate in candidates:
            expected_families = {
                cls._answer_type_family(value) for value in candidate.get("answer_types", [])
            }
            type_match = answer_type in expected_families or answer_type in {
                "unknown", "entity", "string", "answer",
            }
            candidate["type_match"] = bool(type_match)
            candidate["role_score"] = round(
                (2.0 if type_match else 0.0)
                + (1.0 if candidate.get("intent_match") else 0.0)
                + max(0.0, 1.0 - float(candidate.get("position", query_length)) / query_length),
                6,
            )
        best = max(
            candidates,
            key=lambda item: (float(item.get("role_score", 0.0)), -int(item.get("position", query_length))),
        )
        return {
            "resolved": bool(best.get("type_match", False)),
            "source": "query_surface_typed_relation",
            "reason": "resolved" if best.get("type_match", False) else "terminal_answer_type_mismatch",
            "answer_type": answer_type,
            "key": best.get("key", ""),
            "canonical_relation": best.get("canonical_relation", ""),
            "aliases": list(best.get("aliases", []) or []),
            "expected_answer_types": list(best.get("answer_types", []) or []),
            "query_position": int(best.get("position", -1)),
            "role_score": float(best.get("role_score", 0.0)),
        }

    @classmethod
    def _terminal_type_score(cls, value: Any, answer_type: Any) -> float:
        value_norm = cls._norm(value)
        if not value_norm:
            return 0.0
        family = cls._answer_type_family(answer_type)
        tokens = set(tokenize(value_norm))
        has_year = bool(re.search(r"\b\d{3,4}\b", value_norm))
        has_month = bool(tokens & cls._MONTH_NAMES)
        has_number = bool(re.search(r"\d", value_norm))
        if family == "date":
            if has_year or has_month:
                return 1.0
            return 0.45 if has_number else 0.0
        if family == "number":
            return 1.0 if has_number else 0.0
        if family == "boolean":
            return 1.0 if value_norm in {"yes", "no", "true", "false"} else 0.0
        if family in {"location", "person"}:
            if has_year or (has_number and len(tokens) <= 3):
                return 0.10
            return 0.82
        return 0.70

    @classmethod
    def _terminal_predicate_overlap(
        cls,
        terminal_profile: Mapping[str, Any],
        predicate: Any,
    ) -> Tuple[float, List[str], str]:
        best_score = 0.0
        best_hits: List[str] = []
        best_alias = ""
        aliases = list(terminal_profile.get("aliases", []) or [])
        canonical = str(terminal_profile.get("canonical_relation", "") or "")
        if canonical:
            aliases = [canonical] + aliases
        for alias in cls._unique(aliases, limit=12):
            score, hits = cls._relation_overlap(alias, predicate)
            if score > best_score:
                best_score = score
                best_hits = hits
                best_alias = alias
        return best_score, best_hits, best_alias

    @classmethod
    def _valid_binding(cls, entity: Any, query: str, anchors: Sequence[str]) -> bool:
        entity_norm = cls._norm(entity)
        if not entity_norm or len(entity_norm) < 3 or len(entity_norm) > 96:
            return False
        if QueryPlanOntologyStructuralCandidateRetriever.is_generic_anchor(entity_norm):
            return False
        tokens = cls._content_tokens(entity_norm, limit=12)
        if not tokens or all(token.isdigit() for token in tokens):
            return False
        if cls._phrase_hit(query, entity_norm, min_coverage=0.84):
            return False
        if any(cls._phrase_hit(anchor, entity_norm, min_coverage=0.84) for anchor in anchors):
            return False
        return True

    @classmethod
    def _plan_features(cls, query_plan_log: Mapping[str, Any]) -> Dict[str, Any]:
        condition = query_plan_log.get("query_condition", {}) or {}
        if not isinstance(condition, Mapping):
            condition = {}
        anchors: List[Any] = list(query_plan_log.get("anchor_entities", []) or [])
        ordered_relations: List[Any] = []
        fallback_relations: List[Any] = []
        plan_ids: List[Any] = []
        matched_answer_types: List[Any] = []

        for plan in query_plan_log.get("matched_plans", []) or []:
            if not isinstance(plan, Mapping):
                continue
            plan_ids.append(plan.get("id", ""))
            matched_answer_types.append(plan.get("answer_type", ""))
            path_relations: List[Any] = []
            for step in plan.get("path", []) or []:
                if isinstance(step, Mapping):
                    path_relations.append(
                        step.get("relation")
                        or step.get("relation_signature")
                        or step.get("relation_family")
                        or ""
                    )
                else:
                    path_relations.append(step)
            if len(path_relations) >= 2 and not ordered_relations:
                ordered_relations.extend(path_relations)
            fallback_relations.extend(plan.get("relation_signatures", []) or [])
            fallback_relations.extend(plan.get("relation_families", []) or [])

        for path in query_plan_log.get("sample_paths", []) or []:
            if not isinstance(path, Mapping):
                continue
            anchors.append(path.get("anchor_entity", ""))
            path_relations = list(path.get("relations", []) or [])
            if len(path_relations) >= 2 and not ordered_relations:
                ordered_relations.extend(path_relations)
            fallback_relations.extend(path_relations)




        condition_relations = (
            list(condition.get("relation_terms", []) or [])
            + list(condition.get("relation_signatures", []) or [])
            + list(condition.get("relation_families", []) or [])
        )
        fallback_relations = condition_relations + fallback_relations
        answer_type = cls._norm(condition.get("answer_type", ""))
        if not answer_type:
            answer_type = next(
                (value for value in cls._unique(matched_answer_types, limit=4) if value),
                "",
            )
        answer_hints = cls._unique(
            list(condition.get("answer_hints", []) or []) + ([answer_type] if answer_type else []),
            limit=8,
        )
        ordered = cls._unique(ordered_relations, limit=6)
        fallback = cls._unique(fallback_relations, limit=10)
        second_hop_relations = ordered[1:] if len(ordered) >= 2 else fallback
        first_hop_relations = ordered[:1]
        return {
            "anchors": [
                anchor for anchor in cls._unique(anchors, limit=12)
                if not QueryPlanOntologyStructuralCandidateRetriever.is_generic_anchor(anchor)
            ],
            "first_hop_relations": first_hop_relations,
            "second_hop_relations": cls._unique(second_hop_relations, limit=6),
            "answer_type": answer_type,
            "answer_hints": answer_hints,
            "plan_ids": cls._unique(plan_ids, limit=6),
            "ordered_relation_count": len(ordered),
            "fallback_relations": fallback,
        }

    def _apply_relation_roles(self, query: str, plan: Mapping[str, Any]) -> Dict[str, Any]:
        resolved_plan = dict(plan)
        terminal_profile = self._terminal_relation_profile(query, plan)
        resolved_plan["terminal_relation_profile"] = terminal_profile
        if not terminal_profile.get("resolved", False):
            resolved_plan["relation_role_resolved"] = False
            return resolved_plan

        terminal_aliases = list(terminal_profile.get("aliases", []) or [])
        first_relations = list(plan.get("first_hop_relations", []) or [])
        for relation in plan.get("fallback_relations", []) or []:
            terminal_overlap = max(
                (self._relation_overlap(alias, relation)[0] for alias in terminal_aliases),
                default=0.0,
            )
            if terminal_overlap < self.min_terminal_relation_overlap:
                first_relations.append(relation)
        resolved_plan["first_hop_relations"] = self._unique(first_relations, limit=4)
        resolved_plan["second_hop_relations"] = [str(terminal_profile.get("canonical_relation", "") or "")]
        resolved_plan["relation_role_resolved"] = True
        return resolved_plan

    def _extract_bindings(
        self,
        query: str,
        base_doc_ids: Sequence[int],
        plan: Mapping[str, Any],
        get_openie_record: Callable[[int], Mapping[str, Any]],
        get_passage_text: Callable[[int], str],
        rejections: Dict[str, int],
        terminal_profile: Mapping[str, Any],
    ) -> List[Dict[str, Any]]:
        anchors = list(plan.get("anchors", []) or [])
        first_relations = list(plan.get("first_hop_relations", []) or [])
        query_and_anchors = " ".join([query] + anchors)
        binding_by_entity: Dict[str, Dict[str, Any]] = {}

        for source_rank, passage_idx in enumerate(base_doc_ids[: self.anchor_top_k], start=1):
            record = get_openie_record(int(passage_idx)) or {}
            triples = list(record.get("extracted_triples", []) or [])
            if not triples:
                rejections["source_without_openie_triples"] = rejections.get("source_without_openie_triples", 0) + 1
                continue
            passage_text = str(record.get("passage") or get_passage_text(int(passage_idx)) or "")
            passage_query_overlap, _ = self._token_overlap(query, passage_text)
            for triple_index, triple in enumerate(triples):
                if not isinstance(triple, (list, tuple)) or len(triple) != 3:
                    rejections["malformed_binding_triple"] = rejections.get("malformed_binding_triple", 0) + 1
                    continue
                subject, predicate, obj = [str(value or "") for value in triple]
                if self.enable_relation_role_constraints and terminal_profile.get("resolved", False):
                    terminal_predicate_score, _, _ = self._terminal_predicate_overlap(
                        terminal_profile,
                        predicate,
                    )
                    if terminal_predicate_score >= self.min_terminal_relation_overlap:
                        rejections["terminal_predicate_cannot_bind_bridge"] = rejections.get(
                            "terminal_predicate_cannot_bind_bridge", 0
                        ) + 1
                        continue
                subject_known = self._phrase_hit(query_and_anchors, subject, min_coverage=0.76)
                object_known = self._phrase_hit(query_and_anchors, obj, min_coverage=0.76)
                if subject_known == object_known:
                    rejections["binding_triple_without_single_known_side"] = rejections.get(
                        "binding_triple_without_single_known_side", 0
                    ) + 1
                    continue
                binding = obj if subject_known else subject
                known_endpoint = subject if subject_known else obj
                if not self._valid_binding(binding, query, anchors):
                    rejections["invalid_binding_entity"] = rejections.get("invalid_binding_entity", 0) + 1
                    continue

                predicate_query_overlap, predicate_query_hits = self._relation_overlap(predicate, query)
                predicate_plan_overlap = 0.0
                predicate_plan_hits: List[str] = []
                for first_relation in first_relations:
                    score, hits = self._relation_overlap(first_relation, predicate)
                    if score > predicate_plan_overlap:
                        predicate_plan_overlap = score
                        predicate_plan_hits = hits
                entity_specificity = min(len(self._content_tokens(binding, limit=8)) / 3.0, 1.0)
                binding_score = (
                    0.34
                    + 0.18 * predicate_query_overlap
                    + 0.12 * predicate_plan_overlap
                    + 0.12 * (1.0 / max(source_rank, 1))
                    + 0.10 * passage_query_overlap
                    + 0.08 * entity_specificity
                    + 0.06 * (1.0 if anchors and any(self._phrase_hit(known_endpoint, a) for a in anchors) else 0.0)
                )
                binding_score = min(max(float(binding_score), 0.0), 1.0)
                if binding_score < self.min_binding_score:
                    rejections["low_binding_score"] = rejections.get("low_binding_score", 0) + 1
                    continue
                entity_key = self._norm(binding)
                item = {
                    "binding_entity": binding,
                    "binding_key": entity_key,
                    "binding_score": round(binding_score, 6),
                    "source_passage_idx": int(passage_idx),
                    "source_rank": int(source_rank),
                    "source_triple_index": int(triple_index),
                    "source_triple": [subject, predicate, obj],
                    "known_endpoint": known_endpoint,
                    "first_hop_predicate": predicate,
                    "predicate_query_overlap": round(predicate_query_overlap, 6),
                    "predicate_query_hits": predicate_query_hits,
                    "predicate_plan_overlap": round(predicate_plan_overlap, 6),
                    "predicate_plan_hits": predicate_plan_hits,
                }
                previous = binding_by_entity.get(entity_key)
                if previous is None or binding_score > float(previous.get("binding_score", 0.0)):
                    binding_by_entity[entity_key] = item

        bindings = sorted(
            binding_by_entity.values(),
            key=lambda item: (
                float(item.get("binding_score", 0.0)),
                -int(item.get("source_rank", 10**9)),
            ),
            reverse=True,
        )
        return bindings[: self.max_bindings]

    def run(
        self,
        query: str,
        base_doc_ids: Sequence[int],
        query_plan_log: Mapping[str, Any],
        dense_retrieve: Callable[[str], Tuple[Sequence[int], Sequence[float]]],
        get_passage_text: Callable[[int], str],
        get_openie_record: Callable[[int], Mapping[str, Any]],
    ) -> QueryPlanBindProbeJoinResult:
        plan = self._plan_features(query_plan_log)
        if self.enable_relation_role_constraints:
            plan = self._apply_relation_roles(query, plan)
        terminal_profile = dict(plan.get("terminal_relation_profile", {}) or {})
        rejections: Dict[str, int] = {}
        log: Dict[str, Any] = {
            "enabled": True,
            "operator": "ontology_bind_probe_join",
            "anchor_top_k": self.anchor_top_k,
            "max_bindings": self.max_bindings,
            "max_probes": self.max_probes,
            "dense_top_k": self.dense_top_k,
            "max_candidates": self.max_candidates,
            "min_binding_score": self.min_binding_score,
            "min_candidate_score": self.min_candidate_score,
            "min_relation_overlap": self.min_relation_overlap,
            "synthetic_original_rank": self.synthetic_original_rank,
            "relation_role_constraints_enabled": self.enable_relation_role_constraints,
            "require_role_consistent_candidate": self.require_role_consistent_candidate,
            "min_terminal_relation_overlap": self.min_terminal_relation_overlap,
            "min_terminal_type_score": self.min_terminal_type_score,
            "max_role_candidate_dense_rank": self.max_role_candidate_dense_rank,
            "relation_role_resolved": bool(plan.get("relation_role_resolved", False)),
            "terminal_relation_profile": terminal_profile,
            "plan": dict(plan),
            "bindings": [],
            "probes": [],
            "added_candidates": [],
            "binding_count": 0,
            "probe_count": 0,
            "candidate_count": 0,
            "selected_count": 0,
            "rejection_reasons": {},
            "skip_reason": "",
        }
        if not query_plan_log or not query_plan_log.get("enabled", False):
            log["skip_reason"] = "no_query_plan"
            return QueryPlanBindProbeJoinResult(log=log)
        if (
            self.enable_relation_role_constraints
            and self.require_role_consistent_candidate
            and not plan.get("relation_role_resolved", False)
        ):
            log["skip_reason"] = str(terminal_profile.get("reason") or "terminal_relation_unresolved")
            return QueryPlanBindProbeJoinResult(log=log)
        if not plan.get("second_hop_relations") and not plan.get("answer_hints"):
            log["skip_reason"] = "no_unresolved_second_hop"
            return QueryPlanBindProbeJoinResult(log=log)

        bindings = self._extract_bindings(
            query=query,
            base_doc_ids=base_doc_ids,
            plan=plan,
            get_openie_record=get_openie_record,
            get_passage_text=get_passage_text,
            rejections=rejections,
            terminal_profile=terminal_profile,
        )
        log["bindings"] = bindings
        log["binding_count"] = len(bindings)
        if not bindings:
            log["skip_reason"] = "no_grounded_bridge_binding"
            log["rejection_reasons"] = dict(sorted(rejections.items(), key=lambda row: row[1], reverse=True)[:16])
            return QueryPlanBindProbeJoinResult(log=log)

        relations = list(plan.get("second_hop_relations", []) or [])
        answer_hints = list(plan.get("answer_hints", []) or [])
        answer_type = str(plan.get("answer_type", "") or "")
        original_rank_by_doc = {int(doc_id): rank + 1 for rank, doc_id in enumerate(base_doc_ids)}
        selected_by_passage: Dict[int, Dict[str, Any]] = {}
        probes: List[Dict[str, Any]] = []

        for binding in bindings:
            binding_entity = str(binding.get("binding_entity", "") or "")
            relation_pool = relations[:2] or [""]
            for relation in relation_pool:
                probe_query = " ".join(
                    piece for piece in [query, binding_entity, relation, " ".join(answer_hints[:2])]
                    if piece
                )
                probe_key = self._norm(probe_query)
                if not probe_key or any(item.get("probe_key") == probe_key for item in probes):
                    continue
                action_id = (
                    f"bind_probe::{binding.get('source_passage_idx')}::"
                    f"{binding.get('source_triple_index')}::{len(probes)}"
                )
                probe = {
                    "action_id": action_id,
                    "probe_key": probe_key,
                    "query": probe_query,
                    "binding_entity": binding_entity,
                    "second_hop_relation": relation,
                    "terminal_relation_profile": terminal_profile,
                    "answer_type": answer_type,
                    "answer_hints": answer_hints,
                    "binding_source_passage_idx": binding.get("source_passage_idx"),
                    "binding_source_rank": binding.get("source_rank"),
                    "binding_score": binding.get("binding_score"),
                    "estimated_cost": round(1.0 + self.dense_top_k / 100.0, 6),
                }
                probes.append(probe)
                if len(probes) >= self.max_probes:
                    break
            if len(probes) >= self.max_probes:
                break

        log["probes"] = probes
        log["probe_count"] = len(probes)
        binding_by_action = {
            str(probe.get("action_id")): next(
                (
                    binding for binding in bindings
                    if int(binding.get("source_passage_idx", -1))
                    == int(probe.get("binding_source_passage_idx", -2))
                    and self._norm(binding.get("binding_entity"))
                    == self._norm(probe.get("binding_entity"))
                ),
                {},
            )
            for probe in probes
        }

        for probe in probes:
            try:
                dense_doc_ids, dense_scores = dense_retrieve(str(probe.get("query", "")))
            except Exception as exc:
                rejections["dense_retrieve_error"] = rejections.get("dense_retrieve_error", 0) + 1
                log["last_dense_retrieve_error"] = str(exc)[:200]
                continue
            dense_doc_ids = list(dense_doc_ids[: self.dense_top_k])
            dense_scores = list(dense_scores[: self.dense_top_k])
            max_dense_score = max((float(score) for score in dense_scores), default=0.0)
            binding = binding_by_action.get(str(probe.get("action_id")), {}) or {}
            binding_entity = str(probe.get("binding_entity", "") or "")
            relation = str(probe.get("second_hop_relation", "") or "")

            for dense_rank, passage_idx_value in enumerate(dense_doc_ids, start=1):
                if (
                    self.enable_relation_role_constraints
                    and self.max_role_candidate_dense_rank > 0
                    and dense_rank > self.max_role_candidate_dense_rank
                ):
                    rejections["outside_role_dense_access_path"] = rejections.get(
                        "outside_role_dense_access_path", 0
                    ) + len(dense_doc_ids) - dense_rank + 1
                    break
                passage_idx = int(passage_idx_value)
                if passage_idx == int(binding.get("source_passage_idx", -1)):
                    rejections["same_as_binding_source"] = rejections.get("same_as_binding_source", 0) + 1
                    continue
                passage_text = get_passage_text(passage_idx)
                if not self._phrase_hit(passage_text, binding_entity, min_coverage=0.76):
                    rejections["second_hop_binding_miss"] = rejections.get("second_hop_binding_miss", 0) + 1
                    continue

                record = get_openie_record(passage_idx) or {}
                candidate_triples: List[List[str]] = []
                candidate_predicates: List[str] = []
                terminal_entities: List[str] = []
                terminal_triple_profiles: List[Dict[str, Any]] = []
                for triple in record.get("extracted_triples", []) or []:
                    if not isinstance(triple, (list, tuple)) or len(triple) != 3:
                        continue
                    subject, predicate, obj = [str(value or "") for value in triple]
                    if self._phrase_hit(subject, binding_entity, min_coverage=0.76):
                        candidate_triples.append([subject, predicate, obj])
                        candidate_predicates.append(predicate)
                        terminal_entities.append(obj)
                        terminal_entity = obj
                    elif self._phrase_hit(obj, binding_entity, min_coverage=0.76):
                        candidate_triples.append([subject, predicate, obj])
                        candidate_predicates.append(predicate)
                        terminal_entities.append(subject)
                        terminal_entity = subject
                    else:
                        continue
                    terminal_relation_score, terminal_relation_hits, matched_alias = (
                        self._terminal_predicate_overlap(terminal_profile, predicate)
                        if terminal_profile.get("resolved", False)
                        else (0.0, [], "")
                    )
                    terminal_type_score = self._terminal_type_score(
                        terminal_entity,
                        terminal_profile.get("answer_type", answer_type),
                    )
                    terminal_triple_profiles.append({
                        "triple": [subject, predicate, obj],
                        "terminal_entity": terminal_entity,
                        "predicate": predicate,
                        "matched_alias": matched_alias,
                        "relation_score": round(float(terminal_relation_score), 6),
                        "relation_hits": terminal_relation_hits,
                        "type_score": round(float(terminal_type_score), 6),
                        "role_score": round(
                            0.72 * float(terminal_relation_score) + 0.28 * float(terminal_type_score),
                            6,
                        ),
                    })

                best_terminal = max(
                    terminal_triple_profiles,
                    key=lambda item: (
                        float(item.get("role_score", 0.0)),
                        float(item.get("relation_score", 0.0)),
                        float(item.get("type_score", 0.0)),
                    ),
                    default={},
                )
                terminal_relation_score = float(best_terminal.get("relation_score", 0.0) or 0.0)
                terminal_type_score = float(best_terminal.get("type_score", 0.0) or 0.0)
                role_consistent = bool(
                    terminal_profile.get("resolved", False)
                    and terminal_relation_score >= self.min_terminal_relation_overlap
                    and terminal_type_score >= self.min_terminal_type_score
                )

                relation_score, relation_hits = self._relation_overlap(relation, passage_text)
                for predicate in candidate_predicates:
                    score, hits = self._relation_overlap(relation, predicate)
                    if score > relation_score:
                        relation_score = score
                        relation_hits = hits
                if self.enable_relation_role_constraints and terminal_profile.get("resolved", False):
                    relation_score = terminal_relation_score
                    relation_hits = list(best_terminal.get("relation_hits", []) or [])
                answer_match = self._answer_hit(passage_text, answer_hints)
                has_second_hop_fact = bool(candidate_triples)
                if self.require_role_consistent_candidate and not role_consistent:
                    reason = "terminal_relation_miss"
                    if terminal_relation_score >= self.min_terminal_relation_overlap:
                        reason = "terminal_answer_type_miss"
                    rejections[reason] = rejections.get(reason, 0) + 1
                    continue
                if self.require_relation_or_answer and not (
                    relation_score >= self.min_relation_overlap or answer_match
                ):
                    rejections["second_hop_relation_answer_miss"] = rejections.get(
                        "second_hop_relation_answer_miss", 0
                    ) + 1
                    continue

                dense_raw = float(dense_scores[dense_rank - 1]) if dense_rank - 1 < len(dense_scores) else 0.0
                dense_norm = dense_raw / max_dense_score if max_dense_score > 0.0 else 1.0 / max(dense_rank, 1)
                dense_norm = min(max(float(dense_norm), 0.0), 1.0)
                binding_score = min(max(float(binding.get("binding_score", 0.0)), 0.0), 1.0)
                if self.enable_relation_role_constraints:
                    candidate_score = (
                        0.23 * dense_norm
                        + 0.12
                        + 0.25 * terminal_relation_score
                        + 0.14 * terminal_type_score
                        + 0.14 * binding_score
                        + 0.07 * (1.0 if has_second_hop_fact else 0.0)
                        + 0.05 * (1.0 / max(dense_rank, 1))
                    )
                else:
                    candidate_score = (
                        0.28 * dense_norm
                        + 0.20
                        + 0.16 * min(relation_score, 1.0)
                        + 0.12 * (1.0 if answer_match else 0.0)
                        + 0.12 * binding_score
                        + 0.07 * (1.0 if has_second_hop_fact else 0.0)
                        + 0.05 * (1.0 / max(dense_rank, 1))
                    )
                candidate_score = min(max(float(candidate_score), 0.0), 1.0)
                if candidate_score < self.min_candidate_score:
                    rejections["low_candidate_score"] = rejections.get("low_candidate_score", 0) + 1
                    continue

                terminal = str(best_terminal.get("terminal_entity", "") or "")
                if not terminal:
                    terminal = next(
                        (
                            entity for entity in terminal_entities
                            if self._valid_binding(entity, query, list(plan.get("anchors", []) or []))
                            and self._norm(entity) != self._norm(binding_entity)
                        ),
                        "",
                    )
                original_rank = int(original_rank_by_doc.get(passage_idx, self.synthetic_original_rank + dense_rank))
                original_rank = min(original_rank, self.synthetic_original_rank + dense_rank)
                context = {
                    "ontology_bind_probe_join": True,
                    "bind_probe_join": True,
                    "bind_probe_role_constrained": self.enable_relation_role_constraints,
                    "bind_probe_role_consistent": role_consistent,
                    "schema_route_source": "ontology_bind_probe_join",
                    "support_key": f"{probe.get('action_id')}::{passage_idx}",
                    "plan_id": (plan.get("plan_ids") or ["ontology_bind_probe_join"])[0],
                    "anchor_entity": binding_entity,
                    "end_entity": terminal,
                    "bridge_entity": binding_entity,
                    "relations": [relation] if relation else list(candidate_predicates[:1]),
                    "answer_type": answer_type,
                    "fact_indices": [],
                    "fact_position": 1,
                    "path_length": 2,
                    "signature_match_mode": "strict",
                    "plan_effective_precision": candidate_score,
                    "plan_lift_weight_scale": 1.0,
                    "admission_min_confidence": max(0.62, self.min_candidate_score),
                    "binding_source_passage_idx": int(binding.get("source_passage_idx", -1)),
                    "binding_source_rank": int(binding.get("source_rank", 0)),
                    "binding_source_triple": list(binding.get("source_triple", []) or []),
                    "binding_score": binding_score,
                    "probe_query": probe.get("query", ""),
                    "probe_action_id": probe.get("action_id", ""),
                    "probe_estimated_cost": probe.get("estimated_cost", 0.0),
                    "second_hop_relation": relation,
                    "second_hop_relation_overlap": relation_score,
                    "second_hop_relation_hits": relation_hits,
                    "terminal_relation_key": terminal_profile.get("key", ""),
                    "terminal_relation_source": terminal_profile.get("source", ""),
                    "terminal_relation_alias": best_terminal.get("matched_alias", ""),
                    "terminal_relation_overlap": terminal_relation_score,
                    "terminal_answer_type_score": terminal_type_score,
                    "terminal_triple": list(best_terminal.get("triple", []) or []),
                    "second_hop_answer_hit": bool(answer_match),
                    "second_hop_openie_hit": bool(has_second_hop_fact),
                    "role_consistent": role_consistent,
                    "terminal_relation_score": round(float(terminal_relation_score), 6),
                    "terminal_type_score": round(float(terminal_type_score), 6),
                    "terminal_triple": list(best_terminal.get("triple", []) or []),
                    "second_hop_triples": candidate_triples[:3],
                    "score": candidate_score,
                }
                candidate = {
                    "passage_idx": passage_idx,
                    "confidence": round(candidate_score, 6),
                    "score": round(candidate_score + 0.01 / max(dense_rank, 1), 6),
                    "dense_rank": int(dense_rank),
                    "original_rank": int(original_rank),
                    "was_in_baseline": passage_idx in original_rank_by_doc,
                    "action_id": probe.get("action_id", ""),
                    "role_consistent": bool(role_consistent),
                    "terminal_relation_score": round(float(terminal_relation_score), 6),
                    "terminal_type_score": round(float(terminal_type_score), 6),
                    "binding_source_passage_idx": int(binding.get("source_passage_idx", -1)),
                    "binding_source_rank": int(binding.get("source_rank", 0)),
                    "binding": dict(binding),
                    "probe": dict(probe),
                    "matched_binding": True,
                    "relation_score": round(float(relation_score), 6),
                    "relation_hits": relation_hits,
                    "answer_hit": bool(answer_match),
                    "second_hop_openie_hit": bool(has_second_hop_fact),
                    "transaction_action": {
                        "operator": "ontology_bind_probe_join",
                        "action_id": str(probe.get("action_id", "") or ""),
                        "source_passage_idx": int(binding.get("source_passage_idx", -1)),
                        "terminal_passage_idx": int(passage_idx),
                        "source_rank": int(binding.get("source_rank", 0)),
                        "terminal_original_rank": int(original_rank),
                        "dense_rank": int(dense_rank),
                        "estimated_probe_cost": float(probe.get("estimated_cost", 0.0) or 0.0),
                        "role_consistent": bool(role_consistent),
                    },
                    "contexts": [context],
                }
                previous = selected_by_passage.get(passage_idx)
                if previous is None or float(candidate["score"]) > float(previous.get("score", 0.0)):
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
        selected = candidates[: self.max_candidates]
        log["candidate_count"] = len(candidates)
        log["selected_count"] = len(selected)
        log["added_candidates"] = selected
        log["appended_passage_count"] = sum(
            1 for item in selected if int(item.get("passage_idx", -1)) not in original_rank_by_doc
        )
        log["rejection_reasons"] = dict(sorted(rejections.items(), key=lambda row: row[1], reverse=True)[:16])
        log["skip_reason"] = "candidate_only" if selected else "no_bound_second_hop_candidates"
        return QueryPlanBindProbeJoinResult(added_candidates=selected, log=log)
