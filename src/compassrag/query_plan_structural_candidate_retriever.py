from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

from .query_ontology_prior import tokenize
from .query_plan_ontology import relation_signature
from .utils.misc_utils import text_processing


@dataclass
class QueryPlanStructuralCandidateResult:
    passage_scores: Dict[int, float]
    passage_contexts: Dict[int, List[Dict[str, Any]]]
    log: Dict[str, Any] = field(default_factory=dict)


class QueryPlanOntologyStructuralCandidateRetriever:
    """Find ontology-supported passages even when OpenIE source links are sparse.

    The source-passage path lift is precise, but it only fires when executable
    ontology facts have usable source-passage links. This structural retriever
    scans the already-retrieved passage window and looks for passages that
    jointly mention concrete query-plan anchors plus terminal/relation clues.
    It is intentionally lexical and conservative; the downstream candidate
    retriever decides whether a found passage is strong enough to promote.
    """

    GENERIC_ANCHORS = {
        "north", "south", "east", "west", "northern", "southern", "eastern", "western",
        "region", "area", "place", "location", "city", "country", "county", "state",
        "province", "territory", "person", "people", "man", "woman", "film", "films",
        "movie", "movies", "book", "books", "album", "albums", "song", "songs", "work",
        "works", "letter", "letters", "document", "documents", "company", "organization", "school",
        "university", "team", "group", "event", "series", "episode", "river",
        "mountain", "lake", "island", "answer", "entity", "name", "title",
        "both", "second", "first", "third", "last", "same", "different",
    }
    GENERIC_RELATION_TOKENS = {
        "is", "are", "was", "were", "be", "been", "being", "has", "have", "had",
        "of", "in", "on", "at", "to", "from", "by", "with", "and", "or", "the",
        "a", "an", "as", "for", "into", "about", "related", "associated",
    }

    def __init__(
        self,
        scan_top_k: int = 80,
        max_original_rank: int = 120,
        max_added_candidates: int = 8,
        min_score: float = 0.56,
        require_anchor: bool = True,
        require_terminal_or_relation: bool = True,
        min_relation_token_overlap: int = 1,
        reject_generic_anchors: bool = True,
        score_weight: float = 0.85,
        require_existing_source: bool = False,
    ):
        self.scan_top_k = max(int(scan_top_k), 0)
        self.max_original_rank = int(max_original_rank)
        self.max_added_candidates = max(int(max_added_candidates), 0)
        self.min_score = min(max(float(min_score), 0.0), 1.0)
        self.require_anchor = bool(require_anchor)
        self.require_terminal_or_relation = bool(require_terminal_or_relation)
        self.min_relation_token_overlap = max(int(min_relation_token_overlap), 0)
        self.reject_generic_anchors = bool(reject_generic_anchors)
        self.score_weight = max(float(score_weight), 0.0)
        self.require_existing_source = bool(require_existing_source)

    @classmethod
    def is_generic_anchor(cls, value: Any) -> bool:
        text = text_processing(str(value or ""))
        if not text:
            return True
        tokens = tokenize(text)
        if text in cls.GENERIC_ANCHORS:
            return True
        if text.startswith("the ") and text_processing(text[4:]) in cls.GENERIC_ANCHORS:
            return True
        if len(tokens) <= 1 and (text in cls.GENERIC_ANCHORS or len(text) <= 2):
            return True
        if all(token in cls.GENERIC_ANCHORS for token in tokens):
            return True
        if len(tokens) <= 2 and any(token in {"both", "first", "second", "third", "last", "same"} for token in tokens):
            return True
        return False

    @staticmethod
    def _contains_entity(text: str, entity: str, min_coverage: float = 0.75) -> bool:
        entity_key = text_processing(entity)
        if not entity_key:
            return False
        text_key = text_processing(text)
        if entity_key in text_key:
            return True
        entity_tokens = tokenize(entity_key)
        if not entity_tokens:
            return False
        text_tokens = set(tokenize(text_key))
        if not text_tokens:
            return False
        matched = sum(1 for token in entity_tokens if token in text_tokens)
        return matched / max(len(entity_tokens), 1) >= min_coverage

    @classmethod
    def _relation_tokens(cls, values: Sequence[Any]) -> List[str]:
        tokens = []
        for value in values or []:
            text = text_processing(str(value or "").replace("_", " "))
            signature = relation_signature(text)
            for token in tokenize(text):
                if token and token not in cls.GENERIC_RELATION_TOKENS:
                    tokens.append(token)
            for token in tokenize(signature.replace("_", " ")):
                if token and token not in cls.GENERIC_RELATION_TOKENS:
                    tokens.append(token)
        seen = set()
        unique = []
        for token in tokens:
            if token not in seen:
                unique.append(token)
                seen.add(token)
        return unique

    def _extract_plan_terms(
        self,
        plans: Sequence[Mapping[str, Any]],
        anchors: Sequence[Tuple[str, float]],
        executable_paths: Sequence[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        anchor_entities = []
        for entity, score in anchors or []:
            entity = str(entity or "").strip()
            if not entity:
                continue
            if self.reject_generic_anchors and self.is_generic_anchor(entity):
                continue
            anchor_entities.append((entity, float(score or 0.0)))

        terminal_entities = []
        relations = []
        max_plan_precision = 0.0
        max_plan_scale = 0.0
        has_strict = False
        max_path_len = 1

        for plan in plans or []:
            try:
                max_plan_precision = max(max_plan_precision, float(plan.get("precision", 0.0) or 0.0))
            except (TypeError, ValueError):
                pass
            try:
                max_plan_scale = max(max_plan_scale, float(plan.get("lift_weight_scale", plan.get("scale", 0.0)) or 0.0))
            except (TypeError, ValueError):
                pass
            for step in plan.get("path", []) or []:
                relations.append(step.get("relation_family", ""))
                relations.append(step.get("relation_signature", ""))

        for path in executable_paths or []:
            terminal = str(path.get("entity", "") or path.get("end_entity", "") or "").strip()
            if terminal and not self.is_generic_anchor(terminal):
                terminal_entities.append(terminal)
            path_relations = [str(item or "") for item in path.get("relations", []) or []]
            relations.extend(path_relations)
            mode = str(path.get("signature_match_mode", "") or "").lower()
            has_strict = has_strict or mode == "strict"
            max_path_len = max(
                max_path_len,
                len(path.get("fact_indices", []) or []),
                len(path_relations),
            )
            try:
                max_plan_precision = max(
                    max_plan_precision,
                    float(path.get("plan_effective_precision", 0.0) or 0.0),
                )
            except (TypeError, ValueError):
                pass
            try:
                max_plan_scale = max(max_plan_scale, float(path.get("plan_lift_weight_scale", 0.0) or 0.0))
            except (TypeError, ValueError):
                pass

        terminal_entities = list(dict.fromkeys(terminal_entities))
        return {
            "anchors": anchor_entities,
            "terminals": terminal_entities,
            "relation_tokens": self._relation_tokens(relations),
            "plan_precision": min(max(max_plan_precision, 0.0), 1.0),
            "plan_scale": min(max(max_plan_scale or 1.0, 0.0), 1.0),
            "has_strict": has_strict,
            "max_path_len": max_path_len,
        }

    def expand(
        self,
        query: str,
        query_condition: Mapping[str, Any],
        plans: Sequence[Mapping[str, Any]],
        anchors: Sequence[Tuple[str, float]],
        executable_paths: Sequence[Mapping[str, Any]],
        sorted_doc_ids: Sequence[int],
        original_rank_by_doc: Mapping[int, int],
        existing_passage_scores: Mapping[int, float],
        existing_passage_contexts: Mapping[int, Sequence[Mapping[str, Any]]],
        get_passage_text: Callable[[int], str],
    ) -> QueryPlanStructuralCandidateResult:
        log: Dict[str, Any] = {
            "enabled": True,
            "scan_top_k": self.scan_top_k,
            "max_original_rank": self.max_original_rank,
            "max_added_candidates": self.max_added_candidates,
            "min_score": self.min_score,
            "require_anchor": self.require_anchor,
            "require_terminal_or_relation": self.require_terminal_or_relation,
            "min_relation_token_overlap": self.min_relation_token_overlap,
            "reject_generic_anchors": self.reject_generic_anchors,
            "score_weight": self.score_weight,
            "require_existing_source": self.require_existing_source,
        }
        passage_scores = {int(key): float(value) for key, value in (existing_passage_scores or {}).items()}
        passage_contexts = {
            int(key): list(value or [])
            for key, value in (existing_passage_contexts or {}).items()
        }

        if self.scan_top_k <= 0 or self.max_added_candidates <= 0:
            log["skip_reason"] = "disabled_by_zero_limit"
            return QueryPlanStructuralCandidateResult(passage_scores, passage_contexts, log)

        terms = self._extract_plan_terms(plans, anchors, executable_paths)
        anchor_entities = terms["anchors"]
        terminal_entities = terms["terminals"]
        relation_tokens = terms["relation_tokens"]
        log["anchor_count"] = len(anchor_entities)
        log["terminal_count"] = len(terminal_entities)
        log["relation_token_count"] = len(relation_tokens)
        log["plan_precision"] = terms["plan_precision"]
        log["plan_scale"] = terms["plan_scale"]
        log["has_strict_path"] = terms["has_strict"]
        log["max_path_len"] = terms["max_path_len"]

        if self.require_anchor and not anchor_entities:
            log["skip_reason"] = "no_concrete_anchors"
            return QueryPlanStructuralCandidateResult(passage_scores, passage_contexts, log)
        if self.require_terminal_or_relation and not terminal_entities and not relation_tokens:
            log["skip_reason"] = "no_terminal_or_relation_terms"
            return QueryPlanStructuralCandidateResult(passage_scores, passage_contexts, log)

        base_score = max((float(value) for value in passage_scores.values()), default=1.0)
        base_score = max(base_score, 1.0)
        candidates = []
        rejected = {
            "after_rank": 0,
            "empty_passage": 0,
            "missing_anchor": 0,
            "missing_terminal_or_relation": 0,
            "low_score": 0,
            "missing_existing_source": 0,
        }
        scanned = 0
        for raw_doc_id in list(sorted_doc_ids)[: self.scan_top_k]:
            doc_id = int(raw_doc_id)
            if self.require_existing_source and doc_id not in passage_scores:
                rejected["missing_existing_source"] += 1
                continue
            original_rank = int(original_rank_by_doc.get(doc_id, len(sorted_doc_ids) + 1))
            if self.max_original_rank > 0 and original_rank > self.max_original_rank:
                rejected["after_rank"] += 1
                continue
            passage_text = get_passage_text(doc_id)
            if not passage_text:
                rejected["empty_passage"] += 1
                continue
            scanned += 1
            text_tokens = set(tokenize(text_processing(passage_text)))

            matched_anchors = [
                entity
                for entity, _ in anchor_entities
                if self._contains_entity(passage_text, entity, min_coverage=0.80)
            ]
            if self.require_anchor and not matched_anchors:
                rejected["missing_anchor"] += 1
                continue

            matched_terminals = [
                entity
                for entity in terminal_entities[:24]
                if self._contains_entity(passage_text, entity, min_coverage=0.67)
            ]
            relation_overlap = sorted({token for token in relation_tokens if token in text_tokens})
            has_relation_evidence = len(relation_overlap) >= self.min_relation_token_overlap
            if self.require_terminal_or_relation and not matched_terminals and not has_relation_evidence:
                rejected["missing_terminal_or_relation"] += 1
                continue

            anchor_strength = min(len(matched_anchors) / max(min(len(anchor_entities), 2), 1), 1.0)
            terminal_strength = min(len(matched_terminals) / 2.0, 1.0)
            relation_strength = min(len(relation_overlap) / max(self.min_relation_token_overlap + 1, 2), 1.0)
            plan_quality = 0.50 * float(terms["plan_precision"]) + 0.30 * float(terms["plan_scale"])
            if terms["has_strict"]:
                plan_quality += 0.10
            if terms["max_path_len"] >= 2:
                plan_quality += 0.10
            plan_quality = min(max(plan_quality, 0.0), 1.0)

            score = (
                0.36 * anchor_strength
                + 0.30 * terminal_strength
                + 0.22 * relation_strength
                + 0.12 * plan_quality
            )
            if matched_terminals and has_relation_evidence:
                score += 0.04
            score = min(max(score, 0.0), 1.0)
            if score < self.min_score:
                rejected["low_score"] += 1
                continue

            candidates.append({
                "passage_idx": doc_id,
                "score": score,
                "original_rank": original_rank,
                "matched_anchors": matched_anchors[:4],
                "matched_terminals": matched_terminals[:4],
                "relation_overlap": relation_overlap[:8],
                "anchor_strength": anchor_strength,
                "terminal_strength": terminal_strength,
                "relation_strength": relation_strength,
                "plan_quality": plan_quality,
            })

        candidates.sort(
            key=lambda item: (
                float(item["score"]),
                float(item["terminal_strength"]),
                float(item["relation_strength"]),
                -int(item["original_rank"]),
            ),
            reverse=True,
        )
        selected = candidates[: self.max_added_candidates]
        for item in selected:
            passage_idx = int(item["passage_idx"])
            raw_score = base_score * self.score_weight * float(item["score"])
            passage_scores[passage_idx] = max(passage_scores.get(passage_idx, 0.0), raw_score)
            passage_contexts.setdefault(passage_idx, []).append({
                "structural_candidate": True,
                "plan_id": "structural_scan",
                "anchor_entity": item["matched_anchors"][0] if item["matched_anchors"] else "",
                "end_entity": item["matched_terminals"][0] if item["matched_terminals"] else "",
                "relations": item["relation_overlap"],
                "fact_indices": [],
                "plan_intent": str(query_condition.get("intent", "") or ""),
                "plan_answer_type": str(query_condition.get("answer_type", "") or ""),
                "signature_match_mode": "strict" if terms["has_strict"] else "structural",
                "plan_lift_weight_scale": float(terms["plan_scale"]),
                "plan_effective_precision": float(terms["plan_precision"]),
                "score": raw_score,
                "structural_score": float(item["score"]),
                "original_rank": int(item["original_rank"]),
            })

        log.update({
            "scanned_passage_count": scanned,
            "candidate_count": len(candidates),
            "added_candidate_count": len(selected),
            "rejection_reasons": {key: value for key, value in rejected.items() if value},
            "added_candidates": [
                {
                    "passage_idx": int(item["passage_idx"]),
                    "score": round(float(item["score"]), 6),
                    "original_rank": int(item["original_rank"]),
                    "matched_anchors": item["matched_anchors"],
                    "matched_terminals": item["matched_terminals"],
                    "relation_overlap": item["relation_overlap"],
                }
                for item in selected
            ],
        })
        if not selected:
            log["skip_reason"] = "no_structural_candidates"
        return QueryPlanStructuralCandidateResult(passage_scores, passage_contexts, log)
