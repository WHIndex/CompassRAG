import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

try:
    from .query_ontology_prior import (
        HIGH_SIGNAL_RELATION_TERMS,
        INTENT_TERMS,
        STOPWORDS,
        _relation_family,
        normalize_text,
        tokenize,
    )
except ImportError:
    from query_ontology_prior import (
        HIGH_SIGNAL_RELATION_TERMS,
        INTENT_TERMS,
        STOPWORDS,
        _relation_family,
        normalize_text,
        tokenize,
    )


SCHEMA_QUERY_TERMS = (
    set().union(*INTENT_TERMS.values())
    | HIGH_SIGNAL_RELATION_TERMS
    | {
        "actor", "actress", "answer", "author", "award", "book", "born",
        "affiliated", "alma", "attended", "capital", "city", "club", "company",
        "country", "county", "date", "death", "die", "director", "educated",
        "education", "family", "father", "film", "first", "founded",
        "graduated", "husband", "job", "label", "located", "location",
        "married", "member", "mother", "movie", "nationality", "occupation",
        "older", "part", "place", "profession", "producer", "record", "region",
        "released", "same", "school", "signed", "song", "spouse", "state",
        "studied", "team", "university", "when", "where", "which", "who",
        "wife", "writer", "younger",
        "activate", "adverse", "allele", "associate", "biomarker", "bind",
        "cause", "diagnose", "disease", "dosage", "drug", "enzyme",
        "express", "gene", "genetic", "inhibit", "interact", "mechanism",
        "mutation", "pathway", "prevent", "protein", "receptor", "regulate",
        "risk", "screen", "symptom", "therapy", "toxicity", "treat", "variant",
    }
)


GENERIC_ENTITY_TOKENS = {
    "answer", "book", "city", "country", "date", "entity", "event", "film",
    "location", "movie", "organization", "people", "person", "place", "song",
    "state", "team", "work", "year",
}


GENERIC_SCHEMA_NOUNS = {
    "answer", "book", "city", "country", "date", "entity", "event", "film",
    "first", "location", "movie", "people", "person", "place", "same", "song",
    "state", "team", "work", "year",
}


QUERY_PLAN_RELATION_FAMILIES = {
    "birth", "death", "location", "creator", "performer", "production", "family",
    "nationality", "occupation", "membership", "ownership", "award", "alias",
    "adaptation",
    "biomedical_adverse_effect", "biomedical_association", "biomedical_attribute",
    "biomedical_causation", "biomedical_classification", "biomedical_diagnosis",
    "biomedical_dosage", "biomedical_epidemiology", "biomedical_genetics",
    "biomedical_interaction", "biomedical_manifestation", "biomedical_mechanism",
    "biomedical_regulation", "biomedical_treatment", "biomedical_definition",
    "biomedical_composition", "biomedical_localization", "biomedical_inheritance",
    "biomedical_comparison",
}


QUERY_PLAN_RELATION_TERMS = (
    HIGH_SIGNAL_RELATION_TERMS
    | {
        "actor", "actress", "affiliated", "alma", "attended", "author", "born",
        "birth", "birthday", "city", "club", "composer", "composed", "country",
        "county", "death", "die", "died", "directed", "director", "educated",
        "education", "employed", "father", "founded", "graduated",
        "headquartered", "headquarters", "husband", "job", "label", "located",
        "married", "member", "mother", "nationality", "occupation", "part",
        "played", "profession", "producer", "produced", "published", "publisher",
        "record", "region", "release", "released", "school", "served", "signed",
        "spouse", "starring", "starred", "state", "studied", "team",
        "university", "wife", "worked", "writer", "wrote", "written",
        "activate", "adverse", "allele", "associate", "biomarker", "bind",
        "cause", "classify", "correlate", "diagnose", "dosage", "drug",
        "enzyme", "express", "gene", "genetic", "inhibit", "interact",
        "mechanism", "mutation", "pathway", "prevent", "protein", "receptor",
        "regulate", "risk", "screen", "symptom", "therapy", "toxicity",
        "treat", "variant",
    }
) - GENERIC_SCHEMA_NOUNS


def relation_signature(relation: Any, dataset: str = "") -> str:
    relation_text = normalize_text(str(relation or "").replace("_", " ").replace("-", " "))
    tokens = set(tokenize(relation_text))
    family = _relation_family(relation_text)






    if dataset and not str(dataset).lower().startswith("bioasq"):
        if family == "biomedical_localization" and tokens & {
            "locate", "located", "localization", "localize",
        }:
            family = "location"
        elif family.startswith("biomedical_"):
            return relation_text

    if family.startswith("biomedical_"):
        return family

    if family == "birth":
        if tokens & {"date", "year", "on"}:
            return "birth_date"
        if tokens & {"place", "location", "city", "country", "in", "birthplace"}:
            return "birth_place"
        return "birth"
    if family == "death":
        if tokens & {"date", "year", "on"}:
            return "death_date"
        if tokens & {"place", "location", "city", "country", "in"}:
            return "death_place"
        return "death"
    if family == "location":
        if tokens & {"part"}:
            return "location_part_of"
        if tokens & {"county", "province", "region", "state"}:
            return "location_region"
        if tokens & {"country", "nationality", "citizenship"}:
            return "location_country"
        if tokens & {"city", "town"}:
            return "location_city"
        if tokens & {"headquarters", "headquartered"}:
            return "headquarters_location"
        return "location"
    if family == "creator":
        if tokens & {"director", "directed"}:
            return "director"
        if tokens & {"author", "writer", "wrote", "written", "authored"}:
            return "author"
        if tokens & {"composer", "composed"}:
            return "composer"
        return "creator"
    if family == "performer":
        if tokens & {"actor", "actress", "starring", "starred"}:
            return "actor"
        return "performer"
    if family == "production":
        if tokens & {"release", "released"}:
            return "release_date"
        if tokens & {"published", "publication", "publisher"}:
            return "publication"
        if tokens & {"producer", "produced"}:
            return "producer"
        return "production"
    if family == "family":
        for label in ("father", "mother", "son", "daughter", "spouse", "wife", "husband", "married", "sibling"):
            if label in tokens:
                return f"family_{label}"
        return "family"
    if family == "nationality":
        return "nationality"
    if family == "occupation":
        return "occupation"
    if family == "membership":
        if tokens & {"educated", "education", "graduated", "studied", "attended", "alma", "school", "university"}:
            return "education"
        if tokens & {"signed", "label", "record"}:
            return "record_label"
        if tokens & {"team", "club", "played"}:
            return "team_membership"
        if tokens & {"employed", "worked", "served"}:
            return "employment"
        return "membership"
    if family == "ownership":
        if tokens & {"founded", "founder"}:
            return "founder"
        if tokens & {"owned", "owner"}:
            return "owner"
        return "ownership"
    if family == "award":
        return "award"
    if family == "alias":
        return "alias"
    if family == "adaptation":
        return "adaptation"
    return family or relation_text


def factorized_relation_family(
    signature: Any,
    collapse_open_class: bool = False,
) -> str:
    """Map fine-grained relation signatures to reusable operator families."""
    normalized = normalize_text(str(signature or "").replace("_", " "))
    family = _relation_family(normalized)
    if family in {"child", "children", "sibling", "brother", "sister"}:
        return "family"
    if collapse_open_class and family not in QUERY_PLAN_RELATION_FAMILIES:



        return "attribute"
    return family


def plan_schema_sha256(plan: Mapping[str, Any]) -> str:
    """Return a stable content address for the executable plan schema."""
    semantic_fields = (
        "id",
        "intent",
        "question_pattern",
        "triggers",
        "anchor_slots",
        "path",
        "answer_type",
        "answer_constraints",
        "integrity_constraints",
        "parent_schema_id",
        "competency_status",
    )
    payload = {field: plan.get(field) for field in semantic_fields if field in plan}
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_json_or_jsonl(path: str | Path, max_rows: int | None = None) -> List[Dict[str, Any]]:
    path = Path(path)
    if path.suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
                    if max_rows is not None and len(rows) >= max_rows:
                        break
        return rows
    data = json.load(path.open("r", encoding="utf-8"))
    if isinstance(data, list):
        return data[:max_rows] if max_rows is not None else data
    if isinstance(data, dict):
        for key in ("data", "examples", "rows"):
            if isinstance(data.get(key), list):
                rows = data[key]
                return rows[:max_rows] if max_rows is not None else rows
    return []


def text_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in ("text", "paragraph_text", "summary", "value", "answer"):
            nested = value.get(key)
            if isinstance(nested, dict):
                nested_text = text_value(nested)
                if nested_text:
                    return nested_text
            elif nested is not None:
                nested_text = text_value(nested)
                if nested_text:
                    return nested_text
        return ""
    if isinstance(value, list):
        parts = [text_value(item) for item in value]
        return " ".join(part for part in parts if part).strip()
    return str(value).strip()


def get_question(row: Dict[str, Any]) -> str:
    return text_value(row.get("question") or row.get("query") or row.get("input"))


def split_answer_string(value: str) -> List[str]:
    value = value.strip()
    if not value:
        return []
    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return [text_value(item) for item in parsed if text_value(item)]
    except Exception:
        pass
    if ";" in value:
        return [part.strip() for part in value.split(";") if part.strip()]
    return [value]


def get_answers(row: Dict[str, Any]) -> List[str]:
    for key in ("answer", "answers", "best_answer", "correct_answers", "possible_answers", "answer_aliases", "gold_ans"):
        value = row.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            return split_answer_string(value)
        if isinstance(value, list):
            answers = [text_value(item) for item in value if text_value(item)]
            if answers:
                return answers
        if isinstance(value, dict):
            answer = text_value(value)
            if answer:
                return [answer]
    return []


def get_row_id(row: Dict[str, Any]) -> str:
    for key in ("id", "_id", "qid", "question_id"):
        if key in row:
            return str(row[key])
    return normalize_text(get_question(row))


def get_evidence_triples(row: Dict[str, Any]) -> List[Tuple[str, str, str]]:
    question = get_question(row)
    answers = get_answers(row)
    triples = []
    for value in row.get("evidences", []) or []:
        if isinstance(value, (list, tuple)) and len(value) == 3:
            triple = tuple(normalize_text(part) for part in value)
            if all(triple):
                triples.append(triple)
        elif isinstance(value, dict):
            fact = value.get("fact") or value.get("triple")
            if isinstance(fact, (list, tuple)) and len(fact) == 3:
                triple = tuple(normalize_text(part) for part in fact)
                if all(triple):
                    triples.append(triple)
    if triples:
        return triples

    subj = row.get("subj")
    prop = row.get("prop")
    obj = row.get("obj")
    if subj and prop and obj:
        triple = (normalize_text(subj), normalize_text(prop), normalize_text(obj))
        if all(triple):
            return [triple]

    decompositions = row.get("question_decomposition", []) or []
    previous_answers: Dict[str, str] = {}
    for idx, decomposition in enumerate(decompositions, start=1):
        if not isinstance(decomposition, dict):
            continue
        sub_question = str(decomposition.get("question") or "").strip()
        answer = str(decomposition.get("answer") or "").strip()
        if not sub_question or not answer:
            continue
        subject = ""
        relation = ""
        if ">>" in sub_question:
            subject, relation = sub_question.split(">>", 1)
        else:
            ref_match = re.search(r"#(\d+)", sub_question)
            if ref_match:
                reference_id = ref_match.group(1)
                subject = previous_answers.get(
                    reference_id, f"unresolved_binding_{reference_id}"
                )
            else:




                subject = f"query_anchor_{idx}"
            relation = relation_from_decomposition_question(sub_question)
        triple = (normalize_text(subject), normalize_text(relation), normalize_text(answer))
        if all(triple):
            triples.append(triple)
        previous_answers[str(idx)] = answer
        if decomposition.get("id") is not None:
            previous_answers[str(decomposition["id"])] = answer
    if triples:
        return triples

    return text_evidence_triples(row, question, answers)


def relation_from_decomposition_question(question: str) -> str:
    question = re.sub(r"#\d+", " entity ", normalize_text(question))
    removable = {
        "a", "an", "are", "as", "at", "be", "been", "being", "by", "did",
        "do", "does", "entity", "for", "from", "had", "has", "have", "he",
        "her", "his", "in", "is", "it", "its", "of", "on", "she", "that",
        "the", "their", "this", "to", "was", "were", "what", "when", "where",
        "which", "who", "whom", "whose", "with",
    }
    relation_tokens = [token for token in tokenize(question) if token not in removable]
    return " ".join(relation_tokens[:6])


def supporting_fact_refs(row: Dict[str, Any]) -> List[Tuple[str, int | None]]:
    value = row.get("supporting_facts")
    refs: List[Tuple[str, int | None]] = []
    if isinstance(value, dict):
        titles = value.get("title") or value.get("titles") or []
        sent_ids = value.get("sent_id") or value.get("sent_ids") or []
        if not isinstance(titles, list):
            titles = [titles]
        if not isinstance(sent_ids, list):
            sent_ids = [sent_ids]
        for idx, title in enumerate(titles):
            sent_id = sent_ids[idx] if idx < len(sent_ids) else None
            try:
                sent_id = int(sent_id) if sent_id is not None else None
            except (TypeError, ValueError):
                sent_id = None
            refs.append((normalize_text(title), sent_id))
    elif isinstance(value, list):
        for item in value:
            title = ""
            sent_id = None
            if isinstance(item, dict):
                title = text_value(item.get("title"))
                raw_sent_id = item.get("sent_id") or item.get("sentence_id") or item.get("idx")
                try:
                    sent_id = int(raw_sent_id) if raw_sent_id is not None else None
                except (TypeError, ValueError):
                    sent_id = None
            elif isinstance(item, (list, tuple)) and item:
                title = text_value(item[0])
                if len(item) > 1:
                    try:
                        sent_id = int(item[1])
                    except (TypeError, ValueError):
                        sent_id = None
            if title:
                refs.append((normalize_text(title), sent_id))
    return refs


def get_supporting_passages(row: Mapping[str, Any], limit: int = 3) -> List[Dict[str, str]]:
    """Return corpus-verbatim positive passages across supported QA formats."""
    result: List[Dict[str, str]] = []
    seen = set()

    def add(title: Any, text: Any) -> None:
        title_value = str(title or "").strip()
        text_value = str(text or "").strip()
        key = (normalize_text(title_value), normalize_text(text_value))
        if not text_value or key in seen or len(result) >= limit:
            return
        seen.add(key)
        result.append({"title": title_value, "text": text_value[:2400]})

    supporting_titles = {title for title, _ in supporting_fact_refs(dict(row)) if title}
    context = row.get("context", []) or []
    if isinstance(context, Mapping):
        titles = list(context.get("title", []) or [])
        sentence_groups = list(context.get("sentences", []) or [])
        context = [
            [title, sentence_groups[idx] if idx < len(sentence_groups) else []]
            for idx, title in enumerate(titles)
        ]
    if isinstance(context, list):
        for item in context:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            title = str(item[0] or "")
            if supporting_titles and normalize_text(title) not in supporting_titles:
                continue
            sentences = item[1]
            text = " ".join(str(value) for value in sentences) if isinstance(sentences, list) else str(sentences or "")
            add(title, text)

    for key in ("paragraphs", "contexts", "ctxs"):
        for passage in row.get(key, []) or []:
            if not isinstance(passage, Mapping):
                continue
            title = str(passage.get("title", "") or "")
            explicitly_supporting = passage.get("is_supporting", passage.get("supporting"))
            if explicitly_supporting is False:
                continue
            if supporting_titles and normalize_text(title) not in supporting_titles and explicitly_supporting is not True:
                continue
            add(
                title,
                passage.get("text", passage.get("paragraph_text", passage.get("passage", ""))),
            )
    return result


def support_text_entries(row: Dict[str, Any], max_entries: int = 12, max_text_chars: int = 400) -> List[Dict[str, Any]]:
    refs = supporting_fact_refs(row)
    ref_titles = {title for title, _ in refs if title}
    ref_pairs = {(title, sent_id) for title, sent_id in refs if title}
    entries: List[Dict[str, Any]] = []

    def add_entry(title: Any, text: Any, is_supporting: bool = False):
        title_text = text_value(title)
        body = text_value(text)
        if max_text_chars > 0 and len(body) > max_text_chars:
            body = body[:max_text_chars]
        if not title_text and not body:
            return
        entries.append({
            "title": title_text,
            "text": body,
            "is_supporting": bool(is_supporting),
        })

    paragraphs = row.get("paragraphs")
    if isinstance(paragraphs, list):
        support_indices = set()
        for decomposition in row.get("question_decomposition", []) or []:
            if isinstance(decomposition, dict) and decomposition.get("paragraph_support_idx") is not None:
                try:
                    support_indices.add(int(decomposition["paragraph_support_idx"]))
                except (TypeError, ValueError):
                    pass
        for paragraph in paragraphs:
            if not isinstance(paragraph, dict):
                add_entry("", paragraph)
                continue
            idx = paragraph.get("idx")
            try:
                idx = int(idx) if idx is not None else None
            except (TypeError, ValueError):
                idx = None
            is_supporting = bool(paragraph.get("is_supporting")) or (idx in support_indices)
            add_entry(
                paragraph.get("title"),
                paragraph.get("paragraph_text") or paragraph.get("text"),
                is_supporting=is_supporting,
            )

    context = row.get("context")
    if isinstance(context, dict):
        titles = context.get("title") or context.get("titles") or []
        sentences = context.get("sentences") or context.get("text") or []
        if isinstance(titles, list) and isinstance(sentences, list):
            for idx, title in enumerate(titles):
                sentence_list = sentences[idx] if idx < len(sentences) else []
                normalized_title = normalize_text(title)
                selected_sentences = []
                supporting = normalized_title in ref_titles
                for ref_title, sent_id in ref_pairs:
                    if ref_title == normalized_title and sent_id is not None and isinstance(sentence_list, list) and sent_id < len(sentence_list):
                        selected_sentences.append(sentence_list[sent_id])
                if not selected_sentences:
                    selected_sentences = sentence_list if isinstance(sentence_list, list) else [sentence_list]
                add_entry(title, " ".join(text_value(sentence) for sentence in selected_sentences), is_supporting=supporting)
        else:
            add_entry(context.get("title"), context.get("text") or context.get("sentences"), is_supporting=True)
    elif context is not None:
        add_entry("", context, is_supporting=True)

    contexts = row.get("contexts")
    if isinstance(contexts, list):
        for context_item in contexts:
            if isinstance(context_item, dict):
                add_entry(
                    context_item.get("title") or context_item.get("name"),
                    context_item.get("text") or context_item.get("paragraph_text") or context_item.get("content"),
                    is_supporting=True,
                )
            else:
                add_entry("", context_item, is_supporting=True)
    elif contexts is not None:
        add_entry("", contexts, is_supporting=True)

    document = row.get("document")
    if isinstance(document, dict):
        add_entry(
            document.get("title") or document.get("id"),
            "",
            is_supporting=True,
        )
    elif document is not None:
        add_entry("", document, is_supporting=True)

    for key in ("reference", "source"):
        if row.get(key) is not None:
            add_entry("", row.get(key), is_supporting=True)

    supporting_entries = [entry for entry in entries if entry.get("is_supporting") and entry.get("text")]
    if supporting_entries:
        entries = supporting_entries + [entry for entry in entries if entry not in supporting_entries]

    deduped = []
    seen = set()
    for entry in entries:
        key = (normalize_text(entry.get("title", "")), normalize_text(entry.get("text", ""))[:240])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(entry)
        if len(deduped) >= max_entries:
            break
    return deduped


RELATION_CUE_PATTERNS: List[Tuple[str, str]] = [
    (r"\badverse\b|\bside effect\b|\btoxici", "adverse effect"),
    (r"\bdose\b|\bdosage\b|\badministration\b", "dosage"),
    (r"\bdiagnos|\bdetect|\bscreen|\btest\b", "diagnosis"),
    (r"\bbiomarker\b|\bmarker\b", "biomarker"),
    (r"\btreat|\btherap|\bprevent|\bprophyl", "treatment"),
    (r"\bcause|\binduc|\betiolog", "causation"),
    (r"\bgene\b|\bgenetic\b|\bmutation\b|\bvariant\b|\ballele\b", "genetic association"),
    (r"\binhibit|\bactivat|\bregulat|\bexpress", "molecular regulation"),
    (r"\binteract|\bbind|\breceptor\b|\btarget\b", "interaction"),
    (r"\bassociate|\bcorrelat|\brisk factor|\blinked\b", "association"),
    (r"\bsymptom|\bsign\b|\bmanifest|\bphenotype", "manifestation"),
    (r"\bincidence\b|\bprevalence\b|\bmortality\b|\bsurvival\b", "epidemiology"),
    (r"\bmechanis|\bpathway|\bfunction\b|\brole\b", "mechanism"),
    (r"\binherit|\bautosomal|\bdominant|\brecessive", "inheritance"),
    (r"\blocated|\blocali[sz]|\bsecret", "localization"),
    (r"\bcontain|\binclud|\bcompos|\bcomponent", "composition"),
    (r"\bcompar|\bdifference", "comparison"),
    (r"^(?:what|which) (?:is|are)\b|^describe\b|\bstand for\b", "definition"),
    (r"\bbirthplace\b|\bwas born\b|\bborn\b|\bbirth\b", "born"),
    (r"\bdied\b|\bdeath\b", "died"),
    (r"\bfather\b", "father"),
    (r"\bmother\b", "mother"),
    (r"\bson\b", "son"),
    (r"\bdaughter\b", "daughter"),
    (r"\bspouse\b|\bwife\b|\bhusband\b|\bmarried\b", "spouse"),
    (r"\bdirected\b|\bdirector\b", "director"),
    (r"\bauthor\b|\bauthored\b|\bwriter\b|\bwrote\b|\bwritten\b", "author"),
    (r"\bcomposer\b|\bcomposed\b", "composer"),
    (r"\bstarring\b|\bstarred\b|\bactor\b|\bactress\b", "actor"),
    (r"\bperformed\b|\bperformer\b|\bplayed\b", "performed"),
    (r"\bpublished\b|\bpublication\b|\bpublisher\b|\bprinted\b", "published"),
    (r"\breleased\b|\brelease\b", "released"),
    (r"\bfounded\b|\bfounder\b|\bestablished\b|\bformed\b|\bstarted\b|\bbegan\b", "founded"),
    (r"\bowned\b|\bowner\b|\bbelong", "owned"),
    (r"\bheadquartered\b|\bheadquarters\b", "headquartered"),
    (r"\bcountry\b|\bnationality\b|\bcitizenship\b|\bcitizen\b", "citizenship"),
    (r"\bcounty\b|\bstate\b|\bprovince\b|\bregion\b", "region"),
    (r"\bcity\b|\btown\b|\blocated\b|\blocation\b|\bwhere\b|\bplace\b|\bpart of\b", "located"),
    (r"\boccupation\b|\bprofession\b|\bjob\b", "occupation"),
    (r"\beducated\b|\beducation\b|\bgraduated\b|\bstud(?:y|ies|ied|ying)\b|\battended\b|\balma\b|\bschool\b|\buniversity\b", "educated"),
    (r"\bteam\b|\bclub\b", "team"),
    (r"\bsigned\b|\brecord label\b|\blabel\b", "signed"),
    (r"\bmember\b|\baffiliated\b", "member"),
    (r"\baward\b|\bwon\b|\bchampionship\b|\bcompetition\b|\btournament\b", "won"),
    (r"\badapted\b|\bbased on\b", "adapted"),
    (r"\bknown as\b|\balso known\b|\bcalled\b|\bnamed\b", "known as"),
]


def infer_relation_from_text_evidence(question: str, support_entries: List[Dict[str, Any]], answers: List[str]) -> str:
    question_text = normalize_text(question)
    answer_text = normalize_text(" ".join(answers[:4]))
    support_text = normalize_text(" ".join(entry.get("text", "") for entry in support_entries[:4]))
    answer_augmented = f"{question_text} {answer_text}"
    combined = f"{answer_augmented} {support_text}"



    for source in (question_text, answer_augmented, combined):
        for pattern, relation in RELATION_CUE_PATTERNS:
            if re.search(pattern, source):
                return relation

    answer_type = infer_answer_type(question, answers)
    if answer_type == "date":
        return "released"
    if answer_type == "location":
        return "located"
    if answer_type == "person":
        return "member"
    return "occupation"


def extract_anchor_candidates(question: str, support_entries: List[Dict[str, Any]], answers: List[str], max_anchors: int = 3) -> List[str]:
    question_norm = normalize_text(question)
    answer_norms = {normalize_text(answer) for answer in answers if normalize_text(answer)}
    candidates = []
    for entry in support_entries:
        title = text_value(entry.get("title"))
        if not title:
            continue
        title_norm = normalize_text(title)
        if title_norm in answer_norms:
            continue
        title_tokens = set(tokenize(title_norm))
        query_tokens = set(tokenize(question_norm))
        if entry.get("is_supporting") or title_norm in question_norm or (title_tokens and title_tokens & query_tokens):
            candidates.append(title)

    for quoted in re.findall(r"['\"]([^'\"]{2,80})['\"]", question):
        if normalize_text(quoted) not in answer_norms:
            candidates.append(quoted)
    for phrase in re.findall(r"\b[A-Z][\w'’.-]*(?:\s+[A-Z][\w'’.-]*){0,5}", question):
        phrase_norm = normalize_text(phrase)
        if phrase_norm and phrase_norm not in answer_norms and phrase_norm not in {"who", "what", "when", "where", "which"}:
            candidates.append(phrase)

    deduped = []
    seen = set()
    for candidate in candidates:
        normalized = normalize_text(candidate)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
        if len(deduped) >= max_anchors:
            break
    return deduped


def text_evidence_triples(row: Dict[str, Any], question: str, answers: List[str]) -> List[Tuple[str, str, str]]:
    if not question:
        return []
    support_entries = support_text_entries(row)
    if not answers and not support_entries:
        return []
    relation = infer_relation_from_text_evidence(question, support_entries, answers)
    if not _relation_family(relation):
        return []
    anchors = extract_anchor_candidates(question, support_entries, answers)
    if not anchors:
        anchors = ["query anchor"]
    answer = normalize_text(answers[0]) if answers else infer_answer_type(question, answers)
    if answer in {"yes", "no", "true", "false"}:
        answer = "comparison answer"
    if not answer:
        answer = infer_answer_type(question, answers)

    triples = []
    for anchor in anchors:
        triple = (normalize_text(anchor), normalize_text(relation), normalize_text(answer))
        if all(triple):
            triples.append(triple)
    return triples[:3]


def infer_answer_type(question: str, answers: Iterable[str]) -> str:
    question_text = normalize_text(question)
    question_padded = f" {question_text} "

    if re.match(r"^(list|name (?:two|three|four|five)|which (?:drugs|genes|proteins|factors|cells|diseases|methods|classes))\b", question_text):
        return "list"

    if re.match(r"^(is|are|was|were|do|does|did|can|could|has|have)\b", question_text):
        return "boolean"
    if re.search(r"\bhow many\b|\bpercentage\b|\brate\b|\bdose\b|\bdosage\b", question_text):
        return "quantity"
    if re.search(r"\bgene\b|\bmutation\b|\bvariant\b|\ballele\b", question_text):
        return "gene"
    if re.search(r"\bdrug\b|\bmedication\b|\bcompound\b|\btreatment\b|\btherapy\b", question_text):
        return "drug"
    if re.search(r"\bprotein\b|\benzyme\b|\breceptor\b|\bbiomarker\b", question_text):
        return "protein"
    if re.search(r"\bdisease\b|\bdisorder\b|\bsyndrome\b|\bcondition\b", question_text):
        return "disease"
    if re.match(r"^(what|which) (is|are)\b|^describe\b", question_text):
        return "definition"






    if question_text.startswith("when ") or any(
        phrase in question_padded
        for phrase in {" what month ", " which month ", " what year ", " which year ", " what date "}
    ):
        return "date"
    if question_text.startswith(("who ", "whom ", "whose ")):
        return "person"
    if question_text.startswith("where ") or any(
        phrase in question_padded
        for phrase in {" what city ", " which city ", " what country ", " which country ", " what state ", " which state ", " what county ", " which county "}
    ):
        return "location"
    if question_text.startswith("how "):


        return "entity"

    q_tokens = set(tokenize(question))
    answer_text = " ".join(str(answer) for answer in answers)
    answer_tokens = set(tokenize(answer_text))
    if q_tokens & {"when", "date", "year"} or re.search(r"\b\d{3,4}\b", answer_text):
        return "date"
    if q_tokens & {"where", "county", "city", "country", "state", "region", "place", "located", "location"}:
        return "location"
    if q_tokens & {"who", "whose", "father", "mother", "son", "daughter", "wife", "husband", "spouse"}:
        return "person"
    if q_tokens & {"which"} and q_tokens & {"film", "movie", "book", "song", "album", "work"}:
        return "work"
    if answer_tokens & {"american", "british", "french", "german", "italian", "canadian"}:
        return "nationality"
    return "entity"


def canonical_answer_type_family(value: Any) -> str:
    normalized = normalize_text(value)
    tokens = set(tokenize(normalized))
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
    return normalized or "unknown"


def schema_query_terms(question: str, entities: Iterable[str], answers: Iterable[str], max_terms: int = 16) -> List[str]:
    entity_tokens = set()
    for entity in entities:
        entity_tokens.update(tokenize(entity))
    for answer in answers:
        entity_tokens.update(tokenize(answer))

    terms = []
    seen = set()
    for token in tokenize(question):
        if token in seen or token in STOPWORDS or token in entity_tokens:
            continue
        if token in SCHEMA_QUERY_TERMS:
            terms.append(token)
            seen.add(token)
        if len(terms) >= max_terms:
            break
    return terms


def endpoint_mentions_question(endpoint: str, question: str) -> bool:
    endpoint = normalize_text(endpoint)
    question_text = f" {normalize_text(question)} "
    if not endpoint:
        return False
    if f" {endpoint} " in question_text:
        return True
    endpoint_tokens = set(tokenize(endpoint)) - GENERIC_ENTITY_TOKENS
    query_tokens = set(tokenize(question))
    if not endpoint_tokens:
        return False
    return len(endpoint_tokens & query_tokens) / max(len(endpoint_tokens), 1) >= 0.5


def ordered_evidence_path(triples: List[Tuple[str, str, str]], question: str, answers: List[str]) -> List[Tuple[str, str, str]]:
    if len(triples) < 2:
        return triples

    answer_tokens = set()
    for answer in answers:
        answer_tokens.update(tokenize(answer))

    def endpoint_score(endpoint: str) -> float:
        score = 0.0
        if endpoint_mentions_question(endpoint, question):
            score += 2.0
        tokens = set(tokenize(endpoint))
        if answer_tokens and tokens & answer_tokens:
            score -= 1.0
        return score

    remaining = list(triples)
    start_idx = max(
        range(len(remaining)),
        key=lambda idx: max(endpoint_score(remaining[idx][0]), endpoint_score(remaining[idx][2])),
    )
    path = [remaining.pop(start_idx)]
    while remaining:
        endpoints = {path[-1][0], path[-1][2]}
        next_idx = None
        for idx, triple in enumerate(remaining):
            if triple[0] in endpoints or triple[2] in endpoints:
                next_idx = idx
                break
        if next_idx is None:
            path.extend(remaining)
            break
        path.append(remaining.pop(next_idx))
    return path


def signed_evidence_path(
    triples: List[Tuple[str, str, str]],
    question: str,
    answers: List[str],
) -> Tuple[List[Tuple[str, str, str]], List[str]]:
    """Order evidence edges and retain the query-to-answer traversal direction.

    Stored triples describe relation facts, while an executable schema describes
    how a binding flows through those facts.  The two are not equivalent when a
    query starts from the object of a relation.  This helper therefore leaves the
    fact untouched and records ``forward`` (subject to object) or ``reverse``
    (object to subject) for every traversal step.
    """
    if not triples:
        return [], []

    answer_tokens = {
        token for answer in answers for token in tokenize(answer)
    }

    def endpoint_score(endpoint: str) -> float:
        score = 2.0 if endpoint_mentions_question(endpoint, question) else 0.0
        if answer_tokens and set(tokenize(endpoint)) & answer_tokens:
            score -= 1.0
        return score

    remaining = list(triples)
    start_idx = max(
        range(len(remaining)),
        key=lambda idx: max(
            endpoint_score(remaining[idx][0]),
            endpoint_score(remaining[idx][2]),
        ),
    )
    first = remaining.pop(start_idx)
    if endpoint_score(first[2]) > endpoint_score(first[0]):
        direction = "reverse"
        current = first[0]
    else:
        direction = "forward"
        current = first[2]
    path = [first]
    directions = [direction]

    while remaining:
        next_idx = next(
            (
                idx for idx, triple in enumerate(remaining)
                if triple[0] == current or triple[2] == current
            ),
            None,
        )
        if next_idx is None:
            next_idx = max(
                range(len(remaining)),
                key=lambda idx: max(
                    endpoint_score(remaining[idx][0]),
                    endpoint_score(remaining[idx][2]),
                ),
            )
        triple = remaining.pop(next_idx)
        if triple[0] == current:
            direction = "forward"
            current = triple[2]
        elif triple[2] == current:
            direction = "reverse"
            current = triple[0]
        elif endpoint_score(triple[2]) > endpoint_score(triple[0]):
            direction = "reverse"
            current = triple[0]
        else:
            direction = "forward"
            current = triple[2]
        path.append(triple)
        directions.append(direction)
    return path, directions


def question_intent(question: str, answer_type: str, relation_families: Iterable[str]) -> str:
    q_tokens = set(tokenize(question))
    families = [family for family in relation_families if family]
    biomedical_families = [family for family in families if family.startswith("biomedical_")]
    if biomedical_families:
        return biomedical_families[0]
    if q_tokens & {"same", "both"}:
        return "comparison"
    if answer_type == "date":
        return "temporal_attribute"
    if answer_type == "location":
        return "location_attribute"
    if answer_type == "person" and any(family in {"family", "creator", "performer"} for family in families):
        return "person_bridge"
    if len(families) >= 2:
        return "bridge_attribute"
    return "attribute"


def query_routing_features(question: str) -> List[str]:
    """Build query-only features for the learned schema router.

    The feature extractor intentionally has no answer/evidence arguments.  This
    makes the serialized router safe to use on evaluation queries while still
    allowing its parameters to be learned from the induction workload.
    """
    tokens = [
        token for token in tokenize(question)
        if token not in STOPWORDS and len(token) > 1
    ]
    features = [f"u:{token}" for token in tokens]
    features.extend(
        f"b:{left}_{right}" for left, right in zip(tokens, tokens[1:])
    )
    features.append(f"t:{infer_answer_type(question, [])}")
    return features


def build_hierarchical_routing_model(
    plans: Sequence[Mapping[str, Any]],
    dataset: str,
    max_features_per_schema: int = 96,
    negative_weight: float = 1.0,
) -> Dict[str, Any]:
    """Compile positive and wrong-schema Q_ind traces into a schema index."""
    plan_statistics: Dict[str, Counter] = {}
    plan_documents: Dict[str, int] = {}
    plan_negative_statistics: Dict[str, Counter] = {}
    plan_negative_documents: Dict[str, int] = {}
    global_df: Counter = Counter()
    total_documents = 0
    for plan in plans:
        plan_id = str(plan.get("id", "") or "")
        if not plan_id:
            continue
        training = dict(plan.get("routing_training", {}) or {})
        counts = Counter({
            str(key): int(value)
            for key, value in dict(training.get("feature_document_frequency", {}) or {}).items()
            if str(key) and int(value) > 0
        })
        documents = int(training.get("document_count", 0) or 0)
        if not counts:
            for example in plan.get("examples", []) or []:
                if not isinstance(example, Mapping):
                    continue
                features = set(query_routing_features(str(example.get("question", "") or "")))
                counts.update(features)
                documents += 1
        if not counts:
            counts.update(query_routing_features(str(plan.get("question_pattern", "") or "")))
            documents = max(documents, 1)
        plan_statistics[plan_id] = counts
        plan_documents[plan_id] = max(documents, 1)
        plan_negative_statistics[plan_id] = Counter({
            str(key): int(value)
            for key, value in dict(
                training.get("negative_feature_document_frequency", {}) or {}
            ).items()
            if str(key) and int(value) > 0
        })
        plan_negative_documents[plan_id] = int(
            training.get("negative_document_count", 0) or 0
        )
        global_df.update(counts)
        total_documents += max(documents, 1)

    idf = {
        feature: math.log((1.0 + total_documents) / (1.0 + frequency)) + 1.0
        for feature, frequency in global_df.items()
    }

    has_contrastive_negatives = any(plan_negative_documents.values())

    def normalized_vector(
        counts: Mapping[str, int],
        documents: int,
        negative_counts: Mapping[str, int] | None = None,
        negative_documents: int = 0,
    ) -> Dict[str, float]:
        negative_counts = negative_counts or {}
        features = set(counts) | set(negative_counts)

        def signed_weight(feature: str) -> float:
            positive_rate = float(counts.get(feature, 0)) / max(documents, 1)
            negative_rate = float(negative_counts.get(feature, 0)) / max(
                negative_documents, 1
            )
            return (
                positive_rate - max(float(negative_weight), 0.0) * negative_rate
            ) * idf.get(feature, 1.0)

        ranked = sorted(
            features,
            key=lambda feature: abs(signed_weight(feature)),
            reverse=True,
        )[:max(max_features_per_schema, 8)]
        raw = {
            feature: signed_weight(feature)
            for feature in ranked
            if signed_weight(feature) != 0.0
        }
        norm = math.sqrt(sum(value * value for value in raw.values())) or 1.0
        return {feature: round(value / norm, 8) for feature, value in raw.items()}

    plan_vectors: Dict[str, Dict[str, Any]] = {}
    parent_counts: Dict[str, Counter] = defaultdict(Counter)
    parent_documents: Counter = Counter()
    for plan in plans:
        plan_id = str(plan.get("id", "") or "")
        if plan_id not in plan_statistics:
            continue
        intent = str(plan.get("intent", "attribute") or "attribute")
        parent_id = f"intent:{intent}"
        vector = normalized_vector(
            plan_statistics[plan_id],
            plan_documents[plan_id],
            plan_negative_statistics[plan_id],
            plan_negative_documents[plan_id],
        )
        plan_vectors[plan_id] = {
            "parent_id": parent_id,
            "vector": vector,
            "training_documents": plan_documents[plan_id],
            "negative_training_documents": plan_negative_documents[plan_id],
        }
        parent_counts[parent_id].update(plan_statistics[plan_id])
        parent_documents[parent_id] += plan_documents[plan_id]
        if isinstance(plan, dict):
            plan["parent_schema_id"] = parent_id

    parent_vectors = {
        parent_id: normalized_vector(counts, int(parent_documents[parent_id]))
        for parent_id, counts in parent_counts.items()
    }
    retained_features = {
        feature
        for item in plan_vectors.values()
        for feature in item["vector"]
    }
    return {
        "type": (
            "hierarchical_contrastive_tfidf_schema_router"
            if has_contrastive_negatives
            else "hierarchical_tfidf_schema_router"
        ),
        "version": 2 if has_contrastive_negatives else 1,
        "dataset": dataset,
        "training_source": (
            "q_ind_positive_and_ontology_induced_wrong_schema_pairs"
            if has_contrastive_negatives
            else "induction_query_features_only"
        ),
        "inference_uses_gold": False,
        "root_schema_id": "workload_root",
        "idf": {
            feature: round(idf.get(feature, 1.0), 8)
            for feature in sorted(retained_features)
        },
        "parent_vectors": parent_vectors,
        "plan_vectors": plan_vectors,
        "contrastive_training": {
            "enabled": has_contrastive_negatives,
            "negative_weight": float(negative_weight),
            "wrong_schema_pair_count": sum(plan_negative_documents.values()),
        },
        "policy": {
            "parent_top_m": min(3, max(len(parent_vectors), 1)),
            "leaf_top_m": 5,
            "min_score": 0.0,
            "source": "uncalibrated_induction_default",
        },
    }


def build_query_plan_ontology(
    rows: List[Dict[str, Any]],
    dataset: str,
    eval_rows: List[Dict[str, Any]] | None = None,
    max_examples: int = 1000,
    min_support: int = 3,
    min_path_steps: int = 1,
    schema_grouping: str = "lexical",
    retain_rare_multihop: bool = False,
    contrastive_cq_records: Mapping[str, Mapping[str, Any]] | None = None,
) -> Dict[str, Any]:
    if schema_grouping not in {
        "lexical",
        "typed_path",
        "family_path",
        "contract_path",
        "signed_contract_path",
        "signed_family_contract_path",
        "length_adaptive_typed_path",
    }:
        raise ValueError(f"unsupported schema_grouping={schema_grouping}")
    eval_ids = {get_row_id(row) for row in (eval_rows or [])}
    eval_questions = {normalize_text(get_question(row)) for row in (eval_rows or [])}
    buckets: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    used = 0
    skipped_eval_overlap = 0
    skipped_no_evidence = 0
    contrastive_cq_records = contrastive_cq_records or {}
    contrastive_records_used = 0

    for row in rows:
        if used >= max_examples:
            break
        row_id = get_row_id(row)
        contrastive_cq = contrastive_cq_records.get(row_id, {}) or {}
        question = get_question(row)
        if not question:
            continue
        if row_id in eval_ids or normalize_text(question) in eval_questions:
            skipped_eval_overlap += 1
            continue
        structural_repair = dict(
            contrastive_cq.get("structural_repair", {}) or {}
        )
        repaired_path = []
        if structural_repair.get("complete"):
            repaired_path = [
                tuple(normalize_text(part) for part in triple)
                for triple in structural_repair.get("evidence_path", []) or []
                if isinstance(triple, (list, tuple))
                and len(triple) == 3
                and all(normalize_text(part) for part in triple)
            ]
        evidences = repaired_path or get_evidence_triples(row)
        if len(evidences) < min_path_steps:
            skipped_no_evidence += 1
            continue
        answers = get_answers(row)
        if schema_grouping in {
            "signed_contract_path",
            "signed_family_contract_path",
        }:
            if repaired_path:
                ordered_path = repaired_path
                stored_directions = list(
                    structural_repair.get("directions", []) or []
                )
                path_directions = [
                    str(stored_directions[idx] or "forward")
                    if idx < len(stored_directions)
                    else "forward"
                    for idx in range(len(ordered_path))
                ]
            else:
                ordered_path, path_directions = signed_evidence_path(
                    evidences, question, answers
                )
        else:
            ordered_path = ordered_evidence_path(evidences, question, answers)
            path_directions = ["forward"] * len(ordered_path)
        relations = [triple[1] for triple in ordered_path]
        relation_signatures = [
            relation_signature(relation, dataset=dataset) for relation in relations
        ]
        relation_signatures = [signature for signature in relation_signatures if signature]
        relation_families = [
            factorized_relation_family(
                signature,
                collapse_open_class=(
                    schema_grouping == "signed_family_contract_path"
                ),
            )
            for signature in relation_signatures
        ]
        if not dataset.startswith("bioasq"):
            relation_families = [
                "attribute" if family.startswith("biomedical_") else family
                for family in relation_families
            ]
        relation_families = [family for family in relation_families if family]
        if len(relation_families) < min_path_steps:
            skipped_no_evidence += 1
            continue
        answer_type = infer_answer_type(question, answers)
        endpoints = []
        for triple in ordered_path:
            endpoints.extend([triple[0], triple[2]])
        triggers = schema_query_terms(question, endpoints, answers)
        intent = question_intent(question, answer_type, relation_families)





        if schema_grouping in {
            "contract_path",
            "signed_contract_path",
            "signed_family_contract_path",
        }:
            branch_structure = dict(contrastive_cq.get("branch_structure", {}) or {})
            branch_kind = str(
                branch_structure.get("kind")
                or branch_structure.get("type")
                or branch_structure.get("branch_shape")
                or ("comparison" if intent == "comparison" else "chain")
            )
            path_length = min(len(relation_families), 4)
            if schema_grouping == "signed_contract_path":
                signed_relations = tuple(zip(
                    relation_signatures[:4], path_directions[:4]
                ))
                key = (
                    intent,
                    answer_type,
                    path_length,
                    branch_kind,
                    signed_relations,
                )
            elif schema_grouping == "signed_family_contract_path":
                signed_families = tuple(zip(
                    relation_families[:4], path_directions[:4]
                ))
                key = (
                    intent,
                    answer_type,
                    path_length,
                    branch_kind,
                    signed_families,
                )
            else:


                key = (intent, answer_type, path_length, branch_kind)
        elif schema_grouping == "family_path":
            key = (intent, answer_type, tuple(relation_families[:4]))
        elif schema_grouping == "length_adaptive_typed_path" and len(relation_signatures) >= 3:
            key = (intent, answer_type, tuple(relation_families[:4]))
        elif dataset.startswith("bioasq") or schema_grouping == "typed_path":
            key = (intent, answer_type, tuple(relation_signatures[:4]))
        else:
            key = (intent, answer_type, tuple(relation_signatures[:4]), tuple(triggers[:6]))
        if key not in buckets:
            buckets[key] = {
                "intent": intent,
                "answer_type": answer_type,
                "relation_families": relation_families[:4],
                "relation_family_counter": [Counter() for _ in relation_families[:4]],
                "relation_signatures": relation_signatures[:4],
                "relation_alias_counter": [Counter() for _ in relation_families[:4]],
                "relation_signature_counter": [
                    Counter() for _ in relation_families[:4]
                ],
                "direction_counter": [Counter() for _ in relation_families[:4]],
                "trigger_counter": Counter(),
                "support": 0,
                "examples": [],
                "routing_feature_document_frequency": Counter(),
                "cluster_counter": Counter(),
                "hard_negative_type_counter": Counter(),
                "contrastive_constraints": Counter(),
                "source_split_counter": Counter(),
                "structural_architect_seed_support": 0,
                "ontology_induced_negative_count": 0,
            }
        bucket = buckets[key]
        bucket["support"] += 1
        for idx, family in enumerate(relation_families[:len(bucket["relation_family_counter"])]) :
            bucket["relation_family_counter"][idx][family] += 1
        for idx, relation in enumerate(relations[:len(bucket["relation_alias_counter"])]):
            bucket["relation_alias_counter"][idx][relation] += 1
            if idx < len(relation_signatures):
                bucket["relation_signature_counter"][idx][relation_signatures[idx]] += 1
            if idx < len(path_directions):
                bucket["direction_counter"][idx][path_directions[idx]] += 1
        bucket["trigger_counter"].update(triggers)
        bucket["routing_feature_document_frequency"].update(
            set(query_routing_features(question))
        )
        if contrastive_cq:
            contrastive_records_used += 1
            contrastive_source_split = str(
                (contrastive_cq.get("structural_repair", {}) or {}).get(
                    "source_split"
                )
                or (contrastive_cq.get("provenance", {}) or {}).get("split")
                or "q_ind"
            )
            bucket["source_split_counter"][contrastive_source_split] += 1
            bucket["cluster_counter"][str(contrastive_cq.get("cluster_id", ""))] += 1
            bucket["hard_negative_type_counter"].update(
                contrastive_cq.get("hard_negative_types", []) or []
            )
            bucket["contrastive_constraints"].update(
                contrastive_cq.get("hard_negative_failure_reasons", []) or []
            )
            bucket["structural_architect_seed_support"] += int(bool(
                contrastive_cq.get("structural_architect_seed", False)
            ))
            bucket["ontology_induced_negative_count"] += len(
                contrastive_cq.get("ontology_induced_hard_negatives", []) or []
            )
        if len(bucket["examples"]) < 8:
            supporting_passages = get_supporting_passages(row, limit=3)
            example = {
                "id": row_id,
                "question": question,
                "answers": answers[:4],
                "evidence_path": [list(triple) for triple in ordered_path[:4]],
                "supporting_passages": supporting_passages,
            }
            if contrastive_cq:
                example["contrastive_cq"] = {
                    "cluster_id": contrastive_cq.get("cluster_id"),
                    "structural_architect_seed": bool(
                        contrastive_cq.get("structural_architect_seed", False)
                    ),
                    "coverage_cqs": contrastive_cq.get("coverage_cqs", []),
                    "contrastive_cqs": contrastive_cq.get("contrastive_cqs", []),
                    "hard_negatives": contrastive_cq.get("hard_negatives", []),
                    "required_roles": contrastive_cq.get("required_roles", []),
                    "intermediate_bindings": contrastive_cq.get(
                        "intermediate_bindings", []
                    ),
                    "branch_structure": contrastive_cq.get("branch_structure", {}),
                    "structural_constraints": contrastive_cq.get(
                        "structural_constraints", {}
                    ),
                }
            bucket["examples"].append(example)
        used += 1

    plans = []
    for idx, bucket in enumerate(sorted(buckets.values(), key=lambda item: item["support"], reverse=True)):
        rare_multihop = (
            retain_rare_multihop
            and int(bucket["support"]) >= 1
            and len(bucket["relation_signatures"]) >= 3
        )
        if bucket["support"] < min_support and not rare_multihop:
            continue
        path = []
        for step_idx, initial_family in enumerate(bucket["relation_families"]):
            family_counts = bucket.get("relation_family_counter", [])
            family = (
                family_counts[step_idx].most_common(1)[0][0]
                if step_idx < len(family_counts) and family_counts[step_idx]
                else initial_family
            )
            family_alternatives = (
                [value for value, _ in family_counts[step_idx].most_common(8)]
                if step_idx < len(family_counts)
                else [family]
            )
            aliases = [relation for relation, _ in bucket["relation_alias_counter"][step_idx].most_common(8)]
            signature_counts = bucket.get("relation_signature_counter", [])
            representative_signature = (
                signature_counts[step_idx].most_common(1)[0][0]
                if step_idx < len(signature_counts) and signature_counts[step_idx]
                else bucket["relation_signatures"][step_idx]
                if step_idx < len(bucket.get("relation_signatures", []))
                else family
            )
            path.append({
                "step": step_idx,
                "from": "anchor" if step_idx == 0 else f"node_{step_idx}",
                "to": "answer" if step_idx == len(bucket["relation_families"]) - 1 else f"node_{step_idx + 1}",
                "relation_family": family,
                "relation_family_alternatives": family_alternatives,
                "relation_signature": representative_signature,
                "relation_signature_alternatives": (
                    [value for value, _ in signature_counts[step_idx].most_common(12)]
                    if step_idx < len(signature_counts)
                    else [representative_signature]
                ),
                "relation_aliases": aliases,
                "direction": (
                    bucket["direction_counter"][step_idx].most_common(1)[0][0]
                    if bucket.get("direction_counter")
                    and step_idx < len(bucket["direction_counter"])
                    and bucket["direction_counter"][step_idx]
                    else "forward"
                ),
            })
        triggers = [term for term, _ in bucket["trigger_counter"].most_common(16)]
        support = int(bucket["support"])
        confidence = min(0.99, 0.70 + min(support, 30) / 100.0)
        precision = min(0.98, 0.82 + min(support, 20) / 100.0)
        plans.append({
            "id": f"{dataset}_plan_{idx}",
            "intent": bucket["intent"],
            "question_pattern": " ".join(triggers[:8]) or bucket["intent"],
            "triggers": triggers,
            "anchor_slots": [{"name": "anchor", "type": "entity", "source": "query_mention"}],
            "path": path,
            "answer_type": bucket["answer_type"],
            "support": support,
            "confidence": round(confidence, 4),
            "precision": round(precision, 4),
            "examples": bucket["examples"],
            "routing_training": {
                "document_count": support,
                "feature_document_frequency": dict(
                    bucket["routing_feature_document_frequency"].most_common(256)
                ),
                "uses_query_text_only": True,
            },
            "contrastive_induction": {
                "source_split": (
                    bucket["source_split_counter"].most_common(1)[0][0]
                    if bucket["source_split_counter"]
                    else "q_ind"
                ),
                "cluster_ids": [
                    value for value, _ in bucket["cluster_counter"].most_common()
                    if value
                ],
                "hard_negative_types": [
                    value for value, _ in bucket["hard_negative_type_counter"].most_common()
                ],
                "failure_constraints": [
                    value for value, _ in bucket["contrastive_constraints"].most_common(32)
                ],
                "structural_architect_seed_support": int(
                    bucket["structural_architect_seed_support"]
                ),
                "real_corpus_negatives_only": (
                    int(bucket["ontology_induced_negative_count"]) == 0
                ),
                "ontology_induced_negative_count": int(
                    bucket["ontology_induced_negative_count"]
                ),
            },
        })




    plans_by_id = {str(plan.get("id", "")): plan for plan in plans}
    contrastive_wrong_schema_pairs = 0
    for record in contrastive_cq_records.values():
        if not isinstance(record, Mapping):
            continue
        features = set(query_routing_features(str(record.get("query", "") or "")))
        if not features:
            continue
        for negative in record.get("ontology_induced_hard_negatives", []) or []:
            if not isinstance(negative, Mapping):
                continue
            plan = plans_by_id.get(str(negative.get("schema_id", "") or ""))
            if plan is None:
                continue
            training = plan.setdefault("routing_training", {})
            counts = Counter(training.get("negative_feature_document_frequency", {}) or {})
            counts.update(features)
            training["negative_feature_document_frequency"] = dict(
                counts.most_common(256)
            )
            training["negative_document_count"] = int(
                training.get("negative_document_count", 0) or 0
            ) + 1
            training["negative_label"] = "ontology_induced_wrong_schema"
            contrastive_wrong_schema_pairs += 1

    hierarchy = {
        "root": {"id": "workload_root", "children": []},
        "parents": [],
    }
    parent_children: Dict[str, List[str]] = defaultdict(list)
    for plan in plans:
        parent_id = f"intent:{plan.get('intent', 'attribute')}"
        plan["parent_schema_id"] = parent_id
        parent_children[parent_id].append(str(plan.get("id", "")))
    hierarchy["root"]["children"] = sorted(parent_children)
    hierarchy["parents"] = [
        {"id": parent_id, "children": children}
        for parent_id, children in sorted(parent_children.items())
    ]
    routing_model = build_hierarchical_routing_model(plans, dataset)
    architect_skeleton = [
        {
            "schema_id": str(plan.get("id", "") or ""),
            "intent": str(plan.get("intent", "") or ""),
            "answer_type": str(plan.get("answer_type", "") or ""),
            "path": deepcopy(plan.get("path", [])),
            "seed_support": int(
                (plan.get("contrastive_induction", {}) or {}).get(
                    "structural_architect_seed_support", 0
                ) or 0
            ),
        }
        for plan in plans
        if int(
            (plan.get("contrastive_induction", {}) or {}).get(
                "structural_architect_seed_support", 0
            ) or 0
        ) > 0
    ]

    return {
        "version": 1,
        "kind": "query_conditioned_micro_ontology",
        "dataset": dataset,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "builder": "scripts/build_query_plan_ontology.py",
        "supervision": "train_query_answer_evidence_triples",
        "schema_hierarchy": hierarchy,
        "structural_architect_skeleton": architect_skeleton,
        "refinement_contract": {
            "skeleton_source": "diverse_q_ind_seed",
            "seed_only_initializes_skeleton": True,
            "all_q_ind_queries_enter_refinement": True,
        },
        "routing_model": routing_model,
        "plans": plans,
        "diagnostics": {
            "train_rows_seen": len(rows),
            "train_rows_used": used,
            "eval_overlap_skipped": skipped_eval_overlap,
            "no_evidence_skipped": skipped_no_evidence,
            "plan_count": len(plans),
            "min_support": min_support,
            "max_examples": max_examples,
            "min_path_steps": min_path_steps,
            "schema_grouping": schema_grouping,
            "retain_rare_multihop": bool(retain_rare_multihop),
            "contrastive_cq_record_count": len(contrastive_cq_records),
            "contrastive_cq_records_used": contrastive_records_used,
            "contrastive_wrong_schema_pair_count": (
                contrastive_wrong_schema_pairs
            ),
            "contrastive_hard_negatives_grounded_in_corpus": bool(
                contrastive_cq_records
            ),
        },
    }


def routing_parameters_for_scores(
    policy: Mapping[str, Any],
    scores: Sequence[float],
    requested_top_k: int,
    min_route_score: float | None = None,
) -> Tuple[int, float, str, float]:
    """Resolve a query-local route budget from score-distribution features."""
    route_limit = max(int(policy.get("leaf_top_m", requested_top_k) or requested_top_k), 1)
    threshold = (
        float(min_route_score)
        if min_route_score is not None
        else float(policy.get("min_score", 0.0) or 0.0)
    )
    route_class = "global"
    margin = 1.0
    if str(policy.get("routing_mode", "")) == "score_margin_adaptive":
        margin = (
            float(scores[0]) - float(scores[1])
            if len(scores) >= 2 else 1.0
        )
        cutoff = float(policy.get("margin_cutoff", 0.0) or 0.0)
        route_class = "decisive" if margin >= cutoff else "ambiguous"
        class_policy = dict(
            dict(policy.get("class_policies", {}) or {}).get(route_class, {}) or {}
        )
        route_limit = max(
            int(class_policy.get("leaf_top_m", route_limit) or route_limit), 1
        )
        if min_route_score is None:
            threshold = float(class_policy.get("min_score", threshold) or 0.0)
    route_limit = min(max(int(requested_top_k), 0), route_limit)
    return route_limit, threshold, route_class, margin


class QueryPlanOntologyMatcher:
    def __init__(self, ontology: Dict[str, Any]):
        self.ontology = ontology or {}
        self.plans = [
            plan for plan in self.ontology.get("plans", [])
            if isinstance(plan, dict) and plan.get("path")
        ]
        self.routing_model = dict(self.ontology.get("routing_model", {}) or {})
        self._plans_by_id = {
            str(plan.get("id", "")): plan for plan in self.plans if plan.get("id")
        }

    @classmethod
    def load(cls, path: str | Path) -> "QueryPlanOntologyMatcher":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls(json.load(handle))

    def restrict_to_plan_ids(self, plan_ids: Iterable[Any]) -> Dict[str, int]:
        """Apply a deployment mask before routing rather than after top-k.

        Filtering only the final top-k can let rejected schemas consume every
        routing slot.  The physical matcher and its learned vectors therefore
        need to expose the same certified relation from the outset.
        """
        allowed = {str(value) for value in plan_ids or [] if value}
        before = len(self.plans)
        self.plans = [
            plan for plan in self.plans
            if str(plan.get("id", "") or "") in allowed
        ]
        self._plans_by_id = {
            str(plan.get("id", "")): plan
            for plan in self.plans
            if plan.get("id")
        }

        plan_vectors = dict(self.routing_model.get("plan_vectors", {}) or {})
        retained_vectors = {
            str(plan_id): value
            for plan_id, value in plan_vectors.items()
            if str(plan_id) in allowed
        }
        if plan_vectors:
            self.routing_model["plan_vectors"] = retained_vectors
            retained_parents = {
                str((value or {}).get("parent_id", "") or "")
                for value in retained_vectors.values()
                if isinstance(value, Mapping)
            }
            parent_vectors = dict(self.routing_model.get("parent_vectors", {}) or {})
            self.routing_model["parent_vectors"] = {
                str(parent_id): value
                for parent_id, value in parent_vectors.items()
                if str(parent_id) in retained_parents
            }
        self.routing_model.setdefault("policy", {})[
            "deployment_schema_ids"
        ] = sorted(self._plans_by_id)
        return {
            "before": before,
            "after": len(self.plans),
            "removed": before - len(self.plans),
        }

    @staticmethod
    def _cosine_query_score(
        features: Sequence[str],
        vector: Mapping[str, Any],
        idf: Mapping[str, Any],
    ) -> float:
        counts = Counter(features)
        raw = {
            feature: float(count) * float(idf.get(feature, 1.0) or 1.0)
            for feature, count in counts.items()
            if feature in vector
        }
        norm = math.sqrt(sum(value * value for value in raw.values())) or 1.0
        return sum(
            (value / norm) * float(vector.get(feature, 0.0) or 0.0)
            for feature, value in raw.items()
        )

    def rank(self, query: str, min_confidence: float = 0.0) -> List[Tuple[float, Dict[str, Any]]]:
        """Rank leaf schemas, using the learned hierarchy when available."""
        if self.routing_model.get("type") in {
            "hierarchical_tfidf_schema_router",
            "hierarchical_contrastive_tfidf_schema_router",
        }:
            idf = dict(self.routing_model.get("idf", {}) or {})
            features = query_routing_features(query)
            parent_scores = [
                (
                    self._cosine_query_score(features, vector, idf),
                    str(parent_id),
                )
                for parent_id, vector in dict(
                    self.routing_model.get("parent_vectors", {}) or {}
                ).items()
            ]
            parent_scores.sort(reverse=True)
            policy = dict(self.routing_model.get("policy", {}) or {})
            parent_top_m = max(int(policy.get("parent_top_m", 1) or 1), 1)
            admitted_parents = {
                parent_id for _, parent_id in parent_scores[:parent_top_m]
            }
            scored: List[Tuple[float, Dict[str, Any]]] = []
            for plan_id, model_plan in dict(
                self.routing_model.get("plan_vectors", {}) or {}
            ).items():
                plan = self._plans_by_id.get(str(plan_id))
                if plan is None or float(plan.get("confidence", 0.0) or 0.0) < min_confidence:
                    continue
                parent_id = str(model_plan.get("parent_id", "") or "")
                if admitted_parents and parent_id not in admitted_parents:
                    continue
                score = self._cosine_query_score(
                    features,
                    dict(model_plan.get("vector", {}) or {}),
                    idf,
                )
                scored.append((score, plan))
            scored.sort(key=lambda item: (item[0], str(item[1].get("id", ""))), reverse=True)
            return scored

        query_tokens = set(tokenize(query))
        if not query_tokens:
            return []
        answer_type = infer_answer_type(query, [])
        biomedical_workload = str(self.ontology.get("dataset", "")).startswith("bioasq")
        scored = []
        for plan in self.plans:
            if float(plan.get("confidence", 0.0) or 0.0) < min_confidence:
                continue
            if (
                biomedical_workload
                and plan.get("answer_type") not in {answer_type, "entity", "biomedical_entity"}
            ):
                continue
            triggers = set(token for trigger in plan.get("triggers", []) for token in tokenize(trigger))
            family_tokens = set(tokenize(" ".join(
                " ".join([step.get("relation_family", ""), step.get("relation_signature", "")])
                for step in plan.get("path", [])
            )))
            overlap = query_tokens & (triggers | family_tokens)
            if not overlap:
                continue
            score = len(overlap) / max(min(len(query_tokens), len(triggers | family_tokens)), 1)
            if answer_type == plan.get("answer_type"):
                score += 0.25
            score += 0.20 * float(plan.get("precision", 0.0) or 0.0)
            score += 0.10 * min(float(plan.get("support", 0.0) or 0.0) / 20.0, 1.0)
            scored.append((score, plan))
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored

    def match(
        self,
        query: str,
        top_k: int = 6,
        min_confidence: float = 0.8,
        min_route_score: float | None = None,
    ) -> List[Dict[str, Any]]:
        query_tokens = set(tokenize(query))
        if not query_tokens:
            return []
        scored = self.rank(query, min_confidence=min_confidence)
        learned = self.routing_model.get("type") in {
            "hierarchical_tfidf_schema_router",
            "hierarchical_contrastive_tfidf_schema_router",
        }
        policy = dict(self.routing_model.get("policy", {}) or {})
        if learned:
            route_limit, route_threshold, route_class, score_margin = (
                routing_parameters_for_scores(
                    policy,
                    [float(score) for score, _ in scored],
                    requested_top_k=top_k,
                    min_route_score=min_route_score,
                )
            )
        else:
            route_limit = max(int(top_k), 0)
            route_threshold = (
                float(min_route_score) if min_route_score is not None else float("-inf")
            )
            route_class = "legacy"
            score_margin = 1.0
        matched = []
        deployment_ids = {
            str(value)
            for value in (policy.get("deployment_schema_ids", []) or [])
            if value
        }
        for score, plan in scored[:route_limit]:
            if score < route_threshold:
                continue
            if deployment_ids and str(plan.get("id", "") or "") not in deployment_ids:
                continue
            matched.append({
                **plan,
                "match_score": float(score),
                "route_method": (
                    str(self.routing_model.get("type")) if learned else "legacy_lexical_router"
                ),
                "route_class": route_class,
                "route_score_margin": float(score_margin),
            })
        condition = self.infer_query_condition(query)
        return self.filter_endpoint_grounded_route_contracts(matched, condition)

    def match_schema_linked(
        self,
        query: str,
        top_k: int = 6,
        min_confidence: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Route over all leaf schemas using the query relation contract."""
        condition = self.infer_query_condition(query)
        expected_signatures = set(condition.get("relation_signatures", []) or [])
        expected_families = set(condition.get("relation_families", []) or [])
        answer_type = str(condition.get("answer_type", "") or "")
        query_tokens = set(tokenize(re.sub(r"[^\w]+", " ", str(query or ""))))
        features = query_routing_features(query)
        idf = dict(self.routing_model.get("idf", {}) or {})
        plan_vectors = dict(self.routing_model.get("plan_vectors", {}) or {})

        scored: List[Tuple[float, Dict[str, Any]]] = []
        for plan in self.plans:
            if float(plan.get("confidence", 0.0) or 0.0) < min_confidence:
                continue
            route_contract_matched, route_contract_detail = (
                self.endpoint_grounded_route_contract_matches(plan, condition)
            )
            if not route_contract_matched:
                continue
            path = list(plan.get("path", []) or [])
            plan_signatures = {
                str(step.get("relation_signature", "") or relation_signature(step.get("relation", "")))
                for step in path
                if step.get("relation_signature") or step.get("relation")
            }
            plan_signatures.update(
                str(value)
                for step in path
                for value in (step.get("relation_signature_alternatives", []) or [])
                if value
            )
            plan_families = {
                str(step.get("relation_family", "") or _relation_family(str(step.get("relation", ""))))
                for step in path
                if step.get("relation_family") or step.get("relation")
            }
            plan_families.update(
                str(value)
                for step in path
                for value in (step.get("relation_family_alternatives", []) or [])
                if value
            )
            exact_hits = expected_signatures & plan_signatures
            family_hits = expected_families & plan_families
            exact_coverage = (
                len(exact_hits) / max(len(expected_signatures), 1)
                if expected_signatures else 0.0
            )
            family_coverage = (
                len(family_hits) / max(len(expected_families), 1)
                if expected_families else 0.0
            )
            if expected_signatures and not exact_hits and not family_hits:
                continue

            plan_id = str(plan.get("id", "") or "")
            model_plan = dict(plan_vectors.get(plan_id, {}) or {})
            lexical_score = self._cosine_query_score(
                features,
                dict(model_plan.get("vector", {}) or {}),
                idf,
            ) if model_plan else 0.0
            triggers = {
                token
                for trigger in list(plan.get("triggers", []) or [])
                for token in tokenize(str(trigger))
            }
            trigger_overlap = len(query_tokens & triggers) / max(
                len(query_tokens & SCHEMA_QUERY_TERMS), 1
            )
            plan_answer_type = str(plan.get("answer_type", "") or "")
            query_answer_family = canonical_answer_type_family(answer_type)
            plan_answer_family = canonical_answer_type_family(plan_answer_type)
            generic_types = {"", "unknown", "entity", "string", "answer"}
            if query_answer_family in generic_types:
                answer_match = 0.75
            elif query_answer_family == plan_answer_family:
                answer_match = 1.0
            elif plan_answer_family in generic_types:
                answer_match = 0.35
            else:
                answer_match = 0.0
            score = (
                0.25 * lexical_score
                + 0.55 * exact_coverage
                + 0.10 * family_coverage
                + 0.06 * min(trigger_overlap, 1.0)
                + 0.04 * answer_match
            )
            scored.append((score, {
                **plan,
                "match_score": float(score),
                "schema_link_score": float(score),
                "schema_relation_coverage": float(exact_coverage),
                "schema_answer_type_compatibility": float(answer_match),
                "schema_expected_relations": sorted(expected_signatures),
                "schema_sha256": plan_schema_sha256(plan),
                "route_method": "schema_linked_leaf_router",
                "query_condition": {
                    **dict(plan.get("query_condition", {}) or {}),
                    **route_contract_detail,
                },
            }))

        scored.sort(
            key=lambda item: (
                item[1].get("schema_relation_coverage", 0.0),
                item[1].get("schema_answer_type_compatibility", 0.0),
                item[0],
                float(item[1].get("precision", 0.0) or 0.0),
                str(item[1].get("id", "")),
            ),
            reverse=True,
        )
        ranked = [plan for _, plan in scored[:max(int(top_k), 0)]]
        deployment_ids = {
            str(value)
            for value in (
                (self.routing_model.get("policy", {}) or {}).get(
                    "deployment_schema_ids", []
                ) or []
            )
            if value
        }
        if deployment_ids:
            ranked = [
                plan for plan in ranked
                if str(plan.get("id", "") or "") in deployment_ids
            ]
        return ranked

    @staticmethod
    def endpoint_grounded_route_contract_matches(
        plan: Dict[str, Any],
        condition: Dict[str, Any],
    ) -> Tuple[bool, Dict[str, Any]]:
        """Apply an endpoint route contract at every ontology ingress.

        Endpoint-grounded schemas are calibrated from historical queries, so
        their accepted signature sets are hard runtime predicates rather than
        ranking features.  Keeping this check independent of a particular
        router prevents semantic, schema-linked, and conditioned routes from
        disagreeing about schema eligibility.
        """
        contract = dict(plan.get("endpoint_grounded_route_contract", {}) or {})
        query_signatures = {
            relation_signature(value)
            for value in condition.get("relation_signatures", []) or []
            if relation_signature(value)
        }
        accepted_signature_sets: List[set[str]] = []
        for values in contract.get("accepted_query_signature_sets", []) or []:
            normalized = {
                relation_signature(value)
                for value in (values or [])
                if relation_signature(value)
            }
            if normalized:
                accepted_signature_sets.append(normalized)

        enabled = contract.get("enabled") is True
        matched_set: set[str] = set()
        matched = True
        if enabled:
            matched = False
            for required in accepted_signature_sets:
                if required.issubset(query_signatures):
                    matched = True
                    matched_set = required
                    break
        detail = {
            "endpoint_grounded_route_contract_enabled": enabled,
            "endpoint_grounded_route_contract_matched": matched,
            "endpoint_grounded_route_required_signatures": sorted(matched_set),
            "endpoint_grounded_route_query_signatures": sorted(query_signatures),
        }
        return matched, detail

    def filter_endpoint_grounded_route_contracts(
        self,
        plans: List[Dict[str, Any]],
        condition: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        filtered: List[Dict[str, Any]] = []
        for plan in plans:
            matched, detail = self.endpoint_grounded_route_contract_matches(
                plan, condition
            )
            if not matched:
                continue
            updated = dict(plan)
            updated["query_condition"] = {
                **dict(plan.get("query_condition", {}) or {}),
                **detail,
            }
            filtered.append(updated)
        return filtered

    def infer_query_condition(self, query: str) -> Dict[str, Any]:
        normalized_query = normalize_text(re.sub(r"[^\w]+", " ", str(query or "")))
        query_tokens = set(tokenize(normalized_query))
        answer_type = infer_answer_type(query, [])

        relation_signatures = set()
        relation_families = set()
        relation_terms = []

        phrase_hints = {
            "biomedical_adverse_effect": ({"adverse", "toxicity", "contraindication"}, set()),
            "biomedical_dosage": ({"dose", "dosage", "administration"}, set()),
            "biomedical_diagnosis": ({"diagnose", "diagnosis", "detect", "screen", "screening", "biomarker"}, set()),
            "biomedical_treatment": ({"treat", "treatment", "therapy", "prevent", "prevention"}, set()),
            "biomedical_causation": ({"cause", "causes", "causation", "etiology"}, set()),
            "biomedical_genetics": ({"gene", "genetic", "mutation", "variant", "allele"}, set()),
            "biomedical_regulation": ({"regulate", "regulation", "activate", "activation", "inhibit", "inhibition", "express", "expression"}, set()),
            "biomedical_interaction": ({"interact", "interaction", "bind", "binding", "receptor", "target"}, set()),
            "biomedical_association": ({"associate", "associated", "association", "correlate", "correlation", "risk"}, set()),
            "biomedical_manifestation": ({"manifestation", "symptom", "phenotype"}, set()),
            "biomedical_epidemiology": ({"incidence", "prevalence", "mortality", "survival"}, set()),
            "biomedical_mechanism": ({"mechanism", "pathway", "function"}, set()),
            "biomedical_classification": ({"classify", "classification", "subtype"}, set()),
            "biomedical_definition": ({"define", "definition", "describe", "abbreviation"}, set()),
            "biomedical_composition": ({"compose", "composition", "contain", "include", "component"}, set()),
            "biomedical_localization": ({"locate", "located", "localization", "secrete"}, set()),
            "biomedical_inheritance": ({"inherit", "inheritance", "autosomal", "dominant", "recessive"}, set()),
            "biomedical_comparison": ({"compare", "comparison", "difference"}, set()),
            "birth_date": ({"born", "birth", "birthday"}, {"when", "date", "year"}),
            "birth_place": ({"born", "birth", "birthplace"}, {"where", "place", "city", "country", "county", "state", "region", "part"}),
            "death_date": ({"died", "death", "die"}, {"when", "date", "year"}),
            "death_place": ({"died", "death", "die"}, {"where", "place", "city", "country", "county", "state", "region", "part"}),
            "director": ({"director", "directed"}, set()),
            "author": ({"author", "writer", "wrote", "written"}, set()),
            "composer": ({"composer", "composed"}, set()),
            "actor": ({"actor", "actress", "starred", "starring", "played"}, set()),
            "release_date": ({"released", "release"}, {"when", "date", "year"}),
            "publication": ({"published", "publication", "publisher"}, set()),
            "producer": ({"producer", "produced"}, set()),
            "family_father": ({"father"}, set()),
            "family_mother": ({"mother"}, set()),
            "family_spouse": ({"spouse", "wife", "husband", "married"}, set()),
            "location_country": ({"country", "nationality", "citizenship"}, set()),
            "location_city": ({"city", "town"}, set()),
            "location_region": ({"county", "state", "province", "region"}, set()),
            "location_part_of": ({"part"}, {"county", "state", "province", "region", "country", "city", "place", "location"}),
            "headquarters_location": ({"headquarters", "headquartered"}, set()),
            "occupation": ({"occupation", "profession", "job"}, set()),
            "education": ({"educated", "education", "graduated", "study", "studies", "studied", "studying", "attended", "alma", "school", "university"}, set()),
            "record_label": ({"signed", "label", "record"}, set()),
            "team_membership": ({"team", "club", "played"}, set()),
            "employment": ({"employed", "worked", "served"}, set()),
            "membership": ({"member", "affiliated"}, set()),
            "founder": ({"founded", "founder"}, set()),
            "award": ({"award", "won"}, set()),
            "nationality": ({"nationality", "citizenship"}, set()),
        }
        for signature, (primary_terms, context_terms) in phrase_hints.items():
            if query_tokens & primary_terms and (not context_terms or query_tokens & context_terms):
                relation_signatures.add(signature)
                family = _relation_family(signature.replace("_", " ")) or signature.split("_", 1)[0]
                if family in QUERY_PLAN_RELATION_FAMILIES:
                    relation_families.add(family)
                relation_terms.extend(sorted(primary_terms & query_tokens))

        for token in sorted(query_tokens & QUERY_PLAN_RELATION_TERMS):
            signature = relation_signature(token)
            family = _relation_family(token)
            if family not in QUERY_PLAN_RELATION_FAMILIES:
                continue
            if signature and signature not in GENERIC_SCHEMA_NOUNS:
                relation_signatures.add(signature)
            relation_families.add(family)
            relation_terms.append(token)







        if answer_type == "date" and query_tokens & {"born", "birth", "birthday"}:
            relation_signatures.discard("birth")
            relation_signatures.add("birth_date")
            relation_families.add("birth")
        if answer_type == "date" and query_tokens & {"died", "death", "die"}:
            relation_signatures.discard("death")
            relation_signatures.add("death_date")
            relation_families.add("death")
        if re.search(r"\bcause\s+of\s+death\b", normalized_query):
            relation_signatures.add("biomedical_causation")
            relation_families.add("biomedical_causation")
        person_role_signatures = {
            "director", "author", "composer", "actor", "producer",
            "family_father", "family_mother", "family_spouse",
        }
        asks_person_origin = bool(
            "country" in query_tokens
            and (
                re.search(r"\b(?:from|nationality|citizenship)\b", normalized_query)
                or relation_signatures & person_role_signatures
            )
            and not re.search(r"\b(?:located|location|part\s+of)\b", normalized_query)
        )
        if asks_person_origin:
            relation_signatures.add("nationality")
            relation_families.add("nationality")

        if answer_type == "date":
            condition_answer_hints = ["date"]
        elif answer_type == "location":
            condition_answer_hints = ["location"]
        elif answer_type == "person":
            condition_answer_hints = ["person"]
        elif answer_type in {"boolean", "quantity", "gene", "drug", "protein", "disease", "biomedical_entity", "list", "definition"}:
            condition_answer_hints = [answer_type]
        else:
            condition_answer_hints = []

        return {
            "answer_type": answer_type,
            "answer_hints": condition_answer_hints,
            "query_tokens": sorted(query_tokens),
            "relation_signatures": sorted(relation_signatures),
            "relation_families": sorted(relation_families),
            "relation_terms": sorted(set(relation_terms)),
        }

    def condition_plans(
        self,
        query: str,
        plans: List[Dict[str, Any]],
        top_k: int = 6,
        min_condition_score: float = 0.0,
        require_relation_overlap: bool = False,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        condition = self.infer_query_condition(query)
        if not plans:
            return [], condition

        query_signatures = set(condition.get("relation_signatures", []) or [])
        query_families = set(condition.get("relation_families", []) or [])
        answer_type = condition.get("answer_type", "")
        scored = []
        for plan in plans:
            plan_path = plan.get("path", []) or []
            plan_signatures = {
                relation_signature(step.get("relation_signature", "") or step.get("relation_family", ""))
                for step in plan_path
            }
            plan_signatures.update(
                relation_signature(value)
                for step in plan_path
                for value in (step.get("relation_signature_alternatives", []) or [])
            )
            plan_signatures.discard("")
            plan_families = {
                _relation_family(step.get("relation_family", "") or step.get("relation_signature", ""))
                for step in plan_path
            }
            plan_families.update(
                _relation_family(str(value))
                for step in plan_path
                for value in (step.get("relation_family_alternatives", []) or [])
            )
            plan_families.discard("")

            route_contract_matched, route_contract_detail = (
                self.endpoint_grounded_route_contract_matches(plan, condition)
            )
            if not route_contract_matched:
                continue

            signature_overlap = query_signatures & plan_signatures
            family_overlap = query_families & plan_families
            relation_overlap = len(signature_overlap) + 0.5 * len(family_overlap)
            answer_match = 1.0 if answer_type and answer_type == plan.get("answer_type") else 0.0
            trigger_tokens = set(token for trigger in plan.get("triggers", []) for token in tokenize(trigger))
            trigger_overlap = len(set(condition.get("query_tokens", [])) & trigger_tokens)
            base_score = float(plan.get("match_score", 0.0) or 0.0)
            precision = float(plan.get("precision", 0.0) or 0.0)
            plan_scale = float(plan.get("lift_weight_scale", 1.0) or 1.0)

            condition_score = (
                base_score
                + 0.35 * relation_overlap
                + 0.20 * answer_match
                + 0.03 * min(trigger_overlap, 5)
                + 0.10 * precision
            ) * max(plan_scale, 0.0)

            if require_relation_overlap and query_signatures and relation_overlap <= 0:
                continue
            if condition_score < min_condition_score:
                continue

            updated = {**plan}
            updated["query_condition"] = {
                "answer_type": answer_type,
                "relation_signature_overlap": sorted(signature_overlap),
                "relation_family_overlap": sorted(family_overlap),
                "trigger_overlap": sorted(set(condition.get("query_tokens", [])) & trigger_tokens)[:12],
                "condition_score": round(condition_score, 6),
                **route_contract_detail,
            }
            updated["match_score"] = float(condition_score)
            scored.append((condition_score, updated))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [plan for _, plan in scored[:top_k]], condition
