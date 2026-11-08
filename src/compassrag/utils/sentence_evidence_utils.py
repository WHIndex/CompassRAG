import re
from collections import defaultdict
from typing import Any, Dict, List, Tuple


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does",
    "for", "from", "had", "has", "have", "he", "her", "his", "how", "in",
    "is", "it", "its", "of", "on", "or", "she", "that", "the", "their",
    "there", "this", "to", "was", "were", "what", "when", "where", "which",
    "who", "whom", "whose", "why", "with",
}

TOKEN_RE = re.compile(r"[a-z0-9]+")
YEAR_RE = re.compile(r"\b(?:1[0-9]{3}|20[0-9]{2}|21[0-9]{2})\b")
NUMBER_RE = re.compile(r"\b\d+(?:,\d{3})*(?:\.\d+)?\b")
MONTH_RE = re.compile(
    r"\b(?:january|february|march|april|may|june|july|august|september|"
    r"october|november|december|jan\.|feb\.|mar\.|apr\.|jun\.|jul\.|"
    r"aug\.|sep\.|sept\.|oct\.|nov\.|dec\.)\b",
    re.IGNORECASE,
)
TITLE_CASE_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+(?:of|the|and|de|la|[A-Z][a-z]+)){0,5}\b")

LOCATION_WORD_RE = re.compile(
    r"\b(?:city|town|village|county|state|province|country|region|district|"
    r"capital|river|lake|mountain|island|municipality|prefecture|kingdom|"
    r"republic|territory)\b",
    re.IGNORECASE,
)
ORG_WORD_RE = re.compile(
    r"\b(?:university|college|school|institute|company|corporation|inc\.?|ltd\.?|"
    r"club|team|party|association|organization|organisation|agency|committee|"
    r"ministry|department|foundation|league|group|band)\b",
    re.IGNORECASE,
)
WORK_WORD_RE = re.compile(
    r"\b(?:film|movie|album|song|single|book|novel|poem|play|series|episode|"
    r"opera|work|publication|magazine|newspaper|game)\b",
    re.IGNORECASE,
)
OCCUPATION_WORD_RE = re.compile(
    r"\b(?:actor|actress|singer|writer|author|director|producer|composer|poet|"
    r"politician|scientist|engineer|artist|painter|footballer|player|coach|"
    r"journalist|historian|professor|mathematician|physician|lawyer|designer|"
    r"architect|musician|rapper|novelist|screenwriter|businessperson)\b",
    re.IGNORECASE,
)
NATIONALITY_WORD_RE = re.compile(
    r"\b(?:nationality|citizenship|citizen|national|afghan|albanian|algerian|"
    r"american|argentine|armenian|australian|austrian|belgian|brazilian|"
    r"british|bulgarian|canadian|chilean|chinese|colombian|croatian|cuban|"
    r"czech|danish|dutch|egyptian|english|finnish|french|german|ghanaian|"
    r"greek|hungarian|indian|indonesian|iranian|irish|israeli|italian|"
    r"jamaican|japanese|korean|malaysian|mexican|moroccan|nigerian|"
    r"norwegian|pakistani|polish|portuguese|romanian|russian|scottish|"
    r"serbian|singaporean|spanish|swedish|swiss|syrian|taiwanese|thai|"
    r"turkish|ukrainian|vietnamese|welsh)\b",
    re.IGNORECASE,
)


def _doc_title(doc: str) -> str:
    if not doc:
        return ""
    return str(doc).split("\n", 1)[0].strip()


def _doc_body(doc: str) -> str:
    if not doc:
        return ""
    parts = str(doc).split("\n", 1)
    return parts[1] if len(parts) > 1 else parts[0]


def _tokens(text: Any) -> List[str]:
    return [
        token
        for token in TOKEN_RE.findall(str(text or "").lower())
        if len(token) > 1 and token not in STOPWORDS
    ]


def _token_set(text: Any) -> set:
    return set(_tokens(text))


def _split_sentences(text: str, max_body_chars: int = 8000) -> List[str]:
    text = re.sub(r"\s+", " ", str(text or "")[:max_body_chars]).strip()
    if not text:
        return []

    pieces = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", text)
    sentences = []
    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        if len(piece) <= 420:
            sentences.append(piece)
            continue
        clauses = re.split(r"(?<=[,;:])\s+", piece)
        current = ""
        for clause in clauses:
            if len(current) + len(clause) + 1 <= 420:
                current = f"{current} {clause}".strip()
            else:
                if current:
                    sentences.append(current)
                current = clause
        if current:
            sentences.append(current[:420].strip())
    return sentences


def infer_answer_target_type(question: str) -> str:
    q = re.sub(r"\s+", " ", str(question or "").lower()).strip()



    if re.match(r"^(?:who|whose)\b", q):
        return "person"
    if re.match(r"^(?:when|on what date|on which date|in what year|in which year)\b", q):
        return "date"
    if re.match(r"^(?:how many|how much|how long|how old|what percentage|what proportion)\b", q):
        return "number"
    if re.match(r"^where\b", q):
        return "location"
    if re.match(
        r"^(?:which|what) (?:film|movie|album|book|novel|song|work)\b",
        q,
    ):
        return "work"
    if re.match(
        r"^(?:which|what) (?:company|organization|organisation|team|university|school)\b",
        q,
    ):
        return "organization"
    if re.search(r"\b(?:what year|which year|what date|date of (?:birth|death))\b", q):
        return "date"
    if re.search(r"\b(?:number of|population|count|total|percentage)\b", q) or re.match(
        r"^what (?:is|was) (?:the )?age\b", q
    ):
        return "number"
    if re.search(r"\b(?:nationality|citizenship|citizen|national)\b", q):
        return "nationality"
    if re.search(r"\b(?:where|which city|which country|which state|which county|capital|birthplace|located)\b", q):
        return "location"
    if re.search(r"\b(?:which company|which organization|which organisation|which team|which university|which school)\b", q):
        return "organization"
    if re.search(r"\b(?:which film|which movie|which album|which book|which novel|which song|what film|what album|what book)\b", q):
        return "work"
    if re.search(r"\b(?:who|whose|person|director|author|writer|founder|composer|producer|creator|designer)\b", q):
        return "person"
    if re.search(r"\b(?:occupation|profession|job|known for)\b", q):
        return "occupation"
    return "factoid"


def _target_signal(sentence: str, title: str, target_type: str) -> float:
    text = f"{title} {sentence}"
    if target_type == "date":
        return 1.0 if (YEAR_RE.search(text) or MONTH_RE.search(text)) else 0.0
    if target_type == "number":
        return 1.0 if NUMBER_RE.search(text) else 0.0
    if target_type == "location":
        if LOCATION_WORD_RE.search(text):
            return 1.0
        return 0.45 if TITLE_CASE_RE.search(sentence) else 0.0
    if target_type == "person":
        return 1.0 if TITLE_CASE_RE.search(sentence) else 0.0
    if target_type == "organization":
        if ORG_WORD_RE.search(text):
            return 1.0
        return 0.45 if TITLE_CASE_RE.search(sentence) else 0.0
    if target_type == "work":
        if WORK_WORD_RE.search(text) or '"' in sentence or "'" in sentence:
            return 1.0
        return 0.45 if TITLE_CASE_RE.search(sentence) else 0.0
    if target_type == "occupation":
        return 1.0 if OCCUPATION_WORD_RE.search(text) else 0.0
    if target_type == "nationality":
        return 1.0 if NATIONALITY_WORD_RE.search(text) else 0.0
    return 0.35 if (TITLE_CASE_RE.search(sentence) or YEAR_RE.search(sentence) or NUMBER_RE.search(sentence)) else 0.0


def _overlap_ratio(left: set, right: set) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / max(1, len(left))


def _format_sentence_evidence_doc(target_type: str, selected: List[Dict[str, Any]]) -> str:
    lines = [
        "Typed Evidence Grounding",
        f"Target answer type: {target_type}",
        "High-signal evidence sentences:",
    ]
    for item in selected:
        lines.append(f"- {item['title']}: {item['sentence']}")
    return "\n".join(lines)


def build_typed_sentence_evidence(
    question: str,
    docs: List[str],
    enabled: bool = False,
    max_sentences_per_doc: int = 2,
    max_total_sentences: int = 16,
    keep_full_top_docs: int = 5,
    min_sentence_score: float = 0.18,
    bridge_cues: bool = False,
    chain_preserve: bool = True,
    chain_preserve_doc_threshold: int = 5,
) -> Tuple[List[str], Dict[str, Any]]:
    """Create a typed, sentence-level evidence pack before QA.

    The module does not change retrieval results. It extracts high-signal
    answer-bearing sentences from the QA evidence pool. For single-hop
    attribute questions it prepends a compact typed guide. For bridge or
    expanded-evidence questions it preserves the original chain of full
    passages and inserts the typed guide after the leading evidence docs.
    """
    docs = docs or []
    info: Dict[str, Any] = {
        "enabled": bool(enabled),
        "applied": False,
        "reason": "disabled",
        "target_type": infer_answer_target_type(question),
        "original_doc_count": len(docs),
        "final_doc_count": len(docs),
        "selected_sentence_count": 0,
        "keep_full_top_docs": keep_full_top_docs,
        "max_sentences_per_doc": max_sentences_per_doc,
        "max_total_sentences": max_total_sentences,
        "min_sentence_score": min_sentence_score,
        "bridge_cues": bool(bridge_cues),
        "chain_preserve": bool(chain_preserve),
        "chain_preserve_doc_threshold": chain_preserve_doc_threshold,
        "selected_titles": [],
        "compression_ratio": 1.0,
    }

    if not enabled:
        return docs, info
    if not docs:
        info["reason"] = "no_docs"
        return docs, info

    question_tokens = _token_set(question)
    if not question_tokens:
        info["reason"] = "no_question_tokens"
        return docs, info

    max_sentences_per_doc = max(1, int(max_sentences_per_doc))
    max_total_sentences = max(1, int(max_total_sentences))
    keep_full_top_docs = max(0, int(keep_full_top_docs))
    chain_preserve_doc_threshold = max(1, int(chain_preserve_doc_threshold))
    min_sentence_score = max(0.0, float(min_sentence_score))

    title_token_union = set()
    for doc in docs[:max(1, keep_full_top_docs)]:
        title_token_union.update(_token_set(_doc_title(doc)))

    scored = []
    per_doc_candidates: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for doc_index, doc in enumerate(docs):
        title = _doc_title(doc)
        body = _doc_body(doc)
        title_tokens = _token_set(title)
        rank_prior = 1.0 / (doc_index + 1)
        for sentence_index, sentence in enumerate(_split_sentences(body)):
            sentence_tokens = _token_set(sentence)
            if not sentence_tokens:
                continue

            question_overlap = _overlap_ratio(question_tokens, sentence_tokens)
            title_overlap = _overlap_ratio(question_tokens, title_tokens)
            anchor_overlap = _overlap_ratio(title_token_union | question_tokens, sentence_tokens | title_tokens)
            target_signal = _target_signal(sentence, title, info["target_type"])
            early_sentence_prior = 1.0 / (sentence_index + 1)

            score = (
                0.38 * question_overlap
                + 0.18 * title_overlap
                + 0.16 * anchor_overlap
                + 0.18 * target_signal
                + 0.06 * rank_prior
                + 0.04 * early_sentence_prior
            )
            if score < min_sentence_score:
                continue
            item = {
                "doc_index": doc_index,
                "sentence_index": sentence_index,
                "title": title,
                "sentence": sentence,
                "score": score,
                "question_overlap": question_overlap,
                "target_signal": target_signal,
            }
            scored.append(item)
            per_doc_candidates[doc_index].append(item)

    if not scored:
        info["reason"] = "no_high_signal_sentences"
        return docs, info

    scored.sort(key=lambda item: (-item["score"], item["doc_index"], item["sentence_index"]))

    selected = []
    per_doc_selected = defaultdict(int)
    seen_sentences = set()
    for item in scored:
        if len(selected) >= max_total_sentences:
            break
        if per_doc_selected[item["doc_index"]] >= max_sentences_per_doc:
            continue
        sentence_key = re.sub(r"\W+", " ", item["sentence"].lower()).strip()
        if sentence_key in seen_sentences:
            continue
        selected.append(item)
        seen_sentences.add(sentence_key)
        per_doc_selected[item["doc_index"]] += 1

    if not selected:
        info["reason"] = "no_selected_sentences"
        return docs, info

    selected.sort(key=lambda item: (item["doc_index"], item["sentence_index"]))
    evidence_doc = _format_sentence_evidence_doc(info["target_type"], selected)

    preserve_chain = bool(chain_preserve) and (
        bool(bridge_cues) or len(docs) > chain_preserve_doc_threshold
    )
    if preserve_chain:
        final_docs = docs[:keep_full_top_docs] + [evidence_doc] + docs[keep_full_top_docs:]
    else:
        final_docs = [evidence_doc]
        final_docs.extend(docs[:keep_full_top_docs])

    selected_by_doc = defaultdict(list)
    for item in selected:
        selected_by_doc[item["doc_index"]].append(item)

    if not preserve_chain:
        for doc_index, doc in enumerate(docs[keep_full_top_docs:], start=keep_full_top_docs):
            snippets = selected_by_doc.get(doc_index, [])
            if not snippets:
                continue
            title = _doc_title(doc)
            snippet_lines = [f"Evidence snippets from {title}:"]
            for item in snippets[:max_sentences_per_doc]:
                snippet_lines.append(f"- {item['sentence']}")
            final_docs.append(f"{title}\n" + "\n".join(snippet_lines))

    original_chars = sum(len(str(doc)) for doc in docs) or 1
    final_chars = sum(len(str(doc)) for doc in final_docs)
    selected_titles = []
    seen_titles = set()
    for item in selected:
        title = item["title"]
        if title not in seen_titles:
            selected_titles.append(title)
            seen_titles.add(title)

    info.update({
        "applied": True,
        "reason": "selected_chain_preserved" if preserve_chain else "selected",
        "final_doc_count": len(final_docs),
        "selected_sentence_count": len(selected),
        "selected_titles": selected_titles,
        "compression_ratio": round(final_chars / original_chars, 6),
        "top_sentences": [
            {
                "doc_index": item["doc_index"],
                "sentence_index": item["sentence_index"],
                "title": item["title"],
                "score": round(item["score"], 6),
                "question_overlap": round(item["question_overlap"], 6),
                "target_signal": round(item["target_signal"], 6),
                "sentence_preview": item["sentence"][:240],
            }
            for item in sorted(selected, key=lambda x: -x["score"])[:8]
        ],
    })
    return final_docs, info
