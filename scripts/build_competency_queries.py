#!/usr/bin/env python3
"""Build query-specific scoping/validation CQs from positive and negative proofs."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import heapq
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


ROOT = Path(__file__).resolve().parents[1]
COMPASSRAG_SRC = ROOT / "src" / "compassrag"
for value in (str(ROOT / "scripts"), str(COMPASSRAG_SRC)):
    if value not in sys.path:
        sys.path.insert(0, value)

from workload import gold_documents
from construct_evidence_ontology import CachedChatClient
from query_plan_ontology import (
    get_answers,
    get_evidence_triples,
    get_question,
    get_row_id,
    infer_answer_type,
    ordered_evidence_path,
    question_intent,
    relation_signature,
)


TOKEN_RE = re.compile(r"[a-z0-9]+")
STOP = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does",
    "for", "from", "had", "has", "have", "how", "in", "is", "it", "of",
    "on", "or", "that", "the", "to", "was", "were", "what", "when", "where",
    "which", "who", "with",
}


def normalize(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def tokens(value: Any) -> List[str]:
    return [item for item in TOKEN_RE.findall(normalize(value)) if item not in STOP]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_doc_id(title: str, text: str) -> str:
    return "doc:" + hashlib.sha256(
        (normalize(title) + "\n" + normalize(text)).encode("utf-8")
    ).hexdigest()[:20]


def load_list(path: Path) -> List[Dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"expected JSON list: {path}")
    return [dict(item) for item in value if isinstance(item, Mapping)]


class SparseCorpusIndex:
    def __init__(self, documents: Sequence[Mapping[str, Any]]) -> None:
        self.documents: List[Dict[str, Any]] = []
        self.postings: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        document_frequency: Counter[str] = Counter()
        for index, item in enumerate(documents):
            title = str(item.get("title", "") or "")
            text = str(item.get("text", item.get("paragraph_text", "")) or "")
            token_counts = Counter(tokens(title + " " + text))
            document_frequency.update(token_counts)
            self.documents.append({
                "doc_id": stable_doc_id(title, text),
                "title": title,
                "text": text,
                "tokens": set(token_counts),
            })
            for token, count in token_counts.items():
                self.postings[token].append((index, count))
        count = max(len(self.documents), 1)
        self.idf = {
            token: math.log((count + 1.0) / (frequency + 0.5)) + 1.0
            for token, frequency in document_frequency.items()
        }
        self.by_exact = {
            normalize(item["title"] + "\n" + item["text"]): item["doc_id"]
            for item in self.documents
        }

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            sublinear_tf=True,
            max_features=200000,
            dtype=np.float32,
        )
        texts = [item["title"] + " " + item["text"] for item in self.documents]
        self.semantic_matrix = self.vectorizer.fit_transform(texts) if texts else None

    def search(self, query: str, limit: int) -> List[Tuple[float, Dict[str, Any]]]:
        scores: Dict[int, float] = defaultdict(float)
        query_counts = Counter(tokens(query))
        for token, query_tf in query_counts.items():
            idf = self.idf.get(token, 0.0)
            for index, doc_tf in self.postings.get(token, []):
                scores[index] += idf * (1.0 + math.log1p(doc_tf)) * (1.0 + math.log1p(query_tf))
        top = heapq.nlargest(limit, scores.items(), key=lambda item: (item[1], -item[0]))
        return [(score, self.documents[index]) for index, score in top]

    def hybrid_search(self, query: str, limit: int) -> List[Tuple[float, Dict[str, Any]]]:
        """Fuse lexical BM25-like and corpus-level TF-IDF semantic ranks."""
        lexical = self.search(query, max(limit * 2, limit))
        fused: Dict[int, float] = defaultdict(float)
        doc_position = {item["doc_id"]: index for index, item in enumerate(self.documents)}
        for rank, (_, document) in enumerate(lexical):
            fused[doc_position[document["doc_id"]]] += 1.0 / (60.0 + rank)
        if self.semantic_matrix is not None:
            query_vector = self.vectorizer.transform([query])
            similarities = (self.semantic_matrix @ query_vector.T).toarray().ravel()
            semantic_count = min(max(limit * 2, limit), len(similarities))
            if semantic_count:
                indices = np.argpartition(similarities, -semantic_count)[-semantic_count:]
                ordered = indices[np.argsort(similarities[indices])[::-1]]
                for rank, index in enumerate(ordered.tolist()):
                    fused[index] += 1.0 / (60.0 + rank)
        top = heapq.nlargest(limit, fused.items(), key=lambda item: (item[1], -item[0]))
        return [(score, self.documents[index]) for index, score in top]


def split_gold_document(value: str) -> Tuple[str, str]:
    title, separator, text = str(value or "").partition("\n")
    return title.strip(), text.strip() if separator else title.strip()


def content_filter_error(exc: BaseException) -> bool:
    """Identify provider-side prompt filtering without hiding other API faults."""
    if getattr(exc, "status_code", None) != 400:
        return False
    text = str(exc).lower()
    return any(
        marker in text
        for marker in ("content_filter", "content management policy", "response was filtered")
    )


def validate_generated_cqs(
    parsed: Mapping[str, Any] | Any,
    *,
    query_id: str,
) -> Dict[str, List[str]]:
    result: Dict[str, List[str]] = {}
    for field in (
        "scoping_cqs",
        "validation_cqs",
        "positive_requirements",
        "negative_rejection_rules",
    ):
        values = parsed.get(field, []) if isinstance(parsed, Mapping) else []
        result[field] = list(
            dict.fromkeys(str(value).strip() for value in values if str(value).strip())
        )[:8]
        if not result[field]:
            raise ValueError(
                f"CompetencyQueryAgent returned empty {field} for {query_id}"
            )
    return result


def deterministic_structural_cqs(record: Mapping[str, Any]) -> Dict[str, List[str]]:
    """Preserve structural supervision when one provider filters one example."""
    relations = [
        str(value).strip()
        for value in record.get("required_relations", []) or []
        if str(value).strip()
    ]
    relation_chain = " -> ".join(relations) or "the required evidence relation chain"
    answer_type = str(record.get("answer_type", "entity") or "entity")
    shape = record.get("branch_structure", {}) or {}
    branch_shape = str(shape.get("branch_shape", "chain") or "chain")
    negative_types = [
        str(value).replace("_", " ")
        for value in record.get("hard_negative_types", []) or []
        if str(value).strip()
    ]
    rejection_target = ", ".join(negative_types[:4]) or "relation, binding, or branch mismatch"
    return {
        "scoping_cqs": [
            f"Which anchor, intermediate bindings, and {answer_type} answer role execute {relation_chain}?",
            f"Which completeness constraints are required for the {branch_shape} evidence structure?",
        ],
        "validation_cqs": [
            f"Does the candidate proof preserve the ordered relation chain {relation_chain}?",
            f"Does it reject near misses exhibiting {rejection_target}?",
        ],
        "positive_requirements": [
            f"Preserve every executable step in {relation_chain}.",
            f"Bind adjacent steps consistently and terminate at a {answer_type} answer.",
        ],
        "negative_rejection_rules": [
            f"Reject candidates with {rejection_target}.",
            "Reject incomplete paths, broken intermediate bindings, and wrong terminal types.",
        ],
    }


def content_safe_cq_messages(record: Mapping[str, Any]) -> List[Dict[str, str]]:
    """Build a text-free structural retry payload for provider-filtered examples."""
    payload = {
        "query_shape": record.get("branch_structure", {}),
        "answer_type": record.get("answer_type", "entity"),
        "ordered_relation_signatures": record.get("required_relations", []),
        "required_roles": record.get("required_roles", []),
        "intermediate_bindings": record.get("intermediate_bindings", []),
        "negative_failure_types": record.get("hard_negative_types", []),
    }
    return [
        {
            "role": "system",
            "content": (
                "Generate reusable structural competency queries for an evidence ontology. "
                "Use only the supplied abstract roles, relation signatures, bindings, branch "
                "shape, and failure types. Return strict JSON arrays named scoping_cqs, "
                "validation_cqs, positive_requirements, and negative_rejection_rules."
            ),
        },
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def query_specific_cqs(
    record: Mapping[str, Any], client: CachedChatClient
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    positives = [
        {"title": item.get("title", ""), "text": str(item.get("text", ""))[:700]}
        for item in record.get("positive_proof", []) or []
    ]
    negatives = [
        {
            "title": item.get("title", ""),
            "text": str(item.get("text", ""))[:500],
            "failure_types": item.get("negative_types", []),
        }
        for item in record.get("hard_negatives", []) or []
    ]
    messages = [
        {
            "role": "system",
            "content": (
                "You are the competency-query author in an evidence-ontology induction system. "
                "A scoping CQ must state what reusable evidence roles, ordered relations, bindings, "
                "branches, and answer type are needed to support the given query. A validation CQ "
                "must contrast the complete positive proof against the retrieved hard-negative proofs "
                "and ask for an executable constraint that rejects the negatives. Do not mention "
                "dataset names, document ids, gold labels, or evaluation. Return strict JSON with "
                "scoping_cqs, validation_cqs, positive_requirements, and negative_rejection_rules. "
                "Each field is a non-empty array of concise strings."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "query": record.get("query", ""),
                    "query_shape": record.get("branch_structure", {}),
                    "answer_type": record.get("answer_type", "entity"),
                    "grounded_relation_path": record.get("required_relations", []),
                    "positive_proof": positives,
                    "retrieved_counterexamples": negatives,
                },
                ensure_ascii=False,
            ),
        },
    ]
    query_id = str(record.get("query_id", ""))
    try:
        parsed, metadata = client.ask_json(messages, "CompetencyQueryAgent")
        return validate_generated_cqs(parsed, query_id=query_id), metadata
    except Exception as exc:
        if not (content_filter_error(exc) or isinstance(exc, ValueError)):
            raise
        first_error = f"{type(exc).__name__}: {exc}"

    try:
        parsed, metadata = client.ask_json(
            content_safe_cq_messages(record),
            "CompetencyQueryAgentStructuralRetry",
        )
        metadata = {
            **metadata,
            "fallback_mode": "structural_llm_retry",
            "first_error": first_error,
        }
        return validate_generated_cqs(parsed, query_id=query_id), metadata
    except Exception as exc:
        if not (content_filter_error(exc) or isinstance(exc, ValueError)):
            raise
        return deterministic_structural_cqs(record), {
            "cache_hit": False,
            "fallback_mode": "deterministic_structural_cq",
            "first_error": first_error,
            "second_error": f"{type(exc).__name__}: {exc}",
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }


def query_shape(question: str, path: Sequence[Sequence[str]], answer_type: str) -> Dict[str, Any]:
    text = normalize(question)
    comparison = bool(re.search(r"\b(same|both|either|earlier|later|older|younger)\b", text))
    conjunction = " and " in f" {text} "
    return {
        "hop_count": len(path),
        "branch_shape": "comparison" if comparison else "conjunction" if conjunction else "chain",
        "answer_type": answer_type,
    }


def classify_negative(
    document: Mapping[str, Any],
    *,
    query_tokens: set[str],
    relation_tokens: set[str],
    anchor_terms: Sequence[str],
    answer_terms: Sequence[str],
    shape: Mapping[str, Any],
) -> Tuple[List[str], List[str]]:
    doc_tokens = set(document.get("tokens", set()) or set())
    types = ["semantic_near_miss"]
    reasons = ["retrieved_by_sparse_similarity_but_not_in_gold_evidence"]
    anchor_hits = sum(bool(set(tokens(anchor)) & doc_tokens) for anchor in anchor_terms)
    relation_hit = bool(relation_tokens & doc_tokens)
    answer_hit = any(normalize(answer) in normalize(document.get("text", "")) for answer in answer_terms if answer)
    if int(shape.get("hop_count", 0) or 0) >= 2 and (anchor_hits or relation_hit):
        types.append("incomplete_path")
        reasons.append("shares_anchor_or_relation_signal_but_is_not_a_gold_proof_member")
    if int(shape.get("hop_count", 0) or 0) >= 2 and anchor_hits and not answer_hit:
        types.append("wrong_bridge")
        reasons.append("anchor_related_candidate_does_not_contain_the_gold_terminal_answer")
    if shape.get("branch_shape") in {"comparison", "conjunction"} and anchor_hits < 2:
        types.append("branch_incomplete")
        reasons.append("candidate_does_not_cover_multiple_query_branches")
    if relation_hit and anchor_hits:
        types.append("structurally_plausible_incorrect_path")
        reasons.append("candidate_matches_anchor_and_relation_vocabulary_but_is_not_gold_evidence")
    expected = str(shape.get("answer_type", "entity") or "entity")
    text = str(document.get("text", "") or "")
    if expected == "date" and not re.search(r"\b(?:1[0-9]{3}|20[0-9]{2})\b", text):
        types.append("wrong_answer_type")
        reasons.append("expected_date_but_candidate_has_no_year_expression")
    return list(dict.fromkeys(types)), list(dict.fromkeys(reasons))


def feature_set(record: Mapping[str, Any]) -> set[str]:
    features = {f"q:{token}" for token in tokens(record.get("query", ""))}
    features.add(f"intent:{record.get('intent', '')}")
    features.add(f"answer:{record.get('answer_type', '')}")
    shape = record.get("branch_structure", {}) or {}
    features.add(f"hop:{shape.get('hop_count', 0)}")
    features.add(f"branch:{shape.get('branch_shape', 'chain')}")
    for relation in record.get("required_relations", []) or []:
        features.add(f"rel:{relation}")
    return features


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def diverse_seed_indices(records: Sequence[Mapping[str, Any]], count: int) -> List[int]:
    if not records or count <= 0:
        return []
    count = min(count, len(records))
    features = [feature_set(record) for record in records]
    first = max(range(len(records)), key=lambda idx: (len(features[idx]), -idx))
    selected = [first]
    remaining = set(range(len(records))) - {first}
    while remaining and len(selected) < count:
        candidate = max(
            remaining,
            key=lambda idx: (
                min(1.0 - jaccard(features[idx], features[chosen]) for chosen in selected),
                len(features[idx]),
                -idx,
            ),
        )
        selected.append(candidate)
        remaining.remove(candidate)
    return selected


def balanced_clusters(
    records: Sequence[Mapping[str, Any]],
    cluster_count: int,
    max_capacity: int,
    seed_indices: Sequence[int],
) -> List[int]:
    if not records:
        return []
    required = math.ceil(len(records) / max(max_capacity, 1))
    cluster_count = max(cluster_count, required, 1)
    cluster_count = min(cluster_count, len(records))
    features = [feature_set(record) for record in records]
    initial = list(seed_indices[:cluster_count])
    if len(initial) < cluster_count:
        initial.extend(idx for idx in range(len(records)) if idx not in initial)
        initial = initial[:cluster_count]
    members: List[List[int]] = [[idx] for idx in initial]
    assignments = [-1] * len(records)
    for cluster_idx, record_idx in enumerate(initial):
        assignments[record_idx] = cluster_idx
    for record_idx in range(len(records)):
        if assignments[record_idx] >= 0:
            continue
        candidates = [idx for idx, values in enumerate(members) if len(values) < max_capacity]
        if not candidates:
            candidates = list(range(cluster_count))
        cluster_idx = max(
            candidates,
            key=lambda idx: (
                sum(jaccard(features[record_idx], features[value]) for value in members[idx])
                / max(len(members[idx]), 1),
                -len(members[idx]),
                -idx,
            ),
        )
        members[cluster_idx].append(record_idx)
        assignments[record_idx] = cluster_idx
    return assignments


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--q-ind")
    source.add_argument("--q-h")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--retrieve-k", type=int, default=64)
    parser.add_argument("--hard-negatives", type=int, default=8)
    parser.add_argument("--seed-count", type=int, default=48)
    parser.add_argument("--clusters", type=int, default=8)
    parser.add_argument("--max-cluster-capacity", type=int, default=160)
    parser.add_argument(
        "--llm-cqs", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--require-llm-cqs", action="store_true")
    parser.add_argument("--llm-name", default=os.environ.get("MODEL_NAME", "reader-model"))
    parser.add_argument("--llm-base-url", default=os.environ.get("MODEL_BASE_URL"))
    parser.add_argument("--llm-cache", default="outputs/contrastive_cq_agent_cache.sqlite")
    parser.add_argument("--llm-workers", type=int, default=8)
    parser.add_argument("--llm-max-tokens", type=int, default=900)
    args = parser.parse_args()

    source_path = Path(args.q_h or args.q_ind).expanduser().resolve()
    source_split = "q_h" if args.q_h else "q_ind"
    corpus_path = Path(args.corpus).expanduser().resolve()
    rows = load_list(source_path)
    corpus = load_list(corpus_path)
    index = SparseCorpusIndex(corpus)
    records: List[Dict[str, Any]] = []
    for row in rows:
        query = get_question(row)
        if not query:
            continue
        answers = get_answers(row)
        triples = get_evidence_triples(row)
        path = ordered_evidence_path(triples, query, answers)
        answer_type = infer_answer_type(query, answers)
        signatures = [relation_signature(item[1], dataset=args.dataset) for item in path]
        signatures = [value for value in signatures if value]
        relation_terms = {token for value in signatures for token in tokens(value)}
        anchors = list(dict.fromkeys(
            [item[0] for item in path if item and item[0]]
            + [str(row.get("subj", "") or "")]
        ))
        shape = query_shape(query, path, answer_type)
        gold_values = gold_documents(args.dataset, row)
        gold_ids = {
            index.by_exact.get(normalize(value), "") for value in gold_values
        }
        gold_ids.discard("")
        negatives = []
        for score, document in index.hybrid_search(
            query + " " + " ".join(signatures),
            max(args.retrieve_k, args.hard_negatives),
        ):
            if document["doc_id"] in gold_ids:
                continue
            negative_types, reasons = classify_negative(
                document,
                query_tokens=set(tokens(query)),
                relation_tokens=relation_terms,
                anchor_terms=anchors,
                answer_terms=answers,
                shape=shape,
            )
            negatives.append({
                "doc_id": document["doc_id"],
                "title": document["title"],
                "text": str(document.get("text", "") or "")[:1200],
                "retrieval_score": round(float(score), 8),
                "negative_types": negative_types,
                "failure_reasons": reasons,
                "source": "real_corpus_hybrid_tfidf_near_miss",
            })
            if len(negatives) >= args.hard_negatives:
                break
        intent = question_intent(query, answer_type, signatures)
        query_id = get_row_id(row)
        positive_proof = []
        for value in gold_values:
            title, text = split_gold_document(value)
            doc_id = index.by_exact.get(normalize(value), stable_doc_id(title, text))
            positive_proof.append({"doc_id": doc_id, "title": title, "text": text[:1600]})
        counterexample_proofs = []
        for proof_index, negative in enumerate(negatives[: max(len(positive_proof), 1)]):
            retained = [dict(item) for item in positive_proof]
            if retained:
                retained[proof_index % len(retained)] = {
                    "doc_id": negative["doc_id"],
                    "title": negative["title"],
                    "text": negative["text"],
                }
            else:
                retained = [{
                    "doc_id": negative["doc_id"],
                    "title": negative["title"],
                    "text": negative["text"],
                }]
            counterexample_proofs.append({
                "proof_id": f"{query_id}:negative_slate:{proof_index}",
                "documents": retained,
                "failure_types": negative["negative_types"],
                "construction": "replace_one_positive_proof_member_with_retrieved_near_miss",
            })
        record = {
            "query_id": query_id,
            "query": query,
            "intent": intent,
            "anchor_roles": ["anchor"] if anchors else [],
            "answer_role": "answer",
            "answer_type": answer_type,
            "gt_evidence_ids": sorted(gold_ids),
            "positive_proof": positive_proof,
            "counterexample_proofs": counterexample_proofs,
            "gt_path_signature": signatures,
            "hard_negative_ids": [item["doc_id"] for item in negatives],
            "hard_negative_types": sorted({
                value for item in negatives for value in item["negative_types"]
            }),
            "hard_negative_failure_reasons": sorted({
                value for item in negatives for value in item["failure_reasons"]
            }),
            "hard_negatives": negatives,
            "scoping_cqs": [
                "Which reusable anchor, intermediate, and answer roles are required?",
                "Which ordered relations and bindings are required to complete the evidence proof?",
                "Which answer type and branch-completeness constraints must the schema preserve?",
            ],
            "validation_cqs": [
                "Which executable constraint rejects the retrieved near misses while retaining the gold proof?",
                "Which relation direction, bridge binding, answer type, or branch condition distinguishes the proof?",
            ],
            "required_roles": [
                "anchor",
                *[f"intermediate_{idx}" for idx in range(max(len(path) - 1, 0))],
                "answer",
            ],
            "required_relations": signatures,
            "relation_directions": ["forward" for _ in signatures],
            "intermediate_bindings": [
                {"from_step": idx, "to_step": idx + 1, "shared_role": f"intermediate_{idx}"}
                for idx in range(max(len(path) - 1, 0))
            ],
            "branch_structure": shape,
            "structural_constraints": {
                "ordered_path": True,
                "complete_all_branches": shape["branch_shape"] != "chain",
                "answer_type": answer_type,
            },
            "llm_abstraction_payload": {
                "query": query,
                "ground_truth_evidence_ids": sorted(gold_ids),
                "ground_truth_path": [list(item) for item in path],
                "retrieved_hard_negatives": negatives,
            },
            "provenance": {
                "split": source_split,
                "historical_query_path": str(source_path),
                "corpus_path": str(corpus_path),
                "negative_generator": "deterministic_real_corpus_hybrid_tfidf_retrieval",
                "llm_generated_negative": False,
            },
        }
        records.append(record)

    llm_metadata: List[Dict[str, Any]] = []
    if args.llm_cqs:
        client = CachedChatClient(
            model=args.llm_name,
            base_url=args.llm_base_url,
            cache_path=Path(args.llm_cache).expanduser().resolve(),
            temperature=0.0,
            max_tokens=args.llm_max_tokens,
        )

        def enrich(record: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
            generated, metadata = query_specific_cqs(record, client)
            enriched = dict(record)
            enriched.update(generated)

            enriched["coverage_cqs"] = generated["scoping_cqs"]
            enriched["contrastive_cqs"] = generated["validation_cqs"]
            return enriched, metadata

        enriched_records: List[Dict[str, Any] | None] = [None] * len(records)
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.llm_workers) as executor:
            futures = {
                executor.submit(enrich, record): index_value
                for index_value, record in enumerate(records)
            }
            for future in concurrent.futures.as_completed(futures):
                index_value = futures[future]
                enriched_records[index_value], metadata = future.result()
                llm_metadata.append(metadata)
        records = [record for record in enriched_records if record is not None]
    elif args.require_llm_cqs:
        raise ValueError("--require-llm-cqs requires --llm-cqs")

    seeds = diverse_seed_indices(records, args.seed_count)
    assignments = balanced_clusters(
        records,
        args.clusters,
        args.max_cluster_capacity,
        seeds,
    )
    seed_set = set(seeds)
    for index_value, (record, cluster_id) in enumerate(zip(records, assignments)):
        record["cluster_id"] = int(cluster_id)
        record["structural_architect_seed"] = index_value in seed_set
    assert len(records) == len(assignments)
    assert {record["query_id"] for record in records} == {
        get_row_id(row) for row in rows if get_question(row)
    }, "every historical query must enter ontology refinement"

    cluster_counts = Counter(assignments)
    payload = {
        "type": "contrastive_evidence_competency_queries",
        "version": 2,
        "dataset": args.dataset,
        "source_split": source_split,
        "historical_query_sha256": sha256_file(source_path),
        "corpus_sha256": sha256_file(corpus_path),
        "records": records,
        "initialization": {
            "method": "deterministic_greedy_max_min_diversity",
            "seed_query_ids": [records[idx]["query_id"] for idx in seeds],
            "seed_count": len(seeds),
            "seed_only_initializes_skeleton": True,
            "all_historical_queries_refined": True,
        },
        "clustering": {
            "method": "capacity_constrained_query_feature_clustering",
            "cluster_count": len(cluster_counts),
            "max_capacity": args.max_cluster_capacity,
            "cluster_sizes": {str(key): value for key, value in sorted(cluster_counts.items())},
            "features": [
                "query_lexical_semantics", "intent", "answer_type", "hop_count",
                "relation_path_signature", "anchor_roles", "branch_shape",
            ],
        },
        "diagnostics": {
            "historical_query_rows": len(rows),
            "cq_records": len(records),
            "queries_with_gold_evidence_ids": sum(bool(item["gt_evidence_ids"]) for item in records),
            "queries_with_hard_negatives": sum(bool(item["hard_negative_ids"]) for item in records),
            "hard_negative_count": sum(len(item["hard_negative_ids"]) for item in records),
            "positive_proof_count": sum(len(item["positive_proof"]) for item in records),
            "counterexample_proof_count": sum(
                len(item["counterexample_proofs"]) for item in records
            ),
            "query_specific_cqs": bool(args.llm_cqs),
            "llm_calls": sum(not bool(item.get("cache_hit")) for item in llm_metadata),
            "llm_cache_hits": sum(bool(item.get("cache_hit")) for item in llm_metadata),
            "llm_structural_retries": sum(
                item.get("fallback_mode") == "structural_llm_retry"
                for item in llm_metadata
            ),
            "deterministic_structural_fallbacks": sum(
                item.get("fallback_mode") == "deterministic_structural_cq"
                for item in llm_metadata
            ),
        },
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"contrastive_cqs={output}")
    print(json.dumps(payload["diagnostics"], ensure_ascii=False))


if __name__ == "__main__":
    main()
