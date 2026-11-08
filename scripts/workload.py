#!/usr/bin/env python3
"""Create deterministic Q_ind/Q_cal/Q_val/Q_test protocol partitions."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
SOURCE_CANDIDATES = {
    "2wikimultihopqa": ["reproduce/dataset/raw_qa/2wikimultihopqa/train.jsonl"],
    "hotpotqa": ["reproduce/dataset/raw_qa/hotpotqa_distractor/train.jsonl"],
    "lveval": [
        "reproduce/query_splits_80_20/lveval_train.json",
        "reproduce/dataset/raw_qa/hotpotqa_distractor/train.jsonl",
    ],
    "musique": ["reproduce/dataset/raw_qa/musique/train.jsonl"],
    "narrativeqa_dev_10_doc": [
        "reproduce/query_splits_80_20/narrativeqa_dev_10_doc_train.json",
        "reproduce/dataset/raw_qa/narrativeqa/train.jsonl",
    ],



    "nq_rear": [
        "reproduce/query_splits_80_20/nq_rear_train.json",
    ],
    "popqa": ["reproduce/query_splits_80_20/popqa_train.json"],
}
DEFAULT_DATASETS = tuple(SOURCE_CANDIDATES)
DATASET_LIMITS = {
    "narrativeqa_dev_10_doc": 600,
    "popqa": 200,
}
DATASET_SCAN_LIMITS = {


    "narrativeqa_dev_10_doc": 1200,
}


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def text_value(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Mapping):
        for key in ("text", "value", "answer", "summary"):
            nested = text_value(value.get(key))
            if nested:
                return nested
        return ""
    if isinstance(value, list):
        return " ".join(filter(None, (text_value(item) for item in value))).strip()
    return str(value or "").strip()


def stable_id(row: Mapping[str, Any]) -> str:
    explicit = row.get("id") or row.get("_id")
    if explicit:
        return str(explicit)
    document = row.get("document")
    document_id = str(document.get("id", "") or "") if isinstance(document, Mapping) else ""
    query = question(row)
    return f"{document_id}:{query}" if document_id and query else query


def question(row: Mapping[str, Any]) -> str:
    return text_value(row.get("question") or row.get("query") or "")


def numbered_passages(value: str) -> List[Dict[str, Any]]:
    chunks = re.split(r"(?=^###\s+Passage\s+\d+\s*$)", value, flags=re.MULTILINE)
    passages = []
    for index, chunk in enumerate(chunks):
        chunk = chunk.strip()
        if not chunk:
            continue
        lines = chunk.splitlines()
        heading = lines[0].lstrip("# ").strip()
        text = "\n".join(lines[1:]).strip() if lines[0].startswith("###") else chunk
        if text:
            passages.append({
                "idx": index,
                "title": heading or f"Passage {index + 1}",
                "paragraph_text": text,
                "is_supporting": True,
            })
    return passages


def load_rows(path: Path, max_scan_rows: int = 0) -> Iterator[Dict[str, Any]]:
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if max_scan_rows and index >= max_scan_rows:
                    break
                line = line.strip()
                if not line:
                    continue
                value = json.loads(line)
                if isinstance(value, dict):
                    yield value
        return
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict):
        for key in ("data", "examples", "records", "questions"):
            if isinstance(value.get(key), list):
                value = value[key]
                break
    if not isinstance(value, list):
        return
    for index, row in enumerate(value):
        if max_scan_rows and index >= max_scan_rows:
            break
        if isinstance(row, dict):
            yield row


def normalize_row(dataset: str, row: Mapping[str, Any]) -> Dict[str, Any]:
    payload = dict(row)
    if dataset == "lveval" and isinstance(payload.get("context"), str):
        payload["paragraphs"] = numbered_passages(payload["context"])
    if dataset == "narrativeqa_dev_10_doc":
        payload["question"] = question(payload)
        if "answer" not in payload and payload.get("answers") is not None:
            payload["answer"] = payload.get("answers")
        document = payload.get("document")
        if isinstance(document, Mapping):
            summary = text_value(document.get("summary"))
            if not summary:
                summary = text_value(document.get("text"))
            if summary:
                payload["paragraphs"] = [{
                    "idx": 0,
                    "title": text_value(document.get("title") or document.get("id")),
                    "paragraph_text": summary,
                    "is_supporting": True,
                }]
    if dataset in {"2wikimultihopqa", "hotpotqa", "lveval"} and isinstance(
        payload.get("context"), dict
    ):
        context = payload["context"]
        titles = list(context.get("title", []) or [])
        sentences = list(context.get("sentences", []) or [])
        payload["context"] = [
            [title, sentences[index] if index < len(sentences) else []]
            for index, title in enumerate(titles)
        ]
    if dataset in {"2wikimultihopqa", "hotpotqa", "lveval"} and isinstance(
        payload.get("supporting_facts"), dict
    ):
        facts = payload["supporting_facts"]
        titles = list(facts.get("title", []) or [])
        sentence_ids = list(facts.get("sent_id", []) or [])
        payload["supporting_facts"] = [
            [title, sentence_ids[index] if index < len(sentence_ids) else 0]
            for index, title in enumerate(titles)
        ]
    return payload


def choose_source(dataset: str) -> Path:
    for relative in SOURCE_CANDIDATES.get(dataset, []):
        path = ROOT / relative
        if path.is_file() and path.stat().st_size > 0:
            return path
    raise FileNotFoundError(f"no non-empty train source for {dataset}")


def load_eval_identity(
    dataset: str, deployment_dir: Path
) -> Tuple[set[str], set[str], Path | None]:
    candidates = (
        deployment_dir / f"{dataset}_eval.json",
        deployment_dir / f"{dataset}.eval.json",
        deployment_dir / f"{dataset}.json",
    )
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        return set(), set(), None
    rows = list(load_rows(path))
    return (
        {stable_id(row) for row in rows if stable_id(row)},
        {normalize_text(question(row)) for row in rows if question(row)},
        path,
    )


def priority(seed: str, dataset: str, row: Mapping[str, Any]) -> int:
    identity = stable_id(row) or normalize_text(question(row))
    digest = hashlib.sha256(f"{seed}|{dataset}|{identity}".encode("utf-8")).digest()
    return int.from_bytes(digest, "big")


def select_rows(
    dataset: str,
    source: Path,
    limit: int,
    seed: str,
    eval_ids: set[str],
    eval_questions: set[str],
    max_scan_rows: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    heap: List[Tuple[int, int, Dict[str, Any]]] = []
    seen_ids: set[str] = set()
    scanned = overlap = duplicates = 0
    for row in load_rows(source, max_scan_rows=max_scan_rows):
        scanned += 1
        identity = stable_id(row) or normalize_text(question(row))
        if not identity or identity in seen_ids:
            duplicates += 1
            continue
        seen_ids.add(identity)
        if identity in eval_ids or normalize_text(question(row)) in eval_questions:
            overlap += 1
            continue
        rank = priority(seed, dataset, row)
        item = (-rank, scanned, normalize_row(dataset, row))
        if len(heap) < limit:
            heapq.heappush(heap, item)
        elif rank < -heap[0][0]:
            heapq.heapreplace(heap, item)
    selected = [item[2] for item in sorted(heap, key=lambda item: -item[0])]
    return selected, {
        "scanned_rows": scanned,
        "eval_overlap_excluded": overlap,
        "duplicate_rows_excluded": duplicates,
    }


def split_rows(
    rows: Sequence[Dict[str, Any]],
    induction_ratio: float,
    calibration_ratio: float,
) -> Dict[str, List[Dict[str, Any]]]:
    count = len(rows)
    induction_end = int(round(count * induction_ratio))
    calibration_end = induction_end + int(round(count * calibration_ratio))
    induction_end = min(max(induction_end, 1 if count else 0), count)
    calibration_end = min(max(calibration_end, induction_end), count)
    return {
        "induction": list(rows[:induction_end]),
        "calibration": list(rows[induction_end:calibration_end]),
        "cq_holdout": list(rows[calibration_end:]),
    }


def answer_type(row: Mapping[str, Any]) -> str:
    text = normalize_text(question(row))
    raw_answers = row.get("reference", row.get("answer", row.get("obj", "")))
    if isinstance(raw_answers, list):
        answers = [normalize_text(value) for value in raw_answers]
    else:
        answers = [normalize_text(raw_answers)]
    if any(value in {"yes", "no"} for value in answers):
        return "boolean"
    prop = normalize_text(row.get("prop", ""))
    if prop in {"date of birth", "date of death", "publication date", "inception"}:
        return "date"
    if prop in {"place of birth", "place of death", "country", "location"}:
        return "location"
    if re.search(r"\bhow many\b|\bnumber of\b", text):
        return "number"
    if re.match(r"^(when|what year|which year)\b", text):
        return "date"
    if re.match(r"^(where|in which (?:city|country|place))\b", text):
        return "location"
    if re.match(r"^(who|whose)\b", text):
        return "person"
    return "entity"


def query_family(dataset: str, row: Mapping[str, Any]) -> str:
    explicit = normalize_text(row.get("type", ""))
    if explicit:
        return re.sub(r"[^a-z0-9]+", "_", explicit).strip("_") or "unknown"
    prop = normalize_text(row.get("prop", ""))
    if prop:
        return "property_" + re.sub(r"[^a-z0-9]+", "_", prop).strip("_")
    text = normalize_text(question(row))
    if re.search(r"\b(same|both|either|older|younger|earlier|later|first)\b", text):
        return "comparison"
    if re.search(r"\bhow many\b|\bnumber of\b", text):
        return "aggregation"
    decomposition = row.get("question_decomposition", row.get("decomposition", []))
    if isinstance(decomposition, list) and len(decomposition) >= 3:
        return "multi_hop_3plus"
    if isinstance(decomposition, list) and len(decomposition) >= 2:
        return "multi_hop_2"
    if dataset in {"2wikimultihopqa", "hotpotqa", "musique"}:
        return "multi_hop"
    return "single_hop"


def split_stratum(dataset: str, row: Mapping[str, Any]) -> str:
    return f"{query_family(dataset, row)}|{answer_type(row)}"


def anchor_entities(dataset: str, row: Mapping[str, Any]) -> set[str]:
    anchors: set[str] = set()

    def add(value: Any) -> None:
        normalized = re.sub(r"[^a-z0-9]+", " ", normalize_text(value)).strip()
        if normalized and not re.fullmatch(r"passage\s+\d+", normalized):
            anchors.add(normalized)

    for key in ("subj", "s_wiki_title"):
        add(row.get(key, ""))
    document = row.get("document")
    if isinstance(document, Mapping):
        add(document.get("id", ""))
        add(document.get("title", ""))
    facts = row.get("supporting_facts", []) or []
    if isinstance(facts, Mapping):
        for value in facts.get("title", []) or []:
            add(value)
    else:
        for item in facts:
            if isinstance(item, (list, tuple)) and item:
                add(item[0])
    for key in ("contexts", "paragraphs", "ctxs"):
        for item in row.get(key, []) or []:
            if not isinstance(item, Mapping):
                continue
            if item.get("is_supporting", False):
                add(item.get("title", ""))
    if not anchors:
        anchors.add(f"query:{stable_id(row) or normalize_text(question(row))}")
    return anchors


def stratified_entity_partitions(
    rows: Sequence[Dict[str, Any]],
    dataset: str,
    induction_ratio: float,
    calibration_ratio: float,
    seed: str,
) -> Dict[str, List[Dict[str, Any]]]:
    """Split by strata while keeping every shared evidence entity in one split."""
    split_names = ("induction", "calibration", "cq_holdout")
    ratios = {
        "induction": induction_ratio,
        "calibration": calibration_ratio,
        "cq_holdout": 1.0 - induction_ratio - calibration_ratio,
    }
    parent = list(range(len(rows)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    entity_owner: Dict[str, int] = {}
    row_anchors: List[set[str]] = []
    for index, row in enumerate(rows):
        anchors = anchor_entities(dataset, row)
        row_anchors.append(anchors)
        for anchor in anchors:
            owner = entity_owner.setdefault(anchor, index)
            union(index, owner)

    components: Dict[int, List[int]] = defaultdict(list)
    for index in range(len(rows)):
        components[find(index)].append(index)
    total_labels = Counter(split_stratum(dataset, row) for row in rows)
    target_sizes = {name: len(rows) * ratios[name] for name in split_names}
    target_labels = {
        name: {label: count * ratios[name] for label, count in total_labels.items()}
        for name in split_names
    }
    assigned: Dict[str, List[int]] = {name: [] for name in split_names}
    sizes: Counter[str] = Counter()
    labels = {name: Counter() for name in split_names}
    ordered = sorted(
        components.values(),
        key=lambda indices: (
            -len(indices),
            min(total_labels[split_stratum(dataset, rows[index])] for index in indices),
            priority(seed, dataset, rows[indices[0]]),
        ),
    )
    for indices in ordered:
        component_labels = Counter(split_stratum(dataset, rows[index]) for index in indices)

        def deficit(split_name: str) -> Tuple[float, int]:
            size_need = (target_sizes[split_name] - sizes[split_name]) / max(
                target_sizes[split_name], 1.0
            )
            label_need = sum(
                count
                * (target_labels[split_name][label] - labels[split_name][label])
                / max(target_labels[split_name][label], 1.0)
                for label, count in component_labels.items()
            ) / max(sum(component_labels.values()), 1)
            tie_break = int.from_bytes(
                hashlib.sha256(
                    f"{seed}|{dataset}|{split_name}|{indices[0]}".encode("utf-8")
                ).digest(),
                "big",
            )
            return (size_need + label_need) / 2.0, -tie_break

        destination = max(split_names, key=deficit)
        assigned[destination].extend(indices)
        sizes[destination] += len(indices)
        labels[destination].update(component_labels)

    return {
        name: sorted(
            (rows[index] for index in indices),
            key=lambda row: priority(seed, f"{dataset}:{name}", row),
        )
        for name, indices in assigned.items()
    }


def partition_diagnostics(
    dataset: str,
    partitions: Mapping[str, Sequence[Mapping[str, Any]]],
) -> Dict[str, Any]:
    entities = {
        name: set().union(*(anchor_entities(dataset, row) for row in rows)) if rows else set()
        for name, rows in partitions.items()
    }
    names = ("induction", "calibration", "cq_holdout")
    overlaps = {
        f"{names[left]}_{names[right]}": len(
            entities[names[left]] & entities[names[right]]
        )
        for left, right in ((0, 1), (0, 2), (1, 2))
    }
    return {
        "strategy": "stratified_support_entity_component_disjoint",
        "support_entity_counts": {name: len(values) for name, values in entities.items()},
        "support_entity_overlap": overlaps,
        "stratum_counts": {
            name: dict(Counter(split_stratum(dataset, row) for row in rows))
            for name, rows in partitions.items()
        },
    }


def iter_documents(dataset: str, rows: Iterable[Mapping[str, Any]]) -> Iterator[Dict[str, str]]:
    for row in rows:
        if dataset in {"2wikimultihopqa", "hotpotqa", "lveval"}:
            if isinstance(row.get("context"), str):
                for passage in numbered_passages(str(row["context"])):
                    yield {
                        "title": str(passage["title"]),
                        "text": str(passage["paragraph_text"]),
                    }
                continue
            separator = "" if dataset in {"hotpotqa", "lveval"} else " "
            for item in row.get("context", []) or []:
                if isinstance(item, list) and len(item) >= 2:
                    yield {
                        "title": str(item[0]),
                        "text": separator.join(str(sentence) for sentence in item[1]),
                    }
            continue
        for key in ("ctxs", "contexts", "paragraphs"):
            for item in row.get(key, []) or []:
                if not isinstance(item, Mapping):
                    continue
                yield {
                    "title": str(item.get("title", "") or ""),
                    "text": str(item.get("text", item.get("paragraph_text", "")) or ""),
                }


def corpus_rows(dataset: str, rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, str]]:
    result: List[Dict[str, str]] = []
    seen = set()
    for document in iter_documents(dataset, rows):
        key = (document["title"], document["text"])
        if not document["text"] or key in seen:
            continue
        seen.add(key)
        result.append(document)
    return result


def gold_documents(dataset: str, row: Mapping[str, Any]) -> List[str]:
    if "supporting_facts" in row and "context" in row:
        titles = {
            str(item[0])
            for item in row.get("supporting_facts", []) or []
            if isinstance(item, list) and item
        }
        separator = "" if dataset in {"hotpotqa", "lveval"} else " "
        return list({
            str(item[0]) + "\n" + separator.join(str(value) for value in item[1])
            for item in row.get("context", []) or []
            if isinstance(item, list) and len(item) >= 2 and str(item[0]) in titles
        })
    if "contexts" in row:
        return list({
            str(item.get("title", "")) + "\n" + str(item.get("text", ""))
            for item in row.get("contexts", []) or []
            if isinstance(item, Mapping) and item.get("is_supporting")
        })
    if "paragraphs" in row:
        return list({
            str(item.get("title", "")) + "\n" + str(
                item.get("text", item.get("paragraph_text", "")) or ""
            )
            for item in row.get("paragraphs", []) or []
            if isinstance(item, Mapping) and item.get("is_supporting", True) is not False
        })
    return []


def exact_gold_coverage(
    dataset: str,
    rows: Sequence[Mapping[str, Any]],
    corpus: Sequence[Mapping[str, str]],
) -> Dict[str, Any]:
    corpus_values = {str(item["title"]) + "\n" + str(item["text"]) for item in corpus}
    expected = matched = queries_with_gold = fully_covered = 0
    for row in rows:
        gold = gold_documents(dataset, row)
        if not gold:
            continue
        queries_with_gold += 1
        expected += len(gold)
        hits = sum(item in corpus_values for item in gold)
        matched += hits
        fully_covered += int(hits == len(gold))
    return {
        "gold_documents": expected,
        "matched_gold_documents": matched,
        "gold_document_exact_coverage": matched / max(expected, 1),
        "queries_with_gold": queries_with_gold,
        "fully_covered_queries": fully_covered,
        "query_exact_coverage": fully_covered / max(queries_with_gold, 1),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def canonical_split_name(legacy_name: str) -> str:
    return {
        "induction": "q_ind",
        "calibration": "q_cal",
        "cq_holdout": "q_val",
    }[legacy_name]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--deployment-dir",
        default="reproduce/query_splits_80_20",
        help="Directory containing held-out *_eval.json files.",
    )
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--max-workload-rows", type=int, default=3000)
    parser.add_argument("--max-scan-rows", type=int, default=30000)
    parser.add_argument("--induction-ratio", type=float, default=0.70)
    parser.add_argument("--calibration-ratio", type=float, default=0.15)
    parser.add_argument("--seed", default="workload-competency")
    parser.add_argument(
        "--merge-cal-val",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Ablation only: expose Q_cal and Q_val as one dev partition.",
    )
    parser.add_argument(
        "--write-legacy-aliases",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also write induction/calibration/cq_holdout copies for old runners.",
    )
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    if args.induction_ratio <= 0 or args.calibration_ratio <= 0:
        raise SystemExit("split ratios must be positive")
    if args.induction_ratio + args.calibration_ratio >= 1:
        raise SystemExit("induction + calibration ratios must be below 1")

    output_dir = Path(args.output_dir).expanduser().resolve()
    deployment_dir = (ROOT / args.deployment_dir).resolve()
    datasets = [item.strip().lower() for item in args.datasets.split(",") if item.strip()]
    manifest: Dict[str, Any] = {
        "type": "workload_competency_split_contract",
        "version": 3,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "split": "train_only",
        "seed": args.seed,
        "ratios": {
            "induction": args.induction_ratio,
            "calibration": args.calibration_ratio,
            "cq_holdout": 1.0 - args.induction_ratio - args.calibration_ratio,
        },
        "inference_uses_gold": False,
        "partition_strategy": "stratified_support_entity_component_disjoint",
        "protocol": {
            "q_ind": {
                "role": "ontology_structure_induction_only",
                "may_mutate_ontology_structure": True,
                "may_fit_calibration_parameters": False,
            },
            "q_cal": {
                "role": "frozen_ontology_replay_and_deployment_calibration",
                "may_mutate_ontology_structure": False,
                "may_fit_calibration_parameters": True,
            },
            "q_val": {
                "role": "frozen_deployment_certification_and_mask",
                "may_mutate_ontology_structure": False,
                "may_fit_calibration_parameters": False,
            },
            "q_test": {
                "role": "untouched_final_evaluation_only",
                "may_mutate_ontology_structure": False,
                "may_fit_calibration_parameters": False,
            },
        },
        "merge_cal_val_ablation": bool(args.merge_cal_val),
        "datasets": {},
        "contract_failures": [],
    }
    for dataset in datasets:
        source = choose_source(dataset)
        eval_ids, eval_questions, eval_path = load_eval_identity(dataset, deployment_dir)
        if eval_path is None:
            manifest["contract_failures"].append(f"{dataset}:missing_heldout_eval")
        limit = min(args.max_workload_rows, DATASET_LIMITS.get(dataset, args.max_workload_rows))
        scan_limit = args.max_scan_rows
        if dataset in DATASET_SCAN_LIMITS:
            scan_limit = min(scan_limit, DATASET_SCAN_LIMITS[dataset])
        selected, scan_stats = select_rows(
            dataset,
            source,
            limit,
            args.seed,
            eval_ids,
            eval_questions,
            scan_limit,
        )
        partitions = stratified_entity_partitions(
            selected,
            dataset,
            args.induction_ratio,
            args.calibration_ratio,
            args.seed,
        )
        partition_stats = partition_diagnostics(dataset, partitions)
        if not selected:
            manifest["contract_failures"].append(f"{dataset}:empty_workload")
        if len(selected) >= 3 and any(not rows for rows in partitions.values()):
            manifest["contract_failures"].append(f"{dataset}:empty_partition")
        corpus = corpus_rows(dataset, selected)
        corpus_path = output_dir / "corpora" / f"{dataset}_corpus.json"
        write_json(corpus_path, corpus)
        identities: Dict[str, set[str]] = {}
        split_manifest: Dict[str, Any] = {}
        canonical_partitions = {
            canonical_split_name(name): list(rows)
            for name, rows in partitions.items()
        }
        if args.merge_cal_val:
            merged_dev = [
                *canonical_partitions["q_cal"],
                *canonical_partitions["q_val"],
            ]
            canonical_partitions["q_cal_val_dev"] = merged_dev
        for canonical_name, rows in canonical_partitions.items():
            path = output_dir / canonical_name / f"{dataset}.json"
            write_json(path, rows)
            if canonical_name == "q_cal_val_dev":
                continue
            legacy_name = {
                "q_ind": "induction",
                "q_cal": "calibration",
                "q_val": "cq_holdout",
            }[canonical_name]
            identities[legacy_name] = {
                stable_id(row) or normalize_text(question(row)) for row in rows
            }
            coverage = exact_gold_coverage(dataset, rows, corpus)
            split_manifest[canonical_name] = {
                "rows": len(rows),
                "path": str(path),
                "sha256": sha256_file(path),
                "role": manifest["protocol"][canonical_name]["role"],
                **coverage,
            }
            if coverage["gold_documents"] and coverage["gold_document_exact_coverage"] < 1.0:
                manifest["contract_failures"].append(
                    f"{dataset}:{canonical_name}:gold_document_exact_coverage"
                )
            if args.write_legacy_aliases:
                write_json(output_dir / legacy_name / f"{dataset}.json", rows)
        if args.merge_cal_val:
            merged_path = output_dir / "q_cal_val_dev" / f"{dataset}.json"
            split_manifest["q_cal_val_dev"] = {
                "rows": len(canonical_partitions["q_cal_val_dev"]),
                "path": str(merged_path),
                "sha256": sha256_file(merged_path),
                "ablation_only": True,
            }
        overlap_counts = {
            "induction_calibration": len(identities["induction"] & identities["calibration"]),
            "induction_cq_holdout": len(identities["induction"] & identities["cq_holdout"]),
            "calibration_cq_holdout": len(identities["calibration"] & identities["cq_holdout"]),
        }
        if any(overlap_counts.values()):
            manifest["contract_failures"].append(f"{dataset}:partition_overlap")
        if any(partition_stats["support_entity_overlap"].values()):
            manifest["contract_failures"].append(
                f"{dataset}:support_entity_partition_overlap"
            )
        manifest["datasets"][dataset] = {
            "source": str(source),
            "heldout_eval_path": str(eval_path) if eval_path else "",
            "heldout_eval_sha256": sha256_file(eval_path) if eval_path else "",
            "q_test": {
                "path": str(eval_path) if eval_path else "",
                "sha256": sha256_file(eval_path) if eval_path else "",
                "query_count": len(eval_ids),
                "untouched": True,
            },
            "selected_rows": len(selected),
            "corpus_rows": len(corpus),
            "corpus_path": str(corpus_path),
            "corpus_sha256": sha256_file(corpus_path),
            "partition_overlap": overlap_counts,
            "partition_diagnostics": partition_stats,
            "max_scan_rows": scan_limit,
            **scan_stats,
            "splits": split_manifest,
        }
        print(
            f"dataset={dataset} selected={len(selected)} corpus={len(corpus)} "
            f"splits={','.join(f'{key}:{len(value)}' for key, value in partitions.items())}",
            flush=True,
        )

    manifest["contract_ok"] = not manifest["contract_failures"]
    manifest_path = output_dir / "workload_contract.json"
    write_json(manifest_path, manifest)
    print(f"contract={manifest_path}")
    if args.strict and not manifest["contract_ok"]:
        raise SystemExit("workload contract failed: " + ",".join(manifest["contract_failures"]))


if __name__ == "__main__":
    main()
