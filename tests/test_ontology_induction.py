import json
import subprocess
import sys
from pathlib import Path

from compassrag.query_plan_ontology import (
    QueryPlanOntologyMatcher,
    build_query_plan_ontology,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from construct_evidence_ontology import (
    deterministic_contrastive_report,
    hierarchical_aggregate,
    retain_parent_incumbents,
)


ROOT = Path(__file__).resolve().parents[1]


def test_multiagent_refinement_preserves_qind_parent_as_memo_incumbent():
    parent = {
        "id": "schema_parent",
        "support": 12,
        "path": [{"relation_family": "creator"}],
        "examples": [{"id": "q1"}, {"id": "q2"}],
    }

    retained = retain_parent_incumbents([parent])

    assert len(retained) == 1
    assert retained[0]["id"] == "schema_parent"
    assert retained[0]["memo_role"] == "q_ind_parent_incumbent"
    assert retained[0]["global_supervisor_retention"]["deployment_authority"] == (
        "fixed_q_val_schema_mask"
    )
    assert "memo_role" not in parent


def test_hierarchical_aggregation_does_not_merge_distinct_parent_incumbents():
    parents = retain_parent_incumbents([{
        "id": "schema_author",
        "intent": "creator",
        "answer_type": "person",
        "triggers": ["author"],
        "path": [{"relation_family": "creator"}],
    }, {
        "id": "schema_director",
        "intent": "creator",
        "answer_type": "person",
        "triggers": ["director"],
        "path": [{"relation_family": "creator"}],
    }])

    aggregated, _ = hierarchical_aggregate(parents, [])

    assert {item["id"] for item in aggregated} == {
        "schema_author",
        "schema_director",
    }


def test_real_corpus_hard_negatives_exclude_gold_and_cluster_every_qind_query(tmp_path):
    rows = [{
        "id": "q1",
        "question": "Where was the author of Hamlet born?",
        "answer": "Stratford-upon-Avon",
        "supporting_facts": [["Hamlet", 0], ["William Shakespeare", 0]],
        "context": [
            ["Hamlet", ["Hamlet was written by William Shakespeare."]],
            ["William Shakespeare", ["William Shakespeare was born in Stratford-upon-Avon."]],
        ],
        "evidences": [
            ["Hamlet", "author", "William Shakespeare"],
            ["William Shakespeare", "born in", "Stratford-upon-Avon"],
        ],
    }, {
        "id": "q2",
        "question": "Who directed Inception?",
        "answer": "Christopher Nolan",
        "supporting_facts": [["Inception", 0]],
        "context": [["Inception", ["Inception was directed by Christopher Nolan."]]],
        "evidences": [["Inception", "directed by", "Christopher Nolan"]],
    }]
    corpus = [
        {"title": "Hamlet", "text": "Hamlet was written by William Shakespeare."},
        {"title": "William Shakespeare", "text": "William Shakespeare was born in Stratford-upon-Avon."},
        {"title": "Hamlet adaptations", "text": "Hamlet has many film adaptations and directors."},
        {"title": "Inception", "text": "Inception was directed by Christopher Nolan."},
        {"title": "Inception soundtrack", "text": "The Inception soundtrack was composed for the film."},
    ]
    q_ind = tmp_path / "q_ind.json"
    corpus_path = tmp_path / "corpus.json"
    output = tmp_path / "cqs.json"
    q_ind.write_text(json.dumps(rows), encoding="utf-8")
    corpus_path.write_text(json.dumps(corpus), encoding="utf-8")

    result = subprocess.run([
        sys.executable,
        str(ROOT / "scripts/build_competency_queries.py"),
        "--dataset", "hotpotqa",
        "--q-ind", str(q_ind),
        "--corpus", str(corpus_path),
        "--output", str(output),
        "--retrieve-k", "5",
        "--hard-negatives", "2",
        "--seed-count", "1",
        "--clusters", "2",
        "--max-cluster-capacity", "1",
    ], capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert {item["query_id"] for item in payload["records"]} == {"q1", "q2"}
    assert all(item["cluster_id"] >= 0 for item in payload["records"])
    assert payload["initialization"]["all_q_ind_refined"] is True
    for record in payload["records"]:
        assert not set(record["gt_evidence_ids"]) & set(record["hard_negative_ids"])
        assert all(item["source"] == "real_corpus_sparse_near_miss" for item in record["hard_negatives"])


def test_freeze_reads_only_qind_and_removes_runtime_examples(tmp_path):
    q_ind = tmp_path / "q_ind.json"
    q_ind.write_text(json.dumps([{"id": "q1", "question": "Who wrote Hamlet?"}]), encoding="utf-8")
    source = tmp_path / "induced.json"
    source.write_text(json.dumps({
        "dataset": "sample",
        "plans": [{
            "id": "schema_1",
            "intent": "creator",
            "triggers": ["wrote"],
            "anchor_slots": [{"name": "anchor", "type": "work"}],
            "path": [{
                "from": "anchor", "to": "answer",
                "relation_family": "creator",
                "relation_signature": "author",
                "direction": "forward",
            }],
            "answer_type": "person",
            "support": 1,
            "precision": 0.8,
            "confidence": 0.8,
            "examples": [{"id": "q1", "answers": ["William Shakespeare"]}],
        }],
    }), encoding="utf-8")
    output = tmp_path / "frozen.json"

    result = subprocess.run([
        sys.executable,
        str(ROOT / "scripts/freeze_ontology.py"),
        "--input", str(source),
        "--output", str(output),
        "--q-ind", str(q_ind),
    ], capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stderr
    frozen = json.loads(output.read_text(encoding="utf-8"))
    assert frozen["structure_source_split"] == "q_ind"
    assert frozen["q_cal_read"] is False
    assert frozen["q_val_read"] is False
    assert frozen["q_test_read"] is False
    assert "examples" not in frozen["plans"][0]


def test_contract_path_keeps_logical_shape_and_physical_relation_domains():
    rows = [{
        "id": "q1",
        "question": "Where was the author of Hamlet born?",
        "answer": "Stratford-upon-Avon",
        "evidences": [
            ["Hamlet", "author", "William Shakespeare"],
            ["William Shakespeare", "born in", "Stratford-upon-Avon"],
        ],
    }, {
        "id": "q2",
        "question": "Where was the director of Inception born?",
        "answer": "London",
        "evidences": [
            ["Inception", "directed by", "Christopher Nolan"],
            ["Christopher Nolan", "birth place", "London"],
        ],
    }]

    ontology = build_query_plan_ontology(
        rows,
        dataset="sample",
        min_support=1,
        schema_grouping="contract_path",
    )

    assert len(ontology["plans"]) == 1
    path = ontology["plans"][0]["path"]
    assert {"creator"} <= set(path[0]["relation_family_alternatives"])
    assert {"author", "director"} <= set(path[0]["relation_signature_alternatives"])
    assert {"birth_place"} <= set(path[1]["relation_signature_alternatives"])


def test_signed_contract_path_separates_relation_sequences_and_directions():
    rows = [{
        "id": "q1",
        "question": "Where was the author of Hamlet born?",
        "answer": "Stratford-upon-Avon",
        "evidences": [
            ["Hamlet", "author", "William Shakespeare"],
            ["William Shakespeare", "born in", "Stratford-upon-Avon"],
        ],
    }, {
        "id": "q2",
        "question": "Which film was Christopher Nolan the director of?",
        "answer": "Inception",
        "evidences": [["Inception", "directed by", "Christopher Nolan"]],
    }]

    ontology = build_query_plan_ontology(
        rows,
        dataset="sample",
        min_support=1,
        schema_grouping="signed_contract_path",
    )

    assert len(ontology["plans"]) == 2
    paths = [plan["path"] for plan in ontology["plans"]]
    assert any(path[0]["relation_signature"] == "author" for path in paths)
    director_path = next(
        path for path in paths if path[0]["relation_signature"] == "director"
    )
    assert director_path[0]["direction"] == "reverse"


def test_dynamic_wrong_schema_pairs_train_a_contrastive_router():
    rows = [{
        "id": "author_q",
        "question": "Who was the author of Hamlet?",
        "answer": "William Shakespeare",
        "evidences": [["Hamlet", "author", "William Shakespeare"]],
    }, {
        "id": "director_q",
        "question": "Who was the director of Inception?",
        "answer": "Christopher Nolan",
        "evidences": [["Inception", "directed by", "Christopher Nolan"]],
    }]
    initial = build_query_plan_ontology(
        rows,
        dataset="sample",
        min_support=1,
        schema_grouping="signed_contract_path",
    )
    director_id = next(
        plan["id"] for plan in initial["plans"]
        if plan["path"][0]["relation_signature"] == "director"
    )
    records = {
        "author_q": {
            "query": rows[0]["question"],
            "ontology_induced_hard_negatives": [{"schema_id": director_id}],
        },
        "director_q": {"query": rows[1]["question"]},
    }

    ontology = build_query_plan_ontology(
        rows,
        dataset="sample",
        min_support=1,
        schema_grouping="signed_contract_path",
        contrastive_cq_records=records,
    )

    assert ontology["routing_model"]["type"] == (
        "hierarchical_contrastive_tfidf_schema_router"
    )
    assert ontology["diagnostics"]["contrastive_wrong_schema_pair_count"] == 1
    ranked = QueryPlanOntologyMatcher(ontology).rank(rows[0]["question"])
    assert ranked[0][1]["path"][0]["relation_signature"] == "author"


def test_wrong_relation_negative_is_a_deterministic_contrastive_check():
    plan = {
        "intent": "person_bridge",
        "answer_type": "location",
        "path": [{
            "relation_signature": "author",
            "relation_family": "creator",
            "direction": "forward",
            "from": "anchor",
            "to": "answer",
        }],
        "examples": [{
            "contrastive_cq": {
                "hard_negatives": [{
                    "doc_id": "schema:wrong",
                    "negative_types": ["wrong_relation"],
                    "negative_schema_contract": {
                        "intent": "person_bridge",
                        "answer_type": "location",
                        "path": [{
                            "relation_signature": "director",
                            "direction": "forward",
                        }],
                    },
                }],
            },
        }],
    }

    report = deterministic_contrastive_report(plan)

    assert report["testable_hard_negative_count"] == 1
    assert report["hard_negative_rejection_rate"] == 1.0


def test_deployment_mask_filters_original_topk_without_refill():
    plans = [{
        "id": "rejected_top1",
        "path": [{"relation_family": "creator"}],
        "confidence": 0.9,
    }, {
        "id": "certified_rank2",
        "path": [{"relation_family": "creator"}],
        "confidence": 0.9,
    }]
    matcher = QueryPlanOntologyMatcher({
        "plans": plans,
        "routing_model": {
            "type": "hierarchical_tfidf_schema_router",
            "policy": {
                "leaf_top_m": 1,
                "min_score": 0.0,
                "deployment_schema_ids": ["certified_rank2"],
            },
        },
    })
    matcher.rank = lambda *_args, **_kwargs: [(0.9, plans[0]), (0.8, plans[1])]

    assert matcher.match("who created it", top_k=2, min_confidence=0.0) == []
