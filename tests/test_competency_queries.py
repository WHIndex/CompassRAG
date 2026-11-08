import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, relative_path):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


preparer = load_module(
    "v217_workload_preparer_test",
    "scripts/workload.py",
)
validator = load_module(
    "v217_competency_validator_test",
    "scripts/validate_ontology.py",
)
ontology_builder = load_module(
    "query_plan_ontology_grounding_test",
    "src/compassrag/query_plan_ontology.py",
)


def make_plan(plan_id="person", answer_type="person", relation="author"):
    return {
        "id": plan_id,
        "intent": "attribute",
        "triggers": ["who", "wrote"],
        "anchor_slots": [{"name": "work", "type": "work"}],
        "path": [{
            "from": "anchor",
            "to": "answer",
            "relation_family": "creator",
            "relation_signature": relation,
            "direction": "forward",
            "terminal_object_allow_terms": ["William Shakespeare", "person"],
        }],
        "answer_type": answer_type,
        "support": 12,
        "precision": 0.8,
        "confidence": 0.8,
        "answer_constraints": {
            "allowed_terms": ["William Shakespeare", "person"],
            "object_type_rules": ["answer_type=person", "born in 1564"],
        },
        "examples": [{
            "id": "train-1",
            "question": "Who wrote Hamlet?",
            "answers": ["William Shakespeare"],
        }],
        "trace_supervision": {"gold_hits": 1},
    }


def test_hotpot_corpus_is_byte_identical_to_gold_contract():
    rows = [{
        "context": [["Hamlet", ["First sentence. ", "Second sentence."]]],
        "supporting_facts": [["Hamlet", 0]],
    }]
    corpus = preparer.corpus_rows("hotpotqa", rows)
    assert corpus == [{
        "title": "Hamlet",
        "text": "First sentence. Second sentence.",
    }]
    coverage = preparer.exact_gold_coverage("hotpotqa", rows, corpus)
    assert coverage["gold_document_exact_coverage"] == 1.0


def test_lveval_numbered_context_produces_grounding_passages():
    row = preparer.normalize_row("lveval", {
        "id": "lv-1",
        "question": "Who wrote the work?",
        "context": "### Passage 1\nFirst evidence.\n\n### Passage 2\nSecond evidence.",
    })
    corpus = preparer.corpus_rows("lveval", [row])

    assert [item["title"] for item in corpus] == ["Passage 1", "Passage 2"]
    assert row["paragraphs"][1]["paragraph_text"] == "Second evidence."


def test_ontology_examples_retain_paragraph_text_for_agent_grounding():
    ontology = ontology_builder.build_query_plan_ontology(
        rows=[{
            "id": "q1",
            "question": "Who wrote Hamlet?",
            "answer": "William Shakespeare",
            "paragraphs": [{
                "title": "Hamlet",
                "paragraph_text": "Hamlet was written by William Shakespeare.",
                "is_supporting": True,
            }],
        }],
        dataset="narrativeqa_dev_10_doc",
        min_support=1,
        min_path_steps=1,
    )

    passages = ontology["plans"][0]["examples"][0]["supporting_passages"]
    assert passages == [{
        "title": "Hamlet",
        "text": "Hamlet was written by William Shakespeare.",
    }]


def test_open_domain_location_is_not_typed_as_biomedical():
    assert ontology_builder.relation_signature(
        "located", dataset="hotpotqa"
    ) == "location"
    assert ontology_builder.relation_signature(
        "localization", dataset="bioasq_domain"
    ) == "biomedical_localization"


def test_typed_path_grouping_merges_lexical_query_variants():
    rows = [
        {
            "id": "q1",
            "question": "Who wrote Hamlet?",
            "answer": "William Shakespeare",
            "subj": "Hamlet",
            "prop": "author",
            "obj": "William Shakespeare",
        },
        {
            "id": "q2",
            "question": "Who was the writer of Macbeth?",
            "answer": "William Shakespeare",
            "subj": "Macbeth",
            "prop": "author",
            "obj": "William Shakespeare",
        },
    ]
    ontology = ontology_builder.build_query_plan_ontology(
        rows=rows,
        dataset="hotpotqa",
        min_support=1,
        min_path_steps=1,
        schema_grouping="typed_path",
    )

    assert len(ontology["plans"]) == 1
    assert ontology["plans"][0]["support"] == 2
    assert ontology["diagnostics"]["schema_grouping"] == "typed_path"


def test_rare_multihop_admission_uses_structure_not_dataset_name():
    long_chain = {
        "id": "long",
        "question": "Which city contains the school attended by the author of Hamlet?",
        "answer": "London",
        "evidences": [
            ["Hamlet", "author", "William Shakespeare"],
            ["William Shakespeare", "educated at", "King's New School"],
            ["King's New School", "located in", "London"],
        ],
    }
    short_chain = {
        "id": "short",
        "question": "Who wrote Macbeth?",
        "answer": "William Shakespeare",
        "evidences": [["Macbeth", "author", "William Shakespeare"]],
    }
    ontology = ontology_builder.build_query_plan_ontology(
        rows=[long_chain, short_chain],
        dataset="arbitrary_open_domain_workload",
        min_support=2,
        schema_grouping="typed_path",
        retain_rare_multihop=True,
    )

    assert len(ontology["plans"]) == 1
    assert len(ontology["plans"][0]["path"]) == 3
    assert ontology["plans"][0]["support"] == 1


def test_length_adaptive_grouping_factorizes_long_relation_variants():
    rows = [
        {
            "id": "q1",
            "question": "Where was the spouse of the father of Person A born?",
            "answer": "Paris",
            "evidences": [
                ["Person A", "father", "Person B"],
                ["Person B", "spouse", "Person C"],
                ["Person C", "birthplace", "Paris"],
            ],
        },
        {
            "id": "q2",
            "question": "Where was the child of the mother of Person D born?",
            "answer": "Rome",
            "evidences": [
                ["Person D", "mother", "Person E"],
                ["Person E", "child", "Person F"],
                ["Person F", "place of birth", "Rome"],
            ],
        },
    ]
    ontology = ontology_builder.build_query_plan_ontology(
        rows=rows,
        dataset="arbitrary_open_domain_workload",
        min_support=2,
        schema_grouping="length_adaptive_typed_path",
    )

    assert len(ontology["plans"]) == 1
    assert ontology["plans"][0]["support"] == 2


def test_score_margin_router_allocates_budget_by_query_uncertainty():
    policy = {
        "routing_mode": "score_margin_adaptive",
        "margin_cutoff": 0.20,
        "leaf_top_m": 5,
        "min_score": 0.10,
        "class_policies": {
            "decisive": {"leaf_top_m": 2, "min_score": 0.10},
            "ambiguous": {"leaf_top_m": 5, "min_score": 0.10},
        },
    }
    decisive = ontology_builder.routing_parameters_for_scores(
        policy, [0.90, 0.50, 0.40], requested_top_k=8
    )
    ambiguous = ontology_builder.routing_parameters_for_scores(
        policy, [0.70, 0.62, 0.50], requested_top_k=8
    )

    assert decisive[:3] == (2, 0.10, "decisive")
    assert ambiguous[:3] == (5, 0.10, "ambiguous")


def test_workload_partitions_are_disjoint_and_complete():
    rows = [{"id": str(index), "question": f"q{index}"} for index in range(20)]
    partitions = preparer.split_rows(rows, 0.70, 0.15)
    identities = [
        {preparer.stable_id(row) for row in partitions[name]}
        for name in ("induction", "calibration", "cq_holdout")
    ]
    assert [len(value) for value in identities] == [14, 3, 3]
    assert not identities[0] & identities[1]
    assert not identities[0] & identities[2]
    assert not identities[1] & identities[2]
    assert set.union(*identities) == {str(index) for index in range(20)}


def test_generic_split_is_stratified_and_support_entity_disjoint():
    rows = []
    for group in range(30):
        for offset in range(2):
            comparison = group % 2 == 0
            rows.append({
                "id": f"q-{group}-{offset}",
                "question": (
                    f"Which of Entity{group}'s works appeared earlier?"
                    if comparison
                    else f"Who wrote a work associated with Entity{group}?"
                ),
                "answer": "1900" if comparison else f"Person{group}",
                "type": "comparison" if comparison else "bridge",
                "supporting_facts": [[f"Entity{group}", offset]],
            })

    partitions = preparer.stratified_entity_partitions(
        rows, "hotpotqa", 0.70, 0.15, "test-seed"
    )
    diagnostics = preparer.partition_diagnostics("hotpotqa", partitions)
    identities = [
        {preparer.stable_id(row) for row in partitions[name]}
        for name in ("induction", "calibration", "cq_holdout")
    ]

    assert not any(diagnostics["support_entity_overlap"].values())
    assert set.union(*identities) == {preparer.stable_id(row) for row in rows}
    assert all(diagnostics["stratum_counts"][name] for name in partitions)


def test_route_features_do_not_use_gold_answers():
    profile = validator.row_profile({
        "question": "Name the event associated with the archive.",
        "answer": "1999",
        "subj": "archive",
        "prop": "year",
        "obj": "1999",
    })
    assert profile["route_answer_type"] == "entity"
    assert profile["target_answer_type"] == "date"


def test_competency_evaluation_prefers_structurally_valid_schema():
    rows = [{
        "question": "Who wrote Hamlet?",
        "answer": "William Shakespeare",
        "subj": "Hamlet",
        "prop": "author",
        "obj": "William Shakespeare",
    }]
    good = make_plan()
    bad = make_plan("location", "location", "located")
    summary = validator.evaluate([good, bad], rows, top_k=1, min_route_score=0.1)
    assert summary["plans"]["person"]["true_positive"] == 1
    assert summary["plans"]["location"]["true_positive"] == 0
    assert summary["routing_precision"] == 1.0


def test_sanitizer_removes_instances_examples_and_gold_supervision():
    sanitized, stats = validator.sanitize_plan(make_plan())
    serialized = str(sanitized).lower()
    assert "william shakespeare" not in serialized
    assert "born in 1564" not in serialized
    assert "examples" not in sanitized
    assert "trace_supervision" not in sanitized
    assert sanitized["integrity_constraints"]["range_type"] == "person"
    assert sanitized["training_provenance"]["runtime_uses_examples"] is False
    assert stats["removed_supervision_fields"] == 1


def test_global_conflict_reports_the_lower_validation_score():
    strong = make_plan("strong", "person", "author")
    weak = make_plan("weak", "location", "located")
    for plan, tp, fp in ((strong, 8, 1), (weak, 1, 8)):
        plan["competency_validation"] = {
            "cq_holdout": {
                "routed": tp + fp,
                "true_positive": tp,
                "false_positive": fp,
                "posterior_precision": (tp + 1) / (tp + fp + 2),
                "recall": 0.8 if plan is strong else 0.1,
            }
        }
    conflicts = validator.detect_conflicts([strong, weak])
    assert len(conflicts) == 1
    assert conflicts[0]["winner_plan_id"] == "strong"
    assert conflicts[0]["loser_plan_id"] == "weak"
    assert conflicts[0]["score_gap"] > 0.1


def test_plan_admission_uses_calibration_and_not_cq_holdout():
    plan = make_plan()
    plan["competency_validation"] = {
        "calibration": {
            "true_positive": 0,
            "false_positive": 8,
            "posterior_precision": 0.1,
        },
        "cq_holdout": {
            "true_positive": 20,
            "false_positive": 0,
            "posterior_precision": 0.95,
        },
    }
    status, _ = validator.status_for_plan(plan, min_precision=0.5, min_true_positives=1)
    assert status == "rejected"


def test_structural_rare_status_requires_a_long_chain():
    plan = make_plan()
    plan["support"] = 1
    plan["competency_validation"] = {
        "calibration": {"true_positive": 0, "false_positive": 0, "opportunities": 0}
    }
    status, _ = validator.status_for_plan(
        plan,
        min_precision=0.5,
        min_true_positives=2,
        retain_structural_rare=True,
    )
    assert status == "rejected"

    plan["path"] = [dict(plan["path"][0], step=index) for index in range(3)]
    status, reason = validator.status_for_plan(
        plan,
        min_precision=0.5,
        min_true_positives=2,
        retain_structural_rare=True,
    )
    assert status == "structural_rare"
    assert "long_chain" in reason


def test_hierarchical_router_uses_induction_statistics():
    treatment = make_plan("treatment", "drug", "treatment")
    treatment["intent"] = "biomedical_treatment"
    treatment["routing_training"] = {
        "document_count": 10,
        "feature_document_frequency": {
            "u:treatment": 10,
            "u:therapy": 8,
            "u:disease": 6,
            "t:drug": 8,
        },
    }
    genetics = make_plan("genetics", "gene", "genetic association")
    genetics["intent"] = "biomedical_genetics"
    genetics["routing_training"] = {
        "document_count": 10,
        "feature_document_frequency": {
            "u:gene": 10,
            "u:mutation": 8,
            "u:disease": 6,
            "t:gene": 8,
        },
    }
    plans = [treatment, genetics]
    model = validator.build_hierarchical_routing_model(plans, "bioasq_domain")
    matcher = validator.QueryPlanOntologyMatcher({
        "dataset": "bioasq_domain",
        "plans": plans,
        "routing_model": model,
    })
    ranked = matcher.rank("Which therapy is used for this disease?", min_confidence=0.0)
    assert ranked[0][1]["id"] == "treatment"
    assert ranked[0][1]["parent_schema_id"] == "intent:biomedical_treatment"
    policy, diagnostics = validator.calibrate_routing_policy(
        plans,
        [{
            "question": "Which therapy is used for this disease?",
            "answer": "a drug",
            "subj": "disease",
            "prop": "treatment",
            "obj": "drug",
        }],
        model,
        max_top_m=2,
        target_precision=0.5,
    )
    assert policy["parent_top_m"] in {1, 2}
    assert policy["leaf_top_m"] in {1, 2}
    assert diagnostics["candidate_policy_count"] > 0
