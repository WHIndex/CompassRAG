from compassrag.query_plan_factorized_executor import QueryPlanFactorizedExecutor


def _empty_dense(_query):
    return [], []


def _empty_record(_passage_idx):
    return {}


def test_lossless_ir_preserves_variables_directions_and_order():
    executor = QueryPlanFactorizedExecutor(enable_lossless_typed_plan_ir=True)
    typed_ir = executor._compile_lossless_plan_ir(
        query="Where was the director of Inception born?",
        candidate={
            "plan_id": "director_birth_place",
            "path": [
                {
                    "step": 0,
                    "from": "film",
                    "relation_signature": "director",
                    "direction": "forward",
                    "to": "person",
                },
                {
                    "step": 1,
                    "from": "person",
                    "relation_signature": "birth place",
                    "direction": "forward",
                    "to": "answer",
                },
            ],
        },
        anchors=["Inception"],
        answer_type="location",
        comparison=False,
        comparison_branches={},
    )

    assert typed_ir["valid"] is True
    assert typed_ir["initial_bindings"] == {"film": "Inception"}
    assert [
        (
            edge["input_variable"],
            edge["relation"],
            edge["direction"],
            edge["output_variable"],
        )
        for edge in typed_ir["edges"]
    ] == [
        ("film", "director", "forward", "person"),
        ("person", "birth place", "forward", "answer"),
    ]


def test_lossless_ir_recovers_two_branches_from_flat_comparison_schema():
    executor = QueryPlanFactorizedExecutor(enable_lossless_typed_plan_ir=True)
    typed_ir = executor._compile_lossless_plan_ir(
        query="Which film has the younger director, Alpha or Beta?",
        candidate={
            "plan_id": "compare_director_birth_date",
            "path": [
                {"step": 0, "relation_signature": "director", "direction": "forward"},
                {"step": 1, "relation_signature": "birth date", "direction": "forward"},
                {"step": 2, "relation_signature": "director", "direction": "forward"},
                {"step": 3, "relation_signature": "birth date", "direction": "forward"},
            ],
        },
        anchors=["Alpha", "Beta"],
        answer_type="entity",
        comparison=True,
        comparison_branches={},
    )

    assert typed_ir["valid"] is True
    assert typed_ir["topology"] == "dual_branch_comparison"
    assert typed_ir["branch_ids"] == ["branch_0", "branch_1"]
    assert len(typed_ir["edges"]) == 4
    assert [edge["relation"] for edge in typed_ir["edges"]] == [
        "director",
        "birth date",
        "director",
        "birth date",
    ]
    assert typed_ir["projection"]["operator"] == "argmax"


def test_typed_variable_unification_emits_complete_binding_certificate():
    rows = {
        "inception": [{
            "passage_idx": 3,
            "subject": "Inception",
            "predicate": "director",
            "object": "Christopher Nolan",
            "entity_match_score": 1.0,
        }],
        "christopher nolan": [{
            "passage_idx": 7,
            "subject": "Christopher Nolan",
            "predicate": "birth place",
            "object": "London",
            "entity_match_score": 1.0,
        }],
    }
    executor = QueryPlanFactorizedExecutor(
        enable_lossless_typed_plan_ir=True,
        require_typed_variable_unification=True,
        min_relation_score=0.25,
        min_type_score=0.20,
        min_candidate_utility=0.0,
        cost_weight=0.0,
    )
    typed_ir = executor._compile_lossless_plan_ir(
        query="Where was the director of Inception born?",
        candidate={
            "plan_id": "director_birth_place",
            "path": [
                {
                    "step": 0,
                    "from": "film",
                    "relation_signature": "director",
                    "direction": "forward",
                    "to": "person",
                },
                {
                    "step": 1,
                    "from": "person",
                    "relation_signature": "birth place",
                    "direction": "forward",
                    "to": "answer",
                },
            ],
        },
        anchors=["Inception"],
        answer_type="location",
        comparison=False,
        comparison_branches={},
    )

    candidates, groups = executor._typed_variable_unification_join(
        query="Where was the director of Inception born?",
        typed_ir=typed_ir,
        anchors=["Inception"],
        answer_type="location",
        dense_retrieve=_empty_dense,
        get_openie_record=_empty_record,
        lookup_entity_relations=lambda entity, limit: rows.get(entity.lower(), [])[:limit],
        probes=[],
    )

    assert {candidate["passage_idx"] for candidate in candidates} == {3, 7}
    assert len(groups) == 1
    certificate = groups[0]["binding_certificate"]
    assert certificate["valid"] is True
    assert certificate["complete"] is True
    assert certificate["realized_edge_count"] == 2
    assert certificate["variable_bindings"] == {
        "film": "Inception",
        "person": "Christopher Nolan",
        "answer": "London",
    }
    assert [item["direction"] for item in certificate["lineage"]] == [
        "forward",
        "forward",
    ]
