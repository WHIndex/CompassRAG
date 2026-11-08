import json

from compassrag.ontology_evidence_query_optimizer import (
    CostBasedEvidenceQueryOptimizer,
    EvidenceCatalog,
    OntologyEvidenceQueryCompiler,
)
from compassrag.query_plan_factorized_executor import QueryPlanFactorizedExecutor


def _catalog(tmp_path, profiles):
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps({
        "type": "ontology_evidence_catalog",
        "version": 1,
        "prior_strength": 2.0,
        "metadata": {
            "source_split": "train",
            "training_uses_gold": True,
            "inference_uses_gold": False,
        },
        "profiles": profiles,
    }), encoding="utf-8")
    return path


def _profile(operator, shape, strategy="adaptive", **overrides):
    profile = {
        "dataset": "*",
        "operator": operator,
        "query_shape": shape,
        "access_strategy": strategy,
        "sample_count": 100,
        "candidate_hit_rate": 0.55,
        "candidate_precision": 0.60,
        "mean_delta_r2": 0.04,
        "mean_delta_r5": 0.03,
        "loss_rate": 0.02,
        "mean_cost": 1.0,
        "proof_coverage": 0.80,
    }
    profile.update(overrides)
    return profile


def test_compiler_emits_typed_multi_hop_algebra():
    plan = OntologyEvidenceQueryCompiler().compile(
        query="Where was Alice's mother born?",
        dataset="musique",
        anchors=["Alice"],
        relations=["mother", "born in"],
        answer_type="location",
        comparison=False,
        typed_contract={
            "admissible": True,
            "query_predicate_coverage": 1.0,
            "contract_confidence": 0.9,
        },
    )

    assert plan.admissible is True
    assert plan.query_shape == "multi_hop"
    assert plan.required_hop_count == 2
    assert "join_connectivity" in plan.proof_obligations
    assert [item["op"] for item in plan.operators] == [
        "Anchor", "Traverse", "Traverse", "BindJoin", "Verify", "Project"
    ]


def test_compiler_requires_two_comparison_branches():
    plan = OntologyEvidenceQueryCompiler().compile(
        query="Which was founded earlier?",
        dataset="2wikimultihopqa",
        anchors=["Alpha"],
        relations=["founded"],
        answer_type="date",
        comparison=True,
    )

    assert plan.admissible is False
    assert plan.rejection_reason == "logical_plan_incomplete_branches"


def test_catalog_rejects_non_train_artifact(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({
        "type": "ontology_evidence_catalog",
        "version": 1,
        "metadata": {"source_split": "test", "inference_uses_gold": False},
        "profiles": [_profile("entity_relation_index_scan", "single_hop")],
    }), encoding="utf-8")

    try:
        EvidenceCatalog.from_path(str(path))
    except ValueError as exc:
        assert "source_split=train" in str(exc)
    else:
        raise AssertionError("test-split catalog should be rejected")


def test_optimizer_selects_lower_cost_index_plan_when_quality_is_sufficient(tmp_path):
    path = _catalog(tmp_path, [
        _profile("entity_relation_index_scan", "single_hop", mean_cost=4.0),
    ])
    logical = OntologyEvidenceQueryCompiler().compile(
        query="What is the capital of France?",
        dataset="popqa",
        anchors=["France"],
        relations=["capital"],
        answer_type="location",
        comparison=False,
    )
    optimizer = CostBasedEvidenceQueryOptimizer(
        catalog=EvidenceCatalog.from_path(str(path)),
        cost_weight=0.80,
        no_op_utility=-1.0,
    )
    decision = optimizer.optimize(
        logical_plan=logical,
        enabled_operators={"entity_relation_index_scan": True},
    )

    assert decision["selected"]["access_strategy"] == "index_only"
    assert decision["alternative_count"] == 4
    assert decision["selected"]["max_index_rows"] <= 64


def test_optimizer_can_choose_noop_under_proof_constraint(tmp_path):
    path = _catalog(tmp_path, [
        _profile(
            "cascaded_bind_join",
            "multi_hop",
            proof_coverage=0.05,
        ),
    ])
    logical = OntologyEvidenceQueryCompiler().compile(
        query="Where was Alice's mother born?",
        dataset="musique",
        anchors=["Alice"],
        relations=["mother", "born in"],
        answer_type="location",
        comparison=False,
    )
    decision = CostBasedEvidenceQueryOptimizer(
        catalog=EvidenceCatalog.from_path(str(path)),
        min_proof_coverage=0.50,
    ).optimize(
        logical_plan=logical,
        enabled_operators={"cascaded_bind_join": True},
    )

    assert decision["selected"] == {}
    assert decision["decision_reason"] == "no_admissible_physical_plan"


def test_factorized_executor_honors_catalog_selected_access_path(tmp_path):
    path = _catalog(tmp_path, [
        _profile("entity_relation_index_scan", "single_hop", mean_cost=3.0),
    ])
    rows = {
        "france": [{
            "passage_idx": 9,
            "subject": "France",
            "predicate": "capital",
            "object": "Paris",
            "entity_match_score": 1.0,
        }],
    }
    executor = QueryPlanFactorizedExecutor(
        enable_typed_semantic_compiler=True,
        enable_evidence_query_compiler=True,
        enable_cost_based_physical_planner=True,
        evidence_catalog_path=str(path),
        physical_planner_cost_weight=0.8,
        physical_planner_no_op_utility=-1.0,
        min_candidate_utility=0.60,
    )
    result = executor.run(
        query="What is the capital of France?",
        dataset="popqa",
        base_doc_ids=list(range(20)),
        query_plan_log={
            "enabled": True,
            "anchor_entities": ["France"],
            "query_condition": {
                "answer_type": "location",
                "answer_hints": ["location"],
                "relation_terms": ["capital"],
                "relation_signatures": ["capital"],
            },
            "matched_plans": [{
                "id": "capital-plan",
                "path": [{"relation": "capital"}],
            }],
            "sample_paths": [],
        },
        dense_retrieve=lambda _query: ([], []),
        get_openie_record=lambda _passage_idx: {},
        lookup_entity_relations=lambda entity, limit: rows.get(entity.lower(), [])[:limit],
        runtime_statistics={"entity_key_count": 1, "materialized_row_count": 1},
    )

    assert result.candidates[0]["passage_idx"] == 9
    assert result.log["logical_evidence_plan"]["query_shape"] == "single_hop"
    assert result.log["physical_plan_optimizer"]["selected_operator"] == "entity_relation_index_scan"
    assert result.log["physical_access_strategy"] == "index_only"


def test_constrained_optimizer_uses_observed_schema_profile_without_magic_weights(tmp_path):
    profiles = [
        _profile(
            "entity_relation_index_scan",
            "single_hop",
            strategy="adaptive",
            mean_delta_r2=0.01,
            mean_delta_r5=0.02,
            delta_r2_standard_error=0.001,
            delta_r5_standard_error=0.001,
            loss_count=1,
        ),
        _profile(
            "entity_relation_index_scan",
            "single_hop",
            strategy="index_only",
            mean_delta_r2=0.06,
            mean_delta_r5=0.03,
            delta_r2_standard_error=0.001,
            delta_r5_standard_error=0.001,
            loss_count=1,
        ),
    ]
    path = _catalog(tmp_path, profiles)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema_profiles"] = [{
        **profiles[1],
        "dataset": "popqa",
        "schema_id": "capital-plan",
    }]
    path.write_text(json.dumps(payload), encoding="utf-8")
    logical = OntologyEvidenceQueryCompiler().compile(
        query="What is the capital of France?",
        dataset="popqa",
        anchors=["France"],
        relations=["capital"],
        answer_type="location",
        comparison=False,
        source_schema_ids=["capital-plan"],
    )
    decision = CostBasedEvidenceQueryOptimizer(
        catalog=EvidenceCatalog.from_path(str(path)),
        selection_mode="constrained_lexicographic",
        max_loss_probability=0.25,
        min_catalog_samples=5,
    ).optimize(
        logical_plan=logical,
        enabled_operators={"entity_relation_index_scan": True},
    )

    assert decision["selection_mode"] == "constrained_lexicographic"
    assert decision["selected"]["access_strategy"] == "index_only"
    assert decision["selected"]["catalog_match_level"] == "schema_observed:capital-plan"
    assert "weighted" not in decision["decision_rule"]
