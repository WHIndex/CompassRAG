import json
import importlib.util
from types import SimpleNamespace
from pathlib import Path

from compassrag.query_plan_complete_evidence_plan_optimizer import (
    ALTERNATIVE_PLANS,
    FEATURE_NAMES,
    MODEL_TYPE,
    OBJECTIVE_CLASS_EXPERT_PREFIX,
    objective_family_model_key,
    PLAN_BOUND_PROOF,
    PLAN_HYBRID,
    PLAN_LEARNED_RESIDUAL_TOP5,
    PLAN_ONTOLOGY,
    PLAN_ONTOLOGY_PHYSICAL,
    PLAN_ONTOLOGY_STRUCTURAL_TOP5,
    PLAN_ONTOLOGY_WAREHOUSE_TOP5,
    PLAN_TOPOLOGY_MULTI_ACCESS_TOP5,
    PLAN_PROOF_CHAIN,
    PLAN_RANK_PARTITIONED_TOP2,
    PLAN_SEMANTIC,
    PLAN_SIGNATURE_GROUNDED,
    PLAN_QUERY_LOCAL_TOP2_PAIR,
    PLAN_QUERY_LOCAL_TOP5,
    PLAN_CHAIN_COMPLETION,
    PLAN_TOP2_SEMIJOIN,
    PLAN_TOP5_SEMIJOIN,
    TOPOLOGY_ROUTING_BOUNDED_V282,
    TOPOLOGY_ROUTING_BOUNDED_V283,
    TOPOLOGY_ROUTING_WAREHOUSE_V284,
    TOPOLOGY_ROUTING_COMBINED_V285,
    TOPOLOGY_ROUTING_HIERARCHICAL_V286,
    TOPOLOGY_ROUTING_OBJECTIVE_FAMILY_V287,
    TOPOLOGY_ROUTING_MULTI_ACCESS_V289,
    CompleteEvidencePlanPortfolioOptimizer,
    compose_hierarchical_topk,
    materialize_complete_plan_portfolio,
    plan_allowed_for_topology,
    query_plan_topology,
    topology_partitioned_plan_families,
)


def _record():
    semantic_ids = [1, 2, 3, 4, 5, 6, 7]
    candidates = []
    for rank, passage_idx in enumerate([10, 11, 12, 13, 14, 15], start=1):
        candidates.append({
            "passage_idx": passage_idx,
            "original_rank": rank + 5,
            "content_available": True,
            "has_proof_signal": True,
            "local_proof_path_complete": passage_idx in {10, 11},
            "bound_proof_path_complete": passage_idx in {10, 11},
            "complete_provenance": passage_idx in {10, 11},
            "binding": passage_idx in {10, 11},
            "relation_coverage_score": 1.0 if passage_idx in {10, 11} else 0.5,
            "terminal_relation_match": passage_idx in {11, 12},
            "answer_type_score": 1.0 if passage_idx == 11 else 0.0,
            "answer_type_compatible": passage_idx == 11,
            "anchor_coverage": 1.0 if passage_idx == 10 else 0.0,
            "query_coverage": 0.4,
        })
    return {
        "matched_plan_count": 2,
        "strict_executable_path_count": 1,
        "executable_path_count": 2,
        "semantic_plan_incumbent": {
            "doc_ids": semantic_ids[:5],
            "doc_scores": [1.0, 0.9, 0.8, 0.7, 0.6],
            "candidate_window_doc_ids": semantic_ids,
            "candidate_window_doc_scores": [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
        },
        "ontology_physical_plan_incumbent": {
            "doc_ids": [1, 10, 11, 2, 3],
            "doc_scores": [1.0, 0.95, 0.9, 0.85, 0.8],
            "candidate_window_doc_ids": [1, 10, 11, 2, 3, 4, 5],
            "candidate_window_doc_scores": [
                1.0, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6
            ],
        },
        "full_aperture_proof_slate": {
            "candidate_count": len(candidates),
            "relation_slot_count": 2,
            "anchors": ["anchor"],
            "proof_signal_candidate_count": len(candidates),
            "complete_single_passage_proof_count": 2,
            "local_proof_path_count": 1,
            "local_proof_path_passage_count": 2,
            "bound_proof_path_count": 1,
            "bound_proof_path_passage_count": 2,
            "local_proof_paths": [{
                "passage_ids": [10, 11],
                "min_relation_score": 0.9,
            }],
            "bound_proof_paths": [{
                "passage_ids": [10, 11],
                "min_relation_score": 0.9,
                "anchor_endpoint_bound": True,
                "binding_consistent": True,
            }],
            "candidates": candidates,
        },
    }


def _constant_head(value):
    return {
        "type": "json_random_forest_regressor",
        "feature_names": list(FEATURE_NAMES),
        "trees": [{
            "children_left": [-1],
            "children_right": [-1],
            "feature": [-2],
            "threshold": [-2.0],
            "value": [value],
        }],
        "fallback_value": value,
    }


def test_v289_materializes_topology_conditioned_complete_slates():
    record = _record()
    record["full_aperture_proof_slate"]["query_structure_signature"] = {
        "topology": "ordered_multihop_chain",
        "relation_slot_count": 2,
        "anchor_count": 1,
    }
    portfolio = materialize_complete_plan_portfolio(
        record,
        [1, 2, 3, 4, 5, 6, 7],
        [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
        allowed_plan_families={PLAN_TOPOLOGY_MULTI_ACCESS_TOP5},
    )
    plans = [
        plan for plan in portfolio["plans"].values()
        if plan.get("plan_family") == PLAN_TOPOLOGY_MULTI_ACCESS_TOP5
        and plan.get("valid")
    ]
    assert plans
    assert portfolio["topology_multi_access_slate_count"] >= 1
    for plan in plans:
        assert plan["doc_ids"][:2] == [1, 2]
        assert set(plan["doc_ids"][:5]) - {1, 2, 3, 4, 5}
        assert plan["features"][
            "plan_is_topology_conditioned_multi_access_top5"
        ] == 1.0
        assert plan["features"]["query_candidate_count_log"] > 0.0


def test_v289_query_grouped_shortlist_is_shared_by_materializer(monkeypatch):
    record = _record()
    monkeypatch.setenv(
        "QUERY_PLAN_ONTOLOGY_COMPLETE_PLAN_SHORTLIST_PER_FAMILY", "2"
    )
    portfolio = materialize_complete_plan_portfolio(
        record,
        [1, 2, 3, 4, 5, 6, 7],
        [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
        allowed_plan_families={
            PLAN_LEARNED_RESIDUAL_TOP5,
            PLAN_ONTOLOGY_WAREHOUSE_TOP5,
        },
    )
    diagnostics = portfolio["query_grouped_shortlist"]
    assert diagnostics["enabled"]
    assert diagnostics["limit_per_family"] == 2
    for family in (
        PLAN_LEARNED_RESIDUAL_TOP5,
        PLAN_ONTOLOGY_WAREHOUSE_TOP5,
    ):
        valid = [
            plan for plan in portfolio["plans"].values()
            if plan.get("plan_family") == family and plan.get("valid")
        ]
        assert len(valid) <= 2


def test_v289_topology_profile_exposes_multi_access_without_dataset_routing():
    routed = topology_partitioned_plan_families(
        TOPOLOGY_ROUTING_MULTI_ACCESS_V289
    )
    assert PLAN_TOPOLOGY_MULTI_ACCESS_TOP5 in routed["ordered_multihop_chain"]
    assert PLAN_TOPOLOGY_MULTI_ACCESS_TOP5 in routed["branch_comparison"]


def test_materializes_independent_ontology_plan_and_explicit_merge():
    record = _record()
    portfolio = materialize_complete_plan_portfolio(
        record, [1, 2, 3, 4, 5, 10, 11], [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
    )
    plans = portfolio["plans"]
    assert plans[PLAN_SEMANTIC]["doc_ids"] == [1, 2, 3, 4, 5]
    assert plans[PLAN_ONTOLOGY_PHYSICAL]["doc_ids"] == [1, 10, 11, 2, 3]
    assert plans[PLAN_ONTOLOGY_PHYSICAL]["valid"]
    assert plans[PLAN_ONTOLOGY_PHYSICAL]["features"][
        "plan_is_ontology_physical_complete_slate"
    ] == 1.0
    assert plans[PLAN_ONTOLOGY]["doc_ids"][:2] == [10, 11]
    assert not (set(plans[PLAN_ONTOLOGY]["doc_ids"]) & {1, 2, 3, 4, 5})
    assert plans[PLAN_PROOF_CHAIN]["doc_ids"][:2] == [10, 11]
    assert plans[PLAN_BOUND_PROOF]["doc_ids"][:2] == [10, 11]
    assert plans[PLAN_BOUND_PROOF]["valid"]
    assert plans[PLAN_BOUND_PROOF]["features"]["plan_is_bound_proof"] == 1.0
    assert plans[PLAN_TOP2_SEMIJOIN]["doc_ids"][:2] == [1, 10]
    assert plans[PLAN_TOP5_SEMIJOIN]["doc_ids"][:5] == [1, 2, 10, 11, 12]
    assert plans[PLAN_TOP5_SEMIJOIN]["features"][
        "plan_semantic_top2_retained_fraction"
    ] == 1.0
    assert plans[PLAN_TOP2_SEMIJOIN]["features"][
        "plan_top2_displacement_count"
    ] == 1.0
    assert len(plans[PLAN_HYBRID]["doc_ids"]) == len(
        set(plans[PLAN_HYBRID]["doc_ids"])
    )
    assert plans[PLAN_ONTOLOGY]["features"]["plan_is_ontology"] == 1.0
    assert plans[PLAN_HYBRID]["features"]["plan_is_hybrid"] == 1.0


def test_hierarchical_completion_freezes_stage1_top2():
    result = compose_hierarchical_topk(
        [10, 11, 1, 2, 3],
        [1, 2, 20, 21, 22],
        [1, 2, 3, 4, 5],
        top_k=5,
    )
    assert result == [10, 11, 20, 21, 22]
    assert result[:2] == [10, 11]


def test_materialization_prunes_unrouted_plan_families_before_enumeration():
    portfolio = materialize_complete_plan_portfolio(
        _record(),
        [1, 2, 3, 4, 5, 10, 11],
        [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
        allowed_plan_families={PLAN_ONTOLOGY_PHYSICAL},
    )
    assert set(portfolio["plans"]) == {
        PLAN_SEMANTIC,
        PLAN_ONTOLOGY_PHYSICAL,
    }
    assert portfolio["physical_plan_catalog"] == [
        PLAN_SEMANTIC,
        PLAN_ONTOLOGY_PHYSICAL,
    ]


def test_bounded_v282_routes_at_most_two_ontology_alternatives():
    routes = topology_partitioned_plan_families(
        TOPOLOGY_ROUTING_BOUNDED_V282
    )
    assert routes["ordered_multihop_chain"] == [
        PLAN_ONTOLOGY_PHYSICAL,
        PLAN_QUERY_LOCAL_TOP2_PAIR,
    ]
    assert routes["branch_comparison"] == [
        PLAN_ONTOLOGY_PHYSICAL,
        PLAN_LEARNED_RESIDUAL_TOP5,
    ]
    assert max(map(len, routes.values())) == 2


def test_v286_reuses_candidate_families_with_hierarchical_execution():
    assert topology_partitioned_plan_families(
        TOPOLOGY_ROUTING_HIERARCHICAL_V286
    ) == topology_partitioned_plan_families(TOPOLOGY_ROUTING_COMBINED_V285)


def test_v287_keeps_v286_candidate_aperture():
    assert topology_partitioned_plan_families(
        TOPOLOGY_ROUTING_OBJECTIVE_FAMILY_V287
    ) == topology_partitioned_plan_families(TOPOLOGY_ROUTING_HIERARCHICAL_V286)


def test_bounded_v283_routes_two_atomic_rank_objectives():
    routes = topology_partitioned_plan_families(
        TOPOLOGY_ROUTING_BOUNDED_V283
    )
    expected = [
        PLAN_QUERY_LOCAL_TOP2_PAIR,
        PLAN_ONTOLOGY_STRUCTURAL_TOP5,
    ]
    assert routes["ordered_multihop_chain"] == expected
    assert routes["branch_comparison"] == expected
    assert "unresolved" not in routes


def test_v284_routes_query_structure_to_warehouse_completion():
    routes = topology_partitioned_plan_families(
        TOPOLOGY_ROUTING_WAREHOUSE_V284
    )
    expected = [
        PLAN_QUERY_LOCAL_TOP2_PAIR,
        PLAN_ONTOLOGY_WAREHOUSE_TOP5,
    ]
    assert routes["ordered_multihop_chain"] == expected
    assert routes["branch_comparison"] == expected
    assert "unresolved" not in routes


def test_v285_combines_recall_safe_and_warehouse_plan_families():
    routes = topology_partitioned_plan_families(
        TOPOLOGY_ROUTING_COMBINED_V285
    )

    assert routes["ordered_multihop_chain"] == [
        PLAN_SIGNATURE_GROUNDED,
        PLAN_QUERY_LOCAL_TOP2_PAIR,
        PLAN_LEARNED_RESIDUAL_TOP5,
        PLAN_ONTOLOGY_WAREHOUSE_TOP5,
    ]
    assert routes["branch_comparison"] == [
        PLAN_QUERY_LOCAL_TOP2_PAIR,
        PLAN_CHAIN_COMPLETION,
        PLAN_QUERY_LOCAL_TOP5,
        PLAN_LEARNED_RESIDUAL_TOP5,
        PLAN_ONTOLOGY_WAREHOUSE_TOP5,
    ]
    assert "unresolved" not in routes


def test_v284_warehouse_materializes_diverse_complete_slates():
    portfolio = materialize_complete_plan_portfolio(
        _record(),
        [1, 2, 3, 4, 5, 6, 7],
        [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
        allowed_plan_families={PLAN_ONTOLOGY_WAREHOUSE_TOP5},
    )
    instances = [
        value for value in portfolio["plans"].values()
        if value["plan_family"] == PLAN_ONTOLOGY_WAREHOUSE_TOP5
    ]
    assert instances
    assert all(value["doc_ids"][:2] == [1, 2] for value in instances)
    assert {
        value["plan_metadata"]["warehouse_replacement_rank"]
        for value in instances
    } == {3, 4, 5}
    assert all(
        len(set(value["doc_ids"][:5]) - {1, 2, 3, 4, 5}) == 1
        for value in instances
    )
    assert all(value["valid"] for value in instances)
    assert all(
        value["features"]["plan_is_ontology_warehouse_rank5_completion"]
        == 1.0
        for value in instances
    )
    assert portfolio["ontology_warehouse_top5_candidate_count"] > 0


def test_v284_warehouse_composes_top2_stage_before_top5_completion():
    record = _record()
    proof_slate = record["full_aperture_proof_slate"]
    anchor_candidate = dict(proof_slate["candidates"][0])
    anchor_candidate["passage_idx"] = 1
    proof_slate["candidates"].append(anchor_candidate)
    proof_slate["local_proof_paths"][0]["passage_ids"] = [1, 11]
    bound_path = proof_slate["bound_proof_paths"][0]
    bound_path["passage_ids"] = [1, 11]
    bound_path["query_local_compiler_validated"] = True
    bound_path["strict_chain_certificate"] = True
    portfolio = materialize_complete_plan_portfolio(
        record,
        [1, 2, 3, 4, 5, 6, 7],
        [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
        allowed_plan_families={PLAN_ONTOLOGY_WAREHOUSE_TOP5},
    )
    instances = [
        value for value in portfolio["plans"].values()
        if value["plan_family"] == PLAN_ONTOLOGY_WAREHOUSE_TOP5
    ]
    assert instances
    assert all(value["doc_ids"][:2] == [1, 11] for value in instances)
    assert all(
        value["plan_metadata"]["warehouse_stage1_top2_pair"]
        for value in instances
    )


def test_structural_rank5_materializes_one_incremental_ontology_action():
    portfolio = materialize_complete_plan_portfolio(
        _record(),
        [1, 2, 3, 4, 5, 6, 7],
        [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
        allowed_plan_families={PLAN_ONTOLOGY_STRUCTURAL_TOP5},
    )
    assert set(portfolio["plans"]) == {
        PLAN_SEMANTIC,
        PLAN_ONTOLOGY_STRUCTURAL_TOP5,
    }
    plan = portfolio["plans"][PLAN_ONTOLOGY_STRUCTURAL_TOP5]
    assert plan["valid"]
    assert plan["doc_ids"] == [1, 2, 3, 4, 11]
    assert plan["features"][
        "plan_is_ontology_structural_rank5_completion"
    ] == 1.0
    assert plan["features"]["plan_top2_membership_invariant"] == 1.0
    assert plan["features"]["plan_top5_displacement_count"] == 1.0
    assert portfolio["ontology_structural_rank5_candidate_available"]


def test_residual_top5_plan_preserves_top4_and_exposes_candidate_features():
    portfolio = materialize_complete_plan_portfolio(
        _record(),
        [1, 2, 3, 4, 5, 6, 7],
        [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
    )
    instances = [
        value for value in portfolio["plans"].values()
        if value.get("plan_family") == PLAN_LEARNED_RESIDUAL_TOP5
    ]
    assert instances
    first = instances[0]
    assert first["valid"]
    assert first["doc_ids"][:4] == [1, 2, 3, 4]
    assert first["doc_ids"][4] != 5
    assert first["features"]["plan_top2_membership_invariant"] == 1.0
    assert first["features"][
        "plan_is_learned_residual_top5_completion"
    ] == 1.0
    assert first["features"]["residual_candidate_pool_rank_log"] > 0.0


def test_shadow_and_missing_model_preserve_upstream_ranking():
    record = _record()
    current_ids = [1, 10, 2, 11, 3, 4, 5]
    current_scores = [1.0, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6]
    optimizer = CompleteEvidencePlanPortfolioOptimizer("")
    ids, scores, log = optimizer.optimize(
        current_ids, current_scores, record, shadow_only=True
    )
    assert ids == current_ids
    assert log["selected_plan"] == "upstream_shadow"
    assert not log["atomic_execution"]["executed"]

    ids, scores, log = optimizer.optimize(
        current_ids, current_scores, record, shadow_only=False
    )
    assert ids == current_ids
    assert log["selected_plan"] == "upstream_model_unavailable"


def test_certified_model_selects_a_complete_plan_atomically(tmp_path):
    plan_models = {}
    for plan in ALTERNATIVE_PLANS:
        plan_models[plan] = {
            "delta_r2_head": _constant_head(0.2),
            "delta_r5_head": _constant_head(0.1),
            "loss_probability_head": _constant_head(0.0),
            "delta_r2_residual_lcb": 0.0,
            "delta_r5_residual_lcb": 0.0,
        }
    model = {
        "type": MODEL_TYPE,
        "feature_names": list(FEATURE_NAMES),
        "plans": plan_models,
        "policy": {
            "max_loss_probability": 0.05,
            "min_utility_lcb": 0.0,
            "min_delta_r2_lcb": 0.0,
            "min_delta_r5_lcb": 0.0,
        },
        "holdout_certificate": {"passed": True},
    }
    path = tmp_path / "model.json"
    path.write_text(json.dumps(model), encoding="utf-8")
    optimizer = CompleteEvidencePlanPortfolioOptimizer(str(path))
    record = _record()
    current_ids = [1, 2, 3, 4, 5, 10, 11]
    ids, _, log = optimizer.optimize(
        current_ids, [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4], record
    )
    assert log["selected_plan"] in ALTERNATIVE_PLANS
    assert log["atomic_execution"]["executed"]
    assert ids[:5] == log["plans"][log["selected_plan_instance"]]["doc_ids"]
    assert log["selection_contract"] == "whole_topk_physical_plan"
    assert not log["upstream_ranker_preserved"]


def test_v286_executes_top2_then_residual_top5_without_prefix_loss(tmp_path):
    def expert(delta_r2, delta_r5):
        return {
            "delta_r2_head": _constant_head(delta_r2),
            "delta_r5_head": _constant_head(delta_r5),
            "gain_probability_head": _constant_head(0.9),
            "loss_probability_head": _constant_head(0.0),
            "delta_r2_residual_lcb": 0.0,
            "delta_r5_residual_lcb": 0.0,
        }

    class_policy = {
        "selection_mode": "pareto_lcb",
        "calibration_feasible": True,
        "max_loss_probability": 0.05,
        "min_utility_lcb": 0.0,
        "min_delta_r2_lcb": 0.0,
        "min_delta_r5_lcb": 0.0,
        "min_gain_probability": 0.0,
    }
    model = {
        "type": MODEL_TYPE,
        "feature_names": list(FEATURE_NAMES),
        "plans": {
            OBJECTIVE_CLASS_EXPERT_PREFIX + "top2_ordering": expert(0.2, 0.1),
            OBJECTIVE_CLASS_EXPERT_PREFIX + "top5_completion": expert(0.0, 0.2),
        },
        "policy": {
            "allowed_plans": [PLAN_TOP2_SEMIJOIN, PLAN_TOP5_SEMIJOIN],
            "hierarchical_objective_execution": True,
            "objective_class_policies": {
                "top2_ordering": dict(class_policy),
                "top5_completion": dict(class_policy),
            },
        },
        "holdout_certificate": {"passed": True},
    }
    path = tmp_path / "hierarchical_model.json"
    path.write_text(json.dumps(model), encoding="utf-8")
    optimizer = CompleteEvidencePlanPortfolioOptimizer(str(path))

    ids, _, log = optimizer.optimize(
        [1, 2, 3, 4, 5, 10, 11],
        [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
        _record(),
    )

    assert ids[:2] == [1, 10]
    assert ids[:5] == [1, 10, 11, 12, 2]
    assert log["selection_contract"] == "hierarchical_top2_then_top5_completion"
    assert log["hierarchical_execution"]["stage1_plan"] == PLAN_TOP2_SEMIJOIN
    assert log["hierarchical_execution"]["stage2_plan"] == PLAN_TOP5_SEMIJOIN
    assert log["hierarchical_execution"]["top2_frozen_during_stage2"]


def test_v287_executes_family_gated_hierarchical_slate(tmp_path):
    def expert(delta_r2, delta_r5):
        return {
            "delta_r2_head": _constant_head(delta_r2),
            "delta_r5_head": _constant_head(delta_r5),
            "gain_probability_head": _constant_head(0.9),
            "loss_probability_head": _constant_head(0.0),
            "delta_r2_residual_lcb": 0.0,
            "delta_r5_residual_lcb": 0.0,
        }

    def family_policy(plan_name):
        return {
            "selection_mode": "pareto_lcb",
            "calibration_feasible": True,
            "plan_family_policies": {
                plan_name: {
                    "selection_mode": "pareto_lcb",
                    "calibration_feasible": True,
                    "max_loss_probability": 0.05,
                    "min_utility_lcb": 0.0,
                    "min_delta_r2_lcb": 0.0,
                    "min_delta_r5_lcb": 0.0,
                    "min_gain_probability": 0.0,
                }
            },
        }

    model = {
        "type": MODEL_TYPE,
        "feature_names": list(FEATURE_NAMES),
        "plans": {
            objective_family_model_key(
                "top2_ordering", PLAN_TOP2_SEMIJOIN
            ): expert(0.2, 0.1),
            objective_family_model_key(
                "top5_completion", PLAN_TOP5_SEMIJOIN
            ): expert(0.0, 0.2),
        },
        "policy": {
            "allowed_plans": [PLAN_TOP2_SEMIJOIN, PLAN_TOP5_SEMIJOIN],
            "hierarchical_objective_execution": True,
            "objective_class_policies": {
                "top2_ordering": family_policy(PLAN_TOP2_SEMIJOIN),
                "top5_completion": family_policy(PLAN_TOP5_SEMIJOIN),
            },
        },
        "holdout_certificate": {"passed": True},
    }
    path = tmp_path / "objective_family_model.json"
    path.write_text(json.dumps(model), encoding="utf-8")
    optimizer = CompleteEvidencePlanPortfolioOptimizer(str(path))

    ids, _, log = optimizer.optimize(
        [1, 2, 3, 4, 5, 10, 11],
        [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
        _record(),
    )

    assert ids[:5] == [1, 10, 11, 12, 2]
    assert log["hierarchical_execution"]["stage1_plan"] == PLAN_TOP2_SEMIJOIN
    assert log["hierarchical_execution"]["stage2_plan"] == PLAN_TOP5_SEMIJOIN


def test_model_policy_can_limit_runtime_to_bound_join(tmp_path):
    model = {
        "type": MODEL_TYPE,
        "feature_names": list(FEATURE_NAMES),
        "plans": {
            PLAN_BOUND_PROOF: {
                "delta_r2_head": _constant_head(0.2),
                "delta_r5_head": _constant_head(0.1),
                "loss_probability_head": _constant_head(0.0),
                "delta_r2_residual_lcb": 0.0,
                "delta_r5_residual_lcb": 0.0,
            },
            PLAN_ONTOLOGY: {
                "delta_r2_head": _constant_head(1.0),
                "delta_r5_head": _constant_head(1.0),
                "loss_probability_head": _constant_head(0.0),
                "delta_r2_residual_lcb": 0.0,
                "delta_r5_residual_lcb": 0.0,
            },
        },
        "policy": {
            "allowed_plans": [PLAN_BOUND_PROOF],
            "max_loss_probability": 0.05,
            "min_utility_lcb": 0.0,
            "min_delta_r2_lcb": 0.0,
            "min_delta_r5_lcb": 0.0,
        },
        "holdout_certificate": {"passed": True},
    }
    path = tmp_path / "bound_only.json"
    path.write_text(json.dumps(model), encoding="utf-8")
    optimizer = CompleteEvidencePlanPortfolioOptimizer(str(path))
    ids, _, log = optimizer.optimize(
        [1, 2, 3, 4, 5, 10, 11],
        [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
        _record(),
    )

    assert log["selected_plan"] == PLAN_BOUND_PROOF
    assert log["allowed_plans"] == [PLAN_BOUND_PROOF]
    assert ids[:2] == [10, 11]


def test_plan_level_trainer_calibrates_without_dataset_routing():
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "plan_optimizer_core.py"
    )
    spec = importlib.util.spec_from_file_location("v250_trainer", script_path)
    trainer = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(trainer)

    rows = []
    for query_idx in range(24):
        for plan in ALTERNATIVE_PLANS:
            features = {name: 0.0 for name in FEATURE_NAMES}
            features["strict_executable_path_count_log"] = 1.0
            features["complete_proof_candidate_count_log"] = 1.0
            features["plan_is_ontology"] = float(plan == PLAN_ONTOLOGY)
            features["plan_is_hybrid"] = float(plan == PLAN_HYBRID)
            rows.append({
                "query_key": f"q{query_idx}",
                "dataset": "not_used_for_routing",
                "plan_name": plan,
                "features": features,
                "delta_r2": 0.5,
                "delta_r5": 0.5,
                "loss_probability": 0.0,
            })
    args = SimpleNamespace(
        trees=16,
        max_depth=3,
        min_leaf=2,
        max_features=0.8,
        seed=250,
        jobs=1,
        alpha=0.10,
        min_calibration_selected=4,
        max_calibration_loss_rate=0.08,
        selection_mode="pareto_lcb",
        min_delta_r2_mean=0.0,
        min_delta_r5_mean=0.0,
        min_characteristic_selected=3,
        max_characteristic_loss_rate=0.15,
        robust_calibration_objective=False,
    )
    payloads, models = trainer.train_models(rows, rows, args)
    policy, summary = trainer.calibrate_policy(rows, payloads, models, args)
    assert policy["calibration_feasible"]
    assert summary["selected_count"] == 24
    assert summary["delta_r2_sum"] > 0.0


def test_trace_loader_filters_disallowed_plan_families_before_training(tmp_path):
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "plan_optimizer_core.py"
    )
    spec = importlib.util.spec_from_file_location("v250_filtered_trainer", script_path)
    trainer = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(trainer)

    semantic_ids = [1, 2, 3, 4, 5]
    signature_ids = [10, 11, 3, 4, 5]
    noisy_ids = [20, 21, 3, 4, 5]
    record = {
        "query_idx": 0,
        "final_ranking_incumbent": {"doc_ids": semantic_ids},
        "complete_evidence_plan_optimizer": {
            "endpoint_contract_matched": True,
            "plans": {
                PLAN_SIGNATURE_GROUNDED: {
                    "plan_family": PLAN_SIGNATURE_GROUNDED,
                    "valid": True,
                    "doc_ids": signature_ids,
                    "features": {},
                },
                PLAN_RANK_PARTITIONED_TOP2: {
                    "plan_family": PLAN_RANK_PARTITIONED_TOP2,
                    "valid": True,
                    "doc_ids": noisy_ids,
                    "features": {},
                },
            },
        },
        "complete_evidence_plan_training": {
            "labels_valid": True,
            "label_semantics": "official_exact_document_recall",
            "baseline_endpoint": "final_ranking_incumbent",
            "plans": {
                PLAN_SEMANTIC: {
                    "doc_ids": semantic_ids,
                    "recall_at2": 0.0,
                    "recall_at5": 0.0,
                },
                PLAN_SIGNATURE_GROUNDED: {
                    "doc_ids": signature_ids,
                    "recall_at2": 1.0,
                    "recall_at5": 1.0,
                },
                PLAN_RANK_PARTITIONED_TOP2: {
                    "doc_ids": noisy_ids,
                    "recall_at2": 0.0,
                    "recall_at5": 0.0,
                },
            },
        },
    }
    trace_path = tmp_path / "2wikimultihopqa.jsonl"
    trace_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    rows = trainer.rows_from_traces(
        str(tmp_path),
        ["2wikimultihopqa"],
        allowed_plan_names={PLAN_SIGNATURE_GROUNDED},
    )

    assert len(rows) == 1
    assert rows[0]["plan_name"] == PLAN_SIGNATURE_GROUNDED


def test_topology_partition_routes_plan_families_without_dataset_identity():
    routing = topology_partitioned_plan_families()
    ordered = {
        "signature_topology_ordered_chain": 1.0,
        "signature_topology_branch_comparison": 0.0,
    }
    branch = {
        "signature_topology_ordered_chain": 0.0,
        "signature_topology_branch_comparison": 1.0,
    }
    unresolved = {
        "signature_topology_ordered_chain": 0.0,
        "signature_topology_branch_comparison": 0.0,
    }

    assert query_plan_topology(ordered) == "ordered_multihop_chain"
    assert plan_allowed_for_topology(PLAN_SIGNATURE_GROUNDED, ordered, routing)
    assert plan_allowed_for_topology(PLAN_QUERY_LOCAL_TOP2_PAIR, ordered, routing)
    assert not plan_allowed_for_topology(PLAN_QUERY_LOCAL_TOP5, ordered, routing)
    assert query_plan_topology(branch) == "branch_comparison"
    assert plan_allowed_for_topology(PLAN_QUERY_LOCAL_TOP5, branch, routing)
    assert plan_allowed_for_topology(PLAN_CHAIN_COMPLETION, branch, routing)
    assert not plan_allowed_for_topology(PLAN_SIGNATURE_GROUNDED, branch, routing)
    assert query_plan_topology(unresolved) == "unresolved"
    assert not plan_allowed_for_topology(PLAN_CHAIN_COMPLETION, unresolved, routing)


def test_recall_safe_topology_profile_restores_rank_compatible_candidates():
    routing = topology_partitioned_plan_families("recall_safe_v281")
    ordered = {"signature_topology_ordered_chain": 1.0}
    branch = {"signature_topology_branch_comparison": 1.0}

    assert plan_allowed_for_topology(PLAN_QUERY_LOCAL_TOP2_PAIR, branch, routing)
    assert plan_allowed_for_topology(PLAN_LEARNED_RESIDUAL_TOP5, branch, routing)
    assert plan_allowed_for_topology(PLAN_LEARNED_RESIDUAL_TOP5, ordered, routing)
    assert not plan_allowed_for_topology(PLAN_QUERY_LOCAL_TOP5, ordered, routing)


def test_v290_admission_score_search_is_independent_of_empirical_risk_budget():
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "plan_optimizer_core.py"
    )
    spec = importlib.util.spec_from_file_location("v290_trainer", script_path)
    trainer = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(trainer)

    legacy_args = SimpleNamespace(max_calibration_loss_rate=0.08)
    assert trainer.admission_loss_probability_caps(legacy_args)[-1] == 0.08

    decoupled_args = SimpleNamespace(
        max_calibration_loss_rate=0.08,
        max_admission_loss_probability=0.50,
    )
    caps = trainer.admission_loss_probability_caps(decoupled_args)
    assert caps[-1] == 0.50
    assert 0.08 in caps
