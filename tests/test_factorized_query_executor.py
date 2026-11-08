from collections import defaultdict
from types import SimpleNamespace

import numpy as np

from compassrag.system import CompassRAG
from compassrag.query_plan_factorized_executor import QueryPlanFactorizedExecutor
from compassrag.query_plan_delta_admission import (
    FEATURE_NAMES,
    HIERARCHICAL_GATE_FEATURE_NAMES,
    V185_FEATURE_NAMES,
    FactorizedDeltaAdmissionModel,
    factorized_action_features_v185,
)
from compassrag.query_plan_action_replay import materialize_counterfactual_ranking


def _dense_empty(_query):
    return [], []


def _openie_empty(_passage_idx):
    return {}


def _plan(anchors, relations, answer_type="entity"):
    return {
        "enabled": True,
        "anchor_entities": anchors,
        "query_condition": {
            "answer_type": answer_type,
            "answer_hints": [answer_type],
            "relation_terms": relations,
            "relation_signatures": relations,
        },
        "matched_plans": [{
            "id": "test-plan",
            "path": [{"relation": relation} for relation in relations],
        }],
        "sample_paths": [],
    }


def test_entity_relation_index_scan_finds_typed_terminal():
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
        enable_cascaded_bind_join=False,
        min_candidate_utility=0.60,
    )
    result = executor.run(
        query="What is the capital of France?",
        dataset="popqa",
        base_doc_ids=list(range(20)),
        query_plan_log=_plan(["France"], ["capital"], "location"),
        dense_retrieve=_dense_empty,
        get_openie_record=_openie_empty,
        lookup_entity_relations=lambda entity, limit: rows.get(entity.lower(), [])[:limit],
    )

    assert result.log["selected_operator"] == "entity_relation_index_scan"
    assert result.candidates[0]["passage_idx"] == 9
    assert result.candidates[0]["terminal_entity"] == "Paris"
    assert result.candidates[0]["utility"] >= 0.60


def test_inverse_direction_access_recovers_exact_key_openie_orientation():
    rows = [{
        "passage_idx": 9,
        "subject": "Paris",
        "predicate": "capital of",
        "object": "France",
        "entity_match_score": 1.0,
    }]
    executor = QueryPlanFactorizedExecutor(
        enable_inverse_direction_fallback=True,
        inverse_direction_penalty=0.12,
        min_relation_score=0.30,
    )
    executor._active_relation_directions = {"capital": "forward"}

    selected = executor._indexed_access(
        binding="France",
        relation="capital",
        answer_type="location",
        terminal_step=False,
        lookup_entity_relations=lambda _entity, limit: rows[:limit],
    )

    assert selected[0]["terminal_entity"] == "Paris"
    assert selected[0]["expected_direction"] == "forward"
    assert selected[0]["access_direction"] == "reverse"
    assert selected[0]["inverse_direction_fallback"] is True


def test_inverse_direction_access_remains_disabled_by_default():
    rows = [{
        "passage_idx": 9,
        "subject": "Paris",
        "predicate": "capital of",
        "object": "France",
        "entity_match_score": 1.0,
    }]
    executor = QueryPlanFactorizedExecutor(min_relation_score=0.30)
    executor._active_relation_directions = {"capital": "forward"}

    selected = executor._indexed_access(
        binding="France",
        relation="capital",
        answer_type="location",
        terminal_step=False,
        lookup_entity_relations=lambda _entity, limit: rows[:limit],
    )

    assert selected == []


def test_cascaded_bind_join_preserves_join_bindings():
    rows = {
        "alice": [{
            "passage_idx": 10,
            "subject": "Alice",
            "predicate": "mother",
            "object": "Beth",
            "entity_match_score": 1.0,
        }],
        "beth": [{
            "passage_idx": 22,
            "subject": "Beth",
            "predicate": "born in",
            "object": "Paris",
            "entity_match_score": 1.0,
        }],
    }
    executor = QueryPlanFactorizedExecutor(min_candidate_utility=0.60)
    result = executor.run(
        query="Where was Alice's mother born?",
        dataset="musique",
        base_doc_ids=list(range(30)),
        query_plan_log=_plan(["Alice"], ["mother", "born in"], "location"),
        dense_retrieve=_dense_empty,
        get_openie_record=_openie_empty,
        lookup_entity_relations=lambda entity, limit: rows.get(entity.lower(), [])[:limit],
    )

    assert result.log["selected_operator"] == "cascaded_bind_join"
    assert result.groups
    assert result.groups[0]["passage_indices"] == [10, 22]
    assert result.groups[0]["bindings"] == ["alice", "Beth", "Paris"]
    assert result.candidates[0]["passage_idx"] == 22


def test_dual_branch_comparison_join_requires_both_branches():
    rows = {
        "alpha": [{
            "passage_idx": 31,
            "subject": "Alpha",
            "predicate": "founded",
            "object": "1900",
            "entity_match_score": 1.0,
        }],
        "beta": [{
            "passage_idx": 42,
            "subject": "Beta",
            "predicate": "founded",
            "object": "1910",
            "entity_match_score": 1.0,
        }],
    }
    executor = QueryPlanFactorizedExecutor(min_candidate_utility=0.60)
    result = executor.run(
        query="Which was founded earlier, Alpha or Beta?",
        dataset="2wikimultihopqa",
        base_doc_ids=list(range(50)),
        query_plan_log=_plan(["Alpha", "Beta"], ["founded"], "date"),
        dense_retrieve=_dense_empty,
        get_openie_record=_openie_empty,
        lookup_entity_relations=lambda entity, limit: rows.get(entity.lower(), [])[:limit],
    )

    assert result.log["selected_operator"] == "dual_branch_comparison_join"
    assert result.groups[0]["passage_indices"] == [31, 42]
    assert len(result.candidates) == 2


def test_each_physical_operator_is_independently_ablatable():
    executor = QueryPlanFactorizedExecutor(
        enable_entity_relation_index_scan=False,
        enable_cascaded_bind_join=False,
        enable_dual_branch_comparison_join=False,
    )
    result = executor.run(
        query="What is the capital of France?",
        dataset="popqa",
        base_doc_ids=list(range(10)),
        query_plan_log=_plan(["France"], ["capital"], "location"),
        dense_retrieve=_dense_empty,
        get_openie_record=_openie_empty,
        lookup_entity_relations=lambda _entity, _limit: [],
    )

    assert result.log["skip_reason"] == "physical_operator_disabled"
    assert not result.candidates


def test_compassrag_admission_promotes_high_utility_simple_scan_to_top2():
    rag = object.__new__(CompassRAG)
    rag.global_config = SimpleNamespace(
        dataset="popqa",
        query_plan_ontology_enable_factorized_query_executor=True,
        query_plan_ontology_factorized_target_datasets="popqa",
        query_plan_ontology_factorized_enable_entity_relation_index_scan=True,
        query_plan_ontology_factorized_enable_cascaded_bind_join=True,
        query_plan_ontology_factorized_enable_dual_branch_comparison_join=True,
        query_plan_ontology_factorized_min_candidate_utility=0.60,
        query_plan_ontology_factorized_min_top2_utility=0.60,
        query_plan_ontology_factorized_min_top5_utility=0.60,
        query_plan_ontology_factorized_enable_top2_admission=True,
        query_plan_ontology_factorized_enable_top5_admission=True,
    )
    rag.passage_node_keys = [str(idx) for idx in range(10)]
    rag.openie_record_by_passage_key = {
        "8": {
            "extracted_triples": [["France", "capital", "Paris"]],
        },
    }
    rag.dense_passage_retrieval = _dense_empty
    rag.query_plan_ontology_stats = defaultdict(int)

    doc_ids, _, log = rag.apply_query_plan_factorized_query_executor(
        query="What is the capital of France?",
        sorted_doc_ids=np.arange(10, dtype=np.intp),
        sorted_doc_scores=np.linspace(1.0, 0.1, 10, dtype=np.float32),
        query_plan_log=_plan(["France"], ["capital"], "location"),
    )

    assert list(doc_ids[:2]) == [0, 8]
    assert log["route_reason"] == "simple_entity_relation_top2"
    assert log["changed_top2"] is True


def test_typed_compiler_rejects_birth_certificate_false_event():
    rows = {
        "united states": [{
            "passage_idx": 8,
            "subject": "United States",
            "predicate": "legalized birth control for",
            "object": "married couples in 1965",
            "entity_match_score": 1.0,
        }],
    }
    executor = QueryPlanFactorizedExecutor(
        enable_typed_semantic_compiler=True,
        require_exact_entity_keys=True,
        min_relation_score=0.60,
        min_candidate_utility=0.60,
    )
    result = executor.run(
        query="When were birth certificates first issued in the United States?",
        dataset="nq_rear",
        base_doc_ids=list(range(10)),
        query_plan_log=_plan(
            ["birth certificates", "United States"],
            ["birth"],
            "date",
        ),
        dense_retrieve=_dense_empty,
        get_openie_record=_openie_empty,
        lookup_entity_relations=lambda entity, limit: rows.get(entity.lower(), [])[:limit],
    )

    contract = result.log["typed_contract"]
    assert contract["non_event_birth_context"] is True
    assert contract["relations"] == ["issued"]
    assert not result.candidates


def test_typed_compiler_prefers_full_entity_key_over_surname():
    rows = {
        "c n annadurai": [{
            "passage_idx": 4,
            "subject": "C. N. Annadurai",
            "predicate": "was born in",
            "object": "Kanchipuram",
            "entity_match_score": 1.0,
        }],
        "annadurai": [{
            "passage_idx": 7,
            "subject": "Rani Annadurai",
            "predicate": "was born in",
            "object": "Thirumullaivoyal",
            "entity_match_score": 1.0,
        }],
    }
    executor = QueryPlanFactorizedExecutor(
        enable_typed_semantic_compiler=True,
        require_exact_entity_keys=True,
        min_relation_score=0.60,
        min_candidate_utility=0.60,
    )
    result = executor.run(
        query="In what city was C. N. Annadurai born?",
        dataset="popqa",
        base_doc_ids=list(range(10)),
        query_plan_log=_plan(
            ["Annadurai", "C. N. Annadurai"],
            ["birth_place"],
            "location",
        ),
        dense_retrieve=_dense_empty,
        get_openie_record=_openie_empty,
        lookup_entity_relations=lambda entity, limit: rows.get(entity.lower(), [])[:limit],
    )

    assert result.log["anchors"] == ["c n annadurai"]
    assert [item["passage_idx"] for item in result.candidates] == [4]


def test_typed_join_key_rejects_year_literal_bridge():
    rows = {
        "charles stewart": [{
            "passage_idx": 3,
            "subject": "Charles Stewart",
            "predicate": "was born in",
            "object": "1639",
            "entity_match_score": 1.0,
        }],
        "1639": [{
            "passage_idx": 8,
            "subject": "Henry Wenceslaus",
            "predicate": "served as governor until",
            "object": "1639",
            "entity_match_score": 1.0,
        }],
    }
    executor = QueryPlanFactorizedExecutor(
        require_typed_join_keys=True,
        min_candidate_utility=0.60,
    )
    result = executor.run(
        query="Where did Charles Stewart's father graduate from?",
        dataset="2wikimultihopqa",
        base_doc_ids=list(range(10)),
        query_plan_log=_plan(
            ["Charles Stewart"],
            ["was born in", "served as governor until"],
            "location",
        ),
        dense_retrieve=_dense_empty,
        get_openie_record=_openie_empty,
        lookup_entity_relations=lambda entity, limit: rows.get(entity.lower(), [])[:limit],
    )

    assert not result.candidates
    assert result.log["runtime_rejections"]["invalid_intermediate_join_key"] >= 1


def test_typed_admission_keeps_simple_scan_out_of_top2():
    rag = object.__new__(CompassRAG)
    rag.global_config = SimpleNamespace(
        dataset="popqa",
        query_plan_ontology_enable_factorized_query_executor=True,
        query_plan_ontology_factorized_target_datasets="popqa",
        query_plan_ontology_factorized_enable_entity_relation_index_scan=True,
        query_plan_ontology_factorized_enable_cascaded_bind_join=True,
        query_plan_ontology_factorized_enable_dual_branch_comparison_join=True,
        query_plan_ontology_factorized_enable_typed_semantic_compiler=True,
        query_plan_ontology_factorized_require_exact_entity_keys=True,
        query_plan_ontology_factorized_require_typed_join_keys=True,
        query_plan_ontology_factorized_min_query_predicate_coverage=0.75,
        query_plan_ontology_factorized_min_relation_score=0.60,
        query_plan_ontology_factorized_min_candidate_utility=0.60,
        query_plan_ontology_factorized_min_top2_utility=0.60,
        query_plan_ontology_factorized_min_top5_utility=0.60,
        query_plan_ontology_factorized_enable_top2_admission=True,
        query_plan_ontology_factorized_enable_top5_admission=True,
        query_plan_ontology_factorized_enable_simple_top2_admission=False,
        query_plan_ontology_factorized_require_chain_source_top2=True,
        query_plan_ontology_factorized_min_counterfactual_net_benefit=0.60,
    )
    rag.passage_node_keys = [str(idx) for idx in range(10)]
    rag.openie_record_by_passage_key = {
        "8": {"extracted_triples": [["France", "capital", "Paris"]]},
    }
    rag.dense_passage_retrieval = _dense_empty
    rag.query_plan_ontology_stats = defaultdict(int)

    doc_ids, _, log = rag.apply_query_plan_factorized_query_executor(
        query="What is the capital of France?",
        sorted_doc_ids=np.arange(10, dtype=np.intp),
        sorted_doc_scores=np.linspace(1.0, 0.1, 10, dtype=np.float32),
        query_plan_log=_plan(["France"], ["capital"], "location"),
    )

    assert list(doc_ids[:2]) == [0, 1]
    assert list(doc_ids[:5])[-1] == 8
    assert log["route_reason"] == "simple_entity_relation_top5"


def test_branch_complete_compiler_keeps_qualifiers_out_of_branch_keys():
    cases = [
        (
            "Are Wiqu (Lima) and Qillqa (Melgar) both located in the same country?",
            ["wiqu", "lima", "qillqa", "melgar"],
            [("wiqu", ["lima"]), ("qillqa", ["melgar"])],
        ),
        (
            "Which album was released first, Don't Freak Me Out or Love & War (Barlowgirl Album)?",
            ["don t freak me out", "love war", "barlowgirl"],
            [("don t freak me out", []), ("love war", ["barlowgirl"])],
        ),
        (
            "Are both Selo Sanatoriya Imeni Chekhova and Volovo, Lipetsk Oblast located in the same country?",
            ["selo sanatoriya imeni chekhova", "lipetsk oblast", "volovo"],
            [
                ("selo sanatoriya imeni chekhova", []),
                ("volovo", ["lipetsk oblast"]),
            ],
        ),
    ]
    for query, anchors, expected in cases:
        contract = QueryPlanFactorizedExecutor._compile_comparison_branches(query, anchors)
        assert contract["complete"] is True
        assert [
            (branch["entity_key"], branch["qualifiers"])
            for branch in contract["branches"]
        ] == expected


def test_v227_query_surface_compiler_drops_corpus_anchors_and_binds_two_branches():
    executor = QueryPlanFactorizedExecutor(
        enable_typed_semantic_compiler=True,
        enable_executable_ontology_compiler=True,
        min_candidate_utility=0.0,
    )
    result = executor.run(
        query="Are Nahran, East Azerbaijan and Ab Sefid-E Saqaveh both located in the same country?",
        dataset="2wikimultihopqa",
        base_doc_ids=list(range(10)),
        query_plan_log=_plan(
            [
                "Nahran",
                "East Azerbaijan",
                "Ab Sefid-E Saqaveh",
                "10 km east from Tiran",
                "44 km east of Kermanshah",
            ],
            ["located in"],
            "boolean",
        ),
        dense_retrieve=_dense_empty,
        get_openie_record=_openie_empty,
        lookup_entity_relations=lambda _entity, _limit: [],
    )

    binding = result.log["typed_contract"]["query_surface_slot_binding"]
    assert binding["executable_anchors"] == ["nahran", "ab sefid e saqaveh"]
    assert binding["comparison_branches"]["branches"][0]["qualifiers"] == ["east azerbaijan"]
    assert binding["dropped_non_query_anchors"] == [
        "10 km east from tiran",
        "44 km east of kermanshah",
    ]
    assert result.log["typed_contract"]["anchor_grounding"] == 1.0
    assert result.log["skip_reason"] == "no_validated_candidates"


def test_v227_relation_realization_matches_observed_physical_predicate():
    realization_map = {
        "plans": {
            "test-plan": {
                "steps": {
                    "0": {
                        "expected_relation_signature": "country of citizenship",
                        "expected_relation_family": "nationality",
                        "expected_relation_terms": ["country", "citizenship"],
                        "realizations": [{
                            "normalized_relation": "holds nationality",
                            "count": 50,
                            "score": 4.0,
                        }],
                    },
                },
            },
        },
    }
    rows = {
        "alice": [{
            "passage_idx": 8,
            "subject": "Alice",
            "predicate": "holds nationality",
            "object": "France",
            "entity_match_score": 1.0,
        }],
    }
    executor = QueryPlanFactorizedExecutor(
        enable_typed_semantic_compiler=True,
        enable_executable_ontology_compiler=True,
        enable_relation_realization=True,
        relation_realization_map=realization_map,
        min_relation_score=0.90,
        min_type_score=0.0,
        min_candidate_utility=0.50,
    )
    result = executor.run(
        query="What country is Alice associated with?",
        dataset="popqa",
        base_doc_ids=list(range(10)),
        query_plan_log=_plan(["Alice"], ["country of citizenship"], "location"),
        dense_retrieve=_dense_empty,
        get_openie_record=_openie_empty,
        lookup_entity_relations=lambda entity, limit: rows.get(entity.lower(), [])[:limit],
    )

    assert result.candidates[0]["passage_idx"] == 8
    assert result.candidates[0]["matched_relation_surface"] == "holds nationality"
    assert result.log["relation_realization"]["realized_relation_count"] == 1


def test_v227_whole_warehouse_probe_resolves_expanded_entity_key():
    rag = object.__new__(CompassRAG)
    rag.global_config = SimpleNamespace(
        dataset="popqa",
        query_plan_ontology_enable_factorized_query_executor=True,
        query_plan_ontology_factorized_target_datasets="popqa",
        query_plan_ontology_factorized_enable_entity_relation_index_scan=True,
        query_plan_ontology_factorized_enable_cascaded_bind_join=True,
        query_plan_ontology_factorized_enable_dual_branch_comparison_join=True,
        query_plan_ontology_factorized_enable_typed_semantic_compiler=True,
        query_plan_ontology_factorized_enable_executable_ontology_compiler=True,
        query_plan_ontology_factorized_enable_relation_realization=False,
        query_plan_ontology_factorized_enable_whole_warehouse_probe=True,
        query_plan_ontology_factorized_warehouse_max_entity_keys=8,
        query_plan_ontology_factorized_require_exact_entity_keys=True,
        query_plan_ontology_factorized_min_relation_score=0.60,
        query_plan_ontology_factorized_min_type_score=0.0,
        query_plan_ontology_factorized_min_candidate_utility=0.50,
        query_plan_ontology_factorized_min_top2_utility=2.0,
        query_plan_ontology_factorized_min_top5_utility=2.0,
    )
    rag.passage_node_keys = [str(idx) for idx in range(10)]
    rag.openie_record_by_passage_key = {
        "8": {
            "extracted_triples": [["Nahran, East Azerbaijan", "is located in", "Iran"]],
        },
    }
    rag.query_plan_relation_realization_map = {}
    rag.dense_passage_retrieval = _dense_empty
    rag.query_plan_ontology_stats = defaultdict(int)

    _, _, log = rag.apply_query_plan_factorized_query_executor(
        query="Which country is Nahran located in?",
        sorted_doc_ids=np.arange(10, dtype=np.intp),
        sorted_doc_scores=np.linspace(1.0, 0.1, 10, dtype=np.float32),
        query_plan_log=_plan(["Nahran", "unseen corpus alias"], ["located in"], "location"),
    )

    assert log["candidate_count"] == 1
    assert log["candidates"][0]["passage_idx"] == 8
    assert log["candidates"][0]["warehouse_lookup_mode"] == "token_semijoin"
    assert log["candidate_from_beyond_top5_count"] == 1


def test_comparison_branch_qualifier_rejects_ambiguous_album():
    rows = {
        "don t freak me out": [{
            "passage_idx": 1,
            "subject": "Don't Freak Me Out",
            "predicate": "released in",
            "object": "1972",
            "entity_match_score": 1.0,
        }],
        "love war": [
            {
                "passage_idx": 2,
                "subject": "Love + War",
                "predicate": "released on",
                "object": "11 September 2015",
                "entity_match_score": 1.0,
            },
            {
                "passage_idx": 3,
                "subject": "Love & War",
                "predicate": "released on",
                "object": "10 September 2013",
                "entity_match_score": 1.0,
            },
        ],
    }
    passages = {
        1: {"passage": "Don't Freak Me Out is an album released in 1972."},
        2: {"passage": "Love + War is the debut album by Kwabs."},
        3: {"passage": "Love & War is the fourth studio album by BarlowGirl."},
    }
    executor = QueryPlanFactorizedExecutor(
        enable_typed_semantic_compiler=True,
        require_exact_entity_keys=True,
        enable_branch_complete_compiler=True,
        require_branch_qualifier_evidence=True,
        min_relation_score=0.60,
        min_candidate_utility=0.60,
    )
    result = executor.run(
        query="Which album was released first, Don't Freak Me Out or Love & War (Barlowgirl Album)?",
        dataset="2wikimultihopqa",
        base_doc_ids=list(range(10)),
        query_plan_log=_plan(
            ["Don't Freak Me Out", "Love & War", "Barlowgirl"],
            ["released"],
            "entity",
        ),
        dense_retrieve=_dense_empty,
        get_openie_record=lambda passage_idx: passages.get(passage_idx, {}),
        lookup_entity_relations=lambda entity, limit: rows.get(entity.lower(), [])[:limit],
    )

    assert result.log["selected_operator"] == "dual_branch_comparison_join"
    assert result.groups[0]["passage_indices"] == [1, 3]
    assert result.groups[0]["qualifier_coverage"] == 1.0
    assert result.log["runtime_rejections"]["comparison_qualifier_mismatch"] >= 1


def test_learned_delta_admission_uses_gain_and_loss_heads(tmp_path):
    def head(bias):
        return {
            "feature_names": FEATURE_NAMES,
            "weights": [0.0] * len(FEATURE_NAMES),
            "means": [0.0] * len(FEATURE_NAMES),
            "stds": [1.0] * len(FEATURE_NAMES),
            "bias": bias,
        }

    model_path = tmp_path / "delta-model.json"
    model_path.write_text(
        __import__("json").dumps({
            "type": "factorized_delta_admission_model",
            "sections": {
                "top5": {
                    "name": "top5",
                    "enabled": True,
                    "mean_gain": 0.5,
                    "mean_loss_magnitude": 0.5,
                    "gain_head": head(2.0),
                    "loss_head": head(-3.0),
                },
            },
        }),
        encoding="utf-8",
    )
    model = FactorizedDeltaAdmissionModel(str(model_path))
    decision = model.decide(
        {"dataset": "musique", "operator": "cascaded_bind_join", "target": "top5"},
        min_expected_delta=0.0,
        max_loss_probability=0.20,
    )

    assert decision["approved"] is True
    assert decision["gain_probability"] > decision["loss_probability"]


def test_compassrag_learned_delta_admission_fails_closed_without_model():
    rag = object.__new__(CompassRAG)
    rag.global_config = SimpleNamespace(
        dataset="popqa",
        query_plan_ontology_enable_factorized_query_executor=True,
        query_plan_ontology_factorized_target_datasets="popqa",
        query_plan_ontology_factorized_enable_entity_relation_index_scan=True,
        query_plan_ontology_factorized_enable_cascaded_bind_join=True,
        query_plan_ontology_factorized_enable_dual_branch_comparison_join=True,
        query_plan_ontology_factorized_min_candidate_utility=0.60,
        query_plan_ontology_factorized_min_top2_utility=0.60,
        query_plan_ontology_factorized_min_top5_utility=0.60,
        query_plan_ontology_factorized_enable_top2_admission=True,
        query_plan_ontology_factorized_enable_top5_admission=True,
        query_plan_ontology_factorized_enable_simple_top2_admission=False,
        query_plan_ontology_factorized_enable_learned_delta_admission=True,
        query_plan_ontology_factorized_delta_admission_model_path="/missing/delta-model.json",
        query_plan_ontology_factorized_delta_admission_fail_open=False,
        query_plan_ontology_factorized_delta_admission_collect_only=False,
        query_plan_ontology_factorized_entity_scan_mutation_target_datasets="popqa",
    )
    rag.passage_node_keys = [str(idx) for idx in range(10)]
    rag.openie_record_by_passage_key = {
        "8": {"extracted_triples": [["France", "capital", "Paris"]]},
    }
    rag.dense_passage_retrieval = _dense_empty
    rag.query_plan_ontology_stats = defaultdict(int)

    doc_ids, _, log = rag.apply_query_plan_factorized_query_executor(
        query="What is the capital of France?",
        sorted_doc_ids=np.arange(10, dtype=np.intp),
        sorted_doc_scores=np.linspace(1.0, 0.1, 10, dtype=np.float32),
        query_plan_log=_plan(["France"], ["capital"], "location"),
    )

    assert list(doc_ids[:5]) == [0, 1, 2, 3, 4]
    assert log["route_reason"] == "entity_relation_learned_delta_rejected"
    decision = log["counterfactual_actions"][0]["learned_delta_admission"]
    assert decision["approved"] is False
    assert decision["reason"] == "model_not_found"


def test_gold_coverage_counts_distinct_gold_documents():
    class Store:
        rows = {
            "0": {"content": "Alpha is in France."},
            "1": {"content": "Alpha is in France."},
            "2": {"content": "Beta is in Germany."},
        }

        def get_row(self, key):
            return self.rows[key]

    rag = object.__new__(CompassRAG)
    rag.passage_node_keys = ["0", "1", "2"]
    rag.chunk_embedding_store = Store()
    gold_docs = ["Alpha is in France.", "Beta is in Germany."]

    covered = set()
    for passage_idx in [0, 1, 2]:
        covered.update(rag.passage_matching_gold_doc_indices(passage_idx, gold_docs))

    assert covered == {0, 1}
    assert len(covered) / len(gold_docs) == 1.0


def test_comparison_router_requires_two_sided_coordinated_branches():
    false_positive_cases = [
        (
            "What year did the governor whose patron was the same saint take office?",
            ["governor", "saint"],
        ),
        (
            "Who was the second pick after the team selected James first?",
            ["team", "james"],
        ),
        (
            "What country was first to invade the region containing Riga?",
            ["country", "riga"],
        ),
    ]
    for query, anchors in false_positive_cases:
        assert QueryPlanFactorizedExecutor._is_comparison(query, anchors) is False

    assert QueryPlanFactorizedExecutor._is_comparison(
        "Which album was released first, Alpha or Beta?",
        ["alpha", "beta"],
    ) is True
    assert QueryPlanFactorizedExecutor._is_comparison(
        "Are Alpha and Beta both located in the same country?",
        ["alpha", "beta"],
    ) is True


def test_counterfactual_action_materialization_is_independent_and_protects_prefix():
    baseline = list(range(10))
    top2_group = materialize_counterfactual_ranking(
        baseline,
        [7, 8],
        "top2_group",
    )
    top5_single = materialize_counterfactual_ranking(
        baseline,
        [9],
        "top5_single",
    )

    assert baseline == list(range(10))
    assert top2_group["after_top2"] == [7, 8]
    assert top2_group["after_top5"] == [7, 8, 0, 1, 2]
    assert top5_single["after_top2"] == [0, 1]
    assert top5_single["after_top5"] == [0, 1, 2, 3, 9]
    assert top5_single["displaced_passage_indices"] == [4]


def test_action_replay_model_uses_direct_delta_head(tmp_path):
    zero_head = {
        "feature_names": FEATURE_NAMES,
        "weights": [0.0] * len(FEATURE_NAMES),
        "means": [0.0] * len(FEATURE_NAMES),
        "stds": [1.0] * len(FEATURE_NAMES),
        "bias": -4.0,
    }
    positive_delta_head = dict(zero_head)
    positive_delta_head["bias"] = 0.125
    model_path = tmp_path / "action-replay-model.json"
    model_path.write_text(
        __import__("json").dumps({
            "type": "factorized_action_replay_model",
            "sections": {
                "top2": {
                    "name": "top2",
                    "enabled": True,
                    "mean_gain": 0.5,
                    "mean_loss_magnitude": 0.5,
                    "delta_head": positive_delta_head,
                    "gain_head": zero_head,
                    "loss_head": zero_head,
                },
            },
        }),
        encoding="utf-8",
    )
    model = FactorizedDeltaAdmissionModel(str(model_path))
    decision = model.decide(
        {
            "dataset": "2wikimultihopqa",
            "operator": "cascaded_bind_join",
            "target": "top2",
            "action": "atomic_top2_chain_group",
        },
        min_expected_delta=0.01,
        max_loss_probability=0.20,
    )

    assert decision["approved"] is True
    assert decision["expected_delta"] == 0.125
    assert decision["expected_delta_source"] == "direct_counterfactual_regression"


def test_listwise_action_cbo_competes_with_explicit_noop(tmp_path):
    utility_index = FEATURE_NAMES.index("utility")
    weights = [0.0] * len(FEATURE_NAMES)
    weights[utility_index] = 1.0
    model_path = tmp_path / "listwise-model.json"
    model_path.write_text(
        __import__("json").dumps({
            "type": "factorized_listwise_action_cbo_model",
            "metadata": {"supported_datasets": ["2wikimultihopqa"]},
            "dataset_sections": {
                "2wikimultihopqa": {
                    "enabled": True,
                    "min_score_margin": 0.20,
                    "rank_head": {
                        "feature_names": FEATURE_NAMES,
                        "weights": weights,
                        "means": [0.0] * len(FEATURE_NAMES),
                        "stds": [1.0] * len(FEATURE_NAMES),
                        "bias": 0.0,
                    },
                },
            },
        }),
        encoding="utf-8",
    )
    model = FactorizedDeltaAdmissionModel(str(model_path))
    decisions = model.decide_group([
        {"dataset": "2wikimultihopqa", "action": "no_op", "utility": 0.0},
        {"dataset": "2wikimultihopqa", "action": "weak", "utility": 0.1},
        {"dataset": "2wikimultihopqa", "action": "strong", "utility": 0.8},
    ])

    assert model.listwise is True
    assert [item["approved"] for item in decisions] == [True, False, True]
    assert decisions[2]["score_margin"] == 0.8
    assert decisions[2]["expected_delta_source"] == "query_group_listwise_margin"


def test_listwise_action_cbo_abstains_on_unsupported_domain(tmp_path):
    model_path = tmp_path / "listwise-model.json"
    model_path.write_text(
        __import__("json").dumps({
            "type": "factorized_listwise_action_cbo_model",
            "metadata": {"supported_datasets": ["2wikimultihopqa"]},
            "dataset_sections": {},
        }),
        encoding="utf-8",
    )
    model = FactorizedDeltaAdmissionModel(str(model_path))
    decisions = model.decide_group([
        {"dataset": "nq_rear", "action": "no_op"},
        {"dataset": "nq_rear", "action": "warehouse_candidate_top5"},
    ])

    assert decisions[0]["approved"] is True
    assert decisions[0]["reason"] == "explicit_no_op"
    assert decisions[1]["approved"] is False
    assert decisions[1]["reason"] == "unsupported_domain_abstain"
    assert decisions[1]["dataset_supported"] is False


def test_v185_noop_has_an_explicit_non_mutating_feature_contract():
    no_op = factorized_action_features_v185({
        "dataset": "2wikimultihopqa",
        "action": "no_op",
        "semantic_score": 1.0,
    })
    top5 = factorized_action_features_v185({
        "dataset": "2wikimultihopqa",
        "action": "warehouse_candidate_top5",
        "operator": "evidence_warehouse_scan",
        "target": "top5",
        "semantic_score": 0.8,
    })

    assert list(no_op) == V185_FEATURE_NAMES
    assert no_op["action_no_op"] == 1.0
    assert no_op["dataset_2wiki"] == 1.0
    assert no_op["target_top2"] == 0.0
    assert no_op["target_top5"] == 0.0
    assert no_op["action_top5_single"] == 0.0
    assert no_op["semantic_score"] == 0.0
    assert top5["action_no_op"] == 0.0
    assert top5["target_top5"] == 1.0
    assert top5["operator_evidence_warehouse"] == 1.0


def test_v185_hierarchical_cbo_ranks_then_gates_target_winner(tmp_path):
    rank_weights = [0.0] * len(V185_FEATURE_NAMES)
    rank_weights[V185_FEATURE_NAMES.index("utility")] = 1.0

    def gate_head(bias):
        return {
            "feature_names": HIERARCHICAL_GATE_FEATURE_NAMES,
            "weights": [0.0] * len(HIERARCHICAL_GATE_FEATURE_NAMES),
            "means": [0.0] * len(HIERARCHICAL_GATE_FEATURE_NAMES),
            "stds": [1.0] * len(HIERARCHICAL_GATE_FEATURE_NAMES),
            "bias": bias,
        }

    model_path = tmp_path / "hierarchical-model.json"
    model_path.write_text(
        __import__("json").dumps({
            "type": "factorized_hierarchical_action_cbo_model",
            "metadata": {"supported_datasets": ["2wikimultihopqa"]},
            "dataset_sections": {
                "2wikimultihopqa": {
                    "enabled": True,
                    "target_sections": {
                        "top5": {
                            "enabled": True,
                            "rank_head": {
                                "feature_names": V185_FEATURE_NAMES,
                                "weights": rank_weights,
                                "means": [0.0] * len(V185_FEATURE_NAMES),
                                "stds": [1.0] * len(V185_FEATURE_NAMES),
                                "bias": 0.0,
                            },
                            "gain_head": gate_head(4.0),
                            "loss_head": gate_head(-4.0),
                            "mean_gain": 0.5,
                            "mean_loss_magnitude": 0.5,
                            "min_expected_delta": 0.1,
                            "max_loss_probability": 0.2,
                        },
                    },
                },
            },
        }),
        encoding="utf-8",
    )
    model = FactorizedDeltaAdmissionModel(str(model_path))
    decisions = model.decide_group([
        {"dataset": "2wikimultihopqa", "action": "no_op"},
        {
            "dataset": "2wikimultihopqa",
            "action": "weak_top5",
            "target": "top5",
            "utility": 0.1,
        },
        {
            "dataset": "2wikimultihopqa",
            "action": "strong_top5",
            "target": "top5",
            "utility": 0.9,
        },
    ])

    assert model.listwise is True
    assert model.hierarchical is True
    assert decisions[0]["reason"] == "explicit_no_op"
    assert decisions[1]["reason"] == "hierarchical_rank_loser"
    assert decisions[2]["approved"] is True
    assert decisions[2]["reason"] == "hierarchical_opportunity_gain"
    assert decisions[2]["expected_delta_source"] == "hierarchical_gain_loss_gate"


def test_compassrag_action_replay_uses_one_listwise_group_decision(tmp_path):
    utility_index = FEATURE_NAMES.index("utility")
    weights = [0.0] * len(FEATURE_NAMES)
    weights[utility_index] = 1.0
    model_path = tmp_path / "listwise-runtime-model.json"
    model_path.write_text(
        __import__("json").dumps({
            "type": "factorized_listwise_action_cbo_model",
            "metadata": {"supported_datasets": ["popqa"]},
            "dataset_sections": {
                "popqa": {
                    "enabled": True,
                    "min_score_margin": 0.10,
                    "rank_head": {
                        "feature_names": FEATURE_NAMES,
                        "weights": weights,
                        "means": [0.0] * len(FEATURE_NAMES),
                        "stds": [1.0] * len(FEATURE_NAMES),
                        "bias": 0.0,
                    },
                },
            },
        }),
        encoding="utf-8",
    )
    rag = object.__new__(CompassRAG)
    rag.global_config = SimpleNamespace(
        dataset="popqa",
        query_plan_ontology_enable_factorized_query_executor=True,
        query_plan_ontology_factorized_target_datasets="popqa",
        query_plan_ontology_factorized_enable_entity_relation_index_scan=True,
        query_plan_ontology_factorized_enable_cascaded_bind_join=True,
        query_plan_ontology_factorized_enable_dual_branch_comparison_join=True,
        query_plan_ontology_factorized_min_candidate_utility=0.60,
        query_plan_ontology_factorized_min_top2_utility=0.60,
        query_plan_ontology_factorized_min_top5_utility=0.60,
        query_plan_ontology_factorized_enable_top2_admission=True,
        query_plan_ontology_factorized_enable_top5_admission=True,
        query_plan_ontology_factorized_enable_simple_top2_admission=True,
        query_plan_ontology_factorized_enable_counterfactual_action_replay=True,
        query_plan_ontology_factorized_counterfactual_action_replay_max_actions=8,
        query_plan_ontology_factorized_action_replay_include_evidence_warehouse=False,
        query_plan_ontology_factorized_action_replay_enable_atomic_chain_top2=True,
        query_plan_ontology_factorized_enable_learned_delta_admission=True,
        query_plan_ontology_factorized_delta_admission_collect_only=False,
        query_plan_ontology_factorized_delta_admission_model_path=str(model_path),
        query_plan_ontology_factorized_delta_admission_fail_open=False,
    )
    rag.passage_node_keys = [str(idx) for idx in range(10)]
    rag.openie_record_by_passage_key = {
        "8": {
            "passage": "Paris is the capital city of France.",
            "extracted_triples": [["France", "capital", "Paris"]],
        },
    }
    rag.dense_passage_retrieval = _dense_empty
    rag.query_plan_ontology_stats = defaultdict(int)

    doc_ids, _, log = rag.apply_query_plan_factorized_query_executor(
        query="What is the capital of France?",
        sorted_doc_ids=np.arange(10, dtype=np.intp),
        sorted_doc_scores=np.linspace(1.0, 0.1, 10, dtype=np.float32),
        query_plan_log=_plan(["France"], ["capital"], "location"),
    )

    optimizer = log["action_replay_optimizer"]
    assert list(doc_ids[:2]) == [0, 8]
    assert optimizer["selection_mode"] == "domain_listwise_noop_competition"
    assert optimizer["dataset_supported"] is True
    assert optimizer["selected_action"] == "simple_entity_relation_top2"
    assert optimizer["selected_score_margin"] > 0.0


def test_compassrag_action_replay_collects_uncommitted_alternatives():
    rag = object.__new__(CompassRAG)
    rag.global_config = SimpleNamespace(
        dataset="popqa",
        query_plan_ontology_enable_factorized_query_executor=True,
        query_plan_ontology_factorized_target_datasets="popqa",
        query_plan_ontology_factorized_enable_entity_relation_index_scan=True,
        query_plan_ontology_factorized_enable_cascaded_bind_join=True,
        query_plan_ontology_factorized_enable_dual_branch_comparison_join=True,
        query_plan_ontology_factorized_min_candidate_utility=0.60,
        query_plan_ontology_factorized_min_top2_utility=0.60,
        query_plan_ontology_factorized_min_top5_utility=0.60,
        query_plan_ontology_factorized_enable_top2_admission=True,
        query_plan_ontology_factorized_enable_top5_admission=True,
        query_plan_ontology_factorized_enable_simple_top2_admission=True,
        query_plan_ontology_factorized_enable_counterfactual_action_replay=True,
        query_plan_ontology_factorized_counterfactual_action_replay_max_actions=8,
        query_plan_ontology_factorized_action_replay_include_evidence_warehouse=True,
        query_plan_ontology_factorized_action_replay_enable_atomic_chain_top2=True,
        query_plan_ontology_factorized_enable_learned_delta_admission=False,
        query_plan_ontology_factorized_delta_admission_collect_only=True,
        query_plan_ontology_factorized_delta_admission_model_path="",
    )
    rag.passage_node_keys = [str(idx) for idx in range(10)]
    rag.openie_record_by_passage_key = {
        "8": {"extracted_triples": [["France", "capital", "Paris"]]},
    }
    rag.dense_passage_retrieval = _dense_empty
    rag.query_plan_ontology_stats = defaultdict(int)

    doc_ids, _, log = rag.apply_query_plan_factorized_query_executor(
        query="What is the capital of France?",
        sorted_doc_ids=np.arange(10, dtype=np.intp),
        sorted_doc_scores=np.linspace(1.0, 0.1, 10, dtype=np.float32),
        query_plan_log=_plan(["France"], ["capital"], "location"),
    )

    assert list(doc_ids[:5]) == [0, 1, 2, 3, 4]
    assert log["action_replay_optimizer"]["collect_only"] is True
    assert log["action_replay_optimizer"]["candidate_action_count"] == 2
    assert [
        item["action"] for item in log["action_replay_warehouse"]
    ] == ["no_op", "simple_entity_relation_top2", "simple_entity_relation_top5"]
    assert all(
        item["learned_delta_admission"]["reason"] == "collect_only_no_mutation"
        for item in log["action_replay_warehouse"]
        if item["action"] != "no_op"
    )


def test_optimizer_candidate_pool_is_residual_to_frozen_incumbent():
    rag = object.__new__(CompassRAG)
    rag.global_config = SimpleNamespace(
        dataset="2wikimultihopqa",
        query_plan_ontology_enable_factorized_query_executor=True,
        query_plan_ontology_factorized_target_datasets="2wikimultihopqa",
        query_plan_ontology_factorized_enable_entity_relation_index_scan=True,
        query_plan_ontology_factorized_enable_cascaded_bind_join=True,
        query_plan_ontology_factorized_enable_dual_branch_comparison_join=True,
        query_plan_ontology_factorized_enable_counterfactual_action_replay=True,
        query_plan_ontology_factorized_counterfactual_action_replay_max_actions=8,
        query_plan_ontology_factorized_action_replay_include_evidence_warehouse=True,
        query_plan_ontology_factorized_enable_learned_delta_admission=False,
        query_plan_ontology_factorized_delta_admission_collect_only=True,
        query_plan_ontology_factorized_delta_admission_model_path="",
        query_plan_ontology_evidence_set_enable_optimizer_candidate_pool=True,
    )
    rag.passage_node_keys = [str(idx) for idx in range(10)]
    rag.openie_record_by_passage_key = {}
    rag.dense_passage_retrieval = _dense_empty
    rag.query_plan_ontology_stats = defaultdict(int)
    plan = _plan(["France"], ["capital"], "location")
    plan["evidence_set_retriever"] = {
        "candidate_pool_samples": [{
            "passage_idx": 7,
            "confidence": 0.8,
            "text_verifier": {"score": 0.8},
            "path_aware": {"score": 0.7},
        }],
        "optimizer_candidate_pool_samples": [{
            "passage_idx": 8,
            "confidence": 0.9,
            "text_verifier": {"score": 0.9},
            "path_aware": {"score": 0.8},
        }],
    }

    _, _, log = rag.apply_query_plan_factorized_query_executor(
        query="What is the capital of France?",
        sorted_doc_ids=np.arange(10, dtype=np.intp),
        sorted_doc_scores=np.linspace(1.0, 0.1, 10, dtype=np.float32),
        query_plan_log=plan,
    )

    optimizer = log["action_replay_optimizer"]
    actions = log["action_replay_warehouse"]
    assert optimizer["incumbent_base_action_count"] == 3
    assert optimizer["optimizer_candidate_pool_action_source_count"] == 1
    assert optimizer["optimizer_candidate_pool_residual_action_count"] == 2
    assert [
        item["selected_passage_indices"] for item in actions[1:]
    ] == [[7], [7], [8], [8]]


def test_qa_api_concurrency_preserves_input_order():
    import threading
    import time

    class Model:
        def __init__(self):
            self.lock = threading.Lock()
            self.active = 0
            self.max_active = 0

        def infer(self, value):
            with self.lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            time.sleep(0.01 * (4 - int(value)))
            with self.lock:
                self.active -= 1
            return value, {"value": value}, False

    rag = object.__new__(CompassRAG)
    rag.global_config = SimpleNamespace(qa_api_concurrency=4)
    rag.llm_model = Model()
    results = rag._infer_qa_messages([0, 1, 2, 3])

    assert [item[0] for item in results] == [0, 1, 2, 3]
    assert rag.llm_model.max_active > 1


def test_v228_role_aware_slot_binder_rejects_predicates_and_literals():
    contract = QueryPlanFactorizedExecutor._bind_query_surface_slots(
        query="Where did Henrique Salas Feo's father study in 1900?",
        anchors=["Henrique Salas Feo", "father", "study", "1900"],
        enable_branch_compiler=True,
    )

    assert contract["executable_anchors"] == ["henrique salas feo"]
    assert set(contract["dropped_non_entity_role_anchors"]) >= {"father", "study", "1900"}


def test_v230_role_aware_slot_binder_rejects_composite_role_phrase():
    contract = QueryPlanFactorizedExecutor._bind_query_surface_slots(
        query="When was the director of film But What If This Is Love born?",
        anchors=["But What If This Is Love", "director of film"],
        enable_branch_compiler=True,
    )

    assert contract["executable_anchors"] == ["but what if this is love"]
    assert "director of film" in contract["dropped_non_entity_role_anchors"]


def test_v230_provenance_plan_portfolio_retains_valid_prefix_after_join_miss():
    rows = {
        "alice": [{
            "passage_idx": 10,
            "subject": "Alice",
            "predicate": "mother",
            "object": "Beth",
            "entity_match_score": 1.0,
        }],
    }
    executor = QueryPlanFactorizedExecutor(
        enable_provenance_plan_portfolio=True,
        min_candidate_utility=0.60,
    )
    result = executor.run(
        query="Where was Alice's mother born?",
        dataset="musique",
        base_doc_ids=list(range(30)),
        query_plan_log=_plan(["Alice"], ["mother", "born in"], "location"),
        dense_retrieve=_dense_empty,
        get_openie_record=_openie_empty,
        lookup_entity_relations=lambda entity, limit: rows.get(entity.lower(), [])[:limit],
    )

    portfolio = result.log["provenance_plan_portfolio"]
    assert portfolio["triggered"] is True
    assert portfolio["decision_reason"] == "partial_proofs_materialized"
    assert portfolio["covered_relation_slots"] == [0]
    assert result.log["primary_selected_operator"] == "cascaded_bind_join"
    assert result.log["selected_operator"] == "entity_relation_index_scan"
    assert result.candidates[0]["passage_idx"] == 10
    assert result.candidates[0]["provenance_plan_portfolio"] is True
    assert result.candidates[0]["partial_proof_completeness"] == 0.5


def test_v228_selected_schema_limits_realizations_to_exact_hops():
    schema_hash = "schema-education"
    query_plan_log = {
        "enabled": True,
        "anchor_entities": ["Henrique Salas Feo", "father", "study"],
        "query_condition": {
            "answer_type": "location",
            "relation_terms": ["father", "study"],
            "relation_signatures": ["family_father", "education"],
        },
        "matched_plans": [{
            "id": "education-plan",
            "answer_type": "location",
            "schema_sha256": schema_hash,
            "path": [
                {"relation_signature": "family_father"},
                {"relation_signature": "education"},
            ],
            "relation_signatures": ["family_father", "education"],
            "relation_families": ["family", "membership"],
            "integrity_constraints": {"relation_directions": ["forward", "forward"]},
        }, {
            "id": "birth-plan",
            "answer_type": "location",
            "schema_sha256": "schema-birth",
            "path": [
                {"relation_signature": "family_father"},
                {"relation_signature": "birth_place"},
            ],
            "relation_signatures": ["family_father", "birth_place"],
            "relation_families": ["family", "birth"],
        }],
    }
    realization_map = {
        "plans": {
            "education-plan": {
                "schema_sha256": schema_hash,
                "steps": {
                    "0": {
                        "expected_relation_signature": "family father",
                        "expected_direction": "forward",
                        "realizations": [{"normalized_relation": "father", "count": 4, "score": 5.0}],
                    },
                    "1": {
                        "expected_relation_signature": "education",
                        "expected_direction": "forward",
                        "realizations": [{"normalized_relation": "studied at", "count": 3, "score": 4.0}],
                    },
                },
            },
            "birth-plan": {
                "schema_sha256": "schema-birth",
                "steps": {"1": {"realizations": [{"normalized_relation": "plays", "count": 99, "score": 99.0}]}},
            },
        },
    }
    rows = {
        "henrique salas feo": [{
            "passage_idx": 10,
            "subject": "Henrique Salas Feo",
            "predicate": "father",
            "object": "Jose Salas",
            "entity_match_score": 1.0,
        }],
        "jose salas": [{
            "passage_idx": 11,
            "subject": "Jose Salas",
            "predicate": "studied at",
            "object": "University of Carabobo",
            "entity_match_score": 1.0,
        }],
    }
    executor = QueryPlanFactorizedExecutor(
        enable_typed_semantic_compiler=True,
        enable_executable_ontology_compiler=True,
        enable_relation_realization=True,
        relation_realization_map=realization_map,
        require_relation_artifact_contract=True,
        require_exact_entity_keys=True,
        min_relation_score=0.50,
        min_type_score=0.50,
        min_candidate_utility=0.50,
    )
    result = executor.run(
        query="Where did Henrique Salas Feo's father study?",
        dataset="2wikimultihopqa",
        base_doc_ids=list(range(20)),
        query_plan_log=query_plan_log,
        dense_retrieve=_dense_empty,
        get_openie_record=_openie_empty,
        lookup_entity_relations=lambda entity, limit: rows.get(entity.lower(), [])[:limit],
    )

    realization = result.log["typed_contract"]["relation_realization"]
    assert realization["selected_plan_id"] == "education-plan"
    assert realization["artifact_contract_ok"] is True
    assert "plays" not in str(realization)
    assert result.log["typed_contract"]["source_plan_id"] == "education-plan"
    assert result.groups[0]["passage_indices"] == [10, 11]


def test_v229_relation_keyed_catalog_resolves_nonpositional_artifact_step():
    realization_map = {
        "plans": {
            "duplicated-plan": {
                "steps": {
                    "0": {
                        "expected_relation_signature": "director",
                        "realizations": [{"normalized_relation": "directed by", "count": 5, "score": 4.0}],
                    },
                    "1": {
                        "expected_relation_signature": "director",
                        "realizations": [{"normalized_relation": "film director", "count": 3, "score": 3.0}],
                    },
                    "2": {
                        "expected_relation_signature": "birth date",
                        "realizations": [{"normalized_relation": "was born on", "count": 7, "score": 5.0}],
                    },
                },
            },
        },
    }
    runtime_plan = {
        "id": "duplicated-plan",
        "path": [
            {"relation_signature": "director"},
            {"relation_signature": "director"},
            {"relation_signature": "birth_date"},
        ],
    }
    executor = QueryPlanFactorizedExecutor(
        enable_relation_realization=True,
        enable_relation_keyed_catalog=True,
        relation_realization_map=realization_map,
    )

    realization = executor._compile_relation_realizations(
        query_plan_log={"matched_plans": [runtime_plan]},
        relations=["director", "birth date"],
        selected_schema={"plan_id": "duplicated-plan"},
    )

    assert realization["relation_keyed_catalog"] is True
    assert realization["unresolved_relations"] == []
    assert realization["relations"]["birth date"][0]["predicate"] == "was born on"
    assert realization["relations"]["birth date"][0]["source_step"] == 2


def test_v255_certified_predicates_keep_their_own_observed_direction():
    realization_map = {
        "version": 4,
        "enforce_certified_only": True,
        "entity_binding_contract": "safe_canonical_alias_exact_match",
        "plans": {
            "test-plan": {
                "certified_executable": True,
                "steps": {
                    "0": {
                        "expected_relation_signature": "director",
                        "realizations": [
                            {
                                "normalized_relation": "directed by",
                                "count": 20,
                                "score": 5.0,
                                "historical_support": 12,
                                "historical_concept_precision_lcb": 0.80,
                                "semantic_compatibility": 1.0,
                                "direction_consistency": 1.0,
                                "observed_direction": "forward",
                                "certified": True,
                            },
                            {
                                "normalized_relation": "directed",
                                "count": 15,
                                "score": 4.0,
                                "historical_support": 8,
                                "historical_concept_precision_lcb": 0.70,
                                "semantic_compatibility": 1.0,
                                "direction_consistency": 1.0,
                                "observed_direction": "reverse",
                                "certified": True,
                            },
                        ],
                    },
                },
            },
        },
    }
    executor = QueryPlanFactorizedExecutor(
        enable_typed_semantic_compiler=True,
        enable_relation_realization=True,
        relation_realization_map=realization_map,
        require_exact_entity_keys=True,
        min_relation_score=0.90,
        min_type_score=0.0,
    )
    executor._compile_relation_realizations(
        query_plan_log={
            "matched_plans": [{
                "id": "test-plan",
                "path": [{"relation_signature": "director"}],
            }],
        },
        relations=["director"],
        selected_schema={"plan_id": "test-plan"},
    )
    rows = [
        {
            "passage_idx": 1,
            "subject": "Film A",
            "predicate": "directed by",
            "object": "Person A",
        },
        {
            "passage_idx": 2,
            "subject": "Person B",
            "predicate": "directed",
            "object": "Film A",
        },
        {
            "passage_idx": 3,
            "subject": "Film A",
            "predicate": "produced by",
            "object": "Person C",
        },
    ]

    selected = executor._indexed_access(
        binding="Film A",
        relation="director",
        answer_type="person",
        terminal_step=False,
        lookup_entity_relations=lambda _entity, limit: rows[:limit],
    )

    assert {item["terminal_entity"] for item in selected} == {"Person A", "Person B"}
    assert {item["access_direction"] for item in selected} == {"forward", "reverse"}
    assert all(item["predicate"] != "produced by" for item in selected)


def test_v255_safe_alias_binding_is_exact_but_parenthetical_aware():
    assert QueryPlanFactorizedExecutor._exact_endpoint_match(
        "The Orphan (film)",
        "The Orphan",
    )
    assert not QueryPlanFactorizedExecutor._exact_endpoint_match(
        "The Orphanage",
        "The Orphan",
    )


def test_v333_incomplete_plan_retains_certified_prefix_for_terminal_projection():
    realization_map = {
        "version": 4,
        "enforce_certified_only": True,
        "plans": {
            "partial-plan": {
                "certified_executable": False,
                "steps": {
                    "0": {
                        "expected_relation_signature": "director",
                        "certification_relation_terms": ["director"],
                        "realizations": [{
                            "normalized_relation": "directed by",
                            "certified": True,
                            "observed_direction": "forward",
                        }],
                    },
                    "1": {
                        "expected_relation_signature": "birth place",
                        "certification_relation_terms": ["birth place"],
                        "realizations": [],
                    },
                },
            },
        },
    }
    executor = QueryPlanFactorizedExecutor(
        enable_relation_realization=True,
        enable_relation_keyed_catalog=True,
        relation_realization_map=realization_map,
    )
    realization = executor._compile_relation_realizations(
        query_plan_log={
            "matched_plans": [{
                "id": "partial-plan",
                "path": [
                    {"relation_signature": "director"},
                    {"relation_signature": "birth_place"},
                ],
            }],
        },
        relations=["director", "birth place"],
        selected_schema={"plan_id": "partial-plan"},
    )

    assert realization["realized_relation_count"] == 1
    assert realization["relations"]["director"][0]["predicate"] == "directed by"
    assert realization["unresolved_relations"] == ["birth place"]
    assert executor._runtime_rejections["uncertified_incomplete_physical_plan"] == 1


def test_v333_terminal_projection_requires_strict_endpoint_bound_evidence():
    certification = QueryPlanFactorizedExecutor._certify_entity_bound_terminal_projection(
        relations=["director", "cause of death"],
        relation_realization={
            "certified_only": True,
            "unresolved_relations": ["cause of death"],
            "relations": {
                "director": [{"predicate": "directed by", "certified": True}],
                "cause of death": [],
            },
        },
        candidates=[{
            "passage_idx": 57,
            "group_id": "strict-terminal",
            "exact_entity_key": True,
            "provenance_validated": True,
            "candidate_window_path": True,
            "contextual_relation_count": 1,
        }],
        groups=[{
            "group_id": "strict-terminal",
            "strict_chain_certificate": True,
            "provenance_validated": True,
            "schema_to_evidence_grounding": True,
            "exact_anchor": True,
            "chain_completeness": 1.0,
            "contextual_relation_count": 1,
        }],
        endpoint_grounded_source_schema_ids=["director-cause-schema"],
    )

    assert certification["certified"] is True
    assert certification["candidate_doc_ids"] == [57]
    assert certification["operator"] == "entity_bound_passage_projection"


def test_v333_terminal_projection_rejects_noncontextual_bound_passage():
    certification = QueryPlanFactorizedExecutor._certify_entity_bound_terminal_projection(
        relations=["director", "cause of death"],
        relation_realization={
            "certified_only": True,
            "unresolved_relations": ["cause of death"],
            "relations": {
                "director": [{"predicate": "directed by", "certified": True}],
                "cause of death": [],
            },
        },
        candidates=[{
            "passage_idx": 57,
            "group_id": "noncontextual-terminal",
            "exact_entity_key": True,
            "provenance_validated": True,
            "candidate_window_path": True,
            "contextual_relation_count": 0,
        }],
        groups=[{
            "group_id": "noncontextual-terminal",
            "strict_chain_certificate": True,
            "provenance_validated": True,
            "schema_to_evidence_grounding": True,
            "exact_anchor": True,
            "chain_completeness": 1.0,
            "contextual_relation_count": 0,
        }],
        endpoint_grounded_source_schema_ids=["director-cause-schema"],
    )

    assert certification["certified"] is False
    assert certification["reason"] == "no_strict_entity_bound_terminal_passage"


def test_v229_query_terminal_type_overrides_generic_schema_type():
    executor = QueryPlanFactorizedExecutor(
        enable_typed_semantic_compiler=True,
        enable_executable_ontology_compiler=True,
        enable_execution_aware_contract=True,
        min_query_predicate_coverage=0.50,
    )
    contract = executor._compile_typed_contract(
        query="When was the director of But What If This Is Love born?",
        query_plan_log={
            "query_condition": {
                "answer_type": "date",
                "relation_terms": ["director", "birth date"],
                "compiler_schema_plans": [{
                    "id": "generic-schema",
                    "answer_type": "entity",
                    "precision": 0.9,
                    "path": [
                        {"relation_signature": "director"},
                        {"relation_signature": "birth_date"},
                    ],
                }],
            },
        },
        plan={"anchors": ["But What If This Is Love"], "answer_type": "entity"},
    )

    assert contract["answer_type"] == "date"
    assert contract["answer_type_source"] == "query_terminal_relation"
    assert contract["schema_answer_type"] == "entity"
    assert contract["answer_type_compatibility"] == 0.35
    assert contract["execution_contract_valid"] is True
