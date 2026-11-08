import json

from compassrag.query_plan_delta_admission import (
    V186_FEATURE_NAMES,
    V186_HIERARCHICAL_GATE_FEATURE_NAMES,
    FactorizedDeltaAdmissionModel,
    factorized_action_features_v186,
)
from compassrag.query_plan_evidence_set_compiler import (
    compile_ontology_evidence_sets,
)


def _candidate(passage_idx, slot_id, plan_id, rank_score=0.9):
    return {
        "passage_idx": passage_idx,
        "score": 4.0,
        "confidence": rank_score,
        "coverage_gain": 2.0,
        "text_verifier": {"score": rank_score},
        "path_aware": {"score": rank_score},
        "contexts": [{
            "slot_ids": [slot_id],
            "plan_id": plan_id,
            "support_key": f"ontology_path_executor::{plan_id}",
            "anchor_entity": "alpha",
            "end_entity": slot_id,
            "relations": [slot_id],
        }],
    }


def test_compiler_emits_complementary_top2_and_top5_sets():
    result = compile_ontology_evidence_sets(
        warehouse_samples=[
            _candidate(6, "slot:director", "plan-a"),
            _candidate(7, "slot:birth-place", "plan-b"),
            _candidate(8, "slot:country", "plan-c", 0.8),
        ],
        base_doc_ids=list(range(12)),
        probes=[{"matched_count": 8}],
        max_candidates=3,
        beam_width=16,
    )

    top2 = [item for item in result["actions"] if item["target"] == "top2"]
    top5 = [item for item in result["actions"] if item["target"] == "top5"]
    assert top2
    assert top5
    assert len(top2[0]["selected_passage_indices"]) == 2
    assert top2[0]["profile"]["slot_gain"] == 1
    assert top2[0]["profile"]["provenance_diversity"] == 1.0
    assert top5[0]["profile"]["execution_cost"] > 0.0


def test_compiler_rejects_redundant_same_provenance_pairs():
    first = _candidate(6, "slot:same", "plan-a")
    second = _candidate(7, "slot:same", "plan-a")
    result = compile_ontology_evidence_sets(
        warehouse_samples=[first, second],
        base_doc_ids=list(range(10)),
        require_complementarity=True,
    )

    assert result["actions"] == []


def test_v186_feature_contract_exposes_set_quality_and_cost():
    features = factorized_action_features_v186({
        "dataset": "2wikimultihopqa",
        "target": "top5",
        "action": "ontology_evidence_set_top5",
        "set_compiler_score": 0.8,
        "set_slot_coverage": 0.75,
        "set_execution_cost": 0.25,
        "set_member_count": 3,
        "set_external_count": 2,
    })

    assert list(features) == V186_FEATURE_NAMES
    assert features["action_evidence_set"] == 1.0
    assert features["set_slot_coverage"] == 0.75
    assert features["set_member_count"] == 1.0
    assert features["set_external_count"] == 2.0 / 3.0


def test_v186_model_uses_the_set_feature_contract(tmp_path):
    rank_weights = [0.0] * len(V186_FEATURE_NAMES)
    rank_weights[V186_FEATURE_NAMES.index("action_evidence_set")] = 2.0

    def gate_head(bias):
        return {
            "feature_names": V186_HIERARCHICAL_GATE_FEATURE_NAMES,
            "weights": [0.0] * len(V186_HIERARCHICAL_GATE_FEATURE_NAMES),
            "means": [0.0] * len(V186_HIERARCHICAL_GATE_FEATURE_NAMES),
            "stds": [1.0] * len(V186_HIERARCHICAL_GATE_FEATURE_NAMES),
            "bias": bias,
        }

    model_path = tmp_path / "evidence-set-model.json"
    model_path.write_text(json.dumps({
        "type": "factorized_risk_calibrated_set_cbo_model",
        "metadata": {"supported_datasets": ["2wikimultihopqa"]},
        "dataset_sections": {
            "2wikimultihopqa": {
                "enabled": True,
                "target_sections": {
                    "top5": {
                        "enabled": True,
                        "rank_head": {
                            "feature_names": V186_FEATURE_NAMES,
                            "weights": rank_weights,
                            "means": [0.0] * len(V186_FEATURE_NAMES),
                            "stds": [1.0] * len(V186_FEATURE_NAMES),
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
    }), encoding="utf-8")
    model = FactorizedDeltaAdmissionModel(str(model_path))
    decisions = model.decide_group([
        {"dataset": "2wikimultihopqa", "action": "no_op"},
        {
            "dataset": "2wikimultihopqa",
            "target": "top5",
            "action": "warehouse_candidate_top5",
            "hard_admissible": True,
        },
        {
            "dataset": "2wikimultihopqa",
            "target": "top5",
            "action": "ontology_evidence_set_top5",
            "hard_admissible": True,
            "set_member_count": 2,
        },
    ])

    assert model.evidence_set_model is True
    assert decisions[1]["reason"] == "hierarchical_rank_loser"
    assert decisions[2]["approved"] is True
