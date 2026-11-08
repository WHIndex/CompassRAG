import argparse
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import plan_optimizer_core as base
from compassrag.query_plan_complete_evidence_plan_optimizer import (
    CompleteEvidencePlanPortfolioOptimizer,
)
from train_plan_optimizer import (
    build_stage2_residual_rows,
    fit_objective_family_score_alignments,
    load_fold_checkpoint,
    sample_stage2_training_rows,
    selected_for_objective_policy,
    write_fold_checkpoint,
)


def sample_rows():
    return [
        {
            "query_key": f"dataset:{index}",
            "dataset": "dataset",
            "plan_name": base.PLAN_SIGNATURE_GROUNDED,
            "plan_instance": f"candidate:{index}",
            "features": {
                name: float(index % 3) for name in base.FEATURE_NAMES
            },
            "delta_r2": 0.5 if index % 2 else 0.0,
            "delta_r5": 0.5 if index % 3 else 0.0,
            "gain_probability": float(index % 2),
            "loss_probability": 0.0,
        }
        for index in range(12)
    ]


class CrossfitCheckpointTest(unittest.TestCase):
    def test_training_base_exports_all_grouped_oof_routing_profiles(self):
        self.assertEqual(
            base.TOPOLOGY_ROUTING_COMBINED_V285,
            "combined_v285",
        )
        self.assertEqual(
            base.TOPOLOGY_ROUTING_HIERARCHICAL_V286,
            "hierarchical_v286",
        )
        self.assertEqual(
            base.TOPOLOGY_ROUTING_OBJECTIVE_FAMILY_V287,
            "objective_family_v287",
        )

    def test_objective_family_models_separate_rank_objectives_and_families(self):
        rows = []
        for index in range(16):
            for objective_class, invariant in (
                ("top2_ordering", 0.0), ("top5_completion", 1.0)
            ):
                for plan_name in ("family_a", "family_b"):
                    features = {
                        name: float(index % 3) for name in base.FEATURE_NAMES
                    }
                    features["plan_top2_membership_invariant"] = invariant
                    rows.append({
                        "query_key": f"dataset:{index}",
                        "plan_name": plan_name,
                        "features": features,
                        "delta_r2": 0.5 if objective_class == "top2_ordering" else 0.0,
                        "delta_r5": 0.5 if index % 2 else 0.0,
                        "gain_probability": float(index % 2),
                        "loss_probability": 0.0,
                    })
        args = argparse.Namespace(
            allowed_plans="family_a,family_b",
            structural_expert_model=False,
            objective_family_outcome_model=True,
            objective_class_outcome_model=False,
            global_outcome_model=False,
            trees=2,
            max_depth=2,
            min_leaf=1,
            max_features=1.0,
            seed=7,
            jobs=1,
            alpha=0.1,
            balance_outcome_targets=True,
            skip_model_export=True,
            skip_residual_calibration=True,
        )

        payloads, models = base.train_models(rows, rows, args)

        expected = {
            base.objective_family_model_key(objective_class, plan_name)
            for objective_class in ("top2_ordering", "top5_completion")
            for plan_name in ("family_a", "family_b")
        }
        self.assertEqual(set(payloads), expected)
        self.assertEqual(set(models), expected)

    def test_stage2_hard_negative_sampling_keeps_outcomes_and_caps_neutral_rows(self):
        rows = []
        for index in range(20):
            features = {name: 0.0 for name in base.FEATURE_NAMES}
            features["residual_candidate_confidence"] = index / 20.0
            rows.append({
                "query_key": "dataset:0",
                "plan_name": "family_a",
                "plan_instance": f"action:{index}",
                "features": features,
                "gain_probability": float(index in {0, 1}),
                "loss_probability": float(index == 2),
            })

        sampled = sample_stage2_training_rows(rows, 4)

        sampled_instances = {row["plan_instance"] for row in sampled}
        self.assertTrue({"action:0", "action:1", "action:2"} <= sampled_instances)
        self.assertEqual(len(sampled), 7)
        self.assertTrue({"action:16", "action:17", "action:18", "action:19"} <= sampled_instances)

    def test_family_policies_admit_then_compare_across_plan_families(self):
        def row(plan_name, gain, r2):
            return {
                "query_key": "dataset:0",
                "plan_name": plan_name,
                "prediction": {
                    "gain_probability": gain,
                    "loss_probability": 0.0,
                    "safety_margin": gain,
                    "delta_r2": r2,
                    "delta_r5": 0.0,
                    "utility_mean": 2.0 * r2,
                },
            }

        family_policy = {
            "selection_mode": "risk_budgeted_lexicographic",
            "calibration_feasible": True,
            "min_utility_mean": 0.0,
            "max_loss_probability": 0.1,
            "min_delta_r2_mean": 0.0,
            "min_delta_r5_mean": 0.0,
            "min_gain_probability": 0.1,
        }
        policy = {
            "selection_mode": "risk_budgeted_lexicographic",
            "calibration_feasible": True,
            "plan_family_policies": {
                "family_a": dict(family_policy),
                "family_b": dict(family_policy),
            },
        }

        selected = selected_for_objective_policy(
            [row("family_a", 0.8, 0.2), row("family_b", 0.7, 0.4)],
            policy,
            "top2_ordering",
        )

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["plan_name"], "family_b")

    def test_family_policy_routes_admission_by_query_contract_regime(self):
        def row(query_key, comparison):
            return {
                "query_key": query_key,
                "plan_name": "family_a",
                "features": {
                    "query_comparison_cue": float(comparison),
                    "logical_required_branch_count": float(2 if comparison else 1),
                    "relation_slot_count": 2.0,
                },
                "prediction": {
                    "gain_probability": 0.8,
                    "loss_probability": 0.0,
                    "safety_margin": 0.8,
                    "delta_r2": 0.4,
                    "delta_r5": 0.1,
                    "utility_mean": 0.9,
                },
            }

        admitted = {
            "selection_mode": "risk_budgeted_lexicographic",
            "calibration_feasible": True,
            "min_utility_mean": 0.0,
            "max_loss_probability": 0.1,
            "min_delta_r2_mean": 0.0,
            "min_delta_r5_mean": 0.0,
            "min_gain_probability": 0.1,
        }
        policy = {
            "selection_mode": "risk_budgeted_lexicographic",
            "calibration_feasible": True,
            "plan_family_policies": {
                "family_a": {
                    "calibration_feasible": True,
                    "structural_regime_policies": {
                        "explicit_two_hop": admitted,
                        "comparison_branch": {
                            **admitted,
                            "calibration_feasible": False,
                        },
                    },
                },
            },
        }

        selected = selected_for_objective_policy(
            [row("dataset:0", False), row("dataset:1", True)],
            policy,
            "top2_ordering",
        )

        self.assertEqual([value["query_key"] for value in selected], ["dataset:0"])

    def test_query_relative_family_gate_compares_against_top2_noop(self):
        def row(plan_name, regret_r2, regret_r5):
            return {
                "query_key": "dataset:0",
                "plan_name": plan_name,
                "features": {
                    "logical_required_branch_count": 1.0,
                    "relation_slot_count": 2.0,
                },
                "prediction": {
                    "regret_r2_ucb": regret_r2,
                    "regret_r5_ucb": regret_r5,
                },
            }

        gate = {
            "selection_mode": "query_relative_slate_regret",
            "calibration_feasible": True,
            "min_regret_advantage": 0.2,
        }
        noop = {
            "selection_mode": "query_relative_slate_regret",
            "calibration_feasible": True,
            "min_regret_advantage": 0.0,
        }
        policy = {
            "selection_mode": "query_relative_slate_regret",
            "calibration_feasible": True,
            "plan_family_policies": {
                "family_a": {
                    "calibration_feasible": True,
                    "structural_regime_policies": {
                        "explicit_two_hop": gate,
                    },
                },
                base.PLAN_SEMANTIC_TOP2_NOOP: noop,
            },
        }
        alternative = row("family_a", 0.10, 0.10)
        comparator = row(base.PLAN_SEMANTIC_TOP2_NOOP, 0.20, 0.20)

        selected = selected_for_objective_policy(
            [alternative, comparator], policy, "top2_ordering"
        )

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["plan_name"], "family_a")
        policy["plan_family_policies"]["family_a"][
            "structural_regime_policies"
        ]["explicit_two_hop"]["min_regret_advantage"] = 0.31
        selected = selected_for_objective_policy(
            [alternative, comparator], policy, "top2_ordering"
        )
        self.assertEqual(selected[0]["plan_name"], base.PLAN_SEMANTIC_TOP2_NOOP)

    def test_runtime_prefers_objective_family_expert(self):
        def expert(delta_r2):
            head = {"feature_names": [], "trees": [], "fallback_value": delta_r2}
            zero = {"feature_names": [], "trees": [], "fallback_value": 0.0}
            return {
                "delta_r2_head": head,
                "delta_r5_head": zero,
                "gain_probability_head": head,
                "loss_probability_head": zero,
            }

        optimizer = CompleteEvidencePlanPortfolioOptimizer("")
        optimizer.model = {
            "plans": {
                base.OBJECTIVE_CLASS_EXPERT_PREFIX + "top2_ordering": expert(0.1),
                base.objective_family_model_key(
                    "top2_ordering", "family_a"
                ): expert(0.7),
            }
        }

        prediction = optimizer._predict("family_a", {})

        self.assertAlmostEqual(prediction["delta_r2"], 0.7)

    def test_paired_oof_alignment_corrects_final_training_score_scale(self):
        oof_rows = []
        final_rows = []
        for index in range(40):
            features = {name: 0.0 for name in base.FEATURE_NAMES}
            value = index / 39.0
            common = {
                "query_key": f"dataset:{index}",
                "plan_name": "family_a",
                "plan_instance": f"action:{index}",
                "features": features,
            }
            oof_rows.append({
                **common,
                "prediction": {"delta_r2": value},
            })
            final_rows.append({
                **common,
                "prediction": {"delta_r2": 2.0 * value},
            })

        alignments = fit_objective_family_score_alignments(
            oof_rows, final_rows, "top2_ordering"
        )

        mapping = alignments["family_a"]["fields"]["delta_r2"]
        self.assertLess(mapping["mae_after"], mapping["mae_before"])
        self.assertAlmostEqual(mapping["target_knots"][-1], 1.0, places=6)

    def test_runtime_applies_family_score_alignment_before_admission(self):
        optimizer = CompleteEvidencePlanPortfolioOptimizer("")
        optimizer.model = {
            "grouped_oof_score_alignment": {
                "enabled": True,
                "alignments": {
                    "top2_ordering": {
                        "family_a": {
                            "fields": {
                                "utility_mean": {
                                    "source_knots": [0.0, 1.0],
                                    "target_knots": [0.0, 0.5],
                                }
                            }
                        }
                    }
                },
            }
        }

        aligned = optimizer._align_prediction_to_oof(
            {"utility_mean": 0.8}, "top2_ordering", "family_a"
        )

        self.assertAlmostEqual(aligned["utility_mean"], 0.4)
        self.assertAlmostEqual(aligned["raw_utility_mean"], 0.8)
        self.assertTrue(aligned["score_alignment_applied"])

    def test_stage2_labels_are_residual_to_frozen_oof_top2(self):
        top2_features = {name: 0.0 for name in base.FEATURE_NAMES}
        top5_features = dict(top2_features)
        top5_features["plan_top2_membership_invariant"] = 1.0
        stage1 = {
            "query_key": "dataset:0",
            "dataset": "dataset",
            "plan_name": base.PLAN_SIGNATURE_GROUNDED,
            "plan_instance": "top2:0",
            "features": top2_features,
            "doc_ids": [10, 2, 3, 4, 5],
            "delta_r2": 0.5,
            "delta_r5": 0.5,
        }
        completion = {
            "query_key": "dataset:0",
            "dataset": "dataset",
            "plan_name": "learned_residual_top5_completion",
            "plan_instance": "top5:0",
            "features": top5_features,
            "doc_ids": [1, 2, 3, 4, 30],
            "semantic_doc_ids": [1, 2, 3, 4, 5],
            "gold_doc_ids": [10, 30],
            "delta_r2": 0.0,
            "delta_r5": 0.5,
        }

        residual = build_stage2_residual_rows([completion], [stage1])

        self.assertEqual(len(residual), 1)
        self.assertEqual(residual[0]["doc_ids"], [10, 2, 3, 4, 30])
        self.assertEqual(residual[0]["doc_ids"][:2], [10, 2])
        self.assertEqual(residual[0]["delta_r2"], 0.0)
        self.assertEqual(residual[0]["delta_r5"], 0.5)
        self.assertEqual(residual[0]["loss_probability"], 0.0)

    def test_transient_fold_fit_skips_serialized_forests(self):
        rows = sample_rows()
        args = argparse.Namespace(
            allowed_plans=base.PLAN_SIGNATURE_GROUNDED,
            structural_expert_model=False,
            objective_class_outcome_model=False,
            global_outcome_model=False,
            trees=2,
            max_depth=2,
            min_leaf=1,
            max_features=1.0,
            seed=7,
            jobs=1,
            alpha=0.1,
            balance_outcome_targets=True,
            skip_model_export=True,
            skip_residual_calibration=True,
        )

        payloads, models = base.train_models(rows, rows, args)

        payload = payloads[base.PLAN_SIGNATURE_GROUNDED]
        self.assertFalse(any(key.endswith("_head") for key in payload))
        self.assertNotIn("delta_r2_residual_lcb", payload)
        predicted = base.attach_predictions(rows, payloads, models)
        self.assertEqual(len(predicted), len(rows))
        self.assertTrue(all("prediction" in row for row in predicted))

    def test_fold_checkpoint_round_trip_and_contract_guard(self):
        rows = sample_rows()[:4]
        predicted = []
        for index, row in enumerate(rows):
            value = dict(row)
            value["prediction"] = {
                "delta_r2": index / 10.0,
                "delta_r5": index / 20.0,
                "gain_probability": 0.8,
                "loss_probability": 0.1,
            }
            predicted.append(value)
        report = {"fold": 0, "heldout_action_count": len(rows)}

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fold_0.json.gz"
            write_fold_checkpoint(
                path,
                fingerprint="contract-a",
                fold=0,
                predicted=predicted,
                report=report,
            )
            restored, restored_report = load_fold_checkpoint(
                path,
                fingerprint="contract-a",
                fold=0,
                heldout_rows=rows,
            )

            self.assertEqual(restored_report, report)
            self.assertEqual(
                [row["prediction"] for row in restored],
                [row["prediction"] for row in predicted],
            )
            self.assertTrue(all(row["grouped_oof_fold"] == 0 for row in restored))
            with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                load_fold_checkpoint(
                    path,
                    fingerprint="contract-b",
                    fold=0,
                    heldout_rows=rows,
                )


if __name__ == "__main__":
    unittest.main()
