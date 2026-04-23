import unittest

from spectralix_benchmark.evaluation.materialize_benchmark_v3_eval import _to_contract_row
from spectralix_benchmark.evaluation.benchmark_taxonomy import (
    compute_benchmark_taxonomy_metrics,
    get_benchmark_taxonomy_metadata,
)


class BenchmarkTaxonomyTests(unittest.TestCase):
    def test_materialized_contract_row_includes_benchmark_taxonomy_fields(self):
        source_row = {
            "level": "A",
            "task_family": "Reaction Understanding",
            "task_subtype": "reagent_role_identification",
            "difficulty": "medium",
            "coverage_tags": ["reagent_roles", "procedural_text"],
            "input": {"question_text": "Assign reagent roles."},
            "input_text": "Assign reagent roles.",
            "gold": {"reference_answer": {"base": "Et3N"}},
            "source_id": "chemu_2020",
            "source_split": "eval",
            "source_license": "test",
            "record_id": "row-1",
            "metadata": {"seed": 1},
        }
        row = _to_contract_row(source_row)
        self.assertEqual(row["benchmark_suite"], "G")
        self.assertEqual(row["benchmark_subtrack"], "G1")
        self.assertEqual(row["planning_horizon"], "auxiliary")
        self.assertEqual(row["task_mode"], "grounding_extraction")
        self.assertEqual(row["judge_rubric_id"], "g_eval.reaction_description.v1")
        self.assertEqual(row["expected_output_schema"], "role_mapping")
        self.assertIn('"benchmark_suite": "G"', row["question_text"])
        self.assertIn('"judge_rubric_id": "g_eval.reaction_description.v1"', row["question_text"])

    def test_compute_benchmark_taxonomy_metrics_uses_suite_breakdowns_and_macro_depth(self):
        rows = [
            {
                "level": "A",
                "task_subtype": "reaction_center_identification",
                "difficulty": "hard",
                "answer_type": "reaction_description",
                "max_score": 2,
                "final_score": 1.0,
                "student_status": "ok",
                "row_status": "ok",
            },
            {
                "level": "A",
                "task_subtype": "reagent_role_identification",
                "difficulty": "medium",
                "answer_type": "reaction_description",
                "max_score": 2,
                "final_score": 1.5,
                "student_status": "ok",
                "row_status": "ok",
            },
            {
                "level": "B",
                "task_subtype": "immediate_precursor_prediction",
                "difficulty": "easy",
                "answer_type": "text",
                "max_score": 2,
                "final_score": 0.5,
                "student_status": "ok",
                "row_status": "ok",
            },
            {
                "level": "C",
                "task_subtype": "reference_route_planning",
                "difficulty": "hard",
                "answer_type": "full_synthesis",
                "max_score": 4,
                "final_score": None,
                "student_status": "agent_step_error",
                "row_status": "technical_skip",
            },
        ]
        metrics = compute_benchmark_taxonomy_metrics(rows)
        self.assertAlmostEqual(metrics["breakdown_by_suite"]["A"]["quality_score"], 0.5)
        self.assertAlmostEqual(metrics["breakdown_by_suite"]["B"]["quality_score"], 0.25)
        self.assertAlmostEqual(metrics["breakdown_by_suite"]["G"]["quality_score"], 0.75)
        self.assertAlmostEqual(metrics["breakdown_by_suite"]["C"]["end_to_end_score"], 0.0)
        self.assertAlmostEqual(
            metrics["breakdown_by_task_subtype"]["reaction_center_identification"][
                "quality_score"
            ],
            0.5,
        )
        self.assertAlmostEqual(
            metrics["breakdown_by_task_subtype"]["reagent_role_identification"][
                "quality_score"
            ],
            0.75,
        )
        self.assertIsNone(metrics["macro_depth_quality_score"])
        self.assertAlmostEqual(metrics["macro_depth_end_to_end_score"], (0.5 + 0.25 + 0.0) / 3)
        self.assertEqual(metrics["auxiliary_grounding_quality_score"], 0.75)

class TaxonomyMetadataTests(unittest.TestCase):
    def test_unknown_subtype_falls_back_to_generic_level(self):
        metadata = get_benchmark_taxonomy_metadata({"level": "B", "task_subtype": "unknown"})
        self.assertEqual(metadata["benchmark_suite"], "B")
        self.assertEqual(metadata["benchmark_subtrack"], "B0")
        self.assertEqual(metadata["planning_horizon"], "single_step")
        self.assertEqual(metadata["eval_contract_id"], "level_b.plausible_immediate_retrosynthesis.v2")
        self.assertEqual(metadata["judge_rubric_id"], "g_eval.level_b.plausible_immediate_retrosynthesis.v2")

    def test_level_b_disconnection_metadata_uses_plausibility_contract(self):
        metadata = get_benchmark_taxonomy_metadata(
            {"level": "B", "task_subtype": "immediate_precursor_with_disconnection"}
        )
        self.assertEqual(metadata["benchmark_subtrack"], "B2")
        self.assertEqual(metadata["eval_contract_id"], "level_b.plausible_immediate_disconnection.v2")
        self.assertEqual(
            metadata["expected_output_schema"],
            "plausible_immediate_precursor_set_with_disconnection",
        )
        self.assertEqual(metadata["judge_rubric_id"], "g_eval.level_b.plausible_immediate_disconnection.v2")
