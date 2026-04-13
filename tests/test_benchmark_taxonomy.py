import json
import tempfile
import unittest
from pathlib import Path

from scripts.evaluation.migrate_run_artifacts import migrate_run_dir
from scripts.evaluation.materialize_benchmark_v3_eval import _to_contract_row
from scripts.evaluation.benchmark_taxonomy import (
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

    def test_migrate_run_dir_writes_taxonomy_artifacts(self):
        row = {
            "exam_id": "benchmark_v3_a",
            "page_id": "source_eval",
            "question_id": "row-1",
            "level": "A",
            "task_subtype": "mechanistic_classification",
            "difficulty": "medium",
            "answer_type": "reaction_description",
            "max_score": 2,
            "final_score": 1.2,
            "student_status": "ok",
            "row_status": "ok",
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "run_1"
            model_dir = run_dir / "demo-model"
            model_dir.mkdir(parents=True)
            with (model_dir / "llm_judge_output.jsonl").open("w", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            with (model_dir / "student_output.jsonl").open("w", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            with (model_dir / "student_output_verbose.jsonl").open("w", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            write_json = {
                "verbose_output_path": str(model_dir / "student_output_verbose.jsonl"),
            }
            (model_dir / "run_manifest.json").write_text(
                json.dumps(write_json, ensure_ascii=False),
                encoding="utf-8",
            )

            summary_rows = migrate_run_dir(run_dir)

            self.assertEqual(len(summary_rows), 1)
            self.assertTrue((model_dir / "metrics.json").exists())
            self.assertTrue((model_dir / "breakdown_by_suite.json").exists())
            self.assertTrue((model_dir / "breakdown_by_subtrack.json").exists())
            self.assertTrue((model_dir / "breakdown_by_task_subtype.json").exists())
            self.assertTrue((run_dir / "summary.json").exists())
            enriched_rows = [
                json.loads(line)
                for line in (model_dir / "llm_judge_output.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            enriched_student_rows = [
                json.loads(line)
                for line in (model_dir / "student_output.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(enriched_rows[0]["benchmark_suite"], "A")
            self.assertEqual(enriched_rows[0]["benchmark_subtrack"], "A2")
            self.assertEqual(enriched_student_rows[0]["benchmark_suite"], "A")
            self.assertEqual(enriched_student_rows[0]["benchmark_subtrack"], "A2")

    def test_migrate_run_dir_drops_stale_verbose_output_path(self):
        row = {
            "exam_id": "benchmark_v3_a",
            "page_id": "source_eval",
            "question_id": "row-1",
            "level": "A",
            "task_subtype": "mechanistic_classification",
            "difficulty": "medium",
            "answer_type": "reaction_description",
            "max_score": 2,
            "final_score": 1.2,
            "student_status": "ok",
            "row_status": "ok",
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "run_1"
            model_dir = run_dir / "demo-model"
            model_dir.mkdir(parents=True)
            with (model_dir / "llm_judge_output.jsonl").open("w", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            (model_dir / "run_manifest.json").write_text(
                json.dumps({"verbose_output_path": str(model_dir / "missing_verbose.jsonl")}),
                encoding="utf-8",
            )

            migrate_run_dir(run_dir)

            manifest = json.loads((model_dir / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertNotIn("verbose_output_path", manifest)


class TaxonomyMetadataTests(unittest.TestCase):
    def test_unknown_subtype_falls_back_to_generic_level(self):
        metadata = get_benchmark_taxonomy_metadata({"level": "B", "task_subtype": "unknown"})
        self.assertEqual(metadata["benchmark_suite"], "B")
        self.assertEqual(metadata["benchmark_subtrack"], "B0")
        self.assertEqual(metadata["planning_horizon"], "single_step")
