import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from scripts.evaluation import student_validation as student_validation_module
from scripts.evaluation import run_full_matrix as run_full_matrix_module

EXPECTED_STUDENT_OUTPUT_KEYS = {
    "exam_id",
    "page_id",
    "question_id",
    "question_type",
    "question_text",
    "answer_type",
    "student_answer",
    "student_status",
    "student_error",
    "student_elapsed_ms",
}


class _FakeAgentRuntime:
    preflight_calls = 0
    solve_calls = 0

    def __init__(self, *args, **kwargs):
        _ = args
        _ = kwargs

    def preflight(self):
        type(self).preflight_calls += 1
        return None

    def solve_question(self, question):
        type(self).solve_calls += 1
        answer_type = str(question.get("answer_type") or "")
        if answer_type == "single_choice":
            return "Answer: A"
        if answer_type == "ordering":
            return "Answer: 2; 1"
        return "Answer: demo"

    def close(self):
        return None


def _write_benchmark(path: Path) -> None:
    rows = [
        {
            "exam_id": "exam_1",
            "page_id": "1",
            "question_id": "1",
            "question_type": "text",
            "question_text": "Q1",
            "answer_type": "single_choice",
            "canonical_answer": "A",
            "max_score": 1,
        },
        {
            "exam_id": "exam_1",
            "page_id": "1",
            "question_id": "2",
            "question_type": "text",
            "question_text": "Q2",
            "answer_type": "ordering",
            "canonical_answer": "2; 1",
            "max_score": 1,
        },
    ]
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class StudentValidationContractTests(unittest.TestCase):
    def test_student_output_schema_stable(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            benchmark_path = tmp_path / "benchmark.jsonl"
            output_path = tmp_path / "student_output.jsonl"
            _write_benchmark(benchmark_path)

            with mock.patch.object(student_validation_module, "AgentRuntime", _FakeAgentRuntime):
                _FakeAgentRuntime.preflight_calls = 0
                _FakeAgentRuntime.solve_calls = 0
                student_validation_module.run_benchmark_inference(
                    benchmark_path=benchmark_path,
                    output_path=output_path,
                    model_url="http://127.0.0.1:8317/v1",
                    model_name="test-model",
                    api_key="test-key",
                    max_tokens=128,
                    temperature=0.2,
                    timeout=15,
                    max_retries=1,
                )

            rows = _read_jsonl(output_path)
            self.assertEqual(len(rows), 2)
            self.assertEqual(_FakeAgentRuntime.preflight_calls, 1)
            self.assertEqual(_FakeAgentRuntime.solve_calls, 2)
            for row in rows:
                self.assertEqual(set(row.keys()), EXPECTED_STUDENT_OUTPUT_KEYS)
                self.assertEqual(row["student_status"], "ok")

            trace_files = sorted((tmp_path / "traces").glob("*.log"))
            self.assertEqual(len(trace_files), 2)
            trace_sample = trace_files[0].read_text(encoding="utf-8")
            self.assertIn("=== AGENT STDOUT/STDERR TRACE ===", trace_sample)
            self.assertIn("=== RAW MODEL ANSWER ===", trace_sample)


class RunFullMatrixContractTests(unittest.TestCase):
    def test_run_full_matrix_artifacts_and_contract(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            benchmark_path = tmp_path / "benchmark.jsonl"
            output_root = tmp_path / "runs"
            _write_benchmark(benchmark_path)

            def fake_run_benchmark_inference(**kwargs):
                output_path = Path(kwargs["output_path"])
                output_path.parent.mkdir(parents=True, exist_ok=True)
                rows = [
                    {
                        "exam_id": "exam_1",
                        "page_id": "1",
                        "question_id": "1",
                        "question_type": "text",
                        "question_text": "Q1",
                        "answer_type": "single_choice",
                        "student_answer": "A",
                        "student_status": "ok",
                        "student_error": "",
                        "student_elapsed_ms": 10,
                    },
                    {
                        "exam_id": "exam_1",
                        "page_id": "1",
                        "question_id": "2",
                        "question_type": "text",
                        "question_text": "Q2",
                        "answer_type": "ordering",
                        "student_answer": "2; 1",
                        "student_status": "ok",
                        "student_error": "",
                        "student_elapsed_ms": 12,
                    },
                ]
                with output_path.open("w", encoding="utf-8") as f:
                    for row in rows:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")

            def fake_run_llm_judge(
                input_path,
                gold_path,
                output_path,
                model_name,
                max_tokens,
                temperature,
                **kwargs,
            ):
                _ = gold_path
                _ = model_name
                _ = max_tokens
                _ = temperature
                _ = kwargs
                input_rows = _read_jsonl(Path(input_path))
                with Path(output_path).open("w", encoding="utf-8") as f:
                    for row in input_rows:
                        payload = dict(row)
                        payload.update(
                            {
                                "row_status": "ok",
                                "final_score": 1.0,
                                "max_score": 1.0,
                            }
                        )
                        f.write(json.dumps(payload, ensure_ascii=False) + "\n")

            fake_llm_judge_module = types.ModuleType("llm_judge")
            fake_llm_judge_module.run_llm_judge = fake_run_llm_judge

            with mock.patch.object(
                run_full_matrix_module,
                "run_benchmark_inference",
                side_effect=fake_run_benchmark_inference,
            ):
                with mock.patch.dict(sys.modules, {"llm_judge": fake_llm_judge_module}):
                    with mock.patch.object(
                        sys,
                        "argv",
                        [
                            "run_full_matrix.py",
                            "--benchmark-path",
                            str(benchmark_path),
                            "--output-root",
                            str(output_root),
                            "--run-id",
                            "test_run",
                            "--model-url",
                            "http://127.0.0.1:8317/v1",
                            "--models",
                            "test-model",
                        ],
                    ):
                        run_full_matrix_module.main()

            run_dir = output_root / "test_run" / "test-model"
            student_output_path = run_dir / "student_output.jsonl"
            judge_output_path = run_dir / "llm_judge_output.jsonl"
            summary_json_path = output_root / "test_run" / "summary.json"
            summary_csv_path = output_root / "test_run" / "summary.csv"

            self.assertTrue(student_output_path.exists())
            self.assertTrue(judge_output_path.exists())
            self.assertTrue(summary_json_path.exists())
            self.assertTrue(summary_csv_path.exists())

            student_rows = _read_jsonl(student_output_path)
            self.assertEqual(len(student_rows), 2)
            for row in student_rows:
                self.assertEqual(set(row.keys()), EXPECTED_STUDENT_OUTPUT_KEYS)

            judge_rows = _read_jsonl(judge_output_path)
            self.assertEqual(len(judge_rows), 2)
            for row in judge_rows:
                self.assertIn("row_status", row)
                self.assertIn("final_score", row)
                self.assertIn("max_score", row)
                self.assertIn("student_status", row)

            summary_rows = json.loads(summary_json_path.read_text(encoding="utf-8"))
            self.assertEqual(len(summary_rows), 1)
            self.assertEqual(summary_rows[0]["model_name"], "test-model")


if __name__ == "__main__":
    unittest.main()
