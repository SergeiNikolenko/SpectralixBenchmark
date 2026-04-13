import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.evaluation import student_validation as student_validation_module
from scripts.evaluation import run_full_matrix as run_full_matrix_module

EXPECTED_STUDENT_OUTPUT_KEYS = {
    "exam_id",
    "page_id",
    "question_id",
    "level",
    "task_subtype",
    "difficulty",
    "benchmark_suite",
    "benchmark_subtrack",
    "planning_horizon",
    "task_mode",
    "eval_contract_id",
    "expected_output_schema",
    "judge_rubric_id",
    "question_type",
    "question_text",
    "answer_type",
    "student_answer",
    "student_status",
    "student_error",
    "student_elapsed_ms",
    "student_input_tokens",
    "student_output_tokens",
    "student_total_tokens",
    "student_reasoning_tokens",
}
EXPECTED_TRACE_SECTIONS = (
    "=== REASONING SUMMARY ===",
    "=== STEP SUMMARY ===",
    "=== SGR SCHEMA ===",
    "=== SGR CONTRACT CHECK ===",
    "=== SGR FINAL ANSWER CANDIDATE ===",
    "=== AGENT STDOUT/STDERR TRACE ===",
    "=== AGENT RUN DETAILS ===",
    "=== RAW MODEL ANSWER ===",
)


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

    def get_last_run_details(self):
        return {
            "state": "success",
            "steps": [],
            "sgr_schema_name": "sgr.level_a.reaction_center_identification",
            "sgr_validation_status": "valid",
            "sgr_repair_attempted": False,
            "sgr_fallback_used": False,
            "sgr_payload": {
                "level": "A",
                "task_subtype": "reaction_center_identification",
                "contract_check": {
                    "answer_matches_requested_task": True,
                    "answer_matches_requested_depth": True,
                    "answer_matches_exact_benchmark_contract": True,
                    "broader_or_alternative_answer_leak": False,
                },
                "final_answer": {"value": "A"},
            },
        }

    def get_runtime_metadata(self):
        return {"executor_type": "local", "sandbox_runtime": "local_worker"}


class _QuotaAgentRuntime:
    def __init__(self, *args, **kwargs):
        _ = args
        _ = kwargs

    def preflight(self):
        return None

    def solve_question(self, question):
        _ = question
        raise student_validation_module.AgentRuntimeError(
            status="http_error",
            message="Error code: 429 insufficient_quota",
        )

    def close(self):
        return None

    def get_last_run_details(self):
        return {"state": "error", "steps": []}

    def get_runtime_metadata(self):
        return {"executor_type": "local", "sandbox_runtime": "local_worker"}


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


def _run_student_inference(benchmark_path: Path, output_path: Path, **kwargs):
    params = {
        "benchmark_path": benchmark_path,
        "output_path": output_path,
        "model_url": "http://127.0.0.1:8317/v1",
        "model_name": "test-model",
        "api_key": "test-key",
        "timeout": 15,
        "max_retries": 1,
    }
    params.update(kwargs)
    student_validation_module.run_benchmark_inference(**params)


def _assert_student_output_row_contract(test_case: unittest.TestCase, row: dict) -> None:
    test_case.assertEqual(set(row.keys()), EXPECTED_STUDENT_OUTPUT_KEYS)


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
                _run_student_inference(benchmark_path, output_path)

            rows = _read_jsonl(output_path)
            self.assertEqual(len(rows), 2)
            self.assertEqual(_FakeAgentRuntime.preflight_calls, 1)
            self.assertEqual(_FakeAgentRuntime.solve_calls, 2)
            for row in rows:
                _assert_student_output_row_contract(self, row)
                self.assertEqual(row["student_status"], "ok")

            trace_files = sorted((tmp_path / "traces").glob("*.log"))
            self.assertEqual(len(trace_files), 2)
            trace_sample = trace_files[0].read_text(encoding="utf-8")
            for marker in EXPECTED_TRACE_SECTIONS:
                self.assertIn(marker, trace_sample)
            self.assertIn("sgr.level_a.reaction_center_identification", trace_sample)
            self.assertIn("answer_matches_exact_benchmark_contract", trace_sample)

    def test_student_verbose_output_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            benchmark_path = tmp_path / "benchmark.jsonl"
            output_path = tmp_path / "student_output.jsonl"
            verbose_output_path = tmp_path / "student_output_verbose.jsonl"
            _write_benchmark(benchmark_path)

            with mock.patch.object(student_validation_module, "AgentRuntime", _FakeAgentRuntime):
                _run_student_inference(
                    benchmark_path,
                    output_path,
                    verbose_output_enabled=True,
                    verbose_output_path=verbose_output_path,
                    trace_log_enabled=False,
                )

            self.assertTrue(verbose_output_path.exists())
            verbose_rows = _read_jsonl(verbose_output_path)
            self.assertEqual(len(verbose_rows), 2)
            for row in verbose_rows:
                self.assertIn("raw_answer", row)
                self.assertIn("reasoning_summary", row)
                self.assertIn("agent_run_details", row)
                self.assertIn("trace_log_path", row)
                self.assertIn("sgr_schema_name", row)
                self.assertIn("sgr_snapshot", row)
                self.assertEqual(row["sgr_schema_name"], "sgr.level_a.reaction_center_identification")
                self.assertIsInstance(row["sgr_snapshot"], dict)
                self.assertIn("contract_check", row["sgr_snapshot"])

    def test_student_run_fails_fast_on_model_limit(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            benchmark_path = tmp_path / "benchmark.jsonl"
            output_path = tmp_path / "student_output.jsonl"
            _write_benchmark(benchmark_path)

            with mock.patch.object(student_validation_module, "AgentRuntime", _QuotaAgentRuntime):
                with self.assertRaises(student_validation_module.ModelLimitExceededError):
                    _run_student_inference(benchmark_path, output_path)


class RunFullMatrixContractTests(unittest.TestCase):
    def test_run_full_matrix_artifacts_and_contract(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            benchmark_path = tmp_path / "benchmark.jsonl"
            output_root = tmp_path / "runs"
            _write_benchmark(benchmark_path)
            inference_kwargs_seen = {}

            def fake_run_benchmark_inference(**kwargs):
                inference_kwargs_seen.update(kwargs)
                output_path = Path(kwargs["output_path"])
                output_path.parent.mkdir(parents=True, exist_ok=True)
                rows = [
                    {
                        "exam_id": "exam_1",
                        "page_id": "1",
                        "question_id": "1",
                        "level": None,
                        "task_subtype": None,
                        "difficulty": None,
                        "benchmark_suite": None,
                        "benchmark_subtrack": None,
                        "planning_horizon": None,
                        "task_mode": None,
                        "eval_contract_id": None,
                        "expected_output_schema": None,
                        "judge_rubric_id": None,
                        "question_type": "text",
                        "question_text": "Q1",
                        "answer_type": "single_choice",
                        "student_answer": "A",
                        "student_status": "ok",
                        "student_error": "",
                        "student_elapsed_ms": 10,
                        "student_input_tokens": 0,
                        "student_output_tokens": 0,
                        "student_total_tokens": 0,
                        "student_reasoning_tokens": 0,
                    },
                    {
                        "exam_id": "exam_1",
                        "page_id": "1",
                        "question_id": "2",
                        "level": None,
                        "task_subtype": None,
                        "difficulty": None,
                        "benchmark_suite": None,
                        "benchmark_subtrack": None,
                        "planning_horizon": None,
                        "task_mode": None,
                        "eval_contract_id": None,
                        "expected_output_schema": None,
                        "judge_rubric_id": None,
                        "question_type": "text",
                        "question_text": "Q2",
                        "answer_type": "ordering",
                        "student_answer": "2; 1",
                        "student_status": "ok",
                        "student_error": "",
                        "student_elapsed_ms": 12,
                        "student_input_tokens": 0,
                        "student_output_tokens": 0,
                        "student_total_tokens": 0,
                        "student_reasoning_tokens": 0,
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

            with mock.patch.object(
                run_full_matrix_module,
                "run_benchmark_inference",
                side_effect=fake_run_benchmark_inference,
            ):
                with mock.patch.object(
                    run_full_matrix_module,
                    "run_llm_judge",
                    side_effect=fake_run_llm_judge,
                ):
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
                            "--agent-backend",
                            "codex_native",
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
                _assert_student_output_row_contract(self, row)

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
            self.assertEqual(inference_kwargs_seen["agent_backend"], "codex_native")


if __name__ == "__main__":
    unittest.main()
