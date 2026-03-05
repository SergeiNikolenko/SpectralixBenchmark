import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pydantic import ValidationError

from scripts.evaluation.llm_judge import deterministic_score, run_llm_judge
from scripts.pydantic_guard.retry import run_with_retries
from scripts.pydantic_guard.schemas import JudgeResult, ParsedQuestionSchema, StudentGuardOutput
from scripts.pydantic_guard.student_guard import is_answer_invalid


def _write_single_judge_case(input_path: Path, gold_path: Path) -> None:
    student_row = {
        "exam_id": "exam_1",
        "page_id": "1",
        "question_id": "1",
        "question_type": "text",
        "question_text": "Explain result",
        "answer_type": "text",
        "student_answer": "demo",
        "student_status": "ok",
        "student_error": "",
        "student_elapsed_ms": 10,
    }
    gold_row = {
        "exam_id": "exam_1",
        "page_id": "1",
        "question_id": "1",
        "question_type": "text",
        "question_text": "Explain result",
        "answer_type": "text",
        "canonical_answer": "expected",
        "max_score": 5,
    }
    input_path.write_text(json.dumps(student_row) + "\n", encoding="utf-8")
    gold_path.write_text(json.dumps(gold_row) + "\n", encoding="utf-8")


def _read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class SchemaTests(unittest.TestCase):
    def test_judge_result_range_validation(self):
        with self.assertRaises(ValidationError):
            JudgeResult(llm_score=1.2, llm_comment="bad")

    def test_student_guard_output_requires_non_empty_answer(self):
        with self.assertRaises(ValidationError):
            StudentGuardOutput(final_answer="", format_ok=True)

    def test_parsed_question_requires_fields_for_ok_status(self):
        with self.assertRaises(ValidationError):
            ParsedQuestionSchema(status="ok", question_id=1, answer_type="text")

    def test_parsed_question_allows_error_status_minimal(self):
        item = ParsedQuestionSchema(status="error", question_id=1, error_comment="bad parse")
        self.assertEqual(item.status, "error")


class RetryTests(unittest.TestCase):
    def test_retry_succeeds_after_transient_failures(self):
        state = {"attempt": 0}

        def flaky():
            state["attempt"] += 1
            if state["attempt"] < 3:
                raise ValueError("retry")
            return "ok"

        result = run_with_retries(flaky, retries=3, retry_on=(ValueError,))
        self.assertEqual(result, "ok")
        self.assertEqual(state["attempt"], 3)


class StudentGuardHeuristicTests(unittest.TestCase):
    def test_is_answer_invalid(self):
        self.assertTrue(is_answer_invalid("single_choice", ""))
        self.assertTrue(is_answer_invalid("single_choice", "A B"))
        self.assertFalse(is_answer_invalid("single_choice", "A"))
        self.assertFalse(is_answer_invalid("ordering", "2; 1"))


class JudgeDeterministicTests(unittest.TestCase):
    def test_msms_string_match_is_case_insensitive(self):
        result = deterministic_score(
            answer_type="msms_structure_prediction",
            student_answer="CC(=O)Nc1ccccc1",
            canonical_answer="cc(=o)nc1ccccc1",
        )
        self.assertEqual(result["llm_score"], 1.0)


class JudgeStructuredTests(unittest.TestCase):
    def test_structured_quota_error_fails_fast(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            input_path = tmp / "student.jsonl"
            gold_path = tmp / "gold.jsonl"
            output_path = tmp / "judge.jsonl"
            _write_single_judge_case(input_path, gold_path)

            with mock.patch(
                "scripts.evaluation.llm_judge.run_structured_judge",
                side_effect=RuntimeError("insufficient_quota"),
            ):
                with self.assertRaisesRegex(RuntimeError, "limit exceeded"):
                    run_llm_judge(
                        input_path=input_path,
                        gold_path=gold_path,
                        output_path=output_path,
                        model_name="judge-model",
                        max_tokens=128,
                        temperature=0.0,
                        judge_structured_retries=1,
                        judge_api_key="test-key",
                    )

    def test_judge_appends_result_to_existing_trace(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            input_path = tmp / "student.jsonl"
            gold_path = tmp / "gold.jsonl"
            output_path = tmp / "judge.jsonl"
            trace_dir = tmp / "traces"
            trace_dir.mkdir(parents=True, exist_ok=True)
            trace_path = trace_dir / "0001_exam_1_p1_q1.log"
            trace_path.write_text("=== TRACE METADATA ===\n{}\n", encoding="utf-8")
            _write_single_judge_case(input_path, gold_path)

            with mock.patch(
                "scripts.evaluation.llm_judge.run_structured_judge",
                return_value={
                    "llm_score": 0.6,
                    "llm_comment": "structured ok",
                    "judge_request_id": "req_123",
                    "judge_latency_ms": 42,
                },
            ):
                run_llm_judge(
                    input_path=input_path,
                    gold_path=gold_path,
                    output_path=output_path,
                    model_name="judge-model",
                    max_tokens=128,
                    temperature=0.0,
                    judge_structured_retries=1,
                    judge_api_key="test-key",
                    trace_log_enabled=True,
                    trace_log_dir=trace_dir,
                )

            trace_content = trace_path.read_text(encoding="utf-8")
            self.assertIn("=== JUDGE RESULT ===", trace_content)
            self.assertIn("\"judge_mode\": \"llm_judge\"", trace_content)
            self.assertIn("structured ok", trace_content)


if __name__ == "__main__":
    unittest.main()
