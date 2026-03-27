import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from scripts.evaluation.judge_rubrics import get_g_eval_spec
from scripts.pydantic_guard.judge_geval import run_g_eval_judge
from scripts.pydantic_guard.judge_structured import run_structured_judge

DETERMINISTIC_TYPES = {
    "single_choice",
    "multiple_choice",
    "ordering",
    "numeric",
    "msms_structure_prediction",
}
TRUTHY_VALUES = {"1", "true", "yes", "y", "on"}
FALSY_VALUES = {"0", "false", "no", "n", "off"}
MODEL_LIMIT_ERROR_MARKERS = (
    "429",
    "insufficient_quota",
    "quota exceeded",
    "quota_exceeded",
    "exceeded your current quota",
    "billing hard limit",
    "billing_limit_reached",
    "out of credits",
    "credits exhausted",
    "monthly usage limit",
    "rate limit exceeded permanently",
    "rate limit exceeded",
    "model_cooldown",
    "cooling down",
    "all credentials for model",
)


class ModelLimitExceededError(RuntimeError):
    pass


def normalize_answer_type(answer_type: Optional[str]) -> str:
    if not answer_type:
        return ""
    normalized = answer_type.strip().lower()
    if normalized == "single_choise":
        return "single_choice"
    return normalized


def build_key(item: Dict[str, Any]) -> str:
    return f"{item.get('exam_id')}/{item.get('page_id')}/{item.get('question_id')}"


def _clean_text(value: Any) -> str:
    text = str(value or "").strip()
    text = text.replace("```", "")
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"(?im)^\s*answer\s*:\s*", "", text)
    return text.strip()


def _tokenize_sequence(value: Any) -> List[str]:
    text = _clean_text(value)
    text = text.replace("\n", ";").replace(",", ";").replace("|", ";")
    tokens: List[str] = []
    for part in text.split(";"):
        token = part.strip().lower()
        if not token:
            continue
        token = re.sub(r"^\d+[\.)]\s*", "", token)
        if token:
            tokens.append(token)
    return tokens


def _first_token(value: Any) -> str:
    text = _clean_text(value)
    tokens = re.findall(r"[A-Za-z0-9]+", text)
    if not tokens:
        return ""
    return tokens[0].lower()


def _parse_float(value: Any) -> Optional[float]:
    text = _clean_text(value)
    if not text:
        return None
    matches = re.findall(r"-?\d+(?:\.\d+)?", text)
    if not matches:
        return None
    try:
        return float(matches[0])
    except ValueError:
        return None


def _parse_range(value: Any) -> Optional[Tuple[float, float]]:
    text = _clean_text(value)
    match = re.search(r"(-?\d+(?:\.\d+)?)\s*[-–]\s*(-?\d+(?:\.\d+)?)", text)
    if not match:
        return None
    low = float(match.group(1))
    high = float(match.group(2))
    if low > high:
        low, high = high, low
    return (low, high)


def deterministic_score(answer_type: str, student_answer: Any, canonical_answer: Any) -> Dict[str, Any]:
    normalized_type = normalize_answer_type(answer_type)

    if normalized_type == "single_choice":
        student_token = _first_token(student_answer)
        canonical_token = _first_token(canonical_answer)
        if not student_token:
            return {"llm_score": 0.0, "llm_comment": "No parsable single-choice answer"}
        score = 1.0 if student_token == canonical_token else 0.0
        return {
            "llm_score": score,
            "llm_comment": f"single_choice strict match: student={student_token}, canonical={canonical_token}",
        }

    if normalized_type == "multiple_choice":
        student_tokens = _tokenize_sequence(student_answer)
        canonical_tokens = _tokenize_sequence(canonical_answer)
        if not canonical_tokens:
            return {"llm_score": 0.0, "llm_comment": "Canonical answer has no parsable options"}
        if not student_tokens:
            return {"llm_score": 0.0, "llm_comment": "No parsable multiple-choice answer"}

        student_set = set(student_tokens)
        canonical_set = set(canonical_tokens)
        intersection = len(student_set & canonical_set)
        precision = intersection / len(student_set)
        recall = intersection / len(canonical_set)
        if precision + recall == 0:
            score = 0.0
        else:
            score = 2 * precision * recall / (precision + recall)

        return {
            "llm_score": float(round(score, 6)),
            "llm_comment": (
                f"multiple_choice deterministic F1: matched={intersection}, "
                f"student={sorted(student_set)}, canonical={sorted(canonical_set)}"
            ),
        }

    if normalized_type == "ordering":
        student_tokens = _tokenize_sequence(student_answer)
        canonical_tokens = _tokenize_sequence(canonical_answer)
        if not canonical_tokens:
            return {"llm_score": 0.0, "llm_comment": "Canonical answer has no parsable ordering"}
        if not student_tokens:
            return {"llm_score": 0.0, "llm_comment": "No parsable ordering answer"}

        matches = 0
        for idx, canonical_token in enumerate(canonical_tokens):
            if idx < len(student_tokens) and student_tokens[idx] == canonical_token:
                matches += 1
        score = matches / len(canonical_tokens)
        return {
            "llm_score": float(round(score, 6)),
            "llm_comment": f"ordering positional match: {matches}/{len(canonical_tokens)}",
        }

    if normalized_type == "numeric":
        student_value = _parse_float(student_answer)
        if student_value is None:
            return {"llm_score": 0.0, "llm_comment": "No parsable numeric answer"}

        canonical_range = _parse_range(canonical_answer)
        if canonical_range is not None:
            low, high = canonical_range
            score = 1.0 if low <= student_value <= high else 0.0
            return {
                "llm_score": score,
                "llm_comment": f"numeric range check: student={student_value}, range=[{low}, {high}]",
            }

        canonical_value = _parse_float(canonical_answer)
        if canonical_value is None:
            return {"llm_score": 0.0, "llm_comment": "Canonical answer has no parsable numeric value"}

        tolerance = max(abs(canonical_value) * 0.01, 1e-6)
        score = 1.0 if abs(student_value - canonical_value) <= tolerance else 0.0
        return {
            "llm_score": score,
            "llm_comment": (
                f"numeric tolerance check: student={student_value}, canonical={canonical_value}, "
                f"tolerance={tolerance}"
            ),
        }

    if normalized_type == "msms_structure_prediction":
        student_clean = _clean_text(student_answer).replace(" ", "").lower()
        canonical_candidates = [
            token.replace(" ", "") for token in _tokenize_sequence(canonical_answer)
        ]
        if not canonical_candidates:
            canonical_candidates = [_clean_text(canonical_answer).replace(" ", "").lower()]

        if not student_clean:
            return {"llm_score": 0.0, "llm_comment": "No parsable SMILES answer"}

        score = 1.0 if student_clean in canonical_candidates else 0.0
        return {
            "llm_score": score,
            "llm_comment": "msms_structure_prediction strict string match",
        }

    raise ValueError(f"Unsupported deterministic answer type: {answer_type}")


def build_user_prompt(item: Dict[str, Any]) -> str:
    return f"""
<task>
Grade the student answer against the canonical answer for a chemistry exam question.
</task>

<scoring_rules>
- Score in the range [0.0, 1.0].
- Use 1.0 only when the student answer is fully correct for the requested answer_type.
- Use 0.0 when the answer is incorrect, incompatible with the question, or effectively missing.
- Use intermediate scores only for meaningful partial correctness.
- Penalize chemistry mistakes more than wording differences.
- Prefer semantic correctness over style.
</scoring_rules>

<output_contract>
Return strict structured output only.
Keep the comment concise, factual, and evidence-based.
</output_contract>

<question_context>
Question type: {item.get("question_type")}
Answer type: {item.get("answer_type")}
</question_context>

<question>
{item.get("question_text")}
</question>

<canonical_answer>
{item.get("canonical_answer")}
</canonical_answer>

<student_answer>
{item.get("student_answer")}
</student_answer>
""".strip()


def build_g_eval_prompt(item: Dict[str, Any]) -> str:
    spec = get_g_eval_spec(item.get("answer_type"))
    criteria = "\n".join(f"- {entry}" for entry in spec["criteria"])
    evaluation_steps = "\n".join(f"{idx}. {entry}" for idx, entry in enumerate(spec["evaluation_steps"], start=1))
    rubric = "\n".join(f"- {entry}" for entry in spec["rubric"])
    return f"""
<task>
Evaluate the student answer using rubric-guided chemistry grading.
</task>

<criteria>
{criteria}
</criteria>

<evaluation_steps>
{evaluation_steps}
</evaluation_steps>

<rubric>
{rubric}
</rubric>

<output_contract>
Return strict structured output only.
Provide concrete step findings and one final rubric score from 0 to 10.
</output_contract>

<question_context>
Question type: {item.get("question_type")}
Answer type: {item.get("answer_type")}
</question_context>

<question>
{item.get("question_text")}
</question>

<canonical_answer>
{item.get("canonical_answer")}
</canonical_answer>

<student_answer>
{item.get("student_answer")}
</student_answer>
""".strip()


def _is_model_limit_error(error_message: str) -> bool:
    text = str(error_message or "").strip().lower()
    if not text:
        return False
    return any(marker in text for marker in MODEL_LIMIT_ERROR_MARKERS)


def _raise_model_limit_exceeded(
    *,
    exc: Exception,
    model_name: str,
    key: str,
    prefix: str,
) -> None:
    if not _is_model_limit_error(str(exc)):
        return
    raise ModelLimitExceededError(
        f"{prefix}model={model_name}, key={key}, reason={str(exc)[:240]}"
    ) from exc


def _build_llm_success_output(
    *,
    judge_input: Dict[str, Any],
    judge_result: Dict[str, Any],
    max_score: float,
    model_name: str,
    score_method: str = "llm_judge",
) -> Dict[str, Any]:
    return {
        **judge_input,
        "llm_score": judge_result["llm_score"],
        "llm_comment": judge_result["llm_comment"],
        "final_score": max_score * float(judge_result["llm_score"]),
        "max_score": max_score,
        "score_method": score_method,
        "judge_model": model_name,
        "judge_request_id": judge_result.get("judge_request_id"),
        "judge_latency_ms": judge_result.get("judge_latency_ms"),
        "judge_input_tokens": judge_result.get("judge_input_tokens"),
        "judge_output_tokens": judge_result.get("judge_output_tokens"),
        "judge_total_tokens": judge_result.get("judge_total_tokens"),
        "judge_reasoning_tokens": judge_result.get("judge_reasoning_tokens"),
        "judge_requests": judge_result.get("judge_requests"),
        "judge_tool_calls": judge_result.get("judge_tool_calls"),
        "judge_usage_details": judge_result.get("judge_usage_details"),
        "row_status": "ok",
    }


def _build_missing_canonical_output(student: Dict[str, Any]) -> Dict[str, Any]:
    return {
        **student,
        "canonical_answer": None,
        "llm_score": None,
        "llm_comment": "Judging skipped: canonical answer not found",
        "final_score": None,
        "max_score": 0.0,
        "score_method": None,
        "row_status": "judge_error",
    }


def _build_technical_skip_output(
    *,
    judge_input: Dict[str, Any],
    student_status: str,
    max_score: float,
) -> Dict[str, Any]:
    return {
        **judge_input,
        "llm_score": None,
        "llm_comment": f"Technical skip: student_status={student_status}",
        "final_score": None,
        "max_score": max_score,
        "score_method": None,
        "row_status": student_status,
    }


def _build_deterministic_output(
    *,
    judge_input: Dict[str, Any],
    judge_result: Dict[str, Any],
    max_score: float,
) -> Dict[str, Any]:
    return {
        **judge_input,
        "llm_score": judge_result["llm_score"],
        "llm_comment": judge_result["llm_comment"],
        "final_score": max_score * float(judge_result["llm_score"]),
        "max_score": max_score,
        "score_method": "deterministic",
        "row_status": "ok",
    }


def _build_llm_error_output(
    *,
    judge_input: Dict[str, Any],
    max_score: float,
    model_name: str,
    llm_comment: str,
) -> Dict[str, Any]:
    return {
        **judge_input,
        "llm_score": None,
        "llm_comment": llm_comment,
        "final_score": None,
        "max_score": max_score,
        "score_method": "llm_judge",
        "judge_model": model_name,
        "judge_input_tokens": None,
        "judge_output_tokens": None,
        "judge_total_tokens": None,
        "judge_reasoning_tokens": None,
        "judge_requests": None,
        "judge_tool_calls": None,
        "judge_usage_details": None,
        "row_status": "judge_error",
    }


def _parse_max_score(gold_row: Dict[str, Any]) -> float:
    value = gold_row.get("max_score", 0)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _slug_for_filename(value: Any, fallback: str) -> str:
    text = str(value or "").strip()
    if not text:
        text = fallback
    text = re.sub(r"[^a-zA-Z0-9._-]+", "_", text).strip("._-")
    return text or fallback


def _build_trace_log_path(trace_log_dir: Path, item: Dict[str, Any], line_idx: int) -> Path:
    exam_id = _slug_for_filename(item.get("exam_id"), "exam")
    page_id = _slug_for_filename(item.get("page_id"), "page")
    question_id = _slug_for_filename(item.get("question_id"), f"line_{line_idx}")
    return trace_log_dir / f"{line_idx:04d}_{exam_id}_p{page_id}_q{question_id}.log"


def _truncate_text(value: Any, limit: int = 500) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\s+", " ", text)
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def _append_judge_trace(
    *,
    trace_log_path: Path,
    line_idx: int,
    judge_mode: str,
    judge_input: Dict[str, Any],
    judge_output: Dict[str, Any],
    judge_trace_details: Optional[Dict[str, Any]] = None,
) -> None:
    trace_payload = {
        "line_idx": line_idx,
        "judge_mode": judge_mode,
        "row_status": judge_output.get("row_status"),
        "score_method": judge_output.get("score_method"),
        "judge_model": judge_output.get("judge_model"),
        "llm_score": judge_output.get("llm_score"),
        "final_score": judge_output.get("final_score"),
        "max_score": judge_output.get("max_score"),
        "judge_latency_ms": judge_output.get("judge_latency_ms"),
        "judge_request_id": judge_output.get("judge_request_id"),
        "judge_input_tokens": judge_output.get("judge_input_tokens"),
        "judge_output_tokens": judge_output.get("judge_output_tokens"),
        "judge_total_tokens": judge_output.get("judge_total_tokens"),
        "judge_reasoning_tokens": judge_output.get("judge_reasoning_tokens"),
        "judge_requests": judge_output.get("judge_requests"),
        "judge_tool_calls": judge_output.get("judge_tool_calls"),
        "llm_comment": judge_output.get("llm_comment"),
    }
    input_snapshot = {
        "answer_type": judge_input.get("answer_type"),
        "question_text_preview": _truncate_text(judge_input.get("question_text"), limit=400),
        "student_answer_preview": _truncate_text(judge_input.get("student_answer"), limit=300),
        "canonical_answer_preview": _truncate_text(judge_input.get("canonical_answer"), limit=300),
    }
    with trace_log_path.open("a", encoding="utf-8") as f:
        f.write(
            "\n=== JUDGE RESULT ===\n"
            f"{json.dumps(trace_payload, ensure_ascii=False, indent=2)}\n\n"
            "=== JUDGE INPUT SNAPSHOT ===\n"
            f"{json.dumps(input_snapshot, ensure_ascii=False, indent=2)}\n"
        )
        if judge_trace_details:
            f.write(
                "\n=== JUDGE DETAILS ===\n"
                f"{json.dumps(judge_trace_details, ensure_ascii=False, indent=2)}\n"
            )


def _str_to_bool(value: str) -> bool:
    normalized = (value or "").strip().lower()
    if normalized in TRUTHY_VALUES:
        return True
    if normalized in FALSY_VALUES:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def run_llm_judge(
    input_path: Path,
    gold_path: Path,
    output_path: Path,
    model_name: str,
    max_tokens: int,
    temperature: float,
    reasoning_effort: str = "high",
    judge_structured_retries: int = 2,
    judge_method: str = "structured",
    judge_g_eval_fallback_structured: bool = True,
    judge_model_url: Optional[str] = None,
    judge_api_key: Optional[str] = None,
    trace_log_enabled: bool = False,
    trace_log_dir: Optional[Path] = None,
    resume_existing: bool = False,
):
    if not input_path.exists():
        raise FileNotFoundError(f"Student output file not found: {input_path}")
    if not gold_path.exists():
        raise FileNotFoundError(f"Gold benchmark file not found: {gold_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_trace_log_dir = trace_log_dir or (output_path.parent / "traces")
    if trace_log_enabled:
        resolved_trace_log_dir.mkdir(parents=True, exist_ok=True)

    gold: Dict[str, Dict[str, Any]] = {}
    with gold_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            key = build_key(row)
            gold[key] = row

    deterministic_count = 0
    llm_count = 0
    technical_skip_count = 0
    judge_error_count = 0
    completed_keys = set()

    if resume_existing and output_path.exists():
        with output_path.open("r", encoding="utf-8") as existing_f:
            for line in existing_f:
                if not line.strip():
                    continue
                completed_keys.add(build_key(json.loads(line)))

    output_mode = "a" if resume_existing else "w"
    with input_path.open("r", encoding="utf-8") as f_in, output_path.open(output_mode, encoding="utf-8") as f_out:
        for line_idx, line in enumerate(tqdm(f_in, desc="Judging answers"), start=1):
            student = json.loads(line)
            key = build_key(student)
            if key in completed_keys:
                continue
            judge_mode = "unknown"
            judge_trace_details: Optional[Dict[str, Any]] = None

            gold_q = gold.get(key)
            if gold_q is None:
                judge_error_count += 1
                judge_mode = "missing_canonical"
                judge_input = {
                    **student,
                    "canonical_answer": None,
                }
                output = _build_missing_canonical_output(student)
            else:
                max_score = _parse_max_score(gold_q)
                judge_input = {
                    **student,
                    "canonical_answer": gold_q.get("canonical_answer"),
                }

                student_status = str(student.get("student_status", "ok") or "ok")
                if student_status != "ok":
                    technical_skip_count += 1
                    judge_mode = "technical_skip"
                    output = _build_technical_skip_output(
                        judge_input=judge_input,
                        student_status=student_status,
                        max_score=max_score,
                    )
                else:
                    answer_type = normalize_answer_type(student.get("answer_type"))

                    if answer_type in DETERMINISTIC_TYPES:
                        deterministic_count += 1
                        judge_mode = "deterministic"
                        judge_result = deterministic_score(
                            answer_type=answer_type,
                            student_answer=student.get("student_answer"),
                            canonical_answer=gold_q.get("canonical_answer"),
                        )
                        output = _build_deterministic_output(
                            judge_input=judge_input,
                            judge_result=judge_result,
                            max_score=max_score,
                        )
                    else:
                        llm_count += 1
                        judge_mode = "llm_judge"
                        try:
                            if judge_method == "g_eval":
                                judge_mode = "g_eval"
                                try:
                                    judge_result = run_g_eval_judge(
                                        model_name=model_name,
                                        model_url=judge_model_url,
                                        api_key=judge_api_key,
                                        user_prompt=build_g_eval_prompt(judge_input),
                                        retries=judge_structured_retries,
                                        temperature=temperature,
                                        max_tokens=max_tokens,
                                        reasoning_effort=reasoning_effort,
                                    )
                                    judge_trace_details = judge_result.get("g_eval_trace")
                                    score_method = "g_eval"
                                except Exception:
                                    if not judge_g_eval_fallback_structured:
                                        raise
                                    judge_mode = "g_eval_fallback_structured"
                                    judge_result = run_structured_judge(
                                        model_name=model_name,
                                        model_url=judge_model_url,
                                        api_key=judge_api_key,
                                        user_prompt=build_user_prompt(judge_input),
                                        retries=judge_structured_retries,
                                        temperature=temperature,
                                        max_tokens=max_tokens,
                                        reasoning_effort=reasoning_effort,
                                    )
                                    judge_trace_details = {
                                        "fallback_from": "g_eval",
                                        "fallback_to": "structured",
                                    }
                                    score_method = "structured_fallback"
                            else:
                                judge_result = run_structured_judge(
                                    model_name=model_name,
                                    model_url=judge_model_url,
                                    api_key=judge_api_key,
                                    user_prompt=build_user_prompt(judge_input),
                                    retries=judge_structured_retries,
                                    temperature=temperature,
                                    max_tokens=max_tokens,
                                    reasoning_effort=reasoning_effort,
                                )
                                score_method = "llm_judge"
                            output = _build_llm_success_output(
                                judge_input=judge_input,
                                judge_result=judge_result,
                                max_score=max_score,
                                model_name=model_name,
                                score_method=score_method,
                            )
                        except Exception as exc:
                            _raise_model_limit_exceeded(
                                exc=exc,
                                model_name=model_name,
                                key=key,
                                prefix="Judge model limit exceeded; aborting run. ",
                            )
                            judge_error_count += 1
                            judge_mode = "judge_error"
                            output = _build_llm_error_output(
                                judge_input=judge_input,
                                max_score=max_score,
                                model_name=model_name,
                                llm_comment=f"Judging failed: {str(exc)[:240]}",
                            )

            f_out.write(json.dumps(output, ensure_ascii=False) + "\n")
            f_out.flush()
            completed_keys.add(key)
            if trace_log_enabled:
                trace_log_path = _build_trace_log_path(resolved_trace_log_dir, student, line_idx)
                _append_judge_trace(
                    trace_log_path=trace_log_path,
                    line_idx=line_idx,
                    judge_mode=judge_mode,
                    judge_input=judge_input,
                    judge_output=output,
                    judge_trace_details=judge_trace_details,
                )

    tqdm.write(
        "Judging summary: "
        f"deterministic={deterministic_count}, "
        f"llm={llm_count}, "
        f"technical_skips={technical_skip_count}, "
        f"judge_errors={judge_error_count}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate student answers with hybrid deterministic + LLM judge")
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path("student_output.jsonl"),
        help="Path to student output JSONL (default: student_output.jsonl)",
    )
    parser.add_argument(
        "--gold-path",
        type=Path,
        default=Path("benchmark/benchmark_v1_0.jsonl"),
        help="Path to benchmark JSONL containing canonical answers (default: benchmark/benchmark_v1_0.jsonl)",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("llm_judge_output.jsonl"),
        help="Path to write judge output JSONL (default: llm_judge_output.jsonl)",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="gpt-5.2-codex",
        help="Judge model name for non-deterministic answer types (default: gpt-5.2-codex)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=300,
        help="Maximum completion tokens for LLM judge (default: 300)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Judge temperature (default: 0.0)",
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        default="high",
        choices=["low", "medium", "high"],
        help="Reasoning effort for judge model calls (default: high)",
    )
    parser.add_argument(
        "--judge-structured-retries",
        type=int,
        default=2,
        help="PydanticAI structured judge retries (default: 2)",
    )
    parser.add_argument(
        "--judge-method",
        type=str,
        default="g_eval",
        choices=["structured", "g_eval"],
        help="Judge method for open-ended answer types: structured|g_eval (default: g_eval)",
    )
    parser.add_argument(
        "--judge-g-eval-fallback-structured",
        type=_str_to_bool,
        default=True,
        help="Fallback from g_eval to structured judge on error (default: true)",
    )
    parser.add_argument(
        "--judge-model-url",
        type=str,
        default=None,
        help="Optional judge model URL/base URL for OpenAI-compatible endpoints",
    )
    parser.add_argument(
        "--resume-existing",
        type=_str_to_bool,
        default=False,
        help="Append only missing judged rows to an existing judge output JSONL (default: false)",
    )
    parser.add_argument(
        "--judge-api-key",
        type=str,
        default=None,
        help="Optional API key override for judge stage",
    )
    parser.add_argument(
        "--trace-log-enabled",
        type=_str_to_bool,
        default=False,
        help="Append judge result to per-question traces (default: false)",
    )
    parser.add_argument(
        "--trace-log-dir",
        type=Path,
        default=None,
        help="Directory with per-question trace logs (default: <output-dir>/traces)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_llm_judge(
        input_path=args.input_path,
        gold_path=args.gold_path,
        output_path=args.output_path,
        model_name=args.judge_model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        reasoning_effort=args.reasoning_effort,
        judge_structured_retries=args.judge_structured_retries,
        judge_method=args.judge_method,
        judge_g_eval_fallback_structured=args.judge_g_eval_fallback_structured,
        judge_model_url=args.judge_model_url,
        judge_api_key=args.judge_api_key,
        trace_log_enabled=args.trace_log_enabled,
        trace_log_dir=args.trace_log_dir,
        resume_existing=args.resume_existing,
    )
