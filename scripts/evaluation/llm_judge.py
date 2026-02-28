import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import sys

from openai import OpenAI
from tqdm import tqdm

# Ensure repository root is importable when script is run as a file.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.agents.models import parse_model_url
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
)


JUDGE_SYSTEM_PROMPT = """
You are an expert chemistry exam examiner.

Your task is to evaluate a student's answer against the canonical answer.

You MUST follow these rules strictly:
- Do NOT help the student.
- Do NOT explain chemistry theory.
- Judge ONLY correctness and completeness.
- Be strict and conservative.

Return ONLY valid JSON:
{
  "llm_score": <float in [0, 1]>,
  "llm_comment": "<short justification>"
}
""".strip()


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
Question type: {item.get("question_type")}
Answer type: {item.get("answer_type")}

Question:
{item.get("question_text")}

Canonical answer:
{item.get("canonical_answer")}

Student answer:
{item.get("student_answer")}
""".strip()


def parse_llm_judge_json(content: str) -> Dict[str, Any]:
    raw = (content or "").strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw.replace("json\n", "", 1)

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            raise
        parsed = json.loads(match.group(0))

    if not isinstance(parsed, dict):
        raise ValueError("Judge response must be a JSON object")
    if "llm_score" not in parsed:
        raise ValueError("Judge response missing 'llm_score'")
    if "llm_comment" not in parsed:
        parsed["llm_comment"] = ""

    parsed["llm_score"] = float(parsed["llm_score"])
    if parsed["llm_score"] < 0:
        parsed["llm_score"] = 0.0
    if parsed["llm_score"] > 1:
        parsed["llm_score"] = 1.0
    parsed["llm_comment"] = str(parsed["llm_comment"])
    return parsed


def _str_to_bool(value: str) -> bool:
    normalized = (value or "").strip().lower()
    if normalized in TRUTHY_VALUES:
        return True
    if normalized in FALSY_VALUES:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


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


def _ensure_client(
    *,
    client: Optional[OpenAI],
    judge_api_key: Optional[str],
    judge_model_url: Optional[str],
) -> OpenAI:
    if client is not None:
        return client
    return OpenAI(
        api_key=judge_api_key,
        base_url=_resolve_judge_api_base(judge_model_url),
    )


def _build_llm_success_output(
    *,
    judge_input: Dict[str, Any],
    judge_result: Dict[str, Any],
    max_score: float,
    model_name: str,
    comment_suffix: str = "",
) -> Dict[str, Any]:
    return {
        **judge_input,
        "llm_score": judge_result["llm_score"],
        "llm_comment": f"{judge_result['llm_comment']}{comment_suffix}",
        "final_score": max_score * float(judge_result["llm_score"]),
        "max_score": max_score,
        "score_method": "llm_judge",
        "judge_model": model_name,
        "judge_request_id": judge_result.get("judge_request_id"),
        "judge_latency_ms": judge_result.get("judge_latency_ms"),
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
        "row_status": "judge_error",
    }


def _resolve_judge_api_base(model_url: Optional[str]) -> Optional[str]:
    if not model_url:
        return None
    api_base, _ = parse_model_url(model_url)
    return api_base


def call_llm_judge(
    client: OpenAI,
    model_name: str,
    max_tokens: int,
    temperature: float,
    reasoning_effort: str,
    item: Dict[str, Any],
) -> Dict[str, Any]:
    started_at = time.perf_counter()
    response = client.chat.completions.create(
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(item)},
        ],
    )
    elapsed_ms = int((time.perf_counter() - started_at) * 1000)

    content = response.choices[0].message.content or ""
    parsed = parse_llm_judge_json(content)

    result = {
        "llm_score": parsed["llm_score"],
        "llm_comment": parsed["llm_comment"],
        "judge_request_id": getattr(response, "id", None),
        "judge_latency_ms": elapsed_ms,
    }
    return result


def _parse_max_score(gold_row: Dict[str, Any]) -> float:
    value = gold_row.get("max_score", 0)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _call_legacy_judge(
    *,
    client: Optional[OpenAI],
    judge_api_key: Optional[str],
    judge_model_url: Optional[str],
    model_name: str,
    max_tokens: int,
    temperature: float,
    reasoning_effort: str,
    judge_input: Dict[str, Any],
) -> Tuple[Optional[OpenAI], Dict[str, Any]]:
    resolved_client = _ensure_client(
        client=client,
        judge_api_key=judge_api_key,
        judge_model_url=judge_model_url,
    )
    judge_result = call_llm_judge(
        client=resolved_client,
        model_name=model_name,
        max_tokens=max_tokens,
        temperature=temperature,
        reasoning_effort=reasoning_effort,
        item=judge_input,
    )
    return resolved_client, judge_result


def run_llm_judge(
    input_path: Path,
    gold_path: Path,
    output_path: Path,
    model_name: str,
    max_tokens: int,
    temperature: float,
    reasoning_effort: str = "high",
    judge_structured_enabled: bool = True,
    judge_structured_retries: int = 2,
    judge_structured_fallback_legacy: bool = True,
    judge_model_url: Optional[str] = None,
    judge_api_key: Optional[str] = None,
):
    if not input_path.exists():
        raise FileNotFoundError(f"Student output file not found: {input_path}")
    if not gold_path.exists():
        raise FileNotFoundError(f"Gold benchmark file not found: {gold_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    gold: Dict[str, Dict[str, Any]] = {}
    with gold_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            key = build_key(row)
            gold[key] = row

    client: Optional[OpenAI] = None

    deterministic_count = 0
    llm_count = 0
    technical_skip_count = 0
    judge_error_count = 0

    with input_path.open("r", encoding="utf-8") as f_in, output_path.open("w", encoding="utf-8") as f_out:
        for line in tqdm(f_in, desc="Judging answers"):
            student = json.loads(line)
            key = build_key(student)

            gold_q = gold.get(key)
            if gold_q is None:
                judge_error_count += 1
                output = _build_missing_canonical_output(student)
                f_out.write(json.dumps(output, ensure_ascii=False) + "\n")
                continue

            max_score = _parse_max_score(gold_q)
            judge_input = {
                **student,
                "canonical_answer": gold_q.get("canonical_answer"),
            }

            student_status = str(student.get("student_status", "ok") or "ok")
            if student_status != "ok":
                technical_skip_count += 1
                output = _build_technical_skip_output(
                    judge_input=judge_input,
                    student_status=student_status,
                    max_score=max_score,
                )
                f_out.write(json.dumps(output, ensure_ascii=False) + "\n")
                continue

            answer_type = normalize_answer_type(student.get("answer_type"))

            if answer_type in DETERMINISTIC_TYPES:
                deterministic_count += 1
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
                f_out.write(json.dumps(output, ensure_ascii=False) + "\n")
                continue

            llm_count += 1
            try:
                if judge_structured_enabled:
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
                else:
                    client, judge_result = _call_legacy_judge(
                        client=client,
                        judge_api_key=judge_api_key,
                        judge_model_url=judge_model_url,
                        model_name=model_name,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        reasoning_effort=reasoning_effort,
                        judge_input=judge_input,
                    )
                output = _build_llm_success_output(
                    judge_input=judge_input,
                    judge_result=judge_result,
                    max_score=max_score,
                    model_name=model_name,
                )
            except Exception as exc:
                _raise_model_limit_exceeded(
                    exc=exc,
                    model_name=model_name,
                    key=key,
                    prefix="Judge model limit exceeded; aborting run. ",
                )
                if judge_structured_enabled and judge_structured_fallback_legacy:
                    try:
                        client, judge_result = _call_legacy_judge(
                            client=client,
                            judge_api_key=judge_api_key,
                            judge_model_url=judge_model_url,
                            model_name=model_name,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            reasoning_effort=reasoning_effort,
                            judge_input=judge_input,
                        )
                        output = _build_llm_success_output(
                            judge_input=judge_input,
                            judge_result=judge_result,
                            max_score=max_score,
                            model_name=model_name,
                            comment_suffix=" (legacy_fallback)",
                        )
                    except Exception as fallback_exc:
                        _raise_model_limit_exceeded(
                            exc=fallback_exc,
                            model_name=model_name,
                            key=key,
                            prefix="Judge model limit exceeded during legacy fallback; aborting run. ",
                        )
                        judge_error_count += 1
                        output = _build_llm_error_output(
                            judge_input=judge_input,
                            max_score=max_score,
                            model_name=model_name,
                            llm_comment=(
                                f"Judging failed: {str(exc)[:160]}; "
                                f"legacy fallback failed: {str(fallback_exc)[:160]}"
                            ),
                        )
                else:
                    judge_error_count += 1
                    output = _build_llm_error_output(
                        judge_input=judge_input,
                        max_score=max_score,
                        model_name=model_name,
                        llm_comment=f"Judging failed: {str(exc)[:240]}",
                    )

            f_out.write(json.dumps(output, ensure_ascii=False) + "\n")

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
        "--judge-structured-enabled",
        type=_str_to_bool,
        default=True,
        help="Enable PydanticAI structured judge for non-deterministic types (default: true)",
    )
    parser.add_argument(
        "--judge-structured-retries",
        type=int,
        default=2,
        help="PydanticAI structured judge retries (default: 2)",
    )
    parser.add_argument(
        "--judge-structured-fallback-legacy",
        type=_str_to_bool,
        default=True,
        help="Fallback to legacy JSON judge parsing on structured failure (default: true)",
    )
    parser.add_argument(
        "--judge-model-url",
        type=str,
        default=None,
        help="Optional judge model URL/base URL for OpenAI-compatible endpoints",
    )
    parser.add_argument(
        "--judge-api-key",
        type=str,
        default=None,
        help="Optional API key override for judge stage",
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
        judge_structured_enabled=args.judge_structured_enabled,
        judge_structured_retries=args.judge_structured_retries,
        judge_structured_fallback_legacy=args.judge_structured_fallback_legacy,
        judge_model_url=args.judge_model_url,
        judge_api_key=args.judge_api_key,
    )
