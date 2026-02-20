import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from requests import Response
from tqdm import tqdm

STATUS_OK = "ok"


class StudentCallError(Exception):
    def __init__(self, status: str, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def normalize_answer_type(answer_type: Optional[str]) -> str:
    if not answer_type:
        return ""
    normalized = answer_type.strip().lower()
    if normalized == "single_choise":
        return "single_choice"
    return normalized


def load_benchmark_questions(benchmark_path: Path) -> List[Dict[str, Any]]:
    if not benchmark_path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {benchmark_path}")

    questions: List[Dict[str, Any]] = []

    with benchmark_path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            try:
                question = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_idx}: {exc}") from exc
            questions.append(question)

    return questions


def _format_instruction(answer_type: str) -> str:
    if answer_type == "single_choice":
        return "Return only one option label (example: A)."
    if answer_type == "multiple_choice":
        return "Return only option labels separated by '; ' (example: A; D)."
    if answer_type == "ordering":
        return "Return only the ordered labels/numbers separated by '; ' (example: 4; 2; 3; 1)."
    if answer_type in {"msms_structure_prediction", "structure"}:
        return "Return only one SMILES string on a single line."
    return (
        "Start the response with 'Answer: <machine-readable answer>'. "
        "If needed, add a very short explanation after that line."
    )


def build_prompt(question: Dict[str, Any]) -> str:
    answer_type = normalize_answer_type(question.get("answer_type"))
    return (
        "You are solving a chemistry benchmark task. "
        "Follow the output format exactly.\n\n"
        f"Output format rule: {_format_instruction(answer_type)}\n\n"
        "Question:\n"
        f"{question.get('question_text', '')}"
    )


def _extract_answer_payload(raw_text: str) -> str:
    text = raw_text.strip()
    if not text:
        return ""

    text = text.replace("```", "")
    text = re.sub(r"`([^`]*)`", r"\1", text)

    answer_line = re.search(r"(?im)^\s*answer\s*:\s*(.+)$", text)
    if answer_line:
        return answer_line.group(1).strip()
    return text


def _compact_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _normalize_sequence(text: str) -> str:
    sanitized = text.replace("\n", ";")
    sanitized = sanitized.replace(",", ";")
    sanitized = sanitized.replace("|", ";")
    tokens: List[str] = []
    for raw_token in sanitized.split(";"):
        token = raw_token.strip()
        if not token:
            continue
        token = re.sub(r"^\d+[\.)]\s*", "", token)
        if token:
            tokens.append(token)
    return "; ".join(tokens)


def normalize_student_answer(answer_type: str, raw_text: str, max_len: int = 1000) -> str:
    payload = _extract_answer_payload(raw_text)
    if not payload:
        return ""

    normalized_type = normalize_answer_type(answer_type)

    if normalized_type == "single_choice":
        tokens = re.findall(r"[A-Za-z0-9]+", payload)
        if tokens:
            return tokens[0].strip()[:max_len]
        return payload[:max_len]

    if normalized_type in {"multiple_choice", "ordering"}:
        seq = _normalize_sequence(payload)
        if seq:
            return seq[:max_len]
        return _compact_whitespace(payload)[:max_len]

    if normalized_type in {"msms_structure_prediction", "structure"}:
        for line in payload.splitlines():
            candidate = line.strip()
            if candidate:
                return candidate[:max_len]
        return payload[:max_len]

    compact = _compact_whitespace(payload)
    return compact[:max_len]


def _classify_http_error(response: Optional[Response]) -> str:
    if response is None:
        return "http_error"
    if response.status_code in {401, 403}:
        return "auth_error"
    return "http_error"


def _sanitize_error(error: Exception, limit: int = 240) -> str:
    message = str(error).strip() or error.__class__.__name__
    if len(message) > limit:
        return message[: limit - 3] + "..."
    return message


def call_chemllm(
    prompt: str,
    model_url: str,
    model_name: str,
    max_tokens: int,
    temperature: float,
    timeout: int,
    max_retries: int = 3,
) -> str:
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }

    retryable_statuses = {"timeout", "connection_error", "http_error", "parse_error"}

    for attempt in range(max_retries):
        try:
            response = requests.post(
                model_url,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()

            try:
                data = response.json()
            except json.JSONDecodeError as exc:
                content = response.text.strip()
                if content:
                    lines = content.split("\n")
                    full_response = ""
                    for line in lines:
                        if not line.strip():
                            continue
                        try:
                            chunk = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if "message" in chunk and "content" in chunk["message"]:
                            full_response += chunk["message"]["content"]
                        elif "response" in chunk:
                            full_response += chunk["response"]
                    if full_response:
                        return full_response.strip()
                raise StudentCallError("parse_error", f"Failed to parse response as JSON: {exc}")

            if "message" in data and "content" in data["message"]:
                return data["message"]["content"].strip()
            if "choices" in data and data["choices"]:
                return data["choices"][0]["message"]["content"].strip()
            if "response" in data:
                return data["response"].strip()

            raise StudentCallError("parse_error", "Unexpected response format")

        except requests.Timeout as exc:
            status = "timeout"
            final_error = StudentCallError(status, _sanitize_error(exc))
        except requests.ConnectionError as exc:
            status = "connection_error"
            final_error = StudentCallError(status, _sanitize_error(exc))
        except requests.HTTPError as exc:
            status = _classify_http_error(exc.response)
            final_error = StudentCallError(status, _sanitize_error(exc))
        except StudentCallError as exc:
            status = exc.status
            final_error = exc
        except requests.RequestException as exc:
            status = "http_error"
            final_error = StudentCallError(status, _sanitize_error(exc))
        except Exception as exc:
            status = "parse_error"
            final_error = StudentCallError(status, _sanitize_error(exc))

        if attempt == max_retries - 1 or status not in retryable_statuses:
            raise final_error

        wait_time = 2**attempt
        tqdm.write(
            f"Attempt {attempt + 1}/{max_retries} failed with status={status}: "
            f"{final_error.message}. Retrying in {wait_time}s..."
        )
        time.sleep(wait_time)

    raise StudentCallError("parse_error", "Unreachable retry state")


def run_benchmark_inference(
    benchmark_path: Path,
    output_path: Path,
    model_url: str,
    model_name: str,
    max_tokens: int,
    temperature: float,
    timeout: int,
    max_retries: int = 3,
    max_error_len: int = 240,
    limit: Optional[int] = None,
    workers: int = 1,
):
    questions = load_benchmark_questions(benchmark_path=benchmark_path)
    if limit is not None and limit >= 0:
        questions = questions[:limit]
    if workers != 1:
        tqdm.write(
            f"workers={workers} requested; running in sequential mode in this script version."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f_out:
        for line_idx, question in enumerate(
            tqdm(questions, total=len(questions), desc="Validating student answers"),
            start=1,
        ):
            prompt = build_prompt(question)

            started_at = time.perf_counter()
            student_status = STATUS_OK
            student_error = ""
            student_answer = ""

            try:
                raw_answer = call_chemllm(
                    prompt,
                    model_url=model_url,
                    model_name=model_name,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout=timeout,
                    max_retries=max_retries,
                )
                student_answer = normalize_student_answer(
                    question.get("answer_type", ""),
                    raw_answer,
                )
            except StudentCallError as exc:
                student_status = exc.status
                student_error = exc.message
                student_answer = ""
                tqdm.write(f"[ERROR] Line {line_idx}: status={student_status} error={student_error}")
            except Exception as exc:  # Defensive fallback
                student_status = "parse_error"
                student_error = _sanitize_error(exc, limit=max_error_len)
                student_answer = ""
                tqdm.write(f"[ERROR] Line {line_idx}: status={student_status} error={student_error}")

            elapsed_ms = int((time.perf_counter() - started_at) * 1000)

            if student_error:
                student_error = student_error[:max_error_len]

            result = {
                "exam_id": question.get("exam_id"),
                "page_id": question.get("page_id"),
                "question_id": question.get("question_id"),
                "question_type": question.get("question_type"),
                "question_text": question.get("question_text"),
                "answer_type": question.get("answer_type"),
                "student_answer": student_answer,
                "student_status": student_status,
                "student_error": student_error,
                "student_elapsed_ms": elapsed_ms,
            }

            f_out.write(json.dumps(result, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark student answers using a remote LLM model"
    )
    parser.add_argument(
        "--benchmark-path",
        type=Path,
        required=True,
        help="Path to the benchmark JSONL file",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Path to save the output JSONL file",
    )
    parser.add_argument(
        "--model-url",
        type=str,
        required=True,
        help="URL of the model API endpoint",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Name of the model to use",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum number of tokens in the response (default: 256)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.5,
        help="Temperature for sampling (default: 0.5)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Timeout in seconds for API requests (default: 120)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retry attempts for retryable request failures (default: 3)",
    )
    parser.add_argument(
        "--max-error-len",
        type=int,
        default=240,
        help="Maximum stored error length in output JSONL (default: 240)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit of rows",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Reserved for parallel execution tuning (default: 1)",
    )

    args = parser.parse_args()

    run_benchmark_inference(
        benchmark_path=args.benchmark_path,
        output_path=args.output_path,
        model_url=args.model_url,
        model_name=args.model_name,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        timeout=args.timeout,
        max_retries=args.max_retries,
        max_error_len=args.max_error_len,
        limit=args.limit,
        workers=args.workers,
    )
