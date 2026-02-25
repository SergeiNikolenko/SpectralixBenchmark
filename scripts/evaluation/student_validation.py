import argparse
import io
import json
import re
import time
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, TextIO
import sys

from tqdm import tqdm

# Ensure repository root is importable when script is run as a file.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.agents import AgentRuntime, AgentRuntimeError
from scripts.agents.models import ensure_chat_completions_url
from scripts.pydantic_guard.student_guard import is_answer_invalid, run_student_guard

STATUS_OK = "ok"
TRUTHY_VALUES = {"1", "true", "yes", "y", "on"}
FALSY_VALUES = {"0", "false", "no", "n", "off"}


class StudentCallError(Exception):
    def __init__(self, status: str, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


class _TeeStream:
    def __init__(self, *streams: TextIO):
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


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


def _sanitize_error(error: Exception, limit: int = 240) -> str:
    message = str(error).strip() or error.__class__.__name__
    if len(message) > limit:
        return message[: limit - 3] + "..."
    return message


def _slug_for_filename(value: Any, fallback: str) -> str:
    text = str(value or "").strip()
    if not text:
        text = fallback
    text = re.sub(r"[^a-zA-Z0-9._-]+", "_", text).strip("._-")
    return text or fallback


def _build_trace_log_path(trace_log_dir: Path, question: Dict[str, Any], line_idx: int) -> Path:
    exam_id = _slug_for_filename(question.get("exam_id"), "exam")
    page_id = _slug_for_filename(question.get("page_id"), "page")
    question_id = _slug_for_filename(question.get("question_id"), f"line_{line_idx}")
    return trace_log_dir / f"{line_idx:04d}_{exam_id}_p{page_id}_q{question_id}.log"


def _write_trace_log(
    *,
    trace_log_path: Path,
    question: Dict[str, Any],
    model_name: str,
    student_status: str,
    student_error: str,
    student_answer: str,
    raw_answer: str,
    elapsed_ms: int,
    captured_trace: str,
) -> None:
    trace_payload = {
        "exam_id": question.get("exam_id"),
        "page_id": question.get("page_id"),
        "question_id": question.get("question_id"),
        "answer_type": question.get("answer_type"),
        "model_name": model_name,
        "student_status": student_status,
        "student_error": student_error,
        "student_elapsed_ms": elapsed_ms,
    }

    content = (
        "=== TRACE METADATA ===\n"
        f"{json.dumps(trace_payload, ensure_ascii=False, indent=2)}\n\n"
        "=== QUESTION TEXT ===\n"
        f"{question.get('question_text', '')}\n\n"
        "=== AGENT STDOUT/STDERR TRACE ===\n"
        f"{captured_trace.strip() or '<empty>'}\n\n"
        "=== RAW MODEL ANSWER ===\n"
        f"{raw_answer or '<empty>'}\n\n"
        "=== NORMALIZED STUDENT ANSWER ===\n"
        f"{student_answer or '<empty>'}\n"
    )
    trace_log_path.write_text(content, encoding="utf-8")


def _wait_or_raise_retry(
    *,
    attempt: int,
    max_retries: int,
    status: str,
    retryable_statuses: Set[str],
    error: StudentCallError,
) -> None:
    if attempt == max_retries - 1 or status not in retryable_statuses:
        raise error

    wait_time = 2**attempt
    tqdm.write(
        f"Attempt {attempt + 1}/{max_retries} failed with status={status}: "
        f"{error.message}. Retrying in {wait_time}s..."
    )
    time.sleep(wait_time)


def call_agent(
    runtime: AgentRuntime,
    question: Dict[str, Any],
    max_retries: int = 3,
) -> str:
    retryable_statuses = {
        "timeout",
        "http_error",
        "connection_error",
        "parse_error",
        "agent_step_error",
    }

    final_error = StudentCallError("parse_error", "Unknown runtime error")

    for attempt in range(max_retries):
        try:
            return runtime.solve_question(question)
        except AgentRuntimeError as exc:
            status = exc.status or "parse_error"
            final_error = StudentCallError(status, _sanitize_error(exc))
        except Exception as exc:
            status = "parse_error"
            final_error = StudentCallError(status, _sanitize_error(exc))

        _wait_or_raise_retry(
            attempt=attempt,
            max_retries=max_retries,
            status=status,
            retryable_statuses=retryable_statuses,
            error=final_error,
        )

    raise final_error


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
    *,
    agent_enabled: bool = True,
    agent_max_steps: int = 6,
    agent_sandbox: str = "docker",
    agent_tools_profile: str = "full",
    agent_config: Optional[Path] = Path("scripts/agents/agent_config.yaml"),
    api_key: Optional[str] = None,
    student_guard_enabled: bool = True,
    student_guard_mode: str = "on_failure",
    student_guard_retries: int = 2,
    trace_log_enabled: bool = True,
    trace_log_dir: Optional[Path] = None,
):
    _ = max_tokens
    _ = temperature

    questions = load_benchmark_questions(benchmark_path=benchmark_path)
    if limit is not None and limit >= 0:
        questions = questions[:limit]
    if workers != 1:
        tqdm.write(
            f"workers={workers} requested; running in sequential mode in this script version."
        )

    if not agent_enabled:
        tqdm.write("agent_enabled=false is deprecated; legacy backend was removed. Using agent runtime.")

    try:
        runtime = AgentRuntime(
            model_url=model_url,
            model_name=model_name,
            benchmark_path=benchmark_path,
            api_key=api_key,
            config_path=agent_config,
            max_steps=agent_max_steps,
            sandbox=agent_sandbox,
            tools_profile=agent_tools_profile,
            timeout_sec=timeout,
        )
        runtime.preflight()
    except AgentRuntimeError as exc:
        raise RuntimeError(f"Agent preflight failed: {exc.status}: {exc.message}") from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to initialize agent runtime: {exc}") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_trace_log_dir = trace_log_dir or (output_path.parent / "traces")
    if trace_log_enabled:
        resolved_trace_log_dir.mkdir(parents=True, exist_ok=True)
    guard_mode = (student_guard_mode or "on_failure").strip().lower()
    if guard_mode not in {"on_failure", "always", "off"}:
        raise ValueError(f"Invalid student_guard_mode: {student_guard_mode}")

    try:
        with output_path.open("w", encoding="utf-8") as f_out:
            for line_idx, question in enumerate(
                tqdm(questions, total=len(questions), desc="Validating student answers"),
                start=1,
            ):
                started_at = time.perf_counter()
                student_status = STATUS_OK
                student_error = ""
                student_answer = ""
                raw_answer = ""
                trace_buffer = io.StringIO()

                tee_stdout = _TeeStream(sys.stdout, trace_buffer)
                tee_stderr = _TeeStream(sys.stderr, trace_buffer)
                with redirect_stdout(tee_stdout), redirect_stderr(tee_stderr):
                    try:
                        raw_answer = call_agent(
                            runtime=runtime,
                            question=question,
                            max_retries=max_retries,
                        )

                        student_answer = normalize_student_answer(
                            question.get("answer_type", ""),
                            raw_answer,
                        )
                        guard_on_failure = is_answer_invalid(question.get("answer_type", ""), student_answer)
                        should_run_guard = (
                            student_guard_enabled
                            and guard_mode != "off"
                            and (guard_mode == "always" or guard_on_failure)
                        )
                        if should_run_guard:
                            try:
                                guard_output = run_student_guard(
                                    question=question,
                                    raw_answer=raw_answer,
                                    normalized_answer=student_answer,
                                    model_name=model_name,
                                    model_url=model_url,
                                    api_key=api_key,
                                    retries=student_guard_retries,
                                )
                                guarded_answer = normalize_student_answer(
                                    question.get("answer_type", ""),
                                    guard_output.final_answer,
                                )
                                if guarded_answer:
                                    student_answer = guarded_answer
                                elif guard_on_failure:
                                    raise StudentCallError(
                                        "parse_error",
                                        "Student guard returned empty normalized answer",
                                    )
                            except StudentCallError:
                                raise
                            except Exception as exc:
                                if guard_on_failure:
                                    raise StudentCallError(
                                        "parse_error",
                                        f"Student guard failed: {_sanitize_error(exc)}",
                                    ) from exc
                                tqdm.write(f"[WARN] Student guard skipped: {_sanitize_error(exc)}")
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

                if trace_log_enabled:
                    trace_log_path = _build_trace_log_path(
                        trace_log_dir=resolved_trace_log_dir,
                        question=question,
                        line_idx=line_idx,
                    )
                    _write_trace_log(
                        trace_log_path=trace_log_path,
                        question=question,
                        model_name=model_name,
                        student_status=student_status,
                        student_error=student_error,
                        student_answer=student_answer,
                        raw_answer=raw_answer,
                        elapsed_ms=elapsed_ms,
                        captured_trace=trace_buffer.getvalue(),
                    )

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
    finally:
        runtime.close()


def _str_to_bool(value: str) -> bool:
    normalized = (value or "").strip().lower()
    if normalized in TRUTHY_VALUES:
        return True
    if normalized in FALSY_VALUES:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _resolve_model_url(model_url: Optional[str], api_base_url: Optional[str]) -> str:
    if model_url:
        return ensure_chat_completions_url(model_url)
    if api_base_url:
        return ensure_chat_completions_url(api_base_url)
    raise argparse.ArgumentTypeError("Either --model-url or --api-base-url must be provided")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark student answers using an agentic runtime (smolagents)"
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
        default=None,
        help="Model API endpoint URL (OpenAI-compatible chat completion endpoint preferred)",
    )
    parser.add_argument(
        "--api-base-url",
        type=str,
        default=None,
        help="OpenAI-compatible API base URL (alternative to --model-url)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Name of the model to use",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Optional API key override for agent runtime",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum number of tokens in the response (kept for CLI compatibility)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.5,
        help="Temperature for sampling (kept for CLI compatibility)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Timeout in seconds for agent requests (default: 120)",
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

    parser.add_argument(
        "--agent-enabled",
        type=_str_to_bool,
        default=True,
        help="Enable agent runtime (default: true; false is accepted for compatibility)",
    )
    parser.add_argument(
        "--agent-max-steps",
        type=int,
        default=6,
        help="Maximum reasoning steps for agent runtime (default: 6)",
    )
    parser.add_argument(
        "--agent-sandbox",
        type=str,
        default="docker",
        help="Agent executor type (default: docker)",
    )
    parser.add_argument(
        "--agent-tools-profile",
        type=str,
        default="full",
        help="Tools profile from config (default: full)",
    )
    parser.add_argument(
        "--agent-config",
        type=Path,
        default=Path("scripts/agents/agent_config.yaml"),
        help="Path to agent YAML config (default: scripts/agents/agent_config.yaml)",
    )
    parser.add_argument(
        "--student-guard-enabled",
        type=_str_to_bool,
        default=True,
        help="Enable PydanticAI student answer guard (default: true)",
    )
    parser.add_argument(
        "--student-guard-mode",
        type=str,
        default="on_failure",
        choices=["on_failure", "always", "off"],
        help="When to run student guard: on_failure|always|off (default: on_failure)",
    )
    parser.add_argument(
        "--student-guard-retries",
        type=int,
        default=2,
        help="PydanticAI student guard retries (default: 2)",
    )
    parser.add_argument(
        "--trace-log-enabled",
        type=_str_to_bool,
        default=True,
        help="Write per-question detailed traces (stdout/stderr, raw answer, normalized answer)",
    )
    parser.add_argument(
        "--trace-log-dir",
        type=Path,
        default=None,
        help="Directory for per-question trace logs (default: <output-dir>/traces)",
    )

    args = parser.parse_args()

    try:
        resolved_model_url = _resolve_model_url(args.model_url, args.api_base_url)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))

    try:
        run_benchmark_inference(
            benchmark_path=args.benchmark_path,
            output_path=args.output_path,
            model_url=resolved_model_url,
            model_name=args.model_name,
            api_key=args.api_key,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            timeout=args.timeout,
            max_retries=args.max_retries,
            max_error_len=args.max_error_len,
            limit=args.limit,
            workers=args.workers,
            agent_enabled=args.agent_enabled,
            agent_max_steps=args.agent_max_steps,
            agent_sandbox=args.agent_sandbox,
            agent_tools_profile=args.agent_tools_profile,
            agent_config=args.agent_config,
            student_guard_enabled=args.student_guard_enabled,
            student_guard_mode=args.student_guard_mode,
            student_guard_retries=args.student_guard_retries,
            trace_log_enabled=args.trace_log_enabled,
            trace_log_dir=args.trace_log_dir,
        )
    except Exception as exc:
        raise SystemExit(f"[ERROR] {exc}") from exc
