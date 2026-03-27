import argparse
import io
import json
import re
import time
from contextlib import nullcontext, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, TextIO, Tuple
import sys

from tqdm import tqdm

from scripts.agents import AgentRuntime, AgentRuntimeError
from scripts.agents.models import ensure_chat_completions_url
from scripts.pydantic_guard.student_guard import is_answer_invalid, run_student_guard

STATUS_OK = "ok"
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


class StudentCallError(Exception):
    def __init__(self, status: str, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


class ModelLimitExceededError(RuntimeError):
    pass


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


def build_result_key(item: Dict[str, Any]) -> str:
    return f"{item.get('exam_id')}/{item.get('page_id')}/{item.get('question_id')}"


def _load_completed_keys_from_jsonl(path: Path) -> Set[str]:
    completed_keys: Set[str] = set()
    with path.open("r", encoding="utf-8") as existing_f:
        for line_idx, line in enumerate(existing_f, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                tqdm.write(
                    f"[WARN] Skipping malformed existing JSONL row at {path}:{line_idx} "
                    "(likely truncated tail during prior interrupted run)"
                )
                continue
            completed_keys.add(build_result_key(payload))
    return completed_keys


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


def _clip_text(text: str, max_len: Optional[int]) -> str:
    if max_len is None or max_len <= 0:
        return text
    return text[:max_len]


def normalize_student_answer(
    answer_type: str,
    raw_text: str,
    max_len: Optional[int] = None,
) -> str:
    payload = _extract_answer_payload(raw_text)
    if not payload:
        return ""

    normalized_type = normalize_answer_type(answer_type)

    if normalized_type == "single_choice":
        tokens = re.findall(r"[A-Za-z0-9]+", payload)
        if tokens:
            return _clip_text(tokens[0].strip(), max_len)
        return _clip_text(payload, max_len)

    if normalized_type in {"multiple_choice", "ordering"}:
        seq = _normalize_sequence(payload)
        if seq:
            return _clip_text(seq, max_len)
        return _clip_text(_compact_whitespace(payload), max_len)

    if normalized_type in {"msms_structure_prediction", "structure"}:
        for line in payload.splitlines():
            candidate = line.strip()
            if candidate:
                return _clip_text(candidate, max_len)
        return _clip_text(payload, max_len)

    compact = _compact_whitespace(payload)
    return _clip_text(compact, max_len)


def _sanitize_error(error: Exception, limit: int = 240) -> str:
    message = str(error).strip() or error.__class__.__name__
    if len(message) > limit:
        return message[: limit - 3] + "..."
    return message


def _is_model_limit_error(error_message: str) -> bool:
    text = str(error_message or "").strip().lower()
    if not text:
        return False
    return any(marker in text for marker in MODEL_LIMIT_ERROR_MARKERS)


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


def _truncate_text(value: Any, limit: int = 300) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\s+", " ", text)
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def _reasoning_tokens_from_model_output_message(model_output_message: Any) -> int:
    if not isinstance(model_output_message, dict):
        return 0
    raw = model_output_message.get("raw")
    if not isinstance(raw, dict):
        return 0
    usage = raw.get("usage")
    if not isinstance(usage, dict):
        return 0
    completion_details = usage.get("completion_tokens_details")
    if not isinstance(completion_details, dict):
        return 0
    candidate = completion_details.get("reasoning_tokens")
    return candidate if isinstance(candidate, int) else 0


def _compact_run_details(agent_run_details: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(agent_run_details, dict):
        return None

    steps_src = agent_run_details.get("steps")
    compact_steps: List[Dict[str, Any]] = []
    if isinstance(steps_src, list):
        for step in steps_src:
            if not isinstance(step, dict):
                continue
            if "step_number" not in step:
                continue

            step_number = int(step.get("step_number") or 0)
            timing = step.get("timing") if isinstance(step.get("timing"), dict) else {}
            duration_sec = float(timing.get("duration") or 0.0)
            duration_ms = int(duration_sec * 1000)

            model_output_message = step.get("model_output_message")
            thought = ""
            code = ""
            if isinstance(model_output_message, dict):
                content = model_output_message.get("content")
                if isinstance(content, dict):
                    thought = _truncate_text(content.get("thought"), limit=500)
                    code = str(content.get("code") or "")

            tool_calls_src = step.get("tool_calls")
            tool_calls: List[Dict[str, Any]] = []
            if isinstance(tool_calls_src, list):
                for tool_call in tool_calls_src:
                    if not isinstance(tool_call, dict):
                        continue
                    function = tool_call.get("function") if isinstance(tool_call.get("function"), dict) else {}
                    name = str(function.get("name") or "")
                    arguments_preview = _truncate_text(function.get("arguments"), limit=240)
                    tool_calls.append(
                        {
                            "name": name,
                            "arguments_preview": arguments_preview,
                        }
                    )

            error = step.get("error")
            error_message = ""
            if isinstance(error, dict):
                error_message = _truncate_text(error.get("message"), limit=500)
            elif error:
                error_message = _truncate_text(error, limit=500)

            model_output_preview = ""
            if "model_output" in step and step.get("model_output") is not None:
                model_output_preview = _truncate_text(step.get("model_output"), limit=500)

            compact_steps.append(
                {
                    "step_number": step_number,
                    "duration_ms": duration_ms,
                    "thought": thought,
                    "code": code,
                    "tool_calls": tool_calls,
                    "observations_preview": _truncate_text(step.get("observations"), limit=500),
                    "model_output_preview": model_output_preview,
                    "error": error_message,
                    "reasoning_tokens": _reasoning_tokens_from_model_output_message(model_output_message),
                    "is_final_answer": bool(step.get("is_final_answer", False)),
                }
            )

    return {
        "state": str(agent_run_details.get("state") or ""),
        "output_preview": _truncate_text(agent_run_details.get("output"), limit=500),
        "step_count": len(compact_steps),
        "steps": compact_steps,
    }


def _render_step_summary(compact_run_details: Optional[Dict[str, Any]]) -> str:
    if not isinstance(compact_run_details, dict):
        return "<empty>"

    steps = compact_run_details.get("steps")
    if not isinstance(steps, list) or not steps:
        return "<empty>"

    lines: List[str] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        step_number = step.get("step_number")
        duration_ms = step.get("duration_ms")
        reasoning_tokens = step.get("reasoning_tokens")
        lines.append(
            f"Step {step_number} | duration_ms={duration_ms} | reasoning_tokens={reasoning_tokens}"
        )

        thought = str(step.get("thought") or "").strip()
        if thought:
            lines.append(f"  Thought: {thought}")

        code = str(step.get("code") or "").strip()
        if code:
            lines.append("  Code:")
            for line in code.splitlines():
                lines.append(f"    {line}")

        tool_calls = step.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            lines.append("  Tools:")
            for item in tool_calls:
                if not isinstance(item, dict):
                    continue
                lines.append(
                    f"    - {item.get('name')}: {item.get('arguments_preview')}"
                )

        observation = str(step.get("observations_preview") or "").strip()
        if observation:
            lines.append(f"  Observation: {observation}")

        error = str(step.get("error") or "").strip()
        if error:
            lines.append(f"  Error: {error}")

    return "\n".join(lines) if lines else "<empty>"


def _extract_tool_usage_summary(compact_run_details: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    summary = {
        "step_count": 0,
        "steps_with_tool_calls": 0,
        "tool_call_count": 0,
        "tool_names": [],
        "network_tool_names": [],
    }
    if not isinstance(compact_run_details, dict):
        return summary

    steps = compact_run_details.get("steps")
    if not isinstance(steps, list):
        return summary

    tool_names: List[str] = []
    network_tool_names: List[str] = []
    network_markers = ("http", "web", "browser", "search", "visit", "perplexity", "gemini")

    summary["step_count"] = len(steps)
    for step in steps:
        if not isinstance(step, dict):
            continue
        tool_calls = step.get("tool_calls")
        if not isinstance(tool_calls, list) or not tool_calls:
            continue
        summary["steps_with_tool_calls"] += 1
        summary["tool_call_count"] += len(tool_calls)
        for item in tool_calls:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            tool_names.append(name)
            lowered = name.lower()
            if any(marker in lowered for marker in network_markers):
                network_tool_names.append(name)

    summary["tool_names"] = sorted(set(tool_names))
    summary["network_tool_names"] = sorted(set(network_tool_names))
    return summary


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
    compact_run_details: Optional[Dict[str, Any]],
    reasoning_summary: Optional[Dict[str, Any]],
    runtime_metadata: Optional[Dict[str, Any]],
    tool_usage_summary: Optional[Dict[str, Any]],
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
        "runtime_metadata": runtime_metadata,
        "tool_usage_summary": tool_usage_summary,
    }

    step_summary_text = _render_step_summary(compact_run_details)

    content = (
        "=== TRACE METADATA ===\n"
        f"{json.dumps(trace_payload, ensure_ascii=False, indent=2)}\n\n"
        "=== QUESTION TEXT ===\n"
        f"{question.get('question_text', '')}\n\n"
        "=== REASONING SUMMARY ===\n"
        f"{json.dumps(reasoning_summary, ensure_ascii=False, indent=2) if reasoning_summary else '<empty>'}\n\n"
        "=== STEP SUMMARY ===\n"
        f"{step_summary_text}\n\n"
        "=== AGENT RUN DETAILS ===\n"
        f"{json.dumps(compact_run_details, ensure_ascii=False, indent=2) if compact_run_details else '<empty>'}\n\n"
        "=== AGENT STDOUT/STDERR TRACE ===\n"
        f"{captured_trace.strip() or '<empty>'}\n\n"
        "=== RAW MODEL ANSWER ===\n"
        f"{raw_answer or '<empty>'}\n\n"
        "=== NORMALIZED STUDENT ANSWER ===\n"
        f"{student_answer or '<empty>'}\n"
    )
    trace_log_path.write_text(content, encoding="utf-8")


def _extract_reasoning_summary(agent_run_details: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(agent_run_details, dict):
        return None

    steps = agent_run_details.get("steps")
    if not isinstance(steps, list):
        return None

    thoughts: List[str] = []
    provider_summaries: List[str] = []
    reasoning_tokens_total = 0
    step_count = 0

    for step in steps:
        if not isinstance(step, dict):
            continue
        if "step_number" not in step:
            continue
        step_count += 1

        model_output_message = step.get("model_output_message")
        if isinstance(model_output_message, dict):
            reasoning_tokens_total += _reasoning_tokens_from_model_output_message(model_output_message)
            content = model_output_message.get("content")
            if isinstance(content, dict):
                thought = str(content.get("thought") or "").strip()
                if thought:
                    thoughts.append(thought)
            elif isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    text = str(part.get("text") or "").strip()
                    if text and "thought" in text.lower():
                        thoughts.append(text)

            raw = model_output_message.get("raw")
            if isinstance(raw, dict):
                output_items = raw.get("output")
                if isinstance(output_items, list):
                    for item in output_items:
                        if not isinstance(item, dict):
                            continue
                        if item.get("type") != "reasoning":
                            continue
                        summary = item.get("summary")
                        if isinstance(summary, str) and summary.strip():
                            provider_summaries.append(summary.strip())
                        elif isinstance(summary, list):
                            parts: List[str] = []
                            for summary_item in summary:
                                if not isinstance(summary_item, dict):
                                    continue
                                text = str(summary_item.get("text") or "").strip()
                                if text:
                                    parts.append(text)
                            if parts:
                                provider_summaries.append(" ".join(parts))

                choices = raw.get("choices")
                if isinstance(choices, list) and choices:
                    first_choice = choices[0] if isinstance(choices[0], dict) else {}
                    message = first_choice.get("message") if isinstance(first_choice, dict) else None
                    if isinstance(message, dict):
                        reasoning = message.get("reasoning")
                        if isinstance(reasoning, str) and reasoning.strip():
                            provider_summaries.append(reasoning.strip())
                        elif isinstance(reasoning, dict):
                            summary_text = str(reasoning.get("summary") or "").strip()
                            if summary_text:
                                provider_summaries.append(summary_text)

    if not thoughts and not provider_summaries and reasoning_tokens_total == 0:
        return None

    return {
        "step_count": step_count,
        "thoughts": thoughts,
        "provider_reasoning_summaries": provider_summaries,
        "reasoning_tokens_total": reasoning_tokens_total,
        "note": (
            "Best-effort summary from visible traces only. "
            "Hidden provider chain-of-thought is not exposed."
        ),
    }


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


def _collect_run_details(
    runtime: AgentRuntime,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    run_details = runtime.get_last_run_details()
    return _compact_run_details(run_details), _extract_reasoning_summary(run_details)


def _collect_runtime_debug_payload(
    runtime: AgentRuntime,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Dict[str, Any]]:
    compact_run_details, reasoning_summary = _collect_run_details(runtime)
    tool_usage_summary = _extract_tool_usage_summary(compact_run_details)
    return compact_run_details, reasoning_summary, tool_usage_summary


def _maybe_apply_student_guard(
    *,
    question: Dict[str, Any],
    raw_answer: str,
    student_answer: str,
    model_name: str,
    model_url: str,
    api_key: Optional[str],
    student_guard_enabled: bool,
    guard_mode: str,
    student_guard_retries: int,
    student_guard_reasoning_effort: str,
) -> str:
    answer_type = question.get("answer_type", "")
    guard_on_failure = is_answer_invalid(answer_type, student_answer)
    should_run_guard = (
        student_guard_enabled
        and guard_mode != "off"
        and (guard_mode == "always" or guard_on_failure)
    )
    if not should_run_guard:
        return student_answer

    try:
        guard_output = run_student_guard(
            question=question,
            raw_answer=raw_answer,
            normalized_answer=student_answer,
            model_name=model_name,
            model_url=model_url,
            api_key=api_key,
            retries=student_guard_retries,
            reasoning_effort=student_guard_reasoning_effort,
        )
        guarded_answer = normalize_student_answer(
            answer_type,
            guard_output.final_answer,
        )
        if guarded_answer:
            return guarded_answer
        if guard_on_failure:
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

    return student_answer


def _build_student_result_row(
    *,
    question: Dict[str, Any],
    student_answer: str,
    student_status: str,
    student_error: str,
    elapsed_ms: int,
) -> Dict[str, Any]:
    return {
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
        except Exception as exc:
            if _is_model_limit_error(str(exc)):
                raise ModelLimitExceededError(f"Model limit exceeded: {str(exc)}") from exc
            status = "parse_error"
            if isinstance(exc, AgentRuntimeError):
                status = exc.status or "parse_error"
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
    timeout: int,
    max_retries: int = 3,
    max_error_len: int = 240,
    limit: Optional[int] = None,
    *,
    agent_max_steps: int = 6,
    agent_sandbox: str = "docker",
    agent_tools_profile: str = "full",
    agent_config: Optional[Path] = Path("scripts/agents/agent_config.yaml"),
    api_key: Optional[str] = None,
    student_guard_enabled: bool = True,
    student_guard_mode: str = "on_failure",
    student_guard_retries: int = 2,
    student_guard_reasoning_effort: str = "high",
    trace_log_enabled: bool = True,
    trace_log_dir: Optional[Path] = None,
    verbose_output_enabled: bool = False,
    verbose_output_path: Optional[Path] = None,
    resume_existing: bool = False,
):
    questions = load_benchmark_questions(benchmark_path=benchmark_path)
    if limit is not None and limit >= 0:
        questions = questions[:limit]

    try:
        runtime = AgentRuntime(
            model_url=model_url,
            model_name=model_name,
            api_key=api_key,
            config_path=agent_config,
            max_steps=agent_max_steps,
            sandbox=agent_sandbox,
            tools_profile=agent_tools_profile,
            timeout_sec=timeout,
        )
        runtime.preflight()
        runtime_metadata = runtime.get_runtime_metadata()
    except AgentRuntimeError as exc:
        raise RuntimeError(f"Agent preflight failed: {exc.status}: {exc.message}") from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to initialize agent runtime: {exc}") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_verbose_output_path = verbose_output_path or (output_path.parent / "student_output_verbose.jsonl")
    if verbose_output_enabled:
        resolved_verbose_output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_trace_log_dir = trace_log_dir or (output_path.parent / "traces")
    if trace_log_enabled:
        resolved_trace_log_dir.mkdir(parents=True, exist_ok=True)
    guard_mode = (student_guard_mode or "on_failure").strip().lower()
    if guard_mode not in {"on_failure", "always", "off"}:
        raise ValueError(f"Invalid student_guard_mode: {student_guard_mode}")

    completed_keys: Set[str] = set()
    if resume_existing and output_path.exists():
        completed_keys = _load_completed_keys_from_jsonl(output_path)

    try:
        output_mode = "a" if resume_existing else "w"
        verbose_context = (
            resolved_verbose_output_path.open(output_mode, encoding="utf-8")
            if verbose_output_enabled
            else nullcontext()
        )
        with output_path.open(output_mode, encoding="utf-8") as f_out, verbose_context as f_verbose:
            for line_idx, question in enumerate(
                tqdm(questions, total=len(questions), desc="Validating student answers"),
                start=1,
            ):
                question_key = build_result_key(question)
                if question_key in completed_keys:
                    continue
                started_at = time.perf_counter()
                student_status = STATUS_OK
                student_error = ""
                student_answer = ""
                raw_answer = ""
                compact_run_details: Optional[Dict[str, Any]] = None
                reasoning_summary: Optional[Dict[str, Any]] = None
                tool_usage_summary: Optional[Dict[str, Any]] = None
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
                        (
                            compact_run_details,
                            reasoning_summary,
                            tool_usage_summary,
                        ) = _collect_runtime_debug_payload(runtime)

                        student_answer = normalize_student_answer(
                            question.get("answer_type", ""),
                            raw_answer,
                        )
                        student_answer = _maybe_apply_student_guard(
                            question=question,
                            raw_answer=raw_answer,
                            student_answer=student_answer,
                            model_name=model_name,
                            model_url=model_url,
                            api_key=api_key,
                            student_guard_enabled=student_guard_enabled,
                            guard_mode=guard_mode,
                            student_guard_retries=student_guard_retries,
                            student_guard_reasoning_effort=student_guard_reasoning_effort,
                        )
                    except ModelLimitExceededError as exc:
                        raise ModelLimitExceededError(
                            "Model limit exceeded; aborting run. "
                            f"model={model_name}, line={line_idx}, "
                            f"exam_id={question.get('exam_id')}, "
                            f"page_id={question.get('page_id')}, "
                            f"question_id={question.get('question_id')}, "
                            f"reason={_sanitize_error(exc)}"
                        ) from exc
                    except StudentCallError as exc:
                        (
                            compact_run_details,
                            reasoning_summary,
                            tool_usage_summary,
                        ) = _collect_runtime_debug_payload(runtime)
                        student_status = exc.status
                        student_error = exc.message
                        student_answer = ""
                        tqdm.write(f"[ERROR] Line {line_idx}: status={student_status} error={student_error}")
                    except Exception as exc:  # Defensive fallback
                        (
                            compact_run_details,
                            reasoning_summary,
                            tool_usage_summary,
                        ) = _collect_runtime_debug_payload(runtime)
                        student_status = "parse_error"
                        student_error = _sanitize_error(exc, limit=max_error_len)
                        student_answer = ""
                        tqdm.write(f"[ERROR] Line {line_idx}: status={student_status} error={student_error}")

                elapsed_ms = int((time.perf_counter() - started_at) * 1000)

                if student_error:
                    student_error = student_error[:max_error_len]

                trace_log_path: Optional[Path] = None
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
                        compact_run_details=compact_run_details,
                        reasoning_summary=reasoning_summary,
                        runtime_metadata=runtime_metadata,
                        tool_usage_summary=tool_usage_summary,
                    )

                result = _build_student_result_row(
                    question=question,
                    student_answer=student_answer,
                    student_status=student_status,
                    student_error=student_error,
                    elapsed_ms=elapsed_ms,
                )

                f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                f_out.flush()
                completed_keys.add(question_key)

                if verbose_output_enabled and f_verbose is not None:
                    verbose_result = {
                        **result,
                        "raw_answer": raw_answer,
                        "reasoning_summary": reasoning_summary,
                        "runtime_metadata": runtime_metadata,
                        "tool_usage_summary": tool_usage_summary,
                        "agent_run_details": compact_run_details,
                        "trace_log_path": str(trace_log_path) if trace_log_path else None,
                    }
                    f_verbose.write(json.dumps(verbose_result, ensure_ascii=False) + "\n")
                    f_verbose.flush()
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
        "--student-guard-reasoning-effort",
        type=str,
        default="high",
        choices=["low", "medium", "high"],
        help="Reasoning effort for PydanticAI student guard (default: high)",
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
    parser.add_argument(
        "--verbose-output-enabled",
        type=_str_to_bool,
        default=False,
        help="Write extended student_output_verbose.jsonl with raw/model step context (default: false)",
    )
    parser.add_argument(
        "--verbose-output-path",
        type=Path,
        default=None,
        help="Path for extended verbose JSONL (default: <output-dir>/student_output_verbose.jsonl)",
    )
    parser.add_argument(
        "--resume-existing",
        type=_str_to_bool,
        default=False,
        help="Append only missing questions to an existing student output JSONL (default: false)",
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
            timeout=args.timeout,
            max_retries=args.max_retries,
            max_error_len=args.max_error_len,
            limit=args.limit,
            agent_max_steps=args.agent_max_steps,
            agent_sandbox=args.agent_sandbox,
            agent_tools_profile=args.agent_tools_profile,
            agent_config=args.agent_config,
            student_guard_enabled=args.student_guard_enabled,
            student_guard_mode=args.student_guard_mode,
            student_guard_retries=args.student_guard_retries,
            student_guard_reasoning_effort=args.student_guard_reasoning_effort,
            trace_log_enabled=args.trace_log_enabled,
            trace_log_dir=args.trace_log_dir,
            verbose_output_enabled=args.verbose_output_enabled,
            verbose_output_path=args.verbose_output_path,
            resume_existing=args.resume_existing,
        )
    except Exception as exc:
        raise SystemExit(f"[ERROR] {exc}") from exc
