"""CLI for student validation stage."""

from __future__ import annotations

import argparse
from pathlib import Path

from spectralix_benchmark.evaluation.pipeline.student_validation import (
    _resolve_model_url,
    _str_to_bool,
    run_benchmark_inference,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark student answers using an OpenShell-based agentic runtime"
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
        default="openshell",
        help="Agent executor type (default: openshell)",
    )
    parser.add_argument(
        "--agent-backend",
        type=str,
        default=None,
        help="Optional runtime backend selector (supported: openshell_worker)",
    )
    parser.add_argument(
        "--agent-tools-profile",
        type=str,
        default="minimal",
        help="Tools profile from config (default: minimal)",
    )
    parser.add_argument(
        "--agent-config",
        type=Path,
        default=Path("spectralix_benchmark/agents/agent_config.yaml"),
        help="Path to agent YAML config (default: spectralix_benchmark/agents/agent_config.yaml)",
    )
    parser.add_argument(
        "--agent-reasoning-effort",
        type=str,
        default=None,
        choices=["low", "medium", "high"],
        help="Optional override for agent model reasoning effort",
    )
    parser.add_argument(
        "--agent-sgr-enabled",
        type=_str_to_bool,
        default=True,
        help="Enable hidden SGR reasoning phase for student runtime (default: true)",
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
    parser.add_argument(
        "--fail-fast-error-streak",
        type=int,
        default=20,
        help=(
            "Abort after this many consecutive failed rows with empty answers and zero tokens "
            "(default: 20; set 0 to disable)"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        resolved_model_url = _resolve_model_url(args.model_url, args.api_base_url)
    except argparse.ArgumentTypeError as exc:
        raise SystemExit(f"[ERROR] {exc}") from exc

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
            agent_backend=args.agent_backend,
            agent_tools_profile=args.agent_tools_profile,
            agent_config=args.agent_config,
            agent_reasoning_effort=args.agent_reasoning_effort,
            agent_sgr_enabled=args.agent_sgr_enabled,
            student_guard_enabled=args.student_guard_enabled,
            student_guard_mode=args.student_guard_mode,
            student_guard_retries=args.student_guard_retries,
            student_guard_reasoning_effort=args.student_guard_reasoning_effort,
            trace_log_enabled=args.trace_log_enabled,
            trace_log_dir=args.trace_log_dir,
            verbose_output_enabled=args.verbose_output_enabled,
            verbose_output_path=args.verbose_output_path,
            resume_existing=args.resume_existing,
            fail_fast_error_streak=args.fail_fast_error_streak,
        )
    except Exception as exc:
        raise SystemExit(f"[ERROR] {exc}") from exc
