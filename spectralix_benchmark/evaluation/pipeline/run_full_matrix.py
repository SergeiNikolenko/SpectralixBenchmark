import argparse
import csv
import re
import subprocess
import math
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from spectralix_benchmark.agents.models import ensure_chat_completions_url
from spectralix_benchmark.evaluation.benchmark_taxonomy import compute_benchmark_taxonomy_metrics
from spectralix_benchmark.evaluation.io import read_json as _read_json
from spectralix_benchmark.evaluation.io import read_jsonl, write_json
from spectralix_benchmark.evaluation.pipeline.llm_judge import run_llm_judge
from spectralix_benchmark.evaluation.pipeline.student_validation import run_benchmark_inference

TRUTHY_VALUES = {"1", "true", "yes", "y", "on"}
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
    "model limit exceeded",
)

TOKEN_TRACE_PATTERN = re.compile(r"Input tokens: ([0-9,]+) \| Output tokens: ([0-9,]+)")
DEFAULT_PRICING_USD_PER_1M: Dict[str, Dict[str, float]] = {
    "gpt-5.4": {"input": 2.50, "output": 15.00},
    "gpt-5.4-mini": {"input": 0.250, "output": 2.000},
    "gpt-5": {"input": 1.250, "output": 10.000},
    "gpt-5-codex": {"input": 1.250, "output": 10.000},
    "gpt-5-codex-mini": {"input": 0.250, "output": 2.000},
    "gpt-5.1": {"input": 1.250, "output": 10.000},
    "gpt-5.1-codex": {"input": 1.250, "output": 10.000},
    "gpt-5.1-codex-mini": {"input": 0.250, "output": 2.000},
    "gpt-5.1-codex-max": {"input": 1.250, "output": 10.000},
    "gpt-5.2": {"input": 1.750, "output": 14.000},
    "gpt-5.2-codex": {"input": 1.750, "output": 14.000},
}


def _now_utc_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _detect_git_commit(repo_dir: Path) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            check=False,
            timeout=3,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    commit = (result.stdout or "").strip()
    return commit or None


def sanitize_model_name(model_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", model_name)


def _status(row: Dict[str, Any], key: str) -> str:
    return str(row.get(key, "ok") or "ok")


def _status_pair(row: Dict[str, Any]) -> tuple[str, str]:
    return _status(row, "student_status"), _status(row, "row_status")


def _is_ok_pair(student_status: str, row_status: str) -> bool:
    return student_status == "ok" and row_status == "ok"


def _is_truthy(value: Any) -> bool:
    return str(value).strip().lower() in TRUTHY_VALUES


def _is_model_limit_error(error_message: str) -> bool:
    text = str(error_message or "").strip().lower()
    if not text:
        return False
    return any(marker in text for marker in MODEL_LIMIT_ERROR_MARKERS)


def _raise_model_limit_exit(message: str, exc: Exception) -> None:
    if _is_model_limit_error(str(exc)):
        raise SystemExit(f"{message}; reason={str(exc)}") from exc


def _resolve_model_url(model_url: Optional[str], api_base_url: Optional[str]) -> str:
    raw_value = (model_url or api_base_url or "").strip()
    if not raw_value:
        raise ValueError("Either --model-url or --api-base-url must be provided")
    return ensure_chat_completions_url(raw_value)


def _safe_float(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return float(value)


def _resolve_pricing(
    *,
    model_name: str,
    input_price_override: Optional[float],
    output_price_override: Optional[float],
) -> Dict[str, Optional[float]]:
    default = DEFAULT_PRICING_USD_PER_1M.get(model_name, {})
    return {
        "input_per_1m_usd": _safe_float(
            input_price_override if input_price_override is not None else default.get("input")
        ),
        "output_per_1m_usd": _safe_float(
            output_price_override if output_price_override is not None else default.get("output")
        ),
    }


def _estimate_cost_usd(
    *,
    input_tokens: int,
    output_tokens: int,
    input_per_1m_usd: Optional[float],
    output_per_1m_usd: Optional[float],
) -> Optional[float]:
    if input_per_1m_usd is None or output_per_1m_usd is None:
        return None
    return (
        (float(input_tokens) / 1_000_000.0) * input_per_1m_usd
        + (float(output_tokens) / 1_000_000.0) * output_per_1m_usd
    )


def _collect_student_token_totals(trace_log_dir: Path) -> Dict[str, int]:
    totals = {
        "student_input_tokens_total": 0,
        "student_output_tokens_total": 0,
        "student_total_tokens_total": 0,
    }
    if not trace_log_dir.exists():
        return totals

    for trace_path in trace_log_dir.glob("*.log"):
        text = trace_path.read_text(encoding="utf-8", errors="ignore")
        for match in TOKEN_TRACE_PATTERN.finditer(text):
            input_tokens = int(match.group(1).replace(",", ""))
            output_tokens = int(match.group(2).replace(",", ""))
            totals["student_input_tokens_total"] += input_tokens
            totals["student_output_tokens_total"] += output_tokens

    totals["student_total_tokens_total"] = (
        totals["student_input_tokens_total"] + totals["student_output_tokens_total"]
    )
    return totals


def _collect_student_token_totals_from_rows(student_rows: List[Dict[str, Any]]) -> Dict[str, int]:
    input_total = int(sum(int(row.get("student_input_tokens") or 0) for row in student_rows))
    output_total = int(sum(int(row.get("student_output_tokens") or 0) for row in student_rows))
    total_total = int(sum(int(row.get("student_total_tokens") or 0) for row in student_rows))
    if total_total == 0:
        total_total = input_total + output_total
    return {
        "student_input_tokens_total": input_total,
        "student_output_tokens_total": output_total,
        "student_total_tokens_total": total_total,
    }


def _format_usd(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"${value:.4f}"


def _build_skipped_row(model_name: str) -> Dict[str, Any]:
    return {
        "model_name": model_name,
        "total_rows": 0,
        "overall_quality_total_score": 0.0,
        "overall_quality_max_score": 0.0,
        "overall_quality_score": None,
        "overall_end_to_end_score": None,
        "macro_depth_quality_score": None,
        "macro_depth_end_to_end_score": None,
        "auxiliary_grounding_quality_score": None,
        "auxiliary_grounding_end_to_end_score": None,
        "reliability_ok_count": 0,
        "reliability_ok_rate": None,
        "infra_error_count": {},
        "infra_error_rate": {},
        "student_input_tokens_total": None,
        "student_output_tokens_total": None,
        "student_total_tokens_total": None,
        "judge_input_tokens_total": None,
        "judge_output_tokens_total": None,
        "judge_total_tokens_total": None,
        "judge_reasoning_tokens_total": None,
        "judge_requests_total": None,
        "judge_tool_calls_total": None,
        "student_estimated_cost_usd": None,
        "judge_estimated_cost_usd": None,
        "estimated_cost_usd": None,
        "skipped": True,
    }


def _validate_resume_compatibility(
    *,
    existing_manifest: Optional[Dict[str, Any]],
    benchmark_path: Path,
    model_name: str,
    judge_model: str,
    agent_sandbox: str,
    agent_backend: Optional[str],
    agent_tools_profile: str,
    agent_config: Path,
) -> None:
    if not existing_manifest:
        return
    expected_pairs = {
        "benchmark_path": str(benchmark_path),
        "model_name": model_name,
    }
    settings = existing_manifest.get("settings") if isinstance(existing_manifest.get("settings"), dict) else {}
    actual_pairs = {
        "benchmark_path": str(existing_manifest.get("benchmark_path") or ""),
        "model_name": str(existing_manifest.get("model_name") or ""),
        "judge_model": str(settings.get("judge_model") or ""),
        "agent_sandbox": str(settings.get("agent_sandbox") or ""),
        "agent_backend": str(settings.get("agent_backend") or ""),
        "agent_tools_profile": str(settings.get("agent_tools_profile") or ""),
        "agent_config": str(settings.get("agent_config") or ""),
    }
    expected_pairs.update(
        {
            "judge_model": judge_model,
            "agent_sandbox": agent_sandbox,
            "agent_backend": str(agent_backend or ""),
            "agent_tools_profile": agent_tools_profile,
            "agent_config": str(agent_config),
        }
    )
    mismatches = {
        key: {"existing": actual_pairs.get(key), "requested": expected_pairs.get(key)}
        for key in expected_pairs
        if actual_pairs.get(key) and actual_pairs.get(key) != expected_pairs.get(key)
    }
    if mismatches:
        raise ValueError(
            "Resume requested for an incompatible run directory. "
            f"Use a fresh run_id or clear previous artifacts. mismatches={mismatches}"
        )


def _build_summary_row(
    *,
    model_name: str,
    metrics: Dict[str, Any],
    judge_model: str,
    metrics_path: Path,
) -> Dict[str, Any]:
    return {
        "model_name": model_name,
        **_summary_base_fields(metrics),
        "infra_error_count": metrics.get("infra_error_count"),
        "infra_error_rate": metrics.get("infra_error_rate"),
        "judge_model": judge_model,
        "metrics_path": str(metrics_path),
    }


def _summary_base_fields(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "total_rows": row.get("total_rows"),
        "overall_quality_total_score": row.get("overall_quality_total_score"),
        "overall_quality_max_score": row.get("overall_quality_max_score"),
        "overall_quality_score": row.get("overall_quality_score"),
        "overall_end_to_end_score": row.get("overall_end_to_end_score"),
        "macro_depth_quality_score": row.get("macro_depth_quality_score"),
        "macro_depth_end_to_end_score": row.get("macro_depth_end_to_end_score"),
        "auxiliary_grounding_quality_score": row.get("auxiliary_grounding_quality_score"),
        "auxiliary_grounding_end_to_end_score": row.get("auxiliary_grounding_end_to_end_score"),
        "student_input_tokens_total": row.get("student_input_tokens_total"),
        "student_output_tokens_total": row.get("student_output_tokens_total"),
        "student_total_tokens_total": row.get("student_total_tokens_total"),
        "judge_input_tokens_total": row.get("judge_input_tokens_total"),
        "judge_output_tokens_total": row.get("judge_output_tokens_total"),
        "judge_total_tokens_total": row.get("judge_total_tokens_total"),
        "judge_reasoning_tokens_total": row.get("judge_reasoning_tokens_total"),
        "judge_requests_total": row.get("judge_requests_total"),
        "judge_tool_calls_total": row.get("judge_tool_calls_total"),
        "student_estimated_cost_usd": row.get("student_estimated_cost_usd"),
        "judge_estimated_cost_usd": row.get("judge_estimated_cost_usd"),
        "estimated_cost_usd": row.get("estimated_cost_usd"),
        "reliability_ok_count": row.get("reliability_ok_count"),
        "reliability_ok_rate": row.get("reliability_ok_rate"),
    }


def compute_metrics(judge_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_rows = len(judge_rows)

    quality_rows = [
        row
        for row in judge_rows
        if row.get("row_status") == "ok" and row.get("final_score") is not None
    ]

    quality_total_score = float(sum(float(row.get("final_score", 0) or 0) for row in quality_rows))
    quality_max_score = float(sum(float(row.get("max_score", 0) or 0) for row in quality_rows))
    end_to_end_max_score = float(sum(float(row.get("max_score", 0) or 0) for row in judge_rows))
    overall_quality_score: Optional[float]
    if quality_max_score > 0:
        overall_quality_score = quality_total_score / quality_max_score
    else:
        overall_quality_score = None
    overall_end_to_end_score: Optional[float]
    if end_to_end_max_score > 0:
        overall_end_to_end_score = quality_total_score / end_to_end_max_score
    else:
        overall_end_to_end_score = None

    judge_input_tokens_total = int(sum(int(row.get("judge_input_tokens") or 0) for row in judge_rows))
    judge_output_tokens_total = int(sum(int(row.get("judge_output_tokens") or 0) for row in judge_rows))
    judge_total_tokens_total = int(sum(int(row.get("judge_total_tokens") or 0) for row in judge_rows))
    judge_reasoning_tokens_total = int(sum(int(row.get("judge_reasoning_tokens") or 0) for row in judge_rows))
    judge_requests_total = int(sum(int(row.get("judge_requests") or 0) for row in judge_rows))
    judge_tool_calls_total = int(sum(int(row.get("judge_tool_calls") or 0) for row in judge_rows))

    reliability_ok_count = 0
    infra_counts: Counter = Counter()

    for row in judge_rows:
        student_status, row_status = _status_pair(row)
        if _is_ok_pair(student_status, row_status):
            reliability_ok_count += 1
            continue

        if student_status != "ok":
            infra_counts[student_status] += 1
        elif row_status != "ok":
            infra_counts[row_status] += 1

    reliability_ok_rate = (reliability_ok_count / total_rows) if total_rows else None
    infra_error_rate = {
        key: value / total_rows for key, value in sorted(infra_counts.items())
    } if total_rows else {}

    taxonomy_metrics = compute_benchmark_taxonomy_metrics(judge_rows)

    errors_sample: List[Dict[str, Any]] = []
    for row in judge_rows:
        student_status, _ = _status_pair(row)
        if student_status == "ok":
            continue
        errors_sample.append(
            {
                "exam_id": row.get("exam_id"),
                "page_id": row.get("page_id"),
                "question_id": row.get("question_id"),
                "answer_type": row.get("answer_type"),
                "student_status": student_status,
                "student_error": str(row.get("student_error", ""))[:240],
                "question_text_preview": str(row.get("question_text", ""))[:160],
            }
        )

    return {
        "total_rows": total_rows,
        "overall_quality_total_score": quality_total_score,
        "overall_quality_max_score": quality_max_score,
        "overall_quality_score": overall_quality_score,
        "overall_end_to_end_score": overall_end_to_end_score,
        "judge_input_tokens_total": judge_input_tokens_total,
        "judge_output_tokens_total": judge_output_tokens_total,
        "judge_total_tokens_total": judge_total_tokens_total,
        "judge_reasoning_tokens_total": judge_reasoning_tokens_total,
        "judge_requests_total": judge_requests_total,
        "judge_tool_calls_total": judge_tool_calls_total,
        "reliability_ok_count": reliability_ok_count,
        "reliability_ok_rate": reliability_ok_rate,
        "infra_error_count": dict(sorted(infra_counts.items())),
        "infra_error_rate": infra_error_rate,
        "errors_sample": errors_sample,
        **taxonomy_metrics,
    }


def write_summary_csv(path: Path, summary_rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    infra_keys = sorted(
        {
            key
            for row in summary_rows
            for key in (row.get("infra_error_count") or {}).keys()
        }
    )

    fieldnames = [
        "model_name",
        "total_rows",
        "overall_quality_total_score",
        "overall_quality_max_score",
        "overall_quality_score",
        "overall_end_to_end_score",
        "macro_depth_quality_score",
        "macro_depth_end_to_end_score",
        "auxiliary_grounding_quality_score",
        "auxiliary_grounding_end_to_end_score",
        "student_input_tokens_total",
        "student_output_tokens_total",
        "student_total_tokens_total",
        "judge_input_tokens_total",
        "judge_output_tokens_total",
        "judge_total_tokens_total",
        "judge_reasoning_tokens_total",
        "judge_requests_total",
        "judge_tool_calls_total",
        "student_estimated_cost_usd",
        "judge_estimated_cost_usd",
        "estimated_cost_usd",
        "reliability_ok_count",
        "reliability_ok_rate",
    ] + [f"infra_error_count.{key}" for key in infra_keys]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            record = {
                "model_name": row.get("model_name"),
                **_summary_base_fields(row),
            }
            counts = row.get("infra_error_count") or {}
            for key in infra_keys:
                record[f"infra_error_count.{key}"] = counts.get(key, 0)
            writer.writerow(record)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full student inference + hybrid judge matrix")
    parser.add_argument(
        "--benchmark-path",
        type=Path,
        default=Path("benchmark/benchmark_v3_eval.jsonl"),
        help="Path to benchmark JSONL (default: benchmark/benchmark_v3_eval.jsonl)",
    )
    parser.add_argument(
        "--model-url",
        type=str,
        default=None,
        help="Student model API endpoint URL (chat completions endpoint)",
    )
    parser.add_argument(
        "--api-base-url",
        type=str,
        default=None,
        help="OpenAI-compatible API base URL (alternative to --model-url)",
    )
    parser.add_argument(
        "--student-models",
        "--models",
        nargs="+",
        dest="student_models",
        required=True,
        help="One or more student model names",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="gpt-5.2-codex",
        help="Fixed judge model for non-deterministic answer types (default: gpt-5.2-codex)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("runs"),
        help="Directory where run artifacts are written (default: runs)",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=datetime.now().strftime("%Y%m%d_%H%M%S"),
        help="Run identifier; used as runs/<run-id> (default: timestamp)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Student request timeout seconds (default: 120)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Student request retry attempts (default: 3)",
    )
    parser.add_argument(
        "--fail-fast-error-streak",
        type=int,
        default=20,
        help=(
            "Abort a student run after this many consecutive failed rows with empty answers and zero tokens "
            "(default: 20; set 0 to disable)"
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit of rows",
    )
    parser.add_argument(
        "--judge-max-tokens",
        type=int,
        default=220,
        help="Judge completion max tokens (default: 220)",
    )
    parser.add_argument(
        "--judge-temperature",
        type=float,
        default=0.0,
        help="Judge temperature (default: 0.0)",
    )
    parser.add_argument(
        "--judge-reasoning-effort",
        type=str,
        default="medium",
        choices=["low", "medium", "high"],
        help="Judge reasoning effort for LLM calls (default: medium)",
    )
    parser.add_argument(
        "--error-sample-size",
        type=int,
        default=10,
        help="How many technical error examples to save (default: 10)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Optional API key override for student runtime",
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
        help="Tools profile from agent config (default: minimal)",
    )
    parser.add_argument(
        "--agent-config",
        type=Path,
        default=Path("spectralix_benchmark/agents/agent_config.yaml"),
        help="Path to agent config YAML",
    )
    parser.add_argument(
        "--agent-reasoning-effort",
        type=str,
        default=None,
        choices=["low", "medium", "high"],
        help="Optional override for student agent reasoning effort",
    )
    parser.add_argument(
        "--agent-sgr-enabled",
        type=str,
        default="true",
        help="Enable hidden SGR reasoning phase in student runtime (true/false, default: true)",
    )
    parser.add_argument(
        "--student-guard-enabled",
        type=str,
        default="true",
        help="Enable PydanticAI student guard (true/false, default: true)",
    )
    parser.add_argument(
        "--student-guard-mode",
        type=str,
        default="on_failure",
        choices=["on_failure", "always", "off"],
        help="Student guard mode: on_failure|always|off (default: on_failure)",
    )
    parser.add_argument(
        "--student-guard-retries",
        type=int,
        default=2,
        help="Student guard retries (default: 2)",
    )
    parser.add_argument(
        "--trace-log-enabled",
        type=str,
        default="true",
        help="Write per-question student traces (true/false, default: true)",
    )
    parser.add_argument(
        "--trace-log-dir",
        type=Path,
        default=None,
        help="Optional base directory for student traces (per model subdir will be used)",
    )
    parser.add_argument(
        "--verbose-output-enabled",
        type=str,
        default="false",
        help="Write per-model student_output_verbose.jsonl with extended context (true/false, default: false)",
    )
    parser.add_argument(
        "--verbose-output-dir",
        type=Path,
        default=None,
        help="Optional base directory for per-model verbose JSONL outputs",
    )
    parser.add_argument(
        "--resume-existing",
        type=str,
        default="false",
        help="Resume from existing student/judge JSONL files instead of overwriting them (true/false, default: false)",
    )
    parser.add_argument(
        "--judge-structured-retries",
        type=int,
        default=2,
        help="Structured judge retries (default: 2)",
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
        type=str,
        default="true",
        help="Fallback from g_eval to structured judge on error (true/false, default: true)",
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
    parser.add_argument(
        "--student-input-price-per-1m-usd",
        type=float,
        default=None,
        help="Optional override for student input token price in USD per 1M tokens",
    )
    parser.add_argument(
        "--student-output-price-per-1m-usd",
        type=float,
        default=None,
        help="Optional override for student output token price in USD per 1M tokens",
    )
    parser.add_argument(
        "--judge-input-price-per-1m-usd",
        type=float,
        default=None,
        help="Optional override for judge input token price in USD per 1M tokens",
    )
    parser.add_argument(
        "--judge-output-price-per-1m-usd",
        type=float,
        default=None,
        help="Optional override for judge output token price in USD per 1M tokens",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        model_url = _resolve_model_url(args.model_url, args.api_base_url)
    except ValueError as exc:
        raise SystemExit(f"[ERROR] {exc}") from exc

    run_dir = args.output_root / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parents[2]
    git_commit = _detect_git_commit(repo_root)

    summary_rows: List[Dict[str, Any]] = []
    student_guard_enabled = _is_truthy(args.student_guard_enabled)
    trace_log_enabled = _is_truthy(args.trace_log_enabled)
    verbose_output_enabled = _is_truthy(args.verbose_output_enabled)
    judge_g_eval_fallback_structured = _is_truthy(args.judge_g_eval_fallback_structured)
    resume_existing = _is_truthy(args.resume_existing)

    run_manifest_path = run_dir / "run_manifest.json"
    run_manifest: Dict[str, Any] = {
        "run_id": args.run_id,
        "status": "running",
        "started_at_utc": _now_utc_iso(),
        "updated_at_utc": _now_utc_iso(),
        "finished_at_utc": None,
        "git_commit": git_commit,
        "benchmark_path": str(args.benchmark_path),
        "output_root": str(args.output_root),
        "run_dir": str(run_dir),
        "student_models": list(args.student_models),
        "judge_model": args.judge_model,
        "model_url": args.model_url,
        "api_base_url": args.api_base_url,
        "judge_model_url": args.judge_model_url,
        "settings": {
            "timeout": args.timeout,
            "max_retries": args.max_retries,
            "limit": args.limit,
            "agent_max_steps": args.agent_max_steps,
            "agent_sandbox": args.agent_sandbox,
            "agent_backend": args.agent_backend,
            "agent_tools_profile": args.agent_tools_profile,
            "agent_config": str(args.agent_config),
            "judge_max_tokens": args.judge_max_tokens,
            "judge_temperature": args.judge_temperature,
            "judge_reasoning_effort": args.judge_reasoning_effort,
            "judge_structured_retries": args.judge_structured_retries,
            "judge_method": args.judge_method,
            "judge_g_eval_fallback_structured": judge_g_eval_fallback_structured,
            "student_guard_enabled": student_guard_enabled,
            "student_guard_mode": args.student_guard_mode,
            "student_guard_retries": args.student_guard_retries,
            "trace_log_enabled": trace_log_enabled,
            "trace_log_dir": str(args.trace_log_dir) if args.trace_log_dir else None,
            "verbose_output_enabled": verbose_output_enabled,
            "verbose_output_dir": str(args.verbose_output_dir) if args.verbose_output_dir else None,
            "resume_existing": resume_existing,
        },
        "models": [],
    }
    write_json(run_manifest_path, run_manifest)

    def _record_model_manifest(model_manifest: Dict[str, Any], model_manifest_path: Path) -> None:
        write_json(model_manifest_path, model_manifest)
        model_entry = {
            "model_name": model_manifest.get("model_name"),
            "model_slug": model_manifest.get("model_slug"),
            "status": model_manifest.get("status"),
            "student_rows": model_manifest.get("student_rows"),
            "judge_rows": model_manifest.get("judge_rows"),
            "macro_depth_quality_score": model_manifest.get("macro_depth_quality_score"),
            "macro_depth_end_to_end_score": model_manifest.get("macro_depth_end_to_end_score"),
            "auxiliary_grounding_quality_score": model_manifest.get("auxiliary_grounding_quality_score"),
            "auxiliary_grounding_end_to_end_score": model_manifest.get("auxiliary_grounding_end_to_end_score"),
            "overall_quality_score": model_manifest.get("overall_quality_score"),
            "overall_end_to_end_score": model_manifest.get("overall_end_to_end_score"),
            "reliability_ok_rate": model_manifest.get("reliability_ok_rate"),
            "model_manifest_path": str(model_manifest_path),
        }
        existing_models = run_manifest.get("models") or []
        replaced = False
        for idx, existing in enumerate(existing_models):
            if existing.get("model_slug") == model_entry["model_slug"]:
                existing_models[idx] = model_entry
                replaced = True
                break
        if not replaced:
            existing_models.append(model_entry)
        run_manifest["models"] = existing_models
        run_manifest["updated_at_utc"] = _now_utc_iso()
        write_json(run_manifest_path, run_manifest)

    try:
        for model_name in args.student_models:
            model_slug = sanitize_model_name(model_name)
            model_dir = run_dir / model_slug
            model_dir.mkdir(parents=True, exist_ok=True)

            student_output_path = model_dir / "student_output.jsonl"
            judge_output_path = model_dir / "llm_judge_output.jsonl"
            metrics_path = model_dir / "metrics.json"
            breakdown_suite_path = model_dir / "breakdown_by_suite.json"
            breakdown_subtrack_path = model_dir / "breakdown_by_subtrack.json"
            breakdown_task_subtype_path = model_dir / "breakdown_by_task_subtype.json"
            breakdown_task_mode_path = model_dir / "breakdown_by_task_mode.json"
            breakdown_planning_horizon_path = model_dir / "breakdown_by_planning_horizon.json"
            errors_sample_path = model_dir / "errors_sample.json"

            print(f"[INFO] Running student inference for model={model_name}")
            trace_log_dir = (args.trace_log_dir / model_slug) if args.trace_log_dir else (model_dir / "traces")
            verbose_output_path = (
                (args.verbose_output_dir / model_slug / "student_output_verbose.jsonl")
                if args.verbose_output_dir
                else (model_dir / "student_output_verbose.jsonl")
            )
            model_manifest_path = model_dir / "run_manifest.json"
            if resume_existing:
                _validate_resume_compatibility(
                    existing_manifest=_read_json(model_manifest_path),
                    benchmark_path=args.benchmark_path,
                    model_name=model_name,
                    judge_model=args.judge_model,
                    agent_sandbox=args.agent_sandbox,
                    agent_backend=args.agent_backend,
                    agent_tools_profile=args.agent_tools_profile,
                    agent_config=args.agent_config,
                )
            model_manifest: Dict[str, Any] = {
                "run_id": args.run_id,
                "model_name": model_name,
                "model_slug": model_slug,
                "status": "running",
                "started_at_utc": _now_utc_iso(),
                "updated_at_utc": _now_utc_iso(),
                "finished_at_utc": None,
                "git_commit": git_commit,
                "benchmark_path": str(args.benchmark_path),
                "student_output_path": str(student_output_path),
                "judge_output_path": str(judge_output_path),
                "metrics_path": str(metrics_path),
                "breakdown_by_suite_path": str(breakdown_suite_path),
                "breakdown_by_subtrack_path": str(breakdown_subtrack_path),
                "breakdown_by_task_subtype_path": str(breakdown_task_subtype_path),
                "breakdown_by_task_mode_path": str(breakdown_task_mode_path),
                "breakdown_by_planning_horizon_path": str(breakdown_planning_horizon_path),
                "errors_sample_path": str(errors_sample_path),
                "trace_log_dir": str(trace_log_dir),
                "verbose_output_path": str(verbose_output_path),
                "settings": {
                    "agent_sandbox": args.agent_sandbox,
                    "agent_backend": args.agent_backend,
                    "agent_tools_profile": args.agent_tools_profile,
                    "agent_config": str(args.agent_config),
                    "agent_reasoning_effort": args.agent_reasoning_effort,
                    "agent_sgr_enabled": _is_truthy(args.agent_sgr_enabled),
                    "fail_fast_error_streak": args.fail_fast_error_streak,
                    "judge_model": args.judge_model,
                    "judge_reasoning_effort": args.judge_reasoning_effort,
                    "resume_existing": resume_existing,
                    "trace_log_enabled": trace_log_enabled,
                },
                "stages": {
                    "student": {"status": "running", "started_at_utc": _now_utc_iso(), "finished_at_utc": None},
                    "judge": {"status": "pending", "started_at_utc": None, "finished_at_utc": None},
                    "metrics": {"status": "pending", "started_at_utc": None, "finished_at_utc": None},
                },
            }
            _record_model_manifest(model_manifest, model_manifest_path)

            try:
                run_benchmark_inference(
                    benchmark_path=args.benchmark_path,
                    output_path=student_output_path,
                    model_url=model_url,
                    model_name=model_name,
                    api_key=args.api_key,
                    timeout=args.timeout,
                    max_retries=args.max_retries,
                    limit=args.limit,
                    agent_max_steps=args.agent_max_steps,
                    agent_sandbox=args.agent_sandbox,
                    agent_backend=args.agent_backend,
                    agent_tools_profile=args.agent_tools_profile,
                    agent_config=args.agent_config,
                    agent_reasoning_effort=args.agent_reasoning_effort,
                    agent_sgr_enabled=_is_truthy(args.agent_sgr_enabled),
                    student_guard_enabled=student_guard_enabled,
                    student_guard_mode=args.student_guard_mode,
                    student_guard_retries=args.student_guard_retries,
                    trace_log_enabled=trace_log_enabled,
                    trace_log_dir=trace_log_dir,
                    verbose_output_enabled=verbose_output_enabled,
                    verbose_output_path=verbose_output_path,
                    resume_existing=resume_existing,
                    fail_fast_error_streak=args.fail_fast_error_streak,
                )
            except Exception as exc:
                model_manifest["status"] = "failed"
                model_manifest["updated_at_utc"] = _now_utc_iso()
                model_manifest["finished_at_utc"] = _now_utc_iso()
                model_manifest["stages"]["student"]["status"] = "failed"
                model_manifest["stages"]["student"]["finished_at_utc"] = _now_utc_iso()
                model_manifest["error"] = str(exc)[:500]
                _record_model_manifest(model_manifest, model_manifest_path)
                _raise_model_limit_exit(
                    "[ERROR] Model limit exceeded during student stage; run aborted. "
                    f"model={model_name}",
                    exc,
                )
                raise

            model_manifest["stages"]["student"]["status"] = "completed"
            model_manifest["stages"]["student"]["finished_at_utc"] = _now_utc_iso()
            model_manifest["stages"]["judge"]["status"] = "running"
            model_manifest["stages"]["judge"]["started_at_utc"] = _now_utc_iso()
            model_manifest["updated_at_utc"] = _now_utc_iso()
            _record_model_manifest(model_manifest, model_manifest_path)

            student_rows = read_jsonl(student_output_path)
            model_manifest["student_rows"] = len(student_rows)
            model_manifest["updated_at_utc"] = _now_utc_iso()
            _record_model_manifest(model_manifest, model_manifest_path)
            if not student_rows:
                print(
                    f"[WARN] No rows were produced for model={model_name}. "
                    "Skipping judging and metrics."
                )
                skipped_row = _build_skipped_row(model_name)
                write_json(metrics_path, skipped_row)
                write_json(breakdown_suite_path, {})
                write_json(breakdown_subtrack_path, {})
                write_json(breakdown_task_subtype_path, {})
                write_json(breakdown_task_mode_path, {})
                write_json(breakdown_planning_horizon_path, {})
                write_json(errors_sample_path, [])
                summary_rows.append(skipped_row)
                model_manifest["stages"]["judge"]["status"] = "skipped"
                model_manifest["stages"]["judge"]["finished_at_utc"] = _now_utc_iso()
                model_manifest["stages"]["metrics"]["status"] = "completed"
                model_manifest["stages"]["metrics"]["started_at_utc"] = _now_utc_iso()
                model_manifest["stages"]["metrics"]["finished_at_utc"] = _now_utc_iso()
                model_manifest["status"] = "completed"
                model_manifest["finished_at_utc"] = _now_utc_iso()
                model_manifest["updated_at_utc"] = _now_utc_iso()
                _record_model_manifest(model_manifest, model_manifest_path)
                continue

            print(
                f"[INFO] Running hybrid judge for model={model_name} "
                f"with fixed judge={args.judge_model}"
            )
            try:
                run_llm_judge(
                    input_path=student_output_path,
                    gold_path=args.benchmark_path,
                    output_path=judge_output_path,
                    model_name=args.judge_model,
                    max_tokens=args.judge_max_tokens,
                    temperature=args.judge_temperature,
                    reasoning_effort=args.judge_reasoning_effort,
                    judge_structured_retries=args.judge_structured_retries,
                    judge_method=args.judge_method,
                    judge_g_eval_fallback_structured=judge_g_eval_fallback_structured,
                    judge_model_url=args.judge_model_url,
                    judge_api_key=args.judge_api_key or args.api_key,
                    trace_log_enabled=trace_log_enabled,
                    trace_log_dir=trace_log_dir,
                    resume_existing=resume_existing,
                )
            except Exception as exc:
                model_manifest["status"] = "failed"
                model_manifest["updated_at_utc"] = _now_utc_iso()
                model_manifest["finished_at_utc"] = _now_utc_iso()
                model_manifest["stages"]["judge"]["status"] = "failed"
                model_manifest["stages"]["judge"]["finished_at_utc"] = _now_utc_iso()
                model_manifest["error"] = str(exc)[:500]
                _record_model_manifest(model_manifest, model_manifest_path)
                _raise_model_limit_exit(
                    "[ERROR] Model limit exceeded during judge stage; run aborted. "
                    f"student_model={model_name}; judge_model={args.judge_model}",
                    exc,
                )
                raise

            model_manifest["stages"]["judge"]["status"] = "completed"
            model_manifest["stages"]["judge"]["finished_at_utc"] = _now_utc_iso()
            model_manifest["stages"]["metrics"]["status"] = "running"
            model_manifest["stages"]["metrics"]["started_at_utc"] = _now_utc_iso()
            model_manifest["updated_at_utc"] = _now_utc_iso()
            _record_model_manifest(model_manifest, model_manifest_path)

            judge_rows = read_jsonl(judge_output_path)
            model_manifest["judge_rows"] = len(judge_rows)
            metrics = compute_metrics(judge_rows)
            student_token_totals = _collect_student_token_totals_from_rows(student_rows)
            if (
                student_token_totals["student_input_tokens_total"] == 0
                and student_token_totals["student_output_tokens_total"] == 0
            ):
                student_token_totals = _collect_student_token_totals(trace_log_dir)
            student_pricing = _resolve_pricing(
                model_name=model_name,
                input_price_override=args.student_input_price_per_1m_usd,
                output_price_override=args.student_output_price_per_1m_usd,
            )
            judge_pricing = _resolve_pricing(
                model_name=args.judge_model,
                input_price_override=args.judge_input_price_per_1m_usd,
                output_price_override=args.judge_output_price_per_1m_usd,
            )
            student_estimated_cost_usd = _estimate_cost_usd(
                input_tokens=int(student_token_totals.get("student_input_tokens_total") or 0),
                output_tokens=int(student_token_totals.get("student_output_tokens_total") or 0),
                input_per_1m_usd=student_pricing["input_per_1m_usd"],
                output_per_1m_usd=student_pricing["output_per_1m_usd"],
            )
            judge_estimated_cost_usd = _estimate_cost_usd(
                input_tokens=int(metrics.get("judge_input_tokens_total") or 0),
                output_tokens=int(metrics.get("judge_output_tokens_total") or 0),
                input_per_1m_usd=judge_pricing["input_per_1m_usd"],
                output_per_1m_usd=judge_pricing["output_per_1m_usd"],
            )
            estimated_cost_usd = None
            if student_estimated_cost_usd is not None and judge_estimated_cost_usd is not None:
                estimated_cost_usd = student_estimated_cost_usd + judge_estimated_cost_usd
            metrics.update(
                {
                    "model_name": model_name,
                    "judge_model": args.judge_model,
                    "student_output_path": str(student_output_path),
                    "judge_output_path": str(judge_output_path),
                    **student_token_totals,
                    "student_pricing_usd_per_1m": student_pricing,
                    "judge_pricing_usd_per_1m": judge_pricing,
                    "student_estimated_cost_usd": student_estimated_cost_usd,
                    "judge_estimated_cost_usd": judge_estimated_cost_usd,
                    "estimated_cost_usd": estimated_cost_usd,
                }
            )

            top_infra = sorted(
                (metrics.get("infra_error_count") or {}).items(),
                key=lambda kv: kv[1],
                reverse=True,
            )[:3]
            if top_infra:
                print(f"[INFO] Top infra errors for model={model_name}: {top_infra}")
            print(
                "[INFO] Estimated run cost for "
                f"model={model_name}: student={_format_usd(student_estimated_cost_usd)}, "
                f"judge={_format_usd(judge_estimated_cost_usd)}, "
                f"total={_format_usd(estimated_cost_usd)}"
            )

            write_json(metrics_path, metrics)
            write_json(breakdown_suite_path, metrics.get("breakdown_by_suite") or {})
            write_json(breakdown_subtrack_path, metrics.get("breakdown_by_subtrack") or {})
            write_json(
                breakdown_task_subtype_path,
                metrics.get("breakdown_by_task_subtype") or {},
            )
            write_json(breakdown_task_mode_path, metrics.get("breakdown_by_task_mode") or {})
            write_json(
                breakdown_planning_horizon_path,
                metrics.get("breakdown_by_planning_horizon") or {},
            )
            write_json(
                errors_sample_path,
                (metrics.get("errors_sample") or [])[: max(args.error_sample_size, 0)],
            )
            model_manifest["stages"]["metrics"]["status"] = "completed"
            model_manifest["stages"]["metrics"]["finished_at_utc"] = _now_utc_iso()
            model_manifest["status"] = "completed"
            model_manifest["finished_at_utc"] = _now_utc_iso()
            model_manifest["updated_at_utc"] = _now_utc_iso()
            model_manifest["macro_depth_quality_score"] = metrics.get("macro_depth_quality_score")
            model_manifest["macro_depth_end_to_end_score"] = metrics.get("macro_depth_end_to_end_score")
            model_manifest["auxiliary_grounding_quality_score"] = metrics.get(
                "auxiliary_grounding_quality_score"
            )
            model_manifest["auxiliary_grounding_end_to_end_score"] = metrics.get(
                "auxiliary_grounding_end_to_end_score"
            )
            model_manifest["overall_quality_score"] = metrics.get("overall_quality_score")
            model_manifest["overall_end_to_end_score"] = metrics.get("overall_end_to_end_score")
            model_manifest["reliability_ok_rate"] = metrics.get("reliability_ok_rate")
            model_manifest["student_input_tokens_total"] = metrics.get("student_input_tokens_total")
            model_manifest["student_output_tokens_total"] = metrics.get("student_output_tokens_total")
            model_manifest["student_total_tokens_total"] = metrics.get("student_total_tokens_total")
            model_manifest["judge_input_tokens_total"] = metrics.get("judge_input_tokens_total")
            model_manifest["judge_output_tokens_total"] = metrics.get("judge_output_tokens_total")
            model_manifest["judge_total_tokens_total"] = metrics.get("judge_total_tokens_total")
            model_manifest["student_estimated_cost_usd"] = metrics.get("student_estimated_cost_usd")
            model_manifest["judge_estimated_cost_usd"] = metrics.get("judge_estimated_cost_usd")
            model_manifest["estimated_cost_usd"] = metrics.get("estimated_cost_usd")
            _record_model_manifest(model_manifest, model_manifest_path)

            summary_rows.append(
                _build_summary_row(
                    model_name=model_name,
                    metrics=metrics,
                    judge_model=args.judge_model,
                    metrics_path=metrics_path,
                )
            )
    except Exception:
        run_manifest["status"] = "failed"
        run_manifest["finished_at_utc"] = _now_utc_iso()
        run_manifest["updated_at_utc"] = _now_utc_iso()
        write_json(run_manifest_path, run_manifest)
        raise

    summary_json_path = run_dir / "summary.json"
    summary_csv_path = run_dir / "summary.csv"

    write_json(summary_json_path, summary_rows)
    write_summary_csv(summary_csv_path, summary_rows)
    run_manifest["status"] = "completed"
    run_manifest["finished_at_utc"] = _now_utc_iso()
    run_manifest["updated_at_utc"] = _now_utc_iso()
    run_manifest["summary_json_path"] = str(summary_json_path)
    run_manifest["summary_csv_path"] = str(summary_csv_path)
    write_json(run_manifest_path, run_manifest)

    print(f"[INFO] Summary JSON written to: {summary_json_path}")
    print(f"[INFO] Summary CSV written to: {summary_csv_path}")
    print(f"[INFO] Run manifest written to: {run_manifest_path}")


if __name__ == "__main__":
    main()
