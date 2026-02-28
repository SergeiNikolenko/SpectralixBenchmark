import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import sys

# Ensure repository root is importable when script is run as a file.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from scripts.evaluation.student_validation import run_benchmark_inference
except Exception:  # pragma: no cover - script execution fallback
    from student_validation import run_benchmark_inference
from scripts.agents.models import ensure_chat_completions_url

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


def sanitize_model_name(model_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", model_name)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


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


def _build_skipped_row(model_name: str) -> Dict[str, Any]:
    return {
        "model_name": model_name,
        "total_rows": 0,
        "quality_total_score": 0.0,
        "quality_max_score": 0.0,
        "quality_normalized_score": None,
        "reliability_ok_count": 0,
        "reliability_ok_rate": None,
        "infra_error_count": {},
        "infra_error_rate": {},
        "skipped": True,
    }


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
        "quality_total_score": row.get("quality_total_score"),
        "quality_max_score": row.get("quality_max_score"),
        "quality_normalized_score": row.get("quality_normalized_score"),
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
    quality_normalized_score: Optional[float]
    if quality_max_score > 0:
        quality_normalized_score = quality_total_score / quality_max_score
    else:
        quality_normalized_score = None

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

    breakdown: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "count": 0,
            "quality_count": 0,
            "quality_total_score": 0.0,
            "quality_max_score": 0.0,
            "ok_count": 0,
        }
    )

    for row in judge_rows:
        answer_type = str(row.get("answer_type", "unknown") or "unknown")
        bucket = breakdown[answer_type]
        bucket["count"] += 1

        student_status, row_status = _status_pair(row)
        if _is_ok_pair(student_status, row_status):
            bucket["ok_count"] += 1

        if row_status == "ok" and row.get("final_score") is not None:
            bucket["quality_count"] += 1
            bucket["quality_total_score"] += float(row.get("final_score", 0) or 0)
            bucket["quality_max_score"] += float(row.get("max_score", 0) or 0)

    breakdown_by_answer_type: Dict[str, Dict[str, Any]] = {}
    for answer_type, values in sorted(breakdown.items()):
        quality_max = float(values["quality_max_score"])
        if quality_max > 0:
            quality_norm = float(values["quality_total_score"]) / quality_max
        else:
            quality_norm = None

        count = int(values["count"])
        ok_rate = (int(values["ok_count"]) / count) if count else None
        breakdown_by_answer_type[answer_type] = {
            "count": count,
            "quality_count": int(values["quality_count"]),
            "quality_total_score": float(values["quality_total_score"]),
            "quality_max_score": quality_max,
            "quality_normalized_score": quality_norm,
            "ok_rate": ok_rate,
        }

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
        "quality_total_score": quality_total_score,
        "quality_max_score": quality_max_score,
        "quality_normalized_score": quality_normalized_score,
        "reliability_ok_count": reliability_ok_count,
        "reliability_ok_rate": reliability_ok_rate,
        "infra_error_count": dict(sorted(infra_counts.items())),
        "infra_error_rate": infra_error_rate,
        "breakdown_by_answer_type": breakdown_by_answer_type,
        "errors_sample": errors_sample,
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
        "quality_total_score",
        "quality_max_score",
        "quality_normalized_score",
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
        default=Path("benchmark/benchmark_v1_0.jsonl"),
        help="Path to benchmark JSONL (default: benchmark/benchmark_v1_0.jsonl)",
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
        "--max-tokens",
        type=int,
        default=256,
        help="Student generation max tokens (default: 256)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.5,
        help="Student generation temperature (default: 0.5)",
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
        "--workers",
        type=int,
        default=1,
        help="Reserved for student inference parallelism tuning (default: 1)",
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
        default=300,
        help="Judge completion max tokens (default: 300)",
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
        default="high",
        choices=["low", "medium", "high"],
        help="Judge reasoning effort for LLM calls (default: high)",
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
        "--agent-enabled",
        type=str,
        default="true",
        help="Enable smolagents runtime for student inference (true/false, default: true)",
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
        help="Tools profile from agent config (default: full)",
    )
    parser.add_argument(
        "--agent-config",
        type=Path,
        default=Path("scripts/agents/agent_config.yaml"),
        help="Path to agent config YAML",
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
        "--judge-structured-enabled",
        type=str,
        default="true",
        help="Enable structured PydanticAI judge (true/false, default: true)",
    )
    parser.add_argument(
        "--judge-structured-retries",
        type=int,
        default=2,
        help="Structured judge retries (default: 2)",
    )
    parser.add_argument(
        "--judge-structured-fallback-legacy",
        type=str,
        default="true",
        help="Fallback to legacy JSON judge parser on structured failure (true/false, default: true)",
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


def main() -> None:
    args = parse_args()
    try:
        from llm_judge import run_llm_judge
    except ImportError as exc:
        raise SystemExit(
            "[ERROR] Missing dependency for judge stage. Install with: pip install openai"
        ) from exc
    try:
        model_url = _resolve_model_url(args.model_url, args.api_base_url)
    except ValueError as exc:
        raise SystemExit(f"[ERROR] {exc}") from exc

    run_dir = args.output_root / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, Any]] = []
    agent_enabled = _is_truthy(args.agent_enabled)
    student_guard_enabled = _is_truthy(args.student_guard_enabled)
    trace_log_enabled = _is_truthy(args.trace_log_enabled)
    verbose_output_enabled = _is_truthy(args.verbose_output_enabled)
    judge_structured_enabled = _is_truthy(args.judge_structured_enabled)
    judge_structured_fallback_legacy = _is_truthy(args.judge_structured_fallback_legacy)

    for model_name in args.student_models:
        model_slug = sanitize_model_name(model_name)
        model_dir = run_dir / model_slug
        model_dir.mkdir(parents=True, exist_ok=True)

        student_output_path = model_dir / "student_output.jsonl"
        judge_output_path = model_dir / "llm_judge_output.jsonl"
        metrics_path = model_dir / "metrics.json"
        breakdown_path = model_dir / "breakdown_by_answer_type.json"
        errors_sample_path = model_dir / "errors_sample.json"

        print(f"[INFO] Running student inference for model={model_name}")
        trace_log_dir = (args.trace_log_dir / model_slug) if args.trace_log_dir else (model_dir / "traces")
        verbose_output_path = (
            (args.verbose_output_dir / model_slug / "student_output_verbose.jsonl")
            if args.verbose_output_dir
            else (model_dir / "student_output_verbose.jsonl")
        )
        try:
            run_benchmark_inference(
                benchmark_path=args.benchmark_path,
                output_path=student_output_path,
                model_url=model_url,
                model_name=model_name,
                api_key=args.api_key,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                timeout=args.timeout,
                max_retries=args.max_retries,
                limit=args.limit,
                workers=args.workers,
                agent_enabled=agent_enabled,
                agent_max_steps=args.agent_max_steps,
                agent_sandbox=args.agent_sandbox,
                agent_tools_profile=args.agent_tools_profile,
                agent_config=args.agent_config,
                student_guard_enabled=student_guard_enabled,
                student_guard_mode=args.student_guard_mode,
                student_guard_retries=args.student_guard_retries,
                trace_log_enabled=trace_log_enabled,
                trace_log_dir=trace_log_dir,
                verbose_output_enabled=verbose_output_enabled,
                verbose_output_path=verbose_output_path,
            )
        except Exception as exc:
            _raise_model_limit_exit(
                "[ERROR] Model limit exceeded during student stage; run aborted. "
                f"model={model_name}",
                exc,
            )
            raise

        student_rows = read_jsonl(student_output_path)
        if not student_rows:
            print(
                f"[WARN] No rows were produced for model={model_name}. "
                "Skipping judging and metrics."
            )
            skipped_row = _build_skipped_row(model_name)
            write_json(metrics_path, skipped_row)
            write_json(breakdown_path, {})
            write_json(errors_sample_path, [])
            summary_rows.append(skipped_row)
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
                judge_structured_enabled=judge_structured_enabled,
                judge_structured_retries=args.judge_structured_retries,
                judge_structured_fallback_legacy=judge_structured_fallback_legacy,
                judge_model_url=args.judge_model_url,
                judge_api_key=args.judge_api_key or args.api_key,
            )
        except Exception as exc:
            _raise_model_limit_exit(
                "[ERROR] Model limit exceeded during judge stage; run aborted. "
                f"student_model={model_name}; judge_model={args.judge_model}",
                exc,
            )
            raise

        judge_rows = read_jsonl(judge_output_path)
        metrics = compute_metrics(judge_rows)
        metrics.update(
            {
                "model_name": model_name,
                "judge_model": args.judge_model,
                "student_output_path": str(student_output_path),
                "judge_output_path": str(judge_output_path),
            }
        )

        top_infra = sorted(
            (metrics.get("infra_error_count") or {}).items(),
            key=lambda kv: kv[1],
            reverse=True,
        )[:3]
        if top_infra:
            print(f"[INFO] Top infra errors for model={model_name}: {top_infra}")

        write_json(metrics_path, metrics)
        write_json(breakdown_path, metrics.get("breakdown_by_answer_type") or {})
        write_json(
            errors_sample_path,
            (metrics.get("errors_sample") or [])[: max(args.error_sample_size, 0)],
        )

        summary_rows.append(
            _build_summary_row(
                model_name=model_name,
                metrics=metrics,
                judge_model=args.judge_model,
                metrics_path=metrics_path,
            )
        )

    summary_json_path = run_dir / "summary.json"
    summary_csv_path = run_dir / "summary.csv"

    write_json(summary_json_path, summary_rows)
    write_summary_csv(summary_csv_path, summary_rows)

    print(f"[INFO] Summary JSON written to: {summary_json_path}")
    print(f"[INFO] Summary CSV written to: {summary_csv_path}")


if __name__ == "__main__":
    main()
