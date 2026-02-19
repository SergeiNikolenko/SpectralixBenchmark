import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from llm_judge import run_llm_judge
from student_validation import run_benchmark_inference


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
        student_status = str(row.get("student_status", "ok") or "ok")
        row_status = str(row.get("row_status", "ok") or "ok")
        if student_status == "ok" and row_status == "ok":
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

        student_status = str(row.get("student_status", "ok") or "ok")
        row_status = str(row.get("row_status", "ok") or "ok")
        if student_status == "ok" and row_status == "ok":
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
        student_status = str(row.get("student_status", "ok") or "ok")
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
                "total_rows": row.get("total_rows"),
                "quality_total_score": row.get("quality_total_score"),
                "quality_max_score": row.get("quality_max_score"),
                "quality_normalized_score": row.get("quality_normalized_score"),
                "reliability_ok_count": row.get("reliability_ok_count"),
                "reliability_ok_rate": row.get("reliability_ok_rate"),
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
        required=True,
        help="Student model API endpoint URL",
    )
    parser.add_argument(
        "--student-models",
        nargs="+",
        required=True,
        help="One or more student model names",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="gpt-4o-mini",
        help="Fixed judge model for non-deterministic answer types (default: gpt-4o-mini)",
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
        "--error-sample-size",
        type=int,
        default=10,
        help="How many technical error examples to save (default: 10)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_dir = args.output_root / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, Any]] = []

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
        run_benchmark_inference(
            benchmark_path=args.benchmark_path,
            output_path=student_output_path,
            model_url=args.model_url,
            model_name=model_name,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            timeout=args.timeout,
            max_retries=args.max_retries,
            limit=args.limit,
            workers=args.workers,
        )

        student_rows = read_jsonl(student_output_path)
        if not student_rows:
            print(
                f"[WARN] No rows were produced for model={model_name}. "
                "Skipping judging and metrics."
            )
            skipped_row = {
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
            write_json(metrics_path, skipped_row)
            write_json(breakdown_path, {})
            write_json(errors_sample_path, [])
            summary_rows.append(skipped_row)
            continue

        print(
            f"[INFO] Running hybrid judge for model={model_name} "
            f"with fixed judge={args.judge_model}"
        )
        run_llm_judge(
            input_path=student_output_path,
            gold_path=args.benchmark_path,
            output_path=judge_output_path,
            model_name=args.judge_model,
            max_tokens=args.judge_max_tokens,
            temperature=args.judge_temperature,
        )

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
            {
                "model_name": model_name,
                "total_rows": metrics.get("total_rows"),
                "quality_total_score": metrics.get("quality_total_score"),
                "quality_max_score": metrics.get("quality_max_score"),
                "quality_normalized_score": metrics.get("quality_normalized_score"),
                "reliability_ok_count": metrics.get("reliability_ok_count"),
                "reliability_ok_rate": metrics.get("reliability_ok_rate"),
                "infra_error_count": metrics.get("infra_error_count"),
                "infra_error_rate": metrics.get("infra_error_rate"),
                "judge_model": args.judge_model,
                "metrics_path": str(metrics_path),
            }
        )

    summary_json_path = run_dir / "summary.json"
    summary_csv_path = run_dir / "summary.csv"

    write_json(summary_json_path, summary_rows)
    write_summary_csv(summary_csv_path, summary_rows)

    print(f"[INFO] Summary JSON written to: {summary_json_path}")
    print(f"[INFO] Summary CSV written to: {summary_csv_path}")


if __name__ == "__main__":
    main()
