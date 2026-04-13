from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from scripts.evaluation.benchmark_taxonomy import overlay_benchmark_taxonomy_fields
from scripts.evaluation.run_full_matrix import compute_metrics, write_summary_csv


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _iter_model_dirs(run_dir: Path) -> Iterable[Path]:
    for child in sorted(run_dir.iterdir()):
        if not child.is_dir():
            continue
        if (child / "llm_judge_output.jsonl").exists():
            yield child


def migrate_model_dir(model_dir: Path) -> Dict[str, Any]:
    judge_path = model_dir / "llm_judge_output.jsonl"
    judge_rows = [overlay_benchmark_taxonomy_fields(row) for row in read_jsonl(judge_path)]
    write_jsonl(judge_path, judge_rows)

    metrics = compute_metrics(judge_rows)
    existing_metrics = {}
    metrics_path = model_dir / "metrics.json"
    if metrics_path.exists():
        existing_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    for key in (
        "model_name",
        "judge_model",
        "student_output_path",
        "judge_output_path",
        "student_input_tokens_total",
        "student_output_tokens_total",
        "student_total_tokens_total",
        "student_pricing_usd_per_1m",
        "judge_pricing_usd_per_1m",
        "student_estimated_cost_usd",
        "judge_estimated_cost_usd",
        "estimated_cost_usd",
    ):
        if key in existing_metrics and key not in metrics:
            metrics[key] = existing_metrics[key]
        elif key in existing_metrics and metrics.get(key) in (None, 0, {}, []):
            metrics[key] = existing_metrics[key]

    write_json(metrics_path, metrics)
    write_json(model_dir / "breakdown_by_suite.json", metrics.get("breakdown_by_suite") or {})
    write_json(model_dir / "breakdown_by_subtrack.json", metrics.get("breakdown_by_subtrack") or {})
    write_json(model_dir / "breakdown_by_task_mode.json", metrics.get("breakdown_by_task_mode") or {})
    write_json(
        model_dir / "breakdown_by_planning_horizon.json",
        metrics.get("breakdown_by_planning_horizon") or {},
    )

    model_manifest_path = model_dir / "run_manifest.json"
    if model_manifest_path.exists():
        manifest = json.loads(model_manifest_path.read_text(encoding="utf-8"))
        manifest["metrics_path"] = str(metrics_path)
        manifest["breakdown_by_suite_path"] = str(model_dir / "breakdown_by_suite.json")
        manifest["breakdown_by_subtrack_path"] = str(model_dir / "breakdown_by_subtrack.json")
        manifest["breakdown_by_task_mode_path"] = str(model_dir / "breakdown_by_task_mode.json")
        manifest["breakdown_by_planning_horizon_path"] = str(
            model_dir / "breakdown_by_planning_horizon.json"
        )
        manifest.pop("breakdown_path", None)
        manifest.pop("quality_normalized_score", None)
        manifest.pop("quality_end_to_end_score", None)
        manifest["macro_depth_quality_score"] = metrics.get("macro_depth_quality_score")
        manifest["macro_depth_end_to_end_score"] = metrics.get("macro_depth_end_to_end_score")
        manifest["auxiliary_grounding_quality_score"] = metrics.get(
            "auxiliary_grounding_quality_score"
        )
        manifest["auxiliary_grounding_end_to_end_score"] = metrics.get(
            "auxiliary_grounding_end_to_end_score"
        )
        manifest["overall_quality_score"] = metrics.get("overall_quality_score")
        manifest["overall_end_to_end_score"] = metrics.get("overall_end_to_end_score")
        manifest["reliability_ok_rate"] = metrics.get("reliability_ok_rate")
        write_json(model_manifest_path, manifest)

    return {
        "model_name": metrics.get("model_name") or model_dir.name,
        "total_rows": metrics.get("total_rows"),
        "overall_quality_total_score": metrics.get("overall_quality_total_score"),
        "overall_quality_max_score": metrics.get("overall_quality_max_score"),
        "overall_quality_score": metrics.get("overall_quality_score"),
        "overall_end_to_end_score": metrics.get("overall_end_to_end_score"),
        "macro_depth_quality_score": metrics.get("macro_depth_quality_score"),
        "macro_depth_end_to_end_score": metrics.get("macro_depth_end_to_end_score"),
        "auxiliary_grounding_quality_score": metrics.get("auxiliary_grounding_quality_score"),
        "auxiliary_grounding_end_to_end_score": metrics.get("auxiliary_grounding_end_to_end_score"),
        "student_input_tokens_total": metrics.get("student_input_tokens_total"),
        "student_output_tokens_total": metrics.get("student_output_tokens_total"),
        "student_total_tokens_total": metrics.get("student_total_tokens_total"),
        "judge_input_tokens_total": metrics.get("judge_input_tokens_total"),
        "judge_output_tokens_total": metrics.get("judge_output_tokens_total"),
        "judge_total_tokens_total": metrics.get("judge_total_tokens_total"),
        "judge_reasoning_tokens_total": metrics.get("judge_reasoning_tokens_total"),
        "judge_requests_total": metrics.get("judge_requests_total"),
        "judge_tool_calls_total": metrics.get("judge_tool_calls_total"),
        "student_estimated_cost_usd": metrics.get("student_estimated_cost_usd"),
        "judge_estimated_cost_usd": metrics.get("judge_estimated_cost_usd"),
        "estimated_cost_usd": metrics.get("estimated_cost_usd"),
        "reliability_ok_count": metrics.get("reliability_ok_count"),
        "reliability_ok_rate": metrics.get("reliability_ok_rate"),
        "infra_error_count": metrics.get("infra_error_count"),
        "infra_error_rate": metrics.get("infra_error_rate"),
        "judge_model": metrics.get("judge_model"),
        "metrics_path": str(metrics_path),
    }


def migrate_run_dir(run_dir: Path) -> List[Dict[str, Any]]:
    summary_rows = [migrate_model_dir(model_dir) for model_dir in _iter_model_dirs(run_dir)]
    if summary_rows:
        write_json(run_dir / "summary.json", summary_rows)
        write_summary_csv(run_dir / "summary.csv", summary_rows)

        run_manifest_path = run_dir / "run_manifest.json"
        if run_manifest_path.exists():
            run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
            models = run_manifest.get("models") or []
            new_models = []
            summary_by_name = {row["model_name"]: row for row in summary_rows}
            for model in models:
                name = model.get("model_name")
                summary = summary_by_name.get(name, {})
                updated = dict(model)
                updated.pop("quality_normalized_score", None)
                updated.pop("quality_end_to_end_score", None)
                updated["overall_quality_score"] = summary.get("overall_quality_score")
                updated["overall_end_to_end_score"] = summary.get("overall_end_to_end_score")
                updated["macro_depth_quality_score"] = summary.get("macro_depth_quality_score")
                updated["macro_depth_end_to_end_score"] = summary.get("macro_depth_end_to_end_score")
                updated["auxiliary_grounding_quality_score"] = summary.get(
                    "auxiliary_grounding_quality_score"
                )
                updated["auxiliary_grounding_end_to_end_score"] = summary.get(
                    "auxiliary_grounding_end_to_end_score"
                )
                new_models.append(updated)
            run_manifest["models"] = new_models
            run_manifest["summary_json_path"] = str(run_dir / "summary.json")
            run_manifest["summary_csv_path"] = str(run_dir / "summary.csv")
            run_manifest.pop("summary_taxonomy_v2_path", None)
            write_json(run_manifest_path, run_manifest)

        for obsolete_path in (
            run_dir / "summary_taxonomy_v2.json",
        ):
            if obsolete_path.exists():
                obsolete_path.unlink()

        for model_dir in _iter_model_dirs(run_dir):
            for obsolete_path in (
                model_dir / "breakdown_by_answer_type.json",
                model_dir / "taxonomy_v2_rows.jsonl",
                model_dir / "taxonomy_v2_metrics.json",
                model_dir / "taxonomy_v2_breakdown_by_suite.json",
                model_dir / "taxonomy_v2_breakdown_by_subtrack.json",
            ):
                if obsolete_path.exists():
                    obsolete_path.unlink()
    return summary_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Migrate existing run artifacts to the benchmark taxonomy structure."
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("runs"),
        help="Root folder containing run directories (default: runs)",
    )
    parser.add_argument(
        "--run-dirs",
        nargs="*",
        type=Path,
        default=None,
        help="Optional explicit run directories to backfill",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dirs = args.run_dirs or [path for path in sorted(args.runs_root.iterdir()) if path.is_dir()]
    completed = 0
    for run_dir in run_dirs:
        summary_rows = migrate_run_dir(run_dir)
        if summary_rows:
            completed += 1
            print(f"[INFO] Migrated benchmark taxonomy artifacts for {run_dir} ({len(summary_rows)} model dirs)")
    print(f"[INFO] Completed run migration count: {completed}")


if __name__ == "__main__":
    main()
