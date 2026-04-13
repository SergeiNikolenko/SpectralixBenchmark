from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class RunRecord:
    run_id: str
    model_slug: str
    model_name: str
    overall_end_to_end_score: Optional[float]
    macro_depth_end_to_end_score: Optional[float]
    reliability_ok_rate: Optional[float]
    estimated_cost_usd: Optional[float]
    breakdown_by_task_subtype: Dict[str, Dict[str, Any]]


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_run_records(runs_root: Path) -> Iterable[RunRecord]:
    for run_dir in sorted(path for path in runs_root.iterdir() if path.is_dir()):
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            continue
        summary_rows = _read_json(summary_path)
        for row in summary_rows:
            model_name = str(row.get("model_name") or "")
            if not model_name:
                continue
            model_slug = model_name.replace("/", "_")
            model_dir = run_dir / model_slug
            breakdown_path = model_dir / "breakdown_by_task_subtype.json"
            if not breakdown_path.exists():
                continue
            yield RunRecord(
                run_id=run_dir.name,
                model_slug=model_slug,
                model_name=model_name,
                overall_end_to_end_score=row.get("overall_end_to_end_score"),
                macro_depth_end_to_end_score=row.get("macro_depth_end_to_end_score"),
                reliability_ok_rate=row.get("reliability_ok_rate"),
                estimated_cost_usd=row.get("estimated_cost_usd"),
                breakdown_by_task_subtype=_read_json(breakdown_path),
            )


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _format_float(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def build_rows(records: List[RunRecord]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for record in records:
        for subtype, metrics in sorted(record.breakdown_by_task_subtype.items()):
            rows.append(
                {
                    "run_id": record.run_id,
                    "model_name": record.model_name,
                    "task_subtype": subtype,
                    "count": metrics.get("count"),
                    "quality_score": metrics.get("quality_score"),
                    "end_to_end_score": metrics.get("end_to_end_score"),
                    "ok_rate": metrics.get("ok_rate"),
                    "overall_end_to_end_score": record.overall_end_to_end_score,
                    "macro_depth_end_to_end_score": record.macro_depth_end_to_end_score,
                    "reliability_ok_rate": record.reliability_ok_rate,
                    "estimated_cost_usd": record.estimated_cost_usd,
                }
            )
    return rows


def write_markdown(path: Path, records: List[RunRecord]) -> None:
    lines: List[str] = []
    lines.append("# Task Subtype Results")
    lines.append("")
    lines.append("Paper-facing summary of per-run task subtype metrics from local run artifacts.")
    lines.append("")
    lines.append("## Runs")
    lines.append("")
    lines.append("| run_id | model | overall_e2e | macro_depth_e2e | reliability | cost_usd |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for record in records:
        lines.append(
            f"| {record.run_id} | {record.model_name} | "
            f"{_format_float(record.overall_end_to_end_score)} | "
            f"{_format_float(record.macro_depth_end_to_end_score)} | "
            f"{_format_float(record.reliability_ok_rate)} | "
            f"{_format_float(record.estimated_cost_usd)} |"
        )
    lines.append("")
    lines.append("## Task Subtype Breakdown")
    lines.append("")
    for record in records:
        lines.append(f"### {record.run_id} ({record.model_name})")
        lines.append("")
        lines.append("| task_subtype | count | quality_score | end_to_end_score | ok_rate |")
        lines.append("|---|---:|---:|---:|---:|")
        for subtype, metrics in sorted(record.breakdown_by_task_subtype.items()):
            lines.append(
                f"| {subtype} | {metrics.get('count')} | "
                f"{_format_float(metrics.get('quality_score'))} | "
                f"{_format_float(metrics.get('end_to_end_score'))} | "
                f"{_format_float(metrics.get('ok_rate'))} |"
            )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export paper-facing task subtype comparison tables from local run artifacts."
    )
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("reports/task_subtype_report"),
        help="Output prefix without extension",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = list(_iter_run_records(args.runs_root))
    rows = build_rows(records)
    csv_path = args.output_prefix.with_suffix(".csv")
    md_path = args.output_prefix.with_suffix(".md")
    _write_csv(
        csv_path,
        rows,
        [
            "run_id",
            "model_name",
            "task_subtype",
            "count",
            "quality_score",
            "end_to_end_score",
            "ok_rate",
            "overall_end_to_end_score",
            "macro_depth_end_to_end_score",
            "reliability_ok_rate",
            "estimated_cost_usd",
        ],
    )
    write_markdown(md_path, records)
    print(f"[INFO] Wrote {csv_path}")
    print(f"[INFO] Wrote {md_path}")


if __name__ == "__main__":
    main()
