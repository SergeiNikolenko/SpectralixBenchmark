"""CLI for judge stage."""

from __future__ import annotations

import argparse

from spectralix_benchmark.evaluation.pipeline.llm_judge import parse_args, run_llm_judge


def main() -> None:
    args = parse_args()
    run_llm_judge(
        input_path=args.input_path,
        gold_path=args.gold_path,
        output_path=args.output_path,
        model_name=args.judge_model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        reasoning_effort=args.reasoning_effort,
        judge_structured_retries=args.judge_structured_retries,
        judge_method=args.judge_method,
        judge_g_eval_fallback_structured=args.judge_g_eval_fallback_structured,
        judge_model_url=args.judge_model_url,
        judge_api_key=args.judge_api_key,
        trace_log_enabled=args.trace_log_enabled,
        trace_log_dir=args.trace_log_dir,
        resume_existing=args.resume_existing,
    )


__all__ = ["main", "parse_args"]

