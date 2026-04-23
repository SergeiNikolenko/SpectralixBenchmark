#!/usr/bin/env bash
set -euo pipefail
cd /Users/nikolenko/Documents/Projects/SpectralixBenchmark
UV_CACHE_DIR=/tmp/uv-cache-spectralix-glm51-final /Users/nikolenko/.local/bin/uv run spectralix-judge \
  --input-path runs/benchmark_v3_glm51_aitunnel_minimal_no-sgr/judge_gpt54mini/b_v2_rejudge_20260423/judge_shard_01_input.jsonl \
  --gold-path benchmark/benchmark_v3_eval.jsonl \
  --judge-model gpt-5.4-mini \
  --judge-model-url http://127.0.0.1:8318/v1 \
  --judge-api-key ccs-internal-managed \
  --judge-method g_eval \
  --judge-g-eval-fallback-structured true \
  --judge-structured-retries 2 \
  --reasoning-effort medium \
  --resume-existing true \
  --trace-log-enabled true \
  --trace-log-dir runs/benchmark_v3_glm51_aitunnel_minimal_no-sgr/judge_gpt54mini/b_v2_rejudge_20260423/judge_shard_01_traces \
  --output-path runs/benchmark_v3_glm51_aitunnel_minimal_no-sgr/judge_gpt54mini/b_v2_rejudge_20260423/judge_shard_01_output.jsonl
