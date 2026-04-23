#!/usr/bin/env bash
set -euo pipefail
cd ~/work/Projects/SpectralixBenchmark_aitunnel_glm51_src
source ~/.config/spectralix/aitunnel.env
~/.local/bin/uv run spectralix-student \
  --benchmark-path runs/benchmark_v3_glm51_aitunnel_minimal_no-sgr/final_repair6_20260423/shard_00.jsonl \
  --output-path runs/benchmark_v3_glm51_aitunnel_minimal_no-sgr/final_repair6_20260423/shard_00_student_output.jsonl \
  --api-base-url https://api.aitunnel.ru/v1 \
  --model-name glm-5.1 \
  --timeout 1800 \
  --max-retries 1 \
  --agent-sandbox local \
  --agent-backend local_worker \
  --agent-tools-profile minimal \
  --agent-config runs/benchmark_v3_glm51_aitunnel_minimal_no-sgr/agent_config_glm51_high.yaml \
  --agent-sgr-enabled false \
  --student-guard-enabled false \
  --trace-log-enabled true \
  --trace-log-dir runs/benchmark_v3_glm51_aitunnel_minimal_no-sgr/final_repair6_20260423/shard_00_traces \
  --verbose-output-enabled true \
  --verbose-output-path runs/benchmark_v3_glm51_aitunnel_minimal_no-sgr/final_repair6_20260423/shard_00_student_output_verbose.jsonl \
  --resume-existing true
