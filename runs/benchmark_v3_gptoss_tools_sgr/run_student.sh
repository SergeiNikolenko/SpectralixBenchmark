#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/work/Projects/SpectralixBenchmark"
export PATH="$HOME/.local/bin:$PATH"
export OPENAI_API_KEY=ollama
export SPECTRALIX_OPENSHELL_DIRECT_UPSTREAM=true
RUN_DIR="runs/benchmark_v3_gptoss_tools_sgr"
mkdir -p "$RUN_DIR"
uv run spectralix-student \
  --benchmark-path benchmark/benchmark_v3_eval.jsonl \
  --output-path "$RUN_DIR/student_output.jsonl" \
  --api-base-url http://172.17.0.1:11435/v1 \
  --model-name gpt-oss-120b-mxfp4:latest \
  --api-key ollama \
  --agent-sandbox openshell \
  --agent-backend openshell_worker \
  --agent-tools-profile tools \
  --agent-config spectralix_benchmark/agents/agent_config_ollama_gpt_oss.yaml \
  --agent-sgr-enabled true \
  --student-guard-enabled true \
  --trace-log-enabled true \
  --verbose-output-enabled true \
  --resume-existing true \
  --timeout 1800 \
  --max-retries 0 2>&1 | tee "$RUN_DIR/run.log"
touch "$RUN_DIR/stage.done"
