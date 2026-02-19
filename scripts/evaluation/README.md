# Evaluation Pipeline

This folder contains two scripts:

1. `student_validation.py`
Generates `student_answer` for each benchmark question by querying a model endpoint.

2. `llm_judge.py`
Evaluates `student_answer` against the canonical answer and writes judge scores/comments.

3. `run_full_matrix.py`
Runs the full fixed model matrix through both stages and writes per-run artifacts under `scripts/evaluation/runs/<timestamp>/`.

## Requirements

- Python 3.10+
- `requests`, `tqdm`, `openai`
- API key is required for `llm_judge.py` (`--api-key` or `OPENAI_API_KEY` / `AITUNNEL_API_KEY`).

Install dependencies if needed:

```bash
pip install requests tqdm openai
```

## OpenAI-Compatible Endpoint Support (including aitunnel)

Both scripts support:

- Direct endpoint mode (`--model-url`) for `student_validation.py`
- OpenAI-compatible base URL mode (`--api-base-url`) for both scripts
- API key from CLI or environment variables
- Custom API key header name/prefix when required by the gateway
- Low-cost test runs with `--limit` in both scripts

Environment variables supported by both scripts:

- `OPENAI_BASE_URL` or `AITUNNEL_BASE_URL`
- `OPENAI_API_KEY` or `AITUNNEL_API_KEY`
- `OPENAI_API_KEY_HEADER` (default: `Authorization`)
- `OPENAI_API_KEY_PREFIX` (default: `Bearer`)

## Default Gold/Benchmark Path Behavior

- `llm_judge.py` default `--gold-path`:
  - Uses `benchmark/benchmark_v1_0.jsonl` (relative to repo root) when it exists.
  - Falls back to `scripts/parsing/benchmark_gold_standard.jsonl` if needed.
- `student_validation.py` default `--benchmark-path`:
  - Uses `benchmark/benchmark_v1_0.jsonl` when available.

You can always override paths explicitly with CLI flags.

## Exact Run Commands (aitunnel example)

Run from repository root:

```bash
cd /Users/nikolenko/Documents/Code/SpectralixBenchmark
```

Set endpoint + key:

```bash
export OPENAI_BASE_URL="https://api.aitunnel.ru/v1"
export OPENAI_API_KEY="YOUR_AITUNNEL_API_KEY"
```

List available models:

```bash
curl -sS \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  "$OPENAI_BASE_URL/models"
```

Optional (only if your gateway requires non-standard auth header):

```bash
export OPENAI_API_KEY_HEADER="Authorization"
export OPENAI_API_KEY_PREFIX="Bearer"
```

`llm_judge.py` uses the OpenAI SDK. If `OPENAI_API_KEY_HEADER=Authorization`, keep `OPENAI_API_KEY_PREFIX=Bearer`.
For non-Bearer prefixes, use a non-Authorization header name.

### 1) Generate student answers through aitunnel

```bash
python3 scripts/evaluation/student_validation.py \
  --api-base-url "$OPENAI_BASE_URL" \
  --api-key "$OPENAI_API_KEY" \
  --model-name "deepseek-r1-distill-qwen-32b" \
  --limit 10 \
  --output-path scripts/evaluation/student_output.jsonl
```

If you want a different benchmark file:

```bash
python3 scripts/evaluation/student_validation.py \
  --benchmark-path benchmark/benchmark_v1_0.jsonl \
  --api-base-url "$OPENAI_BASE_URL" \
  --api-key "$OPENAI_API_KEY" \
  --model-name "deepseek-r1-distill-qwen-32b" \
  --limit 10 \
  --output-path scripts/evaluation/student_output.jsonl
```

### 2) Run LLM judge through aitunnel

```bash
python3 scripts/evaluation/llm_judge.py \
  --input-path scripts/evaluation/student_output.jsonl \
  --api-base-url "$OPENAI_BASE_URL" \
  --api-key "$OPENAI_API_KEY" \
  --model-name "deepseek-r1-distill-qwen-32b" \
  --limit 10 \
  --output-path scripts/evaluation/llm_judge_output.jsonl
```

Override gold path explicitly if needed:

```bash
python3 scripts/evaluation/llm_judge.py \
  --input-path scripts/evaluation/student_output.jsonl \
  --gold-path benchmark/benchmark_v1_0.jsonl \
  --api-base-url "$OPENAI_BASE_URL" \
  --api-key "$OPENAI_API_KEY" \
  --model-name "deepseek-r1-distill-qwen-32b" \
  --limit 10 \
  --output-path scripts/evaluation/llm_judge_output.jsonl
```

## Notes

- `llm_judge.py` now has robust JSON parsing for judge output:
  - Direct JSON
  - JSON inside markdown code fences
  - First valid JSON object extracted from mixed text
- If parsing still fails, fallback output is used with:
  - `llm_score = 0.0`
  - Clear `llm_comment` with a short raw response preview
  - Technical marker `[judge_error]` in fallback comments to make infra/parser failures easy to detect downstream
- Output JSONL schema is preserved.

## Full Matrix Orchestrator

Run all fixed models through the full pipeline:

```bash
python3 scripts/evaluation/run_full_matrix.py \
  --api-base-url "http://127.0.0.1:8317/v1" \
  --api-key "${CLIPROXY_API_KEY:-your-api-key-1}" \
  --workers 1
```

### Current Target Model Set (agreed)

Default testing policy:

- Always use `gpt-5.3-codex-spark` as the primary test model.

For the next primary run, use this exact subset:

- `gpt-5.2` (`reasoning=default`)
- `gpt-5.2` (`reasoning=low`)
- `gpt-5.2` (`reasoning=medium`)
- `gpt-5.2` (`reasoning=high`)
- `gpt-5.2-codex` (`reasoning=high`)
- `gpt-5.3-codex` (`reasoning=high`)
- `gpt-5.3-codex-spark` (`reasoning=high`)
- `claude-sonnet-4-6` (single Sonnet representative)
- `claude-opus-4-6` (single Opus representative)
- `claude-haiku-4-5-20251001` (Haiku representative)

Recommended launch command (baseline, with `gpt-5.2` at `reasoning=high`):

```bash
python3 scripts/evaluation/run_full_matrix.py \
  --api-base-url "http://127.0.0.1:8317/v1" \
  --api-key "${CLIPROXY_API_KEY:-your-api-key-1}" \
  --models gpt-5.2 gpt-5.2-codex gpt-5.3-codex gpt-5.3-codex-spark claude-sonnet-4-6 claude-opus-4-6 claude-haiku-4-5-20251001 \
  --workers 1
```

Run smoke test (fast):

```bash
python3 scripts/evaluation/run_full_matrix.py \
  --api-base-url "http://127.0.0.1:8317/v1" \
  --api-key "${CLIPROXY_API_KEY:-your-api-key-1}" \
  --models gpt-5.3-codex-spark claude-haiku-4-5-20251001 \
  --limit 3 \
  --workers 1
```

`--models` rules:

- Only models from the fixed list are accepted.
- Duplicates are not allowed in `--models` and fail argument validation.

Exit code behavior:

- `0` when all selected model runs succeed.
- Non-zero when any selected model run fails.

Find latest run and inspect artifacts:

```bash
latest_run=$(ls -td /Users/nikolenko/Documents/Code/SpectralixBenchmark/scripts/evaluation/runs/* | head -1)
echo "$latest_run"
cat "$latest_run/summary.csv"
head -n 2 "$latest_run/summary.csv"
```
