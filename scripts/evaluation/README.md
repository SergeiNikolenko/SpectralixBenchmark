# Evaluation Pipeline

## Components

- `student_validation.py`: student-answer generation (agent runtime by default)
- `student_validation_legacy.py`: archived direct HTTP baseline
- `llm_judge.py`: deterministic + LLM hybrid judge
- `run_full_matrix.py`: orchestrates multi-model runs and aggregates metrics

## Runtime Modes

### Agent mode (default)

- Uses `scripts/agents/runtime.py`
- Executes tool-capable agent with Docker sandbox
- Controlled by `scripts/agents/agent_config.yaml`

### Legacy mode

- Uses direct OpenAI-compatible HTTP endpoint
- Enable with `--agent-enabled false`

## Dependencies

```bash
pip install requests tqdm openai "smolagents[docker]" PyYAML
```

## Environment Variables

Supported API key variables:

- `OPENAI_API_KEY`
- `AITUNNEL_API_KEY`

Optional auth customization:

- `OPENAI_API_KEY_HEADER` (default: `Authorization`)
- `OPENAI_API_KEY_PREFIX` (default: `Bearer`)

## `student_validation.py` CLI

Required:

- `--benchmark-path`
- `--output-path`
- `--model-name`
- one of:
  - `--model-url` (chat endpoint or base URL)
  - `--api-base-url`

Agent flags:

- `--agent-enabled` (default: `true`)
- `--agent-max-steps` (default: `6`)
- `--agent-sandbox` (default: `docker`)
- `--agent-tools-profile` (default: `full`)
- `--agent-config` (default: `scripts/agents/agent_config.yaml`)

## Example: Local Proxy Run (5 rows)

```bash
python3 scripts/evaluation/student_validation.py \
  --benchmark-path benchmark/benchmark_v1_0.jsonl \
  --output-path scripts/evaluation/student_output.jsonl \
  --api-base-url "http://127.0.0.1:8317/v1" \
  --model-name "gpt-5.3-codex-spark" \
  --api-key "ccs-internal-managed" \
  --limit 5
```

## Judge Run

```bash
python3 scripts/evaluation/llm_judge.py \
  --input-path scripts/evaluation/student_output.jsonl \
  --gold-path benchmark/benchmark_v1_0.jsonl \
  --judge-model "gpt-5.3-codex-spark" \
  --output-path scripts/evaluation/llm_judge_output.jsonl
```

## Full Matrix Run

`run_full_matrix.py` accepts `--student-models` and alias `--models`.
It also accepts `--api-base-url` as an alternative to `--model-url`.

```bash
python3 scripts/evaluation/run_full_matrix.py \
  --benchmark-path benchmark/benchmark_v1_0.jsonl \
  --api-base-url "http://127.0.0.1:8317/v1" \
  --api-key "ccs-internal-managed" \
  --models gpt-5.3-codex-spark \
  --judge-model gpt-5.3-codex-spark \
  --agent-enabled true
```

## Output Contracts

Student output (`student_output.jsonl`) schema is stable:

- `exam_id`
- `page_id`
- `question_id`
- `question_type`
- `question_text`
- `answer_type`
- `student_answer`
- `student_status`
- `student_error`
- `student_elapsed_ms`

Judge output (`llm_judge_output.jsonl`) schema remains unchanged.
