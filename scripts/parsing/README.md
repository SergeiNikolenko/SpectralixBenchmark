# Parsing Pipeline

## Components

- `exam-parser-pipeline.py`: PDF-to-questions parsing pipeline
- `benchmark_collection.py`: flattens parsed output into benchmark-style JSONL

## Runtime Behavior

`exam-parser-pipeline.py` uses agent runtime by default:

- Agent mode: OpenShell-backed runtime
- Structured repair mode: `PydanticAI` repairs malformed parser JSON when enabled
- Fallback: legacy OpenAI direct call if agent runtime initialization fails

## Dependencies

```bash
uv sync
```

## Required Inputs

Place exam PDFs in:

```bash
exam_data/exams/
```

## Parser Run

```bash
uv run python scripts/parsing/exam-parser-pipeline.py \
  --agent-enabled true \
  --agent-max-steps 6 \
  --agent-config scripts/agents/agent_config.yaml \
  --model-marker gpt-4.1-mini \
  --openai-base-url "http://127.0.0.1:8317/v1" \
  --api-key "ccs-internal-managed" \
  --parser-structured-repair-enabled true \
  --parser-structured-retries 2
```

Environment:

- `OPENAI_API_KEY` (or use `--api-key`)

## Output

Per exam under `exam_data/output/<exam_id>/`:

- `exam.json`
- `questions.jsonl`

Error reports:

- `exam_data/output/errors/<exam_id>_errors.json`

Pipeline log:

- `exam_data/output/pipeline.log`

## Flatten Parsed Output

```bash
uv run python scripts/parsing/benchmark_collection.py
```

Produces:

- `benchmark_dataset.jsonl`
