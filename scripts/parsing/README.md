# Parsing Pipeline

## Components

- `exam-parser-pipeline.py`: PDF-to-questions parsing pipeline
- `benchmark_collection.py`: flattens parsed output into benchmark-style JSONL

## Runtime Behavior

`exam-parser-pipeline.py` uses agent runtime by default:

- Agent mode: `smolagents` + Docker sandbox
- Fallback: legacy OpenAI direct call if agent runtime initialization fails

## Dependencies

```bash
pip install -r scripts/parsing/requirements.txt
```

## Required Inputs

Place exam PDFs in:

```bash
exam_data/exams/
```

## Parser Run

```bash
python3 scripts/parsing/exam-parser-pipeline.py \
  --agent-enabled true \
  --agent-max-steps 6 \
  --agent-config scripts/agents/agent_config.yaml \
  --model-marker gpt-4.1-mini \
  --openai-base-url "http://127.0.0.1:8317/v1" \
  --api-key "ccs-internal-managed"
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
python3 scripts/parsing/benchmark_collection.py
```

Produces:

- `benchmark_dataset.jsonl`
