# SpectralixBenchmark

Benchmark and evaluation tooling for chemistry-focused AI systems:

- Organic chemistry tasks
- Tandem mass spectrometry (MS2) tasks
- Structured answer evaluation with deterministic + LLM judging

## Benchmark Schema

Primary dataset: `benchmark/benchmark_v1_0.jsonl`

Each row follows this schema:

```json
{
  "exam_id": "string",
  "page_id": "integer | string",
  "question_id": "integer | string",
  "question_type": "string",
  "question_text": "string",
  "answer_type": "single_choice | multiple_choice | numeric | ordering | structure | full_synthesis | reaction_description | property_determination | msms_structure_prediction | text",
  "canonical_answer": "string",
  "max_score": "integer"
}
```

## Runtime Architecture

Student-stage inference supports two backends:

1. Agent runtime (default): `smolagents` with Docker sandbox and allowlisted tools
2. Legacy baseline: direct OpenAI-compatible HTTP calls

Core runtime modules:

- `scripts/agents/runtime.py`
- `scripts/agents/config.py`
- `scripts/agents/models.py`
- `scripts/agents/tool_registry.py`
- `scripts/agents/prompts.py`

Security and operational controls:

- `scripts/agents/agent_config.yaml`
- `docs/security_runbook.md`
- `docs/architecture.md`

## Quick Start (Local Proxy)

Repository root:

```bash
cd /Users/nikolenko/.codex/worktrees/e20d/SpectralixBenchmark
```

Install dependencies:

```bash
pip install -r scripts/parsing/requirements.txt
pip install requests tqdm openai "smolagents[docker]" PyYAML
```

Verify local proxy (example setup):

- API base URL: `http://127.0.0.1:8317/v1`
- API key: `ccs-internal-managed`

```bash
curl -sS \
  -H "Authorization: Bearer ccs-internal-managed" \
  "http://127.0.0.1:8317/v1/models"
```

Run student inference on 5 questions:

```bash
python3 scripts/evaluation/student_validation.py \
  --benchmark-path benchmark/benchmark_v1_0.jsonl \
  --output-path scripts/evaluation/student_output.jsonl \
  --api-base-url "http://127.0.0.1:8317/v1" \
  --model-name "gpt-5.3-codex-spark" \
  --api-key "ccs-internal-managed" \
  --limit 5
```

Run judge:

```bash
python3 scripts/evaluation/llm_judge.py \
  --input-path scripts/evaluation/student_output.jsonl \
  --gold-path benchmark/benchmark_v1_0.jsonl \
  --judge-model "gpt-5.3-codex-spark" \
  --output-path scripts/evaluation/llm_judge_output.jsonl
```

Run full matrix:

```bash
python3 scripts/evaluation/run_full_matrix.py \
  --benchmark-path benchmark/benchmark_v1_0.jsonl \
  --api-base-url "http://127.0.0.1:8317/v1" \
  --api-key "ccs-internal-managed" \
  --models gpt-5.3-codex-spark \
  --judge-model gpt-5.3-codex-spark \
  --agent-enabled true \
  --agent-max-steps 6 \
  --agent-sandbox docker \
  --agent-tools-profile full
```

## Parsing Pipeline

Parse source PDFs and produce per-exam artifacts:

```bash
python3 scripts/parsing/exam-parser-pipeline.py \
  --agent-enabled true \
  --agent-max-steps 6 \
  --agent-config scripts/agents/agent_config.yaml \
  --openai-base-url "http://127.0.0.1:8317/v1" \
  --api-key "ccs-internal-managed"
```

Collect flattened benchmark rows from parser output:

```bash
python3 scripts/parsing/benchmark_collection.py
```

## Contacts

Maintainer (Innopolis University — AI Lab in Chemistry):

- Ivan Golov
- Email: `i.golov@innopolis.university`
- Telegram: [https://t.me/Ione_Golov](https://t.me/Ione_Golov)
