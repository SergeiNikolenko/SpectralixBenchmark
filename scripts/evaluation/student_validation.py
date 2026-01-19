import json
import requests
from pathlib import Path
from typing import Dict, Any
from tqdm import tqdm

# =========================
# CONFIG
# =========================

BENCHMARK_PATH = Path(
    "../parsing/benchmark_gold_standard.jsonl"
)

OUTPUT_PATH = Path(
    "student_output.jsonl"
)

CHEMLLM_URL = "http://10.100.10.105:30888/v1/completions"
MODEL_NAME = "AI4Chem/ChemLLM-7B-Chat-1_5-DPO"

MAX_TOKENS = 256
TEMPERATURE = 0.0
TIMEOUT = 120  # seconds


# =========================
# PROMPT BUILDER
# =========================

def build_prompt(question: dict) -> str:
    return f"""Answer the following chemistry exam question.
Provide ONLY the final answer.

{question["question_text"]}
""".strip()


# =========================
# LLM CALL
# =========================

def call_chemllm(prompt: str) -> str:
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
    }

    response = requests.post(
        CHEMLLM_URL,
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=TIMEOUT,
    )

    response.raise_for_status()
    data = response.json()

    return data["choices"][0]["text"].strip()


# =========================
# MAIN PIPELINE
# =========================

def validate_students():
    if not BENCHMARK_PATH.exists():
        raise FileNotFoundError(f"Benchmark file not found: {BENCHMARK_PATH}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Pre-count lines for tqdm
    with BENCHMARK_PATH.open("r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)

    with BENCHMARK_PATH.open("r", encoding="utf-8") as f_in, \
         OUTPUT_PATH.open("w", encoding="utf-8") as f_out:

        for line_idx, line in enumerate(
            tqdm(f_in, total=total_lines, desc="Validating student answers"),
            start=1
        ):
            question = json.loads(line)
            prompt = build_prompt(question)

            try:
                student_answer = call_chemllm(prompt)
            except Exception as e:
                student_answer = ""
                tqdm.write(f"[ERROR] Line {line_idx}: {e}")

            result = {
                "exam_id": question.get("exam_id"),
                "page_id": question.get("page_id"),
                "question_id": question.get("question_id"),
                "question_type": question.get("question_type"),
                "question_text": question.get("question_text"),
                "answer_type": question.get("answer_type"),
                "student_answer": student_answer,
            }

            f_out.write(json.dumps(result, ensure_ascii=False) + "\n")


# =========================
# ENTRYPOINT
# =========================

if __name__ == "__main__":
    validate_students()