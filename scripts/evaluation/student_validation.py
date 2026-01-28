import json
import time
import requests
import argparse
from pathlib import Path
from typing import Dict, Any
from tqdm import tqdm


# =========================
# PROMPT BUILDER
# =========================

def build_prompt(question: dict) -> str:
    return question["question_text"]


# =========================
# LLM CALL
# =========================

def call_chemllm(
    prompt: str,
    model_url: str,
    model_name: str,
    max_tokens: int,
    temperature: float,
    timeout: int,
    max_retries: int = 3,
) -> str:
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                model_url,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()
            data = response.json()
            
            if "choices" in data and len(data["choices"]) > 0:
                return data["choices"][0]["message"]["content"].strip()
            else:
                raise ValueError("No choices in response")
                
        except (requests.RequestException, KeyError, ValueError) as e:
            if attempt == max_retries - 1:
                raise
            wait_time = 2 ** attempt  # Exponential backoff
            tqdm.write(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
            time.sleep(wait_time)
    
    return ""  # Fallback


# =========================
# MAIN PIPELINE
# =========================

def run_benchmark_inference(
    benchmark_path: Path,
    output_path: Path,
    model_url: str,
    model_name: str,
    max_tokens: int,
    temperature: float,
    timeout: int,
):
    if not benchmark_path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {benchmark_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Pre-count lines for tqdm
    with benchmark_path.open("r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)

    with benchmark_path.open("r", encoding="utf-8") as f_in, \
         output_path.open("w", encoding="utf-8") as f_out:

        for line_idx, line in enumerate(
            tqdm(f_in, total=total_lines, desc="Validating student answers"),
            start=1
        ):
            question = json.loads(line)
            prompt = build_prompt(question)

            try:
                student_answer = call_chemllm(
                    prompt,
                    model_url=model_url,
                    model_name=model_name,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout=timeout,
                )
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
    parser = argparse.ArgumentParser(
        description="Benchmark student answers using a remote LLM model"
    )
    parser.add_argument(
        "--benchmark-path",
        type=Path,
        default="/Users/kokosiknn/Desktop/SpectralixBenchmark/SpectralixBenchmark/benchmark/benchmark_v1_0.jsonl",
        help="Path to the benchmark JSONL file",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Path to save the output JSONL file",
    )
    parser.add_argument(
        "--model-url",
        type=str,
        required=True,
        help="URL of the model API endpoint",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Name of the model to use",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum number of tokens in the response (default: 256)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.5,
        help="Temperature for sampling (default: 0.5)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Timeout in seconds for API requests (default: 120)",
    )

    args = parser.parse_args()

    run_benchmark_inference(
        benchmark_path=args.benchmark_path,
        output_path=args.output_path,
        model_url=args.model_url,
        model_name=args.model_name,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        timeout=args.timeout,
    )