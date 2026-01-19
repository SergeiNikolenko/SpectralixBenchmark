import json
from pathlib import Path
from typing import Dict
from openai import OpenAI
from tqdm import tqdm

# =========================
# CONFIG
# =========================

INPUT_PATH = Path(
    "student_output.jsonl"
)

GOLD_PATH = Path(
    "../parsing/benchmark_gold_standard.jsonl"
)

OUTPUT_PATH = Path(
    "llm_judge_output.jsonl"
)

MODEL_NAME = "gpt-4o-mini"
MAX_TOKENS = 300
TEMPERATURE = 0.0

client = OpenAI()


# =========================
# PROMPT
# =========================

JUDGE_SYSTEM_PROMPT = """
You are an expert chemistry exam examiner.

Your task is to evaluate a student's answer against the canonical answer.

You MUST follow these rules strictly:

GENERAL RULES
- Do NOT help the student.
- Do NOT explain chemistry theory.
- Judge ONLY correctness and completeness.
- Be strict and conservative.

SCORING
- Return a normalized score in the range [0, 1].

ANSWER TYPE RULES
(single_choice, multiple_choice, numeric, ordering, structure, text, full_synthesis,
reaction_description, property_determination)

Follow the evaluation rules for each type exactly.

OUTPUT FORMAT
Return ONLY valid JSON:
{
  "llm_score": <float>,
  "llm_comment": "<short justification>"
}
""".strip()


def build_user_prompt(item: Dict) -> str:
    return f"""
Question type: {item["question_type"]}
Answer type: {item["answer_type"]}

Question:
{item["question_text"]}

Canonical answer:
{item["canonical_answer"]}

Student answer:
{item["student_answer"]}
""".strip()


# =========================
# LLM CALL
# =========================

def call_llm_judge(item: Dict) -> Dict:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(item)},
        ],
    )

    content = response.choices[0].message.content
    return json.loads(content)


# =========================
# MAIN PIPELINE
# =========================

def run_llm_judge():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Load gold answers for lookup
    gold = {}
    with GOLD_PATH.open() as f:
        for line in f:
            q = json.loads(line)
            key = (q["exam_id"], q["page_id"], q["question_id"])
            gold[key] = q

    with INPUT_PATH.open() as f_in, OUTPUT_PATH.open("w") as f_out:
        for line in tqdm(f_in, desc="LLM judging"):
            student = json.loads(line)
            key = (
                student["exam_id"],
                student["page_id"],
                student["question_id"],
            )

            gold_q = gold[key]

            judge_input = {
                **student,
                "canonical_answer": gold_q["canonical_answer"],
            }

            try:
                judge_result = call_llm_judge(judge_input)
            except Exception as e:
                judge_result = {
                    "llm_score": 0.0,
                    "llm_comment": f"Judging failed: {e}",
                }

            max_score = gold_q.get("max_score", 0)
            final_score = max_score * judge_result["llm_score"]

            output = {
                **judge_input,
                "llm_score": judge_result["llm_score"],
                "llm_comment": judge_result["llm_comment"],
                "final_score": final_score,
                "max_score": max_score,
            }

            f_out.write(json.dumps(output, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    run_llm_judge()