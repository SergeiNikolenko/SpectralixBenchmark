import json
import os
from pathlib import Path
from typing import List, Dict

# ================= CONFIGURATION =================

# Directory containing parser pipeline outputs (one subdirectory per exam_id)
INPUT_BASE_DIR = Path("./exam_data/output")

# Consolidated JSONL output file
OUTPUT_FILE = Path("./benchmark_dataset.jsonl")

# ================= MAIN LOGIC =================

def collect_benchmark():
    if not INPUT_BASE_DIR.exists():
        print(f"Error: Directory {INPUT_BASE_DIR} does not exist.")
        return

    print(f"Scanning {INPUT_BASE_DIR} for questions.jsonl files...")
    
    total_questions = 0
    processed_exams = 0
    
    # Open the consolidated output file for writing.
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as out_f:
        
        # Iterate over all exam directories in a deterministic order.
        for exam_dir in sorted(INPUT_BASE_DIR.iterdir()):
            
            if not exam_dir.is_dir() or exam_dir.name == "errors":
                continue

            jsonl_path = exam_dir / "questions.jsonl"
            
            if not jsonl_path.exists():
                print(f"Skipping {exam_dir.name}: questions.jsonl not found.")
                continue

            print(f"Processing {exam_dir.name}...")
            
            try:
                # Read the file page by page.
                with open(jsonl_path, 'r', encoding='utf-8') as in_f:
                    for line_num, line in enumerate(in_f, 1):
                        if not line.strip():
                            continue
                            
                        try:
                            page_data = json.loads(line)
                        except json.JSONDecodeError:
                            print(f"  Warning: Invalid JSON in {jsonl_path} at line {line_num}")
                            continue

                        # Page metadata is attached to every flattened question.
                        page_meta = {
                            "exam_id": page_data.get("exam_id"),
                            "page_id": page_data.get("page_id"),
                            "path_to_page": page_data.get("path_to_page")
                        }
                        
                        questions_list = page_data.get("parsed_questions", [])
                        
                        # Flatten each question on the page into the output stream.
                        for q in questions_list:
                            flat_question = {**page_meta, **q}
                            out_f.write(json.dumps(flat_question) + "\n")
                            total_questions += 1

                processed_exams += 1

            except Exception as e:
                print(f"  Error processing file {jsonl_path}: {e}")

    print("-" * 40)
    print(f"Done! Successfully created {OUTPUT_FILE}")
    print(f"Total exams processed: {processed_exams}")
    print(f"Total questions collected: {total_questions}")

if __name__ == "__main__":
    collect_benchmark()
