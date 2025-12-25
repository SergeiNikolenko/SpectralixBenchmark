import json
import os
from pathlib import Path
from typing import List, Dict

# ================= CONFIGURATION =================

# Папка, где лежат результаты работы пайплайна (папки по exam_id)
INPUT_BASE_DIR = Path("./exam_data/output")

# Итоговый файл
OUTPUT_FILE = Path("./benchmark_dataset.jsonl")

# ================= MAIN LOGIC =================

def collect_benchmark():
    if not INPUT_BASE_DIR.exists():
        print(f"Error: Directory {INPUT_BASE_DIR} does not exist.")
        return

    print(f"Scanning {INPUT_BASE_DIR} for questions.jsonl files...")
    
    total_questions = 0
    processed_exams = 0
    
    # Открываем итоговый файл на запись
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as out_f:
        
        # Перебираем все папки внутри output (exam_1, exam_2, ...)
        # Сортируем, чтобы порядок был детерминированным
        for exam_dir in sorted(INPUT_BASE_DIR.iterdir()):
            
            if not exam_dir.is_dir() or exam_dir.name == "errors":
                continue

            jsonl_path = exam_dir / "questions.jsonl"
            
            if not jsonl_path.exists():
                print(f"Skipping {exam_dir.name}: questions.jsonl not found.")
                continue

            print(f"Processing {exam_dir.name}...")
            
            try:
                # Читаем файл страницы за страницей
                with open(jsonl_path, 'r', encoding='utf-8') as in_f:
                    for line_num, line in enumerate(in_f, 1):
                        if not line.strip():
                            continue
                            
                        try:
                            page_data = json.loads(line)
                        except json.JSONDecodeError:
                            print(f"  Warning: Invalid JSON in {jsonl_path} at line {line_num}")
                            continue

                        # Извлекаем метаданные страницы
                        # Эти поля будут добавлены к каждому вопросу
                        page_meta = {
                            "exam_id": page_data.get("exam_id"),
                            "page_id": page_data.get("page_id"),
                            "path_to_page": page_data.get("path_to_page")
                        }
                        
                        questions_list = page_data.get("parsed_questions", [])
                        
                        # Проходим по каждому вопросу на этой странице
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