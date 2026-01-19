# Parsing and Benchmark Collection Scripts

This folder contains scripts and data for parsing exam PDFs and collecting benchmark datasets from the parsed results.

## Overview

The parsing workflow consists of two main stages:

1. **Exam Parsing Pipeline** - Automatically annotates exam PDFs using OpenAI Vision API
2. **Benchmark Collection** - Aggregates parsed results into a unified benchmark dataset

## Key Files

### Core Scripts

- **`exam-parser-pipeline.py`** - Main pipeline for automated exam parsing
  - Converts PDFs to images
  - Uses OpenAI Vision API (GPT-4.1-mini) to extract questions
  - Validates extracted data for consistency
  - Handles batch processing with error tracking
  - Output: JSON exam data and JSONL question files per exam

- **`benchmark_collection.py`** - Aggregates parsed questions into benchmark dataset
  - Reads all `questions.jsonl` files from parsed exams
  - Combines them into a single standardized `benchmark_dataset.jsonl`
  - Maintains deterministic ordering across runs

- **`analyse.ipynb`** - Jupyter notebook for analyzing parsing results and benchmark quality

### Data Files

- **`benchmark_dataset.jsonl`** - Final benchmark dataset with all parsed questions
- **`benchmark_gold_standard.jsonl`** - Reference gold standard for evaluation (status of parsing = 'ok')
- **`prompt.txt`** - Prompt template used for OpenAI Vision API calls

### Configuration

- **`requirements.txt`** - Python dependencies (openai, PyMuPDF, pdf2image)

## Directory Structure

### `exam_data/`

- **`exams/`** - Source PDF files to be parsed
- **`pages/`** - Extracted page images from PDFs (organized by exam)
- **`output/`** - Parsed results
  - `exam_1/`, `exam_2/`, ... - Per-exam output directories
    - `exam.json` - Extracted exam metadata
    - `questions.jsonl` - Extracted questions in JSONL format
  - `errors/` - Error logs per exam for debugging

### `iterations/`

Historical iteration snapshots of the pipeline:
- `output_1/` through `output-6/` - Different versions of parsing results
- Each contains `comment.txt` for run notes and per-exam outputs

## Usage

Before running the scripts, ensure you have set: export openai_api_key="YOUR-API-KEY-HERE" in your .env file.

### 1. Prepare Exam PDFs
Place PDF files in `exam_data/exams/`

### 2. Run Parsing Pipeline
```bash
python exam-parser-pipeline.py
```
- Converts PDFs to page images
- Sends pages to OpenAI Vision API for extraction
- Generates `exam.json` and `questions.jsonl` per exam
- Logs errors to `errors/` folder

### 3. Collect Benchmark Dataset
```bash
python benchmark_collection.py
```
- Reads all parsed questions from `exam_data/output/`
- Outputs aggregated dataset to `benchmark_dataset.jsonl`

### 4. Analyze Results
Open `analyse.ipynb` for:
- Quality checks on extracted questions
- Statistical analysis of benchmark
- Comparison with gold standard