# AI Assistant Benchmark for Organic Chemistry & Mass Spectrometry (MS2)

**Innopolis University — Artificial Intelligence Laboratory in Chemistry**

---

## 📘 Project Overview

This repository contains the official benchmark dataset, documentation, and evaluation tools for a specialized AI assistant focused on:

- **Organic Chemistry**  
- **Tandem Mass Spectrometry (MS2)**  
- **Chemical Reaction Product Structure Analysis**  

This benchmark is designed to rigorously evaluate AI model performance using real academic tests sourced from:

- Top global universities  
- Undergraduate + graduate chemistry exams  
- Olympiads  
- Specialized MS2 interpretation exercises  
- Official textbook problem sets  

The benchmark supports fine-grained assessment of reasoning, interpretation, and domain-specific problem-solving in chemistry and mass spectrometry.

This project accompanies the “AI Assistant for Organic Chemistry and Mass Spectrometry” project, whose goal is to:
> Develop a digital AI assistant to optimize MS2 data interpretation and improve the accuracy of molecular structure predictions for reaction products.

---

## 🧪 Benchmark Structure

Each entry in the benchmark follows the unified schema:

```json
{
  "exam_id": "string",
  "page_id": "integer",
  "path_to_page": "string",
  "question_id": "string",
  "question_type": "text | mcq | numeric | short_answer",
  "question_text": "string",
  "answer_type": "single_choice | multiple_choice | numeric | text",
  "max_score": "integer",
  "canonical_answer": "string or array",
  "status": "ok | error",
  "error_comment": "string or null"
}

```

**Supported answer types (answer_type):**

- `single_choice` — Single answer choice
- `multiple_choice` — Multiple answer choices
- `numeric` — Numeric answer
- `text` — Text answer
- `ordering` — Ordering/ranking
- `structure` — Structure determination
- `full_synthesis` — Full synthesis
- `reaction_description` — Reaction description
- `property_determination` — Property determination
- `msms_structure_prediction` — MS/MS spectra interpretation

## 📁 Repository Structure

```bash
SpectralixBenchmark/
│
├── README.md
│
├── benchmark/ # Final benchmark dataset or its parts
│
├── data/
│   ├── mass_spec/ # Mass spectrometry data
│   └── organic/ # Organic chemistry data
│
├── scripts/
│   ├── parsing/
│   │   ├── exam_data/
│   │   │   ├── exams/ # Exam PDFs that passed the parser
│   │   │   ├── full_exams/ # Full list of exam PDFs
│   │   │   ├── output/ # Parsed exams
│   │   │   └── pages/ # Exam *png pages
│   │   ├── iterations/
│   │   ├── benchmark_collection.py
│   │   ├── exam-parser-pipeline.py
│   │   ├── requirements.txt
│   │   └── README.md
│   ├── evaluation/ # Evaluation scripts
│
└── evaluation/
    ├── baseline_results/
    │   └── README.md
    └── rag_results/
        └── README.md
```
## 🤝 How to Contribute

We welcome contributions from:

- Chemists  
- Mass spectrometry experts  
- ML researchers  
- Students and engineers  

### 1. Add new exam sources

Place PDFs or links into:

```bash
data/raw/organic/
data/raw/mass_spec/
```

### 2. Add new benchmark questions

Create JSONL entries following the official schema:

```bash
data/benchmark//.jsonl
```

Each entry must include:

- Clean question text  
- University + year + page  
- Canonical answer  
- Rubric  
- Topic tags  
- Difficulty  

### 3. Add new RAG materials

Add textbooks, glossaries, lecture notes into:
```bash
data/rag_corpus/
```

### 4. Propose improvements

Open issues for:

- Schema changes  
- Evaluation methodology  
- Partial credit policy  
- New chemistry domains  
- MS2 spectrum datasets  

---

## 🔬 Project Focus Areas

### Organic Chemistry

- Reaction mechanisms  
- Stereochemistry  
- Functional group transformations  
- Reagents, conditions, selectivity  
- Structure–property relationships  

### Mass Spectrometry (MS2)

- Ionization methods  
- Analyzers  
- Fragmentation patterns  
- Spectrum interpretation  
- Reaction product identification  
- Quantitative analysis  

---

## ✨ Project Goals

- Build a high-quality academic benchmark for chemistry and MS2.  
- Enable evaluation of LLM reasoning in chemical sciences.  
- Support the development of a digital AI assistant integrated with MS2 pipelines.  
- Provide a reliable dataset for researchers, students, and industry practitioners.  

---

## 📬 Contacts

Maintainer (Innopolis University — AI Lab in Chemistry):

**Ivan Golov**  
**Email:** i.golov@innopolis.university  
**Telegram:** [https://t.me/Ione_Golov](https://t.me/Ione_Golov)  

If you’re interested in contributing, collaborating, or integrating the benchmark — feel free to reach out.
