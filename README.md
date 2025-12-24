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
  "example_id": "string",
  "source_file": "string",
  "page": "integer",
  "university": "string",
  "year": "string",
  "question_number": "string",

  "question_type": "mcq | short_answer | numeric | definition | explanation | ordering | mechanism_text | ms_text",

  "question_text": "string",
  "choices": ["string"],
  "context": "string or null",

  "answer_type": "label | numeric | text | order",
  "canonical_answer": {},
  "solution_explanation": "string",

  "rubric": {
    "correct_answer": "string or structured object",
    "partial_credit_steps": { "step_name": 0.5 },
    "tolerances": {
      "numeric_abs": 0.1,
      "numeric_rel": 0.05
    }
  },

  "topic_tags": ["string"],
  "difficulty": 1,
  "notes": "string"
}

```

**Supported question types include:**

- Multiple choice (MCQ)  
- Short answer  
- Numeric problems (with tolerances)  
- Definitions  
- Explanations  
- Ordering tasks  
- Organic chemistry reaction mechanisms (text-based)  
- MS2 spectrum interpretation

## 📁 Repository Structure

```bash
chem-benchmark/
│
├── README.md
├── LICENSE
│
├── docs/
│   ├── benchmark_specification.md
│   ├── annotation_guidelines.md
│   ├── evaluation_methodology.md
│   └── rag_design.md
│
├── data/
│   ├── raw/
│   │   ├── organic/
│   │   └── mass_spec/
│   ├── ai_parsed/
│   ├── benchmark/
│   └── rag_corpus/
│
├── scripts/
│   ├── parsing/
│   ├── evaluation/
│   └── rag/
│
└── evaluation/
    ├── baseline_results/
    ├── rag_results/
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
