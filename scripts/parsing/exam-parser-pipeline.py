# Пайплайн автоматической разметки экзаменов с ChatGPT / OpenAI Vision API

import os
import json
import base64
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime
from enum import Enum
import re

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    logging.warning("PyMuPDF not available. Install with: pip install PyMuPDF")

try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False
    logging.warning("pdf2image not available. Install with: pip install pdf2image")

from openai import OpenAI

# ==================== CONFIGURATION ====================

class Config:
    """Global configuration for the pipeline"""
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-api-key-here")
    MODEL_MARKER = "gpt-4.1-mini"
    MAX_TOKENS_MARKER = 3000
    
    BASE_DIR = Path("./exam_data")
    EXAMS_DIR = BASE_DIR / "exams"
    PAGES_DIR = BASE_DIR / "pages"
    OUTPUT_DIR = BASE_DIR / "output"
    ERRORS_DIR = OUTPUT_DIR / "errors"
    
    IMAGE_FORMAT = "png"
    DPI = 150
    BATCH_SIZE = 5
    TIMEOUT = 30

# ==================== ENUMS ====================

class QuestionType(str, Enum):
    TEXT = "text"
    MULTIMODAL = "multimodal"

class AnswerType(str, Enum):
    SINGLE_CHOICE = "single_choice"
    MULTIPLE_CHOICE = "multiple_choice"
    NUMERIC = "numeric"
    TEXT = "text"
    ORDERING = "ordering"
    STRUCTURE = "structure"
    FULL_SYNTHESIS = "full_synthesis"
    REACTION_DESCRIPTION = "reaction_description"
    PROPERTY_DETERMINATION = "property_determination"

class ParseStatus(str, Enum):
    OK = "ok"
    UNREADABLE = "unreadable"
    ERROR = "error"

# ==================== DATA CLASSES ====================

@dataclass
class Question:
    """Parsed question from exam page"""
    question_id: int
    question_type: Optional[QuestionType] = None
    question_text: Optional[str] = None
    answer_type: Optional[AnswerType] = None
    max_score: Optional[int] = None
    canonical_answer: Optional[str] = None
    status: ParseStatus = ParseStatus.OK
    error_comment: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        d = {
            'question_id': self.question_id,
            'question_type': self.question_type.value if self.question_type else None,
            'question_text': self.question_text,
            'answer_type': self.answer_type.value if self.answer_type else None,
            'max_score': self.max_score,
            'canonical_answer': self.canonical_answer,
            'status': self.status.value,
            'error_comment': self.error_comment,
        }
        return d

@dataclass
class PageParseResult:
    """Result of parsing a single exam page"""
    exam_id: str
    page_id: int
    path_to_page: str
    parsed_questions: List[Question] = field(default_factory=list)
    processing_time: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict:
        return {
            "exam_id": self.exam_id,
            "page_id": self.page_id,
            "path_to_page": self.path_to_page,
            "parsed_questions": [q.to_dict() for q in self.parsed_questions],
            "processing_time": self.processing_time,
            "timestamp": self.timestamp,
            "stats": {
                "total_questions": len(self.parsed_questions),
                "ok_count": sum(1 for q in self.parsed_questions if q.status == ParseStatus.OK),
                "error_count": sum(1 for q in self.parsed_questions if q.status != ParseStatus.OK),
            }
        }

@dataclass
class ExamMetadata:
    """Exam metadata"""
    exam_id: str
    source: str
    exam_pdf_path: str
    key_pdf_path: Optional[str] = None
    pages_dir: str = ""
    total_score: int = 100
    language: str = "en"

    def to_dict(self) -> Dict:
        return {
            "exam_id": self.exam_id,
            "source": self.source,
            "exam_pdf_path": self.exam_pdf_path,
            "key_pdf_path": self.key_pdf_path,
            "pages_dir": self.pages_dir,
            "total_score": self.total_score,
            "language": self.language,
        }

# ==================== PDF PROCESSING ====================

class PDFProcessor:
    """Handles PDF to image conversion"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._check_dependencies()

    def _check_dependencies(self):
        """Check if required dependencies are available"""
        if not PYMUPDF_AVAILABLE and not PDF2IMAGE_AVAILABLE:
            raise RuntimeError(
                "Neither PyMuPDF nor pdf2image is installed. "
                "Install one: pip install PyMuPDF  OR  pip install pdf2image"
            )

    def pdf_to_images(self, pdf_path: str, output_dir: str) -> List[Tuple[int, str]]:
        """Convert PDF to images (one per page)"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        try:
            if PYMUPDF_AVAILABLE:
                return self._pdf_to_images_pymupdf(pdf_path, output_dir)
            else:
                return self._pdf_to_images_pdf2image(pdf_path, output_dir)
        except Exception as e:
            self.logger.error(f"Error converting PDF {pdf_path}: {e}")
            raise

    def _pdf_to_images_pymupdf(self, pdf_path: str, output_dir: str) -> List[Tuple[int, str]]:
        """Use PyMuPDF for conversion"""
        page_images = []
        with fitz.open(pdf_path) as doc:
            for page_num in range(len(doc)):
                page = doc[page_num]
                pix = page.get_pixmap(matrix=fitz.Matrix(Config.DPI/72, Config.DPI/72))
                image_path = Path(output_dir) / f"page_{page_num + 1}.{Config.IMAGE_FORMAT}"
                pix.save(str(image_path))
                page_images.append((page_num + 1, str(image_path)))
                self.logger.info(f"Extracted page {page_num + 1} from {Path(pdf_path).name}")
        return page_images

    def _pdf_to_images_pdf2image(self, pdf_path: str, output_dir: str) -> List[Tuple[int, str]]:
        """Use pdf2image for conversion"""
        page_images = []
        images = convert_from_path(pdf_path, dpi=Config.DPI)
        for page_num, image in enumerate(images, 1):
            image_path = Path(output_dir) / f"page_{page_num}.{Config.IMAGE_FORMAT}"
            image.save(str(image_path), Config.IMAGE_FORMAT.upper())
            page_images.append((page_num, str(image_path)))
            self.logger.info(f"Extracted page {page_num} from {Path(pdf_path).name}")
        return page_images

# ==================== EXAM MARKER ====================

class ExamMarker:
    """Uses OpenAI Vision to extract and mark questions"""
    
    MARKER_PROMPT = """
    You are an expert chemistry exam OCR specialist.

Your goal is to extract ALL questions and their corresponding answers from the provided exam page image and return them in a structured JSON format.

Analyze the image carefully and identify ALL distinct questions/tasks and answers visible on the page.

────────────────────────
GENERAL RULES
────────────────────────

• Extract ONLY what is explicitly visible in the image.
• DO NOT infer, guess, or hallucinate missing content.
• Even if a question is partially unreadable, extract all readable parts.
• If a page contains ONLY instructions, rules, or reference tables (no actual questions), return an empty array: [].

────────────────────────
QUESTION IDENTIFICATION
────────────────────────

question_id
• Represents the main question number as shown in the exam.
• Use only the main number: “1”, “2”, “3”, etc.

SUB-QUESTIONS (a, b, c, …):

DEFAULT RULE
• Treat sub-questions as ONE SINGLE COMBINED QUESTION.

EXCEPTION — SPLIT sub-questions ONLY IF ALL conditions apply:
• Sub-questions are logically independent
• Each sub-question expects its OWN answer type
• Combining them would cause loss of answer_type correctness

If split is required:
• Create separate questions with the SAME question_id
• Preserve sub-labels inside question_text:
“A) …”, “B) …”

Otherwise:
• Merge all sub-parts into ONE question_text
• Preserve structure using:
“A) … B) … C) …”

Example (combined):
“3. (a) Name the product. (b) Explain the mechanism.”
→ ONE question:
question_id: “3”
question_text: “A) Name the product. B) Explain the mechanism.”

────────────────────────
QUESTION TYPE
────────────────────────

question_type — choose ONE:

“text”
• Question contains ONLY text.
• No visual elements are required to answer.

“multimodal”
• Question includes text PLUS required visual information such as:
	•	Chemical structures
	•	Reaction schemes
	•	Graphs or plots
	•	Spectra (NMR, MS, IR, etc.)
	•	Diagrams or images
	•	Visually presented answer options

If non-textual information is required to understand or answer → use “multimodal”.

────────────────────────
QUESTION TEXT
────────────────────────

question_text
• MUST include the FULL question content AND ALL available context.
• ALWAYS merge context into text, including:
	•	Answer options
	•	Labels
	•	Given values
	•	Reaction schemes (described textually)
• Include all sub-parts using:
“A) … B) … C) …”

IMPORTANT — CONTEXT DESCRIPTION RULE
• NEVER write vague placeholders like:
“(scheme shown)”, “(image above)”, “(see diagram)”
• ALWAYS attempt to describe the visible context in words:
	•	reagents
	•	arrows
	•	positions of blanks
	•	labels near structures

If context cannot be adequately described → mark status = “unreadable”.

MANDATORY WORD NORMALIZATION
If ANY of the following appear in the image:
circle, highlight, mark, underline, box, blank

They MUST be replaced in question_text with:
choose / select / explicit placeholders {box_1}, {box_2}

Example:
“Circle the correct answer” → “Choose the correct answer”

Answer options MUST be normalized as:
“A) … B) … C) …” OR “1) … 2) … 3) …”

────────────────────────
ANSWER TYPE
────────────────────────

answer_type — choose ONE
(choose the MOST COMPLEX type if sub-parts differ):

single_choice
• Exactly ONE correct option
• canonical_answer: “{A}”

multiple_choice
• Multiple correct options
• canonical_answer: “{A;B;D}”

numeric
• Numerical value(s): mass, m/z, yield, chemical shift, etc.
• “{3.5}” or “{23;49;4.5}”

text
• Short free-text answer
• “{text}”

ordering
• Ordering / ranking task
• canonical_answer MUST contain ONLY indices
• “{1;4;6;5}”
• NEVER include item names

structure
• Chemical structure identification
• “{text}”

full_synthesis
• Multi-step synthesis with reagents
• “{text}”

reaction_description
• Reaction or mechanism description
• “{text}”

property_determination
• Chemical properties (acidity, aromaticity, stereochemistry, spectra)

FILL-IN CASES
If placeholders exist, canonical_answer MUST map values explicitly.

Example:
question_text:
“Complete the reaction: Na + {box_1} → Cr + {box_2}”

canonical_answer:
“{box_1=Cl; box_2=Mg}”

────────────────────────
MAX SCORE
────────────────────────

max_score
• Extract TOTAL points for the question if visible
• If missing or unclear → set to 0

────────────────────────
CANONICAL ANSWER
────────────────────────

canonical_answer — ALWAYS REQUIRED.

Rules:
• ONLY the answer, NO explanations
• Preserve sub-structure if present:
“A) …; B) …; C) …”
• If answer cannot be extracted → “” (empty string)
• NEVER omit this field

────────────────────────
STATUS AND ERRORS
────────────────────────

status — one of:
“ok”
“unreadable”

Set status = “unreadable” IF:
	1.	Parts of question_text are unreadable or missing
	2.	canonical_answer cannot be confidently extracted
	3.	The task requires drawing / sketching / handwriting

IMPORTANT
• Even if status = “unreadable”, extract ALL readable fields.

error_comment
• Explain EXACTLY what could not be read
• Otherwise set to null

────────────────────────
OUTPUT FORMAT
────────────────────────

Return ONLY a valid JSON array.

ALL fields are REQUIRED for EACH question:
question_id, question_type, question_text, answer_type, max_score, canonical_answer, status, error_comment

────────────────────────
EXAMPLE OUTPUT
────────────────────────

[
  {
    "question_id": "1",
    "question_type": "text",
    "question_text": "What is the IUPAC name of 2-methylpropene?",
    "answer_type": "text",
    "max_score": 5,
    "canonical_answer": "2-methylpropene or isobutylene",
    "status": "ok",
    "error_comment": null
  },
{
  "question_id": "2",
  "question_type": "multimodal",
  "question_text": "A) Complete the reaction: phenol + Br2 → ? (reaction scheme shown)",
  "answer_type": "structure",
  "max_score": 8,
  "canonical_answer": "A) 2,4,6-tribromophenol",
  "status": "ok",
  "error_comment": null
},
  {
    "question_id": "3",
    "question_type": null,
    "question_text": null,
    "answer_type": null,
    "max_score": null,
    "canonical_answer": "",
    "status": "unreadable",
    "error_comment": "Chemical structure diagram is too blurry to read"
  }
]
"""

    def __init__(self):
        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
        self.logger = logging.getLogger(__name__)

    def encode_image(self, image_path: str) -> str:
        """Encode image to base64"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def parse_page(self, image_path: str, exam_id: str, page_id: int) -> PageParseResult:
        """Parse exam page using OpenAI Vision"""
        import time
        start_time = time.time()

        try:
            base64_image = self.encode_image(image_path)

            response = self.client.chat.completions.create(
                model=Config.MODEL_MARKER,
                max_tokens=Config.MAX_TOKENS_MARKER,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": self.MARKER_PROMPT,
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}",
                                },
                            },
                        ],
                    }
                ],
            )

            response_text = response.choices[0].message.content
            questions = self._parse_response(response_text, exam_id, page_id)

            processing_time = time.time() - start_time
            self.logger.info(
                f"Parsed page {page_id} for exam {exam_id}: "
                f"{len(questions)} questions in {processing_time:.2f}s"
            )

            return PageParseResult(
                exam_id=exam_id,
                page_id=page_id,
                path_to_page=image_path,
                parsed_questions=questions,
                processing_time=processing_time,
            )

        except Exception as e:
            self.logger.error(f"Error parsing page {page_id}: {e}")
            return PageParseResult(
                exam_id=exam_id,
                page_id=page_id,
                path_to_page=image_path,
                parsed_questions=[],
                processing_time=time.time() - start_time,
            )

    def _parse_response(self, response_text: str, exam_id: str, page_id: int) -> List[Question]:
        """Parse JSON response from model"""
        questions = []
        
        try:
            # Extract JSON from response
            json_match = re.search(r'\[[\s\S]*\]', response_text)
            if not json_match:
                self.logger.warning(f"No JSON array found in response for page {page_id}")
                return []

            json_str = json_match.group(0)
            parsed_data = json.loads(json_str)

            if not isinstance(parsed_data, list):
                self.logger.warning(f"Expected list, got {type(parsed_data)}")
                return []

            # Return empty if no questions (skip instruction pages)
            if len(parsed_data) == 0:
                self.logger.info(f"Page {page_id} contains no questions (skipped)")
                return []

            for item in parsed_data:
                question = self._construct_question(item)
                if question:
                    questions.append(question)

            return questions

        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parsing error on page {page_id}: {e}")
            return []
        except Exception as e:
            self.logger.error(f"Unexpected error parsing response: {e}")
            return []

    def _construct_question(self, item: Dict) -> Optional[Question]:
        """Construct Question object from parsed item"""
        try:
            # Extract status first
            status_str = item.get('status', 'ok')
            if status_str not in [ps.value for ps in ParseStatus]:
                status_str = 'ok'

            # If error/unreadable, return minimal question
            if status_str in ['error', 'unreadable']:
                return Question(
                    question_id=item.get('question_id'),
                    status=ParseStatus(status_str),
                    error_comment=item.get('error_comment'),
                )

            # Validate required fields for OK status
            question_id = item.get('question_id')
            if not question_id:
                self.logger.warning("Missing question_id")
                return None

            question_text = item.get('question_text', '').strip()
            if not question_text:
                return Question(
                    question_id=question_id,
                    status=ParseStatus.ERROR,
                    error_comment="Missing question_text",
                )

            # max_score: если отсутствует или не парсится → 0
            max_score = item.get('max_score')
            try:
                max_score = int(max_score) if max_score else 0
            except (ValueError, TypeError):
                max_score = 0

            # canonical_answer: может быть пустой строкой
            canonical_answer = item.get('canonical_answer', '')
            if canonical_answer is None:
                canonical_answer = ''
            canonical_answer = str(canonical_answer).strip()

            # Extract and validate question_type
            q_type = item.get('question_type', 'text')
            try:
                q_type = QuestionType(q_type)
            except ValueError:
                q_type = QuestionType.TEXT

            # Extract and validate answer_type
            answer_type = item.get('answer_type', 'text')
            try:
                answer_type = AnswerType(answer_type)
            except ValueError:
                answer_type = AnswerType.TEXT

            return Question(
                question_id=question_id,
                question_type=q_type,
                question_text=question_text,
                answer_type=answer_type,
                max_score=max_score,
                canonical_answer=canonical_answer,
                status=ParseStatus.OK,
                error_comment=None,
            )

        except Exception as e:
            self.logger.error(f"Failed to construct question: {e}")
            return None

# ==================== PIPELINE ORCHESTRATOR ====================

class ExamPipeline:
    """Main orchestrator for the exam marking pipeline"""
    
    def __init__(self):
        self.logger = self._setup_logging()
        self.pdf_processor = PDFProcessor()
        self.marker = ExamMarker()
        Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        Config.ERRORS_DIR.mkdir(parents=True, exist_ok=True)

    def _setup_logging(self) -> logging.Logger:
        """Configure logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(Config.OUTPUT_DIR / 'pipeline.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)

    def process_exam(self, 
                    pdf_path: str, 
                    exam_id: Optional[str] = None,
                    key_pdf_path: Optional[str] = None,
                    source: str = "unknown") -> Dict:
        """Process a single exam PDF"""
        if exam_id is None:
            exam_id = self._generate_exam_id(pdf_path)
        
        self.logger.info(f"Processing exam: {exam_id} from {pdf_path}")
        
        exam_pages_dir = Config.PAGES_DIR / exam_id
        exam_output_dir = Config.OUTPUT_DIR / exam_id
        exam_pages_dir.mkdir(parents=True, exist_ok=True)
        exam_output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Convert PDF to images
            self.logger.info(f"Converting PDF to images: {pdf_path}")
            page_images = self.pdf_processor.pdf_to_images(
                pdf_path, 
                str(exam_pages_dir)
            )
            
            if not page_images:
                self.logger.error(f"Failed to extract images from {pdf_path}")
                return self._error_result(exam_id, "No pages extracted from PDF")

            # Create exam metadata
            metadata = ExamMetadata(
                exam_id=exam_id,
                source=source,
                exam_pdf_path=pdf_path,
                key_pdf_path=key_pdf_path,
                pages_dir=str(exam_pages_dir),
                language="en"
            )
            
            metadata_path = exam_output_dir / "exam.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata.to_dict(), f, indent=2)
            
            self.logger.info(f"Saved exam metadata to {metadata_path}")

            # Parse each page
            all_results = []
            all_questions = []
            error_questions = []

            for page_num, image_path in page_images:
                self.logger.info(f"Parsing page {page_num}/{len(page_images)}")
                result = self.marker.parse_page(image_path, exam_id, page_num)
                all_results.append(result)
                all_questions.extend(result.parsed_questions)
                
                error_questions.extend([
                    q for q in result.parsed_questions 
                    if q.status in [ParseStatus.ERROR, ParseStatus.UNREADABLE]
                ])

            # Save results
            self._save_results(exam_id, exam_output_dir, all_questions)
            self._save_error_report(exam_id, error_questions)

            # Generate summary
            summary = self._generate_summary(exam_id, all_questions)
            
            return {
                "status": "success",
                "exam_id": exam_id,
                "summary": summary,
                "output_dir": str(exam_output_dir)
            }

        except Exception as e:
            self.logger.error(f"Critical error processing exam {exam_id}: {e}")
            return self._error_result(exam_id, str(e))

    def _save_results(self, exam_id: str, output_dir: Path, 
                     all_questions: List[Question]):
        """Save results to JSON files"""
        # Save questions.jsonl
        questions_file = output_dir / "questions.jsonl"
        with open(questions_file, 'w') as f:
            for q in all_questions:
                f.write(json.dumps(q.to_dict()) + '\n')

        self.logger.info(f"Saved {len(all_questions)} questions to {questions_file}")

    def _save_error_report(self, exam_id: str, error_questions: List[Question]):
        """Save error report for manual review"""
        if not error_questions:
            return

        error_report = {
            "exam_id": exam_id,
            "timestamp": datetime.now().isoformat(),
            "total_errors": len(error_questions),
            "errors": [q.to_dict() for q in error_questions]
        }

        error_file = Config.ERRORS_DIR / f"{exam_id}_errors.json"
        with open(error_file, 'w') as f:
            json.dump(error_report, f, indent=2)

        self.logger.warning(
            f"Found {len(error_questions)} error(s) for exam {exam_id}. "
            f"See {error_file}"
        )

    def _generate_summary(self, exam_id: str, 
                         all_questions: List[Question]) -> Dict:
        """Generate processing summary"""
        total_questions = len(all_questions)
        ok_count = sum(1 for q in all_questions if q.status == ParseStatus.OK)
        error_count = sum(1 for q in all_questions if q.status != ParseStatus.OK)
        total_score = sum(q.max_score or 0 for q in all_questions if q.status == ParseStatus.OK)
        
        by_question_type = {}
        by_answer_type = {}
        
        for q in all_questions:
            if q.question_type and q.status == ParseStatus.OK:
                by_question_type[q.question_type.value] = by_question_type.get(q.question_type.value, 0) + 1
            if q.answer_type and q.status == ParseStatus.OK:
                by_answer_type[q.answer_type.value] = by_answer_type.get(q.answer_type.value, 0) + 1

        return {
            "exam_id": exam_id,
            "total_questions": total_questions,
            "ok_questions": ok_count,
            "error_questions": error_count,
            "error_rate": f"{(error_count/total_questions*100):.1f}%" if total_questions > 0 else "0%",
            "total_score": total_score,
            "by_question_type": by_question_type,
            "by_answer_type": by_answer_type,
        }

    def _generate_exam_id(self, pdf_path: str) -> str:
        """Generate unique exam ID from PDF"""
        filename = Path(pdf_path).stem
        file_hash = hashlib.md5(open(pdf_path, 'rb').read()).hexdigest()[:8]
        return f"{filename}_{file_hash}"

    def _error_result(self, exam_id: str, error_msg: str) -> Dict:
        """Return error result"""
        return {
            "status": "error",
            "exam_id": exam_id,
            "error": error_msg
        }

# ==================== MAIN ====================

def main():
    """Example usage"""
    pipeline = ExamPipeline()
    
    pdf_path = "./exam_data/exams/exam_1.pdf"
    result = pipeline.process_exam(
        pdf_path=pdf_path,
        exam_id="exam_1",
        source="mit_spring_2005"
    )
    
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
