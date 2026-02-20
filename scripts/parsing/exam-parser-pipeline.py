# Automated exam parsing pipeline with OpenAI Vision/API backends

import argparse
import os
import json
import base64
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import re
import sys
import time

# Ensure repository root is importable when script is run as a file.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.agents import AgentRuntime, AgentRuntimeError

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

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# ==================== CONFIGURATION ====================

class Config:
    """Global configuration for the pipeline"""
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
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


TRUTHY_VALUES = {"1", "true", "yes", "y", "on"}
FALSY_VALUES = {"0", "false", "no", "n", "off"}
PARSE_STATUS_VALUES = {status.value for status in ParseStatus}

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
        return {
            'question_id': self.question_id,
            'question_type': self.question_type.value if self.question_type else None,
            'question_text': self.question_text,
            'answer_type': self.answer_type.value if self.answer_type else None,
            'max_score': self.max_score,
            'canonical_answer': self.canonical_answer,
            'status': self.status.value,
            'error_comment': self.error_comment,
        }

@dataclass
class PageParseResult:
    """Result of parsing a single exam page"""
    exam_id: str
    page_id: int
    path_to_page: str
    parsed_questions: List[Question] = field(default_factory=list)
    processing_time: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict_for_jsonl(self) -> Dict:
        """
        Formats the object strictly according to the requirement:
        Contains exam_id, page_id, path_to_page, and list of questions.
        """
        return {
            "exam_id": self.exam_id,
            "page_id": self.page_id,
            "path_to_page": self.path_to_page,
            "parsed_questions": [q.to_dict() for q in self.parsed_questions]
        }

@dataclass
class ExamMetadata:
    exam_id: str
    source: str
    exam_pdf_path: str
    pages_dir: str
    questions_path: str  # ADDED: path to the questions.jsonl file
    total_score: int = 100
    language: str = "en"
    # REMOVED: key_pdf_path

    def to_dict(self) -> Dict:
        return {
            "exam_id": self.exam_id,
            "source": self.source,
            "exam_pdf_path": self.exam_pdf_path,
            "pages_dir": self.pages_dir,
            "questions_path": self.questions_path,
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

    PROMPT_PATH = Path(__file__).parent / "prompt.txt"

    if not PROMPT_PATH.exists():
        raise FileNotFoundError(
            f"MARKER_PROMPT file not found at {PROMPT_PATH}"
        )

    MARKER_PROMPT = PROMPT_PATH.read_text(encoding="utf-8")

    def __init__(self, agent_runtime: Optional[AgentRuntime] = None):
        self.agent_runtime = agent_runtime
        self.client = (
            OpenAI(api_key=Config.OPENAI_API_KEY, base_url=Config.OPENAI_BASE_URL)
            if OpenAI is not None
            else None
        )
        if self.client is None and self.agent_runtime is None:
            raise RuntimeError(
                "openai package is required for legacy parser mode. "
                "Install with: pip install openai"
            )
        self.logger = logging.getLogger(__name__)

    def encode_image(self, image_path: str) -> str:
        """Encode image to base64"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def parse_page(self, image_path: str, exam_id: str, page_id: int) -> PageParseResult:
        start_time = time.time()

        try:
            if self.agent_runtime is not None:
                response_text = self.agent_runtime.parse_page(
                    image_path=image_path,
                    exam_id=exam_id,
                    page_id=page_id,
                    marker_prompt=self.MARKER_PROMPT,
                )
            else:
                if self.client is None:
                    raise RuntimeError("openai package is required for legacy parser mode")
                base64_image = self.encode_image(image_path)

                # o1 models require max_completion_tokens.
                token_param = "max_completion_tokens" if "o1" in Config.MODEL_MARKER else "max_tokens"

                payload = {
                    "model": Config.MODEL_MARKER,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": self.MARKER_PROMPT},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                                },
                            ],
                        }
                    ],
                }
                payload[token_param] = Config.MAX_TOKENS_MARKER

                response = self.client.chat.completions.create(**payload)
                response_text = response.choices[0].message.content

            questions = self._parse_response(response_text, page_id)

            processing_time = time.time() - start_time
            self.logger.info(f"Parsed page {page_id} for exam {exam_id}: {len(questions)} questions")

            return PageParseResult(
                exam_id=exam_id,
                page_id=page_id,
                path_to_page=image_path,
                parsed_questions=questions,
                processing_time=processing_time,
            )

        except AgentRuntimeError as e:
            self.logger.error(f"Agent error while parsing page {page_id}: status={e.status} error={e.message}")
            return PageParseResult(exam_id=exam_id, page_id=page_id, path_to_page=image_path)
        except Exception as e:
            self.logger.error(f"Error parsing page {page_id}: {e}")
            return PageParseResult(exam_id=exam_id, page_id=page_id, path_to_page=image_path)

    def _parse_response(self, response_text: str, page_id: int) -> List[Question]:
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
            status_str = str(item.get("status", ParseStatus.OK.value) or ParseStatus.OK.value)
            if status_str not in PARSE_STATUS_VALUES:
                status_str = ParseStatus.OK.value

            # If error/unreadable, return minimal question
            if status_str in {ParseStatus.ERROR.value, ParseStatus.UNREADABLE.value}:
                return Question(
                    question_id=item.get("question_id"),
                    status=ParseStatus(status_str),
                    error_comment=item.get("error_comment"),
                )

            # Validate required fields for OK status
            question_id = item.get("question_id")
            if not question_id:
                self.logger.warning("Missing question_id")
                return None

            question_text = item.get("question_text", "").strip()
            if not question_text:
                return Question(
                    question_id=question_id,
                    status=ParseStatus.ERROR,
                    error_comment="Missing question_text",
                )

            # max_score: if missing or invalid, default to 0
            max_score = item.get("max_score")
            try:
                max_score = int(max_score) if max_score else 0
            except (ValueError, TypeError):
                max_score = 0

            # canonical_answer can be an empty string
            canonical_answer = item.get("canonical_answer", "")
            if canonical_answer is None:
                canonical_answer = ""
            canonical_answer = str(canonical_answer).strip()

            # Extract and validate question_type
            q_type = item.get("question_type", QuestionType.TEXT.value)
            try:
                q_type = QuestionType(q_type)
            except ValueError:
                q_type = QuestionType.TEXT

            # Extract and validate answer_type
            answer_type = item.get("answer_type", AnswerType.TEXT.value)
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
    def __init__(
        self,
        *,
        agent_enabled: bool = True,
        agent_max_steps: int = 6,
        agent_config_path: Optional[Path] = None,
    ):
        Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        Config.ERRORS_DIR.mkdir(parents=True, exist_ok=True)
        Config.PAGES_DIR.mkdir(parents=True, exist_ok=True)
        
        self.logger = self._setup_logging()
        self.pdf_processor = PDFProcessor()
        agent_runtime: Optional[AgentRuntime] = None

        if agent_enabled:
            parser_model_url = self._resolve_parser_model_url()
            try:
                agent_runtime = AgentRuntime(
                    model_url=parser_model_url,
                    model_name=Config.MODEL_MARKER,
                    api_key=Config.OPENAI_API_KEY,
                    config_path=agent_config_path,
                    max_steps=agent_max_steps,
                    sandbox="docker",
                    tools_profile="full",
                    timeout_sec=Config.TIMEOUT,
                )
                self.logger.info(
                    "Parser agent runtime initialized successfully "
                    f"(model={Config.MODEL_MARKER}, sandbox=docker)"
                )
            except Exception as exc:
                self.logger.error(f"Failed to initialize parser agent runtime: {exc}")
                self.logger.error("Falling back to legacy direct OpenAI parser mode.")
                agent_runtime = None

        self.marker = ExamMarker(agent_runtime=agent_runtime)

    def _resolve_parser_model_url(self) -> str:
        base = (Config.OPENAI_BASE_URL or "").strip().rstrip("/")
        if not base:
            base = "https://api.openai.com/v1"
        if base.endswith("/chat/completions"):
            return base
        return f"{base}/chat/completions"

    def _setup_logging(self) -> logging.Logger:
        log_file = Config.OUTPUT_DIR / "pipeline.log"

        logger = logging.getLogger("ExamPipeline")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            fh = logging.FileHandler(log_file)
            sh = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            fh.setFormatter(formatter)
            sh.setFormatter(formatter)
            logger.addHandler(fh)
            logger.addHandler(sh)

        return logger

    def process_exam(self, 
                    pdf_path: str, 
                    exam_id: Optional[str] = None,
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

            # Define path for questions.jsonl
            questions_jsonl_path = exam_output_dir / "questions.jsonl"

            # Create exam metadata (MODIFIED: removed key_pdf_path, added questions_path)
            metadata = ExamMetadata(
                exam_id=exam_id,
                source=source,
                exam_pdf_path=pdf_path,
                pages_dir=str(exam_pages_dir),
                questions_path=str(questions_jsonl_path), # Added
                language="en"
            )
            
            metadata_path = exam_output_dir / "exam.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata.to_dict(), f, indent=2)
            
            self.logger.info(f"Saved exam metadata to {metadata_path}")

            # Parse each page
            all_page_results = [] # Store PageParseResult objects
            all_questions_flat = [] # Keep for summary stats
            error_questions = []

            for page_num, image_path in page_images:
                self.logger.info(f"Parsing page {page_num}/{len(page_images)}")
                result = self.marker.parse_page(image_path, exam_id, page_num)
                
                all_page_results.append(result)
                all_questions_flat.extend(result.parsed_questions)
                
                error_questions.extend([
                    q for q in result.parsed_questions 
                    if q.status in [ParseStatus.ERROR, ParseStatus.UNREADABLE]
                ])

            # Save results (MODIFIED: Pass full page results, not flat questions)
            self._save_results(exam_output_dir, all_page_results)
            self._save_error_report(exam_id, error_questions)

            # Generate summary
            summary = self._generate_summary(exam_id, all_questions_flat)
            
            return {
                "status": "success",
                "exam_id": exam_id,
                "summary": summary,
                "output_dir": str(exam_output_dir)
            }

        except Exception as e:
            self.logger.error(f"Critical error processing exam {exam_id}: {e}")
            return self._error_result(exam_id, str(e))

    def _save_results(self, output_dir: Path, page_results: List[PageParseResult]):
        """Save results to JSON files in the requested grouped format"""
        # Save questions.jsonl (MODIFIED STRUCTURE)
        questions_file = output_dir / "questions.jsonl"
        with open(questions_file, 'w') as f:
            for page_res in page_results:
                # Use to_dict_for_jsonl to match exact required structure
                f.write(json.dumps(page_res.to_dict_for_jsonl()) + '\n')

        self.logger.info(f"Saved {len(page_results)} pages to {questions_file}")

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
        with open(pdf_path, "rb") as pdf_file:
            file_hash = hashlib.md5(pdf_file.read()).hexdigest()[:8]
        return f"{filename}_{file_hash}"

    def _error_result(self, exam_id: str, error_msg: str) -> Dict:
        """Return error result"""
        return {
            "status": "error",
            "exam_id": exam_id,
            "error": error_msg
        }

# ==================== MAIN ====================

def _str_to_bool(value: str) -> bool:
    normalized = (value or "").strip().lower()
    if normalized in TRUTHY_VALUES:
        return True
    if normalized in FALSY_VALUES:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def main():
    parser = argparse.ArgumentParser(description="Exam parsing pipeline with optional smolagents runtime")
    parser.add_argument(
        "--agent-enabled",
        type=_str_to_bool,
        default=True,
        help="Enable agent runtime for parsing (default: true)",
    )
    parser.add_argument(
        "--agent-max-steps",
        type=int,
        default=6,
        help="Maximum reasoning steps for parser agent (default: 6)",
    )
    parser.add_argument(
        "--agent-config",
        type=Path,
        default=Path("scripts/agents/agent_config.yaml"),
        help="Path to agent config YAML",
    )
    parser.add_argument(
        "--model-marker",
        type=str,
        default=Config.MODEL_MARKER,
        help="Model name used by parser (default: gpt-4.1-mini)",
    )
    parser.add_argument(
        "--openai-base-url",
        type=str,
        default=Config.OPENAI_BASE_URL,
        help="OpenAI-compatible base URL (default: OPENAI_BASE_URL env or https://api.openai.com/v1)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=Config.OPENAI_API_KEY,
        help="Optional API key override (default: OPENAI_API_KEY env)",
    )
    args = parser.parse_args()

    Config.MODEL_MARKER = args.model_marker
    Config.OPENAI_BASE_URL = args.openai_base_url
    Config.OPENAI_API_KEY = args.api_key

    pipeline = ExamPipeline(
        agent_enabled=args.agent_enabled,
        agent_max_steps=args.agent_max_steps,
        agent_config_path=args.agent_config,
    )

    exams_dir = Config.EXAMS_DIR
    results = []

    if not exams_dir.exists():
        logging.warning(f"Exams directory {exams_dir} does not exist. Please create it and add PDFs.")
        return

    for pdf_path in sorted(exams_dir.glob("*.pdf")):
        exam_id = pdf_path.stem

        result = pipeline.process_exam(
            pdf_path=str(pdf_path),
            exam_id=exam_id,
            source=""
        )

        results.append(result)

    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
