from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


AnswerTypeLiteral = Literal[
    "single_choice",
    "multiple_choice",
    "numeric",
    "ordering",
    "structure",
    "full_synthesis",
    "reaction_description",
    "property_determination",
    "msms_structure_prediction",
    "text",
]

QuestionTypeLiteral = Literal["text", "multimodal"]
StatusLiteral = Literal["ok", "unreadable", "error"]


class JudgeResult(BaseModel):
    llm_score: float = Field(ge=0.0, le=1.0)
    llm_comment: str = Field(default="")


class GEvalJudgeResult(BaseModel):
    criteria_steps: list[str] = Field(default_factory=list)
    step_findings: list[str] = Field(default_factory=list)
    rubric_score_0_to_10: int = Field(ge=0, le=10)
    llm_comment: str = Field(min_length=1)


class StudentGuardOutput(BaseModel):
    final_answer: str = Field(min_length=1)
    format_ok: bool = True


class ParsedQuestionSchema(BaseModel):
    question_id: int | str | None = None
    question_type: Optional[QuestionTypeLiteral] = None
    question_text: Optional[str] = None
    answer_type: Optional[AnswerTypeLiteral] = None
    max_score: int = 0
    canonical_answer: str = ""
    status: StatusLiteral = "ok"
    error_comment: Optional[str] = None

    @field_validator("max_score", mode="before")
    @classmethod
    def _normalize_max_score(cls, value):
        if value in (None, ""):
            return 0
        return int(value)

    @field_validator("canonical_answer", mode="before")
    @classmethod
    def _normalize_canonical_answer(cls, value):
        if value is None:
            return ""
        return str(value)

    @field_validator("question_text", mode="before")
    @classmethod
    def _normalize_question_text(cls, value):
        if value is None:
            return None
        return str(value).strip()

    @field_validator("error_comment", mode="before")
    @classmethod
    def _normalize_error_comment(cls, value):
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @model_validator(mode="after")
    def _validate_status_fields(self):
        if self.status == "ok":
            if self.question_id in (None, ""):
                raise ValueError("question_id is required when status=ok")
            if not (self.question_text or "").strip():
                raise ValueError("question_text is required when status=ok")
            if self.answer_type is None:
                raise ValueError("answer_type is required when status=ok")
        return self


def parsed_questions_to_dicts(items: list[ParsedQuestionSchema]) -> list[dict]:
    return [item.model_dump() for item in items]
