from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

MAX_TEXT_LENGTH = 20_000


def to_camel(value: str) -> str:
    parts = value.split("_")
    return parts[0] + "".join(part.capitalize() for part in parts[1:])


class CamelModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        extra="forbid",
        populate_by_name=True,
        use_enum_values=True,
    )


class ConceptType(StrEnum):
    SYMPTOM = "SYMPTOM"
    LAB_RESULT = "LAB_RESULT"
    DISEASE = "DISEASE"
    MEDICATION = "MEDICATION"
    PATIENT_INFO = "PATIENT_INFO"


class Polarity(StrEnum):
    PRESENT = "PRESENT"
    NEGATED = "NEGATED"
    POSSIBLE = "POSSIBLE"


class Subject(StrEnum):
    PATIENT = "PATIENT"
    FAMILY = "FAMILY"
    OTHER = "OTHER"


class Temporality(StrEnum):
    CURRENT = "CURRENT"
    HISTORICAL = "HISTORICAL"
    HYPOTHETICAL = "HYPOTHETICAL"


class RelationType(StrEnum):
    TREATS = "TREATS"
    HAS_DOSAGE = "HAS_DOSAGE"
    HAS_VALUE = "HAS_VALUE"
    ASSOCIATED_WITH = "ASSOCIATED_WITH"


class AnalyzeOptions(CamelModel):
    allow_external_inference: bool = False
    confidence_threshold: float = Field(default=0.80, ge=0, le=1)
    include_unmapped: bool = True


class AnalyzeRequest(CamelModel):
    text: str = Field(min_length=1, max_length=MAX_TEXT_LENGTH)
    document_id: str | None = None
    document_type: str | None = None
    encounter_date: str | None = None
    language: str = "vi"
    options: AnalyzeOptions = Field(default_factory=AnalyzeOptions)

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be blank")
        return value


class NormalizedConcept(CamelModel):
    code_system: str | None = None
    code: str | None = None
    preferred_term: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    source_url: str | None = None


class ContextAttributes(CamelModel):
    polarity: Polarity = Polarity.PRESENT
    subject: Subject = Subject.PATIENT
    temporality: Temporality = Temporality.CURRENT


class Concept(CamelModel):
    concept_id: str
    text: str
    start_offset: int = Field(ge=0)
    end_offset: int = Field(ge=0)
    concept_type: ConceptType
    normalized: NormalizedConcept = Field(default_factory=NormalizedConcept)
    context: ContextAttributes = Field(default_factory=ContextAttributes)
    confidence: float = Field(ge=0, le=1)
    source: str


class Relation(CamelModel):
    relation_id: str
    type: RelationType
    source_concept_id: str
    target_concept_id: str | None = None
    confidence: float = Field(ge=0, le=1)
    evidence_start_offset: int = Field(ge=0)
    evidence_end_offset: int = Field(ge=0)
    value: str | None = None


class ReviewFlag(CamelModel):
    concept_id: str | None = None
    reason: str
    requires_human_review: bool = True
    message: str


class WarningMessage(CamelModel):
    code: str
    message: str


class ProcessingMetadata(CamelModel):
    pipeline_version: str = "0.1.0"
    model_versions: dict[str, str] = Field(default_factory=dict)
    terminology_releases: dict[str, str] = Field(default_factory=dict)
    processed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    external_inference_used: bool = False
    deidentification_applied: bool = False


class AnalyzeResponse(CamelModel):
    document_id: str | None = None
    concepts: list[Concept] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)
    review_flags: list[ReviewFlag] = Field(default_factory=list)
    warnings: list[WarningMessage] = Field(default_factory=list)
    processing_metadata: ProcessingMetadata = Field(default_factory=ProcessingMetadata)


class ErrorDetail(CamelModel):
    loc: list[str | int]
    msg: str
    type: str


class APIErrorBody(CamelModel):
    code: str
    message: str
    details: list[ErrorDetail] | None = None


class APIError(CamelModel):
    error: APIErrorBody


ExternalEntity = dict[str, Any]
