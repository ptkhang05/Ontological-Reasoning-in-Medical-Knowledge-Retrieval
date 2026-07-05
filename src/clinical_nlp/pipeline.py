from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Protocol

from clinical_nlp.context import infer_context
from clinical_nlp.deidentification import Deidentifier
from clinical_nlp.external import build_external_extractor_from_env
from clinical_nlp.extraction import CandidateConcept, RuleBasedExtractor, remove_overlaps
from clinical_nlp.relations import extract_relations
from clinical_nlp.review import build_review_flags
from clinical_nlp.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    Concept,
    ConceptType,
    ExternalEntity,
    NormalizedConcept,
    ProcessingMetadata,
    WarningMessage,
)
from clinical_nlp.terminology import TerminologyStore

EXTERNAL_TYPE_MAP = {
    "SYMPTOM": ConceptType.SYMPTOM,
    "TRIỆU_CHỨNG": ConceptType.SYMPTOM,
    "TRIEU_CHUNG": ConceptType.SYMPTOM,
    "LAB_RESULT": ConceptType.LAB_RESULT,
    "LAB_NAME": ConceptType.LAB_RESULT,
    "LAB_VALUE": ConceptType.LAB_RESULT,
    "TÊN_XÉT_NGHIỆM": ConceptType.LAB_RESULT,
    "TEN_XET_NGHIEM": ConceptType.LAB_RESULT,
    "KẾT_QUẢ_XÉT_NGHIỆM": ConceptType.LAB_RESULT,
    "KET_QUA_XET_NGHIEM": ConceptType.LAB_RESULT,
    "DISEASE": ConceptType.DISEASE,
    "DIAGNOSIS": ConceptType.DISEASE,
    "CHẨN_ĐOÁN": ConceptType.DISEASE,
    "CHAN_DOAN": ConceptType.DISEASE,
    "MEDICATION": ConceptType.MEDICATION,
    "DRUG": ConceptType.MEDICATION,
    "THUỐC": ConceptType.MEDICATION,
    "THUOC": ConceptType.MEDICATION,
}


class ExternalExtractor(Protocol):
    def extract(self, text: str) -> list[ExternalEntity]:
        raise NotImplementedError


class ClinicalPipeline:
    def __init__(
        self,
        terminology: TerminologyStore | None = None,
        deidentifier: Deidentifier | None = None,
        external_extractor: ExternalExtractor | None = None,
    ) -> None:
        terminology_path = Path("data/terminologies")
        self._terminology = terminology or TerminologyStore.default(terminology_path)
        self._deidentifier = deidentifier or Deidentifier()
        self._extractor = RuleBasedExtractor(self._terminology)
        self._external_extractor = external_extractor or build_external_extractor_from_env()

    def analyze(self, request: AnalyzeRequest) -> AnalyzeResponse:
        deidentified = self._deidentifier.deidentify(request.text)
        external_used = False
        warnings: list[WarningMessage] = []

        if request.language.lower() not in {"en", "vi"}:
            warnings.append(
                WarningMessage(
                    code="LANGUAGE_NOT_DEFAULT",
                    message=(
                        "Only English and Vietnamese are configured; "
                        "text is processed best-effort."
                    ),
                )
            )

        candidates = self._extractor.extract(deidentified.processed_text)

        if request.options.allow_external_inference:
            if self._external_extractor is None:
                warnings.append(
                    WarningMessage(
                        code="EXTERNAL_INFERENCE_NOT_CONFIGURED",
                        message="External inference was requested but no provider is configured.",
                    )
                )
            else:
                try:
                    external_entities = self._external_extractor.extract(
                        deidentified.processed_text
                    )
                except Exception:
                    warnings.append(
                        WarningMessage(
                            code="EXTERNAL_INFERENCE_FAILED",
                            message="External inference failed; rule-based output was used.",
                        )
                    )
                else:
                    external_used = True
                    candidates = remove_overlaps(
                        [
                            *candidates,
                            *self._external_entities_to_candidates(
                                deidentified.processed_text,
                                external_entities,
                            ),
                        ]
                    )

        concepts = [
            self._candidate_to_concept(request.text, candidate) for candidate in candidates
        ]
        relations = extract_relations(request.text, concepts)
        warnings.extend(self._normalization_warnings(concepts))
        review_flags = build_review_flags(
            request.text,
            concepts,
            confidence_threshold=request.options.confidence_threshold,
            external_inference_used=external_used,
        )

        return AnalyzeResponse(
            document_id=request.document_id,
            concepts=concepts,
            relations=relations,
            review_flags=review_flags,
            warnings=warnings,
            processing_metadata=ProcessingMetadata(
                model_versions=self._model_versions(external_used),
                terminology_releases=self._terminology.releases(),
                external_inference_used=external_used,
                deidentification_applied=deidentified.changed,
            ),
        )

    def _candidate_to_concept(self, text: str, candidate: CandidateConcept) -> Concept:
        original_text = text[candidate.start_offset : candidate.end_offset]
        normalized = NormalizedConcept()
        if candidate.concept_type in {ConceptType.DISEASE, ConceptType.MEDICATION}:
            entry = self._terminology.lookup(original_text, candidate.concept_type)
            if entry is None and candidate.source == "fuzzy_terminology_match":
                entry = self._terminology.lookup_fuzzy(
                    original_text,
                    candidate.concept_type,
                )
            if entry is not None:
                normalized = NormalizedConcept(
                    code_system=entry.code_system,
                    code=entry.code,
                    preferred_term=entry.preferred_term,
                    confidence=0.93,
                    source_url=entry.source_url,
                )

        return Concept(
            concept_id=str(uuid.uuid4()),
            text=original_text,
            start_offset=candidate.start_offset,
            end_offset=candidate.end_offset,
            concept_type=candidate.concept_type,
            normalized=normalized,
            context=infer_context(text, candidate.start_offset, candidate.end_offset),
            confidence=candidate.confidence,
            source=candidate.source,
        )

    def _normalization_warnings(self, concepts: list[Concept]) -> list[WarningMessage]:
        warnings: list[WarningMessage] = []
        for concept in concepts:
            if (
                concept.concept_type in {ConceptType.DISEASE, ConceptType.MEDICATION}
                and (concept.normalized.code_system is None or concept.normalized.code is None)
            ):
                warnings.append(
                    WarningMessage(
                        code="UNMAPPED_CODE",
                        message=f"{concept.concept_type} concept could not be normalized.",
                    )
                )
        return warnings

    def _external_entities_to_candidates(
        self, processed_text: str, entities: list[ExternalEntity]
    ) -> list[CandidateConcept]:
        candidates: list[CandidateConcept] = []
        for entity in entities:
            candidate = self._external_entity_to_candidate(processed_text, entity)
            if candidate is not None:
                candidates.append(candidate)
        return candidates

    def _external_entity_to_candidate(
        self, processed_text: str, entity: ExternalEntity
    ) -> CandidateConcept | None:
        entity_text_value = entity.get("text")
        entity_type_value = entity.get("type")
        if not isinstance(entity_text_value, str) or not isinstance(entity_type_value, str):
            return None

        entity_text = entity_text_value.strip()
        concept_type = EXTERNAL_TYPE_MAP.get(entity_type_value.strip().upper())
        if not entity_text or concept_type is None:
            return None

        position = _external_position(entity.get("position"))
        span = _resolve_external_span(processed_text, entity_text, position)
        if span is None:
            return None

        start, end = span
        return CandidateConcept(
            text=processed_text[start:end],
            start_offset=start,
            end_offset=end,
            concept_type=concept_type,
            confidence=_external_confidence(entity.get("confidence")),
            source="external_inference",
        )

    def _model_versions(self, external_used: bool) -> dict[str, str]:
        versions = {"rule_based_extractor": "0.1.0"}
        if external_used and self._external_extractor is not None:
            versions["external_extractor"] = type(self._external_extractor).__name__
        return versions


def _external_position(value: object) -> tuple[int, int] | None:
    if not isinstance(value, list) or len(value) != 2:
        return None
    start_raw, end_raw = value
    if type(start_raw) is not int or type(end_raw) is not int:
        return None
    return start_raw, end_raw


def _resolve_external_span(
    processed_text: str,
    entity_text: str,
    position: tuple[int, int] | None,
) -> tuple[int, int] | None:
    if position is not None:
        start, end = position
        if 0 <= start < end <= len(processed_text):
            source_slice = processed_text[start:end]
            if source_slice == entity_text:
                return start, end
            trimmed = source_slice.strip()
            if trimmed == entity_text:
                leading_spaces = len(source_slice) - len(source_slice.lstrip())
                fixed_start = start + leading_spaces
                return fixed_start, fixed_start + len(entity_text)

    spans = _find_text_spans(processed_text, entity_text)
    if not spans:
        return None
    if position is None:
        return spans[0]
    return min(spans, key=lambda span: abs(span[0] - position[0]))


def _find_text_spans(processed_text: str, entity_text: str) -> list[tuple[int, int]]:
    spans = [
        (match.start(), match.end())
        for match in re.finditer(re.escape(entity_text), processed_text)
    ]
    if spans:
        return spans
    return [
        (match.start(), match.end())
        for match in re.finditer(
            re.escape(entity_text),
            processed_text,
            flags=re.IGNORECASE,
        )
    ]


def _external_confidence(value: object) -> float:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return max(0.0, min(1.0, float(value)))
    return 0.76
