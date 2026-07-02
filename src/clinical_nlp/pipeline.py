from __future__ import annotations

import uuid
from pathlib import Path
from typing import Protocol

from clinical_nlp.context import infer_context
from clinical_nlp.deidentification import Deidentifier
from clinical_nlp.extraction import CandidateConcept, RuleBasedExtractor
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
        self._external_extractor = external_extractor

    def analyze(self, request: AnalyzeRequest) -> AnalyzeResponse:
        deidentified = self._deidentifier.deidentify(request.text)
        external_used = False
        warnings: list[WarningMessage] = []

        if request.language.lower() != "en":
            warnings.append(
                WarningMessage(
                    code="LANGUAGE_NOT_DEFAULT",
                    message="English is the v1 default; non-English text is processed best-effort.",
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
                self._external_extractor.extract(deidentified.processed_text)
                external_used = True

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
                model_versions={"rule_based_extractor": "0.1.0"},
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
