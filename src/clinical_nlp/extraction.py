from __future__ import annotations

import re
from dataclasses import dataclass

from clinical_nlp.schemas import ConceptType
from clinical_nlp.terminology import TerminologyStore


@dataclass(frozen=True)
class CandidateConcept:
    text: str
    start_offset: int
    end_offset: int
    concept_type: ConceptType
    confidence: float
    source: str


SYMPTOM_TERMS = (
    "shortness of breath",
    "chest pain",
    "headache",
    "fever",
    "cough",
    "nausea",
    "vomiting",
    "dizziness",
)

LAB_NAMES = (
    "hemoglobin a1c",
    "hba1c",
    "glucose",
    "sodium",
    "potassium",
    "creatinine",
)


class RuleBasedExtractor:
    def __init__(self, terminology: TerminologyStore) -> None:
        self._terminology = terminology

    def extract(self, text: str) -> list[CandidateConcept]:
        candidates: list[CandidateConcept] = []
        candidates.extend(
            self._extract_terms(
                text, SYMPTOM_TERMS, ConceptType.SYMPTOM, 0.86, "rule_symptom"
            )
        )
        candidates.extend(self._extract_labs(text))
        candidates.extend(
            self._extract_terms(
                text,
                self._terminology.search_terms_for(ConceptType.DISEASE),
                ConceptType.DISEASE,
                0.90,
                "terminology_match",
            )
        )
        candidates.extend(
            self._extract_terms(
                text,
                self._terminology.search_terms_for(ConceptType.MEDICATION),
                ConceptType.MEDICATION,
                0.90,
                "terminology_match",
            )
        )
        candidates.extend(self._extract_patient_info(text))
        candidates.extend(self._extract_unknown_medications(text, candidates))
        candidates.extend(self._extract_unknown_diseases(text, candidates))
        return remove_overlaps(candidates)

    def _extract_terms(
        self,
        text: str,
        terms: tuple[str, ...] | list[str],
        concept_type: ConceptType,
        confidence: float,
        source: str,
    ) -> list[CandidateConcept]:
        candidates: list[CandidateConcept] = []
        for term in sorted(set(terms), key=lambda value: (-len(value), value.lower())):
            pattern = re.compile(rf"(?<!\w){re.escape(term)}(?!\w)", re.I)
            for match in pattern.finditer(text):
                candidates.append(
                    CandidateConcept(
                        text=text[match.start() : match.end()],
                        start_offset=match.start(),
                        end_offset=match.end(),
                        concept_type=concept_type,
                        confidence=confidence,
                        source=source,
                    )
                )
        return candidates

    def _extract_labs(self, text: str) -> list[CandidateConcept]:
        candidates: list[CandidateConcept] = []
        for lab_name in LAB_NAMES:
            pattern = re.compile(rf"(?<!\w){re.escape(lab_name)}(?!\w)", re.I)
            for match in pattern.finditer(text):
                candidates.append(
                    CandidateConcept(
                        text=text[match.start() : match.end()],
                        start_offset=match.start(),
                        end_offset=match.end(),
                        concept_type=ConceptType.LAB_RESULT,
                        confidence=0.84,
                        source="rule_lab",
                    )
                )
        return candidates

    def _extract_patient_info(self, text: str) -> list[CandidateConcept]:
        patterns = (
            re.compile(r"\b\d{1,3}[- ]year[- ]old\b", re.I),
            re.compile(r"\b(?:male|female|man|woman)\b", re.I),
        )
        candidates: list[CandidateConcept] = []
        for pattern in patterns:
            for match in pattern.finditer(text):
                candidates.append(
                    CandidateConcept(
                        text=text[match.start() : match.end()],
                        start_offset=match.start(),
                        end_offset=match.end(),
                        concept_type=ConceptType.PATIENT_INFO,
                        confidence=0.82,
                        source="rule_patient_info",
                    )
                )
        return candidates

    def _extract_unknown_medications(
        self, text: str, existing: list[CandidateConcept]
    ) -> list[CandidateConcept]:
        candidates: list[CandidateConcept] = []
        pattern = re.compile(r"\b(?:takes|taking|started|on)\s+([A-Za-z][A-Za-z-]{2,})\b", re.I)
        for match in pattern.finditer(text):
            start, end = match.span(1)
            if self._span_has_existing(start, end, existing):
                continue
            token = text[start:end]
            if token.lower() in {term.lower() for term in SYMPTOM_TERMS}:
                continue
            candidates.append(
                CandidateConcept(
                    text=token,
                    start_offset=start,
                    end_offset=end,
                    concept_type=ConceptType.MEDICATION,
                    confidence=0.55,
                    source="heuristic_medication",
                )
            )
        return candidates

    def _extract_unknown_diseases(
        self, text: str, existing: list[CandidateConcept]
    ) -> list[CandidateConcept]:
        candidates: list[CandidateConcept] = []
        pattern = re.compile(
            r"\b(?:diagnosed with|history of|has)\s+([A-Za-z][A-Za-z-]{4,})\b",
            re.I,
        )
        symptom_terms = {term.lower() for term in SYMPTOM_TERMS}
        for match in pattern.finditer(text):
            start, end = match.span(1)
            token = text[start:end]
            if self._span_has_existing(start, end, existing) or token.lower() in symptom_terms:
                continue
            candidates.append(
                CandidateConcept(
                    text=token,
                    start_offset=start,
                    end_offset=end,
                    concept_type=ConceptType.DISEASE,
                    confidence=0.55,
                    source="heuristic_disease",
                )
            )
        return candidates

    @staticmethod
    def _span_has_existing(start: int, end: int, existing: list[CandidateConcept]) -> bool:
        return any(
            start < candidate.end_offset and candidate.start_offset < end
            for candidate in existing
        )


def remove_overlaps(candidates: list[CandidateConcept]) -> list[CandidateConcept]:
    chosen: list[CandidateConcept] = []
    for candidate in sorted(
        candidates,
        key=lambda item: (
            -(item.end_offset - item.start_offset),
            -item.confidence,
            item.start_offset,
        ),
    ):
        if any(
            candidate.start_offset < existing.end_offset
            and existing.start_offset < candidate.end_offset
            for existing in chosen
        ):
            continue
        chosen.append(candidate)
    return sorted(chosen, key=lambda item: (item.start_offset, item.end_offset))
