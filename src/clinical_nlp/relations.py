from __future__ import annotations

import re
import uuid

from clinical_nlp.context import sentence_bounds
from clinical_nlp.schemas import Concept, ConceptType, Relation, RelationType

DOSAGE_PATTERN = re.compile(
    r"\s*(\d+(?:\.\d+)?\s*(?:mg|mcg|g|units?|mL|viên|ống)(?:\s*x\s*\d+)?)\b",
    re.I,
)
LAB_VALUE_PATTERN = re.compile(
    r"[\s:)]*(?:\([^)]*\)\s*)?"
    r"(?:(?:cho\s+thấy|ghi\s+nhận|kết\s+quả)\s+[^.;,\n]{0,80}?\s+)?"
    r"(?:is|=|:|là|tăng\s+là|tăng\s+nhẹ\s+lên|tăng\s+lên|tăng|"
    r"giảm\s+còn|giảm|cao\s+là|trả\s+về\s+là)?\s*"
    r"("
    r"\d+(?:[.,]\d+)*(?:\s*(?:mg/dL|mg/dl|mmol/L|mmol/l|mEq/L|G/L|g/l|%))?"
    r"|không\s+có\s+gì\s+đáng\s+chú\s+ý"
    r"|không\s+có\s+bất\s+thường(?:\s+[^.;,\n()]+)?"
    r"|không\s+ghi\s+nhận\s+gì\s+bất\s+thường"
    r"|không\s+phát\s+hiện(?:\s+[^.;,\n()]+)?"
    r"|chưa\s+phát\s+hiện(?:\s+[^.;,\n()]+)?"
    r"|không\s+thấy(?:\s+[^.;,\n()]+)?"
    r"|tế\s+bào\s+bất\s+thường"
    r"|bất\s+thường"
    r"|nhịp\s+xoang"
    r"|âm\s+tính"
    r"|dương\s+tính"
    r"|bình\s+thường"
    r"|tăng\s+nhẹ"
    r"|tăng\s+cao"
    r"|tăng"
    r"|giảm"
    r"|cao"
    r"|thấp"
    r")\b",
    re.I,
)
LAB_VALUE_SEARCH_PATTERN = re.compile(
    r"("
    r"\d+(?:[.,]\d+)*(?:\s*(?:mg/dL|mg/dl|mmol/L|mmol/l|mEq/L|G/L|g/l|%))?"
    r"|không\s+có\s+gì\s+đáng\s+chú\s+ý"
    r"|không\s+có\s+bất\s+thường(?:\s+[^.;,\n()]+)?"
    r"|không\s+ghi\s+nhận\s+gì\s+bất\s+thường"
    r"|không\s+phát\s+hiện(?:\s+[^.;,\n()]+)?"
    r"|chưa\s+phát\s+hiện(?:\s+[^.;,\n()]+)?"
    r"|không\s+thấy(?:\s+[^.;,\n()]+)?"
    r"|tế\s+bào\s+bất\s+thường"
    r"|bất\s+thường"
    r"|nhịp\s+xoang"
    r"|âm\s+tính"
    r"|dương\s+tính"
    r"|bình\s+thường"
    r"|tăng\s+nhẹ"
    r"|tăng\s+cao"
    r"|tăng"
    r"|giảm"
    r"|cao"
    r"|thấp"
    r")\b",
    re.I,
)


def extract_relations(text: str, concepts: list[Concept]) -> list[Relation]:
    relations: list[Relation] = []
    medications = [
        concept for concept in concepts if concept.concept_type == ConceptType.MEDICATION
    ]
    diseases = [concept for concept in concepts if concept.concept_type == ConceptType.DISEASE]
    labs = [concept for concept in concepts if concept.concept_type == ConceptType.LAB_RESULT]

    for medication in medications:
        relations.extend(_dosage_relations(text, medication))
        for disease in diseases:
            if _same_sentence(text, medication, disease) and _has_treatment_cue(
                text, medication, disease
            ):
                evidence_start, evidence_end = _combined_evidence(text, medication, disease)
                relations.append(
                    Relation(
                        relation_id=str(uuid.uuid4()),
                        type=RelationType.TREATS,
                        source_concept_id=medication.concept_id,
                        target_concept_id=disease.concept_id,
                        confidence=0.76,
                        evidence_start_offset=evidence_start,
                        evidence_end_offset=evidence_end,
                    )
                )

    for lab in labs:
        relations.extend(_lab_value_relations(text, lab))

    return relations


def _dosage_relations(text: str, medication: Concept) -> list[Relation]:
    _, sentence_end = sentence_bounds(text, medication.start_offset, medication.end_offset)
    tail = text[medication.end_offset:sentence_end]
    match = DOSAGE_PATTERN.match(tail)
    if match is None:
        return []
    value_start = medication.end_offset + match.start(1)
    value_end = medication.end_offset + match.end(1)
    return [
        Relation(
            relation_id=str(uuid.uuid4()),
            type=RelationType.HAS_DOSAGE,
            source_concept_id=medication.concept_id,
            target_concept_id=None,
            confidence=0.82,
            evidence_start_offset=medication.start_offset,
            evidence_end_offset=value_end,
            value=text[value_start:value_end],
        )
    ]


def _lab_value_relations(text: str, lab: Concept) -> list[Relation]:
    _, sentence_end = sentence_bounds(text, lab.start_offset, lab.end_offset)
    tail = text[lab.end_offset:sentence_end]
    match = LAB_VALUE_PATTERN.match(tail)
    if match is None:
        match = LAB_VALUE_SEARCH_PATTERN.search(tail)
    if match is None:
        return []
    value_start = lab.end_offset + match.start(1)
    value_end = lab.end_offset + match.end(1)
    return [
        Relation(
            relation_id=str(uuid.uuid4()),
            type=RelationType.HAS_VALUE,
            source_concept_id=lab.concept_id,
            target_concept_id=None,
            confidence=0.82,
            evidence_start_offset=lab.start_offset,
            evidence_end_offset=value_end,
            value=text[value_start:value_end],
        )
    ]


def _same_sentence(text: str, left: Concept, right: Concept) -> bool:
    left_bounds = sentence_bounds(text, left.start_offset, left.end_offset)
    right_bounds = sentence_bounds(text, right.start_offset, right.end_offset)
    return left_bounds == right_bounds


def _has_treatment_cue(text: str, medication: Concept, disease: Concept) -> bool:
    start = min(medication.end_offset, disease.end_offset)
    end = max(medication.start_offset, disease.start_offset)
    between = text[start:end].lower()
    return any(cue in between for cue in (" for ", " treat", " due to ", " cho ", " điều trị "))


def _combined_evidence(text: str, left: Concept, right: Concept) -> tuple[int, int]:
    sentence_start, sentence_end = sentence_bounds(
        text,
        min(left.start_offset, right.start_offset),
        max(left.end_offset, right.end_offset),
    )
    return sentence_start, sentence_end
