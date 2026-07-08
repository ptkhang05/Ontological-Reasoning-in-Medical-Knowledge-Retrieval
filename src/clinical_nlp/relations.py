from __future__ import annotations

import re
import uuid

from clinical_nlp.context import sentence_bounds
from clinical_nlp.schemas import Concept, ConceptType, Relation, RelationType

LAB_VALUE_UNIT = r"mg/dL|mg/dl|mmol/L|mmol/l|mEq/L|G/L|g/l|cm|mm|%"
LAB_NUMERIC_CORE = r"(?:[<>]=?|≥|≤)?\s*\d+(?:[.,]\d+)*"
LAB_NUMERIC_VALUE = (
    rf"{LAB_NUMERIC_CORE}"
    rf"(?:\s*(?:-->|->|[-–—]|đến|lên|xuống)\s*{LAB_NUMERIC_CORE})?"
    rf"(?:\s*(?:{LAB_VALUE_UNIT}))?"
)
DOSAGE_PATTERN = re.compile(
    r"\s*(\d+(?:\.\d+)?\s*(?:mg|mcg|g|units?|mL|viên|ống)(?:\s*x\s*\d+)?)\b",
    re.I,
)
LAB_VALUE_PATTERN = re.compile(
    r"[\s:)]*(?:\([^)]*\)\s*)?"
    r"(?:(?:toàn\s+phần|bắt\s+đầu)\s+){0,2}"
    r"(?:(?:cho\s+thấy|ghi\s+nhận|kết\s+quả)\s+[^.;,\n]{0,80}?\s+)?"
    r"(?:is|=|:|là|tăng\s+từ|tăng\s+lên\s+đạt\s+đỉnh|tăng\s+là|"
    r"tăng\s+nhẹ\s+lên|tăng\s+lên|tăng(?=\s*(?:[<>]=?|≥|≤)?\d)|"
    r"vẫn\s+giảm\s+xuống|giảm\s+xuống|giảm\s+còn|"
    r"giảm(?=\s*(?:[<>]=?|≥|≤)?\d)|nâng\s+cao\s+lên|cao\s+là|"
    r"trả\s+về\s+là)?\s*"
    r"("
    + LAB_NUMERIC_VALUE +
    r""
    r"|không\s+có\s+gì\s+đáng\s+chú\s+ý"
    r"|không\s+có\s+bất\s+thường(?:\s+[^.;,\n()]+)?"
    r"|không\s+ghi\s+nhận\s+gì\s+bất\s+thường"
    r"|không\s+phát\s+hiện(?:\s+[^.;,\n()]+)?"
    r"|chưa\s+phát\s+hiện(?:\s+[^.;,\n()]+)?"
    r"|không\s+thấy(?:\s+[^.;,\n()]+)?"
    r"|tế\s+bào\s+bất\s+thường"
    r"|bất\s+thường"
    r"|đang\s+chờ"
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
    r")(?=$|[\s.;,\n)])",
    re.I,
)
LAB_VALUE_SEARCH_PATTERN = re.compile(
    r"("
    + LAB_NUMERIC_VALUE +
    r""
    r"|không\s+có\s+gì\s+đáng\s+chú\s+ý"
    r"|không\s+có\s+bất\s+thường(?:\s+[^.;,\n()]+)?"
    r"|không\s+ghi\s+nhận\s+gì\s+bất\s+thường"
    r"|không\s+phát\s+hiện(?:\s+[^.;,\n()]+)?"
    r"|chưa\s+phát\s+hiện(?:\s+[^.;,\n()]+)?"
    r"|không\s+thấy(?:\s+[^.;,\n()]+)?"
    r"|tế\s+bào\s+bất\s+thường"
    r"|bất\s+thường"
    r"|đang\s+chờ"
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
    r")(?=$|[\s.;,\n)])",
    re.I,
)
PREFERRED_LAB_VALUE_SEARCH_PATTERN = re.compile(
    r"(xác\s+suất\s+thấp\s+thuyên\s+tắc\s+phổi)(?=$|[\s.;,\n)])",
    re.I,
)
LAB_VALUE_UNIT_PATTERN = re.compile(
    rf"(?:\b(?:{LAB_VALUE_UNIT.replace('|%', '')})\b|%)",
    re.I,
)
LAB_NUMERIC_PATTERN = re.compile(
    rf"^{LAB_NUMERIC_VALUE}$",
    re.I,
)
LAB_NUMERIC_CUE_PATTERN = re.compile(
    r"(?:=|:|\blà\b|\btăng\s+từ\b|\btăng\s+lên\s+đạt\s+đỉnh\b|"
    r"\btăng(?:\s+(?:là|nhẹ\s+lên|lên))?\b|"
    r"\bvẫn\s+giảm\s+xuống\b|\bgiảm(?:\s+(?:xuống|còn))?\b|"
    r"\bnâng\s+cao\s+lên\b|\bcao\s+là\b|"
    r"\btrả\s+về\s+là\b)",
    re.I,
)
NON_LAB_NUMERIC_CONTEXT_PATTERN = re.compile(
    r"(?:\bđộ\s*$|\bgrade\s*$|\bthứ\s*$|\bcấp\s*$|"
    r"\bxương\s+sườn\s*$|\bgãy\s*$|\bc[0-9]?\s*$)",
    re.I,
)
NON_LAB_NUMERIC_CONTEXT_ANYWHERE_PATTERN = re.compile(
    r"\b(?:xương\s+sườn|hình\s+ảnh\s+gãy|cột\s+sống)\b",
    re.I,
)
NON_LAB_COUNT_AFTER_PATTERN = re.compile(
    r"^\s*(?:viên|lần|stent|ống|mẫu|thủ\s+thuật|tuần|ngày|tháng|năm|giờ|phút)\b",
    re.I,
)
SHORT_TEXTUAL_TREND_VALUES = {"tăng", "giảm", "cao", "thấp"}
NON_LAB_TEXTUAL_TREND_CONTEXT_PATTERN = re.compile(r"(?:\bgóc\s*$|\blàm\s*$)", re.I)
PRECEDING_LAB_VALUE_PATTERN = re.compile(
    rf"({LAB_NUMERIC_VALUE}|âm\s+tính|dương\s+tính|bình\s+thường|"
    r"bất\s+thường|đang\s+chờ)\s*$",
    re.I,
)
PRECEDING_VALUE_LAB_NAMES = {
    "bạch cầu",
    "cấy máu",
    "chem 7",
    "điện giải",
    "guaiac",
    "hồng cầu",
    "lymphocyte",
    "neutrophil",
    "nitrite",
    "xét nghiệm phân tìm cryptosporidium",
}


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
    preceding_relations = _preceding_lab_value_relations(text, lab)
    if preceding_relations:
        return preceding_relations
    tail = text[lab.end_offset:sentence_end]
    match = _find_valid_lab_value_match(text, lab, tail)
    if match is None:
        return []
    value_start, value_end = _trimmed_lab_value_span(text, lab.end_offset, match)
    if value_start >= value_end:
        return []
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


def _preceding_lab_value_relations(text: str, lab: Concept) -> list[Relation]:
    if text[lab.start_offset : lab.end_offset].lower() not in PRECEDING_VALUE_LAB_NAMES:
        return []
    sentence_start, _ = sentence_bounds(text, lab.start_offset, lab.end_offset)
    head = text[sentence_start : lab.start_offset]
    match = PRECEDING_LAB_VALUE_PATTERN.search(head)
    if match is None:
        return []
    value_start = sentence_start + match.start(1)
    value_end = sentence_start + match.end(1)
    raw_value = text[value_start:value_end]
    value_start += len(raw_value) - len(raw_value.lstrip())
    value_end -= len(raw_value) - len(raw_value.rstrip())
    if value_start >= value_end:
        return []
    return [
        Relation(
            relation_id=str(uuid.uuid4()),
            type=RelationType.HAS_VALUE,
            source_concept_id=lab.concept_id,
            target_concept_id=None,
            confidence=0.82,
            evidence_start_offset=value_start,
            evidence_end_offset=lab.end_offset,
            value=text[value_start:value_end],
        )
    ]


def _find_valid_lab_value_match(
    text: str, lab: Concept, tail: str
) -> re.Match[str] | None:
    preferred_match = PREFERRED_LAB_VALUE_SEARCH_PATTERN.search(tail)
    if preferred_match is not None and _is_valid_lab_value_match(
        text, lab, preferred_match
    ):
        return preferred_match

    match = LAB_VALUE_PATTERN.match(tail)
    if match is not None and _is_valid_lab_value_match(text, lab, match):
        return match

    for search_match in LAB_VALUE_SEARCH_PATTERN.finditer(tail):
        if _is_valid_lab_value_match(text, lab, search_match):
            return search_match
    return None


def _is_valid_lab_value_match(
    text: str, lab: Concept, match: re.Match[str]
) -> bool:
    value = match.group(1).strip()
    if LAB_NUMERIC_PATTERN.fullmatch(value) is None:
        value_start, _ = _trimmed_lab_value_span(text, lab.end_offset, match)
        return not _is_non_lab_textual_value_context(text, lab, value, value_start)

    value_start, value_end = _trimmed_lab_value_span(text, lab.end_offset, match)
    between_lab_and_value = text[lab.end_offset:value_start]
    if _is_non_lab_numeric_context(text, value_start, value_end, between_lab_and_value):
        return False
    if LAB_VALUE_UNIT_PATTERN.search(value):
        return True
    if "." in value or "," in value:
        return True
    return True


def _is_non_lab_textual_value_context(
    text: str, lab: Concept, value: str, value_start: int
) -> bool:
    if value.lower() not in SHORT_TEXTUAL_TREND_VALUES:
        return False
    before_value = text[lab.end_offset:value_start].lower()
    return NON_LAB_TEXTUAL_TREND_CONTEXT_PATTERN.search(before_value[-100:]) is not None


def _trimmed_lab_value_span(
    text: str, base_offset: int, match: re.Match[str]
) -> tuple[int, int]:
    raw_start = base_offset + match.start(1)
    raw_end = base_offset + match.end(1)
    raw_value = text[raw_start:raw_end]
    leading_trim = len(raw_value) - len(raw_value.lstrip())
    trailing_trim = len(raw_value) - len(raw_value.rstrip())
    return raw_start + leading_trim, raw_end - trailing_trim


def _is_non_lab_numeric_context(
    text: str, value_start: int, value_end: int, before_value: str
) -> bool:
    stripped_before = before_value.rstrip()
    after_value = text[value_end : min(len(text), value_end + 12)]
    previous_character = text[value_start - 1] if value_start > 0 else ""
    next_character = text[value_end] if value_end < len(text) else ""
    before_context = stripped_before[-80:]

    if (
        next_character in ".,"
        and value_end + 1 < len(text)
        and text[value_end + 1].isdigit()
    ):
        return True
    if previous_character.isalpha() or previous_character in "-/":
        return True
    if next_character in "-/":
        return True
    if NON_LAB_COUNT_AFTER_PATTERN.search(after_value):
        return True
    if NON_LAB_NUMERIC_CONTEXT_ANYWHERE_PATTERN.search(before_context):
        return True
    if re.match(r"\s*[/,-]", after_value, re.I) and NON_LAB_NUMERIC_CONTEXT_PATTERN.search(
        before_context
    ):
        return True
    return NON_LAB_NUMERIC_CONTEXT_PATTERN.search(before_context) is not None


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
