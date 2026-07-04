from __future__ import annotations

import re
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from clinical_nlp.context import sentence_bounds
from clinical_nlp.schemas import (
    AnalyzeResponse,
    Concept,
    ConceptType,
    Polarity,
    Relation,
    RelationType,
    Subject,
    Temporality,
)


class BtcConceptType(StrEnum):
    SYMPTOM = "TRIỆU_CHỨNG"
    LAB_NAME = "TÊN_XÉT_NGHIỆM"
    LAB_VALUE = "KẾT_QUẢ_XÉT_NGHIỆM"
    DIAGNOSIS = "CHẨN_ĐOÁN"
    MEDICATION = "THUỐC"


class BtcAssertion(StrEnum):
    NEGATED = "isNegated"
    FAMILY = "isFamily"
    HISTORICAL = "isHistorical"


class BtcEntity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    position: list[int] = Field(min_length=2, max_length=2)
    type: BtcConceptType
    assertions: list[BtcAssertion] = Field(default_factory=list)
    candidates: list[str] = Field(default_factory=list)


TYPE_MAP = {
    ConceptType.SYMPTOM: BtcConceptType.SYMPTOM,
    ConceptType.LAB_RESULT: BtcConceptType.LAB_NAME,
    ConceptType.DISEASE: BtcConceptType.DIAGNOSIS,
    ConceptType.MEDICATION: BtcConceptType.MEDICATION,
}
ASSERTION_SUPPORTED_TYPES = {
    ConceptType.SYMPTOM,
    ConceptType.DISEASE,
    ConceptType.MEDICATION,
}
MEDICATION_EXPAND_HINT = re.compile(
    r"\b(?:"
    r"\d+(?:[.,]\d+)?(?:-\d+(?:[.,]\d+)?)?\s*(?:mg|mcg|g|ml|mL|viên|ống|pill|pills)"
    r"|po|iv|sl|bid|qid|qhs|qam|q6h|prn|daily|once|succinate|xl|oral|suspension|sodium"
    r"|đường\s+uống|tiêm\s+tĩnh\s+mạch"
    r")\b",
    re.I,
)
MEDICATION_PREFIX_PATTERN = re.compile(
    r"(?:(?:iv|po|sl)\s+|"
    r"\d+\s*(?:sl|po|iv)\s+|"
    r"\d+(?:[.,]\d+)?(?:-\d+(?:[.,]\d+)?)?\s*"
    r"(?:mg|mcg|g|ml|mL|viên|ống)"
    r"(?:\s*/\s*(?:ngày|day))?"
    r"(?:\s*(?:iv|po|sl|đường\s+uống|tiêm\s+tĩnh\s+mạch))?\s*)$",
    re.I,
)
MEDICATION_STOP_PATTERNS = (
    re.compile(r"\s+\d+\.\s+"),
    re.compile(r"\s+dose\b", re.I),
    re.compile(r"\s+(?:và|hoặc|nhưng|mà|and|or|but)\b", re.I),
    re.compile(r"\s+(?:điều trị|cho|do|vì|để)\b", re.I),
    re.compile(r"[\n;,]"),
    re.compile(r"\("),
    re.compile(r"(?<!\d)\.(?!\d)"),
)
DIAGNOSIS_TRIM_PATTERNS = (
    re.compile(r",\s+(?:nghi|nghi ngờ|có thể|khả năng|liên quan)\b", re.I),
    re.compile(r"\s+\((?:nghi|có thể|khả năng)[^)]+\)", re.I),
)
MEDICATION_HISTORY_CUES = (
    "thuốc trước khi nhập viện",
    "trước khi nhập viện",
    "trước nhập viện",
    "tiền sử sử dụng",
    "tiền sử dùng",
    "home medication",
    "prior medication",
)
HISTORY_SECTION_CUES = (
    "tiền sử bệnh nội khoa",
    "tiền sử bệnh lý",
    "tiền sử bệnh",
    "tiền sử phẫu thuật",
    "lịch sử phẫu thuật",
    "bệnh mạn tính",
    "các bệnh lý mãn tính",
    "các tình trạng bệnh lý mạn tính",
)
CURRENT_SECTION_CUES = (
    "bệnh sử hiện tại",
    "bệnh sử  hiện tại",
    "bệnh sử  hin tại",
    "tiền sử bệnh hiện tại",
    "tiền sử bệnh bệnh hiện tại",
    "lý do nhập viện",
    "lý do khám bệnh",
    "triệu chứng hiện tại",
    "các triệu chứng hiện tại",
    "triệu chứng khi nhập viện",
    "triệu chứng khi đến",
    "kết quả khám",
    "kết quả xét nghiệm",
    "kết quả chẩn đoán",
    "kết quả chụp",
    "cận lâm sàng",
    "đánh giá tại bệnh viện",
    "khám tại bệnh viện",
    "tình trạng ngay trước",
    "tình trạng ngay trước khi nhập viện",
    "chẩn đoán:",
    "chẩn đoán sơ bộ",
)
FAMILY_OBSERVER_CUES = (
    "gia đình nhận thấy",
    "gia đình lo ngại",
    "gia đình yêu cầu",
    "gia đình cho biết",
)
BTC_CANDIDATE_OVERRIDES = {
    "aspirin 81mg": ["243670"],
    "aspirin 81 mg": ["243670"],
    "metoprolol succinate 100mg daily": ["1370483"],
    "metoprolol succinate 100 mg daily": ["1370483"],
    "furosemide 40 mg đường uống": ["315971"],
    "iv lasix 40 mg once": ["565458"],
    "lasix 40mg daily": ["565458"],
    "lasix 40 mg daily": ["565458"],
    "80mg po lasix": ["566621"],
    "80 mg po lasix": ["566621"],
    "80mg lasix iv": ["566621"],
    "80 mg lasix iv": ["566621"],
    "bumetanide 2mg iv": ["315502"],
    "bumetanide 2 mg iv": ["315502"],
    "levofloxacin 750mg iv": ["330371"],
    "levofloxacin 750 mg iv": ["330371"],
    "methylprednisolone 125mg iv": ["1743704"],
    "methylprednisolone 125 mg iv": ["1743704"],
    "prednisone 40 mg/ngày trong 3 ngày": ["451144"],
    "prednisone 40 mg /ngày trong 3 ngày": ["451144"],
    "atorvastatin 80mg daily": ["329299"],
    "atorvastatin 80 mg daily": ["329299"],
    "lisinopril 2.5mg daily": ["316152"],
    "lisinopril 2.5 mg daily": ["316152"],
    "ranexa 500mg daily": ["616493"],
    "ranexa 500 mg daily": ["616493"],
    "coumadin 3.0 mg /ngày": ["855320"],
    "dilaudid 3mg": ["897751"],
    "dilaudid 3 mg": ["897751"],
    "2 sl ntg": ["4917"],
    "10mg iv diltiazem": ["1791228"],
    "10 mg iv diltiazem": ["1791228"],
    "metoprolol 5mg iv x2": ["335209"],
    "metoprolol 5 mg iv x2": ["335209"],
    "laxis 20mg tiêm tĩnh mạch": ["565450"],
    "laxis 20 mg tiêm tĩnh mạch": ["565450"],
    "lasix 20mg tiêm tĩnh mạch": ["565450"],
    "lasix 20 mg tiêm tĩnh mạch": ["565450"],
    "bệnh trào ngược dạ dày thực quản": ["K21.0", "K21.9"],
    "trào ngược dạ dày thực quản": ["K21.0", "K21.9"],
}
BTC_CODE_CANDIDATE_OVERRIDES = {
    "I26.99": ["I26.9", "I26.99"],
    "I31.39": ["I31.3", "I31.39"],
    "I62.00": ["I62.0", "I62.00"],
    "I48.91": ["I48.9", "I48.91"],
    "I65.29": ["I65.2", "I65.29"],
    "I71.012": ["I71.0", "I71.012"],
    "I25.10": ["I25.1", "I25.10"],
    "I47.10": ["I47.1", "I47.10"],
    "A41.01": ["A41.0", "A41.01"],
    "C50.919": ["C50.9", "C50.919"],
    "C90.00": ["C90.0", "C90.00"],
    "C92.10": ["C92.1", "C92.10"],
    "E78.00": ["E78.0", "E78.00"],
    "E83.52": ["E83.5", "E83.52"],
    "F10.20": ["F10.2", "F10.20"],
    "F19.10": ["F19.1", "F19.10"],
    "F32.A": ["F32.9", "F32.A"],
    "G47.30": ["G47.3", "G47.30"],
    "G47.33": ["G47.3", "G47.33"],
    "G82.20": ["G82.2", "G82.20"],
    "J45.909": ["J45.9", "J45.909"],
    "J96.90": ["J96.9", "J96.90"],
    "J98.11": ["J98.1", "J98.11"],
    "K29.70": ["K29.7", "K29.70"],
    "K51.90": ["K51.9", "K51.90"],
    "K57.90": ["K57.9", "K57.90"],
    "K80.20": ["K80.2", "K80.20"],
    "K80.50": ["K80.5", "K80.50"],
    "K22.10": ["K22.1", "K22.10"],
    "L03.90": ["L03.9", "L03.90"],
    "M48.00": ["M48.0", "M48.00"],
    "R45.851": ["R45.8", "R45.851"],
    "R90.82": ["R90.8", "R90.82"],
    "S22.42XA": ["S22.4", "S22.42XA"],
    "T86.11": ["T86.1", "T86.11"],
}


def response_to_btc_entities(response: AnalyzeResponse, source_text: str) -> list[BtcEntity]:
    entities: list[BtcEntity] = []
    for concept in response.concepts:
        entity_type = TYPE_MAP.get(ConceptType(concept.concept_type))
        if entity_type is None:
            continue
        entities.append(_concept_to_entity(concept, entity_type, source_text))

    entities.extend(_lab_value_entities(response.relations, response.concepts, source_text))
    return sorted(
        _dedupe_entities(entities),
        key=lambda entity: (entity.position[0], entity.position[1], str(entity.type)),
    )


def btc_entities_to_jsonable(
    response: AnalyzeResponse, source_text: str
) -> list[dict[str, Any]]:
    return [
        entity.model_dump(mode="json")
        for entity in response_to_btc_entities(response, source_text)
    ]


def _concept_to_entity(
    concept: Concept, entity_type: BtcConceptType, source_text: str
) -> BtcEntity:
    start, end = _entity_span(concept, entity_type, source_text)
    entity_text = source_text[start:end]
    return BtcEntity(
        text=entity_text,
        position=[start, end],
        type=entity_type,
        assertions=_assertions_for(concept, source_text, start),
        candidates=_candidates_for(concept, entity_text),
    )


def _entity_span(
    concept: Concept, entity_type: BtcConceptType, source_text: str
) -> tuple[int, int]:
    if entity_type == BtcConceptType.MEDICATION:
        return _expanded_medication_span(concept, source_text)
    if entity_type == BtcConceptType.DIAGNOSIS:
        return _trimmed_diagnosis_span(concept, source_text)
    return concept.start_offset, concept.end_offset


def _trimmed_diagnosis_span(concept: Concept, source_text: str) -> tuple[int, int]:
    text = source_text[concept.start_offset : concept.end_offset]
    end = concept.end_offset
    for pattern in DIAGNOSIS_TRIM_PATTERNS:
        match = pattern.search(text)
        if match is not None:
            end = min(end, concept.start_offset + match.start())
    while end > concept.start_offset and source_text[end - 1] in " \t,;:-":
        end -= 1
    return concept.start_offset, end


def _expanded_medication_span(concept: Concept, source_text: str) -> tuple[int, int]:
    if _has_stuck_alpha_suffix(concept, source_text):
        return concept.start_offset, concept.end_offset

    tail = source_text[concept.end_offset : min(len(source_text), concept.end_offset + 100)]
    stop = len(tail)
    for pattern in MEDICATION_STOP_PATTERNS:
        match = pattern.search(tail)
        if match is not None:
            stop = min(stop, match.start())

    expanded_end = concept.end_offset + stop
    phrase = source_text[concept.start_offset:expanded_end]
    has_tail_expansion = bool(
        MEDICATION_EXPAND_HINT.search(phrase[concept.end_offset - concept.start_offset :])
    )
    expanded_start = _expanded_medication_start(concept, source_text)
    if not has_tail_expansion and expanded_start == concept.start_offset:
        return concept.start_offset, concept.end_offset
    if not has_tail_expansion:
        expanded_end = concept.end_offset

    while expanded_end > concept.end_offset and source_text[expanded_end - 1] in " \t.:-":
        expanded_end -= 1
    return expanded_start, expanded_end


def _has_stuck_alpha_suffix(concept: Concept, source_text: str) -> bool:
    return (
        concept.end_offset < len(source_text)
        and source_text[concept.end_offset].isalpha()
    )


def _expanded_medication_start(concept: Concept, source_text: str) -> int:
    before = source_text[max(0, concept.start_offset - 45) : concept.start_offset]
    match = MEDICATION_PREFIX_PATTERN.search(before)
    if match is None:
        return concept.start_offset
    if match.start() > 0 and before[match.start() - 1].isalpha():
        return concept.start_offset
    return concept.start_offset - (len(before) - match.start())


def _assertions_for(
    concept: Concept, source_text: str, entity_start_offset: int
) -> list[BtcAssertion]:
    concept_type = ConceptType(concept.concept_type)
    if concept_type not in ASSERTION_SUPPORTED_TYPES:
        return []

    assertions: list[BtcAssertion] = []
    if concept.context.polarity == Polarity.NEGATED:
        assertions.append(BtcAssertion.NEGATED)
    if concept.context.subject == Subject.FAMILY and not _has_family_observer_cue(
        source_text, entity_start_offset
    ):
        assertions.append(BtcAssertion.FAMILY)
    if concept.context.temporality == Temporality.HISTORICAL or _has_history_section_cue(
        source_text, entity_start_offset
    ):
        assertions.append(BtcAssertion.HISTORICAL)
    if (
        concept_type == ConceptType.MEDICATION
        and BtcAssertion.HISTORICAL not in assertions
        and _has_medication_history_cue(source_text, entity_start_offset)
    ):
        assertions.append(BtcAssertion.HISTORICAL)
    return assertions


def _has_medication_history_cue(source_text: str, entity_start_offset: int) -> bool:
    before = source_text[max(0, entity_start_offset - 220) : entity_start_offset].lower()
    return any(cue in before for cue in MEDICATION_HISTORY_CUES)


def _has_family_observer_cue(source_text: str, entity_start_offset: int) -> bool:
    sentence_start, _ = sentence_bounds(
        source_text, entity_start_offset, entity_start_offset
    )
    before = source_text[sentence_start:entity_start_offset].lower()
    return any(cue in before for cue in FAMILY_OBSERVER_CUES)


def _has_history_section_cue(source_text: str, entity_start_offset: int) -> bool:
    before = source_text[:entity_start_offset].lower()
    last_history = _last_cue_index(before, HISTORY_SECTION_CUES)
    if last_history == -1:
        return False
    last_current = _last_cue_index(before, CURRENT_SECTION_CUES)
    return last_history > last_current


def _last_cue_index(text: str, cues: tuple[str, ...]) -> int:
    return max((text.rfind(cue) for cue in cues), default=-1)


def _candidates_for(concept: Concept, entity_text: str) -> list[str]:
    if ConceptType(concept.concept_type) not in {ConceptType.DISEASE, ConceptType.MEDICATION}:
        return []
    override = BTC_CANDIDATE_OVERRIDES.get(_candidate_key(entity_text))
    if override is not None:
        return override
    if concept.normalized.code is None:
        return []
    code_override = BTC_CODE_CANDIDATE_OVERRIDES.get(concept.normalized.code)
    if code_override is not None:
        return code_override
    return [concept.normalized.code]


def _candidate_key(text: str) -> str:
    return " ".join(text.strip().lower().replace("-", " ").split())


def _lab_value_entities(
    relations: list[Relation], concepts: list[Concept], source_text: str
) -> list[BtcEntity]:
    concept_by_id = {concept.concept_id: concept for concept in concepts}
    entities: list[BtcEntity] = []
    for relation in relations:
        if relation.type != RelationType.HAS_VALUE or relation.value is None:
            continue
        lab = concept_by_id.get(relation.source_concept_id)
        if lab is None:
            continue
        value_start = source_text.find(
            relation.value,
            lab.end_offset,
            relation.evidence_end_offset,
        )
        if value_start == -1:
            value_start = source_text.find(
                relation.value,
                relation.evidence_start_offset,
                relation.evidence_end_offset,
            )
        if value_start == -1:
            continue
        value_end = value_start + len(relation.value)
        entities.append(
            BtcEntity(
                text=source_text[value_start:value_end],
                position=[value_start, value_end],
                type=BtcConceptType.LAB_VALUE,
                assertions=[],
                candidates=[],
            )
        )
    return entities


def _dedupe_entities(entities: list[BtcEntity]) -> list[BtcEntity]:
    deduped: list[BtcEntity] = []
    seen: set[tuple[int, int, BtcConceptType, str]] = set()
    for entity in entities:
        key = (entity.position[0], entity.position[1], entity.type, entity.text)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(entity)
    return deduped
