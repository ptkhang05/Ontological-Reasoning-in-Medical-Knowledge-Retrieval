from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

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


def response_to_btc_entities(response: AnalyzeResponse, source_text: str) -> list[BtcEntity]:
    entities: list[BtcEntity] = []
    for concept in response.concepts:
        entity_type = TYPE_MAP.get(ConceptType(concept.concept_type))
        if entity_type is None:
            continue
        entities.append(_concept_to_entity(concept, entity_type))

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


def _concept_to_entity(concept: Concept, entity_type: BtcConceptType) -> BtcEntity:
    return BtcEntity(
        text=concept.text,
        position=[concept.start_offset, concept.end_offset],
        type=entity_type,
        assertions=_assertions_for(concept),
        candidates=_candidates_for(concept),
    )


def _assertions_for(concept: Concept) -> list[BtcAssertion]:
    if ConceptType(concept.concept_type) not in ASSERTION_SUPPORTED_TYPES:
        return []

    assertions: list[BtcAssertion] = []
    if concept.context.polarity == Polarity.NEGATED:
        assertions.append(BtcAssertion.NEGATED)
    if concept.context.subject == Subject.FAMILY:
        assertions.append(BtcAssertion.FAMILY)
    if concept.context.temporality == Temporality.HISTORICAL:
        assertions.append(BtcAssertion.HISTORICAL)
    return assertions


def _candidates_for(concept: Concept) -> list[str]:
    if ConceptType(concept.concept_type) not in {ConceptType.DISEASE, ConceptType.MEDICATION}:
        return []
    if concept.normalized.code is None:
        return []
    return [concept.normalized.code]


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
