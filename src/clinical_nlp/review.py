from __future__ import annotations

from clinical_nlp.context import sentence_bounds
from clinical_nlp.schemas import Concept, ConceptType, ReviewFlag

SAFETY_CUES = ("allergy", "allergic", "contraindication", "contraindicated")


def build_review_flags(
    text: str,
    concepts: list[Concept],
    confidence_threshold: float,
    external_inference_used: bool,
) -> list[ReviewFlag]:
    flags: list[ReviewFlag] = []
    seen: set[tuple[str | None, str]] = set()

    for concept in concepts:
        if concept.confidence < confidence_threshold:
            _append_once(
                flags,
                seen,
                ReviewFlag(
                    concept_id=concept.concept_id,
                    reason="LOW_CONFIDENCE",
                    message="Concept confidence is below the configured review threshold.",
                ),
            )

        if (
            concept.concept_type in {ConceptType.DISEASE, ConceptType.MEDICATION}
            and (concept.normalized.code_system is None or concept.normalized.code is None)
        ):
            _append_once(
                flags,
                seen,
                ReviewFlag(
                    concept_id=concept.concept_id,
                    reason="UNMAPPED_CODE",
                    message="Disease or medication concept could not be normalized.",
                ),
            )

        sentence_start, sentence_end = sentence_bounds(
            text, concept.start_offset, concept.end_offset
        )
        sentence = text[sentence_start:sentence_end].lower()
        if concept.concept_type == ConceptType.MEDICATION and any(
            cue in sentence for cue in SAFETY_CUES
        ):
            _append_once(
                flags,
                seen,
                ReviewFlag(
                    concept_id=concept.concept_id,
                    reason="SAFETY_SENSITIVE",
                    message="Medication appears in an allergy or contraindication context.",
                ),
            )

    if external_inference_used:
        _append_once(
            flags,
            seen,
            ReviewFlag(
                concept_id=None,
                reason="EXTERNAL_INFERENCE_USED",
                message="External inference was used after de-identification; review is required.",
            ),
        )

    return flags


def _append_once(
    flags: list[ReviewFlag], seen: set[tuple[str | None, str]], flag: ReviewFlag
) -> None:
    key = (flag.concept_id, flag.reason)
    if key not in seen:
        flags.append(flag)
        seen.add(key)
