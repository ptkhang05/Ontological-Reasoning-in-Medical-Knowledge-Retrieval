from __future__ import annotations

from clinical_nlp.schemas import ContextAttributes, Polarity, Subject, Temporality

NEGATION_CUES = ("no", "denies", "denied", "without", "negative for")
POSSIBLE_CUES = ("possible", "probable", "suspected", "rule out", "r/o")
FAMILY_CUES = ("mother", "father", "sister", "brother", "family history", "parent")
HISTORY_CUES = ("history of", "past medical history", "prior", "previous", "previously")
HYPOTHETICAL_CUES = ("if", "planned", "will", "rule out", "r/o")


def infer_context(text: str, start: int, end: int) -> ContextAttributes:
    sentence_start, sentence_end = sentence_bounds(text, start, end)
    sentence = text[sentence_start:sentence_end].lower()
    before = text[max(sentence_start, start - 80) : start].lower()

    polarity = Polarity.PRESENT
    if any(cue in before for cue in NEGATION_CUES):
        polarity = Polarity.NEGATED
    elif any(cue in sentence for cue in POSSIBLE_CUES):
        polarity = Polarity.POSSIBLE

    subject = Subject.FAMILY if any(cue in sentence for cue in FAMILY_CUES) else Subject.PATIENT

    temporality = Temporality.CURRENT
    if any(cue in sentence for cue in HISTORY_CUES):
        temporality = Temporality.HISTORICAL
    elif any(cue in sentence for cue in HYPOTHETICAL_CUES):
        temporality = Temporality.HYPOTHETICAL

    return ContextAttributes(polarity=polarity, subject=subject, temporality=temporality)


def sentence_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    del end
    left = max(text.rfind(".", 0, start), text.rfind("\n", 0, start))
    sentence_start = 0 if left == -1 else left + 1
    right_dot = text.find(".", start)
    right_newline = text.find("\n", start)
    right_candidates = [index for index in (right_dot, right_newline) if index != -1]
    sentence_end = min(right_candidates) if right_candidates else len(text)
    return sentence_start, sentence_end
