from __future__ import annotations

import re

from clinical_nlp.schemas import ContextAttributes, Polarity, Subject, Temporality

NEGATION_CUES = (
    "no",
    "denies",
    "denied",
    "without",
    "negative for",
    "không",
    "không có",
    "không ghi nhận",
    "chưa",
    "chưa ghi nhận",
    "chưa phát hiện",
    "phủ nhận bất kỳ",
    "phủ nhận",
)
IGNORED_VI_NEGATION_PREFIXES = (
    "khỏe",
    "rõ",
    "thuốc cản quang",
    "tự chủ",
    "đáp ứng",
    "dung nạp",
    "liên quan",
    "cải thiện",
    "ngon miệng",
)
NEGATION_SCOPE_BOUNDARY_PATTERN = re.compile(
    r"[\n.;]|(?:\bkết\s+quả\b|\bcận\s+lâm\s+sàng\b|\bhình\s+ảnh\b|"
    r"\bchẩn\s+đoán\s+hình\s+ảnh\b)",
    re.I,
)
POSSIBLE_CUES = (
    "possible",
    "probable",
    "suspected",
    "rule out",
    "r/o",
    "nghi ngờ",
    "có thể",
)
FAMILY_CUES = (
    "mother",
    "father",
    "sister",
    "brother",
    "family history",
    "parent",
    "mẹ",
    "cha",
    "bố",
    "chị gái",
    "anh trai",
    "em gái",
    "em trai",
    "gia đình",
)
HISTORY_CUES = (
    "history of",
    "past medical history",
    "prior",
    "previous",
    "previously",
    "tiền sử",
    "trước đây",
    "mãn tính",
    "đã điều trị",
)
HYPOTHETICAL_CUES = (
    "if",
    "planned",
    "will",
    "rule out",
    "r/o",
    "dự kiến",
    "sẽ",
    "lên lịch",
)


def infer_context(text: str, start: int, end: int) -> ContextAttributes:
    sentence_start, sentence_end = sentence_bounds(text, start, end)
    sentence = text[sentence_start:sentence_end].lower()
    before = text[max(sentence_start, start - 80) : start].lower()

    polarity = Polarity.PRESENT
    if is_negated(before):
        polarity = Polarity.NEGATED
    elif has_any_cue(sentence, POSSIBLE_CUES):
        polarity = Polarity.POSSIBLE

    subject = Subject.FAMILY if has_any_cue(sentence, FAMILY_CUES) else Subject.PATIENT

    temporality = Temporality.CURRENT
    if has_any_cue(sentence, HISTORY_CUES):
        temporality = Temporality.HISTORICAL
    elif has_any_cue(sentence, HYPOTHETICAL_CUES):
        temporality = Temporality.HYPOTHETICAL

    return ContextAttributes(polarity=polarity, subject=subject, temporality=temporality)


def has_any_cue(text: str, cues: tuple[str, ...]) -> bool:
    return any(has_cue(text, cue) for cue in cues)


def is_negated(before: str) -> bool:
    for cue in sorted(NEGATION_CUES, key=len, reverse=True):
        for match in reversed(list(re.finditer(rf"(?<!\w){re.escape(cue)}(?!\w)", before, re.I))):
            following = before[match.end() :].strip(" \t:-,;")
            if _negation_following_in_scope(cue, following):
                return True
            break
    return False


def has_actionable_vietnamese_negation(before: str) -> bool:
    for match in reversed(list(re.finditer(r"(?<!\w)không(?!\w)", before, re.I))):
        following = before[match.end() :].strip(" \t:-,;")
        if _negation_following_in_scope("không", following):
            return True
    return False


def _negation_following_in_scope(cue: str, following: str) -> bool:
    if not following:
        return True
    if any(
        following.startswith(prefix)
        for prefix in IGNORED_VI_NEGATION_PREFIXES
    ):
        return False
    if NEGATION_SCOPE_BOUNDARY_PATTERN.search(following):
        return False
    return not (_contains_vietnamese_character(cue) and len(following) > 55)


def _contains_vietnamese_character(text: str) -> bool:
    return any(ord(character) > 127 for character in text)


def has_cue(text: str, cue: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(cue)}(?!\w)", text, re.I) is not None


def sentence_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    left = max(_rfind_sentence_dot(text, start), text.rfind("\n", 0, start))
    sentence_start = 0 if left == -1 else left + 1
    right_dot = _find_sentence_dot(text, end)
    right_newline = text.find("\n", end)
    right_candidates = [index for index in (right_dot, right_newline) if index != -1]
    sentence_end = min(right_candidates) if right_candidates else len(text)
    return sentence_start, sentence_end


def _find_sentence_dot(text: str, start: int) -> int:
    index = text.find(".", start)
    while index != -1:
        if not _is_decimal_dot(text, index):
            return index
        index = text.find(".", index + 1)
    return -1


def _rfind_sentence_dot(text: str, end: int) -> int:
    index = text.rfind(".", 0, end)
    while index != -1:
        if not _is_decimal_dot(text, index):
            return index
        index = text.rfind(".", 0, index)
    return -1


def _is_decimal_dot(text: str, index: int) -> bool:
    return (
        index > 0
        and index + 1 < len(text)
        and text[index - 1].isdigit()
        and text[index + 1].isdigit()
    )
