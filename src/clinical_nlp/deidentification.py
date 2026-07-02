from __future__ import annotations

import re
from dataclasses import dataclass
from re import Pattern


@dataclass(frozen=True)
class SpanMapping:
    original_start: int
    original_end: int
    processed_start: int
    processed_end: int
    label: str


@dataclass(frozen=True)
class DeidentifiedText:
    original_text: str
    processed_text: str
    mappings: tuple[SpanMapping, ...]

    @property
    def changed(self) -> bool:
        return self.original_text != self.processed_text


@dataclass(frozen=True)
class PIIPattern:
    label: str
    pattern: Pattern[str]


class Deidentifier:
    def __init__(self) -> None:
        self._patterns = (
            PIIPattern("EMAIL", re.compile(r"\b[\w.\-+]+@[\w.\-]+\.\w+\b")),
            PIIPattern(
                "PHONE",
                re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b"),
            ),
            PIIPattern("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
            PIIPattern(
                "MRN",
                re.compile(r"\b(?:MRN|medical record)\s*[:#]?\s*[A-Za-z0-9-]+\b", re.I),
            ),
            PIIPattern("NAME", re.compile(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b(?=,)")),
        )

    def deidentify(self, text: str) -> DeidentifiedText:
        chars = list(text)
        mappings: list[SpanMapping] = []
        occupied: list[range] = []

        for pii_pattern in self._patterns:
            for match in pii_pattern.pattern.finditer(text):
                span = range(match.start(), match.end())
                if any(overlaps(span, existing) for existing in occupied):
                    continue
                occupied.append(span)
                for index in span:
                    if chars[index] != " ":
                        chars[index] = "X"
                mappings.append(
                    SpanMapping(
                        original_start=match.start(),
                        original_end=match.end(),
                        processed_start=match.start(),
                        processed_end=match.end(),
                        label=pii_pattern.label,
                    )
                )

        return DeidentifiedText(
            original_text=text,
            processed_text="".join(chars),
            mappings=tuple(sorted(mappings, key=lambda mapping: mapping.original_start)),
        )


def overlaps(left: range, right: range) -> bool:
    return left.start < right.stop and right.start < left.stop
