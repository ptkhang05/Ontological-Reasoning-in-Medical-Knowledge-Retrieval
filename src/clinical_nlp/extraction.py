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
    "khó thở khi gắng sức",
    "đánh trống ngực",
    "khó thở",
    "thắt chặt ngực",
    "khó chịu vùng ngực",
    "đau ngực",
    "buồn nôn",
    "nôn ra máu",
    "nôn",
    "đổ mồ hôi qua đêm",
    "đổ mồ hôi",
    "mệt mỏi",
    "mệt mỏi nhiều",
    "giảm dung nạp gắng sức",
    "đau vùng hạ sườn phải",
    "đau hạ sườn phải",
    "đau bụng vùng thượng vị",
    "khó chịu vùng bụng",
    "đau quanh vết mổ",
    "đau ấn vùng mổ",
    "đau khi sờ nắn",
    "đau hố chậu",
    "đau bụng",
    "tiêu chảy",
    "đi ngoài ra máu",
    "phân có máu",
    "phân đen",
    "đờm có màu hồng",
    "đờm trắng",
    "đờm vàng",
    "đờm",
    "ho có đờm",
    "ho khan",
    "ho nhiều",
    "ho",
    "chóng mặt",
    "ngất xỉu",
    "phù ngoại vi",
    "phù chi dưới",
    "phù",
    "sốt cao",
    "sốt",
    "đau đầu",
    "ớn lạnh",
    "khò khè nhiều",
    "khò khè",
    "nghẹt ngực",
    "đau lưng",
    "đau cổ",
    "đau cổ và vai phải",
    "đau vai",
    "đau hai bàn chân",
    "đau chân",
    "đau cơ",
    "đau họng",
    "ho ra máu",
    "khó nuốt",
    "khàn tiếng",
    "chướng bụng",
    "chán ăn",
    "ăn uống kém",
    "uống kém",
    "mất ăn ngon",
    "toàn trạng suy kiệt",
    "suy kiệt",
    "khó chịu",
    "yếu cơ",
    "sưng quanh khớp",
    "sưng",
    "ban đỏ",
    "ngứa da",
    "ngứa",
    "mất trí nhớ",
    "ảo giác",
    "tiểu khó",
    "tiểu đêm",
    "táo bón",
    "choáng váng",
    "hoa mắt",
    "mờ mắt",
    "mất ngủ",
)

LAB_NAMES = (
    "hemoglobin a1c",
    "hba1c",
    "troponin",
    "inr",
    "bạch cầu",
    "wbc",
    "alt",
    "ast",
    "kali",
    "hemoglobin",
    "bnp",
    "bun",
    "ure",
    "tiểu cầu",
    "phosphate",
    "anion gap",
    "pcr",
    "natri",
    "lactate",
    "bilirubin",
    "glucose",
    "sodium",
    "potassium",
    "creatinine",
    "cea",
    "kháng nguyên ung thư phôi",
    "canxi toàn phần",
    "canxi ion hóa",
    "canxi",
    "creatinin",
    "cr",
    "phân tích nước tiểu",
)

VI_DISEASE_TERMS = (
    "xơ gan",
    "hội chứng não gan",
    "tăng calci máu",
    "tăng canxi máu",
    "cường cận giáp nguyên phát",
    "xơ vữa động mạch",
    "cơn đau thắt ngực ổn định",
    "u ác của đại tràng",
    "u ác đại tràng",
    "khối u trực tràng",
    "u ác trực tràng",
    "u tuyến",
    "viêm tuyến mồ hôi",
    "ngoại tâm thu nhĩ",
    "ngoại tâm thu thất",
    "nhồi máu cơ tim vùng vách liên thất, mạn tính và đỉnh",
    "nhồi máu cơ tim vùng dưới cũ",
    "nhồi máu cũ nhỏ ở vỏ não đỉnh trái",
    "nhồi máu cũ nhỏ ở vỏ não",
    "nhồi máu cơ tim cũ",
    "nhồi máu cơ tim",
    "nhồi máu",
)

VI_MEDICATION_TERMS = (
    "metoprolol",
    "doxycycline",
    "atenolol",
    "aspirin",
    "nitroglycerin",
    "laxis",
    "natriclori",
    "natri clorid",
    "ceftazidime",
    "zosyn",
    "nac",
    "dilaudid",
    "plavix",
    "klonopin",
    "clonidine",
    "advil",
    "bumetanide",
    "albuterol",
    "ipratropium",
    "desmopressin",
    "allopurinol",
    "prograf",
    "flagyl",
)

UNKNOWN_MEDICATION_STOPWORDS = {
    "bằng",
    "thuốc",
    "trước",
    "kháng",
    "nhiễm",
    "viêm",
    "chứng",
    "tiếp",
    "ống",
    "tại",
    "một",
    "đều",
    "theo",
    "được",
    "rượu",
    "bán",
    "thêm",
    "lợi",
    "intravenous",
    "bảo",
    "nội",
    "ngoại",
    "khám",
    "chẩn",
    "chống",
    "biến",
    "phình",
    "máy",
    "bất",
    "tình",
    "các",
    "băng",
    "quả",
    "ngay",
    "đặt",
    "bắt",
    "bệnh",
    "gậy",
    "thì",
    "bipap",
    "nitrateskhi",
}
UNKNOWN_DISEASE_STOPWORDS = {
    "thuyên",
    "nhiễm",
    "trước",
    "nghẽn",
}


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
                VI_DISEASE_TERMS,
                ConceptType.DISEASE,
                0.78,
                "vietnamese_medical_dictionary",
            )
        )
        candidates.extend(
            self._extract_terms(
                text,
                VI_MEDICATION_TERMS,
                ConceptType.MEDICATION,
                0.80,
                "vietnamese_medication_dictionary",
            )
        )
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
        candidates.extend(self._extract_compacted_medications(text, candidates))
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
            re.compile(r"\b\d{1,3}\s*tuổi\b", re.I),
            re.compile(r"\b(?:nam giới|nữ giới|nữ)\b", re.I),
            re.compile(r"\bbệnh nhân\s+(?:nam|nữ)\b", re.I),
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

    def _extract_compacted_medications(
        self, text: str, existing: list[CandidateConcept]
    ) -> list[CandidateConcept]:
        terms = {
            term.lower()
            for term in (
                *VI_MEDICATION_TERMS,
                *self._terminology.search_terms_for(ConceptType.MEDICATION),
            )
            if len(term) >= 5 and " " not in term
        }
        token_pattern = re.compile(r"[A-Za-zÀ-ỹ][A-Za-zÀ-ỹ-]{7,}", re.I)
        candidates: list[CandidateConcept] = []
        seen: set[tuple[int, int]] = set()
        for token_match in token_pattern.finditer(text):
            token = token_match.group(0)
            token_lower = token.lower()
            if token_lower in terms:
                continue
            for term in sorted(terms, key=lambda value: (-len(value), value)):
                search_start = 0
                while True:
                    index = token_lower.find(term, search_start)
                    if index == -1:
                        break
                    start = token_match.start() + index
                    end = start + len(term)
                    search_start = index + 1
                    if (start, end) in seen or self._span_has_existing(start, end, existing):
                        continue
                    seen.add((start, end))
                    candidates.append(
                        CandidateConcept(
                            text=text[start:end],
                            start_offset=start,
                            end_offset=end,
                            concept_type=ConceptType.MEDICATION,
                            confidence=0.82,
                            source="compacted_medication",
                        )
                    )
        return candidates

    def _extract_unknown_medications(
        self, text: str, existing: list[CandidateConcept]
    ) -> list[CandidateConcept]:
        candidates: list[CandidateConcept] = []
        pattern = re.compile(
            r"\b(?:takes|taking|started|on|dùng|sử dụng|điều trị)\s+"
            r"([A-Za-zÀ-ỹ][A-Za-zÀ-ỹ-]{2,})\b",
            re.I,
        )
        for match in pattern.finditer(text):
            start, end = match.span(1)
            if self._span_has_existing(start, end, existing):
                continue
            token = text[start:end]
            token_key = token.lower()
            if (
                token_key in UNKNOWN_MEDICATION_STOPWORDS
                or token_key in {term.lower() for term in SYMPTOM_TERMS}
            ):
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
            r"\b(?:diagnosed with|history of|has|tiền sử|nghi ngờ|chẩn đoán)\s+"
            r"([A-Za-zÀ-ỹ][A-Za-zÀ-ỹ-]{4,})\b",
            re.I,
        )
        symptom_terms = {term.lower() for term in SYMPTOM_TERMS}
        for match in pattern.finditer(text):
            start, end = match.span(1)
            token = text[start:end]
            token_key = token.lower()
            if (
                self._span_has_existing(start, end, existing)
                or token_key in symptom_terms
                or token_key in UNKNOWN_DISEASE_STOPWORDS
            ):
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
