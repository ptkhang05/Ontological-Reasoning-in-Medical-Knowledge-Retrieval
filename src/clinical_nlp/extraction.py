from __future__ import annotations

import re
from dataclasses import dataclass

from rapidfuzz import fuzz

from clinical_nlp.schemas import ConceptType
from clinical_nlp.terminology import TerminologyStore, fold_term


@dataclass(frozen=True)
class CandidateConcept:
    text: str
    start_offset: int
    end_offset: int
    concept_type: ConceptType
    confidence: float
    source: str


@dataclass(frozen=True)
class FuzzySearchTerm:
    text: str
    folded: str


FUZZY_TERMINOLOGY_TERM_LIMIT = 2_000
TOKEN_PATTERN = re.compile(r"[^\W_]+(?:[-/][^\W_]+)?", re.UNICODE)

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
    "khó thở tăng dần",
    "nhịp thở nhanh",
    "thiếu oxy",
    "đánh trống ngực",
    "khó thở",
    "tiểu tiện không tự chủ",
    "sa âm đạo",
    "bàng quang căng",
    "cảm giác bí tiểu",
    "cảm giác bí tiếu",
    "bí tiểu",
    "bí tiếu",
    "cảm giác có khối lồi trong âm đạo",
    "cảm giác có khối sa lồi ra âm đạo",
    "cảm giác căng tức vùng âm đạo",
    "tiểu không tự chủ",
    "tiểu dắt",
    "tiểu buốt",
    "khô âm đạo",
    "đau khi giao hợp",
    "cơn co tử cung",
    "ra huyết âm đạo",
    "ra huyết âm đạo lượng ít",
    "vỡ ối",
    "rỉ ối",
    "thắt chặt ngực",
    "khó chịu vùng ngực",
    "đau ngực",
    "buồn nôn",
    "buồn nôn hoặc nôn",
    "nôn ra máu",
    "nôn",
    "đổ mồ hôi qua đêm",
    "đổ mồ hôi",
    "mệt mỏi",
    "mệt mỏi nhiều",
    "giảm dung nạp gắng sức",
    "sốt và run rẩy",
    "chủ quan sốt",
    "đau âm ỉ vùng quanh rốn",
    "đau âm ỉ",
    "đau tăng dần",
    "đau dữ dội",
    "đau ngực trái cấp tính",
    "đau sau xương ức lan ra sau lưng",
    "đau ngực lan ra sau lưng",
    "đau ngực trái",
    "đau sau xương ức",
    "đau lan ra sau lưng",
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
    "thay đổi thói quen đại tiện",
    "đại tiện ra máu đỏ tươi",
    "đi ngoài ra máu",
    "phân có máu",
    "phân đen",
    "phân lỏng",
    "đờm có màu hồng",
    "đờm trắng",
    "đờm vàng",
    "đờm",
    "cảm giác có đờm ở cổ họng",
    "ho có đờm",
    "ho khan",
    "ho nhiều",
    "ho tăng lên",
    "ho",
    "chảy nước mũi",
    "chóng mặt",
    "ngất xỉu",
    "phù ngoại vi",
    "phù ngoại vi tăng dần",
    "phù chi dưới",
    "sưng phù hai mắt cá chân",
    "phù mắt cá chân",
    "phù hai bên",
    "phù chân trái",
    "phù 2 chân",
    "phù chân",
    "phù phù",
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
    "đau đầu gối phải",
    "đau vùng xương bánh chè – đùi phải dữ dội",
    "đau vùng xương bánh chè - đùi phải",
    "đau bánh chè đùi phải dữ dội",
    "đau hai bàn chân",
    "đau bẹn trái",
    "đau chân",
    "đau cơ",
    "đau họng",
    "ho ra máu",
    "khó nuốt",
    "khàn tiếng",
    "giọng khàn",
    "tổn thương dây thanh quản",
    "triệu chứng trào ngược",
    "chướng bụng",
    "vết thương thấu bụng giữa bên phải",
    "vết thương thấu bụng",
    "bầm máu vùng bẹn trái",
    "chán ăn",
    "ăn uống kém",
    "uống kém",
    "mất ăn ngon",
    "mất cảm giác ngon miệng",
    "ăn không ngon miệng",
    "tăng cân trở lại",
    "tăng cân",
    "giảm cân",
    "toàn trạng suy kiệt",
    "suy kiệt",
    "suy nhược toàn thân",
    "suy nhược",
    "khó thở liên tục",
    "hồi hộp",
    "run rẩy",
    "khó chịu",
    "yếu cơ",
    "loét mới ở ngón chân út bên phải",
    "loét đau",
    "loét đỏ",
    "loét sưng",
    "sưng quanh khớp",
    "sưng nề",
    "sưng",
    "đỏ",
    "chảy mủ",
    "chảy dịch liên tục từ mặt trong của bàn chân trái",
    "chảy dịch",
    "dịch rò rỉ",
    "dịch rỉ",
    "ran nổ",
    "ban đỏ",
    "ngứa da",
    "ngứa da toàn thân nhiều",
    "ngứa",
    "mất trí nhớ",
    "mất trí nhớ chi tiết",
    "tình trạng tri giác giảm sút",
    "yếu nửa người trái",
    "yếu sức",
    "yếu sức nửa người bên phải",
    "không thể tự đứng dậy",
    "không thể chịu lực",
    "khuỵu chân",
    "ảo giác",
    "rối loạn ý thức",
    "mất định hướng",
    "lú lẫn",
    "ảo thanh",
    "ảo giác thị giác",
    "cảm giác nặng ở chân trái",
    "cảm giác châm chích",
    "cảm giác tê",
    "tiểu khó",
    "khó chịu khi đi tiểu",
    "khó chịu khi đại tiện",
    "tiểu đêm",
    "thiểu niệu",
    "táo bón",
    "choáng váng",
    "hoa mắt",
    "phù gai thị",
    "mờ mắt",
    "nhìn mờ",
    "nhìn song thị",
    "khó nhìn gần",
    "vụng về",
    "kéo lê chân",
    "khó khăn khi ra khỏi ghế tựa",
    "khó khăn khi ăn",
    "khó khăn khi cài cúc áo",
    "khó khăn khi ước lượng vị trí ngồi xuống ghế ăn trưa",
    "cánh tay trái được cho là lơ lửng",
    "cảm giác bất thường ở bên phải đầu",
    "khó nằm",
    "nhịp tim nhanh",
    "mạch nhanh",
    "mất ngủ",
    "gãy xương sườn trái",
    "tổn thương rõ ràng",
)

LAB_NAMES = (
    "hemoglobin a1c",
    "hba1c",
    "công thức máu",
    "cbc",
    "hct",
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
    "tổng phân tích nước tiểu",
    "ua (urinalysis - tổng phân tích nước tiểu)",
    "ua",
    "urinalysis",
    "guaiac",
    "huyết thanh",
    "xét nghiệm tế bào học",
    "tế bào học",
    "monitor holter",
    "sinh thiết",
    "lấy mẫu bằng bàn chải",
    "chọc hút bằng kim nhỏ",
    "chọc hút",
    "chọc dò dịch não tủy",
    "dịch não tủy",
    "xét nghiệm chức năng gan",
    "chức năng gan",
    "men gan",
    "chụp x-quang ngực",
    "xquang ngực thẳng",
    "x-quang ngực",
    "chụp x-quang",
    "x-quang",
    "chụp ct sọ não không thuốc cản quang",
    "chụp ct sọ não",
    "ct sọ não",
    "chụp ct ngực không thuốc cản quang",
    "chụp ct bụng, chậu, không thuốc cản quang",
    "chụp ct bụng chậu",
    "chụp ct bụng",
    "ct bụng và chậu",
    "ct bụng",
    "chụp ct ngực",
    "ct ngực",
    "chụp ct",
    "chụp cắt lớp vi tính (ct) bụng và chậu",
    "chụp cắt lớp vi tính (ct) lồng ngực",
    "chụp cắt lớp vi tính (ct) cột sống cổ",
    "chụp cắt lớp vi tính (ct)",
    "chụp cắt lớp vi tính mạch máu",
    "chụp cắt lớp vi tính",
    "chụp cộng hưởng từ (mri)",
    "chụp cộng hưởng từ",
    "mri ngoại trú",
    "mri",
    "siêu âm tim qua thành ngực (siêu âm tim qua thành ngực)",
    "siêu âm tim qua thành ngực",
    "siêu âm tim gắng sức",
    "siêu âm vùng gan mật",
    "siêu âm vùng cổ phải",
    "siêu âm mạch máu chi trên",
    "siêu âm thận",
    "siêu âm bụng có doppler",
    "siêu âm doppler",
    "siêu âm tim",
    "siêu âm",
    "điện tâm đồ (ecg)",
    "điện tâm đồ (ekg)",
    "điện tâm đồ",
    "ekg",
    "ecg",
    "công thức máu (cbc)",
    "nội soi thực quản - dạ dày - tá tràng",
    "nội soi mật tụy ngược dòng (ercp)",
    "nội soi mật tuỵ ngược dòng (ercp)",
    "nội soi mật tụy ngược dòng",
    "nội soi mật tuỵ ngược dòng",
    "nội soi đại tràng",
    "nội soi",
    "ercp",
    "cholangiogram",
    "xạ hình thông khí - tưới máu phổi",
    "xạ hình tưới máu cơ tim",
    "nghiệm pháp gắng sức",
)

VI_DISEASE_TERMS = (
    "bệnh trào ngược dạ dày - thực quản",
    "trào ngược dạ dày thực quản",
    "trào ngược dạ dày - thực quản",
    "bệnh bạch cầu dòng tủy mãn tính",
    "bệnh bạch cầu dòng tủy mạn tính",
    "xuất huyết nội sọ không do chấn thương",
    "khối máu tụ dưới màng cứng",
    "não úng thủy khác",
    "não úng thuỷ khác",
    "não úng tuỷ khác",
    "não úng thủy",
    "não úng thuỷ",
    "não úng tuỷ",
    "ung thư biểu mô tế bào mật không thể cắt bỏ",
    "ung thư biểu mô tế bào mật",
    "ung thư đường mật không thể cắt bỏ",
    "ung thư đường mật",
    "tắc nghẽn kéo dài gần chỗ nối mật tụy",
    "tắc nghẽn kéo dài gần chỗ nối mật tuỵ",
    "tắc nghẽn đường mật",
    "nang tụy",
    "nang tuỵ",
    "gãy xương sườn trái",
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
TERMINOLOGY_DISEASE_BLOCK_TERMS = (
    "sử dụng rượu",
    "sử dụng thuốc lá",
    "tai nạn",
    "kháng vancomycin",
)
TERMINOLOGY_MEDICATION_BLOCK_TERMS = (
    "albumin",
    "alanine",
    "aspartate",
    "bilirubin",
    "caffeine",
    "calcium",
    "chloride",
    "cholesterol",
    "creatinine",
    "glucose",
    "guaiac",
    "hemoglobin",
    "lactate",
    "lipase",
    "phosphate",
    "phosphorus",
    "potassium",
    "sodium",
    "talc",
    "triglycerides",
    "troponin",
    "urea",
)


class RuleBasedExtractor:
    def __init__(self, terminology: TerminologyStore) -> None:
        self._terminology = terminology
        self._disease_terminology_terms = _extractable_disease_terminology_terms(
            self._terminology.search_terms_for(ConceptType.DISEASE)
        )
        self._medication_terminology_terms = _extractable_medication_terminology_terms(
            self._terminology.search_terms_for(ConceptType.MEDICATION)
        )
        self._disease_terminology_pattern = _compile_term_pattern(
            self._disease_terminology_terms
        )
        self._medication_terminology_pattern = _compile_term_pattern(
            self._medication_terminology_terms
        )
        fuzzy_disease_terms = (
            self._disease_terminology_terms
            if len(self._disease_terminology_terms) <= FUZZY_TERMINOLOGY_TERM_LIMIT
            else []
        )
        self._fuzzy_disease_terms = _build_fuzzy_terms(
            fuzzy_disease_terms
        )

    def extract(self, text: str) -> list[CandidateConcept]:
        candidates: list[CandidateConcept] = []
        candidates.extend(
            self._extract_terms(
                text, SYMPTOM_TERMS, ConceptType.SYMPTOM, 0.86, "rule_symptom"
            )
        )
        candidates.extend(self._extract_contextual_symptoms(text))
        candidates.extend(self._extract_contextual_medications(text))
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
            self._extract_compiled_terms(
                text,
                self._disease_terminology_pattern,
                ConceptType.DISEASE,
                0.90,
                "terminology_match",
            )
        )
        candidates.extend(
            self._extract_compiled_terms(
                text,
                self._medication_terminology_pattern,
                ConceptType.MEDICATION,
                0.90,
                "terminology_match",
            )
        )
        candidates.extend(
            self._extract_fuzzy_terms(
                text,
                self._fuzzy_disease_terms,
                ConceptType.DISEASE,
                0.82,
                "fuzzy_terminology_match",
                candidates,
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
                if concept_type == ConceptType.MEDICATION and _is_blocked_medication_context(
                    text, match.start(), match.end()
                ):
                    continue
                if concept_type == ConceptType.SYMPTOM and _is_blocked_symptom_context(
                    text, match.start(), match.end()
                ):
                    continue
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

    def _extract_compiled_terms(
        self,
        text: str,
        pattern: re.Pattern[str] | None,
        concept_type: ConceptType,
        confidence: float,
        source: str,
    ) -> list[CandidateConcept]:
        if pattern is None:
            return []
        candidates: list[CandidateConcept] = []
        for match in pattern.finditer(text):
            if concept_type == ConceptType.MEDICATION and _is_blocked_medication_context(
                text, match.start(), match.end()
            ):
                continue
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

    def _extract_fuzzy_terms(
        self,
        text: str,
        terms_by_token_count: dict[int, list[FuzzySearchTerm]],
        concept_type: ConceptType,
        confidence: float,
        source: str,
        existing: list[CandidateConcept],
    ) -> list[CandidateConcept]:
        token_matches = list(TOKEN_PATTERN.finditer(text))
        if not token_matches:
            return []

        token_folds = [fold_term(match.group(0)) for match in token_matches]
        candidates: list[CandidateConcept] = []
        seen: set[tuple[int, int]] = set()
        for start_index in range(len(token_matches)):
            for token_count, terms in terms_by_token_count.items():
                end_index = start_index + token_count
                if end_index > len(token_matches):
                    continue
                start = token_matches[start_index].start()
                end = token_matches[end_index - 1].end()
                if (start, end) in seen or self._span_has_existing(
                    start, end, [*existing, *candidates]
                ):
                    continue
                folded_window = " ".join(token_folds[start_index:end_index])
                if len(folded_window) < 8:
                    continue
                best_score = 0.0
                best_term: FuzzySearchTerm | None = None
                for term in terms:
                    if abs(len(folded_window) - len(term.folded)) > max(
                        4, int(len(term.folded) * 0.15)
                    ):
                        continue
                    score = fuzz.ratio(folded_window, term.folded)
                    if score > best_score:
                        best_score = score
                        best_term = term
                if best_term is None or best_score < 96.0:
                    continue
                seen.add((start, end))
                candidates.append(
                    CandidateConcept(
                        text=text[start:end],
                        start_offset=start,
                        end_offset=end,
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
                if _is_blocked_lab_context(
                    lab_name, text, match.start(), match.end()
                ):
                    continue
                if lab_name == "bạch cầu" and _is_leukemia_context(
                    text, match.start(), match.end()
                ):
                    continue
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

    def _extract_contextual_symptoms(self, text: str) -> list[CandidateConcept]:
        candidates: list[CandidateConcept] = []
        weakness_patterns = (
            re.compile(
                r"(?im)(?:^|[.;\n])\s*(?:[-*]\s*)?"
                r"(?:lý do (?:nhập viện|vào viện|khám bệnh)|"
                r"(?:các\s+)?triệu chứng(?:\s+hiện tại)?|triệu chứng khi đến)"
                r"\s*:\s*(yếu)\b"
            ),
            re.compile(r"(?im)^\s*[-*]\s*(yếu)\s*$"),
        )
        for pattern in weakness_patterns:
            for match in pattern.finditer(text):
                start, end = match.span(1)
                candidates.append(
                    CandidateConcept(
                        text=text[start:end],
                        start_offset=start,
                        end_offset=end,
                        concept_type=ConceptType.SYMPTOM,
                        confidence=0.84,
                        source="contextual_symptom",
                    )
                )
        return candidates

    def _extract_contextual_medications(self, text: str) -> list[CandidateConcept]:
        candidates: list[CandidateConcept] = []
        patterns = (
            re.compile(r"\bthay\s+thế\s+(bicarbonate)\b", re.I),
            re.compile(
                r"\btruyền\s+dịch\s*:?\s*(?:\d+\s*ml\s*)?(NS\s*0[.,]9\s*%)",
                re.I,
            ),
            re.compile(r"\b(?:nhận|được\s+cho|dùng)\s+(40\s*meq\s+(?:po|iv)\s+k)\b", re.I),
        )
        for pattern in patterns:
            for match in pattern.finditer(text):
                start, end = match.span(1)
                candidates.append(
                    CandidateConcept(
                        text=text[start:end],
                        start_offset=start,
                        end_offset=end,
                        concept_type=ConceptType.MEDICATION,
                        confidence=0.84,
                        source="contextual_medication",
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
                *self._medication_terminology_terms,
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
        known_single_token_medications = {
            term.lower()
            for term in (
                *VI_MEDICATION_TERMS,
                *self._medication_terminology_terms,
            )
            if len(term) >= 5 and " " not in term
        }
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
                or _contains_known_medication_fragment(
                    token_key, known_single_token_medications
                )
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


def _is_blocked_medication_context(text: str, start: int, end: int) -> bool:
    term = text[start:end].lower()
    if term not in {"oxy", "oxygen"}:
        return False

    before = text[max(0, start - 35) : start].lower()
    after = text[end : min(len(text), end + 25)].lower()
    before_with_term = f"{before}{term}"
    nearby = f"{before_with_term}{after}"

    treatment_cues = (
        "dùng oxy",
        "không dùng oxy",
        "thở oxy",
        "không thở oxy",
        "cho bệnh nhân thở oxy",
        "điều trị bằng oxy",
    )
    if any(cue in before_with_term for cue in treatment_cues):
        return False
    if term == "oxygen" and after.lstrip().startswith("therapy"):
        return False

    measurement_cues = (
        "độ bão hòa",
        "bão hòa",
        "spo2",
        "sao2",
        "chỉ số",
        "thiếu oxy",
        "khí oxy",
    )
    return any(cue in nearby for cue in measurement_cues)


def _is_blocked_symptom_context(text: str, start: int, end: int) -> bool:
    term = text[start:end].lower()
    after = text[end : min(len(text), end + 12)].lower()
    if term == "phù":
        return after.lstrip().startswith("hợp")
    if term == "đỏ":
        return after.lstrip().startswith("tươi")
    return term == "ho" and _is_history_of_abbreviation_context(text, start, end)


def _is_blocked_lab_context(lab_name: str, text: str, start: int, end: int) -> bool:
    term = lab_name.lower()
    before = text[max(0, start - 45) : start].lower().rstrip()
    if term == "huyết thanh":
        return re.search(r"\bdịch(?:\s+rỉ)?$", before) is not None
    if term == "nội soi":
        if re.search(r"\bphẫu\s+thuật$", before):
            return True
        return re.search(r"\bcắt(?:\s+[\wÀ-ỹ-]+){0,6}$", before) is not None
    return False


def _is_history_of_abbreviation_context(text: str, start: int, end: int) -> bool:
    line_start = max(text.rfind("\n", 0, start) + 1, 0)
    prefix = text[line_start:start].strip().lower()
    if prefix not in {"-", "•", ""}:
        return False
    following = text[end : min(len(text), end + 45)].lstrip().lower()
    return following.startswith(
        (
            "bệnh ",
            "đái tháo đường",
            "rối loạn ",
            "rung nhĩ",
            "tăng ",
            "suy ",
            "ung thư",
            "hội chứng",
        )
    )


def _is_leukemia_context(text: str, start: int, end: int) -> bool:
    context = text[max(0, start - 12) : min(len(text), end + 35)].lower()
    return re.search(r"\bbệnh\s+bạch\s+cầu\b", context) is not None


def _contains_known_medication_fragment(token: str, known_medications: set[str]) -> bool:
    return any(term != token and term in token for term in known_medications)


def _build_fuzzy_terms(terms: list[str]) -> dict[int, list[FuzzySearchTerm]]:
    grouped: dict[int, list[FuzzySearchTerm]] = {}
    for term in terms:
        if not _has_non_ascii(term):
            continue
        folded = fold_term(term)
        token_count = len(folded.split())
        if len(folded) < 8 or token_count < 2:
            continue
        grouped.setdefault(token_count, []).append(
            FuzzySearchTerm(text=term, folded=folded)
        )
    return grouped


def _compile_term_pattern(terms: list[str]) -> re.Pattern[str] | None:
    unique_terms = sorted(
        {term for term in terms if term},
        key=lambda value: (-len(value), value.lower()),
    )
    if not unique_terms:
        return None
    alternates = "|".join(re.escape(term) for term in unique_terms)
    return re.compile(rf"(?<!\w)(?:{alternates})(?!\w)", re.I)


def _extractable_disease_terminology_terms(terms: list[str]) -> list[str]:
    blocked_folds = {
        fold_term(term)
        for term in (*SYMPTOM_TERMS, *LAB_NAMES, *TERMINOLOGY_DISEASE_BLOCK_TERMS)
        if term
    }
    curated_disease_folds = {fold_term(term) for term in VI_DISEASE_TERMS if term}
    extractable: list[str] = []
    for term in terms:
        folded = fold_term(term)
        if not folded:
            continue
        if folded in blocked_folds and folded not in curated_disease_folds:
            continue
        if _has_non_ascii(term) and len(folded.split()) < 2:
            continue
        extractable.append(term)
    return extractable


def _extractable_medication_terminology_terms(terms: list[str]) -> list[str]:
    blocked_folds = {fold_term(term) for term in TERMINOLOGY_MEDICATION_BLOCK_TERMS}
    return [term for term in terms if fold_term(term) not in blocked_folds]


def _has_non_ascii(text: str) -> bool:
    return any(ord(character) > 127 for character in text)


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
