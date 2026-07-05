from __future__ import annotations

import csv
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from rapidfuzz import fuzz

from clinical_nlp.schemas import ConceptType

CMS_ICD10_URL = "https://www.cms.gov/medicare/coding-billing/icd-10-codes"
RXNORM_URL = "https://www.nlm.nih.gov/research/umls/rxnorm/overview.html"
RXNORM_API_URL = "https://lhncbc.nlm.nih.gov/RxNav/APIs/RxNormAPIs.html"
WHO_ATC_URL = "https://www.who.int/tools/atc-ddd-toolkit/atc-classification"


@dataclass(frozen=True)
class TerminologyEntry:
    concept_type: ConceptType
    code_system: str
    code: str
    preferred_term: str
    synonyms: tuple[str, ...]
    release_id: str
    source_url: str

    def search_terms(self) -> tuple[str, ...]:
        return (self.preferred_term, *self.synonyms)


class TerminologyStore:
    def __init__(self, entries: list[TerminologyEntry]) -> None:
        self._entries = entries
        self._index: dict[tuple[ConceptType, str], TerminologyEntry] = {}
        self._folded_index: dict[tuple[ConceptType, str], TerminologyEntry] = {}
        for entry in entries:
            for term in entry.search_terms():
                normalized = normalize_term(term)
                if normalized:
                    self._index[(entry.concept_type, normalized)] = entry
                folded = fold_term(term)
                if folded:
                    self._folded_index[(entry.concept_type, folded)] = entry

    @classmethod
    def default(cls, directory: Path | None = None) -> TerminologyStore:
        if directory is not None and directory.exists():
            loaded = cls.from_directory(directory)
            if loaded._entries:
                return loaded
        return cls(demo_entries())

    @classmethod
    def from_directory(cls, directory: Path) -> TerminologyStore:
        entries: list[TerminologyEntry] = []
        for csv_path in sorted(directory.glob("*.csv")):
            entries.extend(load_entries_from_csv(csv_path))
        return cls(entries)

    def lookup(self, text: str, concept_type: str | ConceptType) -> TerminologyEntry | None:
        normalized_type = ConceptType(concept_type)
        return self._index.get((normalized_type, normalize_term(text))) or self._folded_index.get(
            (normalized_type, fold_term(text))
        )

    def lookup_fuzzy(
        self,
        text: str,
        concept_type: str | ConceptType,
        *,
        score_cutoff: float = 96.0,
    ) -> TerminologyEntry | None:
        normalized_type = ConceptType(concept_type)
        folded_text = fold_term(text)
        if len(folded_text) < 8 or len(folded_text.split()) < 2:
            return None

        best_entry: TerminologyEntry | None = None
        best_score = 0.0
        for (entry_type, folded_term), entry in self._folded_index.items():
            if entry_type != normalized_type:
                continue
            if len(folded_term) < 8 or len(folded_term.split()) < 2:
                continue
            if abs(len(folded_text) - len(folded_term)) > max(
                4, int(len(folded_term) * 0.15)
            ):
                continue
            score = fuzz.ratio(folded_text, folded_term)
            if score > best_score:
                best_score = score
                best_entry = entry

        if best_score < score_cutoff:
            return None
        return best_entry

    def search_terms_for(self, concept_type: ConceptType) -> list[str]:
        terms: set[str] = set()
        for entry in self._entries:
            if entry.concept_type == concept_type:
                terms.update(term for term in entry.search_terms() if term)
        return sorted(terms, key=lambda term: (-len(term), term.lower()))

    def releases(self) -> dict[str, str]:
        releases: dict[str, str] = {}
        for entry in self._entries:
            releases.setdefault(entry.code_system, entry.release_id)
        releases.setdefault("ATC", "not-loaded")
        return releases


def load_entries_from_csv(path: Path) -> list[TerminologyEntry]:
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        return [entry_from_row(row) for row in reader]


def entry_from_row(row: dict[str, str]) -> TerminologyEntry:
    code_system = row["code_system"].strip()
    synonyms = tuple(
        synonym.strip() for synonym in row.get("synonyms", "").split("|") if synonym.strip()
    )
    return TerminologyEntry(
        concept_type=concept_type_for_code_system(code_system),
        code_system=code_system,
        code=row["code"].strip(),
        preferred_term=row["preferred_term"].strip(),
        synonyms=synonyms,
        release_id=row.get("release_id", "unknown").strip() or "unknown",
        source_url=row.get("source_url", "").strip(),
    )


def concept_type_for_code_system(code_system: str) -> ConceptType:
    normalized = code_system.upper()
    if normalized.startswith("ICD"):
        return ConceptType.DISEASE
    if normalized in {"RXNORM", "ATC"}:
        return ConceptType.MEDICATION
    raise ValueError(f"Unsupported code system: {code_system}")


def normalize_term(term: str) -> str:
    return " ".join(term.strip().lower().replace("-", " ").split())


def fold_term(term: str) -> str:
    normalized = unicodedata.normalize("NFD", term.lower().replace("đ", "d"))
    without_marks = "".join(
        character
        for character in normalized
        if unicodedata.category(character) != "Mn"
    )
    return " ".join(re.findall(r"[a-z0-9]+", without_marks))


def _icd10_entry(
    code: str, preferred_term: str, synonyms: tuple[str, ...]
) -> TerminologyEntry:
    return TerminologyEntry(
        concept_type=ConceptType.DISEASE,
        code_system="ICD-10-CM",
        code=code,
        preferred_term=preferred_term,
        synonyms=synonyms,
        release_id="clinical-seed-see-cms",
        source_url=CMS_ICD10_URL,
    )


def demo_entries() -> list[TerminologyEntry]:
    return [
        TerminologyEntry(
            concept_type=ConceptType.DISEASE,
            code_system="ICD-10-CM",
            code="E11.9",
            preferred_term="Type 2 diabetes mellitus without complications",
            synonyms=(
                "type 2 diabetes",
                "t2dm",
                "diabetes mellitus type 2",
                "đái tháo đường",
                "đái tháo đường típ 2",
                "đái tháo đường type 2",
                "tiểu đường",
            ),
            release_id="demo-seed-see-cms",
            source_url=CMS_ICD10_URL,
        ),
        _icd10_entry(
            "I48.91",
            "Unspecified atrial fibrillation",
            (
                "rung nhĩ",
                "rung nhĩ kèm đáp ứng thất nhanh",
                "rung nhĩ điển hình kèm theo đáp ứng thất nhanh",
            ),
        ),
        _icd10_entry(
            "I48.0",
            "Paroxysmal atrial fibrillation",
            ("rung nhĩ kịch phát",),
        ),
        _icd10_entry(
            "N18.9",
            "Chronic kidney disease, unspecified",
            ("bệnh thận mạn", "bệnh thận mạn tính"),
        ),
        _icd10_entry(
            "N18.5",
            "Chronic kidney disease, stage 5",
            (
                "suy thận mạn giai đoạn v",
                "suy thận mạn giai v",
                "bệnh thận mạn giai đoạn v",
                "bệnh thận mạn giai đoạn 5",
            ),
        ),
        _icd10_entry(
            "N17.9",
            "Acute kidney failure, unspecified",
            ("suy thận cấp",),
        ),
        _icd10_entry(
            "N19",
            "Unspecified kidney failure",
            ("suy thận",),
        ),
        _icd10_entry(
            "I50.9",
            "Heart failure, unspecified",
            ("suy tim",),
        ),
        _icd10_entry(
            "I25.10",
            "Atherosclerotic heart disease of native coronary artery without angina pectoris",
            (
                "bệnh động mạch vành",
                "bệnh mạch vành",
                "bệnh tim mạch do xơ vữa động mạch",
            ),
        ),
        _icd10_entry(
            "I25.9",
            "Chronic ischemic heart disease, unspecified",
            ("thiếu máu cơ tim", "bệnh thiếu máu cơ tim"),
        ),
        _icd10_entry(
            "E78.5",
            "Hyperlipidemia, unspecified",
            ("tăng lipid máu", "rối loạn lipid máu"),
        ),
        _icd10_entry(
            "E78.00",
            "Pure hypercholesterolemia, unspecified",
            ("tăng cholesterol máu",),
        ),
        _icd10_entry(
            "J44.9",
            "Chronic obstructive pulmonary disease, unspecified",
            (
                "bệnh phổi tắc nghẽn mạn tính, không xác định",
                "bệnh phổi tắc nghẽn mạn tính",
                "copd",
            ),
        ),
        _icd10_entry(
            "J45.909",
            "Unspecified asthma, uncomplicated",
            ("hen suyễn", "hen phế quản"),
        ),
        _icd10_entry(
            "E05.0",
            "Thyrotoxicosis with diffuse goiter",
            ("bệnh graves", "graves"),
        ),
        _icd10_entry(
            "M10.9",
            "Gout, unspecified",
            ("bệnh gút", "gút"),
        ),
        _icd10_entry(
            "N81.4",
            "Uterovaginal prolapse, unspecified",
            ("sa âm đạo", "khối sa âm đạo"),
        ),
        _icd10_entry(
            "J18.9",
            "Pneumonia, unspecified organism",
            (
                "viêm phổi",
                "viêm phổi thùy dưới phải",
                "viêm phổi thùy dưới phải (rll pna)",
            ),
        ),
        _icd10_entry(
            "J81",
            "Pulmonary edema",
            ("phù phổi",),
        ),
        _icd10_entry(
            "A41.9",
            "Sepsis, unspecified organism",
            ("nhiễm trùng huyết",),
        ),
        _icd10_entry(
            "L03.90",
            "Cellulitis, unspecified",
            ("viêm mô tế bào",),
        ),
        _icd10_entry(
            "C18.9",
            "Malignant neoplasm of colon, unspecified",
            (
                "ung thư đại tràng",
                "ung thư biểu mô đại tràng",
                "ung thư biểu mô tuyến đại tràng",
            ),
        ),
        _icd10_entry(
            "I26.99",
            "Other pulmonary embolism without acute cor pulmonale",
            ("thuyên tắc phổi",),
        ),
        _icd10_entry(
            "J96.90",
            "Respiratory failure, unspecified, unspecified whether with hypoxia or hypercapnia",
            ("suy hô hấp",),
        ),
        _icd10_entry(
            "E87.5",
            "Hyperkalemia",
            ("tăng kali máu",),
        ),
        _icd10_entry(
            "E87.6",
            "Hypokalemia",
            ("hạ kali máu",),
        ),
        _icd10_entry(
            "E66.9",
            "Obesity, unspecified",
            ("béo phì",),
        ),
        _icd10_entry(
            "G47.33",
            "Obstructive sleep apnea (adult) (pediatric)",
            ("ngưng thở khi ngủ do tắc nghẽn",),
        ),
        _icd10_entry(
            "G47.30",
            "Sleep apnea, unspecified",
            ("ngưng thở khi ngủ",),
        ),
        _icd10_entry(
            "M86.9",
            "Osteomyelitis, unspecified",
            ("viêm tủy xương", "viêm tuỷ xương"),
        ),
        _icd10_entry(
            "I63.9",
            "Cerebral infarction, unspecified",
            ("đột quỵ",),
        ),
        _icd10_entry(
            "K81.0",
            "Acute cholecystitis",
            (
                "viêm túi mật cấp",
                "viêm túi mật cấp tính",
                "viêm túi mật thủng cấp tính",
            ),
        ),
        _icd10_entry(
            "K81.9",
            "Cholecystitis, unspecified",
            ("viêm túi mật", "viêm túi mật không biến chứng"),
        ),
        _icd10_entry(
            "K29.70",
            "Gastritis, unspecified, without bleeding",
            ("viêm dạ dày",),
        ),
        _icd10_entry(
            "A08.4",
            "Viral intestinal infection, unspecified",
            ("viêm dạ dày ruột do virus",),
        ),
        _icd10_entry(
            "C61",
            "Malignant neoplasm of prostate",
            ("u ác của tuyến tiền liệt", "ung thư tuyến tiền liệt"),
        ),
        _icd10_entry(
            "C24.0",
            "Malignant neoplasm of extrahepatic bile duct",
            (
                "ung thư đường mật",
                "ung thư đường mật không thể cắt bỏ",
                "ung thư biểu mô tế bào mật",
                "ung thư biểu mô tế bào mật không thể cắt bỏ",
            ),
        ),
        _icd10_entry(
            "C34.90",
            "Malignant neoplasm of unspecified part of unspecified bronchus or lung",
            (
                "ung thư phổi",
                "ung thư phổi không tế bào nhỏ",
            ),
        ),
        _icd10_entry(
            "C79.31",
            "Secondary malignant neoplasm of brain",
            (
                "di căn não",
                "di căn não vùng trán phải",
            ),
        ),
        _icd10_entry(
            "C78.0",
            "Secondary malignant neoplasm of lung",
            ("ung thư di căn theo đường bạch huyết ở hai phổi",),
        ),
        _icd10_entry(
            "C60.9",
            "Malignant neoplasm of penis, unspecified",
            (
                "ung thư biểu mô tế bào vảy xâm nhập của dương vật",
                "ung thư biểu mô tế bào vảy xâm nhập của dương vậtbiệt hóa kém",
            ),
        ),
        _icd10_entry(
            "Q96.9",
            "Turner's syndrome, unspecified",
            ("hội chứng turner",),
        ),
        _icd10_entry(
            "C90.00",
            "Multiple myeloma not having achieved remission",
            ("đa u tủy xương", "đa u tuỷ xương", "bệnh đa u tuỷ xương"),
        ),
        _icd10_entry(
            "D64.9",
            "Anemia, unspecified",
            ("thiếu máu", "thiếu máu hồng cầu nhỏ"),
        ),
        _icd10_entry(
            "J98.11",
            "Atelectasis",
            ("xẹp phổi",),
        ),
        _icd10_entry(
            "J90",
            "Pleural effusion, not elsewhere classified",
            ("tràn dịch màng phổi",),
        ),
        _icd10_entry(
            "N39.0",
            "Urinary tract infection, site not specified",
            (
                "nhiễm trùng đường tiết niệu kháng thuốc",
                "nhiễm trùng đường tiết niệu",
                "nhiễm khuẩn đường tiết niệu",
                "nhiễm khuẩn đường tiết niệu, vị trí không xác định",
                "nhiễm trùng đường vào tiết niệu",
            ),
        ),
        _icd10_entry(
            "J06.9",
            "Acute upper respiratory infection, unspecified",
            ("nhiễm trùng đường hô hấp trên cấp",),
        ),
        _icd10_entry(
            "K80.20",
            "Calculus of gallbladder without cholecystitis without obstruction",
            ("sỏi mật",),
        ),
        _icd10_entry(
            "K57.90",
            "Diverticulosis of intestine, part unspecified, without perforation or abscess",
            ("bệnh túi thừa",),
        ),
        _icd10_entry(
            "K51.90",
            "Ulcerative colitis, unspecified, without complications",
            ("viêm loét đại tràng",),
        ),
        _icd10_entry(
            "M48.00",
            "Spinal stenosis, site unspecified",
            ("hẹp ống sống",),
        ),
        _icd10_entry(
            "M54.1",
            "Radiculopathy",
            (
                "bệnh rễ thần kinh tuỷ sống",
                "bệnh rễ thần kinh tuỷ sống ở ngón tay cái",
            ),
        ),
        _icd10_entry(
            "M99.7",
            "Connective tissue and disc stenosis of intervertebral foramina",
            ("hẹp lỗ liên hợp",),
        ),
        _icd10_entry(
            "G62.9",
            "Polyneuropathy, unspecified",
            (
                "bệnh lý thần kinh ngoại biên",
                "viêm thần kinh ngoại biên",
            ),
        ),
        _icd10_entry(
            "F32.A",
            "Depression, unspecified",
            ("trầm cảm", "rối loạn cảm xúc trầm cảm"),
        ),
        _icd10_entry(
            "F20.9",
            "Schizophrenia, unspecified",
            ("tâm thần phân liệt",),
        ),
        _icd10_entry(
            "F30.9",
            "Manic episode, unspecified",
            ("hưng cảm",),
        ),
        _icd10_entry(
            "F31.9",
            "Bipolar disorder, unspecified",
            ("rối loạn lưỡng cực",),
        ),
        _icd10_entry(
            "R45.851",
            "Suicidal ideations",
            ("ý định tự tử",),
        ),
        _icd10_entry(
            "C92.10",
            "Chronic myeloid leukemia, BCR/ABL-positive, not having achieved remission",
            (
                "cml",
                "bệnh bạch cầu mạn tính dòng tủy",
                "bệnh bạch cầu dòng tủy mãn tính",
                "bệnh bạch cầu dòng tủy mạn tính",
            ),
        ),
        _icd10_entry(
            "I51.7",
            "Cardiomegaly",
            ("tim to",),
        ),
        _icd10_entry(
            "I31.39",
            "Other pericardial effusion (noninflammatory)",
            ("tràn dịch màng tim", "tràn dịch màng ngoài tim"),
        ),
        _icd10_entry(
            "J43.9",
            "Emphysema, unspecified",
            ("khí phế thủng",),
        ),
        _icd10_entry(
            "K44.9",
            "Diaphragmatic hernia without obstruction or gangrene",
            ("thoát vị hoành", "thoát vị cạnh thực quản"),
        ),
        _icd10_entry(
            "K21.9",
            "Gastro-esophageal reflux disease without esophagitis",
            (
                "bệnh trào ngược dạ dày - thực quản",
                "trào ngược dạ dày thực quản",
                "trào ngược dạ dày - thực quản",
            ),
        ),
        _icd10_entry(
            "I35.0",
            "Nonrheumatic aortic valve stenosis",
            ("hẹp van động mạch chủ", "hẹp van động mạch chủ nghiêm trọng"),
        ),
        _icd10_entry(
            "I34.0",
            "Nonrheumatic mitral (valve) insufficiency",
            ("hở van hai lá",),
        ),
        _icd10_entry(
            "I34.1",
            "Nonrheumatic mitral (valve) prolapse",
            ("sa van hai lá",),
        ),
        _icd10_entry(
            "I36.1",
            "Nonrheumatic tricuspid (valve) insufficiency",
            ("hở van ba lá",),
        ),
        _icd10_entry(
            "I31.4",
            "Cardiac tamponade",
            ("chèn ép tim",),
        ),
        _icd10_entry(
            "I60.9",
            "Nontraumatic subarachnoid hemorrhage, unspecified",
            ("xuất huyết dưới nhện", "xuất huyết dưới nhện vùng trán phải"),
        ),
        _icd10_entry(
            "I62.00",
            "Nontraumatic subdural hemorrhage, unspecified",
            (
                "xuất huyết dưới màng cứng",
                "tụ máu dưới màng cứng",
                "khối máu tụ dưới màng cứng",
            ),
        ),
        _icd10_entry(
            "I62.9",
            "Nontraumatic intracranial hemorrhage, unspecified",
            ("xuất huyết nội sọ không do chấn thương",),
        ),
        _icd10_entry(
            "I62.1",
            "Nontraumatic extradural hemorrhage",
            ("tụ máu ngoài màng cứng",),
        ),
        _icd10_entry(
            "G91.8",
            "Other hydrocephalus",
            (
                "não úng thủy khác",
                "não úng thuỷ khác",
                "não úng tuỷ khác",
            ),
        ),
        _icd10_entry(
            "G91.9",
            "Hydrocephalus, unspecified",
            (
                "não úng thủy",
                "não úng thuỷ",
                "não úng tuỷ",
            ),
        ),
        _icd10_entry(
            "I47.10",
            "Supraventricular tachycardia, unspecified",
            ("nhịp nhanh trên thất",),
        ),
        _icd10_entry(
            "R00.1",
            "Bradycardia, unspecified",
            (
                "nhịp tim chậm",
                "nhịp tim chậm tương đối",
                "nhịp tim chậm nặng",
                "nhịp chậm xoang",
            ),
        ),
        _icd10_entry(
            "K26.9",
            "Duodenal ulcer, unspecified as acute or chronic, without hemorrhage or perforation",
            ("loét tá tràng",),
        ),
        _icd10_entry(
            "K22.10",
            "Ulcer of esophagus without bleeding",
            ("loét thực quản",),
        ),
        _icd10_entry(
            "K83.1",
            "Obstruction of bile duct",
            (
                "tắc nghẽn đường mật",
                "tắc nghẽn đường dẫn mật",
                "tắc nghẽn kéo dài gần chỗ nối mật tụy",
                "tắc nghẽn kéo dài gần chỗ nối mật tuỵ",
            ),
        ),
        _icd10_entry(
            "K86.2",
            "Cyst of pancreas",
            (
                "nang tụy",
                "nang tuỵ",
            ),
        ),
        _icd10_entry(
            "K83.8",
            "Other specified diseases of biliary tract",
            ("giãn đường mật", "giãn đường dẫn mật"),
        ),
        _icd10_entry(
            "K80.50",
            "Calculus of bile duct without cholangitis or cholecystitis without obstruction",
            (
                "sỏi ống mật",
                "sỏi ống dẫn mật chung",
                "sỏi ống dẫn mật chung đoạn cuối",
            ),
        ),
        _icd10_entry(
            "E04.1",
            "Nontoxic single thyroid nodule",
            ("nốt tuyến giáp",),
        ),
        _icd10_entry(
            "D25.9",
            "Leiomyoma of uterus, unspecified",
            ("u xơ tử cung", "u cơ trơn tử cung"),
        ),
        _icd10_entry(
            "N31.9",
            "Neuromuscular dysfunction of bladder, unspecified",
            ("bàng quang thần kinh",),
        ),
        _icd10_entry(
            "G82.20",
            "Paraplegia, unspecified",
            ("liệt hai chi dưới", "liệt hai chân"),
        ),
        _icd10_entry(
            "F41.9",
            "Anxiety disorder, unspecified",
            ("rối loạn lo âu, không biệt định nghiêm trọng", "rối loạn lo âu"),
        ),
        _icd10_entry(
            "F10.20",
            "Alcohol dependence, uncomplicated",
            ("hội chứng nghiện rượu",),
        ),
        _icd10_entry(
            "F19.10",
            "Other psychoactive substance abuse, uncomplicated",
            ("lạm dụng chất kích thích", "lạm dụng chất gây nghiện opioid"),
        ),
        _icd10_entry(
            "K76.6",
            "Portal hypertension",
            ("tăng áp lực tĩnh mạch cửa", "áp lực tĩnh mạch cửa"),
        ),
        _icd10_entry(
            "K70.9",
            "Alcoholic liver disease, unspecified",
            ("bệnh gan do rượu",),
        ),
        _icd10_entry(
            "R18.8",
            "Other ascites",
            ("cổ trướng",),
        ),
        _icd10_entry(
            "M00.9",
            "Pyogenic arthritis, unspecified",
            ("viêm khớp nhiễm trùng",),
        ),
        _icd10_entry(
            "I65.29",
            "Occlusion and stenosis of unspecified carotid artery",
            (
                "hẹp động mạch cảnh",
                "nghẽn tắc và hẹp động mạch cảnh",
                "hẹp động mạch cảnh chung trái",
                "hẹp khoảng 65% của động mạch cảnh chung trái",
            ),
        ),
        TerminologyEntry(
            concept_type=ConceptType.DISEASE,
            code_system="ICD-10-CM",
            code="I10",
            preferred_term="Essential (primary) hypertension",
            synonyms=("hypertension", "high blood pressure", "tăng huyết áp"),
            release_id="demo-seed-see-cms",
            source_url=CMS_ICD10_URL,
        ),
        TerminologyEntry(
            concept_type=ConceptType.DISEASE,
            code_system="ICD-10-CM",
            code="C50.919",
            preferred_term="Malignant neoplasm of unspecified site of unspecified female breast",
            synonyms=("breast cancer", "ung thư vú"),
            release_id="demo-seed-see-cms",
            source_url=CMS_ICD10_URL,
        ),
        TerminologyEntry(
            concept_type=ConceptType.DISEASE,
            code_system="ICD-10-CM",
            code="K74.6",
            preferred_term="Other and unspecified cirrhosis of liver",
            synonyms=("xơ gan",),
            release_id="clinical-seed-see-cms",
            source_url=CMS_ICD10_URL,
        ),
        TerminologyEntry(
            concept_type=ConceptType.DISEASE,
            code_system="ICD-10-CM",
            code="E83.52",
            preferred_term="Hypercalcemia",
            synonyms=("tăng calci máu", "tăng canxi máu"),
            release_id="clinical-seed-see-cms",
            source_url=CMS_ICD10_URL,
        ),
        TerminologyEntry(
            concept_type=ConceptType.DISEASE,
            code_system="ICD-10-CM",
            code="C20",
            preferred_term="Malignant neoplasm of rectum",
            synonyms=("u ác trực tràng", "khối u trực tràng"),
            release_id="clinical-seed-see-cms",
            source_url=CMS_ICD10_URL,
        ),
        TerminologyEntry(
            concept_type=ConceptType.DISEASE,
            code_system="ICD-10-CM",
            code="C18.9",
            preferred_term="Malignant neoplasm of colon, unspecified",
            synonyms=("u ác đại tràng", "u ác của đại tràng"),
            release_id="clinical-seed-see-cms",
            source_url=CMS_ICD10_URL,
        ),
        TerminologyEntry(
            concept_type=ConceptType.DISEASE,
            code_system="ICD-10-CM",
            code="E21.0",
            preferred_term="Primary hyperparathyroidism",
            synonyms=("cường cận giáp nguyên phát",),
            release_id="clinical-seed-see-cms",
            source_url=CMS_ICD10_URL,
        ),
        TerminologyEntry(
            concept_type=ConceptType.DISEASE,
            code_system="ICD-10-CM",
            code="L73.2",
            preferred_term="Hidradenitis suppurativa",
            synonyms=("viêm tuyến mồ hôi",),
            release_id="clinical-seed-see-cms",
            source_url=CMS_ICD10_URL,
        ),
        TerminologyEntry(
            concept_type=ConceptType.DISEASE,
            code_system="ICD-10-CM",
            code="I70.9",
            preferred_term="Generalized and unspecified atherosclerosis",
            synonyms=("xơ vữa động mạch",),
            release_id="clinical-seed-see-cms",
            source_url=CMS_ICD10_URL,
        ),
        TerminologyEntry(
            concept_type=ConceptType.DISEASE,
            code_system="ICD-10-CM",
            code="I49.1",
            preferred_term="Atrial premature depolarization",
            synonyms=("ngoại tâm thu nhĩ",),
            release_id="clinical-seed-see-cms",
            source_url=CMS_ICD10_URL,
        ),
        TerminologyEntry(
            concept_type=ConceptType.DISEASE,
            code_system="ICD-10-CM",
            code="I49.3",
            preferred_term="Ventricular premature depolarization",
            synonyms=("ngoại tâm thu thất",),
            release_id="clinical-seed-see-cms",
            source_url=CMS_ICD10_URL,
        ),
        TerminologyEntry(
            concept_type=ConceptType.DISEASE,
            code_system="ICD-10-CM",
            code="I20.8",
            preferred_term="Other forms of angina pectoris",
            synonyms=("cơn đau thắt ngực ổn định",),
            release_id="clinical-seed-see-cms",
            source_url=CMS_ICD10_URL,
        ),
        TerminologyEntry(
            concept_type=ConceptType.DISEASE,
            code_system="ICD-10-CM",
            code="K76.82",
            preferred_term="Hepatic encephalopathy",
            synonyms=("hội chứng não gan",),
            release_id="clinical-seed-see-cms",
            source_url=CMS_ICD10_URL,
        ),
        TerminologyEntry(
            concept_type=ConceptType.DISEASE,
            code_system="ICD-10-CM",
            code="I25.2",
            preferred_term="Old myocardial infarction",
            synonyms=(
                "nhồi máu cơ tim cũ",
                "nhồi máu cơ tim vùng dưới cũ",
                "nhồi máu cơ tim vùng vách liên thất, mạn tính và đỉnh",
            ),
            release_id="clinical-seed-see-cms",
            source_url=CMS_ICD10_URL,
        ),
        TerminologyEntry(
            concept_type=ConceptType.DISEASE,
            code_system="ICD-10-CM",
            code="I21.9",
            preferred_term="Acute myocardial infarction, unspecified",
            synonyms=("nhồi máu cơ tim",),
            release_id="clinical-seed-see-cms",
            source_url=CMS_ICD10_URL,
        ),
        TerminologyEntry(
            concept_type=ConceptType.DISEASE,
            code_system="ICD-10-CM",
            code="I63.9",
            preferred_term="Cerebral infarction, unspecified",
            synonyms=("nhồi máu cũ nhỏ ở vỏ não", "nhồi máu cũ nhỏ ở vỏ não đỉnh trái"),
            release_id="clinical-seed-see-cms",
            source_url=CMS_ICD10_URL,
        ),
        TerminologyEntry(
            concept_type=ConceptType.DISEASE,
            code_system="ICD-10-CM",
            code="D36.9",
            preferred_term="Benign neoplasm, unspecified site",
            synonyms=("u tuyến",),
            release_id="clinical-seed-see-cms",
            source_url=CMS_ICD10_URL,
        ),
        _icd10_entry(
            "Q61.3",
            "Polycystic kidney, unspecified",
            ("bệnh thận đa nang",),
        ),
        _icd10_entry(
            "T86.11",
            "Kidney transplant rejection",
            (
                "đào thải ghép thận",
                "đào thải kháng thể hữu hình cấp tính",
                "đào thải kháng thể hữu hình cấp tính (trong sinh thiết)",
            ),
        ),
        _icd10_entry(
            "R90.82",
            "White matter disease, unspecified",
            ("bệnh lý chất trắng",),
        ),
        _icd10_entry(
            "A41.01",
            "Sepsis due to Methicillin susceptible Staphylococcus aureus",
            (
                "nhiễm khuẩn huyết do tụ cầu vàng nhạy cảm methicillin",
                (
                    "nhiễm khuẩn huyết do tụ cầu vàng nhạy cảm methicillin, "
                    "nghi liên quan đến đường truyền tĩnh mạch trung tâm"
                ),
            ),
        ),
        _icd10_entry(
            "I71.012",
            "Dissection of descending thoracic aorta",
            (
                "tách thành động mạch chủ",
                "bóc tách động mạch chủ loại b",
                "tổn thương bóc tách động mạch chủ stanford loại b",
            ),
        ),
        _icd10_entry(
            "I71.6",
            "Thoracoabdominal aortic aneurysm, without rupture",
            (
                "phình động mạch chủ ngực – bụng",
                "phình động mạch chủ ngực bụng",
            ),
        ),
        _icd10_entry(
            "I71.9",
            "Aortic aneurysm of unspecified site, without rupture",
            ("phình động mạch chủ nhỏ",),
        ),
        _icd10_entry(
            "Q66.0",
            "Congenital talipes equinovarus",
            ("bàn chân vẹo bẩm sinh",),
        ),
        _icd10_entry(
            "I70.1",
            "Atherosclerosis of renal artery",
            (
                "hẹp động mạch thận",
                "tắc hẹp 80% động mạch thận trái",
            ),
        ),
        _icd10_entry(
            "I77.0",
            "Arteriovenous fistula, acquired",
            (
                "rò động - tĩnh mạch đùi phải",
                "rò động - tĩnh mạch đùi phải do thủ thuật can thiệp",
            ),
        ),
        _icd10_entry(
            "I73.9",
            "Peripheral vascular disease, unspecified",
            ("bệnh mạch máu ngoại biên",),
        ),
        _icd10_entry(
            "S22.42XA",
            "Multiple fractures of ribs, left side, initial encounter for closed fracture",
            ("gãy xương sườn trái",),
        ),
        TerminologyEntry(
            concept_type=ConceptType.MEDICATION,
            code_system="RxNorm",
            code="6809",
            preferred_term="metformin",
            synonyms=("metformin",),
            release_id="demo-seed-see-rxnorm",
            source_url=RXNORM_URL,
        ),
        TerminologyEntry(
            concept_type=ConceptType.MEDICATION,
            code_system="RxNorm",
            code="1191",
            preferred_term="aspirin",
            synonyms=("aspirin", "asa"),
            release_id="demo-seed-see-rxnorm",
            source_url=RXNORM_URL,
        ),
        TerminologyEntry(
            concept_type=ConceptType.MEDICATION,
            code_system="RxNorm",
            code="317300",
            preferred_term="aspirin 325 mg",
            synonyms=("aspirin 325mg", "aspirin 325mg x 1"),
            release_id="rxnav-api-2026-07-02",
            source_url=RXNORM_API_URL,
        ),
        TerminologyEntry(
            concept_type=ConceptType.MEDICATION,
            code_system="RxNorm",
            code="1370489",
            preferred_term="metoprolol 25 mg",
            synonyms=("metoprolol 25mg", "metoprolol 25mg po bid"),
            release_id="rxnav-api-2026-07-02",
            source_url=RXNORM_API_URL,
        ),
        TerminologyEntry(
            concept_type=ConceptType.MEDICATION,
            code_system="RxNorm",
            code="6918",
            preferred_term="metoprolol",
            synonyms=("metoprolol",),
            release_id="rxnav-api-2026-07-02",
            source_url=RXNORM_API_URL,
        ),
        TerminologyEntry(
            concept_type=ConceptType.MEDICATION,
            code_system="RxNorm",
            code="3640",
            preferred_term="doxycycline",
            synonyms=("doxycycline", "doxycyclin"),
            release_id="rxnav-api-2026-07-02",
            source_url=RXNORM_API_URL,
        ),
        TerminologyEntry(
            concept_type=ConceptType.MEDICATION,
            code_system="RxNorm",
            code="10831",
            preferred_term="sulfamethoxazole / trimethoprim",
            synonyms=("cotrimoxazol", "cotrimoxazole", "trimethoprim sulfamethoxazole"),
            release_id="clinical-seed-see-rxnorm",
            source_url=RXNORM_API_URL,
        ),
        TerminologyEntry(
            concept_type=ConceptType.MEDICATION,
            code_system="RxNorm",
            code="325642",
            preferred_term="ertapenem",
            synonyms=("ertapenem",),
            release_id="clinical-seed-see-rxnorm",
            source_url=RXNORM_API_URL,
        ),
        TerminologyEntry(
            concept_type=ConceptType.MEDICATION,
            code_system="RxNorm",
            code="7052",
            preferred_term="morphine",
            synonyms=("morphine",),
            release_id="clinical-seed-see-rxnorm",
            source_url=RXNORM_API_URL,
        ),
        TerminologyEntry(
            concept_type=ConceptType.MEDICATION,
            code_system="RxNorm",
            code="42844",
            preferred_term="Percocet",
            synonyms=("percocet",),
            release_id="rxnav-api-2026-07-03",
            source_url=RXNORM_API_URL,
        ),
        TerminologyEntry(
            concept_type=ConceptType.MEDICATION,
            code_system="RxNorm",
            code="83553",
            preferred_term="Seroquel",
            synonyms=("seroquel",),
            release_id="rxnav-api-2026-07-03",
            source_url=RXNORM_API_URL,
        ),
        TerminologyEntry(
            concept_type=ConceptType.MEDICATION,
            code_system="RxNorm",
            code="90176",
            preferred_term="iron",
            synonyms=("iron",),
            release_id="rxnav-api-2026-07-03",
            source_url=RXNORM_API_URL,
        ),
        TerminologyEntry(
            concept_type=ConceptType.MEDICATION,
            code_system="RxNorm",
            code="35827",
            preferred_term="ketorolac",
            synonyms=("ketorolac", "toradol"),
            release_id="rxnav-api-2026-07-03",
            source_url=RXNORM_API_URL,
        ),
        TerminologyEntry(
            concept_type=ConceptType.MEDICATION,
            code_system="RxNorm",
            code="4839",
            preferred_term="glucocorticoids",
            synonyms=("corticoid", "corticoids", "glucocorticoids"),
            release_id="rxnav-approximate-2026-07-03",
            source_url=RXNORM_API_URL,
        ),
        TerminologyEntry(
            concept_type=ConceptType.MEDICATION,
            code_system="RxNorm",
            code="214182",
            preferred_term="hydrocodone / acetaminophen",
            synonyms=("hydrocodone acetaminophen", "vicodin"),
            release_id="rxnav-api-2026-07-03",
            source_url=RXNORM_API_URL,
        ),
        TerminologyEntry(
            concept_type=ConceptType.MEDICATION,
            code_system="RxNorm",
            code="1202",
            preferred_term="atenolol",
            synonyms=("atenolol",),
            release_id="rxnav-api-2026-07-02",
            source_url=RXNORM_API_URL,
        ),
        TerminologyEntry(
            concept_type=ConceptType.MEDICATION,
            code_system="RxNorm",
            code="392085",
            preferred_term="guaifenesin ml po q6h:prn",
            synonyms=("guaifenesin", "guaifenesin ml po q6h:prn"),
            release_id="btc-example",
            source_url=RXNORM_URL,
        ),
        TerminologyEntry(
            concept_type=ConceptType.MEDICATION,
            code_system="RxNorm",
            code="315266",
            preferred_term="acetaminophen 500 mg",
            synonyms=("acetaminophen 500mg",),
            release_id="rxnav-api-2026-07-02",
            source_url=RXNORM_API_URL,
        ),
        TerminologyEntry(
            concept_type=ConceptType.MEDICATION,
            code_system="RxNorm",
            code="161",
            preferred_term="acetaminophen",
            synonyms=("acetaminophen",),
            release_id="rxnav-api-2026-07-02",
            source_url=RXNORM_API_URL,
        ),
        TerminologyEntry(
            concept_type=ConceptType.MEDICATION,
            code_system="RxNorm",
            code="316365",
            preferred_term="nitroglycerin 0.4 mg",
            synonyms=("nitroglycerin 0.4mg",),
            release_id="rxnav-api-2026-07-02",
            source_url=RXNORM_API_URL,
        ),
        TerminologyEntry(
            concept_type=ConceptType.MEDICATION,
            code_system="RxNorm",
            code="4917",
            preferred_term="nitroglycerin",
            synonyms=("nitroglycerin", "nitro", "ntg"),
            release_id="rxnav-api-2026-07-02",
            source_url=RXNORM_API_URL,
        ),
        TerminologyEntry(
            concept_type=ConceptType.MEDICATION,
            code_system="RxNorm",
            code="315970",
            preferred_term="furosemide 20 mg",
            synonyms=("furosemide 20mg", "lasix 20mg", "laxis 20mg"),
            release_id="rxnav-api-2026-07-02",
            source_url=RXNORM_API_URL,
        ),
        TerminologyEntry(
            concept_type=ConceptType.MEDICATION,
            code_system="RxNorm",
            code="202991",
            preferred_term="Lasix",
            synonyms=("lasix", "laxis"),
            release_id="rxnav-api-2026-07-02",
            source_url=RXNORM_API_URL,
        ),
        TerminologyEntry(
            concept_type=ConceptType.MEDICATION,
            code_system="RxNorm",
            code="9863",
            preferred_term="sodium chloride",
            synonyms=("natri clorid", "natriclori", "sodium chloride"),
            release_id="rxnav-api-2026-07-02",
            source_url=RXNORM_API_URL,
        ),
        TerminologyEntry(
            concept_type=ConceptType.MEDICATION,
            code_system="RxNorm",
            code="202433",
            preferred_term="Tylenol",
            synonyms=("tylenol",),
            release_id="rxnav-api-2026-07-02",
            source_url=RXNORM_API_URL,
        ),
        TerminologyEntry(
            concept_type=ConceptType.MEDICATION,
            code_system="RxNorm",
            code="152633",
            preferred_term="CellCept",
            synonyms=("cellcept",),
            release_id="rxnav-api-2026-07-02",
            source_url=RXNORM_API_URL,
        ),
        TerminologyEntry(
            concept_type=ConceptType.MEDICATION,
            code_system="RxNorm",
            code="11124",
            preferred_term="vancomycin",
            synonyms=("vancomycin",),
            release_id="rxnav-api-2026-07-02",
            source_url=RXNORM_API_URL,
        ),
        TerminologyEntry(
            concept_type=ConceptType.MEDICATION,
            code_system="RxNorm",
            code="202421",
            preferred_term="Coumadin",
            synonyms=("coumadin",),
            release_id="rxnav-api-2026-07-02",
            source_url=RXNORM_API_URL,
        ),
        TerminologyEntry(
            concept_type=ConceptType.MEDICATION,
            code_system="RxNorm",
            code="7646",
            preferred_term="omeprazole",
            synonyms=("omeprazole",),
            release_id="rxnav-api-2026-07-02",
            source_url=RXNORM_API_URL,
        ),
        TerminologyEntry(
            concept_type=ConceptType.MEDICATION,
            code_system="RxNorm",
            code="8640",
            preferred_term="prednisone",
            synonyms=("prednisone", "prednisone 40 mg/ngày trong 3 ngày"),
            release_id="rxnav-api-2026-07-02",
            source_url=RXNORM_API_URL,
        ),
        TerminologyEntry(
            concept_type=ConceptType.MEDICATION,
            code_system="RxNorm",
            code="5224",
            preferred_term="heparin",
            synonyms=("heparin",),
            release_id="rxnav-api-2026-07-02",
            source_url=RXNORM_API_URL,
        ),
        TerminologyEntry(
            concept_type=ConceptType.MEDICATION,
            code_system="RxNorm",
            code="352990",
            preferred_term="Suboxone",
            synonyms=("suboxone",),
            release_id="rxnav-api-2026-07-02",
            source_url=RXNORM_API_URL,
        ),
        TerminologyEntry(
            concept_type=ConceptType.MEDICATION,
            code_system="RxNorm",
            code="82122",
            preferred_term="levofloxacin",
            synonyms=("levofloxacin",),
            release_id="rxnav-api-2026-07-02",
            source_url=RXNORM_API_URL,
        ),
        TerminologyEntry(
            concept_type=ConceptType.MEDICATION,
            code_system="RxNorm",
            code="151399",
            preferred_term="Bactrim",
            synonyms=("bactrim",),
            release_id="rxnav-api-2026-07-02",
            source_url=RXNORM_API_URL,
        ),
        TerminologyEntry(
            concept_type=ConceptType.MEDICATION,
            code_system="RxNorm",
            code="1364436",
            preferred_term="Eliquis",
            synonyms=("eliquis",),
            release_id="rxnav-api-2026-07-02",
            source_url=RXNORM_API_URL,
        ),
        TerminologyEntry(
            concept_type=ConceptType.MEDICATION,
            code_system="RxNorm",
            code="282386",
            preferred_term="Gleevec",
            synonyms=("gleevec",),
            release_id="rxnav-api-2026-07-02",
            source_url=RXNORM_API_URL,
        ),
        TerminologyEntry(
            concept_type=ConceptType.MEDICATION,
            code_system="RxNorm",
            code="38413",
            preferred_term="torsemide",
            synonyms=("torsemide",),
            release_id="rxnav-api-2026-07-02",
            source_url=RXNORM_API_URL,
        ),
        TerminologyEntry(
            concept_type=ConceptType.MEDICATION,
            code_system="RxNorm",
            code="253182",
            preferred_term="insulin",
            synonyms=("insulin",),
            release_id="rxnav-api-2026-07-02",
            source_url=RXNORM_API_URL,
        ),
        TerminologyEntry(
            concept_type=ConceptType.MEDICATION,
            code_system="RxNorm",
            code="274783",
            preferred_term="insulin glargine",
            synonyms=("glargine", "insulin glargine"),
            release_id="rxnav-api-2026-07-02",
            source_url=RXNORM_API_URL,
        ),
        TerminologyEntry(
            concept_type=ConceptType.MEDICATION,
            code_system="RxNorm",
            code="6813",
            preferred_term="methadone",
            synonyms=("methadone",),
            release_id="rxnav-api-2026-07-02",
            source_url=RXNORM_API_URL,
        ),
        TerminologyEntry(
            concept_type=ConceptType.MEDICATION,
            code_system="RxNorm",
            code="18631",
            preferred_term="azithromycin",
            synonyms=("azithromycin",),
            release_id="rxnav-api-2026-07-02",
            source_url=RXNORM_API_URL,
        ),
        TerminologyEntry(
            concept_type=ConceptType.MEDICATION,
            code_system="RxNorm",
            code="82000",
            preferred_term="Colace",
            synonyms=("colace",),
            release_id="rxnav-api-2026-07-02",
            source_url=RXNORM_API_URL,
        ),
        TerminologyEntry(
            concept_type=ConceptType.MEDICATION,
            code_system="RxNorm",
            code="203563",
            preferred_term="Cipro",
            synonyms=("cipro",),
            release_id="rxnav-api-2026-07-02",
            source_url=RXNORM_API_URL,
        ),
        TerminologyEntry(
            concept_type=ConceptType.MEDICATION,
            code_system="RxNorm",
            code="7806",
            preferred_term="oxygen",
            synonyms=("oxygen", "oxy"),
            release_id="rxnav-api-2026-07-02",
            source_url=RXNORM_API_URL,
        ),
        TerminologyEntry(
            concept_type=ConceptType.MEDICATION,
            code_system="RxNorm",
            code="2191",
            preferred_term="ceftazidime",
            synonyms=("ceftazidime",),
            release_id="rxnav-api-2026-07-03",
            source_url=RXNORM_API_URL,
        ),
        TerminologyEntry(
            concept_type=ConceptType.MEDICATION,
            code_system="RxNorm",
            code="74170",
            preferred_term="Zosyn",
            synonyms=("zosyn",),
            release_id="rxnav-api-2026-07-03",
            source_url=RXNORM_API_URL,
        ),
        TerminologyEntry(
            concept_type=ConceptType.MEDICATION,
            code_system="RxNorm",
            code="197",
            preferred_term="acetylcysteine",
            synonyms=("acetylcysteine", "nac"),
            release_id="rxnav-api-2026-07-03",
            source_url=RXNORM_API_URL,
        ),
        TerminologyEntry(
            concept_type=ConceptType.MEDICATION,
            code_system="RxNorm",
            code="224913",
            preferred_term="Dilaudid",
            synonyms=("dilaudid",),
            release_id="rxnav-api-2026-07-03",
            source_url=RXNORM_API_URL,
        ),
        TerminologyEntry(
            concept_type=ConceptType.MEDICATION,
            code_system="RxNorm",
            code="174742",
            preferred_term="Plavix",
            synonyms=("plavix",),
            release_id="rxnav-api-2026-07-03",
            source_url=RXNORM_API_URL,
        ),
        TerminologyEntry(
            concept_type=ConceptType.MEDICATION,
            code_system="RxNorm",
            code="202585",
            preferred_term="Klonopin",
            synonyms=("klonopin",),
            release_id="rxnav-api-2026-07-03",
            source_url=RXNORM_API_URL,
        ),
        TerminologyEntry(
            concept_type=ConceptType.MEDICATION,
            code_system="RxNorm",
            code="2599",
            preferred_term="clonidine",
            synonyms=("clonidine",),
            release_id="rxnav-api-2026-07-03",
            source_url=RXNORM_API_URL,
        ),
        TerminologyEntry(
            concept_type=ConceptType.MEDICATION,
            code_system="RxNorm",
            code="153010",
            preferred_term="Advil",
            synonyms=("advil",),
            release_id="rxnav-api-2026-07-03",
            source_url=RXNORM_API_URL,
        ),
        TerminologyEntry(
            concept_type=ConceptType.MEDICATION,
            code_system="RxNorm",
            code="1808",
            preferred_term="bumetanide",
            synonyms=("bumetanide",),
            release_id="rxnav-api-2026-07-03",
            source_url=RXNORM_API_URL,
        ),
        TerminologyEntry(
            concept_type=ConceptType.MEDICATION,
            code_system="RxNorm",
            code="435",
            preferred_term="albuterol",
            synonyms=("albuterol",),
            release_id="rxnav-api-2026-07-03",
            source_url=RXNORM_API_URL,
        ),
        TerminologyEntry(
            concept_type=ConceptType.MEDICATION,
            code_system="RxNorm",
            code="7213",
            preferred_term="ipratropium",
            synonyms=("ipratropium",),
            release_id="rxnav-api-2026-07-03",
            source_url=RXNORM_API_URL,
        ),
        TerminologyEntry(
            concept_type=ConceptType.MEDICATION,
            code_system="RxNorm",
            code="3251",
            preferred_term="desmopressin",
            synonyms=("desmopressin",),
            release_id="rxnav-api-2026-07-03",
            source_url=RXNORM_API_URL,
        ),
        TerminologyEntry(
            concept_type=ConceptType.MEDICATION,
            code_system="RxNorm",
            code="519",
            preferred_term="allopurinol",
            synonyms=("allopurinol",),
            release_id="rxnav-api-2026-07-03",
            source_url=RXNORM_API_URL,
        ),
        TerminologyEntry(
            concept_type=ConceptType.MEDICATION,
            code_system="RxNorm",
            code="196463",
            preferred_term="Prograf",
            synonyms=("prograf",),
            release_id="rxnav-api-2026-07-03",
            source_url=RXNORM_API_URL,
        ),
        TerminologyEntry(
            concept_type=ConceptType.MEDICATION,
            code_system="RxNorm",
            code="202866",
            preferred_term="Flagyl",
            synonyms=("flagyl",),
            release_id="rxnav-api-2026-07-03",
            source_url=RXNORM_API_URL,
        ),
        TerminologyEntry(
            concept_type=ConceptType.MEDICATION,
            code_system="RxNorm",
            code="5032",
            preferred_term="guaifenesin",
            synonyms=("guaifenesin",),
            release_id="rxnav-api-2026-07-03",
            source_url=RXNORM_API_URL,
        ),
        TerminologyEntry(
            concept_type=ConceptType.MEDICATION,
            code_system="RxNorm",
            code="4603",
            preferred_term="furosemide",
            synonyms=("furosemide",),
            release_id="rxnav-api-2026-07-03",
            source_url=RXNORM_API_URL,
        ),
        TerminologyEntry(
            concept_type=ConceptType.MEDICATION,
            code_system="RxNorm",
            code="83367",
            preferred_term="atorvastatin",
            synonyms=("atorvastatin",),
            release_id="rxnav-api-2026-07-03",
            source_url=RXNORM_API_URL,
        ),
        TerminologyEntry(
            concept_type=ConceptType.MEDICATION,
            code_system="RxNorm",
            code="32968",
            preferred_term="clopidogrel",
            synonyms=("clopidogrel",),
            release_id="rxnav-api-2026-07-03",
            source_url=RXNORM_API_URL,
        ),
        TerminologyEntry(
            concept_type=ConceptType.MEDICATION,
            code_system="RxNorm",
            code="356778",
            preferred_term="Ranexa",
            synonyms=("ranexa",),
            release_id="rxnav-api-2026-07-03",
            source_url=RXNORM_API_URL,
        ),
        TerminologyEntry(
            concept_type=ConceptType.MEDICATION,
            code_system="RxNorm",
            code="6057",
            preferred_term="isosorbide",
            synonyms=("isosorbide",),
            release_id="rxnav-api-2026-07-03",
            source_url=RXNORM_API_URL,
        ),
        TerminologyEntry(
            concept_type=ConceptType.MEDICATION,
            code_system="RxNorm",
            code="723",
            preferred_term="amoxicillin",
            synonyms=("amoxicillin",),
            release_id="rxnav-api-2026-07-03",
            source_url=RXNORM_API_URL,
        ),
        TerminologyEntry(
            concept_type=ConceptType.MEDICATION,
            code_system="RxNorm",
            code="6902",
            preferred_term="methylprednisolone",
            synonyms=("methylprednisolone",),
            release_id="rxnav-api-2026-07-03",
            source_url=RXNORM_API_URL,
        ),
        TerminologyEntry(
            concept_type=ConceptType.MEDICATION,
            code_system="RxNorm",
            code="202479",
            preferred_term="Ativan",
            synonyms=("ativan",),
            release_id="rxnav-api-2026-07-03",
            source_url=RXNORM_API_URL,
        ),
        TerminologyEntry(
            concept_type=ConceptType.MEDICATION,
            code_system="RxNorm",
            code="6470",
            preferred_term="lorazepam",
            synonyms=("lorazepam",),
            release_id="rxnav-api-2026-07-03",
            source_url=RXNORM_API_URL,
        ),
        TerminologyEntry(
            concept_type=ConceptType.MEDICATION,
            code_system="RxNorm",
            code="225036",
            preferred_term="Lovenox",
            synonyms=("lovenox",),
            release_id="rxnav-api-2026-07-03",
            source_url=RXNORM_API_URL,
        ),
        TerminologyEntry(
            concept_type=ConceptType.MEDICATION,
            code_system="RxNorm",
            code="29046",
            preferred_term="lisinopril",
            synonyms=("lisinopril",),
            release_id="demo-seed-see-rxnorm",
            source_url=RXNORM_URL,
        ),
        TerminologyEntry(
            concept_type=ConceptType.MEDICATION,
            code_system="RxNorm",
            code="3443",
            preferred_term="diltiazem",
            synonyms=("diltiazem",),
            release_id="rxnav-api-2026-07-03",
            source_url=RXNORM_API_URL,
        ),
        TerminologyEntry(
            concept_type=ConceptType.MEDICATION,
            code_system="RxNorm",
            code="1256",
            preferred_term="azathioprine",
            synonyms=("azathioprine",),
            release_id="rxnav-api-2026-07-05",
            source_url=RXNORM_API_URL,
        ),
        TerminologyEntry(
            concept_type=ConceptType.MEDICATION,
            code_system="RxNorm",
            code="301542",
            preferred_term="rosuvastatin",
            synonyms=("rosuvastatin",),
            release_id="rxnav-api-2026-07-05",
            source_url=RXNORM_API_URL,
        ),
        TerminologyEntry(
            concept_type=ConceptType.MEDICATION,
            code_system="RxNorm",
            code="320864",
            preferred_term="Crestor",
            synonyms=("crestor",),
            release_id="rxnav-api-2026-07-05",
            source_url=RXNORM_API_URL,
        ),
        TerminologyEntry(
            concept_type=ConceptType.MEDICATION,
            code_system="RxNorm",
            code="20352",
            preferred_term="carvedilol",
            synonyms=("carvedilol",),
            release_id="rxnav-api-2026-07-05",
            source_url=RXNORM_API_URL,
        ),
        TerminologyEntry(
            concept_type=ConceptType.MEDICATION,
            code_system="RxNorm",
            code="703",
            preferred_term="amiodarone",
            synonyms=("amiodarone",),
            release_id="rxnav-api-2026-07-05",
            source_url=RXNORM_API_URL,
        ),
        TerminologyEntry(
            concept_type=ConceptType.MEDICATION,
            code_system="RxNorm",
            code="214599",
            preferred_term="guaifenesin / pseudoephedrine",
            synonyms=("mucinex d",),
            release_id="rxnav-api-2026-07-05",
            source_url=RXNORM_API_URL,
        ),
        TerminologyEntry(
            concept_type=ConceptType.MEDICATION,
            code_system="RxNorm",
            code="750149",
            preferred_term="Z-PAK",
            synonyms=("z-pack", "z-pak"),
            release_id="rxnav-api-2026-07-05",
            source_url=RXNORM_API_URL,
        ),
        TerminologyEntry(
            concept_type=ConceptType.MEDICATION,
            code_system="RxNorm",
            code="613391",
            preferred_term="prasugrel",
            synonyms=("prasugrel",),
            release_id="rxnav-api-2026-07-05",
            source_url=RXNORM_API_URL,
        ),
        TerminologyEntry(
            concept_type=ConceptType.MEDICATION,
            code_system="RxNorm",
            code="20481",
            preferred_term="cefepime",
            synonyms=("cefepime", "cefepim"),
            release_id="rxnav-api-2026-07-05",
            source_url=RXNORM_API_URL,
        ),
        TerminologyEntry(
            concept_type=ConceptType.MEDICATION,
            code_system="RxNorm",
            code="221099",
            preferred_term="coagulation factor IX, human",
            synonyms=("yếu tố ix đậm đặc", "yếu tố IX đậm đặc"),
            release_id="rxnav-api-2026-07-05",
            source_url=RXNORM_API_URL,
        ),
        TerminologyEntry(
            concept_type=ConceptType.MEDICATION,
            code_system="RxNorm",
            code="82122",
            preferred_term="levofloxacin",
            synonyms=("levafloxacin",),
            release_id="rxnav-api-2026-07-05",
            source_url=RXNORM_API_URL,
        ),
        TerminologyEntry(
            concept_type=ConceptType.MEDICATION,
            code_system="RxNorm",
            code="2231",
            preferred_term="cephalexin",
            synonyms=("cephalexin",),
            release_id="rxnav-api-2026-07-05",
            source_url=RXNORM_API_URL,
        ),
        TerminologyEntry(
            concept_type=ConceptType.MEDICATION,
            code_system="RxNorm",
            code="1665021",
            preferred_term="cefTRIAXone 1 GM Injection",
            synonyms=("ceftriaxone 1 gram", "ceftriaxone 1 g"),
            release_id="rxnav-api-2026-07-05",
            source_url=RXNORM_API_URL,
        ),
        TerminologyEntry(
            concept_type=ConceptType.MEDICATION,
            code_system="RxNorm",
            code="196466",
            preferred_term="Taxol",
            synonyms=("taxol",),
            release_id="rxnav-api-2026-07-05",
            source_url=RXNORM_API_URL,
        ),
        TerminologyEntry(
            concept_type=ConceptType.MEDICATION,
            code_system="RxNorm",
            code="282357",
            preferred_term="fulvestrant",
            synonyms=("fulvestrant",),
            release_id="rxnav-api-2026-07-05",
            source_url=RXNORM_API_URL,
        ),
        TerminologyEntry(
            concept_type=ConceptType.MEDICATION,
            code_system="RxNorm",
            code="203546",
            preferred_term="Compazine",
            synonyms=("compazine",),
            release_id="rxnav-api-2026-07-05",
            source_url=RXNORM_API_URL,
        ),
        TerminologyEntry(
            concept_type=ConceptType.MEDICATION,
            code_system="RxNorm",
            code="215101",
            preferred_term="Aleve",
            synonyms=("aleve",),
            release_id="rxnav-api-2026-07-05",
            source_url=RXNORM_API_URL,
        ),
        TerminologyEntry(
            concept_type=ConceptType.MEDICATION,
            code_system="RxNorm",
            code="4850",
            preferred_term="dextrose",
            synonyms=("dextrose",),
            release_id="rxnav-api-2026-07-05",
            source_url=RXNORM_API_URL,
        ),
        TerminologyEntry(
            concept_type=ConceptType.MEDICATION,
            code_system="RxNorm",
            code="151539",
            preferred_term="Combivent",
            synonyms=("combivent", "combivent nebs"),
            release_id="rxnav-api-2026-07-05",
            source_url=RXNORM_API_URL,
        ),
        TerminologyEntry(
            concept_type=ConceptType.MEDICATION,
            code_system="RxNorm",
            code="6574",
            preferred_term="magnesium",
            synonyms=("magnesium",),
            release_id="rxnav-api-2026-07-05",
            source_url=RXNORM_API_URL,
        ),
        TerminologyEntry(
            concept_type=ConceptType.MEDICATION,
            code_system="RxNorm",
            code="227559",
            preferred_term="Levophed",
            synonyms=("levophed",),
            release_id="rxnav-api-2026-07-05",
            source_url=RXNORM_API_URL,
        ),
        TerminologyEntry(
            concept_type=ConceptType.MEDICATION,
            code_system="RxNorm",
            code="8782",
            preferred_term="propofol",
            synonyms=("propofol",),
            release_id="rxnav-api-2026-07-05",
            source_url=RXNORM_API_URL,
        ),
        TerminologyEntry(
            concept_type=ConceptType.MEDICATION,
            code_system="RxNorm",
            code="8153",
            preferred_term="phentolamine",
            synonyms=("phentolamine",),
            release_id="rxnav-api-2026-07-05",
            source_url=RXNORM_API_URL,
        ),
        TerminologyEntry(
            concept_type=ConceptType.MEDICATION,
            code_system="RxNorm",
            code="151392",
            preferred_term="Augmentin",
            synonyms=("augmentin",),
            release_id="rxnav-api-2026-07-05",
            source_url=RXNORM_API_URL,
        ),
    ]
