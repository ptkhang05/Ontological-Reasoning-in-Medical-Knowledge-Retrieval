from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

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
        for entry in entries:
            for term in entry.search_terms():
                normalized = normalize_term(term)
                if normalized:
                    self._index[(entry.concept_type, normalized)] = entry

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
        return self._index.get((normalized_type, normalize_term(text)))

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


def demo_entries() -> list[TerminologyEntry]:
    return [
        TerminologyEntry(
            concept_type=ConceptType.DISEASE,
            code_system="ICD-10-CM",
            code="E11.9",
            preferred_term="Type 2 diabetes mellitus without complications",
            synonyms=("type 2 diabetes", "t2dm", "diabetes mellitus type 2"),
            release_id="demo-seed-see-cms",
            source_url=CMS_ICD10_URL,
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
            synonyms=("aspirin",),
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
            synonyms=("doxycycline",),
            release_id="rxnav-api-2026-07-02",
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
            synonyms=("nitroglycerin", "nitro"),
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
            synonyms=("cipro", "ciproflagyl"),
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
    ]
