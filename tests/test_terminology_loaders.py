from fastapi.testclient import TestClient

from clinical_nlp.api.app import create_app
from clinical_nlp.pipeline import ClinicalPipeline
from clinical_nlp.terminology import TerminologyStore


def test_terminology_store_loads_local_csv_files(tmp_path) -> None:  # type: ignore[no-untyped-def]
    (tmp_path / "icd10cm.csv").write_text(
        "code_system,code,preferred_term,synonyms,release_id,source_url\n"
        "ICD-10-CM,E11.9,Type 2 diabetes mellitus without complications,"
        "\"type 2 diabetes|t2dm\",FY2026,https://www.cms.gov/medicare/coding-billing/icd-10-codes\n",
        encoding="utf-8",
    )
    (tmp_path / "rxnorm.csv").write_text(
        "code_system,code,preferred_term,synonyms,release_id,source_url\n"
        "RxNorm,6809,metformin,metformin,DEMO,"
        "https://www.nlm.nih.gov/research/umls/rxnorm/overview.html\n",
        encoding="utf-8",
    )

    store = TerminologyStore.from_directory(tmp_path)

    disease = store.lookup("type 2 diabetes", "DISEASE")
    assert disease is not None
    assert disease.code_system == "ICD-10-CM"
    assert disease.code == "E11.9"

    medication = store.lookup("metformin", "MEDICATION")
    assert medication is not None
    assert medication.code_system == "RxNorm"
    assert medication.code == "6809"


def test_default_terminology_augments_seed_entries_with_local_csv(tmp_path) -> None:  # type: ignore[no-untyped-def]
    (tmp_path / "icd10_tt06.generated.csv").write_text(
        "code_system,code,preferred_term,synonyms,release_id,source_url\n"
        "ICD-10-TT06,A15,Bệnh lao,bệnh lao phổi|lao phổi,TT06-2026,"
        "https://icd.kcb.vn/icd-10-tt06/icd10-tt06\n",
        encoding="utf-8",
    )

    store = TerminologyStore.default(tmp_path)

    disease = store.lookup("bệnh lao phổi", "DISEASE")
    assert disease is not None
    assert disease.code_system == "ICD-10-TT06"
    assert disease.code == "A15"

    medication = store.lookup("metformin", "MEDICATION")
    assert medication is not None
    assert medication.code_system == "RxNorm"
    assert medication.code == "6809"


def test_default_terminology_prefers_local_csv_on_duplicate_terms(tmp_path) -> None:  # type: ignore[no-untyped-def]
    (tmp_path / "icd10_tt06.generated.csv").write_text(
        "code_system,code,preferred_term,synonyms,release_id,source_url\n"
        "ICD-10-TT06,E11.9,Đái tháo đường type 2,,TT06-2026,"
        "https://icd.kcb.vn/icd-10-tt06/icd10-tt06\n",
        encoding="utf-8",
    )

    store = TerminologyStore.default(tmp_path)

    disease = store.lookup("đái tháo đường type 2", "DISEASE")
    assert disease is not None
    assert disease.code_system == "ICD-10-TT06"
    assert disease.code == "E11.9"


def test_diabetes_type_qualifiers_win_over_local_generic_icd_term(tmp_path) -> None:  # type: ignore[no-untyped-def]
    (tmp_path / "icd10_tt06.generated.csv").write_text(
        "code_system,code,preferred_term,synonyms,release_id,source_url\n"
        "ICD-10-TT06,E10,Đái tháo đường,,TT06-2026,"
        "https://icd.kcb.vn/icd-10-tt06/icd10-tt06\n",
        encoding="utf-8",
    )
    pipeline = ClinicalPipeline(terminology=TerminologyStore.default(tmp_path))

    with TestClient(create_app(pipeline)) as client:
        response = client.post(
            "/v1/analyze/btc",
            json={"text": "Tiền sử Đái tháo đường type II. Tiểu đường loại 1."},
        )

    assert response.status_code == 200
    diagnosis_by_text = {
        entity["text"].lower(): entity
        for entity in response.json()
        if entity["type"] == "CHẨN_ĐOÁN"
    }
    assert diagnosis_by_text["đái tháo đường type ii"]["candidates"] == ["E11.9"]
    assert diagnosis_by_text["tiểu đường loại 1"]["candidates"] == ["E10"]
