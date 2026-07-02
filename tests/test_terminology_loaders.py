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
