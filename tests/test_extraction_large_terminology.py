from __future__ import annotations

from clinical_nlp.extraction import RuleBasedExtractor
from clinical_nlp.schemas import ConceptType
from clinical_nlp.terminology import TerminologyEntry, TerminologyStore


def test_large_terminology_keeps_exact_matches_without_fuzzy_expansion() -> None:
    entries = [
        TerminologyEntry(
            concept_type=ConceptType.DISEASE,
            code_system="ICD-10-TT06",
            code=f"A{index:02d}",
            preferred_term=f"Bệnh thử nghiệm {index}",
            synonyms=(),
            release_id="TT06-2026",
            source_url="https://icd.kcb.vn/icd-10-tt06/icd10-tt06",
        )
        for index in range(2100)
    ]
    entries.append(
        TerminologyEntry(
            concept_type=ConceptType.DISEASE,
            code_system="ICD-10-TT06",
            code="B99",
            preferred_term="Bệnh đặc hiệu",
            synonyms=(),
            release_id="TT06-2026",
            source_url="https://icd.kcb.vn/icd-10-tt06/icd10-tt06",
        )
    )
    extractor = RuleBasedExtractor(TerminologyStore(entries))

    exact_matches = extractor.extract("Chẩn đoán: Bệnh đặc hiệu.")
    assert any(
        candidate.text == "Bệnh đặc hiệu"
        and candidate.concept_type == ConceptType.DISEASE
        and candidate.source == "terminology_match"
        for candidate in exact_matches
    )

    fuzzy_matches = extractor.extract("Chan doan: Benh dac hieu.")
    assert not any(candidate.source == "fuzzy_terminology_match" for candidate in fuzzy_matches)


def test_icd_symptom_terms_do_not_override_symptom_extraction() -> None:
    store = TerminologyStore(
        [
            TerminologyEntry(
                concept_type=ConceptType.DISEASE,
                code_system="ICD-10-TT06",
                code="R06.0",
                preferred_term="Khó thở",
                synonyms=("ho",),
                release_id="TT06-2026",
                source_url="https://icd.kcb.vn/icd-10-tt06/icd10-tt06",
            ),
            TerminologyEntry(
                concept_type=ConceptType.DISEASE,
                code_system="ICD-10-TT06",
                code="J18.9",
                preferred_term="Viêm phổi",
                synonyms=(),
                release_id="TT06-2026",
                source_url="https://icd.kcb.vn/icd-10-tt06/icd10-tt06",
            ),
        ]
    )
    extractor = RuleBasedExtractor(store)

    candidates = extractor.extract("Bệnh nhân khó thở và ho do viêm phổi.")

    assert any(
        candidate.text.lower() == "khó thở"
        and candidate.concept_type == ConceptType.SYMPTOM
        for candidate in candidates
    )
    assert any(
        candidate.text.lower() == "viêm phổi"
        and candidate.concept_type == ConceptType.DISEASE
        for candidate in candidates
    )
    assert not any(
        candidate.text.lower() in {"khó thở", "ho"}
        and candidate.concept_type == ConceptType.DISEASE
        for candidate in candidates
    )


def test_icd_social_external_and_symptom_terms_are_not_extracted_as_diseases() -> None:
    store = TerminologyStore(
        [
            TerminologyEntry(
                concept_type=ConceptType.DISEASE,
                code_system="ICD-10-TT06",
                code=code,
                preferred_term=term,
                synonyms=(),
                release_id="TT06-2026",
                source_url="https://icd.kcb.vn/icd-10-tt06/icd10-tt06",
            )
            for code, term in [
                ("Z72.1", "sử dụng rượu"),
                ("Z72.0", "sử dụng thuốc lá"),
                ("V01", "tai nạn"),
                ("U83.0", "kháng vancomycin"),
                ("R11", "buồn nôn hoặc nôn"),
                ("R19.4", "thay đổi thói quen đại tiện"),
            ]
        ]
    )
    extractor = RuleBasedExtractor(store)

    candidates = extractor.extract(
        "Có sử dụng rượu, sử dụng thuốc lá sau tai nạn. "
        "Ghi nhận kháng vancomycin, buồn nôn hoặc nôn và thay đổi thói quen đại tiện."
    )

    diagnosis_texts = {
        candidate.text.lower()
        for candidate in candidates
        if candidate.concept_type == ConceptType.DISEASE
    }
    assert diagnosis_texts.isdisjoint(
        {
            "sử dụng rượu",
            "sử dụng thuốc lá",
            "tai nạn",
            "kháng vancomycin",
            "buồn nôn hoặc nôn",
            "thay đổi thói quen đại tiện",
        }
    )
    symptom_texts = {
        candidate.text.lower()
        for candidate in candidates
        if candidate.concept_type == ConceptType.SYMPTOM
    }
    assert "buồn nôn hoặc nôn" in symptom_texts
    assert "thay đổi thói quen đại tiện" in symptom_texts


def test_rxnorm_lab_analytes_are_not_extracted_as_medications() -> None:
    store = TerminologyStore(
        [
            TerminologyEntry(
                concept_type=ConceptType.MEDICATION,
                code_system="RxNorm",
                code=code,
                preferred_term=term,
                synonyms=(),
                release_id="RxNorm-test",
                source_url="https://www.nlm.nih.gov/research/umls/rxnorm/docs/rxnormfiles.html",
            )
            for code, term in [
                ("426", "alanine"),
                ("42543", "aspartate"),
                ("1886", "caffeine"),
                ("214275", "creatinine"),
                ("4850", "glucose"),
                ("2364", "guaiac"),
                ("6406", "lipase"),
                ("10323", "talc"),
                ("7617", "octreotide"),
                ("202433", "Tylenol"),
            ]
        ]
    )
    extractor = RuleBasedExtractor(store)

    candidates = extractor.extract(
        "Kết quả xét nghiệm: glucose 537, cr (creatinine) 1.2. "
        "AST (aspartate aminotransferase) là 319, ALT (alanine aminotransferase) là 690, "
        "lipase là tăng. Phân dương tính guaiac. "
        "Uống cà phê có caffeine. Không thực hiện gây dính màng phổi bằng talc. "
        "Dùng octreotide và Tylenol khi đau."
    )

    medication_texts = {
        candidate.text.lower()
        for candidate in candidates
        if candidate.concept_type == ConceptType.MEDICATION
    }
    assert {"octreotide", "tylenol"}.issubset(medication_texts)
    assert medication_texts.isdisjoint(
        {
            "alanine",
            "aspartate",
            "caffeine",
            "creatinine",
            "glucose",
            "guaiac",
            "lipase",
            "talc",
        }
    )


def test_ho_history_abbreviation_is_not_extracted_as_cough() -> None:
    extractor = RuleBasedExtractor(TerminologyStore([]))

    history_candidates = extractor.extract(
        "Các bệnh lý mạn tính\n"
        "- ho đái tháo đường\n"
        "- ho Bệnh bạch cầu dòng tủy mãn tính\n"
        "- ho Rối loạn cảm xúc\n"
    )
    history_symptoms = {
        candidate.text.lower()
        for candidate in history_candidates
        if candidate.concept_type == ConceptType.SYMPTOM
    }
    assert "ho" not in history_symptoms

    current_candidates = extractor.extract("Triệu chứng hiện tại\n- ho\n- ho nhẹ")
    current_symptoms = [
        candidate.text.lower()
        for candidate in current_candidates
        if candidate.concept_type == ConceptType.SYMPTOM
    ]
    assert "ho" in current_symptoms
