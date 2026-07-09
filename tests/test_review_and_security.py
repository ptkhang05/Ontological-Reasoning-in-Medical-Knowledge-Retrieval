import logging

import pytest
from fastapi.testclient import TestClient

from clinical_nlp.deidentification import Deidentifier
from clinical_nlp.pipeline import ClinicalPipeline
from clinical_nlp.schemas import AnalyzeOptions, AnalyzeRequest


class RecordingExternalExtractor:
    def __init__(self) -> None:
        self.seen_text: str | None = None

    def extract(self, text: str) -> list[dict[str, object]]:
        self.seen_text = text
        return []


class StaticExternalExtractor:
    def __init__(self, entities: list[dict[str, object]]) -> None:
        self.entities = entities

    def extract(self, text: str) -> list[dict[str, object]]:
        del text
        return self.entities


def test_review_flags_unmapped_and_safety_sensitive_concepts(client: TestClient) -> None:
    text = "Patient takes zorbium. Allergy to aspirin."

    response = client.post("/v1/analyze", json={"text": text})

    assert response.status_code == 200
    body = response.json()
    reasons = {flag["reason"] for flag in body["reviewFlags"]}
    assert "UNMAPPED_CODE" in reasons
    assert "SAFETY_SENSITIVE" in reasons

    zorbium = next(concept for concept in body["concepts"] if concept["text"] == "zorbium")
    assert zorbium["normalized"]["codeSystem"] is None
    assert zorbium["normalized"]["code"] is None


def test_external_inference_receives_deidentified_text_only() -> None:
    external = RecordingExternalExtractor()
    pipeline = ClinicalPipeline(external_extractor=external)
    request = AnalyzeRequest(
        text="John Doe, phone 555-123-4567, reports fever.",
        options=AnalyzeOptions(allow_external_inference=True),
    )

    response = pipeline.analyze(request)

    assert response.processing_metadata.external_inference_used is True
    assert external.seen_text is not None
    assert "John Doe" not in external.seen_text
    assert "555-123-4567" not in external.seen_text
    assert any(flag.reason == "EXTERNAL_INFERENCE_USED" for flag in response.review_flags)


def test_external_inference_redacts_names_without_trailing_comma() -> None:
    external = RecordingExternalExtractor()
    pipeline = ClinicalPipeline(external_extractor=external)
    request = AnalyzeRequest(
        text="John Doe reports fever. Patient Mary Smith denies cough.",
        options=AnalyzeOptions(allow_external_inference=True),
    )

    pipeline.analyze(request)

    assert external.seen_text is not None
    assert "John Doe" not in external.seen_text
    assert "Mary Smith" not in external.seen_text
    assert "fever" in external.seen_text
    assert "cough" in external.seen_text


def test_deidentifier_preserves_offsets_when_masking_names_without_commas() -> None:
    text = "John Doe reports fever."
    deidentified = Deidentifier().deidentify(text)

    assert len(deidentified.processed_text) == len(text)
    assert text.index("fever") == deidentified.processed_text.index("fever")
    assert "John Doe" not in deidentified.processed_text


def test_external_inference_entities_are_merged_after_offset_validation() -> None:
    text = "Bệnh nhân mô tả dị cảm ngón út khi nhập viện."
    extractor = StaticExternalExtractor(
        [
            {
                "text": "dị cảm ngón út",
                "position": [18, 31],
                "type": "TRIỆU_CHỨNG",
            }
        ]
    )
    pipeline = ClinicalPipeline(external_extractor=extractor)

    response = pipeline.analyze(
        AnalyzeRequest(
            text=text,
            options=AnalyzeOptions(allow_external_inference=True),
        )
    )

    concept = next(concept for concept in response.concepts if concept.text == "dị cảm ngón út")
    assert concept.start_offset == text.index("dị cảm ngón út")
    assert concept.end_offset == concept.start_offset + len("dị cảm ngón út")
    assert concept.source == "external_inference"


def test_external_inference_cannot_replace_rule_based_spans() -> None:
    text = "Bệnh nhân sốt và khó thở khi nhập viện."
    extractor = StaticExternalExtractor(
        [
            {
                "text": "sốt và khó thở",
                "position": [10, 24],
                "type": "TRIỆU_CHỨNG",
            }
        ]
    )
    pipeline = ClinicalPipeline(external_extractor=extractor)

    response = pipeline.analyze(
        AnalyzeRequest(
            text=text,
            options=AnalyzeOptions(allow_external_inference=True),
        )
    )

    concepts = {(concept.text, concept.source) for concept in response.concepts}
    assert ("sốt", "rule_symptom") in concepts
    assert ("khó thở", "rule_symptom") in concepts
    assert all(concept.text != "sốt và khó thở" for concept in response.concepts)


def test_external_inference_skips_section_headings() -> None:
    text = "Tiền sử bệnh hiện tại\nBệnh nhân mô tả dị cảm ngón út."
    extractor = StaticExternalExtractor(
        [
            {
                "text": "Tiền sử bệnh hiện tại",
                "position": [0, 21],
                "type": "TRIỆU_CHỨNG",
            },
            {
                "text": "dị cảm ngón út",
                "position": [39, 52],
                "type": "TRIỆU_CHỨNG",
            },
        ]
    )
    pipeline = ClinicalPipeline(external_extractor=extractor)

    response = pipeline.analyze(
        AnalyzeRequest(
            text=text,
            options=AnalyzeOptions(allow_external_inference=True),
        )
    )

    assert all(concept.text != "Tiền sử bệnh hiện tại" for concept in response.concepts)
    assert any(
        concept.text == "dị cảm ngón út" and concept.source == "external_inference"
        for concept in response.concepts
    )


def test_external_inference_repairs_position_by_exact_text_match() -> None:
    text = "Bệnh nhân mô tả dị cảm ngón út khi nhập viện."
    extractor = StaticExternalExtractor(
        [
            {
                "text": "dị cảm ngón út",
                "position": [0, 5],
                "type": "SYMPTOM",
            }
        ]
    )
    pipeline = ClinicalPipeline(external_extractor=extractor)

    response = pipeline.analyze(
        AnalyzeRequest(
            text=text,
            options=AnalyzeOptions(allow_external_inference=True),
        )
    )

    concept = next(concept for concept in response.concepts if concept.text == "dị cảm ngón út")
    assert concept.start_offset == text.index("dị cảm ngón út")


def test_external_inference_skips_unaligned_entities() -> None:
    extractor = StaticExternalExtractor(
        [
            {
                "text": "không tồn tại trong văn bản",
                "position": [0, 5],
                "type": "TRIỆU_CHỨNG",
            }
        ]
    )
    pipeline = ClinicalPipeline(external_extractor=extractor)

    response = pipeline.analyze(
        AnalyzeRequest(
            text="Bệnh nhân ổn định.",
            options=AnalyzeOptions(allow_external_inference=True),
        )
    )

    assert all(concept.text != "không tồn tại trong văn bản" for concept in response.concepts)


def test_api_does_not_log_raw_clinical_text(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    raw_text = "John Doe, phone 555-123-4567, reports fever."

    response = client.post("/v1/analyze", json={"text": raw_text})

    assert response.status_code == 200
    assert "John Doe" not in caplog.text
    assert "555-123-4567" not in caplog.text
