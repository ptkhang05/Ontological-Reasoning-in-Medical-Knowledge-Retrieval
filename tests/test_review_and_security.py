import logging

import pytest
from fastapi.testclient import TestClient

from clinical_nlp.pipeline import ClinicalPipeline
from clinical_nlp.schemas import AnalyzeRequest


class RecordingExternalExtractor:
    def __init__(self) -> None:
        self.seen_text: str | None = None

    def extract(self, text: str) -> list[dict[str, object]]:
        self.seen_text = text
        return []


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
        options={"allowExternalInference": True},
    )

    response = pipeline.analyze(request)

    assert response.processing_metadata.external_inference_used is True
    assert external.seen_text is not None
    assert "John Doe" not in external.seen_text
    assert "555-123-4567" not in external.seen_text
    assert any(flag.reason == "EXTERNAL_INFERENCE_USED" for flag in response.review_flags)


def test_api_does_not_log_raw_clinical_text(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    raw_text = "John Doe, phone 555-123-4567, reports fever."

    response = client.post("/v1/analyze", json={"text": raw_text})

    assert response.status_code == 200
    assert "John Doe" not in caplog.text
    assert "555-123-4567" not in caplog.text
