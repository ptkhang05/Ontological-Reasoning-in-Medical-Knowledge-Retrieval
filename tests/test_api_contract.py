from fastapi.testclient import TestClient


def test_root_redirects_to_interactive_docs(client: TestClient) -> None:
    response = client.get("/", follow_redirects=False)

    assert response.status_code == 307
    assert response.headers["location"] == "/docs"


def test_favicon_does_not_emit_not_found(client: TestClient) -> None:
    response = client.get("/favicon.ico")

    assert response.status_code == 204


def test_analyze_returns_stable_contract_and_original_offsets(client: TestClient) -> None:
    text = (
        "Patient denies fever. History of type 2 diabetes. "
        "Metformin 500 mg for type 2 diabetes. Glucose 250 mg/dL."
    )

    response = client.post(
        "/v1/analyze",
        json={
            "documentId": "note-1",
            "documentType": "discharge_summary",
            "language": "en",
            "text": text,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {
        "documentId",
        "concepts",
        "relations",
        "reviewFlags",
        "warnings",
        "processingMetadata",
    }
    assert body["documentId"] == "note-1"
    assert body["processingMetadata"]["externalInferenceUsed"] is False
    assert "ICD-10-CM" in body["processingMetadata"]["terminologyReleases"]
    assert "RxNorm" in body["processingMetadata"]["terminologyReleases"]

    for concept in body["concepts"]:
        assert text[concept["startOffset"] : concept["endOffset"]] == concept["text"]
        assert concept["confidence"] >= 0
        assert concept["confidence"] <= 1

    disease = next(
        concept for concept in body["concepts"] if concept["text"].lower() == "type 2 diabetes"
    )
    assert disease["conceptType"] == "DISEASE"
    assert disease["normalized"]["codeSystem"] == "ICD-10-CM"
    assert disease["normalized"]["code"] == "E11.9"

    medication = next(concept for concept in body["concepts"] if concept["text"] == "Metformin")
    assert medication["conceptType"] == "MEDICATION"
    assert medication["normalized"]["codeSystem"] == "RxNorm"
    assert medication["normalized"]["code"] == "6809"


def test_analyze_rejects_empty_text(client: TestClient) -> None:
    response = client.post("/v1/analyze", json={"text": "   "})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_analyze_rejects_oversized_text(client: TestClient) -> None:
    response = client.post("/v1/analyze", json={"text": "x" * 20001})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_analyze_rejects_malformed_json_without_echoing_payload(client: TestClient) -> None:
    response = client.post(
        "/v1/analyze",
        content='{"text": "John Doe has fever"',
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 422
    body = response.json()
    assert body["error"]["code"] == "VALIDATION_ERROR"
    assert "John Doe" not in str(body)
