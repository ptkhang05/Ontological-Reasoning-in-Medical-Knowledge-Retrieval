from fastapi.testclient import TestClient


def test_context_detects_negation_family_subject_and_history(client: TestClient) -> None:
    text = "No fever. Mother had breast cancer. History of hypertension."

    response = client.post("/v1/analyze", json={"text": text})

    assert response.status_code == 200
    concepts = response.json()["concepts"]

    fever = next(concept for concept in concepts if concept["text"].lower() == "fever")
    assert fever["context"]["polarity"] == "NEGATED"
    assert fever["context"]["subject"] == "PATIENT"

    breast_cancer = next(
        concept for concept in concepts if concept["text"].lower() == "breast cancer"
    )
    assert breast_cancer["context"]["subject"] == "FAMILY"

    hypertension = next(
        concept for concept in concepts if concept["text"].lower() == "hypertension"
    )
    assert hypertension["context"]["temporality"] == "HISTORICAL"


def test_relations_capture_treatment_dosage_and_lab_value(client: TestClient) -> None:
    text = "Metformin 500 mg for type 2 diabetes. Glucose 250 mg/dL."

    response = client.post("/v1/analyze", json={"text": text})

    assert response.status_code == 200
    relations = response.json()["relations"]
    relation_types = {relation["type"] for relation in relations}
    assert {"TREATS", "HAS_DOSAGE", "HAS_VALUE"}.issubset(relation_types)

    treats = next(relation for relation in relations if relation["type"] == "TREATS")
    assert treats["sourceConceptId"] != treats["targetConceptId"]
    assert text[treats["evidenceStartOffset"] : treats["evidenceEndOffset"]]

    dosage = next(relation for relation in relations if relation["type"] == "HAS_DOSAGE")
    assert dosage["value"] == "500 mg"

    lab_value = next(relation for relation in relations if relation["type"] == "HAS_VALUE")
    assert lab_value["value"] == "250 mg/dL"
