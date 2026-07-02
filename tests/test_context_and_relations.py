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


def test_family_subject_cues_do_not_match_inside_words(client: TestClient) -> None:
    text = "Bệnh nhân khó thở khi xem TV và đang dùng furosemide."

    response = client.post("/v1/analyze", json={"text": text})

    assert response.status_code == 200
    concepts = response.json()["concepts"]

    dyspnea = next(concept for concept in concepts if concept["text"].lower() == "khó thở")
    assert dyspnea["context"]["subject"] == "PATIENT"

    furosemide = next(
        concept for concept in concepts if concept["text"].lower() == "furosemide"
    )
    assert furosemide["context"]["subject"] == "PATIENT"


def test_vietnamese_negation_ignores_non_entity_negated_phrases(
    client: TestClient,
) -> None:
    text = (
        "Bắt đầu cảm thấy không khỏe kèm theo sốt, mệt mỏi. "
        "Sốt không đáp ứng với tylenol và advil. "
        "Tụt huyết áp không rõ nguyên nhân và mệt mỏi. "
        "Không sốt. Không dùng oxy."
    )

    response = client.post("/v1/analyze", json={"text": text})

    assert response.status_code == 200
    concepts = response.json()["concepts"]

    non_negated_texts = {"tylenol", "advil"}
    for concept in concepts:
        if concept["text"].lower() in non_negated_texts:
            assert concept["context"]["polarity"] == "PRESENT"

    first_fever = next(concept for concept in concepts if concept["text"].lower() == "sốt")
    assert first_fever["context"]["polarity"] == "PRESENT"

    fatigue_mentions = [
        concept for concept in concepts if concept["text"].lower() == "mệt mỏi"
    ]
    assert fatigue_mentions
    assert all(concept["context"]["polarity"] == "PRESENT" for concept in fatigue_mentions)

    last_fever = [
        concept for concept in concepts if concept["text"].lower() == "sốt"
    ][-1]
    assert last_fever["context"]["polarity"] == "NEGATED"

    oxygen = next(concept for concept in concepts if concept["text"].lower() == "oxy")
    assert oxygen["context"]["polarity"] == "NEGATED"


def test_vietnamese_non_tolerance_phrase_does_not_negate_entities(
    client: TestClient,
) -> None:
    text = (
        "Bệnh nhân không dung nạp amoxicillin do xuất hiện tiêu chảy "
        "nên được chuyển sang sử dụng azithromycin."
    )

    response = client.post("/v1/analyze", json={"text": text})

    assert response.status_code == 200
    concepts = response.json()["concepts"]
    for text_value in {"amoxicillin", "tiêu chảy", "azithromycin"}:
        concept = next(
            concept for concept in concepts if concept["text"].lower() == text_value
        )
        assert concept["context"]["polarity"] == "PRESENT"


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
