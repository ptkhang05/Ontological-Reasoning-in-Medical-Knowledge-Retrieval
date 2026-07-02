import json
import zipfile

from fastapi.testclient import TestClient

from clinical_nlp.cli.batch import package_submission


def test_vietnamese_record_extracts_core_contest_concepts(client: TestClient) -> None:
    text = (
        "Bệnh nhân nữ không có khó thở. Nghi ngờ xơ gan do rượu. "
        "Được chỉ định điều trị aspirin 325mg x 1. "
        "Kết quả xét nghiệm: creatinine 1.2."
    )

    response = client.post("/v1/analyze", json={"text": text})

    assert response.status_code == 200
    body = response.json()
    concepts = body["concepts"]

    dyspnea = next(concept for concept in concepts if concept["text"].lower() == "khó thở")
    assert dyspnea["conceptType"] == "SYMPTOM"
    assert dyspnea["context"]["polarity"] == "NEGATED"

    cirrhosis = next(concept for concept in concepts if concept["text"].lower() == "xơ gan")
    assert cirrhosis["conceptType"] == "DISEASE"
    assert cirrhosis["context"]["polarity"] == "POSSIBLE"

    aspirin = next(
        concept for concept in concepts if concept["text"].lower() == "aspirin 325mg x 1"
    )
    assert aspirin["conceptType"] == "MEDICATION"
    assert aspirin["normalized"]["codeSystem"] == "RxNorm"
    assert aspirin["normalized"]["code"] == "317300"

    lab = next(concept for concept in concepts if concept["text"].lower() == "creatinine")
    assert lab["conceptType"] == "LAB_RESULT"

    sex = next(concept for concept in concepts if concept["text"].lower() == "bệnh nhân nữ")
    assert sex["conceptType"] == "PATIENT_INFO"

    relation_types = {relation["type"] for relation in body["relations"]}
    assert "HAS_VALUE" in relation_types


def test_batch_cli_packages_viettel_zip(tmp_path) -> None:  # type: ignore[no-untyped-def]
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "1.txt").write_text("Bệnh nhân ho. Điều trị aspirin 325mg.", encoding="utf-8")
    (input_dir / "2.txt").write_text("Nghi ngờ xơ gan. Creatinine 1.2.", encoding="utf-8")

    output_zip = package_submission(input_dir, tmp_path / "submission.zip")

    assert output_zip.exists()
    with zipfile.ZipFile(output_zip) as archive:
        assert archive.namelist() == ["output/1.json", "output/2.json"]
        first_payload = json.loads(archive.read("output/1.json").decode("utf-8"))
        second_payload = json.loads(archive.read("output/2.json").decode("utf-8"))

    assert isinstance(first_payload, list)
    assert isinstance(second_payload, list)
    assert all(
        set(entity) == {"text", "position", "type", "assertions", "candidates"}
        for entity in first_payload
    )
    aspirin = next(
        entity for entity in first_payload if entity["text"].lower() == "aspirin 325mg"
    )
    assert aspirin["type"] == "THUỐC"
    assert aspirin["candidates"] == ["317300"]
    assert "submission.json" not in archive.namelist()


def test_btc_serializer_expands_medication_text_and_historical_context(
    client: TestClient,
) -> None:
    text = "Thuốc trước khi nhập viện\n- metoprolol 25mg po bid\n- doxycycline cho viêm da."

    response = client.post("/v1/analyze/btc", json={"text": text})

    assert response.status_code == 200
    entities = response.json()

    metoprolol = next(
        entity for entity in entities if entity["text"].lower() == "metoprolol 25mg po bid"
    )
    assert metoprolol["type"] == "THUỐC"
    assert metoprolol["candidates"] == ["1370489"]
    assert metoprolol["position"] == [
        text.index("metoprolol"),
        text.index("metoprolol") + len("metoprolol 25mg po bid"),
    ]
    assert metoprolol["assertions"] == ["isHistorical"]

    doxycycline = next(entity for entity in entities if entity["text"].lower() == "doxycycline")
    assert doxycycline["type"] == "THUỐC"
    assert doxycycline["assertions"] == ["isHistorical"]


def test_btc_endpoint_returns_competition_schema(client: TestClient) -> None:
    text = (
        "Tiền sử xơ gan. Bệnh nhân không có khó thở. "
        "Điều trị aspirin 325mg. Creatinine 1.2."
    )

    response = client.post("/v1/analyze/btc", json={"text": text})

    assert response.status_code == 200
    entities = response.json()
    assert isinstance(entities, list)
    assert all(
        set(entity) == {"text", "position", "type", "assertions", "candidates"}
        for entity in entities
    )

    dyspnea = next(entity for entity in entities if entity["text"].lower() == "khó thở")
    assert dyspnea["type"] == "TRIỆU_CHỨNG"
    assert dyspnea["position"] == [text.index("khó thở"), text.index("khó thở") + len("khó thở")]
    assert dyspnea["assertions"] == ["isNegated"]
    assert dyspnea["candidates"] == []

    cirrhosis = next(entity for entity in entities if entity["text"].lower() == "xơ gan")
    assert cirrhosis["type"] == "CHẨN_ĐOÁN"
    assert cirrhosis["assertions"] == ["isHistorical"]

    aspirin = next(entity for entity in entities if entity["text"].lower() == "aspirin 325mg")
    assert aspirin["type"] == "THUỐC"
    assert aspirin["candidates"] == ["317300"]

    lab_name = next(entity for entity in entities if entity["text"].lower() == "creatinine")
    assert lab_name["type"] == "TÊN_XÉT_NGHIỆM"

    lab_value = next(entity for entity in entities if entity["text"] == "1.2")
    assert lab_value["type"] == "KẾT_QUẢ_XÉT_NGHIỆM"
    assert lab_value["position"] == [text.index("1.2"), text.index("1.2") + len("1.2")]
