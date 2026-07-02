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


def test_heuristic_extractor_filters_vietnamese_connector_words(client: TestClient) -> None:
    text = "Bệnh nhân dùng thuốc tại nhà. Điều trị bằng oxy và theo dõi sát."

    response = client.post("/v1/analyze/btc", json={"text": text})

    assert response.status_code == 200
    medication_texts = {
        entity["text"].lower()
        for entity in response.json()
        if entity["type"] == "THUỐC"
    }
    assert "thuốc" not in medication_texts
    assert "bằng" not in medication_texts
    assert "tại" not in medication_texts
    assert "theo" not in medication_texts


def test_common_vietnamese_diagnoses_get_icd_candidates(client: TestClient) -> None:
    text = "Tiền sử xơ gan và tăng calci máu. Chẩn đoán u ác trực tràng."

    response = client.post("/v1/analyze/btc", json={"text": text})

    assert response.status_code == 200
    entities = response.json()

    cirrhosis = next(entity for entity in entities if entity["text"].lower() == "xơ gan")
    assert cirrhosis["type"] == "CHẨN_ĐOÁN"
    assert cirrhosis["candidates"] == ["K74.6"]

    hypercalcemia = next(
        entity for entity in entities if entity["text"].lower() == "tăng calci máu"
    )
    assert hypercalcemia["type"] == "CHẨN_ĐOÁN"
    assert hypercalcemia["candidates"] == ["E83.52"]

    rectal_cancer = next(
        entity for entity in entities if entity["text"].lower() == "u ác trực tràng"
    )
    assert rectal_cancer["type"] == "CHẨN_ĐOÁN"
    assert rectal_cancer["candidates"] == ["C20"]


def test_public_cardioliver_diagnoses_get_icd_candidates(client: TestClient) -> None:
    text = (
        "Triệu chứng hiện tại: hội chứng não gan. "
        "Điện tâm đồ gợi ý nhồi máu cơ tim vùng dưới cũ."
    )

    response = client.post("/v1/analyze/btc", json={"text": text})

    assert response.status_code == 200
    entities = response.json()

    hepatic_encephalopathy = next(
        entity for entity in entities if entity["text"].lower() == "hội chứng não gan"
    )
    assert hepatic_encephalopathy["type"] == "CHẨN_ĐOÁN"
    assert hepatic_encephalopathy["candidates"] == ["K76.82"]

    old_mi = next(
        entity
        for entity in entities
        if entity["text"].lower() == "nhồi máu cơ tim vùng dưới cũ"
    )
    assert old_mi["type"] == "CHẨN_ĐOÁN"
    assert old_mi["candidates"] == ["I25.2"]


def test_unspecified_adenoma_gets_icd_candidate(client: TestClient) -> None:
    text = "Sinh thiết chỉ cho thấy một u tuyến."

    response = client.post("/v1/analyze/btc", json={"text": text})

    assert response.status_code == 200
    entities = response.json()
    adenoma = next(entity for entity in entities if entity["text"].lower() == "u tuyến")
    assert adenoma["type"] == "CHẨN_ĐOÁN"
    assert adenoma["candidates"] == ["D36.9"]


def test_common_public_medications_get_rxnorm_candidates(client: TestClient) -> None:
    text = (
        "Đang dùng Tylenol, vancomycin, omeprazole, heparin, Suboxone, "
        "Eliquis, Gleevec, torsemide, insulin glargine và azithromycin."
    )

    response = client.post("/v1/analyze/btc", json={"text": text})

    assert response.status_code == 200
    entities = response.json()

    expected = {
        "tylenol": "202433",
        "vancomycin": "11124",
        "omeprazole": "7646",
        "heparin": "5224",
        "suboxone": "352990",
        "eliquis": "1364436",
        "gleevec": "282386",
        "torsemide": "38413",
        "insulin glargine": "274783",
        "azithromycin": "18631",
    }
    for text_value, code in expected.items():
        entity = next(entity for entity in entities if entity["text"].lower() == text_value)
        assert entity["type"] == "THUỐC"
        assert entity["candidates"] == [code]


def test_compacted_public_medications_are_split_and_mapped(client: TestClient) -> None:
    text = (
        "Ở nhà bệnh nhân đã sử dụng atenololtrong ngày. "
        "Đã điều trị vancomycinvà ceftazidime trong 7 ngày. "
        "Tiếp tục sử dụng doxycyclinebactrim và sau đó dùng zosyn. "
        "Bệnh nhân cần dùng gậy hỗ trợ đi lại."
    )

    response = client.post("/v1/analyze/btc", json={"text": text})

    assert response.status_code == 200
    entities = response.json()
    medication_by_text = {
        entity["text"].lower(): entity
        for entity in entities
        if entity["type"] == "THUỐC"
    }

    assert medication_by_text["atenolol"]["candidates"] == ["1202"]
    assert medication_by_text["vancomycin"]["candidates"] == ["11124"]
    assert medication_by_text["ceftazidime"]["candidates"] == ["2191"]
    assert medication_by_text["doxycycline"]["candidates"] == ["3640"]
    assert medication_by_text["bactrim"]["candidates"] == ["151399"]
    assert medication_by_text["zosyn"]["candidates"] == ["74170"]

    assert "atenololtrong" not in medication_by_text
    assert "vancomycinvà" not in medication_by_text
    assert "doxycyclinebactrim" not in medication_by_text
    assert "gậy" not in medication_by_text


def test_medication_expansion_stops_at_conjunctions(client: TestClient) -> None:
    text = (
        "Bệnh nhân được dùng guaifenesin và furosemide 40 mg đường uống "
        "nhưng triệu chứng không cải thiện. Đau ngực giảm sau khi dùng nitro và dilaudid 3mg."
    )

    response = client.post("/v1/analyze/btc", json={"text": text})

    assert response.status_code == 200
    medication_by_text = {
        entity["text"].lower(): entity
        for entity in response.json()
        if entity["type"] == "THUỐC"
    }

    assert medication_by_text["guaifenesin"]["candidates"] == ["5032"]
    assert medication_by_text["furosemide 40 mg đường uống"]["candidates"] == ["4603"]
    assert medication_by_text["nitro"]["candidates"] == ["4917"]
    assert medication_by_text["dilaudid 3mg"]["candidates"] == ["224913"]
    assert not any("không cải thiện" in text for text in medication_by_text)
    assert "nitro và dilaudid 3mg" not in medication_by_text


def test_common_public_labs_extract_names_and_values(client: TestClient) -> None:
    text = (
        "Kết quả xét nghiệm: troponin là 0.03. INR là 1.4. "
        "Bạch cầu tăng là 39.2. ALT là 176 và AST là 287. "
        "Kali là 6.6 mmol/l. Hemoglobin 7.8. BNP 21,000."
    )

    response = client.post("/v1/analyze/btc", json={"text": text})

    assert response.status_code == 200
    entities = response.json()
    lab_names = {
        entity["text"].lower()
        for entity in entities
        if entity["type"] == "TÊN_XÉT_NGHIỆM"
    }
    lab_values = {
        entity["text"]
        for entity in entities
        if entity["type"] == "KẾT_QUẢ_XÉT_NGHIỆM"
    }

    assert {"troponin", "inr", "bạch cầu", "alt", "ast", "kali", "hemoglobin", "bnp"}.issubset(
        lab_names
    )
    assert {"0.03", "1.4", "39.2", "176", "287", "6.6 mmol/l", "7.8", "21,000"}.issubset(
        lab_values
    )


def test_common_public_symptoms_are_extracted_with_assertions(
    client: TestClient,
) -> None:
    text = (
        "Bệnh nhân không có ớn lạnh hoặc khò khè. "
        "Hiện có ho ra máu, đau lưng, đau cổ, chướng bụng, táo bón, "
        "ban đỏ, sưng, ngứa, ảo giác và mất trí nhớ."
    )

    response = client.post("/v1/analyze/btc", json={"text": text})

    assert response.status_code == 200
    symptom_by_text = {
        entity["text"].lower(): entity
        for entity in response.json()
        if entity["type"] == "TRIỆU_CHỨNG"
    }

    expected_symptoms = {
        "ớn lạnh",
        "khò khè",
        "ho ra máu",
        "đau lưng",
        "đau cổ",
        "chướng bụng",
        "táo bón",
        "ban đỏ",
        "sưng",
        "ngứa",
        "ảo giác",
        "mất trí nhớ",
    }
    assert expected_symptoms.issubset(symptom_by_text)
    assert symptom_by_text["ớn lạnh"]["assertions"] == ["isNegated"]
    assert symptom_by_text["khò khè"]["assertions"] == ["isNegated"]
    assert symptom_by_text["ho ra máu"]["assertions"] == []


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
