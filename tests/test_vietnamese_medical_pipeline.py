import json
import zipfile

from fastapi.testclient import TestClient

from clinical_nlp.cli.batch import package_submission, validate_submission_zip


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
    assert validate_submission_zip(output_zip, input_dir) == []


def test_batch_validator_reports_schema_and_offset_errors(tmp_path) -> None:  # type: ignore[no-untyped-def]
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "1.txt").write_text("Bệnh nhân ho.", encoding="utf-8")
    output_zip = tmp_path / "bad.zip"
    bad_payload = [
        {
            "text": "sai",
            "position": [0, 3],
            "type": "TRIỆU_CHỨNG",
            "assertions": [],
            "candidates": [],
        },
        {
            "text": "ho",
            "position": [9, 11],
            "type": "SAI_NHÃN",
            "assertions": ["isMaybe"],
            "candidates": "bad",
        },
    ]
    with zipfile.ZipFile(output_zip, "w") as archive:
        archive.writestr("output/1.json", json.dumps(bad_payload, ensure_ascii=False))

    errors = validate_submission_zip(output_zip, input_dir)

    assert any("does not match source text" in error for error in errors)
    assert any("invalid type" in error for error in errors)
    assert any("invalid assertion" in error for error in errors)
    assert any("candidates must be a list" in error for error in errors)


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
    assert hypercalcemia["candidates"] == ["E83.5"]

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
    assert hepatic_encephalopathy["candidates"] == ["K76.8"]

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


def test_common_public_chronic_diagnoses_get_icd_candidates(client: TestClient) -> None:
    text = (
        "Tiền sử đái tháo đường, rung nhĩ, bệnh thận mạn tính, suy tim, "
        "bệnh động mạch vành và bệnh phổi tắc nghẽn mạn tính. "
        "Chẩn đoán viêm phổi, thuyên tắc phổi, tăng kali máu và béo phì."
    )

    response = client.post("/v1/analyze/btc", json={"text": text})

    assert response.status_code == 200
    entities = response.json()

    expected = {
        "đái tháo đường": ["E11.9"],
        "rung nhĩ": ["I48.9"],
        "bệnh thận mạn tính": ["N18.9"],
        "suy tim": ["I50.9"],
        "bệnh động mạch vành": ["I25.1"],
        "bệnh phổi tắc nghẽn mạn tính": ["J44.9"],
        "viêm phổi": ["J18.9"],
        "thuyên tắc phổi": ["I26.9"],
        "tăng kali máu": ["E87.5"],
        "béo phì": ["E66.9"],
    }
    diagnosis_by_text = {
        entity["text"].lower(): entity
        for entity in entities
        if entity["type"] == "CHẨN_ĐOÁN"
    }
    for text_value, candidates in expected.items():
        entity = diagnosis_by_text[text_value]
        assert entity["candidates"] == candidates


def test_btc_candidates_use_admin_vietnamese_icd10_for_common_cm_specific_codes(
    client: TestClient,
) -> None:
    text = (
        "Chẩn đoán rung nhĩ, bệnh động mạch vành, ung thư vú, trầm cảm, "
        "thuyên tắc phổi, hen suyễn, viêm mô tế bào, "
        "bệnh bạch cầu dòng tủy mãn tính, loét thực quản, "
        "nhịp nhanh trên thất và gãy xương sườn trái."
    )

    response = client.post("/v1/analyze/btc", json={"text": text})

    assert response.status_code == 200
    diagnosis_by_text = {
        entity["text"].lower(): entity
        for entity in response.json()
        if entity["type"] == "CHẨN_ĐOÁN"
    }

    assert diagnosis_by_text["rung nhĩ"]["candidates"] == ["I48.9"]
    assert diagnosis_by_text["bệnh động mạch vành"]["candidates"] == ["I25.1"]
    assert diagnosis_by_text["ung thư vú"]["candidates"] == ["C50.9"]
    assert diagnosis_by_text["trầm cảm"]["candidates"] == ["F32.9"]
    assert diagnosis_by_text["thuyên tắc phổi"]["candidates"] == ["I26.9"]
    assert diagnosis_by_text["hen suyễn"]["candidates"] == ["J45.9"]
    assert diagnosis_by_text["viêm mô tế bào"]["candidates"] == ["L03.9"]
    assert diagnosis_by_text["bệnh bạch cầu dòng tủy mãn tính"]["candidates"] == [
        "C92.1"
    ]
    assert diagnosis_by_text["loét thực quản"]["candidates"] == ["K22.1"]
    assert diagnosis_by_text["nhịp nhanh trên thất"]["candidates"] == ["I47.1"]
    assert diagnosis_by_text["gãy xương sườn trái"]["candidates"] == ["S22.4"]


def test_btc_icd_candidates_follow_admin_vietnamese_icd10_judgement(
    client: TestClient,
) -> None:
    text = (
        "Chẩn đoán thuyên tắc phổi, ung thư phổi không tế bào nhỏ, "
        "di căn não vùng trán phải và gãy xương sườn trái."
    )

    response = client.post("/v1/analyze/btc", json={"text": text})

    assert response.status_code == 200
    diagnosis_by_text = {
        entity["text"].lower(): entity
        for entity in response.json()
        if entity["type"] == "CHẨN_ĐOÁN"
    }

    assert diagnosis_by_text["thuyên tắc phổi"]["candidates"] == ["I26.9"]
    assert diagnosis_by_text["ung thư phổi không tế bào nhỏ"]["candidates"] == ["C34.9"]
    assert diagnosis_by_text["di căn não vùng trán phải"]["candidates"] == ["C79.3"]
    assert diagnosis_by_text["gãy xương sườn trái"]["candidates"] == ["S22.4"]


def test_common_public_acute_diagnoses_get_icd_candidates(client: TestClient) -> None:
    text = (
        "Chẩn đoán suy thận cấp, nhiễm trùng huyết, viêm mô tế bào, "
        "ung thư đại tràng, suy hô hấp, hạ kali máu, ngưng thở khi ngủ "
        "và viêm tủy xương. Có tiền sử đột quỵ."
    )

    response = client.post("/v1/analyze/btc", json={"text": text})

    assert response.status_code == 200
    entities = response.json()

    expected = {
        "suy thận cấp": ["N17.9"],
        "nhiễm trùng huyết": ["A41.9"],
        "viêm mô tế bào": ["L03.9"],
        "ung thư đại tràng": ["C18.9"],
        "suy hô hấp": ["J96.9"],
        "hạ kali máu": ["E87.6"],
        "ngưng thở khi ngủ": ["G47.3"],
        "viêm tủy xương": ["M86.9"],
        "đột quỵ": ["I63.9"],
    }
    diagnosis_by_text = {
        entity["text"].lower(): entity
        for entity in entities
        if entity["type"] == "CHẨN_ĐOÁN"
    }
    for text_value, candidates in expected.items():
        entity = diagnosis_by_text[text_value]
        assert entity["candidates"] == candidates


def test_additional_public_diagnoses_get_icd_candidates(client: TestClient) -> None:
    text = (
        "Chẩn đoán viêm túi mật cấp, viêm dạ dày, viêm dạ dày ruột do virus, "
        "u ác của tuyến tiền liệt, đa u tủy xương, thiếu máu, xẹp phổi, "
        "tràn dịch màng phổi, nhiễm trùng đường tiết niệu và sỏi mật."
    )

    response = client.post("/v1/analyze/btc", json={"text": text})

    assert response.status_code == 200
    entities = response.json()

    expected = {
        "viêm túi mật cấp": ["K81.0"],
        "viêm dạ dày": ["K29.7"],
        "viêm dạ dày ruột do virus": ["A08.4"],
        "u ác của tuyến tiền liệt": ["C61"],
        "đa u tủy xương": ["C90.0"],
        "thiếu máu": ["D64.9"],
        "xẹp phổi": ["J98.1"],
        "tràn dịch màng phổi": ["J90"],
        "nhiễm trùng đường tiết niệu": ["N39.0"],
        "sỏi mật": ["K80.2"],
    }
    diagnosis_by_text = {
        entity["text"].lower(): entity
        for entity in entities
        if entity["type"] == "CHẨN_ĐOÁN"
    }
    for text_value, candidates in expected.items():
        entity = diagnosis_by_text[text_value]
        assert entity["candidates"] == candidates


def test_high_confidence_public_diagnosis_findings_get_icd_candidates(
    client: TestClient,
) -> None:
    text = (
        "Hình ảnh cho thấy hẹp van động mạch chủ nghiêm trọng, "
        "sỏi ống dẫn mật chung đoạn cuối, bệnh lý chất trắng và bệnh thận đa nang. "
        "Chẩn đoán nhiễm khuẩn huyết do tụ cầu vàng nhạy cảm methicillin."
    )

    response = client.post("/v1/analyze/btc", json={"text": text})

    assert response.status_code == 200
    diagnosis_by_text = {
        entity["text"].lower(): entity
        for entity in response.json()
        if entity["type"] == "CHẨN_ĐOÁN"
    }

    expected = {
        "hẹp van động mạch chủ nghiêm trọng": ["I35.0"],
        "sỏi ống dẫn mật chung đoạn cuối": ["K80.5"],
        "bệnh lý chất trắng": ["R90.8"],
        "bệnh thận đa nang": ["Q61.3"],
        "nhiễm khuẩn huyết do tụ cầu vàng nhạy cảm methicillin": [
            "A41.0",
        ],
    }
    for text_value, candidates in expected.items():
        entity = diagnosis_by_text[text_value]
        assert entity["candidates"] == candidates


def test_fuzzy_diagnosis_matching_handles_missing_or_wrong_diacritics(
    client: TestClient,
) -> None:
    text = (
        "Chẩn đoán hep van dong mach chu nghiem trong. "
        "Nghi ngờ nhiễm khuẩn đường tiết niẹu."
    )

    response = client.post("/v1/analyze/btc", json={"text": text})

    assert response.status_code == 200
    diagnosis_by_text = {
        entity["text"].lower(): entity
        for entity in response.json()
        if entity["type"] == "CHẨN_ĐOÁN"
    }

    assert diagnosis_by_text["hep van dong mach chu nghiem trong"]["candidates"] == [
        "I35.0"
    ]
    assert diagnosis_by_text["nhiễm khuẩn đường tiết niẹu"]["candidates"] == ["N39.0"]


def test_public_imaging_and_chronic_findings_get_icd_candidates(client: TestClient) -> None:
    text = (
        "Kết quả chẩn đoán: tim to, tràn dịch màng tim, khí phế thủng, thoát vị hoành, "
        "hẹp van động mạch chủ, hở van hai lá, chèn ép tim, xuất huyết dưới nhện, "
        "tụ máu dưới màng cứng, loét tá tràng, tắc nghẽn đường mật, nốt tuyến giáp, "
        "u xơ tử cung, bàng quang thần kinh, liệt hai chi dưới, rối loạn lo âu và cổ trướng."
    )

    response = client.post("/v1/analyze/btc", json={"text": text})

    assert response.status_code == 200
    diagnosis_by_text = {
        entity["text"].lower(): entity
        for entity in response.json()
        if entity["type"] == "CHẨN_ĐOÁN"
    }

    expected = {
        "tim to": ["I51.7"],
        "tràn dịch màng tim": ["I31.3"],
        "khí phế thủng": ["J43.9"],
        "thoát vị hoành": ["K44.9"],
        "hẹp van động mạch chủ": ["I35.0"],
        "hở van hai lá": ["I34.0"],
        "chèn ép tim": ["I31.4"],
        "xuất huyết dưới nhện": ["I60.9"],
        "tụ máu dưới màng cứng": ["I62.0"],
        "loét tá tràng": ["K26.9"],
        "tắc nghẽn đường mật": ["K83.1"],
        "nốt tuyến giáp": ["E04.1"],
        "u xơ tử cung": ["D25.9"],
        "bàng quang thần kinh": ["N31.9"],
        "liệt hai chi dưới": ["G82.2"],
        "rối loạn lo âu": ["F41.9"],
        "cổ trướng": ["R18.8"],
    }
    for text_value, candidates in expected.items():
        assert diagnosis_by_text[text_value]["candidates"] == candidates


def test_low_output_public_records_get_verified_terms(client: TestClient) -> None:
    text = (
        "Bệnh lý mãn tính: tách thành động mạch chủ biến chứng bởi liệt hai chân. "
        "Tiền sử bàn chân vẹo bẩm sinh. "
        "Kết quả khám: tràn dịch màng ngoài tim mức độ trung bình. "
        "Hình ảnh ghi nhận thoát vị cạnh thực quản khá lớn. "
        "Chụp mạch máu ghi nhận hẹp khoảng 65% của động mạch cảnh chung trái. "
        "Đã đặt stent graft cho phình động mạch chủ ngực – bụng. "
        "Bệnh nhân nhiều lần giảm cân rồi tăng cân trở lại."
    )

    response = client.post("/v1/analyze/btc", json={"text": text})

    assert response.status_code == 200
    entities = response.json()
    diagnosis_by_text = {
        entity["text"].lower(): entity
        for entity in entities
        if entity["type"] == "CHẨN_ĐOÁN"
    }
    symptom_texts = {
        entity["text"].lower()
        for entity in entities
        if entity["type"] == "TRIỆU_CHỨNG"
    }

    assert diagnosis_by_text["tách thành động mạch chủ"]["candidates"] == [
        "I71.0",
    ]
    assert diagnosis_by_text["liệt hai chân"]["candidates"] == ["G82.2"]
    assert diagnosis_by_text["bàn chân vẹo bẩm sinh"]["candidates"] == ["Q66.0"]
    assert diagnosis_by_text["tràn dịch màng ngoài tim"]["candidates"] == ["I31.3"]
    assert diagnosis_by_text["thoát vị cạnh thực quản"]["candidates"] == ["K44.9"]
    assert diagnosis_by_text["hẹp khoảng 65% của động mạch cảnh chung trái"][
        "candidates"
    ] == ["I65.2"]
    assert diagnosis_by_text["phình động mạch chủ ngực – bụng"]["candidates"] == [
        "I71.6"
    ]
    assert "giảm cân" in symptom_texts
    assert "tăng cân trở lại" in symptom_texts


def test_low_output_neuro_and_wound_records_get_terms(client: TestClient) -> None:
    text = (
        "Lý do nhập viện: bệnh rễ thần kinh tuỷ sống ở ngón tay cái. "
        "MRI cột sống cổ cho thấy hẹp lỗ liên hợp. "
        "Triệu chứng khi đến: vết thương thấu bụng giữa bên phải."
    )

    response = client.post("/v1/analyze/btc", json={"text": text})

    assert response.status_code == 200
    entities = response.json()
    diagnosis_by_text = {
        entity["text"].lower(): entity
        for entity in entities
        if entity["type"] == "CHẨN_ĐOÁN"
    }
    symptom_texts = {
        entity["text"].lower()
        for entity in entities
        if entity["type"] == "TRIỆU_CHỨNG"
    }

    assert diagnosis_by_text["bệnh rễ thần kinh tuỷ sống ở ngón tay cái"][
        "candidates"
    ] == ["M54.1"]
    assert diagnosis_by_text["hẹp lỗ liên hợp"]["candidates"] == ["M99.7"]
    assert "vết thương thấu bụng giữa bên phải" in symptom_texts


def test_low_output_oncology_psych_and_vascular_records_get_terms(
    client: TestClient,
) -> None:
    text = (
        "Các bệnh lý mãn tính: Giai đoạn IV Ung thư phổi không tế bào nhỏ; "
        "Di căn não vùng trán phải; Sa van hai lá; tâm thần phân liệt. "
        "Triệu chứng hiện tại: đau bẹn trái. "
        "Không có cảm giác tê, cảm giác châm chích hoặc cảm giác nặng ở chân trái. "
        "Dấu hiệu lâm sàng: Bầm máu vùng bẹn trái. "
        "Bệnh mạn tính: bệnh mạch máu ngoại biên và "
        "Ung thư biểu mô tế bào vảy xâm nhập của dương vật."
    )

    response = client.post("/v1/analyze/btc", json={"text": text})

    assert response.status_code == 200
    entities = response.json()
    diagnosis_by_text = {
        entity["text"].lower(): entity
        for entity in entities
        if entity["type"] == "CHẨN_ĐOÁN"
    }
    symptoms_by_text = {
        entity["text"].lower(): entity
        for entity in entities
        if entity["type"] == "TRIỆU_CHỨNG"
    }

    assert diagnosis_by_text["ung thư phổi không tế bào nhỏ"]["candidates"] == [
        "C34.9",
    ]
    assert diagnosis_by_text["di căn não vùng trán phải"]["candidates"] == ["C79.3"]
    assert diagnosis_by_text["sa van hai lá"]["candidates"] == ["I34.1"]
    assert diagnosis_by_text["tâm thần phân liệt"]["candidates"] == ["F20.9"]
    assert diagnosis_by_text["bệnh mạch máu ngoại biên"]["candidates"] == ["I73.9"]
    assert diagnosis_by_text[
        "ung thư biểu mô tế bào vảy xâm nhập của dương vật"
    ]["candidates"] == ["C60.9"]
    assert "đau bẹn trái" in symptoms_by_text
    assert symptoms_by_text["cảm giác tê"]["assertions"] == ["isNegated"]
    assert symptoms_by_text["cảm giác châm chích"]["assertions"] == ["isNegated"]
    assert symptoms_by_text["cảm giác nặng ở chân trái"]["assertions"] == [
        "isNegated"
    ]
    assert "bầm máu vùng bẹn trái" in symptoms_by_text


def test_public_ocr_spacing_and_long_symptom_spans(client: TestClient) -> None:
    text = (
        "Bệnh mạn tính: Ung thư biểu mô tế bào vảy xâm nhập của "
        "dương vậtbiệt hóa kém. "
        "Triệu chứng hiện tại: đau ngực trái cấp tính, "
        "đau sau xương ức lan ra sau lưng, đau ngực lan ra sau lưng. "
        "Lý do nhập viện: suy nhược toàn thân. Bệnh nhân khó thở liên tục và hồi hộp."
    )

    response = client.post("/v1/analyze/btc", json={"text": text})

    assert response.status_code == 200
    entities = response.json()
    diagnosis_by_text = {
        entity["text"].lower(): entity
        for entity in entities
        if entity["type"] == "CHẨN_ĐOÁN"
    }
    symptom_texts = {
        entity["text"].lower()
        for entity in entities
        if entity["type"] == "TRIỆU_CHỨNG"
    }

    assert diagnosis_by_text[
        "ung thư biểu mô tế bào vảy xâm nhập của dương vậtbiệt hóa kém"
    ]["candidates"] == ["C60.9"]
    assert "đau ngực trái cấp tính" in symptom_texts
    assert "đau sau xương ức lan ra sau lưng" in symptom_texts
    assert "đau ngực lan ra sau lưng" in symptom_texts
    assert "suy nhược toàn thân" in symptom_texts
    assert "khó thở liên tục" in symptom_texts
    assert "hồi hộp" in symptom_texts


def test_public_single_file_terms_keep_long_spans(client: TestClient) -> None:
    text = (
        "Tiền sử bệnh: hội chứng turner. "
        "Triệu chứng hiện tại: triệu chứng trào ngược, "
        "loét mới ở ngón chân út bên phải, cảm giác có đờm ở cổ họng, "
        "khó thở tăng dần, phù ngoại vi tăng dần và ho tăng lên."
    )

    response = client.post("/v1/analyze/btc", json={"text": text})

    assert response.status_code == 200
    entities = response.json()
    diagnosis_by_text = {
        entity["text"].lower(): entity
        for entity in entities
        if entity["type"] == "CHẨN_ĐOÁN"
    }
    symptom_texts = {
        entity["text"].lower()
        for entity in entities
        if entity["type"] == "TRIỆU_CHỨNG"
    }

    assert diagnosis_by_text["hội chứng turner"]["candidates"] == ["Q96.9"]
    assert "triệu chứng trào ngược" in symptom_texts
    assert "loét mới ở ngón chân út bên phải" in symptom_texts
    assert "cảm giác có đờm ở cổ họng" in symptom_texts
    assert "khó thở tăng dần" in symptom_texts
    assert "phù ngoại vi tăng dần" in symptom_texts
    assert "ho tăng lên" in symptom_texts


def test_public_cardiac_long_diagnosis_spans_use_correct_icd(
    client: TestClient,
) -> None:
    text = (
        "Các bệnh lý mạn tính: bệnh tim mạch do xơ vữa động mạch. "
        "Chẩn đoán thiếu máu cơ tim vùng dưới và thành bên. "
        "Điện tâm đồ cho thấy rung nhĩ kèm đáp ứng thất nhanh mới "
        "và nhịp tim chậm tương đối. Sau đó có nhịp tim chậm nặng. "
        "Ghi nhận rung nhĩ kèm nhịp nhanh trên thất. "
        "Phát hiện rung nhĩ điển hình kèm theo đáp ứng thất nhanh."
    )

    response = client.post("/v1/analyze/btc", json={"text": text})

    assert response.status_code == 200
    entities = response.json()
    diagnosis_by_text = {
        entity["text"].lower(): entity
        for entity in entities
        if entity["type"] == "CHẨN_ĐOÁN"
    }

    assert diagnosis_by_text["bệnh tim mạch do xơ vữa động mạch"][
        "candidates"
    ] == ["I25.1"]
    assert diagnosis_by_text["thiếu máu cơ tim"]["candidates"] == ["I25.9"]
    assert diagnosis_by_text["rung nhĩ kèm đáp ứng thất nhanh"]["candidates"] == [
        "I48.9"
    ]
    assert diagnosis_by_text["nhịp nhanh trên thất"]["candidates"] == ["I47.1"]
    assert diagnosis_by_text[
        "rung nhĩ điển hình kèm theo đáp ứng thất nhanh"
    ]["candidates"] == ["I48.9"]
    assert diagnosis_by_text["nhịp tim chậm tương đối"]["candidates"] == ["R00.1"]
    assert diagnosis_by_text["nhịp tim chậm nặng"]["candidates"] == ["R00.1"]
    assert "xơ vữa động mạch" not in diagnosis_by_text
    assert "thiếu máu" not in diagnosis_by_text


def test_public_lung_metastasis_gets_exact_diagnosis_candidate(
    client: TestClient,
) -> None:
    text = "Kết quả chẩn đoán hình ảnh: ung thư di căn theo đường bạch huyết ở hai phổi."

    response = client.post("/v1/analyze/btc", json={"text": text})

    assert response.status_code == 200
    diagnosis_by_text = {
        entity["text"].lower(): entity
        for entity in response.json()
        if entity["type"] == "CHẨN_ĐOÁN"
    }

    assert diagnosis_by_text["ung thư di căn theo đường bạch huyết ở hai phổi"][
        "candidates"
    ] == ["C78.0"]


def test_lab_values_keep_comparators_and_units_from_source_text(
    client: TestClient,
) -> None:
    text = (
        "Siêu âm Doppler ghi nhận tỷ số PSV/EDV > 7. "
        "Chụp cắt lớp vi tính (ct) cho thấy tắc hẹp 80% động mạch thận trái. "
        "Siêu âm cho thấy ống mật chủ 15 mm. "
        "ERCP cho thấy khối phình to 2 cm. "
        "Nội soi cho thấy nhiều loét tá tràng và hồi tràng 2 tuần sau điều trị."
    )

    response = client.post("/v1/analyze/btc", json={"text": text})

    assert response.status_code == 200
    lab_values = {
        entity["text"]
        for entity in response.json()
        if entity["type"] == "KẾT_QUẢ_XÉT_NGHIỆM"
    }

    assert {"> 7", "80%", "15 mm", "2 cm"}.issubset(lab_values)
    assert lab_values.isdisjoint({"7", "80", "15", "2"})


def test_lab_values_trim_surrounding_whitespace_from_source_span(
    client: TestClient,
) -> None:
    text = (
        "Cận lâm sàng: creatinine   1.2. "
        "Creatinine ổn định ở mức 1.4-1.6. "
        "Lactate 1.1-->0.8. "
        "Siêu âm:  không phát hiện huyết khối tĩnh mạch sâu \n"
    )

    response = client.post("/v1/analyze/btc", json={"text": text})

    assert response.status_code == 200
    lab_values = [
        entity for entity in response.json() if entity["type"] == "KẾT_QUẢ_XÉT_NGHIỆM"
    ]
    entity_by_text = {entity["text"]: entity for entity in lab_values}

    assert "1.2" in entity_by_text
    assert entity_by_text["1.2"]["position"] == [
        text.index("1.2"),
        text.index("1.2") + len("1.2"),
    ]
    assert "1.4-1.6" in entity_by_text
    assert "1.1-->0.8" in entity_by_text
    assert "1" not in entity_by_text
    textual_value = "không phát hiện huyết khối tĩnh mạch sâu"
    assert textual_value in entity_by_text
    assert entity_by_text[textual_value]["position"] == [
        text.index(textual_value),
        text.index(textual_value) + len(textual_value),
    ]


def test_history_section_marks_conditions_historical(client: TestClient) -> None:
    text = (
        "1. Tiền sử bệnh nội khoa\n"
        "- tăng huyết áp\n"
        "- đái tháo đường\n"
        "2. Bệnh sử hiện tại\n"
        "Bệnh nhân khó thở và đau ngực."
    )

    response = client.post("/v1/analyze/btc", json={"text": text})

    assert response.status_code == 200
    entities = response.json()
    entity_by_text = {entity["text"].lower(): entity for entity in entities}

    assert entity_by_text["tăng huyết áp"]["assertions"] == ["isHistorical"]
    assert entity_by_text["đái tháo đường"]["assertions"] == ["isHistorical"]
    assert entity_by_text["khó thở"]["assertions"] == []
    assert entity_by_text["đau ngực"]["assertions"] == []


def test_family_observer_sentences_do_not_mark_patient_symptoms_as_family(
    client: TestClient,
) -> None:
    text = (
        "Gia đình nhận thấy khó khăn khi cài cúc áo và kéo khóa quần. "
        "Mẹ có tiền sử tăng huyết áp."
    )

    response = client.post("/v1/analyze/btc", json={"text": text})

    assert response.status_code == 200
    entity_by_text = {entity["text"].lower(): entity for entity in response.json()}

    assert entity_by_text["khó khăn khi cài cúc áo"]["assertions"] == []
    assert "isFamily" in entity_by_text["tăng huyết áp"]["assertions"]


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
        "Nhập viện trước đây bắt đầu bằng ciproflagyl. "
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
    assert medication_by_text["cipro"]["candidates"] == ["203563"]
    assert medication_by_text["flagyl"]["candidates"] == ["202866"]

    assert "atenololtrong" not in medication_by_text
    assert "vancomycinvà" not in medication_by_text
    assert "doxycyclinebactrim" not in medication_by_text
    assert "ciproflagyl" not in medication_by_text
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
    assert medication_by_text["furosemide 40 mg đường uống"]["candidates"] == ["315971"]
    assert medication_by_text["nitro"]["candidates"] == ["4917"]
    assert medication_by_text["dilaudid 3mg"]["candidates"] == ["897751"]
    assert not any("không cải thiện" in text for text in medication_by_text)
    assert "nitro và dilaudid 3mg" not in medication_by_text


def test_oxygen_saturation_context_is_not_medication(client: TestClient) -> None:
    text = "SpO2 99%, độ bão hòa oxy ổn định. Không dùng oxy."

    response = client.post("/v1/analyze/btc", json={"text": text})

    assert response.status_code == 200
    oxygen_medications = [
        entity
        for entity in response.json()
        if entity["type"] == "THUỐC" and entity["text"].lower() == "oxy"
    ]
    assert len(oxygen_medications) == 1
    assert oxygen_medications[0]["position"] == [
        text.rindex("oxy"),
        text.rindex("oxy") + len("oxy"),
    ]
    assert oxygen_medications[0]["assertions"] == ["isNegated"]


def test_prefixed_medication_dose_and_route_are_included(client: TestClient) -> None:
    text = (
        "Các thủ thuật đã thực hiện\n"
        "- Nhận 80mg lasix iv\n"
        "- Nhận 2 sl ntg\n"
        "- Nhận asa\n"
        "- Được cho 10mg iv diltiazem\n"
        "- Được cho metoprolol 5mg iv x2\n"
        "- đã dùng Laxis 20mg tiêm tĩnh mạch"
    )

    response = client.post("/v1/analyze/btc", json={"text": text})

    assert response.status_code == 200
    medication_by_text = {
        entity["text"].lower(): entity
        for entity in response.json()
        if entity["type"] == "THUỐC"
    }

    assert medication_by_text["80mg lasix iv"]["candidates"] == ["566621"]
    assert medication_by_text["2 sl ntg"]["candidates"] == ["4917"]
    assert medication_by_text["asa"]["candidates"] == ["1191"]
    assert medication_by_text["10mg iv diltiazem"]["candidates"] == ["1791228"]
    assert medication_by_text["metoprolol 5mg iv x2"]["candidates"] == ["335209"]
    assert medication_by_text["laxis 20mg tiêm tĩnh mạch"]["candidates"] == ["565450"]


def test_medication_expansion_does_not_include_dose_change_sentence(
    client: TestClient,
) -> None:
    text = "Prograf dose decreased from 5mg bid to 1mg bid. Tiếp tục corticoid liều cao."

    response = client.post("/v1/analyze/btc", json={"text": text})

    assert response.status_code == 200
    medication_by_text = {
        entity["text"].lower(): entity
        for entity in response.json()
        if entity["type"] == "THUỐC"
    }

    assert medication_by_text["prograf"]["candidates"] == ["196463"]
    assert "prograf dose decreased from 5mg bid to 1mg bid" not in medication_by_text
    assert medication_by_text["corticoid"]["candidates"] == ["4839"]


def test_public_medication_aliases_get_rxnorm_candidates(client: TestClient) -> None:
    text = (
        "Đang điều trị cotrimoxazol, doxycyclin, ertapenem, morphine, "
        "Percocet, Seroquel, iron, toradol và vicodin."
    )

    response = client.post("/v1/analyze/btc", json={"text": text})

    assert response.status_code == 200
    medication_by_text = {
        entity["text"].lower(): entity
        for entity in response.json()
        if entity["type"] == "THUỐC"
    }

    assert medication_by_text["cotrimoxazol"]["candidates"] == ["10831"]
    assert medication_by_text["doxycyclin"]["candidates"] == ["3640"]
    assert medication_by_text["ertapenem"]["candidates"] == ["325642"]
    assert medication_by_text["morphine"]["candidates"] == ["7052"]
    assert medication_by_text["percocet"]["candidates"] == ["42844"]
    assert medication_by_text["seroquel"]["candidates"] == ["83553"]
    assert medication_by_text["iron"]["candidates"] == ["90176"]
    assert medication_by_text["toradol"]["candidates"] == ["35827"]
    assert medication_by_text["vicodin"]["candidates"] == ["214182"]


def test_missed_public_medication_section_terms_get_2026_rxnorm_candidates(
    client: TestClient,
) -> None:
    text = (
        "Thuốc trước khi nhập viện\n"
        "- azathioprine\n"
        "- Rosuvastatin (Crestor): đã hết thuốc\n"
        "- Carvedilol: đã hết thuốc\n"
        "- amiodarone\n"
        "- mucinex d\n"
        "- z-pack\n"
        "- clopidogrel (prasugrel chuyển sang clopidogrel)\n"
        "Điều trị: Kháng sinh Cefepim và truyền dịch yếu tố IX đậm đặc."
    )

    response = client.post("/v1/analyze/btc", json={"text": text})

    assert response.status_code == 200
    medication_by_text = {
        entity["text"].lower(): entity
        for entity in response.json()
        if entity["type"] == "THUỐC"
    }

    expected = {
        "azathioprine": ["1256"],
        "rosuvastatin": ["301542"],
        "crestor": ["320864"],
        "carvedilol": ["20352"],
        "amiodarone": ["703"],
        "mucinex d": ["214599"],
        "z-pack": ["750149"],
        "prasugrel": ["613391"],
        "cefepim": ["20481"],
        "yếu tố ix đậm đặc": ["221099"],
    }
    for text_value, candidates in expected.items():
        assert medication_by_text[text_value]["candidates"] == candidates


def test_discontinued_medication_mentions_are_marked_historical(
    client: TestClient,
) -> None:
    text = (
        "Bệnh sử hiện tại: Các loại thuốc (isosorbide, crestor, carvedilol) "
        "đã hết khi đi khám PCP. Hết isosorbide, crestor, carvedilol khoảng "
        "3 tuần trước."
    )

    response = client.post("/v1/analyze/btc", json={"text": text})

    assert response.status_code == 200
    medication_entities = [
        entity for entity in response.json() if entity["type"] == "THUỐC"
    ]
    assert medication_entities
    assert all(entity["assertions"] == ["isHistorical"] for entity in medication_entities)


def test_bicarbonate_replacement_is_medication_only_in_treatment_context(
    client: TestClient,
) -> None:
    text = (
        "Thuốc trước khi nhập viện: thay thế bicarbonate. "
        "Kết quả xét nghiệm: hco3- (bicarbonate) 20."
    )

    response = client.post("/v1/analyze/btc", json={"text": text})

    assert response.status_code == 200
    bicarbonate_medications = [
        entity
        for entity in response.json()
        if entity["type"] == "THUỐC" and entity["text"].lower() == "bicarbonate"
    ]

    assert bicarbonate_medications == [
        {
            "text": "bicarbonate",
            "position": [
                text.index("bicarbonate"),
                text.index("bicarbonate") + len("bicarbonate"),
            ],
            "type": "THUỐC",
            "assertions": ["isHistorical"],
            "candidates": ["36676"],
        }
    ]


def test_normal_saline_abbreviation_is_medication_only_after_infusion_cue(
    client: TestClient,
) -> None:
    text = "Điều trị: Truyền dịch : 4000 ml NS 0.9 %. Theo dõi chỉ số NS khác."

    response = client.post("/v1/analyze/btc", json={"text": text})

    assert response.status_code == 200
    saline_medications = [
        entity
        for entity in response.json()
        if entity["type"] == "THUỐC" and "ns 0.9" in entity["text"].lower()
    ]

    assert saline_medications == [
        {
            "text": "4000 ml NS 0.9 %",
            "position": [
                text.index("4000 ml NS 0.9 %"),
                text.index("4000 ml NS 0.9 %") + len("4000 ml NS 0.9 %"),
            ],
            "type": "THUỐC",
            "assertions": [],
            "candidates": ["313002"],
        }
    ]


def test_more_public_medication_mentions_get_verified_rxnorm_candidates(
    client: TestClient,
) -> None:
    text = (
        "Bệnh nhân được cho về với đơn thuốci levafloxacin và cephalexin. "
        "Ceftriaxone 1 gram dùng 1 liều. "
        "Đang dùng thuốc weekly taxol. "
        "Nhận 40meq po k. Nhận 40meq iv k. "
        "Kết quả xét nghiệm kali (K) 6.3."
    )

    response = client.post("/v1/analyze/btc", json={"text": text})

    assert response.status_code == 200
    medication_by_text = {
        entity["text"].lower(): entity
        for entity in response.json()
        if entity["type"] == "THUỐC"
    }

    expected = {
        "levafloxacin": ["82122"],
        "cephalexin": ["2231"],
        "ceftriaxone 1 gram": ["1665021"],
        "taxol": ["196466"],
        "40meq po k": ["2728723"],
        "40meq iv k": ["2728723"],
    }
    for text_value, candidates in expected.items():
        assert medication_by_text[text_value]["candidates"] == candidates

    assert "k" not in medication_by_text


def test_public_oncology_medications_and_stuck_aleve_get_rxnorm_candidates(
    client: TestClient,
) -> None:
    text = (
        "Các bệnh mãn tính: Ung thư vú di căn ER+/HER2− đang dùng thuốc "
        "weekly taxol và fulvestrant để điều trị. "
        "Điều trị tại bệnh viện: compazine và alevenhưng vẫn còn đau."
    )

    response = client.post("/v1/analyze/btc", json={"text": text})

    assert response.status_code == 200
    medication_by_text = {
        entity["text"].lower(): entity
        for entity in response.json()
        if entity["type"] == "THUỐC"
    }

    expected = {
        "taxol": ["196466"],
        "fulvestrant": ["282357"],
        "compazine": ["203546"],
        "aleve": ["215101"],
    }
    for text_value, candidates in expected.items():
        assert medication_by_text[text_value]["candidates"] == candidates

    assert "alevenhưng" not in medication_by_text


def test_public_treatment_medications_dextrose_combivent_and_iv_magnesium(
    client: TestClient,
) -> None:
    text = (
        "Các thủ thuật đã thực hiện: Được dùng insulin và dextrose cho tăng kali máu. "
        "Điều trị: combivent nebs x3 every 20 minutes và iv magnesium."
    )

    response = client.post("/v1/analyze/btc", json={"text": text})

    assert response.status_code == 200
    medication_by_text = {
        entity["text"].lower(): entity
        for entity in response.json()
        if entity["type"] == "THUỐC"
    }

    expected = {
        "insulin": ["253182"],
        "dextrose": ["4850"],
        "combivent nebs": ["151539"],
        "iv magnesium": ["6574"],
    }
    for text_value, candidates in expected.items():
        assert medication_by_text[text_value]["candidates"] == candidates


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


def test_public_imaging_and_qualitative_results_are_extracted(
    client: TestClient,
) -> None:
    text = (
        "Kết quả chẩn đoán hình ảnh: chụp x-quang ngực không phát hiện viêm phổi. "
        "Điện tâm đồ (ECG) bình thường. "
        "Chụp ct sọ não: âm tính. "
        "Siêu âm thận cho thấy chỉ số kháng trở bình thường. "
        "Công thức máu (CBC) tăng nhẹ lên 11.3."
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
        entity["text"].lower()
        for entity in entities
        if entity["type"] == "KẾT_QUẢ_XÉT_NGHIỆM"
    }

    assert {
        "chụp x-quang ngực",
        "điện tâm đồ (ecg)",
        "chụp ct sọ não",
        "siêu âm thận",
        "công thức máu (cbc)",
    }.issubset(lab_names)
    assert {
        "không phát hiện viêm phổi",
        "bình thường",
        "âm tính",
        "11.3",
    }.issubset(lab_values)


def test_public_procedure_labs_and_abnormal_results_are_extracted(
    client: TestClient,
) -> None:
    text = (
        "Monitor holter cho thấy nhịp xoang. Sinh thiết và lấy mẫu bằng bàn chải "
        "cho thấy tế bào bất thường. Chọc hút bằng kim nhỏ nốt tuyến giáp ghi nhận bất thường."
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
        entity["text"].lower()
        for entity in entities
        if entity["type"] == "KẾT_QUẢ_XÉT_NGHIỆM"
    }

    assert {
        "monitor holter",
        "sinh thiết",
        "lấy mẫu bằng bàn chải",
        "chọc hút bằng kim nhỏ",
    }.issubset(lab_names)
    assert {"nhịp xoang", "tế bào bất thường", "bất thường"}.issubset(lab_values)


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


def test_long_public_symptom_spans_are_preferred(client: TestClient) -> None:
    text = (
        "Bệnh nhân khó thở khi gắng sức, nôn ra máu, phù chi dưới, ho khan, "
        "sốt cao, uống kém, toàn trạng suy kiệt, mờ mắt và mất ngủ."
    )

    response = client.post("/v1/analyze/btc", json={"text": text})

    assert response.status_code == 200
    symptom_texts = {
        entity["text"].lower()
        for entity in response.json()
        if entity["type"] == "TRIỆU_CHỨNG"
    }

    assert {
        "khó thở khi gắng sức",
        "nôn ra máu",
        "phù chi dưới",
        "ho khan",
        "sốt cao",
        "uống kém",
        "toàn trạng suy kiệt",
        "mờ mắt",
        "mất ngủ",
    }.issubset(symptom_texts)
    assert "khó thở" not in symptom_texts
    assert "phù" not in symptom_texts


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


def test_ambiguous_weakness_terms_are_context_limited(client: TestClient) -> None:
    text = (
        "Các yếu tố nguy cơ liên quan: không rõ. "
        "Khó thở chủ yếu khi gắng sức. "
        "Lý do nhập viện: yếu"
    )

    response = client.post("/v1/analyze/btc", json={"text": text})

    assert response.status_code == 200
    symptom_entities = [
        entity
        for entity in response.json()
        if entity["type"] == "TRIỆU_CHỨNG" and entity["text"].lower() == "yếu"
    ]
    assert symptom_entities == [
        {
            "text": "yếu",
            "position": [text.rindex("yếu"), text.rindex("yếu") + len("yếu")],
            "type": "TRIỆU_CHỨNG",
            "assertions": [],
            "candidates": [],
        }
    ]


def test_low_recall_public_records_get_obstetric_and_urology_symptoms(
    client: TestClient,
) -> None:
    text = (
        "Lý do nhập viện: tiểu tiện không tự chủ và sa âm đạo\n"
        "Triệu chứng hiện tại\n"
        "- tiểu tiện không tự chủ\n"
        "- sa âm đạo\n"
        "- bàng quang căng,\n"
        "- cảm giác bí tiếu liên tục\n"
        "Bệnh sử: Thai 41 tuần. Hiện có cơn co tử cung, thai máy tốt, "
        "chưa ra huyết âm đạo, chưa vỡ ối/rỉ ối."
    )

    response = client.post("/v1/analyze/btc", json={"text": text})

    assert response.status_code == 200
    entities = response.json()
    entity_by_text = {
        (entity["text"].lower(), entity["position"][0]): entity for entity in entities
    }

    for term in (
        "tiểu tiện không tự chủ",
        "sa âm đạo",
        "bàng quang căng",
        "cảm giác bí tiếu",
        "cơn co tử cung",
        "ra huyết âm đạo",
        "vỡ ối",
        "rỉ ối",
    ):
        start = text.lower().index(term)
        entity = entity_by_text[(term, start)]
        assert entity["position"] == [start, start + len(term)]
        assert entity["type"] in {"TRIỆU_CHỨNG", "CHẨN_ĐOÁN"}

    assert entity_by_text[("ra huyết âm đạo", text.lower().index("ra huyết âm đạo"))][
        "assertions"
    ] == ["isNegated"]
    assert entity_by_text[("vỡ ối", text.lower().index("vỡ ối"))]["assertions"] == [
        "isNegated"
    ]
    assert entity_by_text[("rỉ ối", text.lower().index("rỉ ối"))]["assertions"] == [
        "isNegated"
    ]


def test_lab_names_do_not_split_leukemia_diagnoses(client: TestClient) -> None:
    text = "Tiền sử bệnh nội khoa: Bệnh bạch cầu dòng tủy mãn tính."

    response = client.post("/v1/analyze/btc", json={"text": text})

    assert response.status_code == 200
    entities = response.json()
    assert not any(
        entity["type"] == "TÊN_XÉT_NGHIỆM" and entity["text"].lower() == "bạch cầu"
        for entity in entities
    )
    leukemia = next(
        entity
        for entity in entities
        if entity["text"].lower() == "bệnh bạch cầu dòng tủy mãn tính"
    )
    assert leukemia["type"] == "CHẨN_ĐOÁN"
    assert leukemia["candidates"] == ["C92.1"]


def test_long_diagnosis_spans_are_trimmed_before_secondary_cues(
    client: TestClient,
) -> None:
    text = (
        "Bệnh nhân được chẩn đoán nhiễm khuẩn huyết do tụ cầu vàng nhạy cảm "
        "methicillin, nghi liên quan đến đường truyền tĩnh mạch trung tâm."
    )

    response = client.post("/v1/analyze/btc", json={"text": text})

    assert response.status_code == 200
    diagnosis_by_text = {
        entity["text"].lower(): entity
        for entity in response.json()
        if entity["type"] == "CHẨN_ĐOÁN"
    }
    diagnosis = diagnosis_by_text["nhiễm khuẩn huyết do tụ cầu vàng nhạy cảm methicillin"]
    assert diagnosis["candidates"] == ["A41.0"]
    assert not any("nghi liên quan" in text for text in diagnosis_by_text)


def test_btc_serializer_can_return_multiple_candidates(client: TestClient) -> None:
    text = "Chẩn đoán bệnh trào ngược dạ dày - thực quản."

    response = client.post("/v1/analyze/btc", json={"text": text})

    assert response.status_code == 200
    diagnosis = next(
        entity
        for entity in response.json()
        if entity["text"].lower() == "bệnh trào ngược dạ dày - thực quản"
    )
    assert diagnosis["type"] == "CHẨN_ĐOÁN"
    assert diagnosis["candidates"] == ["K21.0", "K21.9"]


def test_medication_serializer_trims_stuck_route_and_next_drug(client: TestClient) -> None:
    text = "Đã dùng iv morphineiv morphine and toradol. Sau đó dùng morphineoral."

    response = client.post("/v1/analyze/btc", json={"text": text})

    assert response.status_code == 200
    medication_texts = {
        entity["text"].lower()
        for entity in response.json()
        if entity["type"] == "THUỐC"
    }

    assert "morphine" in medication_texts
    assert "toradol" in medication_texts
    assert "iv morphineiv morphine and toradol" not in medication_texts
    assert "morphineoral" not in medication_texts


def test_imaging_and_procedure_numbers_are_not_lab_values(client: TestClient) -> None:
    text = (
        "Chụp cộng hưởng từ (mri) cột sống cổ cho thấy hẹp ống sống C4-5. "
        "Nội soi cho thấy loét thực quản độ 2/6. "
        "Chụp cắt lớp vi tính (ct) lồng ngực Hình ảnh gãy 3 xương sườn "
        "9, 10 và 11. "
        "Nội soi mật tụy ngược dòng (ERCP) lấy thành công 02 viên sỏi."
    )

    response = client.post("/v1/analyze/btc", json={"text": text})

    assert response.status_code == 200
    lab_values = {
        entity["text"]
        for entity in response.json()
        if entity["type"] == "KẾT_QUẢ_XÉT_NGHIỆM"
    }
    assert lab_values.isdisjoint({"02", "2", "3", "4", "5", "6", "9", "10", "11"})


def test_long_eye_finding_span_is_preferred(client: TestClient) -> None:
    text = "Khám nhãn khoa không phát hiện phù gai thị. Sau đó ghi nhận Phù gai thị."

    response = client.post("/v1/analyze/btc", json={"text": text})

    assert response.status_code == 200
    entities = response.json()
    symptom_texts = [
        entity["text"].lower()
        for entity in entities
        if entity["type"] == "TRIỆU_CHỨNG"
    ]
    assert symptom_texts.count("phù gai thị") == 2
    assert "phù" not in symptom_texts
    negated = next(entity for entity in entities if entity["text"].lower() == "phù gai thị")
    assert negated["assertions"] == ["isNegated"]


def test_edema_spans_are_context_limited_and_longest_match(
    client: TestClient,
) -> None:
    text = (
        "Hướng điều trị phù hợp. "
        "Triệu chứng hiện tại: phù mắt cá chân, phù hai bên và phù chân trái. "
        "Chụp x-quang ngực không phát hiện viêm phổi hoặc phù phổi."
    )

    response = client.post("/v1/analyze/btc", json={"text": text})

    assert response.status_code == 200
    entities = response.json()
    symptom_texts = [
        entity["text"].lower()
        for entity in entities
        if entity["type"] == "TRIỆU_CHỨNG"
    ]
    diagnosis_by_text = {
        entity["text"].lower(): entity
        for entity in entities
        if entity["type"] == "CHẨN_ĐOÁN"
    }

    assert "phù" not in symptom_texts
    assert "phù mắt cá chân" in symptom_texts
    assert "phù hai bên" in symptom_texts
    assert "phù chân trái" in symptom_texts
    assert diagnosis_by_text["phù phổi"]["candidates"] == ["J81"]
    assert diagnosis_by_text["phù phổi"]["assertions"] == ["isNegated"]


def test_negation_does_not_cross_into_imaging_result(client: TestClient) -> None:
    text = (
        "Không có rối loạn thần kinh trước nhập viện. "
        "Kết quả chẩn đoán hình ảnh: chụp ct sọ não không thuốc cản quang "
        "Hình ảnh khối máu tụ dưới màng cứng bán cấp hai bên."
    )

    response = client.post("/v1/analyze/btc", json={"text": text})

    assert response.status_code == 200
    diagnosis = next(
        entity
        for entity in response.json()
        if entity["text"].lower() == "khối máu tụ dưới màng cứng"
    )
    assert "isNegated" not in diagnosis["assertions"]


def test_low_coded_public_diagnoses_get_verified_candidates(
    client: TestClient,
) -> None:
    text = (
        "Bệnh lý mãn tính: não úng thuỷ khác từ thời kỳ sơ sinh. "
        "MRCP cho thấy nhiều nang tụy nhỏ. "
        "Kết quả chẩn đoán hình ảnh: tắc nghẽn đường mật. "
        "Lý do nhập viện: chấn thương gãy xương sườn trái."
    )

    response = client.post("/v1/analyze/btc", json={"text": text})

    assert response.status_code == 200
    diagnosis_by_text = {
        entity["text"].lower(): entity
        for entity in response.json()
        if entity["type"] == "CHẨN_ĐOÁN"
    }

    assert diagnosis_by_text["não úng thuỷ khác"]["candidates"] == ["G91.8"]
    assert diagnosis_by_text["nang tụy"]["candidates"] == ["K86.2"]
    assert diagnosis_by_text["tắc nghẽn đường mật"]["candidates"] == ["K83.1"]
    assert diagnosis_by_text["gãy xương sườn trái"]["candidates"] == ["S22.4"]


def test_public_kidney_graves_gout_and_colon_cancer_terms_get_candidates(
    client: TestClient,
) -> None:
    text = (
        "Các bệnh lý mạn tính: bệnh gút không đặc hiệu, bệnh Graves. "
        "Tiền sử ung thư biểu mô tuyến đại tràng. "
        "Bệnh mạn tính: Suy thận mạn giai đoạn V do đái tháo đường."
    )

    response = client.post("/v1/analyze/btc", json={"text": text})

    assert response.status_code == 200
    diagnosis_by_text = {
        entity["text"].lower(): entity
        for entity in response.json()
        if entity["type"] == "CHẨN_ĐOÁN"
    }

    assert diagnosis_by_text["bệnh gút"]["candidates"] == ["M10.9"]
    assert diagnosis_by_text["bệnh graves"]["candidates"] == ["E05.0"]
    assert diagnosis_by_text["ung thư biểu mô tuyến đại tràng"]["candidates"] == [
        "C18.9"
    ]
    assert diagnosis_by_text["suy thận mạn giai đoạn v"]["candidates"] == ["N18.5"]
    assert "suy thận" not in diagnosis_by_text


def test_public_procedure_medications_get_verified_rxnorm_candidates(
    client: TestClient,
) -> None:
    text = (
        "Các thủ thuật đã thực hiện: levophed, propofol để an thần, "
        "Đã đặt phentolamine. Ra viện với đơn Augmentin đường uống trong 10 ngày."
    )

    response = client.post("/v1/analyze/btc", json={"text": text})

    assert response.status_code == 200
    medication_by_text = {
        entity["text"].lower(): entity
        for entity in response.json()
        if entity["type"] == "THUỐC"
    }

    expected = {
        "levophed": ["227559"],
        "propofol": ["8782"],
        "phentolamine": ["8153"],
        "augmentin đường uống": ["151392"],
    }
    for text_value, candidates in expected.items():
        assert medication_by_text[text_value]["candidates"] == candidates


def test_non_appetite_symptom_does_not_negate_later_symptoms(
    client: TestClient,
) -> None:
    text = (
        "Bệnh nhân kể cảm thấy khó chịu mệt mỏi nhiều, ăn không ngon miệng, "
        "ngứa da toàn thân nhiều, và mất trí nhớ chi tiết."
    )

    response = client.post("/v1/analyze/btc", json={"text": text})

    assert response.status_code == 200
    symptoms_by_text = {
        entity["text"].lower(): entity
        for entity in response.json()
        if entity["type"] == "TRIỆU_CHỨNG"
    }

    assert symptoms_by_text["ăn không ngon miệng"]["assertions"] == []
    assert symptoms_by_text["ngứa da toàn thân nhiều"]["assertions"] == []
    assert symptoms_by_text["mất trí nhớ chi tiết"]["assertions"] == []
    assert "ngứa da" not in symptoms_by_text
    assert "mất trí nhớ" not in symptoms_by_text


def test_public_missing_diagnosis_mentions_get_candidates(
    client: TestClient,
) -> None:
    text = (
        "Tiền sử bệnh: phình động mạch chủ nhỏ, u cơ trơn tử cung không đặc hiệu. "
        "Bệnh lý mãn tính: bệnh lý thần kinh ngoại biên. "
        "Bệnh sử tâm thần: Rối loạn lưỡng cực."
    )

    response = client.post("/v1/analyze/btc", json={"text": text})

    assert response.status_code == 200
    diagnosis_by_text = {
        entity["text"].lower(): entity
        for entity in response.json()
        if entity["type"] == "CHẨN_ĐOÁN"
    }

    assert diagnosis_by_text["phình động mạch chủ nhỏ"]["candidates"] == ["I71.9"]
    assert diagnosis_by_text["u cơ trơn tử cung"]["candidates"] == ["D25.9"]
    assert diagnosis_by_text["bệnh lý thần kinh ngoại biên"]["candidates"] == [
        "G62.9"
    ]
    assert diagnosis_by_text["rối loạn lưỡng cực"]["candidates"] == ["F31.9"]


def test_public_neuro_knee_and_blood_color_spans_are_not_fragmented(
    client: TestClient,
) -> None:
    text = (
        "Lý do nhập viện: Yếu nửa người trái. "
        "Diễn biến: yếu nửa người trái, yếu sức, khó khăn khi ra khỏi ghế tựa, "
        "cánh tay trái được cho là lơ lửng, khó khăn khi ước lượng vị trí ngồi "
        "xuống ghế ăn trưa, cảm giác bất thường ở bên phải đầu. "
        "Tiền sử: đau vùng xương bánh chè – đùi phải dữ dội. "
        "Hiện tại: Đau bánh chè đùi phải dữ dội. "
        "Lý do khác: đại tiện ra máu đỏ tươi gián đoạn. Màu sắc: đỏ tươi."
    )

    response = client.post("/v1/analyze/btc", json={"text": text})

    assert response.status_code == 200
    symptom_texts = {
        entity["text"].lower()
        for entity in response.json()
        if entity["type"] == "TRIỆU_CHỨNG"
    }

    expected = {
        "yếu nửa người trái",
        "yếu sức",
        "khó khăn khi ra khỏi ghế tựa",
        "cánh tay trái được cho là lơ lửng",
        "khó khăn khi ước lượng vị trí ngồi xuống ghế ăn trưa",
        "cảm giác bất thường ở bên phải đầu",
        "đau vùng xương bánh chè – đùi phải dữ dội",
        "đau bánh chè đùi phải dữ dội",
        "đại tiện ra máu đỏ tươi",
    }
    assert expected.issubset(symptom_texts)
    assert "yếu" not in symptom_texts
    assert "đỏ" not in symptom_texts


def test_public_long_diagnosis_variants_keep_full_text_with_same_codes(
    client: TestClient,
) -> None:
    text = (
        "Các bệnh lý mạn tính: bệnh phổi tắc nghẽn mạn tính, không xác định. "
        "Bệnh mạn tính: Nhiễm trùng đường tiết niệu kháng thuốc. "
        "Bệnh lý mãn tính: rối loạn lo âu, không biệt định nghiêm trọng. "
        "Bằng chứng Bệnh gan do rượu ở vị trí 35cm."
    )

    response = client.post("/v1/analyze/btc", json={"text": text})

    assert response.status_code == 200
    diagnosis_by_text = {
        entity["text"].lower(): entity
        for entity in response.json()
        if entity["type"] == "CHẨN_ĐOÁN"
    }

    expected = {
        "bệnh phổi tắc nghẽn mạn tính, không xác định": ["J44.9"],
        "nhiễm trùng đường tiết niệu kháng thuốc": ["N39.0"],
        "rối loạn lo âu, không biệt định nghiêm trọng": ["F41.9"],
        "bệnh gan do rượu": ["K70.9"],
    }
    for text_value, candidates in expected.items():
        assert diagnosis_by_text[text_value]["candidates"] == candidates
