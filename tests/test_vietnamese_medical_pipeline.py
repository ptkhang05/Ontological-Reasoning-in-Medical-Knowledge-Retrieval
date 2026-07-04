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
    assert hypercalcemia["candidates"] == ["E83.5", "E83.52"]

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
        "rung nhĩ": ["I48.9", "I48.91"],
        "bệnh thận mạn tính": ["N18.9"],
        "suy tim": ["I50.9"],
        "bệnh động mạch vành": ["I25.1", "I25.10"],
        "bệnh phổi tắc nghẽn mạn tính": ["J44.9"],
        "viêm phổi": ["J18.9"],
        "thuyên tắc phổi": ["I26.9", "I26.99"],
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


def test_btc_candidates_include_icd10_parent_for_common_cm_specific_codes(
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

    assert diagnosis_by_text["rung nhĩ"]["candidates"] == ["I48.9", "I48.91"]
    assert diagnosis_by_text["bệnh động mạch vành"]["candidates"] == ["I25.1", "I25.10"]
    assert diagnosis_by_text["ung thư vú"]["candidates"] == ["C50.9", "C50.919"]
    assert diagnosis_by_text["trầm cảm"]["candidates"] == ["F32.9", "F32.A"]
    assert diagnosis_by_text["thuyên tắc phổi"]["candidates"] == ["I26.9", "I26.99"]
    assert diagnosis_by_text["hen suyễn"]["candidates"] == ["J45.9", "J45.909"]
    assert diagnosis_by_text["viêm mô tế bào"]["candidates"] == ["L03.9", "L03.90"]
    assert diagnosis_by_text["bệnh bạch cầu dòng tủy mãn tính"]["candidates"] == [
        "C92.1",
        "C92.10",
    ]
    assert diagnosis_by_text["loét thực quản"]["candidates"] == ["K22.1", "K22.10"]
    assert diagnosis_by_text["nhịp nhanh trên thất"]["candidates"] == ["I47.1", "I47.10"]
    assert diagnosis_by_text["gãy xương sườn trái"]["candidates"] == ["S22.4", "S22.42XA"]


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
        "viêm mô tế bào": ["L03.9", "L03.90"],
        "ung thư đại tràng": ["C18.9"],
        "suy hô hấp": ["J96.9", "J96.90"],
        "hạ kali máu": ["E87.6"],
        "ngưng thở khi ngủ": ["G47.3", "G47.30"],
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
        "viêm dạ dày": ["K29.7", "K29.70"],
        "viêm dạ dày ruột do virus": ["A08.4"],
        "u ác của tuyến tiền liệt": ["C61"],
        "đa u tủy xương": ["C90.0", "C90.00"],
        "thiếu máu": ["D64.9"],
        "xẹp phổi": ["J98.1", "J98.11"],
        "tràn dịch màng phổi": ["J90"],
        "nhiễm trùng đường tiết niệu": ["N39.0"],
        "sỏi mật": ["K80.2", "K80.20"],
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
        "sỏi ống dẫn mật chung đoạn cuối": ["K80.5", "K80.50"],
        "bệnh lý chất trắng": ["R90.8", "R90.82"],
        "bệnh thận đa nang": ["Q61.3"],
        "nhiễm khuẩn huyết do tụ cầu vàng nhạy cảm methicillin": [
            "A41.0",
            "A41.01",
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
        "tràn dịch màng tim": ["I31.3", "I31.39"],
        "khí phế thủng": ["J43.9"],
        "thoát vị hoành": ["K44.9"],
        "hẹp van động mạch chủ": ["I35.0"],
        "hở van hai lá": ["I34.0"],
        "chèn ép tim": ["I31.4"],
        "xuất huyết dưới nhện": ["I60.9"],
        "tụ máu dưới màng cứng": ["I62.0", "I62.00"],
        "loét tá tràng": ["K26.9"],
        "tắc nghẽn đường mật": ["K83.1"],
        "nốt tuyến giáp": ["E04.1"],
        "u xơ tử cung": ["D25.9"],
        "bàng quang thần kinh": ["N31.9"],
        "liệt hai chi dưới": ["G82.2", "G82.20"],
        "rối loạn lo âu": ["F41.9"],
        "cổ trướng": ["R18.8"],
    }
    for text_value, candidates in expected.items():
        assert diagnosis_by_text[text_value]["candidates"] == candidates


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
    assert leukemia["candidates"] == ["C92.1", "C92.10"]


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
    assert diagnosis["candidates"] == ["A41.0", "A41.01"]
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
        "9, 10 và 11."
    )

    response = client.post("/v1/analyze/btc", json={"text": text})

    assert response.status_code == 200
    lab_values = {
        entity["text"]
        for entity in response.json()
        if entity["type"] == "KẾT_QUẢ_XÉT_NGHIỆM"
    }
    assert lab_values.isdisjoint({"2", "3", "4", "5", "6", "9", "10", "11"})


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
    assert diagnosis_by_text["gãy xương sườn trái"]["candidates"] == ["S22.4", "S22.42XA"]
