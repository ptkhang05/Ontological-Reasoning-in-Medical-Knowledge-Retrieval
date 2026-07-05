from clinical_nlp.external import parse_llm_entities


def test_parse_llm_entities_accepts_json_array_in_code_fence() -> None:
    content = """```json
[
  {"text": "ho", "position": [0, 2], "type": "TRIỆU_CHỨNG"}
]
```"""

    entities = parse_llm_entities(content)

    assert entities == [{"text": "ho", "position": [0, 2], "type": "TRIỆU_CHỨNG"}]


def test_parse_llm_entities_accepts_entities_wrapper() -> None:
    content = '{"entities": [{"text": "sốt", "position": [4, 7], "type": "SYMPTOM"}]}'

    entities = parse_llm_entities(content)

    assert entities == [{"text": "sốt", "position": [4, 7], "type": "SYMPTOM"}]


def test_parse_llm_entities_rejects_non_json_text() -> None:
    assert parse_llm_entities("không có json hợp lệ") == []
