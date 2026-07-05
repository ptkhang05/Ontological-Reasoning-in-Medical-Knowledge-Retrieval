from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

BUILDER_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_icd10_tt06.py"
BUILDER_SPEC = importlib.util.spec_from_file_location("build_icd10_tt06", BUILDER_PATH)
assert BUILDER_SPEC is not None
assert BUILDER_SPEC.loader is not None
builder = importlib.util.module_from_spec(BUILDER_SPEC)
BUILDER_SPEC.loader.exec_module(builder)


def test_parallel_icd10_tt06_crawl_exports_sorted_rows(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    root = [
        {
            "model": "chapter",
            "id": "I",
            "is_leaf": False,
            "data": {"code": "I", "name": "Chương I"},
        }
    ]
    children_by_node = {
        ("chapter", "I"): [
            {
                "model": "section",
                "id": "A00",
                "is_leaf": False,
                "data": {"code": "A00", "name": "Bệnh tả"},
            }
        ],
        ("section", "A00"): [
            {
                "model": "type",
                "id": "A00.0",
                "is_leaf": True,
                "data": {"code": "A00.0", "name": "Bệnh tả do Vibrio cholerae 01"},
            }
        ],
    }

    def fake_fetch_root(language: str) -> list[dict[str, object]]:
        assert language == "vi"
        return root

    def fake_fetch_children(
        model: str, node_id: str, language: str
    ) -> list[dict[str, object]]:
        assert language == "vi"
        return children_by_node.get((model, node_id), [])

    monkeypatch.setattr(builder, "fetch_root", fake_fetch_root)
    monkeypatch.setattr(builder, "fetch_children", fake_fetch_children)

    rows = builder.crawl_icd10_tt06(delay_seconds=0, workers=2)

    assert [row["code"] for row in rows] == ["A00", "A00.0"]
    assert rows[0]["code_system"] == "ICD-10-TT06"
    assert rows[0]["release_id"] == "TT06-2026"


def test_icd10_tt06_crawl_rejects_invalid_worker_count() -> None:
    with pytest.raises(ValueError, match="workers must be >= 1"):
        builder.crawl_icd10_tt06(workers=0)
