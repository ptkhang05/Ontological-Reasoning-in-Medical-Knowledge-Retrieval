from __future__ import annotations

import csv
import importlib.util
import zipfile
from pathlib import Path

from clinical_nlp.terminology import TerminologyStore

BUILDER_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_rxnorm.py"
BUILDER_SPEC = importlib.util.spec_from_file_location("build_rxnorm", BUILDER_PATH)
assert BUILDER_SPEC is not None
assert BUILDER_SPEC.loader is not None
builder = importlib.util.module_from_spec(BUILDER_SPEC)
BUILDER_SPEC.loader.exec_module(builder)


def test_rxnorm_builder_exports_active_english_rxnorm_terms(tmp_path: Path) -> None:
    rxnorm_dir = tmp_path / "rxnorm" / "rrf"
    rxnorm_dir.mkdir(parents=True)
    (rxnorm_dir / "RXNCONSO.RRF").write_text(
        _rxnconso_line("1191", "ENG", "RXNORM", "IN", "Y", "aspirin", "N")
        + _rxnconso_line(
            "243670", "ENG", "RXNORM", "SCD", "Y", "aspirin 81 MG Oral Tablet", "N"
        )
        + _rxnconso_line(
            "243670", "ENG", "RXNORM", "SCD", "N", "aspirin 81mg oral tablet", "N"
        )
        + _rxnconso_line("999", "SPA", "RXNORM", "IN", "Y", "aspirina", "N")
        + _rxnconso_line("888", "ENG", "RXNORM", "IN", "Y", "suppressed", "Y")
        + _rxnconso_line("777", "ENG", "SNOMEDCT_US", "IN", "Y", "not rxnorm", "N"),
        encoding="utf-8",
    )
    output_path = tmp_path / "rxnorm.generated.csv"

    row_count = builder.build_rxnorm_csv(
        tmp_path / "rxnorm",
        output_path,
        release_id="RxNorm-test",
        included_ttys=("IN", "SCD"),
    )

    assert row_count == 2
    rows = list(csv.DictReader(output_path.open(encoding="utf-8", newline="")))
    assert [row["code"] for row in rows] == ["1191", "243670"]
    assert rows[1] == {
        "code_system": "RxNorm",
        "code": "243670",
        "preferred_term": "aspirin 81 MG Oral Tablet",
        "synonyms": "aspirin 81mg oral tablet",
        "release_id": "RxNorm-test",
        "source_url": builder.RXNORM_FILES_URL,
    }

    store = TerminologyStore.from_directory(tmp_path)
    medication = store.lookup("aspirin 81mg oral tablet", "MEDICATION")
    assert medication is not None
    assert medication.code_system == "RxNorm"
    assert medication.code == "243670"


def test_rxnorm_builder_reads_nested_zip_and_infers_release_id(tmp_path: Path) -> None:
    zip_path = tmp_path / "RxNorm_full_07062026.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr(
            "rrf/RXNCONSO.RRF",
            _rxnconso_line("6809", "ENG", "RXNORM", "IN", "Y", "metformin", "N"),
        )
    output_path = tmp_path / "rxnorm.generated.csv"

    row_count = builder.build_rxnorm_csv(
        zip_path,
        output_path,
        included_ttys=("IN",),
    )

    assert row_count == 1
    rows = list(csv.DictReader(output_path.open(encoding="utf-8", newline="")))
    assert rows[0]["code"] == "6809"
    assert rows[0]["release_id"] == "RxNorm-07062026"


def _rxnconso_line(
    rxcui: str,
    lat: str,
    sab: str,
    tty: str,
    is_pref: str,
    term: str,
    suppress: str,
) -> str:
    fields = [""] * 18
    fields[0] = rxcui
    fields[1] = lat
    fields[2] = "P"
    fields[3] = f"L{rxcui}"
    fields[4] = "PF"
    fields[5] = f"S{rxcui}"
    fields[6] = is_pref
    fields[7] = f"A{rxcui}"
    fields[11] = sab
    fields[12] = tty
    fields[13] = rxcui
    fields[14] = term
    fields[16] = suppress
    return "|".join(fields) + "|\n"
