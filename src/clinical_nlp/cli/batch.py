from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Any, cast

from clinical_nlp.btc import btc_entities_to_jsonable
from clinical_nlp.pipeline import ClinicalPipeline
from clinical_nlp.schemas import AnalyzeOptions, AnalyzeRequest

BTC_ENTITY_KEYS = {"text", "position", "type", "assertions", "candidates"}
BTC_TYPES = {
    "TRIỆU_CHỨNG",
    "TÊN_XÉT_NGHIỆM",
    "KẾT_QUẢ_XÉT_NGHIỆM",
    "CHẨN_ĐOÁN",
    "THUỐC",
}
BTC_ASSERTIONS = {"isNegated", "isFamily", "isHistorical"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze a folder of clinical .txt files and package a Viettel phase ZIP."
    )
    parser.add_argument("input_dir", type=Path, help="Folder containing numbered .txt files.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/output.zip"),
        help="ZIP path to write. Defaults to output/output.zip.",
    )
    parser.add_argument("--language", default="vi", help="Language code passed to the analyzer.")
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip BTC ZIP schema and offset validation after packaging.",
    )
    parser.add_argument(
        "--allow-external-inference",
        action="store_true",
        help=(
            "Use a configured local external extractor, for example a self-hosted "
            "OpenAI-compatible LLM endpoint from CLINICAL_NLP_LOCAL_LLM_BASE_URL."
        ),
    )
    args = parser.parse_args()

    package_submission(
        input_dir=args.input_dir,
        output_zip=args.output,
        language=args.language,
        validate=not args.skip_validation,
        allow_external_inference=args.allow_external_inference,
    )


def package_submission(
    input_dir: Path,
    output_zip: Path,
    language: str = "vi",
    validate: bool = True,
    allow_external_inference: bool = False,
) -> Path:
    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    pipeline = ClinicalPipeline()
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(input_dir.glob("*.txt"), key=numeric_stem):
            text = path.read_text(encoding="utf-8")
            request = AnalyzeRequest(
                document_id=path.stem,
                document_type="viettel-medical-2026-public",
                language=language,
                text=text,
                options=AnalyzeOptions(allow_external_inference=allow_external_inference),
            )
            response = pipeline.analyze(request)
            archive.writestr(
                f"output/{path.stem}.json",
                json.dumps(
                    btc_entities_to_jsonable(response, text),
                    ensure_ascii=False,
                    indent=2,
                ),
            )
    if validate:
        errors = validate_submission_zip(output_zip, input_dir)
        if errors:
            preview = "\n".join(errors[:20])
            raise SystemExit(f"Invalid BTC submission ZIP:\n{preview}")
    return output_zip


def numeric_stem(path: Path) -> int:
    try:
        return int(path.stem)
    except ValueError:
        return 10**9


def validate_submission_zip(output_zip: Path, input_dir: Path) -> list[str]:
    errors: list[str] = []
    input_paths = sorted(input_dir.glob("*.txt"), key=numeric_stem)
    expected_names = [f"output/{path.stem}.json" for path in input_paths]
    source_text_by_name = {
        f"output/{path.stem}.json": path.read_text(encoding="utf-8") for path in input_paths
    }

    if not output_zip.exists():
        return [f"ZIP does not exist: {output_zip}"]

    try:
        with zipfile.ZipFile(output_zip) as archive:
            actual_names = sorted(
                [name for name in archive.namelist() if name.endswith(".json")],
                key=_archive_numeric_name,
            )
            if actual_names != expected_names:
                errors.append(
                    "ZIP JSON files do not match input files: "
                    f"expected={expected_names}, actual={actual_names}"
                )

            for name in actual_names:
                errors.extend(_validate_archive_payload(archive, name, source_text_by_name))
    except zipfile.BadZipFile:
        return [f"Bad ZIP file: {output_zip}"]
    return errors


def _validate_archive_payload(
    archive: zipfile.ZipFile, name: str, source_text_by_name: dict[str, str]
) -> list[str]:
    errors: list[str] = []
    try:
        payload = json.loads(archive.read(name).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return [f"{name}: invalid JSON: {exc}"]

    if not isinstance(payload, list):
        return [f"{name}: payload must be a list"]

    source_text = source_text_by_name.get(name)
    if source_text is None:
        errors.append(f"{name}: no matching source .txt file")
        source_text = ""

    for index, entity in enumerate(payload):
        prefix = f"{name}[{index}]"
        if not isinstance(entity, dict):
            errors.append(f"{prefix}: entity must be an object")
            continue
        errors.extend(_validate_entity(prefix, entity, source_text))
    return errors


def _validate_entity(prefix: str, entity: dict[str, Any], source_text: str) -> list[str]:
    errors: list[str] = []
    if set(entity) != BTC_ENTITY_KEYS:
        errors.append(f"{prefix}: invalid keys {sorted(entity)}")

    text = entity.get("text")
    if not isinstance(text, str) or not text:
        errors.append(f"{prefix}: text must be a non-empty string")

    position = entity.get("position")
    if not _is_two_int_list(position):
        errors.append(f"{prefix}: position must be [start, end]")
    else:
        start, end = cast(list[int], position)
        if not 0 <= start < end <= len(source_text):
            errors.append(f"{prefix}: position out of bounds {position}")
        elif isinstance(text, str) and source_text[start:end] != text:
            errors.append(
                f"{prefix}: text does not match source text at {position}: "
                f"{source_text[start:end]!r} != {text!r}"
            )

    entity_type = entity.get("type")
    if entity_type not in BTC_TYPES:
        errors.append(f"{prefix}: invalid type {entity_type!r}")

    assertions = entity.get("assertions")
    if not isinstance(assertions, list):
        errors.append(f"{prefix}: assertions must be a list")
    else:
        for assertion in assertions:
            if assertion not in BTC_ASSERTIONS:
                errors.append(f"{prefix}: invalid assertion {assertion!r}")

    candidates = entity.get("candidates")
    if not isinstance(candidates, list):
        errors.append(f"{prefix}: candidates must be a list")
    elif not all(isinstance(candidate, str) for candidate in candidates):
        errors.append(f"{prefix}: candidates must contain only strings")
    return errors


def _is_two_int_list(value: object) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and all(isinstance(item, int) for item in value)
    )


def _archive_numeric_name(name: str) -> tuple[int, str]:
    try:
        return int(Path(name).stem), name
    except ValueError:
        return 10**9, name


if __name__ == "__main__":
    main()
