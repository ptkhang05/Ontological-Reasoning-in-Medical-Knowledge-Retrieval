from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

from clinical_nlp.btc import btc_entities_to_jsonable
from clinical_nlp.pipeline import ClinicalPipeline
from clinical_nlp.schemas import AnalyzeRequest


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
    args = parser.parse_args()

    package_submission(input_dir=args.input_dir, output_zip=args.output, language=args.language)


def package_submission(input_dir: Path, output_zip: Path, language: str = "vi") -> Path:
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
    return output_zip


def numeric_stem(path: Path) -> int:
    try:
        return int(path.stem)
    except ValueError:
        return 10**9


if __name__ == "__main__":
    main()
