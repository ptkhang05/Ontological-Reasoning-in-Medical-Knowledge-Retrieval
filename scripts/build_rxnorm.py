from __future__ import annotations

import argparse
import csv
import re
import zipfile
from collections import defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import NamedTuple

RXNORM_FILES_URL = "https://www.nlm.nih.gov/research/umls/rxnorm/docs/rxnormfiles.html"
CODE_SYSTEM = "RxNorm"
DEFAULT_OUTPUT = Path("data/terminologies/rxnorm.generated.csv")
DEFAULT_TTYS = (
    "SCD",
    "SBD",
    "GPCK",
    "BPCK",
    "SCDC",
    "SCDF",
    "SBDF",
    "IN",
    "PIN",
    "MIN",
    "BN",
)
RXNCONSO_FIELD_COUNT = 18
RXCUI_INDEX = 0
LAT_INDEX = 1
ISPREF_INDEX = 6
SAB_INDEX = 11
TTY_INDEX = 12
STR_INDEX = 14
SUPPRESS_INDEX = 16
TTY_PRIORITY = {tty: index for index, tty in enumerate(DEFAULT_TTYS)}


class RxNormTerm(NamedTuple):
    rxcui: str
    tty: str
    value: str
    is_preferred: bool


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a TerminologyStore CSV from a licensed local RxNorm release. "
            "Input may be RxNorm_full_*.zip, an extracted release directory, or RXNCONSO.RRF."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to RxNorm_full_*.zip, extracted RxNorm directory, or RXNCONSO.RRF.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"CSV path to write. Defaults to {DEFAULT_OUTPUT}.",
    )
    parser.add_argument(
        "--release-id",
        default=None,
        help="Release identifier to write into CSV. Defaults to RxNorm-<date> from input name.",
    )
    parser.add_argument(
        "--tty",
        nargs="*",
        default=list(DEFAULT_TTYS),
        help=(
            "RxNorm term types to export. Defaults to common clinical drug, ingredient, "
            "brand, dose-form, and pack TTYs."
        ),
    )
    parser.add_argument(
        "--min-term-length",
        type=int,
        default=3,
        help="Skip terms shorter than this many characters. Defaults to 3.",
    )
    parser.add_argument(
        "--max-synonyms-per-code",
        type=int,
        default=100,
        help="Maximum synonyms kept per RxCUI after the preferred term. Defaults to 100.",
    )
    args = parser.parse_args()

    row_count = build_rxnorm_csv(
        args.input,
        args.output,
        release_id=args.release_id,
        included_ttys=_parse_ttys(args.tty),
        min_term_length=args.min_term_length,
        max_synonyms_per_code=args.max_synonyms_per_code,
    )
    print(f"Wrote {row_count} RxNorm rows to {args.output}")


def build_rxnorm_csv(
    input_path: Path,
    output_path: Path,
    *,
    release_id: str | None = None,
    included_ttys: Iterable[str] = DEFAULT_TTYS,
    min_term_length: int = 3,
    max_synonyms_per_code: int = 100,
) -> int:
    if min_term_length < 1:
        raise ValueError("min_term_length must be >= 1")
    if max_synonyms_per_code < 0:
        raise ValueError("max_synonyms_per_code must be >= 0")

    included_tty_set = {tty.strip().upper() for tty in included_ttys if tty.strip()}
    if not included_tty_set:
        raise ValueError("included_ttys must not be empty")

    terms_by_rxcui: dict[str, list[RxNormTerm]] = defaultdict(list)
    for line_number, line in enumerate(iter_rxnconso_lines(input_path), start=1):
        term = parse_rxnconso_line(
            line,
            line_number=line_number,
            included_ttys=included_tty_set,
            min_term_length=min_term_length,
        )
        if term is not None:
            terms_by_rxcui[term.rxcui].append(term)

    rows = [
        _terminology_row(
            rxcui,
            terms,
            release_id=release_id or infer_release_id(input_path),
            max_synonyms_per_code=max_synonyms_per_code,
        )
        for rxcui, terms in sorted(
            terms_by_rxcui.items(),
            key=lambda item: _rxcui_sort_key(item[0]),
        )
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                "code_system",
                "code",
                "preferred_term",
                "synonyms",
                "release_id",
                "source_url",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def iter_rxnconso_lines(input_path: Path) -> Iterator[str]:
    if input_path.is_file() and zipfile.is_zipfile(input_path):
        with zipfile.ZipFile(input_path) as archive:
            member = _rxnconso_member_name(archive)
            with archive.open(member) as stream:
                for raw_line in stream:
                    yield raw_line.decode("utf-8").rstrip("\r\n")
        return

    rxnconso_path = find_rxnconso_path(input_path)
    with rxnconso_path.open("r", encoding="utf-8", newline="") as file:
        for line in file:
            yield line.rstrip("\r\n")


def find_rxnconso_path(input_path: Path) -> Path:
    if input_path.is_file() and input_path.name.upper() == "RXNCONSO.RRF":
        return input_path
    if input_path.is_dir():
        matches = sorted(
            path for path in input_path.rglob("*") if path.name.upper() == "RXNCONSO.RRF"
        )
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Could not find RXNCONSO.RRF under {input_path}")


def parse_rxnconso_line(
    line: str,
    *,
    line_number: int,
    included_ttys: set[str],
    min_term_length: int,
) -> RxNormTerm | None:
    fields = line.split("|")
    if fields and fields[-1] == "":
        fields = fields[:-1]
    if len(fields) < RXNCONSO_FIELD_COUNT:
        raise ValueError(
            f"RXNCONSO.RRF line {line_number} has {len(fields)} fields; "
            f"expected at least {RXNCONSO_FIELD_COUNT}."
        )

    lat = fields[LAT_INDEX].strip().upper()
    sab = fields[SAB_INDEX].strip().upper()
    tty = fields[TTY_INDEX].strip().upper()
    suppress = fields[SUPPRESS_INDEX].strip().upper()
    value = _clean_term(fields[STR_INDEX])
    if (
        lat != "ENG"
        or sab != "RXNORM"
        or suppress != "N"
        or tty not in included_ttys
        or len(value) < min_term_length
    ):
        return None

    rxcui = fields[RXCUI_INDEX].strip()
    if not rxcui:
        return None
    return RxNormTerm(
        rxcui=rxcui,
        tty=tty,
        value=value,
        is_preferred=fields[ISPREF_INDEX].strip().upper() == "Y",
    )


def infer_release_id(input_path: Path) -> str:
    match = re.search(r"(\d{8})", input_path.name)
    if match is not None:
        return f"RxNorm-{match.group(1)}"
    return "RxNorm-local"


def _terminology_row(
    rxcui: str,
    terms: list[RxNormTerm],
    *,
    release_id: str,
    max_synonyms_per_code: int,
) -> dict[str, str]:
    sorted_terms = sorted(
        terms,
        key=lambda term: (
            TTY_PRIORITY.get(term.tty, len(TTY_PRIORITY)),
            not term.is_preferred,
            len(term.value),
            term.value.lower(),
        ),
    )
    unique_terms: list[str] = []
    seen: set[str] = set()
    for term in sorted_terms:
        key = _term_key(term.value)
        if key in seen:
            continue
        seen.add(key)
        unique_terms.append(term.value)

    preferred_term = unique_terms[0]
    synonyms = unique_terms[1 : max_synonyms_per_code + 1]
    return {
        "code_system": CODE_SYSTEM,
        "code": rxcui,
        "preferred_term": preferred_term,
        "synonyms": "|".join(synonyms),
        "release_id": release_id,
        "source_url": RXNORM_FILES_URL,
    }


def _rxnconso_member_name(archive: zipfile.ZipFile) -> str:
    matches = sorted(
        name
        for name in archive.namelist()
        if name.replace("\\", "/").upper().endswith("RXNCONSO.RRF")
    )
    if not matches:
        raise FileNotFoundError("Could not find RXNCONSO.RRF in the RxNorm zip file.")
    return matches[0]


def _parse_ttys(values: Iterable[str]) -> tuple[str, ...]:
    parsed: list[str] = []
    for value in values:
        parsed.extend(part.strip().upper() for part in value.split(",") if part.strip())
    return tuple(parsed)


def _clean_term(value: str) -> str:
    return " ".join(value.strip().split())


def _term_key(value: str) -> str:
    return _clean_term(value).lower()


def _rxcui_sort_key(value: str) -> tuple[int, int | str]:
    if value.isdigit():
        return (0, int(value))
    return (1, value)


if __name__ == "__main__":
    main()
