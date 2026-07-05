from __future__ import annotations

import argparse
import csv
import json
import re
import time
import urllib.parse
import urllib.request
from collections import deque
from collections.abc import Iterable
from pathlib import Path
from typing import Any

BASE_URL = "https://ccs.whiteneuron.com/api/ICD10_TT06"
SOURCE_URL = "https://icd.kcb.vn/icd-10-tt06/icd10-tt06"
RELEASE_ID = "TT06-2026"
CODE_SYSTEM = "ICD-10-TT06"
DEFAULT_OUTPUT = Path("data/terminologies/icd10_tt06.generated.csv")
ICD_CODE_PATTERN = re.compile(r"^[A-Z][0-9]{2}(?:\.[0-9A-Z]{1,4})?$")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a TerminologyStore CSV from the public ICD-10 TT06 API."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"CSV path to write. Defaults to {DEFAULT_OUTPUT}.",
    )
    parser.add_argument(
        "--language",
        default="vi",
        choices=("vi", "dual"),
        help="Language parameter used for tree traversal. Detail pages are not fetched.",
    )
    parser.add_argument(
        "--leaf-only",
        action="store_true",
        help=(
            "Only export leaf disease nodes. By default sections and type/category "
            "nodes are included too."
        ),
    )
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=0.05,
        help="Polite delay between API calls.",
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=0,
        help="Debug limit for crawled nodes. 0 means no limit.",
    )
    args = parser.parse_args()

    rows = crawl_icd10_tt06(
        language=args.language,
        leaf_only=args.leaf_only,
        delay_seconds=args.delay_seconds,
        max_nodes=args.max_nodes,
    )
    write_terminology_csv(args.output, rows)
    print(f"Wrote {len(rows)} ICD-10 TT06 rows to {args.output}")


def crawl_icd10_tt06(
    *,
    language: str = "vi",
    leaf_only: bool = False,
    delay_seconds: float = 0.05,
    max_nodes: int = 0,
) -> list[dict[str, str]]:
    queue: deque[dict[str, Any]] = deque(fetch_root(language))
    seen_nodes: set[tuple[str, str]] = set()
    rows_by_code_and_term: dict[tuple[str, str], dict[str, str]] = {}
    visited_count = 0

    while queue:
        node = queue.popleft()
        model = str(node.get("model", "")).strip()
        node_id = str(node.get("id", "")).strip()
        if not model or not node_id:
            continue

        key = (model, node_id)
        if key in seen_nodes:
            continue
        seen_nodes.add(key)
        visited_count += 1

        row = terminology_row_from_node(node, leaf_only=leaf_only)
        if row is not None:
            rows_by_code_and_term[(row["code"], row["preferred_term"])] = row

        if max_nodes and visited_count >= max_nodes:
            break

        if not bool(node.get("is_leaf", False)):
            children = fetch_children(model, node_id, language)
            queue.extend(children)
            if delay_seconds > 0:
                time.sleep(delay_seconds)

    return sorted(
        rows_by_code_and_term.values(),
        key=lambda row: (_code_sort_key(row["code"]), row["preferred_term"]),
    )


def fetch_root(language: str) -> list[dict[str, Any]]:
    return _fetch_data(f"{BASE_URL}/root?{urllib.parse.urlencode({'lang': language})}")


def fetch_children(model: str, node_id: str, language: str) -> list[dict[str, Any]]:
    query = urllib.parse.urlencode({"id": node_id, "lang": language})
    return _fetch_data(f"{BASE_URL}/childs/{urllib.parse.quote(model)}?{query}")


def _fetch_data(url: str) -> list[dict[str, Any]]:
    request = urllib.request.Request(url, headers={"User-Agent": "clinical-nlp-prototype/0.1"})
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict) or payload.get("status") != "success":
        return []
    data = payload.get("data")
    if not isinstance(data, list):
        return []
    return [item for item in data if isinstance(item, dict)]


def terminology_row_from_node(
    node: dict[str, Any],
    *,
    leaf_only: bool = False,
) -> dict[str, str] | None:
    model = str(node.get("model", "")).strip()
    if model == "chapter":
        return None
    if leaf_only and not bool(node.get("is_leaf", False)):
        return None

    data = node.get("data")
    if not isinstance(data, dict):
        return None
    code = str(data.get("code", "")).strip()
    preferred_term = clean_term(str(data.get("name", "")).strip())
    if not ICD_CODE_PATTERN.fullmatch(code) or not preferred_term:
        return None

    return {
        "code_system": CODE_SYSTEM,
        "code": code,
        "preferred_term": preferred_term,
        "synonyms": "|".join(generate_synonyms(preferred_term)),
        "release_id": RELEASE_ID,
        "source_url": SOURCE_URL,
    }


def generate_synonyms(preferred_term: str) -> list[str]:
    synonyms: list[str] = []
    for candidate in _synonym_candidates(preferred_term):
        candidate = clean_term(candidate)
        if candidate and candidate != preferred_term and candidate not in synonyms:
            synonyms.append(candidate)
    return synonyms


def _synonym_candidates(preferred_term: str) -> Iterable[str]:
    without_brackets = re.sub(r"\[[^\]]+\]", "", preferred_term)
    if without_brackets != preferred_term:
        yield without_brackets

    slash_normalized = preferred_term.replace("và/ hoặc", "hoặc").replace("và/hoặc", "hoặc")
    if slash_normalized != preferred_term:
        yield slash_normalized


def clean_term(value: str) -> str:
    return " ".join(value.replace("\xa0", " ").split())


def write_terminology_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "code_system",
        "code",
        "preferred_term",
        "synonyms",
        "release_id",
        "source_url",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _code_sort_key(code: str) -> tuple[str, int, str]:
    match = re.match(r"^([A-Z])([0-9]{2})(.*)$", code)
    if match is None:
        return code, -1, ""
    return match.group(1), int(match.group(2)), match.group(3)


if __name__ == "__main__":
    main()
