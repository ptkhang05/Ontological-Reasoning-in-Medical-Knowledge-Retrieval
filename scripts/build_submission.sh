#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT_DIR="${1:-${ROOT_DIR}/input/input}"
OUTPUT_ZIP="${2:-${ROOT_DIR}/output/output.zip}"

export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

if command -v python3 >/dev/null 2>&1; then
  PYTHON_CMD=(python3)
elif command -v python >/dev/null 2>&1; then
  PYTHON_CMD=(python)
elif command -v py >/dev/null 2>&1; then
  PYTHON_CMD=(py -3)
else
  echo "Could not find python3, python, or py on PATH." >&2
  exit 127
fi

if ! "${PYTHON_CMD[@]}" -c "import fastapi, pydantic, rapidfuzz" >/dev/null 2>&1; then
  echo "Missing Python dependencies. Run this first:" >&2
  echo "  ${PYTHON_CMD[*]} -m pip install -e \"${ROOT_DIR}[dev]\"" >&2
  exit 1
fi

"${PYTHON_CMD[@]}" -m clinical_nlp.cli.batch "${INPUT_DIR}" --output "${OUTPUT_ZIP}"
echo "Wrote BTC submission ZIP: ${OUTPUT_ZIP}"
