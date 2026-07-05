# Clinical Concept Normalization Prototype API

Prototype FastAPI service for Viettel AI Race 2026 Medical Track: extracting
clinical concepts from Vietnamese free-text records, assigning context, and
normalizing diseases to ICD-10-format codes and medications to RxNorm where source-backed
codes are available. This is not a diagnostic, treatment, or billing-authoritative
system.

## Commands

```powershell
python -m pip install -e ".[dev]"
python -m uvicorn clinical_nlp.api.app:app --reload
python -m pytest
python -m ruff check .
python -m mypy
python -m clinical_nlp.cli.batch input/input --output output/output.zip
```

Terminology files can be placed under `data/terminologies`. Do not commit large
or licensed terminology dumps. Local contest data under `input/` is ignored by
git.

See `docs/API.md` for the request/response contract,
`docs/TERMINOLOGY.md` for terminology loader details, and
`docs/VIETTEL_MEDICAL_2026.md` for contest-specific alignment notes.
