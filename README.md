# Clinical Concept Normalization Prototype API

Prototype FastAPI service for extracting clinical concepts from free-text notes,
assigning context, and normalizing diseases to ICD-10-CM and medications to
RxNorm. This is not a diagnostic, treatment, or billing-authoritative system.

## Commands

```powershell
python -m pip install -e ".[dev]"
python -m uvicorn clinical_nlp.api.app:app --reload
python -m pytest
python -m ruff check .
python -m mypy
```

Terminology files can be placed under `data/terminologies`. Do not commit large
or licensed terminology dumps.

See `docs/API.md` for the request/response contract and `docs/TERMINOLOGY.md`
for terminology loader details.
