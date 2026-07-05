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
python scripts/build_icd10_tt06.py --output data/terminologies/icd10_tt06.generated.csv
python -m clinical_nlp.cli.batch input/input --output output/output.zip
./scripts/build_submission.sh input/input output/output.zip
```

Optional local LLM/entity proposal mode is available for self-hosted,
OpenAI-compatible servers such as LM Studio, Ollama, or vLLM:

```powershell
$env:CLINICAL_NLP_LOCAL_LLM_BASE_URL="http://127.0.0.1:1234/v1"
$env:CLINICAL_NLP_LOCAL_LLM_MODEL="qwen2.5-7b-instruct"
python -m clinical_nlp.cli.batch input/input --output output/output.zip --allow-external-inference
```

Use this only with a BTC-compliant self-hosted model. The adapter receives
de-identified text, and every proposed span is validated against the source
text before it can affect output.

Terminology files can be placed under `data/terminologies`. Do not commit large
or licensed terminology dumps. Local contest data under `input/` is ignored by
git.

See `docs/API.md` for the request/response contract,
`docs/TERMINOLOGY.md` for terminology loader details, and
`docs/VIETTEL_MEDICAL_2026.md` for contest-specific alignment notes.
