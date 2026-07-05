# Viettel Medical 2026 Alignment

Local requirement Markdown snapshots checked on 2026-07-06:

- `docs/btc/NLP_1.md`: contest overview, problem statement, entity schema,
  public data description, and timeline.
- `docs/btc/NLP_2.md`: phase 1 submission format, output example, scoring
  formulas, resource rules, and submission limits.

The Markdown snapshots preserve the same phase 1 output contract and scoring
formula currently implemented in this repo. No API/schema migration is required
from this update.

Public facts from the updated Markdown files:

- Contest name: `Bài 2 - Ontological Reasoning in Medical Knowledge Retrieval`.
- Phase 1: `Vòng 1 - Sơ loại`, running from 2026-07-02 to 2026-07-30.
- Phase 1 submission type: one ZIP file.
- Phase 1 submission limit: 5 submissions/day.
- The judge runs on GPU infrastructure, but contestants prepare their own
  compute resources.
- If using LLM/agent solutions, external APIs are not allowed; self-hosted
  model size is capped at 9B parameters.
- Before phase 1 ends, BTC may ask roughly the top 15 teams to submit source
  code, data used, model weights, and a README so BTC can rebuild and evaluate
  on private test data. This is explicitly meant to prevent hard-coded outputs.

Implementation choices in this repo:

- Vietnamese is now the default language for `AnalyzeRequest`.
- The rule-based extractor includes Vietnamese clinical terms observed in the
  local `input/input/*.txt` files.
- `/v1/analyze/btc` returns the BTC-compatible list-of-entities schema:
  `text`, `position`, `type`, `assertions`, and `candidates`.
- `python -m clinical_nlp.cli.batch input/input --output output/output.zip`
  creates a ZIP containing one `output/` folder with one JSON file per input
  record, for example `output/1.json`, `output/2.json`, ..., `output/100.json`.
  Each JSON file is a direct list of extracted entities.
- `input/` and generated `output/` files are not intended to be committed, so
  contest data and submissions stay local.

BTC output contract:

- The submitted ZIP must contain an `output/` folder with one JSON file per
  input file: `output/1.json`, `output/2.json`, ..., `output/100.json`.
- Each JSON file is a direct list of dictionaries, not an object wrapper.
- Each entity dictionary uses these fields:
  - `text`: exact source text span for the concept.
  - `position`: two integers `[start, end]`, zero-based character offsets.
  - `type`: one of `TRIỆU_CHỨNG`, `TÊN_XÉT_NGHIỆM`,
    `KẾT_QUẢ_XÉT_NGHIỆM`, `CHẨN_ĐOÁN`, `THUỐC`.
  - `assertions`: list containing zero or more of `isNegated`, `isFamily`,
    `isHistorical`. BTC only evaluates this for symptoms, diagnoses, and
    medications.
  - `candidates`: ICD-10 codes for `CHẨN_ĐOÁN`, RxNorm codes for `THUỐC`;
    other entity types should use an empty list in this implementation.
- If the same medical concept appears at multiple positions, BTC expects one
  JSON object per occurrence with its own `position`.
- Lab result text values such as `dương tính`, `âm tính`, and `bình thường`
  are valid. Include the unit only when the source text includes the unit.
- BTC clarified that RxNorm version 2026 is used, and ICD-10 uses the
  Vietnamese ICD-10 standard. BTC does not provide additional terminology files.

Latest scoring notes from the local BTC requirement Markdown files:

- `final_score = 0.3 * text_score + 0.3 * assertions_score + 0.4 * candidates_score`.
- `text_score` is based on `1 - WER` over the predicted `text` fields.
- `assertions_score` uses Jaccard similarity for the `assertions` lists.
- `candidates_score` uses Jaccard similarity for ICD-10/RxNorm `candidates`
  and carries the largest weight. BTC weights each sample by
  `sum(len(ground_truth(k)) + 1)` over candidates in that sample.
- If the same span is predicted with the wrong `type`, BTC counts it as a
  missed ground-truth concept and an extra predicted concept, so type precision
  matters as much as span quality.

The BTC serializer follows the requirement files provided in the local
workspace. If BTC publishes a stricter schema later, update only the serializer
and CLI packaging layer unless extraction behavior also needs to change.
