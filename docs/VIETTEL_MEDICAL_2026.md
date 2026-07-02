# Viettel Medical 2026 Alignment

Source checked on 2026-07-02:

- Contest API: `https://competition.viettel.vn/api/contests/medical-2026`
- Phase API: `https://competition.viettel.vn/api/phases/019e649f-4e5d-70ed-b221-7a10f537281e`

Public facts returned by the API:

- Contest name: `Bài 2 - Ontological Reasoning in Medical Knowledge Retrieval`.
- Goal: extract and normalize medical concepts from free-text clinical records,
  including symptoms, lab results, diseases, medications, patient information,
  ICD-10 disease mapping, RxNorm medication mapping, contextual reasoning, and
  concept relations.
- Phase 1: `Vòng 1 - Sơ loại`.
- Phase 1 submission type: `FILE_ZIP`.
- Phase 1 worker type: `GPU`.

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
- `input/` is ignored by git so public/private contest data stays local.

Latest scoring notes from the local BTC requirement PDF:

- `final_score = 0.3 * text_score + 0.3 * assertions_score + 0.4 * candidates_score`.
- `text_score` is based on `1 - WER` over the predicted `text` fields.
- `assertions_score` uses Jaccard similarity for the `assertions` lists.
- `candidates_score` uses Jaccard similarity for ICD-10/RxNorm `candidates`
  and carries the largest weight.
- If the same span is predicted with the wrong `type`, BTC counts it as a
  missed ground-truth concept and an extra predicted concept, so type precision
  matters as much as span quality.

The public phase endpoint does not expose the full judge schema without
authentication. The BTC serializer follows the requirement files provided in
the local workspace; if BTC publishes a stricter schema later, update only the
serializer and CLI packaging layer.
