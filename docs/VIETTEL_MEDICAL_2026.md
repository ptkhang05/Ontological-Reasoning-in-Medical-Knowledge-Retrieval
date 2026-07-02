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
  public `input/input/*.txt` files.
- `python -m clinical_nlp.cli.batch input/input --output output/submission.zip`
  creates a ZIP containing `submission.json`.

The public phase endpoint does not expose the full judge schema without
authentication. The ZIP payload is therefore intentionally explicit and easy to
adapt if BTC publishes a stricter required schema.
