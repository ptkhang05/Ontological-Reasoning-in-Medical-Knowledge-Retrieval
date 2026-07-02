# Terminology Sources

Terminology lookup is local-first. Place source-backed CSV files under
`data/terminologies` with this schema:

```csv
code_system,code,preferred_term,synonyms,release_id,source_url
```

Supported v1 systems:

- `ICD-10-CM` for disease normalization.
- `RxNorm` for medication normalization.
- `ATC` as a future alternate medication code system, not a v1 replacement for
  RxNorm.

The built-in entries are demo seeds so tests and local development work without
shipping large terminology dumps. Replace them with official local files before
using this against real workflows.

Reference sources:

- CMS ICD-10-CM: https://www.cms.gov/medicare/coding-billing/icd-10-codes
- NLM RxNorm overview: https://www.nlm.nih.gov/research/umls/rxnorm/overview.html
- NLM RxNorm API: https://lhncbc.nlm.nih.gov/RxNav/APIs/RxNormAPIs.html
- WHO ATC classification: https://www.who.int/tools/atc-ddd-toolkit/atc-classification
