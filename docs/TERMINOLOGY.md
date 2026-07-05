# Terminology Sources

Terminology lookup is local-first. Place source-backed CSV files under
`data/terminologies` with this schema:

```csv
code_system,code,preferred_term,synonyms,release_id,source_url
```

Supported v1 systems:

- `ICD-10-CM` for disease normalization.
- `ICD-10-TT06` for Vietnamese ICD-10 disease normalization from the public
  06/2026/TT-BYT browser.
- `RxNorm` for medication normalization.
- `ATC` as a future alternate medication code system, not a v1 replacement for
  RxNorm.

The built-in entries are demo seeds so tests and local development work without
shipping large terminology dumps. Replace them with official local files before
using this against real workflows.

Reference sources:

- Vietnamese ICD-10 TT06 browser: https://icd.kcb.vn/icd-10-tt06/icd10-tt06
- Vietnamese ICD-10 TT06 API root:
  https://ccs.whiteneuron.com/api/ICD10_TT06/root?lang=vi
- CMS ICD-10-CM: https://www.cms.gov/medicare/coding-billing/icd-10-codes
- NLM RxNorm overview: https://www.nlm.nih.gov/research/umls/rxnorm/overview.html
- NLM RxNorm API: https://lhncbc.nlm.nih.gov/RxNav/APIs/RxNormAPIs.html
- WHO ATC classification: https://www.who.int/tools/atc-ddd-toolkit/atc-classification

To build a local Vietnamese ICD CSV from the public TT06 API:

```powershell
python scripts/build_icd10_tt06.py --output data/terminologies/icd10_tt06.generated.csv
```

Generated terminology CSV files named `*.generated.csv` are ignored by git.
Review licensing/source requirements before committing any downloaded ontology
dump.
