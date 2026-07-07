# Terminology Data

Place local, source-backed terminology files here.

Supported CSV columns:

```csv
code_system,code,preferred_term,synonyms,release_id,source_url
```

Use pipe-separated synonyms, for example `type 2 diabetes|t2dm`.

Local CSV rows augment the built-in seed terminology. When a local row uses the
same term as a seed row, the local source-backed row is preferred.

Large or licensed dumps such as RxNorm RRF files should not be committed.
After your UMLS/RxNorm license is approved, download the RxNorm release locally
and convert it into the project CSV format with:

```powershell
python scripts/build_rxnorm.py --input C:\path\to\RxNorm_full_07062026.zip --output data/terminologies/rxnorm.generated.csv
```

The generated CSV is ignored by git and should stay local to your laptop or VM.
