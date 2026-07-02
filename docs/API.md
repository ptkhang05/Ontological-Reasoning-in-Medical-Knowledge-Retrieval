# API Contract

## `POST /v1/analyze`

Accepts a clinical free-text note and returns extracted concepts, relations,
review flags, warnings, and processing metadata.

Request fields:

- `text` (required): clinical note text, 1-20000 characters.
- `documentId` (optional): caller-provided document identifier.
- `documentType` (optional): caller-provided document type.
- `encounterDate` (optional): caller-provided encounter date.
- `language` (optional): defaults to `vi`; English and Vietnamese are supported
  by the prototype rules.
- `options.allowExternalInference` (optional): defaults to `false`.
- `options.confidenceThreshold` (optional): defaults to `0.80`.
- `options.includeUnmapped` (optional): defaults to `true`.

Errors use this stable shape:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid request.",
    "details": []
  }
}
```

The service intentionally avoids echoing raw clinical text in validation errors.

## `POST /v1/analyze/btc`

Accepts the same request body as `/v1/analyze`, but returns the BTC-compatible
entity list directly. This endpoint intentionally omits prototype metadata,
relations, warnings, and review flags.

Each returned entity has exactly these keys:

```json
{
  "text": "aspirin",
  "position": [42, 49],
  "type": "THUỐC",
  "assertions": [],
  "candidates": ["1191"]
}
```

Supported `type` values are `TRIỆU_CHỨNG`, `TÊN_XÉT_NGHIỆM`,
`KẾT_QUẢ_XÉT_NGHIỆM`, `CHẨN_ĐOÁN`, and `THUỐC`. Supported assertion values are
`isNegated`, `isFamily`, and `isHistorical`.
