# API Contract

## `POST /v1/analyze`

Accepts a clinical free-text note and returns extracted concepts, relations,
review flags, warnings, and processing metadata.

Request fields:

- `text` (required): clinical note text, 1-20000 characters.
- `documentId` (optional): caller-provided document identifier.
- `documentType` (optional): caller-provided document type.
- `encounterDate` (optional): caller-provided encounter date.
- `language` (optional): defaults to `en`; non-English is best-effort in v1.
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
