# Annotated-book JSON schema (`schema_version` 0.1)

**Source of truth:** `src/vllm_doc_processing/models.py` (Pydantic). Full example: [`examples/annotated_book.example.json`](../examples/annotated_book.example.json). Machine-readable JSON Schema: `AnnotatedBook.model_json_schema()`.

One run processes **one book** from one directory of **ordered scans** and writes one JSON object. It is a custom format, not MetaKat's `MetakatIO`. Unknown values are `null` or empty lists; nothing is guessed. All objects reject unknown keys.

## Layers
| Key | Content | Mutability |
|---|---|---|
| `scans[]` | Inventory (ID, index, filename, hash, size) + the vision model's `observation` of that one image. | Never edited by reconciliation. With escalation (#9) only the stronger model's observation is kept; the cheaper model's call stays in `run.calls`. |
| `resolved` | Document-level reconciled view: bibliography, per-scan page type/side/labels, chapter structure, warnings and change log. `null` until reconciliation has run. | Produced by the reconciliation stage (#7). |
| `run` | Provenance: provider, models, prompt versions, non-secret parameters, every API call attempt and usage totals. | Appended during the run. |

```text
AnnotatedBook
├─ schema_version: "0.1"   book_id   source {input_directory?, order_file?, scan_count}
├─ scans[]: ScanRecord {scan_id, scan_index, filename, image_sha256?, width?, height?,
│            observation: ScanObservation?, observation_call_id?}
├─ resolved: ResolvedBook? {bibliography, scans[]: ResolvedScan, structure[]: StructureNode,
│            warnings[], changes[]}
└─ run: RunInfo {provider, base_url, vision_model, postprocess_model, prompt_versions,
         parameters, calls[]: CallRecord, totals: UsageTotals, warnings[]}
```

## Scans and observations
- Scan order comes from an **order file** (one image name without extension per line, in physical order; recorded in `source.order_file`). Filenames are usually UUIDs and are not sorted.
- `scan_index` is the zero-based line position in the order file. `scan_id` is the listed name (e.g. the page UUID), so it is stable even if scans are added; `filename` is the matching file including its extension. Neither is a printed page number.
- `image_sha256` is the hash of the original file bytes; `width`/`height` are pixel dimensions of the upright original (EXIF orientation applied), not of the possibly downscaled upload. With `--max-pages N`, `scans` holds only the first N listed scans and `source.scan_count == N`.
- Exactly one `observation` per scan. `null` means the scan was not (successfully) observed; failed attempts are in `run.calls`. `observation_call_id` names the call that produced it: when a scan is escalated (#9) the stronger model's output replaces the cheaper one, and the cheaper call stays only in `run.calls`.
- `ScanObservation` is also the vision model's response contract (#5):

| Field | Type | Notes |
|---|---|---|
| `page_type`, `page_type_confidence` | `PageType?`, `float?` | Whole-image label (comparable with MetaKat's per-image label). |
| `side`, `side_confidence` | `left \| right \| both`?, `float?` | `both` = two-page spread in one image; unknown/not applicable (cover, spine) = `null`. |
| `subpages[]` | `{side: left\|right, page_type?, confidence?}` | Optional per-page types; only allowed when `side = both`, distinct sides. |
| `printed_numbers[]` | `PrintedNumber` | `side: left\|right\|null` (never `both`), `raw` exactly as printed (`[12]`, `xii`), `normalized`, `numeric_value`, `numeral_system: arabic\|roman\|other\|null`, `confidence`, `notes`. A spread holds two independent observations. Empty list = no number seen, which is *not* evidence of a missing scan. |
| `headings[]` | `{text, level?, side?, confidence?}` | Chapter/section headings visible on the scan. |
| `toc_entries[]` | `{title, printed_page_reference?, level?, confidence?}` | Short TOC evidence; target page as printed. |
| `bibliographic_candidates[]` | `{field: BiblioField, value, confidence?, notes?}` | Candidates as printed; competing candidates are all kept. |
| `notes` | `str?` | Short free-text remarks (illegibility etc.). |

Observations carry no scan IDs: their evidence is the enclosing scan.

## Resolved view and provenance
Every resolved assertion is a `Claim`:
```json
{"value": "Cesty po Šumavě", "origin": "observed", "source_scan_ids": ["5b1f0c2e-7d4a-4e8b-9c3f-000000000002"], "confidence": null, "notes": null}
```
`origin`: `observed` (read on a scan), `inferred` (derived, e.g. a page label counted from a sequence), `catalogued` (reserved for external metadata). Confidence is self-reported/heuristic, **not** calibrated.

- `bibliography` — field names equal the `BiblioField` vocabulary. Single claims: `title`, `subtitle`, `part_name`, `part_number`, `edition`, `publication_date`. Lists of claims: `series_name`, `series_number`, `publisher`, `publication_place`, `manufacture_publisher`, `manufacture_place`, `author`, `editor`, `translator`, `illustrator`, `photographer`.
- `scans[]` (`ResolvedScan`) — exactly one record per input scan, in scan order (values may stay `null`): `scan_id`, `page_type: Claim[PageType]?`, `side: Claim[ScanSide]?`, `subpages[]` (`{side: left|right, page_type: Claim[PageType]?}`, only when the resolved side is `both`), `page_labels[]` (`side?`, `label`, `numeric_value?`, `numeral_system?` + origin/source/confidence/notes). Inferred labels for unnumbered pages use `origin: inferred`.
- `structure[]` (`StructureNode`) — flat list; hierarchy by `parent_id` and `level` (1 = top). `title`/`subtitle`/`part_number` claims, `printed_page_reference` (from TOC), `toc_scan_ids`, `heading_scan_ids`, `start_scan_id`, `end_scan_id` (null when unresolved, e.g. target not scanned), `origin`, `confidence`, `notes`. A TOC entry is evidence, not proof that the destination was captured.
- `warnings[]` — `{code, message, detected_by: llm|check, field_path?, scan_ids[]}`; conflicts and unresolved questions are flagged here rather than silently fixed.
- `changes[]` — audit log `{field_path, old_value, new_value, reason, source_scan_ids}` wherever the resolved value differs from (or adds to) the observations.

## Run provenance
`CallRecord` is one request **attempt**: `call_id`, `stage: observe|reconcile|escalate|revisit`, `scan_id?`, `provider`, `model`, `attempt` (retries > 1), `status: ok|invalid_response|error`, `started_at`, `latency_s`, `prompt_tokens`, `completion_tokens`, `cost_usd` (as reported; `null` if not — OpenRouter reports cost, OpenAI does not), `response_id` (provider response/generation ID), `served_model` (model name in the response, e.g. a dated snapshot), `upstream_provider` (serving provider behind OpenRouter), `error` (redacted, truncated; never contains prompts, images or keys). `prompt_versions` holds `observe` (observation prompt) and `context` (format of the earlier-scan context, `context.CONTEXT_VERSION`). `totals` must equal `UsageTotals.from_calls(calls)` (call `run.refresh_totals()` after adding calls); `cost_complete: false` means some calls lacked a reported cost. `parameters` must never contain secrets.

## Validated invariants
- `schema_version` is `"0.1"`; `source.scan_count == len(scans)`.
- `scans` in order-file order, `scan_index` contiguous from 0; `scan_id`, `filename`, `call_id` unique.
- Every scan ID referenced anywhere (`resolved.*`, `run.calls[].scan_id`) exists.
- If `resolved` is present, `resolved.scans` lists every input scan exactly once, in scan order.
- `observation_call_id` (only set together with an `observation`) refers to a successful (`ok`) `observe` or `escalate` call for the same scan.
- `run.totals` is consistent with `run.calls`.
- `dump_json()` re-validates the whole document, so in-place edits that break an invariant fail before writing.
- Printed numbers and subpages never use `side: both`; subpages only on spreads (observed and resolved), with distinct sides.
- Structure IDs unique; a parent is listed before its children and has a lower `level`; `start_scan_id` is not after `end_scan_id`. Unresolved bounds are allowed.
- Confidence values in [0, 1]; non-negative numeric page values, tokens and costs.

## MetaKat mapping
Reference: [base_objects.py](https://github.com/DCGM/MetaKat/blob/main/metakat/schemas/base_objects.py). MetaKat is not a runtime dependency.

| MetaKat | This schema |
|---|---|
| `PageType` (38 values) | `PageType`, identical strings. Unknown is `null`, not `NormalPage`. |
| `PageSideType` `left\|right\|single_page` | `ScanSide` `left\|right\|both\|null`. MetaKat `single_page` has no direct equivalent (a single non-facing leaf is `left`/`right` or `null`); `both` (spread) has no MetaKat value. Needs explicit handling in evaluation (#10). |
| `MetakatPage.pageIndex` / `batch_index` | `scan_index` |
| `MetakatPage.pageNumber` | `printed_numbers[]` (observed) / `resolved.scans[].page_labels[]` — may be two per spread. |
| `MetakatTitle` + `MetakatVolume`: `title`, `subTitle`, `partName`, `partNumber`, `edition`, `dateIssued`, `placeTerm`, `publisher`, `manufacturePublisher`, `manufacturePlaceTerm`, `seriesName`, `seriesNumber`, `author`, `editor`, `translator`, `illustrator`, `photographer` | `Bibliography` fields in snake_case (`subtitle`, `publication_date`, `publication_place`, `manufacture_place`, …). MetaKat `(value, confidence, detection_id)` tuples become `Claim` with `source_scan_ids`. `placeTerm` is a list here (several places may be printed). Periodical/issue fields and `redaktor` are out of scope. |
| `MetakatChapter`: `title`, `title_destination_page`, `subTitle`, `partNumber`, `pageNumber`, `pageIndexToc`, `pageIndexStart`, `pageIndexEnd` | `StructureNode`: `title`, heading text in observations of `heading_scan_ids`, `subtitle`, `part_number`, `printed_page_reference`, `toc_scan_ids`, `start_scan_id`, `end_scan_id`; plus `parent_id`/`level` (MetaKat `Level1Title`/`Level2Title`). |
| `imageDim` | `width`, `height` |
| Bounding boxes, ALTO, detections | Out of scope. |

## Versioning
Breaking changes bump `schema_version` and are listed here with a migration note.
