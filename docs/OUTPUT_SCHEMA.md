# Annotated-book JSON schema (`schema_version` 0.3)

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
├─ schema_version: "0.3"   book_id   source {input_directory?, order_file?, ocr_directory?, scan_count}
├─ scans[]: ScanRecord {scan_id, scan_index, filename, image_sha256?, width?, height?, ocr: OcrInput?,
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
- `ocr` (#14) describes the OCR sidecar sent with the scan; it is `null` in every scan of a run without `--ocr-dir` (`source.ocr_directory: null`) and set in every scan of a run with one. `OcrInput`: `status: ok|missing|error`, `filename` (in `source.ocr_directory`), `format: txt|alto`, `sha256` and `size_bytes` of the sidecar bytes, `chars` (length of the normalized text), `sent_chars` (length sent, ≤ `ocr_max_chars`), `truncated`, `error`. `missing` = no sidecar (all other fields `null`); `error` = the sidecar could not be used (`error` says why: ambiguous match, unreadable, not UTF-8, malformed or non-ALTO XML; file details as far as known, no text lengths). Both are sent as image only and listed in `run.warnings`. The OCR text itself is never stored. It is request input, not evidence: observations are not attributed to OCR.
- Exactly one `observation` per scan. `null` means the scan was not (successfully) observed; failed attempts are in `run.calls`. `observation_call_id` names the call that produced it: when a scan is escalated (#9) the stronger model's output replaces the cheaper one, and the cheaper call stays only in `run.calls`.
- `ScanObservation` is also the vision model's response contract (#5):

| Field | Type | Notes |
|---|---|---|
| `page_type`, `page_type_confidence`, `page_type_reason` | `PageType?`, `float?`, `str?` | Whole-image label from the NDK vocabulary (below); `page_type_reason` = the visible evidence in a few words. |
| `side`, `side_confidence` | `left \| right \| both`?, `float?` | `both` = two-page spread in one image; unknown/not applicable (cover, spine) = `null`. |
| `side_reason` | `str?` | Visible evidence for the side. |
| `leaf`, `leaf_reason` | `book_block \| plate \| binding \| loose`?, `str?` | Physical kind of the sheet: part of the book block, an inserted plate (different paper, printed on one side, tipped in), binding (covers, pastedowns, endpapers, jacket) or a loose sheet. Plates, binding and loose sheets are outside the page count (NDK §1.1.4). On a spread it applies to both pages unless a subpage gives its own. |
| `subpages[]` | `{side: left\|right, page_type?, confidence?, leaf?}` | Optional per-page types and leaf kinds (when the two pages differ, e.g. a text page facing a plate); only allowed when `side = both`, distinct sides. |
| `printed_numbers[]` | `PrintedNumber` | `side: left\|right\|null` (never `both`), `raw` exactly as printed (`[12]`, `xii`), `normalized`, `numeric_value`, `numeral_system: arabic\|roman\|other\|null`, `position: top_left\|top_center\|top_right\|bottom_left\|bottom_center\|bottom_right\|other\|null`, `confidence`, `notes`. A spread holds two independent observations. Empty list = no number seen, which is *not* evidence of a missing scan. |
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
- `scans[]` (`ResolvedScan`) — exactly one record per input scan, in scan order (values may stay `null`): `scan_id`, `page_type: Claim[PageType]?`, `side: Claim[ScanSide]?`, `leaf: Claim[Leaf]?`, `subpages[]` (`{side: left|right, page_type: Claim[PageType]?, leaf: Claim[Leaf]?}`, only when the resolved side is `both`; a spread's scan-level `leaf` is copied to subpages without their own, adding subpage entries as needed, and only the subpage leaf kinds count), `page_labels[]` (one per page: `side?`, `label`, `numeric_value?`, `numeral_system?` + origin/source/confidence/notes) and `page_number` (the whole scan's label, `page_labels` joined by `,`; `null` if a page is unresolved). Labels follow the Czech NDK notation ([Pravidla pro popis monografií 2.4](https://standardy.ndk.cz/ndk/standardy-digitalizace/ppp_mono_2.4_final.pdf/at_download/file), §1.1), as in Kramerius `ORDERLABEL` and MetaKat `pageNumber`:

| Page | Label | `origin` | `numeric_value` |
|---|---|---|---|
| printed number in sequence | `12`, `XII` (canonical roman, upper case) | observed | 12 |
| printed number wrong inside an intact sequence (§1.1.2) | the correct number, `notes: "printed '31'"`, warning `page_number_conflict` | inferred | correct value |
| counted page without printed number (§1.1.4 "chybí v číselné řadě") | `[13]`; at the start `[1]`, `[2]`, …; at the end `[70]`, `[71]`, … (binding parts included) | inferred | 13 |
| page outside the count: binding parts, jacket, loose leaves, frontispiece, inserted plates (§1.1.4 "nechybí v číselné řadě"); after the last printed number all pages are counted instead (§1.1.4 c) | `[1a]`, `[1b]` before page 1 (`[Ia]` before roman front matter); `[26a]`, `[26b]` after page 26 | inferred | `null` |
| spread (§1.1.6) | `page_number` `5,6`, `[4],5` | | |
| several or non-numeric printed numbers on one page | as observed | observed | as observed |

NDK practice brackets every number that is not printed (the rules allow omitting brackets in the middle of the book); we always bracket. Leaf numbering (`1r`, `1v`), column numbering and the optional `55 [58]` form are not produced. Which unnumbered pages are counted is decided by `pagination.py` (docs/PROMPTS.md).
- `structure[]` (`StructureNode`) — flat list; hierarchy by `parent_id` and `level` (1 = top). `title`/`subtitle`/`part_number` claims, `printed_page_reference` (from TOC), `toc_scan_ids`, `heading_scan_ids`, `start_scan_id`, `end_scan_id` (null when unresolved, e.g. target not scanned), `origin`, `confidence`, `notes`. A TOC entry is evidence, not proof that the destination was captured.
- `warnings[]` — `{code, message, detected_by: llm|check, field_path?, scan_ids[]}`; conflicts and unresolved questions are flagged here rather than silently fixed. Codes from the checks (#7): `page_number_conflict`, `correction_of_unobserved_scan`, `page_number_gap`, `page_number_sequence_break`, `competing_observed_values`, `competing_values`, `ungrounded_value`, `leaf_type_conflict`, `ungrounded_chapter`, `ungrounded_title`, `ungrounded_reference`, `unmatched_heading`, `unmatched_toc_entry`, `invalid_scan_reference`, `unresolved_toc_reference`, `ambiguous_toc_reference`, `toc_heading_mismatch`, `contradictory_order`, `duplicate_chapter` (see [PROMPTS.md](PROMPTS.md#document-wide-reconciliation-issue-7)); `llm` warnings use the model's own slugs.
- `changes[]` — audit log `{field_path, old_value, new_value, reason, source_scan_ids}` wherever a resolved page type or side was corrected (`old_value`: the observed value), or a resolved bibliographic value or chapter title is not literally observed (`old_value`: the values observed for that field, or `null`). Computed page labels are not repeated here; they are marked by `origin: inferred`.

## Run provenance
`CallRecord` is one request **attempt**: `call_id`, `stage: observe|reconcile|escalate|revisit`, `scan_id?`, `provider`, `model`, `attempt` (retries > 1), `status: ok|invalid_response|error`, `started_at`, `latency_s`, `prompt_tokens`, `completion_tokens`, `cost_usd` (as reported; `null` if not — OpenRouter reports cost, OpenAI does not), `response_id` (provider response/generation ID), `served_model` (model name in the response, e.g. a dated snapshot), `upstream_provider` (serving provider behind OpenRouter), `error` (redacted, truncated; never contains prompts, images or keys). `prompt_versions` holds `observe` (observation prompt), `context` (format of the earlier-scan context, `context.CONTEXT_VERSION`), `ocr` (OCR block of the observation request, only in runs with an OCR directory) and, after reconciliation, `reconcile` (reconciliation prompt and input format). `totals` must equal `UsageTotals.from_calls(calls)` (call `run.refresh_totals()` after adding calls); `cost_complete: false` means some calls lacked a reported cost. `parameters` must never contain secrets (on resume: the settings of the last session). A resumed run (#8) keeps the calls of all its sessions, including failed and repeated reconciliations, and the first session's `started_at`. `finished_at` is set only in a written output, never in a checkpoint. `run.warnings` lists scans left without an observation and a skipped reconciliation (`--skip-postprocess`).

The checkpoint file (`--checkpoint`, #8) is `{"checkpoint_version": 1, "identity": {...}, "book": AnnotatedBook}` with `book.resolved` and `book.run.finished_at` always `null`; `identity` holds the settings (including `ocr`: `null` or the OCR format and `max_chars`), prompt versions, schema version and order-file hash that must match for `--resume` (`checkpoint.run_identity`); images and OCR sidecars are compared per scan (`scans[].image_sha256`, `scans[].ocr`). It is an internal format; only the output is versioned by `schema_version`.

## Validated invariants
- `schema_version` is `"0.3"`; `source.scan_count == len(scans)`.
- `scans[].ocr` is set in every scan if `source.ocr_directory` is set, else `null` in every scan; an `ok` OCR input has all file details and text lengths, a `missing` one none, an `error` one an `error` message and no text lengths.
- `scans` in order-file order, `scan_index` contiguous from 0; `scan_id`, `filename`, `call_id` unique.
- Every scan ID referenced anywhere (`resolved.*`, `run.calls[].scan_id`) exists.
- If `resolved` is present, `resolved.scans` lists every input scan exactly once, in scan order.
- `observation_call_id` (only set together with an `observation`) refers to a successful (`ok`) `observe` or `escalate` call for the same scan.
- `run.totals` is consistent with `run.calls`.
- `dump_json()` re-validates the whole document, so in-place edits that break an invariant fail before writing.
- `resolved.scans[].page_number` is `null` or its `page_labels` joined by `,`.
- Printed numbers and subpages never use `side: both`; subpages only on spreads (observed and resolved), with distinct sides.
- Structure IDs unique; a parent is listed before its children and has a lower `level`; `start_scan_id` is not after `end_scan_id`. Unresolved bounds are allowed.
- Confidence values in [0, 1]; non-negative numeric page values, tokens and costs.

## MetaKat mapping
Reference: [base_objects.py](https://github.com/DCGM/MetaKat/blob/main/metakat/schemas/base_objects.py). MetaKat is not a runtime dependency.

| MetaKat | This schema |
|---|---|
| `PageType` (38 values, PascalCase) | `PageType`: the 37 NDK page types written as in NDK (lowerCamelCase), see below. Unknown is `null`, not `normalPage`. |
| `PageSideType` `left\|right\|single_page` | `ScanSide` `left\|right\|both\|null`. MetaKat `single_page` has no direct equivalent (a single non-facing leaf is `left`/`right` or `null`); `both` (spread) has no MetaKat value. The evaluator treats `single_page` as incomparable ([EVALUATION.md](EVALUATION.md)). |
| `MetakatPage.pageIndex` / `batch_index` | `scan_index` |
| `MetakatPage.pageNumber` | `resolved.scans[].page_number` (NDK label of the scan, e.g. `[1a]`, `5,6`); per page in `page_labels[]`; observed numbers in `printed_numbers[]`. |
| `MetakatTitle` + `MetakatVolume`: `title`, `subTitle`, `partName`, `partNumber`, `edition`, `dateIssued`, `placeTerm`, `publisher`, `manufacturePublisher`, `manufacturePlaceTerm`, `seriesName`, `seriesNumber`, `author`, `editor`, `translator`, `illustrator`, `photographer` | `Bibliography` fields in snake_case (`subtitle`, `publication_date`, `publication_place`, `manufacture_place`, …). MetaKat `(value, confidence, detection_id)` tuples become `Claim` with `source_scan_ids`. `placeTerm` is a list here (several places may be printed). Periodical/issue fields and `redaktor` are out of scope. |
| `MetakatChapter`: `title`, `title_destination_page`, `subTitle`, `partNumber`, `pageNumber`, `pageIndexToc`, `pageIndexStart`, `pageIndexEnd` | `StructureNode`: `title`, heading text in observations of `heading_scan_ids`, `subtitle`, `part_number`, `printed_page_reference`, `toc_scan_ids`, `start_scan_id`, `end_scan_id`; plus `parent_id`/`level` (MetaKat `Level1Title`/`Level2Title`). |
| `imageDim` | `width`, `height` |
| Bounding boxes, ALTO, detections | Out of scope. |

## Page types
`PageType` is the NDK page-type vocabulary of [Pravidla pro popis monografií 2.4](https://standardy.ndk.cz/ndk/standardy-digitalizace/ppp_mono_2.4_final.pdf/at_download/file), table 1.2.2 (the values used in NDK METS `TYPE` and MODS `genre type`): `frontJacket`, `cover`, `frontCover`, `backCover`, `frontEndSheet`, `backEndSheet`, `frontEndPaper`, `backEndPaper`, `titlePage`, `preface`, `introduction`, `normalPage`, `blank`, `illustration`, `map`, `table`, `advertisement`, `impressum`, `colophon`, `frontispiece`, `imprimatur`, `dedication`, `errata`, `sheetMusic`, `appendix`, `bibliography`, `afterword`, `conclusion`, `tableOfContents`, `index`, `listOfIllustrations`, `listOfMaps`, `listOfTables`, `edge`, `spine`, `jacket`, `flyleaf`. Definitions (our English summary of §1.2.1) are in `prompts.PAGE_TYPE_DESCRIPTIONS`.

Mapping to MetaKat's `PageType` for evaluation (#10, implemented in `predictions.page_type`; types without an NDK equivalent are incomparable, never equal). Kramerius data mixes both spellings, so compare case-insensitively:

| MetaKat | NDK (this schema) |
|---|---|
| same name in PascalCase (`TitlePage`, `FrontEndSheet`, …) | same name in lowerCamelCase (`titlePage`, `frontEndSheet`, …) |
| `FlyLeaf` | `flyleaf` (NDK: a loose leaf, not a blank protective leaf) |
| `Abstract`, `Obituary` | no page type (NDK uses them only for chapters); usually `normalPage` |
| `CalibrationTable`, `CustomInclude`, `FragmentsOfBookbinding` | no NDK page type; closest `flyleaf` (loose inserts) or exclude from scoring |
| — | `colophon` (closest MetaKat `Impressum`), `introduction`, `afterword`, `conclusion` (closest `NormalPage`/`Preface`) |

## Versioning
Breaking changes bump `schema_version` and are listed here with a migration note.

- **0.3** (#14): added `source.ocr_directory`, `ScanRecord.ocr` (`OcrInput`) and `run.prompt_versions.ocr`, all absent/`null` in image-only runs. Migration of 0.2 files: set `schema_version` to `"0.3"`; nothing else changes. `vllm-doc evaluate` reads 0.2 outputs directly.

- **0.2** (#21): `PageType` values switched from MetaKat's vocabulary to NDK's (renamed to lowerCamelCase, `FlyLeaf` → `flyleaf`; `Abstract`, `Obituary`, `CalibrationTable`, `CustomInclude`, `FragmentsOfBookbinding` removed; `colophon`, `introduction`, `afterword`, `conclusion` added). Added `ScanObservation.page_type_reason`, `side_reason`, `leaf`, `leaf_reason`, `SubpageObservation.leaf`, `ResolvedScan.leaf`, `ResolvedSubpage.leaf` and `PrintedNumber.position`. Migration of 0.1 files: rename page types per the table above (removed types → `null`) and set `schema_version` to `"0.2"`; no 0.1 results were published.
