# Prompts

All prompt texts live in [`src/vllm_doc_processing/prompts.py`](../src/vllm_doc_processing/prompts.py). They are plain text, avoid provider-specific features and rely only on a strict JSON-schema response format (see README, *API requests*), so they are model-independent. `PROMPT_VERSIONS` records `<number>-<sha256 prefix of the text>` for each prompt; bump the number on intended changes, the hash changes with any edit anyway.

## Observation prompt (`observe`, issue #5)
One request per scan: a system prompt, then a user message containing **one image** and a short text:
```text
Scan position: 5 of 120 in physical scanning order (not a page number).

Context from earlier scans (automatic, may contain errors):
<bounded summary from #6; omitted when empty>

Annotate the attached scan.
```
The response must be a `ScanObservation` (docs/OUTPUT_SCHEMA.md). The system prompt:
- asks for observations **of this image only**, short quoted evidence instead of OCR, exact original spelling, and `null`/empty lists instead of guesses;
- states that the scan position is not a printed page number and that context may be wrong and must not be copied into the observation;
- defines `side` (`left`/`right`/`both`/`null`, with gutter and page-number-corner clues);
- lists all 38 MetaKat page types with one-line definitions (`PAGE_TYPE_DESCRIPTIONS`); for spreads the more specific type is the image label and each page gets a `subpages` entry;
- defines printed page numbers (raw as printed, normalized, numeric value, numeral system, side) and what is *not* a page number (signature marks, chapter/footnote/plate numbers, years); illegible numbers go to `notes`;
- restricts `headings` to headings starting on the scan (no running headers), `toc_entries` to TOC pages, and `bibliographic_candidates` to pages presenting the book itself (title page, cover, imprint, series page), one value per entry, without role phrases, keeping competing forms.

Invalid answers — bad JSON, schema mismatch or violated invariants such as `subpages` on a single page — are recorded as `invalid_response` calls and retried by the adapter; after `max_retries` the scan fails with an error carrying all attempts.

The page-type definitions are our own reading of the MetaKat/Czech NDK vocabulary (MetaKat publishes only the labels). In particular `FrontEndSheet`/`BackEndSheet` = pastedown (přídeští) and `FrontEndPaper`/`BackEndPaper` = free endpaper leaf, and half-title pages are labelled `TitlePage`; check these against the MetaKat ground truth during benchmarking (#10).

Use `vllm-doc observe` (README) to try the prompt on selected scans.

## Manual check (issue #5)
