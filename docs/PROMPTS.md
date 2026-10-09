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
Run 2026-10-09 with `vllm-doc observe` on 16 scans picked from 9 local digitized documents (not committed): title pages of a 1902 Czech monograph and a 1965 geophysics offprint in a series, journal covers, two bilingual TOC pages, a blank page, a nearly invisible mirrored show-through page, text pages with chapter headings and with printer's signature marks, a 17th-century Latin occasional print, and a fold-out map. No real two-page spreads were available, so spread handling is untested. Each scan was annotated independently (no context), via OpenRouter with default settings, prompt versions 1–3, `openai/gpt-4.1-mini` and `openai/gpt-5.4-mini`.

| | gpt-4.1-mini | gpt-5.4-mini |
|---|---|---|
| Valid `ScanObservation` on first attempt | 48/48 | 48/48 |
| Prompt / completion tokens per scan (mean) | ~5.4k / ~200 | ~4.9k / ~240 |
| Cost per scan (OpenRouter-reported) | ~$0.0017–0.0019 | ~$0.0029–0.0031 |
| Median latency | ~4 s | ~3 s |

About 4k of the prompt tokens are the system prompt (identical for every scan, so cacheable by providers that support it).

**Successes (both models, prompt v3):** title page → correct title, subtitle, author, place, year and publisher (Czech diacritics preserved); text pages → correct arabic page numbers and side; chapter openings → headings ("Úvod.", "Dodatek."); TOC pages → complete entries with page references, including Cyrillic; truly blank and near-invisible show-through pages → `Blank` or `null` page type with empty lists (no invented text); catchwords ("SPON") recognised as not page numbers.

**Failures observed:**
- *Signature marks as page numbers:* "1\*" (v1, both models) and "B ij" (gpt-5.4-mini, all versions — even when its own note says it is a signature). Explicit examples in v2/v3 fixed "1\*" but not "B ij".
- *TOC page references as page numbers* of the TOC page itself (gpt-4.1-mini, all versions, varying between runs); fixed for gpt-5.4-mini from v2.
- *False spreads:* a single bilingual TOC page (two stacked language versions) and a journal cover were labelled `both`; the fold-out map was `both` until v3 (gpt-5.4-mini) and remains `both` with gpt-4.1-mini.
- *TOC section captions as headings* ("ARTICLES", "Contents No. 6/1985").
- *Weak cases:* the Latin occasional print's dedicatory title page was labelled NormalPage/TitlePage/CustomInclude across runs, with the dedicatee and opening phrase once reported as `author`/`publisher` (v1/v2); journal issue numbers land in `series_number`/`part_number` (periodicals are out of scope).
- *Run-away output:* one gpt-4.1-mini call (v2) produced 27.5k completion tokens (whitespace before a valid 470-character JSON) and took 194 s, costing as much as ~25 normal scans. Set an output cap, e.g. `"request_params": {"max_completion_tokens": 4000}`; a capped response is then rejected as truncated and retried.

Outputs vary between runs at default temperature, so these are qualitative observations, not accuracy figures; measured accuracy is the subject of #10.
