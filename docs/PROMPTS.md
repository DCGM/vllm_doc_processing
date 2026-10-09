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

## Context from earlier scans (issue #6)
`pipeline.observe_book` observes the scans strictly in order; each request carries only the current image plus a text context built by `context.build_context` from the stored observations of **all** earlier scans. The context is rebuilt from scratch for every scan, deterministically (same observations → same text), never edits the observations, and is not stored: it can be regenerated from `scans[].observation`. Its format version is recorded as `run.prompt_versions.context`. With `use_context: false` no context is sent (every scan is observed on its own, as with `vllm-doc observe`), so the effect of the context can be measured; the setting is recorded in `run.parameters`. Example (scan 6 of a book):
```text
Scans 1-5 so far; "scan N" is a scan position, not a page number.
Printed page numbers, latest: scan 2: XII (left), 13 (right); scan 4: 13; scan 5: 14.
Current chapter (latest headings): level 1 "KAPITOLA I. Úvod" (scan 4).
Unresolved: scan 3 not observed (request failed); page number 13 on scan 4 after 13 on scan 2.
Table of contents entries seen on scan 2 (1 entries).
Bibliographic data seen (field: value (scans)):
- title: "Cesty po Šumavě" (scan 1)
- author: "Karel Klostermann" (scan 1)
Previous scans:
- scan 1: TitlePage; right; no page number
- scan 2: TableOfContents; both (left NormalPage, right TableOfContents); page XII (left), 13 (right); 1 TOC entries
- scan 3: not observed
- scan 4: NormalPage; right; page 13; heading "KAPITOLA I. Úvod"
- scan 5: NormalPage; left; page 14
```
Sections, in priority order: the current document state — the latest printed numbers; the current heading per level (a heading closes deeper levels; unknown level counts as 1; at most 4 levels); unresolved points (failed scans, several forms of a single-valued bibliographic field, page numbers that do not increase within one numeral system); where TOC entries were seen (counts only, not the entries) — then bibliographic candidates per field (distinct values in first-seen order, with the scans showing them; fields ordered title, author, subtitle, part, edition, date, publisher, place, series, other contributors, printer), then one line per recent scan.

**Bounding rule:** quoted values are cut to 80 characters; at most 3 values per bibliographic field, the last 5 numbered scans, 4 heading levels, 2 numbering irregularities, 5 failed scans and `context_recent_scans` previous-scan lines are shown. If the text is still longer than `context_max_chars`, the oldest previous-scan lines are dropped first, then bibliographic fields from the least important one (if all are dropped, the section says so), then whole lines from the end. The document state therefore survives a crowded title page, and the context never exceeds `context_max_chars` (default 2000 characters ≈ 500–700 tokens), however long the book.

The observation system prompt already tells the model that the context is automatic, may be wrong and must not be copied into the observation; whether the context improves or degrades observations has not been measured yet.

## Document-wide reconciliation (issue #7)
`reconcile.reconcile_book` runs after all scans are observed and fills `resolved`; `scans[].observation` is never modified. The LLM does only what needs judgement or language understanding, and everything it returns is checked against the observations. The conventions follow the Czech NDK rules for describing monographs ([Pravidla pro popis monografií 2.4](https://standardy.ndk.cz/ndk/standardy-digitalizace/ppp_mono_2.4_final.pdf/at_download/file), "PPM"), which the MetaKat/Kramerius ground truth also follows.

| Step | Done by |
|---|---|
| Page type and side per scan | observed values, unless the LLM lists a `scan_corrections` entry for a clearly wrong one (judged from neighbouring scans, numbering and content, using the PPM 1.2 conventions in the prompt: order of binding parts, `FlyLeaf` = loose leaf, priorities such as TitlePage before TableOfContents, alternating sides with odd numbers on the right). A correction becomes `origin: inferred` with the reason in `notes` and a `changes[]` entry; corrections of unknown or unobserved scans are ignored with a warning. |
| Page labels (NDK notation, see OUTPUT_SCHEMA) | code (`pagination.py`), from the printed numbers and the *corrected* page types and sides, see below |
| Bibliography: choose, merge and deduplicate forms | LLM; then each value must equal an observed candidate (ignoring case, spacing and trailing punctuation) or cite a scan with bibliographic candidates, else it is dropped (`ungrounded_value`). Values not observed literally in that field get `origin: inferred` and a `changes[]` entry. A second value for a single-valued field is dropped (`competing_values`). |
| Chapter list: TOC entry ↔ heading matching, titles, levels | LLM; cited scans must hold TOC entries/headings, page references must be printed in those TOC entries, chapters citing neither are dropped. |
| Hierarchy, start and end scans | code: parent = previous chapter of a lower level; start = heading scan, else the unique scan whose page label (printed or computed) equals the TOC page reference; none → `unresolved_toc_reference` (start stays `null`), several → `ambiguous_toc_reference`, heading and TOC disagree → `toc_heading_mismatch` (heading wins). End = scan before the next chapter of the same or higher level (that scan itself if it is a spread), `null` when unknown or for the last chapter. Starts going backwards → `contradictory_order`. |
| Other doubts | LLM `issues` → warnings with `detected_by: llm` |

**Page labels** (`pagination.paginate`, PPM §1.1). Every page (a spread is two pages) is classified from the printed numbers:
1. A page with exactly one printed arabic/roman number is an *anchor*. An anchor disagreeing with both neighbours that agree with each other is a misprint or misreading inside an intact sequence: it is labelled with the correct number (§1.1.2) and flagged `page_number_conflict`.
2. Before the first anchor (value *v*), the last *v*−1 countable pages get `[1]` … `[v-1]`; earlier pages and binding parts are lettered `[1a]`, `[1b]`, … (§1.1.4). With no printed numbers at all, countable pages are `[1]`, `[2]`, … (§1.1.4 d).
3. Between anchors *a* and *b* of the same system, the *b*−*a*−1 missing numbers are given to some of the unnumbered pages and the rest are lettered after the preceding number (`[16a]`). The choice minimizes a cost: counting a binding part, jacket, loose leaf, frontispiece or calibration target (`UNCOUNTED_TYPES`) costs 10, a number on the wrong side costs 1 (odd numbers on the side that the observed numbers mostly show, by default right). Ties put lettered pages first (`16, [16a], [16b], [17], 18`, the first PPM variant).
4. More missing numbers than pages (`page_number_gap`: scans missing or a number misread): pages between stay unlabelled (`page_number: null`); nothing is invented. A change of numeral system (roman front matter → arabic body) or a restart starts a new sequence as in step 2, lettering the leftovers after the previous number; a non-increasing step is flagged `page_number_sequence_break` (PPM: "nekonzistence v paginaci").
5. After the last anchor, countable pages continue `[v+1]`, … until the first binding part or insert; everything after it is lettered.

Not handled yet: leaf numbering (`1r`, `1v`), column numbers, the optional `55 [58]` form, and the evidence a page itself gives about being an inserted plate (that needs observation fields; see the follow-up issue).

The prompt (`RECONCILE_SYSTEM`, version `reconcile`) is text-only and refers to scans by position. Its input lists the numbering summary and **every** scan in one line (observed type, side, printed numbers, the label computed from the observations) with headings, TOC entries and bibliographic data indented below; values are never shortened. Example (the book of `tests/test_reconcile.py`, abridged):
```text
Book with 20 scans; "scan N" is the position in scanning order, not a page number.

Printed page numbering (computed from the observations):
- scans 5-7: roman III-V
- scans 8-15: arabic 1-6
- scans 17-18: arabic 9-10
- page_number_conflict: scan 11 shows page number 31, but the numbers on scan 10 (3) and scan 12 (5) imply 4; labelled 4 (NDK 1.1.2)
- page_number_gap: page numbers jump from 6 (scan 15) to 9 (scan 17) with 1 unnumbered page(s) in between: scans may be missing or a number misread; pages in between left unlabelled

Scans (type; side; printed page numbers; computed NDK page label):
scan 1: FrontCover; side unknown; no printed number; label [Ia]
  title: "CESTY PO ŠUMAVĚ"
scan 2: Blank; side unknown; no printed number; label [Ib]
scan 3: TitlePage; left; no printed number; label [I]
  title: "Cesty po Šumavě"
  author: "Karel Klostermann"
  publication_place: "V Praze"
...
scan 5: TableOfContents; side unknown; printed III; label III
  TOC entry level 1: "Úvod" -> "1"
  TOC entry level 1: "Kapitola II. Na horách" -> "6"
  TOC entry level 1: "Kapitola III. Domů" -> "40"
...
scan 13: Illustration; side unknown; no printed number; label [5a]
scan 14: Blank; side unknown; no printed number; label [5b]
scan 15: NormalPage; side unknown; printed 6; label 6
  heading level 1: "Kapitola II. Na horách"
```
The answer is `{scan_corrections: [{scan, page_type|null, side: keep|left|right|both|none, reason}], bibliography: [{field, value, source_scans, notes}], chapters: [{title, level, toc_scans, printed_page_reference, heading_scan, notes}], issues: [{code, message, scans}]}`, in one request with `reconcile_max_output_tokens`. With one line per scan, a 500-scan book needs about 30k characters. If the input exceeds `reconcile_max_chars` the call fails **before** the request with a clear message; splitting a book into several requests is not implemented. Observation `notes`, confidences and chapter subtitles/part numbers are not passed on. Bump `RECONCILE_PROMPT_NUMBER` when the prompt or the input format changes. Not measured on real books yet.

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
- *Run-away output:* one gpt-4.1-mini call (v2) produced 27.5k completion tokens (whitespace before a valid 470-character JSON) and took 194 s, costing as much as ~25 normal scans. Since then `max_output_tokens` (default 4000) caps every request; a capped response is rejected as truncated and retried.

Outputs vary between runs at default temperature, so these are qualitative observations, not accuracy figures; measured accuracy is the subject of #10.
