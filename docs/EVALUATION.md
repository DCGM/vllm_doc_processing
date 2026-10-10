# Evaluation: gold annotations, imports and scoring (issue #10)

`vllm-doc evaluate` scores predictions offline (no API calls) against **human-verified gold annotations** and, separately, reports **agreement** between this tool and MetaKat/Kramerius outputs. MetaKat and Kramerius are predictions/comparators, never ground truth: they are scored against gold like any other system, and agreement with them is never called accuracy.

Sources of truth: `src/vllm_doc_processing/gold.py` (gold format), `predictions.py` (imports and vocabulary mapping), `evaluation.py` (scoring and reports). Example: [`examples/gold.example.json`](../examples/gold.example.json), which annotates [`examples/annotated_book.example.json`](../examples/annotated_book.example.json).

```bash
# 1. manifest + empty labels for one book (hashes every listed image; never overwrites)
vllm-doc gold-template --input BOOK_DIR [--order-file PATH] --book-id ID --source URL \
  [--txt-dir DIR] [--alto-dir DIR] --output gold/ID.json
# 2. fill in labels by hand (status per field, see below)
# 3. score one configuration (one output per book) and optional comparators
vllm-doc evaluate --gold gold/*.json --prediction results/*/book.json --name gpt-mini-image-only \
  --comparator metakat/*.json data/*.kramerius.json --json report.json --markdown report.md
# agreement only, before gold labels exist:
vllm-doc evaluate --prediction results/*/book.json --comparator data/*.kramerius.json
```
Without `--json`/`--markdown` the Markdown report goes to stdout. Reports are written atomically (missing directories are created). Exit code `2` for unreadable or invalid inputs, `1` if a report cannot be written. The report records the SHA-256 of exactly the gold and prediction bytes that were parsed. Different configurations (models, prompts, image-only vs TXT vs ALTO) are evaluated by separate invocations on the same gold files; repeated runs and their variance are outside the tool.

## Gold file (`gold_version` 1)
One JSON file per book. It is also the **corpus manifest**: it binds the labels to scan IDs, scan order and image hashes, and records available OCR sidecars.

| Key | Meaning |
|---|---|
| `gold_version` | `"1"` |
| `book_id` | Any stable ID (e.g. the Kramerius document UUID). Predictions are matched to gold books by **shared scan IDs**, not by book ID. |
| `dataset_version`, `source`, `reviewers[]`, `notes` | Data version, origin of the scans, annotators, remarks. |
| `scan_count` | Number of scans of the whole book (length of its order file). |
| `scans[]` | All scans or a selected subset, sorted by `scan_index`: `scan_id` (order-file name), `scan_index` (zero-based position in the whole order file), `filename?`, `image_sha256` (required; labels are valid only for this image), `txt?`/`alto?` (`{filename, sha256}` of an OCR sidecar), labels `page_type`, `side`, `leaf`, `printed_numbers`, `page_number`, `notes`. |
| `bibliography` | `{field: label}` for `BiblioField` names (`title`, `author`, `publisher`, …); every value is a **list** of strings (one element for single-valued fields). Missing fields are not reviewed. |
| `structure` | Label whose value is the chapter list `[{title, level, start_scan_id?, printed_page_reference?}]`. Reviewable only when every scan of the book is listed (`len(scans) == scan_count`). |

Every label is `{status, value?, alternatives[], reviewer?, notes?}`:

| `status` | Meaning | Scored as |
|---|---|---|
| `verified` | `value` (non-null, non-empty) is correct. | prediction must equal `value` |
| `absent` | Verified that there is no value (no printed number, no subtitle, no chapters). | prediction must be null/empty |
| `not_applicable` | The field has no meaning here (e.g. side of a spine). | prediction must be null (the output schema uses null for "not applicable") |
| `ambiguous` | Any of `alternatives` is acceptable (`null` allowed as an alternative). Without alternatives: counted, not scored. | prediction must equal one alternative |
| `not_reviewed` (default) | Not checked. | **never scored**, only counted |

A label that is omitted is `not_reviewed`, so a template scores nothing until labels are filled in. Only `verified` may carry a `value`, only `ambiguous` may carry `alternatives`.

Scan labels: `page_type` (NDK `PageType`), `side` (`left|right|both`), `leaf` (`book_block|plate|binding|loose`), `printed_numbers` (numbers **exactly as printed**, left to right, e.g. `["xii"]`, `["4", "5"]`; not page labels), `page_number` (**NDK page label** of the scan as in Kramerius `ORDERLABEL`/MetaKat `pageNumber`, e.g. `[1a]`, `[4],5`). Printed numbers and page labels are different concepts and are scored separately. Spreads are labeled at scan level (no per-page gold types yet).

## Predictions
Formats are detected from the JSON content:

| Input | Role | Provides |
|---|---|---|
| Annotated book (`schema_version` `"0.2"`; other versions are rejected) via `--prediction` | system, two **layers**: `observed` (`scans[].observation`) and `resolved` (if reconciled) | observed: `page_type`, `side`, `leaf`, printed numbers. resolved: `page_type`, `side`, `leaf`, `page_number`, bibliography, structure. Plus run provenance. |
| MetakatIO JSON (`batch_id`, `elements`) via `--comparator` | comparator `metakat` | per page `pageType`, `side`, `pageNumber`; bibliography from the (at most one) `volume` and `title` element (volume preferred per field); chapters (`title`, level from the `parent_id` chain, `pageIndexStart`, `pageNumber` as TOC reference). Scan IDs: stems of `page_to_image_mapping` file names, else page UUIDs. Scan index and chapter `pageIndex*` refer to `pageIndex` (or `batch_index` if pages have none). Null values in MetaKat tuples are dropped; a chapter `parent_id` cycle is an error. |
| `*.kramerius.json` of `scripts/kramerius_order.py` via `--comparator` | comparator `kramerius` | per page `page_type`, `page_number` (NDK label). No side, leaf, printed numbers, bibliography or structure. |

Each source/layer declares the scan fields it outputs (the "Provides" column; `predictions.py` `*_FIELDS`). Other fields are listed as `fields_not_provided` and not scored. A scan whose observation failed provides no values for the declared fields and counts as `no_prediction` (lowering coverage, even if every scan failed), which differs from a predicted `null` ("unknown"/"none").

### Vocabulary mapping (explicit, never silently equal)
- **Page types:** case-insensitive match to NDK values, so MetaKat PascalCase (`TitlePage`, `FlyLeaf`) and Kramerius' mixed spellings map to `titlePage`, `flyleaf`. Values without an NDK page type (`Abstract`, `Obituary`, `CalibrationTable`, `CustomInclude`, `FragmentsOfBookbinding`, unknown strings) become **incomparable**: counted, listed under `incomparable_values`, never correct. `null` is an "unknown" prediction.
- **Side:** MetaKat `left`/`right` map directly; `single_page` has no equivalent in `left|right|both|null` and is incomparable.
- **Page labels:** whitespace removed, otherwise exact (brackets and roman case matter, as in NDK). On import from Kramerius/MetaKat the older notation for unprinted numbers in round brackets is rewritten to NDK square brackets per page (`(171a)` → `[171a]`, `(4),5` → `[4],5`); leaf numbering (`A 1r`) is kept and does not match labels of this tool.
- **Empty lists** (no printed numbers, no bibliography values) are the same as `null`, also inside `alternatives`.
- **Printed numbers:** `printed_numbers_exact` compares the multiset of raw strings; `printed_numbers_normalized` strips brackets/dots/dashes, drops leading zeros and canonicalizes roman numerals (`[xii].` → `XII`).
- **Bibliography:** compared as sets of normalized strings (Unicode NFC, case-folded, whitespace collapsed, outer punctuation removed; diacritics kept).
- **Chapters:** gold chapters are matched to predicted chapters by normalized title (first unused equal title).

## Scoring
Per field, each gold item gets one outcome:

| Outcome | In accuracy denominator | Meaning |
|---|---|---|
| `correct` | yes | prediction acceptable |
| `wrong_value` | yes | both have a value, different |
| `missing_value` | yes | gold has a value, prediction null |
| `spurious_value` | yes | gold expects null (`absent`/`not_applicable`), prediction has a value |
| `incomparable` | no | prediction has no equivalent in the vocabulary |
| `no_prediction` | no | the system gave no record for the scan/field |
| `ambiguous_unscorable` | no | `ambiguous` without alternatives |
| `scan_missing` | no | gold scan/book not in the prediction |
| `hash_mismatch` | no | gold and prediction `image_sha256` differ: labels may not belong to that image |
| `document_incompatible` | no | bibliography of a prediction that fails the document check below |

`eligible` = all outcomes above; `scored` = the first four; `accuracy` = correct / scored; `coverage` = scored / eligible. `not_reviewed` items are not eligible; `by_reference_status` breaks results down by gold status (e.g. how often `absent` printed numbers were correctly left empty). Accuracy/coverage are `null` when nothing was scored. Hashes are checked when both sides have them (MetaKat/Kramerius have none: `hash_unchecked`); `scan_index_differs` counts scans whose order differs.

**Document check** (`documents[]`, per matched book): bibliography and structure are scored only if the prediction covers the **whole book** (exactly `scan_count` scans, including every gold scan) and **no** gold scan has a different image hash. Otherwise bibliography fields get `document_incompatible` and structure `scored: false` with that reason (`partial prediction`, e.g. a `--max-pages` run; `scan count differs`; `image hash mismatch`). `hashes_checked` counts scans whose image hashes were present on both sides and compared; `hashes_total` is the book's `scan_count`. `hashes_verified` is true **only if all scans of the book have matching reference and prediction hashes**. A partial gold manifest can still permit document-level scoring, but `hashes_verified` remains false (e.g. 10/100 checked), even if all gold-listed hashes match. When hashes are unavailable (MetaKat, Kramerius), results are still scored with `hashes_checked: 0` and `hashes_verified: false`; image identity could not be verified. Per-scan fields are scored for the scans present regardless.

Structure (only `verified`, `absent` or `not_applicable` gold structure of a complete book, prediction passing the document check): title recall/precision, start-scan accuracy for matched chapters with a gold start, level accuracy, TOC reference accuracy (normalized numbers), recall/precision of the set of chapter start scans, and lists of unmatched titles and wrong starts/references. For a book verified to have no chapters, `no_chapters_correct` says whether the prediction has none. Repeated chapter titles are matched to the first unused equal title, which can pair them wrongly (known limitation).

**Agreement** (`agreement[]`): each `--prediction` layer against each comparator, over the comparator's scans, for `page_type`, `side`, `leaf`, `page_number` and bibliography. The comparator's non-null values act as reference; its nulls are not compared, its incomparable values are counted as `incomparable`. Bibliography agreement uses the same document check, with the comparator's page list as the whole book. Reported as `agreeing`/`agreement`, never accuracy.

## Report
JSON (`--json`) and Markdown (`--markdown`) are deterministic for the same inputs (no timestamps):
- `evaluator_version`; `inputs` with paths, SHA-256 of every gold and prediction file, book IDs, gold/dataset versions;
- `gold_status_counts` per field and status;
- `provenance` per system: per output file vision/postprocess model, provider, served models, prompt/context/schema versions, image policy, context settings, OCR mode (`image-only` unless the run parameters contain `ocr*` settings), start/finish/duration, summed latency, usage/cost totals, retry attempts, error rate, unobserved scans, reconciliation warnings; plus summed `usage`;
- `incomparable_values` per system and field;
- `accuracy[]` per system and layer: `scans` summary, `documents` (document checks), `fields`, `bibliography`, `structure`, and every non-correct item in `errors` (`book_id`, `scan_id`, `scan_index`, `field`, `outcome`, `reference_status`, `reference`, `predicted`); Markdown shows the first 100 per section;
- `agreement[]` with `disagreements`.
