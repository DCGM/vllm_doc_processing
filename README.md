# VLM document processing

Experimental **API-only vision-language model processing of digitized books**. A folder of ordered book scans is analyzed image by image using a vision model with context derived from earlier extracted pages. A final text LLM pass reconciles bibliographic metadata, page numbering, page types, sides, table of contents and chapter structure into a custom JSON.

**Status: early implementation.** The output data model (`src/vllm_doc_processing/models.py`, issue #1) the CLI/configuration scaffold (issue #2) and the scan inventory/image preparation (issue #3) exist; the CLI can validate a run with `--dry-run` but does not process scans yet. Start with the dependency-ordered [issues](https://github.com/DCGM/vllm_doc_processing/issues) (#1–#8 for MVP); code follows in separate PRs.

## Scope
- Input: directory of book images (one image per scan: single page or facing-page spread) plus an order file listing the image names without extensions, one per line, in physical scan order. Filenames (usually UUIDs) carry no order.
- Provider: direct OpenAI or OpenRouter using OpenAI-compatible APIs; vision model configurable.
- Output: versioned custom JSON with bibliography, per-scan annotations, printed numbering (separate from scan index), page sides/types, chapter hierarchy, evidence, reconciled values and request metrics.
- Processing: sequential per scan with bounded summary of previously extracted data, followed by a book-level consistency pass.
- Optional *later*: small-model-to-large-model escalation, selective image revisits, MetaKat benchmark.

Not in the first version: periodicals/newspapers, full OCR/ALTO processing, page-region boxes, a web UI, a server, a database, or production deployment.

## Usage
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
export OPENROUTER_API_KEY=... # or OPENAI_API_KEY
vllm-doc process --input /data/scanned-book --output /data/book.json \
  --provider openrouter --model '<vision-model-id>' \
  --postprocess-model '<text-model-id>' --dry-run
```
`--dry-run` validates configuration, paths and the scan inventory (see [Input scans](#input-scans)), prints the effective (non-secret) settings and scan counts and makes no API calls; a missing API key is reported as a warning (`api_key_set: false`), so it also works without credentials. Without `--dry-run` the command currently exits with an error: processing lands with issues #3–#8. Model names are intentionally not fixed until live benchmarking.

### `vllm-doc process` options
| Flag | Meaning |
|---|---|
| `--input BOOK_DIR` | Directory with the scans of one book (required). |
| `--order-file PATH` | Image names without extensions, one per line, in scan order. Default `BOOK_DIR/order.txt`. |
| `--output PATH` | Annotated book JSON to write (required); must not lie inside `BOOK_DIR`. |
| `--config PATH` | Optional JSON configuration file (see below). |
| `--provider openai\|openrouter` | API provider. |
| `--base-url URL` | Override the provider's default OpenAI-compatible base URL. |
| `--model ID` | Vision model for per-scan observation. |
| `--postprocess-model ID` | Text model for reconciliation; defaults to `--model`. |
| `--max-pages N` | Process only the first N scans of the order file (cheap experiments). |
| `--dry-run` | Validate and print effective settings; no API calls. |

### Configuration
Precedence: **built-in defaults < `--config` JSON file < command-line flags**. The config file is a flat JSON object whose keys match the settings below; unknown keys are rejected so typos fail loudly. See [examples/config.example.json](examples/config.example.json).

| Key | Required | Default |
|---|---|---|
| `provider` | yes (`openai` or `openrouter`) | — |
| `model` | yes | — |
| `postprocess_model` | no | same as `model` |
| `base_url` | no | `https://api.openai.com/v1` (openai), `https://openrouter.ai/api/v1` (openrouter) |
| `max_pages` | no | `null` (all scans) |
| `image_max_side` | no | `2048` px; uploads are downscaled so the longest side fits (`null` = never downscale, minimum 256) |
| `image_format` | no | `jpeg` (quality 90); encoding of converted or downscaled uploads, or `png` (lossless, larger) |

### Input scans
- The order file lists image names without extensions, one per line, in physical scan order; surrounding whitespace and blank lines are ignored. Empty order files, duplicate names and names containing path separators are errors.
- Each listed name must match exactly one file `<name><ext>` directly in `BOOK_DIR` (non-recursive), with `ext` one of `.jpg .jpeg .png .webp .tif .tiff` (case-insensitive). Missing, ambiguous (`a.jpg` + `a.png`) or unsupported-only (`a.gif`) matches are errors, all reported at once. Supported images that are not listed are reported as a warning and ignored; other files (such as `order.txt`) are ignored.
- Every selected image is hashed (SHA-256) and fully decoded; corrupt and multi-frame images are errors, as are images above Pillow's decompression-bomb limit (≈179 Mpx; a warning is printed above ≈89 Mpx), which bounds decoding memory to well under 1 GB per scan. With `--max-pages` the whole order file is still matched, but only the first N images are decoded and processed.
- Recorded width/height are those of the upright image (EXIF orientation applied).
- Uploads are prepared in memory; originals are never modified. JPEG/PNG/WebP files in RGB or grayscale that need no rotation or downscaling are sent byte-for-byte (including any embedded metadata). Everything else — TIFF, CMYK, 16-bit, transparency, EXIF-rotated or larger than `image_max_side` — is converted to 8-bit RGB/grayscale (high-bit-depth grayscale is scaled by the smallest bit depth holding its maximum, so 12-bit data in 16-bit TIFFs keeps its white), rotated upright, downscaled (never upscaled) and re-encoded as `image_format` without metadata. File names and paths are never part of the upload.
- `image_max_side` trades legibility of small print against upload size and cost. The default 2048 is a starting point, not a measured optimum: some providers downscale large images internally anyway, while others bill and see more detail at higher resolution. Tune it per model (or `null`) during benchmarking.

Credentials are read **only** from the environment: `OPENAI_API_KEY` for `openai`, `OPENROUTER_API_KEY` for `openrouter`; an `api_key` entry in the config file is an error. A custom `base_url` still uses the selected provider's key variable. If `--provider` overrides a different provider from the config file, the file's `base_url` is discarded (the new provider's default is used unless `--base-url` is also given), so a key is never sent to another provider's endpoint. Exit codes: `0` success, `2` invalid arguments, configuration or paths, `1` runtime failure.

## Development
```bash
pip install -e '.[dev]'
pytest            # offline tests only
```
The output JSON format is defined by Pydantic models in `src/vllm_doc_processing/models.py`, documented in [docs/OUTPUT_SCHEMA.md](docs/OUTPUT_SCHEMA.md), with a validated example in [examples/annotated_book.example.json](examples/annotated_book.example.json).

## Design and contributions
- [Implementation plan and backlog](docs/IMPLEMENTATION_PLAN.md)
- [JSON output schema and MetaKat mapping](docs/OUTPUT_SCHEMA.md)
- [Agent development instructions](AGENTS.md)
- [MetaKat](https://github.com/DCGM/MetaKat) — basis for comparison and page-type/metadata vocabulary

**Privacy and cost:** Book images are uploaded to a selected external API provider, which may have its own retention, billing and routing policies. Do not use sensitive/rights-restricted images without authorization; never commit credentials, scanned datasets, or raw API payloads.
