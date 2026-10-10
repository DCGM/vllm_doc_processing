# VLM document processing

Experimental **API-only vision-language model processing of digitized books**. A folder of ordered book scans is analyzed image by image using a vision model with context derived from earlier extracted pages. A final text LLM pass reconciles bibliographic metadata, page numbering, page types, sides, table of contents and chapter structure into a custom JSON.

**Status: first runnable version (MVP issues #1–#8).** `vllm-doc process` annotates a whole book end to end: scan inventory, sequential observation of every scan with bounded context from earlier scans, document-wide reconciliation and the final JSON, with a checkpoint after every scan and `--resume`. `vllm-doc evaluate` scores outputs against human-verified gold annotations and reports agreement with MetaKat/Kramerius separately (#10, [docs/EVALUATION.md](docs/EVALUATION.md)); accuracy has not been measured yet because the gold set (#26) is still being prepared. Optional experiments are issues #9 and #11.

## Scope
- Input: directory of book images (one image per scan: single page or facing-page spread) plus an order file listing the image names without extensions, one per line, in physical scan order. Filenames (usually UUIDs) carry no order.
- Provider: direct OpenAI or OpenRouter using OpenAI-compatible APIs; vision model configurable.
- Output: versioned custom JSON with bibliography, per-scan annotations, printed numbering (separate from scan index) and page labels in Czech NDK notation (`[1a]`, `[4],5`, comparable with MetaKat/Kramerius), page sides/types, chapter hierarchy, evidence, reconciled values and request metrics.
- Processing: sequential per scan with bounded summary of previously extracted data, followed by a book-level consistency pass.
- Optional OCR input (#14): existing per-scan OCR sidecars (plain text or ALTO XML) can be sent with each image as bounded, fallible text; no OCR engine is run and the image stays the primary input.
- Evaluation: offline scoring against human-verified gold annotations; MetaKat and Kramerius outputs are imported as comparators, not ground truth.
- Optional *later*: small-model-to-large-model escalation, selective image revisits.

Not in the first version: periodicals/newspapers, running OCR or coordinate-aware ALTO layout, page-region boxes, a web UI, a server, a database, or production deployment.

## Usage
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
export OPENROUTER_API_KEY=... # or OPENAI_API_KEY
# check settings, paths and images without any API call (works without a key)
vllm-doc process --input /data/scanned-book --output results/book.json \
  --provider openrouter --model '<vision-model-id>' --postprocess-model '<text-model-id>' --dry-run
# cheap trial on the first 10 scans, observations only
vllm-doc process --input /data/scanned-book --output results/book.json \
  --provider openrouter --model '<vision-model-id>' --max-pages 10 --skip-postprocess
# continue the same run with the whole book and reconcile (the 10 scans are not requested again)
vllm-doc process --input /data/scanned-book --output results/book.json \
  --provider openrouter --model '<vision-model-id>' --postprocess-model '<text-model-id>' --resume
```
`--dry-run` validates configuration, paths and the scan inventory (see [Input scans](#input-scans)) and, with `--resume`, the checkpoint (without it, it only reports whether one exists); it prints the effective (non-secret) settings and scan counts and makes no API calls. A missing API key is reported as a warning (`api_key_set: false`), so it also works without credentials. Model names are intentionally not fixed until live benchmarking.

### Processing, checkpoints and resume
1. **Inventory**: the order file and images are validated and hashed.
2. **Observation**: scans are sent one at a time, in order, each with a bounded text summary of earlier observations. A scan that still fails after retries is left without an observation and the run continues. The checkpoint is rewritten atomically after every scan.
3. **Reconciliation**: one text-only request over all observations (skipped with `--skip-postprocess`, leaving `resolved: null`).
4. **Output**: written atomically only at the end, with `run.finished_at` set; `run.warnings` lists scans that were not observed and a skipped reconciliation. Progress (one line per request and per scan, then a summary with tokens and cost) goes to stderr.

The checkpoint (default: the output path with suffix `.checkpoint.json`, e.g. `results/book.checkpoint.json`; `--checkpoint PATH`) holds the observation stage of the book in the output format (`resolved: null`, every API call so far) plus the identity it was made with. It is kept after a successful run. If it exists, a run must say what to do with it:
- `--resume` observes only scans without an observation (scans not reached yet and scans whose requests failed), then reconciles again; earlier calls stay in `run.calls`, so totals cover the whole run. Resuming a finished run therefore only repeats the reconciliation (e.g. with another `--postprocess-model`). Resume is refused (exit `2`, no request) if anything that shapes the observations changed: the order file, any image (SHA-256), provider/base URL, `model`, image settings, `max_output_tokens`, context settings (if `use_context`), `request_params`, the observation prompt, context or OCR prompt version, OCR on/off, `ocr_format`, `ocr_max_chars`, any used OCR sidecar (SHA-256, or a sidecar added or removed), or the schema version. Reconciliation settings, timeouts/retries and a larger `--max-pages` may change; a smaller one is refused.
- `--fresh` discards it and starts over.

Exit codes of `process`: `0` all scans observed and the output written; `1` the output was written but some scans were not observed (retry them with `--resume`), or a request failed so that no output was written (reconciliation failure; the checkpoint keeps everything); `2` invalid arguments, configuration, paths, images or checkpoint; `130` interrupted (Ctrl-C; the checkpoint holds every completed scan, the interrupted request is not recorded).

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
| `--ocr-dir DIR` | Optional existing OCR sidecars, sent as text with each scan's image (see [OCR sidecars](#optional-ocr-sidecars)). |
| `--ocr-format auto\|txt\|alto` | Which sidecars to use; overrides `ocr_format` (requires `--ocr-dir`). |
| `--checkpoint PATH` | Observation checkpoint, saved after every scan. Default: OUTPUT with suffix `.checkpoint.json`; must not lie inside `BOOK_DIR`. |
| `--resume` | Continue from the checkpoint (see above). |
| `--fresh` | Discard an existing checkpoint and start over. |
| `--skip-postprocess` | Skip reconciliation; the output holds observations only (`resolved: null`). |
| `--dry-run` | Validate and print effective settings; no API calls. |

### `vllm-doc observe` (prompt checks)
```bash
vllm-doc observe --input /data/scanned-book --provider openrouter --model '<vision-model-id>' \
  --scans <scan-id> <scan-id> > observations.jsonl     # or --max-pages N
```
Annotates the selected scans **independently** (no context from other scans) with the observation prompt and prints one JSON line per scan, in scan order: `scan_id`, `scan_index`, `filename`, `model`, `prompt_version`, `observation` (a `ScanObservation`, or `null`), `error` and `calls` (every attempt as a `CallRecord`). Per-call log lines and a usage/cost summary go to stderr. Each line also has `ocr` (the scan's OCR input metadata, or `null` without `--ocr-dir`). It accepts `--input`, `--order-file`, `--config`, `--provider`, `--base-url`, `--model`, `--ocr-dir`, `--ocr-format`; `max_pages` from a config file is ignored. With `--scans` only the selected images are matched and decoded, so other listed scans may be missing or corrupt; scan positions still come from the whole order file. Exit code `1` if any scan failed after retries. Paid: every selected scan is one or more requests. The prompt is described in [docs/PROMPTS.md](docs/PROMPTS.md).

### `vllm-doc evaluate` and `vllm-doc gold-template` (offline)
```bash
vllm-doc gold-template --input /data/scanned-book --book-id ID --output gold/ID.json   # manifest + empty labels
vllm-doc evaluate --gold gold/*.json --prediction results/*/book.json --name my-config \
  --comparator metakat/*.json data/*.kramerius.json --json report.json --markdown report.md
vllm-doc evaluate --gold examples/gold.example.json --prediction examples/annotated_book.example.json
```
`gold-template` hashes every listed scan and writes a gold file (the corpus manifest, with optional `--txt-dir`/`--alto-dir` OCR sidecars `<scan_id>.txt|.xml`) in which every label is `not_reviewed`; it never overwrites an existing file. `evaluate` scores the observed and resolved layers of annotated-book outputs (`--prediction`, one configuration, at most one output per book) and MetakatIO / `*.kramerius.json` imports (`--comparator`) against the gold files, only on reviewed fields, and reports the agreement of this tool with each comparator separately. Predictions are matched to gold books by shared scan IDs; scans whose image SHA-256 differs from the gold are not scored. Without `--gold`, only agreement is reported. Gold format, vocabulary mapping (NDK vs MetaKat page types, MetaKat `single_page`), metrics and report contents are described in [docs/EVALUATION.md](docs/EVALUATION.md). Exit codes: `0` report written, `2` invalid inputs, `1` report not writable.

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
| `image_detail` | no | `auto`; image `detail` hint sent with each scan (`auto`, `low`, `high`); `high` may help with small print at higher cost; providers other than OpenAI may ignore it |
| `request_timeout_s` | no | `180`; timeout of one request attempt in seconds |
| `max_retries` | no | `3`; retries after rate limits (429), timeouts/connection errors, 408/409/5xx and responses that fail schema validation |
| `max_output_tokens` | no | `4000`; output token cap of each request, sent as `max_completion_tokens` (openai) or `max_tokens` (openrouter); includes reasoning tokens, so raise it for high reasoning effort. A truncated answer is rejected and retried, so the cap bounds the cost of run-away output. `null` = no cap |
| `use_context` | no | `true`; send a bounded text summary of earlier scans with each scan; `false` observes every scan on its own (for comparing the effect of context) |
| `context_recent_scans` | no | `5`; number of earlier scans summarized one line each in the text context sent with the next scan (0–50, see [docs/PROMPTS.md](docs/PROMPTS.md#context-from-earlier-scans-issue-6)) |
| `context_max_chars` | no | `2000`; hard limit on the length of that context text (minimum 200); oldest scan lines are dropped first |
| `ocr_format` | no | `auto`; with `--ocr-dir`, use `<scan_id>.txt` (UTF-8 text), `<scan_id>.xml` (ALTO) or either (`auto`; both present is an error) |
| `ocr_max_chars` | no | `6000`; longest OCR text sent with one scan (minimum 200); a longer text keeps its start and end (where running heads and page numbers are) with a visible omission marker in between |
| `reconcile_max_chars` | no | `100000`; longest allowed text input of the reconciliation request (minimum 1000). A longer book fails before the request with a clear error (splitting is not implemented); raise it if the postprocess model's context allows |
| `reconcile_max_output_tokens` | no | `16000`; output token cap of the reconciliation request (replaces `max_output_tokens` there, since a long chapter list needs more); `null` = no cap |
| `request_params` | no | `{}`; extra request body fields, e.g. `{"temperature": 0, "reasoning_effort": "low"}`; with OpenRouter also `provider` routing preferences. `model`, `messages`, `response_format`, `stream`, `n`, `tools`, `tool_choice`, `max_tokens`, `max_completion_tokens` are rejected |

### Input scans
- The order file lists image names without extensions, one per line, in physical scan order; surrounding whitespace and blank lines are ignored. Empty order files, duplicate names and names containing path separators are errors.
- Each listed name must match exactly one file `<name><ext>` directly in `BOOK_DIR` (non-recursive), with `ext` one of `.jpg .jpeg .png .webp .tif .tiff` (case-insensitive). Missing, ambiguous (`a.jpg` + `a.png`) or unsupported-only (`a.gif`) matches are errors, all reported at once. Supported images that are not listed are reported as a warning and ignored; other files (such as `order.txt`) are ignored.
- Every selected image is hashed (SHA-256) and fully decoded; corrupt and multi-frame images are errors, as are images above Pillow's decompression-bomb limit (≈179 Mpx; a warning is printed above ≈89 Mpx), which bounds decoding memory to well under 1 GB per scan. With `--max-pages` the whole order file is still matched, but only the first N images are decoded and processed.
- Recorded width/height are those of the upright image (EXIF orientation applied).
- Uploads are prepared in memory; originals are never modified. JPEG/PNG/WebP files in RGB or grayscale that need no rotation or downscaling are sent byte-for-byte (including any embedded metadata). Everything else — TIFF, CMYK, 16-bit, transparency, EXIF-rotated or larger than `image_max_side` — is converted to 8-bit RGB/grayscale (high-bit-depth grayscale is scaled by the smallest bit depth holding its maximum, so 12-bit data in 16-bit TIFFs keeps its white), rotated upright, downscaled (never upscaled) and re-encoded as `image_format` without metadata. File names and paths are never part of the upload.
- `image_max_side` trades legibility of small print against upload size and cost. The default 2048 is a starting point, not a measured optimum: some providers downscale large images internally anyway, while others bill and see more detail at higher resolution. Tune it per model (or `null`) during benchmarking.

### Optional OCR sidecars
- `--ocr-dir DIR` points to existing OCR results of the same book, one file per scan, matched by the exact scan ID from the order file: `<scan_id>.txt` (UTF-8, optional BOM) or `<scan_id>.xml` (ALTO, any namespace version or none), directly in `DIR`, extension case-insensitive. The directory may be `BOOK_DIR` itself. No OCR is run; source files are only read.
- A scan without a sidecar is sent as image only (counted in a startup line on stderr and in `run.warnings`). An ambiguous match (`a.txt` + `a.xml` with `ocr_format: auto`, or `a.txt` + `a.TXT`), an unreadable file, invalid UTF-8, malformed XML or a non-ALTO XML stops the run before any request (exit `2`), listing every problem; nothing is silently substituted.
- ALTO is reduced to text in document order: one line per `TextLine` (`String/@CONTENT` joined by spaces, `HYP` appended), a blank line between `TextBlock`s; coordinates, styles and `SUBS_CONTENT` are ignored. Both formats get normalized line breaks and at most one blank line in a row.
- Each observation request carries the current image, the bounded context from earlier observations and the OCR text of **that scan only**, cut to `ocr_max_chars`. The prompt calls the OCR fallible and the image authoritative ([docs/PROMPTS.md](docs/PROMPTS.md#ocr-text-issue-14)). OCR of earlier scans is never added to later requests (the context is built from observations only).
- The output records `source.ocr_directory`, per scan `ocr` (sidecar filename, format, SHA-256, size, normalized and sent length, truncation, or `missing`) and `run.prompt_versions.ocr`; the OCR text itself is not stored in the output, checkpoint or logs. Observations are not attributed to OCR. Without `--ocr-dir` requests and output are as before (image-only).

Credentials are read **only** from the environment: `OPENAI_API_KEY` for `openai`, `OPENROUTER_API_KEY` for `openrouter`; an `api_key` entry in the config file is an error. A custom `base_url` still uses the selected provider's key variable. If `--provider` overrides a different provider from the config file, the file's `base_url` is discarded (the new provider's default is used unless `--base-url` is also given), so a key is never sent to another provider's endpoint. Exit codes: `0` success, `2` invalid arguments, configuration or paths, `1` runtime failure.

### API requests
- Every request uses Chat Completions with a strict `json_schema` `response_format` generated from a Pydantic model; the answer is validated locally again. Images are sent as base64 data URLs, one per request.
- With OpenRouter, `provider.require_parameters` is always set to `true` (merged into any `request_params.provider` preferences), so requests are only routed to endpoints that support structured outputs instead of silently dropping the schema. A model/route without vision or `json_schema` support fails immediately with an explicit error (HTTP 400/404), as do authentication errors; those are not retried.
- `require_parameters` applies to **every** request parameter, not only the schema: anything in `request_params` that a route does not support (e.g. `temperature` for many reasoning models, `top_k` for OpenAI models) removes that route, and if none is left the request fails with HTTP 404 "No endpoints found that can handle the requested parameters". Keep `request_params` minimal and check the model's supported parameters on OpenRouter.
- Transient failures and invalid answers (bad JSON, schema mismatch, refusal, truncated output) are retried up to `max_retries` times with exponential backoff (honouring `Retry-After`). Each attempt — including failed but billed ones — is recorded as a `CallRecord` with tokens, provider-reported cost (OpenRouter only), latency and served model, and logged as one line without prompts, image data or keys.

## Development
```bash
pip install -e '.[dev]'
pytest            # offline tests only
# opt-in paid smoke test of one vision + structured-output request:
VLLM_DOC_LIVE_TEST=1 VLLM_DOC_LIVE_PROVIDER=openrouter VLLM_DOC_LIVE_MODEL='<vision-model-id>' pytest tests/test_llm_live.py -s
# (passed with openai/gpt-4.1-nano via OpenRouter, ~$0.00002)
```
The output JSON format is defined by Pydantic models in `src/vllm_doc_processing/models.py`, documented in [docs/OUTPUT_SCHEMA.md](docs/OUTPUT_SCHEMA.md), with a validated example in [examples/annotated_book.example.json](examples/annotated_book.example.json).

## Design and contributions
- [Implementation plan and backlog](docs/IMPLEMENTATION_PLAN.md)
- [JSON output schema and MetaKat mapping](docs/OUTPUT_SCHEMA.md)
- [Evaluation: gold format, imports, metrics](docs/EVALUATION.md)
- [Prompts](docs/PROMPTS.md)
- [Agent development instructions](AGENTS.md)
- [MetaKat](https://github.com/DCGM/MetaKat) — basis for comparison and bibliographic field vocabulary
- [NDK rules for describing monographs](https://standardy.ndk.cz/ndk/standardy-digitalizace/ppp_mono_2.4_final.pdf/at_download/file) — page types and page-number notation

**Privacy and cost:** Book images are uploaded to a selected external API provider, which may have its own retention, billing and routing policies. Do not use sensitive/rights-restricted images without authorization; never commit credentials, scanned datasets, or raw API payloads.
