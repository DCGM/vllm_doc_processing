# VLM document processing

Experimental **API-only vision-language model processing of digitized books**. A folder of ordered book scans is analyzed image by image using a vision model with context derived from earlier extracted pages. A final text LLM pass reconciles bibliographic metadata, page numbering, page types, sides, table of contents and chapter structure into a custom JSON.

**Status: planning.** No runnable CLI implementation exists yet. Start with the dependency-ordered [issues](https://github.com/DCGM/vllm_doc_processing/issues) (#1–#8 for MVP); code follows in separate PRs.

## Scope
- Input: directory of naturally sorted book images, one image per scan (single page or facing-page spread).
- Provider: direct OpenAI or OpenRouter using OpenAI-compatible APIs; vision model configurable.
- Output: versioned custom JSON with bibliography, per-scan annotations, printed numbering (separate from scan index), page sides/types, chapter hierarchy, evidence, reconciled values and request metrics.
- Processing: sequential per scan with bounded summary of previously extracted data, followed by a book-level consistency pass.
- Optional *later*: small-model-to-large-model escalation, selective image revisits, MetaKat benchmark.

Not in the first version: periodicals/newspapers, full OCR/ALTO processing, page-region boxes, a web UI, a server, a database, or production deployment.

## Planned usage (not executable yet)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
export OPENROUTER_API_KEY=... # or OPENAI_API_KEY
vllm-doc process --input /data/scanned-book --output /data/book.json \
  --provider openrouter --model '<vision-model-id>' \
  --postprocess-model '<text-model-id>'
```
CLI flags, configuration rules and working examples will be refined by issue #2 and #8. Model names are intentionally not fixed until live benchmarking.

## Design and contributions
- [Implementation plan and backlog](docs/IMPLEMENTATION_PLAN.md)
- [Proposed JSON output schema and MetaKat mapping](docs/OUTPUT_SCHEMA.md)
- [Agent development instructions](AGENTS.md)
- [MetaKat](https://github.com/DCGM/MetaKat) — basis for comparison and page-type/metadata vocabulary

**Privacy and cost:** Book images are uploaded to a selected external API provider, which may have its own retention, billing and routing policies. Do not use sensitive/rights-restricted images without authorization; never commit credentials, scanned datasets, or raw API payloads.
