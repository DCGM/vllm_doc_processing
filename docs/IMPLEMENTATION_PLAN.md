# Implementation plan: VLM book annotation experiment

## Goal and boundaries
Determine whether remotely hosted vision-language models can approximate MetaKat-level annotations for digitized **books** in an inexpensive, small Python CLI. Input is a directory of ordered scans; output is one custom JSON. This is a feasibility experiment, **not** a replacement for MetaKat or a generalized document AI platform.

In scope: monographs/volumes; bibliographic fields, page-type labels, scan side (left/right/both), observed printed numbers, logical chapter hierarchy and TOC references; structured evidence and provenance. Out of scope initially: periodicals, newspapers, articles, full text transcription/running OCR, coordinate-aware ALTO layout, bounding-box detection, METS/MODS export, UI, database, distributed workers and training.

## Architectural decisions
1. **Python >=3.11, CLI only.** Keep install dependencies small: `openai`, `pydantic`, `Pillow`. Use `argparse`; optional testing tools in a dev extra.
2. **OpenAI-compatible Chat Completions** with images supplied as base64 data URLs. Use the official OpenAI SDK for `openai` and `openrouter` with configurable base URL. Prefer JSON-schema structured response; validate locally as well. OpenRouter should specify `provider.require_parameters=true` to avoid routes that ignore requested features. Model support varies: no hard-coded model ID.
3. **Image per request, ordered sequential processing.** Only the current image is attached; pass a bounded textual `BookContext` synthesized from prior observations (summary of candidates, prior N pages, page numbering sequence, chapter state). Maintain all raw observations locally.
4. **Two-stage inference.** First collect per-scan observations, then run a text-only document-wide LLM reconciliation over compact structured records. Support hierarchical chunking if the input exceeds its context budget; never silently drop pages. Apply deterministic invariant checks afterward.
5. **Uncertain data is not fabricated.** The extracted raw evidence, original model calls and any later reconciled values are distinguishable. Null ≠ blank page ≠ missing printed number.
6. **Budget and auditability.** Capture per-call model/provider, stage, token use, latency, estimated/returned cost where available; retry safely and checkpoint after each successful page. Live API calls are opt-in in tests.
7. **Separate optional experiments.** Adaptive small-to-large VLM escalation, and visual reinspection during postprocessing, only after measuring a simple one-model baseline.

## Pipeline
```text
book_dir/ (image files) + order file (names without extensions, in scan order)
  -> manifest (ordered scan IDs, checksums, dimensions)
  -> for each scan in order:
       image + bounded context from previous structured observations
       -> vision model -> schema validation -> raw observation
       -> update context and atomic checkpoint
  -> all raw page observations
       -> document-level reconciliation LLM (text-only by default)
       -> deterministic consistency checks / warning list
  -> final versioned annotated-book JSON + run/usage metadata
```

### Initial CLI shape
```bash
pip install -e .
export OPENROUTER_API_KEY=...  # alternatively OPENAI_API_KEY
vllm-doc process --input ./my-book --output ./results/book.json \
  --provider openrouter --model '<vision-model-id>' \
  --postprocess-model '<text-model-id>'
```
Implemented by #2: `--config config.json` (precedence defaults < file < CLI), `--order-file` (default `BOOK_DIR/order.txt`), `--base-url`, `--dry-run`; by #3: `--max-pages 20`, `image_max_side`/`image_format` settings; by #4: `image_detail`, `request_timeout_s`, `max_retries`, `request_params` settings; by #5: `max_output_tokens` setting and `vllm-doc observe` (independent per-scan annotation for prompt checks); by #6: `use_context`, `context_recent_scans`, `context_max_chars` settings (sequential observation with bounded context, `pipeline.observe_book`); by #7: `reconcile_max_chars`, `reconcile_max_output_tokens` settings (document-wide reconciliation, `reconcile.reconcile_book`); by #8: end-to-end `process` with `--checkpoint`, `--resume`, `--fresh`, `--skip-postprocess` (checkpoint after every scan; resume refused when images, order file or observation settings changed); by #10: offline `evaluate` (gold accuracy, comparator agreement) and `gold-template` (docs/EVALUATION.md). Expected later: `--escalation-model`. Actual supported flags are tracked in README and `--help` as issues land; new settings are added to `config.Config` by the issue that needs them.

### Document reasoning specifics
- Scan index is an ordered physical image index, starting at zero. Printed number is separately observed and may use Arabic or Roman numerals, omit numbers, repeat, or restart.
- `side=both` means a two-page opening scanned into one image. Represent left and right printed numbers separately. A cover, blank image or single scan might not have a meaningful side: use null.
- Page type uses the NDK page-type vocabulary (see OUTPUT_SCHEMA; mapped to MetaKat's older labels for evaluation). If a spread contains different types, allow each visible leaf page its own classification or optional subpage labels.
- The per-image prompt should detect bibliography on title/colophon pages, TOC entries, chapter headings, printed numbers and legible relevant evidence; avoid full OCR.
- Text-only postprocessing should consolidate bibliographic evidence, align TOC entries with destination chapters, infer chapter start/end scan IDs, detect conflicting/missing page-number sequences and explain uncertainty.
- Page labels, page types and sides follow the Czech NDK rules for describing monographs ([Pravidla pro popis monografií](https://standardy.ndk.cz/ndk/standardy-digitalizace/ppp_mono_2.4_final.pdf/at_download/file)), which also underlie the MetaKat/Kramerius ground truth: printed numbers plain, computed numbers in brackets, pages outside the count lettered (`[1a]`, `[26a]`). Reconciliation computes the labels deterministically and may correct clearly wrong page types and sides.
- Distinguish observed facts, derived facts and unresolved conflicts. Keep postprocessing edits in a separate reconciled view or explicit change record.

## Milestones and GitHub issues
**M1 — Usable one-model baseline (in dependency order):**
- #1 Versioned annotated-book JSON schema; MetaKat parity
- #2 Minimal CLI and config
- #3 Deterministic scan inventory and image preprocessing
- #4 OpenAI/OpenRouter structured-output API adapter
- #5 Current-image extraction prompt and result validation
- #6 Sequential state and bounded prior-page context
- #7 Document-wide reconciliation and deterministic validation
- #8 End-to-end CLI, checkpoints/resume and final JSON

**M2 — Experimental comparisons:**
- #9 Optional small-to-large model escalation
- #10 Evaluation framework (gold annotations, MetaKat/Kramerius as comparators) and benchmark
- #11 Selective postprocessing image revisits

**Phase 2:** #14 Optional pre-existing OCR sidecars (TXT / ALTO) as bounded text next to the image.

Prefer delivering #1–#6 before polishing optional features, and record a run on a small real book at #8. Implementation issues are autonomous work units and should use separate, small PRs.

## Evaluation
Start with 2–5 different books with a modest number of annotated representative scans, then expand only if results warrant it. Preserve at least one baseline config/model. Measure page type and side accuracy, printed page numbers (exact match including roman/arabic and absent numbering), bibliography correctness, chapter/TOC alignment, number of unresolved conflicts, invalid JSON/repair rate, request latency and estimated or reported USD cost. Compare initial per-page observations and final reconciled output separately. Compare with MetaKat annotations when available, accounting for ontology mismatches (such as its `single_page` side versus this project's `both`).

## Risks and mitigations
- **Unbounded context:** send a rolling summary, not complete previous pages.
- **Vision model confidence:** self-reported probabilities may be poorly calibrated; evaluate them rather than treating them as truth.
- **OpenRouter provider compatibility:** use a route with vision + JSON-schema support, validate responses and report compatibility failures.
- **Data leakage/cost:** user chooses provider; warn about uploading scans to third parties; never log base64 or secrets; support page/request budget.
- **Very long books:** don't force the full book into one prompt; chunk and reconcile with explicit coverage accounting.
- **Model drift:** pin model IDs and record prompts, parameters and timestamps in run metadata; historical reproductions may not be perfectly deterministic.
- **Missing scans or odd foliation:** flag inconsistencies and gaps, but don't synthesize physical pages.

## MVP exit criteria
Given a sample folder of book scans, one command outputs a valid JSON containing every input scan and at least the MetaKat-comparable fields: bibliographic facts, page types, sides, printed numbers and chapter/TOC structure, plus clear unknowns. The processing uses preceding context, performs a documented final reconciliation pass, can resume interrupted jobs and reports model usage/cost when available. A small manual evaluation run demonstrates failure modes and approximate economics.

## References
- MetaKat schema: https://github.com/DCGM/MetaKat/blob/main/metakat/schemas/base_objects.py
- MetaKat pipeline: https://github.com/DCGM/MetaKat/blob/main/metakat/README.md
- OpenRouter structured outputs: https://openrouter.ai/docs/guides/features/structured-outputs
- OpenRouter image inputs: https://openrouter.ai/docs/guides/overview/multimodal/overview
- OpenAI images/vision: https://developers.openai.com/api/docs/guides/images-vision
