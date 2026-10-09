# Agent instructions

## Mission
This repository is a small **research experiment**, not a production pipeline. Determine how accurately and cheaply API-hosted vision-language models can annotate digitized **books**, relative to [MetaKat](https://github.com/DCGM/MetaKat). Keep the implementation simple, observable, and reproducible.

## Read first
1. `README.md` — scope and example workflow.
2. `docs/IMPLEMENTATION_PLAN.md` — stages, dependencies, decisions and exit criteria.
3. `docs/OUTPUT_SCHEMA.md` — versioned JSON data model, ontology and evidence semantics.
4. Relevant GitHub issue; do not silently broaden its scope.
5. MetaKat's [schema](https://github.com/DCGM/MetaKat/blob/main/metakat/schemas/base_objects.py) when working on field parity.

## Non-negotiable design constraints
- Python >=3.11; simple installable CLI. Prefer stdlib `argparse`, `pathlib`, `json`, `logging` + `pydantic`, `Pillow`, and official `openai` SDK. No LangChain, agents framework, DB, server, queue, orchestration stack, or cloud infrastructure.
- Input: **one** directory of scans of **one book** plus an order file (image names without extensions, one per line, in scan order); filenames (usually UUIDs) are not sortable. Preserve original filenames and use the listed names as stable scan IDs; never confuse scan index with printed pagination.
- Analyze scans **sequentially**, one scan at a time, with the current image and bounded **text/structured context** from prior results. Never pass every previous image or endlessly append the full conversation.
- Use OpenAI-compatible hosted APIs, especially OpenRouter and OpenAI. Model identifiers, provider/base URL, image detail/resizing, request parameters, context budget, retry limits, and escalation policy must be configurable, not hard-coded.
- Use structured JSON responses wherever supported; parse and validate with Pydantic. Fail explicitly if a provider/model lacks required capabilities. OpenRouter routing must not silently ignore required JSON-schema parameters.
- Book-only extraction: bibliographic metadata, page types, physical scan side (`left`, `right`, `both` or unknown), printed page numbers (potentially two on a spread), contents and chapter hierarchy. Target MetaKat's page type vocabulary and monograph/volume metadata.
- Make uncertain facts null/unknown. Record raw observations, evidence source scans, and a separate reconciled/inferred view. Do not overwrite initial observations or fabricate absent content.
- Final document-wide LLM consistency pass happens *after* scanning. Targeted image revisits and cheap-model-first escalation are optional experimental stages, not prerequisites to the first runnable version.
- Every paid request must be observable (provider, model, stage, page, tokens and cost when available). Support checkpoint/resume and sensible spending limits.
- Do not commit API keys, complete copyrighted scans, local datasets, API payload logs, checkpoints, generated results, or embedded base64 images.

## Agent workflow
- Work on **one issue at a time**, ideally one focused branch/PR per issue. Put `Closes #N` in PR description when complete. Respect issue dependencies.
- Before edits, inspect only the relevant code and docs; avoid unrelated refactoring and abstraction.
- Add only a few high-value **offline** tests using mocked API responses for each behavior (schemas, sorting, retry, context, resume, invariants). Live API tests must be explicit opt-in.
- Update README, schema docs and example files whenever CLI/JSON behavior changes; keep the examples executable. Preserve schema_version and document migrations when breaking JSON format.
- Prefer concise functions and typed data classes/Pydantic models over plugins and generic pipelines. Avoid premature optimization, dependency proliferation, heavy logging, and unnecessary backward compatibility.
- Before claiming completion, run available local checks and report what ran, what failed, and any unresolved limits. Never claim live model accuracy without actual measured runs.

## Suggested package layout (adapt when justified)
```
src/vllm_doc_processing/
    cli.py          # argparse entry point
    config.py       # provider/model/budget configuration
    images.py       # order file, image inventory and conversion
    models.py       # Pydantic request/output contracts
    llm.py          # OpenAI-compatible API adapter
    prompts.py      # versioned extraction and reconciliation prompts
    context.py      # bounded prior-page context
    pipeline.py     # sequential orchestration/checkpointing
    reconcile.py    # document-wide LLM and deterministic checks
    evaluation.py   # optional offline comparisons
tests/             # small offline tests
docs/
```

## Documentation and source of truth
- The published MetaKat schema is a **reference**, not a runtime dependency.
- `src/vllm_doc_processing/models.py` is the source of truth for the JSON format; `docs/OUTPUT_SCHEMA.md` documents it and must be updated together with it.
- GitHub issues #1–#8 are the MVP. #9–#11 are optional experiments.
- Any proposed change that expands scope beyond books or requires OCR/ALTO, a database, or a service needs an explicit issue first.
