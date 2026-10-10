"""Observation checkpoint of one run: the intermediate book plus the identity it was produced under.

A checkpoint is written atomically after every scan. It holds only the observation stage
(``book.resolved`` is always None); reconciliation is cheap (one request) and is run again on resume.
Resuming is allowed only when everything that shapes an observation is unchanged: the order file,
the images (hashes), the observation model and its settings, the prompt and context versions and
the schema version. Settings that only affect reconciliation, retries or ``max_pages`` may change.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Literal

from pydantic import JsonValue, ValidationError

from .config import Config, ConfigError
from .context import CONTEXT_VERSION
from .images import Inventory
from .models import SCHEMA_VERSION, AnnotatedBook, StrictModel
from .prompts import PROMPT_VERSIONS

OBSERVATION_SETTINGS = (
    "provider",
    "model",
    "image_max_side",
    "image_format",
    "image_detail",
    "max_output_tokens",
    "use_context",
    "context_recent_scans",
    "context_max_chars",
    "request_params",
)
"""``Config`` keys that change what the vision model sees or answers; part of the checkpoint identity."""

SCAN_KEYS = ("scan_id", "scan_index", "filename", "image_sha256", "width", "height")


class CheckpointError(ConfigError):
    """Missing, unreadable or incompatible checkpoint; nothing was sent to the API."""


class Checkpoint(StrictModel):
    checkpoint_version: Literal[1] = 1
    identity: dict[str, JsonValue]
    book: AnnotatedBook


def run_identity(config: Config, inventory: Inventory, order_names: list[str]) -> dict[str, JsonValue]:
    """Everything an observation depends on except the images themselves (compared per scan)."""
    settings = {key: getattr(config, key) for key in OBSERVATION_SETTINGS}
    if not config.use_context:  # irrelevant without context
        del settings["context_recent_scans"], settings["context_max_chars"]
    return {
        "schema_version": SCHEMA_VERSION,
        "prompt_versions": {"observe": PROMPT_VERSIONS["observe"], "context": CONTEXT_VERSION},
        "base_url": config.effective_base_url,
        **settings,
        # The scan count and positions are part of every prompt, so the whole order file must match.
        "order_sha256": hashlib.sha256("\n".join(order_names).encode()).hexdigest(),
        "listed_scans": inventory.total_listed,
    }


def load_checkpoint(path: Path, identity: dict[str, JsonValue], inventory: Inventory) -> Checkpoint:
    """Read ``path`` and check that it can be resumed with ``identity`` and ``inventory``."""
    try:
        checkpoint = Checkpoint.model_validate_json(path.read_bytes())
    except FileNotFoundError:
        raise CheckpointError(f"no checkpoint to resume at {path}; run without --resume") from None
    except OSError as exc:
        raise CheckpointError(f"cannot read checkpoint {path}: {exc}") from None
    except ValidationError as exc:
        raise CheckpointError(
            f"checkpoint {path} is not a valid checkpoint of this version ({exc.error_count()} error(s)); "
            "use --fresh to start over"
        ) from None
    changed = sorted(k for k in identity.keys() | checkpoint.identity.keys() if identity.get(k) != checkpoint.identity.get(k))
    if changed:
        raise CheckpointError(
            f"checkpoint {path} was made with different {', '.join(changed)}; "
            "use the original settings to resume, or --fresh to start over"
        )
    done, selected = checkpoint.book.scans, inventory.scans
    if len(done) > len(selected):
        raise CheckpointError(
            f"checkpoint {path} covers {len(done)} scans but only {len(selected)} are selected; "
            "raise --max-pages or use --fresh"
        )
    for old, new in zip(done, selected):
        if old.model_dump(include=set(SCAN_KEYS)) != new.model_dump(include=set(SCAN_KEYS)):
            raise CheckpointError(
                f"scan {new.scan_index + 1} ({new.filename}) differs from the checkpoint {path} "
                "(changed image or order); use --fresh to start over"
            )
    return checkpoint


def save_checkpoint(path: Path, identity: dict[str, JsonValue], book: AnnotatedBook) -> None:
    """Only the observation stage is saved (no ``resolved``, ``finished_at`` or reconciliation provenance;
    reconciliation calls stay in ``run.calls``). Re-validated, so a broken book fails here, not on resume."""
    prompt_versions = {k: v for k, v in book.run.prompt_versions.items() if k != "reconcile"}
    run = book.run.model_copy(update={"finished_at": None, "postprocess_model": None, "prompt_versions": prompt_versions})
    data = {"identity": identity, "book": book.model_copy(update={"resolved": None, "run": run}).model_dump()}
    write_atomic(path, Checkpoint.model_validate(data).model_dump_json(indent=1) + "\n")


def write_atomic(path: Path, text: str) -> None:
    """Write via a temporary file in the same directory, so ``path`` is never left half-written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

