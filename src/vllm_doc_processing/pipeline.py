"""Sequential observation of a whole book (one scan at a time, in order, with bounded text context)
and the end-to-end run: observation with a checkpoint after every scan, then reconciliation."""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from pydantic import JsonValue

from .checkpoint import Checkpoint, save_checkpoint
from .context import build_context
from .images import Inventory
from .llm import LLMClient, LLMError
from .models import AnnotatedBook, OcrInput, RunInfo, SourceInfo
from .observe import observe_prompt_versions, observe_scan
from .ocr import ocr_problems, ocr_summary
from .reconcile import reconcile_book

log = logging.getLogger(__name__)


def new_book(client: LLMClient, inventory: Inventory, order_file: Path | None = None) -> AnnotatedBook:
    """Intermediate book with the inventory's scans, no observations yet and the run provenance."""
    config = client.config
    scans = [s.model_copy(deep=True) for s in inventory.scans]
    return AnnotatedBook(
        book_id=inventory.book_dir.resolve().name or "book",
        source=SourceInfo(
            input_directory=str(inventory.book_dir),
            order_file=str(order_file) if order_file else None,
            ocr_directory=str(inventory.ocr_dir) if inventory.ocr_dir else None,
            scan_count=len(scans),
        ),
        scans=scans,
        run=RunInfo(
            started_at=datetime.now(UTC),
            provider=config.provider,
            base_url=config.effective_base_url,
            vision_model=config.model,
            prompt_versions=observe_prompt_versions(inventory),
            parameters=config.model_dump(mode="json"),
        ),
    )


def observe_book(
    client: LLMClient,
    inventory: Inventory,
    order_file: Path | None = None,
    *,
    book: AnnotatedBook | None = None,
    on_scan: Callable[[AnnotatedBook], None] | None = None,
) -> AnnotatedBook:
    """Observe the scans in scan order and return the intermediate book (``resolved`` is None).

    Each request carries only the current image and (unless ``use_context`` is off) a context built
    from the observations of all earlier scans; stored observations are never changed afterwards. A scan that fails after retries
    keeps ``observation=None`` (its attempts stay in ``run.calls``) and processing continues.
    With ``book`` (e.g. from a checkpoint; changed in place), scans that already have an observation
    are skipped. ``on_scan`` is called with the book after every attempted scan.
    """
    config = client.config
    book = book or new_book(client, inventory, order_file)
    scans = book.scans
    for i, scan in enumerate(scans):
        if scan.observation is not None:
            continue
        context = (
            build_context(scans[:i], recent_scans=config.context_recent_scans, max_chars=config.context_max_chars)
            if config.use_context
            else None
        )
        try:
            result = observe_scan(client, inventory, scan, context)
            scan.observation, scan.observation_call_id = result.value, result.call.call_id
            calls, status = result.calls, "ok"
        except LLMError as exc:
            calls, status = exc.calls, f"failed: {exc}"
        book.run.calls += calls
        book.run.refresh_totals()
        log.info(
            "scan %d/%d id=%s context_chars=%d ocr=%s attempts=%d status=%s",
            i + 1,
            len(scans),
            scan.scan_id,
            len(context or ""),
            _ocr_state(scan.ocr),
            len(calls),
            status,
        )
        if on_scan:
            on_scan(book)
    book.run.finished_at = datetime.now(UTC)
    return book


def _ocr_state(ocr: OcrInput | None) -> str:
    if ocr is None:
        return "-"
    if ocr.status != "ok":
        return ocr.status
    return f"{ocr.format}:{ocr.sent_chars}" + ("(truncated)" if ocr.truncated else "")


def resume_book(client: LLMClient, inventory: Inventory, order_file: Path, checkpoint: Checkpoint) -> AnnotatedBook:
    """The checkpoint's book extended to the selected scans (``max_pages`` may have grown)."""
    book = checkpoint.book.model_copy(deep=True)
    book.scans += [s.model_copy(deep=True) for s in inventory.scans[len(book.scans) :]]
    book.source.scan_count = len(book.scans)
    book.source.input_directory, book.source.order_file = str(inventory.book_dir), str(order_file)
    book.source.ocr_directory = str(inventory.ocr_dir) if inventory.ocr_dir else None
    book.run.parameters = client.config.model_dump(mode="json")
    log.info(
        "resuming: %d of %d scan(s) already observed, %d earlier request(s)",
        sum(s.observation is not None for s in book.scans),
        len(book.scans),
        book.run.totals.requests,
    )
    return book


def process_book(
    client: LLMClient,
    inventory: Inventory,
    order_file: Path,
    checkpoint_path: Path,
    identity: dict[str, JsonValue],
    *,
    resume_from: Checkpoint | None = None,
    skip_postprocess: bool = False,
) -> AnnotatedBook:
    """Observe (checkpointing after every scan), then reconcile; return the finished book.

    Only failed scans and scans not in the checkpoint are requested. Reconciliation always runs again
    on resume. If it fails, the ``LLMError``/``ReconcileError`` propagates after its attempts are
    saved to the checkpoint, so a later resume repeats only the reconciliation.
    """

    def save(b: AnnotatedBook) -> None:
        save_checkpoint(checkpoint_path, identity, b)

    book = resume_book(client, inventory, order_file, resume_from) if resume_from else new_book(client, inventory, order_file)
    save(book)
    book = observe_book(client, inventory, order_file, book=book, on_scan=save)
    warnings = []
    unobserved = [s.scan_id for s in book.scans if s.observation is None]
    if unobserved:
        warnings.append(f"{len(unobserved)} scan(s) not observed (failed requests): {', '.join(unobserved)}")
    ocr = ocr_summary(book.scans)
    if ocr["missing"]:
        warnings.append(f"{ocr['missing']} scan(s) had no OCR sidecar and were sent as image only")
    warnings += ocr_problems(book.scans)
    if ocr["truncated"]:
        warnings.append(f"OCR text of {ocr['truncated']} scan(s) was shortened to ocr_max_chars")
    if skip_postprocess:
        warnings.append("reconciliation skipped (--skip-postprocess); resolved is null")
    else:
        try:
            book = reconcile_book(client, book)
        except LLMError as exc:
            book.run.calls += exc.calls
            book.run.refresh_totals()
            save(book)
            raise
        save(book)  # records the reconciliation calls for later resumes
    book.run.warnings += warnings
    book.run.finished_at = datetime.now(UTC)
    return book
