"""Sequential observation of a whole book: one scan at a time, in order, with bounded text context."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

from .context import CONTEXT_VERSION, build_context
from .images import Inventory
from .llm import LLMClient, LLMError
from .models import AnnotatedBook, RunInfo, SourceInfo
from .observe import observe_scan
from .prompts import PROMPT_VERSIONS

log = logging.getLogger(__name__)


def observe_book(client: LLMClient, inventory: Inventory, order_file: Path | None = None) -> AnnotatedBook:
    """Observe ``inventory.scans`` in scan order and return the intermediate book (``resolved`` is None).

    Each request carries only the current image and (unless ``use_context`` is off) a context built
    from the observations of all earlier scans; stored observations are never changed afterwards. A scan that fails after retries
    keeps ``observation=None`` (its attempts stay in ``run.calls``) and processing continues.
    """
    config = client.config
    scans = [s.model_copy(deep=True) for s in inventory.scans]
    book = AnnotatedBook(
        book_id=inventory.book_dir.resolve().name or "book",
        source=SourceInfo(
            input_directory=str(inventory.book_dir),
            order_file=str(order_file) if order_file else None,
            scan_count=len(scans),
        ),
        scans=scans,
        run=RunInfo(
            started_at=datetime.now(UTC),
            provider=config.provider,
            base_url=config.effective_base_url,
            vision_model=config.model,
            prompt_versions={"observe": PROMPT_VERSIONS["observe"], "context": CONTEXT_VERSION},
            parameters=config.model_dump(mode="json"),
        ),
    )
    for i, scan in enumerate(scans):
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
        log.info(
            "scan %d/%d id=%s context_chars=%d attempts=%d status=%s",
            i + 1,
            len(scans),
            scan.scan_id,
            len(context or ""),
            len(calls),
            status,
        )
    book.run.refresh_totals()
    book.run.finished_at = datetime.now(UTC)
    return book
