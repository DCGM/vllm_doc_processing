"""Single-scan annotation: current image + optional text context and OCR text -> validated ``ScanObservation``."""

from __future__ import annotations

from .context import CONTEXT_VERSION
from .images import Inventory, prepare_image
from .llm import LLMClient, LLMResult
from .models import ScanObservation, ScanRecord
from .prompts import OBSERVE_SYSTEM, PROMPT_VERSIONS, observe_user_prompt


def observe_prompt_versions(inventory: Inventory) -> dict[str, str]:
    """Versions of everything in an observation request's text; ``ocr`` only if OCR is attached."""
    versions = {"observe": PROMPT_VERSIONS["observe"], "context": CONTEXT_VERSION}
    if inventory.ocr_dir is not None:
        versions["ocr"] = PROMPT_VERSIONS["ocr"]
    return versions


def observe_scan(
    client: LLMClient, inventory: Inventory, scan: ScanRecord, context: str | None = None
) -> LLMResult[ScanObservation]:
    """Annotate one scan; raises ``LLMError`` (carrying every attempt's ``CallRecord``) if no valid answer.

    Invalid answers (bad JSON, schema or invariant violations such as subpages on a single page) are
    recorded as ``invalid_response`` calls and retried by the client. The scan's own bounded OCR
    text (``inventory.ocr_texts``), if any, is sent with the image; never that of other scans.
    """
    config = client.config
    image = prepare_image(inventory.path(scan), config.image_max_side, config.image_format)
    return client.request(
        ScanObservation,
        stage="observe",
        model=config.model,
        system=OBSERVE_SYSTEM,
        user=observe_user_prompt(
            scan.scan_index, inventory.total_listed, context, inventory.ocr_texts.get(scan.scan_id)
        ),
        image=image,
        scan_id=scan.scan_id,
    )
