"""Single-scan annotation: current image + optional text context -> validated ``ScanObservation``."""

from __future__ import annotations

from .images import Inventory, prepare_image
from .llm import LLMClient, LLMResult
from .models import ScanObservation, ScanRecord
from .prompts import OBSERVE_SYSTEM, observe_user_prompt


def observe_scan(
    client: LLMClient, inventory: Inventory, scan: ScanRecord, context: str | None = None
) -> LLMResult[ScanObservation]:
    """Annotate one scan; raises ``LLMError`` (carrying every attempt's ``CallRecord``) if no valid answer.

    Invalid answers (bad JSON, schema or invariant violations such as subpages on a single page) are
    recorded as ``invalid_response`` calls and retried by the client.
    """
    config = client.config
    image = prepare_image(inventory.path(scan), config.image_max_side, config.image_format)
    return client.request(
        ScanObservation,
        stage="observe",
        model=config.model,
        system=OBSERVE_SYSTEM,
        user=observe_user_prompt(scan.scan_index, inventory.total_listed, context),
        image=image,
        scan_id=scan.scan_id,
    )
