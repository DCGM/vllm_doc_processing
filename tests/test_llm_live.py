"""Opt-in live smoke test (paid, a fraction of a cent). Never runs by default.

    VLLM_DOC_LIVE_TEST=1 VLLM_DOC_LIVE_PROVIDER=openrouter VLLM_DOC_LIVE_MODEL=<vision-model-id> \\
        OPENROUTER_API_KEY=... pytest tests/test_llm_live.py -s
"""

import io
import os
from typing import Literal

import pytest
from PIL import Image
from pydantic import BaseModel

from vllm_doc_processing.config import build_config
from vllm_doc_processing.images import PreparedImage
from vllm_doc_processing.llm import LLMClient

pytestmark = pytest.mark.skipif(os.environ.get("VLLM_DOC_LIVE_TEST") != "1", reason="set VLLM_DOC_LIVE_TEST=1")


class DarkHalf(BaseModel):
    dark_half: Literal["left", "right"]


def test_live_vision_structured_output():
    config = build_config(
        {"provider": os.environ.get("VLLM_DOC_LIVE_PROVIDER", "openrouter"), "model": os.environ["VLLM_DOC_LIVE_MODEL"]},
        {},
    )
    image = Image.new("L", (256, 128), 255)
    image.paste(0, (128, 0, 256, 128))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    result = LLMClient(config).request(
        DarkHalf,
        stage="observe",
        model=config.model,
        system="You describe images.",
        user="Which half of the image is black?",
        image=PreparedImage(buffer.getvalue(), "image/png", 256, 128, reencoded=False),
    )
    print(result.call.model_dump_json(indent=2))
    assert result.value.dark_half == "right"
    assert result.call.status == "ok" and result.call.prompt_tokens
