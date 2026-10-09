import json
import logging
from typing import Literal

import openai
import pytest
from openai.types.chat import ChatCompletion
from pydantic import BaseModel

from vllm_doc_processing.config import ConfigError, build_config
from vllm_doc_processing.images import PreparedImage
from vllm_doc_processing.llm import LLMClient, LLMError

try:  # HTTP library of the installed OpenAI SDK (httpx2 since SDK 3.x)
    import httpx2 as httpx
except ImportError:
    import httpx

IMAGE = PreparedImage(b"\xff\xd8fake-jpeg", "image/jpeg", 10, 20, reencoded=False)
KEY = "sk-or-v1-0123456789abcdef"


class Answer(BaseModel):
    side: Literal["left", "right"] | None
    note: str | None


def completion(content, *, finish_reason="stop", cost=0.0012, provider="Google", choices=True):
    body = {
        "id": "gen-123",
        "object": "chat.completion",
        "created": 0,
        "model": "vendor/model-2026-01-01",
        "choices": [
            {"index": 0, "finish_reason": finish_reason, "message": {"role": "assistant", "content": content}}
        ]
        if choices
        else [],
        "usage": {"prompt_tokens": 1000, "completion_tokens": 50, "total_tokens": 1050, "cost": cost},
        "provider": provider,
    }
    return ChatCompletion.model_validate(body)


def http_error(cls, status, message, headers=None):
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    return cls(message, response=httpx.Response(status, request=request, headers=headers), body=None)


class FakeClient:
    """Stands in for ``openai.OpenAI``: returns or raises the queued outcomes, records the requests."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.requests = []
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        self.requests.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def make_client(fake, provider="openrouter", **settings):
    config = build_config({"provider": provider, "model": "vendor/model", **settings}, {})
    sleeps = []
    return LLMClient(config, client=fake, sleep=sleeps.append), sleeps


def ask(client, image=IMAGE):
    return client.request(Answer, stage="observe", model="vendor/model", system="sys", user="Which side?",
                          image=image, scan_id="s1")


def test_image_request_strict_schema_routing_and_usage():
    fake = FakeClient(completion('{"side": "left", "note": null}'))
    client, sleeps = make_client(fake, request_params={"temperature": 0, "provider": {"order": ["google"]}})
    result = ask(client)

    assert result.value == Answer(side="left", note=None)
    req = fake.requests[0]
    assert req["model"] == "vendor/model"
    fmt = req["response_format"]
    assert fmt["type"] == "json_schema" and fmt["json_schema"]["strict"] is True
    assert fmt["json_schema"]["schema"]["additionalProperties"] is False
    assert sorted(fmt["json_schema"]["schema"]["required"]) == ["note", "side"]
    image_part, text_part = req["messages"][1]["content"]
    assert image_part["image_url"] == {"url": IMAGE.data_url(), "detail": "high"}
    assert text_part == {"type": "text", "text": "Which side?"}
    assert req["extra_body"] == {"temperature": 0, "provider": {"order": ["google"], "require_parameters": True}}

    call = result.call
    assert (call.stage, call.scan_id, call.provider, call.model, call.status) == (
        "observe", "s1", "openrouter", "vendor/model", "ok")
    assert (call.prompt_tokens, call.completion_tokens, call.cost_usd) == (1000, 50, 0.0012)
    assert (call.response_id, call.served_model, call.upstream_provider) == ("gen-123", "vendor/model-2026-01-01", "Google")
    assert call.latency_s is not None and call.started_at is not None and sleeps == []


def test_text_only_request_on_openai():
    fake = FakeClient(completion('{"side": null, "note": "n/a"}', cost=None, provider=None))
    client, _ = make_client(fake, provider="openai")
    result = ask(client, image=None)

    assert result.value.note == "n/a"
    req = fake.requests[0]
    assert req["messages"][1]["content"] == [{"type": "text", "text": "Which side?"}]
    assert req["extra_body"] == {}
    assert result.call.cost_usd is None and result.call.upstream_provider is None


def test_invalid_json_is_retried_and_paid_attempts_are_recorded():
    fake = FakeClient(
        completion("not json"),
        completion('{"side": "middle", "note": null}'),
        completion("", finish_reason="length"),
        completion(None, choices=False),
        completion('{"side": "right", "note": null}'),
    )
    client, sleeps = make_client(fake, max_retries=4)
    result = ask(client)

    assert result.value.side == "right"
    assert [c.status for c in result.calls] == ["invalid_response"] * 4 + ["ok"]
    assert [c.attempt for c in result.calls] == [1, 2, 3, 4, 5]
    assert all(c.prompt_tokens == 1000 for c in result.calls)  # failed but billed
    assert "side" in result.calls[1].error and "finish_reason=length" in result.calls[2].error
    assert len(sleeps) == 4
    assert len({c.call_id for c in result.calls}) == 5


def test_unsupported_structured_outputs_fails_without_retry():
    error = http_error(openai.NotFoundError, 404, "No endpoints found that can handle the requested parameters.")
    client, sleeps = make_client(FakeClient(error))
    with pytest.raises(LLMError, match="json_schema") as exc:
        ask(client)
    assert [c.status for c in exc.value.calls] == ["error"]
    assert "HTTP 404" in exc.value.calls[0].error and sleeps == []


def test_transient_errors_retry_with_backoff_then_give_up():
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    fake = FakeClient(
        http_error(openai.RateLimitError, 429, "slow down", headers={"retry-after": "7"}),
        openai.APITimeoutError(request=request),
        http_error(openai.InternalServerError, 502, "bad gateway"),
    )
    client, sleeps = make_client(fake, max_retries=2)
    with pytest.raises(LLMError, match="after 3 attempt") as exc:
        ask(client)
    assert [c.status for c in exc.value.calls] == ["error"] * 3
    assert sleeps[0] == 7.0 and len(sleeps) == 2
    assert all(c.prompt_tokens is None for c in exc.value.calls)


def test_logs_and_errors_never_contain_keys_or_image_data(caplog):
    echo = f"bad request: key {KEY} payload {IMAGE.data_url()}"
    fake = FakeClient(completion('{"side": "left", "note": null}'), http_error(openai.BadRequestError, 400, echo))
    client, _ = make_client(fake)
    with caplog.at_level(logging.INFO, logger="vllm_doc_processing.llm"):
        ask(client)
        with pytest.raises(LLMError) as exc:
            ask(client)
    dumped = caplog.text + str(exc.value) + json.dumps([c.model_dump(mode="json") for c in exc.value.calls])
    assert len(caplog.records) == 2 and "status=ok" in caplog.records[0].getMessage()
    assert KEY not in dumped and "base64," not in dumped and "Which side?" not in dumped
    assert "<redacted-key>" in exc.value.calls[0].error


def test_request_params_cannot_override_adapter_fields():
    for params, provider in (({"response_format": {"type": "text"}}, "openrouter"), ({"provider": {}}, "openai")):
        with pytest.raises(ConfigError, match="request_params"):
            build_config({"provider": provider, "model": "m", "request_params": params}, {})
