"""OpenAI-compatible chat client (OpenAI, OpenRouter): optional image + text in, one validated Pydantic model out.

Every request attempt becomes a ``CallRecord`` (retries included) and one log line; neither ever
contains the API key, the prompt or image data.
"""

from __future__ import annotations

import json
import logging
import random
import re
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from types import SimpleNamespace
from datetime import UTC, datetime
from typing import Any, Generic, TypeVar

from openai import APIConnectionError, APIResponseValidationError, APIStatusError, OpenAI

# Same strict-schema conversion that ``client.chat.completions.parse()`` uses; we call ``create()``
# ourselves so that token usage and cost are recorded even when the response fails validation.
from openai.lib._pydantic import to_strict_json_schema
from pydantic import BaseModel, ValidationError

from .config import Config, require_api_key
from .images import PreparedImage
from .models import CallRecord, Stage

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

RETRYABLE_STATUS = {408, 409, 429}
"""Plus every 5xx; other HTTP errors (bad request, auth, unsupported model/parameters) fail at once."""
BACKOFF_BASE_S = 2.0
BACKOFF_MAX_S = 60.0
MAX_ERROR_CHARS = 500

_DATA_URL = re.compile(r"data:[\w/+.-]+;base64,[A-Za-z0-9+/=]+")
_KEY_LIKE = re.compile(r"\bsk-[A-Za-z0-9_-]{8,}")


class LLMError(Exception):
    """A request failed for good. The message is safe to log; ``calls`` holds every attempt."""

    def __init__(self, message: str, calls: list[CallRecord]):
        super().__init__(message)
        self.calls = calls


class _InvalidResponse(Exception):
    """The provider answered, but not with a usable instance of the response model."""


@dataclass
class LLMResult(Generic[T]):
    value: T
    calls: list[CallRecord]
    """All attempts in order; the last one is the successful call."""

    @property
    def call(self) -> CallRecord:
        return self.calls[-1]


class LLMClient:
    """One shared interface for vision (``image`` given) and text-only structured requests."""

    def __init__(self, config: Config, *, client: Any = None, sleep: Callable[[float], None] = time.sleep):
        self.config = config
        self._secret: str | None = None
        if client is None:
            self._secret = require_api_key(config)
            # SDK retries are disabled so that every attempt is recorded and logged here.
            client = OpenAI(
                api_key=self._secret,
                base_url=config.effective_base_url,
                timeout=config.request_timeout_s,
                max_retries=0,
            )
        self._client = client
        self._sleep = sleep

    def request(
        self,
        response_model: type[T],
        *,
        stage: Stage,
        model: str,
        system: str,
        user: str,
        image: PreparedImage | None = None,
        scan_id: str | None = None,
    ) -> LLMResult[T]:
        try:
            schema = to_strict_json_schema(response_model)
        except Exception as exc:
            raise LLMError(f"cannot build a strict JSON schema for {response_model.__name__}: {exc}", []) from None
        kwargs = self._build_request(response_model.__name__, schema, model, system, user, image)

        calls: list[CallRecord] = []
        for attempt in range(1, self.config.max_retries + 2):
            started_at, t0 = datetime.now(UTC), time.monotonic()
            response, value, error, retry_after = None, None, None, None
            status, retryable = "ok", False
            try:
                response = self._client.chat.completions.create(**kwargs)
                value = _parse(response, response_model)
            except _InvalidResponse as exc:
                status, error, retryable = "invalid_response", str(exc), True
            except APIResponseValidationError as exc:
                # The SDK could not parse the response; the raw body may still carry usage and cost.
                response = _raw_body(exc.body)
                status, error, retryable = "invalid_response", f"{type(exc).__name__}: {exc.message}", True
            except APIStatusError as exc:
                status, error = "error", f"HTTP {exc.status_code}: {exc.message}"
                retryable = exc.status_code in RETRYABLE_STATUS or exc.status_code >= 500
                retry_after = _retry_after(exc)
            except APIConnectionError as exc:  # includes APITimeoutError
                status, error, retryable = "error", f"{type(exc).__name__}: {exc}", True
            record = self._record(stage, scan_id, model, attempt, status, started_at, t0, response, error)
            calls.append(record)
            _log_call(record)
            if value is not None:
                return LLMResult(value, calls)
            if not retryable or attempt > self.config.max_retries:
                hint = "" if retryable else _hint(error or "")
                raise LLMError(f"{stage} request failed after {attempt} attempt(s): {record.error}{hint}", calls)
            self._sleep(retry_after if retry_after is not None else _backoff(attempt))
        raise AssertionError("unreachable")

    def _build_request(
        self, name: str, schema: dict[str, Any], model: str, system: str, user: str, image: PreparedImage | None
    ) -> dict[str, Any]:
        content: list[dict[str, Any]] = []
        if image is not None:
            content.append(
                {"type": "image_url", "image_url": {"url": image.data_url(), "detail": self.config.image_detail}}
            )
        content.append({"type": "text", "text": user})
        extra_body = dict(self.config.request_params)
        if self.config.provider == "openrouter":
            # Route only to endpoints honouring every parameter, so json_schema is never silently dropped.
            extra_body["provider"] = {**extra_body.get("provider", {}), "require_parameters": True}
        return {
            "model": model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}],
            "response_format": {"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": schema}},
            "extra_body": extra_body,
        }

    def _record(
        self,
        stage: Stage,
        scan_id: str | None,
        model: str,
        attempt: int,
        status: str,
        started_at: datetime,
        t0: float,
        response: Any,
        error: str | None,
    ) -> CallRecord:
        usage = getattr(response, "usage", None)
        cost = getattr(usage, "cost", None)
        upstream = getattr(response, "provider", None)
        return CallRecord(
            call_id=uuid.uuid4().hex,
            stage=stage,
            scan_id=scan_id,
            provider=self.config.provider,
            model=model,
            attempt=attempt,
            status=status,
            started_at=started_at,
            latency_s=round(time.monotonic() - t0, 3),
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            cost_usd=float(cost) if isinstance(cost, int | float) and cost >= 0 else None,
            response_id=getattr(response, "id", None) or None,
            served_model=getattr(response, "model", None) or None,
            upstream_provider=upstream if isinstance(upstream, str) else None,
            error=None if error is None else safe_text(error, self._secret),
        )


def _parse(response: Any, response_model: type[T]) -> T:
    choices = getattr(response, "choices", None)
    if not choices:
        # OpenRouter can report upstream failures as HTTP 200 with an ``error`` object and no choices.
        detail = getattr(response, "error", None)
        raise _InvalidResponse(f"response has no choices{f': {detail}' if detail else ''}")
    choice = choices[0]
    message = choice.message
    if getattr(message, "refusal", None):
        raise _InvalidResponse(f"model refused: {message.refusal}")
    if choice.finish_reason == "length":
        raise _InvalidResponse("output truncated (finish_reason=length); raise the output token limit")
    if not message.content:
        raise _InvalidResponse(f"empty response content (finish_reason={choice.finish_reason})")
    try:
        return response_model.model_validate_json(message.content)
    except ValidationError as exc:
        errors = exc.errors(include_input=False, include_url=False)
        first = errors[0]
        where = ".".join(map(str, first["loc"])) or "response"
        raise _InvalidResponse(
            f"response does not match {response_model.__name__} ({len(errors)} error(s)); {where}: {first['msg']}"
        ) from None


def _raw_body(body: object) -> Any:
    """Attribute view of a raw JSON response body for ``_record``; None if it is not a JSON object."""
    if not isinstance(body, dict):
        return None
    try:
        return json.loads(json.dumps(body), object_hook=lambda d: SimpleNamespace(**d))
    except (TypeError, ValueError):
        return None


def _retry_after(exc: APIStatusError) -> float | None:
    try:
        return min(BACKOFF_MAX_S, max(0.0, float(exc.response.headers["retry-after"])))
    except (KeyError, ValueError, TypeError, AttributeError):
        return None


def _backoff(attempt: int) -> float:
    return min(BACKOFF_MAX_S, BACKOFF_BASE_S * 2 ** (attempt - 1)) * random.uniform(0.5, 1.0)


def _hint(error: str) -> str:
    if error.startswith(("HTTP 400", "HTTP 404", "HTTP 422")):
        return (
            " (check that the model and its provider route support image input, if used, and strict "
            "json_schema structured outputs, and that request_params are valid)"
        )
    if error.startswith(("HTTP 401", "HTTP 403")):
        return " (check the API key environment variable and account permissions)"
    return ""


def safe_text(text: str, secret: str | None = None) -> str:
    """Redact base64 data URLs and API-key-like strings, then truncate."""
    if secret:
        text = text.replace(secret, "<redacted-key>")
    text = _KEY_LIKE.sub("<redacted-key>", _DATA_URL.sub("data:<redacted>", text))
    return text if len(text) <= MAX_ERROR_CHARS else text[:MAX_ERROR_CHARS] + "..."


def _log_call(call: CallRecord) -> None:
    level = logging.INFO if call.status == "ok" else logging.WARNING
    log.log(
        level,
        "%s scan=%s model=%s attempt=%d status=%s latency=%.1fs tokens=%s/%s cost=%s%s",
        call.stage,
        call.scan_id,
        call.model,
        call.attempt,
        call.status,
        call.latency_s or 0.0,
        call.prompt_tokens,
        call.completion_tokens,
        call.cost_usd,
        f" error={call.error}" if call.error else "",
    )
