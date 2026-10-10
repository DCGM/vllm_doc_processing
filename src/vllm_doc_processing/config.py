"""Run configuration: built-in defaults < JSON config file < command-line flags.

Credentials are never part of the configuration; they are read only from the
provider's environment variable (see ``API_KEY_ENV``).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError, field_validator, model_validator

Provider = Literal["openai", "openrouter"]
OcrFormat = Literal["auto", "txt", "alto"]

DEFAULT_BASE_URLS: dict[str, str] = {
    "openai": "https://api.openai.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
}
API_KEY_ENV: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}
RESERVED_REQUEST_PARAMS = frozenset(
    {"model", "messages", "response_format", "stream", "n", "tools", "tool_choice", "max_tokens", "max_completion_tokens"}
)
"""Request fields set by the adapter itself; ``request_params`` must not override them."""
OUTPUT_LIMIT_PARAM: dict[str, str] = {
    "openai": "max_completion_tokens",  # OpenAI reasoning models reject max_tokens
    "openrouter": "max_tokens",
}


class ConfigError(Exception):
    """Invalid configuration or input; message is safe to show (no secrets)."""


class Config(BaseModel):
    """Effective, non-secret settings of one run (serializable into ``run.parameters``)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: Provider
    model: str = Field(min_length=1, description="Vision model ID used for per-scan observation.")
    postprocess_model: str | None = Field(
        default=None, min_length=1, description="Text model ID for reconciliation; None means reuse `model`."
    )
    base_url: str | None = Field(default=None, description="None means the provider's default base URL.")
    max_pages: int | None = Field(default=None, ge=1, description="Process only the first N listed scans; None = all.")
    image_max_side: int | None = Field(
        default=2048, ge=256, description="Downscale uploads so the longest side is at most this; None = never."
    )
    image_format: Literal["jpeg", "png"] = Field(default="jpeg", description="Encoding of converted/resized uploads.")
    image_detail: Literal["auto", "low", "high"] = Field(default="auto", description="Image `detail` sent with each scan.")
    request_timeout_s: float = Field(default=180.0, gt=0, description="Timeout of one API request attempt.")
    max_retries: int = Field(default=3, ge=0, le=10, description="Retries after a transient or invalid response.")
    max_output_tokens: int | None = Field(
        default=4000, ge=1, description="Output token cap per request (includes reasoning tokens); None = no cap."
    )
    use_context: bool = Field(
        default=True, description="Send a text summary of earlier scans with each scan (False: every scan alone)."
    )
    context_recent_scans: int = Field(
        default=5, ge=0, le=50, description="Earlier scans summarized one line each in the context of the next scan."
    )
    context_max_chars: int = Field(
        default=2000, ge=200, description="Hard limit on the length of the earlier-scan context text."
    )
    ocr_format: OcrFormat = Field(
        default="auto",
        description="Sidecars read from the OCR directory: <scan_id>.txt, <scan_id>.xml (ALTO) or either (auto).",
    )
    ocr_max_chars: int = Field(
        default=6000, ge=200, description="Longest OCR text sent with one scan; longer texts lose their middle part."
    )
    reconcile_max_chars: int = Field(
        default=100_000, ge=1000, description="Longest reconciliation input text; longer books fail before the request."
    )
    reconcile_max_output_tokens: int | None = Field(
        default=16000, ge=1, description="Output token cap of the reconciliation request; None = no cap."
    )
    request_params: dict[str, JsonValue] = Field(
        default_factory=dict,
        description="Extra request body fields, e.g. temperature, max_completion_tokens, reasoning settings.",
    )

    @field_validator("base_url")
    @classmethod
    def _check_base_url(cls, value: str | None) -> str | None:
        if value is not None and not value.startswith(("https://", "http://")):
            raise ValueError("must start with http:// or https://")
        return value

    @model_validator(mode="after")
    def _check_request_params(self) -> Config:
        reserved = sorted(RESERVED_REQUEST_PARAMS & self.request_params.keys())
        if reserved:
            raise ValueError(f"request_params must not set {reserved}; they are set by the tool (use max_output_tokens)")
        if "provider" in self.request_params and not (
            self.provider == "openrouter" and isinstance(self.request_params["provider"], dict)
        ):
            raise ValueError("request_params.provider must be an object (OpenRouter provider routing only)")
        return self

    @property
    def effective_base_url(self) -> str:
        return self.base_url or DEFAULT_BASE_URLS[self.provider]

    @property
    def effective_postprocess_model(self) -> str:
        return self.postprocess_model or self.model

    @property
    def api_key_env(self) -> str:
        return API_KEY_ENV[self.provider]


def read_config_file(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ConfigError(f"config file not found: {path}") from None
    except (OSError, UnicodeDecodeError) as exc:
        raise ConfigError(f"cannot read config file {path}: {exc}") from None
    except json.JSONDecodeError as exc:
        raise ConfigError(f"config file {path} is not valid JSON: {exc}") from None
    if not isinstance(data, dict):
        raise ConfigError(f"config file {path} must contain a JSON object")
    return data


def build_config(file_values: dict[str, Any], cli_values: dict[str, Any]) -> Config:
    """Merge config-file values with CLI overrides (``None`` means "not given") and validate."""
    if "api_key" in file_values:
        raise ConfigError(
            "do not put API keys in the config file; set OPENAI_API_KEY or OPENROUTER_API_KEY in the environment"
        )
    cli_given = {k: v for k, v in cli_values.items() if v is not None}
    file_values = dict(file_values)
    if file_values.get("provider") not in (None, cli_given.get("provider", file_values.get("provider"))):
        # A base URL from the file belongs to the file's provider; never pair it with another provider's key.
        file_values.pop("base_url", None)
    merged = {**file_values, **cli_given}
    try:
        return Config.model_validate(merged)
    except ValidationError as exc:
        # Format without input values so a misplaced secret is never echoed.
        problems = "; ".join(f"{'.'.join(map(str, e['loc'])) or 'config'}: {e['msg']}" for e in exc.errors())
        raise ConfigError(f"invalid configuration: {problems}") from None


def api_key(config: Config) -> str | None:
    return os.environ.get(config.api_key_env, "").strip() or None


def require_api_key(config: Config) -> str:
    key = api_key(config)
    if not key:
        raise ConfigError(f"missing API key: set the {config.api_key_env} environment variable")
    return key
