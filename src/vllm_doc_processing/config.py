"""Run configuration: built-in defaults < JSON config file < command-line flags.

Credentials are never part of the configuration; they are read only from the
provider's environment variable (see ``API_KEY_ENV``).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

Provider = Literal["openai", "openrouter"]

DEFAULT_BASE_URLS: dict[str, str] = {
    "openai": "https://api.openai.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
}
API_KEY_ENV: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
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

    @field_validator("base_url")
    @classmethod
    def _check_base_url(cls, value: str | None) -> str | None:
        if value is not None and not value.startswith(("https://", "http://")):
            raise ValueError("must start with http:// or https://")
        return value

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
