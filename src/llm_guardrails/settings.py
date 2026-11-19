"""Settings from environment variables and an optional `.env` file.

A tiny loader instead of pydantic-settings: one less dependency, and the rules are visible
here. Real environment variables win over `.env`. Secret fields are hidden from repr so a
stray print or log of the settings object can never leak a key.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, get_args

from llm_guardrails.types import ConfigError

LLMMode = Literal["replay", "fake", "bedrock"]


def read_env_file(path: Path) -> dict[str, str]:
    """Parse KEY=VALUE lines; ignores blanks and comments; strips matching quotes."""
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key.strip()] = value
    return values


@dataclass(frozen=True)
class Settings:
    aws_region: str = "ap-south-1"
    bearer_token: str = field(default="", repr=False)
    access_key_id: str = field(default="", repr=False)
    secret_access_key: str = field(default="", repr=False)
    chat_model_id: str = "global.amazon.nova-2-lite-v1:0"
    judge_model_id: str = "global.amazon.nova-2-lite-v1:0"
    embed_model_id: str = "amazon.titan-embed-text-v2:0"
    embed_dimensions: int = 1024
    embed_requests_per_second: float = 1.0
    llm_mode: LLMMode = "replay"
    live_budget_usd: float = 5.0
    log_hash_salt: str = field(default="", repr=False)

    @property
    def auth_method(self) -> Literal["bearer", "iam", "none"]:
        if self.bearer_token:
            return "bearer"
        if self.access_key_id:
            return "iam"
        return "none"


def _number(env: Mapping[str, str], key: str, default: float, cast: type) -> float:
    raw = env.get(key, "").strip()
    if not raw:
        return default
    try:
        return cast(raw)
    except ValueError as exc:
        raise ConfigError(f"{key} must be a number") from exc


def load_settings(
    env_file: Path | None = Path(".env"), environ: Mapping[str, str] | None = None
) -> Settings:
    """Build Settings. `.env` values are used only where the real environment has none."""
    env: dict[str, str] = read_env_file(env_file) if env_file else {}
    for key, value in (os.environ if environ is None else environ).items():
        if value != "":
            env[key] = value

    bearer = env.get("AWS_BEARER_TOKEN_BEDROCK", "")
    key_id = env.get("AWS_ACCESS_KEY_ID", "")
    secret = env.get("AWS_SECRET_ACCESS_KEY", "")
    if bearer and (key_id or secret):
        # A beginner cannot tell which credential boto3 would use, so refuse to guess.
        raise ConfigError(
            "Set either AWS_BEARER_TOKEN_BEDROCK or AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY, "
            "not both"
        )
    if bool(key_id) != bool(secret):
        raise ConfigError("AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY must be set together")

    mode = env.get("GUARDRAILS_LLM_MODE", "replay") or "replay"
    if mode not in get_args(LLMMode):
        raise ConfigError(f"GUARDRAILS_LLM_MODE must be one of {get_args(LLMMode)}")

    budget = _number(env, "LIVE_BUDGET_USD", 5.0, float)
    if budget < 0:
        raise ConfigError("LIVE_BUDGET_USD must not be negative")

    return Settings(
        aws_region=env.get("AWS_REGION", "") or "ap-south-1",
        bearer_token=bearer,
        access_key_id=key_id,
        secret_access_key=secret,
        chat_model_id=env.get("BEDROCK_CHAT_MODEL_ID", "") or Settings.chat_model_id,
        judge_model_id=env.get("BEDROCK_JUDGE_MODEL_ID", "") or Settings.judge_model_id,
        embed_model_id=env.get("BEDROCK_EMBED_MODEL_ID", "") or Settings.embed_model_id,
        embed_dimensions=int(_number(env, "EMBED_DIMENSIONS", 1024, int)),
        embed_requests_per_second=_number(env, "EMBED_REQUESTS_PER_SECOND", 1.0, float),
        llm_mode=mode,  # type: ignore[arg-type]
        live_budget_usd=budget,
        log_hash_salt=env.get("LOG_HASH_SALT", ""),
    )
