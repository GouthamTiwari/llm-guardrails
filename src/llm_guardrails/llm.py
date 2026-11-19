"""Chat model clients behind one small protocol (docs/SPEC.md 2.2, 13.5).

- `BedrockLLM`: real Amazon Bedrock Converse calls, costed and budget-checked.
- `FakeLLM`: scripted answers for unit tests (never touches the network).
- `RecordingLLM` (in recording.py): records real answers once, then replays them offline.

Every guard that needs a model (injection classifier, entailment judge, schema retries)
depends only on `LLMClient`, so tests swap in a fake and the eval swaps in replay.
"""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from llm_guardrails.pricing import cost_usd, estimate_tokens
from llm_guardrails.settings import Settings
from llm_guardrails.spend import SpendLedger
from llm_guardrails.types import ConfigError

BEARER_ENV_VAR = "AWS_BEARER_TOKEN_BEDROCK"


@dataclass(frozen=True)
class Message:
    role: Literal["user", "assistant"]
    text: str = field(repr=False)


@dataclass(frozen=True)
class LLMResult:
    text: str = field(repr=False)
    stop_reason: str
    tokens_in: int
    tokens_out: int
    cost_usd: float
    latency_ms: float
    model_id: str
    replayed: bool = False


class LLMClient(Protocol):
    model_id: str

    async def converse(
        self,
        *,
        system: str,
        messages: Sequence[Message],
        temperature: float = 0.0,
        max_tokens: int = 512,
        purpose: str,
    ) -> LLMResult: ...


# ----------------------------------------------------------------------------- Bedrock


def make_runtime_client(settings: Settings) -> Any:
    """A `bedrock-runtime` boto3 client with short timeouts and adaptive retries."""
    import boto3
    from botocore.config import Config

    config = Config(
        retries={"max_attempts": 4, "mode": "adaptive"},
        connect_timeout=5,
        read_timeout=30,
    )
    if settings.auth_method == "bearer":
        # boto3 reads the Bedrock API key only from the process environment.
        os.environ[BEARER_ENV_VAR] = settings.bearer_token
        return boto3.client("bedrock-runtime", region_name=settings.aws_region, config=config)
    if settings.auth_method == "iam":
        # An empty bearer variable would still count as "a token exists" to botocore.
        if not os.environ.get(BEARER_ENV_VAR, "").strip():
            os.environ.pop(BEARER_ENV_VAR, None)
        return boto3.client(
            "bedrock-runtime",
            region_name=settings.aws_region,
            aws_access_key_id=settings.access_key_id,
            aws_secret_access_key=settings.secret_access_key,
            config=config,
        )
    raise ConfigError(
        f"No Bedrock credentials: set {BEARER_ENV_VAR} or AWS_ACCESS_KEY_ID + "
        "AWS_SECRET_ACCESS_KEY in .env"
    )


def _response_text(response: dict) -> str:
    content = response.get("output", {}).get("message", {}).get("content", [])
    return "".join(block.get("text", "") for block in content if isinstance(block, dict))


class BedrockLLM:
    def __init__(
        self,
        settings: Settings,
        *,
        model_id: str | None = None,
        ledger: SpendLedger | None = None,
        client: Any = None,
    ) -> None:
        self.model_id = model_id or settings.chat_model_id
        self.ledger = ledger or SpendLedger(limit_usd=settings.live_budget_usd)
        self._settings = settings
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = make_runtime_client(self._settings)
        return self._client

    def _call(
        self, system: str, messages: Sequence[Message], temperature: float, max_tokens: int,
        purpose: str,
    ) -> LLMResult:  # fmt: skip
        prompt_chars = len(system) + sum(len(m.text) for m in messages)
        estimate = cost_usd(self.model_id, estimate_tokens("x" * prompt_chars), max_tokens)
        self.ledger.check(estimate)
        request: dict[str, Any] = {
            "modelId": self.model_id,
            "messages": [{"role": m.role, "content": [{"text": m.text}]} for m in messages],
            "inferenceConfig": {"maxTokens": max_tokens, "temperature": temperature},
        }
        if system:
            request["system"] = [{"text": system}]
        start = time.perf_counter()
        response = self.client.converse(**request)
        latency_ms = (time.perf_counter() - start) * 1000
        usage = response.get("usage", {})
        tokens_in = int(usage.get("inputTokens", 0))
        tokens_out = int(usage.get("outputTokens", 0))
        usd = cost_usd(self.model_id, tokens_in, tokens_out)
        self.ledger.record(
            purpose=purpose, model_id=self.model_id, tokens_in=tokens_in, tokens_out=tokens_out,
            usd=usd,
        )  # fmt: skip
        return LLMResult(
            text=_response_text(response),
            stop_reason=str(response.get("stopReason", "")),
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=usd,
            latency_ms=latency_ms,
            model_id=self.model_id,
        )

    async def converse(
        self,
        *,
        system: str,
        messages: Sequence[Message],
        temperature: float = 0.0,
        max_tokens: int = 512,
        purpose: str,
    ) -> LLMResult:
        # boto3 is blocking; a thread keeps the event loop (and guard timeouts) responsive.
        return await asyncio.to_thread(
            self._call, system, list(messages), temperature, max_tokens, purpose
        )


# ----------------------------------------------------------------------------- fake

Script = str | LLMResult | BaseException
Responder = Callable[[str, Sequence[Message]], Script]


@dataclass
class FakeCall:
    system: str = field(repr=False)
    messages: list[Message] = field(repr=False)
    temperature: float
    max_tokens: int
    purpose: str


class FakeLLM:
    """Returns scripted answers in order (or from a function). Raises scripted exceptions.

    `calls` records every request so tests can assert, for example, that a blocked request
    never reached the model.
    """

    def __init__(
        self,
        script: Sequence[Script] | Responder | None = None,
        *,
        model_id: str = "fake-llm",
        stop_reason: str = "end_turn",
        delay_s: float = 0.0,
    ) -> None:
        self.model_id = model_id
        self.stop_reason = stop_reason
        self.delay_s = delay_s
        self.calls: list[FakeCall] = []
        self._responder = script if callable(script) else None
        self._queue = list(script) if script is not None and not callable(script) else []

    async def converse(
        self,
        *,
        system: str,
        messages: Sequence[Message],
        temperature: float = 0.0,
        max_tokens: int = 512,
        purpose: str,
    ) -> LLMResult:
        self.calls.append(FakeCall(system, list(messages), temperature, max_tokens, purpose))
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        if self._responder is not None:
            item = self._responder(system, messages)
        elif self._queue:
            item = self._queue.pop(0)
        else:
            item = "ok"
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, LLMResult):
            return item
        tokens_in = estimate_tokens(system + "".join(m.text for m in messages))
        tokens_out = estimate_tokens(item)
        return LLMResult(
            text=item,
            stop_reason=self.stop_reason,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost_usd(self.model_id, tokens_in, tokens_out),
            latency_ms=0.0,
            model_id=self.model_id,
        )
