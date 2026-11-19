"""Builds the model, embedder and pipelines the API and CLI use, per GUARDRAILS_LLM_MODE.

- `fake`: no AWS at all. A deterministic stand-in plays every model role (answers with the
  first sentence of the first document and cites it; the injection classifier says benign;
  the grounding judge says supported). For CI, the quickstart and trying the API locally.
  Embedding similarity is off (its corpus vectors come from Titan, and fake vectors would
  compare meaninglessly against them), so injection relies on rules (+ the fake classifier).
- `replay`: no AWS either. Model calls are answered from `data/recordings/runtime.jsonl`;
  a call with no recording fails, and the guard that made it fails closed. Useful only for
  inputs that were recorded before, so the CLI says when that happened.
- `bedrock`: real Nova and Titan calls, paid, counted against the spend cap.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from llm_guardrails.guards.grounding import ABSTENTION
from llm_guardrails.llm import FakeLLM, Message
from llm_guardrails.logging_utils import set_hash_salt
from llm_guardrails.pipeline import Pipeline
from llm_guardrails.policy_config import Policy, bundled_policy_names, load_policy
from llm_guardrails.recording import RecordingLLM
from llm_guardrails.settings import Settings
from llm_guardrails.types import ConfigError

RECORDINGS = Path("data") / "recordings" / "runtime.jsonl"
_DOC = re.compile(r'<<<UNTRUSTED_DOCUMENT id="([^"]+)" nonce="[0-9a-f]+">>>\n(.*?)\n<<<END_', re.S)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s")


def _fake_answer(user: str) -> str:
    """First sentence of the first wrapped document, cited; the abstention if none."""
    m = _DOC.search(user)
    if not m:
        return ABSTENTION
    doc_id, text = m.group(1), m.group(2).strip()
    first = _SENTENCE_END.split(text, maxsplit=1)[0].rstrip(".!?")
    return f"{first} [{doc_id}]." if first else ABSTENTION


def _fake_respond(system: str, messages: list[Message]) -> str:
    if system.startswith("You are a security classifier"):
        return json.dumps({"label": "benign", "confidence": 0.6, "category": "benign",
                           "reason": "fake mode"})  # fmt: skip
    if system.startswith("You check whether"):
        return json.dumps({"label": "supported", "reason": "fake mode"})
    if system.startswith("You summarise customer support tickets"):
        return json.dumps({"kind": "ticket", "title": "Customer support request",
                           "severity": "medium", "tags": ["support"], "due_days": 3})  # fmt: skip
    return _fake_answer(messages[-1].text)


def fake_llm() -> FakeLLM:
    return FakeLLM(_fake_respond)


def make_clients(settings: Settings) -> tuple[Any, Any]:
    """(llm, embedder) for the configured mode."""
    if settings.llm_mode == "fake":
        return fake_llm(), None
    if settings.llm_mode == "replay":
        return RecordingLLM(RECORDINGS, mode="replay", model_id=settings.chat_model_id), None
    if settings.auth_method == "none":
        raise ConfigError("GUARDRAILS_LLM_MODE=bedrock needs Bedrock credentials in .env")
    from llm_guardrails.embeddings import TitanEmbedder
    from llm_guardrails.llm import BedrockLLM
    from llm_guardrails.spend import SpendLedger

    ledger = SpendLedger(limit_usd=settings.live_budget_usd)
    return BedrockLLM(settings, ledger=ledger), TitanEmbedder(settings, ledger=ledger)


class Pipelines:
    """One Pipeline per (policy, schema), built on first use and shared after that.
    Pipelines keep no per-request state, so sharing them across requests is safe."""

    def __init__(self, settings: Settings, llm: Any = None, embedder: Any = None) -> None:
        self.settings = settings
        # The salt from .env makes log hash prefixes comparable across runs; without this
        # call a random per-process salt was used even when LOG_HASH_SALT was set.
        set_hash_salt(settings.log_hash_salt)
        if llm is None:
            llm, embedder = make_clients(settings)
        self.llm = llm
        self.embedder = embedder
        self._cache: dict[tuple[str, str | None], Pipeline] = {}
        self._policies: dict[str, Policy] = {}

    def policy(self, name: str) -> Policy:
        if name not in bundled_policy_names():
            # Only bundled names: a request must not be able to load an arbitrary file path.
            raise ConfigError(f"unknown policy {name!r}")
        if name not in self._policies:
            self._policies[name] = load_policy(name)
        return self._policies[name]

    def get(self, policy_name: str = "default", schema_name: str | None = None) -> Pipeline:
        key = (policy_name, schema_name)
        if key not in self._cache:
            self._cache[key] = Pipeline(self.policy(policy_name), self.llm, self.embedder,
                                        schema_name=schema_name)  # fmt: skip
        return self._cache[key]
