"""Record and replay model calls so the eval runs offline, free and byte-for-byte repeatable
(docs/SPEC.md 13.5).

- record mode: call the real client, append the answer to `data/recordings/<suite>.jsonl`.
- replay mode: answer from that file; a missing entry raises `ReplayMissing` (never a silent
  fake answer, which would quietly change the numbers).

The key is sha256 of the model, purpose and the canonical request, so changing a prompt
changes the key and forces a fresh recording.

Nonces are masked in the key. The guards wrap untrusted text between markers carrying a
fresh random code per call (`<<<TEXT 9f3a1c2e>>>`), so the exact prompt never repeats; keyed
on the raw text, no recorded call could ever be replayed. Every 8-hex code that appears
inside a `<<<...>>>` marker is replaced by a placeholder (everywhere in the request) before
hashing. Attacker text cannot add codes to that set: the guards turn `<<<` inside untrusted
text into `‹‹‹` before wrapping it.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from llm_guardrails.llm import LLMClient, LLMResult, Message
from llm_guardrails.types import ReplayMissing

Mode = Literal["record", "replay"]


# The nonce is the 8-hex code right before ">>>": `<<<TEXT 9f3a1c2e>>>`,
# `<<<UNTRUSTED_DOCUMENT id="S1" nonce="9f3a1c2e">>>`.
_MARKER_NONCE = re.compile(r'<<<[A-Z][^<>\n]{0,80}?[ "=]([0-9a-f]{8})"?>>>')
NONCE_PLACEHOLDER = "<nonce>"


# The leak guard's per-request canary (spec 12.1) sits in the system prompt: same problem.
_CANARY = re.compile(r"cnry-[0-9a-f]{24}")
CANARY_PLACEHOLDER = "cnry-<canary>"


def mask_nonces(texts: Sequence[str]) -> list[str]:
    """Replace this request's marker nonces and canary with fixed placeholders (see the
    module docstring)."""
    texts = [_CANARY.sub(CANARY_PLACEHOLDER, t) for t in texts]
    nonces = {n for t in texts for n in _MARKER_NONCE.findall(t)}
    if not nonces:
        return texts
    pattern = re.compile("|".join(sorted(nonces)))
    return [pattern.sub(NONCE_PLACEHOLDER, t) for t in texts]


def request_key(
    model_id: str,
    purpose: str,
    system: str,
    messages: Sequence[Message],
    temperature: float,
    max_tokens: int,
) -> str:
    system, *texts = mask_nonces([system, *(m.text for m in messages)])
    canonical = json.dumps(
        {
            "model": model_id,
            "purpose": purpose,
            "system": system,
            "messages": [[m.role, t] for m, t in zip(messages, texts, strict=True)],
            "temperature": round(float(temperature), 3),
            "max_tokens": int(max_tokens),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class RecordingLLM:
    def __init__(
        self, path: Path | str, *, mode: Mode, inner: LLMClient | None = None, model_id: str = ""
    ) -> None:
        if mode == "record" and inner is None:
            raise ValueError("record mode needs a real client to record from")
        self.path = Path(path)
        self.mode = mode
        self.inner = inner
        self.model_id = inner.model_id if inner is not None else model_id
        self._lock = threading.Lock()
        self._entries: dict[str, dict] = {}
        # Every key a replay could not find. Guards fail closed, so a ReplayMissing raised
        # inside one becomes a quiet BLOCK and would change eval numbers without anyone
        # noticing; the eval runner calls ensure_complete() after a run to catch that.
        self.missing: list[str] = []
        if self.path.is_file():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    self._entries[row["key"]] = row

    def __len__(self) -> int:
        return len(self._entries)

    def __bool__(self) -> bool:
        # With __len__ defined, an EMPTY recorder would be falsy, and `llm or fallback`
        # would silently drop it: the injection guard then saw "no classifier" and allowed a
        # grey-zone input instead of failing closed. A client object is always "present".
        return True

    def ensure_complete(self) -> None:
        """Raise ReplayMissing if any call during the run had no recording."""
        if self.missing:
            raise ReplayMissing(list(self.missing))

    async def converse(
        self,
        *,
        system: str,
        messages: Sequence[Message],
        temperature: float = 0.0,
        max_tokens: int = 512,
        purpose: str,
    ) -> LLMResult:
        key = request_key(self.model_id, purpose, system, messages, temperature, max_tokens)
        row = self._entries.get(key)
        if row is not None:
            return LLMResult(
                text=row["text"],
                stop_reason=row["stop_reason"],
                tokens_in=row["tokens_in"],
                tokens_out=row["tokens_out"],
                cost_usd=row["cost_usd"],
                latency_ms=row["latency_ms"],
                model_id=row["model_id"],
                replayed=True,
            )
        if self.mode == "replay":
            with self._lock:
                if key not in self.missing:
                    self.missing.append(key)
            raise ReplayMissing([key])
        assert self.inner is not None
        result = await self.inner.converse(
            system=system, messages=messages, temperature=temperature, max_tokens=max_tokens,
            purpose=purpose,
        )  # fmt: skip
        row = {
            "key": key,
            "purpose": purpose,
            "model_id": result.model_id,
            "text": result.text,
            "stop_reason": result.stop_reason,
            "tokens_in": result.tokens_in,
            "tokens_out": result.tokens_out,
            "cost_usd": result.cost_usd,
            "latency_ms": round(result.latency_ms, 1),
        }
        with self._lock:
            self._entries[key] = row
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return result
