"""Record and replay model calls so the eval runs offline, free and byte-for-byte repeatable
(docs/SPEC.md 13.5).

- record mode: call the real client, append the answer to `data/recordings/<suite>.jsonl`.
- replay mode: answer from that file; a missing entry raises `ReplayMissing` (never a silent
  fake answer, which would quietly change the numbers).

The key is sha256 of the model, purpose and the canonical request, so changing a prompt
changes the key and forces a fresh recording.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from llm_guardrails.llm import LLMClient, LLMResult, Message
from llm_guardrails.types import ReplayMissing

Mode = Literal["record", "replay"]


def request_key(
    model_id: str,
    purpose: str,
    system: str,
    messages: Sequence[Message],
    temperature: float,
    max_tokens: int,
) -> str:
    canonical = json.dumps(
        {
            "model": model_id,
            "purpose": purpose,
            "system": system,
            "messages": [[m.role, m.text] for m in messages],
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
        if self.path.is_file():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    self._entries[row["key"]] = row

    def __len__(self) -> int:
        return len(self._entries)

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
