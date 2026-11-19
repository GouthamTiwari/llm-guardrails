"""Text embeddings for the injection guard's similarity layer (docs/SPEC.md 2.2, 7.2, 13.5).

- `TitanEmbedder`: Amazon Titan Text Embeddings v2 (one InvokeModel call per text), paced to
  stay under the 60-requests-per-minute quota, memoised on disk in `.embed_cache/` so no
  text is ever paid for twice, and budget-checked.
- `FakeEmbedder`: deterministic vectors for unit tests (optionally hand-made ones).
- `VectorBundle`: a committed `.npy` + `.json` index of recorded vectors, so the eval can
  run offline from a fresh clone.

Vectors are float32 and unit length, so cosine similarity is a plain dot product.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from llm_guardrails.llm import make_runtime_client
from llm_guardrails.pricing import cost_usd, estimate_tokens
from llm_guardrails.settings import Settings
from llm_guardrails.spend import SpendLedger
from llm_guardrails.types import ReplayMissing

DEFAULT_MEMO_DIR = Path(".embed_cache")


class Embedder(Protocol):
    model_id: str
    dims: int

    async def embed(self, texts: Sequence[str], *, purpose: str) -> np.ndarray: ...


def embed_key(model_id: str, dims: int, text: str) -> str:
    return hashlib.sha256(f"{model_id}|{dims}|{text}".encode()).hexdigest()


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    norm = float(np.linalg.norm(v))
    return v / norm if norm > 0 else v


class Pacer:
    """Allows at most `rate` calls per second across threads (simple minimum spacing)."""

    def __init__(self, rate: float) -> None:
        self.interval = 1.0 / rate if rate > 0 else 0.0
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            delay = self._next - now
            self._next = max(now, self._next) + self.interval
        if delay > 0:
            time.sleep(delay)


class TitanEmbedder:
    def __init__(
        self,
        settings: Settings,
        *,
        ledger: SpendLedger | None = None,
        client: Any = None,
        memo_dir: Path | str | None = DEFAULT_MEMO_DIR,
        pacer: Pacer | None = None,
    ) -> None:
        self.model_id = settings.embed_model_id
        self.dims = settings.embed_dimensions
        self.ledger = ledger or SpendLedger(limit_usd=settings.live_budget_usd)
        self.memo_dir = Path(memo_dir) if memo_dir else None
        self.pacer = pacer or Pacer(settings.embed_requests_per_second)
        self._settings = settings
        self._client = client
        self.api_calls = 0

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = make_runtime_client(self._settings)
        return self._client

    def _memo_path(self, key: str) -> Path | None:
        return self.memo_dir / key[:2] / f"{key}.npy" if self.memo_dir else None

    def _embed_one(self, text: str, purpose: str) -> np.ndarray:
        key = embed_key(self.model_id, self.dims, text)
        path = self._memo_path(key)
        if path is not None and path.is_file():
            return np.load(path)
        self.ledger.check(cost_usd(self.model_id, estimate_tokens(text)))
        self.pacer.wait()
        body = json.dumps({"inputText": text, "dimensions": self.dims, "normalize": True})
        response = self.client.invoke_model(
            modelId=self.model_id,
            body=body,
            contentType="application/json",
            accept="application/json",
        )
        self.api_calls += 1
        payload = json.loads(response["body"].read())
        vector = _unit(np.array(payload["embedding"], dtype=np.float32))
        tokens = int(payload.get("inputTextTokenCount", 0))
        self.ledger.record(
            purpose=purpose, model_id=self.model_id, tokens_in=tokens, tokens_out=0,
            usd=cost_usd(self.model_id, tokens),
        )  # fmt: skip
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, vector)
        return vector

    def _embed_all(self, texts: Sequence[str], purpose: str) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dims), dtype=np.float32)
        return np.stack([self._embed_one(t, purpose) for t in texts])

    async def embed(self, texts: Sequence[str], *, purpose: str) -> np.ndarray:
        return await asyncio.to_thread(self._embed_all, list(texts), purpose)


class FakeEmbedder:
    """Deterministic unit vectors seeded by the text's hash, or hand-made ones from `table`."""

    def __init__(
        self, dims: int = 32, table: dict[str, Sequence[float]] | None = None, *,
        fail: BaseException | None = None,
    ) -> None:  # fmt: skip
        self.model_id = "fake-embed"
        self.dims = dims
        self.table = {k: _unit(np.array(v)) for k, v in (table or {}).items()}
        self.fail = fail
        self.calls: list[list[str]] = []

    def vector(self, text: str) -> np.ndarray:
        if text in self.table:
            return self.table[text]
        seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
        return _unit(np.random.default_rng(seed).standard_normal(self.dims))

    async def embed(self, texts: Sequence[str], *, purpose: str) -> np.ndarray:
        self.calls.append(list(texts))
        if self.fail is not None:
            raise self.fail
        if not texts:
            return np.zeros((0, self.dims), dtype=np.float32)
        return np.stack([self.vector(t) for t in texts])


class VectorBundle:
    """Recorded vectors committed next to a dataset: `<prefix>.npy` + `<prefix>.json`.

    The json holds {model_id, dims, keys: [...]} where row i of the .npy belongs to keys[i].
    """

    def __init__(self, prefix: Path | str, model_id: str, dims: int) -> None:
        self.prefix = Path(prefix)
        self.model_id = model_id
        self.dims = dims
        self._rows: dict[str, np.ndarray] = {}
        npy, index = self.prefix.with_suffix(".npy"), self.prefix.with_suffix(".json")
        if npy.is_file() and index.is_file():
            meta = json.loads(index.read_text(encoding="utf-8"))
            if meta["model_id"] == model_id and meta["dims"] == dims:
                matrix = np.load(npy)
                self._rows = {k: matrix[i] for i, k in enumerate(meta["keys"])}

    def __len__(self) -> int:
        return len(self._rows)

    def get(self, text: str) -> np.ndarray | None:
        return self._rows.get(embed_key(self.model_id, self.dims, text))

    def put(self, text: str, vector: np.ndarray) -> None:
        self._rows[embed_key(self.model_id, self.dims, text)] = _unit(vector)

    def save(self) -> None:
        self.prefix.parent.mkdir(parents=True, exist_ok=True)
        keys = sorted(self._rows)
        matrix = (
            np.stack([self._rows[k] for k in keys]).astype(np.float32)
            if keys
            else np.zeros((0, self.dims), dtype=np.float32)
        )
        np.save(self.prefix.with_suffix(".npy"), matrix)
        meta = {"model_id": self.model_id, "dims": self.dims, "keys": keys}
        self.prefix.with_suffix(".json").write_text(json.dumps(meta), encoding="utf-8")


class RecordedEmbedder:
    """Replay vectors from a VectorBundle; in record mode, fill gaps from a real embedder."""

    def __init__(self, bundle: VectorBundle, *, inner: Embedder | None = None) -> None:
        self.bundle = bundle
        self.inner = inner
        self.model_id = bundle.model_id
        self.dims = bundle.dims

    async def embed(self, texts: Sequence[str], *, purpose: str) -> np.ndarray:
        missing = [t for t in texts if self.bundle.get(t) is None]
        if missing:
            if self.inner is None:
                raise ReplayMissing([embed_key(self.model_id, self.dims, t) for t in missing])
            vectors = await self.inner.embed(missing, purpose=purpose)
            for text, vec in zip(missing, vectors, strict=True):
                self.bundle.put(text, vec)
        if not texts:
            return np.zeros((0, self.dims), dtype=np.float32)
        return np.stack([self.bundle.get(t) for t in texts])
