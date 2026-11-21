"""Duck-typed shapes that other projects plug in (docs/SPEC.md 16). Nothing here imports the
sibling repos: any object with the right methods works."""

from __future__ import annotations

from collections.abc import Awaitable
from typing import Any, Protocol, runtime_checkable

from llm_guardrails.types import SourceDoc


@runtime_checkable
class RagAnswerFn(Protocol):
    """A RAG answer function: (question, sources) -> answer text, sync or async. The answer
    should cite the sources by id, like [S1], so the grounding guard can check it."""

    def __call__(self, question: str, sources: list[SourceDoc]) -> str | Awaitable[str]: ...


@runtime_checkable
class CacheLike(Protocol):
    """Any key-value cache, e.g. the Semantic Cache project's: get returns None on a miss."""

    def get(self, key: str) -> Any: ...

    def set(self, key: str, value: Any) -> None: ...
