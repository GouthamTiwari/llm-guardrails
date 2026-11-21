"""Plug the guardrails into other projects without depending on them (docs/SPEC.md 16)."""

from llm_guardrails.adapters.cache import should_cache
from llm_guardrails.adapters.decorator import GuardedCall, guarded
from llm_guardrails.adapters.protocols import CacheLike, RagAnswerFn
from llm_guardrails.guards.policy import check_tool_call

__all__ = ["CacheLike", "GuardedCall", "RagAnswerFn", "check_tool_call", "guarded",
           "should_cache"]  # fmt: skip
