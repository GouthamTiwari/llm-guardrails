"""The Guard protocol and `run_guard`, the single place where timeouts and failures are handled.

Why one wrapper: fail-open bugs usually hide in scattered try/except blocks. Every guard runs
through `run_guard`, so "an exception or timeout never becomes ALLOW for a security guard"
is enforced (and tested) in exactly one place. See docs/SPEC.md 4.2 and 4.3.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Literal, Protocol, runtime_checkable

from llm_guardrails.types import Action, GuardContext, Reason, Verdict

logger = logging.getLogger("llm_guardrails.guards")

FailMode = Literal["closed", "open"]


@runtime_checkable
class Guard(Protocol):
    name: str
    security: bool  # security guards default to fail-closed

    async def check(self, ctx: GuardContext) -> Verdict: ...


def default_fail_mode(guard: Guard) -> FailMode:
    return "closed" if guard.security else "open"


def _failure_verdict(
    guard_name: str, fail_mode: FailMode, kind: Literal["error", "timeout"]
) -> Verdict:
    reason = Reason.GUARD_TIMEOUT if kind == "timeout" else Reason.GUARD_ERROR
    action = Action.BLOCK if fail_mode == "closed" else Action.ALLOW
    return Verdict(guard=guard_name, action=action, reasons=[reason], error_kind=kind)


async def run_guard(
    guard: Guard,
    ctx: GuardContext,
    timeout_s: float,
    fail_mode: FailMode | None = None,
) -> Verdict:
    """Run one guard with a timeout. Never raises (except on cancellation).

    - timeout          -> BLOCK (closed) or ALLOW (open), reason guard.timeout
    - any exception    -> same, reason guard.error; only the exception TYPE is logged,
                          because its message may contain user text
    - not a Verdict    -> treated as an exception (a buggy guard must not slip through)
    """
    mode: FailMode = fail_mode or default_fail_mode(guard)
    name = getattr(guard, "name", type(guard).__name__)
    start = time.perf_counter()
    try:
        verdict = await asyncio.wait_for(guard.check(ctx), timeout=timeout_s)
        if not isinstance(verdict, Verdict):
            raise TypeError("guard returned something that is not a Verdict")
        verdict = verdict.model_copy(update={"guard": name})
    except TimeoutError:
        logger.warning("guard %s timed out after %.3fs (fail_mode=%s)", name, timeout_s, mode)
        verdict = _failure_verdict(name, mode, "timeout")
    except asyncio.CancelledError:
        # Cancellation means the caller gave up on the whole request; let it propagate.
        raise
    except Exception as exc:  # noqa: BLE001 - every failure must become a verdict
        logger.warning("guard %s failed with %s (fail_mode=%s)", name, type(exc).__name__, mode)
        verdict = _failure_verdict(name, mode, "error")
    verdict.latency_ms = (time.perf_counter() - start) * 1000
    return verdict
