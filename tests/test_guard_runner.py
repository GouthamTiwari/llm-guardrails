"""run_guard: timeouts and exceptions must never turn into ALLOW for a security guard."""

import asyncio
import logging

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from llm_guardrails.guards import default_fail_mode, run_guard
from llm_guardrails.types import Action, GuardContext, Reason, Verdict

FAKE_AADHAAR = "2345 6789 0123"


class ScriptedGuard:
    def __init__(self, behaviour, *, security: bool = True, name: str = "scripted") -> None:
        self.name = name
        self.security = security
        self.behaviour = behaviour

    async def check(self, ctx: GuardContext) -> Verdict:
        return await self.behaviour(ctx)


def ctx() -> GuardContext:
    return GuardContext(request_id="r1", text="hello")


async def _allow(_ctx):
    return Verdict(guard="wrong-name", action=Action.ALLOW)


async def _raise(_ctx):
    raise ValueError(f"user said {FAKE_AADHAAR}")


async def _hang(_ctx):
    await asyncio.sleep(10)


async def _not_a_verdict(_ctx):
    return {"action": "allow"}


async def test_normal_verdict_passes_through_with_name_and_latency() -> None:
    v = await run_guard(ScriptedGuard(_allow), ctx(), timeout_s=1)
    assert v.action is Action.ALLOW
    assert v.guard == "scripted"
    assert v.latency_ms >= 0
    assert v.error_kind is None


async def test_exception_blocks_and_never_stores_message(caplog) -> None:
    caplog.set_level(logging.WARNING)
    v = await run_guard(ScriptedGuard(_raise), ctx(), timeout_s=1)
    assert v.action is Action.BLOCK
    assert v.reasons == [Reason.GUARD_ERROR]
    assert v.error_kind == "error"
    assert FAKE_AADHAAR not in v.model_dump_json()
    assert FAKE_AADHAAR not in caplog.text
    assert "ValueError" in caplog.text


async def test_timeout_blocks() -> None:
    v = await run_guard(ScriptedGuard(_hang), ctx(), timeout_s=0.01)
    assert v.action is Action.BLOCK
    assert v.reasons == [Reason.GUARD_TIMEOUT]
    assert v.error_kind == "timeout"


async def test_non_verdict_result_is_treated_as_error() -> None:
    v = await run_guard(ScriptedGuard(_not_a_verdict), ctx(), timeout_s=1)
    assert v.action is Action.BLOCK and v.error_kind == "error"


async def test_fail_open_allows_but_records_error() -> None:
    guard = ScriptedGuard(_raise, security=False)
    assert default_fail_mode(guard) == "open"
    v = await run_guard(guard, ctx(), timeout_s=1)
    assert v.action is Action.ALLOW
    assert v.error_kind == "error" and v.reasons == [Reason.GUARD_ERROR]


async def test_explicit_fail_mode_overrides_default() -> None:
    v = await run_guard(ScriptedGuard(_raise, security=True), ctx(), 1, fail_mode="open")
    assert v.action is Action.ALLOW
    v = await run_guard(ScriptedGuard(_raise, security=False), ctx(), 1, fail_mode="closed")
    assert v.action is Action.BLOCK


async def test_cancellation_propagates() -> None:
    task = asyncio.create_task(run_guard(ScriptedGuard(_hang), ctx(), timeout_s=5))
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


EXCEPTIONS = [
    ValueError,
    KeyError,
    RuntimeError,
    TypeError,
    ZeroDivisionError,
    OSError,
    ConnectionError,
    TimeoutError,
    MemoryError,
    RecursionError,
    UnicodeDecodeError,
    NotImplementedError,
]


def _make_exc(kind):
    if kind is UnicodeDecodeError:
        return UnicodeDecodeError("utf-8", b"\xff", 0, 1, "bad")
    return kind(FAKE_AADHAAR)


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    failure=st.sampled_from(["raise", "hang"]),
    exc=st.sampled_from(EXCEPTIONS),
    delay=st.floats(min_value=0, max_value=0.005),
)
def test_security_guard_never_allows_on_failure(failure, exc, delay) -> None:
    async def behaviour(_ctx):
        await asyncio.sleep(delay)
        if failure == "raise":
            raise _make_exc(exc)
        await asyncio.sleep(10)

    guard = ScriptedGuard(behaviour, security=True)
    verdict = asyncio.run(run_guard(guard, ctx(), timeout_s=0.02))
    assert verdict.action is Action.BLOCK
    assert verdict.error_kind in {"error", "timeout"}
    assert FAKE_AADHAAR not in verdict.model_dump_json()
