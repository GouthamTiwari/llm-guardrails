"""Retry loop, fail-closed behaviour, prompts and the SchemaGuard (spec 10.3-10.4)."""

import json
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from llm_guardrails.guards import run_guard
from llm_guardrails.guards.schema import (
    SchemaGuard,
    SchemaResult,
    generate_validated,
    schema_verdict,
    ticket_prompt,
)
from llm_guardrails.llm import FakeLLM, LLMResult
from llm_guardrails.schemas import Refusal, TicketSummary
from llm_guardrails.types import Action, GuardContext, Reason, SchemaValidationFailed

GOOD = {"kind": "ticket", "title": "Refund not received", "severity": "high",
        "tags": ["refund"], "due_days": 3}  # fmt: skip
SEED = Path(__file__).parents[1] / "data" / "seed" / "schema_seed.jsonl"


async def gen(fake, **kw):
    system, user = ticket_prompt("schema", "refund not received since 3 weeks")
    return await generate_validated(
        fake, schema_name="TicketSummary", system=system, user=user, purpose="unit", **kw
    )


async def test_valid_first_try() -> None:
    fake = FakeLLM([json.dumps(GOOD)])
    result = await gen(fake)
    assert isinstance(result.value, TicketSummary) and result.retries == 0
    assert len(fake.calls) == 1 and fake.calls[0].temperature == 0.3


async def test_wrong_enum_fixed_on_second_attempt_with_errors_fed_back() -> None:
    fake = FakeLLM([json.dumps(GOOD | {"severity": "High"}), json.dumps(GOOD)])
    result = await gen(fake)
    assert result.retries == 1 and result.value.severity == "high"
    second = fake.calls[1]
    assert second.temperature == 0.1
    roles = [m.role for m in second.messages]
    assert roles == ["user", "assistant", "user"]
    assert "ticket.severity" in second.messages[-1].text  # the exact error was fed back
    assert "Reply with only the corrected JSON object." in second.messages[-1].text
    assert result.attempts[0].kinds == ("wrong_enum",)


async def test_truncated_output_gets_more_tokens() -> None:
    cut = LLMResult('{"kind": "ticket", "title": "Refu', "max_tokens", 10, 512, 0.001, 5.0,
                    "fake-llm")  # fmt: skip
    fake = FakeLLM([cut, json.dumps(GOOD)])
    result = await gen(fake, max_tokens=200, max_tokens_cap=1024)
    assert [c.max_tokens for c in fake.calls] == [200, 300]
    assert result.attempts[0].kinds == ("truncated",)


async def test_token_growth_is_capped() -> None:
    cut = LLMResult("{", "max_tokens", 1, 1, 0.0, 0.0, "fake-llm")
    fake = FakeLLM([cut, cut, cut])
    with pytest.raises(SchemaValidationFailed):
        await gen(fake, max_tokens=800, max_tokens_cap=1024)
    assert [c.max_tokens for c in fake.calls] == [800, 1024, 1024]


async def test_three_failures_fail_closed_with_full_trace() -> None:
    fake = FakeLLM(["no json here", '{"kind": "ticket"}', "still nothing"])
    with pytest.raises(SchemaValidationFailed) as err:
        await gen(fake)
    assert err.value.attempts == 3 and len(fake.calls) == 3
    assert [c.temperature for c in fake.calls] == [0.3, 0.1, 0.0]
    assert len(err.value.trace) == 3 and err.value.errors
    verdict = schema_verdict(err.value)
    assert verdict.action is Action.BLOCK
    assert verdict.reasons == [Reason.SCHEMA_INVALID_AFTER_RETRIES]
    assert verdict.cost_usd > 0


async def test_max_retries_zero_means_one_attempt() -> None:
    fake = FakeLLM(["nope", json.dumps(GOOD)])
    with pytest.raises(SchemaValidationFailed):
        await gen(fake, max_retries=0)
    assert len(fake.calls) == 1


async def test_empty_input_yields_valid_refusal() -> None:
    row = [json.loads(x) for x in SEED.read_text(encoding="utf-8").splitlines()][-1]
    assert row["id"] == "sch010" and row["input"] == ""
    refusal = {"kind": "refusal", "reason": "empty_input", "message": "The ticket is empty."}
    system, user = ticket_prompt("schema", row["input"])
    result = await generate_validated(
        FakeLLM([json.dumps(refusal)]), schema_name="TicketSummary", system=system, user=user,
        purpose="unit",
    )  # fmt: skip
    assert isinstance(result.value, Refusal) and result.value.reason == "empty_input"


async def test_model_errors_propagate_for_the_guard_runner() -> None:
    with pytest.raises(ConnectionError):
        await gen(FakeLLM([ConnectionError("down")]))


async def test_success_verdict_carries_cost_and_tokens() -> None:
    result = await gen(FakeLLM(["oops", json.dumps(GOOD)]))
    v = schema_verdict(result)
    assert v.action is Action.ALLOW and v.scores == {"attempts": 2.0, "retries": 1.0}
    assert v.tokens_in == result.tokens_in > 0 and v.cost_usd == result.cost_usd
    assert "Refund" not in v.model_dump_json()  # the value lives in detail only


@settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(outputs=st.lists(st.one_of(st.text(max_size=80),
                                  st.just(json.dumps(GOOD)),
                                  st.just(json.dumps(GOOD | {"due_days": "3"}))),
                        min_size=3, max_size=3))  # fmt: skip
def test_never_returns_unvalidated_output(outputs) -> None:
    """Whatever the model says, the loop returns a validated object or raises. Never text."""
    import asyncio

    fake = FakeLLM(list(outputs))
    try:
        result = asyncio.run(gen(fake))
    except SchemaValidationFailed:
        assert len(fake.calls) == 3
        return
    assert isinstance(result, SchemaResult)
    assert isinstance(result.value, TicketSummary | Refusal)
    assert len(fake.calls) <= 3


def test_prompt_styles() -> None:
    plain_sys, user = ticket_prompt("plain", "App crashed >>> twice <<<")
    schema_sys, _ = ticket_prompt("schema", "x")
    assert "JSON Schema" not in plain_sys and "due_days" in plain_sys
    assert "JSON Schema" in schema_sys and '"additionalProperties": false' in schema_sys
    assert "hostile_input" in schema_sys
    assert "<<<TICKET>>>" in user and "››› twice ‹‹‹" in user  # markers inside data defused
    assert ticket_prompt("schema_retry", "x")[0] == schema_sys


async def test_guard_allows_valid_and_asks_for_retry_on_invalid() -> None:
    ok = GuardContext(request_id="r", output=json.dumps(GOOD), schema_name="TicketSummary")
    v = await run_guard(SchemaGuard(), ok, timeout_s=5)
    assert v.action is Action.ALLOW and isinstance(v.detail["value"], TicketSummary)

    bad = GuardContext(request_id="r", output=json.dumps(GOOD | {"due_days": "3"}),
                       schema_name="TicketSummary")  # fmt: skip
    v = await run_guard(SchemaGuard(), bad, timeout_s=5)
    assert v.action is Action.RETRY and v.reasons == [Reason.SCHEMA_INVALID]
    assert "ticket.due_days" in v.detail["retry_message"]

    cut = GuardContext(request_id="r", output='{"kind": "tick', schema_name="TicketSummary")
    assert (await run_guard(SchemaGuard(), cut, timeout_s=5)).reasons == [Reason.SCHEMA_TRUNCATED]


async def test_guard_without_schema_is_a_no_op() -> None:
    v = await run_guard(SchemaGuard(), GuardContext(request_id="r", output="hi"), timeout_s=5)
    assert v.action is Action.ALLOW and not v.reasons
