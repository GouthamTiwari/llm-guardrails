"""Every security step, broken in turn, must make the request BLOCK (docs/SPEC.md 4.3).

Composition is where fail-open bugs hide: a guard can be perfect on its own and still be
skipped, or have its error swallowed, by the code that strings guards together. So each
guard is replaced by one that raises, then by one that hangs past its timeout, and the
whole request must come back blocked, with the error visible in the report.
"""

import asyncio

import pytest
from pipeline_helpers import S1, TICKET, bank_llm, policy

from llm_guardrails import pipeline as pipeline_module
from llm_guardrails.llm import FakeLLM
from llm_guardrails.pipeline import Pipeline
from llm_guardrails.types import GUARD_NAMES, Reason

QUESTION = "What is the minimum balance for a savings account?"


class Broken:
    security = True

    def __init__(self, name: str, how: str) -> None:
        self.name = name
        self.how = how

    async def check(self, ctx):
        if self.how == "raise":
            raise RuntimeError("guard crashed")
        await asyncio.sleep(5)  # far past the 0.05 s budget set below


def pipeline_for(name: str, how: str) -> Pipeline:
    p = policy()
    p.timeouts_s[name] = 0.05
    # The schema guard only runs when a schema is set; grounding only when there isn't one.
    schema = "TicketSummary" if name == "schema" else None
    llm = bank_llm(TICKET if schema else None)
    pipe = Pipeline(p, llm, schema_name=schema, guards={name: Broken(name, how)})
    if not schema:
        pipe.llm = bank_llm()
    return pipe


def test_every_security_guard_is_covered() -> None:
    assert set(GUARD_NAMES) == {"normalize", "policy", "pii", "injection", "context", "schema",
                                "grounding", "leak"}  # fmt: skip


@pytest.mark.parametrize("how", ["raise", "hang"])
@pytest.mark.parametrize("name", GUARD_NAMES)
async def test_broken_guard_blocks_the_request(name, how) -> None:
    pipe = pipeline_for(name, how)
    result = await pipe.run(QUESTION, [S1])
    assert result.decision == "block", f"{name} {how} let the request through"
    assert result.answer is None and result.data is None
    v = next(v for v in result.report.verdicts if v.guard == name)
    assert v.error_kind == ("error" if how == "raise" else "timeout")
    assert v.reasons == [Reason.GUARD_ERROR if how == "raise" else Reason.GUARD_TIMEOUT]
    assert result.report.verdicts[-1].guard == name  # nothing ran after it


@pytest.mark.parametrize("how", ["raise", "hang"])
@pytest.mark.parametrize("name", ["schema", "grounding", "leak"])
async def test_broken_output_guard_blocks_check_output(name, how) -> None:
    p = policy()
    p.timeouts_s[name] = 0.05
    pipe = Pipeline(p, FakeLLM(), guards={name: Broken(name, how)})
    schema = "TicketSummary" if name == "schema" else None
    result = await pipe.check_output(TICKET if schema else "Rs 5,000 [S1].", sources=[S1],
                                     schema_name=schema)  # fmt: skip
    assert result.decision == "block" and result.output is None


@pytest.mark.parametrize("how", ["raise", "hang"])
@pytest.mark.parametrize("name", ["normalize", "policy", "pii", "injection", "context"])
async def test_broken_input_guard_blocks_check_input(name, how) -> None:
    p = policy()
    p.timeouts_s[name] = 0.05
    pipe = Pipeline(p, FakeLLM(), guards={name: Broken(name, how)})
    result = await pipe.check_input(QUESTION, [S1])
    assert result.decision == "block" and result.sanitized_text is None


async def test_broken_document_redaction_blocks(monkeypatch) -> None:
    def boom(*a, **k):
        raise RuntimeError("detector crashed")

    monkeypatch.setattr(pipeline_module, "_redact_documents", boom)
    llm = bank_llm()
    result = await Pipeline(policy(), llm).run(QUESTION, [S1])
    assert result.decision == "block" and result.report.verdicts[-1].guard == "pii_docs"
    assert llm.calls == []


@pytest.mark.parametrize("how", ["raise", "hang"])
async def test_broken_model_call_blocks(how) -> None:
    p = policy()
    p.timeouts_s["model"] = 0.05
    llm = FakeLLM([RuntimeError("down")]) if how == "raise" else bank_llm(delay_s=5)
    result = await Pipeline(p, llm).run(QUESTION, [S1])
    assert result.decision == "block" and result.report.verdicts[-1].guard == "model"


async def test_policy_cannot_be_used_to_fail_open_silently() -> None:
    """fail_mode 'open' is an explicit, visible operator choice: the error is still
    reported, and the request goes on to the remaining guards."""
    p = policy()
    p.fail_mode["normalize"] = "open"
    pipe = Pipeline(p, bank_llm(), guards={"normalize": Broken("normalize", "raise")})
    result = await pipe.run(QUESTION, [S1])
    v = result.report.verdicts[0]
    assert v.error_kind == "error" and v.reasons == [Reason.GUARD_ERROR]
