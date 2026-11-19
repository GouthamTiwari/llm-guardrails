"""Real Amazon Bedrock calls. Skipped by default; run with `uv run pytest -m live`.

Tiny calls: one chat reply of a few tokens and one embedding (well under USD 0.001), and
three grounding-judge calls (about USD 0.001; at most six with retries). All of them are
recorded in the project's spend ledger like any other live call.
"""

import numpy as np
import pytest

from llm_guardrails.embeddings import TitanEmbedder
from llm_guardrails.guards.grounding import check_grounding_async
from llm_guardrails.llm import BedrockLLM, Message
from llm_guardrails.settings import load_settings
from llm_guardrails.spend import SpendLedger
from llm_guardrails.types import SourceDoc

pytestmark = pytest.mark.live


@pytest.fixture(scope="module")
def settings():
    s = load_settings()
    if s.auth_method == "none":
        pytest.skip("no Bedrock credentials in .env")
    return s


async def test_live_converse(settings) -> None:
    ledger = SpendLedger(limit_usd=settings.live_budget_usd)
    before = ledger.total()
    llm = BedrockLLM(settings, ledger=ledger)
    result = await llm.converse(
        system="Reply with exactly one word.",
        messages=[Message("user", "Say: ok")],
        max_tokens=5,
        purpose="live_test",
    )
    assert result.text.strip()
    assert result.tokens_in > 0 and result.tokens_out > 0
    assert 0 < result.cost_usd < 0.001
    assert ledger.total() == pytest.approx(before + result.cost_usd)


async def test_live_titan_embedding(settings, tmp_path) -> None:
    emb = TitanEmbedder(settings, memo_dir=tmp_path)  # fresh memo: forces one real call
    vec = await emb.embed(["guardrails live test"], purpose="live_test")
    assert vec.shape == (1, settings.embed_dimensions)
    assert np.linalg.norm(vec[0]) == pytest.approx(1.0, rel=1e-3)
    assert emb.api_calls == 1


async def test_live_entailment_judge(settings) -> None:
    """Nova as the grounding judge on three sentences that all pass the deterministic checks:
    one faithful, one with a changed condition, one with an added fact."""
    source = SourceDoc(
        id="S2",
        text="Example Bank card disputes must be raised within 60 days of the statement "
        "date. Refunds for approved disputes are credited within 10 working days.",
    )
    answer = (
        "You must raise the dispute within 60 days of the statement date [S2]. "
        "Disputes must be raised within 60 days of the payment date [S2]. "
        "Refunds are credited within 10 working days by NEFT transfer [S2]."
    )
    ledger = SpendLedger(limit_usd=settings.live_budget_usd)
    llm = BedrockLLM(settings, ledger=ledger)
    result = await check_grounding_async(
        answer, [source], mode="strip", entailment="required", llm=llm
    )
    labels = [s.entailment for s in result.sentences]
    print("judge labels:", labels, "cost:", round(result.cost_usd, 6))
    assert result.entailment == "used" and len(labels) == 3
    assert labels[0] == "supported"
    assert labels[1] != "supported" and labels[2] != "supported"
    assert 0 < result.cost_usd < 0.005
