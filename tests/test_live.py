"""Real Amazon Bedrock calls. Skipped by default; run with `uv run pytest -m live`.

Two tiny calls (one chat reply of a few tokens, one embedding), well under USD 0.001.
Both are recorded in the project's spend ledger like any other live call.
"""

import numpy as np
import pytest

from llm_guardrails.embeddings import TitanEmbedder
from llm_guardrails.llm import BedrockLLM, Message
from llm_guardrails.settings import load_settings
from llm_guardrails.spend import SpendLedger

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
