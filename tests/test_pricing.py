import pytest

from llm_guardrails.pricing import (
    PRICES_CHECKED_ON,
    cost_usd,
    estimate_tokens,
    guardrails_cost_usd,
    price_for,
)
from llm_guardrails.types import UnknownModelPrice


def test_known_prices() -> None:
    assert price_for("global.amazon.nova-2-lite-v1:0").output_per_1m == 2.95
    assert cost_usd("global.amazon.nova-2-lite-v1:0", 1_000_000, 1_000_000) == pytest.approx(3.30)
    assert cost_usd("amazon.titan-embed-text-v2:0", 1_000_000) == pytest.approx(0.024)
    assert PRICES_CHECKED_ON.year == 2026


def test_unknown_model_raises_instead_of_costing_zero() -> None:
    with pytest.raises(UnknownModelPrice):
        cost_usd("anthropic.some-model", 10, 10)


def test_negative_tokens_rejected() -> None:
    with pytest.raises(ValueError):
        cost_usd("fake-llm", -1, 0)


def test_estimate_is_on_the_high_side() -> None:
    assert estimate_tokens("") == 1
    assert estimate_tokens("x" * 300) == 100


def test_guardrails_cost_counts_started_units() -> None:
    assert guardrails_cost_usd(1, ["content_filter"]) == pytest.approx(0.00015)
    assert guardrails_cost_usd(1001, ["content_filter", "sensitive_information"]) == (
        pytest.approx(2 * 0.00015 + 2 * 0.0001)
    )
