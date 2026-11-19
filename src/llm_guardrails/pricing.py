"""On-demand Bedrock prices and per-call cost (docs/SPEC.md 2.2).

USD, region ap-south-1 (Mumbai). Prices change: update the table and PRICES_CHECKED_ON
together. An unknown model raises instead of silently costing $0, so cost reports can't lie.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from llm_guardrails.types import UnknownModelPrice

PRICES_CHECKED_ON = date(2026, 11, 16)
PRICE_REGION = "ap-south-1"


@dataclass(frozen=True)
class ModelPrice:
    input_per_1m: float = 0.0
    output_per_1m: float = 0.0


PRICES: dict[str, ModelPrice] = {
    "amazon.titan-embed-text-v2:0": ModelPrice(input_per_1m=0.024),
    # Global cross-region inference profile, billed at the global rate.
    "global.amazon.nova-2-lite-v1:0": ModelPrice(input_per_1m=0.35, output_per_1m=2.95),
    # Test doubles are priced like the real models so cost accounting is exercised end to end.
    "fake-llm": ModelPrice(input_per_1m=0.35, output_per_1m=2.95),
    "fake-embed": ModelPrice(input_per_1m=0.024),
}

# Amazon Bedrock Guardrails (optional baseline), USD per 1,000 text units of 1,000 chars.
GUARDRAILS_PRICE_PER_1K_UNITS: dict[str, float] = {
    "content_filter": 0.15,  # includes the prompt-attack filter
    "sensitive_information": 0.10,
}
GUARDRAILS_PRICES_CHECKED_ON = date(2026, 11, 19)


def price_for(model_id: str) -> ModelPrice:
    try:
        return PRICES[model_id]
    except KeyError:
        raise UnknownModelPrice(
            f"No price for model {model_id!r}; add it to llm_guardrails/pricing.py"
        ) from None


def cost_usd(model_id: str, tokens_in: int = 0, tokens_out: int = 0) -> float:
    if tokens_in < 0 or tokens_out < 0:
        raise ValueError("token counts cannot be negative")
    p = price_for(model_id)
    return (tokens_in * p.input_per_1m + tokens_out * p.output_per_1m) / 1_000_000


def estimate_tokens(text: str) -> int:
    """Deliberately high estimate: 3 characters per token (English averages about 4), at
    least 1. Only used to check the budget BEFORE a call; the real count from the response
    is what gets recorded."""
    return max(1, len(text) // 3)


def guardrails_cost_usd(chars: int, policies: list[str]) -> float:
    units = max(1, -(-chars // 1000))  # ceil: every started 1,000 characters is a unit
    return sum(units * GUARDRAILS_PRICE_PER_1K_UNITS[p] / 1000 for p in policies)
