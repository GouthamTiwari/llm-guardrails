"""Target schemas for structured output (docs/SPEC.md 10.1).

All models are strict on purpose:
- no type coercion: "3" is not an int and "High" is not "high". Downstream code gets exactly
  the declared types, and the retry loop fixes the model's mistakes instead of hiding them;
- extra keys are forbidden: a surprise field is either a model error or an injection trying
  to add something, and neither should pass silently.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


Tag = Annotated[str, Field(pattern=r"^[a-z0-9-]{1,24}$")]


class TicketSummary(_Strict):
    kind: Literal["ticket"]
    title: str = Field(min_length=5, max_length=80)
    severity: Literal["low", "medium", "high", "critical"]
    tags: list[Tag] = Field(max_length=5)
    due_days: int = Field(ge=1, le=30)


class Refusal(_Strict):
    """What the model returns instead of a ticket for empty, hostile or off-topic input."""

    kind: Literal["refusal"]
    reason: Literal["empty_input", "hostile_input", "out_of_scope"]
    message: str = Field(min_length=1, max_length=200)


# The "kind" field decides which model applies, so errors point at the right fields.
TicketResult = Annotated[TicketSummary | Refusal, Field(discriminator="kind")]


class InjectionJudgement(_Strict):
    """Strict verdict from the injection classifier (spec 7.3)."""

    label: Literal["attack", "benign"]
    confidence: float = Field(ge=0.0, le=1.0)
    category: str = Field(max_length=40)
    reason: str = Field(max_length=200)


class EntailmentJudgement(_Strict):
    """Strict verdict from the grounding entailment judge (spec 11.4)."""

    label: Literal["supported", "contradicted", "not_enough_info"]
    reason: str = Field(max_length=200)


SCHEMAS: dict[str, TypeAdapter[Any]] = {
    # The public name is TicketSummary (datasets use it); it accepts a ticket OR a refusal.
    "TicketSummary": TypeAdapter(TicketResult),
    "InjectionJudgement": TypeAdapter(InjectionJudgement),
    "EntailmentJudgement": TypeAdapter(EntailmentJudgement),
}


def get_adapter(schema_name: str) -> TypeAdapter[Any]:
    try:
        return SCHEMAS[schema_name]
    except KeyError:
        raise KeyError(f"unknown schema {schema_name!r}; known: {sorted(SCHEMAS)}") from None


def json_schema(schema_name: str) -> dict[str, Any]:
    """The JSON Schema Pydantic generates, pasted into prompts so the model sees the rules."""
    return get_adapter(schema_name).json_schema()
