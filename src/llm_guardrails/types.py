"""Core types shared by every guard: actions, verdicts, reports, reason codes and errors.

See docs/SPEC.md section 4. Anything that may be logged is a plain Pydantic field; anything
that could carry user text (modified text, detail) is excluded from serialisation.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Action(StrEnum):
    ALLOW = "allow"
    REDACT = "redact"
    RETRY = "retry"
    BLOCK = "block"


# Precedence when combining verdicts: BLOCK > RETRY > REDACT > ALLOW.
ACTION_RANK: dict[Action, int] = {
    Action.ALLOW: 0,
    Action.REDACT: 1,
    Action.RETRY: 2,
    Action.BLOCK: 3,
}


def strongest_action(actions: Iterable[Action]) -> Action:
    """Return the highest-precedence action; ALLOW when there are none."""
    return max(actions, key=ACTION_RANK.__getitem__, default=Action.ALLOW)


# Entity types known before the detector registry exists (Task 04 extends this).
BUILTIN_ENTITY_TYPES: frozenset[str] = frozenset(
    {"AADHAAR", "PAN", "GSTIN", "CREDIT_CARD", "PHONE_IN", "EMAIL", "UPI_ID", "IFSC"}
)

# Every guard the pipeline knows about. Used to validate policy keys and metric labels.
GUARD_NAMES: tuple[str, ...] = (
    "normalize",
    "policy",
    "pii",
    "injection",
    "context",
    "schema",
    "grounding",
    "leak",
)


class Reason:
    """Stable reason codes (spec 4.1). Metric labels may only use these values."""

    POLICY_TOO_LONG = "policy.too_long"
    POLICY_TOO_MANY_DOCS = "policy.too_many_docs"
    POLICY_OFF_TOPIC = "policy.off_topic"
    POLICY_DENIED_ACTION = "policy.denied_action"
    POLICY_RISKY_TOOL = "policy.risky_tool_needs_approval"

    INJ_EMBED = "inj.embed"
    INJ_LLM = "inj.llm"
    INJ_OBFUSCATION = "inj.obfuscation"

    CTX_HIDDEN_CONTENT = "ctx.hidden_content"
    CTX_LINE_STRIPPED = "ctx.line_stripped"
    CTX_DOC_DROPPED = "ctx.doc_dropped"
    CTX_INJECTION = "ctx.injection"

    SCHEMA_INVALID = "schema.invalid"
    SCHEMA_TRUNCATED = "schema.truncated"
    SCHEMA_INVALID_AFTER_RETRIES = "schema.invalid_after_retries"

    GROUNDING_UNCITED = "grounding.uncited"
    GROUNDING_FABRICATED = "grounding.fabricated_citation"
    GROUNDING_NUMBER = "grounding.number_mismatch"
    GROUNDING_LOW_OVERLAP = "grounding.low_overlap"
    GROUNDING_QUANTIFIER = "grounding.quantifier_mismatch"
    GROUNDING_QUOTE = "grounding.quote_mismatch"
    GROUNDING_ENTAILMENT_UNAVAILABLE = "grounding.entailment_unavailable"
    GROUNDING_EMPTY_AFTER_STRIP = "grounding.empty_after_strip"

    LEAK_CANARY = "leak.canary"
    LEAK_SYSTEM_PROMPT = "leak.system_prompt"
    LEAK_EXFIL_IMAGE = "leak.exfil_image"
    LEAK_EXFIL_LINK = "leak.exfil_link"

    GUARD_ERROR = "guard.error"
    GUARD_TIMEOUT = "guard.timeout"

    # Templated codes: the suffix is a fixed vocabulary (entity type, rule category, ...).
    @staticmethod
    def pii_block(entity: str) -> str:
        return f"pii.block.{entity}"

    @staticmethod
    def pii_redact(entity: str) -> str:
        return f"pii.redact.{entity}"

    @staticmethod
    def inj_rule(category: str) -> str:
        return f"inj.rule.{category}"

    @staticmethod
    def grounding_entailment(label: str) -> str:
        return f"grounding.entailment_{label}"

    @staticmethod
    def leak_secret(kind: str) -> str:
        return f"leak.secret.{kind}"

    @staticmethod
    def leak_pii(entity: str) -> str:
        return f"leak.pii.{entity}"


_REASON_SHAPE = re.compile(r"^[a-z]+\.[a-z_]+(\.[A-Za-z0-9_]+)?$")


def is_valid_reason(code: str) -> bool:
    """Reason codes are short dotted identifiers, never free text (they become metric labels)."""
    return bool(_REASON_SHAPE.match(code)) and len(code) <= 64


class EntitySummary(BaseModel):
    """What may be logged about a detected entity: never the value itself."""

    type: str
    count: int
    hash_prefixes: list[str] = []


class Verdict(BaseModel):
    guard: str
    action: Action
    reasons: list[str] = []
    scores: dict[str, float] = {}
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    entities: list[EntitySummary] = []
    error_kind: Literal["error", "timeout"] | None = None
    # These can hold user text, so they never leave memory: excluded from model_dump/JSON.
    modified_text: str | None = Field(default=None, exclude=True, repr=False)
    detail: dict[str, Any] = Field(default_factory=dict, exclude=True, repr=False)

    @field_validator("reasons")
    @classmethod
    def _reasons_are_codes(cls, value: list[str]) -> list[str]:
        bad = [r for r in value if not is_valid_reason(r)]
        if bad:
            # Show only the count: a bad "reason" might be user text by mistake.
            raise ValueError(f"{len(bad)} reason(s) are not valid reason codes")
        return value


class PipelineReport(BaseModel):
    request_id: str
    decision: Literal["allow", "block"]
    stage: Literal["input", "output", "complete"]
    verdicts: list[Verdict]
    latency_ms: float
    cost_usd: float
    tokens_in: int
    tokens_out: int
    policy_name: str
    version: str


_DOC_ID = re.compile(r"^[A-Za-z0-9_-]{1,32}$")


class SourceDoc(BaseModel):
    """A retrieved document or source the model may cite, e.g. id 'S1'."""

    model_config = ConfigDict(frozen=True)

    id: str
    text: str

    @field_validator("id")
    @classmethod
    def _id_shape(cls, value: str) -> str:
        if not _DOC_ID.match(value):
            raise ValueError("source id must be 1-32 chars of letters, digits, '_' or '-'")
        return value


class ScanText(BaseModel):
    """Normaliser output (spec 5). `scan` and `decoded` are for detection only."""

    clean: str
    scan: str
    decoded: list[str] = []
    flags: set[str] = set()


@dataclass
class GuardContext:
    """Everything a guard may read for one request. Lives in memory only.

    `llm`, `embedder` and `policy` are typed loosely here to avoid import cycles; their
    protocols are defined in llm.py, embeddings.py and policy_config.py.
    """

    request_id: str
    text: str = ""
    scan: ScanText | None = None
    context_docs: list[SourceDoc] = field(default_factory=list)
    system_prompt: str = ""
    canary: str = ""
    output: str | None = None
    sources: list[SourceDoc] = field(default_factory=list)
    schema_name: str | None = None
    policy: Any = None
    pii_map: dict[str, str] = field(default_factory=dict, repr=False)
    llm: Any = None
    embedder: Any = None


# ----------------------------------------------------------------------------- errors


class GuardrailsError(Exception):
    """Base class for every error this package raises on purpose."""


class GuardTimeout(GuardrailsError):
    pass


class BudgetExceeded(GuardrailsError):
    pass


class UnknownModelPrice(GuardrailsError):
    pass


class SchemaValidationFailed(GuardrailsError):
    def __init__(self, attempts: int, errors: list[str], trace: list[Any] | None = None) -> None:
        super().__init__(f"output still invalid after {attempts} attempt(s)")
        self.attempts = attempts
        self.errors = errors
        # Per-attempt tokens, cost and error kinds, so a failure is still fully costed.
        self.trace = trace or []


class PolicyError(GuardrailsError):
    pass


class ReplayMissing(GuardrailsError):
    def __init__(self, keys: list[str]) -> None:
        super().__init__(f"{len(keys)} recorded response(s) missing; rerun with --live")
        self.keys = keys


class ConfigError(GuardrailsError):
    pass
