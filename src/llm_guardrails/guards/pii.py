"""PII guard: block, redact with typed placeholders, or allow, per entity type (spec 6.5-6.6).

- Redaction gives each distinct value one placeholder per request: `[AADHAAR_1]`, `[EMAIL_2]`.
  The same number written with different spacing gets the same placeholder.
- The placeholder map (placeholder -> original) lives only in memory, in the request's
  GuardContext. It is never logged, serialised or cached.
- Restore puts originals back into the final answer only for types the policy lists, and
  only for placeholders issued in this request.
- Placeholder-shaped text typed by the user is defused first, so an attacker cannot type
  `[EMAIL_1]` and get someone's real email restored into it.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Literal

from llm_guardrails.detectors.entities import EntityMatch, detect_entities, get_detector
from llm_guardrails.logging_utils import summarize_entities
from llm_guardrails.policy_config import Policy, load_policy
from llm_guardrails.types import Action, EntitySummary, GuardContext, Reason, Verdict

Stage = Literal["input", "output"]

PLACEHOLDER = re.compile(r"\[([A-Z][A-Z_]*)_(\d+)\]")

# Fixed, value-free messages shown to the user when a request is blocked for PII.
BLOCK_MESSAGES: dict[str, str] = {
    "CREDIT_CARD": "Please don't share card numbers here. Remove it and try again.",
    "AADHAAR": "Please don't share your Aadhaar number here. Remove it and try again.",
    "PAN": "Please don't share your PAN here. Remove it and try again.",
}
DEFAULT_BLOCK_MESSAGE = "Your message contains personal data that can't be accepted here."


def block_message(entity_types: list[str]) -> str:
    for etype in entity_types:
        if etype in BLOCK_MESSAGES:
            return BLOCK_MESSAGES[etype]
    return DEFAULT_BLOCK_MESSAGE


def escape_placeholders(text: str) -> str:
    """`[EMAIL_1]` typed by a user becomes `(EMAIL_1)`, which restore never touches."""
    return PLACEHOLDER.sub(lambda m: f"({m.group(1)}_{m.group(2)})", text)


@dataclass
class Redactor:
    """Issues placeholders for one request. Rebuildable from the request's pii_map so the
    input text and every context document share one numbering."""

    pii_map: dict[str, str] = field(default_factory=dict, repr=False)  # placeholder -> raw
    _by_value: dict[tuple[str, str], str] = field(default_factory=dict, repr=False)
    _counters: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_map(cls, pii_map: dict[str, str]) -> Redactor:
        r = cls()
        for placeholder, raw in pii_map.items():
            m = PLACEHOLDER.fullmatch(placeholder)
            if not m:
                continue
            etype, n = m.group(1), int(m.group(2))
            r.pii_map[placeholder] = raw
            r._by_value[(etype, get_detector(etype).normalize(raw))] = placeholder
            r._counters[etype] = max(r._counters.get(etype, 0), n)
        return r

    def placeholder_for(self, match: EntityMatch) -> str:
        key = (match.type, match.value)
        if key not in self._by_value:
            n = self._counters.get(match.type, 0) + 1
            self._counters[match.type] = n
            placeholder = f"[{match.type}_{n}]"
            self._by_value[key] = placeholder
            self.pii_map[placeholder] = match.raw
        return self._by_value[key]


@dataclass
class PiiOutcome:
    action: Action
    reasons: list[str]
    text: str  # redacted text (unchanged when nothing was redacted)
    entities: list[EntitySummary]
    blocked_types: list[str]
    suppressed: int


def apply_pii_policy(text: str, policy: Policy, stage: Stage, redactor: Redactor) -> PiiOutcome:
    """Detect, then apply the policy's action for each entity type at this stage.

    Reused by the leak guard on model output (stage="output").
    """
    result = detect_entities(
        text, min_confidence=policy.pii.min_confidence, types=policy.pii.entities.keys()
    )
    blocked: set[str] = set()
    redacted: set[str] = set()
    pieces: list[str] = []
    cursor = 0
    for m in result.matches:
        action = getattr(policy.pii.entities[m.type], stage)
        if action == "block":
            blocked.add(m.type)
        elif action == "redact":
            redacted.add(m.type)
            pieces.append(text[cursor : m.start])
            pieces.append(redactor.placeholder_for(m))
            cursor = m.end
    pieces.append(text[cursor:])
    entities = summarize_entities((m.type, m.value) for m in result.matches)

    if blocked:
        return PiiOutcome(
            action=Action.BLOCK,
            reasons=[Reason.pii_block(t) for t in sorted(blocked)],
            text=text,
            entities=entities,
            blocked_types=sorted(blocked),
            suppressed=result.suppressed,
        )
    if redacted:
        return PiiOutcome(
            action=Action.REDACT,
            reasons=[Reason.pii_redact(t) for t in sorted(redacted)],
            text="".join(pieces),
            entities=entities,
            blocked_types=[],
            suppressed=result.suppressed,
        )
    return PiiOutcome(Action.ALLOW, [], text, entities, [], result.suppressed)


def restore(text: str, pii_map: dict[str, str], restore_types: list[str] | set[str]) -> str:
    """Put originals back for allowed types; unknown or other placeholders stay as they are."""
    allowed = set(restore_types)
    if not allowed or not pii_map:
        return text

    def put_back(m: re.Match[str]) -> str:
        token = m.group(0)
        if m.group(1) in allowed and token in pii_map:
            return pii_map[token]
        return token

    return PLACEHOLDER.sub(put_back, text)


class PiiGuard:
    """Input-stage PII guard. Output-stage PII is handled by the leak guard (spec 12.4)."""

    name = "pii"
    security = True

    def __init__(self, policy: Policy | None = None) -> None:
        self.policy = policy

    def _run(self, ctx: GuardContext) -> Verdict:
        policy: Policy = ctx.policy or self.policy or load_policy("default")
        redactor = Redactor.from_map(ctx.pii_map)
        text = escape_placeholders(ctx.text)
        outcome = apply_pii_policy(text, policy, "input", redactor)
        detail = {
            "pii_map": dict(redactor.pii_map),
            "suppressed": outcome.suppressed,
            "blocked_types": outcome.blocked_types,
        }
        if outcome.action is Action.BLOCK:
            detail["user_message"] = block_message(outcome.blocked_types)
        return Verdict(
            guard=self.name,
            action=outcome.action,
            reasons=outcome.reasons,
            scores={"entities": float(sum(e.count for e in outcome.entities))},
            entities=outcome.entities,
            modified_text=None if outcome.action is Action.BLOCK else outcome.text,
            detail=detail,
        )

    async def check(self, ctx: GuardContext) -> Verdict:
        return await asyncio.to_thread(self._run, ctx)
