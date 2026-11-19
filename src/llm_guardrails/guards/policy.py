"""Policy guard: cheap structural checks that run before anything expensive (spec 9.2).

1. Size limits: oversize input or too many/too large documents are blocked first, which
   also bounds the time every later regex can take.
2. Denied actions: regexes for destructive or unconfirmed actions ("DROP TABLE",
   "transfer ... without confirmation"), matched on the normaliser's scan text so spacing
   and leetspeak tricks don't hide them.
3. Topic allow-list (only when the policy asks for it): a banking bot should not be a free
   general-purpose assistant. Very short follow-ups ("and for GST?") are exempt.

`check_tool_call` is the same idea for agents: risky tools need a human, and tool
arguments are checked against the denied actions.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from llm_guardrails.guards.normalize import normalize
from llm_guardrails.policy_config import Policy, load_policy
from llm_guardrails.types import Action, GuardContext, Reason, Verdict

SHORT_FOLLOW_UP_WORDS = 4


def _keyword_hit(text: str, keyword: str) -> bool:
    keyword = keyword.lower()
    if keyword.isascii():
        return re.search(rf"(?<![a-z0-9]){re.escape(keyword)}(?![a-z0-9])", text) is not None
    return keyword in text  # Devanagari: word boundaries are unreliable, substring is fine


def topic_hits(policy: Policy, scan_text: str) -> int:
    """Number of allowed-topic keywords present in the (lower-case) scan text."""
    return sum(
        1 for topic in policy.topics.allowed for kw in topic.keywords if _keyword_hit(scan_text, kw)
    )


def denied_action_hit(policy: Policy, scan_text: str) -> bool:
    return any(p.search(scan_text) for p in policy.compiled_denied_actions())


class PolicyGuard:
    name = "policy"
    security = True

    def __init__(self, policy: Policy | None = None) -> None:
        self.policy = policy

    def _run(self, ctx: GuardContext) -> Verdict:
        policy: Policy = ctx.policy or self.policy or load_policy("default")
        limits = policy.limits
        text = ctx.text
        scores = {"chars": float(len(text)), "docs": float(len(ctx.context_docs))}

        def block(reason: str) -> Verdict:
            return Verdict(guard=self.name, action=Action.BLOCK, reasons=[reason], scores=scores)

        if len(text) > limits.max_input_chars:
            return block(Reason.POLICY_TOO_LONG)
        if len(ctx.context_docs) > limits.max_context_docs:
            return block(Reason.POLICY_TOO_MANY_DOCS)
        if any(len(doc.text) > limits.max_doc_chars for doc in ctx.context_docs):
            return block(Reason.POLICY_TOO_LONG)

        # Size is bounded now, so scanning is safe. Reuse the normaliser's output if present.
        scan = ctx.scan.scan if ctx.scan is not None else normalize(text).scan
        if denied_action_hit(policy, scan):
            return block(Reason.POLICY_DENIED_ACTION)

        if policy.topics.mode == "allow_list":
            hits = topic_hits(policy, scan)
            scores["topic_hits"] = float(hits)
            short = len(text.split()) <= SHORT_FOLLOW_UP_WORDS
            if hits < policy.topics.min_keyword_hits and not short:
                return block(Reason.POLICY_OFF_TOPIC)

        return Verdict(guard=self.name, action=Action.ALLOW, scores=scores)

    async def check(self, ctx: GuardContext) -> Verdict:
        return await asyncio.to_thread(self._run, ctx)


def check_tool_call(name: str, args: Any, policy: Policy | None = None) -> Verdict:
    """Gate an agent's tool call before it runs.

    Risky tools always need a human approval step (this guard cannot give one), and the
    JSON-dumped arguments are scanned for denied actions such as DROP TABLE.
    """
    policy = policy or load_policy("default")
    if name in policy.tools.risky:
        return Verdict(guard="policy", action=Action.BLOCK, reasons=[Reason.POLICY_RISKY_TOOL])
    dumped = json.dumps(args, ensure_ascii=False, default=str)
    if denied_action_hit(policy, normalize(dumped).scan):
        return Verdict(guard="policy", action=Action.BLOCK, reasons=[Reason.POLICY_DENIED_ACTION])
    return Verdict(guard="policy", action=Action.ALLOW)
