"""Injection guard (docs/SPEC.md 7). This file grows layer by layer:

- Layer 1 (this task): rules from `rules/injection_rules.yaml`. Fast, free, explainable,
  and brittle on their own: they catch phrasings someone thought of in advance.
- Layer 2 (Task 09): similarity to a corpus of known attacks (catches paraphrases).
- Layer 3 (Task 11): a cheap LLM classifier, only for the grey zone.

Until layers 2 and 3 exist, the guard blocks on a strong rule and allows everything else.
"""

from __future__ import annotations

import asyncio
import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Literal

import yaml

from llm_guardrails.guards.normalize import normalize
from llm_guardrails.types import Action, GuardContext, Reason, ScanText, Verdict

WEAK_RULE_WEIGHT = 0.35  # three weak hits are about one strong hit
OBFUSCATION_FLAG_WEIGHT = 0.15
# Flags that suggest deliberate hiding. Leetspeak and mixed script are left out on purpose:
# they are common in normal chat and would raise false positives.
OBFUSCATION_FLAGS = frozenset({"zero_width", "homoglyph", "spaced_letters", "base64", "hex"})
CATEGORIES = frozenset(
    {
        "direct_override", "roleplay_jailbreak", "system_prompt_extraction",
        "delimiter_confusion", "encoding_obfuscation", "multilingual", "indirect_document",
        "exfiltration", "tool_abuse",
    }
)  # fmt: skip


@dataclass(frozen=True)
class Rule:
    id: str
    category: str
    lang: str
    strength: Literal["strong", "weak"]
    pattern: re.Pattern[str]
    match: tuple[str, ...] = ()
    no_match: tuple[str, ...] = ()


def _parse_rules(text: str) -> tuple[Rule, ...]:
    data = yaml.safe_load(text)
    rules: list[Rule] = []
    seen: set[str] = set()
    for raw in data["rules"]:
        rid = raw["id"]
        if rid in seen:
            raise ValueError(f"duplicate rule id {rid}")
        seen.add(rid)
        if raw["category"] not in CATEGORIES:
            raise ValueError(f"rule {rid}: unknown category {raw['category']}")
        if raw["strength"] not in ("strong", "weak"):
            raise ValueError(f"rule {rid}: strength must be strong or weak")
        # NFKC the pattern exactly like the text it runs on, so Devanagari nukta letters
        # (ज़ can be one code point or two) match however the rule file was typed.
        pattern = unicodedata.normalize("NFKC", raw["pattern"])
        rules.append(
            Rule(
                id=rid,
                category=raw["category"],
                lang=raw["lang"],
                strength=raw["strength"],
                pattern=re.compile(pattern, re.IGNORECASE),
                match=tuple(raw.get("match", ())),
                no_match=tuple(raw.get("no_match", ())),
            )
        )
    return tuple(rules)


@lru_cache(maxsize=4)
def load_rules(path: str | None = None) -> tuple[Rule, ...]:
    """Bundled rules by default; a path lets a deployment ship its own rule file."""
    if path is not None:
        return _parse_rules(Path(path).read_text(encoding="utf-8"))
    resource = resources.files("llm_guardrails.rules").joinpath("injection_rules.yaml")
    return _parse_rules(resource.read_text(encoding="utf-8"))


@dataclass
class RuleScore:
    score: float
    strong: list[Rule] = field(default_factory=list)
    weak: list[Rule] = field(default_factory=list)
    from_decoded: bool = False  # a strong rule fired on decoded base64/hex text

    @property
    def categories(self) -> list[str]:
        return sorted({r.category for r in self.strong})

    @property
    def rule_ids(self) -> list[str]:
        return sorted({r.id for r in self.strong + self.weak})


def _main_scan(scan: ScanText) -> str:
    """The scan text without the decoded segments the normaliser appended."""
    if not scan.decoded:
        return scan.scan
    tail = sum(len(seg) + 1 for seg in scan.decoded)
    return scan.scan[: len(scan.scan) - tail]


def score_rules(scan: ScanText, rules: tuple[Rule, ...] | None = None) -> RuleScore:
    """Layer-1 score: 1.0 on any strong rule, else weak hits and obfuscation flags add up."""
    rules = rules if rules is not None else load_rules()
    main = _main_scan(scan)
    result = RuleScore(score=0.0)
    for rule in rules:
        in_main = rule.pattern.search(main) is not None
        in_decoded = any(rule.pattern.search(seg) for seg in scan.decoded)
        if not (in_main or in_decoded):
            continue
        if rule.strength == "strong":
            result.strong.append(rule)
            if in_decoded and not in_main:
                result.from_decoded = True
        else:
            result.weak.append(rule)
    if result.strong:
        result.score = 1.0
    else:
        flags = len(scan.flags & OBFUSCATION_FLAGS)
        weighted = WEAK_RULE_WEIGHT * len(result.weak) + OBFUSCATION_FLAG_WEIGHT * flags
        result.score = min(1.0, weighted)
    return result


class InjectionGuard:
    name = "injection"
    security = True

    def __init__(self, rules: tuple[Rule, ...] | None = None) -> None:
        self.rules = rules

    def _run(self, ctx: GuardContext) -> Verdict:
        scan = ctx.scan if ctx.scan is not None else normalize(ctx.text)
        rs = score_rules(scan, self.rules)
        scores = {"rules": rs.score}
        detail = {"rule_ids": rs.rule_ids}
        if rs.score >= 1.0:
            reasons = [Reason.inj_rule(c) for c in rs.categories]
            if rs.from_decoded:
                reasons.append(Reason.INJ_OBFUSCATION)
            return Verdict(
                guard=self.name, action=Action.BLOCK, reasons=reasons, scores=scores, detail=detail
            )
        return Verdict(guard=self.name, action=Action.ALLOW, scores=scores, detail=detail)

    async def check(self, ctx: GuardContext) -> Verdict:
        return await asyncio.to_thread(self._run, ctx)
