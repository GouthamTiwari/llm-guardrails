"""Injection guard (docs/SPEC.md 7). This file grows layer by layer:

- Layer 1: rules from `rules/injection_rules.yaml`. Fast, free, explainable, and brittle
  on their own: they catch phrasings someone thought of in advance.
- Layer 2: similarity to a corpus of known attacks, minus closeness to known normal
  requests (catches paraphrases the rules miss).
- Layer 3 (Task 11): a cheap LLM classifier, only for the grey zone.

Until layer 3 exists: block on a strong rule or on an embedding score >= t_block, allow
everything else, and block (fail closed) if the embedding layer itself fails.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml

from llm_guardrails.guards.normalize import normalize
from llm_guardrails.types import (
    Action,
    ConfigError,
    GuardContext,
    Reason,
    ScanText,
    Verdict,
)

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


# ----------------------------------------------------------------------------- layer 2

CORPUS_FILES = ("attack_corpus", "benign_anchors")
EMBED_MAX_CHARS = 2000  # one embedding call up to this length
WINDOW_CHARS = 1000  # longer texts are split into windows...
WINDOW_OVERLAP = 200  # ...that overlap, so an attack on a boundary is not cut in half


@dataclass(frozen=True)
class InjectionConfig:
    """Thresholds (spec 7.4). Starting values; Task 19's dev-split sweep replaces them."""

    t_block: float = 0.60  # embedding score that blocks on its own (provisional, decision 4)
    t_allow: float = 0.35  # below this, nothing suspicious (Task 11: grey zone in between)
    t_llm: float = 0.5  # classifier score that blocks (Task 11)
    benign_lambda: float = 0.5  # how hard a closer benign anchor pulls the score down
    use_embeddings: bool = True
    use_llm: bool = True

    @classmethod
    def from_policy(cls, policy: Any, base: InjectionConfig | None = None) -> InjectionConfig:
        base = base or cls()
        inj = getattr(policy, "injection", None)
        if inj is None:
            return base
        overrides = {
            k: v
            for k, v in {
                "t_block": inj.t_block,
                "t_allow": inj.t_allow,
                "t_llm": inj.t_llm,
                "benign_lambda": inj.benign_lambda,
            }.items()
            if v is not None
        }
        overrides |= {"use_embeddings": inj.use_embeddings, "use_llm": inj.use_llm}
        return InjectionConfig(**(base.__dict__ | overrides))


def embedding_text(text: str | ScanText) -> str:
    """What gets embedded: the normaliser's scan text, for corpus rows and inputs alike."""
    return text.scan if isinstance(text, ScanText) else normalize(text).scan


def embedding_windows(text: str) -> list[str]:
    if len(text) <= EMBED_MAX_CHARS:
        return [text]
    step = WINDOW_CHARS - WINDOW_OVERLAP
    return [text[i : i + WINDOW_CHARS] for i in range(0, len(text) - WINDOW_OVERLAP, step)]


def text_sha256(path: Path) -> str:
    """Fingerprint of a text file with line endings normalised to LF, so a Windows
    checkout (CRLF) and a Linux checkout give the same hash."""
    data = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def default_data_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "data"


@dataclass(frozen=True)
class AttackCorpus:
    attack: np.ndarray  # (n, dims), unit rows
    benign: np.ndarray  # (m, dims), unit rows
    attack_categories: tuple[str, ...]
    model_id: str
    dims: int

    @classmethod
    def load(cls, data_dir: Path | None = None) -> AttackCorpus:
        """Load committed vectors; refuse if the texts changed since they were embedded."""
        data_dir = data_dir or default_data_dir()
        meta_path = data_dir / "corpus_meta.json"
        if not meta_path.is_file():
            raise ConfigError("no corpus vectors: run python -m llm_guardrails.eval.corpus embed")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        arrays, categories = {}, ()
        for name in CORPUS_FILES:
            jsonl = data_dir / f"{name}.jsonl"
            if text_sha256(jsonl) != meta["files"][name]["sha256"]:
                raise ConfigError(f"{name}.jsonl changed since it was embedded: rebuild corpus")
            arrays[name] = np.load(data_dir / f"{name}.npy").astype(np.float32)
            if name == "attack_corpus":
                rows = [json.loads(x) for x in jsonl.read_text(encoding="utf-8").splitlines() if x]
                categories = tuple(r["category"] for r in rows)
        return cls(
            attack=arrays["attack_corpus"],
            benign=arrays["benign_anchors"],
            attack_categories=categories,
            model_id=meta["model_id"],
            dims=int(meta["dims"]),
        )


@lru_cache(maxsize=2)
def _cached_corpus(data_dir: str) -> AttackCorpus:
    return AttackCorpus.load(Path(data_dir))


@dataclass(frozen=True)
class EmbedScore:
    score: float
    s_attack: float
    s_benign: float
    nearest_category: str


def embed_score_from_vectors(
    windows: np.ndarray, corpus: AttackCorpus, benign_lambda: float
) -> EmbedScore:
    """score = s_attack - lambda * max(0, s_benign - s_attack), clipped to [0, 1], taking
    the window with the highest score. Vectors are unit length, so dot = cosine."""
    best: EmbedScore | None = None
    for vec in windows:
        attack_sims = corpus.attack @ vec
        i = int(np.argmax(attack_sims))
        s_attack = float(attack_sims[i])
        s_benign = float(np.max(corpus.benign @ vec)) if len(corpus.benign) else 0.0
        raw = s_attack - benign_lambda * max(0.0, s_benign - s_attack)
        cand = EmbedScore(
            score=min(1.0, max(0.0, raw)),
            s_attack=s_attack,
            s_benign=s_benign,
            nearest_category=corpus.attack_categories[i] if corpus.attack_categories else "",
        )
        if best is None or cand.score > best.score:
            best = cand
    assert best is not None
    return best


async def score_embeddings(
    scan: ScanText, embedder: Any, corpus: AttackCorpus, benign_lambda: float
) -> EmbedScore:
    if embedder.model_id != corpus.model_id or embedder.dims != corpus.dims:
        raise ConfigError("embedder does not match the corpus vectors (model or dims)")
    windows = embedding_windows(embedding_text(scan))
    vectors = await embedder.embed(windows, purpose="injection_embed")
    return embed_score_from_vectors(vectors, corpus, benign_lambda)


# ----------------------------------------------------------------------------- guard


class InjectionGuard:
    name = "injection"
    security = True

    def __init__(
        self,
        rules: tuple[Rule, ...] | None = None,
        *,
        config: InjectionConfig | None = None,
        corpus: AttackCorpus | None = None,
        embedder: Any = None,
        data_dir: Path | None = None,
    ) -> None:
        self.rules = rules
        self.config = config
        self.corpus = corpus
        self.embedder = embedder
        self.data_dir = data_dir

    def _corpus(self) -> AttackCorpus:
        return self.corpus or _cached_corpus(str(self.data_dir or default_data_dir()))

    async def check(self, ctx: GuardContext) -> Verdict:
        config = self.config or InjectionConfig()
        if ctx.policy is not None:
            config = InjectionConfig.from_policy(ctx.policy, config)
        scan = ctx.scan if ctx.scan is not None else normalize(ctx.text)
        rs = await asyncio.to_thread(score_rules, scan, self.rules)
        scores = {"rules": rs.score}
        detail: dict[str, Any] = {"rule_ids": rs.rule_ids}

        def verdict(action: Action, reasons: list[str]) -> Verdict:
            return Verdict(
                guard=self.name, action=action, reasons=reasons, scores=scores, detail=detail
            )

        if rs.score >= 1.0:
            reasons = [Reason.inj_rule(c) for c in rs.categories]
            if rs.from_decoded:
                reasons.append(Reason.INJ_OBFUSCATION)
            return verdict(Action.BLOCK, reasons)

        embedder = ctx.embedder or self.embedder
        if config.use_embeddings and embedder is not None:
            try:
                es = await score_embeddings(scan, embedder, self._corpus(), config.benign_lambda)
            except Exception as exc:  # noqa: BLE001
                # Spec 7.5: escalate to the LLM layer; until Task 11 adds it, fail closed.
                detail["embed_unavailable"] = type(exc).__name__
                return verdict(Action.BLOCK, [Reason.GUARD_ERROR])
            scores |= {"embed": es.score, "s_attack": es.s_attack, "s_benign": es.s_benign}
            detail["nearest_category"] = es.nearest_category
            if es.score >= config.t_block:
                return verdict(Action.BLOCK, [Reason.INJ_EMBED])
        return verdict(Action.ALLOW, [])
