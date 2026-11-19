"""Grounding guard: an answer may only say what its cited sources say (docs/SPEC.md 11).

A RAG answer arrives with inline citations like "[S1]". The guard splits it into sentences,
decides which sentences make factual claims, and checks each claim against the sources it
cites, cheapest and most certain check first:

1. uncited            - a factual claim with no citation at all;
2. fabricated citation - it cites a source id that was never retrieved;
3. numbers            - every number must appear in a cited source ("Rs 10,000" vs "Rs 5,000");
4. quotes             - quoted text must appear word for word in a cited source;
5. quantifiers        - "every" / "always" / "never" the sources do not say (overgeneralising);
6. overlap            - most content words must come from the cited sources.

These checks are deterministic: no model call, no cost, same answer every time. They catch
the common, dangerous failures (wrong numbers, invented sources, missing citations) but not
a sentence that reuses the source's words to say something different; the optional
entailment judge (Task 14) is measured against that gap.

Hardening beyond the letter of spec 11.2 (SPEC decision 7): a sentence that contains a
number or a citation is always checked as factual, so a claim cannot slip past by being
phrased as a question, prefixed with "Please note", or tucked into an abstention.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Literal

from llm_guardrails.types import (
    Action,
    GuardContext,
    GuardrailsError,
    Reason,
    SourceDoc,
    Verdict,
)

# The fixed answer used when strip mode removes every factual sentence (spec 11.5).
ABSTENTION = "I could not find this in the provided documents."
DEFAULT_MIN_OVERLAP = 0.5

Mode = Literal["reject", "strip"]
Kind = Literal["factual", "abstention", "non_factual"]

# ----------------------------------------------------------------------------- normalising

# Format characters (zero-width space/joiner, bidi marks, soft hyphen...) are invisible, but
# they would split "every" into "ev ery" and hide it from the quantifier check.


def _drop_format_chars(text: str) -> str:
    return "".join(ch for ch in text if unicodedata.category(ch) != "Cf")


def clean(text: str) -> str:
    """NFKC (full-width digits become ASCII) and no invisible format characters."""
    text = _drop_format_chars(unicodedata.normalize("NFKC", text))
    return text.replace("’", "'").replace("‘", "'")


# ----------------------------------------------------------------------------- citations

_ID = r"[A-Za-z0-9_-]{1,32}"
# "[S1]", "[S1, S2]", "[S1][S2]". Any bracket holding id-shaped tokens counts: a bracket
# like "[note]" is then read as a citation to a source that does not exist and is
# rejected, which errs on the safe side.
CITATION = re.compile(rf"\[\s*({_ID}(?:\s*[,;]\s*{_ID})*)\s*\]")


def citations(sentence: str) -> list[str]:
    """Cited source ids in order of first appearance, without duplicates."""
    found: list[str] = []
    for m in CITATION.finditer(sentence):
        for cid in re.split(r"\s*[,;]\s*", m.group(1).strip()):
            if cid not in found:
                found.append(cid)
    return found


def strip_citations(sentence: str) -> str:
    return re.sub(r"\s+", " ", CITATION.sub(" ", sentence)).strip()


# ----------------------------------------------------------------------------- sentences

# Words whose full stop does not end a sentence. Matched case-insensitively just before
# the stop ("Rs. 5,000", "e.g. NEFT").
_ABBREVIATIONS = ("rs", "no", "e.g", "i.e", "etc", "vs", "dr", "mr", "ms")
_ABBREV_BEFORE = re.compile(
    r"(?:^|(?<=[\s(\[\"'“]))(?:" + "|".join(re.escape(a) for a in _ABBREVIATIONS) + r")$",
    re.IGNORECASE,
)
# A sentence ends at . ? ! (or the Hindi danda), optionally followed by closing quotes or
# brackets and by citations - "... days [S2]." and "... days. [S2]" both keep [S2] with
# their sentence - when what follows is whitespace + an uppercase letter, a digit, a
# non-ASCII letter (Hindi has no case), or the end of the text. "8.5" never splits: no
# whitespace follows the dot.
_BOUNDARY = re.compile(
    r"[.?!।][\"'”’)]*(?:\s*\[[^\[\]\n]{1,80}\])*"
    r"(?=\s+[\"“'(\[]?(?:[A-Z0-9]|[^\x00-\x7f])|\s*$)"
)
_BULLET = re.compile(r"^\s*(?:[-*•]|\d{1,2}[.)])\s+")


@dataclass(frozen=True)
class Sentence:
    text: str  # with its citations, as written
    line: int  # which input line it came from, so strip mode can rebuild the layout


def split_sentences(answer: str) -> list[Sentence]:
    """Spec 11.1. Lines (bullet items) are separate sentences; inside a line, split at
    sentence-ending punctuation unless it closes a protected abbreviation."""
    out: list[Sentence] = []
    for line_no, raw_line in enumerate(clean(answer).splitlines()):
        line = _BULLET.sub("", raw_line).strip()
        start = 0
        for m in _BOUNDARY.finditer(line):
            if line[m.start()] == "." and _ABBREV_BEFORE.search(line[start : m.start()]):
                continue
            piece = line[start : m.end()].strip()
            if piece:
                out.append(Sentence(piece, line_no))
            start = m.end()
        tail = line[start:].strip()
        if tail:
            out.append(Sentence(tail, line_no))
    return out


# ----------------------------------------------------------------------------- numbers

_WORD_VALUES = {
    w: i
    for i, w in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve thirteen "
        "fourteen fifteen sixteen seventeen eighteen nineteen twenty".split()
    )
    if i > 0
}
_MULTIPLIERS = {
    "hundred": 100, "thousand": 1_000, "lakh": 100_000, "lakhs": 100_000, "lac": 100_000,
    "lacs": 100_000, "crore": 10_000_000, "crores": 10_000_000,
}  # fmt: skip
# "one" is also a pronoun: "no one", "each one", "one of the conditions" are not numbers.
_ONE_NOT_A_NUMBER_BEFORE = {"no", "each", "any", "which", "the", "this", "that", "every"}

# Digits with optional thousands separators (Indian "1,00,000" too) and decimals. A digit
# glued to a letter on either side is not read: "S1" is a citation id, "10th" an ordinal,
# and "5k" shorthand is deliberately NOT parsed (documented limitation, pinned by a test).
_NUMBER_TOKEN = re.compile(r"(?<![\w.,])(\d+(?:,\d+)*(?:\.\d+)?)(?![\w]|[.,]\d)|([A-Za-z]+)")
_RUPEE_GLUED = re.compile(r"(?i)\b(rs|inr)\.?(?=\d)")


def _value(digits: str) -> float:
    return round(float(digits.replace(",", "")), 6)


def parse_numbers(text: str) -> set[float]:
    """Spec 11.3 check 3: every number in the text, normalised to a value.

    "Rs 5,000" / "₹5000" / "Rs.5000" -> 5000; "8.5%" -> 8.5; "60 days" -> 60;
    "two lakh" -> 200000; "5 crore" -> 50000000; "twenty" -> 20.
    """
    text = _RUPEE_GLUED.sub(r"\1 ", clean(text))
    values: set[float] = set()
    pending: float | None = None  # a base number that a multiplier word may still scale
    prev_word = ""

    def flush() -> None:
        nonlocal pending
        if pending is not None:
            values.add(round(pending, 6))
            pending = None

    for m in _NUMBER_TOKEN.finditer(text):
        digits, word = m.group(1), m.group(2)
        if digits is not None:
            flush()
            pending = _value(digits)
            prev_word = ""
            continue
        w = word.lower()
        if w in _MULTIPLIERS and pending is not None:
            pending *= _MULTIPLIERS[w]
        elif w in _WORD_VALUES and not (w == "one" and prev_word in _ONE_NOT_A_NUMBER_BEFORE):
            flush()
            pending = float(_WORD_VALUES[w])
        else:
            flush()
            if w in _MULTIPLIERS:  # "a lakh", "lakhs of" - a bare multiplier is its value
                values.add(float(_MULTIPLIERS[w]))
        prev_word = w
    flush()
    return values


# ----------------------------------------------------------------------------- words

# Fixed English stopword list (spec 11.3 check 6). Quantifiers are in it on purpose: they
# have their own check, and "all"/"any" carry no topic.
STOPWORDS = frozenset(
    """a about above after again against all also am an and any are as at be because been
    before being below between both but by can could did do does doing down during each
    either few for from further had has have having he her here hers him his how i if in
    into is it its itself just may me might more most must my no nor not now of off on
    once only or other our ours out over own per same shall she should so some such than
    that the their theirs them then there these they this those through to too under until
    up upon us very was we were what when where which while who whom why will with within
    would yes you your yours every always never none everyone each anyone someone""".split()
)
_WORD = re.compile(r"[\wऀ-ॿ]+")


def _stem(word: str) -> str:
    """Light plural strip ("loans" -> "loan"), applied to both sides so it only has to be
    consistent, not linguistically right."""
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def words(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(clean(text))]


def content_words(text: str) -> set[str]:
    """Lowercased, stemmed, no stopwords, no pure numbers (numbers have their own check)."""
    return {
        _stem(w)
        for w in words(text)
        if w not in STOPWORDS and not w.replace("_", "").isdigit() and w != "_"
    }


QUANTIFIERS = ("all", "every", "any", "always", "never", "none", "no one", "everyone", "each")


def quantifiers_in(text: str) -> set[str]:
    padded = " " + " ".join(words(text)) + " "
    return {q for q in QUANTIFIERS if f" {q} " in padded}


_QUOTE = re.compile(r"\"([^\"\n]+)\"|“([^”\n]+)”")


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", clean(text)).strip().lower()


def quotes_in(text: str) -> list[str]:
    """Quoted passages of at least 3 words (shorter quotes are usually terms, not claims)."""
    out = []
    for m in _QUOTE.finditer(clean(text)):
        q = (m.group(1) or m.group(2)).strip()
        if len(q.split()) >= 3:
            out.append(q)
    return out


# ----------------------------------------------------------------------------- classes

ABSTENTION_PATTERN = re.compile(
    r"not in the provided (?:documents|sources)|could not find|couldn't find|cannot find"
    r"|can't find|don'?t have (?:enough )?information|do not have (?:enough )?information"
    r"|the (?:documents|sources) do(?:es)? not (?:mention|say|contain)"
    r"|\bno information\b|दस्तावेज़ों में नहीं",
    re.IGNORECASE,
)
# Courtesy phrases must be the WHOLE sentence. "Please note that all loans are free" is a
# claim with a courtesy prefix, not courtesy, so a prefix cannot switch the checks off.
COURTESY_PATTERN = re.compile(
    r"(?:i )?hope (?:this|that|it) helps|(?:please )?let me know(?: if .*)?|please note"
    r"|feel free to (?:ask|reach out|contact)(?: .*)?|thank you(?: .*)?|thanks(?: .*)?"
    r"|happy to help",
    re.IGNORECASE,
)


def classify(sentence: str) -> Kind:
    """Spec 11.2, with SPEC decision 7: numbers and citations force 'factual'."""
    body = strip_citations(sentence)
    if citations(sentence) or parse_numbers(body):
        return "factual"
    if ABSTENTION_PATTERN.search(body):
        return "abstention"
    bare = body.rstrip(".!?।\"'” ").strip()
    if body.endswith("?") or len(bare.split()) < 4 or COURTESY_PATTERN.fullmatch(bare):
        return "non_factual"
    return "factual"


# ----------------------------------------------------------------------------- checks


@dataclass
class SentenceResult:
    index: int
    kind: Kind
    outcome: str  # "supported", "skipped", "abstention", or a grounding.* reason code
    overlap: float | None = None

    @property
    def failed(self) -> bool:
        return self.outcome.startswith("grounding.")


@dataclass(frozen=True)
class SourceFacts:
    """What the checks need from one source, worked out once per answer, not per sentence
    (a 2,000-sentence answer took 8 s when every sentence re-read its sources)."""

    numbers: frozenset[float]
    squashed: str
    quantifiers: frozenset[str]
    words: frozenset[str]

    @classmethod
    def of(cls, doc: SourceDoc) -> SourceFacts:
        return cls(
            numbers=frozenset(parse_numbers(doc.text)),
            squashed=_squash(doc.text),
            quantifiers=frozenset(quantifiers_in(doc.text)),
            words=frozenset(content_words(doc.text)),
        )


def check_sentence(
    sentence: str,
    sources: dict[str, SourceDoc],
    min_overlap: float = DEFAULT_MIN_OVERLAP,
    facts: dict[str, SourceFacts] | None = None,
) -> tuple[str, float | None]:
    """Spec 11.3 checks 1-7, in order; the first failure is the sentence's outcome."""
    cited = citations(sentence)
    if not cited:
        return Reason.GROUNDING_UNCITED, None
    if any(c not in sources for c in cited):
        return Reason.GROUNDING_FABRICATED, None
    body = strip_citations(sentence)
    facts = {} if facts is None else facts  # the caller's dict is filled in, as a cache
    for c in cited:
        if c not in facts:
            facts[c] = SourceFacts.of(sources[c])
    cited_facts = [facts[c] for c in cited]

    if not parse_numbers(body) <= set().union(*(f.numbers for f in cited_facts)):
        return Reason.GROUNDING_NUMBER, None

    # Each quote must sit inside ONE cited source: a quote stitched together from the end
    # of one document and the start of another was never said by either.
    for quote in quotes_in(body):
        if not any(_squash(quote) in f.squashed for f in cited_facts):
            return Reason.GROUNDING_QUOTE, None

    if not quantifiers_in(body) <= set().union(*(f.quantifiers for f in cited_facts)):
        return Reason.GROUNDING_QUANTIFIER, None

    claim = content_words(body)
    source_words = set().union(*(f.words for f in cited_facts))
    overlap = len(claim & source_words) / len(claim) if claim else 1.0
    if overlap < min_overlap:
        return Reason.GROUNDING_LOW_OVERLAP, overlap
    return "supported", overlap


@dataclass
class GroundingResult:
    action: Action
    outcome: str  # supported | correct_abstention | no_claims | rejected | stripped |
    # empty_after_strip
    reasons: list[str]
    sentences: list[SentenceResult] = field(default_factory=list)
    text: str | None = None  # the answer to show, when strip mode changed it

    def sentence_outcomes(self) -> list[dict[str, object]]:
        """What may be logged: index, class and outcome. Never the sentence text."""
        return [{"index": s.index, "kind": s.kind, "outcome": s.outcome} for s in self.sentences]


def _rebuild(kept: list[Sentence]) -> str:
    parts: list[str] = []
    for i, s in enumerate(kept):
        if i:
            parts.append("\n" if s.line != kept[i - 1].line else " ")
        parts.append(s.text)
    return "".join(parts)


def check_grounding(
    answer: str,
    sources: list[SourceDoc],
    *,
    mode: Mode = "reject",
    min_overlap: float = DEFAULT_MIN_OVERLAP,
) -> GroundingResult:
    """Spec 11.5: run every sentence, then decide for the whole answer."""
    by_id = {s.id: s for s in sources}
    facts: dict[str, SourceFacts] = {}  # filled lazily: only cited sources are analysed
    split = split_sentences(answer)
    results: list[SentenceResult] = []
    for i, sentence in enumerate(split):
        kind = classify(sentence.text)
        if kind == "factual":
            outcome, overlap = check_sentence(sentence.text, by_id, min_overlap, facts)
            results.append(SentenceResult(i, kind, outcome, overlap))
        else:
            results.append(
                SentenceResult(i, kind, "abstention" if kind == "abstention" else "skipped")
            )

    failed = [r for r in results if r.failed]
    reasons = sorted({r.outcome for r in failed})
    factual = [r for r in results if r.kind == "factual"]

    if not failed:
        if factual:
            outcome = "supported"
        elif any(r.kind == "abstention" for r in results) and not citations(answer):
            outcome = "correct_abstention"
        else:
            outcome = "no_claims"
        return GroundingResult(Action.ALLOW, outcome, [], results)

    if mode == "reject":
        return GroundingResult(Action.BLOCK, "rejected", reasons, results)

    kept = [s for s, r in zip(split, results, strict=True) if not r.failed]
    if not any(r.kind == "factual" and not r.failed for r in results):
        reasons = sorted({*reasons, Reason.GROUNDING_EMPTY_AFTER_STRIP})
        return GroundingResult(Action.REDACT, "empty_after_strip", reasons, results, ABSTENTION)
    return GroundingResult(Action.REDACT, "stripped", reasons, results, _rebuild(kept))


# ----------------------------------------------------------------------------- guard


class GroundingGuard:
    """Output guard. Reads the answer from `ctx.output` and the sources from `ctx.sources`;
    mode and min_overlap come from the policy's `grounding` section."""

    name = "grounding"
    security = True

    def __init__(self, *, mode: Mode | None = None, min_overlap: float | None = None) -> None:
        self.mode = mode
        self.min_overlap = min_overlap

    async def check(self, ctx: GuardContext) -> Verdict:
        if ctx.output is None:
            # Called before there is an answer: a wiring bug. Raising makes run_guard fail
            # closed instead of silently passing an answer nobody checked.
            raise GuardrailsError("grounding guard needs ctx.output")
        cfg = ctx.policy.grounding if ctx.policy is not None else None
        if cfg is not None and not cfg.enabled:
            return Verdict(guard=self.name, action=Action.ALLOW, detail={"outcome": "disabled"})
        mode: Mode = self.mode or (cfg.mode if cfg is not None else "reject")
        min_overlap = (
            self.min_overlap
            if self.min_overlap is not None
            else (cfg.min_overlap if cfg is not None else DEFAULT_MIN_OVERLAP)
        )

        result = check_grounding(ctx.output, ctx.sources, mode=mode, min_overlap=min_overlap)
        factual = [s for s in result.sentences if s.kind == "factual"]
        overlaps = [s.overlap for s in factual if s.overlap is not None]
        return Verdict(
            guard=self.name,
            action=result.action,
            reasons=result.reasons,
            scores={
                "sentences": float(len(result.sentences)),
                "factual": float(len(factual)),
                "failed": float(sum(s.failed for s in result.sentences)),
                "min_overlap_seen": min(overlaps) if overlaps else 1.0,
            },
            modified_text=result.text,
            detail={
                "outcome": result.outcome,
                "mode": mode,
                "sentence_outcomes": result.sentence_outcomes(),
            },
        )
