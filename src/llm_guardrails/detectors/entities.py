"""PII entity detectors: regex candidates + validation + context (docs/SPEC.md 6.1-6.4).

Pipeline per text:
1. every registered detector proposes candidate spans (regex);
2. each candidate must pass the detector's validation (checksum, layout, IIN prefix...);
3. context words nudge confidence up, and for number types a nearby "order id"/"tracking"
   word suppresses the match (about 10% of random 12-digit strings pass Verhoeff, so a
   checksum alone is not proof). Context words never CREATE a detection;
4. overlapping matches are resolved: longer span first, then higher priority
   (a GSTIN contains a PAN; a card number contains Aadhaar-length runs).

Matches hide their raw values from repr so a stray print cannot leak them.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

from llm_guardrails.detectors.checksums import (
    gstin_checksum_valid,
    luhn_valid,
    verhoeff_valid,
)

# A match may not sit inside a longer alphanumeric token.
_L = r"(?<![0-9A-Za-z])"
_R = r"(?![0-9A-Za-z])"
# Digit types additionally may not be glued to another digit group by one separator
# ("4111 1111 1111 1112" must not yield an Aadhaar from its first three groups).
_LD = _L + r"(?<!\d[ -])"
_RD = _R + r"(?![ -]\d)"

POSITIVE_WINDOW = 40  # characters before and after the match
NEGATIVE_WINDOW = 30  # characters before the match
POSITIVE_BONUS = 0.1

NEGATIVE_WORDS: tuple[str, ...] = (
    "order", "order id", "tracking", "awb", "invoice", "inv", "ref", "reference", "txn",
    "transaction id", "utr", "ticket", "case", "booking", "pnr", "sku", "serial",
)  # fmt: skip

HOLDER_TYPES = frozenset("ABCFGHJLPT")  # PAN 4th character (Income Tax Department list)
GST_STATE_CODES = frozenset(f"{n:02d}" for n in range(1, 39)) | {"97", "99"}


@dataclass(frozen=True)
class EntityMatch:
    type: str
    start: int
    end: int
    confidence: float
    raw: str = field(repr=False)  # exactly as written in the text
    value: str = field(repr=False)  # canonical form, used for hashing and dedupe


@dataclass
class DetectionResult:
    matches: list[EntityMatch]
    suppressed: int = 0  # dropped by negative context
    low_confidence: int = 0  # dropped by min_confidence


@dataclass(frozen=True)
class Candidate:
    start: int
    end: int
    raw: str


def _contains_word(window: str, word: str) -> bool:
    if word.isascii():
        return re.search(rf"(?<![a-z0-9]){re.escape(word)}(?![a-z0-9])", window) is not None
    return word in window  # Devanagari: word boundaries are unreliable, substring is fine


class EntityDetector:
    """Base class. A new entity type is one subclass plus `register(...)`."""

    type: str = ""
    priority: int = 0
    base_confidence: float = 0.7
    has_checksum: bool = False
    pattern: re.Pattern[str]
    positive_context: tuple[str, ...] = ()
    negative_context: tuple[str, ...] = ()  # non-empty only for number types
    # Lower-case variant allowed only when one of these words is near (e.g. "pan").
    case_insensitive_if_near: tuple[str, ...] = ()

    def candidates(self, text: str, lowered: str) -> Iterator[Candidate]:
        seen: set[tuple[int, int]] = set()
        for m in self.pattern.finditer(text):
            seen.add(m.span())
            yield Candidate(m.start(), m.end(), m.group(0))
        if self.case_insensitive_if_near:
            for m in self.pattern.finditer(text.upper()):
                if m.span() in seen:
                    continue
                window = lowered[max(0, m.start() - POSITIVE_WINDOW) : m.end() + POSITIVE_WINDOW]
                if any(_contains_word(window, w) for w in self.case_insensitive_if_near):
                    yield Candidate(m.start(), m.end(), text[m.start() : m.end()])

    def validate(self, raw: str) -> bool:
        return True

    def normalize(self, raw: str) -> str:
        return re.sub(r"[ -]", "", raw)

    def confidence(self, lowered: str, start: int, end: int, use_context: bool) -> float | None:
        """Final confidence, or None when negative context suppresses the match."""
        if not use_context:
            return self.base_confidence
        around = lowered[max(0, start - POSITIVE_WINDOW) : end + POSITIVE_WINDOW]
        positive = any(_contains_word(around, w) for w in self.positive_context)
        if self.negative_context and not positive:
            before = lowered[max(0, start - NEGATIVE_WINDOW) : start]
            if any(_contains_word(before, w) for w in self.negative_context):
                return None
        bonus = POSITIVE_BONUS if positive else 0.0
        return min(1.0, self.base_confidence + bonus)


class GstinDetector(EntityDetector):
    type = "GSTIN"
    priority = 90
    base_confidence = 0.95
    has_checksum = True
    pattern = re.compile(_L + r"\d{2}[A-Z]{5}\d{4}[A-Z][1-9A-Z]Z[0-9A-Z]" + _R)
    positive_context = ("gst", "gstin")
    case_insensitive_if_near = ("gst", "gstin")

    def validate(self, raw: str) -> bool:
        g = raw.upper()
        return (
            g[:2] in GST_STATE_CODES
            and g[5] in HOLDER_TYPES  # chars 3-12 are the holder's PAN
            and g[13] == "Z"
            and gstin_checksum_valid(g)
        )

    def normalize(self, raw: str) -> str:
        return raw.upper()


# Card network prefixes (IIN). Luhn alone accepts 1 in 10 random digit strings.
_IIN_RANGES: tuple[tuple[int, int, int], ...] = (
    # (prefix length, low, high)
    (1, 4, 4),  # Visa
    (2, 51, 55),  # Mastercard
    (4, 2221, 2720),  # Mastercard 2-series
    (2, 34, 34),  # Amex
    (2, 37, 37),  # Amex
    (2, 60, 60),  # RuPay
    (2, 65, 65),  # RuPay / Discover
    (2, 81, 82),  # RuPay
    (3, 508, 508),  # RuPay
    (4, 6011, 6011),  # Discover
    (3, 644, 649),  # Discover
    (4, 3528, 3589),  # JCB
    (2, 36, 36),  # Diners
    (2, 38, 38),  # Diners
    (3, 300, 305),  # Diners
)


def known_card_prefix(digits: str) -> bool:
    return any(len(digits) >= n and lo <= int(digits[:n]) <= hi for n, lo, hi in _IIN_RANGES)


class CardDetector(EntityDetector):
    type = "CREDIT_CARD"
    priority = 80
    base_confidence = 0.9
    has_checksum = True
    # Maximal runs of digit groups; card candidates are 13-19-digit sub-runs of whole groups.
    pattern = re.compile(_L + r"\d+(?:[ -]\d+)*" + _R)
    positive_context = ("card", "credit", "debit", "visa", "mastercard", "rupay", "amex")
    negative_context = NEGATIVE_WORDS

    def candidates(self, text: str, lowered: str) -> Iterator[Candidate]:
        for run in self.pattern.finditer(text):
            groups = list(re.finditer(r"\d+", run.group(0)))
            for i in range(len(groups)):
                digits = 0
                for j in range(i, len(groups)):
                    digits += len(groups[j].group(0))
                    if digits > 19:
                        break
                    if digits >= 13:
                        start = run.start() + groups[i].start()
                        end = run.start() + groups[j].end()
                        yield Candidate(start, end, text[start:end])

    def validate(self, raw: str) -> bool:
        digits = self.normalize(raw)
        return luhn_valid(digits) and known_card_prefix(digits)


class AadhaarDetector(EntityDetector):
    type = "AADHAAR"
    priority = 70
    base_confidence = 0.9
    has_checksum = True
    # Separators must be consistent: all spaces, all hyphens, or none.
    pattern = re.compile(_LD + r"[2-9]\d{3}(?P<sep>[ -]?)\d{4}(?P=sep)\d{4}" + _RD)
    positive_context = ("aadhaar", "aadhar", "uid", "आधार")
    negative_context = NEGATIVE_WORDS

    def validate(self, raw: str) -> bool:
        digits = self.normalize(raw)
        return len(digits) == 12 and digits[0] in "23456789" and verhoeff_valid(digits)


class PanDetector(EntityDetector):
    type = "PAN"
    priority = 60
    base_confidence = 0.7
    pattern = re.compile(_L + r"[A-Z]{3}[ABCFGHJLPT][A-Z]\d{4}[A-Z]" + _R)
    positive_context = ("pan", "permanent account")
    case_insensitive_if_near = ("pan", "permanent account")

    def validate(self, raw: str) -> bool:
        # The 10th character is a check letter whose algorithm is not public: no checksum.
        pan = raw.upper()
        # Format templates such as "AAAAA9999A" describe the layout; they are not a PAN.
        template = len(set(pan[:5])) == 1 and len(set(pan[5:9])) == 1
        return pan[3] in HOLDER_TYPES and not template

    def normalize(self, raw: str) -> str:
        return raw.upper()


class IfscDetector(EntityDetector):
    type = "IFSC"
    priority = 50
    base_confidence = 0.75
    pattern = re.compile(_L + r"[A-Z]{4}0[A-Z0-9]{6}" + _R)
    positive_context = ("ifsc", "branch")


class PhoneInDetector(EntityDetector):
    type = "PHONE_IN"
    priority = 40
    base_confidence = 0.7
    pattern = re.compile(
        r"(?<![0-9A-Za-z+])(?<!\d[ -])"
        r"(?:\+91[ -]?|0)?"
        r"(?:[6-9]\d{4}[ -]?\d{5}|[6-9]\d{2}[ -]?\d{3}[ -]?\d{4})" + _RD
    )
    positive_context = ("phone", "mobile", "call", "whatsapp", "contact", "फ़ोन", "मोबाइल")
    negative_context = NEGATIVE_WORDS

    def validate(self, raw: str) -> bool:
        national = self.normalize(raw)
        return len(national) == 10 and national[0] in "6789" and len(set(national)) > 1

    def normalize(self, raw: str) -> str:
        digits = re.sub(r"\D", "", raw)
        if raw.lstrip().startswith("+91"):
            digits = digits[2:]
        elif len(digits) == 11 and digits.startswith("0"):
            digits = digits[1:]
        return digits


class EmailDetector(EntityDetector):
    type = "EMAIL"
    priority = 30
    base_confidence = 0.95
    pattern = re.compile(
        r"(?<![A-Za-z0-9._%+-])[A-Za-z0-9._%+-]{1,64}@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*"
        r"\.[A-Za-z]{2,24}" + _R
    )

    def normalize(self, raw: str) -> str:
        return raw.lower()


KNOWN_UPI_HANDLES = frozenset(
    {"okaxis", "oksbi", "okhdfcbank", "okicici", "ybl", "ibl", "axl", "paytm", "upi", "apl",
     "ptyes", "ptsbi", "pthdfc", "ptaxis"}
)  # fmt: skip


class UpiDetector(EntityDetector):
    type = "UPI_ID"
    priority = 20
    base_confidence = 0.7
    # Handle has no dot (otherwise it is an email domain).
    pattern = re.compile(
        r"(?<![A-Za-z0-9._-])[A-Za-z0-9._-]{2,256}@[A-Za-z][A-Za-z0-9]{1,63}(?![A-Za-z0-9.@])"
    )
    positive_context = ("upi", "vpa", "gpay", "phonepe", "paytm")

    def normalize(self, raw: str) -> str:
        return raw.lower()

    def confidence(self, lowered: str, start: int, end: int, use_context: bool) -> float | None:
        score = super().confidence(lowered, start, end, use_context)
        handle = lowered[start:end].rsplit("@", 1)[-1]
        if score is not None and handle in KNOWN_UPI_HANDLES:
            score = max(score, 0.85)
        return score


# ----------------------------------------------------------------------------- registry

_REGISTRY: dict[str, EntityDetector] = {}


def register(detector: EntityDetector) -> EntityDetector:
    if not detector.type:
        raise ValueError("detector needs a type name")
    _REGISTRY[detector.type] = detector
    return detector


def registered_types() -> frozenset[str]:
    return frozenset(_REGISTRY)


def get_detector(entity_type: str) -> EntityDetector:
    return _REGISTRY[entity_type]


for _detector in (
    GstinDetector(),
    CardDetector(),
    AadhaarDetector(),
    PanDetector(),
    IfscDetector(),
    PhoneInDetector(),
    EmailDetector(),
    UpiDetector(),
):
    register(_detector)


def detect_entities(
    text: str,
    *,
    min_confidence: float = 0.6,
    types: Iterable[str] | None = None,
    validate: bool = True,
    use_context: bool = True,
) -> DetectionResult:
    """Find PII entities in `text` (pass the normaliser's `clean` text).

    `validate=False` and `use_context=False` exist only for the eval's ablation
    (regex-only vs regex + validation vs + context).
    """
    wanted = registered_types() if types is None else frozenset(types)
    lowered = text.lower()
    found: list[EntityMatch] = []
    suppressed = low = 0
    for etype in sorted(wanted):
        det = _REGISTRY[etype]
        for cand in det.candidates(text, lowered):
            if validate and not det.validate(cand.raw):
                continue
            conf = det.confidence(lowered, cand.start, cand.end, use_context)
            if conf is None:
                suppressed += 1
                continue
            if conf < min_confidence:
                low += 1
                continue
            found.append(
                EntityMatch(
                    type=etype,
                    start=cand.start,
                    end=cand.end,
                    confidence=round(conf, 4),
                    raw=cand.raw,
                    value=det.normalize(cand.raw),
                )
            )
    return DetectionResult(_resolve_overlaps(found), suppressed=suppressed, low_confidence=low)


def _resolve_overlaps(matches: list[EntityMatch]) -> list[EntityMatch]:
    ordered = sorted(
        matches, key=lambda m: (-(m.end - m.start), -_REGISTRY[m.type].priority, m.start)
    )
    kept: list[EntityMatch] = []
    for m in ordered:
        if all(m.end <= k.start or m.start >= k.end for k in kept):
            kept.append(m)
    return sorted(kept, key=lambda m: m.start)
