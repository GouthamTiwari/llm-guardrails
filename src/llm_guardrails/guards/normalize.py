"""Normaliser: undo obfuscation tricks before any detector looks at the text (docs/SPEC.md 5).

Two outputs, on purpose:
- `clean` is what the model will see: Unicode-normalised, invisible characters removed,
  digits folded to ASCII. Nothing else changes, so meaning is preserved.
- `scan` is a detection-only copy with every trick undone (homoglyphs, leetspeak, spaced
  letters, decoded base64/hex). It is never sent to the model and never logged, because
  aggressive folding would garble normal text.

Decoding is for scanning only: decoded text is never executed or forwarded.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import re
import unicodedata

from llm_guardrails.types import Action, GuardContext, ScanText, Verdict

# Step 2: zero-width, bidi-control and other invisible format characters.
_INVISIBLE = re.compile("[​-‏‪-‮⁠-⁤﻿­]")
# C0/C1 control characters except newline and tab (\r is dropped: \r\n becomes \n).
_CONTROL = re.compile("[\x00-\x08\x0b-\x1f\x7f-\x9f]")

# Step 5: lowercase Cyrillic/Greek letters that look like Latin ones. Applied only to tokens
# that also contain Latin letters, so genuine Russian or Greek words are left alone.
_HOMOGLYPHS = str.maketrans(
    {
        # Cyrillic
        "а": "a", "в": "b", "г": "r", "е": "e", "ё": "e", "к": "k", "м": "m",
        "о": "o", "р": "p", "с": "c", "т": "t", "у": "y", "х": "x", "ь": "b", "і": "i",
        "ї": "i", "ј": "j", "ѕ": "s", "ԁ": "d", "һ": "h", "ӏ": "l", "ԛ": "q", "ԝ": "w",
        # Greek
        "α": "a", "β": "b", "ε": "e", "η": "n", "ι": "i", "κ": "k", "ν": "v", "ο": "o",
        "ρ": "p", "τ": "t", "υ": "u", "χ": "x", "ϲ": "c", "ϳ": "j",
    }
)  # fmt: skip
_HOMOGLYPH_CHARS = frozenset(k for k in map(chr, _HOMOGLYPHS))
_WORD = re.compile(r"\w+")
_LATIN = re.compile(r"[a-z]")
# Greek, Cyrillic and Cyrillic Supplement: the scripts with Latin look-alikes. Accented Latin
# ("café") and Devanagari are not tricks, so they never count as mixed script.
_NON_LATIN_LETTER = re.compile("[Ͱ-ϿЀ-ԯ]")

# Step 6: leetspeak, only inside tokens that already contain a letter ("1gn0re", not "2026").
_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a",
                       "$": "s"})  # fmt: skip
_LEET_CHARS = frozenset("013457@$")
_TOKEN = re.compile(r"\S+")
# URLs and email addresses keep their digits and "@": folding them would turn
# "attacker@example.com" into "attackeraexample.com" and hide it from exfiltration rules.
_URL_OR_EMAIL = re.compile(r"://|^www\.|[^\s@]@[^\s@]+\.[a-z]{2,}", re.IGNORECASE)

# Step 7: "s y s t e m", "s.y.s.t.e.m", "s-y-s-t-e-m" (4+ single letters).
_SPACED = re.compile(r"(?<!\S)(?:[a-z] {1,2}){3,}[a-z](?!\S)")
_DOTTED = re.compile(r"(?<![\w.\-])(?:[a-z][.\-]){3,}[a-z](?![\w])")

# Step 8: encoded segments worth decoding for the scan.
_B64 = re.compile(r"[A-Za-z0-9+/]{16,}={0,2}")
_HEX = re.compile(r"(?:[0-9a-fA-F]{2}){8,}")
MAX_DECODE_DEPTH = 2
MAX_DECODED_SEGMENTS = 5
MAX_DECODED_CHARS = 4000
MIN_PRINTABLE_RATIO = 0.85


def _remove_invisible(text: str, flags: set[str]) -> str:
    stripped = _INVISIBLE.sub("", text)
    if stripped != text:
        flags.add("zero_width")
    text = stripped.replace("\r\n", "\n")
    stripped = _CONTROL.sub("", text)
    if stripped != text:
        flags.add("control_chars")
    return stripped


def _fold_digits(text: str) -> str:
    """Any Unicode decimal digit (e.g. Devanagari ४) becomes its ASCII digit."""
    if text.isascii():
        return text
    out = []
    for ch in text:
        if not ("0" <= ch <= "9"):
            d = unicodedata.decimal(ch, None)
            if d is not None:
                out.append(str(d))
                continue
        out.append(ch)
    return "".join(out)


def clean_text(raw: str, flags: set[str] | None = None) -> str:
    """Steps 1-3: what the model sees. Idempotent."""
    flags = set() if flags is None else flags
    # Remove invisible characters BEFORE NFKC: otherwise a removed joiner could let NFKC
    # compose characters on a second pass and `clean` would not be stable.
    text = _remove_invisible(raw, flags)
    text = unicodedata.normalize("NFKC", text)
    text = _remove_invisible(text, flags)  # NFKC never adds them, but stay safe
    return _fold_digits(text)


def _fold_homoglyphs(text: str, flags: set[str]) -> str:
    def fix(match: re.Match[str]) -> str:
        token = match.group(0)
        if not _LATIN.search(token) or not _NON_LATIN_LETTER.search(token):
            return token
        flags.add("mixed_script")
        if any(ch in _HOMOGLYPH_CHARS for ch in token):
            flags.add("homoglyph")
            return token.translate(_HOMOGLYPHS)
        return token

    return _WORD.sub(fix, text)


def _fold_leet(text: str, flags: set[str]) -> str:
    def fix(match: re.Match[str]) -> str:
        token = match.group(0)
        if not any(ch.isalpha() for ch in token) or not any(ch in _LEET_CHARS for ch in token):
            return token
        if _URL_OR_EMAIL.search(token):
            return token
        folded = token.translate(_LEET)
        if folded != token:
            flags.add("leetspeak")
        return folded

    return _TOKEN.sub(fix, text)


def _join_spaced(text: str, flags: set[str]) -> str:
    def spaced(match: re.Match[str]) -> str:
        flags.add("spaced_letters")
        # Two or more spaces mark a word break: "s y s t e m  p r o m p t" -> "system prompt".
        groups = re.split(r" {2,}", match.group(0))
        return " ".join(g.replace(" ", "") for g in groups)

    def dotted(match: re.Match[str]) -> str:
        flags.add("spaced_letters")
        return re.sub(r"[.\-]", "", match.group(0))

    return _DOTTED.sub(dotted, _SPACED.sub(spaced, text))


def _scan_fold(clean: str, flags: set[str]) -> str:
    """Steps 4-7 on an already-clean string."""
    text = clean.lower()
    text = _fold_homoglyphs(text, flags)
    text = _fold_leet(text, flags)
    return _join_spaced(text, flags)


def _printable_text(data: bytes) -> str | None:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return None
    if len(text) < 4:
        return None
    ok = sum(1 for ch in text if ch.isprintable() or ch in "\n\t")
    return text if ok / len(text) >= MIN_PRINTABLE_RATIO else None


def _decode_candidates(text: str, flags: set[str]) -> list[str]:
    """Decoded base64/hex segments found in `text` (one level, no recursion here)."""
    found: list[str] = []
    for match in _B64.finditer(text):
        token = match.group(0)
        # Plain long words ("internationalisation") also fit the alphabet; real base64 of
        # text nearly always mixes case or has digits/symbols.
        has_mix = (any(c.isupper() for c in token) and any(c.islower() for c in token)) or any(
            c in "0123456789+/=" for c in token
        )
        if len(token) % 4 or not has_mix:
            continue
        try:
            decoded = _printable_text(base64.b64decode(token, validate=True))
        except (binascii.Error, ValueError):
            continue
        if decoded:
            flags.add("base64")
            found.append(decoded)
    for match in _HEX.finditer(text):
        decoded = _printable_text(bytes.fromhex(match.group(0)))
        if decoded:
            flags.add("hex")
            found.append(decoded)
    return found


def _decode_all(clean: str, flags: set[str]) -> list[str]:
    segments: list[str] = []
    total = 0
    frontier = [clean]
    for _depth in range(MAX_DECODE_DEPTH):
        next_frontier: list[str] = []
        for text in frontier:
            for seg in _decode_candidates(text, flags):
                if len(segments) >= MAX_DECODED_SEGMENTS or total >= MAX_DECODED_CHARS:
                    return segments
                seg = clean_text(seg)[: MAX_DECODED_CHARS - total]
                segments.append(seg)
                total += len(seg)
                next_frontier.append(seg)
        frontier = next_frontier
        if not frontier:
            break
    return segments


def normalize(raw: str) -> ScanText:
    """Build the clean text, the detection-only scan text, decoded segments and flags."""
    flags: set[str] = set()
    clean = clean_text(raw, flags)
    scan = _scan_fold(clean, flags)
    decoded_raw = _decode_all(clean, flags)
    decoded = [_scan_fold(seg, flags) for seg in decoded_raw]
    if decoded:
        scan = scan + "".join("\n" + seg for seg in decoded)
    return ScanText(clean=clean, scan=scan, decoded=decoded, flags=flags)


class NormalizeGuard:
    """Always ALLOW: it only prepares data. Obfuscation flags are signals for the injection
    guard, never verdicts, because leetspeak and odd Unicode are common in normal chat.

    The ScanText is returned in `detail["scan"]` and the clean text in `modified_text`; the
    pipeline copies them into the GuardContext.
    """

    name = "normalize"
    security = True

    async def check(self, ctx: GuardContext) -> Verdict:
        # CPU work runs in a thread so run_guard's timeout can still fire.
        scan = await asyncio.to_thread(normalize, ctx.text)
        return Verdict(
            guard=self.name,
            action=Action.ALLOW,
            scores={"obfuscation_flags": float(len(scan.flags))},
            modified_text=scan.clean,
            detail={"scan": scan},
        )
