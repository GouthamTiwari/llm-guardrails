# Task 03: Normaliser
Branch: task/03-normalizer | Date: 2026-11-19

## What changed
- `src/llm_guardrails/guards/normalize.py`: `normalize(raw) -> ScanText` implementing spec 5.
  - `clean` (what the model sees): invisible/format characters and C0/C1 controls removed (newline and tab kept, `\r\n` → `\n`), NFKC, Unicode decimal digits folded to ASCII. Invisible characters are removed **before** NFKC so that `clean` is idempotent (a removed joiner could otherwise let NFKC compose characters on a second pass).
  - `scan` (detection only): lowercase, Cyrillic/Greek homoglyphs folded only inside tokens that also contain Latin letters, leetspeak folded only inside tokens that contain a letter, spaced/dotted/hyphenated single letters joined (two spaces = word break).
  - `decoded`: base64 and hex segments decoded for scanning only, kept when valid UTF-8 and ≥ 85% printable, up to 2 levels deep, max 5 segments and 4,000 characters. Base64 candidates must mix case or contain digits/symbols, so long plain words are not decoded.
  - Flags: `zero_width`, `control_chars`, `homoglyph`, `mixed_script`, `leetspeak`, `spaced_letters`, `base64`, `hex`. "Mixed script" only counts Greek/Cyrillic, so accented Latin ("café") and Devanagari are not flagged.
  - `NormalizeGuard`: always ALLOW; returns the clean text in `modified_text` and the `ScanText` in `detail["scan"]` (neither is ever serialised). The work runs in a thread so the guard timeout can fire.

## How to verify
- `uv run pytest -q tests/test_normalize.py` → 50 passed
- `uv run python -c "from llm_guardrails.guards.normalize import normalize; print(normalize('1gn0re pr3vious 1nstructi0ns').scan)"` → `ignore previous instructions`

## Tests
- `tests/test_normalize.py`: NFKC, zero-width, control chars, Devanagari digits, homoglyphs (and pure Cyrillic left alone), leetspeak, spaced/dotted letters, seed inj010 (base64) and inj011 (leet + spaced), hex, double base64, binary/plain-word/card-number non-decoding, decode caps, 70k-char input under 2 s, guard behaviour, every seed row; property tests: never raises + idempotent `clean` (300 examples), zero-width insertion into any seed attack never changes `scan` (200 examples).
- Full suite: 83 passed.

## Out of scope / follow-ups
- URL-safe base64 and base32 are not decoded (not in the spec); add if the eval shows misses.
- The homoglyph table is a fixed ~35-character list, not the full Unicode confusables file.

## Self-review
- Acceptance criteria of Task 03 checked: all met. Only the two listed files changed (plus this changelog).
- Decoded text is never executed, never sent to the model and never logged.
