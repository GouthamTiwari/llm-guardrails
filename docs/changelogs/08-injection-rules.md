# Task 08: Injection layer 1, rules
Branch: task/08-injection-rules | Date: 2026-11-19

## What changed
- `src/llm_guardrails/rules/injection_rules.yaml` (+ package `__init__.py`): 48 rules (37 strong, 11 weak) covering all nine categories and every family in spec 7.1: override (English), new authority, role-play jailbreak, extraction, role-tag and chat-template spoofing, end-of-document markers, indirect "note to AI" and HTML-comment instructions, exfiltration (markdown images with query strings, send-to-address, append user data, paste placeholders), tool abuse (SQL/shell destruction, actions without confirmation, approve-all), encoding hints, Hindi, Hinglish and Spanish. Every rule carries its own `match` and `no_match` examples. The `no_match` examples are the hard negatives: "ignore my previous message", "don't follow the old GST rules", "drop the table booking", "transfer money without a debit card", "enable dark mode".
- `src/llm_guardrails/guards/injection.py`: rule loading and validation (unique ids, known categories, strong/weak; patterns NFKC-normalised like the text they run on, so Devanagari nukta letters match however they were typed), `score_rules` (1.0 on any strong rule, else 0.35 per weak rule + 0.15 per obfuscation flag among zero_width/homoglyph/spaced_letters/base64/hex, capped at 1.0; main text and decoded segments checked separately so a strong hit that exists only in decoded text sets `from_decoded`), and `InjectionGuard` (BLOCK with `inj.rule.<category>` reasons, plus `inj.obfuscation` for decoded hits; rule ids kept in `detail`, never serialised). Layers 2 and 3 arrive in Tasks 09 and 11.
- `src/llm_guardrails/guards/normalize.py` (Task 03 file, outside the list): leetspeak folding now skips URL and email tokens. Before, "attacker@example.com" became "attackeraexample.com" in the scan text, so no exfiltration rule could ever see an address. Test added to `tests/test_normalize.py`.

## How to verify
- `uv run pytest -q tests/test_injection_rules.py` → 123 passed

## Tests
- `tests/test_injection_rules.py` (123): rule count and category/language coverage; every rule's own positive and negative examples (48 parametrised cases); the 10 seed attacks the task requires score 1.0; all 10 seed benign rows (6 hard benign) score below 1.0; layer-1 coverage of all 20 seed attacks; decoded-only detection flagged as obfuscation; weak-rule and flag arithmetic; guard verdicts; rule-file validation; every rule runs in under 50 ms on a 20,000-character adversarial string and on 20,000 repeated characters (48 parametrised cases).
- Full suite: 325 passed (2 live tests deselected).

## Honest note on the seed numbers
- The rules catch all 20 seed attacks and none of the 10 seed benign rows. That is **not** a result. The rules were written while reading those same seed rows (they are dev data). The real measure is the held-out test split built in Task 18, which the rules will never have seen. Expect lower recall there, especially for paraphrases (layer 2's job) and for languages beyond Hindi, Hinglish and Spanish.

## Out of scope / follow-ups
- The decision with embeddings and the LLM grey zone (Task 11). Today the guard blocks only on strong rules.

## Self-review
- Acceptance criteria of Task 08 checked: at least 35 rules (48), positive and negative example per rule, seed thresholds, obfuscation reason, speed test. All met.
