# Task 07: Policy guard
Branch: task/07-policy-guard | Date: 2026-11-19

## What changed
- `src/llm_guardrails/guards/policy.py`: `PolicyGuard` (spec 9.2), in order:
  - size limits: input over `max_input_chars` → `policy.too_long`, more than `max_context_docs` → `policy.too_many_docs`, any document over `max_doc_chars` → `policy.too_long`. These run first, so every later regex works on bounded input.
  - denied actions matched on the normaliser's scan text (so `DR0P  T4BLE` and `d r o p table` are still caught); the guard normalises by itself if the pipeline has not.
  - topic allow-list only for `allow_list` policies; inputs of 4 words or fewer are exempt (follow-ups like "and for GST?"); keyword matching uses word boundaries for ASCII and substrings for Devanagari.
  - `check_tool_call(name, args, policy)`: risky tools → `policy.risky_tool_needs_approval`; JSON-dumped arguments scanned for denied actions.
- `src/llm_guardrails/policies/banking_tax.yaml` (outside the file list): added a `hindi_hinglish` topic (खाता, खाते, केवाईसी, बैंक, जीएसटी, रिफंड, लोन, ऋण, ब्याज, टैक्स, आयकर, khata, paisa, paise, byaj, karz, jama). Without it the seed Hindi KYC question (inj029) was blocked as off-topic. The short stem "कर" is deliberately left out because it also matches "करें" ("do").

## How to verify
- `uv run pytest -q tests/test_policy_guard.py` → 21 passed

## Tests
- `tests/test_policy_guard.py` (21): seed tool-abuse rows inj019/inj020 blocked; four look-alike normal requests allowed (transfer to sister, OTP confirmation, "drop the table booking", approved transfer); obfuscated denied actions; all size limits at and over the boundary; topics enforced only by `banking_tax`; short follow-up exemption; 8 seed benign/hard-benign banking rows on-topic (English, Hindi, Hinglish); word-boundary and Devanagari keyword rules; tool-call gate; no user text in the verdict.
- Full suite: 201 passed (2 live tests deselected).

## Out of scope / follow-ups
- Under `banking_tax`, general questions such as "explain prompt injection" (seed inj022/inj023) are blocked as off-topic by design. The end-to-end report (Task 20) must break blocks down by reason so these are not mistaken for injection false positives.

## Self-review
- Acceptance criteria of Task 07 checked: all met.
