# Task 05: PII guard: block, redact, restore, safe logging
Branch: task/05-pii-guard | Date: 2026-11-19

## What changed
- `src/llm_guardrails/guards/pii.py`:
  - `apply_pii_policy(text, policy, stage, redactor)`: detects entities (policy types and `min_confidence`), applies the policy's `input` or `output` action per type; any block → BLOCK, else any redact → REDACT, else ALLOW. Reused later by the leak guard for model output.
  - `Redactor`: one typed placeholder per distinct canonical value, numbered per type in order of appearance (`[AADHAAR_1]`, `[EMAIL_2]`); the same number written with different spacing gets the same placeholder. It can be rebuilt from the request's `pii_map`, so the question and every context document share one numbering.
  - `escape_placeholders`: user-typed `[EMAIL_1]` becomes `(EMAIL_1)` before redaction, so restore can never be tricked into filling an attacker's placeholder.
  - `restore(text, pii_map, types)`: puts originals back only for listed types and only for placeholders issued in this request.
  - `PiiGuard` (input stage): returns the redacted text in `modified_text` (None when blocked), the updated `pii_map`, a value-free `user_message` on block, entity summaries (type, count, salted hash prefixes); runs in a thread so the timeout works.
- `src/llm_guardrails/logging_utils.py`: `hash_value` (first 8 hex of sha256(salt + value)), `set_hash_salt` with fallback to `LOG_HASH_SALT` and then to a random per-process salt (one warning), `summarize_entities`, `safe_record` (the spec 14.1 audit record built from an allow-list of fields, with score keys, reason codes, routes and hash prefixes re-checked), `audit_log` (one JSON line on `llm_guardrails.audit`).

## How to verify
- `uv run pytest -q tests/test_pii_guard.py tests/test_logging_utils.py` → 19 passed
- `uv run python -c "from llm_guardrails.guards.pii import apply_pii_policy, Redactor; from llm_guardrails.policy_config import load_policy; print(apply_pii_policy('mail a@example.com, call 9876543210', load_policy('default'), 'input', Redactor()).text)"` → `mail [EMAIL_1], call [PHONE_IN_1]`

## Tests
- `tests/test_pii_guard.py` (12): typed placeholders, same value/same placeholder across spacing, card block with value-free message, allowed GSTIN/IFSC counted but untouched, numbering across calls, placeholder spoofing defused end to end, escape rules, restore rules (none / EMAIL only / banking_tax policy), redactor rebuild, output-stage actions, expected action for all 20 seed rows, and a dump test: every verdict (JSON, repr, dict), report and audit line for every seed row plus all captured logs contain none of the gold values (spaced or unspaced).
- `tests/test_logging_utils.py` (7): salted/stable/short hashes, random-salt warning once, env fallback, summaries, exact audit-record field lists with unsafe score keys and non-hex hashes dropped, entity merging, one JSON line per call.
- Full suite: 154 passed.

## Out of scope / follow-ups
- Running PII redaction over context documents and wiring `set_hash_salt(settings.log_hash_salt)` at startup happen in the pipeline (Task 16).

## Self-review
- Acceptance criteria of Task 05 checked: all met. Only the four listed files changed (plus this changelog).
- The placeholder map is only in `Verdict.detail` (excluded from every dump) and in memory.
