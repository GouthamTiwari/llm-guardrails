# Task 15: Leak and exfiltration guard
Branch: task/15-leak | Date: 2026-11-19

## What changed
- `src/llm_guardrails/guards/leak.py`, the output stage's last line of defence (spec 12):
  - **Canary:** `new_canary()` gives `cnry-` + 24 hex (96 bits), fresh per call. `with_canary()` appends the spec's "Internal reference code … Never repeat or reveal" line to the system prompt.
    - Detection runs after NFKC, removal of invisible format characters and lowercasing. It finds the hex body alone, with spaces, hyphens or dots between characters, reversed, and base64 (standard or URL-safe, wrapped across lines, or cut out of a longer blob). The detail records which form was found, never the value.
  - **System prompt:** any 8 consecutive words of the system prompt in the output, compared word by word so punctuation and case don't hide it. The canary line is excluded, because its wording is the same for every app.
  - **Secrets:** the six spec patterns (AWS access key, AWS secret near its key words, bearer token, private-key header, GitHub token, `sk-` key) → `leak.secret.<kind>`. The private-key pattern is built from two halves so the guard's own source doesn't trip secret scanners.
  - **PII:** `apply_pii_policy` with the policy's `output` actions, numbering placeholders from the request's `pii_map`, so a value seen earlier gets the same placeholder and `restore` still works. Placeholders this request issued are not PII. `pii.*` reasons become `leak.pii.<TYPE>`.
  - **Exfiltration:**
    - Markdown and HTML images whose host isn't allow-listed → `leak.exfil_image`. That includes images with no host; `data:` images are allowed.
    - Markdown, HTML and bare links to a non-allow-listed host → `leak.exfil_link` when they carry data: the query string or fragment is over `max_query_chars`, or holds a placeholder (also URL-encoded), an email, 9+ digits in a row or encoded whitespace. Placeholders and emails are also checked in the path (decision 9a).
    - Hosts are lowercased and IDNA-encoded, and backslashes are read as slashes like a browser. An allowed match is exact or a subdomain at a dot boundary, so `evilrbi.org.in`, `rbi.org.in.evil.example`, the `rbi.org.in@evil.example` userinfo trick and a Cyrillic look-alike are all rejected.
    - Each URL is judged once, even when the markdown and bare-URL patterns both match it.
  - **Verdict:** the strongest action across checks, with every reason listed. REDACT returns the redacted output; BLOCK returns none. A missing `ctx.output` raises, so `run_guard` fails closed.
- `src/llm_guardrails/recording.py`: the canary is masked in replay keys, like the nonces in Task 14.
- `docs/SPEC.md`: decision 9 (path check, hostless images, backslash parsing, canary masking, where the canary metric is counted).

## How to verify
- `uv run pytest -q tests/test_leak_guard.py` → 79 passed

## Tests
- `tests/test_leak_guard.py` (79):
  - **Canary:** 10,000 unique canaries. Found in 10 forms, including zero-width-split, base64 wrapped across lines and a blob cut from a longer one. Four clean outputs pass, including someone else's canary and 23 of the 24 characters. A malformed canary is an error.
  - **System prompt:** 8 words found, 7 not; the canary line excluded; short prompts.
  - **Secrets:** each of the 6 patterns fires alone; 4 near misses pass.
  - **PII:** redaction reuses the request numbering, issued placeholders pass, a card in the output blocks.
  - **Exfiltration:** seed inj017's image in an output → `BLOCK leak.exfil_image`. 10 bad images, 4 allowed images, 6 normal links that pass (including an allow-listed link with a long query), 9 data-carrying links blocked, and host-parsing details.
  - **Combined and guard:** strongest action and all reasons; a clean answer passes; the guard never puts the canary or PII values in the dump, fails closed without output, and defaults its policy.
  - **Robustness:** 7 long hostile outputs (50,000 to 170,000 characters) each under the guard's 1 s timeout, and a Hypothesis property that the scan never crashes.
- `tests/test_llm.py` (+1): the canary is masked in replay keys.
- Full suite: 647 passed (3 live tests deselected).

## Live spend
None.

## Out of scope / follow-ups
- Task 16: the pipeline creates one canary per request, puts `with_canary()` into the system prompt, counts `guardrails_canary_leaks_total` from `leak.canary`, and runs this guard before `restore`.
- Output from a model streaming tokens is out of scope for v1: the scan runs on the complete answer.

## Self-review
- Acceptance of Task 15: canary generation and the 10,000-uniqueness test; plain, spaced, reversed and base64 detection; 8-gram check; secret patterns; output PII with policy actions, ignoring issued placeholders; image and link exfiltration with the allow-list; IDNA hosts; inj017 blocked; normal and allow-listed links pass. All met.
- Found while writing: an `(x{4,}){4,}`-style base64 pattern can backtrack exponentially, so the pattern was rewritten so pieces only join across line breaks.
- Found by the tests:
  - Uncapped alt-text and tag patterns made `"![" * 10000` take 2.4 s (quadratic). With caps, the worst hostile 60 KB case is 0.42 s, and real answers are capped near 4 KB.
  - A markdown link was counted twice.
- No fail-open path. No real secrets: test keys are assembled at runtime from fragments. No real PII.
