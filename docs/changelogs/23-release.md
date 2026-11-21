# Task 23: README, polish, release
Branch: task/23-release | Date: 2026-11-21

## What changed
- `README.md`, the 90-second read:
  - the pitch;
  - a 3-command quickstart that needs no AWS;
  - a Mermaid diagram of the pipeline;
  - the threat table;
  - the generated results block, now also stating the whole-project live spend;
  - how the numbers were made (dev-only tuning, the test split evaluated once, the pre-test label and rule fix);
  - what still gets through;
  - design decisions in plain words, including Nova vs Claude Haiku and the PAN checksum caveat;
  - what this does not do;
  - using it in another project;
  - data and review status, with deepset credited;
  - the project layout.
- `docs/THREAT_MODEL.md`: a "where each threat is measured" table linking every threat to its report or test.
- `reports/spend.json` (new): the spend ledger aggregated by purpose (counts and USD only), so the README's project-cost line obeys the numbers rule. `guardrails spend --report` writes it, and the summary reads it.
- `src/llm_guardrails/eval/report.py`, `cli.py`: the spend report and its summary line.
- Version 1.0.0 (`pyproject.toml`, `__version__`, `uv.lock`); `tests/test_smoke.py` checks the two agree.
- All reports regenerated offline; the README block regenerated.

## Definition of done (spec 19)
- [x] `uv run pytest` green with Bedrock mocked: 904 passed.
- [ ] `uv run pytest -m live` with real Bedrock: 3 tests, about USD 0.001, run only after Goutham's yes (decision 1).
- [x] Aadhaar, GSTIN and card validators agree with the reference on 10,000 Hypothesis cases each (`tests/test_checksums.py`); look-alike seed rows pass untouched.
- [x] Injection test recall and benign FPR reported together, per layer and combined, with `what_got_through.md`.
- [x] Schema first-try vs after-retry validity reported; fail-closed tested.
- [x] Grounding reject precision/recall reported; correct abstention passes.
- [x] Guard exception or timeout → block, for every security guard (`tests/test_fail_closed.py`).
- [x] No raw PII in logs (`tests/test_no_pii_in_logs.py`).
- [x] README sections as listed.
- [x] CI green; no secrets in history; audit OK (checked at merge).
- [x] No employer code, data, tool names or screenshots.

### Checks run for this task
- **Secret scan of the full history:** the spec's loose pattern has 3 hits, all documentation (the spec's own detection regex, the task list describing the empty `AWS_BEARER_TOKEN_BEDROCK=` line, and the scan command itself). A strict scan for real key shapes (`ABSK…`, `AKIA…`, `ASIA…`, a filled bearer variable, an AWS secret) finds **0**.
- **Employer details:** the private kit's job details appear nowhere in tracked files. "Amazon" appears only as the AWS product name and as a shop name inside synthetic customer tickets.

## How to verify
- `uv sync`
- `uv run guardrails check "My PAN is ABCPE1234F"`
- `uv run guardrails eval --suite all`
- `uv run pytest -q tests/test_reports.py` (the README block equals the generated summary).

## Live spend
None in this task so far. The whole project is USD 0.6477 of the USD 5 cap (`reports/spend.json`).

## Out of scope / follow-ups
- Tagging `v1.0.0` and making the repository public are Goutham's call; nothing outward-facing happens without his yes.
- v1.1 list: the document classifier on by default (confirm on new documents); the two UPI detector fixes; routing more inputs to the classifier for out-of-domain recall; an optional `AnthropicLLM` client (decision 2) as a separate report row; the human spot-check of the 99-row review sample.

## Self-review
- Every number in the README is inside the generated block (checked by `tests/test_reports.py`) or is a qualitative statement pointing at its report; I checked the prose by hand.
- No AI attribution anywhere; authorship is Goutham Tiwari only (Rule 0).
