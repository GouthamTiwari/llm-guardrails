# Task 21: Optional baseline, Amazon Bedrock Guardrails
Branch: task/21-baseline-bedrock | Date: 2026-11-21 (run after a separate cost approval)

## What changed
- `src/llm_guardrails/eval/baseline_bedrock.py`:
  - **Guardrail:** creates one uniquely named, tagged guardrail with the prompt-attack filter at HIGH on input and the three managed PII types that overlap ours (EMAIL, PHONE, CREDIT_DEBIT_CARD_NUMBER), with input evaluation explicitly enabled. It waits for READY, then calls `ApplyGuardrail(source="INPUT")` on every injection and PII test text.
  - **Cleanup:** **deletes the guardrail in `finally`**, so it runs even if a call crashes, then confirms with `ListGuardrails` that nothing with our name prefix is left.
  - **Recording:** responses are recorded (scrubbed of the account-identifying ARN) so the report rebuilds offline; several runs merge (a later response replaces an earlier one, every run's header is kept).
  - **Scoring:** prompt-attack recall and false positives against ours; per-type PII precision/recall on the overlapping types; look-alike rows flagged; cost from the billed units; ApplyGuardrail latency.
  - **No permission or no API in the region:** the report says so and the task closes.
- `src/llm_guardrails/cli.py`: `guardrails baseline bedrock [--live] [--only injection|pii]`.
- Reports: `reports/baseline_bedrock_guardrails.json/.md`, a summary row, and the README results block.
- `data/recordings/baseline_bedrock.jsonl` (no account id, no ARNs).
- `docs/SPEC.md`: decision 15.

## Results (test split; ours = the shipped configuration)

| | Bedrock Guardrails | This project |
|---|---|---|
| Prompt attacks blocked | 38.0% (38/100) | 95.0% (95/100) |
| Normal messages blocked | 10.8% (11/102) | 5.9% (6/102) |
| Tricky-but-harmless blocked | 20.8% (11/53) | 11.3% (6/53) |
| EMAIL precision / recall | 64.7% / 100% | 100% / 100% |
| PHONE precision / recall | 56.2% / 90.0% | 90.9% / 100% |
| Card precision / recall | 61.1% / 100% | 100% / 100% |
| Rows without PII that were flagged | 21.9% (14/64) | 0.0% (0/64) |
| Aadhaar, PAN, GSTIN, UPI ID, IFSC | no managed type | supported |

- **ApplyGuardrail latency:** p50 155 ms, p95 266 ms.
- **What it means:** Amazon's managed PII types find the common identifiers but also flag order numbers, tracking codes and declined-payment references, because checksums are what separate an ID from a look-alike. Its prompt-attack filter catches generic jailbreak phrasing, not the banking-specific attacks in this test set.
- **Fairness caveat (for the README):** the test set was written for this project's threat model (Indian banking, Hindi and Hinglish). On generic English prompts (the deepset check) our own recall falls to 28%, so neither system wins everywhere.

## How to verify
- `uv run guardrails baseline bedrock` rebuilds the report from recordings, offline.
- `uv run pytest -q tests/test_baseline_bedrock.py` → 11 passed (botocore Stubber validates every request and response shape; no AWS).

## Tests
- `tests/test_baseline_bedrock.py` (11):
  - **Cleanup:** delete still runs when an apply call raises (a ThrottlingException stubbed on the second call); a full run records responses and verifies deletion; an unverified deletion is reported.
  - **Unavailable API:** permission denied on create raises `BaselineUnavailable`, and `run()` still writes an "unavailable" report.
  - **Config:** overlapping managed types only; the PII filter is evaluated on input.
  - **Recordings:** a second run replaces its rows and keeps both headers; `scrub` removes the ARN, and the committed recordings contain no ARN.
  - **Scoring helpers.**
- `tests/test_eval_reproducible.py` (+1): the baseline report rebuilds byte for byte from the recordings.
- `tests/test_cli.py`: the old stub test that would have run `baseline bedrock --live` (real AWS calls with `.env`) is replaced by an offline check in a temporary folder.
- Full suite: 882 passed (3 live tests deselected).

## Live spend
Separately approved; estimate given as under USD 0.10.
- **Run 1:** 349 ApplyGuardrail calls, USD 0.0524.
- **Run 2:** 147 calls, USD 0.0367.
- **Task total USD 0.0891.** Project live spend is USD 0.6477 of the USD 5 cap.
- **Cleanup:** both guardrails deleted; afterwards ListGuardrails shows 0 guardrails in the account (ap-south-1).

## Findings and follow-ups
- **My mistake, caught by checking a suspicious number:** the first run's PII config. A 0% recall from a mature managed service was not believable; the recorded usage showed 0 PII units billed, which traced it to the config. The fix and a regression test are in.
- **Privacy catch:** ApplyGuardrail responses carry the account id inside the guardrail ARN. It is scrubbed before saving and pinned by a test; the repo and its history were checked for the id (none).

## Self-review
- Acceptance of Task 21: create → `ApplyGuardrail` on the injection and PII test sets → delete in `finally` → `ListGuardrails` confirms none left, with cost under USD 0.10 logged; a stubbed-boto3 unit test proves delete runs when an apply call raises; a missing permission or API is reported and the task closes. All met.
- No secrets or account identifiers in commits; all data is synthetic.
