# Baseline: Amazon Bedrock Guardrails vs this project

Mode: replay. Temporary guardrails (prompt-attack filter HIGH on input; managed PII types EMAIL, PHONE, CREDIT_DEBIT_CARD_NUMBER) applied to the 349 injection and PII test-split texts with ApplyGuardrail; each was deleted at the end of its run (deletion verified with ListGuardrails: yes).

| Run | Texts | PII filter evaluated on input | USD | Deleted and verified |
|---|---|---|---|---|
| 1 | 349 | no | 0.0524 | yes |
| 2 | 147 | yes | 0.0367 | yes |

Run 1 set only the legacy PII `action` field, so the PII filter was never evaluated on input (0 sensitive-information units billed). Run 2 repeated the PII rows with the input fields set; its responses replace run 1's for those rows. Injection results come from run 1, whose prompt-attack filter was configured correctly.

## Prompt injection (test split)

| System | Attacks blocked | Normal messages blocked | Tricky-but-harmless blocked |
|---|---|---|---|
| Bedrock Guardrails, prompt-attack HIGH | 38.0% (29.1–47.8, 38/100) | 10.8% (6.1–18.3, 11/102) | 20.8% (12.0–33.5, 11/53) |
| This project (shipped) | 95.0% (88.8–97.9, 95/100) | 5.9% (2.7–12.2, 6/102) | 11.3% (5.3–22.6, 6/53) |

## PII, types both systems support (test split)

| Type | Bedrock precision / recall | Ours precision / recall |
|---|---|---|
| EMAIL / EMAIL | 64.7% / 100.0% | 100.0% / 100.0% |
| PHONE_IN / PHONE | 56.2% / 90.0% | 90.9% / 100.0% |
| CREDIT_CARD / CREDIT_DEBIT_CARD_NUMBER | 61.1% / 100.0% | 100.0% / 100.0% |

Rows without PII that Bedrock flagged: 21.9% (13.5–33.4, 14/64).
No managed equivalent (not scored for Bedrock): AADHAAR, PAN, GSTIN, UPI_ID, IFSC.

Cost of both runs: USD 0.0891; ApplyGuardrail latency p50 155.2 ms, p95 266.4 ms (live, one run).
