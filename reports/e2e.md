# End to end: the whole pipeline on every test-split request

432 requests (test splits of injection, PII and grounding) through the full pipeline with the default policy. The answering model is a deterministic stand-in; classifier and embedding calls are real recordings.

Decision matched the label on 96.1% (93.8–97.5, 415/432) of requests.

| What | Result |
|---|---|
| Attacks blocked | 97.0% (91.5–99.0, 97/100) |
| Normal and tricky messages blocked | 5.9% (2.7–12.2, 6/102) |
| Compromised model: attacks that reached it, caught at the output | 100.0% (43.9–100.0, 3/3) |
| Raw PII that reached the model | 0.0% (0.0–5.8, 0/63) |
| Raw PII in an answer | 0.0% (0.0–5.8, 0/63) |
| Card numbers blocked (policy: block) | 100.0% (74.1–100.0, 11/11) |
| Other PII requests blocked | 2.2% (0.8–6.3, 3/136) |
| Grounding: bad answers rejected | 93.7% (84.8–97.5, 59/63) |
| Grounding: good answers passed | 95.0% (76.4–99.1, 19/20) |

Compromised-model catches by guard: {'grounding': 3}. Normal messages blocked by: {'injection': 6}.

## Document scanning: LLM classifier off vs on

40 clean source documents and 12 indirect-injection attacks used as documents.

| Setting | Poisoned documents neutralised | Clean documents changed |
|---|---|---|
| Classifier off (default) | 16.7% (4.7–44.8, 2/12) | 0.0% (0.0–8.8, 0/40) |
| Classifier on | 83.3% (55.2–95.3, 10/12) | 0.0% (0.0–8.8, 0/40) |

## Cost per guard (USD per 1,000 requests)

| Guard | USD | Runs |
|---|---|---|
| context | 0.0 | 315 |
| grounding | 0.0 | 315 |
| injection | 0.183321 | 420 |
| leak | 0.0 | 255 |
| model | 0.0 | 315 |
| normalize | 0.0 | 432 |
| pii | 0.0 | 432 |
| pii_docs | 0.0 | 420 |
| policy | 0.0 | 432 |

## Latency

Live sample: 30 requests with real Nova answers (live Bedrock from ap-south-1 on a home connection; one sample, not a benchmark).

| Guard | p50 ms | p95 ms |
|---|---|---|
| **end to end** | 747.71 | 2742.72 |
| context | 1.59 | 3.16 |
| grounding | 0.55 | 0.82 |
| injection | 701.34 | 1133.47 |
| model | 921.62 | 1564.13 |
| normalize | 0.29 | 0.48 |
| pii | 0.27 | 0.43 |
| pii_docs | 0.46 | 0.67 |
| policy | 0.19 | 0.32 |

Local replay timings (model calls replayed) are in e2e.json; they measure this machine, not Bedrock.
