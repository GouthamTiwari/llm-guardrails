# LLM Guardrails & Output Validation

Layered guardrails for LLM apps: prompt-injection and Indian-ID PII screening, JSON-schema validation with retries, and grounding checks that reject uncited answers. Python, Pydantic, Amazon Bedrock.

**The model proposes, deterministic checks decide.** Every check sits at one of the two trust boundaries of an LLM app (what goes in, what comes out), returns an explicit verdict (`allow`, `redact`, `retry` or `block`), fails closed, and is measured: attack block rate is always reported next to the false-positive rate on normal users.

> Status: in progress. The specification is in [`docs/SPEC.md`](docs/SPEC.md), the threat model in [`docs/THREAT_MODEL.md`](docs/THREAT_MODEL.md) and the build plan in [`docs/TASKS.md`](docs/TASKS.md).

## Results

Every number in this section is generated from the committed files in `reports/` by `guardrails report readme`.

<!-- RESULTS:START -->
| Check | Result (test split, 95% interval, counts) | Report |
|---|---|---|
| Prompt injection: attacks blocked | 95.0% (88.8–97.9, 95/100) | [injection](reports/injection.md) |
| Prompt injection: normal messages blocked | 5.9% (2.7–12.2, 6/102) | [injection](reports/injection.md) |
| Prompt injection: tricky-but-harmless messages blocked | 11.3% (5.3–22.6, 6/53) | [injection](reports/injection.md) |
| Prompt injection, out of domain (deepset test split): attacks blocked / normal blocked | 28.3% (18.5–40.8, 17/60) / 0.0% (0.0–6.4, 0/56) | [injection](reports/injection.md) |
| PII detection (8 Indian ID types): precision / recall / F1 | 98.8% / 97.6% / 98.2% (TP 82, FP 1, FN 2) | [pii](reports/pii.md) |
| PII look-alikes flagged: regex only → with checksums | 32/64 → 0/64 | [pii](reports/pii.md) |
| Structured output valid on first try: plain prompt → schema in prompt | 0.0% (0.0–3.5, 0/105) → 95.2% (89.3–98.0, 100/105) | [schema](reports/schema.md) |
| Structured output valid after error-fed retries | 100.0% (96.5–100.0, 105/105) | [schema](reports/schema.md) |
| Grounding: bad answers rejected (deterministic → + LLM judge) | 93.7% (84.8–97.5, 59/63) → 100.0% (94.2–100.0, 63/63) | [grounding](reports/grounding.md) |
| Grounding: good answers passed (deterministic → + LLM judge) | 95.0% (76.4–99.1, 19/20) → 95.0% (76.4–99.1, 19/20) | [grounding](reports/grounding.md) |
| End to end (432 requests): decision matched the label | 96.1% (93.8–97.5, 415/432) | [e2e](reports/e2e.md) |
| End to end, compromised model: leaks caught at the output | 100.0% (43.9–100.0, 3/3) | [e2e](reports/e2e.md) |
| End to end: raw PII that reached the model | 0.0% (0.0–5.8, 0/63) | [e2e](reports/e2e.md) |
| Poisoned retrieved documents neutralised: document classifier off (default) → on | 16.7% (4.7–44.8, 2/12) → 83.3% (55.2–95.3, 10/12) | [e2e](reports/e2e.md) |
| Latency, live sample of 30 requests: p50 / p95 end to end | 748 ms / 2743 ms | [e2e](reports/e2e.md) |

All data is synthetic or licence-clean; labels are assistant-reviewed, human spot-check pending. Thresholds were tuned on the dev split only and the test split was evaluated once. Model calls recorded in these reports cost USD 0.3335; `guardrails eval --suite all` replays them offline for free.
<!-- RESULTS:END -->

---

Built by Goutham Tiwari.
