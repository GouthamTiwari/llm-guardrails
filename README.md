# LLM Guardrails & Output Validation

Layered guardrails for LLM apps: prompt-injection and Indian-ID PII screening, JSON-schema validation with retries, and grounding checks that reject uncited answers. Python, Pydantic, Amazon Bedrock.

**The model proposes, deterministic checks decide.** Every check sits at one of the two trust boundaries of an LLM app (what goes in, what comes out), returns an explicit verdict (`allow`, `redact`, `retry` or `block`), fails closed, and is measured: attack block rate is always reported next to the false-positive rate on normal users.

> Status: in progress. The specification is in [`docs/SPEC.md`](docs/SPEC.md), the threat model in [`docs/THREAT_MODEL.md`](docs/THREAT_MODEL.md) and the build plan in [`docs/TASKS.md`](docs/TASKS.md).

## Results

Every number in this section is generated from the committed files in `reports/` by `guardrails report readme`.

<!-- RESULTS:START -->
_No results yet._
<!-- RESULTS:END -->

---

Built by Goutham Tiwari.
