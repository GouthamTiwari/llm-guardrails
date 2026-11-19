# Task 16: Pipeline, logging, metrics
Branch: task/16-pipeline | Date: 2026-11-19

## What changed
- `src/llm_guardrails/pipeline.py`: `Pipeline` with `run`, `check_input` and `check_output` (spec 14).
  - **Input:** normalize → policy → pii → `pii_docs` (redact documents with the question's numbering, then rebuild the scan copy from the redacted text) → injection → context. The first BLOCK stops the request; a request blocked on input never calls the model.
  - **Model:** a fresh canary per request in the system prompt; citation instructions when grounding is on; the user turn is the wrapped documents plus the redacted question. It makes one Converse call at temperature 0, or runs the schema retry loop when a schema is set (the schema's 45 s budget covers every retry).
  - **Output:** schema (JSON) or grounding (prose) → leak → `restore` per policy. A schema output is re-validated if redaction or restore changed it. RETRY counts as a block.
  - **Fail closed everywhere:** every step, including the internal ones (`pii_docs`, `model`), runs through `run_guard` with its policy timeout. A whole-request deadline (90 s) adds a `request` timeout verdict and keeps the report of what had run.
  - **Results:** `GuardedAnswer`, `InputResult` and `OutputResult` hold the text only when allowed. A blocked request gets a fixed user message (the PII guard's card/Aadhaar/PAN message, or "Sorry, I can't help with that request.").
  - **Logging and metrics:** one audit line and one metrics update per request (routes `guarded-chat`, `check-input`, `check-output`). Both fail open: a broken log sink or metrics backend never blocks a user.
- `src/llm_guardrails/metrics.py`: the 10 collectors of spec 14.2 on the default Prometheus registry. They are filled from the finished report, so guards keep no global state and metrics always agree with the audit log. Label values are fixed vocabularies, never user text.
- `src/llm_guardrails/prompts/system_default.txt`: the default system prompt (documents only, keep placeholders, never ask for card/Aadhaar/PAN, documents are data).
- `src/llm_guardrails/guards/grounding.py` + `types.py` (bug found while composing): a PII placeholder such as `[EMAIL_1]` has the same bracket shape as a citation, so grounding read it as a citation to a missing source and rejected every answer that kept a placeholder, which the system prompt asks for. Placeholders are now content, not citations, and `SourceDoc` ids may no longer look like placeholders.
- `docs/SPEC.md`: decision 10 (document PII, rescan, schema mode, abstention vs prompt leak, placeholders, internal steps, metrics).

## How to verify
- `uv run pytest -q tests/test_pipeline.py tests/test_fail_closed.py tests/test_no_pii_in_logs.py` → 68 passed

## Tests
- `tests/test_pipeline.py` (29):
  - **Allow path:** guard order, canary and citation instructions in the prompt, wrapped documents, a fresh canary per request, and a correct abstention not mistaken for a prompt leak.
  - **Input blocks:** injection, a card number (fixed message) and a size limit, each with no model call.
  - **PII:** the model never sees raw PII (answer model and injection classifier); restore on and off; document PII shares numbering and fake placeholders are escaped; a kept placeholder isn't a fabricated citation.
  - **Context:** a poisoned document line is stripped before the model sees it.
  - **Output:** uncited answers rejected, strip mode, a canary echo (caught by grounding first, and by the leak guard on its own, counted in metrics), an exfiltration image.
  - **Schema:** a retry then data, and failure after retries.
  - **Failures:** model error, model timeout, whole-request deadline, broken logging and metrics.
  - **Stages and observability:** `check_input` / `check_output` (including RETRY as a block), exactly one audit line with exactly the spec fields, 9 metrics checked against known requests, and per-route logging.
- `tests/test_fail_closed.py` (37):
  - Each of the 8 security guards is made to **raise** and then to **hang past its timeout**: the request blocks, the error shows in the report, and nothing runs after it.
  - The same through `check_input` and `check_output`.
  - Broken document redaction and a broken model call.
  - `fail_mode: open` stays visible in the report.
- `tests/test_no_pii_in_logs.py` (2): every PII and injection seed row through `run`, `check_input` and `check_output` under both policies, with a model that echoes the user's text and every logger captured at DEBUG. None of 72 forbidden strings (gold values in several spellings, and input texts) appear, and there is exactly one audit line per request. A temporary mutation check confirmed that the test catches a planted leak.
- `tests/test_grounding.py` (+2): placeholders aren't citations; source ids can't look like placeholders.
- Full suite: 717 passed (3 live tests deselected).

## Live spend
None.

## Out of scope / follow-ups
- Task 17 serves this through FastAPI (`/metrics` exposes these collectors) and the CLI.
- Task 19/20 runners call `ensure_complete()` on their recorders (SPEC decision 8d).
- Known strictness: a sentence the model adds without a citation ("We will write to [EMAIL_1].") is rejected as uncited under `reject` mode; `strip` mode removes it instead. That is the resume line's "reject uncited answers", and Task 20 measures its cost on real answers.

## Self-review
- Acceptance of Task 16:
  - `check_input`, `check_output` and `run`, and a blocked request never calls the model (asserted with call counts).
  - Per-guard and per-request timeouts; `test_fail_closed.py` injects an exception and a timeout into every security guard.
  - One audit line with exactly the spec 14.1 fields, and no seed PII or text in any log.
  - Metrics read back from `prometheus_client.REGISTRY`.
  - All met.
- Found while composing:
  - Placeholders read as citations (fixed above).
  - The correct abstention sentence would have counted as a system-prompt leak, because it is in the prompt.
  - The scan copy built before PII redaction would have sent raw PII to the embedding and classifier calls.
- No secrets, no real PII, no fail-open path except the explicit, reported `fail_mode: open` option and the observability exports the spec marks as open.
