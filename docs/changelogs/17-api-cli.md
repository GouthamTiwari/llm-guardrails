# Task 17: HTTP API and CLI
Branch: task/17-api-cli | Date: 2026-11-19

## What changed
- `src/llm_guardrails/api.py`: FastAPI app (spec 15.1).
  - **Endpoints:** `POST /v1/guarded-chat`, `/v1/check-input`, `/v1/check-output`, `GET /metrics` (Prometheus text) and `GET /healthz`.
  - **Status codes:**
    - A block is a normal 200 with `decision: "block"`.
    - 422 returns `{error: "invalid_request", fields: [...]}` and never echoes the input, because FastAPI's default 422 body does and the input may hold PII.
    - 500 returns `{request_id, error: "internal_error"}` and logs only the exception type.
  - **Input limits:** only bundled policy names are accepted (no file paths), clients may request only `TicketSummary`, canaries must have the canary shape, and unknown fields are rejected.
- `src/llm_guardrails/runtime.py` (new, shared by the API and CLI): builds the model and embedder for `GUARDRAILS_LLM_MODE`.
  - `fake` uses a deterministic stand-in for every model role, with no AWS. It answers with the first sentence of the first document plus its citation.
  - `replay` answers from recordings and fails closed on a miss.
  - `bedrock` refuses to start without credentials.
  - `Pipelines` caches one pipeline per (policy, schema) and applies the log-hash salt from settings.
- `src/llm_guardrails/cli.py`: Typer CLI with every command from spec 15.2.
  - **Working now:** `check` (input or `--output`, `--sources`, `--policy`, `--json`; exit 0 allowed / 1 blocked), `serve` (binds 127.0.0.1), `spend`, `version`, `corpus generate|embed`.
  - **Stubs:** `dataset …`, `sweep …`, `eval`, `baseline bedrock` and `report readme` exit 2 with "implemented in Task NN".
  - **Safety rules enforced already:** `sweep` refuses `--split test`, and `eval --split test` for injection needs `reports/operating_point.json`.
- **Bugs found and fixed** (outside the file list, recorded as SPEC decision 11):
  - An empty `RecordingLLM` was falsy, because it defines `__len__`. `ctx.llm or self.llm` then dropped it, and the injection guard allowed a grey-zone input instead of failing closed. Recorders are now always truthy, and `guards/injection.py`, `context.py` and `grounding.py` pick clients with `is not None` checks.
  - `LOG_HASH_SALT` from `.env` was never applied (only the process environment was read), so hash prefixes changed every run. `Pipelines` now sets it.
- `docs/SPEC.md`: decision 11.

## How to verify
- `uv run pytest -q tests/test_api.py tests/test_cli.py` → 47 passed
- In fake mode, `guardrails check "My PAN is ABCPE1234F"` prints `decision: allow` and `text: My PAN is [PAN_1]`, with `pii.redact.PAN`.
- Also run by hand: `guardrails serve` started as a real process on a free port, with the health check, a grounded answer, an attack blocked and `/metrics` (194 series lines) all working over HTTP.

## Tests
- `tests/test_api.py` (20):
  - **Endpoints:** health, metrics, an allowed chat with a cited answer, a block as 200, chat with a schema, no redacted PII in the report.
  - **Stages:** check-input (allow, and a block that returns no text), check-output (allow, canary leak, schema RETRY as a block).
  - **Errors:** 7 malformed requests each return 422 naming only the field; a 422 doesn't echo the input; a 500 carries no details.
  - **Modes:** bedrock without credentials refuses to start; replay without recordings fails closed.
- `tests/test_cli.py` (27):
  - **check:** the redacted result, a block with exit 1, `--json`, `--output` with a sources file, the banking policy, bad sources files, an unknown policy, and the replay-mode note.
  - **Guard rules:** sweep refuses the test split; the test eval needs the operating point.
  - **Stubs and others:** all 12 later-task stubs say which task, `spend` reads the ledger, and help lists every command.
- `tests/test_llm.py` (+1): an empty recorder is still truthy.
- Full suite: 765 passed (3 live tests deselected). The one warning is a deprecation notice inside Starlette's test client, not our code.

## Live spend
None.

## Out of scope / follow-ups
- Tasks 18–21 fill in the dataset, sweep, eval, baseline and report commands.
- Auth, rate limiting and multi-tenancy stay non-goals (spec 15.1); the README says to put the service behind a gateway.

## Self-review
- Acceptance of Task 17: endpoints and status codes per 15.1, tested with TestClient in fake mode; every 15.2 command exists, stubs say which task, and `guardrails check "My PAN is ABCPE1234F"` prints a redacted result. All met.
- Found while testing: the empty-recorder fail-open bug and the ignored hash salt, both fixed and tested. Also checked that the API never echoes request values in error bodies and never accepts a policy file path from a request.
- No secrets, no real PII, no new fail-open path.
