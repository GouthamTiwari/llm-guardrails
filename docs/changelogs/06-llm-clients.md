# Task 06: Pricing, spend cap, LLM and embedding clients, record/replay
Branch: task/06-llm-clients | Date: 2026-11-19

## What changed
- `src/llm_guardrails/pricing.py`: price table (Nova 2 Lite global profile, Titan v2, fake test doubles priced like the real models), `PRICES_CHECKED_ON`, `cost_usd` (unknown model → `UnknownModelPrice`), a deliberately high `estimate_tokens` for pre-call budget checks, and Bedrock Guardrails per-text-unit prices for the optional baseline.
- `src/llm_guardrails/spend.py`: `SpendLedger` appends `{ts, purpose, model, tokens_in, tokens_out, usd}` to `.spend/ledger.jsonl` (gitignored, never text) and raises `BudgetExceeded` before any call that would cross `LIVE_BUDGET_USD`. Amounts are kept to 12 decimals because one embedding costs about 1e-7 USD (8 decimals distorted it by 4%; caught by a test).
- `src/llm_guardrails/llm.py`: `LLMClient` protocol, `Message`, `LLMResult` (text hidden from repr), `make_runtime_client` (bearer key or IAM keys, adaptive retries, 5 s connect / 30 s read timeouts, refuses to run without credentials), `BedrockLLM` (Converse in a thread, budget check first, real token counts and cost recorded), `FakeLLM` (scripted answers, exceptions, responder function, delay, call log).
- `src/llm_guardrails/recording.py`: `RecordingLLM` record/replay keyed by sha256 of model, purpose and canonical request; replay of a missing key raises `ReplayMissing`.
- `src/llm_guardrails/embeddings.py`: `Embedder` protocol, `TitanEmbedder` (InvokeModel per text, unit-length float32, disk memo in `.embed_cache/`, `Pacer` for the 60/min quota, budget check), `FakeEmbedder` (hash-seeded or hand-made vectors, optional failure), `VectorBundle` (`.npy` + `.json` index, ignored if model or size differ) and `RecordedEmbedder` (replay from a bundle, fill gaps in record mode).

## How to verify
- `uv run pytest -q` → 180 passed, 2 deselected
- `uv run pytest -m live -q` → 2 passed (real Bedrock; cost approved first)

## Tests
- `tests/test_pricing.py` (5), `tests/test_spend.py` (4, including 200 concurrent records), `tests/test_llm.py` (10, boto3 `Stubber` checks the exact Converse request shape, cost, ledger, empty system prompt, `max_tokens` stop reason, budget refusal before any AWS call, auth modes, repr hiding, fake model, record → replay, missing replay, key sensitivity), `tests/test_embeddings.py` (7, exact InvokeModel body, normalisation, memo prevents a second call, budget refusal, pacer spacing, fake determinism, bundle round trip and model/size isolation).
- `tests/test_live.py` (2, `-m live`): ran once on 2026-11-19. Nova 2 Lite: 55 input + 2 output tokens = $0.00002515; Titan: 5 tokens = $0.00000012. **Project live spend so far: $0.00002527.**
- Full suite: 180 passed.

## Out of scope / follow-ups
- Choosing the client from `GUARDRAILS_LLM_MODE` (bedrock / fake / replay) happens where the pipeline and CLI are built (Tasks 16-17).

## Self-review
- Acceptance criteria of Task 06 checked: all met. The bearer token is only placed in this process's environment for boto3; it is never printed, logged or written to a file.
