# Task 22: Adapters and examples
Branch: task/22-adapters | Date: 2026-11-21

## What changed
- `src/llm_guardrails/adapters/decorator.py`: `@guarded(policy, grounding, schema, pipeline=None)` for sync and async `fn(question, sources) -> str`.
  - **Order:** input stage (a block never calls the function) → the function, with the sanitised question and cleaned, PII-redacted documents → output stage (grounding or schema, leak scan, restore).
  - **Fails closed:** an exception or a non-string return blocks (guard `upstream`), never an unchecked answer.
  - **Result:** a `GuardedCall` (a `GuardedAnswer` plus both stage results) with one merged report.
  - **Sync use:** the sync wrapper refuses to run inside an event loop, with a clear message.
- `src/llm_guardrails/adapters/cache.py`: `should_cache(input_result, output_result)`. It returns False for a block anywhere, any redact, retry or block verdict, any detected PII entity (even one the policy allows), or no passing grounding verdict.
- `src/llm_guardrails/adapters/protocols.py`: `RagAnswerFn` and `CacheLike`, duck-typed and runtime-checkable.
- `src/llm_guardrails/adapters/__init__.py`: exports these plus `check_tool_call`.
- `examples/regulatory_rag_adapter.py`: RAG chunks `{chunk_id, text}` → `S1, S2, ...` with a map back to the original ids. It shows a guarded answer passing, an uncited answer blocked, a PAN redacted before the RAG function sees it, and the grounding guard called directly.
- `examples/semantic_cache_adapter.py`: a dict-backed `CacheLike`. Only the clean exchange is stored; the PII one and the blocked one are not.
- `examples/agent_tool_gate.py`: `check_tool_call` blocks `payments.transfer` (needs a human) and `DROP TABLE`, and runs safe calls.
- `docs/SPEC.md`: decision 16.

## How to verify
- `uv run python examples/regulatory_rag_adapter.py`
- `uv run python examples/semantic_cache_adapter.py`
- `uv run python examples/agent_tool_gate.py`
- `uv run pytest -q tests/test_adapters.py` → 22 passed

All three examples run in fake mode by default, with no AWS.

## Tests
- `tests/test_adapters.py` (22):
  - **@guarded:** sync and async end to end; the function only sees redacted PII; blocked input never calls it; an uncited answer is blocked at output; an exception or non-string return fails closed; grounding can be switched off; the sync wrapper inside an event loop raises; the decorator builds its pipeline from the environment.
  - **should_cache:** True for a clean, grounded exchange. False for PII on the input side, PII the policy allows (GSTIN), PII in the answer, redacted documents, blocked input, blocked output, no grounding verdict, and missing results.
  - **Protocols:** duck typing.
  - **Examples:** all 3 run as subprocesses in fake mode with their expected output.
- Full suite: 904 passed (3 live tests deselected).

## Live spend
None.

## Out of scope / follow-ups
- Wiring the decorator into the real Regulatory RAG and Semantic Cache repos is their change, not this repo's; the examples show the exact shape.

## Self-review
- Acceptance of Task 22: `@guarded` for sync and async functions; `should_cache` returns False for PII, redaction, block and failed grounding, each tested; the examples run with `GUARDRAILS_LLM_MODE=fake`. All met.
- No fail-open path (an upstream error blocks). No dependency on the sibling repos. No secrets or real data.
