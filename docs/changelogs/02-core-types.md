# Task 02: Core types, guard runner, settings, policy
Branch: task/02-core-types | Date: 2026-11-19

## What changed
- `src/llm_guardrails/types.py`: `Action` + `ACTION_RANK` + `strongest_action`, `Verdict` (modified text and detail excluded from every dump and repr), `EntitySummary`, `PipelineReport`, `SourceDoc` (id shape enforced), `ScanText`, `GuardContext`, the `Reason` code catalogue with a shape check (codes become metric labels, so free text is rejected), and all error classes. `ScanText` and `SourceDoc` live here rather than in their guards to avoid import cycles.
- `src/llm_guardrails/guards/base.py`: `Guard` protocol and `run_guard`: timeout or any exception becomes BLOCK (fail-closed) or ALLOW (fail-open) with `error_kind`; only the exception type name is logged; a non-Verdict result is treated as an error; cancellation propagates.
- `src/llm_guardrails/settings.py`: env + `.env` loader (real env wins), secrets hidden from repr, both-auth-options and half-IAM-keys rejected, mode and budget validated.
- `src/llm_guardrails/policy_config.py`: Pydantic policy schema with `extra="forbid"` everywhere, regex/timeout/guard-name/entity-type validation, restore rules (only redacted types can be restored), default timeouts from spec 14, fail-closed defaults with a WARNING when a policy opens a security guard, loader by bundled name or by path.
- `src/llm_guardrails/policies/default.yaml`, `banking_tax.yaml`: spec 9.1 policies. **Location note:** placed inside the package (not a top-level `policies/`) so they ship with the installed package and load by name; `grounding.entailment: "off"` is quoted because YAML reads bare `off` as `false`.
- `pyproject.toml`: set `asyncio_default_fixture_loop_scope` to silence a pytest-asyncio deprecation warning (outside the task's file list; one line).

## How to verify
- `uv run pytest -q` → 33 passed
- `uv run python -c "from llm_guardrails.policy_config import load_policy; print(load_policy('banking_tax').pii.restore)"` → `['EMAIL', 'PHONE_IN']`

## Tests
- `tests/test_types.py` (7), `tests/test_guard_runner.py` (8, including a Hypothesis property test over 12 exception types and timeouts: a security guard never returns ALLOW), `tests/test_policy_config.py` (11), `tests/test_settings.py` (6, extra file not in the task list). Result: 33 passed (with the smoke test).

## Out of scope / follow-ups
- Metric increments inside `run_guard` (spec 4.2 step 4) arrive with `metrics.py` in Task 16; `run_guard` already fills `latency_ms` and `error_kind` for them.
- `known_entity_types()` returns the 8 built-in types until Task 04's detector registry replaces it.

## Self-review
- Acceptance criteria of Task 02 checked one by one: all met.
- Pydantic `ValidationError` text includes the offending input value; that is why `run_guard` logs only exception type names. Noted in a test comment.
- No secrets, no network, no PII beyond fake test strings.
