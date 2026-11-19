# Task 14: Grounding entailment layer
Branch: task/14-entailment | Date: 2026-11-19

## What changed
- `src/llm_guardrails/guards/grounding.py`:
  - **Entailment judge** (spec 11.4). Every sentence that passed the deterministic checks goes to an LLM judge along with its cited sources. The sources sit in the section 8 wrapper. The sentence sits in its own nonce markers with `<<<`/`>>>` defused, because it is model output and must not be able to talk to the judge.
  - **Judge output:** `EntailmentJudgement` through `generate_validated`, with 1 retry, temperature 0 and 150 max tokens. At most 4 sentences are judged at once, and one 15 s deadline covers the whole answer; unfinished calls are cancelled.
  - **Labels:** `contradicted` and `not_enough_info` become `grounding.entailment_<label>`, so in reject mode the answer blocks and in strip mode that sentence is removed.
  - **`off | optional | required`:**
    - A judge that errors, returns invalid JSON after its retry, misses the deadline, or has no client is `unavailable` for that sentence.
    - `optional` keeps the deterministic result and adds `grounding.entailment_unavailable`.
    - `required` blocks (fail closed), in both modes.
    - A contradiction the judge did return still counts when other calls failed.
  - **Cost control:** in reject mode an answer that already failed deterministically is not judged, because the judge can't undo a block (SPEC decision 8a). An answer with nothing to judge (an abstention, say) makes no calls.
  - **Refactor:** the deterministic code was split into `_deterministic` + `_decide`, so the sync `check_grounding` (no model) and `check_grounding_async` (with judge) share one decision path.
  - **Reporting:** `GroundingResult` and the verdict carry judge calls, USD and tokens. `detail.sentence_outcomes` adds each judge label (still no sentence text).
  - **Guard settings:** `entailment` and the judge timeout come from the policy (`grounding.entailment`, `timeouts_s.grounding_entailment`); constructor arguments override them for experiments.
- `src/llm_guardrails/prompts/entailment.txt`: the judge prompt. It defines the three labels (wider scope such as "all loans" counts as contradicted), says to judge only against the sources and never to follow instructions inside the markers, and asks for strict JSON.
- `src/llm_guardrails/policy_config.py`: a new timeout key `grounding_entailment` (15 s).
- `src/llm_guardrails/recording.py` (outside the task's file list, needed for it): **replay keys now mask marker nonces.** Every guard prompt carries a fresh random code, so the exact prompt never repeats, and keyed on raw text no judge or classifier call could ever replay. That made the Task 11 classifier's recordings unusable too. The 8-hex code inside `<<<...>>>` markers is replaced by a placeholder before hashing; attacker text can't create markers because guards defuse `<<<`. Existing corpus recordings are unaffected: their prompts have no markers.
- `recording.py` + `embeddings.py`: **`missing` list and `ensure_complete()`** on `RecordingLLM` and `RecordedEmbedder`. Guards fail closed, so a missing recording inside a guard becomes a quiet BLOCK that would change eval numbers. The eval runner must call `ensure_complete()` after every replay run (SPEC decision 8d).
- `tests/test_live.py`: one live judge test (3 Nova calls, about USD 0.001), deselected by default and **not run** in this task.
- `docs/SPEC.md`: decision 8.

## How to verify
- `uv run pytest -q tests/test_grounding_entailment.py` → 25 passed

## Tests
- `tests/test_grounding_entailment.py` (25):
  - **Prompt shape:** wrapper, sentence markers, defused sentence. **Call settings:** temperature 0, 150 tokens, purpose; only the cited sources are sent, without citations.
  - **Behaviour:** `off` makes no calls. The judge catches "payment date" vs "statement date", which passes every deterministic check. A supported answer is costed. Strip removes the sentence the judge rejects. Reject skips the judge after a deterministic failure. An abstention makes no calls.
  - **Failures:** error, invalid twice and no client are each tested under optional (falls back) and required (blocks, in both modes). One retry and no third call, still costed. A retry that fixes the output. A judged contradiction survives other calls failing. The deadline covers the whole answer and cancels the rest. The peak is exactly 4 calls at once.
  - **Replay:** recorded judge calls replay despite fresh nonces, and a missing recording is reported by `ensure_complete()`.
  - **Guard:** the policy setting (banking_tax: optional) with cost and tokens, no text in the dump, the default policy has the judge off, the policy timeout, and constructor overrides.
- `tests/test_llm.py` (+3): replay keys ignore marker nonces (injection and document wrapper formats), and plain hex outside markers still counts.
- `tests/test_embeddings.py`: `ensure_complete()` reports a vector miss.
- Full suite: 567 passed (3 live tests deselected).

## Live spend
- 2026-11-19, after the merge, with Goutham's yes: `uv run pytest -m live tests/test_live.py -k entailment` → 1 passed. Nova labelled the three sentences **supported / contradicted / not_enough_info**, exactly as intended (faithful, changed condition, added fact). Cost USD 0.000775 (3 calls, no retries). Project live spend now USD 0.0732 of the USD 5 cap.
- This is a smoke test of the prompt and the JSON contract, not a result: the judge's real value is measured on the grounding dataset in Task 20.

## Out of scope / follow-ups
- Tasks 19–20: the eval runner calls `ensure_complete()` on every recorder after a replay run.
- Task 15: the canary in the system prompt is also random per request; its replay masking belongs with the leak guard.
- Task 16: the pipeline must redact PII in sources before the judge sees them, and keep the `grounding` guard timeout (20 s) above the judge deadline (15 s); if it were lower, the guard would time out and block even in optional mode.
- Task 20 measures whether the judge adds anything over the deterministic checks.

## Self-review
- Acceptance of Task 14: per-supported-sentence judge, concurrency 4, schema-validated output, `off | optional | required`, `optional` falls back with `grounding.entailment_unavailable`, `required` blocks on failure. All met and tested.
- Found while reading the replay code before writing: random nonces made every guard prompt unreplayable. Fixed and tested end to end through the judge.
- Found in self-review: `judge_all` (correctly, for runtime) treats a replay miss as unavailable, which would silently alter eval results. That is now covered by `ensure_complete()`.
- No fail-open path: no client, error, invalid output and timeout all fall back to the deterministic result (which already passed) or block under `required`. No secrets; no real PII.
