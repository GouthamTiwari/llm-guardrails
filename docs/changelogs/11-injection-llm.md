# Task 11: Injection layer 3 and the combined decision
Branch: task/11-injection-llm | Date: 2026-11-19

## What changed
- `src/llm_guardrails/prompts/injection_classifier.txt`: the classifier's instructions. The text is data between nonce-tagged markers and must never be followed. Definitions of the 9 attack categories (with Hindi/Hinglish noted), explicit hard-benign examples ("ignore my previous message, I meant GST", questions *about* security, role-plays that keep the rules), and a strict JSON reply format.
- `src/llm_guardrails/guards/injection.py`:
  - `classifier_user`: wraps the text as `<<<TEXT nonce>>> … <<<END TEXT nonce>>>` with a fresh 8-hex nonce per call and `<<<`/`>>>` defused, so the text can't close its own wrapper. Decoded base64/hex goes in a second labelled block. Texts over 4,000 characters keep their head and tail.
  - `score_llm`: calls the schema guard's `generate_validated` with `InjectionJudgement`, temperature 0, at most 1 corrective retry, 200 tokens. Score = confidence when the label is "attack", else 1 − confidence.
  - `InjectionGuard.check`: the full spec 7.4 table. Strong rule → block with no model calls; embedding score ≥ `t_block` → block; max(rules, embedding) < `t_allow` → allow; otherwise the classifier decides against `t_llm`. Per-layer timeouts come from the policy (`injection_embed`, `injection_llm`). The verdict carries the classifier's tokens and cost and a `path` in detail (rules_block, embed_block, below_t_allow, llm_block, llm_allow, grey_zone_no_classifier, …) for the eval.
  - Failure rules (spec 7.5): an embedding error or timeout sends **even clear-looking text** to the classifier, because there is no evidence either way. A classifier error, timeout or invalid verdict (after its retry) → BLOCK. With no classifier (policy `use_llm: false` or none given) the grey zone is allowed. That's the "rules + embeddings" ablation. The exception: if the embedding layer failed, the guard then fails closed.
- `src/llm_guardrails/guards/schema.py` (outside the file list): `generate_validated` takes a `temperatures` sequence, so the classifier runs at a constant 0.0.

## How to verify
- `uv run pytest -q tests/test_injection_guard.py` → 21 passed

## Tests
- `tests/test_injection_guard.py` (21): clear text never calls the classifier (cost 0); an obvious attack is blocked by embeddings with no classifier call; a strong rule calls neither model; grey-zone attack blocked (temperature 0, purpose, cost on the verdict); grey-zone benign allowed, with the score as attack probability; an unsure attack label below `t_llm` allowed; `t_llm` from config; weak rules alone reach the grey zone with embeddings off; classifier switched off allows the grey zone; embedding failure escalates even clear text; embedding failure without a classifier fails closed; three classifier failure modes fail closed (exception, invalid JSON twice, out-of-range values twice); invalid-then-valid verdict is used; classifier timeout fails closed; embedding timeout escalates; wrapper can't be closed by the text, decoded block present, long text truncated; a fresh nonce per call; prompt covers categories and hard negatives; no text in the serialised verdict.
- Full suite: 418 passed (2 live tests deselected).

## Out of scope / follow-ups
- Real classifier quality, its cost per request and the thresholds are measured in Task 19 (dev sweep, then one test run). No live calls in this task.

## Self-review
- Acceptance criteria of Task 11 checked: nonce wrapping, `InjectionJudgement` via the schema guard with 1 retry, the decision table with configurable thresholds, the classifier only in the grey band (call counts asserted), and failure handling per 7.5. All met.
