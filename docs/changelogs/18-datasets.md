# Task 18: Datasets
Branch: task/18-datasets | Date: 2026-11-19

## What changed
- `src/llm_guardrails/eval/synth.py`: seeded synthetic identifiers.
  - **Valid values** pass the real checks: Verhoeff for Aadhaar (some in Devanagari digits), Luhn plus a real card-network prefix, the GSTIN check character, a valid PAN holder letter, `example.com/org/net` emails, fictional UPI handles and IFSC bank codes.
  - **Look-alikes** fail the check they imitate: mistyped Aadhaar, order ids, tracking codes, declined-payment references, PAN-shaped promo codes, bad GSTINs, toll-free numbers, invoice and reference numbers.
  - **Slot filling** turns model-written templates into text plus gold entities.
- `src/llm_guardrails/eval/generate.py`: Nova prompts per suite at temperature 0.9, recorded to `data/recordings/generate_<suite>.jsonl` and replayed by default.
  - **Injection prompts** show only the seed rows of a category, never the attack corpus (decision 5).
  - **PII:** the model writes slot templates and never a value.
  - **Grounding** rows must pass a deterministic label check.
  - **Clean-up:** items are validated one by one, exact and near-duplicates are dropped, and real email providers are rewritten to `attacker.example`.
- `src/llm_guardrails/eval/datasets.py`: row schemas for the four suites and validation.
  - **Validation:** ids, labels, every entity value present and passing its check, every look-alike failing its check, example domains only, fictional UPI handles, grounding label sanity.
  - **Split:** 50/50 dev/test per stratum with seed 7. Seeds stay in dev, near-duplicate groups stay together, and injection rows close to the corpus stay in dev.
- `src/llm_guardrails/eval/leakage.py`: Jaccard ≥ 0.6 against dev and the corpus; cosine ≥ 0.90 against the corpus and ≥ 0.95 against dev, using Titan vectors stored in `data/injection_vectors.*`. A stale vector file now fails the check.
- `src/llm_guardrails/eval/review.py` and `docs/DATASET_REVIEW.md`: the seeded 10% review sample (`data/review/review_sample.csv`, with a BOM so Excel shows Hindi) and `apply-review` (`ok` / `drop` / `edit:<label>`). Dropped rows stay in the file, marked `human-dropped`.
- `src/llm_guardrails/cli.py`: `dataset generate | embed | split | validate | leakcheck | review-sample | apply-review` are now real.
- **Data:**

  | File | Rows | Dev | Test | Notes |
  |---|---|---|---|---|
  | `data/injection.jsonl` | 407 | 205 | 202 | 202 attacks, 205 benign (106 hard) |
  | `data/pii.jsonl` | 308 | 161 | 147 | 41.6% with no entity; 22–26 per type |
  | `data/grounding.jsonl` | 172 | 89 | 83 | 21–24 per label |
  | `data/schema_inputs.jsonl` | 105 | 54 | 51 | normal, multiple-issue, number-heavy, Hindi/Hinglish, hostile, empty, very long (about 5,000 characters), off-topic |

  Plus the recordings, `data/injection_vectors.*` and the review sample.
- `tests/test_corpus.py`: aligned with spec 13.1 (corpus vs seeds and test rows; any corpus-like row must be in dev), decision 12f.
- `docs/SPEC.md`: decision 12.

## How to verify
- `uv run guardrails dataset validate --suite all` → all OK
- `uv run guardrails dataset leakcheck --suite all` → all OK
- `uv run guardrails dataset generate --suite pii` → rebuilds the same file from recordings, free and offline.
- `uv run pytest -q tests/test_datasets.py` → 62 passed

## Tests
- `tests/test_datasets.py` (62):
  - **synth:** 8 types pass, 9 look-alike kinds fail, fill, fictional domains, Devanagari digits.
  - **Generation with a fake model**, for all four suites: generate → validate → split → validate → leakcheck. Also the PII mix, bad items counted, a replay miss raises, injection prompts contain seeds and never corpus texts, template checks, lenient JSON parsing.
  - **Validation:** each kind of problem is caught.
  - **Grounding:** label sanity (8 cases).
  - **Split:** duplicate groups and seeds stay together, forced-dev rows, leakage findings, missing vectors fail.
  - **Review:** sample and apply, covering ok, edit, drop, an invalid edit and the doc.
  - **Committed data:** validates, has no leakage, has full vector coverage, and meets the sizes.
- `tests/test_cli.py`: dataset command argument checks replace the Task 18 stubs.
- Full suite: 822 passed (3 live tests deselected).

## Live spend
Approved by Goutham before running; estimate given as about USD 0.20, at most USD 0.48.
- **Nova:** 90 calls, USD 0.1745.
  - Injection: 33 calls (26 in round 1, 7 top-up).
  - PII: 20 calls (19, plus one re-ask after malformed JSON).
  - Schema: 7 calls.
  - Grounding: 30 calls. The first run was replaced for label quality, and one identical rerun (USD 0.028) was **my mistake**: a failed prompt edit didn't stop the next command. Live steps now run from `set -e` scripts.
- **Titan:** 379 calls, USD 0.0003.
- **Task total USD 0.1748.** Project live spend is USD 0.2481 of the USD 5 cap.

## Findings
- Bedrock's content filter blocked 3 attack-writing batches, and 6 of 33 injection batches came back empty in total, even for a detector's test set. The red-team framing in round 2 recovered most categories (decision 12a).
- The first grounding run had wrong labels (an "unanswerable" question that S2 answered, contradictions labelled partial) and dollar amounts. The deterministic sanity check can't see those, so it was regenerated with a stricter prompt (decision 12b). The human spot-check exists for exactly this kind of error.
- Only 3 of 407 injection rows had to stay in dev for resembling the corpus, which is evidence that the generated test split really is new to layer 2.

## Out of scope / follow-ups
- **Your spot-check:** fill `verdict` in `data/review/review_sample.csv` (99 rows), then run `guardrails dataset apply-review`. Until then the README says "assistant-reviewed, human spot-check pending".
- Task 19 uses `data/injection_vectors.*` for the layer-2 eval and must call `ensure_complete()` after replay runs (decision 8d).
- The external deepset test split is downloaded at eval time (Task 19), not committed.

## Self-review
- Acceptance of Task 18: generation per 13.1 (slot filling for PII, label checks for grounding), sizes reached, recordings committed; `validate` and `leakcheck` pass and run as pytest tests; a 10% review sample written; `apply-review` works on a test CSV; `DATASET_REVIEW.md` records "pending"; live spend recorded. All met.
- Found while checking the data:
  - A real email provider in one row; fixed in the generator and scrubbed from its recording.
  - The validator rejected legitimate example subdomains; fixed.
  - The leakage check silently skipped the meaning-level comparison when one vector was missing; now a failure.
- No secret-shaped strings in any data file (scanned), no real person's data, no employer details.
