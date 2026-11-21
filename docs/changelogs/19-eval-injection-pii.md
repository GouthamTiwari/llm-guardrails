# Task 19: Eval, injection (sweep, test, external) and PII
Branch: task/19-eval-injection-pii | Dates: 2026-11-19 to 2026-11-20

## What changed
- `src/llm_guardrails/eval/metrics.py`: attack recall, benign false-positive rate (FPR), and precision/recall/F1, each with counts and 95% Wilson intervals; grouping by category and language; markdown formatting.
- `src/llm_guardrails/eval/run_injection.py`: every row is scored once per layer.
  - **Layers:** rules; the best attack and benign cosine per embedding window (so λ can be swept for free); the classifier, on every dev row but only on grey-zone rows for test and external.
  - **Outputs:** layer-alone and combined results, per category and language, the hard-benign FPR, classifier calls, USD per 1,000 requests, a dev reference including "classifier on every input", the external deepset row, and `what_got_through.md`.
  - **Recording:** every call is recorded (`data/recordings/injection_eval.jsonl`, `injection_external.jsonl`), and the run ends with `ensure_complete()` (decision 8d).
- `src/llm_guardrails/eval/sweep.py`: exhaustive dev-only grid (69,000+ points) with the fixed objective: maximise recall subject to benign FPR ≤ 2% and hard-benign FPR ≤ 10%. It writes `reports/operating_point.json`.
- `src/llm_guardrails/eval/run_pii.py`: per-type precision/recall/F1 (typed, normalised, multiset matching, Hindi digits folded), look-alike false positives, and the three-step ablation. No model calls.
- `src/llm_guardrails/eval/reports.py`: shared report metadata (`generated_at`, `git_sha`, dataset hashes, operating point, mode, spend).
- `src/llm_guardrails/guards/injection.py`: `InjectionConfig` defaults = the dev operating point (T_BLOCK 0.53, T_ALLOW 0.22, T_LLM 0.80, λ 0.5).
- `src/llm_guardrails/rules/injection_rules.yaml`: the role-play rule no longer fires on "password reset/change/policy..." topics unless an attack verb precedes them, with 3 new examples (decision 13b).
- `data/injection.jsonl`: `injg0283` relabelled `direct_override` after review, recorded in `data/review/decisions_2026-11-20.csv` and `docs/DATASET_REVIEW.md`.
- `src/llm_guardrails/cli.py`:
  - `sweep injection [--live]` (refuses `--split test`).
  - `eval --suite injection|pii [--live]`, which defaults to the test split, refuses `--split dev`, and needs `operating_point.json`.
  - `dataset download-external` fetches the deepset test split into git-ignored `data/raw/`.
- **Reports:** `reports/operating_point.json`, `injection.json/.md`, `pii.json/.md`, `what_got_through.md`, all generated in replay mode.
- `docs/SPEC.md`: decision 13.

## Results (from the committed reports)

**Injection, test split, evaluated once:**

| Configuration | Attack recall | Benign FPR |
|---|---|---|
| Rules only | 18.0% (18/100) | 1.0% (1/102) |
| Embeddings only | 18.0% (18/100) | 1.0% (1/102) |
| Rules + embeddings | 31.0% (31/100) | 2.0% (2/102) |
| **All three (shipped)** | **95.0% (88.8–97.9)** | **5.9% (2.7–12.2)** |

- The hard-benign FPR is 11.3% (6/53), above the 10% the sweep met on dev.
- The classifier is called on 62.4% of inputs, and costs about USD 0.22 per 1,000 requests.
- 5 attacks got through: 4 "encoding" rows (whose decoded text is mostly harmless, a likely generator label issue for the spot-check) and 1 indirect-document row.

**Dev reference:** shipped 97.1% recall at 2.0% FPR; classifier alone 96.1% at 2.0%.

**External (deepset test split, out of domain):** shipped 28.3% recall (17/60) at 0% FPR (0/56). Only 27 of 116 inputs reached the classifier, because the rules and the banking-focused corpus put few generic prompts in the grey zone.

**PII, test split, shipped configuration:**
- Micro precision 98.8%, recall 97.6%, F1 98.2%.
- The ablation shows checksums do the work: false positives fall from 33 to 1, and rows without PII that got a detection fall from 32/64 to 0/64. Context words add nothing more on this split.
- The 3 errors are 2 real detector bugs (decision 13i).

## How the operating point was chosen (dev only)

| Dev configuration | Attack recall | Benign FP |
|---|---|---|
| Spec objective, data and rules as generated | 58/102 | 2/103 |
| Allowing 3% FPR instead (not used) | 95/102 | 3/103 |
| `injg0283` relabelled as an attack | 96/103 | 2/102 |
| + role-play rule false positive fixed (**chosen**) | **100/103** | **2/102** |

The last two changes were decided before the test split was evaluated. **One row moved dev recall from 57% to 93%**: with about 100 benign rows, a 2% budget is 2 rows, so small labelling choices swing the headline. The intervals above are what make that visible.

## How to verify
- `uv run guardrails eval --suite injection` and `uv run guardrails eval --suite pii` regenerate the reports offline.
- `uv run pytest -q tests/test_eval_reproducible.py` → 4 passed: byte-for-byte except `generated_at` and `git_sha`, also without the raw deepset file.
- `uv run pytest -q tests/test_eval_injection_pii.py` → 30 passed

## Tests
- `tests/test_eval_injection_pii.py` (30):
  - **Metrics:** Wilson intervals against known values, rates and formatting, recall/FPR grouping, precision/recall/F1.
  - **Decisions:** the λ margin, 10 decision cases across the layers (a failed classifier fails closed), and a missing classifier score is an error.
  - **Sweep:** hand-made rows with a known best answer; refuses when nothing is feasible or a score is missing.
  - **PII:** matching (Hindi digits, multiset, look-alikes) and an offline report.
  - **End to end:** fake-model sweep then eval writing every report, external numbers kept, the external loader.
  - **Defaults:** `InjectionConfig` defaults equal `operating_point.json`.
- `tests/test_eval_reproducible.py` (4): operating point, injection report and markdown, PII report, and the external numbers without raw data.
- `tests/test_injection_rules.py`: +3 role-play examples. `tests/test_cli.py`: sweep/eval/download behaviour.
- Full suite: 854 passed (3 live tests deselected).

## Live spend
Cost approved first (download plus live run, estimate about USD 0.13, at most USD 0.45).
- **Dev sweep:** 205 Nova classifier calls, USD 0.071. That includes USD 0.020 for 56 calls before a ThrottlingException stopped the first attempt; they were recorded and replayed on resume.
- **Test split:** 126 classifier calls, USD 0.044.
- **External:** 116 Titan embeddings and 27 classifier calls, USD 0.009.
- **Task total USD 0.1246.** Project live spend is USD 0.3727 of the USD 5 cap.
- **Download:** deepset/prompt-injections test split, 116 rows, 19 KB, into git-ignored `data/raw/`.

## Out of scope / follow-ups
- Task 20: the context-scan ablation (classifier on/off for documents) in the end-to-end run; `report readme`.
- v1.1: fix the two UPI detector bugs and re-measure on new data. Consider routing more inputs to the classifier, since out-of-domain recall was 28% because few inputs reach it.
- Human spot-check: the 99-row sample, plus a look at the generated "encoding_obfuscation" attacks whose decoded text is harmless.

## Self-review
- Acceptance of Task 19:
  - Metrics with Wilson intervals, unit-tested on toy data.
  - `sweep injection` runs on dev only and refuses test, writing `operating_point.json`; the defaults are tested to equal it.
  - The test split is evaluated once: per layer, combined, per category and language, hard-benign FPR, the external deepset row, and `what_got_through.md`.
  - PII per-type precision/recall/F1 and the three-step ablation.
  - Both reports reproduce offline byte for byte apart from `generated_at`.
  - All met.
- **Integrity:** no test row was read, changed or used for tuning. The only data and rule changes came from dev evidence and were decided before the test run, and the PII bugs found on test are reported, not fixed.
- **Leftovers:** two stray report files that an old CLI test had written into `reports/` during development were deleted before any real report was generated.
- No secrets in recordings or reports (scanned). The raw external data is not committed.
