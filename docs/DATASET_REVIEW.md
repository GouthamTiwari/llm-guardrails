# Dataset review

Status: **pending** (updated 2026-11-19).

All rows were generated or written with an AI assistant and checked by deterministic validators (`guardrails dataset validate`, `guardrails dataset leakcheck`). A random 10% sample (99 rows, seed 7) is in `data/review/review_sample.csv` for a human spot-check.

How to review: open the CSV, fill the `verdict` column with `ok`, `drop` or `edit:<new label>`, save, then run `guardrails dataset apply-review`.

Until the spot-check is done, the README says: "assistant-reviewed, human spot-check pending".
