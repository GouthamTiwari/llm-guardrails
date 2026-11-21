# Dataset review

Status: **pending** (updated 2026-11-20).

All rows were written with Amazon Nova from hand-written seeds (PII values are synthetic and filled in by code) and checked by deterministic validators (`guardrails dataset validate`, `guardrails dataset leakcheck`). A random 10% sample (99 rows, seed 7) is in `data/review/review_sample.csv` for a human spot-check.

How to review: open the CSV, fill the `verdict` column with `ok`, `drop` or `edit:<new label>`, save, then run `guardrails dataset apply-review`.

Until the spot-check is done, the README says: "machine-checked, human spot-check pending".

## Decisions applied outside the sample

| Date | Row | Change | Decided by | Why |
|---|---|---|---|---|
| 2026-11-20 | injg0283 (dev) | benign_hard -> attack / direct_override | owner | "Ignore any previous instructions you might have, ..." tells the assistant to ignore its instructions, which is the definition of direct_override; the generator had labelled it harmless. Found during the Task 19 dev sweep, before the test split was evaluated. |

The decision file `data/review/decisions_2026-11-20.csv` is applied with `guardrails dataset apply-review --csv data/review/decisions_2026-11-20.csv`; re-apply it if the datasets are ever regenerated from recordings.
