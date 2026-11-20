# Task 20: Eval, schema, grounding, end to end; report generator
Branch: task/20-eval-schema-grounding-e2e | Dates: 2026-11-20 to 2026-11-21

## What changed
- `src/llm_guardrails/eval/run_schema.py`: plain vs schema vs schema + error-fed retries on all 105 ticket inputs.
  - **Metrics:** first-try and final validity, mean retries, extra tokens and USD from retries, added latency p50/p95, first-attempt errors by type, and whether empty, hostile or off-topic inputs are refused while real tickets become tickets.
- `src/llm_guardrails/eval/run_grounding.py`:
  - **Sweep:** the dev-only `min_overlap` sweep, writing `reports/grounding_operating_point.json`.
  - **Comparison:** deterministic checks vs deterministic + entailment judge on test and dev, with reject precision/recall, good answers passed, a confusion matrix (gold label × which check fired), and the judge's extra cost and latency.
- `src/llm_guardrails/eval/mock_upstream.py`: a deterministic answering model.
  - **Normal mode:** grounding rows get their dataset answer; other requests get a cited first sentence or the abstention.
  - **Compromised mode:** an attack that got past the input stage leaks the canary or an exfiltration image.
  - It also keeps every user turn, so leaks of raw PII to the model can be counted.
- `src/llm_guardrails/eval/run_e2e.py`: all 432 test-split requests through the full pipeline.
  - **Runs:** normal and compromised modes; PII leak checks; the document-scan ablation (classifier off vs on); cost per guard; local timings.
  - **Live latency:** a 30-request sample with real Nova answers.
  - **Live recording:** `--part` records one part at a time, so each paid run stays short.
- `src/llm_guardrails/eval/report.py`: `reports/summary.md` built only from the report JSONs, plus the README block helpers.
- `src/llm_guardrails/cli.py`: `eval --suite schema|grounding|e2e|all` (`--part`, `--live-latency`), `sweep grounding`, and `report readme` (works on `README.md` in the current folder).
- **Defaults:** `guards/grounding.py`, `policy_config.py` and `policies/*.yaml` set `min_overlap` to 0.80 from the dev sweep (decision 14b).
- **Plumbing:** `recording.py` gains `also_read` (replay other evals' recordings without paying twice); `guards/grounding.py` results carry the judge's model latency; `eval/run_injection.py` gains a shared `recorder()` helper.
- `README.md`: the results block, generated.
- `docs/SPEC.md`: decision 14.

## Results (test split, from the committed reports)

**Structured output:**

| Style | Valid on first try | Valid after retries |
|---|---|---|
| Plain prompt | 0.0% (0/105) | 0.0% |
| Schema in prompt | 95.2% (100/105) | 95.2% |
| **Schema + retries (shipped)** | 95.2% | **100.0% (105/105)** |

Only 5 inputs needed a retry, at about 3.2 s each. All 20 empty, hostile and off-topic inputs were refused, and all 85 real tickets became tickets.

**Grounding:**
- Deterministic checks reject 93.7% of bad answers (59/63) at 98.3% precision.
- **Adding the judge reaches 100% (63/63)** for USD 0.00004 per answer and about 2 s per judged answer.
- Good answers pass 95% (19/20) either way.

**End to end (432 requests):**
- Decisions match labels on 96.1%; attacks blocked 97% (97/100); normal messages blocked 5.9%.
- The compromised model's leaks were caught at the output 3/3, all by grounding.
- Raw PII reached the model in 0/63 rows, and card numbers were blocked 11/11.
- **Document scanning:** poisoned documents neutralised 2/12 with the classifier off (the default) vs 10/12 with it on.
- **Live latency:** p50 748 ms, p95 2,743 ms.

## How to verify
- `uv run guardrails eval --suite all` regenerates every report offline in about 5 seconds, then `uv run guardrails report readme`.
- `uv run pytest -q tests/test_eval_reproducible.py tests/test_reports.py` → 10 passed.
- `uv run pytest -q tests/test_eval_schema_grounding_e2e.py` → 12 passed.

## Tests
- `tests/test_eval_schema_grounding_e2e.py` (12):
  - **Schema:** styles on one input, the shared first call (the retry style's first attempt is free), summary metrics, a report with fakes.
  - **Grounding:** reject metrics and confusion; the sweep equals the committed point and every default (code, both policies, the YAML files); a report with fakes.
  - **End to end:** MockUpstream modes; a 432-request report with fakes; the pipeline leaks no PII beyond what the detector misses.
  - **Summary:** rows only for reports that exist, and the README block is idempotent and keeps the text around it.
- `tests/test_reports.py` (3): summary.md equals what the JSONs generate, the README block equals the summary, and every linked report exists. This is the numbers rule in CI.
- `tests/test_eval_reproducible.py` (+3): the schema and grounding reports reproduce, and every end-to-end number except local timings reproduces.
- `tests/test_cli.py`: the Task 20 commands run in a temporary folder.
- `tests/test_grounding.py` and `tests/test_pipeline.py`: four tests about other checks pin `min_overlap=0.5`.
- Full suite: 870 passed (3 live tests deselected).

## Live spend
Approved by Goutham (estimate about USD 0.21, at most USD 0.45).
- **Schema experiment:** 215 Nova calls, USD 0.1322.
- **Grounding judge:** 29 calls, USD 0.0077.
- **End-to-end parts:** USD 0.0351 (classifier calls not already recorded, plus Titan).
- **Live latency sample:** 30 requests with real answers, USD 0.0109.
- **Task total USD 0.1859.** Project live spend is USD 0.5586 of the USD 5 cap.

## Findings and follow-ups
- **v1.1, document scanning:** turn the document classifier on by default, and confirm on NEW documents that 10/12 is not luck.
- **v1.1, UPI bug:** fix it (a UPI ID followed by a full stop). In the end-to-end run the affected row was stopped earlier by the injection guard, so no PII reached the model, but the detector bug stands.
- **Small dev margin:** `min_overlap` 0.80 vs 0.60–0.70 is within one answer on dev; the report shows the whole curve.
- **Task 21** (optional Bedrock Guardrails baseline) needs a separate yes, per decision 1.

## Self-review
- Acceptance of Task 20:
  - Schema: three styles × 105 inputs, live once then replay, metrics per 13.3.
  - Grounding: deterministic vs + entailment, a dev sweep of `min_overlap`, and a test confusion matrix.
  - End to end: the full stream with normal and compromised modes, per-guard cost and latency, and a separate 30-request live latency table.
  - `guardrails report readme` fills the README, and `test_reports.py` asserts block == summary.
  - All met (432 requests instead of 500, decision 14c).
- **Found during the work:**
  - The PII leak metric first counted policy-allowed GSTIN/IFSC as leaks; fixed and tested.
  - Old stub tests ran real commands inside the repo folder; `report readme` could have overwritten the real README from another folder. It is now folder-relative, and the tests use temporary folders.
  - A test assertion matched "old" inside "Thresholds"; replaced with a unique placeholder.
  - The schema run finished 7 seconds inside the 10-minute limit. Recordings make any such interruption resumable for free.
- No test row was used to choose a setting; the document-classifier finding is reported, not acted on. No secrets in recordings or reports. All data is synthetic or Apache-2.0.
