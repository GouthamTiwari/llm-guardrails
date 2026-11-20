# Grounding: does the answer say only what its cited sources say?

Mode: replay. min_overlap 0.8 (dev sweep). Positive class = answers that should be rejected (uncited, fabricated citation, unsupported, contradicted, overgeneralised, partially supported).

## Test split (83 answers)

| Checks | Reject precision | Reject recall | F1 | Good answers passed |
|---|---|---|---|---|
| Deterministic checks | 98.3% (91.1–99.7, 59/60) | 93.7% (84.8–97.5, 59/63) | 0.9593 | 95.0% (76.4–99.1, 19/20) |
| Deterministic + LLM entailment judge | 98.4% (91.7–99.7, 63/64) | 100.0% (94.2–100.0, 63/63) | 0.9921 | 95.0% (76.4–99.1, 19/20) |

The judge ran on 13 answers (13 calls), adding USD 0.000042 per answer on average and 1962.6 ms per judged answer; 0 fell back to the deterministic result.

### Confusion matrix, test, deterministic (gold label x outcome)

| Gold label | fabricated_citation | low_overlap | number_mismatch | pass | quantifier_mismatch | uncited |
|---|---|---|---|---|---|---|
| supported | 0 | 1 | 0 | 9 | 0 | 0 |
| correct_abstention | 0 | 0 | 0 | 10 | 0 | 0 |
| uncited | 0 | 0 | 0 | 0 | 0 | 10 |
| fabricated_citation | 10 | 0 | 0 | 0 | 0 | 0 |
| unsupported | 0 | 2 | 8 | 0 | 0 | 0 |
| contradicted | 0 | 1 | 6 | 2 | 0 | 0 |
| overgeneralised | 0 | 0 | 0 | 0 | 11 | 0 |
| partially_supported | 0 | 7 | 2 | 2 | 0 | 2 |

### Confusion matrix, test, + entailment (gold label x outcome)

| Gold label | entailment_contradicted | fabricated_citation | low_overlap | number_mismatch | pass | quantifier_mismatch | uncited |
|---|---|---|---|---|---|---|---|
| supported | 0 | 0 | 1 | 0 | 9 | 0 | 0 |
| correct_abstention | 0 | 0 | 0 | 0 | 10 | 0 | 0 |
| uncited | 0 | 0 | 0 | 0 | 0 | 0 | 10 |
| fabricated_citation | 0 | 10 | 0 | 0 | 0 | 0 | 0 |
| unsupported | 0 | 0 | 2 | 8 | 0 | 0 | 0 |
| contradicted | 2 | 0 | 1 | 6 | 0 | 0 | 0 |
| overgeneralised | 0 | 0 | 0 | 0 | 0 | 11 | 0 |
| partially_supported | 2 | 0 | 7 | 2 | 0 | 0 | 2 |

Dev split (min_overlap was chosen on it): deterministic F1 0.9524, with entailment F1 0.9767.
