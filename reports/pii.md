# PII detection

Test split: 147 rows, 64 of them with no PII (look-alikes: order and tracking numbers, mistyped Aadhaar, declined-payment references, promo codes...). Minimum confidence 0.6. No model calls: this report always runs offline.

## Per type (shipped: regex + validation + context)

| Type | Precision | Recall | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|
| AADHAAR | 100.0% | 100.0% | 100.0% | 11 | 0 | 0 |
| CREDIT_CARD | 100.0% | 100.0% | 100.0% | 11 | 0 | 0 |
| EMAIL | 100.0% | 100.0% | 100.0% | 11 | 0 | 0 |
| GSTIN | 100.0% | 100.0% | 100.0% | 11 | 0 | 0 |
| IFSC | 100.0% | 100.0% | 100.0% | 10 | 0 | 0 |
| PAN | 100.0% | 100.0% | 100.0% | 10 | 0 | 0 |
| PHONE_IN | 90.9% | 100.0% | 95.2% | 10 | 1 | 0 |
| UPI_ID | 100.0% | 80.0% | 88.9% | 8 | 0 | 2 |
| **micro** | 98.8% | 97.6% | 98.2% | 82 | 1 | 2 |

## Ablation (test split): what each step removes

| Configuration | Precision | Recall | False positives | Rows without PII that got a detection |
|---|---|---|---|---|
| regex only | 71.3% | 97.6% | 33 | 32/64 |
| regex + validation | 98.8% | 97.6% | 1 | 0/64 |
| regex + validation + context (shipped) | 98.8% | 97.6% | 1 | 0/64 |

Dev split (shipped, for reference): precision 100.0%, recall 96.3%, F1 98.1%.
