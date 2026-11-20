# Structured output: plain vs schema vs schema + retries

Mode: replay. 105 support-ticket inputs (normal, multiple-issue, number-heavy, Hindi/Hinglish, hostile, empty, very long, off-topic), validated strictly against `TicketSummary | Refusal` (no type coercion, no extra keys).

| Style | Valid on first try | Valid after retries | Mean retries | USD per input |
|---|---|---|---|---|
| Plain prompt | 0.0% (0.0–3.5, 0/105) | 0.0% (0.0–3.5, 0/105) | 0.0 | 0.000722 |
| Schema in prompt | 95.2% (89.3–98.0, 100/105) | 95.2% (89.3–98.0, 100/105) | 0.0 | 0.000501 |
| Schema + error-fed retries (shipped) | 95.2% (89.3–98.0, 100/105) | 100.0% (96.5–100.0, 105/105) | 0.0476 | 0.000537 |

Retries cost 78.33 extra tokens and USD 0.000036 per input on average; 5 input(s) needed a retry (added latency p50 3198.4 ms, p95 3775.1 ms).

Refusal behaviour (schema + retries, valid outputs only): refused empty, hostile or off-topic input 100.0% (83.9–100.0, 20/20); produced a ticket for real tickets 100.0% (95.7–100.0, 85/85).

## First-attempt errors by type

| Style | constraint | missing_field | no_json | truncated |
|---|---|---|---|---|
| Plain prompt | 0 | 97 | 4 | 4 |
| Schema in prompt | 5 | 0 | 0 | 0 |
| Schema + error-fed retries (shipped) | 5 | 0 | 0 | 0 |
