# Injection detection

Mode: replay. Operating point from the dev sweep: T_BLOCK 0.53, T_ALLOW 0.22, T_LLM 0.8, λ 0.5. Rates show the 95% Wilson interval and the counts.

## Test split (202 rows, evaluated once)

| Layers | Attack recall | Benign false-positive rate |
|---|---|---|
| Rules only (layer 1) | 18.0% (11.7–26.7, 18/100) | 1.0% (0.2–5.3, 1/102) |
| Embeddings only (layer 2) | 18.0% (11.7–26.7, 18/100) | 1.0% (0.2–5.3, 1/102) |
| Rules + embeddings (grey zone allowed) | 31.0% (22.8–40.6, 31/100) | 2.0% (0.5–6.9, 2/102) |
| Rules + embeddings + LLM classifier (shipped) | 95.0% (88.8–97.9, 95/100) | 5.9% (2.7–12.2, 6/102) |

Hard-benign false-positive rate: 11.3% (5.3–22.6, 6/53). Classifier calls: 126 of 202 inputs (62.4%), 0 failed closed.

Cost per 1,000 requests (USD): rules 0.0, embeddings 0.000736, llm_classifier 0.216143

### Per category (shipped configuration)

| Category | Attack recall | Benign FPR |
|---|---|---|
| benign | – | 0.0% (0.0–7.3, 0/49) |
| benign_hard | – | 11.3% (5.3–22.6, 6/53) |
| delimiter_confusion | 100.0% (67.6–100.0, 8/8) | – |
| direct_override | 100.0% (72.2–100.0, 10/10) | – |
| encoding_obfuscation | 63.6% (35.4–84.8, 7/11) | – |
| exfiltration | 100.0% (75.8–100.0, 12/12) | – |
| indirect_document | 91.7% (64.6–98.5, 11/12) | – |
| multilingual | 100.0% (74.1–100.0, 11/11) | – |
| roleplay_jailbreak | 100.0% (75.8–100.0, 12/12) | – |
| system_prompt_extraction | 100.0% (74.1–100.0, 11/11) | – |
| tool_abuse | 100.0% (77.2–100.0, 13/13) | – |

### Per language (shipped configuration)

| Language | Attack recall | Benign FPR |
|---|---|---|
| en | 91.7% (81.9–96.4, 55/60) | 7.5% (3.5–15.4, 6/80) |
| es | 100.0% (79.6–100.0, 15/15) | 0.0% (0.0–43.5, 0/5) |
| hi | 100.0% (83.2–100.0, 19/19) | 0.0% (0.0–19.4, 0/16) |
| hinglish | 100.0% (56.5–100.0, 5/5) | – |
| mixed | 100.0% (20.6–100.0, 1/1) | 0.0% (0.0–79.3, 0/1) |

## Dev split, reference (205 rows; the sweep tuned on it)

| Layers | Attack recall | Benign false-positive rate |
|---|---|---|
| Rules only (layer 1) | 36.9% (28.2–46.5, 38/103) | 0.0% (0.0–3.6, 0/102) |
| Embeddings only (layer 2) | 21.4% (14.5–30.2, 22/103) | 0.0% (0.0–3.6, 0/102) |
| Rules + embeddings (grey zone allowed) | 45.6% (36.3–55.2, 47/103) | 0.0% (0.0–3.6, 0/102) |
| Rules + embeddings + LLM classifier (shipped) | 97.1% (91.8–99.0, 100/103) | 2.0% (0.5–6.9, 2/102) |
| LLM classifier on every input (reference) | 96.1% (90.4–98.5, 99/103) | 2.0% (0.5–6.9, 2/102) |

## External check: deepset/prompt-injections test split

deepset/prompt-injections, test split (Apache-2.0), downloaded at eval time into data/raw/ and not committed. Generic English and German prompts, not banking: an out-of-domain check.

| Layers | Attack recall | Benign false-positive rate |
|---|---|---|
| Rules only (layer 1) | 1.7% (0.3–8.9, 1/60) | 0.0% (0.0–6.4, 0/56) |
| Embeddings only (layer 2) | 0.0% (0.0–6.0, 0/60) | 0.0% (0.0–6.4, 0/56) |
| Rules + embeddings (grey zone allowed) | 1.7% (0.3–8.9, 1/60) | 0.0% (0.0–6.4, 0/56) |
| Rules + embeddings + LLM classifier (shipped) | 28.3% (18.5–40.8, 17/60) | 0.0% (0.0–6.4, 0/56) |

Classifier calls: 27 of 116 inputs. The classifier only sees inputs that the rules or the (banking-focused) embedding corpus put in the grey zone, so out-of-domain attacks mostly never reach it.
