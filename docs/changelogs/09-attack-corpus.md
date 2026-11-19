# Task 09: Attack corpus, benign anchors, injection layer 2
Branch: task/09-attack-corpus | Date: 2026-11-19

## What changed
- `data/attack_corpus.jsonl` (319 rows) and `data/benign_anchors.jsonl` (249 rows):
  - 134 handwritten attacks across all 9 categories in 10 languages (English, Hindi, Hinglish, Spanish, French, German, Tamil, Telugu, Marathi, Bengali) and 90 handwritten normal requests, 58 of them hard-benign.
  - Nova 2 Lite then wrote one reworded version per row in the same language (95 attacks, 89 benign kept), plus Hindi translations of the English rows on **both** sides (90 attacks, 70 benign), so Hindi itself doesn't look like an attack.
  - Encoded-attack rows are never reworded, because rewording would break the payload.
- `data/attack_corpus.npy`, `data/benign_anchors.npy`, `data/corpus_meta.json`: Titan v2 vectors (1024 dims, unit length) of the normaliser's scan text, plus the sha256 of each jsonl. Loading refuses vectors whose texts changed afterwards.
- `data/recordings/corpus_generate.jsonl`: every generation call recorded, so `python -m llm_guardrails.eval.corpus generate` rebuilds the corpus offline, for free, with identical output.
- `data/corpus_review.json`: rows dropped after reading every generated row, with reasons (4 rows: meaning reversed or lost).
- `src/llm_guardrails/eval/similarity.py`: character 5-gram Jaccard near-duplicate detection (no model needed; the same for Hindi and English). Task 18's leakage check reuses it.
- `src/llm_guardrails/eval/corpus.py`: generation (batched **per language**, with an explicit "do not translate"), a script check per language (output letters must be in the expected script, and no third script allowed), a Hindi translation pass, the review filter, a near-duplicate filter against each other and the seed/eval data, embedding, and meta. Entry point: `python -m llm_guardrails.eval.corpus generate|embed [--live]` (the `guardrails corpus` CLI wrapper arrives with the CLI in Task 17).
- `src/llm_guardrails/guards/injection.py`: layer 2 (`InjectionConfig`, `AttackCorpus` with tamper check, `embedding_text`, overlapping 1,000-char windows for texts over 2,000 chars, `embed_score = s_attack − λ·max(0, s_benign − s_attack)` clipped to [0, 1], best window wins). The guard blocks on a strong rule or embed score ≥ `t_block`. An embedding failure or a model/dims mismatch **fails closed** until Task 11 routes it to the LLM layer. Policy can switch embeddings off or override thresholds.
- `docs/SPEC.md`: decisions 4 (provisional thresholds, wider sweep grids) and 5 (corpus composition, test-set independence, cosine ≥ 0.90 leakage flag).

## Quality problems found and fixed during the task
- **First generation pass discarded.** Mixed-language batches made Nova translate English attacks into Hindi and Hinglish into Devanagari, so the language labels were wrong. Fixed with per-language batches, an explicit "do not translate", and the script check, which rejected 35 more off-script outputs on the second pass. The discarded pass cost $0.027.
- **Script check bug:** Hindi vowel signs aren't "letters", so a Hindi sentence containing "system prompt" looked mostly Latin. Vowel signs are now counted (40% threshold for non-Latin scripts).
- **Thresholds:** see decision 4. On the 30 seed rows (dev, sanity only) layer-2 scores were 0.19–0.76 for attacks and 0.00–0.30 for benign rows, so the spec's starting `t_block` 0.80 would almost never fire. Provisional defaults are now 0.60 / 0.35 until the Task 19 sweep.

## Live spend for this task
- Generation: 70 Nova calls, $0.0720 (including the discarded first pass). Embedding: 568 Titan calls, $0.00043. Sanity check: 30 Titan calls, $0.00002. **Project total so far: $0.0725.**

## How to verify
- `uv run pytest -q tests/test_injection_embed.py tests/test_corpus.py` → all pass
- `python -m llm_guardrails.eval.corpus generate` (no `--live`) → rebuilds the same two jsonl files from recordings

## Tests
- `tests/test_injection_embed.py` (12): score formula and benign pull-down, clipping, best window, windowing covers the text, a paraphrase missed by rules is blocked by embeddings, a strong rule never calls the embedder, embedding failure and model mismatch fail closed, thresholds and switches from policy, corpus load and tamper detection.
- `tests/test_corpus.py` (18): similarity, script check (8 cases including Cyrillic inside Hindi), per-language batching with a fake model, the Hindi pass, the near-duplicate filter, and committed-corpus invariants (well-formed, size and language spread with Hindi on both sides, reviewed drops absent, no near-copy of seed/eval data or of each other, vectors match texts and are unit length).
- Full suite: 355 passed (2 live tests deselected), including a test that the fingerprint ignores line endings.

## Out of scope / follow-ups
- Grey zone and the LLM classifier (Task 11); the threshold sweep (Task 19).
- Indirect injections behind a long harmless opening score low on whole-text similarity (seed inj015: 0.19). The context guard's per-line scan (Task 12) targets this.

## Self-review
- Acceptance criteria of Task 09 checked: corpus sizes (≈200/≈150 exceeded), all categories and languages, no copies of seed data, vectors + meta with mismatch error, formula tested with fake and hand-made vectors. The CLI name differs (module entry point now; CLI in Task 17).

## Review log
- Round 1 (squash merge, before push): the test run on `main` failed. Git's `* text=auto` checked the `.jsonl` files out with CRLF on Windows, so their sha256 no longer matched `corpus_meta.json`. The unpushed squash commit was undone (`git reset --soft HEAD~1`). Fixes: `.gitattributes` pins LF for `*.jsonl/*.json/*.yaml/*.yml/*.csv` and marks `*.npy` binary; the fingerprint is now computed on LF-normalised bytes (`text_sha256`), with a test; the tamper test's helper uses the same function (it wrote CRLF files on Windows). Nothing had reached GitHub.
