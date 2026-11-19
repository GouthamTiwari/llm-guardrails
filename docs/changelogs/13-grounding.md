# Task 13: Grounding guard, deterministic checks
Branch: task/13-grounding | Date: 2026-11-19

## What changed
- `src/llm_guardrails/guards/grounding.py`:
  - **Sentence splitter** (spec 11.1): splits at `.`, `?`, `!` (and the Hindi danda) when what follows is whitespace plus an uppercase letter, digit or non-ASCII letter, or the end of the text. Protects `Rs.`, `No.`, `e.g.`, `i.e.`, `etc.`, `vs.`, `Dr.`, `Mr.`, `Ms.` and decimals. Bullet lines are separate sentences. Citations after the full stop (`... days. [S2]`) stay with their sentence.
  - **Citations:** `[S1]`, `[S1, S2]`, `[S1][S2]`. Any bracket holding id-shaped tokens counts as a citation, so `[note]` is read as a citation to a missing source and rejected, which errs on the safe side.
  - **Classes** (spec 11.2): abstention, non-factual (questions, under 4 words, courtesy) and factual. Hardened per SPEC decision 7: a number or a citation always makes a sentence factual, and a courtesy phrase must be the whole sentence.
  - **Checks** (spec 11.3, in order, first failure wins): uncited, fabricated citation, numbers, quotes (3+ words, verbatim inside one cited source), universal quantifiers, content-word overlap (`min_overlap`, default 0.5).
  - **Number parser:** `Rs 5,000`, `₹5000`, `Rs.5000`, `8.5%`, `60 days`, Indian grouping `1,00,000`, `two lakh`, `5 crore`, number words one to twenty, full-width and Devanagari digits. Not parsed: `5k` (documented limitation, pinned by a test), ordinals (`10th`), citation ids (`S1`), and the pronoun "one" (`no one`, `each one`).
  - **Answer outcome** (spec 11.5): `reject` blocks with every failing reason. `strip` removes failing sentences and keeps the line layout. If nothing factual survives, the answer becomes the fixed abstention with `grounding.empty_after_strip` (REDACT). An all-abstention answer without citations passes as `correct_abstention`.
  - **Guard:** reads `ctx.output` and `ctx.sources`, with mode and `min_overlap` from the policy. A missing answer raises, so `run_guard` fails closed. `detail.sentence_outcomes` holds index, class and outcome only, never the sentence text.
  - **Hardening:** NFKC plus removal of invisible format characters, so `ev​ery` cannot hide a quantifier. Each source is analysed once per answer: a 2,000-sentence answer went from 8.4 s to 0.16 s.
- `docs/SPEC.md`: decision 7 (sentence-class hardening, per-source quotes, the known short-claim gap).

## How to verify
- `uv run pytest -q tests/test_grounding.py` → 87 passed

## Tests
- `tests/test_grounding.py` (87):
  - All 12 seed rows get the expected decision. The expected first reason holds for gr005, gr007, gr008, gr009 and gr010; gr011 fails on number or overlap; gr012 is a correct abstention.
  - Number parser: 16 parsed forms plus 6 pinned non-parses, including `5k`. Number words compare equal to digits.
  - Splitter: abbreviations and decimals, citations after the stop, bullets, no split before a lowercase letter, all citation formats.
  - 15 class cases, including the decision-7 bypass attempts.
  - Check order, numbers checked against the cited sources only, pooled citations, quotes (verbatim, curly, short terms, stitched across sources), quantifiers and zero-width hiding, overlap threshold, content-word stemming.
  - Reject and strip modes, layout kept, empty after strip, a good answer left alone.
  - The guard reads the policy, can be disabled, fails closed without an answer and defaults to reject; no answer text appears in the dumped verdict.
  - Hypothesis properties: the parser and the guard never crash; strip and reject agree on what passes; an answer citing a missing source is never allowed.
  - A speed test on a 2,000-sentence answer.
- Full suite: 539 passed (2 live tests deselected).

## Live spend
None. No AWS calls.

## Out of scope / follow-ups
- Entailment judge (spec 11.4) is Task 14.
- `min_overlap` is swept on dev data in Task 20; 0.5 stays until then.
- Known limits for the README: shorthand numbers (`5k`), and uncited claims under 4 words without a number are skipped (spec 11.2). An answer that wrongly abstains passes; that is a quality failure, not a safety one (spec 11.6).

## Self-review
- Acceptance of Task 13: splitter, classes, checks 1–7, reject/strip modes. All 12 seed rows and the named first reasons are covered. Number parser cases including the pinned `5k`. All met.
- Found while reviewing: the literal spec let a claim escape by being phrased as a question, prefixed with "Please note", or placed inside an abstention. That is closed by decision 7 and tested.
- Found while reviewing: quotes were matched against all cited sources joined together, so a quote stitched across two documents passed. Each quote must now sit inside one source.
- Long hostile inputs (50,000 spaces, brackets, commas, dots, open quotes) each take under 0.1 s, with no backtracking blow-up.
- No secrets, no real PII (bank text is fictional), no fail-open path: a missing answer blocks, and a disabled guard is an explicit policy choice.
