# Tasks

Build order for `docs/SPEC.md`. One task at a time, in this order, on branch `task/NN-slug`, following `CLAUDE.md`. Each task lists the files it creates or edits; touching anything else needs a line in its changelog saying why. "Spec" references point to sections of `docs/SPEC.md`.

Every task's acceptance criteria implicitly include: `uv run ruff check .`, `uv run ruff format --check .` and `uv run pytest` pass; no network calls in unit tests; changelog written; no raw PII or secrets committed.

## Before Task 01 (Goutham, about 5 minutes)

Nothing new is needed in the AWS console: Bedrock (Nova 2 Lite + Titan v2 in `ap-south-1`) already works from this PC for the other two projects, and the same API key is reused.

1. Confirm the AWS budget alarm (USD 10, alerts at 50% and 80%) still exists: Billing → Budgets.
2. Say yes when Task 01 asks to create the private GitHub repo `GouthamTiwari/llm-guardrails` (or create it yourself on github.com/new with nothing ticked).

---

## 01 · Repo setup
**Files:** `.gitignore`, `.gitattributes`, `.githooks/*`, `.claude/settings.json`, `.env.example`, `.env` (local only), `pyproject.toml`, `uv.lock`, `LICENSE`, `README.md`, `CLAUDE.md`, `docs/CHANGELOG.md`, `docs/changelogs/01-repo-setup.md`, `.github/workflows/ci.yml`, `src/llm_guardrails/__init__.py`, `tests/test_smoke.py`, `tests/reference/reference_checksums.py`, `data/seed/*.jsonl`, `data/seed/make_seeds.py`
**Depends on:** –
**Why:** a safe, reproducible base: identity and hooks from the first commit, secrets ignored before any exist.
**Steps and acceptance:**
- [ ] `git init -b main`; `git config user.name "Goutham Tiwari"`, `git config user.email "227244284+GouthamTiwari@users.noreply.github.com"`, `git config core.hooksPath .githooks` (explain each in one line).
- [ ] Copy `.githooks/`, `.gitattributes`, `.claude/settings.json` from `_kit/llm-guardrails-kit/`; change `OWNER_EMAIL` in `_check.sh` and the email in `settings.json` to the noreply address. Do **not** copy `.claude/agents/` (single-session workflow).
- [ ] `.gitignore` = kit's list + `_kit/`, `.spend/`, `data/raw/`, `.embed_cache/`, `reports/tmp/`, `.env`.
- [ ] `.env.example` with: `AWS_REGION=ap-south-1`, `AWS_BEARER_TOKEN_BEDROCK=`, `AWS_ACCESS_KEY_ID=`, `AWS_SECRET_ACCESS_KEY=`, `BEDROCK_CHAT_MODEL_ID=global.amazon.nova-2-lite-v1:0`, `BEDROCK_JUDGE_MODEL_ID=global.amazon.nova-2-lite-v1:0`, `BEDROCK_EMBED_MODEL_ID=amazon.titan-embed-text-v2:0`, `EMBED_DIMENSIONS=1024`, `EMBED_REQUESTS_PER_SECOND=1`, `GUARDRAILS_LLM_MODE=replay`, `LIVE_BUDGET_USD=5`, `LOG_HASH_SALT=`.
- [ ] Create local `.env` from `.env.example`, copying the `AWS_BEARER_TOKEN_BEDROCK` value from `..\semantic-cache\.env` with a script that **never prints it**; generate a random `LOG_HASH_SALT`.
- [ ] `pyproject.toml`: name `llm-guardrails`, version `0.1.0`, `requires-python = ">=3.12"`, `authors = [{name = "Goutham Tiwari", email = "227244284+GouthamTiwari@users.noreply.github.com"}]`, license MIT, src layout, script `guardrails = "llm_guardrails.cli:app"`; deps from spec 2.1; dev deps `pytest`, `pytest-asyncio`, `hypothesis`, `ruff`; ruff line length 100, rules `E,F,I,UP,B,SIM`; pytest marker `live` deselected by default (`addopts = "-m 'not live'"`).
- [ ] `LICENSE` MIT, `Copyright (c) 2026 Goutham Tiwari`. README skeleton: title, one-paragraph pitch, "status: in progress", empty `<!-- RESULTS:START -->` / `<!-- RESULTS:END -->` markers, footer "Built by Goutham Tiwari".
- [ ] Copy seed jsonl files + `make_seeds.py` to `data/seed/`, `reference_checksums.py` to `tests/reference/`. `python -I tests/reference/reference_checksums.py` prints `checksum self-test passed`.
- [ ] CI (`ubuntu-latest`, Python 3.12 via `astral-sh/setup-uv`): `uv sync`, ruff check, ruff format --check, pytest; job 2 runs `bash .githooks/audit-authors.sh` with `fetch-depth: 0`.
- [ ] `tests/test_smoke.py`: package imports and exposes `__version__`.
- [ ] First commit; `git log --format=fuller` shows author and committer Goutham Tiwari with no trailer; audit prints OK.
- [ ] Ask Goutham, then `gh repo create GouthamTiwari/llm-guardrails --private --source . --remote origin`; push.

## 02 · Core types, guard runner, settings, policy
**Files:** `src/llm_guardrails/types.py`, `guards/__init__.py`, `guards/base.py`, `settings.py`, `policy_config.py`, `policies/default.yaml`, `policies/banking_tax.yaml`, `tests/test_types.py`, `tests/test_guard_runner.py`, `tests/test_policy_config.py`
**Depends on:** 01
**Why:** every later guard plugs into these; fail-closed is decided here once.
**Acceptance:**
- [ ] Types exactly as spec 4, including `ACTION_RANK`, reason-code constants (4.1) and error classes.
- [ ] `run_guard` per 4.2: timeout and exception → `BLOCK` for fail-closed, `ALLOW` for fail-open, both with `error_kind`; exception message never stored (test with an exception whose message contains a fake Aadhaar).
- [ ] Hypothesis property test: for any guard that raises a random exception type or sleeps past its timeout, a security guard with default config never returns `ALLOW`.
- [ ] `settings.py` reads env + `.env`; both auth options set → `ConfigError`; values are never included in `repr`.
- [ ] Policy loader: both YAML files load; unknown keys rejected; unknown entity type rejected; setting a security guard to fail-open logs one WARNING.

## 03 · Normaliser
**Files:** `guards/normalize.py`, `tests/test_normalize.py`
**Depends on:** 02
**Why:** obfuscation tricks must be undone before any detector looks at text.
**Acceptance:**
- [ ] Steps 1–8 of spec 5, `ScanText` with flags.
- [ ] Tests: zero-width inside "ignore", Cyrillic `о` in "ignоre", `1gn0re`, `s y s t e m`, `s.y.s.t.e.m`, base64 of an attack (decoded segment present, `clean` unchanged), hex, double-encoded base64, non-printable decode rejected, Devanagari digits folded, pure Cyrillic word untouched, length caps on decoding.
- [ ] Property tests: never raises; idempotent on `clean`; inserting zero-width characters into any seed attack does not change `scan`.

## 04 · Checksums and entity detectors
**Files:** `detectors/__init__.py`, `detectors/checksums.py`, `detectors/entities.py`, `tests/test_checksums.py`, `tests/test_detectors.py`
**Depends on:** 02 (can be done before 03)
**Why:** checksums turn noisy regexes into precise detectors.
**Acceptance:**
- [ ] Verhoeff, Luhn, GSTIN check char; agree with `tests/reference/reference_checksums.py` on 10,000 hypothesis cases each (valid generated and random).
- [ ] The 8 detectors, patterns, validation, IIN prefixes, priorities, context words and overlap resolution exactly as spec 6.2–6.4; registry with `register()`.
- [ ] All 20 seed PII rows give exactly their gold entities (look-alikes pii013–pii020 give none).
- [ ] Extra tests: order id that passes Verhoeff is suppressed by "order id"; same number after "Aadhaar" is detected; GSTIN does not also yield a PAN; email does not yield a UPI; `1800 123 4567` is not a phone; mixed separators in Aadhaar rejected; Devanagari-digit Aadhaar detected (after normalising).

## 05 · PII guard: block, redact, restore, safe logging
**Files:** `guards/pii.py`, `logging_utils.py`, `tests/test_pii_guard.py`, `tests/test_logging_utils.py`
**Depends on:** 03, 04
**Why:** detection is useless without the right action and without logs that stay clean.
**Acceptance:**
- [ ] Actions per policy (spec 6.5–6.6); consistent placeholders per value; placeholder-spoofing escape; restore only for `policy.pii.restore` types and only placeholders issued this request.
- [ ] `safe_record()` allow-list logger; salted hash prefixes; random salt + one warning when `LOG_HASH_SALT` empty.
- [ ] Test: dump verdicts, reports and captured logs for every seed PII row; no gold value appears anywhere.

## 06 · Pricing, spend cap, LLM and embedding clients, record/replay
**Files:** `pricing.py`, `spend.py`, `llm.py`, `embeddings.py`, `recording.py`, `tests/test_pricing.py`, `tests/test_spend.py`, `tests/test_llm.py`, `tests/test_embeddings.py`, `tests/test_live.py`
**Depends on:** 02
**Why:** every model call must be mockable, costed, capped and replayable.
**Acceptance:**
- [ ] Price table + `PRICES_CHECKED_ON`; unknown model → `UnknownModelPrice`.
- [ ] Ledger in `.spend/ledger.jsonl`; `BudgetExceeded` raised before the AWS call.
- [ ] `LLMClient` protocol; `BedrockLLM` (Converse; botocore retries `adaptive`, max 4 attempts; connect 5 s / read 30 s) returning `LLMResult(text, stop_reason, tokens_in, tokens_out, cost_usd, latency_ms)`; `FakeLLM` (scripted responses); record/replay wrapper per spec 13.5 with `ReplayMissing`.
- [ ] `TitanEmbedder` (1024 dims, normalize) with disk memo in `.embed_cache/`, 1 req/s pacer; `FakeEmbedder` (deterministic hash-seeded unit vectors).
- [ ] Unit tests mock boto3 with `botocore.stub.Stubber`. `tests/test_live.py` (`@pytest.mark.live`): one Converse call, one embedding, cost logged; passes with the real key; total < $0.01.

## 07 · Policy guard
**Files:** `guards/policy.py`, `tests/test_policy_guard.py`
**Depends on:** 03
**Why:** cheap structural limits run first and stop tool abuse.
**Acceptance:**
- [ ] Spec 9.2: size limits, denied actions, topic allow-list with short follow-up exemption, `check_tool_call`.
- [ ] inj019 (DROP TABLE) and inj020 (transfer without confirmation) blocked; "how do I transfer money to my sister?" allowed; off-topic blocked only under `banking_tax.yaml`.

## 08 · Injection layer 1: rules
**Files:** `rules/injection_rules.yaml`, `guards/injection.py` (rules part), `tests/test_injection_rules.py`
**Depends on:** 03
**Why:** fast, free, explainable first layer.
**Acceptance:**
- [ ] ≥ 35 rules covering every family in spec 7.1, each with a positive and a negative test case.
- [ ] Layer-1 score per spec 7.1; strong match in a decoded segment adds `inj.obfuscation`.
- [ ] All seed hard-benign rows (inj021–inj026) and benign rows score < 1.0; seed attacks inj001–003, 006, 008–013 score 1.0 (others may be left to later layers; the changelog lists which).
- [ ] No regex has nested unbounded quantifiers (test runs every rule against a 20,000-char adversarial string in < 50 ms each).

## 09 · Attack corpus, benign anchors, layer 2 embeddings
**Files:** `data/attack_corpus.jsonl`, `data/benign_anchors.jsonl`, `data/attack_corpus.npy`, `data/benign_anchors.npy`, `data/corpus_meta.json`, `guards/injection.py` (embed part), `eval/corpus.py`, `tests/test_injection_embed.py`
**Depends on:** 06, 08
**Why:** catches paraphrases the rules miss.
**Acceptance:**
- [ ] ≈200 attack and ≈150 benign-anchor texts (handwritten + generated with `--live`), all categories and languages, none copied from seed test candidates.
- [ ] `guardrails corpus embed --live` writes the `.npy` files + meta; meta/hash mismatch → `ConfigError`.
- [ ] Windowing and `embed_score` formula per spec 7.2, tested with `FakeEmbedder` and hand-made vectors.

## 10 · Schema guard and target schemas
**Files:** `schemas.py`, `guards/schema.py`, `prompts/ticket_plain.txt`, `prompts/ticket_schema.txt`, `tests/test_schema_extract.py`, `tests/test_schema_guard.py`
**Depends on:** 06
**Why:** the injection classifier and entailment judge both need validated JSON, so this comes before them.
**Acceptance:**
- [ ] Schemas exactly per spec 10.1 (strict, extra forbidden, duplicate keys rejected).
- [ ] Extraction per 10.2: fences, stray prose, nested braces in strings, zero/multiple objects.
- [ ] Retry loop per 10.3 with `FakeLLM`: valid on attempt 2 after a wrong enum; truncated → bigger `maxTokens`; 3 failures → `SchemaValidationFailed` and `BLOCK`; never returns unvalidated data (property test with random garbage outputs).
- [ ] Empty input (sch010) → valid `Refusal`.

## 11 · Injection layer 3 and the combined decision
**Files:** `guards/injection.py` (classifier + combine), `prompts/injection_classifier.txt`, `tests/test_injection_guard.py`
**Depends on:** 09, 10
**Why:** a cheap model only where rules and embeddings disagree.
**Acceptance:**
- [ ] Classifier prompt with nonce wrapping; `InjectionJudgement` via the schema guard (1 retry).
- [ ] Decision table of spec 7.4 with configurable thresholds; classifier only called in the grey band (test counts calls).
- [ ] Failure handling per 7.5: embed failure escalates to LLM; LLM failure → `BLOCK`.

## 12 · Context guard
**Files:** `guards/context.py`, `tests/test_context_guard.py`, `tests/fixtures/context_docs/*.txt`
**Depends on:** 08, 09
**Why:** indirect injection is the dangerous one because nobody typed it.
**Acceptance:**
- [ ] Spec 8: hidden-content strip, per-doc and per-line scan, three `on_injection` modes, nonce wrapper, escaping of `<<<`/`>>>`, fixed preamble, doc id validation.
- [ ] Fixtures: HTML comment instruction, white-on-white span, `display:none` div, "NOTE TO AI" line, fake end marker, inj015/inj016 as documents, and a clean document that passes unchanged.

## 13 · Grounding guard, deterministic checks
**Files:** `guards/grounding.py`, `tests/test_grounding.py`
**Depends on:** 02
**Why:** this is what the Regulatory RAG project will call on every answer.
**Acceptance:**
- [ ] Sentence splitter, classes and checks 1–7 of spec 11.1–11.3; `reject` and `strip` modes (11.5).
- [ ] All 12 seed grounding rows get the expected pass/reject (gr001–004 and gr012 pass; the rest reject), with the expected first reason code for gr005 (number), gr007 (quantifier), gr008 (uncited), gr009 (fabricated), gr010 (number), gr011 (number or overlap).
- [ ] Number parser tests: `Rs 5,000`, `₹5000`, `8.5%`, `two lakh`, `60 days`. Shorthand like `5k` is not parsed (documented limitation, with a test that pins this behaviour).

## 14 · Grounding entailment layer
**Files:** `guards/grounding.py` (entailment), `prompts/entailment.txt`, `tests/test_grounding_entailment.py`
**Depends on:** 10, 13
**Why:** measures whether an LLM judge adds anything over deterministic checks.
**Acceptance:**
- [ ] Spec 11.4: per supported sentence, concurrency 4, schema-validated output, `off | optional | required` behaviour; `optional` falls back with `grounding.entailment_unavailable`, `required` blocks on failure.

## 15 · Leak and exfiltration guard
**Files:** `guards/leak.py`, `tests/test_leak_guard.py`
**Depends on:** 05
**Why:** the output stage's last line of defence.
**Acceptance:**
- [ ] Spec 12: canary generation (unique per call, 10,000-uniqueness test) and detection variants (plain, spaced, reversed, base64), 8-gram system-prompt check, secret patterns, output PII with policy actions ignoring issued placeholders, image/link exfiltration with allow-list, IDNA host comparison.
- [ ] inj017's image URL in an output → `BLOCK leak.exfil_image`; a normal link to an allowed or short-query site passes.

## 16 · Pipeline, logging, metrics
**Files:** `pipeline.py`, `metrics.py`, `prompts/system_default.txt`, `tests/test_pipeline.py`, `tests/test_no_pii_in_logs.py`, `tests/test_fail_closed.py`
**Depends on:** 05, 07, 11, 12, 14, 15
**Why:** composition is where fail-open bugs usually hide.
**Acceptance:**
- [ ] `check_input`, `check_output`, `run` per spec 14; blocked request never calls the model (FakeLLM call count 0).
- [ ] Timeouts per guard and per request; `test_fail_closed.py` injects an exception and a timeout into **every** security guard in turn and asserts `decision == "block"`.
- [ ] One audit log line per request with exactly the spec 14.1 fields; `test_no_pii_in_logs.py` greps captured logs for every seed PII value and every seed input text: zero hits.
- [ ] Metrics from spec 14.2 increment correctly (read via `prometheus_client.REGISTRY`).

## 17 · HTTP API and CLI
**Files:** `api.py`, `cli.py`, `tests/test_api.py`, `tests/test_cli.py`
**Depends on:** 16
**Why:** the service and the three-command quickstart.
**Acceptance:**
- [ ] Endpoints and status codes per spec 15.1 (block = 200; internal error = 500 with no details), tested with `httpx`/`TestClient` in `GUARDRAILS_LLM_MODE=fake`.
- [ ] CLI commands from 15.2 exist (later tasks fill in eval/dataset bodies; stubs exit with a clear "implemented in task NN" message). `guardrails check "My PAN is ABCPE1234F"` prints a redacted result.

## 18 · Datasets: synthesis, generation, validation, splits, review
**Files:** `eval/synth.py`, `eval/generate.py`, `eval/datasets.py`, `eval/leakage.py`, `eval/review.py`, `data/injection.jsonl`, `data/pii.jsonl`, `data/grounding.jsonl`, `data/schema_inputs.jsonl`, `data/recordings/generate_*.jsonl`, `data/review/review_sample.csv`, `docs/DATASET_REVIEW.md`, `tests/test_datasets.py`
**Depends on:** 04, 06, 17
**Why:** the numbers are only as good as the data.
**Acceptance:**
- [ ] Generation per spec 13.1 (slot filling for PII, label sanity checks for grounding), sizes reached, recordings committed.
- [ ] `dataset validate` and `dataset leakcheck` pass and run as pytest tests (spec 13.2).
- [ ] 10% review sample written; `apply-review` works on a test CSV; `DATASET_REVIEW.md` records status (pending or applied).
- [ ] Live spend for this task recorded in the changelog (expected ≈ $0.15).

## 19 · Eval: injection (sweep, test, external) and PII
**Files:** `eval/metrics.py`, `eval/run_injection.py`, `eval/run_pii.py`, `eval/sweep.py`, `reports/operating_point.json`, `reports/injection.*`, `reports/pii.*`, `reports/what_got_through.md`, `data/embeddings/*`, `data/recordings/injection_*.jsonl`
**Depends on:** 11, 12, 18
**Why:** the headline numbers.
**Acceptance:**
- [ ] Metrics with formulas and Wilson intervals per spec 13.3 (unit-tested on toy data).
- [ ] `sweep injection` on dev only (refuses test) → `operating_point.json`; `InjectionConfig` defaults updated and tested to equal it.
- [ ] Test split evaluated once; per-layer, combined, per category, per language, hard-benign FPR; external deepset row (downloaded to `data/raw/`, not committed); `what_got_through.md`.
- [ ] PII per-type P/R/F1 and the three-step ablation.
- [ ] `guardrails eval --suite injection` and `--suite pii` reproduce the reports offline (replay) byte-for-byte except `generated_at`.

## 20 · Eval: schema, grounding, end to end; report generator
**Files:** `eval/run_schema.py`, `eval/run_grounding.py`, `eval/run_e2e.py`, `eval/mock_upstream.py`, `eval/report.py`, `reports/schema.*`, `reports/grounding.*`, `reports/e2e.*`, `reports/summary.md`, `data/recordings/*`, `README.md` (results block only), `tests/test_reports.py`
**Depends on:** 16, 19
**Why:** completes every experiment in spec 13.4 (1–5).
**Acceptance:**
- [ ] Schema: three styles × 100 inputs, live once then replay; metrics per 13.3.
- [ ] Grounding: deterministic vs + entailment, dev sweep of `min_overlap`, test confusion matrix.
- [ ] E2E: 500-request stream with `MockUpstream` (normal + compromised modes), per-guard p50/p95 and cost; separate 30-request live latency table.
- [ ] `guardrails report readme` fills the README block; `test_reports.py` asserts README block == generated summary.

## 21 · Optional baseline: Amazon Bedrock Guardrails
**Files:** `eval/baseline_bedrock.py`, `reports/baseline_bedrock_guardrails.*`, `tests/test_baseline_bedrock.py`
**Depends on:** 19
**Why:** "our layers vs the managed service" is the comparison an interviewer asks for.
**Acceptance:**
- [ ] Spec 13.4 item 6: create → `ApplyGuardrail` on the injection and PII test sets → delete in `finally` → `ListGuardrails` confirms none left; cost < $0.10 logged.
- [ ] Unit test (stubbed boto3) proves delete runs even when an apply call raises.
- [ ] If the API key lacks permission or the region lacks the API, the report states it and the task still closes.

## 22 · Adapters and examples
**Files:** `adapters/__init__.py`, `adapters/decorator.py`, `adapters/protocols.py`, `examples/regulatory_rag_adapter.py`, `examples/semantic_cache_adapter.py`, `examples/agent_tool_gate.py`, `tests/test_adapters.py`
**Depends on:** 16
**Why:** shows how the sibling projects plug in, without depending on them.
**Acceptance:**
- [ ] Spec 16: `@guarded` for sync and async functions; `should_cache` returns False for PII, redaction, block or failed grounding (tests for each); examples run with `GUARDRAILS_LLM_MODE=fake`.

## 23 · README, polish, release
**Files:** `README.md`, `docs/THREAT_MODEL.md` (links only), `docs/CHANGELOG.md`, `docs/changelogs/23-release.md`
**Depends on:** 20, 21, 22
**Why:** the 90-second recruiter read.
**Acceptance:**
- [ ] README: pitch ("the model proposes, deterministic checks decide"), Mermaid diagram of spec 3, 3-command quickstart (`uv sync`, `uv run guardrails check "..."`, `uv run guardrails eval --suite all`), threat model table, results block from `reports/`, "what still gets through" excerpt, design decisions (spec 17 items in plain words), "what this does not do" (spec 18), Nova-vs-Haiku note, honest dataset-review status, cost of the whole project from the ledger.
- [ ] Every number in the README is inside the generated block or quoted from a `reports/` file (checked by hand and by `test_reports.py`).
- [ ] CI green on GitHub; secret scan of full history (`git log -p | grep -E 'AKIA|ASIA|BEGIN .*PRIVATE KEY|AWS_BEARER_TOKEN_BEDROCK=.+'` → nothing); audit OK; tag `v1.0.0`; then tell Goutham the repo is ready to make public.
