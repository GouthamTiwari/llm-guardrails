# Task 01: Repo setup
Branch: main (first commit) | Date: 2026-11-19

## What changed
- `.githooks/`, `.gitattributes`, `.claude/settings.json`: copied from the kit; owner email changed to the GitHub noreply address (spec decision 17.2). The kit's `.claude/agents/` was not copied because this project uses the single-session workflow.
- `.gitignore`: kit list plus `_kit/`, `.spend/`, `.hypothesis/`.
- `.env.example`: all settings from spec 2 with safe defaults and empty secrets. A local `.env` (ignored) was built by a script that copied only the Bedrock key from the Semantic Cache project and generated a random `LOG_HASH_SALT`; no value was printed.
- `pyproject.toml`: package `llm-guardrails` / import `llm_guardrails`, Goutham Tiwari as the only author, dependencies from spec 2.1, ruff and pytest config (`live` marker deselected by default).
- `LICENSE` (MIT, Goutham Tiwari), `README.md` skeleton with the results markers, `CLAUDE.md` (single-session rules).
- `.github/workflows/ci.yml`: lint, tests, checksum self-test, authorship audit.
- `data/seed/`: the kit's seed datasets and `make_seeds.py`. `tests/reference/reference_checksums.py`: the kit's tested checksum code, kept unchanged as the oracle for Task 04.
- `docs/SPEC.md`, `docs/TASKS.md`, `docs/THREAT_MODEL.md`: the approved plan, including the decisions log (section 20).
- `src/llm_guardrails/__init__.py`, `tests/test_smoke.py`.

## How to verify
- `uv sync` → installs without errors
- `uv run ruff check .` and `uv run ruff format --check .` → clean
- `uv run pytest` → 1 passed
- `uv run python -I tests/reference/reference_checksums.py` → `checksum self-test passed`
- `bash .githooks/audit-authors.sh` → `OK: all N commits authored by Goutham Tiwari, no co-author lines`
- To regenerate the seeds: `cd data/seed` then `PYTHONPATH=../../tests/reference python make_seeds.py` (writes the four jsonl files in place).

## Tests
- `tests/test_smoke.py` (1 test).

## Out of scope / follow-ups
- GitHub repo creation and first push wait for Goutham's yes.

## Self-review
- No secrets staged (checked `git diff --cached` for key shapes). `.env` and `_kit/` confirmed ignored.
