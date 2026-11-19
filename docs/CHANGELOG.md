# Changelog

One line per merged task: date, task, summary. Details live in `docs/changelogs/`.

- 2026-11-19 · 01 repo-setup · uv project, ruff, pytest, CI, authorship hooks (noreply identity), seeds and checksum reference copied from the kit, spec/tasks/threat model committed.
- 2026-11-19 · 02 core-types · Verdict/Action types, reason codes, fail-closed run_guard (property-tested), settings loader, policy schema + default and banking_tax policies.
- 2026-11-19 · 03 normalizer · NFKC, invisible-char strip, digit folding; scan copy with homoglyph/leetspeak/spaced-letter folding and safe base64/hex decoding (property-tested).
- 2026-11-19 · 04 pii-detectors · Verhoeff/Luhn/GSTIN checksums (10k-case agreement with reference), 8 entity detectors with IIN prefixes, context suppression and overlap rules; 2 seed GSTINs corrected (spec decision 3).
- 2026-11-19 · 05 pii-guard · per-type block/redact/allow, consistent typed placeholders, spoof-proof restore, salted hashes and an allow-listed audit record; dump test proves no seed PII value reaches any log or report.
- 2026-11-19 · 06 llm-clients · price table, persistent USD 5 spend cap, Bedrock Converse + Titan clients (stub-tested), fakes, record/replay for offline evals; first live calls cost $0.000025.
- 2026-11-19 · 07 policy-guard · size limits first, denied actions on the scan text (obfuscation-proof), topic allow-list with Hindi/Hinglish keywords, tool-call gate for agents.
- 2026-11-19 · 08 injection-rules · 48 multilingual rules (37 strong, 11 weak) each with must-match and must-not-match examples, rule scoring with obfuscation signals, speed-tested regexes; normaliser no longer folds URLs/emails.
- 2026-11-19 · fix pacer-clock · Titan pacer used a 15.6 ms Windows clock and could fire early (flaky test, 5/40 short runs); now perf_counter (0/40).
- 2026-11-19 · 09 attack-corpus · 319 attacks + 249 benign anchors (10 languages, reviewed), Titan vectors committed, layer-2 similarity with benign pull-down, fail-closed on embed errors; provisional thresholds from a dev sanity check (decision 4); spend $0.073.
