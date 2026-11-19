# Changelog

One line per merged task: date, task, summary. Details live in `docs/changelogs/`.

- 2026-11-19 · 01 repo-setup · uv project, ruff, pytest, CI, authorship hooks (noreply identity), seeds and checksum reference copied from the kit, spec/tasks/threat model committed.
- 2026-11-19 · 02 core-types · Verdict/Action types, reason codes, fail-closed run_guard (property-tested), settings loader, policy schema + default and banking_tax policies.
- 2026-11-19 · 03 normalizer · NFKC, invisible-char strip, digit folding; scan copy with homoglyph/leetspeak/spaced-letter folding and safe base64/hex decoding (property-tested).
- 2026-11-19 · 04 pii-detectors · Verhoeff/Luhn/GSTIN checksums (10k-case agreement with reference), 8 entity detectors with IIN prefixes, context suppression and overlap rules; 2 seed GSTINs corrected (spec decision 3).
