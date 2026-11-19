# Task 04: Checksums and entity detectors
Branch: task/04-pii-detectors | Date: 2026-11-19

## What changed
- `src/llm_guardrails/detectors/checksums.py`: Verhoeff (valid + check digit), Luhn (valid + check digit), GSTIN base-36 check character. Malformed input returns False instead of raising, because detectors run on untrusted text.
- `src/llm_guardrails/detectors/entities.py`: `EntityDetector` base class, registry (`register`, `registered_types`, `get_detector`) and the 8 detectors of spec 6.2 with priorities, base confidences, positive/negative context (6.3) and overlap resolution (longer span, then priority). Details worth knowing:
  - Digit types use extra lookarounds so a match cannot be glued to another digit group by one separator (a Luhn-failing "4111 1111 1111 1112" yields no Aadhaar from its first 12 digits).
  - Cards are found as 13-19-digit sub-runs of whole digit groups, then need Luhn **and** a known network prefix (IIN).
  - PAN and GSTIN also match lower case, but only when "pan"/"gst"/"gstin" is within 40 characters.
  - PAN rejects format templates such as `AAAAA9999A` (all letters the same and all digits the same), found by the "talking about the format" test.
  - `EntityMatch` hides `raw` and `value` from repr.
  - `detect_entities(..., validate=False, use_context=False)` exist for the eval's ablation.
- `src/llm_guardrails/detectors/__init__.py`: public exports.
- `src/llm_guardrails/policy_config.py` (outside the file list, as planned in Task 02): `known_entity_types()` now reads the registry.
- `data/seed/pii_seed.jsonl` (outside the file list): the GSTINs in pii008 and pii012 replaced, see spec decision 3. The kit's generator picked the embedded PAN's 4th letter at random (`Q`, `V`), which no real GSTIN can have.
- `docs/SPEC.md`: decision 3 added to section 20.

## How to verify
- `uv run pytest -q tests/test_checksums.py tests/test_detectors.py` → all pass
- `uv run python -c "from llm_guardrails.detectors import detect_entities as d; print([(m.type, m.confidence) for m in d('PAN ABCPE1234F, mobile 9876543210').matches])"` → `[('PAN', 0.8), ('PHONE_IN', 0.8)]`

## Tests
- `tests/test_checksums.py`: textbook values, malformed input, agreement with `tests/reference/reference_checksums.py` on 10,000 Hypothesis cases each for Verhoeff, Luhn and GSTIN, check-digit round trips, 10,000 reference-generated values of each type.
- `tests/test_detectors.py`: all 20 seed rows give exactly their gold entities; registry and plug-in type; Aadhaar formats, mixed separators, bad checksum, bad first digit, order-id suppression and positive-word override, Devanagari digits; card spacing, unknown IIN, card-beats-Aadhaar, card inside a longer run, Luhn-failing tracking number; PAN layout/case/template, GSTIN not also PAN, GSTIN state and checksum; phone formats and rejects (toll-free, 5-start, repeated digit, invoice context); email vs UPI; UPI known handle; IFSC; confidence bonus, min-confidence, ablation switches, repr hiding.
- Full suite: 135 passed.

## Out of scope / follow-ups
- Card length is not checked per network (only 13-19 overall); the eval will show whether that matters.
- UIN/NRI/TDS GSTIN variants are not detected (spec 6.4).

## Self-review
- Acceptance criteria of Task 04 checked: all met. All PII in tests is synthetic, built from check-digit functions or the kit's seeds.
