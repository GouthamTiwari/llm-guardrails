"""Entity detectors on the seed rows and on targeted traps. All values are synthetic."""

import json
import re
from pathlib import Path

import pytest

from llm_guardrails.detectors import (
    EntityDetector,
    detect_entities,
    get_detector,
    register,
    registered_types,
)
from llm_guardrails.detectors.checksums import luhn_check_digit, verhoeff_check_digit
from llm_guardrails.detectors.entities import _REGISTRY, known_card_prefix
from llm_guardrails.guards.normalize import normalize
from llm_guardrails.policy_config import known_entity_types

SEED = Path(__file__).parents[1] / "data" / "seed" / "pii_seed.jsonl"
ROWS = [json.loads(line) for line in SEED.read_text(encoding="utf-8").splitlines()]

# Synthetic numbers built here so their validity is certain.
AADHAAR = "23456789012" + verhoeff_check_digit("23456789012")  # passes Verhoeff
AADHAAR_SPACED = f"{AADHAAR[:4]} {AADHAAR[4:8]} {AADHAAR[8:]}"


def found(text: str, **kw) -> list[tuple[str, str]]:
    return [(m.type, m.raw) for m in detect_entities(text, **kw).matches]


@pytest.mark.parametrize("row", ROWS, ids=[r["id"] for r in ROWS])
def test_seed_rows_exact(row) -> None:
    want = sorted((e["type"], e["value"]) for e in row["entities"])
    assert sorted(found(row["text"])) == want


def test_registry_has_eight_types_and_policy_uses_it() -> None:
    assert registered_types() == {
        "AADHAAR", "PAN", "GSTIN", "CREDIT_CARD", "PHONE_IN", "EMAIL", "UPI_ID", "IFSC",
    }  # fmt: skip
    assert known_entity_types() == registered_types()


def test_new_type_is_one_small_class() -> None:
    class EmployeeId(EntityDetector):
        type = "TEST_EMP_ID"
        priority = 10
        pattern = re.compile(r"\bEMP-\d{6}\b")

    try:
        register(EmployeeId())
        assert found("badge EMP-123456 ok") == [("TEST_EMP_ID", "EMP-123456")]
        assert "TEST_EMP_ID" in known_entity_types()
    finally:
        _REGISTRY.pop("TEST_EMP_ID")


# ----------------------------------------------------------------------- Aadhaar


def test_aadhaar_formats() -> None:
    assert found(f"Aadhaar {AADHAAR}") == [("AADHAAR", AADHAAR)]
    assert found(f"aadhaar {AADHAAR_SPACED}") == [("AADHAAR", AADHAAR_SPACED)]
    hyphen = AADHAAR_SPACED.replace(" ", "-")
    assert found(f"UID {hyphen}") == [("AADHAAR", hyphen)]


def test_aadhaar_mixed_separators_rejected() -> None:
    mixed = f"{AADHAAR[:4]} {AADHAAR[4:8]}-{AADHAAR[8:]}"
    assert found(f"Aadhaar {mixed}") == []


def test_aadhaar_bad_checksum_and_bad_first_digit() -> None:
    bad = AADHAAR[:-1] + str((int(AADHAAR[-1]) + 1) % 10)
    assert found(f"Aadhaar {bad}") == []
    one = "1" + AADHAAR[1:]
    assert found(f"Aadhaar {one}") == []


def test_order_id_that_passes_verhoeff_is_suppressed() -> None:
    result = detect_entities(f"My order id is {AADHAAR}, where is it?")
    assert result.matches == [] and result.suppressed == 1
    assert found(f"Tracking number {AADHAAR}") == []
    # A positive word nearby wins over the negative one.
    assert found(f"order id {AADHAAR} is also my aadhaar") == [("AADHAAR", AADHAAR)]
    # Without context words at all, a valid number is still detected.
    assert found(f"number {AADHAAR}") == [("AADHAAR", AADHAAR)]


def test_devanagari_digit_aadhaar_detected_after_normalising() -> None:
    deva = AADHAAR_SPACED.translate(str.maketrans("0123456789", "०१२३४५६७८९"))
    clean = normalize(f"मेरा आधार {deva} है").clean
    assert [m.type for m in detect_entities(clean).matches] == ["AADHAAR"]


# ----------------------------------------------------------------------- cards


def _card(prefix: str, length: int = 16) -> str:
    body = prefix + "1" * (length - len(prefix) - 1)
    return body + luhn_check_digit(body)


def test_card_spaced_and_hyphenated() -> None:
    c = _card("4")
    spaced = " ".join(c[i : i + 4] for i in range(0, 16, 4))
    assert found(f"card {spaced}") == [("CREDIT_CARD", spaced)]
    assert found(f"card {c[:4]}-{c[4:8]}-{c[8:12]}-{c[12:]}")[0][0] == "CREDIT_CARD"


def test_card_needs_known_network_prefix() -> None:
    unknown = _card("99")  # Luhn-valid, but no network uses 99
    assert found(f"card {unknown}") == []
    for prefix in ["4", "51", "2221", "34", "37", "60", "65", "81", "508", "6011", "3528", "36"]:
        assert known_card_prefix(_card(prefix)), prefix


def test_card_beats_aadhaar_inside_it() -> None:
    c = _card("4", 16)
    assert [t for t, _ in found(f"card {c}")] == ["CREDIT_CARD"]


def test_card_found_inside_longer_digit_run() -> None:
    c = _card("4")
    assert ("CREDIT_CARD", c) in found(f"numbers 123 {c}")
    # "ref" right before it is negative context, so the same number is suppressed there.
    assert detect_entities(f"ref 123 {c}").suppressed >= 1


def test_tracking_number_failing_luhn_is_not_aadhaar_either() -> None:
    assert found("Reference 4111 1111 1111 1112 was declined") == []


# ----------------------------------------------------------------------- PAN / GSTIN


def test_pan_layout_and_case() -> None:
    assert found("PAN ABCPE1234F") == [("PAN", "ABCPE1234F")]
    assert found("code ABCDE1234F") == []  # D is not a holder type
    assert found("my pan is abcpe1234f") == [("PAN", "abcpe1234f")]
    assert found("coupon abcpe1234f") == []  # lower case needs the word "pan" nearby
    assert found("company ABCPE1234F")[0][0] == "PAN"  # "company" does not count as "pan"


def test_gstin_does_not_also_yield_pan() -> None:
    gstin = next(e["value"] for r in ROWS for e in r["entities"] if e["type"] == "GSTIN")
    assert found(f"GSTIN {gstin}") == [("GSTIN", gstin)]
    assert found(f"gst no {gstin.lower()}") == [("GSTIN", gstin.lower())]


def test_gstin_rules() -> None:
    assert found("27AAPFU0939F1ZV") == [("GSTIN", "27AAPFU0939F1ZV")]
    assert found("27AAPFU0939F1ZW") == []  # checksum
    assert found("40AAPFU0939F1ZV") == []  # no state 40


# ----------------------------------------------------------------------- phone


def test_phone_formats() -> None:
    for text, raw in [
        ("Call me on +91 98765 43210 after 5 pm.", "+91 98765 43210"),
        ("mobile 9876543210", "9876543210"),
        ("ph 09876543210", "09876543210"),
        ("whatsapp 987-654-3210", "987-654-3210"),
        ("+91-9876543210", "+91-9876543210"),
    ]:
        assert found(text) == [("PHONE_IN", raw)], text
    assert get_detector("PHONE_IN").normalize("+91 98765 43210") == "9876543210"


def test_phone_rejects() -> None:
    assert found("dial 1800 123 4567 for help") == []
    assert found("number 5876543210") == []  # mobiles start 6-9
    assert found("mobile 9999999999") == []  # one repeated digit
    assert found("invoice 9876543210") == []  # negative context


# ----------------------------------------------------------------------- email / UPI / IFSC


def test_email_is_not_upi_and_upi_is_not_email() -> None:
    assert found("mail ravi.kumar@example.com") == [("EMAIL", "ravi.kumar@example.com")]
    assert found("pay ravi@okbank") == [("UPI_ID", "ravi@okbank")]


def test_upi_known_handle_raises_confidence() -> None:
    plain = detect_entities("send to ravi@okbank").matches[0]
    known = detect_entities("send to ravi@oksbi").matches[0]
    assert plain.confidence == 0.7 and known.confidence == 0.85


def test_ifsc() -> None:
    assert found("IFSC SBIN0001234") == [("IFSC", "SBIN0001234")]
    assert found("code SBIN1001234") == []  # 5th character must be 0


# ----------------------------------------------------------------------- general


def test_positive_context_adds_confidence() -> None:
    assert detect_entities("ABCPE1234F").matches[0].confidence == 0.7
    assert detect_entities("PAN ABCPE1234F").matches[0].confidence == 0.8


def test_min_confidence_filter() -> None:
    result = detect_entities("ABCPE1234F", min_confidence=0.75)
    assert result.matches == [] and result.low_confidence == 1


def test_ablation_switches() -> None:
    bad = AADHAAR[:-1] + str((int(AADHAAR[-1]) + 1) % 10)
    assert found(f"id {bad}", validate=False) == [("AADHAAR", bad)]
    assert found(f"order id {AADHAAR}", use_context=False) == [("AADHAAR", AADHAAR)]


def test_values_hidden_from_repr() -> None:
    match = detect_entities(f"Aadhaar {AADHAAR}").matches[0]
    assert AADHAAR not in repr(match)
    assert match.value == AADHAAR


def test_talking_about_formats_finds_nothing() -> None:
    text = (
        "Aadhaar numbers have 12 digits, PAN is AAAAA9999A style and a GSTIN has 15 "
        "characters; cards use Luhn."
    )
    assert found(text) == []
