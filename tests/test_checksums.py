"""Our checksums must agree with the kit's tested reference implementation everywhere."""

import importlib.util
import random
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from llm_guardrails.detectors.checksums import (
    BASE36,
    gstin_check_char,
    gstin_checksum_valid,
    luhn_check_digit,
    luhn_valid,
    verhoeff_check_digit,
    verhoeff_valid,
)

_spec = importlib.util.spec_from_file_location(
    "reference_checksums", Path(__file__).parent / "reference" / "reference_checksums.py"
)
ref = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ref)

DIGITS = st.text(alphabet="0123456789", min_size=1, max_size=20)


def test_textbook_examples() -> None:
    assert verhoeff_valid("2363") and not verhoeff_valid("2364")
    assert verhoeff_check_digit("236") == "3"
    assert luhn_valid("4111111111111111") and not luhn_valid("4111111111111112")
    assert gstin_checksum_valid("27AAPFU0939F1ZV")  # public example GSTIN


def test_malformed_input_is_false_not_an_error() -> None:
    for bad in ["", "12a4", "१२३४", "1234 5678"]:
        assert not verhoeff_valid(bad)
        assert not luhn_valid(bad)
    assert not gstin_checksum_valid("27aapfu0939f1zv")  # lower case is not base-36 here
    assert not gstin_checksum_valid("27AAPFU0939F1Z")


@settings(max_examples=10_000, deadline=None)
@given(DIGITS)
def test_verhoeff_matches_reference(number: str) -> None:
    assert verhoeff_valid(number) == ref.verhoeff_valid(number)


@settings(max_examples=10_000, deadline=None)
@given(DIGITS)
def test_luhn_matches_reference(number: str) -> None:
    assert luhn_valid(number) == ref.luhn_valid(number)


@settings(max_examples=10_000, deadline=None)
@given(st.text(alphabet=BASE36, min_size=15, max_size=15))
def test_gstin_matches_reference(gstin: str) -> None:
    assert gstin_checksum_valid(gstin) == ref.gstin_valid(gstin)
    assert gstin_check_char(gstin[:14]) == ref.gstin_check_char(gstin[:14])


@settings(max_examples=2_000, deadline=None)
@given(st.text(alphabet="0123456789", min_size=1, max_size=18))
def test_check_digit_generators_round_trip(payload: str) -> None:
    assert verhoeff_valid(payload + verhoeff_check_digit(payload))
    assert verhoeff_check_digit(payload) == ref.verhoeff_check_digit(payload)
    full = payload + luhn_check_digit(payload)
    if 13 <= len(full) <= 19:
        assert luhn_valid(full)


def test_reference_generated_values_validate() -> None:
    rng = random.Random(7)
    for _ in range(10_000):
        assert verhoeff_valid(ref.make_aadhaar(rng))
        assert gstin_checksum_valid(ref.make_gstin(rng))
        assert luhn_valid(ref.make_card(rng))
