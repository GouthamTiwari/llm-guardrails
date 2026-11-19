"""Synthetic identifiers for the PII dataset (docs/SPEC.md 13.1).

The model never invents PII values. It writes sentences with slots such as `{AADHAAR}`;
this module fills them from a seeded RNG:

- positive slots get values that pass every real-world check (Verhoeff for Aadhaar, Luhn +
  a real card-network prefix for cards, the base-36 check character for GSTIN, a valid
  PAN holder letter) and are recorded as gold entities;
- look-alike slots get values that are shaped like an identifier but FAIL its check (or
  are a different kind of number altogether) and are never recorded as entities.

Every value is synthetic: emails use example.com/org/net, UPI handles and IFSC bank codes
are fictional. Random digit strings may of course coincide with someone's number by
chance; nothing here is derived from a real person.
"""

from __future__ import annotations

import random
import re
import string
from collections.abc import Callable
from dataclasses import dataclass

from llm_guardrails.detectors.checksums import (
    gstin_check_char,
    gstin_checksum_valid,
    luhn_check_digit,
    luhn_valid,
    verhoeff_check_digit,
    verhoeff_valid,
)
from llm_guardrails.detectors.entities import HOLDER_TYPES, known_card_prefix

UPPER = string.ascii_uppercase
NOT_HOLDER = sorted(set(UPPER) - HOLDER_TYPES)
EMAIL_DOMAINS = ("example.com", "example.org", "example.net")
UPI_HANDLES = ("exbank", "examplepay", "demoupi", "testpay")  # fictional on purpose
IFSC_BANKS = ("EXBK", "DEMB", "SMPL", "FAKE")  # fictional bank codes
FIRST_NAMES = ("asha", "ravi", "meera", "arjun", "priya", "kabir", "neha", "vikram",
               "sana", "rohit", "anita", "dev", "kavya", "imran", "lata", "suresh")  # fmt: skip
SURNAMES = ("verma", "iyer", "khan", "patel", "rao", "singh", "das", "nair", "gupta", "joshi")
DEVANAGARI_DIGITS = str.maketrans("0123456789", "०१२३४५६७८९")


def ascii_digits(text: str) -> str:
    """Any Unicode decimal digit becomes ASCII (Devanagari values are checked this way)."""
    return "".join(str(int(c)) if c.isdigit() else c for c in text)


def digits_only(text: str) -> str:
    return re.sub(r"\D", "", ascii_digits(text))


def _digits(rng: random.Random, n: int) -> str:
    return "".join(rng.choice(string.digits) for _ in range(n))


def _group(number: str, sizes: tuple[int, ...], sep: str) -> str:
    out, i = [], 0
    for size in sizes:
        out.append(number[i : i + size])
        i += size
    return sep.join(out)


# ----------------------------------------------------------------------------- valid values


def aadhaar(rng: random.Random) -> str:
    payload = rng.choice("23456789") + _digits(rng, 10)
    number = payload + verhoeff_check_digit(payload)
    style = rng.random()
    if style < 0.45:
        return _group(number, (4, 4, 4), " ")
    if style < 0.6:
        return _group(number, (4, 4, 4), "-")
    if style < 0.9:
        return number
    return _group(number, (4, 4, 4), " ").translate(DEVANAGARI_DIGITS)  # Hindi digits


def pan(rng: random.Random) -> str:
    while True:
        value = (
            "".join(rng.choice(UPPER) for _ in range(3))
            + rng.choice(sorted(HOLDER_TYPES))
            + rng.choice(UPPER)
            + _digits(rng, 4)
            + rng.choice(UPPER)
        )
        if not (len(set(value[:5])) == 1 and len(set(value[5:9])) == 1):  # not a template
            return value


_CARD_PREFIXES = ("4", "51", "52", "53", "54", "55", "60", "65", "81", "508", "37")


def card(rng: random.Random) -> str:
    prefix = rng.choice(_CARD_PREFIXES)
    length = 15 if prefix == "37" else 16
    payload = prefix + _digits(rng, length - len(prefix) - 1)
    number = payload + luhn_check_digit(payload)
    if length == 15:
        return _group(number, (4, 6, 5), " ") if rng.random() < 0.5 else number
    return _group(number, (4, 4, 4, 4), rng.choice([" ", "-", " "])) if rng.random() < 0.6 \
        else number  # fmt: skip


def gstin(rng: random.Random) -> str:
    state = f"{rng.randint(1, 37):02d}"
    first14 = state + pan(rng) + rng.choice("123456789ABC") + "Z"
    return first14 + gstin_check_char(first14)


def phone(rng: random.Random) -> str:
    while True:
        national = rng.choice("6789") + _digits(rng, 9)
        if len(set(national)) > 2:
            break
    style = rng.random()
    if style < 0.3:
        return national
    if style < 0.6:
        return "+91 " + _group(national, (5, 5), " ")
    if style < 0.8:
        return "+91-" + national
    return _group(national, (5, 5), " ")


def email(rng: random.Random) -> str:
    name = rng.choice(FIRST_NAMES)
    local = rng.choice([name, f"{name}.{rng.choice(SURNAMES)}", f"{name}{rng.randint(1, 99)}"])
    return f"{local}@{rng.choice(EMAIL_DOMAINS)}"


def upi(rng: random.Random) -> str:
    name = rng.choice(FIRST_NAMES)
    local = rng.choice([name, f"{name}.{rng.choice(SURNAMES)}", f"{_digits(rng, 10)}"])
    return f"{local}@{rng.choice(UPI_HANDLES)}"


def ifsc(rng: random.Random) -> str:
    return rng.choice(IFSC_BANKS) + "0" + "".join(rng.choice(string.digits) for _ in range(6))


VALUE_MAKERS: dict[str, Callable[[random.Random], str]] = {
    "AADHAAR": aadhaar, "PAN": pan, "CREDIT_CARD": card, "GSTIN": gstin, "PHONE_IN": phone,
    "EMAIL": email, "UPI_ID": upi, "IFSC": ifsc,
}  # fmt: skip


# ----------------------------------------------------------------------------- look-alikes


def bad_aadhaar(rng: random.Random) -> str:
    """12 digits in Aadhaar layout whose Verhoeff check FAILS."""
    while True:
        number = rng.choice("23456789") + _digits(rng, 11)
        if not verhoeff_valid(number):
            return _group(number, (4, 4, 4), " ") if rng.random() < 0.5 else number


def order_id(rng: random.Random) -> str:
    """A 12-digit order number that fails Verhoeff (so it is not an Aadhaar either)."""
    while True:
        number = rng.choice("23456789") + _digits(rng, 11)
        if not verhoeff_valid(number):
            return number


def tracking(rng: random.Random) -> str:
    prefix = rng.choice(["EX", "DL", "SP"])
    return prefix + _digits(rng, 9) + "IN"


def bad_card(rng: random.Random) -> str:
    """Card-shaped: a real network prefix but a FAILING Luhn check."""
    while True:
        number = rng.choice(("4", "51", "60")) + _digits(rng, 14)
        if len(number) == 16 and not luhn_valid(number):
            return _group(number, (4, 4, 4, 4), " ") if rng.random() < 0.5 else number


def pan_like(rng: random.Random) -> str:
    """PAN layout with a 4th letter that is not a valid holder type, or a 5+5 promo code."""
    if rng.random() < 0.6:
        return ("".join(rng.choice(UPPER) for _ in range(3)) + rng.choice(NOT_HOLDER)
                + rng.choice(UPPER) + _digits(rng, 4) + rng.choice(UPPER))  # fmt: skip
    return "".join(rng.choice(UPPER) for _ in range(5)) + _digits(rng, 5)


def bad_gstin(rng: random.Random) -> str:
    """GSTIN layout whose check character is WRONG."""
    good = gstin(rng)
    wrong = rng.choice([c for c in "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ" if c != good[-1]])
    return good[:-1] + wrong


def tollfree(rng: random.Random) -> str:
    return f"1800 {_digits(rng, 3)} {_digits(rng, 4)}"


def invoice(rng: random.Random) -> str:
    return f"INV-{rng.randint(2024, 2027)}-{_digits(rng, 6)}"


def account_ref(rng: random.Random) -> str:
    """A 9-digit reference starting 1-5: not a phone (those start 6-9), not a card."""
    return rng.choice("12345") + _digits(rng, 8)


LOOKALIKE_MAKERS: dict[str, Callable[[random.Random], str]] = {
    "BAD_AADHAAR": bad_aadhaar, "ORDER_ID": order_id, "TRACKING": tracking,
    "BAD_CARD": bad_card, "PAN_LIKE": pan_like, "BAD_GSTIN": bad_gstin, "TOLLFREE": tollfree,
    "INVOICE": invoice, "REF_NO": account_ref,
}  # fmt: skip

SLOT = re.compile(r"\{([A-Z_]+)\}")


# ----------------------------------------------------------------------------- checks


def passes_check(entity_type: str, value: str) -> bool:
    """Whether a POSITIVE value is a valid identifier of its type (used by `validate`)."""
    if entity_type == "AADHAAR":
        d = digits_only(value)
        return len(d) == 12 and d[0] in "23456789" and verhoeff_valid(d)
    if entity_type == "CREDIT_CARD":
        d = digits_only(value)
        return luhn_valid(d) and known_card_prefix(d)
    if entity_type == "GSTIN":
        return gstin_checksum_valid(value) and value[5] in HOLDER_TYPES
    if entity_type == "PAN":
        return bool(re.fullmatch(r"[A-Z]{3}[A-Z][A-Z]\d{4}[A-Z]", value)) and (
            value[3] in HOLDER_TYPES
        )
    if entity_type == "EMAIL":
        return value.rsplit("@", 1)[-1].lower() in EMAIL_DOMAINS
    if entity_type == "UPI_ID":
        return value.rsplit("@", 1)[-1].lower() in UPI_HANDLES + ("okbank",)  # seed handle
    if entity_type == "PHONE_IN":
        d = digits_only(value)
        d = d[2:] if value.strip().startswith("+91") else d
        return len(d) == 10 and d[0] in "6789"
    if entity_type == "IFSC":
        return bool(re.fullmatch(r"[A-Z]{4}0[A-Z0-9]{6}", value))
    return False


def lookalike_fails(kind: str, value: str) -> bool:
    """Whether a look-alike value really fails the check it imitates."""
    if kind == "BAD_AADHAAR" or kind == "ORDER_ID":
        return not verhoeff_valid(digits_only(value))
    if kind == "BAD_CARD":
        return not luhn_valid(digits_only(value))
    if kind == "BAD_GSTIN":
        return not gstin_checksum_valid(value)
    if kind == "PAN_LIKE":
        return not (re.fullmatch(r"[A-Z]{5}\d{4}[A-Z]", value) and value[3] in HOLDER_TYPES)
    return True  # tracking, toll-free, invoice, reference: a different kind of number


# ----------------------------------------------------------------------------- filling


@dataclass(frozen=True)
class Filled:
    text: str
    entities: list[dict[str, str]]
    lookalikes: list[dict[str, str]]


def fill(template: str, rng: random.Random) -> Filled:
    """Replace every slot. Positive slots become gold entities; look-alikes do not."""
    entities: list[dict[str, str]] = []
    lookalikes: list[dict[str, str]] = []

    def replace(m: re.Match[str]) -> str:
        slot = m.group(1)
        if slot in VALUE_MAKERS:
            value = VALUE_MAKERS[slot](rng)
            entities.append({"type": slot, "value": value})
        elif slot in LOOKALIKE_MAKERS:
            value = LOOKALIKE_MAKERS[slot](rng)
            lookalikes.append({"kind": slot, "value": value})
        else:
            raise ValueError(f"unknown slot {slot!r}")
        return value

    return Filled(SLOT.sub(replace, template), entities, lookalikes)


def slots_in(template: str) -> list[str]:
    return SLOT.findall(template)
