"""Check-digit algorithms used by Indian IDs and payment cards (docs/SPEC.md 6.4).

- Aadhaar: Verhoeff (catches every single-digit error and every adjacent swap).
- Cards: Luhn (mod 10 with doubling).
- GSTIN: base-36 weighted sum (mod 36).

All functions take the bare characters (no spaces or hyphens) and return False on anything
malformed instead of raising, because detectors call them on untrusted text.
"""

from __future__ import annotations

# Verhoeff tables: multiplication in the dihedral group D5, the position permutation, and
# the inverse. These are the standard published tables.
_VERHOEFF_D = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 2, 3, 4, 0, 6, 7, 8, 9, 5),
    (2, 3, 4, 0, 1, 7, 8, 9, 5, 6),
    (3, 4, 0, 1, 2, 8, 9, 5, 6, 7),
    (4, 0, 1, 2, 3, 9, 5, 6, 7, 8),
    (5, 9, 8, 7, 6, 0, 4, 3, 2, 1),
    (6, 5, 9, 8, 7, 1, 0, 4, 3, 2),
    (7, 6, 5, 9, 8, 2, 1, 0, 4, 3),
    (8, 7, 6, 5, 9, 3, 2, 1, 0, 4),
    (9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
)
_VERHOEFF_P = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 5, 7, 6, 2, 8, 3, 0, 9, 4),
    (5, 8, 0, 3, 7, 9, 6, 1, 4, 2),
    (8, 9, 1, 6, 0, 4, 3, 5, 2, 7),
    (9, 4, 5, 3, 1, 2, 6, 8, 7, 0),
    (4, 2, 8, 6, 5, 7, 3, 9, 0, 1),
    (2, 7, 9, 3, 8, 0, 6, 4, 1, 5),
    (7, 0, 4, 6, 9, 1, 3, 2, 5, 8),
)
_VERHOEFF_INV = (0, 4, 3, 2, 1, 5, 6, 7, 8, 9)

BASE36 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_B36_INDEX = {ch: i for i, ch in enumerate(BASE36)}


def _is_ascii_digits(s: str) -> bool:
    return bool(s) and s.isascii() and s.isdigit()


def verhoeff_valid(number: str) -> bool:
    """True when the last digit is a correct Verhoeff check digit for the rest."""
    if not _is_ascii_digits(number):
        return False
    c = 0
    for i, ch in enumerate(reversed(number)):
        c = _VERHOEFF_D[c][_VERHOEFF_P[i % 8][int(ch)]]
    return c == 0


def verhoeff_check_digit(payload: str) -> str:
    """The digit to append to `payload` so that the result is Verhoeff-valid."""
    if not _is_ascii_digits(payload):
        raise ValueError("payload must be ASCII digits")
    c = 0
    for i, ch in enumerate(reversed(payload)):
        c = _VERHOEFF_D[c][_VERHOEFF_P[(i + 1) % 8][int(ch)]]
    return str(_VERHOEFF_INV[c])


def luhn_valid(number: str) -> bool:
    """Luhn check over 13-19 digits (the card number lengths in use)."""
    if not _is_ascii_digits(number) or not 13 <= len(number) <= 19:
        return False
    total = 0
    for i, ch in enumerate(reversed(number)):
        v = int(ch)
        if i % 2 == 1:  # every second digit from the right is doubled
            v *= 2
            if v > 9:
                v -= 9
        total += v
    return total % 10 == 0


def luhn_check_digit(payload: str) -> str:
    """The digit to append to `payload` so that the result passes Luhn."""
    if not _is_ascii_digits(payload):
        raise ValueError("payload must be ASCII digits")
    for d in "0123456789":
        total = 0
        for i, ch in enumerate(reversed(payload + d)):
            v = int(ch)
            if i % 2 == 1:
                v = v * 2 - 9 if v * 2 > 9 else v * 2
            total += v
        if total % 10 == 0:
            return d
    raise AssertionError("unreachable: one of ten digits always works")


def gstin_check_char(first14: str) -> str:
    """GSTIN check character: weights alternate 1, 2 from the left; each product adds its
    base-36 quotient and remainder; the check is what brings the total to a multiple of 36."""
    if len(first14) != 14 or any(ch not in _B36_INDEX for ch in first14):
        raise ValueError("need 14 base-36 characters (0-9, A-Z)")
    total = 0
    for i, ch in enumerate(first14):
        v = _B36_INDEX[ch] * (1 if i % 2 == 0 else 2)
        total += v // 36 + v % 36
    return BASE36[(36 - total % 36) % 36]


def gstin_checksum_valid(gstin: str) -> bool:
    """Only the check character (format rules live in the GSTIN detector)."""
    if len(gstin) != 15 or any(ch not in _B36_INDEX for ch in gstin):
        return False
    return gstin_check_char(gstin[:14]) == gstin[14]
