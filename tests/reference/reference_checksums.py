"""Reference implementations of the checksums used by Indian ID formats and cards.

Purpose: a worked, tested example for the builder (Aadhaar = Verhoeff, GSTIN = base-36 mod,
cards = Luhn). The real implementation lives in the project under src/. Everything the seed
files contain is SYNTHETIC: random digits that happen to pass the checksum, not real people's IDs.
"""
import random

_D = [[0,1,2,3,4,5,6,7,8,9],[1,2,3,4,0,6,7,8,9,5],[2,3,4,0,1,7,8,9,5,6],[3,4,0,1,2,8,9,5,6,7],
      [4,0,1,2,3,9,5,6,7,8],[5,9,8,7,6,0,4,3,2,1],[6,5,9,8,7,1,0,4,3,2],[7,6,5,9,8,2,1,0,4,3],
      [8,7,6,5,9,3,2,1,0,4],[9,8,7,6,5,4,3,2,1,0]]
_P = [[0,1,2,3,4,5,6,7,8,9],[1,5,7,6,2,8,3,0,9,4],[5,8,0,3,7,9,6,1,4,2],[8,9,1,6,0,4,3,5,2,7],
      [9,4,5,3,1,2,6,8,7,0],[4,2,8,6,5,7,3,9,0,1],[2,7,9,3,8,0,6,4,1,5],[7,0,4,6,9,1,3,2,5,8]]
_INV = [0,4,3,2,1,5,6,7,8,9]
_B36 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def verhoeff_valid(number: str) -> bool:
    c = 0
    for i, ch in enumerate(reversed(number)):
        c = _D[c][_P[i % 8][int(ch)]]
    return c == 0


def verhoeff_check_digit(payload: str) -> str:
    c = 0
    for i, ch in enumerate(reversed(payload)):
        c = _D[c][_P[(i + 1) % 8][int(ch)]]
    return str(_INV[c])


def aadhaar_valid(number: str) -> bool:
    """12 digits, first digit 2-9, Verhoeff checksum. Spaces/hyphens are stripped first."""
    n = number.replace(" ", "").replace("-", "")
    return len(n) == 12 and n.isdigit() and n[0] in "23456789" and verhoeff_valid(n)


def luhn_valid(number: str) -> bool:
    n = number.replace(" ", "").replace("-", "")
    if not n.isdigit() or not 13 <= len(n) <= 19:
        return False
    total = 0
    for i, ch in enumerate(reversed(n)):
        v = int(ch)
        if i % 2 == 1:
            v *= 2
            if v > 9:
                v -= 9
        total += v
    return total % 10 == 0


def gstin_check_char(first14: str) -> str:
    total = 0
    for i, ch in enumerate(first14):
        v = _B36.index(ch) * (1 if i % 2 == 0 else 2)
        total += v // 36 + v % 36
    return _B36[(36 - total % 36) % 36]


def gstin_valid(g: str) -> bool:
    return len(g) == 15 and all(c in _B36 for c in g) and gstin_check_char(g[:14]) == g[14]


def make_aadhaar(rng: random.Random) -> str:
    payload = str(rng.randint(2, 9)) + "".join(str(rng.randint(0, 9)) for _ in range(10))
    return payload + verhoeff_check_digit(payload)


def make_gstin(rng: random.Random, state: str = "36") -> str:
    letters = "ABCDEFGHJKLMNPQRSTUVWXYZ"
    pan = "".join(rng.choice(letters) for _ in range(5)) + "".join(str(rng.randint(0, 9)) for _ in range(4)) + rng.choice(letters)
    first14 = state + pan + "1" + "Z"
    return first14 + gstin_check_char(first14)


def make_card(rng: random.Random, prefix: str = "4") -> str:
    body = prefix + "".join(str(rng.randint(0, 9)) for _ in range(14 - len(prefix) + 1))
    for check in "0123456789":
        if luhn_valid(body + check):
            return body + check
    raise AssertionError("unreachable")


if __name__ == "__main__":
    assert verhoeff_valid("2363") and not verhoeff_valid("2364")      # textbook Verhoeff example
    assert luhn_valid("4111111111111111")                              # standard public test card
    rng = random.Random(7)
    assert aadhaar_valid(make_aadhaar(rng)) and gstin_valid(make_gstin(rng)) and luhn_valid(make_card(rng))
    print("checksum self-test passed")
