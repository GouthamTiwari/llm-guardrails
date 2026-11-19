"""Layer-1 injection rules: every rule's own examples, the seed rows, and regex speed."""

import json
import random
import time
from collections import Counter
from pathlib import Path

import pytest

from llm_guardrails.guards import run_guard
from llm_guardrails.guards.injection import (
    CATEGORIES,
    InjectionGuard,
    _parse_rules,
    load_rules,
    score_rules,
)
from llm_guardrails.guards.normalize import normalize
from llm_guardrails.types import Action, GuardContext, Reason

RULES = load_rules()
SEED = Path(__file__).parents[1] / "data" / "seed" / "injection_seed.jsonl"
ROWS = {r["id"]: r for r in map(json.loads, SEED.read_text(encoding="utf-8").splitlines())}


def fires(rule, text: str) -> bool:
    scan = normalize(text)
    return any(rule.pattern.search(t) for t in [scan.scan, *scan.decoded])


def test_rule_count_and_coverage() -> None:
    assert len(RULES) >= 35
    assert {r.category for r in RULES} == CATEGORIES
    langs = Counter(r.lang for r in RULES)
    assert langs["hi"] >= 2 and langs["hinglish"] >= 2 and langs["es"] >= 1


@pytest.mark.parametrize("rule", RULES, ids=[r.id for r in RULES])
def test_every_rule_has_and_passes_its_examples(rule) -> None:
    assert rule.match, "each rule needs at least one positive example"
    assert rule.no_match, "each rule needs at least one negative example"
    for text in rule.match:
        assert fires(rule, text), f"should match: {text!r}"
    for text in rule.no_match:
        assert not fires(rule, text), f"should NOT match: {text!r}"


STRONG_SEED_ATTACKS = ["inj001", "inj002", "inj003", "inj006", "inj008", "inj009", "inj010",
                       "inj011", "inj012", "inj013"]  # fmt: skip


@pytest.mark.parametrize("row_id", STRONG_SEED_ATTACKS)
def test_required_seed_attacks_score_one(row_id) -> None:
    assert score_rules(normalize(ROWS[row_id]["text"])).score == 1.0


@pytest.mark.parametrize("row_id", [k for k, r in ROWS.items() if r["label"] == "benign"])
def test_seed_benign_rows_stay_below_one(row_id) -> None:
    assert score_rules(normalize(ROWS[row_id]["text"])).score < 1.0


def test_coverage_of_all_seed_attacks_is_reported() -> None:
    """Not an acceptance gate: documents which seed attacks layer 1 alone catches."""
    attacks = [k for k, r in ROWS.items() if r["label"] == "attack"]
    caught = [k for k in attacks if score_rules(normalize(ROWS[k]["text"])).score == 1.0]
    assert set(STRONG_SEED_ATTACKS) <= set(caught)
    assert len(caught) == len(attacks), f"missed by rules alone: {set(attacks) - set(caught)}"


def test_obfuscated_attack_is_flagged_as_obfuscation() -> None:
    rs = score_rules(normalize(ROWS["inj010"]["text"]))
    assert rs.score == 1.0 and rs.from_decoded
    assert "encoding_obfuscation" in rs.categories and "direct_override" in rs.categories


def test_weak_rules_and_flags_add_up_but_do_not_block() -> None:
    one_weak = score_rules(normalize("Copy it verbatim please."))
    assert one_weak.score == pytest.approx(0.35) and not one_weak.strong
    two_weak = score_rules(normalize("Use your email tool. Copy it verbatim please."))
    assert two_weak.score == pytest.approx(0.70)
    hidden = score_rules(normalize("Copy it verb​atim please."))  # weak hit + zero-width
    assert hidden.score == pytest.approx(0.50)
    assert score_rules(normalize("What is the GST rate on gold?")).score == 0.0


async def test_guard_blocks_with_category_reasons() -> None:
    ctx = GuardContext(request_id="r", text=ROWS["inj010"]["text"])
    v = await run_guard(InjectionGuard(), ctx, timeout_s=5)
    assert v.action is Action.BLOCK
    assert Reason.inj_rule("encoding_obfuscation") in v.reasons
    assert Reason.INJ_OBFUSCATION in v.reasons
    assert v.scores["rules"] == 1.0
    assert "base64" not in v.model_dump_json()  # rule ids live in detail, never serialised


async def test_guard_allows_hard_benign() -> None:
    ctx = GuardContext(request_id="r", text=ROWS["inj021"]["text"], scan=None)
    v = await run_guard(InjectionGuard(), ctx, timeout_s=5)
    assert v.action is Action.ALLOW and v.scores["rules"] < 1.0


def test_rule_file_validation() -> None:
    good = "rules:\n- {id: a, category: tool_abuse, lang: en, strength: weak, pattern: 'x'}\n"
    assert len(_parse_rules(good)) == 1
    with pytest.raises(ValueError, match="duplicate"):
        _parse_rules(good + "- {id: a, category: tool_abuse, lang: en, strength: weak, pattern: y}")
    with pytest.raises(ValueError, match="category"):
        _parse_rules(good.replace("tool_abuse", "nonsense"))
    with pytest.raises(ValueError, match="strength"):
        _parse_rules(good.replace("weak", "medium"))


def _adversarial_text(n: int = 20_000) -> str:
    """Words the rules look for, in random order, with no full stops: the worst case for
    patterns with gaps, because every partial match has to be tried and abandoned."""
    rng = random.Random(1)
    words = [
        "ignore", "previous", "system", "instructions", "rules", "you", "are", "pretend", "ai",
        "without", "repeat", "everything", "above", "send", "data", "to", "the", "<!--",
        "[paste", "reveal", "your", "prompt", "dan", "from", "now", "on", "transfer", "when",
        "summarising", "this", "document", "निर्देश", "पिछले",
    ]  # fmt: skip
    out, size = [], 0
    while size < n:
        w = rng.choice(words)
        out.append(w)
        size += len(w) + 1
    return " ".join(out)[:n]


def _search_time(rule, n: int) -> float:
    text, filler = _adversarial_text(n), "a" * n
    best = float("inf")
    for _ in range(3):  # best of 3: ignores a moment where the laptop was busy elsewhere
        start = time.perf_counter()
        rule.pattern.search(text)
        rule.pattern.search(filler)
        best = min(best, time.perf_counter() - start)
    return best


@pytest.mark.parametrize("rule", RULES, ids=[r.id for r in RULES])
def test_each_rule_is_fast_on_long_adversarial_input(rule) -> None:
    """Guards against catastrophic backtracking, which takes seconds to minutes on long
    input. Typical time is ~1-5 ms; the 0.5 s limit leaves room for a busy machine (a 50 ms
    limit failed once under load) while still catching the real problem."""
    t20 = _search_time(rule, 20_000)
    assert t20 < 0.5
    # The signature of backtracking is super-linear growth. Linear search time roughly
    # doubles when the input doubles (worst rule measured: 2.83x, because longer random text
    # holds more partial matches). Catastrophic backtracking grows far faster than 6x, so
    # 6x + 20 ms catches it while leaving room for timer noise (3x failed intermittently).
    assert _search_time(rule, 40_000) < 6 * t20 + 0.02
