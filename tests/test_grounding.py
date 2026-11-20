"""Grounding guard, deterministic layer (docs/SPEC.md 11.1-11.3, 11.5; SPEC decision 7)."""

import json
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from llm_guardrails.guards import run_guard
from llm_guardrails.guards.grounding import (
    ABSTENTION,
    GroundingGuard,
    check_grounding,
    check_sentence,
    citations,
    classify,
    content_words,
    parse_numbers,
    split_sentences,
)
from llm_guardrails.policy_config import load_policy
from llm_guardrails.types import Action, GuardContext, Reason, SourceDoc

SEED = Path(__file__).parents[1] / "data" / "seed" / "grounding_seed.jsonl"
ROWS = {r["id"]: r for r in map(json.loads, SEED.read_text(encoding="utf-8").splitlines())}

S1 = SourceDoc(
    id="S1",
    text="Example Bank savings accounts require a minimum monthly average balance of "
    "Rs 5,000. A fee of Rs 200 applies if the balance falls below this.",
)
S2 = SourceDoc(
    id="S2",
    text="Example Bank card disputes must be raised within 60 days of the statement date. "
    "Refunds for approved disputes are credited within 10 working days.",
)
S3 = SourceDoc(
    id="S3",
    text="Example Bank offers home loans for tenures up to 25 years. Prepayment charges are "
    "not applicable on floating-rate loans.",
)
SOURCES = {s.id: s for s in (S1, S2, S3)}


def sources_of(row: dict) -> list[SourceDoc]:
    return [SourceDoc(**s) for s in row["sources"]]


# ----------------------------------------------------------------------------- seed rows

PASS_ROWS = ["gr001", "gr002", "gr003", "gr004", "gr012"]
FIRST_REASON = {
    "gr005": Reason.GROUNDING_NUMBER,
    "gr007": Reason.GROUNDING_QUANTIFIER,
    "gr008": Reason.GROUNDING_UNCITED,
    "gr009": Reason.GROUNDING_FABRICATED,
    "gr010": Reason.GROUNDING_NUMBER,
}


@pytest.mark.parametrize("row_id", sorted(ROWS))
def test_seed_rows_get_expected_decision(row_id) -> None:
    row = ROWS[row_id]
    result = check_grounding(row["answer"], sources_of(row))
    if row_id in PASS_ROWS:
        assert result.action is Action.ALLOW, result.sentences
    else:
        assert result.action is Action.BLOCK
        assert result.reasons


@pytest.mark.parametrize("row_id", sorted(FIRST_REASON))
def test_seed_rows_fail_for_the_expected_reason(row_id) -> None:
    row = ROWS[row_id]
    first_failure = next(s for s in check_grounding(row["answer"], sources_of(row)).sentences
                         if s.failed)  # fmt: skip
    assert first_failure.outcome == FIRST_REASON[row_id]


def test_gr011_fails_on_number_or_overlap() -> None:
    row = ROWS["gr011"]
    reasons = check_grounding(row["answer"], sources_of(row)).reasons
    assert set(reasons) & {Reason.GROUNDING_NUMBER, Reason.GROUNDING_LOW_OVERLAP}


def test_gr012_is_a_correct_abstention() -> None:
    row = ROWS["gr012"]
    assert check_grounding(row["answer"], sources_of(row)).outcome == "correct_abstention"


# ----------------------------------------------------------------------------- numbers


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Rs 5,000", {5000.0}),
        ("₹5000", {5000.0}),
        ("Rs.5000", {5000.0}),
        ("INR 5,000.50", {5000.5}),
        ("8.5%", {8.5}),
        ("two lakh", {200000.0}),
        ("Rs 2 lakh", {200000.0}),
        ("1,00,000", {100000.0}),
        ("5 crore", {50_000_000.0}),
        ("60 days", {60.0}),
        ("twenty", {20.0}),
        ("a lakh", {100000.0}),
        ("５０００", {5000.0}),  # full-width digits, NFKC
        ("५०००", {5000.0}),  # Devanagari digits
        ("The fee is Rs 5,000.", {5000.0}),
        ("Between 8.5% and 9.25% [S1]", {8.5, 9.25}),
    ],
)
def test_number_parser(text, expected) -> None:
    assert parse_numbers(text) == expected


@pytest.mark.parametrize("text", ["5k", "Rs 5k", "10th", "S1", "no one came", "each one of them"])
def test_number_parser_does_not_read_these(text) -> None:
    """Pinned limitation: shorthand like '5k' is NOT parsed, so a sentence saying 'Rs 5k' has
    no number for check 3 to compare (overlap still runs). Ordinals, citation ids and the
    pronoun 'one' are not numbers either."""
    assert parse_numbers(text) == set()


def test_number_word_and_digits_compare_equal() -> None:
    src = SourceDoc(id="S9", text="Loans up to Rs 2,00,000 need no collateral.")
    # min_overlap pinned: this test is about numbers, and "rupees" is an extra content word.
    assert check_sentence("Loans up to two lakh rupees need no collateral [S9].",
                          {"S9": src}, min_overlap=0.5)[0] == "supported"  # fmt: skip


# ----------------------------------------------------------------------------- sentences


def texts(answer: str) -> list[str]:
    return [s.text for s in split_sentences(answer)]


def test_splitter_protects_abbreviations_and_decimals() -> None:
    answer = "Fees are Rs. 200, e.g. for NEFT. The rate is 8.5% per year. Ask Dr. Rao. Done now."
    assert texts(answer) == [
        "Fees are Rs. 200, e.g. for NEFT.",
        "The rate is 8.5% per year.",
        "Ask Dr. Rao.",
        "Done now.",
    ]


def test_splitter_keeps_citations_with_their_sentence() -> None:
    assert texts("Disputes take 60 days [S2]. Refunds take 10 days. [S2] Fine.") == [
        "Disputes take 60 days [S2].",
        "Refunds take 10 days. [S2]",
        "Fine.",
    ]


def test_splitter_treats_bullets_and_lines_as_sentences() -> None:
    answer = "Summary:\n- Fee is Rs 200 [S1]\n* Balance is Rs 5,000 [S1]\n2. Window is 60 days [S2]"
    assert texts(answer) == [
        "Summary:",
        "Fee is Rs 200 [S1]",
        "Balance is Rs 5,000 [S1]",
        "Window is 60 days [S2]",
    ]


def test_splitter_does_not_split_before_lowercase() -> None:
    assert texts("See section 4. it applies to all.") == ["See section 4. it applies to all."]


def test_pii_placeholders_are_content_not_citations() -> None:
    assert citations("Write to [EMAIL_1] or [PHONE_IN_2] [S1].") == ["S1"]
    result = check_grounding("We will write to [EMAIL_1] about the fee of Rs 200 [S1].", [S1])
    assert result.sentences[0].outcome != Reason.GROUNDING_FABRICATED
    # A sentence whose only bracket is a placeholder is uncited, not "supported".
    assert check_sentence("The fee of Rs 200 goes to [EMAIL_1].", SOURCES)[0] == (
        Reason.GROUNDING_UNCITED
    )


def test_source_ids_cannot_look_like_placeholders() -> None:
    with pytest.raises(ValueError, match="placeholder"):
        SourceDoc(id="EMAIL_1", text="x")
    assert SourceDoc(id="doc_1", text="x").id == "doc_1"  # lower case is fine


def test_citation_formats() -> None:
    assert citations("x [S1, S2] y [S3][S1] z [ S4 ; S5 ]") == ["S1", "S2", "S3", "S4", "S5"]


# ----------------------------------------------------------------------------- classes


@pytest.mark.parametrize(
    ("sentence", "kind"),
    [
        ("I could not find this in the provided documents.", "abstention"),
        ("The documents do not mention gold loans.", "abstention"),
        ("I don't have enough information to answer that.", "abstention"),
        ("दस्तावेज़ों में नहीं है।", "abstention"),
        ("What else would you like to know?", "non_factual"),
        ("Yes, it does.", "non_factual"),
        ("I hope this helps.", "non_factual"),
        ("Let me know if you have other questions.", "non_factual"),
        ("Thank you for asking about your account.", "non_factual"),
        ("Prepayment charges do not apply to floating-rate loans.", "factual"),
        # SPEC decision 7: a number or a citation always makes a sentence factual.
        ("Did you know the fee is Rs 900?", "factual"),
        ("Fee: Rs 900.", "factual"),
        ("Please note that every loan is free [S3].", "factual"),
        ("I could not find the rate, but the fee is Rs 900.", "factual"),
        # A courtesy phrase must be the whole sentence to be skipped.
        ("Please note that all home loans carry no prepayment charge.", "factual"),
    ],
)
def test_sentence_classes(sentence, kind) -> None:
    assert classify(sentence) == kind


def test_answer_without_claims_passes_as_no_claims() -> None:
    result = check_grounding("I hope this helps. Let me know.", [S1])
    assert result.action is Action.ALLOW and result.outcome == "no_claims"


def test_cited_abstention_is_not_a_correct_abstention() -> None:
    result = check_grounding("The documents do not mention gold loans [S1].", [S1])
    assert result.outcome != "correct_abstention"


def test_question_with_a_fabricated_number_is_rejected() -> None:
    result = check_grounding("Did you know the fee is Rs 900?", [S1])
    assert result.action is Action.BLOCK
    assert result.reasons == [Reason.GROUNDING_UNCITED]


# ----------------------------------------------------------------------------- checks


def test_check_order_uncited_before_everything() -> None:
    assert check_sentence("Every fee is Rs 999.", SOURCES)[0] == Reason.GROUNDING_UNCITED


def test_check_order_fabricated_before_numbers() -> None:
    assert check_sentence("Every fee is Rs 999 [S1, S9].", SOURCES)[0] == (
        Reason.GROUNDING_FABRICATED
    )


def test_numbers_are_checked_against_cited_sources_only() -> None:
    # 60 is in S2, but the sentence cites S1.
    assert check_sentence("Disputes are allowed for 60 days [S1].", SOURCES)[0] == (
        Reason.GROUNDING_NUMBER
    )
    assert check_sentence("Disputes are raised within 60 days [S2].", SOURCES)[0] == "supported"


def test_multiple_citations_pool_their_sources() -> None:
    sentence = "The minimum balance is Rs 5,000 and disputes need 60 days [S1][S2]."
    assert check_sentence(sentence, SOURCES)[0] == "supported"


def test_quotes_must_appear_verbatim() -> None:
    ok = 'The bank says "Prepayment charges are  NOT applicable" on floating loans [S3].'
    assert check_sentence(ok, SOURCES)[0] == "supported"
    curly = "The bank says “prepayment charges are always waived” on loans [S3]."
    assert check_sentence(curly, SOURCES)[0] == Reason.GROUNDING_QUOTE
    short = 'Loans are "floating-rate" here [S3].'  # under 3 words: a term, not a quote
    assert check_sentence(short, SOURCES)[0] != Reason.GROUNDING_QUOTE


def test_quote_stitched_across_two_sources_is_rejected() -> None:
    a = SourceDoc(id="A", text="Fees are waived for senior citizens")
    b = SourceDoc(id="B", text="on all premium accounts this year.")
    stitched = 'The bank says "waived for senior citizens on all premium" accounts [A][B].'
    assert check_sentence(stitched, {"A": a, "B": b})[0] == Reason.GROUNDING_QUOTE


def test_long_answers_stay_fast() -> None:
    """Sources are analysed once per answer, not once per sentence (was 8 s for this)."""
    import time

    source = SourceDoc(id="S1", text="The fee is Rs 5,000 for every account. " * 200)
    answer = "The fee is Rs 5,000 for every account [S1]. " * 2_000
    start = time.perf_counter()
    result = check_grounding(answer, [source])
    assert result.action is Action.ALLOW
    assert time.perf_counter() - start < 3.0


def test_quantifier_mismatch_and_zero_width_hiding() -> None:
    assert check_sentence("Prepayment is free on all loans [S3].", SOURCES)[0] == (
        Reason.GROUNDING_QUANTIFIER
    )
    hidden = "Prepayment is free on ev​ery loan [S3]."
    assert check_sentence(hidden, SOURCES)[0] == Reason.GROUNDING_QUANTIFIER
    src = SourceDoc(id="S8", text="All branches are closed on Sundays.")
    # min_overlap pinned: "close" vs "closed" is a paraphrase the stricter default rejects.
    assert check_sentence("All branches close on Sundays [S8].", {"S8": src},
                          min_overlap=0.5)[0] == "supported"  # fmt: skip


def test_low_overlap_uses_min_overlap() -> None:
    sentence = "Gold loans carry attractive festive interest discounts [S3]."
    outcome, overlap = check_sentence(sentence, SOURCES)
    assert outcome == Reason.GROUNDING_LOW_OVERLAP and overlap is not None and overlap < 0.5
    assert check_sentence(sentence, SOURCES, min_overlap=0.0)[0] == "supported"


def test_content_words_strip_plurals_stopwords_and_numbers() -> None:
    assert content_words("The loans and the 25 years of charges") == {"loan", "year", "charge"}


# ----------------------------------------------------------------------------- modes


def test_reject_mode_lists_every_reason_and_logs_indexes_only() -> None:
    answer = "Balance is Rs 5,000 [S1]. Disputes take 90 days [S2]. Gold loans are also available."
    result = check_grounding(answer, [S1, S2])
    assert result.action is Action.BLOCK and result.outcome == "rejected"
    assert result.reasons == [Reason.GROUNDING_NUMBER, Reason.GROUNDING_UNCITED]
    outcomes = result.sentence_outcomes()
    assert [o["index"] for o in outcomes] == [0, 1, 2]
    assert outcomes[0]["outcome"] == "supported"
    assert "Disputes" not in json.dumps(outcomes)


def test_strip_mode_removes_failing_sentences_and_keeps_layout() -> None:
    answer = (
        "Balance is Rs 5,000 [S1]. Disputes take 90 days [S2].\n- Refunds take 10 working days [S2]"
    )
    result = check_grounding(answer, [S1, S2], mode="strip", min_overlap=0.5)  # about layout
    assert result.action is Action.REDACT and result.outcome == "stripped"
    assert result.reasons == [Reason.GROUNDING_NUMBER]
    assert result.text == "Balance is Rs 5,000 [S1].\nRefunds take 10 working days [S2]"


def test_strip_mode_with_nothing_factual_left_becomes_the_abstention() -> None:
    result = check_grounding("Gold loans are at 9% [S3]. I hope this helps.", [S3], mode="strip")
    assert result.action is Action.REDACT and result.outcome == "empty_after_strip"
    assert result.text == ABSTENTION
    assert result.reasons == [Reason.GROUNDING_EMPTY_AFTER_STRIP, Reason.GROUNDING_NUMBER]


def test_strip_mode_leaves_a_good_answer_alone() -> None:
    result = check_grounding(ROWS["gr001"]["answer"], [S1], mode="strip")
    assert result.action is Action.ALLOW and result.text is None


# ----------------------------------------------------------------------------- guard


async def test_guard_rejects_with_reasons_and_no_text_in_the_dump() -> None:
    row = ROWS["gr005"]
    ctx = GuardContext(request_id="r", output=row["answer"], sources=sources_of(row),
                       policy=load_policy("default"))  # fmt: skip
    v = await run_guard(GroundingGuard(), ctx, timeout_s=5)
    assert v.action is Action.BLOCK and v.reasons == [Reason.GROUNDING_NUMBER]
    assert v.detail["outcome"] == "rejected" and v.detail["mode"] == "reject"
    assert v.scores["factual"] == 1.0 and v.scores["failed"] == 1.0
    dumped = v.model_dump_json()
    assert "10,000" not in dumped and "minimum" not in dumped


async def test_guard_takes_mode_and_overlap_from_policy() -> None:
    policy = load_policy("default").model_copy(deep=True)
    policy.grounding.mode = "strip"
    ctx = GuardContext(request_id="r", output="Gold loans are at 9% [S3].", sources=[S3],
                       policy=policy)  # fmt: skip
    v = await run_guard(GroundingGuard(), ctx, timeout_s=5)
    assert v.action is Action.REDACT and v.modified_text == ABSTENTION

    policy.grounding.mode = "reject"
    policy.grounding.min_overlap = 0.0
    ctx.output = "Gold loans carry attractive festive interest discounts [S3]."
    assert (await run_guard(GroundingGuard(), ctx, timeout_s=5)).action is Action.ALLOW


async def test_guard_disabled_by_policy_allows() -> None:
    policy = load_policy("default").model_copy(deep=True)
    policy.grounding.enabled = False
    ctx = GuardContext(request_id="r", output="Made up claim.", policy=policy)
    v = await run_guard(GroundingGuard(), ctx, timeout_s=5)
    assert v.action is Action.ALLOW and v.detail["outcome"] == "disabled"


async def test_guard_without_an_answer_fails_closed() -> None:
    v = await run_guard(GroundingGuard(), GuardContext(request_id="r"), timeout_s=5)
    assert v.action is Action.BLOCK and v.reasons == [Reason.GUARD_ERROR]


async def test_guard_without_policy_defaults_to_reject() -> None:
    ctx = GuardContext(request_id="r", output=ROWS["gr008"]["answer"], sources=[S1])
    v = await run_guard(GroundingGuard(), ctx, timeout_s=5)
    assert v.action is Action.BLOCK and v.reasons == [Reason.GROUNDING_UNCITED]


# ----------------------------------------------------------------------------- properties


@given(st.text(max_size=300))
def test_number_parser_never_crashes(text) -> None:
    assert all(isinstance(v, float) for v in parse_numbers(text))


@given(st.text(max_size=300))
def test_grounding_never_crashes_and_strip_never_grows_claims(text) -> None:
    reject = check_grounding(text, [S1, S2])
    strip = check_grounding(text, [S1, S2], mode="strip")
    assert reject.action in {Action.ALLOW, Action.BLOCK}
    assert strip.action in {Action.ALLOW, Action.REDACT}
    assert (reject.action is Action.ALLOW) == (strip.action is Action.ALLOW)


@given(st.text(min_size=1, max_size=80))
def test_any_answer_citing_a_missing_source_is_never_allowed(text) -> None:
    result = check_grounding(f"{text} [S9].", [S1])
    assert result.action is Action.BLOCK
