"""JSON extraction and strict validation: every trap from spec 10.1-10.2."""

import json

import pytest

from llm_guardrails.guards.schema import extract_json, validate_output
from llm_guardrails.schemas import Refusal, TicketSummary, json_schema

GOOD = {"kind": "ticket", "title": "App crashes at payment", "severity": "high",
        "tags": ["payments", "crash"], "due_days": 2}  # fmt: skip
GOOD_JSON = json.dumps(GOOD)


def test_plain_object() -> None:
    assert extract_json(GOOD_JSON).json_text == GOOD_JSON
    assert not extract_json(GOOD_JSON).prose_stripped


@pytest.mark.parametrize(
    "wrapped",
    [
        f"```json\n{GOOD_JSON}\n```",
        f"```\n{GOOD_JSON}\n```",
        f"Sure! Here is the summary:\n{GOOD_JSON}\nHope this helps.",
        f"Here you go: ```json {GOOD_JSON} ``` done",
    ],
)
def test_fences_and_stray_prose_tolerated(wrapped) -> None:
    out = extract_json(wrapped)
    assert json.loads(out.json_text) == GOOD and out.prose_stripped


def test_braces_inside_strings_do_not_confuse_the_scanner() -> None:
    text = '{"kind": "ticket", "title": "Login {fails} \\" here }", "severity": "low", ' \
           '"tags": [], "due_days": 1}'  # fmt: skip
    assert extract_json(text).json_text == text


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("I cannot help with that.", "no_json"),
        ("", "no_json"),
        (f"{GOOD_JSON}\n{GOOD_JSON}", "multiple_objects"),
        ('{"kind": "ticket", "title": "App cra', "truncated"),
        ('```json\n{"kind": "tic', "truncated"),
    ],
)
def test_extraction_failures(text, kind) -> None:
    out = extract_json(text)
    assert out.json_text is None and out.kind == kind


def test_valid_ticket_and_refusal() -> None:
    v = validate_output(GOOD_JSON, "TicketSummary")
    assert v.ok and isinstance(v.value, TicketSummary) and v.value.due_days == 2
    r = validate_output(
        '{"kind": "refusal", "reason": "empty_input", "message": "No ticket text."}',
        "TicketSummary",
    )
    assert isinstance(r.value, Refusal)


@pytest.mark.parametrize(
    ("change", "kind"),
    [
        ({"due_days": "3"}, "wrong_type"),  # number as a string: strict, no coercion
        ({"due_days": 3.0}, "wrong_type"),  # float where an int is required
        ({"severity": "High"}, "wrong_enum"),  # enum case mismatch
        ({"severity": "urgent"}, "wrong_enum"),
        ({"extra": "field"}, "extra_key"),
        ({"due_days": 45}, "constraint"),
        ({"tags": ["a", "b", "c", "d", "e", "f"]}, "constraint"),  # more than 5 tags
        ({"tags": ["Has Space"]}, "constraint"),
        ({"title": "Hi"}, "constraint"),
        ({"kind": "memo"}, "wrong_enum"),
    ],
)
def test_strict_traps(change, kind) -> None:
    v = validate_output(json.dumps(GOOD | change), "TicketSummary")
    assert not v.ok and kind in v.kinds, (v.errors, v.kinds)


def test_missing_field_and_missing_discriminator() -> None:
    no_title = {k: v for k, v in GOOD.items() if k != "title"}
    assert "missing_field" in validate_output(json.dumps(no_title), "TicketSummary").kinds
    no_kind = {k: v for k, v in GOOD.items() if k != "kind"}
    assert "missing_field" in validate_output(json.dumps(no_kind), "TicketSummary").kinds


def test_duplicate_keys_rejected() -> None:
    text = GOOD_JSON[:-1] + ', "severity": "low"}'
    v = validate_output(text, "TicketSummary")
    assert v.kinds == ["duplicate_key"] and "severity" in v.errors[0]


def test_bad_json_and_truncation_by_stop_reason() -> None:
    assert validate_output("{'single': 'quotes'}", "TicketSummary").kinds == ["invalid_json"]
    cut = validate_output('{"kind": "ticket"', "TicketSummary", stop_reason="max_tokens")
    assert cut.kinds == ["truncated"]


def test_errors_name_fields_but_never_echo_values() -> None:
    secret = "my aadhaar is 2345 6789 0123"
    v = validate_output(json.dumps(GOOD | {"severity": secret}), "TicketSummary")
    assert any(e.startswith("ticket.severity") for e in v.errors)
    assert secret not in " ".join(v.errors)


def test_judge_schemas() -> None:
    ok = '{"label": "attack", "confidence": 0.9, "category": "direct_override", "reason": "x"}'
    assert validate_output(ok, "InjectionJudgement").ok
    assert validate_output(ok.replace("0.9", "1"), "InjectionJudgement").ok  # int is a float
    assert not validate_output(ok.replace("0.9", "1.5"), "InjectionJudgement").ok
    assert validate_output('{"label": "supported", "reason": "r"}', "EntailmentJudgement").ok


def test_json_schema_for_prompts() -> None:
    schema = json_schema("TicketSummary")
    text = json.dumps(schema)
    assert "due_days" in text and "refusal" in text and "additionalProperties" in text


def test_unknown_schema_name() -> None:
    with pytest.raises(KeyError):
        validate_output("{}", "Nope")
