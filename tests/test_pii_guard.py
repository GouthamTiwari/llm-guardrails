"""PII guard actions, placeholders, restore, and proof that values never leave memory."""

import json
import logging
from pathlib import Path

import pytest

from llm_guardrails.detectors.checksums import luhn_check_digit, verhoeff_check_digit
from llm_guardrails.guards import run_guard
from llm_guardrails.guards.pii import (
    PiiGuard,
    Redactor,
    apply_pii_policy,
    block_message,
    escape_placeholders,
    restore,
)
from llm_guardrails.logging_utils import audit_log, set_hash_salt
from llm_guardrails.policy_config import load_policy
from llm_guardrails.types import Action, GuardContext, PipelineReport

SEED = Path(__file__).parents[1] / "data" / "seed" / "pii_seed.jsonl"
ROWS = [json.loads(line) for line in SEED.read_text(encoding="utf-8").splitlines()]
DEFAULT = load_policy("default")
BANKING = load_policy("banking_tax")

AADHAAR = "23456789012" + verhoeff_check_digit("23456789012")
CARD = "411111111111111" + luhn_check_digit("411111111111111")


@pytest.fixture(autouse=True)
def _salt():
    set_hash_salt("test-salt")


async def run(text: str, policy=DEFAULT, pii_map=None):
    ctx = GuardContext(request_id="r1", text=text, policy=policy, pii_map=pii_map or {})
    return await run_guard(PiiGuard(), ctx, timeout_s=5)


async def test_redacts_with_typed_placeholders() -> None:
    v = await run(f"My Aadhaar is {AADHAAR} and email ravi.kumar@example.com")
    assert v.action is Action.REDACT
    assert v.modified_text == "My Aadhaar is [AADHAAR_1] and email [EMAIL_1]"
    assert v.reasons == ["pii.redact.AADHAAR", "pii.redact.EMAIL"]
    assert v.detail["pii_map"] == {"[AADHAAR_1]": AADHAAR, "[EMAIL_1]": "ravi.kumar@example.com"}
    assert {e.type: e.count for e in v.entities} == {"AADHAAR": 1, "EMAIL": 1}


async def test_same_value_same_placeholder_even_with_different_spacing() -> None:
    spaced = f"{AADHAAR[:4]} {AADHAAR[4:8]} {AADHAAR[8:]}"
    v = await run(f"aadhaar {AADHAAR}, again: aadhaar {spaced}, mail a@example.com b@example.com")
    assert v.modified_text == (
        "aadhaar [AADHAAR_1], again: aadhaar [AADHAAR_1], mail [EMAIL_1] [EMAIL_2]"
    )
    aadhaar = next(e for e in v.entities if e.type == "AADHAAR")
    assert aadhaar.count == 2 and len(aadhaar.hash_prefixes) == 1


async def test_card_blocks_with_value_free_message() -> None:
    v = await run(f"Charge my card {CARD} and mail a@example.com")
    assert v.action is Action.BLOCK
    assert v.reasons == ["pii.block.CREDIT_CARD"]
    assert v.modified_text is None
    assert v.detail["user_message"] == block_message(["CREDIT_CARD"])
    assert CARD not in v.detail["user_message"]


async def test_allowed_types_pass_but_are_counted() -> None:
    v = await run("IFSC SBIN0001234 and GSTIN 27AAPFU0939F1ZV")
    assert v.action is Action.ALLOW
    assert v.modified_text == "IFSC SBIN0001234 and GSTIN 27AAPFU0939F1ZV"
    assert {e.type for e in v.entities} == {"GSTIN", "IFSC"}


async def test_numbering_continues_from_existing_request_map() -> None:
    first = await run("mail a@example.com")
    second = await run("mail b@example.com and a@example.com", pii_map=first.detail["pii_map"])
    assert second.modified_text == "mail [EMAIL_2] and [EMAIL_1]"


async def test_user_typed_placeholder_is_defused() -> None:
    v = await run("Please send it to [EMAIL_1] and to c@example.com")
    assert v.modified_text == "Please send it to (EMAIL_1) and to [EMAIL_1]"
    # Restore can only ever fill the placeholder the guard itself issued.
    restored = restore(v.modified_text, v.detail["pii_map"], ["EMAIL"])
    assert restored == "Please send it to (EMAIL_1) and to c@example.com"


def test_escape_placeholders() -> None:
    assert escape_placeholders("x [AADHAAR_12] [not_one] [EMAIL_]") == (
        "x (AADHAAR_12) [not_one] [EMAIL_]"
    )


def test_restore_only_listed_types_and_issued_placeholders() -> None:
    pii_map = {"[EMAIL_1]": "a@example.com", "[PHONE_IN_1]": "9876543210"}
    text = "Mail [EMAIL_1], call [PHONE_IN_1], also [EMAIL_9]."
    assert restore(text, pii_map, []) == text
    assert restore(text, pii_map, ["EMAIL"]) == (
        "Mail a@example.com, call [PHONE_IN_1], also [EMAIL_9]."
    )
    assert restore(text, pii_map, BANKING.pii.restore) == (
        "Mail a@example.com, call 9876543210, also [EMAIL_9]."
    )


def test_redactor_rebuilds_from_map() -> None:
    r = Redactor.from_map({"[EMAIL_3]": "Ravi@Example.com", "junk": "x"})
    assert r._counters == {"EMAIL": 3}
    out = apply_pii_policy("write to ravi@example.com or z@example.com", DEFAULT, "input", r)
    assert out.text == "write to [EMAIL_3] or [EMAIL_4]"


def test_output_stage_uses_output_actions() -> None:
    policy = DEFAULT.model_copy(deep=True)
    policy.pii.entities["EMAIL"].output = "block"
    out = apply_pii_policy("contact a@example.com", policy, "output", Redactor())
    assert out.action is Action.BLOCK and out.reasons == ["pii.block.EMAIL"]
    out = apply_pii_policy("contact a@example.com", policy, "input", Redactor())
    assert out.action is Action.REDACT


async def test_seed_rows_get_expected_actions() -> None:
    for row in ROWS:
        v = await run(row["text"])
        types = {e["type"] for e in row["entities"]}
        if "CREDIT_CARD" in types:
            assert v.action is Action.BLOCK, row["id"]
        elif types - {"GSTIN", "IFSC"}:
            assert v.action is Action.REDACT, row["id"]
            for e in row["entities"]:
                if e["type"] not in {"GSTIN", "IFSC"}:
                    assert e["value"] not in v.modified_text, row["id"]
        else:
            assert v.action is Action.ALLOW, row["id"]
            assert v.modified_text == row["text"], row["id"]


async def test_no_pii_value_reaches_any_serialised_form(caplog) -> None:
    """Dump every verdict, report and audit log line for every seed row and grep for values."""
    caplog.set_level(logging.DEBUG)
    dumped: list[str] = []
    for row in ROWS:
        v = await run(row["text"])
        report = PipelineReport(
            request_id="r1",
            decision="block" if v.action is Action.BLOCK else "allow",
            stage="input",
            verdicts=[v],
            latency_ms=1.0,
            cost_usd=0.0,
            tokens_in=0,
            tokens_out=0,
            policy_name="default",
            version="0.1.0",
        )
        dumped += [v.model_dump_json(), repr(v), report.model_dump_json(), str(v.model_dump())]
        dumped.append(json.dumps(audit_log(report, "check-input")))
    dumped.append(caplog.text)
    blob = "\n".join(dumped)
    values = {e["value"] for r in ROWS for e in r["entities"]}
    values |= {e["value"].replace(" ", "") for r in ROWS for e in r["entities"]}
    leaked = [val for val in values if val in blob]
    assert leaked == [], f"{len(leaked)} value(s) leaked"
