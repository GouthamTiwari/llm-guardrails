import json
import logging

import pytest

import llm_guardrails.logging_utils as lu
from llm_guardrails.types import Action, EntitySummary, PipelineReport, Verdict


@pytest.fixture(autouse=True)
def _reset_salt():
    lu.set_hash_salt("")
    lu._warned_random_salt = False
    yield
    lu.set_hash_salt("")


def report(*verdicts: Verdict) -> PipelineReport:
    return PipelineReport(
        request_id="0f3c",
        decision="allow",
        stage="input",
        verdicts=list(verdicts),
        latency_ms=1.23456,
        cost_usd=0.000012345,
        tokens_in=5,
        tokens_out=7,
        policy_name="default",
        version="0.1.0",
    )


def test_hash_is_salted_short_and_stable() -> None:
    lu.set_hash_salt("salt-a")
    a1, a2 = lu.hash_value("9876543210"), lu.hash_value("9876543210")
    lu.set_hash_salt("salt-b")
    b = lu.hash_value("9876543210")
    assert a1 == a2 and a1 != b
    assert len(a1) == 8 and all(c in "0123456789abcdef" for c in a1)


def test_empty_salt_uses_random_salt_and_warns_once(monkeypatch, caplog) -> None:
    monkeypatch.delenv("LOG_HASH_SALT", raising=False)
    caplog.set_level(logging.WARNING, logger="llm_guardrails")
    lu.hash_value("x")
    lu.hash_value("y")
    assert caplog.text.count("LOG_HASH_SALT is empty") == 1


def test_env_salt_used_when_not_set(monkeypatch) -> None:
    monkeypatch.setenv("LOG_HASH_SALT", "from-env")
    h = lu.hash_value("v")
    lu.set_hash_salt("from-env")
    assert lu.hash_value("v") == h


def test_summarize_entities_counts_and_dedupes() -> None:
    lu.set_hash_salt("s")
    out = lu.summarize_entities([("EMAIL", "a@x.com"), ("EMAIL", "a@x.com"), ("PAN", "P")])
    assert [(e.type, e.count, len(e.hash_prefixes)) for e in out] == [
        ("EMAIL", 2, 1),
        ("PAN", 1, 1),
    ]


def test_safe_record_has_exactly_the_spec_fields() -> None:
    v = Verdict(
        guard="pii",
        action=Action.REDACT,
        reasons=["pii.redact.EMAIL"],
        scores={"entities": 1, "Bad Key with text": 2.0},
        entities=[EntitySummary(type="EMAIL", count=1, hash_prefixes=["abcdef12", "not-hex!"])],
        modified_text="mail [EMAIL_1]",
        detail={"pii_map": {"[EMAIL_1]": "a@example.com"}},
    )
    rec = lu.safe_record(report(v), "guarded-chat")
    assert list(rec) == [
        "ts", "request_id", "route", "decision", "stage", "policy", "version", "latency_ms",
        "tokens_in", "tokens_out", "cost_usd", "guards", "entities",
    ]  # fmt: skip
    assert list(rec["guards"][0]) == [
        "guard", "action", "reasons", "scores", "latency_ms", "cost_usd", "error_kind",
    ]  # fmt: skip
    assert rec["guards"][0]["scores"] == {"entities": 1.0}
    assert rec["entities"] == [{"type": "EMAIL", "count": 1, "hash_prefixes": ["abcdef12"]}]
    blob = json.dumps(rec)
    assert "a@example.com" not in blob and "[EMAIL_1]" not in blob


def test_entities_merged_across_verdicts() -> None:
    v1 = Verdict(
        guard="pii",
        action=Action.ALLOW,
        entities=[EntitySummary(type="PAN", count=1, hash_prefixes=["aaaaaaaa"])],
    )
    v2 = Verdict(
        guard="leak",
        action=Action.ALLOW,
        entities=[EntitySummary(type="PAN", count=2, hash_prefixes=["aaaaaaaa", "bbbbbbbb"])],
    )
    rec = lu.safe_record(report(v1, v2), "check-output")
    assert rec["entities"] == [
        {"type": "PAN", "count": 3, "hash_prefixes": ["aaaaaaaa", "bbbbbbbb"]}
    ]  # fmt: skip


def test_audit_log_writes_one_json_line(caplog) -> None:
    caplog.set_level(logging.INFO, logger="llm_guardrails.audit")
    lu.audit_log(report(Verdict(guard="pii", action=Action.ALLOW)), "check-input")
    lines = [r.getMessage() for r in caplog.records if r.name == "llm_guardrails.audit"]
    assert len(lines) == 1
    assert json.loads(lines[0])["route"] == "check-input"
