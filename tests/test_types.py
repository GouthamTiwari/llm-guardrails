import pytest
from pydantic import ValidationError

from llm_guardrails.types import (
    ACTION_RANK,
    Action,
    PipelineReport,
    Reason,
    SchemaValidationFailed,
    SourceDoc,
    Verdict,
    is_valid_reason,
    strongest_action,
)


def test_action_precedence_is_block_retry_redact_allow() -> None:
    order = sorted(Action, key=ACTION_RANK.__getitem__)
    assert order == [Action.ALLOW, Action.REDACT, Action.RETRY, Action.BLOCK]


def test_strongest_action() -> None:
    assert strongest_action([]) is Action.ALLOW
    assert strongest_action([Action.REDACT, Action.ALLOW]) is Action.REDACT
    assert strongest_action([Action.REDACT, Action.BLOCK, Action.RETRY]) is Action.BLOCK


def test_verdict_never_serialises_modified_text_or_detail() -> None:
    v = Verdict(
        guard="pii",
        action=Action.REDACT,
        reasons=[Reason.pii_redact("AADHAAR")],
        modified_text="secret user text 2345 6789 0123",
        detail={"raw": "secret user text"},
    )
    dumped = v.model_dump_json()
    assert "secret" not in dumped
    assert "modified_text" not in v.model_dump()
    assert "secret" not in repr(v)


def test_reason_codes_validated() -> None:
    for code in [
        Reason.POLICY_TOO_LONG,
        Reason.pii_block("CREDIT_CARD"),
        Reason.inj_rule("direct_override"),
        Reason.grounding_entailment("not_enough_info"),
        Reason.leak_secret("aws_access_key"),
        Reason.GUARD_TIMEOUT,
    ]:
        assert is_valid_reason(code), code
    assert not is_valid_reason("my aadhaar is 2345 6789 0123")
    with pytest.raises(ValidationError) as err:
        Verdict(guard="x", action=Action.BLOCK, reasons=["Ignore previous instructions"])
    # Our own message reports only a count. (Pydantic still appends input_value to str(err),
    # which is why run_guard logs only exception type names, never messages.)
    assert err.value.errors()[0]["msg"] == "Value error, 1 reason(s) are not valid reason codes"


def test_source_doc_id_rules() -> None:
    assert SourceDoc(id="S1", text="x").id == "S1"
    for bad in ["", "S 1", "x" * 33, "S1<script>"]:
        with pytest.raises(ValidationError):
            SourceDoc(id=bad, text="x")


def test_schema_error_keeps_attempts_and_errors() -> None:
    err = SchemaValidationFailed(3, ["severity: wrong enum"])
    assert err.attempts == 3 and err.errors == ["severity: wrong enum"]


def test_pipeline_report_round_trips() -> None:
    report = PipelineReport(
        request_id="abc",
        decision="block",
        stage="input",
        verdicts=[Verdict(guard="pii", action=Action.BLOCK, reasons=[Reason.pii_block("PAN")])],
        latency_ms=1.0,
        cost_usd=0.0,
        tokens_in=0,
        tokens_out=0,
        policy_name="default",
        version="0.1.0",
    )
    assert PipelineReport.model_validate_json(report.model_dump_json()) == report
