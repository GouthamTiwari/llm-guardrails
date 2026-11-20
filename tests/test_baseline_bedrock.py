"""Bedrock Guardrails baseline with stubbed boto3 (botocore Stubber checks every request and
response against the real API shapes; nothing reaches AWS)."""

import datetime as dt

import boto3
import pytest
from botocore.stub import ANY, Stubber

from llm_guardrails.eval import baseline_bedrock as bb

NOW = dt.datetime(2026, 11, 21, tzinfo=dt.UTC)


def clients():
    kw = {"region_name": "ap-south-1", "aws_access_key_id": "testing",
          "aws_secret_access_key": "testing"}  # fmt: skip
    return boto3.client("bedrock", **kw), boto3.client("bedrock-runtime", **kw)


def guardrail(status="READY", gid="g123"):
    return {"name": bb.NAME_PREFIX + "abcd1234", "guardrailId": gid,
            "guardrailArn": f"arn:aws:bedrock:ap-south-1:123456789012:guardrail/{gid}",
            "version": "DRAFT", "status": status, "createdAt": NOW, "updatedAt": NOW,
            "blockedInputMessaging": "x", "blockedOutputsMessaging": "x"}  # fmt: skip


def summary(gid="g123"):
    g = guardrail(gid=gid)
    return {
        k: g[k]
        for k in (
            "guardrailId",
            "guardrailArn",
            "status",
            "name",
            "version",
            "createdAt",
            "updatedAt",
        )
    } | {"id": gid, "arn": g["guardrailArn"]}


def stub_create(sc):
    sc.add_response("create_guardrail", {"guardrailId": "g123", "guardrailArn": guardrail()[
        "guardrailArn"], "version": "DRAFT", "createdAt": NOW}, {**{k: ANY for k in (
            "name", "description", "contentPolicyConfig", "sensitiveInformationPolicyConfig",
            "blockedInputMessaging", "blockedOutputsMessaging", "tags")}})  # fmt: skip
    sc.add_response("get_guardrail", guardrail(), {"guardrailIdentifier": "g123"})


def stub_delete(sc, left_over=False):
    sc.add_response("delete_guardrail", {}, {"guardrailIdentifier": "g123"})
    page = {"guardrails": [_listed()] if left_over else []}
    sc.add_response("list_guardrails", page, {})


def _listed():
    g = guardrail()
    return {"id": "g123", "arn": g["guardrailArn"], "status": "DELETING", "name": g["name"],
            "version": "DRAFT", "createdAt": NOW, "updatedAt": NOW}  # fmt: skip


USAGE = {"topicPolicyUnits": 0, "contentPolicyUnits": 1, "wordPolicyUnits": 0,
         "sensitiveInformationPolicyUnits": 1, "sensitiveInformationPolicyFreeUnits": 0,
         "contextualGroundingPolicyUnits": 0}  # fmt: skip
ATTACK_FILTER = {"type": "PROMPT_ATTACK", "confidence": "HIGH", "action": "BLOCKED",
                 "detected": True}  # fmt: skip
APPLY_OK = {"usage": USAGE, "action": "GUARDRAIL_INTERVENED", "outputs": [],
            "assessments": [{"contentPolicy": {"filters": [ATTACK_FILTER]}}]}  # fmt: skip


def test_delete_runs_even_when_an_apply_call_raises() -> None:
    control, runtime = clients()
    with Stubber(control) as sc, Stubber(runtime) as sr:
        stub_create(sc)
        sr.add_response("apply_guardrail", APPLY_OK, {
            "guardrailIdentifier": "g123", "guardrailVersion": "DRAFT", "source": "INPUT",
            "content": ANY})  # fmt: skip
        sr.add_client_error("apply_guardrail", "ThrottlingException", http_status_code=429)
        stub_delete(sc)
        with pytest.raises(Exception, match="ThrottlingException"):
            bb.run_live(control, runtime, {"a": "first text", "b": "second text"},
                        sleep=lambda s: None)  # fmt: skip
        sc.assert_no_pending_responses()  # delete_guardrail and list_guardrails were called


def test_full_run_records_responses_and_verifies_deletion() -> None:
    control, runtime = clients()
    with Stubber(control) as sc, Stubber(runtime) as sr:
        stub_create(sc)
        for _ in range(2):
            sr.add_response("apply_guardrail", APPLY_OK, {
                "guardrailIdentifier": "g123", "guardrailVersion": "DRAFT",
                "source": "INPUT", "content": ANY})  # fmt: skip
        stub_delete(sc)
        out = bb.run_live(control, runtime, {"a": "x", "b": "y"}, sleep=lambda s: None)
    assert out["deleted_and_verified"] is True
    assert set(out["responses"]) == {"a", "b"} and "ResponseMetadata" not in out["responses"]["a"]


def test_deletion_not_verified_is_reported(monkeypatch) -> None:
    monkeypatch.setattr(bb, "DELETE_TIMEOUT_S", 0)  # no second look
    control, runtime = clients()
    with Stubber(control) as sc, Stubber(runtime):
        stub_create(sc)
        sc.add_response("delete_guardrail", {}, {"guardrailIdentifier": "g123"})
        out = bb.run_live(control, runtime, {}, sleep=lambda s: None)
    assert out["deleted_and_verified"] is False


def test_permission_denied_on_create_is_reported_not_raised(tmp_path, monkeypatch) -> None:
    control, runtime = clients()
    with Stubber(control) as sc:
        sc.add_client_error("create_guardrail", "AccessDeniedException", http_status_code=403)
        with pytest.raises(bb.BaselineUnavailable, match="AccessDeniedException"):
            bb.run_live(control, runtime, {"a": "x"}, sleep=lambda s: None)


class _Settings:
    live_budget_usd = 5.0


def test_unavailable_run_still_writes_a_report(tmp_path, monkeypatch) -> None:
    control, runtime = clients()
    monkeypatch.setattr(bb, "eval_texts", lambda suites=(): {"a": "x"})

    class _Ledger:
        def __init__(self, **kw):
            pass

        def check(self, estimate):
            pass

    monkeypatch.setattr("llm_guardrails.spend.SpendLedger", _Ledger)
    with Stubber(control) as sc:
        sc.add_client_error("create_guardrail", "AccessDeniedException", http_status_code=403)
        r = bb.run(live=True, settings=_Settings(), reports_dir=tmp_path,
                   clients=(control, runtime))  # fmt: skip
    assert r["status"] == {"available": False, "reason": "AccessDeniedException"}
    md = (tmp_path / "baseline_bedrock_guardrails.md").read_text("utf-8")
    assert "Not available: AccessDeniedException" in md


def test_config_uses_the_overlapping_managed_types_only() -> None:
    cfg = bb.guardrail_config(bb.NAME_PREFIX + "x")
    types = {e["type"] for e in cfg["sensitiveInformationPolicyConfig"]["piiEntitiesConfig"]}
    assert types == {"EMAIL", "PHONE", "CREDIT_DEBIT_CARD_NUMBER"}
    attack = cfg["contentPolicyConfig"]["filtersConfig"][0]
    assert attack == {"type": "PROMPT_ATTACK", "inputStrength": "HIGH", "outputStrength": "NONE"}


def test_scoring_helpers() -> None:
    assert bb.prompt_attack_detected(APPLY_OK)
    assert not bb.prompt_attack_detected({"assessments": []})
    resp = {"assessments": [{"sensitiveInformationPolicy": {"piiEntities": [
        {"type": "PHONE", "match": "+91 98765 43210", "action": "ANONYMIZED"},
        {"type": "EMAIL", "match": "Asha@Example.com", "action": "ANONYMIZED"}]}}]}  # fmt: skip
    found = bb.pii_found(resp)
    assert ("PHONE", "9876543210") in found and ("EMAIL", "asha@example.com") in found
    assert bb.units_cost(APPLY_OK) == pytest.approx((0.15 + 0.10) / 1000)


def test_pii_filter_is_evaluated_on_input() -> None:
    """The first live run set only the legacy `action` field and the PII filter never ran
    on input (0 units billed). The input fields must be explicit."""
    cfg = bb.guardrail_config(bb.NAME_PREFIX + "x")
    for entity in cfg["sensitiveInformationPolicyConfig"]["piiEntitiesConfig"]:
        assert entity["inputEnabled"] is True and entity["inputAction"] == "ANONYMIZE"
        assert entity["outputEnabled"] is False


def test_a_second_run_replaces_its_rows_and_keeps_both_headers(tmp_path) -> None:
    path = tmp_path / "rec.jsonl"
    first = {
        "guardrail_name": "a",
        "deleted_and_verified": True,
        "pii_input_enabled": False,
        "responses": {"k1": {"usage": USAGE, "v": 1}, "k2": {"usage": USAGE, "v": 1}},
    }
    second = {"guardrail_name": "b", "deleted_and_verified": True,
              "responses": {"k2": {"usage": USAGE, "v": 2}}}  # fmt: skip
    bb.save_recordings(first, path)
    out = bb.save_recordings(second, path)
    assert [r["guardrail_name"] for r in out["runs"]] == ["a", "b"]
    assert out["runs"][0]["pii_input_enabled"] is False and out["runs"][1]["pii_input_enabled"]
    assert out["responses"]["k1"]["v"] == 1 and out["responses"]["k2"]["v"] == 2
    assert out["deleted_and_verified"] is True


def test_scrub_removes_the_account_identifying_arn() -> None:
    resp = {
        "assessments": [
            {
                "appliedGuardrailDetails": {
                    "guardrailArn": "arn:aws:bedrock:ap-south-1:123456789012:guardrail/g"
                },
                "x": 1,
            }
        ]
    }
    clean = bb.scrub(resp)
    assert "123456789012" not in str(clean) and clean["assessments"] == [{"x": 1}]


def test_committed_recordings_contain_no_account_id() -> None:
    text = bb.RECORDINGS.read_text(encoding="utf-8")
    assert "arn:aws" not in text and "appliedGuardrailDetails" not in text
