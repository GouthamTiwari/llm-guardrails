"""Bedrock client with a stubbed boto3 (no network), the fake model, and record/replay."""

import boto3
import pytest
from botocore.stub import Stubber

from llm_guardrails.llm import BedrockLLM, FakeLLM, LLMResult, Message, make_runtime_client
from llm_guardrails.recording import RecordingLLM, request_key
from llm_guardrails.settings import Settings
from llm_guardrails.spend import SpendLedger
from llm_guardrails.types import BudgetExceeded, ConfigError, ReplayMissing

NOVA = "global.amazon.nova-2-lite-v1:0"


def stubbed_client():
    client = boto3.client(
        "bedrock-runtime",
        region_name="ap-south-1",
        aws_access_key_id="testing",
        aws_secret_access_key="testing",
    )
    return client, Stubber(client)


def converse_response(text="hello", tokens_in=12, tokens_out=3, stop="end_turn"):
    return {
        "output": {"message": {"role": "assistant", "content": [{"text": text}]}},
        "stopReason": stop,
        "usage": {"inputTokens": tokens_in, "outputTokens": tokens_out,
                  "totalTokens": tokens_in + tokens_out},
        "metrics": {"latencyMs": 5},
    }  # fmt: skip


async def test_bedrock_converse_request_shape_cost_and_ledger(tmp_path) -> None:
    client, stub = stubbed_client()
    stub.add_response(
        "converse",
        converse_response(),
        expected_params={
            "modelId": NOVA,
            "system": [{"text": "be brief"}],
            "messages": [{"role": "user", "content": [{"text": "hi"}]}],
            "inferenceConfig": {"maxTokens": 50, "temperature": 0.2},
        },
    )
    ledger = SpendLedger(tmp_path / "l.jsonl", limit_usd=1)
    llm = BedrockLLM(Settings(), ledger=ledger, client=client)
    with stub:
        result = await llm.converse(
            system="be brief", messages=[Message("user", "hi")], temperature=0.2,
            max_tokens=50, purpose="unit",
        )  # fmt: skip
    assert result.text == "hello" and result.stop_reason == "end_turn"
    assert (result.tokens_in, result.tokens_out) == (12, 3)
    assert result.cost_usd == pytest.approx((12 * 0.35 + 3 * 2.95) / 1e6)
    assert ledger.total() == pytest.approx(result.cost_usd)
    assert "hi" not in (tmp_path / "l.jsonl").read_text()


async def test_bedrock_omits_empty_system_and_reports_truncation(tmp_path) -> None:
    client, stub = stubbed_client()
    stub.add_response(
        "converse",
        converse_response(text='{"a": 1', stop="max_tokens"),
        expected_params={
            "modelId": NOVA,
            "messages": [{"role": "user", "content": [{"text": "json"}]}],
            "inferenceConfig": {"maxTokens": 5, "temperature": 0.0},
        },
    )
    llm = BedrockLLM(Settings(), ledger=SpendLedger(tmp_path / "l", 1), client=client)
    with stub:
        result = await llm.converse(
            system="", messages=[Message("user", "json")], max_tokens=5, purpose="unit"
        )
    assert result.stop_reason == "max_tokens"


async def test_budget_checked_before_calling_aws(tmp_path) -> None:
    client, stub = stubbed_client()  # no responses queued: any call would fail
    ledger = SpendLedger(tmp_path / "l.jsonl", limit_usd=0.0)
    llm = BedrockLLM(Settings(), ledger=ledger, client=client)
    with stub, pytest.raises(BudgetExceeded):
        await llm.converse(system="", messages=[Message("user", "hi")], purpose="unit")
    stub.assert_no_pending_responses()


def test_make_runtime_client_auth_modes(monkeypatch) -> None:
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    with pytest.raises(ConfigError):
        make_runtime_client(Settings())
    iam = Settings(access_key_id="AKIDEXAMPLE", secret_access_key="secret")
    assert make_runtime_client(iam).meta.region_name == "ap-south-1"
    make_runtime_client(Settings(bearer_token="fake-token-for-test"))
    import os

    assert os.environ["AWS_BEARER_TOKEN_BEDROCK"] == "fake-token-for-test"


def test_results_hide_text_in_repr() -> None:
    r = LLMResult("secret answer", "end_turn", 1, 1, 0.0, 0.0, "fake-llm")
    assert "secret answer" not in repr(r)
    assert "secret" not in repr(Message("user", "secret question"))


async def test_fake_llm_script_responder_errors_and_calls() -> None:
    fake = FakeLLM(["one", RuntimeError("boom"), "three"])
    assert (await fake.converse(system="s", messages=[], purpose="t")).text == "one"
    with pytest.raises(RuntimeError):
        await fake.converse(system="s", messages=[], purpose="t")
    assert (await fake.converse(system="s", messages=[], purpose="t")).text == "three"
    assert (await fake.converse(system="s", messages=[], purpose="t")).text == "ok"
    assert len(fake.calls) == 4 and fake.calls[0].purpose == "t"

    echo = FakeLLM(lambda system, messages: messages[-1].text.upper())
    out = await echo.converse(system="", messages=[Message("user", "abc")], purpose="t")
    assert out.text == "ABC" and out.cost_usd > 0


async def test_recording_records_once_then_replays(tmp_path) -> None:
    path = tmp_path / "rec.jsonl"
    inner = FakeLLM(["first answer"])
    rec = RecordingLLM(path, mode="record", inner=inner)
    kwargs = dict(system="s", messages=[Message("user", "q")], temperature=0.0, max_tokens=9,
                  purpose="unit")  # fmt: skip
    a = await rec.converse(**kwargs)
    b = await rec.converse(**kwargs)  # served from the recording, not the model
    assert a.text == b.text == "first answer" and len(inner.calls) == 1 and b.replayed

    replay = RecordingLLM(path, mode="replay", model_id="fake-llm")
    c = await replay.converse(**kwargs)
    assert c.text == "first answer" and c.replayed and len(replay) == 1


async def test_replay_missing_raises_instead_of_inventing(tmp_path) -> None:
    replay = RecordingLLM(tmp_path / "none.jsonl", mode="replay", model_id="fake-llm")
    with pytest.raises(ReplayMissing) as err:
        await replay.converse(system="s", messages=[Message("user", "new")], purpose="unit")
    assert len(err.value.keys) == 1 and len(err.value.keys[0]) == 64


def test_request_key_changes_with_any_input() -> None:
    base = dict(model_id="m", purpose="p", system="s", messages=[Message("user", "q")],
                temperature=0.0, max_tokens=10)  # fmt: skip
    k = request_key(**base)
    assert k == request_key(**base)
    for field, value in [("purpose", "x"), ("system", "t"), ("temperature", 0.1),
                         ("max_tokens", 11), ("messages", [Message("user", "r")])]:  # fmt: skip
        assert request_key(**(base | {field: value})) != k


def _key(user: str, system: str = "s") -> str:
    return request_key("m", "p", system, [Message("user", user)], 0.0, 10)


def test_request_key_ignores_marker_nonces() -> None:
    """Guards put a fresh random code in every prompt; without masking, nothing recorded
    could ever replay."""
    a = "Classify with code 1a2b3c4d.\n<<<TEXT 1a2b3c4d>>>\nhello\n<<<END TEXT 1a2b3c4d>>>"
    b = a.replace("1a2b3c4d", "99887766")
    assert _key(a) == _key(b)
    assert _key(a) != _key(a.replace("hello", "hullo"))  # the content still counts


def test_request_key_masks_document_wrapper_nonces() -> None:
    doc = (
        '<<<UNTRUSTED_DOCUMENT id="S1" nonce="{n}">>>\nfee\n'
        '<<<END_UNTRUSTED_DOCUMENT nonce="{n}">>>'
    )
    assert _key(doc.format(n="0badf00d")) == _key(doc.format(n="deadbeef"))


def test_request_key_masks_the_canary() -> None:
    a = "You help bank customers.\n\nInternal reference code: cnry-" + "ab" * 12 + "."
    b = a.replace("ab" * 12, "cd" * 12)
    assert _key("hi", system=a) == _key("hi", system=b)


def test_request_key_does_not_mask_hex_outside_markers() -> None:
    # Untrusted text cannot create markers (guards defuse "<<<"), so plain hex is content.
    assert _key("order 1a2b3c4d") != _key("order 99887766")
    assert _key("‹‹‹TEXT 1a2b3c4d›››") != _key("‹‹‹TEXT 99887766›››")


def test_record_mode_needs_a_real_client(tmp_path) -> None:
    with pytest.raises(ValueError):
        RecordingLLM(tmp_path / "x.jsonl", mode="record")
