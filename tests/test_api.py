"""HTTP API (docs/SPEC.md 15.1) in fake mode: no AWS."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from llm_guardrails.api import create_app
from llm_guardrails.runtime import Pipelines
from llm_guardrails.settings import Settings
from llm_guardrails.types import ConfigError

SEED = Path(__file__).parents[1] / "data" / "seed" / "injection_seed.jsonl"
INJ = {r["id"]: r for r in map(json.loads, SEED.read_text(encoding="utf-8").splitlines())}
S1 = {"id": "S1", "text": "Example Bank savings accounts require a minimum monthly average "
      "balance of Rs 5,000. A fee of Rs 200 applies if the balance falls below this."}  # fmt: skip
FAKE = Settings(llm_mode="fake")


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(create_app(FAKE))


# ----------------------------------------------------------------------------- health


def test_healthz(client) -> None:
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["llm_mode"] == "fake" and body["version"]
    assert {"default", "banking_tax"} <= set(body["policy_names"])


def test_metrics_endpoint(client) -> None:
    client.post("/v1/check-input", json={"text": "What is the fee?"})
    r = client.get("/metrics")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/plain")
    assert 'guardrails_requests_total{decision="allow",route="check-input"}' in r.text


# ----------------------------------------------------------------------------- chat


def test_chat_allow_with_a_cited_answer(client) -> None:
    r = client.post("/v1/guarded-chat",
                    json={"question": "What is the minimum balance?", "context": [S1]})  # fmt: skip
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"request_id", "decision", "answer", "data", "user_message", "report"}
    assert body["decision"] == "allow" and body["user_message"] is None
    assert body["answer"].endswith("[S1].") and "Rs 5,000" in body["answer"]
    assert body["report"]["stage"] == "complete"
    assert [v["guard"] for v in body["report"]["verdicts"]][-2:] == ["grounding", "leak"]


def test_chat_block_is_a_200(client) -> None:
    r = client.post("/v1/guarded-chat", json={"question": INJ["inj001"]["text"]})
    assert r.status_code == 200
    body = r.json()
    assert body["decision"] == "block" and body["answer"] is None
    assert body["user_message"] == "Sorry, I can't help with that request."
    assert body["report"]["stage"] == "input"


def test_chat_with_schema_returns_data(client) -> None:
    r = client.post("/v1/guarded-chat", json={"question": "My card is blocked abroad",
                                              "schema": "TicketSummary"})  # fmt: skip
    body = r.json()
    assert r.status_code == 200 and body["decision"] == "allow"
    assert body["data"]["kind"] == "ticket" and body["data"]["severity"] == "medium"


def test_response_never_carries_redacted_pii_in_the_report(client) -> None:
    r = client.post("/v1/guarded-chat", json={
        "question": "Mail asha.verma@example.com: what is the minimum balance?", "context": [S1]
    })  # fmt: skip
    assert r.status_code == 200
    assert "asha.verma" not in json.dumps(r.json()["report"])


# ----------------------------------------------------------------------------- stages


def test_check_input(client) -> None:
    r = client.post("/v1/check-input", json={"text": "My PAN is ABCPE1234F", "context": [S1]})
    body = r.json()
    assert r.status_code == 200 and set(body) == {"request_id", "decision", "sanitized_text",
                                                  "wrapped_context", "report"}  # fmt: skip
    assert body["decision"] == "allow" and body["sanitized_text"] == "My PAN is [PAN_1]"
    assert '<<<UNTRUSTED_DOCUMENT id="S1"' in body["wrapped_context"]
    assert "ABCPE1234F" not in r.text


def test_check_input_block_returns_no_text(client) -> None:
    r = client.post("/v1/check-input", json={"text": "Card 4111 1111 1111 1111 please"})
    body = r.json()
    assert body["decision"] == "block" and body["sanitized_text"] is None
    assert "4111" not in r.text


def test_check_output(client) -> None:
    canary = "cnry-" + "ab" * 12
    ok = client.post("/v1/check-output", json={
        "output": "The minimum monthly average balance is Rs 5,000 [S1].", "sources": [S1],
    }).json()  # fmt: skip
    assert set(ok) == {"request_id", "decision", "output", "data", "report"}
    assert ok["decision"] == "allow"
    leak = client.post("/v1/check-output", json={"output": f"Code {canary} [S1].",
                                                 "sources": [S1], "canary": canary})  # fmt: skip
    assert leak.json()["decision"] == "block" and leak.json()["output"] is None
    retry = client.post("/v1/check-output", json={"output": '{"kind": "ticket"}',
                                                  "schema": "TicketSummary"})  # fmt: skip
    assert retry.status_code == 200 and retry.json()["decision"] == "block"


# ----------------------------------------------------------------------------- 422 / 500


@pytest.mark.parametrize(
    ("path", "body", "field"),
    [
        ("/v1/guarded-chat", {"question": "hi", "schema": "Nope"}, "body.schema"),
        ("/v1/guarded-chat", {"question": "hi", "schema": "EntailmentJudgement"}, "body.schema"),
        ("/v1/guarded-chat", {"question": "hi", "policy": "../../etc/passwd"}, "body.policy"),
        (
            "/v1/guarded-chat",
            {"question": "hi", "context": [{"id": "S 1", "text": "x"}]},
            "body.context.0.id",
        ),
        ("/v1/guarded-chat", {}, "body.question"),
        ("/v1/check-output", {"output": "x", "canary": "not-a-canary"}, "body.canary"),
        ("/v1/check-input", {"text": "x", "surprise": 1}, "body.surprise"),
    ],
)
def test_bad_requests_are_422_naming_only_the_field(client, path, body, field) -> None:
    r = client.post(path, json=body)
    assert r.status_code == 422
    assert r.json() == {"error": "invalid_request", "fields": [field]}


def test_422_does_not_echo_the_input(client) -> None:
    r = client.post("/v1/guarded-chat",
                    json={"question": "My Aadhaar is 3588 1399 8697", "extra": "x"})  # fmt: skip
    assert r.status_code == 422 and "3588" not in r.text


class ExplodingPipelines(Pipelines):
    def get(self, *args, **kwargs):
        raise RuntimeError("internal detail with PAN ABCPE1234F")


def test_internal_error_is_500_without_details() -> None:
    app = create_app(FAKE, ExplodingPipelines(FAKE))
    r = TestClient(app, raise_server_exceptions=False).post(
        "/v1/guarded-chat", json={"question": "hello there"}
    )
    assert r.status_code == 500
    body = r.json()
    assert set(body) == {"request_id", "error"} and body["error"] == "internal_error"
    assert "ABCPE" not in r.text and "RuntimeError" not in r.text


# ----------------------------------------------------------------------------- modes


def test_bedrock_mode_without_credentials_refuses_to_start() -> None:
    with pytest.raises(ConfigError, match="credentials"):
        create_app(Settings(llm_mode="bedrock"))


def test_replay_mode_without_recordings_fails_closed(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)  # no data/recordings here
    client = TestClient(create_app(Settings(llm_mode="replay")))
    question = {"question": "What is the minimum balance?", "context": [S1]}
    body = client.post("/v1/guarded-chat", json=question).json()
    assert body["decision"] == "block"
    assert body["report"]["verdicts"][-1]["guard"] == "model"
