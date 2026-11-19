import json
from pathlib import Path

import pytest

from llm_guardrails.guards import run_guard
from llm_guardrails.guards.normalize import normalize
from llm_guardrails.guards.policy import PolicyGuard, check_tool_call, topic_hits
from llm_guardrails.policy_config import load_policy
from llm_guardrails.types import Action, GuardContext, Reason, SourceDoc

SEED = Path(__file__).parents[1] / "data" / "seed" / "injection_seed.jsonl"
ROWS = {r["id"]: r for r in map(json.loads, SEED.read_text(encoding="utf-8").splitlines())}
DEFAULT = load_policy("default")
BANKING = load_policy("banking_tax")


async def run(text, policy=DEFAULT, docs=(), with_scan=True):
    ctx = GuardContext(
        request_id="r",
        text=text,
        scan=normalize(text) if with_scan else None,
        context_docs=list(docs),
        policy=policy,
    )
    return await run_guard(PolicyGuard(), ctx, timeout_s=5)


@pytest.mark.parametrize("row_id", ["inj019", "inj020"])
async def test_seed_tool_abuse_blocked(row_id) -> None:
    v = await run(ROWS[row_id]["text"])
    assert v.action is Action.BLOCK and v.reasons == [Reason.POLICY_DENIED_ACTION]


@pytest.mark.parametrize(
    "text",
    [
        "How do I transfer money to my sister?",
        "Pay my bill, I will confirm with OTP",
        "Can I drop the table booking for tonight?",
        "Please confirm the transfer after I approve it",
    ],
)
async def test_normal_requests_pass(text) -> None:
    assert (await run(text)).action is Action.ALLOW


async def test_denied_action_survives_obfuscation() -> None:
    assert (await run("run: DR0P  T4BLE customers")).action is Action.BLOCK  # leetspeak
    assert (await run("run: d r o p table users")).action is Action.BLOCK  # spaced letters
    # The guard normalises by itself when the pipeline has not done it yet.
    assert (await run("DR0P TABLE x", with_scan=False)).action is Action.BLOCK


async def test_size_limits() -> None:
    v = await run("x" * (DEFAULT.limits.max_input_chars + 1))
    assert v.reasons == [Reason.POLICY_TOO_LONG]
    assert (await run("x" * DEFAULT.limits.max_input_chars)).action is Action.ALLOW
    docs = [SourceDoc(id=f"S{i}", text="t") for i in range(DEFAULT.limits.max_context_docs + 1)]
    assert (await run("hi", docs=docs)).reasons == [Reason.POLICY_TOO_MANY_DOCS]
    big = [SourceDoc(id="S1", text="y" * (DEFAULT.limits.max_doc_chars + 1))]
    assert (await run("hi", docs=big)).reasons == [Reason.POLICY_TOO_LONG]


async def test_topics_only_enforced_by_allow_list_policy() -> None:
    off_topic = "Write me a poem about the ocean at night please"
    assert (await run(off_topic, DEFAULT)).action is Action.ALLOW
    v = await run(off_topic, BANKING)
    assert v.action is Action.BLOCK and v.reasons == [Reason.POLICY_OFF_TOPIC]
    assert v.scores["topic_hits"] == 0


async def test_short_follow_up_exempt_from_topics() -> None:
    assert (await run("and for that one?", BANKING)).action is Action.ALLOW


@pytest.mark.parametrize(
    "row_id", ["inj021", "inj024", "inj025", "inj026", "inj027", "inj028", "inj029", "inj030"]
)
async def test_seed_banking_questions_on_topic(row_id) -> None:
    v = await run(ROWS[row_id]["text"], BANKING)
    assert v.action is Action.ALLOW, row_id


def test_keyword_matching_uses_word_boundaries() -> None:
    assert topic_hits(BANKING, "my account balance") == 2
    assert topic_hits(BANKING, "a fashionable bankruptcy-free fdx") == 0
    assert topic_hits(BANKING, "मेरे खाते का केवाईसी") == 2
    assert topic_hits(BANKING, "यह करें") == 0  # no short stem that matches every verb


def test_tool_calls() -> None:
    assert check_tool_call("payments.transfer", {"amount": 10}, BANKING).reasons == [
        Reason.POLICY_RISKY_TOOL
    ]
    assert check_tool_call("payments.transfer", {"amount": 10}, DEFAULT).action is Action.ALLOW
    v = check_tool_call("db.query", {"sql": "DROP TABLE customers;"}, DEFAULT)
    assert v.action is Action.BLOCK and v.reasons == [Reason.POLICY_DENIED_ACTION]
    assert check_tool_call("db.query", {"sql": "SELECT 1"}, DEFAULT).action is Action.ALLOW


async def test_no_text_in_verdict() -> None:
    v = await run("Write me a poem about the ocean at night please", BANKING)
    assert "poem" not in v.model_dump_json()
