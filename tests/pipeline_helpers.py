"""Shared helpers for the pipeline tests: a fake model that plays every role, and policies."""

import json
import re

from llm_guardrails.llm import FakeLLM
from llm_guardrails.policy_config import Policy, load_policy
from llm_guardrails.types import SourceDoc

S1 = SourceDoc(
    id="S1",
    text="Example Bank savings accounts require a minimum monthly average balance of "
    "Rs 5,000. A fee of Rs 200 applies if the balance falls below this.",
)
GOOD_ANSWER = "The minimum monthly average balance is Rs 5,000 [S1]."
TICKET = json.dumps(
    {
        "kind": "ticket",
        "title": "Card blocked after travel",
        "severity": "high",
        "tags": ["cards"],
        "due_days": 2,
    }  # fmt: skip
)


def policy(name: str = "default") -> Policy:
    return load_policy(name).model_copy(deep=True)


def canary_in(system: str) -> str:
    return re.search(r"cnry-[0-9a-f]{24}", system).group(0)


def bank_llm(answer=GOOD_ANSWER, *, classifier: str = "benign", delay_s: float = 0.0) -> FakeLLM:
    """One fake model for every role, told apart by the system prompt it receives:
    the injection classifier, the grounding judge, and the answering model. `answer` is a
    string, a list of strings (one per answer call), or a function (system, user) -> str."""
    queue = list(answer) if isinstance(answer, list) else None

    def respond(system, messages):
        if system.startswith("You are a security classifier"):
            label = {"label": classifier, "confidence": 0.95, "category": "benign", "reason": "t"}
            return json.dumps(label)
        if system.startswith("You check whether"):
            return json.dumps({"label": "supported", "reason": "matches"})
        if queue is not None:
            return queue.pop(0)
        if callable(answer):
            return answer(system, messages[-1].text)
        return answer

    return FakeLLM(respond, delay_s=delay_s)


def answer_calls(llm: FakeLLM):
    return [c for c in llm.calls if c.purpose == "answer"]
