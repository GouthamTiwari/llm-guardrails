"""No user text, model text or PII value ever reaches a log (docs/SPEC.md 14.1).

Every PII and injection seed row goes through the full pipeline with every log captured
at DEBUG level, using a model that echoes the user's text back (the worst case: the raw
text is then also in the answer). None of the gold PII values (in any spacing) or input
texts may appear anywhere in the captured logs.
"""

import json
import logging
import re
from pathlib import Path

import pytest
from pipeline_helpers import S1, bank_llm, policy

from llm_guardrails.pipeline import Pipeline

SEED = Path(__file__).parents[1] / "data" / "seed"
PII_ROWS = [json.loads(x) for x in (SEED / "pii_seed.jsonl").read_text("utf-8").splitlines()]
INJ_ROWS = [json.loads(x) for x in (SEED / "injection_seed.jsonl").read_text("utf-8").splitlines()]


def forbidden_strings() -> set[str]:
    out: set[str] = set()
    for row in PII_ROWS:
        for e in row.get("entities", []):
            value = e["value"]
            out |= {value, re.sub(r"[\s-]", "", value), value.lower()}
    for row in PII_ROWS + INJ_ROWS:
        text = row["text"].strip()
        if len(text) >= 12:  # very short texts could collide with log keys by chance
            out.add(text)
    return {s for s in out if len(s) >= 6}


def echo(system: str, user: str) -> str:
    return "You said: " + user.rsplit("Question: ", 1)[-1]


@pytest.mark.parametrize("policy_name", ["default", "banking_tax"])
async def test_no_raw_text_or_pii_in_any_log(policy_name, caplog) -> None:
    caplog.set_level(logging.DEBUG)  # every logger, every level
    pipe = Pipeline(policy(policy_name), bank_llm(echo))
    for row in PII_ROWS + INJ_ROWS:
        await pipe.run(row["text"], [S1])
        await pipe.check_input(row["text"], [S1])
        await pipe.check_output(row["text"], sources=[S1])

    captured = caplog.text + "\n".join(
        json.dumps(r.args, default=str) for r in caplog.records if r.args
    )
    audit = [r for r in caplog.records if r.name == "llm_guardrails.audit"]
    assert len(audit) == 3 * len(PII_ROWS + INJ_ROWS)  # one line per request, no more
    leaks = sorted(s for s in forbidden_strings() if s in captured)
    assert leaks == [], f"{len(leaks)} forbidden string(s) found in logs"
