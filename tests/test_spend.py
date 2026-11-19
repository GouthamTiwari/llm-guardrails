import json
import threading

import pytest

from llm_guardrails.spend import SpendLedger
from llm_guardrails.types import BudgetExceeded


def test_records_and_totals(tmp_path) -> None:
    ledger = SpendLedger(tmp_path / "l.jsonl", limit_usd=1.0)
    assert ledger.total() == 0.0
    ledger.record(purpose="unit", model_id="fake-llm", tokens_in=10, tokens_out=5, usd=0.25)
    ledger.record(purpose="unit", model_id="fake-llm", tokens_in=10, tokens_out=5, usd=0.5)
    assert ledger.total() == 0.75
    row = json.loads((tmp_path / "l.jsonl").read_text().splitlines()[0])
    assert set(row) == {"ts", "purpose", "model", "tokens_in", "tokens_out", "usd"}


def test_check_raises_before_the_cap_is_crossed(tmp_path) -> None:
    ledger = SpendLedger(tmp_path / "l.jsonl", limit_usd=1.0)
    ledger.record(purpose="unit", model_id="fake-llm", tokens_in=1, tokens_out=1, usd=0.9)
    ledger.check(0.1)  # exactly at the cap is fine
    with pytest.raises(BudgetExceeded):
        ledger.check(0.11)


def test_purpose_must_be_an_identifier(tmp_path) -> None:
    ledger = SpendLedger(tmp_path / "l.jsonl")
    with pytest.raises(ValueError):
        ledger.record(purpose="user said hi", model_id="m", tokens_in=1, tokens_out=1, usd=0)


def test_concurrent_records_are_all_kept(tmp_path) -> None:
    ledger = SpendLedger(tmp_path / "l.jsonl", limit_usd=100)

    def work() -> None:
        for _ in range(50):
            ledger.record(purpose="t", model_id="m", tokens_in=1, tokens_out=1, usd=0.01)

    threads = [threading.Thread(target=work) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(ledger.entries()) == 200
    assert ledger.total() == pytest.approx(2.0)
