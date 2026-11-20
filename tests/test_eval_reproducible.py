"""The committed reports regenerate offline, byte for byte (Task 19 acceptance; spec 13.5).

Only `generated_at` and `git_sha` may differ (they say when and from which commit a report
was made); the operating point's `date` likewise. Everything else - every count, interval
and markdown table - must come out identical from the committed recordings and vectors,
with no AWS account. On a machine without the downloaded deepset file (CI), the external
numbers are carried over from the committed report, which this test also exercises.
"""

import json
import shutil

import pytest

from llm_guardrails.eval import run_injection, run_pii, sweep
from llm_guardrails.eval.reports import REPORTS_DIR, stable
from llm_guardrails.settings import load_settings


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture()
def reports_copy(tmp_path):
    for name in ("operating_point.json", "injection.json"):
        shutil.copy(REPORTS_DIR / name, tmp_path / name)
    return tmp_path


async def test_operating_point_reproduces(tmp_path) -> None:
    await sweep.run(live=False, settings=load_settings(env_file=None), reports_dir=tmp_path)
    new, old = _load(tmp_path / "operating_point.json"), _load(REPORTS_DIR / "operating_point.json")
    new.pop("date"), old.pop("date")
    assert new == old


async def test_injection_report_reproduces(reports_copy) -> None:
    await run_injection.run(live=False, settings=load_settings(env_file=None),
                            reports_dir=reports_copy)  # fmt: skip
    assert stable(_load(reports_copy / "injection.json")) == stable(
        _load(REPORTS_DIR / "injection.json")
    )
    for name in ("injection.md", "what_got_through.md"):
        assert (reports_copy / name).read_text(encoding="utf-8") == (REPORTS_DIR / name).read_text(
            encoding="utf-8"
        )


def test_pii_report_reproduces(tmp_path) -> None:
    run_pii.run(reports_dir=tmp_path)
    assert stable(_load(tmp_path / "pii.json")) == stable(_load(REPORTS_DIR / "pii.json"))
    assert (tmp_path / "pii.md").read_text(encoding="utf-8") == (REPORTS_DIR / "pii.md").read_text(
        encoding="utf-8"
    )


async def test_external_numbers_survive_without_the_raw_file(reports_copy, monkeypatch) -> None:
    monkeypatch.setattr(run_injection, "load_external", lambda path=None: None)
    await run_injection.run(live=False, settings=load_settings(env_file=None),
                            reports_dir=reports_copy)  # fmt: skip
    assert (
        _load(reports_copy / "injection.json")["external"]
        == _load(REPORTS_DIR / "injection.json")["external"]
    )
