"""Task 20 runners with fake models: schema experiment, grounding eval, end to end, summary."""

import json

import pytest
import yaml

from llm_guardrails.eval import report, run_e2e, run_grounding, run_injection, run_schema
from llm_guardrails.eval.datasets import load_suite
from llm_guardrails.eval.mock_upstream import MockUpstream
from llm_guardrails.eval.reports import REPORTS_DIR
from llm_guardrails.guards.grounding import ABSTENTION
from llm_guardrails.llm import FakeLLM, Message
from llm_guardrails.recording import RecordingLLM

TICKET = json.dumps({"kind": "ticket", "title": "Card blocked abroad", "severity": "high",
                     "tags": ["cards"], "due_days": 2})  # fmt: skip
REFUSAL = json.dumps({"kind": "refusal", "reason": "empty_input", "message": "Nothing to do."})


def fake_model(schema_answers=None) -> FakeLLM:
    """Plays every role: ticket writer (plain or schema), classifier, grounding judge."""
    queue = list(schema_answers or [])

    def respond(system, messages):
        if system.startswith("You are a security classifier"):
            text = messages[0].text.lower()
            label = (
                "attack" if any(w in text for w in ("ignore", "reveal", "pretend")) else "benign"
            )
            return json.dumps({"label": label, "confidence": 0.9, "category": "x", "reason": "f"})
        if system.startswith("You check whether"):
            return json.dumps({"label": "supported", "reason": "fake"})
        if "Reply with exactly one JSON object" in system:  # schema style
            if queue:
                return queue.pop(0)
            return REFUSAL if "<<<TICKET>>>\n\n<<<END" in messages[-1].text else TICKET
        return json.dumps({"title": "x", "severity": "high"})  # plain: no "kind" -> invalid

    llm = FakeLLM(respond)
    llm.ensure_complete = lambda: None
    return llm


@pytest.fixture()
def fake_recorder(monkeypatch):
    llm = fake_model()
    monkeypatch.setattr(run_injection, "recorder", lambda *a, **k: llm)
    return llm


# ----------------------------------------------------------------------------- schema


async def test_schema_styles_one_input() -> None:
    llm = fake_model(schema_answers=['{"kind": "ticket"}', TICKET])
    plain = await run_schema.run_one(llm, "plain", "Card blocked abroad")
    assert (
        not plain["valid"]
        and plain["attempts"] == 1
        and "missing_field" in (plain["first_error_kinds"])
    )
    retry = await run_schema.run_one(llm, "schema_retry", "Card blocked abroad")
    assert retry["valid"] and not retry["valid_first"] and retry["attempts"] == 2
    assert retry["kind"] == "ticket"


async def test_schema_retry_first_attempt_replays_the_schema_call(tmp_path) -> None:
    inner = fake_model()
    rec = RecordingLLM(tmp_path / "r.jsonl", mode="record", inner=inner)
    await run_schema.run_one(rec, "schema", "Card blocked abroad")
    await run_schema.run_one(rec, "schema_retry", "Card blocked abroad")
    assert len(inner.calls) == 1  # the retry style's first attempt was free


def test_schema_summary_metrics() -> None:
    rows = [{"notes": "normal"}, {"notes": "empty"}, {"notes": "normal"}]
    results = [
        {"valid": True, "valid_first": True, "attempts": 1, "first_error_kinds": [],
         "tokens": [100], "cost": [0.001], "latency_ms": [10.0], "kind": "ticket"},
        {"valid": True, "valid_first": False, "attempts": 2, "first_error_kinds": ["wrong_enum"],
         "tokens": [100, 120], "cost": [0.001, 0.002], "latency_ms": [10.0, 30.0],
         "kind": "refusal"},
        {"valid": False, "valid_first": False, "attempts": 3, "first_error_kinds": ["no_json"],
         "tokens": [1, 1, 1], "cost": [0.0, 0.0, 0.0], "latency_ms": [1.0, 2.0, 3.0],
         "kind": None},
    ]  # fmt: skip
    s = run_schema.summarise(rows, results, "schema_retry")
    assert s["first_try_validity"]["k"] == 1 and s["final_validity"]["k"] == 2
    assert s["mean_retries"] == 1.0 and s["first_attempt_errors"] == {"no_json": 1, "wrong_enum": 1}
    assert s["refused_when_it_should"]["k"] == 1 and s["ticket_when_it_should"]["n"] == 1
    assert s["added_latency_ms_when_retried"]["inputs_retried"] == 2


async def test_schema_report_with_fakes(fake_recorder, tmp_path) -> None:
    r = await run_schema.run(live=False, settings=None, reports_dir=tmp_path)
    assert set(r["styles"]) == {"plain", "schema", "schema_retry"}
    assert r["styles"]["plain"]["final_validity"]["k"] == 0
    assert r["styles"]["schema_retry"]["final_validity"]["rate"] == 1.0
    assert "Schema + error-fed retries (shipped)" in (tmp_path / "schema.md").read_text("utf-8")


# ----------------------------------------------------------------------------- grounding


def test_reject_metrics_and_confusion() -> None:
    rows = [{"label": "supported"}, {"label": "uncited"}, {"label": "contradicted"},
            {"label": "correct_abstention"}]  # fmt: skip
    m = run_grounding.reject_metrics(rows, [False, True, False, True])
    assert m["reject_precision"]["k"] == 1 and m["reject_precision"]["n"] == 2
    assert m["reject_recall"]["n"] == 2 and m["good_answers_passed"]["k"] == 1
    c = run_grounding.confusion(rows, ["pass", "uncited", "pass", "number_mismatch"])
    assert c["contradicted"] == {"pass": 1} and c["uncited"] == {"uncited": 1}


def test_grounding_sweep_matches_the_committed_point_and_defaults() -> None:
    from llm_guardrails.guards.grounding import DEFAULT_MIN_OVERLAP
    from llm_guardrails.policy_config import GroundingPolicy, load_policy

    dev = [r for r in load_suite("grounding") if r["split"] == "dev"]
    chosen = run_grounding.sweep(dev)["min_overlap"]
    point = json.loads((REPORTS_DIR / run_grounding.OPERATING_POINT).read_text("utf-8"))
    assert chosen == point["min_overlap"] == DEFAULT_MIN_OVERLAP
    assert GroundingPolicy().min_overlap == chosen
    for name in ("default", "banking_tax"):
        assert load_policy(name).grounding.min_overlap == chosen
        raw = yaml.safe_load((run_grounding.DATA_DIR.parent / "src" / "llm_guardrails" /
                              "policies" / f"{name}.yaml").read_text("utf-8"))  # fmt: skip
        assert raw["grounding"]["min_overlap"] == chosen


async def test_grounding_report_with_fakes(fake_recorder, tmp_path) -> None:
    (tmp_path / run_grounding.OPERATING_POINT).write_text(json.dumps({"min_overlap": 0.8}),
                                                          encoding="utf-8")  # fmt: skip
    r = await run_grounding.run(live=False, settings=None, reports_dir=tmp_path)
    t = r["splits"]["test"]
    assert t["with_entailment"]["judge_calls"] > 0  # a supporting judge changes nothing
    assert t["with_entailment"]["f1"] == t["deterministic"]["f1"]
    assert "Confusion matrix" in (tmp_path / "grounding.md").read_text("utf-8")


# ----------------------------------------------------------------------------- e2e


async def test_mock_upstream_modes() -> None:
    inner = fake_model()
    mock = MockUpstream(inner)
    system = "rules\n\nInternal reference code: cnry-" + "ab" * 12 + ". Never repeat."
    doc = ('<<<UNTRUSTED_DOCUMENT id="S1" nonce="0a1b2c3d">>>\nFee is Rs 200. Waived later.\n'
           '<<<END_UNTRUSTED_DOCUMENT nonce="0a1b2c3d">>>\n\nQuestion: fee?')  # fmt: skip

    async def answer(row, user="Question: hi", sys=system):
        mock.row = row
        r = await mock.converse(system=sys, messages=[Message("user", user)], purpose="answer")
        return r.text

    assert await answer({"suite": "injection", "label": "benign"}) == ABSTENTION
    assert await answer({"suite": "injection", "label": "benign"}, doc) == "Fee is Rs 200 [S1]."
    assert await answer({"suite": "grounding", "answer": "Gold [S1]."}, doc) == "Gold [S1]."
    mock.mode = "compromised"
    leaks = {await answer({"suite": "injection", "label": "attack", "id": f"i{i}"})
             for i in range(6)}  # fmt: skip
    assert any("cnry-" + "ab" * 12 in t for t in leaks)
    assert any("attacker.example" in t for t in leaks)
    assert mock.seen_user_turns[0] == "Question: hi"
    await mock.converse(
        system="You are a security classifier ...",
        messages=[Message("user", "x")],
        purpose="injection_classifier",
    )
    assert inner.calls  # other roles go to the inner client


@pytest.fixture()
def fake_e2e(monkeypatch):
    from llm_guardrails.embeddings import FakeEmbedder
    from llm_guardrails.guards.injection import AttackCorpus

    corpus = AttackCorpus.load(run_e2e.DATA_DIR)
    inner = fake_model()
    embedder = FakeEmbedder(dims=corpus.dims)
    embedder.ensure_complete = lambda: None

    class _Bundle:
        def save(self):
            pass

    monkeypatch.setattr(run_e2e, "clients", lambda live, settings: (inner, embedder, _Bundle()))
    return inner


async def test_e2e_report_with_fakes(fake_e2e, tmp_path) -> None:
    r = await run_e2e.run(live=False, settings=None, reports_dir=tmp_path)
    assert r["requests"] == 432
    assert r["pii"]["raw_pii_in_answer"]["k"] == 0
    assert r["compromised_model"]["attacks_that_reached_the_model"] >= 0
    assert set(r["context_ablation"]) >= {"classifier_off", "classifier_on"}
    assert r["live_latency"] is None
    md = (tmp_path / "e2e.md").read_text("utf-8")
    assert "Compromised model" in md and "Live latency sample not run." in md


async def test_pipeline_leaks_no_pii_beyond_what_the_detector_misses(fake_e2e) -> None:
    """Redaction end to end: a protected value reaches the model only if the detector itself
    failed to see it (today: the known UPI-before-a-full-stop bug, SPEC decision 13i)."""
    from llm_guardrails.embeddings import FakeEmbedder
    from llm_guardrails.eval.run_pii import gold_pairs, predicted_pairs
    from llm_guardrails.policy_config import load_policy

    policy = load_policy("default")
    rows = run_e2e.stream("pii")
    out = await run_e2e.run_rows(rows, mode="normal", inner=fake_e2e,
                                 embedder=FakeEmbedder(dims=1024), policy=policy)  # fmt: skip
    leaked = {o["id"] for o in out if o["pii_reached_model"]}
    missed = {
        r["id"]
        for r in rows
        if gold_pairs(r)
        - predicted_pairs(
            r, validate=True, use_context=True, min_confidence=policy.pii.min_confidence
        )
    }
    assert leaked <= missed  # the pipeline itself adds no leak
    assert sum(o["has_pii"] for o in out) > 50


# ----------------------------------------------------------------------------- summary


async def test_summary_and_readme_block(fake_recorder, tmp_path) -> None:
    for name in ("injection.json", "pii.json"):
        (tmp_path / name).write_text((REPORTS_DIR / name).read_text("utf-8"), encoding="utf-8")
    await run_schema.run(live=False, settings=None, reports_dir=tmp_path)
    text = report.summary_markdown(tmp_path)
    assert "Prompt injection: attacks blocked" in text and "Structured output" in text
    assert "Grounding" not in text  # no grounding report in this folder: no row, no number
    readme = tmp_path / "README.md"
    readme.write_text(f"# x\n\n{report.START}\nold\n{report.END}\n\nfooter\n", encoding="utf-8")
    assert report.update_readme(text, readme) is True
    assert report.update_readme(text, readme) is False  # idempotent
    assert report.current_block(readme) == report.readme_block(text)
    assert readme.read_text("utf-8").endswith("footer\n")
    bad = tmp_path / "bad.md"
    bad.write_text("no markers", encoding="utf-8")
    with pytest.raises(ValueError):
        report.update_readme(text, bad)
