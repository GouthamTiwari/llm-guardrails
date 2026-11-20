"""Task 19: metrics, the dev sweep, the injection and PII evaluations (offline, fakes)."""

import json

import pytest

from llm_guardrails.eval import metrics, run_injection, run_pii, sweep
from llm_guardrails.eval.run_injection import OperatingPoint, RowScore
from llm_guardrails.llm import FakeLLM

# ----------------------------------------------------------------------------- metrics


@pytest.mark.parametrize(
    ("k", "n", "lo", "hi"),
    [(0, 10, 0.0, 0.2775), (10, 10, 0.7225, 1.0), (5, 10, 0.2366, 0.7634),
     (1, 100, 0.0018, 0.0545), (0, 0, 0.0, 1.0)],
)  # fmt: skip
def test_wilson_interval_known_values(k, n, lo, hi) -> None:
    a, b = metrics.wilson(k, n)
    assert a == pytest.approx(lo, abs=1e-4) and b == pytest.approx(hi, abs=1e-4)


def test_wilson_rejects_impossible_counts() -> None:
    with pytest.raises(ValueError):
        metrics.wilson(11, 10)


def test_rate_and_pct_formatting() -> None:
    block = metrics.rate(9, 10)
    assert block == {"k": 9, "n": 10, "rate": 0.9, "ci95": [0.5958, 0.9821]}
    assert metrics.pct(block) == "90.0% (59.6–98.2, 9/10)"
    assert metrics.pct(metrics.rate(0, 0)) == "–"


def test_recall_fpr_and_grouping() -> None:
    o = metrics.Outcome
    rows = [
        o("attack", True, "a", "en"),
        o("attack", False, "a", "hi"),
        o("benign", True, "b", "en"),
        o("benign", False, "b", "en"),
        o("benign", False, "b", "hi"),
    ]
    r = metrics.recall_and_fpr(rows)
    assert r["attack_recall"]["rate"] == 0.5 and r["benign_fpr"]["rate"] == 0.3333
    by_lang = metrics.by_key(rows, "language")
    assert by_lang["hi"]["attack_recall"]["k"] == 0 and by_lang["en"]["benign_fpr"]["n"] == 2


def test_precision_recall_f1() -> None:
    m = metrics.precision_recall_f1(8, 2, 2)
    assert (m["precision"], m["recall"], m["f1"]) == (0.8, 0.8, 0.8)
    empty = metrics.precision_recall_f1(0, 0, 0)
    assert empty["precision"] is None and empty["f1"] is None


# ----------------------------------------------------------------------------- decisions

OP = OperatingPoint(t_block=0.6, t_allow=0.35, t_llm=0.5, benign_lambda=0.5)


def row(label="attack", rules=0.0, sa=0.0, sb=0.0, llm=None, err=False, cat="direct_override"):
    return RowScore(
        id="x", label=label, category=cat, language="en", text="t", clean="t", decoded=[],
        rules=rules, windows=[(sa, sb)], llm=llm, llm_error=err,
    )  # fmt: skip


def test_embed_score_applies_the_benign_margin() -> None:
    r = RowScore(id="x", label="attack", category="", language="", text="", clean="",
                 decoded=[], windows=[(0.5, 0.7), (0.6, 0.2)])  # fmt: skip
    assert r.embed(0.0) == pytest.approx(0.6)
    assert r.embed(1.0) == pytest.approx(0.6)  # the second window has no benign penalty
    assert row(sa=0.5, sb=0.7).embed(1.0) == pytest.approx(0.3)


@pytest.mark.parametrize(
    ("r", "layers", "blocked"),
    [
        (row(rules=1.0), "all", True),
        (row(sa=0.65), "all", True),
        (row(sa=0.2), "all", False),
        (row(sa=0.45, llm=0.9), "all", True),
        (row(sa=0.45, llm=0.1), "all", False),
        (row(sa=0.45, llm=0.1, err=True), "all", True),  # classifier failure fails closed
        (row(sa=0.45, llm=0.9), "rules_embed", False),
        (row(rules=1.0, sa=0.0), "embed", False),
        (row(sa=0.65), "rules", False),
        (row(sa=0.1, llm=0.9), "llm", True),
    ],
)
def test_decisions_per_layer(r, layers, blocked) -> None:
    assert r.blocked(OP, layers) is blocked


def test_grey_zone_without_a_classifier_score_is_an_error() -> None:
    with pytest.raises(RuntimeError):
        row(sa=0.45).blocked(OP)
    assert row(sa=0.45).in_grey_zone(OP) and not row(sa=0.2).in_grey_zone(OP)


# ----------------------------------------------------------------------------- sweep


def _sweep_rows() -> list[RowScore]:
    rows = [row("attack", sa=0.7, llm=0.9) for _ in range(10)]  # caught by embeddings
    rows += [row("attack", sa=0.45, llm=0.8) for _ in range(10)]  # only the classifier
    rows += [row("attack", sa=0.1, llm=0.2) for _ in range(5)]  # undetectable
    rows += [row("benign", sa=0.3, sb=0.3, llm=0.1, cat="benign") for _ in range(60)]
    rows += [row("benign", sa=0.5, sb=0.6, llm=0.6, cat="benign_hard") for _ in range(20)]
    return rows


def test_sweep_maximises_recall_within_the_fpr_limits() -> None:
    result = sweep.sweep(_sweep_rows())
    assert result["dev"]["attack_recall"]["k"] == 20  # all detectable attacks
    assert result["dev"]["benign_fpr"]["rate"] <= 0.02
    assert result["dev"]["hard_benign_fpr"]["rate"] <= 0.10
    assert result["feasible_points"] > 0
    # Hard benign rows sit at 0.5 similarity with a 0.6 classifier score: only a classifier
    # threshold above 0.6 keeps them out while catching the 0.8 attacks.
    assert 0.6 < result["t_llm"] <= 0.8


def test_sweep_refuses_when_nothing_is_feasible() -> None:
    rows = [row("attack", rules=1.0, llm=0.9)] + [row("benign", rules=1.0, llm=0.1, cat="benign")]
    with pytest.raises(RuntimeError, match="no threshold"):
        sweep.sweep(rows)
    with pytest.raises(ValueError, match="classifier score"):
        sweep.sweep([row("attack")])


# ----------------------------------------------------------------------------- PII


def _pii(text, entities):
    return {"id": "p", "text": text, "entities": entities, "split": "test"}


def test_pii_matching_is_typed_normalised_and_counted() -> None:
    rows = [
        _pii("My Aadhaar is ३५८८ १३९९ ८६९७", [{"type": "AADHAAR", "value": "३५८८ १३९९ ८६९७"}]),
        _pii("Mail ravi@example.com or RAVI@example.com",
             [{"type": "EMAIL", "value": "ravi@example.com"}]),
        _pii("Order 234567890123 has shipped", []),
        _pii("Reference 4111 1111 1111 1112 was declined", []),
    ]  # fmt: skip
    r = run_pii.evaluate(rows, validate=True, use_context=True, min_confidence=0.6)
    assert r["per_type"]["AADHAAR"]["tp"] == 1  # Hindi digits folded before matching
    assert r["per_type"]["EMAIL"]["tp"] == 1 and r["per_type"]["EMAIL"]["fp"] == 1  # 2nd copy
    assert r["rows_without_pii_with_a_detection"]["k"] == 0
    regex_only = run_pii.evaluate(rows, validate=False, use_context=False, min_confidence=0.6)
    assert regex_only["rows_without_pii_with_a_detection"]["k"] >= 1  # validation matters


def test_pii_report_runs_offline(tmp_path) -> None:
    r = run_pii.run(reports_dir=tmp_path)
    assert set(r["test"]["ablation"]) == set(run_pii.STEPS)
    md = (tmp_path / "pii.md").read_text(encoding="utf-8")
    assert "regex + validation + context (shipped)" in md and "| **micro** |" in md
    assert r["mode"] == "offline" and r["spend_usd"] == 0.0


# ----------------------------------------------------------------------------- end to end


def fake_classifier() -> FakeLLM:
    def respond(system, messages):
        text = messages[0].text.lower()
        attack = any(w in text for w in ("ignore", "reveal", "system", "pretend", "send"))
        return json.dumps({"label": "attack" if attack else "benign", "confidence": 0.9,
                           "category": "x", "reason": "fake"})  # fmt: skip

    return FakeLLM(respond)


@pytest.fixture()
def fake_clients(monkeypatch):
    from llm_guardrails.embeddings import FakeEmbedder
    from llm_guardrails.guards.injection import AttackCorpus

    corpus = AttackCorpus.load(run_injection.DATA_DIR)
    embedder = FakeEmbedder(dims=corpus.dims)
    embedder.ensure_complete = lambda: None
    llm = fake_classifier()
    llm.ensure_complete = lambda: None

    class _Bundle:
        def save(self):
            pass

    def clients(live, recordings, vectors, settings):
        return llm, embedder, _Bundle(), corpus

    monkeypatch.setattr(run_injection, "_clients", clients)
    monkeypatch.setattr(run_injection, "load_external", lambda path=None: None)
    return llm


async def test_sweep_then_eval_writes_every_report(fake_clients, tmp_path) -> None:
    point = await sweep.run(live=False, settings=None, reports_dir=tmp_path)
    assert point["source"] == "dev sweep" and (tmp_path / "operating_point.json").is_file()
    calls_after_sweep = len(fake_clients.calls)
    report = await run_injection.run(live=False, settings=None, reports_dir=tmp_path)
    assert set(report["test"]["layers"]) == {"rules", "embed", "rules_embed", "all"}
    assert "llm" in report["dev_reference"]["layers"]  # LLM-alone reference: dev only
    assert report["external"] is None
    for name in ("injection.json", "injection.md", "what_got_through.md"):
        assert (tmp_path / name).is_file()
    # The test run calls the classifier only for grey-zone rows, never for every row.
    grey_calls = len(fake_clients.calls) - calls_after_sweep
    assert grey_calls <= report["test"]["rows"] + report["dev_reference"]["rows"]
    md = (tmp_path / "injection.md").read_text(encoding="utf-8")
    assert "## Test split" in md and "Not run (data not downloaded)." in md


async def test_external_numbers_are_kept_when_the_raw_data_is_absent(fake_clients,
                                                                     tmp_path) -> None:  # fmt: skip
    (tmp_path / "injection.json").write_text(json.dumps({"external": {"rows": 116}}),
                                             encoding="utf-8")  # fmt: skip
    kept = await run_injection.run_external(OP, live=False, settings=None, reports_dir=tmp_path)
    assert kept == {"rows": 116}


def test_external_loader_maps_labels(tmp_path) -> None:
    path = tmp_path / "x.json"
    path.write_text(json.dumps([{"text": "a", "label": 1}, {"text": "b", "label": 0}]),
                    encoding="utf-8")  # fmt: skip
    rows = run_injection.load_external(path)
    assert [r["label"] for r in rows] == ["attack", "benign"]
    assert run_injection.load_external(tmp_path / "missing.json") is None


def test_injection_defaults_equal_the_dev_operating_point() -> None:
    """Spec 7.4: the sweep's choice becomes the package default, and the two never drift."""
    from llm_guardrails.eval.reports import REPORTS_DIR
    from llm_guardrails.guards.injection import InjectionConfig

    point = json.loads((REPORTS_DIR / "operating_point.json").read_text(encoding="utf-8"))
    assert point["source"] == "dev sweep"
    cfg = InjectionConfig()
    assert (cfg.t_block, cfg.t_allow, cfg.t_llm, cfg.benign_lambda) == (
        point["t_block"],
        point["t_allow"],
        point["t_llm"],
        point["benign_lambda"],
    )
