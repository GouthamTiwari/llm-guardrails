"""Injection evaluation (docs/SPEC.md 13.3-13.4): layers alone and combined, on the test
split, at the operating point the dev sweep chose; plus a dev reference and an external
check on the deepset/prompt-injections test split.

Every row is scored ONCE per layer, then decisions are computed from those scores:
- layer 1: the rule score (free, deterministic);
- layer 2: for each embedding window, the best cosine to the attack corpus and to the
  benign anchors (the benign margin λ is applied later, so λ can be swept for free);
- layer 3: the grey-zone classifier's attack probability. On dev it is fetched for every
  row (the sweep needs it wherever a threshold might put the grey zone, and it doubles as
  the "LLM alone" reference); on test only for rows in the grey zone at the operating
  point, exactly the calls the real guard would make.

Model and embedding calls go through recorders: `--live` records, the default replays, so
`guardrails eval --suite injection` reproduces the report offline.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from llm_guardrails.eval.datasets import DATA_DIR, FILES, load_suite
from llm_guardrails.eval.metrics import Outcome, by_key, pct, rate, recall_and_fpr
from llm_guardrails.eval.reports import meta, write_json, write_text
from llm_guardrails.guards.injection import (
    AttackCorpus,
    embedding_text,
    embedding_windows,
    score_llm,
    score_rules,
)
from llm_guardrails.guards.normalize import normalize
from llm_guardrails.pricing import cost_usd, estimate_tokens

RECORDINGS = DATA_DIR / "recordings" / "injection_eval.jsonl"
EXTERNAL_RECORDINGS = DATA_DIR / "recordings" / "injection_external.jsonl"
DATASET_VECTORS = DATA_DIR / "injection_vectors"  # Task 18 bundle, extended with windows
EXTERNAL_VECTORS = DATA_DIR / "embeddings" / "deepset_test"
EXTERNAL_RAW = DATA_DIR / "raw" / "deepset_prompt_injections_test.json"
OPERATING_POINT = "operating_point.json"
# One at a time: four parallel Nova calls hit this account's on-demand quota
# (ThrottlingException after boto's retries). Replay is instant either way.
CLASSIFIER_CONCURRENCY = 1


@dataclass(frozen=True)
class OperatingPoint:
    t_block: float
    t_allow: float
    t_llm: float
    benign_lambda: float

    @classmethod
    def load(cls, reports_dir: Path) -> OperatingPoint:
        data = json.loads((reports_dir / OPERATING_POINT).read_text(encoding="utf-8"))
        return cls(data["t_block"], data["t_allow"], data["t_llm"], data["benign_lambda"])

    def as_dict(self) -> dict[str, float]:
        return {"t_block": self.t_block, "t_allow": self.t_allow, "t_llm": self.t_llm,
                "benign_lambda": self.benign_lambda}  # fmt: skip


@dataclass
class RowScore:
    id: str
    label: str
    category: str
    language: str
    text: str = field(repr=False)
    clean: str = field(repr=False)
    decoded: list[str] = field(repr=False)
    rules: float = 0.0
    windows: list[tuple[float, float]] = field(default_factory=list)  # (s_attack, s_benign)
    embed_cost: float = 0.0
    llm: float | None = None  # None: not fetched
    llm_error: bool = False
    llm_cost: float = 0.0

    def embed(self, lam: float) -> float:
        """Layer-2 score at margin λ: the best window of clip(sa - λ·max(0, sb - sa))."""
        return max((min(1.0, max(0.0, sa - lam * max(0.0, sb - sa))) for sa, sb in self.windows),
                   default=0.0)  # fmt: skip

    def in_grey_zone(self, op: OperatingPoint) -> bool:
        e = self.embed(op.benign_lambda)
        return self.rules < 1.0 and e < op.t_block and max(self.rules, e) >= op.t_allow

    def blocked(self, op: OperatingPoint, layers: str = "all") -> bool:
        """layers: rules | embed | rules_embed | all | llm (classifier on everything)."""
        e = self.embed(op.benign_lambda)
        if layers == "rules":
            return self.rules >= 1.0
        if layers == "embed":
            return e >= op.t_block
        if layers == "llm":
            return self._llm_blocks(op)
        if self.rules >= 1.0 or e >= op.t_block:
            return True
        if layers == "rules_embed" or max(self.rules, e) < op.t_allow:
            return False
        return self._llm_blocks(op)  # grey zone

    def _llm_blocks(self, op: OperatingPoint) -> bool:
        if self.llm is None:
            raise RuntimeError(f"{self.id}: classifier score needed but not fetched")
        return self.llm_error or self.llm >= op.t_llm  # a failed classifier fails closed


# ----------------------------------------------------------------------------- scoring


async def score_rows(rows: list[dict], embedder: Any, corpus: AttackCorpus) -> list[RowScore]:
    """Layers 1 and 2 for every row (one embedding call per window, recorded)."""
    out: list[RowScore] = []
    for r in rows:
        scan = normalize(r["text"])
        windows = embedding_windows(embedding_text(scan))
        vectors = np.asarray(await embedder.embed(windows, purpose="injection_embed"))
        sims = [(float(np.max(corpus.attack @ v)), float(np.max(corpus.benign @ v)))
                for v in vectors]  # fmt: skip
        out.append(RowScore(
            id=r["id"], label=r["label"], category=r.get("category", ""),
            language=r.get("language", ""), text=r["text"], clean=scan.clean,
            decoded=list(scan.decoded), rules=score_rules(scan).score, windows=sims,
            embed_cost=sum(cost_usd(corpus.model_id, estimate_tokens(w)) for w in windows),
        ))  # fmt: skip
    return out


async def add_llm_scores(rows: list[RowScore], llm: Any) -> None:
    """Classifier score for each given row, CLASSIFIER_CONCURRENCY at a time."""
    from llm_guardrails.types import SchemaValidationFailed

    semaphore = asyncio.Semaphore(CLASSIFIER_CONCURRENCY)

    async def one(row: RowScore) -> None:
        async with semaphore:
            try:
                s = await score_llm(row.clean, row.decoded, llm)
            except SchemaValidationFailed as e:
                row.llm, row.llm_error = 1.0, True
                row.llm_cost = sum(getattr(a, "cost_usd", 0.0) for a in e.trace)
                return
            row.llm, row.llm_cost = s.score, s.cost_usd

    await asyncio.gather(*(one(r) for r in rows if r.llm is None))


# ----------------------------------------------------------------------------- results


def outcomes(rows: list[RowScore], op: OperatingPoint, layers: str) -> list[Outcome]:
    return [Outcome(r.label, r.blocked(op, layers), r.category, r.language) for r in rows]


LAYER_NAMES = {
    "rules": "Rules only (layer 1)",
    "embed": "Embeddings only (layer 2)",
    "rules_embed": "Rules + embeddings (grey zone allowed)",
    "all": "Rules + embeddings + LLM classifier (shipped)",
    "llm": "LLM classifier on every input (reference)",
}


def evaluate(rows: list[RowScore], op: OperatingPoint, layers: tuple[str, ...]) -> dict:
    n = len(rows)
    grey = [r for r in rows if r.in_grey_zone(op)]
    llm_cost = sum(r.llm_cost for r in grey)
    embed_cost = sum(r.embed_cost for r in rows)
    shipped = outcomes(rows, op, "all")
    return {
        "rows": n,
        "layers": {name: recall_and_fpr(outcomes(rows, op, name)) for name in layers},
        "hard_benign_fpr": rate(
            sum(o.blocked for o in shipped if o.category == "benign_hard"),
            sum(o.category == "benign_hard" for o in shipped),
        ),  # fmt: skip
        "per_category": by_key(shipped, "category"),
        "per_language": by_key(shipped, "language"),
        "classifier_calls": len(grey),
        "classifier_call_rate": round(len(grey) / n, 4) if n else None,
        "classifier_errors_failed_closed": sum(r.llm_error for r in grey),
        "usd_per_1000_requests": {
            "rules": 0.0,
            "embeddings": round(1000 * embed_cost / n, 6) if n else None,
            "llm_classifier": round(1000 * llm_cost / n, 6) if n else None,
        },
    }


# ----------------------------------------------------------------------------- external


def load_external(path: Path = EXTERNAL_RAW) -> list[dict] | None:
    """deepset/prompt-injections test split, if downloaded (label 1 = injection)."""
    if not path.is_file():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [{"id": f"ext{i:03d}", "text": r["text"],
             "label": "attack" if int(r["label"]) == 1 else "benign"}
            for i, r in enumerate(raw)]  # fmt: skip


def download_external(path: Path = EXTERNAL_RAW) -> int:
    """Fetch the 116-row test split through the Hugging Face datasets-server API (JSON, no
    new dependency) into data/raw/ (git-ignored: the repo links to it, never copies it)."""
    import httpx

    url = "https://datasets-server.huggingface.co/rows"
    rows: list[dict] = []
    offset = 0
    while True:
        params = {"dataset": "deepset/prompt-injections", "config": "default", "split": "test",
                  "offset": offset, "length": 100}  # fmt: skip
        resp = httpx.get(url, params=params, timeout=30)
        resp.raise_for_status()
        batch = [r["row"] for r in resp.json()["rows"]]
        rows += [{"text": r["text"], "label": int(r["label"])} for r in batch]
        if len(batch) < 100:
            break
        offset += 100
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    return len(rows)


# ----------------------------------------------------------------------------- run


def _clients(live: bool, recordings: Path, vectors: Path, settings: Any):
    from llm_guardrails.embeddings import RecordedEmbedder, TitanEmbedder, VectorBundle
    from llm_guardrails.llm import BedrockLLM
    from llm_guardrails.recording import RecordingLLM
    from llm_guardrails.spend import SpendLedger

    corpus = AttackCorpus.load(DATA_DIR)
    bundle = VectorBundle(vectors, corpus.model_id, corpus.dims)
    if live:
        ledger = SpendLedger(limit_usd=settings.live_budget_usd)
        llm = RecordingLLM(recordings, mode="record", inner=BedrockLLM(settings, ledger=ledger))
        embedder = RecordedEmbedder(bundle, inner=TitanEmbedder(settings, ledger=ledger))
    else:
        llm = RecordingLLM(recordings, mode="replay", model_id=settings.chat_model_id)
        embedder = RecordedEmbedder(bundle)
    return llm, embedder, bundle, corpus


async def score_split(split: str, *, live: bool, settings: Any,
                      op: OperatingPoint | None) -> tuple[list[RowScore], Any]:  # fmt: skip
    """Score one split. With op=None (the sweep) the classifier runs on every row."""
    llm, embedder, bundle, corpus = _clients(live, RECORDINGS, DATASET_VECTORS, settings)
    rows = [r for r in load_suite("injection") if r["split"] == split]
    scores = await score_rows(rows, embedder, corpus)
    targets = scores if op is None else [s for s in scores if s.in_grey_zone(op)]
    await add_llm_scores(targets, llm)
    if live:
        bundle.save()
    llm.ensure_complete()  # a replay miss must fail the run, not become a quiet block
    embedder.ensure_complete()
    return scores, llm


async def run(*, live: bool, settings: Any, reports_dir: Path) -> dict:
    from llm_guardrails.eval.sweep import load_dev_scores

    op = OperatingPoint.load(reports_dir)
    test, _ = await score_split("test", live=live, settings=settings, op=op)
    dev = await load_dev_scores(live=live, settings=settings)

    report: dict[str, Any] = {
        "test": evaluate(test, op, ("rules", "embed", "rules_embed", "all")),
        "dev_reference": evaluate(dev, op, ("rules", "embed", "rules_embed", "all", "llm")),
    }
    report["external"] = await run_external(op, live=live, settings=settings,
                                            reports_dir=reports_dir)  # fmt: skip
    spend = sum(r.llm_cost for r in test if r.in_grey_zone(op))
    spend += sum(r.llm_cost for r in dev) + sum(r.embed_cost for r in test + dev)
    full = meta(paths=[DATA_DIR / FILES["injection"], DATA_DIR / "attack_corpus.jsonl",
                       DATA_DIR / "benign_anchors.jsonl"],
                mode="live" if live else "replay", spend_usd=spend,
                operating_point=op.as_dict()) | report  # fmt: skip
    write_json("injection.json", full, reports_dir)
    write_text("injection.md", injection_markdown(full), reports_dir)
    write_text("what_got_through.md", what_got_through(test, op), reports_dir)
    return full


async def run_external(op: OperatingPoint, *, live: bool, settings: Any,
                       reports_dir: Path) -> dict | None:  # fmt: skip
    """External row; without the downloaded data (a fresh clone) the committed numbers are
    kept as they are, so the report still reproduces byte for byte."""
    rows = load_external()
    if rows is None:
        previous = reports_dir / "injection.json"
        if previous.is_file():
            return json.loads(previous.read_text(encoding="utf-8")).get("external")
        return None
    llm, embedder, bundle, corpus = _clients(live, EXTERNAL_RECORDINGS, EXTERNAL_VECTORS,
                                             settings)  # fmt: skip
    scores = await score_rows(rows, embedder, corpus)
    await add_llm_scores([s for s in scores if s.in_grey_zone(op)], llm)
    if live:
        bundle.save()
    llm.ensure_complete()
    embedder.ensure_complete()
    result = evaluate(scores, op, ("rules", "embed", "rules_embed", "all"))
    for key in ("per_category", "per_language", "hard_benign_fpr"):
        result.pop(key)  # deepset has no categories, languages or hard-benign label
    result["source"] = ("deepset/prompt-injections, test split (Apache-2.0), downloaded at "
                        "eval time into data/raw/ and not committed")  # fmt: skip
    return result


# ----------------------------------------------------------------------------- markdown


def _layer_table(block: dict) -> list[str]:
    lines = ["| Layers | Attack recall | Benign false-positive rate |", "|---|---|---|"]
    for name, m in block["layers"].items():
        lines.append(f"| {LAYER_NAMES[name]} | {pct(m['attack_recall'])} | "
                     f"{pct(m['benign_fpr'])} |")  # fmt: skip
    return lines


def injection_markdown(r: dict) -> str:
    op = r["operating_point"]
    t = r["test"]
    lines = [
        "# Injection detection",
        "",
        f"Mode: {r['mode']}. Operating point from the dev sweep: T_BLOCK {op['t_block']}, "
        f"T_ALLOW {op['t_allow']}, T_LLM {op['t_llm']}, λ {op['benign_lambda']}. "
        "Rates show the 95% Wilson interval and the counts.",
        "",
        f"## Test split ({t['rows']} rows, evaluated once)",
        "",
        *_layer_table(t),
        "",
        f"Hard-benign false-positive rate: {pct(t['hard_benign_fpr'])}. "
        f"Classifier calls: {t['classifier_calls']} of {t['rows']} inputs "
        f"({100 * t['classifier_call_rate']:.1f}%), "
        f"{t['classifier_errors_failed_closed']} failed closed.",
        "",
        "Cost per 1,000 requests (USD): " + ", ".join(
            f"{k} {v}" for k, v in t["usd_per_1000_requests"].items()),
        "",
        "### Per category (shipped configuration)",
        "",
        "| Category | Attack recall | Benign FPR |",
        "|---|---|---|",
        *[f"| {k} | {pct(v['attack_recall'])} | {pct(v['benign_fpr'])} |"
          for k, v in t["per_category"].items()],
        "",
        "### Per language (shipped configuration)",
        "",
        "| Language | Attack recall | Benign FPR |",
        "|---|---|---|",
        *[f"| {k} | {pct(v['attack_recall'])} | {pct(v['benign_fpr'])} |"
          for k, v in t["per_language"].items()],
        "",
        f"## Dev split, reference ({r['dev_reference']['rows']} rows; the sweep tuned on it)",
        "",
        *_layer_table(r["dev_reference"]),
        "",
    ]  # fmt: skip
    ext = r.get("external")
    lines += ["## External check: deepset/prompt-injections test split", ""]
    if ext:
        lines += [f"{ext['source']}. Generic English and German prompts, not banking: an "
                  "out-of-domain check.", "", *_layer_table(ext), "",
                  f"Classifier calls: {ext['classifier_calls']} of {ext['rows']} inputs. The "
                  "classifier only sees inputs that the rules or the (banking-focused) "
                  "embedding corpus put in the grey zone, so out-of-domain attacks mostly "
                  "never reach it.", ""]  # fmt: skip
    else:
        lines += ["Not run (data not downloaded).", ""]
    return "\n".join(lines)


def what_got_through(rows: list[RowScore], op: OperatingPoint) -> str:
    missed = [r for r in rows if r.label == "attack" and not r.blocked(op)]
    lines = [
        "# What got through",
        "",
        f"Every test-split attack the shipped configuration allowed: {len(missed)} of "
        f"{sum(r.label == 'attack' for r in rows)}. The text is our own synthetic test data.",
        "",
        "| id | category | language | rules | embedding | classifier | text |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in sorted(missed, key=lambda x: (x.category, x.id)):
        text = r.text.replace("|", "\\|").replace("\n", " ")[:300]
        llm = "–" if r.llm is None else f"{r.llm:.2f}"
        lines.append(f"| {r.id} | {r.category} | {r.language} | {r.rules:.2f} | "
                     f"{r.embed(op.benign_lambda):.2f} | {llm} | {text} |")  # fmt: skip
    return "\n".join(lines) + "\n"
