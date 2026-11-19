"""Leakage between splits and against the layer-2 corpus (docs/SPEC.md 13.2, decision 5).

A test number is only honest if the test rows are new to the system. So a test row fails
the check when it is:
- a near-duplicate (character 5-gram Jaccard >= 0.6) of any dev row, or (injection) of any
  attack-corpus text or benign anchor;
- with vectors present (injection): the same meaning as a corpus/anchor text (cosine >=
  0.90, decision 5) or as a dev row (cosine >= 0.95). Meaning-level closeness matters
  because the corpus was built by rewording: a reworded attack shares few 5-grams.

Vectors use the same `embedding_text` as the layer-2 corpus, so the comparison is
like for like, and they are stored next to the dataset (`data/injection_vectors.*`).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from llm_guardrails.eval.datasets import DATA_DIR, load_jsonl, load_suite, row_text
from llm_guardrails.eval.similarity import char_ngrams, jaccard

COSINE_REFERENCE = 0.90  # SPEC decision 5 (stricter than the 0.95 in 13.2)
COSINE_DEV = 0.95
VECTOR_PREFIX = "injection_vectors"


def near_reference(texts: Sequence[str], references: Sequence[str],
                   threshold: float = 0.6) -> set[int]:  # fmt: skip
    """Indexes of texts at or above `threshold` Jaccard with any reference text."""
    refs = [char_ngrams(r) for r in references]
    out = set()
    for i, t in enumerate(texts):
        g = char_ngrams(t)
        if any(jaccard(g, r) >= threshold for r in refs):
            out.add(i)
    return out


def corpus_texts(data_dir: Path = DATA_DIR) -> list[str]:
    return [r["text"] for name in ("attack_corpus", "benign_anchors")
            for r in load_jsonl(data_dir / f"{name}.jsonl")]  # fmt: skip


# ----------------------------------------------------------------------------- vectors


def _bundle(data_dir: Path):
    from llm_guardrails.embeddings import VectorBundle
    from llm_guardrails.guards.injection import AttackCorpus

    corpus = AttackCorpus.load(data_dir)
    return VectorBundle(data_dir / VECTOR_PREFIX, corpus.model_id, corpus.dims), corpus


def dataset_vectors(texts: Sequence[str], data_dir: Path = DATA_DIR) -> np.ndarray | None:
    """Stored vectors for these texts, or None if any is missing (then only Jaccard runs)."""
    from llm_guardrails.guards.injection import embedding_text

    try:
        bundle, _ = _bundle(data_dir)
    except Exception:  # noqa: BLE001 - no corpus vectors: no meaning-level check possible
        return None
    rows = [bundle.get(embedding_text(t)) for t in texts]
    if not rows or any(r is None for r in rows):
        return None
    return np.stack(rows)


def missing_vectors(texts: Sequence[str], data_dir: Path = DATA_DIR) -> int:
    """How many texts have no stored vector. 0 when nothing was embedded yet (no vector file:
    a fresh dataset) or there is no corpus; a vector file that does not cover every row is
    stale, and that must not pass silently. The committed dataset is held to full coverage
    by tests/test_datasets.py."""
    from llm_guardrails.guards.injection import embedding_text

    if not (data_dir / f"{VECTOR_PREFIX}.json").is_file():
        return 0
    try:
        bundle, _ = _bundle(data_dir)
    except Exception:  # noqa: BLE001 - no corpus vectors at all: nothing to compare against
        return 0
    return sum(bundle.get(embedding_text(t)) is None for t in texts)


def reference_vectors(data_dir: Path = DATA_DIR) -> np.ndarray | None:
    try:
        _, corpus = _bundle(data_dir)
    except Exception:  # noqa: BLE001
        return None
    return np.concatenate([corpus.attack, corpus.benign])


async def embed_texts(texts: Sequence[str], embedder, data_dir: Path = DATA_DIR) -> int:
    """Embed (and store) the texts that have no vector yet. Returns how many were new."""
    from llm_guardrails.embeddings import RecordedEmbedder
    from llm_guardrails.guards.injection import embedding_text

    bundle, _ = _bundle(data_dir)
    keys = sorted({embedding_text(t) for t in texts})
    missing = [k for k in keys if bundle.get(k) is None]
    if missing:
        await RecordedEmbedder(bundle, inner=embedder).embed(missing, purpose="dataset_embed")
        bundle.save()
    return len(missing)


def near_corpus_by_meaning(texts: Sequence[str], data_dir: Path = DATA_DIR) -> set[int]:
    vecs, refs = dataset_vectors(texts, data_dir), reference_vectors(data_dir)
    if vecs is None or refs is None:
        return set()
    return {int(i) for i in np.where((vecs @ refs.T).max(axis=1) >= COSINE_REFERENCE)[0]}


# ----------------------------------------------------------------------------- check


def leak_findings(suite: str, rows: Sequence[dict], data_dir: Path = DATA_DIR) -> list[str]:
    """Every test row that is too close to dev or (injection) to the corpus. [] = clean."""
    dev_rows = [r for r in rows if r.get("split") == "dev"]
    test_rows = [r for r in rows if r.get("split") == "test"]
    dev = [row_text(suite, r) for r in dev_rows]
    test = [row_text(suite, r) for r in test_rows]
    found = [f"{test_rows[i]['id']}: near-duplicate of a dev row"
             for i in near_reference(test, dev)]  # fmt: skip
    if suite == "injection" and test:
        missing = missing_vectors([row_text(suite, r) for r in rows], data_dir)
        if missing:
            # Without vectors the meaning-level check would silently not run: say so loudly.
            found.append(f"{missing} injection row(s) have no vector: run "
                         "`guardrails dataset embed --live`, then split again")  # fmt: skip
        refs = corpus_texts(data_dir)
        found += [f"{test_rows[i]['id']}: near-duplicate of a corpus/anchor text"
                  for i in near_reference(test, refs)]  # fmt: skip
        found += [f"{test_rows[i]['id']}: same meaning as a corpus/anchor text"
                  for i in near_corpus_by_meaning(test, data_dir)]  # fmt: skip
        tv, dv = dataset_vectors(test, data_dir), dataset_vectors(dev, data_dir)
        if tv is not None and dv is not None and len(dev):
            close = np.where((tv @ dv.T).max(axis=1) >= COSINE_DEV)[0]
            found += [f"{test_rows[int(i)]['id']}: same meaning as a dev row" for i in close]
    return sorted(set(found))


def check_suite(suite: str, data_dir: Path = DATA_DIR) -> list[str]:
    return leak_findings(suite, load_suite(suite, data_dir), data_dir)
