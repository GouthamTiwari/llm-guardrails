"""Evaluation datasets: row schemas, loading, validation and the dev/test split
(docs/SPEC.md 13.1-13.2).

The split rules exist so the test numbers are honest:
- seed rows stay in dev (the guards were built while looking at them);
- near-duplicates (character 5-gram Jaccard >= 0.6) are grouped and a group never straddles
  the split, otherwise "test" would contain copies of dev rows;
- injection rows close to the layer-2 attack corpus or benign anchors (Jaccard >= 0.6, or
  cosine >= 0.90 when vectors exist, SPEC decision 5) are kept in dev: the test split must
  measure attacks the corpus does not already contain;
- within those constraints, every stratum (category / entity mix / label / input kind) is
  split 50/50 with seed 7.
"""

from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from llm_guardrails.eval.similarity import NEAR_DUPLICATE, char_ngrams, jaccard
from llm_guardrails.eval.synth import EMAIL_DOMAINS, UPI_HANDLES, passes_check
from llm_guardrails.types import BUILTIN_ENTITY_TYPES, SourceDoc

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data"
SEED = 7
SUITES = ("injection", "pii", "grounding", "schema")
FILES = {"injection": "injection.jsonl", "pii": "pii.jsonl", "grounding": "grounding.jsonl",
         "schema": "schema_inputs.jsonl"}  # fmt: skip
SEED_FILES = {"injection": "injection_seed.jsonl", "pii": "pii_seed.jsonl",
              "grounding": "grounding_seed.jsonl", "schema": "schema_seed.jsonl"}  # fmt: skip

ATTACK_CATEGORIES = ("direct_override", "roleplay_jailbreak", "system_prompt_extraction",
                     "delimiter_confusion", "encoding_obfuscation", "multilingual",
                     "indirect_document", "exfiltration", "tool_abuse")  # fmt: skip
BENIGN_CATEGORIES = ("benign", "benign_hard")
LANGUAGES = ("en", "hi", "hinglish", "es", "mixed")
GROUNDING_LABELS = tuple(
    "supported correct_abstention uncited fabricated_citation unsupported contradicted "
    "overgeneralised partially_supported".split()
)
SCHEMA_KINDS = ("normal", "multiple_issue", "number_heavy", "hindi_hinglish", "hostile",
                "empty", "very_long", "off_topic")  # fmt: skip
Review = Literal["unreviewed", "human-ok", "human-edited", "human-dropped"]
Split = Literal["dev", "test"]


class _Row(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z]{2,4}[0-9a-z_-]{1,20}$")
    split: Split | None = None
    review: Review = "unreviewed"


class InjectionRow(_Row):
    text: str = Field(min_length=1, max_length=4000)
    label: Literal["attack", "benign"]
    category: Literal[ATTACK_CATEGORIES + BENIGN_CATEGORIES]  # type: ignore[valid-type]
    language: Literal[LANGUAGES]  # type: ignore[valid-type]
    source: Literal["seed", "generated", "handwritten", "deepset"]

    @field_validator("category")
    @classmethod
    def _label_matches(cls, value: str, info: Any) -> str:
        label = info.data.get("label")
        if label is not None and (label == "benign") != (value in BENIGN_CATEGORIES):
            raise ValueError("category does not match label")
        return value


class Entity(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal[tuple(sorted(BUILTIN_ENTITY_TYPES))]  # type: ignore[valid-type]
    value: str = Field(min_length=1)


class PiiRow(_Row):
    text: str = Field(min_length=1, max_length=2000)
    entities: list[Entity]
    lookalikes: list[dict[str, str]] = []  # which look-alike kinds the text holds (no PII)


class GroundingRow(_Row):
    question: str = Field(min_length=1)
    sources: list[SourceDoc] = Field(min_length=1)
    answer: str = Field(min_length=1)
    label: Literal[GROUNDING_LABELS]  # type: ignore[valid-type]


class SchemaInputRow(_Row):
    input: str = Field(max_length=20000)
    target_schema: Literal["TicketSummary"]
    notes: Literal[SCHEMA_KINDS]  # type: ignore[valid-type]


ROW_MODELS: dict[str, type[_Row]] = {
    "injection": InjectionRow,
    "pii": PiiRow,
    "grounding": GroundingRow,
    "schema": SchemaInputRow,
}


# ----------------------------------------------------------------------------- io


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def write_jsonl(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    path.write_text(text, encoding="utf-8", newline="\n")


def load_suite(suite: str, data_dir: Path = DATA_DIR, *, include_dropped: bool = False):
    rows = load_jsonl(data_dir / FILES[suite])
    return [r for r in rows if include_dropped or r.get("review") != "human-dropped"]


def row_text(suite: str, row: dict) -> str:
    """The text that near-duplicate and leakage checks compare."""
    if suite == "grounding":
        return f"{row['question']}\n{row['answer']}"
    if suite == "schema":
        return row["input"]
    return row["text"]


SEED_ID = re.compile(r"^(inj|pii|gr|sch)\d{3}$")  # ids from data/seed/; generated ids differ


def is_seed(row: dict) -> bool:
    return bool(SEED_ID.match(str(row.get("id", ""))))


def stratum(suite: str, row: dict) -> str:
    if suite == "injection":
        return row["category"]
    if suite == "pii":
        return "+".join(sorted({e["type"] for e in row["entities"]})) or "none"
    if suite == "grounding":
        return row["label"]
    return row["notes"]


# ----------------------------------------------------------------------------- validate

_ANY_EMAIL = re.compile(r"[\w.+-]+@([\w-]+(?:\.[\w-]+)+)")
_ANY_HANDLE = re.compile(r"[\w.-]+@([A-Za-z][A-Za-z0-9]{1,63})(?![\w.@])")


def validate_rows(suite: str, rows: Sequence[dict], *, require_split: bool = True) -> list[str]:
    """Every problem found, as short messages naming the row id (never row text)."""
    problems: list[str] = []
    seen: set[str] = set()
    model = ROW_MODELS[suite]
    for row in rows:
        rid = str(row.get("id", "?"))
        if rid in seen:
            problems.append(f"{rid}: duplicate id")
        seen.add(rid)
        try:
            parsed = model.model_validate(row)
        except ValidationError as exc:
            fields = sorted({".".join(map(str, e["loc"])) for e in exc.errors()})
            problems.append(f"{rid}: invalid fields {fields}")
            continue
        if require_split and parsed.split is None:
            problems.append(f"{rid}: no split assigned")
        if suite == "pii":
            problems += _validate_pii(rid, parsed)  # type: ignore[arg-type]
        if suite == "grounding":
            problems += [f"{rid}: {p}" for p in grounding_sanity(row)]
        text = row_text(suite, row) if suite != "grounding" else json.dumps(row)
        for domain in _ANY_EMAIL.findall(text):
            if not is_example_domain(domain):
                problems.append(f"{rid}: email domain is not an example domain")
    return problems


def is_example_domain(domain: str) -> bool:
    """example.com/org/net and their subdomains, or the reserved .example top-level domain
    (RFC 2606): none of them can belong to a real person."""
    d = domain.lower().rstrip(".")
    return d.endswith(".example") or any(d == e or d.endswith("." + e) for e in EMAIL_DOMAINS)


def _validate_pii(rid: str, row: PiiRow) -> list[str]:
    from llm_guardrails.eval.synth import lookalike_fails

    out = []
    for e in row.entities:
        if e.value not in row.text:
            out.append(f"{rid}: {e.type} value not in text")
        elif not passes_check(e.type, e.value):
            out.append(f"{rid}: {e.type} value fails its check")
    for look in row.lookalikes:
        if look["value"] not in row.text:
            out.append(f"{rid}: look-alike value not in text")
        elif not lookalike_fails(look["kind"], look["value"]):
            out.append(f"{rid}: look-alike {look['kind']} passes the check it imitates")
    for handle in _ANY_HANDLE.findall(row.text):
        if "." not in handle and handle.lower() not in UPI_HANDLES + ("okbank",):
            out.append(f"{rid}: UPI handle is not a fictional one")
    return out


def grounding_sanity(row: dict) -> list[str]:
    """Deterministic consistency between a grounding row's label and its answer (13.1)."""
    from llm_guardrails.guards.grounding import ABSTENTION_PATTERN, citations

    cited = citations(row["answer"])
    ids = {s["id"] if isinstance(s, dict) else s.id for s in row["sources"]}
    label = row["label"]
    if label == "correct_abstention":
        if cited or not ABSTENTION_PATTERN.search(row["answer"]):
            return ["correct_abstention must abstain and cite nothing"]
        return []
    if label == "uncited":
        return ["uncited answer has citations"] if cited else []
    if label == "fabricated_citation":
        return [] if any(c not in ids for c in cited) else ["no citation to a missing source"]
    if not cited:
        return [f"{label} answer cites nothing"]
    if any(c not in ids for c in cited):
        return [f"{label} answer cites a missing source"]
    return []


# ----------------------------------------------------------------------------- split


def _groups(texts: Sequence[str], threshold: float = NEAR_DUPLICATE) -> list[int]:
    """Union-find over near-duplicate pairs: group index per text."""
    parent = list(range(len(texts)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    grams = [char_ngrams(t) for t in texts]
    for i in range(len(grams)):
        for j in range(i + 1, len(grams)):
            if jaccard(grams[i], grams[j]) >= threshold:
                parent[find(i)] = find(j)
    return [find(i) for i in range(len(texts))]


def assign_splits(
    suite: str,
    rows: list[dict],
    *,
    forced_dev: set[int] | None = None,
    seed: int = SEED,
) -> list[dict]:
    """50/50 dev/test per stratum, whole near-duplicate groups at a time (13.2)."""
    texts = [row_text(suite, r) for r in rows]
    group_of = _groups(texts)
    forced = set(forced_dev or set()) | {i for i, r in enumerate(rows) if is_seed(r)}
    members: dict[int, list[int]] = defaultdict(list)
    for i, g in enumerate(group_of):
        members[g].append(i)
    dev_groups = {g for g, idx in members.items() if any(i in forced for i in idx)}

    by_stratum: dict[str, list[int]] = defaultdict(list)
    for g, idx in members.items():
        by_stratum[stratum(suite, rows[idx[0]])].append(g)

    rng = random.Random(seed)
    split_of: dict[int, str] = {}
    for key in sorted(by_stratum):
        groups = sorted(by_stratum[key])
        total = sum(len(members[g]) for g in groups)
        target_test = total // 2
        free = [g for g in groups if g not in dev_groups]
        rng.shuffle(free)
        test_count = 0
        for g in free:
            size = len(members[g])
            if test_count + size <= target_test:
                split_of[g] = "test"
                test_count += size
            else:
                split_of[g] = "dev"
        for g in groups:
            split_of.setdefault(g, "dev")
    return [{**r, "split": split_of[group_of[i]]} for i, r in enumerate(rows)]


# ----------------------------------------------------------------------------- commands


def split_suite(suite: str, data_dir: Path = DATA_DIR) -> dict[str, int]:
    """Assign dev/test to every row of data/<suite>.jsonl and rewrite it. Injection rows
    close to the layer-2 corpus (by wording, or by meaning when vectors exist) stay in dev."""
    from llm_guardrails.eval.leakage import corpus_texts, near_corpus_by_meaning, near_reference

    rows = load_suite(suite, data_dir, include_dropped=True)
    forced: set[int] = set()
    if suite == "injection":
        texts = [row_text(suite, r) for r in rows]
        forced = near_reference(texts, corpus_texts(data_dir)) | near_corpus_by_meaning(
            texts, data_dir
        )
    rows = assign_splits(suite, rows, forced_dev=forced)
    write_jsonl(data_dir / FILES[suite], rows)
    counts = {"dev": 0, "test": 0, "kept_in_dev_near_corpus": len(forced)}
    for r in rows:
        counts[r["split"]] += 1
    return counts


def validate_suite(suite: str, data_dir: Path = DATA_DIR) -> list[str]:
    rows = load_suite(suite, data_dir, include_dropped=True)
    if not rows:
        return [f"{FILES[suite]}: missing or empty"]
    return validate_rows(suite, rows)
