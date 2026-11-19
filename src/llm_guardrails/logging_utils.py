"""Safe logging: salted hashes for identifiers and an allow-listed audit record.

Why allow-listing: trying to scrub PII out of a log line after the fact misses things. Here
the audit record is BUILT from a fixed list of safe fields (ids, actions, reason codes,
numbers, entity types and hash prefixes), so user text has no way in. See docs/SPEC.md 6.5
and 14.1.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import secrets
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

from llm_guardrails.types import EntitySummary, PipelineReport, Verdict, is_valid_reason

logger = logging.getLogger("llm_guardrails")
audit_logger = logging.getLogger("llm_guardrails.audit")

HASH_PREFIX_LEN = 8
_salt: str | None = None
_warned_random_salt = False
_SCORE_KEY = re.compile(r"^[a-z][a-z0-9_]{0,40}$")


def set_hash_salt(salt: str) -> None:
    """Called once at startup with LOG_HASH_SALT from settings."""
    global _salt
    _salt = salt or None


def _current_salt() -> str:
    global _salt, _warned_random_salt
    if _salt is None:
        env = os.environ.get("LOG_HASH_SALT", "")
        if env:
            _salt = env
        else:
            # Hashes stay safe, but cannot be compared across processes or runs.
            _salt = secrets.token_hex(16)
            if not _warned_random_salt:
                logger.warning("LOG_HASH_SALT is empty; using a random per-process salt")
                _warned_random_salt = True
    return _salt


def hash_value(value: str) -> str:
    """First 8 hex characters of sha256(salt + value). Enough to spot repeats, useless to
    reverse: without the salt, even a 12-digit Aadhaar cannot be brute-forced from it."""
    digest = hashlib.sha256((_current_salt() + value).encode("utf-8")).hexdigest()
    return digest[:HASH_PREFIX_LEN]


def summarize_entities(pairs: Iterable[tuple[str, str]]) -> list[EntitySummary]:
    """(type, canonical value) pairs -> per-type count and distinct hash prefixes."""
    by_type: dict[str, list[str]] = {}
    for etype, value in pairs:
        by_type.setdefault(etype, []).append(value)
    out = []
    for etype in sorted(by_type):
        values = by_type[etype]
        hashes = sorted({hash_value(v) for v in values})
        out.append(EntitySummary(type=etype, count=len(values), hash_prefixes=hashes))
    return out


def _safe_scores(scores: dict[str, float]) -> dict[str, float]:
    return {
        k: round(float(v), 4)
        for k, v in scores.items()
        if isinstance(k, str) and _SCORE_KEY.match(k) and isinstance(v, int | float)
    }


def _safe_verdict(v: Verdict) -> dict[str, Any]:
    return {
        "guard": v.guard if _SCORE_KEY.match(v.guard) else "unknown",
        "action": v.action.value,
        "reasons": [r for r in v.reasons if is_valid_reason(r)],
        "scores": _safe_scores(v.scores),
        "latency_ms": round(v.latency_ms, 3),
        "cost_usd": round(v.cost_usd, 8),
        "error_kind": v.error_kind,
    }


def safe_record(report: PipelineReport, route: str) -> dict[str, Any]:
    """The one JSON log line per request (spec 14.1). Built only from allow-listed fields."""
    entities: dict[str, EntitySummary] = {}
    for v in report.verdicts:
        for e in v.entities:
            if e.type in entities:
                prev = entities[e.type]
                entities[e.type] = EntitySummary(
                    type=e.type,
                    count=prev.count + e.count,
                    hash_prefixes=sorted(set(prev.hash_prefixes) | set(e.hash_prefixes)),
                )
            else:
                entities[e.type] = e
    return {
        "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
        "request_id": report.request_id,
        "route": route if _SCORE_KEY.match(route.replace("-", "_")) else "unknown",
        "decision": report.decision,
        "stage": report.stage,
        "policy": report.policy_name,
        "version": report.version,
        "latency_ms": round(report.latency_ms, 3),
        "tokens_in": report.tokens_in,
        "tokens_out": report.tokens_out,
        "cost_usd": round(report.cost_usd, 8),
        "guards": [_safe_verdict(v) for v in report.verdicts],
        "entities": [
            {
                "type": e.type,
                "count": e.count,
                "hash_prefixes": [h for h in e.hash_prefixes if re.fullmatch(r"[0-9a-f]{8}", h)],
            }
            for e in sorted(entities.values(), key=lambda e: e.type)
        ],
    }


def audit_log(report: PipelineReport, route: str) -> dict[str, Any]:
    record = safe_record(report, route)
    audit_logger.info(json.dumps(record, ensure_ascii=False, sort_keys=False))
    return record
