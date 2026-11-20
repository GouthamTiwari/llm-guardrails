"""Report files under reports/ (docs/SPEC.md 13.7): shared metadata and writers.

Every report JSON carries {generated_at, git_sha, dataset_sha256, operating_point, mode,
spend_usd}. `generated_at` and `git_sha` describe WHEN and FROM WHICH COMMIT a report was
made, so they are the only fields allowed to differ when a replay reproduces a report;
everything else must match byte for byte (tests/test_eval_reproducible.py).
"""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from llm_guardrails.eval.datasets import REPO_ROOT
from llm_guardrails.guards.injection import text_sha256

REPORTS_DIR = REPO_ROOT / "reports"
VOLATILE_FIELDS = ("generated_at", "git_sha")


def git_sha() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                             capture_output=True, text=True, timeout=10, check=True)  # fmt: skip
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def dataset_sha256(paths: list[Path]) -> dict[str, str]:
    return {p.name: text_sha256(p) for p in paths if p.is_file()}


def meta(*, paths: list[Path], mode: str, spend_usd: float,
         operating_point: dict[str, Any] | None) -> dict[str, Any]:  # fmt: skip
    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_sha": git_sha(),
        "dataset_sha256": dataset_sha256(paths),
        "operating_point": operating_point,
        "mode": mode,
        "spend_usd": round(spend_usd, 6),
    }


def write_json(name: str, data: dict[str, Any], reports_dir: Path = REPORTS_DIR) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / name
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
                    newline="\n")  # fmt: skip
    return path


def write_text(name: str, text: str, reports_dir: Path = REPORTS_DIR) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / name
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def stable(report: dict[str, Any]) -> dict[str, Any]:
    """The report without the fields that legitimately change between runs."""
    return {k: v for k, v in report.items() if k not in VOLATILE_FIELDS}
