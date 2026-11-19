"""Hard cap on live Bedrock spend across the whole project (docs/SPEC.md 2.3).

Every real call first checks that `spent + estimate` fits under LIVE_BUDGET_USD, then appends
its actual cost to `.spend/ledger.jsonl` (gitignored). The file outlives any one run, so the
cap holds across days. Lines carry purpose, model, token counts and USD only: never text.
"""

from __future__ import annotations

import json
import re
import threading
from datetime import UTC, datetime
from pathlib import Path

from llm_guardrails.types import BudgetExceeded

DEFAULT_LEDGER = Path(".spend/ledger.jsonl")
_PURPOSE = re.compile(r"^[a-z][a-z0-9_.-]{0,40}$")


class SpendLedger:
    def __init__(self, path: Path | str = DEFAULT_LEDGER, limit_usd: float = 5.0) -> None:
        self.path = Path(path)
        self.limit_usd = limit_usd
        self._lock = threading.Lock()

    def entries(self) -> list[dict]:
        if not self.path.is_file():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(json.loads(line))
        return out

    def total(self) -> float:
        return round(sum(float(e["usd"]) for e in self.entries()), 12)

    def check(self, estimate_usd: float) -> None:
        """Raise BEFORE a call whose estimated cost would push spend over the cap."""
        spent = self.total()
        if spent + estimate_usd > self.limit_usd:
            raise BudgetExceeded(
                f"live budget would be exceeded: spent ${spent:.4f} + estimated "
                f"${estimate_usd:.4f} > cap ${self.limit_usd:.2f} (LIVE_BUDGET_USD)"
            )

    def record(
        self, *, purpose: str, model_id: str, tokens_in: int, tokens_out: int, usd: float
    ) -> None:
        if not _PURPOSE.match(purpose):
            raise ValueError("purpose must be a short lowercase identifier")
        line = json.dumps(
            {
                "ts": datetime.now(UTC).isoformat(timespec="seconds"),
                "purpose": purpose,
                "model": model_id,
                "tokens_in": int(tokens_in),
                "tokens_out": int(tokens_out),
                "usd": round(float(usd), 12),  # one embedding costs ~1e-7 USD
            }
        )
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8", newline="\n") as f:
                f.write(line + "\n")
