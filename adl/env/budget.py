"""Budgets, metering and the ledger.

Every agent action is charged against a :class:`Budget` and recorded in the
:class:`Ledger`. The ledger is research data: it records hypothesis, edit,
prediction, outcome and acceptance for each step, plus audit flags (e.g.
fitting operations detected in sandbox code).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class Budget:
    forward_passes: int = 200      # probe / evaluate calls (one batch each)
    evaluations: int = 60          # evaluate() calls on dev
    weight_values_read: int = 200_000
    sandbox_runs: int = 100
    edits: int = 200
    used: dict = field(default_factory=lambda: {
        "forward_passes": 0, "evaluations": 0, "weight_values_read": 0,
        "sandbox_runs": 0, "edits": 0})

    def charge(self, kind: str, amount: int = 1) -> None:
        limit = getattr(self, kind)
        if self.used[kind] + amount > limit:
            raise BudgetExceeded(
                f"budget '{kind}' exhausted ({self.used[kind]}/{limit})")
        self.used[kind] += amount

    def remaining(self) -> dict:
        return {k: getattr(self, k) - v for k, v in self.used.items()}


@dataclass
class LedgerEntry:
    step: int
    tool: str
    args: dict
    result_summary: str
    rationale: str = ""
    flags: list = field(default_factory=list)
    t: float = field(default_factory=time.time)


class Ledger:
    def __init__(self, path: str | Path | None = None):
        self.entries: list[LedgerEntry] = []
        self.path = Path(path) if path else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, tool: str, args: dict, result_summary: str,
            rationale: str = "", flags: list | None = None) -> None:
        e = LedgerEntry(len(self.entries), tool, _jsonable(args),
                        result_summary[:2000], rationale, flags or [])
        self.entries.append(e)
        if self.path:
            with self.path.open("a") as f:
                f.write(json.dumps(asdict(e), ensure_ascii=False) + "\n")

    def flags(self) -> list[str]:
        return [fl for e in self.entries for fl in e.flags]


def _jsonable(obj):
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        if isinstance(obj, dict):
            return {k: _jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_jsonable(v) for v in obj]
        return repr(obj)[:200]
