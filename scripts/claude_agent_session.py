"""Replay driver for the session in which Claude itself acted as the ADL agent.

Claude (the assistant in this session) called the environment's tools directly
from Python instead of through a local VLM. Because each shell call is a fresh
process, the session is stored as an ordered list of tool calls
(``actions.json``); this script rebuilds the environment and replays every call
deterministically. Only the environment's public tools (``env.call``) are used,
so budgets, guardrails, the hypothesis/evaluate/decide protocol and the sealed
heldout/shift splits all apply exactly as for the VLM agent.

    python scripts/claude_agent_session.py runs/claude_agent            # replay, print new steps
    python scripts/claude_agent_session.py runs/claude_agent --final    # also score sealed splits

Images returned by tools are written to <dir>/img/step_XXX_k.png so they can be
looked at.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from adl.data.shapes import ConceptSpec, make_splits
from adl.env.budget import Budget, Ledger
from adl.env.environment import ADLEnvironment
from adl.env.guard import Guard
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_experiment import load_base  # noqa: E402


def build_env(run_dir: Path) -> ADLEnvironment:
    base = load_base(0, Path("runs") / "base")
    splits = make_splits(ConceptSpec(kind="named", bar_angle_deg=45.0), n_train=200)
    ledger_path = run_dir / "ledger.jsonl"
    if ledger_path.exists():
        ledger_path.unlink()
    return ADLEnvironment(base, splits, Budget(forward_passes=200, evaluations=60),
                          Ledger(ledger_path), condition="full", guard=Guard(max_base_drop=0.02))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--from-step", type=int, default=None, help="print steps from this index (default: last batch)")
    ap.add_argument("--final", action="store_true")
    a = ap.parse_args()
    run_dir = Path(a.run_dir)
    actions = json.loads((run_dir / "actions.json").read_text())
    env = build_env(run_dir)
    img_dir = run_dir / "img"
    img_dir.mkdir(parents=True, exist_ok=True)
    start = a.from_step if a.from_step is not None else max(
        [i for i, s in enumerate(actions) if s.get("batch_start")] or [0])
    results = []
    for i, step in enumerate(actions):
        res = env.call(step["tool"], step.get("args", {}), step.get("rationale", ""))
        imgs = []
        for k, png in enumerate(res.images):
            p = img_dir / f"step_{i:03d}_{k}.png"
            p.write_bytes(png)
            imgs.append(str(p))
        results.append({"step": i, "tool": step["tool"], "args": step.get("args", {}),
                        "rationale": step.get("rationale", ""), "text": res.text, "images": imgs})
        if i >= start:
            print(f"--- [{i}] {step['tool']} {json.dumps(step.get('args', {}), ensure_ascii=False)[:300]}")
            print(res.text[:4000])
            for p in imgs:
                print("   image:", p)
    (run_dir / "transcript.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print("\n" + env.protocol_status())
    print("budget remaining:", env.budget.remaining())
    if a.final:
        rep = env.final_report()
        (run_dir / "report.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False))
        print(json.dumps({k: v for k, v in rep.items() if k not in ("cycles",)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
