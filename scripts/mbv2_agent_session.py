"""Replay driver for a session in which Claude itself acts as the agent on
MobileNetV2 / CIFAR-10 (the method that worked in PR 5).

Claude writes its tool calls to <run_dir>/actions.json, one batch at a time;
this script rebuilds the environment from the start point and replays every
call through ``env.call`` (so budgets, protocol, guard and sealed splits apply).
Images returned by tools go to <run_dir>/img/.

    python scripts/mbv2_agent_session.py runs/agent_S1 --start runs/mbv2/S1.pt
    python scripts/mbv2_agent_session.py runs/agent_S1 --start runs/mbv2/S1.pt --final

Replaying re-runs every forward pass; on CPU keep --pool-n / --dev-n small, or
run on the GPU.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from adl.data.cifar10 import fake_cifar, make_cifar_splits
from adl.env.budget import Budget, Ledger
from adl.env.mbv2_env import MBV2Environment, MBV2Guard


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--start", required=True)
    ap.add_argument("--data", default="data")
    ap.add_argument("--fake-data", action="store_true")
    ap.add_argument("--condition", default="full")
    ap.add_argument("--guard-drop", type=float, default=0.02, help="negative = no accuracy guard (model 2)")
    ap.add_argument("--no-protocol", action="store_true")
    ap.add_argument("--budget-images", type=int, default=2_000_000)
    ap.add_argument("--evaluations", type=int, default=60)
    ap.add_argument("--pool-n", type=int, default=5000)
    ap.add_argument("--dev-n", type=int, default=5000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--from-step", type=int, default=None)
    a = ap.parse_args()
    run = Path(a.run_dir)
    actions = json.loads((run / "actions.json").read_text())
    splits = fake_cifar(2000, 500) if a.fake_data else make_cifar_splits(a.data, dev_n=a.dev_n)
    model = torch.load(a.start, map_location=a.device, weights_only=False)
    guard = None if a.no_protocol else MBV2Guard(max_acc_drop=None if a.guard_drop < 0 else a.guard_drop)
    ledger = run / "ledger.jsonl"
    if ledger.exists():
        ledger.unlink()
    env = MBV2Environment(model, splits, Budget(evaluations=a.evaluations, forward_images=a.budget_images,
                                                forward_passes=10**9),
                          Ledger(ledger), condition=a.condition, guard=guard, pool_n=a.pool_n, device=a.device)
    img = run / "img"
    img.mkdir(parents=True, exist_ok=True)
    start = a.from_step if a.from_step is not None else max(
        [i for i, s in enumerate(actions) if s.get("batch_start")] or [0])
    results = []
    for i, step in enumerate(actions):
        res = env.call(step["tool"], step.get("args", {}), step.get("rationale", ""))
        paths = []
        for k, png in enumerate(res.images):
            p = img / f"step_{i:03d}_{k}.png"
            p.write_bytes(png)
            paths.append(str(p))
        results.append({"step": i, "tool": step["tool"], "args": step.get("args", {}),
                        "rationale": step.get("rationale", ""), "text": res.text, "images": paths})
        if i >= start:
            print(f"--- [{i}] {step['tool']} {json.dumps(step.get('args', {}), ensure_ascii=False)[:300]}")
            print(res.text[:4000])
            for p in paths:
                print("   image:", p)
    (run / "transcript.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print("\n" + env.protocol_status())
    print("budget remaining:", env.budget.remaining())
    if a.final:
        rep = env.final_report()
        (run / "report.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False))
        print(json.dumps({k: v for k, v in rep.items() if k != "cycles"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
