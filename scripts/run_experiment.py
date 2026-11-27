"""Run one condition of E0/E1 and write a JSON report.

Examples
--------
# baselines (no LLM needed)
python scripts/run_experiment.py --concept glyph --method matched_head
python scripts/run_experiment.py --concept glyph --method es --forward-budget 200

# agent (needs a local Ollama server with a vision + tools model)
python scripts/run_experiment.py --concept glyph --method agent --condition full
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from adl.baselines import methods as B
from adl.data.shapes import ConceptSpec, make_splits
from adl.env.budget import Budget, Ledger
from adl.env.environment import ADLEnvironment
from adl.models.cnn import EditableCNN
from adl.train_base import train_base


def load_base(seed: int, root: Path) -> EditableCNN:
    path = root / f"base_seed{seed}.pt"
    m = EditableCNN(seed=seed)
    if path.exists():
        m.load_state_dict(torch.load(path))
        return m
    m, info = train_base(seed=seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(m.state_dict(), path)
    path.with_suffix(".json").write_text(json.dumps(info, indent=2))
    return m


def concept_from_args(a) -> ConceptSpec:
    if a.concept == "named":
        return ConceptSpec(kind="named", bar_angle_deg=a.bar_angle)
    return ConceptSpec(kind="glyph", glyph_size=a.glyph_size, concept_seed=a.concept_seed,
                       rotate=a.rotate)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--concept", choices=["glyph", "named"], default="glyph")
    ap.add_argument("--concept-seed", type=int, default=0)
    ap.add_argument("--glyph-size", type=int, default=5)
    ap.add_argument("--rotate", action="store_true")
    ap.add_argument("--bar-angle", type=float, default=45.0)
    ap.add_argument("--n-train", type=int, default=200)
    ap.add_argument("--base-seed", type=int, default=0)
    ap.add_argument("--method", default="matched_head",
                    choices=["none", "matched_head", "matched_channel", "random", "es",
                             "backprop_head", "backprop_channel", "backprop_full", "agent"])
    ap.add_argument("--condition", default="full", choices=["full", "no_semantics", "no_observation"])
    ap.add_argument("--forbid-fitting", action="store_true")
    ap.add_argument("--forward-budget", type=int, default=200)
    ap.add_argument("--eval-budget", type=int, default=60)
    ap.add_argument("--max-turns", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs")
    a = ap.parse_args()

    root = Path(a.out)
    base = load_base(a.base_seed, root / "base")
    splits = make_splits(concept_from_args(a), n_train=a.n_train)
    tag = f"{a.concept}{a.concept_seed}_{a.method}_{a.condition}_s{a.seed}_{int(time.time())}"
    run_dir = root / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    budget = Budget(forward_passes=a.forward_budget, evaluations=a.eval_budget)
    t0 = time.time()

    if a.method == "agent":
        from adl.agent.loop import run_agent
        env = ADLEnvironment(base, splits, budget, Ledger(run_dir / "ledger.jsonl"),
                             condition=a.condition, forbid_fitting=a.forbid_fitting)
        report = run_agent(env, max_turns=a.max_turns, transcript_path=str(run_dir / "transcript.json"))
    else:
        fns = {
            "none": lambda: base,
            "matched_head": lambda: B.matched_filter_head(base, splits["train"]),
            "matched_channel": lambda: B.matched_filter_channel(base, splits["train"]),
            "random": lambda: B.random_edits(base, splits["dev"], budget, seed=a.seed)[0],
            "es": lambda: B.evolution_strategy(base, splits["dev"], budget, seed=a.seed)[0],
            "backprop_head": lambda: B.backprop_reference(base, splits["train"], head_only=True, seed=a.seed),
            "backprop_channel": lambda: B.backprop_reference(base, splits["train"], seed=a.seed),
            "backprop_full": lambda: B.backprop_full(base, splits["train"], seed=a.seed),
        }
        model = fns[a.method]()
        env = ADLEnvironment(model, splits, Budget(), Ledger())
        env.base_acc_initial = ADLEnvironment(base, splits, Budget(), Ledger()).base_acc_initial
        report = env.final_report()
        report["budget_used"] = dict(budget.used)
    report.update({"method": a.method, "concept": a.concept, "concept_seed": a.concept_seed,
                   "seed": a.seed, "n_train": a.n_train, "wall_s": round(time.time() - t0, 1)})
    (run_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in report.items() if k != "diff_from_initial"}, indent=2))


if __name__ == "__main__":
    main()
