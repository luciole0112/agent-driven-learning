"""Reference runs in the SAME edit space Claude used as the agent.

Claude's edits added a new pathway (2 conv1 channels -> 1 conv2 channel ->
1 conv3 channel -> head_concept) and left every existing weight untouched.
The stock baselines (ES, backprop_channel) only edit one new conv3 channel, so
they cannot build an orientation detector at conv1. To separate "the edit
space" from "the knowledge used to fill it", this script fills the same
pathway (same shapes, random init) by

* backprop (gradient reference, not an ADL method) on the 200 train images, and
* the same (1+lambda)-ES as adl.baselines, with fitness = dev AUROC and the
  same budget of 200 forward passes over dev.

    python scripts/same_space_reference.py --out runs/same_space
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_experiment import load_base  # noqa: E402

from adl.data.shapes import ConceptSpec, make_splits  # noqa: E402
from adl.env.budget import Budget, Ledger  # noqa: E402
from adl.env.environment import ADLEnvironment  # noqa: E402
from adl.env.metrics import auroc  # noqa: E402
from adl.models.cnn import EditableCNN  # noqa: E402


class PathwayParam:
    """theta <-> the new pathway's weights (existing weights stay fixed)."""

    def __init__(self, base: EditableCNN):
        m = base.clone()
        self.c1 = [m.add_channel("conv1"), m.add_channel("conv1")]
        self.c2 = m.add_channel("conv2")
        self.c3 = m.add_channel("conv3")
        self.base = m
        p = m.params
        self.slots = [
            ("conv1.weight", (slice(self.c1[0], self.c1[1] + 1),), (2, 3, 5, 5)),
            ("conv1.bias", (slice(self.c1[0], self.c1[1] + 1),), (2,)),
            ("conv2.weight", (self.c2,), tuple(p["conv2.weight"].shape[1:])),
            ("conv2.bias", (self.c2,), ()),
            ("conv3.weight", (self.c3,), tuple(p["conv3.weight"].shape[1:])),
            ("conv3.bias", (self.c3,), ()),
            ("head_concept.weight", (0, self.c3), ()),
            ("head_concept.bias", (0,), ()),
        ]
        self.sizes = [int(np.prod(s)) if s else 1 for _, _, s in self.slots]
        self.dim = sum(self.sizes)

    def params(self, theta: torch.Tensor) -> dict:
        out = dict(self.base.params)
        i = 0
        for (name, idx, shape), n in zip(self.slots, self.sizes):
            t = out[name].clone()
            t[idx] = theta[i:i + n].reshape(shape) if shape else theta[i]
            out[name] = t
            i += n
        return out

    def build(self, theta) -> EditableCNN:
        m = EditableCNN.__new__(EditableCNN)
        m.params = {k: v.detach().clone() for k, v in self.params(torch.as_tensor(theta, dtype=torch.float32)).items()}
        return m

    def init(self, rng: np.random.Generator) -> np.ndarray:
        parts = []
        for (name, _, shape), n in zip(self.slots, self.sizes):
            if name.endswith("weight") and name != "head_concept.weight":
                fan_in = int(np.prod(shape[1:] if len(shape) == 4 else shape))
                parts.append(rng.normal(0, (2.0 / fan_in) ** 0.5, n))
            elif name == "head_concept.weight":
                parts.append(np.full(n, 0.5))
            elif name.endswith("bias") and name != "head_concept.bias":
                parts.append(np.full(n, 0.1))
            else:
                parts.append(np.zeros(n))
        return np.concatenate(parts)


def logits(pp: PathwayParam, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    m = EditableCNN.__new__(EditableCNN)
    m.params = pp.params(theta)
    return torch.cat([m(x[i:i + 256])[1] for i in range(0, len(x), 256)])


def score(model: EditableCNN, base: EditableCNN, splits: dict) -> dict:
    env = ADLEnvironment(model, splits, Budget(), Ledger())
    rep = env.final_report()
    return {k: rep[k] for k in ("auroc_train", "auroc_dev", "auroc_heldout", "auroc_shift", "base_acc_final")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/same_space")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--no-es", action="store_true")
    a = ap.parse_args()
    base = load_base(0, Path("runs") / "base")
    splits = make_splits(ConceptSpec(kind="named", bar_angle_deg=45.0), n_train=200)
    pp = PathwayParam(base)
    xtr = torch.from_numpy(splits["train"]["x"])
    ytr = torch.from_numpy(splits["train"]["y_concept"]).float()
    xdv = splits["dev"]["x"]
    ydv = splits["dev"]["y_concept"]
    results = {"dim": pp.dim, "lr": a.lr, "steps": a.steps, "backprop": [], "es": []}
    for seed in range(a.seeds):
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        theta = torch.tensor(pp.init(rng), dtype=torch.float32, requires_grad=True)
        opt = torch.optim.Adam([theta], lr=a.lr)
        for _ in range(a.steps):
            loss = F.binary_cross_entropy_with_logits(logits(pp, theta, xtr), ytr)
            opt.zero_grad(); loss.backward(); opt.step()
        r = score(pp.build(theta.detach()), base, splits)
        r["seed"] = seed
        results["backprop"].append(r)
        print("backprop", r, flush=True)

        if a.no_es:
            continue
        # (1+lambda)-ES, same rule as adl.baselines.methods.evolution_strategy
        budget = Budget(forward_passes=200)
        xdv_t = torch.from_numpy(xdv)

        def fit(th):
            budget.charge("forward_passes")
            with torch.no_grad():
                return auroc(logits(pp, torch.as_tensor(th, dtype=torch.float32), xdv_t).numpy(), ydv)
        mean = pp.init(rng)
        best = fit(mean)
        sigma = 0.3
        while budget.remaining()["forward_passes"] >= 1:
            succ = 0
            for _ in range(8):
                if budget.remaining()["forward_passes"] < 1:
                    break
                cand = mean + sigma * rng.standard_normal(pp.dim)
                f = fit(cand)
                if f > best:
                    mean, best, succ = cand, f, succ + 1
            sigma *= 1.5 if succ / 8 > 0.2 else 0.85
        r = score(pp.build(mean), base, splits)
        r["seed"] = seed
        results["es"].append(r)
        print("es", r, flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
