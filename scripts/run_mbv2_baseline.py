"""Run one non-agent comparison method from a start point.

    python scripts/run_mbv2_baseline.py --start runs/mbv2/S1.pt --method es --space head \
        --budget-images 2000000 --device cuda

Methods: es | random | backprop | ncm | ridge. ``--space`` picks the parameter
subset (see adl/baselines/mbv2_methods.py). Writes a JSON report with the dev
curve against forward images, and the sealed heldout / shift accuracy at the end.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from adl.baselines import mbv2_methods as M
from adl.data.cifar10 import fake_cifar, make_cifar_splits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--method", required=True, choices=["es", "random", "backprop", "ncm", "ridge"])
    ap.add_argument("--space", default="head")
    ap.add_argument("--budget-images", type=int, default=1_000_000)
    ap.add_argument("--pop", type=int, default=20)
    ap.add_argument("--sigma", type=float, default=0.01)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--bn-train", action="store_true", help="backprop with batch statistics (random init)")
    ap.add_argument("--recal-every", type=int, default=0, help="ES: re-estimate BN every N generations")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default="data")
    ap.add_argument("--fake-data", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    splits = M_splits(a)
    model = torch.load(a.start, map_location=a.device, weights_only=False).to(a.device).eval()
    kw = dict(device=a.device)
    if a.method == "es":
        m, rep = M.es(model, splits, a.space, a.budget_images, pop=a.pop, sigma=a.sigma, lr=a.lr or 0.02,
                      batch=a.batch, seed=a.seed, recal_every=a.recal_every, **kw)
    elif a.method == "random":
        m, rep = M.random_search(model, splits, a.space, a.budget_images, sigma=a.sigma, batch=a.batch,
                                 seed=a.seed, **kw)
    elif a.method == "backprop":
        m, rep = M.backprop(model, splits, a.space, a.budget_images, lr=a.lr or 0.05, batch=a.batch,
                            bn_train=a.bn_train, seed=a.seed, **kw)
    elif a.method == "ncm":
        m, rep = M.ncm_head(model, splits, **kw)
    else:
        m, rep = M.ridge_head(model, splits, **kw)
    rep["start"] = a.start
    rep["seed"] = a.seed
    for name in ("dev", "heldout", "shift"):
        rep[f"acc_{name}_initial"] = M.accuracy(model, splits[name], a.device)
        rep[f"acc_{name}_final"] = M.accuracy(m, splits[name], a.device)
    out = Path(a.out or f"runs/mbv2/baselines/{Path(a.start).stem}_{a.method}_{a.space.replace(':', '')}_s{a.seed}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2))
    print(json.dumps({k: v for k, v in rep.items() if k != "curve"}, indent=2))


def M_splits(a):
    return fake_cifar(2000, 500) if a.fake_data else make_cifar_splits(a.data)


if __name__ == "__main__":
    main()
