"""Build the start points of the MobileNetV2 / CIFAR-10 experiments.

    python scripts/prepare_mbv2_cifar.py --start S0 S1 S2 R0 --device cuda

Start points (docs/cifar_mbv2_design_v1.md, section 3):

* S0: ImageNet MobileNetV2 trunk + zero head (chance accuracy).
* S1: S0 + head trained by backprop on frozen features (linear probe). Main start of model 1.
* S2: S1 fine-tuned end to end by backprop on CIFAR-10. Start of the later "interpret and modify" phase.
* R0: random init (CIFAR variant), BN statistics as initialised.
* R0c: R0 with BN statistics re-estimated by forward passes (no gradient).

Backprop is used here only to build the starting models, never on the agent's run.
Use ``--fake-data`` to try the pipeline without downloading CIFAR-10.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from adl.baselines.mbv2_methods import accuracy, backprop, linear_probe, recalibrate_bn
from adl.data.cifar10 import fake_cifar, make_cifar_splits
from adl.models.mobilenetv2 import MBV2


def load_splits(a):
    return fake_cifar(2000, 500) if a.fake_data else make_cifar_splits(a.data)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", nargs="+", default=["S0", "S1", "R0", "R0c"])
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="runs/mbv2")
    ap.add_argument("--input-size", type=int, default=128)
    ap.add_argument("--weights-path", default=None, help="local ImageNet weights (.pth) if offline")
    ap.add_argument("--ft-epochs", type=float, default=10)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fake-data", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    splits = load_splits(a)
    info = {}

    def save(name, m, extra):
        m = m.cpu().eval()
        torch.save(m, out / f"{name}.pt")
        m.to(a.device)
        info[name] = {"dev_acc": accuracy(m, splits["dev"], a.device), "n_params": m.n_params(), **extra}
        m.cpu()
        print(name, info[name], flush=True)

    need_imnet = {"S0", "S1", "S2"} & set(a.start)
    if need_imnet:
        s0 = MBV2.imagenet_pretrained(a.input_size, weights_path=a.weights_path).to(a.device)
        if "S0" in a.start:
            save("S0", s0, {"input_size": a.input_size})
        s1, rep = linear_probe(s0, splits, device=a.device, seed=a.seed)
        if "S1" in a.start or "S2" in a.start:
            save("S1", s1, rep)
        if "S2" in a.start:
            budget = int(a.ft_epochs * len(splits["train"]["y"]) * 3)
            s2, rep = backprop(s1, splits, "all", budget, lr=0.01, batch=128, bn_train=True,
                               device=a.device, seed=a.seed)
            rep.pop("curve")
            save("S2", s2, rep)
    if {"R0", "R0c"} & set(a.start):
        r0 = MBV2.random_init(a.seed).to(a.device)
        if "R0" in a.start:
            save("R0", r0, {"seed": a.seed})
        if "R0c" in a.start:
            recalibrate_bn(r0, splits["train"]["x"], 2000, a.device)
            save("R0c", r0, {"seed": a.seed, "bn_recalibrated_on": 2000})
    prev = json.loads((out / "start_points.json").read_text()) if (out / "start_points.json").exists() else {}
    (out / "start_points.json").write_text(json.dumps({**prev, **info}, indent=2))


if __name__ == "__main__":
    main()
