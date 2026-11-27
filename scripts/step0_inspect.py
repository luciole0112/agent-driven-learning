"""Step 0 measurements that are not a comparison method (docs/cifar_mbv2_design_v1.md, section 7).

    python scripts/step0_inspect.py --out runs/mbv2/step0_inspect.json

* S1: dev accuracy, per-class accuracy and confusion on dev (heldout stays sealed
  here; it is scored only in the comparison runs at their end).
* R0 / R0c: mean |activation| per layer and the spread of the logits on dev
  (model 2's initial state).
* Throughput (images/s) of forward and forward+backward for S1 (128px) and R0c (32px).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from adl.baselines.mbv2_methods import accuracy
from adl.data.cifar10 import make_cifar_splits
from adl.models.mobilenetv2 import CLASSES


@torch.no_grad()
def predict(m, x, dev, bs=500):
    return np.concatenate([m(torch.from_numpy(x[i:i + bs]).to(dev)).cpu().numpy() for i in range(0, len(x), bs)])


@torch.no_grad()
def act_profile(m, x, dev):
    store, hooks = {}, []
    for name in m.layer_names(include_out=True):
        mod = m.modules_of(name).get("act")
        if mod is None:
            continue
        hooks.append(mod.register_forward_hook(
            lambda _m, _i, o, n=name: store.__setitem__(n, float(o.abs().mean()))))
    m(torch.from_numpy(x).to(dev))
    for h in hooks:
        h.remove()
    return store


def throughput(m, dev, size_batch=128, reps=10):
    x = torch.randint(0, 255, (size_batch, 3, 32, 32), dtype=torch.uint8, device=dev)
    y = torch.randint(0, 10, (size_batch,), device=dev)
    m.eval()
    with torch.no_grad():
        m(x)
        torch.cuda.synchronize()
        t = time.time()
        for _ in range(reps):
            m(x)
        torch.cuda.synchronize()
        fwd = reps * size_batch / (time.time() - t)
    ps = list(m.parameters())
    for p in ps:
        p.requires_grad_(True)
    m.train()
    F.cross_entropy(m(x), y).backward()
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(reps):
        F.cross_entropy(m(x), y).backward()
    torch.cuda.synchronize()
    bwd = reps * size_batch / (time.time() - t)
    for p in ps:
        p.requires_grad_(False)
        p.grad = None
    m.eval()
    return {"forward_img_per_s": round(fwd), "train_step_img_per_s": round(bwd)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/mbv2")
    ap.add_argument("--out", default="runs/mbv2/step0_inspect.json")
    a = ap.parse_args()
    dev = "cuda"
    sp = make_cifar_splits("data")
    load = lambda n: torch.load(f"{a.runs}/{n}.pt", map_location=dev, weights_only=False).to(dev).eval()
    rep = {}
    s1 = load("S1")
    lg = predict(s1, sp["dev"]["x"], dev)
    pr, y = lg.argmax(1), sp["dev"]["y"]
    conf = np.zeros((10, 10), int)
    np.add.at(conf, (y, pr), 1)
    pairs = sorted(((conf[i, j] + conf[j, i], CLASSES[i], CLASSES[j]) for i in range(10) for j in range(i + 1, 10)),
                   reverse=True)
    rep["S1"] = {"dev_acc": float((pr == y).mean()),
                 "train_acc": accuracy(s1, sp["train"], dev),
                 "per_class_dev": {CLASSES[c]: float(conf[c, c] / conf[c].sum()) for c in range(10)},
                 "confusion_dev": conf.tolist(),
                 "top_confused_pairs_dev": [{"pair": f"{a_}<->{b_}", "errors_both_ways": int(n)} for n, a_, b_ in pairs[:8]],
                 "throughput_128px": throughput(s1, dev)}
    for n in ("R0", "R0c"):
        m = load(n)
        lg = predict(m, sp["dev"]["x"], dev)
        rep[n] = {"dev_acc": float((lg.argmax(1) == y).mean()),
                  "pred_class_hist_dev": np.bincount(lg.argmax(1), minlength=10).tolist(),
                  "logit_std_across_images": float(lg.std(0).mean()),
                  "logit_abs_mean": float(np.abs(lg).mean()),
                  "mean_abs_activation": act_profile(m, sp["dev"]["x"][:500], dev)}
    rep["R0c"]["throughput_32px"] = throughput(load("R0c"), dev)
    Path(a.out).write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
