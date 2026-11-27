"""Comparison methods for the MobileNetV2 / CIFAR-10 experiments.

All methods spend the same unit as the agent: ``forward_images`` (images
pushed through the network; a backward pass counts as 2 more). They act on a
chosen *parameter subset* so that the agent's edit space can be matched:

* ``"head"``           : head weight + bias (12,810 values)
* ``"last+head"``      : ``last`` conv + BN affine + head
* ``"bn"``             : every BN gamma/beta (the "affine-only" space)
* ``"blocks:a-b"``     : every weight of blocks a..b (plus their BN affine)
* ``"stem"``           : stem conv + BN affine (864 + 64 values)
* ``"all"``            : every trainable parameter
* a comma list of agent tensor names (``b3.dw.weight,head.weight``)

Methods: ``es`` (antithetic OpenAI-ES), ``random`` (random-perturbation hill
climbing), ``backprop`` (SGD with momentum; the reference that uses
gradients), and fixed rules for the head (``ncm``: nearest class mean,
``ridge``: closed-form ridge regression on pooled features).

These functions work on a copy of the model and never touch the agent's one.
"""

from __future__ import annotations

import copy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from adl.models.mobilenetv2 import MBV2


# ------------------------------------------------------------------ helpers
def param_subset(model: MBV2, spec: str) -> list[torch.Tensor]:
    named = model.named_tensors()
    affine = {k: v for k, v in named.items() if not k.endswith(("running_mean", "running_var"))}
    if spec == "all":
        return list(model.net.parameters())
    if spec == "head":
        keys = ["head.weight", "head.bias"]
    elif spec == "last+head":
        keys = ["last.weight", "last.bn.weight", "last.bn.bias", "head.weight", "head.bias"]
    elif spec == "bn":
        keys = [k for k in affine if ".bn." in k]
    elif spec == "stem":
        keys = ["stem.weight", "stem.bn.weight", "stem.bn.bias"]
    elif spec.startswith("blocks:"):
        a, b = (int(v) for v in spec.split(":")[1].split("-"))
        keys = [k for k in affine if k.startswith("b") and k[1:].split(".")[0].isdigit()
                and a <= int(k[1:].split(".")[0]) <= b]
    else:
        keys = [k.strip() for k in spec.split(",")]
    missing = [k for k in keys if k not in affine]
    if missing:
        raise KeyError(f"unknown tensors {missing}")
    return [affine[k] for k in keys]


def n_values(params: list[torch.Tensor]) -> int:
    return int(sum(p.numel() for p in params))


def _flat(params) -> torch.Tensor:
    return torch.cat([p.reshape(-1) for p in params])


def _assign(params, vec: torch.Tensor) -> None:
    i = 0
    for p in params:
        n = p.numel()
        p.copy_(vec[i:i + n].view_as(p))
        i += n


def _to(x: np.ndarray, device) -> torch.Tensor:
    return torch.from_numpy(x).to(device)


@torch.no_grad()
def accuracy(model: MBV2, d: dict, device="cpu", bs: int = 512) -> float:
    model.eval()
    pr = [model(_to(d["x"][i:i + bs], device)).argmax(1).cpu() for i in range(0, len(d["y"]), bs)]
    return float((torch.cat(pr).numpy() == d["y"]).mean())


@torch.no_grad()
def pooled_features(model: MBV2, x: np.ndarray, device="cpu", bs: int = 512) -> np.ndarray:
    model.eval()
    return torch.cat([model.features(_to(x[i:i + bs], device)).cpu() for i in range(0, len(x), bs)]).numpy()


@torch.no_grad()
def recalibrate_bn(model: MBV2, x: np.ndarray, n: int = 2000, device="cpu", bs: int = 250) -> int:
    """Forward-only re-estimation of BN running statistics (cumulative average). Returns images used."""
    bns = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    saved = [m.momentum for m in bns]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None
    model.train()
    for i in range(0, min(n, len(x)), bs):
        model(_to(x[i:i + bs], device))
    model.eval()
    for m, mo in zip(bns, saved):
        m.momentum = mo
    return min(n, len(x))


class Meter:
    """Counts forward images and records an accuracy curve."""

    def __init__(self, budget_images: int, eval_every_images: int, dev: dict, device):
        self.budget, self.every, self.dev, self.device = budget_images, eval_every_images, dev, device
        self.used, self.next_eval, self.curve = 0, 0, []
        self.t0 = time.time()

    def charge(self, n: int) -> None:
        self.used += n

    def left(self) -> bool:
        return self.used < self.budget

    def maybe_eval(self, model, force: bool = False) -> None:
        if force or self.used >= self.next_eval:
            self.curve.append({"forward_images": self.used, "dev_acc": accuracy(model, self.dev, self.device),
                               "sec": round(time.time() - self.t0, 1)})
            self.next_eval = self.used + self.every


# ------------------------------------------------------------------ zero-order
@torch.no_grad()
def es(model: MBV2, splits: dict, spec: str, budget_images: int, pop: int = 20, sigma: float = 0.01,
       lr: float = 0.02, batch: int = 128, seed: int = 0, device="cpu", eval_every: int | None = None,
       recal_every: int = 0) -> tuple[MBV2, dict]:
    """Antithetic OpenAI-ES on train-minibatch cross-entropy, rank-normalised
    fitness, Adam on the search gradient estimate. ``recal_every`` > 0 re-estimates
    BN statistics every that many generations (charged)."""
    m = copy.deepcopy(model).to(device).eval()
    params = param_subset(m, spec)
    theta = _flat(params).clone()
    g = torch.Generator(device="cpu").manual_seed(seed)
    rng = np.random.default_rng(seed)
    meter = Meter(budget_images, eval_every or max(budget_images // 20, 1), splits["dev"], device)
    adam_m, adam_v, t = torch.zeros_like(theta), torch.zeros_like(theta), 0
    tr = splits["train"]
    half = pop // 2
    meter.maybe_eval(m)
    gen = 0
    while meter.left():
        idx = rng.choice(len(tr["y"]), batch, replace=False)
        xb, yb = _to(tr["x"][idx], device), _to(tr["y"][idx], device)
        eps = torch.randn(half, theta.numel(), generator=g).to(theta)
        losses = torch.zeros(2 * half)
        for i in range(half):
            for s, j in ((1, i), (-1, half + i)):
                _assign(params, theta + s * sigma * eps[i])
                losses[j] = F.cross_entropy(m(xb), yb).item()
        meter.charge(2 * half * batch)
        ranks = torch.empty(2 * half)
        ranks[losses.argsort()] = torch.linspace(0.5, -0.5, 2 * half)    # low loss -> high utility
        util = ranks[:half] - ranks[half:]
        grad = -(util.to(theta) @ eps) / (2 * half * sigma)              # descent direction on loss
        t += 1
        adam_m = 0.9 * adam_m + 0.1 * grad
        adam_v = 0.999 * adam_v + 0.001 * grad ** 2
        theta -= lr * (adam_m / (1 - 0.9 ** t)) / (torch.sqrt(adam_v / (1 - 0.999 ** t)) + 1e-8)
        _assign(params, theta)
        gen += 1
        if recal_every and gen % recal_every == 0:
            meter.charge(recalibrate_bn(m, tr["x"][rng.permutation(len(tr["y"]))[:1000]], 1000, device))
        meter.maybe_eval(m)
    meter.maybe_eval(m, force=True)
    return m, {"method": "es", "space": spec, "n_values": theta.numel(), "pop": pop, "sigma": sigma,
               "lr": lr, "batch": batch, "generations": gen, "forward_images": meter.used, "curve": meter.curve}


@torch.no_grad()
def random_search(model: MBV2, splits: dict, spec: str, budget_images: int, sigma: float = 0.01,
                  batch: int = 256, seed: int = 0, device="cpu", eval_every: int | None = None) -> tuple[MBV2, dict]:
    """(1+1) random-perturbation hill climbing: keep a perturbation if the loss
    on a fresh train minibatch (evaluated for both) improves."""
    m = copy.deepcopy(model).to(device).eval()
    params = param_subset(m, spec)
    theta = _flat(params).clone()
    g = torch.Generator(device="cpu").manual_seed(seed)
    rng = np.random.default_rng(seed)
    meter = Meter(budget_images, eval_every or max(budget_images // 20, 1), splits["dev"], device)
    tr = splits["train"]
    accepted = steps = 0
    meter.maybe_eval(m)
    while meter.left():
        idx = rng.choice(len(tr["y"]), batch, replace=False)
        xb, yb = _to(tr["x"][idx], device), _to(tr["y"][idx], device)
        _assign(params, theta)
        l0 = F.cross_entropy(m(xb), yb).item()
        cand = theta + sigma * torch.randn(theta.numel(), generator=g).to(theta)
        _assign(params, cand)
        l1 = F.cross_entropy(m(xb), yb).item()
        meter.charge(2 * batch)
        steps += 1
        if l1 < l0:
            theta, accepted = cand, accepted + 1
        _assign(params, theta)
        meter.maybe_eval(m)
    meter.maybe_eval(m, force=True)
    return m, {"method": "random", "space": spec, "n_values": theta.numel(), "sigma": sigma, "steps": steps,
               "accepted": accepted, "forward_images": meter.used, "curve": meter.curve}


# ------------------------------------------------------------------ gradient reference
def backprop(model: MBV2, splits: dict, spec: str, budget_images: int, lr: float = 0.05,
             batch: int = 128, momentum: float = 0.9, wd: float = 4e-5, bn_train: bool = False,
             augment: bool = True, seed: int = 0, device="cpu", eval_every: int | None = None) -> tuple[MBV2, dict]:
    """SGD on the subset. Each step costs 3 x batch forward images (forward +
    backward counted as two). ``bn_train`` uses batch statistics (needed from
    random init); otherwise BN is frozen as for the agent."""
    torch.manual_seed(seed)
    m = copy.deepcopy(model).to(device)
    params = param_subset(m, spec)
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=wd)
    steps_total = max(1, budget_images // (3 * batch))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps_total)
    rng = np.random.default_rng(seed)
    meter = Meter(budget_images, eval_every or max(budget_images // 20, 1), splits["dev"], device)
    tr = splits["train"]
    meter.maybe_eval(m)
    steps = 0
    while meter.left():
        idx = rng.choice(len(tr["y"]), batch, replace=False)
        xb = _to(tr["x"][idx], device)
        if augment:
            xb = _augment(xb, rng)
        m.train(bn_train)
        loss = F.cross_entropy(m(xb), _to(tr["y"][idx], device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        meter.charge(3 * batch)
        steps += 1
        m.eval()
        meter.maybe_eval(m)
    for p in m.parameters():
        p.requires_grad_(False)
        p.grad = None
    meter.maybe_eval(m, force=True)
    return m, {"method": "backprop", "space": spec, "n_values": n_values(params), "lr": lr, "batch": batch,
               "bn_train": bn_train, "steps": steps, "forward_images": meter.used, "curve": meter.curve}


def _augment(xb: torch.Tensor, rng) -> torch.Tensor:
    """Random crop (pad 4) and horizontal flip, on uint8 batches."""
    p = F.pad(xb, (4, 4, 4, 4), mode="reflect" if xb.dtype != torch.uint8 else "constant")
    out = torch.empty_like(xb)
    for i in range(len(xb)):
        dy, dx = rng.integers(0, 9, 2)
        img = p[i, :, dy:dy + 32, dx:dx + 32]
        out[i] = img.flip(-1) if rng.random() < 0.5 else img
    return out


# ------------------------------------------------------------------ fixed rules for the head
@torch.no_grad()
def ncm_head(model: MBV2, splits: dict, device="cpu") -> tuple[MBV2, dict]:
    """Nearest-class-mean head on pooled features (cosine; no gradient, no fitting)."""
    m = copy.deepcopy(model).to(device).eval()
    f = pooled_features(m, splits["train"]["x"], device)
    y = splits["train"]["y"]
    mu = f.mean(0)
    cm = np.stack([f[y == c].mean(0) - mu for c in range(10)])
    cm /= np.linalg.norm(cm, axis=1, keepdims=True) + 1e-8
    m.net.classifier[1].weight.copy_(torch.from_numpy(cm.astype(np.float32)))
    m.net.classifier[1].bias.copy_(torch.from_numpy((-cm @ mu).astype(np.float32)))
    return m, {"method": "ncm", "space": "head", "forward_images": len(y)}


@torch.no_grad()
def ridge_head(model: MBV2, splits: dict, lam: float = 1.0, device="cpu") -> tuple[MBV2, dict]:
    """Closed-form ridge regression of one-hot labels on pooled features (a fitting rule)."""
    m = copy.deepcopy(model).to(device).eval()
    f = pooled_features(m, splits["train"]["x"], device).astype(np.float64)
    y = splits["train"]["y"]
    mu, sd = f.mean(0), f.std(0) + 1e-6
    z = (f - mu) / sd
    t = np.eye(10)[y] - 0.1
    w = np.linalg.solve(z.T @ z + lam * len(y) * np.eye(z.shape[1]), z.T @ t)      # (d, 10)
    W = (w / sd[:, None]).T
    b = -(W @ mu)
    m.net.classifier[1].weight.copy_(torch.from_numpy(W.astype(np.float32)))
    m.net.classifier[1].bias.copy_(torch.from_numpy(b.astype(np.float32)))
    return m, {"method": "ridge", "space": "head", "lam": lam, "forward_images": len(y)}


def linear_probe(model: MBV2, splits: dict, epochs: int = 30, lr: float = 1e-3, wd: float = 1e-4,
                 device="cpu", seed: int = 0) -> tuple[MBV2, dict]:
    """Start point S1: train only the head by backprop on frozen pooled features (no augmentation)."""
    torch.manual_seed(seed)
    m = copy.deepcopy(model).to(device).eval()
    f = torch.from_numpy(pooled_features(m, splits["train"]["x"], device)).to(device)
    y = torch.from_numpy(splits["train"]["y"]).to(device)
    lin = m.net.classifier[1]
    lin.weight.requires_grad_(True)
    lin.bias.requires_grad_(True)
    opt = torch.optim.AdamW(lin.parameters(), lr=lr, weight_decay=wd)
    for _ in range(epochs):
        perm = torch.randperm(len(y), device=device)
        for i in range(0, len(y), 256):
            j = perm[i:i + 256]
            loss = F.cross_entropy(lin(f[j]), y[j])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    lin.weight.requires_grad_(False)
    lin.bias.requires_grad_(False)
    lin.weight.grad = lin.bias.grad = None
    return m, {"method": "linear_probe", "epochs": epochs, "dev_acc": accuracy(m, splits["dev"], device)}
