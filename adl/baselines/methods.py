"""Non-agent comparison methods (conditions D, E, F, G of the research plan).

All of them act on the *same operation space* that the agent's core
structural move uses: one new conv3 channel (kernel over conv2 features +
bias) connected to ``head_concept`` with a weight and the head bias. Head-only
variants act on ``head_concept`` alone.

They charge the same :class:`Budget` (forward passes = dev evaluations) so
that sample efficiency can be compared at equal budget.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from adl.env.budget import Budget
from adl.env.metrics import auroc
from adl.models.cnn import EditableCNN


def _pooled(model: EditableCNN, x: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        xt = torch.from_numpy(x)
        return torch.cat([model(xt[i:i + 256], return_acts=True)[2]["pooled"]
                          for i in range(0, len(xt), 256)]).numpy()


def _concept_scores(model: EditableCNN, x: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        xt = torch.from_numpy(x)
        return torch.cat([model(xt[i:i + 256])[1] for i in range(0, len(xt), 256)]).numpy()


# --------------------------------------------------------------- F: matched
def matched_filter_head(model: EditableCNN, train: dict) -> EditableCNN:
    """Closed-form, fixed rule: head_concept.w = mean(pos) - mean(neg) of the
    frozen pooled features, bias at the midpoint. Tests whether the frozen
    representation already contains the concept (if this succeeds the concept
    is *not* novel to the model and the task is a mere re-routing)."""
    m = model.clone()
    f = _pooled(m, train["x"])
    y = train["y_concept"].astype(bool)
    d = f[y].mean(0) - f[~y].mean(0)
    mid = 0.5 * (f[y].mean(0) + f[~y].mean(0))
    m.params["head_concept.weight"] = torch.from_numpy(d[None].astype(np.float32))
    m.params["head_concept.bias"] = torch.tensor([-float(d @ mid)])
    return m


def matched_filter_channel(model: EditableCNN, train: dict) -> EditableCNN:
    """Fixed rule one level deeper: a new conv3 channel whose kernel is the
    mean conv2-feature patch at each positive image's most distinctive
    location minus the mean patch over negatives, connected to the head."""
    m = model.clone()
    with torch.no_grad():
        xt = torch.from_numpy(train["x"])
        # conv2 output after pooling (= conv3 input)
        acts = torch.cat([F.max_pool2d(m(xt[i:i + 256], return_acts=True)[2]["conv2"], 2)
                          for i in range(0, len(xt), 256)]).numpy()
    y = train["y_concept"].astype(bool)
    k = m.params["conv3.weight"].shape[-1]
    n, c, h, w = acts.shape
    pad = k // 2
    ap = np.pad(acts, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    patches = np.lib.stride_tricks.sliding_window_view(ap, (k, k), axis=(2, 3))  # n,c,h,w,k,k
    patches = patches.transpose(0, 2, 3, 1, 4, 5).reshape(n, h * w, c * k * k)
    neg_mean = patches[~y].reshape(-1, c * k * k).mean(0)
    # pick, per positive image, the location most unlike the negative mean
    dist = np.linalg.norm(patches[y] - neg_mean, axis=-1)
    sel = patches[y][np.arange(y.sum()), dist.argmax(1)]
    kern = (sel.mean(0) - neg_mean)
    kern = kern / (np.linalg.norm(kern) + 1e-8)
    resp_pos = (sel @ kern).mean()
    idx = m.add_channel("conv3", torch.from_numpy(kern.reshape(c, k, k).astype(np.float32)),
                        bias=-0.5 * float(resp_pos))
    m.params["head_concept.weight"][0, idx] = 1.0
    m.params["head_concept.bias"][0] = -0.0
    return m


# --------------------------------------------------- shared parametrisation
class ChannelParam:
    """Vector <-> (new conv3 kernel, bias, head weight, head bias)."""

    def __init__(self, model: EditableCNN):
        self.base = model.clone()
        self.idx = self.base.add_channel("conv3")
        self.kshape = self.base.params["conv3.weight"][self.idx].shape
        self.dim = int(np.prod(self.kshape)) + 3

    def build(self, theta: np.ndarray) -> EditableCNN:
        m = self.base.clone()
        nk = int(np.prod(self.kshape))
        t = torch.from_numpy(theta.astype(np.float32))
        m.params["conv3.weight"][self.idx] = t[:nk].reshape(self.kshape)
        m.params["conv3.bias"][self.idx] = t[nk]
        m.params["head_concept.weight"][0, self.idx] = t[nk + 1]
        m.params["head_concept.bias"][0] = t[nk + 2]
        return m


def _fitness(model: EditableCNN, dev: dict, budget: Budget) -> float:
    budget.charge("forward_passes")
    return auroc(_concept_scores(model, dev["x"]), dev["y_concept"])


# ------------------------------------------------------------------ D: random
def random_edits(model: EditableCNN, dev: dict, budget: Budget, sigma: float = 0.3,
                 seed: int = 0) -> tuple[EditableCNN, list[float]]:
    rng = np.random.default_rng(seed)
    p = ChannelParam(model)
    best, best_f, hist = None, -1.0, []
    while budget.remaining()["forward_passes"] > 0:
        th = rng.normal(0, sigma, p.dim)
        cand = p.build(th)
        f = _fitness(cand, dev, budget)
        if f > best_f:
            best, best_f = cand, f
        hist.append(best_f)
    return best, hist


# --------------------------------------------------------------- E: ES (1+λ)
def evolution_strategy(model: EditableCNN, dev: dict, budget: Budget, pop: int = 8,
                       sigma: float = 0.3, seed: int = 0) -> tuple[EditableCNN, list[float]]:
    """Simple (1+λ)-ES with 1/5th-success step-size adaptation.

    Fitness is dev AUROC, so it uses exactly the signal the agent gets from
    ``evaluate``; every candidate costs one forward pass over dev."""
    rng = np.random.default_rng(seed)
    p = ChannelParam(model)
    mean = rng.normal(0, 0.05, p.dim)
    best_f = _fitness(p.build(mean), dev, budget)
    hist = [best_f]
    while budget.remaining()["forward_passes"] >= 1:
        succ = 0
        for _ in range(pop):
            if budget.remaining()["forward_passes"] < 1:
                break
            cand = mean + sigma * rng.standard_normal(p.dim)
            f = _fitness(p.build(cand), dev, budget)
            if f > best_f:
                mean, best_f, succ = cand, f, succ + 1
            hist.append(best_f)
        sigma *= 1.5 if succ / pop > 0.2 else 0.85
    return p.build(mean), hist


# ------------------------------------------------------- G: backprop (ref.)
def backprop_reference(model: EditableCNN, train: dict, steps: int = 1000, lr: float = 0.001,
                       head_only: bool = False, seed: int = 0) -> EditableCNN:
    """Upper reference only. Uses gradients; not an ADL method."""
    torch.manual_seed(seed)
    if head_only:
        m = model.clone()
        f = torch.from_numpy(_pooled(m, train["x"]))
        w = torch.zeros(f.shape[1], requires_grad=True)
        b = torch.zeros(1, requires_grad=True)
        opt = torch.optim.Adam([w, b], lr=lr)
        y = torch.from_numpy(train["y_concept"]).float()
        for _ in range(steps):
            loss = F.binary_cross_entropy_with_logits(f @ w + b, y)
            opt.zero_grad(); loss.backward(); opt.step()
        m.params["head_concept.weight"] = w.detach()[None].clone()
        m.params["head_concept.bias"] = b.detach().clone()
        return m
    p = ChannelParam(model)
    theta = torch.zeros(p.dim)
    theta[:-3].normal_(0, 0.1)
    theta[-3] = 0.1   # small positive bias keeps the new ReLU channel alive
    theta[-2] = 0.5   # non-zero head weight so the channel receives gradient
    theta.requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=lr)
    x = torch.from_numpy(train["x"])
    y = torch.from_numpy(train["y_concept"]).float()
    nk = p.dim - 3
    base = p.base
    for _ in range(steps):
        params = dict(base.params)
        wk = base.params["conv3.weight"].clone()
        wk[p.idx] = theta[:nk].reshape(p.kshape)
        bk = base.params["conv3.bias"].clone()
        bk[p.idx] = theta[nk]
        hw = base.params["head_concept.weight"].clone()
        hw[0, p.idx] = theta[nk + 1]
        params.update({"conv3.weight": wk, "conv3.bias": bk, "head_concept.weight": hw,
                       "head_concept.bias": theta[nk + 2:nk + 3]})
        tmp = EditableCNN.__new__(EditableCNN)
        tmp.params = params
        _, logit = tmp(x)
        loss = F.binary_cross_entropy_with_logits(logit, y)
        opt.zero_grad(); loss.backward(); opt.step()
    return p.build(theta.detach().numpy())


def backprop_full(model: EditableCNN, train: dict, steps: int = 400, lr: float = 1e-3,
                  seed: int = 0) -> EditableCNN:
    """Upper reference: fine-tune *all* parameters with backprop on the concept
    loss plus base-class rehearsal on the same images. Not an ADL method."""
    torch.manual_seed(seed)
    m = model.clone()
    for k in m.params:
        m.params[k] = m.params[k].clone().requires_grad_(True)
    with torch.no_grad():
        m.params["head_concept.weight"].normal_(0, 0.05)
    opt = torch.optim.Adam(list(m.params.values()), lr=lr)
    x = torch.from_numpy(train["x"])
    yc = torch.from_numpy(train["y_concept"]).float()
    yb = torch.from_numpy(train["y_base"])
    for _ in range(steps):
        base, logit = m(x)
        loss = F.binary_cross_entropy_with_logits(logit, yc) + F.cross_entropy(base, yb)
        opt.zero_grad(); loss.backward(); opt.step()
    for k in m.params:
        m.params[k] = m.params[k].detach()
    return m
