"""CIFAR-10 splits for the MobileNetV2 experiments (docs/cifar_mbv2_design_v1.md, section 2.1).

Every split is a dict ``{"x": uint8 (N, 3, 32, 32), "y": int64 (N,)}``.

* ``train``   : 45,000 of the official train images (the agent's pool is the
  first 5,000, the retention monitor the last 1,000; chosen by the environment).
* ``dev``     : the remaining 5,000 official train images, class balanced (500 per class).
* ``heldout`` : the official 10,000 test images (sealed).
* ``shift``   : the test images with one of four corruptions each (sealed). ``kind``
  holds 0 noise, 1 blur, 2 contrast, 3 brightness, assigned by a fixed seed.

``fake_cifar`` makes CIFAR-shaped stand-in data with a weak class signal, for
offline tests.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

SHIFT_KINDS = ("noise", "blur", "contrast", "brightness")


def _to_nchw(x: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(x.transpose(0, 3, 1, 2)).astype(np.uint8)


def _blur(x: np.ndarray) -> np.ndarray:
    """3x3 box blur on float (N, 3, H, W), edge padded."""
    p = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="edge")
    h, w = x.shape[2:]
    return sum(p[:, :, i:i + h, j:j + w] for i in range(3) for j in range(3)) / 9.0


def corrupt(x: np.ndarray, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Apply one of four corruptions to each image (deterministic). Returns (x, kind)."""
    rng = np.random.default_rng(seed)
    kind = rng.integers(0, len(SHIFT_KINDS), len(x))
    f = x.astype(np.float32)
    out = f.copy()
    m = kind == 0
    out[m] = f[m] + rng.normal(0, 0.08 * 255, f[m].shape)
    m = kind == 1
    out[m] = _blur(f[m])
    m = kind == 2
    mu = f[m].mean(axis=(1, 2, 3), keepdims=True)
    out[m] = (f[m] - mu) * 0.4 + mu
    m = kind == 3
    out[m] = f[m] + 0.3 * 255
    return np.clip(np.rint(out), 0, 255).astype(np.uint8), kind.astype(np.int64)


def _split_train(x: np.ndarray, y: np.ndarray, dev_n: int, seed: int):
    rng = np.random.default_rng(seed)
    per = dev_n // 10
    dev_idx = np.concatenate([rng.permutation(np.flatnonzero(y == c))[:per] for c in range(10)])
    dev_idx = rng.permutation(dev_idx)
    mask = np.ones(len(y), bool)
    mask[dev_idx] = False
    tr_idx = rng.permutation(np.flatnonzero(mask))
    return {"x": x[tr_idx], "y": y[tr_idx]}, {"x": x[dev_idx], "y": y[dev_idx]}


def make_cifar_splits(root: str | Path = "data", dev_n: int = 5000, seed: int = 0,
                      download: bool = True) -> dict:
    from torchvision.datasets import CIFAR10

    tr = CIFAR10(str(root), train=True, download=download)
    te = CIFAR10(str(root), train=False, download=download)
    x, y = _to_nchw(tr.data), np.asarray(tr.targets, dtype=np.int64)
    train, dev = _split_train(x, y, dev_n, seed)
    hx, hy = _to_nchw(te.data), np.asarray(te.targets, dtype=np.int64)
    sx, kind = corrupt(hx, seed=seed + 1)
    return {"train": train, "dev": dev, "heldout": {"x": hx, "y": hy},
            "shift": {"x": sx, "y": hy.copy(), "kind": kind}}


def fake_cifar(n_train: int = 2000, n_dev: int = 500, seed: int = 0) -> dict:
    """CIFAR-shaped random images whose mean colour depends weakly on the class."""
    rng = np.random.default_rng(seed)
    palette = rng.uniform(60, 200, (10, 3))

    def make(n):
        y = rng.integers(0, 10, n).astype(np.int64)
        x = rng.normal(0, 40, (n, 3, 32, 32)) + palette[y][:, :, None, None]
        return np.clip(np.rint(x), 0, 255).astype(np.uint8), y

    tx, ty = make(n_train)
    dx, dy = make(n_dev)
    hx, hy = make(n_dev)
    sx, kind = corrupt(hx, seed=seed + 1)
    return {"train": {"x": tx, "y": ty}, "dev": {"x": dx, "y": dy}, "heldout": {"x": hx, "y": hy},
            "shift": {"x": sx, "y": hy.copy(), "kind": kind}}
