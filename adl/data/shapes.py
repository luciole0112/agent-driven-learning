"""Synthetic image generator for ADL experiments.

Each image contains one *base shape* (circle / square / triangle), used to
pre-train the base CNN by ordinary backprop, and optionally one *concept
pattern* whose presence defines the new binary concept the agent must teach
the frozen model.

Two kinds of concept are supported:

* ``named``  : an oriented bar with a fixed orientation (E0 pilot). Negatives
  contain bars of other orientations.
* ``glyph``  : a procedurally generated, unnamed k x k colour glyph (E1).
  Negatives contain a *distractor* glyph built from exactly the same colours
  and the same number of pixels of each colour, with the pixels permuted, so
  colour statistics alone cannot solve the task.

All randomness flows from explicit seeds so that every split is reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

BASE_CLASSES = ("circle", "square", "triangle")


@dataclass
class ConceptSpec:
    kind: str = "glyph"  # "glyph" | "named"
    glyph_size: int = 5
    n_colors: int = 3
    concept_seed: int = 0
    scales: tuple[int, ...] = (1, 2)
    rotate: bool = False
    bar_angle_deg: float = 45.0  # for kind == "named"
    _glyph: np.ndarray | None = field(default=None, repr=False)
    _palette: np.ndarray | None = field(default=None, repr=False)

    def build(self) -> "ConceptSpec":
        rng = np.random.default_rng(self.concept_seed)
        self._palette = rng.uniform(0.15, 1.0, size=(self.n_colors, 3))
        k = self.glyph_size
        # index map: -1 transparent, otherwise palette index
        g = rng.integers(0, self.n_colors, size=(k, k))
        mask = rng.random((k, k)) < 0.8
        g = np.where(mask, g, -1)
        # make sure the glyph is not trivially symmetric
        if np.array_equal(g, g[:, ::-1]) or np.array_equal(g, g[::-1, :]):
            g[0, 0] = (g[0, 0] + 1) % self.n_colors
        self._glyph = g
        return self

    @property
    def glyph(self) -> np.ndarray:
        if self._glyph is None:
            self.build()
        return self._glyph

    @property
    def palette(self) -> np.ndarray:
        if self._palette is None:
            self.build()
        return self._palette


def _draw_shape(img: np.ndarray, cls: int, rng: np.random.Generator) -> None:
    h, w, _ = img.shape
    color = rng.uniform(0.2, 1.0, size=3)
    size = rng.integers(10, 22)
    cy, cx = rng.integers(size, h - size), rng.integers(size, w - size)
    yy, xx = np.mgrid[0:h, 0:w]
    if cls == 0:
        m = (yy - cy) ** 2 + (xx - cx) ** 2 <= size**2
    elif cls == 1:
        m = (np.abs(yy - cy) <= size) & (np.abs(xx - cx) <= size)
    else:
        m = (yy >= cy - size) & (yy <= cy + size) & (
            np.abs(xx - cx) <= (yy - (cy - size)) / 2
        )
    img[m] = 0.5 * img[m] + 0.5 * color


def _stamp(img: np.ndarray, pattern_idx: np.ndarray, palette: np.ndarray,
           scale: int, rng: np.random.Generator) -> tuple[int, int, int, int]:
    p = np.kron(pattern_idx, np.ones((scale, scale), dtype=int))
    ph, pw = p.shape
    h, w, _ = img.shape
    y0, x0 = rng.integers(0, h - ph), rng.integers(0, w - pw)
    region = img[y0:y0 + ph, x0:x0 + pw]
    on = p >= 0
    region[on] = palette[p[on]]
    return y0, x0, ph, pw


def _distractor(glyph: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    flat = glyph.flatten()
    for _ in range(100):
        perm = rng.permutation(flat).reshape(glyph.shape)
        if not np.array_equal(perm, glyph):
            return perm
    return np.rot90(glyph).copy()


def _bar(angle_deg: float, length: int = 9) -> np.ndarray:
    k = length
    g = -np.ones((k, k), dtype=int)
    c = (k - 1) / 2
    t = np.deg2rad(angle_deg)
    for s in np.linspace(-c, c, 4 * k):
        y, x = int(round(c - s * np.sin(t))), int(round(c + s * np.cos(t)))
        g[y, x] = 0
    return g


def make_dataset(n: int, seed: int, concept: ConceptSpec | None = None,
                 p_concept: float = 0.5, size: int = 64,
                 shift: bool = False) -> dict[str, np.ndarray]:
    """Return images (N,3,H,W) float32, base labels (N,), concept labels (N,).

    ``shift=True`` produces a distribution-shifted split: brighter, noisier
    backgrounds and a larger stamp scale than any seen in training.
    """
    rng = np.random.default_rng(seed)
    X = np.zeros((n, size, size, 3), dtype=np.float32)
    yb = rng.integers(0, len(BASE_CLASSES), size=n)
    yc = np.zeros(n, dtype=np.int64)
    if concept is not None:
        concept.build()
    for i in range(n):
        noise = 0.12 if not shift else 0.22
        base = rng.uniform(0.0, 0.3 if not shift else 0.45, size=3)
        img = np.clip(base + noise * rng.standard_normal((size, size, 3)), 0, 1)
        _draw_shape(img, int(yb[i]), rng)
        if concept is not None:
            positive = rng.random() < p_concept
            yc[i] = int(positive)
            scales = concept.scales if not shift else (max(concept.scales) + 1,)
            scale = int(rng.choice(scales))
            if concept.kind == "glyph":
                pat = concept.glyph if positive else _distractor(concept.glyph, rng)
                if concept.rotate:
                    pat = np.rot90(pat, int(rng.integers(0, 4)))
                _stamp(img, pat, concept.palette, scale, rng)
            elif concept.kind == "named":
                if positive:
                    ang = concept.bar_angle_deg
                else:
                    ang = concept.bar_angle_deg + float(rng.choice([45, 90, 135]))
                col = rng.uniform(0.6, 1.0, size=(1, 3))
                _stamp(img, _bar(ang), col, scale, rng)
            else:
                raise ValueError(concept.kind)
        X[i] = img
    return {
        "x": X.transpose(0, 3, 1, 2).copy(),
        "y_base": yb.astype(np.int64),
        "y_concept": yc,
    }


SPLIT_SEEDS = {"base_train": 1, "base_val": 2, "train": 11, "dev": 12,
               "heldout": 13, "shift": 14}


def make_splits(concept: ConceptSpec, n_train: int = 200, n_dev: int = 400,
                n_heldout: int = 1000, n_shift: int = 500,
                seed_offset: int = 0) -> dict[str, dict[str, np.ndarray]]:
    """Concept splits. ``heldout`` and ``shift`` must stay sealed from the agent."""
    s = {k: v + 100 * seed_offset for k, v in SPLIT_SEEDS.items()}
    return {
        "train": make_dataset(n_train, s["train"], concept),
        "dev": make_dataset(n_dev, s["dev"], concept),
        "heldout": make_dataset(n_heldout, s["heldout"], concept),
        "shift": make_dataset(n_shift, s["shift"], concept, shift=True),
    }
