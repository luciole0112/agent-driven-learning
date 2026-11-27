"""Small, editable CNN.

The network is deliberately simple (no BatchNorm, no residuals) so that every
edit has a transparent effect:

    conv1 (3 -> c1, 5x5) - ReLU - maxpool2
    conv2 (c1 -> c2, 3x3) - ReLU - maxpool2
    conv3 (c2 -> c3, 3x3) - ReLU - global max pool
    head_base    : Linear(c3 -> 3)   base shape classes (pre-trained)
    head_concept : Linear(c3 -> 1)   new concept logit (starts at zero)

Parameters are plain tensors held in a dict so that tools can change shapes
(add / remove channels) without rebuilding modules.
"""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F

CONV_LAYERS = ("conv1", "conv2", "conv3")
HEADS = ("head_base", "head_concept")


class EditableCNN:
    def __init__(self, c1: int = 8, c2: int = 16, c3: int = 32,
                 n_base: int = 3, seed: int = 0):
        g = torch.Generator().manual_seed(seed)

        def init(*shape, fan_in):
            return torch.randn(*shape, generator=g) * (2.0 / fan_in) ** 0.5

        self.params: dict[str, torch.Tensor] = {
            "conv1.weight": init(c1, 3, 5, 5, fan_in=3 * 25),
            "conv1.bias": torch.zeros(c1),
            "conv2.weight": init(c2, c1, 3, 3, fan_in=c1 * 9),
            "conv2.bias": torch.zeros(c2),
            "conv3.weight": init(c3, c2, 3, 3, fan_in=c2 * 9),
            "conv3.bias": torch.zeros(c3),
            "head_base.weight": init(n_base, c3, fan_in=c3),
            "head_base.bias": torch.zeros(n_base),
            "head_concept.weight": torch.zeros(1, c3),
            "head_concept.bias": torch.zeros(1),
        }

    # ------------------------------------------------------------------ utils
    def clone(self) -> "EditableCNN":
        m = EditableCNN.__new__(EditableCNN)
        m.params = {k: v.detach().clone() for k, v in self.params.items()}
        return m

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {k: v.detach().clone() for k, v in self.params.items()}

    def load_state_dict(self, sd: dict[str, torch.Tensor]) -> None:
        self.params = {k: v.detach().clone() for k, v in sd.items()}

    def n_params(self) -> int:
        return sum(v.numel() for v in self.params.values())

    def channels(self, layer: str) -> int:
        return self.params[f"{layer}.weight"].shape[0]

    def describe(self) -> dict:
        return {k: list(v.shape) for k, v in self.params.items()}

    # ---------------------------------------------------------------- forward
    def forward(self, x: torch.Tensor, return_acts: bool = False,
                ablate: dict[str, list[int]] | None = None,
                act_override: dict[str, callable] | None = None):
        p = self.params
        acts = {}
        h = x
        for i, layer in enumerate(CONV_LAYERS):
            pad = p[f"{layer}.weight"].shape[-1] // 2
            h = F.conv2d(h, p[f"{layer}.weight"], p[f"{layer}.bias"], padding=pad)
            h = F.relu(h)
            if ablate and layer in ablate:
                h = h.clone()
                h[:, ablate[layer]] = 0.0
            if act_override and layer in act_override:
                h = act_override[layer](h)
            acts[layer] = h
            if i < 2:
                h = F.max_pool2d(h, 2)
        feat = h.amax(dim=(2, 3))
        acts["pooled"] = feat
        base = feat @ p["head_base.weight"].T + p["head_base.bias"]
        concept = (feat @ p["head_concept.weight"].T + p["head_concept.bias"])[:, 0]
        if return_acts:
            return base, concept, acts
        return base, concept

    __call__ = forward

    # ------------------------------------------------------- structural edits
    def add_channel(self, layer: str, kernel: torch.Tensor | None = None,
                    bias: float = 0.0) -> int:
        """Append one output channel to ``layer``.

        Its outgoing weights (next conv layer input slice, or head columns for
        conv3) are zero, so the network function is unchanged until the new
        channel is connected. Returns the new channel index.
        """
        w = self.params[f"{layer}.weight"]
        cin, k = w.shape[1], w.shape[-1]
        if kernel is None:
            kernel = torch.zeros(cin, k, k)
        kernel = torch.as_tensor(kernel, dtype=w.dtype).reshape(1, cin, k, k)
        self.params[f"{layer}.weight"] = torch.cat([w, kernel], 0)
        self.params[f"{layer}.bias"] = torch.cat(
            [self.params[f"{layer}.bias"], torch.tensor([bias], dtype=w.dtype)])
        idx = CONV_LAYERS.index(layer)
        if idx + 1 < len(CONV_LAYERS):
            nxt = CONV_LAYERS[idx + 1]
            wn = self.params[f"{nxt}.weight"]
            z = torch.zeros(wn.shape[0], 1, *wn.shape[2:], dtype=wn.dtype)
            self.params[f"{nxt}.weight"] = torch.cat([wn, z], 1)
        else:
            for head in HEADS:
                wh = self.params[f"{head}.weight"]
                self.params[f"{head}.weight"] = torch.cat(
                    [wh, torch.zeros(wh.shape[0], 1, dtype=wh.dtype)], 1)
        return self.channels(layer) - 1

    def remove_channel(self, layer: str, ch: int) -> None:
        keep = [i for i in range(self.channels(layer)) if i != ch]
        self.params[f"{layer}.weight"] = self.params[f"{layer}.weight"][keep]
        self.params[f"{layer}.bias"] = self.params[f"{layer}.bias"][keep]
        idx = CONV_LAYERS.index(layer)
        if idx + 1 < len(CONV_LAYERS):
            nxt = CONV_LAYERS[idx + 1]
            self.params[f"{nxt}.weight"] = self.params[f"{nxt}.weight"][:, keep]
        else:
            for head in HEADS:
                self.params[f"{head}.weight"] = self.params[f"{head}.weight"][:, keep]


def deepcopy_model(m: EditableCNN) -> EditableCNN:
    return copy.deepcopy(m)
