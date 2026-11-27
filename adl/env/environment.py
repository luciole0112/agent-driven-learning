"""The ADL environment: tools through which an agent observes and edits a CNN.

Design rules (see docs/research_plan.md):

* No gradient ever reaches the target network from inside this environment.
  All computation runs under ``torch.no_grad()`` and parameters never require
  grad. :func:`assert_no_grad` is checked after every tool call.
* Temporary interventions (``probe_*``) never change the model. Persistent
  edits (``edit_*``, ``add_channel`` ...) change it and are charged.
* ``heldout`` and ``shift`` splits are sealed: no tool can read them. They are
  only used by :meth:`ADLEnvironment.final_report` after the agent submits.
* Every call is charged to the :class:`Budget` and written to the
  :class:`Ledger`.

Each tool returns ``ToolResult(text, images)``; images are PNG bytes so a
VLM can look at them.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

import numpy as np
import torch
from PIL import Image

from adl.data.shapes import SPLIT_SEEDS, make_dataset
from adl.env.budget import Budget, BudgetExceeded, Ledger
from adl.env.metrics import auroc
from adl.env.sandbox import SandboxViolation, run_code
from adl.models.cnn import CONV_LAYERS, EditableCNN

# receptive field (size, stride) of each conv layer's output in input pixels
RECEPTIVE = {"conv1": (5, 1), "conv2": (10, 2), "conv3": (18, 4)}

OBSERVATION_TOOLS = {
    "get_tensor", "render_kernels", "show_examples", "feature_map",
    "top_activations", "channel_stats", "capture_activations", "probe_ablate",
    "run_sandbox",
}
VISUAL_TOOLS = {"render_kernels", "show_examples", "feature_map", "top_activations"}


@dataclass
class ToolResult:
    text: str
    images: list[bytes] = field(default_factory=list)


class ToolError(RuntimeError):
    pass


def assert_no_grad(model: EditableCNN) -> None:
    for k, v in model.params.items():
        if v.requires_grad or v.grad is not None:
            raise RuntimeError(f"gradient state found on parameter {k}")


def _png(arr: np.ndarray, scale: int = 4) -> bytes:
    """arr: (H,W,3) or (H,W) in [0,1]."""
    a = np.clip(arr, 0, 1)
    if a.ndim == 2:
        a = np.stack([a] * 3, -1)
    img = Image.fromarray((a * 255).astype(np.uint8))
    if scale != 1:
        img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _grid(tiles: list[np.ndarray], ncol: int = 8, pad: int = 1) -> np.ndarray:
    h, w = tiles[0].shape[:2]
    nrow = (len(tiles) + ncol - 1) // ncol
    g = np.ones((nrow * (h + pad) + pad, ncol * (w + pad) + pad, 3)) * 0.5
    for i, t in enumerate(tiles):
        r, c = divmod(i, ncol)
        if t.ndim == 2:
            t = np.stack([t] * 3, -1)
        g[pad + r * (h + pad): pad + r * (h + pad) + h,
          pad + c * (w + pad): pad + c * (w + pad) + w] = t
    return g


def _diverging(m: np.ndarray) -> np.ndarray:
    """Map a 2D array to RGB: red positive, blue negative."""
    s = np.abs(m).max() + 1e-8
    v = m / s
    rgb = np.ones(m.shape + (3,))
    rgb[..., 1] = 1 - np.abs(v)
    rgb[..., 0] = np.where(v < 0, 1 - np.abs(v), 1)
    rgb[..., 2] = np.where(v > 0, 1 - np.abs(v), 1)
    return rgb


class ADLEnvironment:
    def __init__(self, model: EditableCNN, splits: dict, budget: Budget | None = None,
                 ledger: Ledger | None = None, condition: str = "full",
                 forbid_fitting: bool = False, base_val_n: int = 500,
                 device: str = "cpu", max_values_per_read: int = 2048):
        assert condition in {"full", "no_semantics", "no_observation"}
        self.model = model
        for k in self.model.params:
            self.model.params[k] = self.model.params[k].detach().to(device)
        self._initial = model.state_dict()
        self.splits = {k: splits[k] for k in ("train", "dev")}
        self._sealed = {k: splits[k] for k in ("heldout", "shift") if k in splits}
        self.base_val = make_dataset(base_val_n, SPLIT_SEEDS["base_val"] + 7)
        self.budget = budget or Budget()
        self.ledger = ledger or Ledger()
        self.condition = condition
        self.forbid_fitting = forbid_fitting
        self.device = device
        self.max_values = max_values_per_read
        self.checkpoints: dict[str, dict] = {"initial": self._initial}
        self.sandbox_arrays: dict[str, np.ndarray] = {}
        self.sandbox_out: dict[str, np.ndarray] = {}
        self.notes: list[str] = []
        self.submitted = False
        self.base_acc_initial = self._base_acc()

    # ================================================================ helpers
    def available_tools(self) -> list[str]:
        tools = [n for n in TOOL_SPECS]
        if self.condition == "no_observation":
            tools = [t for t in tools if t not in OBSERVATION_TOOLS]
        return tools

    def call(self, name: str, args: dict | None = None, rationale: str = "") -> ToolResult:
        args = args or {}
        if name not in self.available_tools():
            res = ToolResult(f"ERROR: tool '{name}' is not available in condition '{self.condition}'")
            self.ledger.log(name, args, res.text, rationale, ["unavailable_tool"])
            return res
        if self.submitted:
            return ToolResult("ERROR: already submitted")
        fn = getattr(self, f"tool_{name}")
        flags: list[str] = []
        try:
            with torch.no_grad():
                res = fn(**args)
            if isinstance(res, tuple):
                res, flags = res
        except (BudgetExceeded, ToolError, SandboxViolation) as e:
            res = ToolResult(f"ERROR: {e}")
        except Exception as e:  # surface agent mistakes as tool errors
            res = ToolResult(f"ERROR: {type(e).__name__}: {e}")
        assert_no_grad(self.model)
        self.ledger.log(name, args, res.text, rationale, flags)
        return res

    def _x(self, split: str, ids=None) -> torch.Tensor:
        if split not in self.splits:
            raise ToolError(f"split '{split}' is not accessible (use 'train' or 'dev')")
        x = self.splits[split]["x"]
        if ids is not None:
            x = x[np.asarray(ids)]
        return torch.from_numpy(x).to(self.device)

    def _forward(self, x: torch.Tensor, **kw):
        outs = [self.model(x[i:i + 256], return_acts=True, **kw) for i in range(0, len(x), 256)]
        base = torch.cat([o[0] for o in outs]).cpu().numpy()
        concept = torch.cat([o[1] for o in outs]).cpu().numpy()
        acts = {k: torch.cat([o[2][k] for o in outs]).cpu().numpy() for k in outs[0][2]}
        return base, concept, acts

    def _base_acc(self, model: EditableCNN | None = None) -> float:
        m = model or self.model
        x = torch.from_numpy(self.base_val["x"]).to(self.device)
        with torch.no_grad():
            preds = torch.cat([m(x[i:i + 256])[0].argmax(1) for i in range(0, len(x), 256)])
        return float((preds.cpu().numpy() == self.base_val["y_base"]).mean())

    def _layer(self, layer: str) -> str:
        if layer not in CONV_LAYERS:
            raise ToolError(f"unknown layer '{layer}', choose from {CONV_LAYERS}")
        return layer

    def _param(self, name: str) -> torch.Tensor:
        if name not in self.model.params:
            raise ToolError(f"unknown parameter '{name}'. Known: {list(self.model.params)}")
        return self.model.params[name]

    def _no_images(self) -> bool:
        return self.condition == "no_semantics"

    # ========================================================== observation
    def tool_inspect_model(self) -> ToolResult:
        lines = ["EditableCNN (no BatchNorm):",
                 "conv1(5x5) -> ReLU -> maxpool2 -> conv2(3x3) -> ReLU -> maxpool2 -> "
                 "conv3(3x3) -> ReLU -> global max pool -> head_base (3 base classes), "
                 "head_concept (1 logit for the new concept)"]
        for k, s in self.model.describe().items():
            lines.append(f"  {k}: {s}")
        lines.append(f"receptive field (size, stride) in input pixels: {RECEPTIVE}")
        lines.append(f"budget remaining: {self.budget.remaining()}")
        return ToolResult("\n".join(lines))

    def tool_get_tensor(self, name: str, index: list[int] | None = None,
                        stats_only: bool = False) -> ToolResult:
        t = self._param(name)
        if index:
            for i in index:
                t = t[i]
        a = t.cpu().numpy()
        stats = (f"shape={list(a.shape)} mean={a.mean():.4g} std={a.std():.4g} "
                 f"min={a.min():.4g} max={a.max():.4g} l2={np.linalg.norm(a):.4g}")
        if stats_only:
            self.budget.charge("weight_values_read", 1)
            return ToolResult(stats)
        if a.size > self.max_values:
            raise ToolError(f"slice has {a.size} values (> {self.max_values}); use index or stats_only")
        self.budget.charge("weight_values_read", int(a.size))
        return ToolResult(stats + "\n" + np.array2string(a, precision=3, suppress_small=True,
                                                         max_line_width=160, threshold=10**6))

    def tool_render_kernels(self, layer: str, out_channel: int | None = None) -> ToolResult:
        """conv1: all RGB kernels as colour tiles. Deeper layers: for one output
        channel, one diverging heat map per input channel."""
        layer = self._layer(layer)
        w = self.model.params[f"{layer}.weight"].cpu().numpy()
        self.budget.charge("weight_values_read", int(w.size if out_channel is None else w[0].size))
        if self._no_images():
            return self.tool_get_tensor(f"{layer}.weight", [out_channel] if out_channel is not None else None,
                                        stats_only=out_channel is None)
        if layer == "conv1" and out_channel is None:
            tiles = []
            for k in w:
                k = k.transpose(1, 2, 0)
                tiles.append((k - k.min()) / (k.max() - k.min() + 1e-8))
            return ToolResult(f"{len(tiles)} conv1 kernels (RGB, min-max normalised per kernel, "
                              f"index order left-to-right, top-to-bottom)", [_png(_grid(tiles), 12)])
        if out_channel is None:
            raise ToolError("for conv2/conv3 give out_channel")
        tiles = [_diverging(k) for k in w[out_channel]]
        return ToolResult(f"{layer} out_channel {out_channel}: {len(tiles)} input-channel kernels "
                          f"(red=positive, blue=negative, shared scale per tile)", [_png(_grid(tiles), 12)])

    def tool_show_examples(self, split: str = "train", label: int = 1, n: int = 8,
                           offset: int = 0) -> ToolResult:
        d = self.splits.get(split)
        if d is None:
            raise ToolError("only 'train' or 'dev'")
        ids = np.where(d["y_concept"] == label)[0][offset:offset + n]
        if self._no_images():
            x = d["x"][ids]
            return ToolResult(f"ids={ids.tolist()} (images withheld in this condition). "
                              f"per-image mean RGB: {np.round(x.mean((2, 3)), 3).tolist()}")
        tiles = [d["x"][i].transpose(1, 2, 0) for i in ids]
        return ToolResult(f"{split} examples with concept label {label}: ids={ids.tolist()}",
                          [_png(_grid(tiles, ncol=4), 3)])

    def tool_feature_map(self, layer: str, channel: int, ids: list[int],
                         split: str = "train") -> ToolResult:
        layer = self._layer(layer)
        self.budget.charge("forward_passes")
        x = self._x(split, ids)
        _, _, acts = self._forward(x)
        a = acts[layer][:, channel]
        text = (f"{layer}[{channel}] on {split} ids={ids}: max per image="
                f"{np.round(a.reshape(len(ids), -1).max(1), 3).tolist()}")
        if self._no_images():
            return ToolResult(text)
        tiles = []
        for img, fm in zip(x.cpu().numpy(), a):
            img = img.transpose(1, 2, 0)
            up = np.kron(fm, np.ones((img.shape[0] // fm.shape[0],) * 2))
            up = up / (a.max() + 1e-8)
            over = 0.5 * img + 0.5 * np.stack([up, np.zeros_like(up), np.zeros_like(up)], -1)
            tiles += [img, over]
        return ToolResult(text + " (pairs: image | activation overlay in red, shared scale)",
                          [_png(_grid(tiles, ncol=4), 3)])

    def tool_top_activations(self, layer: str, channel: int, k: int = 12,
                             split: str = "train") -> ToolResult:
        layer = self._layer(layer)
        self.budget.charge("forward_passes")
        x = self._x(split)
        _, _, acts = self._forward(x)
        a = acts[layer][:, channel]
        n, h, w = a.shape
        flat = a.reshape(n, -1)
        best = flat.max(1)
        order = np.argsort(-best)[:k]
        rf, stride = RECEPTIVE[layer]
        scale = x.shape[-1] // h
        patches, info = [], []
        for i in order:
            yx = np.unravel_index(flat[i].argmax(), (h, w))
            cy, cx = yx[0] * scale + scale // 2, yx[1] * scale + scale // 2
            y0, x0 = max(0, cy - rf // 2), max(0, cx - rf // 2)
            p = x[i, :, y0:y0 + rf, x0:x0 + rf].cpu().numpy().transpose(1, 2, 0)
            pp = np.zeros((rf, rf, 3))
            pp[:p.shape[0], :p.shape[1]] = p
            patches.append(pp)
            info.append((int(i), float(best[i]), int(self.splits[split]["y_concept"][i])))
        lab = self.splits[split]["y_concept"]
        text = (f"top-{k} {split} images for {layer}[{channel}] as (id, max_act, concept_label): {info}. "
                f"mean max_act: concept=1 {best[lab == 1].mean():.3f}, concept=0 {best[lab == 0].mean():.3f}")
        if self._no_images():
            return ToolResult(text)
        return ToolResult(text + f" | image: receptive-field patches ({rf}x{rf}px) at the argmax location",
                          [_png(_grid(patches, ncol=6), 6)])

    def tool_channel_stats(self, layer: str, split: str = "train") -> ToolResult:
        """Per channel: mean of spatial-max activation for concept=1 vs concept=0, ranked."""
        layer = self._layer(layer)
        self.budget.charge("forward_passes")
        _, _, acts = self._forward(self._x(split))
        m = acts[layer].reshape(*acts[layer].shape[:2], -1).max(-1)
        lab = self.splits[split]["y_concept"]
        pos, neg = m[lab == 1].mean(0), m[lab == 0].mean(0)
        sd = m.std(0) + 1e-8
        d = (pos - neg) / sd
        order = np.argsort(-np.abs(d))
        rows = [f"ch{c}: pos={pos[c]:.3f} neg={neg[c]:.3f} d'={d[c]:+.3f}" for c in order]
        return ToolResult(f"{layer} channels ranked by |standardised difference| ({split}):\n" + "\n".join(rows))

    def tool_capture_activations(self, layer: str, key: str, split: str = "train",
                                 ids: list[int] | None = None) -> ToolResult:
        """Store activations (and labels) into the sandbox as A[key], A[key+'_labels']."""
        if layer != "input":
            layer = self._layer(layer)
        self.budget.charge("forward_passes")
        x = self._x(split, ids)
        if layer == "input":
            arr = x.cpu().numpy()
        else:
            _, _, acts = self._forward(x)
            arr = acts[layer]
        lab = self.splits[split]["y_concept"]
        if ids is not None:
            lab = lab[np.asarray(ids)]
        self.sandbox_arrays[key] = arr
        self.sandbox_arrays[key + "_labels"] = lab.copy()
        return ToolResult(f"A['{key}'] shape={list(arr.shape)}, A['{key}_labels'] shape={list(lab.shape)}")

    def tool_probe_ablate(self, layer: str, channels: list[int], split: str = "dev") -> ToolResult:
        layer = self._layer(layer)
        self.budget.charge("forward_passes", 2)
        x = self._x(split)
        lab = self.splits[split]["y_concept"]
        _, c0, _ = self._forward(x)
        _, c1, _ = self._forward(x, ablate={layer: list(channels)})
        return ToolResult(f"ablate {layer}{list(channels)} on {split}: concept AUROC "
                          f"{auroc(c0, lab):.4f} -> {auroc(c1, lab):.4f} (model unchanged)")

    def tool_run_sandbox(self, code: str):
        self.budget.charge("sandbox_runs")

        def W(name):
            t = self._param(name)
            self.budget.charge("weight_values_read", int(t.numel()))
            return t.cpu().numpy().copy()

        text, out, flags = run_code(code, W, self.sandbox_arrays, self.forbid_fitting)
        self.sandbox_out.update(out)
        msg = text or "(no output)"
        if out:
            msg += f"\nstored outputs: { {k: list(np.shape(v)) for k, v in out.items()} }"
        if flags:
            msg += f"\naudit flags: {flags}"
        return ToolResult(msg), flags

    # ================================================================ edits
    def _values(self, values=None, from_sandbox: str | None = None) -> np.ndarray:
        if from_sandbox is not None:
            if from_sandbox not in self.sandbox_out:
                raise ToolError(f"no sandbox output '{from_sandbox}'")
            return np.asarray(self.sandbox_out[from_sandbox], dtype=np.float32)
        if values is None:
            raise ToolError("give values or from_sandbox")
        return np.asarray(values, dtype=np.float32)

    def _set(self, name: str, index, arr: np.ndarray) -> str:
        t = self._param(name)
        view = t
        for i in index or []:
            view = view[i]
        if tuple(view.shape) != tuple(np.shape(arr)) and np.size(arr) != 1:
            raise ToolError(f"shape mismatch: target {list(view.shape)} vs values {list(np.shape(arr))}")
        before = view.clone()
        view.copy_(torch.as_tensor(arr, dtype=t.dtype).expand_as(view))
        return f"changed {view.numel()} values, |delta|_2={float((view - before).norm()):.4g}"

    def tool_set_weights(self, name: str, index: list[int] | None = None, values=None,
                         from_sandbox: str | None = None) -> ToolResult:
        self.budget.charge("edits")
        return ToolResult(self._set(name, index, self._values(values, from_sandbox)))

    def tool_add_to_weights(self, name: str, index: list[int] | None = None, values=None,
                            from_sandbox: str | None = None) -> ToolResult:
        self.budget.charge("edits")
        t = self._param(name)
        view = t
        for i in index or []:
            view = view[i]
        arr = self._values(values, from_sandbox)
        return ToolResult(self._set(name, index, view.cpu().numpy() + arr))

    def tool_scale_channel(self, layer: str, channel: int, factor: float) -> ToolResult:
        layer = self._layer(layer)
        self.budget.charge("edits")
        self.model.params[f"{layer}.weight"][channel] *= factor
        self.model.params[f"{layer}.bias"][channel] *= factor
        return ToolResult(f"scaled {layer}[{channel}] kernel and bias by {factor}")

    def tool_lowrank_update(self, name: str, u_key: str, v_key: str, alpha: float = 1.0) -> ToolResult:
        """W <- W + alpha * outer(u, v) on the matrix view (out, rest)."""
        self.budget.charge("edits")
        t = self._param(name)
        u = self._values(from_sandbox=u_key).reshape(-1)
        v = self._values(from_sandbox=v_key).reshape(-1)
        mat = t.reshape(t.shape[0], -1)
        if u.size != mat.shape[0] or v.size != mat.shape[1]:
            raise ToolError(f"need u of size {mat.shape[0]} and v of size {mat.shape[1]}")
        mat += alpha * torch.from_numpy(np.outer(u, v)).to(t)
        return ToolResult(f"rank-1 update on {name}, |delta|_F={abs(alpha) * np.linalg.norm(u) * np.linalg.norm(v):.4g}")

    def tool_add_channel(self, layer: str, kernel=None, from_sandbox: str | None = None,
                         bias: float = 0.0) -> ToolResult:
        layer = self._layer(layer)
        self.budget.charge("edits")
        k = None
        if kernel is not None or from_sandbox is not None:
            k = torch.from_numpy(self._values(kernel, from_sandbox))
        idx = self.model.add_channel(layer, k, bias)
        return ToolResult(f"added {layer}[{idx}]; its outgoing weights are zero until you connect it. "
                          f"new shapes: {self.model.describe()}")

    def tool_remove_channel(self, layer: str, channel: int) -> ToolResult:
        layer = self._layer(layer)
        self.budget.charge("edits")
        self.model.remove_channel(layer, channel)
        return ToolResult(f"removed {layer}[{channel}]; indices above it shift down by one")

    def tool_connect(self, src_layer: str, src_channel: int, dst: str, dst_channel: int,
                     values=None, from_sandbox: str | None = None) -> ToolResult:
        """Set the weight(s) from one channel of src_layer into dst.

        dst is the next conv layer (values: k x k kernel slice) or, for conv3,
        'head_concept' / 'head_base' (values: scalar)."""
        src_layer = self._layer(src_layer)
        self.budget.charge("edits")
        arr = self._values(values, from_sandbox)
        i = CONV_LAYERS.index(src_layer)
        if dst in ("head_concept", "head_base"):
            if src_layer != "conv3":
                raise ToolError("only conv3 connects to the heads")
            return ToolResult(self._set(f"{dst}.weight", [dst_channel, src_channel], arr.reshape(())))
        if i + 1 >= len(CONV_LAYERS) or dst != CONV_LAYERS[i + 1]:
            raise ToolError(f"{src_layer} can only connect to {CONV_LAYERS[i + 1] if i + 1 < 3 else 'heads'}")
        return ToolResult(self._set(f"{dst}.weight", [dst_channel, src_channel], arr))

    # ========================================================= verification
    def tool_evaluate(self, split: str = "dev") -> ToolResult:
        self.budget.charge("evaluations")
        self.budget.charge("forward_passes")
        x = self._x(split)
        lab = self.splits[split]["y_concept"]
        _, c, _ = self._forward(x)
        acc = self._base_acc()
        return ToolResult(f"{split}: concept AUROC={auroc(c, lab):.4f}; base accuracy={acc:.4f} "
                          f"(initial {self.base_acc_initial:.4f}); budget remaining {self.budget.remaining()}")

    def tool_checkpoint(self, tag: str) -> ToolResult:
        self.checkpoints[tag] = self.model.state_dict()
        return ToolResult(f"saved checkpoint '{tag}'")

    def tool_rollback(self, tag: str) -> ToolResult:
        if tag not in self.checkpoints:
            raise ToolError(f"no checkpoint '{tag}', have {list(self.checkpoints)}")
        self.model.load_state_dict(self.checkpoints[tag])
        return ToolResult(f"restored '{tag}'")

    def tool_diff(self, tag: str = "initial") -> ToolResult:
        ref = self.checkpoints.get(tag)
        if ref is None:
            raise ToolError(f"no checkpoint '{tag}'")
        rows = []
        for k, v in self.model.params.items():
            r = ref.get(k)
            if r is None or r.shape != v.shape:
                rows.append(f"{k}: shape {None if r is None else list(r.shape)} -> {list(v.shape)}")
            else:
                d = (v - r)
                nz = int((d != 0).sum())
                if nz:
                    rows.append(f"{k}: {nz} values changed, |delta|_2={float(d.norm()):.4g}")
        return ToolResult("\n".join(rows) or "no change")

    def tool_note(self, text: str) -> ToolResult:
        self.notes.append(text)
        return ToolResult(f"noted ({len(self.notes)} notes)")

    def tool_submit(self) -> ToolResult:
        self.submitted = True
        return ToolResult("submitted; the current weights will be scored on sealed splits")

    # ====================================================== sealed scoring
    def final_report(self) -> dict:
        """Score on sealed splits. Not a tool: the agent never sees this."""
        rep = {"base_acc_initial": self.base_acc_initial, "base_acc_final": self._base_acc(),
               "budget_used": dict(self.budget.used), "n_params": self.model.n_params(),
               "audit_flags": self.ledger.flags(), "condition": self.condition}
        with torch.no_grad():
            for name, d in {**self.splits, **self._sealed}.items():
                x = torch.from_numpy(d["x"]).to(self.device)
                c = torch.cat([self.model(x[i:i + 256])[1] for i in range(0, len(x), 256)]).cpu().numpy()
                rep[f"auroc_{name}"] = auroc(c, d["y_concept"])
        rep["diff_from_initial"] = self.tool_diff("initial").text
        return rep


# JSON-schema style specs used by the agent loop. Keep in sync with tool_* methods.
_S, _I, _N, _A = {"type": "string"}, {"type": "integer"}, {"type": "number"}, {"type": "array"}
_LAYER = {"type": "string", "enum": list(CONV_LAYERS)}
_SPLIT = {"type": "string", "enum": ["train", "dev"]}
TOOL_SPECS: dict[str, dict] = {
    "inspect_model": {"desc": "Model structure, shapes, receptive fields, budget.", "params": {}},
    "get_tensor": {"desc": "Read raw values (or stats) of a parameter slice.",
                   "params": {"name": _S, "index": {"type": "array", "items": _I}, "stats_only": {"type": "boolean"}},
                   "required": ["name"]},
    "render_kernels": {"desc": "Image of conv kernels (conv1: all; conv2/3: one out_channel).",
                       "params": {"layer": _LAYER, "out_channel": _I}, "required": ["layer"]},
    "show_examples": {"desc": "Show training images with a given concept label (1 = has concept).",
                      "params": {"split": _SPLIT, "label": _I, "n": _I, "offset": _I}},
    "feature_map": {"desc": "Activation maps of one channel overlaid on chosen images.",
                    "params": {"layer": _LAYER, "channel": _I, "ids": {"type": "array", "items": _I}, "split": _SPLIT},
                    "required": ["layer", "channel", "ids"]},
    "top_activations": {"desc": "Receptive-field patches that most activate a channel.",
                        "params": {"layer": _LAYER, "channel": _I, "k": _I, "split": _SPLIT},
                        "required": ["layer", "channel"]},
    "channel_stats": {"desc": "Rank channels of a layer by how differently they respond to concept 1 vs 0.",
                      "params": {"layer": _LAYER, "split": _SPLIT}, "required": ["layer"]},
    "capture_activations": {"desc": "Store activations ('input' or a conv layer) into sandbox A[key].",
                            "params": {"layer": _S, "key": _S, "split": _SPLIT, "ids": {"type": "array", "items": _I}},
                            "required": ["layer", "key"]},
    "probe_ablate": {"desc": "Temporarily zero channels and report concept AUROC change. Model unchanged.",
                     "params": {"layer": _LAYER, "channels": {"type": "array", "items": _I}, "split": _SPLIT},
                     "required": ["layer", "channels"]},
    "run_sandbox": {"desc": "Run NumPy code. W(name) reads a parameter, A[key] captured arrays, "
                            "put arrays in out[key] to use them in edits. No torch, no training.",
                    "params": {"code": _S}, "required": ["code"]},
    "set_weights": {"desc": "Overwrite a parameter slice with values or a sandbox output.",
                    "params": {"name": _S, "index": {"type": "array", "items": _I}, "values": _A, "from_sandbox": _S},
                    "required": ["name"]},
    "add_to_weights": {"desc": "Add values (or a sandbox output) to a parameter slice.",
                       "params": {"name": _S, "index": {"type": "array", "items": _I}, "values": _A, "from_sandbox": _S},
                       "required": ["name"]},
    "scale_channel": {"desc": "Multiply one channel's kernel and bias.",
                      "params": {"layer": _LAYER, "channel": _I, "factor": _N}, "required": ["layer", "channel", "factor"]},
    "lowrank_update": {"desc": "W += alpha * outer(u, v) with u, v from sandbox outputs.",
                       "params": {"name": _S, "u_key": _S, "v_key": _S, "alpha": _N}, "required": ["name", "u_key", "v_key"]},
    "add_channel": {"desc": "Append a new output channel to a conv layer (kernel from values or sandbox).",
                    "params": {"layer": _LAYER, "kernel": _A, "from_sandbox": _S, "bias": _N}, "required": ["layer"]},
    "remove_channel": {"desc": "Delete a channel.", "params": {"layer": _LAYER, "channel": _I},
                       "required": ["layer", "channel"]},
    "connect": {"desc": "Set weights from src_layer channel into the next layer's channel or a head.",
                "params": {"src_layer": _LAYER, "src_channel": _I, "dst": _S, "dst_channel": _I,
                           "values": _A, "from_sandbox": _S},
                "required": ["src_layer", "src_channel", "dst", "dst_channel"]},
    "evaluate": {"desc": "Concept AUROC on a split and base-class accuracy. Charged.",
                 "params": {"split": _SPLIT}},
    "checkpoint": {"desc": "Save current weights under a tag.", "params": {"tag": _S}, "required": ["tag"]},
    "rollback": {"desc": "Restore a checkpoint ('initial' always exists).", "params": {"tag": _S}, "required": ["tag"]},
    "diff": {"desc": "What changed relative to a checkpoint.", "params": {"tag": _S}},
    "note": {"desc": "Write a note to your memory (hypotheses, results).", "params": {"text": _S}, "required": ["text"]},
    "submit": {"desc": "Finish. Current weights are scored on sealed data.", "params": {}},
}
