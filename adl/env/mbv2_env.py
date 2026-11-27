"""Agent environment for MobileNetV2 on CIFAR-10.

Same research rules as :mod:`adl.env.environment` (the small-CNN environment):

* No gradient reaches the target network: every tool runs under
  ``torch.no_grad()`` and :func:`assert_no_grad` is checked after each call.
* ``probe`` is a temporary intervention; ``edit`` tools change the model and
  are charged; ``heldout`` (CIFAR-10 test) and ``shift`` (corrupted test) are
  sealed and only scored by :meth:`MBV2Environment.final_report`.
* Every call is charged to a :class:`Budget` and logged to the :class:`Ledger`.
  ``forward_images`` (images pushed through the network) is the common unit
  shared with ES, random search and backprop.

Labels: the agent sees images, labels and predictions of the ``train`` pool,
and only aggregate numbers on ``dev``.

Layers are addressed by the names in :mod:`adl.models.mobilenetv2`
(``stem``, ``b3.expand``, ``b3.dw``, ``b3.project``, ``b3.out``, ``last``,
``head``). Activations are the layer's output after BN and ReLU6 (after BN
only for ``project``; after the residual add for ``bK.out``).
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F

from adl.env.budget import Budget, BudgetExceeded, Ledger
from adl.env.environment import ToolError, ToolResult, _diverging, _grid, _png
from adl.env.guard import ArgError, validate_args
from adl.env.sandbox import SandboxViolation, run_code
from adl.models.mobilenetv2 import CLASSES, MBV2

EDIT_TOOLS = {"set_weights", "add_to_weights", "scale_channel", "lowrank_update",
              "add_hidden_channel", "remove_hidden_channel", "add_feature_channel",
              "recalibrate_bn"}
OBSERVATION_TOOLS = {"get_tensor", "render_kernels", "show_examples", "feature_map",
                     "top_activations", "channel_stats", "capture_activations", "probe",
                     "run_sandbox", "confusion", "stimulus_response", "probe_readout"}
PROTOCOL_TOOLS = {"hypothesis", "decide"}


@dataclass
class MBV2Guard:
    """Protocol and rollback for model 1 (pre-trained). For model 2 (random
    init) set ``max_acc_drop=None``: there is no ability to protect yet."""
    max_acc_drop: float | None = 0.02   # absolute drop of retention-set accuracy below initial
    require_hypothesis: bool = True
    require_contrast: bool = True       # a hypothesis needs show_examples to have been used
    max_scale_factor: float = 2.0

    def as_dict(self) -> dict:
        return asdict(self)


def assert_no_grad(model: MBV2) -> None:
    for k, v in model.named_parameters():
        if v.requires_grad or v.grad is not None:
            raise RuntimeError(f"gradient state found on parameter {k}")


class MBV2Environment:
    def __init__(self, model: MBV2, splits: dict, budget: Budget | None = None,
                 ledger: Ledger | None = None, condition: str = "full",
                 forbid_fitting: bool = False, guard: MBV2Guard | None = None,
                 pool_n: int = 5000, retention_n: int = 1000, batch_size: int = 256,
                 device: str = "cpu", max_values_per_read: int = 4096):
        assert condition in {"full", "no_semantics", "no_observation"}
        self.model = model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device, self.bs = device, batch_size
        tr = splits["train"]
        pool_n = min(pool_n, len(tr["y"]) - retention_n) if len(tr["y"]) > retention_n else len(tr["y"])
        self.splits = {"train": {"x": tr["x"][:pool_n], "y": tr["y"][:pool_n]}, "dev": splits["dev"]}
        self.retention = ({"x": tr["x"][-retention_n:], "y": tr["y"][-retention_n:]}
                          if len(tr["y"]) > retention_n else splits["dev"])
        self._sealed = {k: splits[k] for k in ("heldout", "shift") if k in splits}
        self.budget = budget or Budget()
        self.ledger = ledger or Ledger()
        self.condition, self.forbid_fitting, self.guard = condition, forbid_fitting, guard
        self.max_values = max_values_per_read
        self.version = 0
        self._cache: dict = {}
        self.checkpoints: dict[str, MBV2] = {"initial": copy.deepcopy(self.model)}
        self.sandbox_arrays: dict[str, np.ndarray] = {}
        self.sandbox_out: dict[str, np.ndarray] = {}
        self.notes: list[str] = []
        self.submitted = False
        self.retention_acc_initial = self._acc(self.retention)
        self.seen_examples = False
        self.hypothesis: dict | None = None
        self.pending: dict | None = None
        self.last_eval: dict[str, float] = {}
        self.cycles: list[dict] = []
        self.guard_events = {"invalid_args": 0, "protocol_blocks": 0, "auto_rollbacks": 0}

    # ================================================================ dispatch
    def available_tools(self) -> list[str]:
        tools = [n for n in TOOL_SPECS if n not in PROTOCOL_TOOLS or self.guard is not None]
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
        try:
            args = validate_args(self, name, args, TOOL_SPECS[name])
            self._check_ranges(name, args)
        except (ArgError, KeyError) as e:
            self.guard_events["invalid_args"] += 1
            res = ToolResult(f"ERROR (invalid arguments, nothing changed): {e}")
            self.ledger.log(name, args, res.text, rationale, ["invalid_args"])
            return res
        block = self._protocol_block(name) if self.guard else None
        if block:
            self.guard_events["protocol_blocks"] += 1
            res = ToolResult(f"BLOCKED by protocol: {block}\n{self.protocol_status()}")
            self.ledger.log(name, args, res.text, rationale, ["protocol_block"])
            return res
        snapshot = copy.deepcopy(self.model) if name in EDIT_TOOLS else None
        flags: list[str] = []
        ok = True
        try:
            with torch.no_grad():
                res = getattr(self, f"tool_{name}")(**args)
            if isinstance(res, tuple):
                res, flags = res
        except (BudgetExceeded, ToolError, SandboxViolation) as e:
            res, ok = ToolResult(f"ERROR: {e}"), False
        except Exception as e:  # surface agent mistakes as tool errors
            res, ok = ToolResult(f"ERROR: {type(e).__name__}: {e}"), False
        if snapshot is not None:
            if not ok:
                self._restore(snapshot)
            else:
                self.version += 1
                if self.guard is not None:
                    res, more = self._after_edit(name, args, res, snapshot)
                    flags = flags + more
        if self.guard is not None and ok:
            res = self._after_call(name, args, res)
        assert_no_grad(self.model)
        self.ledger.log(name, args, res.text, rationale, flags)
        return res

    def _check_ranges(self, name: str, a: dict) -> None:
        if "layer" in a and name not in {"capture_activations"}:
            self.model.layer_info(a["layer"])          # KeyError on unknown layer
        if "layer" in a and "channel" in a:
            n = self.model.channels(a["layer"])
            if not 0 <= a["channel"] < n:
                raise ArgError(f"channel {a['channel']} out of range: {a['layer']} has 0..{n - 1}")
        for key in ("channels",):
            if key in a and "layer" in a:
                n = self.model.channels(a["layer"])
                bad = [c for c in a[key] if not 0 <= c < n]
                if bad:
                    raise ArgError(f"channels {bad} out of range: {a['layer']} has 0..{n - 1}")
        if "block" in a and not 2 <= a["block"] <= 17:
            raise ArgError("block must be 2..17 (block 1 has no expansion layer)")
        if name == "scale_channel" and self.guard is not None:
            f, lim = abs(a["factor"]), self.guard.max_scale_factor
            if not 1 / lim <= f <= lim:
                raise ArgError(f"factor magnitude must be within [{1 / lim:.3g}, {lim:g}]")

    def _restore(self, snap: MBV2) -> None:
        self.model = copy.deepcopy(snap)
        self.version += 1

    # ================================================================ protocol
    def _protocol_block(self, name: str) -> str | None:
        g, p = self.guard, self.pending
        if p is not None:
            if p["stage"] == "verify" and name in EDIT_TOOLS | {"submit", "rollback", "hypothesis", "decide"}:
                return "the last edit is not verified yet. Call evaluate(split='dev') now."
            if p["stage"] == "decide" and name in EDIT_TOOLS | {"submit", "rollback", "hypothesis"}:
                return "decide on the last edit first: decide(verdict='accept'|'reject', reason=...)."
        if name == "decide" and (p is None or p["stage"] != "decide"):
            return "there is no evaluated edit to decide on."
        if name in EDIT_TOOLS and g.require_hypothesis and self.hypothesis is None:
            return "record a hypothesis first with `hypothesis`."
        if (name == "hypothesis" and g.require_contrast and not self.seen_examples
                and "show_examples" in self.available_tools()):
            return "look at data first: call show_examples (e.g. the misclassified images of a class)."
        return None

    def _after_edit(self, name, args, res, snapshot):
        acc = self._acc(self.retention)
        cycle = {"hypothesis": self.hypothesis, "edit": {"tool": name, "args": _short(args)},
                 "retention_acc_after_edit": acc}
        g = self.guard
        if g.max_acc_drop is not None and acc < self.retention_acc_initial - g.max_acc_drop:
            self._restore(snapshot)
            self.guard_events["auto_rollbacks"] += 1
            cycle["verdict"] = "auto_rollback"
            self.cycles.append(cycle)
            return ToolResult(f"{res.text}\nAUTO-ROLLBACK: retention accuracy fell to {acc:.4f} "
                              f"(initial {self.retention_acc_initial:.4f}, allowed drop {g.max_acc_drop}). "
                              f"The edit was undone. Your hypothesis stays active.", res.images), ["auto_rollback"]
        self.pending = {"stage": "verify", "snapshot": snapshot, "cycle": cycle,
                        "eval_before": dict(self.last_eval)}
        return ToolResult(f"{res.text}\nretention accuracy after edit: {acc:.4f} (initial "
                          f"{self.retention_acc_initial:.4f}). Next: evaluate(split='dev').", res.images), []

    def _after_call(self, name, args, res):
        if name == "show_examples":
            self.seen_examples = True
        elif name == "evaluate" and self.pending and self.pending["stage"] == "verify":
            cyc = self.pending["cycle"]
            cyc["eval"] = {"acc_before": self.pending["eval_before"].get("dev"),
                           "acc_after": self.last_eval.get("dev")}
            self.pending["stage"] = "decide"
            b = cyc["eval"]["acc_before"]
            pred = (self.hypothesis or {}).get("prediction", "")
            res = ToolResult(f"{res.text}\ndev accuracy before this edit: "
                             f"{'unknown' if b is None else f'{b:.4f}'}. Your prediction was {pred!r}. "
                             f"Now decide(verdict='accept'|'reject', reason=...).", res.images)
        return res

    def protocol_status(self) -> str:
        if self.guard is None:
            return ""
        p = self.pending
        if p is not None and p["stage"] == "verify":
            step = "VERIFY: evaluate(split='dev')."
        elif p is not None:
            step = "DECIDE: decide(verdict=..., reason=...)."
        elif self.hypothesis is None:
            step = "OBSERVE, then record a hypothesis."
        else:
            step = f"EDIT: one edit that tests {self.hypothesis['statement'][:100]!r}."
        n_acc = sum(c.get("verdict") == "accept" for c in self.cycles)
        return (f"[protocol] cycle {len(self.cycles) + 1}; accepted {n_acc}; auto-rollbacks "
                f"{self.guard_events['auto_rollbacks']}. Next -> {step}")

    # ================================================================ forward
    def _xs(self, split: str, ids=None) -> np.ndarray:
        if split not in self.splits:
            raise ToolError(f"split '{split}' is not accessible (use 'train' or 'dev')")
        x = self.splits[split]["x"]
        return x if ids is None else x[np.asarray(ids)]

    def _run(self, x: np.ndarray, capture: dict | None = None, intervene: dict | None = None,
             model: MBV2 | None = None, charge: bool = True):
        """Forward ``x`` (uint8). capture: {layer: (channels|None, mode)} with mode
        'mean' | 'max' (also returns argmax location) | 'map'. intervene:
        {layer: (channels, factor)} multiplies those activations temporarily."""
        m = model or self.model
        capture, intervene = capture or {}, intervene or {}
        if charge:
            self.budget.charge("forward_images", len(x))
        store: dict[str, list] = {k: [] for k in capture}
        hooks = []
        for layer, (chs, factor) in intervene.items():
            mod = m.modules_of(layer)["act"]
            idx = torch.as_tensor(list(chs), dtype=torch.long)

            def ihook(_m, _i, out, idx=idx, f=factor):
                out = out.clone()
                out[:, idx] = out[:, idx] * f
                return out
            hooks.append(mod.register_forward_hook(ihook))
        for layer, (chs, mode) in capture.items():
            mod = m.modules_of(layer)["act"]

            def chook(_m, _i, out, layer=layer, chs=chs, mode=mode):
                o = out if chs is None else out[:, list(chs)]
                if o.dim() == 2:
                    store[layer].append(o.cpu().numpy())
                elif mode == "mean":
                    store[layer].append(o.mean((2, 3)).cpu().numpy())
                elif mode == "grid":
                    store[layer].append(F.adaptive_avg_pool2d(o, 2).flatten(1).cpu().numpy())
                elif mode == "max":
                    flat = o.flatten(2)
                    v, i = flat.max(-1)
                    store[layer].append(np.stack([v.cpu().numpy(), i.cpu().numpy().astype(np.float32)], -1))
                else:
                    store[layer].append(o.cpu().numpy())
            hooks.append(mod.register_forward_hook(chook))
        logits = []
        try:
            for i in range(0, len(x), self.bs):
                xb = torch.from_numpy(x[i:i + self.bs]).to(self.device)
                logits.append(m(xb).cpu().numpy())
        finally:
            for h in hooks:
                h.remove()
        return np.concatenate(logits), {k: np.concatenate(v) for k, v in store.items()}

    def _acc(self, d: dict, model: MBV2 | None = None) -> float:
        lg, _ = self._run(d["x"], model=model, charge=False)
        return float((lg.argmax(1) == d["y"]).mean())

    def _pool_logits(self) -> np.ndarray:
        key = ("pool_logits", self.version)
        if key not in self._cache:
            self._cache = {k: v for k, v in self._cache.items() if k[1] == self.version}
            self._cache[key] = self._run(self.splits["train"]["x"])[0]
        return self._cache[key]

    def _pool_capture(self, layer: str, chs, mode: str) -> np.ndarray:
        key = ("cap", self.version, layer, None if chs is None else tuple(chs), mode)
        if key not in self._cache:
            self._cache[key] = self._run(self.splits["train"]["x"], {layer: (chs, mode)})[1][layer]
        return self._cache[key]

    def _no_images(self) -> bool:
        return self.condition == "no_semantics"

    def _tensor(self, name: str) -> torch.Tensor:
        t = self.model.named_tensors()
        if name not in t:
            raise ToolError(f"unknown tensor '{name}'. Examples: stem.weight, b3.dw.weight, "
                            f"b3.expand.bn.weight, head.weight, head.bias")
        return t[name]

    # ================================================================ observation
    def tool_inspect_model(self) -> ToolResult:
        m = self.model
        lines = [f"MobileNetV2 ({m.variant} variant), input {m.input_size}px "
                 f"(CIFAR 32px images are upsampled inside the model if needed), {m.n_params():,} parameters.",
                 "Each block K: expand 1x1 (+BN, ReLU6) -> depthwise 3x3 (+BN, ReLU6) -> project 1x1 (+BN, "
                 "linear); blocks marked 'res' add their input to the output. "
                 "Then last 1x1 -> global average pool -> head (10 classes: " + ", ".join(CLASSES) + ").",
                 "layer: kind in->out stride | receptive field in CIFAR pixels"]
        scale = m.input_size / 32
        for li in m.layers():
            rf = min(32, li.rf / scale)
            lines.append(f"  {li.name}: {li.kind} {li.in_ch}->{li.out_ch} s{li.stride}"
                         f"{' res' if li.residual else ''} | rf {rf:.1f}px")
        lines.append("Tensor names: <layer>.weight, <layer>.bn.{weight,bias,running_mean,running_var}, "
                     "head.weight, head.bias. Activations can also be read at bK.out (block output).")
        lines.append(f"budget remaining: {self.budget.remaining()}")
        return ToolResult("\n".join(lines))

    def tool_get_tensor(self, name: str, index: list[int] | None = None, stats_only: bool = False) -> ToolResult:
        t = self._tensor(name)
        for i in index or []:
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

    def tool_render_kernels(self, layer: str, channels: list[int] | None = None,
                            fold_bn: bool = True) -> ToolResult:
        """stem: RGB tiles; dw: one 3x3 map per channel; 1x1 layers / head: the
        weight matrix rows of the chosen output channels (top inputs as text)."""
        li = self.model.layer_info(layer)
        if li.kind == "out":
            raise ToolError("bK.out has no kernel; use bK.project")
        if fold_bn and li.kind != "head":
            w, _ = self.model.folded_kernel(layer)
        else:
            w = self.model.modules_of(layer)["conv"].weight
        w = w.cpu().numpy()
        chs = list(channels) if channels else list(range(min(64, w.shape[0])))
        sel = w[chs]
        self.budget.charge("weight_values_read", int(sel.size))
        tag = "BN-folded " if fold_bn and li.kind != "head" else ""
        if li.kind in {"stem", "dw"}:
            if self._no_images():
                return ToolResult(f"{layer} {tag}kernels {chs}: per-kernel l2 "
                                  f"{np.round(np.linalg.norm(sel.reshape(len(chs), -1), axis=1), 3).tolist()}")
            if li.kind == "stem":
                tiles = [((k - k.min()) / (k.max() - k.min() + 1e-8)).transpose(1, 2, 0) for k in sel]
                note = "RGB, min-max normalised per kernel"
            else:
                tiles = [_diverging(k[0]) for k in sel]
                note = "red=positive, blue=negative, own scale per tile"
            return ToolResult(f"{layer} {tag}3x3 kernels for channels {chs} ({note}; order left-to-right, "
                              f"top-to-bottom)", [_png(_grid(tiles, ncol=16), 12)])
        mat = sel.reshape(len(chs), -1)
        rows = []
        for c, r in zip(chs, mat):
            top = np.argsort(-np.abs(r))[:6]
            rows.append(f"out {c}: l2={np.linalg.norm(r):.3f}; strongest inputs " +
                        ", ".join(f"{j}:{r[j]:+.3f}" for j in top))
        text = f"{layer} {tag}1x1 weights ({mat.shape[1]} inputs)\n" + "\n".join(rows)
        if self._no_images():
            return ToolResult(text)
        return ToolResult(text + "\nimage: rows = chosen output channels, columns = input channels",
                          [_png(_diverging(mat), 4)])

    def tool_show_examples(self, cls: int | None = None, pred: int | None = None,
                           only_errors: bool = False, n: int = 16, offset: int = 0) -> ToolResult:
        """Train-pool images filtered by true class and/or predicted class."""
        y = self.splits["train"]["y"]
        p = self._pool_logits()
        pr = p.argmax(1)
        conf = _softmax(p).max(1)
        m = np.ones(len(y), bool)
        if cls is not None:
            m &= y == cls
        if pred is not None:
            m &= pr == pred
        if only_errors:
            m &= pr != y
        ids = np.where(m)[0][offset:offset + min(n, 32)]
        info = [f"{i}:{CLASSES[y[i]]}->{CLASSES[pr[i]]}({conf[i]:.2f})" for i in ids]
        text = (f"{int(m.sum())} train-pool images match; showing ids (true->pred(conf)): " + ", ".join(info))
        if self._no_images() or len(ids) == 0:
            return ToolResult(text)
        tiles = [self.splits["train"]["x"][i].transpose(1, 2, 0) / 255.0 for i in ids]
        return ToolResult(text, [_png(_grid(tiles, ncol=8), 3)])

    def tool_confusion(self, split: str = "train") -> ToolResult:
        if split == "dev":
            self.budget.charge("evaluations")
            lg, _ = self._run(self.splits["dev"]["x"])
        else:
            lg = self._pool_logits()
        y, pr = self.splits[split]["y"], lg.argmax(1)
        cm = np.zeros((10, 10), int)
        np.add.at(cm, (y, pr), 1)
        per = cm.diagonal() / np.maximum(cm.sum(1), 1)
        off = [(cm[i, j], i, j) for i in range(10) for j in range(10) if i != j]
        off.sort(reverse=True)
        lines = [f"{split}: accuracy {per.mean():.4f}" if split == "dev" else
                 f"{split} pool: accuracy {(pr == y).mean():.4f}",
                 "per-class accuracy: " + ", ".join(f"{CLASSES[c]} {per[c]:.3f}" for c in range(10)),
                 "largest confusions (count true->pred): " +
                 ", ".join(f"{n} {CLASSES[i]}->{CLASSES[j]}" for n, i, j in off[:10]),
                 "matrix rows=true, cols=pred:\n" + "\n".join(" ".join(f"{v:4d}" for v in r) for r in cm)]
        return ToolResult("\n".join(lines))

    def tool_feature_map(self, layer: str, channel: int, ids: list[int]) -> ToolResult:
        ids = list(ids)[:16]
        x = self._xs("train", ids)
        _, cap = self._run(x, {layer: ([channel], "map")})
        a = cap[layer][:, 0]
        text = f"{layer}[{channel}] on train ids={ids}: max per image={np.round(a.reshape(len(ids), -1).max(1), 3).tolist()}"
        if self._no_images():
            return ToolResult(text)
        tiles = []
        for img, fm in zip(x, a):
            img = img.transpose(1, 2, 0) / 255.0
            up = F.interpolate(torch.from_numpy(fm)[None, None], size=32, mode="nearest")[0, 0].numpy()
            up = np.clip(up / (a.max() + 1e-8), 0, 1)
            tiles += [img, 0.5 * img + 0.5 * np.stack([up, 0 * up, 0 * up], -1)]
        return ToolResult(text + " (pairs: image | activation in red, shared scale)",
                          [_png(_grid(tiles, ncol=8), 3)])

    def tool_top_activations(self, layer: str, channel: int, k: int = 16) -> ToolResult:
        """Train-pool images whose spatial max of this channel is highest, cropped to the receptive field."""
        k = min(k, 48)
        li = self.model.layer_info(layer)
        cap = self._pool_capture(layer, [channel], "max")[:, 0]
        val, loc = cap[:, 0], cap[:, 1].astype(int)
        y = self.splits["train"]["y"]
        order = np.argsort(-val)
        top = order[:k]
        hist = np.bincount(y[order[:100]], minlength=10)
        text = (f"top-{k} train ids for {layer}[{channel}] (id, max_act, class): "
                f"{[(int(i), round(float(val[i]), 3), CLASSES[y[i]]) for i in top]}. "
                f"class histogram of top-100: " + ", ".join(f"{CLASSES[c]} {hist[c]}" for c in np.argsort(-hist) if hist[c]))
        if self._no_images():
            return ToolResult(text)
        scale = self.model.input_size / 32
        fm_size = self._fmap_size(layer)
        rf = int(min(32, max(3, round(li.rf / scale))))
        patches = []
        for i in top:
            cy, cx = divmod(int(loc[i]), fm_size)
            py, px = (cy + 0.5) * 32 / fm_size, (cx + 0.5) * 32 / fm_size
            y0 = int(np.clip(round(py - rf / 2), 0, 32 - rf))
            x0 = int(np.clip(round(px - rf / 2), 0, 32 - rf))
            patches.append(self.splits["train"]["x"][i, :, y0:y0 + rf, x0:x0 + rf].transpose(1, 2, 0) / 255.0)
        return ToolResult(text + f" | image: {rf}x{rf}px receptive-field crops at the argmax",
                          [_png(_grid(patches, ncol=8), max(2, 96 // rf))])

    def _fmap_size(self, layer: str) -> int:
        key = ("fsize", self.version, layer)
        if key not in self._cache:
            _, cap = self._run(self.splits["train"]["x"][:1], {layer: ([0], "map")}, charge=False)
            self._cache[key] = cap[layer].shape[-1]
        return self._cache[key]

    def tool_channel_stats(self, layer: str, cls_a: int | None = None, cls_b: int | None = None,
                           top: int = 20) -> ToolResult:
        """Spatial-mean activation per class on the train pool. With cls_a/cls_b:
        channels ranked by d' between the two classes; otherwise by class selectivity."""
        a = self._pool_capture(layer, None, "mean")
        y = self.splits["train"]["y"]
        means = np.stack([a[y == c].mean(0) if (y == c).any() else np.zeros(a.shape[1]) for c in range(10)])
        sd = a.std(0) + 1e-8
        if cls_a is not None and cls_b is not None:
            d = (means[cls_a] - means[cls_b]) / sd
            order = np.argsort(-np.abs(d))[:top]
            rows = [f"ch{c}: {CLASSES[cls_a]}={means[cls_a, c]:.3f} {CLASSES[cls_b]}={means[cls_b, c]:.3f} d'={d[c]:+.3f}"
                    for c in order]
            head = f"{layer}: channels ranked by |d'| between {CLASSES[cls_a]} and {CLASSES[cls_b]}"
        else:
            best = means.argmax(0)
            sel = (means.max(0) - (means.sum(0) - means.max(0)) / 9) / sd
            order = np.argsort(-sel)[:top]
            rows = [f"ch{c}: prefers {CLASSES[best[c]]} (selectivity {sel[c]:.3f}, mean {a[:, c].mean():.3f})"
                    for c in order]
            dead = int((a.max(0) <= 1e-6).sum())
            head = f"{layer}: most class-selective channels ({dead} of {a.shape[1]} channels never activate)"
        return ToolResult(head + "\n" + "\n".join(rows))

    def tool_capture_activations(self, layer: str, key: str, ids: list[int] | None = None,
                                 mode: str = "mean", channels: list[int] | None = None) -> ToolResult:
        """Store activations of train-pool images in sandbox A[key] (+ A[key+'_y'] labels,
        A[key+'_pred'] predictions). layer='input' stores the images in [0,1]."""
        x = self._xs("train", ids)
        y = self.splits["train"]["y"] if ids is None else self.splits["train"]["y"][np.asarray(ids)]
        if layer == "input":
            arr = x.astype(np.float32) / 255.0
            self.budget.charge("forward_images", 0)
            lg = self._pool_logits() if ids is None else self._run(x)[0]
        else:
            self.model.layer_info(layer)
            if mode == "map" and len(x) > 512:
                raise ToolError("mode='map' is limited to 512 images; pass ids")
            lg, cap = self._run(x, {layer: (channels, mode)})
            arr = cap[layer]
        self.sandbox_arrays[key] = arr
        self.sandbox_arrays[key + "_y"] = y.copy()
        self.sandbox_arrays[key + "_pred"] = lg.argmax(1)
        return ToolResult(f"A['{key}'] shape={list(arr.shape)}; A['{key}_y'], A['{key}_pred'] shape={[len(y)]}")

    def tool_probe(self, layer: str, channels: list[int], factor: float = 0.0,
                   split: str = "dev") -> ToolResult:
        """Temporarily multiply channel activations by factor (0 = ablate) and report
        how accuracy and each class change. The model is not changed."""
        d = self.splits[split]
        if split == "dev":
            self.budget.charge("evaluations")
            base = self._run(d["x"])[0]
        else:
            base = self._pool_logits()
        new = self._run(d["x"], intervene={layer: (list(channels), factor)})[0]
        y = d["y"]
        pc0 = _per_class(base.argmax(1), y)
        pc1 = _per_class(new.argmax(1), y)
        delta = pc1 - pc0
        order = np.argsort(delta)
        return ToolResult(
            f"probe {layer}{list(channels)} x{factor} on {split}: accuracy {(base.argmax(1) == y).mean():.4f} -> "
            f"{(new.argmax(1) == y).mean():.4f} (model unchanged). per-class change: " +
            ", ".join(f"{CLASSES[c]} {delta[c]:+.3f}" for c in order))

    def tool_stimulus_response(self, layer: str, channel: int) -> ToolResult:
        """Response (spatial max) of one channel to synthetic stimuli: oriented
        gratings (8 orientations x 3 spatial periods), uniform colours, and a centre
        dot / ring. A tool for testing a hypothesis about what a filter extracts."""
        if layer == "head":
            raise ToolError("use a conv layer (stem ... last)")
        stim, names = _stimuli()
        _, cap = self._run(stim, {layer: ([channel], "max")})
        r = cap[layer][:, 0, 0]
        order = np.argsort(-r)
        text = (f"{layer}[{channel}] response to {len(r)} stimuli (highest first): " +
                ", ".join(f"{names[i]}={r[i]:.3f}" for i in order[:12]) +
                f" ... lowest: " + ", ".join(f"{names[i]}={r[i]:.3f}" for i in order[-4:]))
        ori = np.array([[r[names.index(f"grating ori={o} period={p}")] for o in range(0, 180, 22)] for p in (4, 8, 16)])
        text += "\norientation tuning (rows: period 4/8/16px, cols: 0..157.5 deg): " + str(np.round(ori, 3).tolist())
        if self._no_images():
            return ToolResult(text)
        tiles = [stim[i].transpose(1, 2, 0) / 255.0 for i in order[:16]]
        return ToolResult(text + " | image: the 16 strongest stimuli", [_png(_grid(tiles, ncol=8), 3)])

    def tool_probe_readout(self, layer: str, lam: float = 1.0) -> ToolResult:
        """Measurement instrument: how linearly decodable the 10 classes are from a
        layer. Features = 2x2 average-pooled activations; a closed-form ridge
        readout is fitted on the train pool by the environment and scored on dev.
        Nothing in the model changes. Charged as an evaluation."""
        if layer == "head":
            raise ToolError("use a conv layer (stem ... last)")
        self.budget.charge("evaluations")
        tr = self._pool_capture(layer, None, "grid")
        _, cap = self._run(self.splits["dev"]["x"], {layer: (None, "grid")})
        acc = _ridge_acc(tr, self.splits["train"]["y"], cap[layer], self.splits["dev"]["y"], lam)
        return ToolResult(f"linear readout of {layer} ({tr.shape[1]} features, ridge lam={lam}): "
                          f"dev accuracy {acc:.4f}")

    def tool_run_sandbox(self, code: str):
        self.budget.charge("sandbox_runs")

        def W(name):
            t = self._tensor(name)
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
                raise ToolError(f"no sandbox output '{from_sandbox}', have {list(self.sandbox_out)}")
            return np.asarray(self.sandbox_out[from_sandbox], dtype=np.float32)
        if values is None:
            raise ToolError("give values or from_sandbox")
        return np.asarray(values, dtype=np.float32)

    def _slice_write(self, name: str, index, arr: np.ndarray, add: bool) -> str:
        t = self._tensor(name)
        view = t
        for i in index or []:
            view = view[i]
        if arr.size == 1:
            arr = np.full(view.shape, float(arr.reshape(-1)[0]), np.float32)
        if tuple(arr.shape) != tuple(view.shape):
            if arr.size == view.numel():
                arr = arr.reshape(view.shape)
            else:
                raise ToolError(f"shape mismatch: target {list(view.shape)}, got {list(arr.shape)}")
        new = torch.from_numpy(arr).to(view)
        if add:
            view += new
        else:
            view.copy_(new)
        return f"{'added to' if add else 'set'} {name}{index or ''} ({view.numel()} values)"

    def tool_set_weights(self, name: str, index: list[int] | None = None, values=None,
                         from_sandbox: str | None = None) -> ToolResult:
        self.budget.charge("edits")
        return ToolResult(self._slice_write(name, index, self._values(values, from_sandbox), add=False))

    def tool_add_to_weights(self, name: str, index: list[int] | None = None, values=None,
                            from_sandbox: str | None = None) -> ToolResult:
        self.budget.charge("edits")
        return ToolResult(self._slice_write(name, index, self._values(values, from_sandbox), add=True))

    def tool_scale_channel(self, layer: str, channel: int, factor: float) -> ToolResult:
        """Multiply one channel's output (after BN, before ReLU6): BN gamma and beta
        are scaled. For head: the class row and bias."""
        self.budget.charge("edits")
        m = self.model.modules_of(layer)
        if m["bn"] is None:
            if layer != "head":
                raise ToolError("bK.out has no parameters; scale bK.project")
            m["conv"].weight[channel] *= factor
            m["conv"].bias[channel] *= factor
        else:
            m["bn"].weight[channel] *= factor
            m["bn"].bias[channel] *= factor
        return ToolResult(f"scaled {layer}[{channel}] by {factor}")

    def tool_lowrank_update(self, name: str, u_key: str, v_key: str, alpha: float = 1.0) -> ToolResult:
        """W += alpha * outer(u, v) for a 1x1 conv (out x in) or the head."""
        self.budget.charge("edits")
        t = self._tensor(name)
        if t.dim() == 4 and t.shape[2:] != (1, 1):
            raise ToolError("lowrank_update is for 1x1 convs and the head")
        u = self._values(from_sandbox=u_key).reshape(-1)
        v = self._values(from_sandbox=v_key).reshape(-1)
        if (len(u), len(v)) != (t.shape[0], t.shape[1]):
            raise ToolError(f"need u:{t.shape[0]}, v:{t.shape[1]}, got {len(u)}, {len(v)}")
        t += alpha * torch.from_numpy(np.outer(u, v)).to(t).view(t.shape)
        return ToolResult(f"{name} += {alpha} * u v^T")

    def tool_add_hidden_channel(self, block: int, expand_key: str, dw_key: str, project_key: str,
                                expand_bn: list[float] | None = None, dw_bn: list[float] | None = None) -> ToolResult:
        """Add one channel to block's expansion: expand row (in_ch), depthwise 3x3,
        project column (out_ch). BN as [gamma, beta, mean, var] (default identity)."""
        self.budget.charge("edits")
        e = torch.from_numpy(self._values(from_sandbox=expand_key).reshape(-1))
        d = torch.from_numpy(self._values(from_sandbox=dw_key).reshape(-1))
        p = torch.from_numpy(self._values(from_sandbox=project_key).reshape(-1))
        li = self.model.layer_info(f"b{block}.expand")
        lp = self.model.layer_info(f"b{block}.project")
        if len(e) != li.in_ch or len(d) != 9 or len(p) != lp.out_ch:
            raise ToolError(f"need expand {li.in_ch}, dw 9, project {lp.out_ch} values; "
                            f"got {len(e)}, {len(d)}, {len(p)}")
        c = self.model.add_hidden_channel(block, e, d.view(3, 3), p, tuple(expand_bn or (1, 0, 0, 1)),
                                          tuple(dw_bn or (1, 0, 0, 1)))
        return ToolResult(f"added hidden channel b{block}.expand[{c}] (now {c + 1} channels)")

    def tool_remove_hidden_channel(self, block: int, channel: int) -> ToolResult:
        self.budget.charge("edits")
        n = self.model.channels(f"b{block}.expand")
        if not 0 <= channel < n:
            raise ToolError(f"channel out of range 0..{n - 1}")
        self.model.remove_hidden_channel(block, channel)
        return ToolResult(f"removed b{block}.expand[{channel}]")

    def tool_add_feature_channel(self, last_key: str, head_key: str,
                                 bn: list[float] | None = None) -> ToolResult:
        """Add one channel to ``last`` (weights over the 320 inputs) and its 10 head weights."""
        self.budget.charge("edits")
        w = torch.from_numpy(self._values(from_sandbox=last_key).reshape(-1))
        h = torch.from_numpy(self._values(from_sandbox=head_key).reshape(-1))
        li = self.model.layer_info("last")
        if len(w) != li.in_ch or len(h) != 10:
            raise ToolError(f"need last {li.in_ch} and head 10 values, got {len(w)}, {len(h)}")
        c = self.model.add_feature_channel(w, h, tuple(bn or (1, 0, 0, 1)))
        return ToolResult(f"added last[{c}] with its head column")

    def tool_recalibrate_bn(self, n: int = 2000, layers: list[str] | None = None):
        """Re-estimate BN running mean/var from n train-pool images (forward only).
        A fixed statistical rule: flagged in the ledger."""
        n = min(n, len(self.splits["train"]["y"]))
        names = layers or [li.name for li in self.model.layers() if li.kind != "head"]
        x = self.splits["train"]["x"][:n]
        sums: dict[str, list] = {}
        hooks = []
        for name in names:
            bn = self.model.modules_of(name)["bn"]
            if bn is None:
                continue

            def h(_m, inp, _o, name=name):
                z = inp[0]
                sums.setdefault(name, []).append((z.mean((0, 2, 3)), (z ** 2).mean((0, 2, 3)), z.shape[0]))
            hooks.append(bn.register_forward_hook(h))
        try:
            self._run(x)
        finally:
            for hk in hooks:
                hk.remove()
        for name, parts in sums.items():
            tot = sum(p[2] for p in parts)
            mu = sum(p[0] * p[2] for p in parts) / tot
            var = sum(p[1] * p[2] for p in parts) / tot - mu ** 2
            bn = self.model.modules_of(name)["bn"]
            bn.running_mean.copy_(mu)
            bn.running_var.copy_(var.clamp_min(1e-5))
        # note: statistics are taken in one pass with the old statistics in
        # place for later layers; call again to settle deeper layers.
        return ToolResult(f"recalibrated BN statistics of {len(sums)} layers on {n} images"), \
            ["fixed_rule:bn_recalibration"]

    # ================================================================ verification
    def tool_evaluate(self, split: str = "dev") -> ToolResult:
        self.budget.charge("evaluations")
        d = self.splits[split]
        lg = self._pool_logits() if split == "train" else self._run(d["x"])[0]
        pr = lg.argmax(1)
        acc = float((pr == d["y"]).mean())
        nll = float(-np.log(_softmax(lg)[np.arange(len(pr)), d["y"]] + 1e-12).mean())
        self.last_eval[split] = acc
        pc = _per_class(pr, d["y"])
        return ToolResult(f"{split}: accuracy={acc:.4f} nll={nll:.4f}; per-class " +
                          ", ".join(f"{CLASSES[c]} {pc[c]:.3f}" for c in range(10)) +
                          f"; budget remaining {self.budget.remaining()}")

    def tool_checkpoint(self, tag: str) -> ToolResult:
        self.checkpoints[tag] = copy.deepcopy(self.model)
        return ToolResult(f"saved checkpoint '{tag}'")

    def tool_rollback(self, tag: str) -> ToolResult:
        if tag not in self.checkpoints:
            raise ToolError(f"no checkpoint '{tag}', have {list(self.checkpoints)}")
        self._restore(self.checkpoints[tag])
        return ToolResult(f"restored '{tag}'")

    def tool_diff(self, tag: str = "initial") -> ToolResult:
        if tag not in self.checkpoints:
            raise ToolError(f"no checkpoint '{tag}'")
        ref = self.checkpoints[tag].named_tensors()
        rows = []
        for k, v in self.model.named_tensors().items():
            r = ref.get(k)
            if r is None or r.shape != v.shape:
                rows.append(f"{k}: shape {None if r is None else list(r.shape)} -> {list(v.shape)}")
            else:
                nz = int((v != r).sum())
                if nz:
                    rows.append(f"{k}: {nz} values changed, |delta|_2={float((v - r).norm()):.4g}")
        return ToolResult("\n".join(rows) or "no change")

    def tool_hypothesis(self, statement: str, evidence: str, edit_plan: str, prediction: str) -> ToolResult:
        self.hypothesis = {"statement": statement, "evidence": evidence,
                           "edit_plan": edit_plan, "prediction": prediction}
        self.notes.append(f"HYPOTHESIS: {statement} | plan: {edit_plan} | predicts: {prediction}")
        return ToolResult("hypothesis recorded. Next: ONE edit that follows the plan.")

    def tool_decide(self, verdict: str, reason: str) -> ToolResult:
        p = self.pending
        cyc = p["cycle"]
        cyc["verdict"], cyc["reason"] = verdict, reason
        if verdict == "reject":
            self._restore(p["snapshot"])
            self.last_eval = dict(p["eval_before"])
            msg = "rejected: the edit was undone."
        else:
            msg = "accepted: the edit stays."
        self.cycles.append(cyc)
        self.pending, self.hypothesis = None, None
        self.notes.append(f"RESULT ({verdict}): {cyc['edit']['tool']} -> {cyc.get('eval')} ; {reason}")
        return ToolResult(msg + " Start the next cycle.")

    def tool_note(self, text: str) -> ToolResult:
        self.notes.append(text)
        return ToolResult(f"noted ({len(self.notes)} notes)")

    def tool_submit(self) -> ToolResult:
        self.submitted = True
        return ToolResult("submitted; the current weights will be scored on sealed splits")

    # ================================================================ sealed scoring
    def final_report(self) -> dict:
        """Score on all splits, sealed ones included. Not a tool."""
        rep = {"condition": self.condition, "n_params": self.model.n_params(),
               "budget_used": dict(self.budget.used), "audit_flags": self.ledger.flags(),
               "retention_acc_initial": self.retention_acc_initial,
               "retention_acc_final": self._acc(self.retention)}
        init = self.checkpoints["initial"]
        for name, d in {**self.splits, **self._sealed}.items():
            for tag, m in (("final", self.model), ("initial", init)):
                lg, _ = self._run(d["x"], model=m, charge=False)
                pr = lg.argmax(1)
                rep[f"acc_{name}_{tag}"] = float((pr == d["y"]).mean())
                if tag == "final" and name in ("heldout", "shift"):
                    rep[f"per_class_{name}"] = _per_class(pr, d["y"]).round(4).tolist()
                    if name == "shift":
                        rep["acc_shift_by_kind"] = {k: float((pr[d["kind"] == i] == d["y"][d["kind"] == i]).mean())
                                                    for i, k in enumerate(("noise", "blur", "contrast", "brightness"))}
        rep["diff_from_initial"] = self.tool_diff("initial").text
        rep["guard"] = self.guard.as_dict() if self.guard else None
        rep["guard_events"] = dict(self.guard_events)
        if self.guard:
            rep["cycles"] = self.cycles
            rep["cycle_verdicts"] = {v: sum(c.get("verdict") == v for c in self.cycles)
                                     for v in ("accept", "reject", "auto_rollback")}
            rep["unverified_edit_at_end"] = self.pending is not None
        return rep


def _stimuli() -> tuple[np.ndarray, list[str]]:
    yy, xx = np.mgrid[0:32, 0:32].astype(np.float32)
    ims, names = [], []
    for p in (4, 8, 16):
        for o in range(0, 180, 22):
            t = np.deg2rad(o)
            g = 0.5 + 0.5 * np.sin(2 * np.pi * (np.cos(t) * xx + np.sin(t) * yy) / p)
            ims.append(np.stack([g] * 3))
            names.append(f"grating ori={o} period={p}")
    for nm, c in (("red", (1, 0, 0)), ("green", (0, 1, 0)), ("blue", (0, 0, 1)), ("yellow", (1, 1, 0)),
                  ("cyan", (0, 1, 1)), ("magenta", (1, 0, 1)), ("white", (1, 1, 1)), ("black", (0, 0, 0)),
                  ("gray", (0.5, 0.5, 0.5))):
        ims.append(np.ones((3, 32, 32), np.float32) * np.array(c, np.float32)[:, None, None])
        names.append(f"colour {nm}")
    r = np.sqrt((yy - 15.5) ** 2 + (xx - 15.5) ** 2)
    for nm, m in (("dot", r < 3), ("ring", (r > 6) & (r < 9)), ("square", (np.abs(yy - 15.5) < 6) & (np.abs(xx - 15.5) < 6))):
        for pol, (fg, bg) in (("light", (1.0, 0.0)), ("dark", (0.0, 1.0))):
            ims.append(np.stack([np.where(m, fg, bg)] * 3).astype(np.float32))
            names.append(f"{pol} {nm}")
    return (np.stack(ims) * 255).round().astype(np.uint8), names


def _ridge_acc(ftr, ytr, fdv, ydv, lam: float) -> float:
    mu, sd = ftr.mean(0), ftr.std(0) + 1e-6
    z, zd = (ftr - mu) / sd, (fdv - mu) / sd
    t = np.eye(10)[ytr] - 0.1
    w = np.linalg.solve(z.T @ z + lam * len(ytr) * np.eye(z.shape[1]), z.T @ t)
    return float(((zd @ w).argmax(1) == ydv).mean())


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def _per_class(pred: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.array([(pred[y == c] == c).mean() if (y == c).any() else np.nan for c in range(10)])


def _short(args: dict) -> dict:
    return {k: (v if not isinstance(v, list) or len(v) <= 16 else f"<{len(v)} values>") for k, v in args.items()}


# JSON-schema style specs. Keep in sync with the tool_* methods.
_S, _I, _N, _A, _B = {"type": "string"}, {"type": "integer"}, {"type": "number"}, {"type": "array"}, {"type": "boolean"}
_IL = {"type": "array", "items": _I}
_NL = {"type": "array", "items": _N}
_SPLIT = {"type": "string", "enum": ["train", "dev"]}
TOOL_SPECS: dict[str, dict] = {
    "inspect_model": {"desc": "Layers, shapes, residual blocks, receptive fields, budget.", "params": {}},
    "get_tensor": {"desc": "Raw values (or stats) of a tensor slice, e.g. b3.dw.weight.",
                   "params": {"name": _S, "index": _IL, "stats_only": _B}, "required": ["name"]},
    "render_kernels": {"desc": "Image of kernels (stem: RGB; dw: 3x3 maps; 1x1 / head: weight rows). BN folded by default.",
                       "params": {"layer": _S, "channels": _IL, "fold_bn": _B}, "required": ["layer"]},
    "show_examples": {"desc": "Train images filtered by true class, predicted class, errors only.",
                      "params": {"cls": _I, "pred": _I, "only_errors": _B, "n": _I, "offset": _I}},
    "confusion": {"desc": "Per-class accuracy and largest confusions (train pool, or dev: charged as an evaluation).",
                  "params": {"split": _SPLIT}},
    "feature_map": {"desc": "Activation map of one channel on chosen train images.",
                    "params": {"layer": _S, "channel": _I, "ids": _IL}, "required": ["layer", "channel", "ids"]},
    "top_activations": {"desc": "Receptive-field crops that most activate a channel, with their class histogram.",
                        "params": {"layer": _S, "channel": _I, "k": _I}, "required": ["layer", "channel"]},
    "channel_stats": {"desc": "Class-selective channels of a layer, or channels separating two classes.",
                      "params": {"layer": _S, "cls_a": _I, "cls_b": _I, "top": _I}, "required": ["layer"]},
    "capture_activations": {"desc": "Store activations ('input' or a layer) of train images in sandbox A[key].",
                            "params": {"layer": _S, "key": _S, "ids": _IL, "mode": {"type": "string", "enum": ["mean", "max", "map"]},
                                       "channels": _IL}, "required": ["layer", "key"]},
    "stimulus_response": {"desc": "Response of a channel to synthetic gratings, colours and shapes (orientation tuning).",
                          "params": {"layer": _S, "channel": _I}, "required": ["layer", "channel"]},
    "probe_readout": {"desc": "Instrument: dev accuracy of a ridge readout fitted by the environment on a layer's pooled activations.",
                      "params": {"layer": _S, "lam": _N}, "required": ["layer"]},
    "probe": {"desc": "Temporarily scale channels (0 = ablate) and see accuracy per class. Model unchanged.",
              "params": {"layer": _S, "channels": _IL, "factor": _N, "split": _SPLIT}, "required": ["layer", "channels"]},
    "run_sandbox": {"desc": "Run NumPy code: W(name) reads a tensor, A[key] captured arrays; put arrays in out[key].",
                    "params": {"code": _S}, "required": ["code"]},
    "set_weights": {"desc": "Overwrite a tensor slice with values or a sandbox output.",
                    "params": {"name": _S, "index": _IL, "values": _A, "from_sandbox": _S}, "required": ["name"]},
    "add_to_weights": {"desc": "Add values (or a sandbox output) to a tensor slice.",
                       "params": {"name": _S, "index": _IL, "values": _A, "from_sandbox": _S}, "required": ["name"]},
    "scale_channel": {"desc": "Multiply a channel's output (BN gamma and beta; for head the class row).",
                      "params": {"layer": _S, "channel": _I, "factor": _N}, "required": ["layer", "channel", "factor"]},
    "lowrank_update": {"desc": "W += alpha * outer(u, v) on a 1x1 conv or the head, u/v from sandbox.",
                       "params": {"name": _S, "u_key": _S, "v_key": _S, "alpha": _N}, "required": ["name", "u_key", "v_key"]},
    "add_hidden_channel": {"desc": "New channel in block K's expansion: expand row, depthwise 3x3, project column (sandbox keys).",
                           "params": {"block": _I, "expand_key": _S, "dw_key": _S, "project_key": _S,
                                      "expand_bn": _NL, "dw_bn": _NL},
                           "required": ["block", "expand_key", "dw_key", "project_key"]},
    "remove_hidden_channel": {"desc": "Delete one expansion channel of block K.",
                              "params": {"block": _I, "channel": _I}, "required": ["block", "channel"]},
    "add_feature_channel": {"desc": "New channel in `last` (320 weights) with its 10 head weights (sandbox keys).",
                            "params": {"last_key": _S, "head_key": _S, "bn": _NL}, "required": ["last_key", "head_key"]},
    "recalibrate_bn": {"desc": "Re-estimate BN statistics from train images (forward only; flagged as a fixed rule).",
                       "params": {"n": _I, "layers": {"type": "array", "items": _S}}},
    "evaluate": {"desc": "Accuracy, NLL and per-class accuracy on a split. Charged.", "params": {"split": _SPLIT}},
    "checkpoint": {"desc": "Save the current model under a tag.", "params": {"tag": _S}, "required": ["tag"]},
    "rollback": {"desc": "Restore a checkpoint ('initial' always exists).", "params": {"tag": _S}, "required": ["tag"]},
    "diff": {"desc": "What changed relative to a checkpoint.", "params": {"tag": _S}},
    "note": {"desc": "Write a note to memory.", "params": {"text": _S}, "required": ["text"]},
    "hypothesis": {"desc": "Register what the next edit tests (required before every edit).",
                   "params": {"statement": _S, "evidence": _S, "edit_plan": _S, "prediction": _S},
                   "required": ["statement", "evidence", "edit_plan", "prediction"]},
    "decide": {"desc": "After evaluate: accept, or reject (undoes the edit).",
               "params": {"verdict": {"type": "string", "enum": ["accept", "reject"]}, "reason": _S},
               "required": ["verdict", "reason"]},
    "submit": {"desc": "Finish. The model is scored on sealed data.", "params": {}},
}
