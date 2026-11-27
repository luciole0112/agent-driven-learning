"""MobileNetV2 for CIFAR-10, with a layer catalogue an agent can address by name.

Two variants (see docs/cifar_mbv2_design_v1.md):

* ``variant="imagenet"``: torchvision's MobileNetV2 with the ImageNet layout
  (stride-2 stem). CIFAR images (32x32) are upsampled to ``input_size`` inside
  :meth:`MBV2.forward`. Used with ImageNet pre-trained weights (model 1).
* ``variant="cifar"``: the same blocks with the stem and the first 24-channel
  block at stride 1, so a 32x32 input ends at 4x4. Used from random init
  (model 2). 2.24M parameters with a 10-class head.

Every conv/BN/linear the agent may touch gets a short name:

=================  ==========================================================
``stem``           3x3 conv 3->32 (+BN, ReLU6)
``b1.dw``          block 1 depthwise 3x3 (+BN, ReLU6)   (block 1 has no expand)
``b1.project``     block 1 pointwise 1x1 (+BN, no activation)
``bK.expand``      block K>=2 pointwise 1x1 expand (+BN, ReLU6)
``bK.dw``          block K depthwise 3x3 (+BN, ReLU6)
``bK.project``     block K pointwise 1x1 project (+BN, linear)
``bK.out``         block K output (after the residual add, if the block has one)
``last``           1x1 conv 320->1280 (+BN, ReLU6)
``head``           linear 1280->10 (after global average pooling)
=================  ==========================================================

Parameter names are ``<layer>.weight`` for the conv/linear weight, ``head.bias``,
and ``<layer>.bn.{weight,bias,running_mean,running_var}`` for BatchNorm.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLASSES = ("airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck")


@dataclass
class LayerInfo:
    name: str
    kind: str            # stem | expand | dw | project | last | head
    block: int           # 0 for stem, 1..17, 18 for last, 19 for head
    in_ch: int
    out_ch: int
    stride: int
    residual: bool       # project layer of a block with a skip connection
    rf: int = 0          # receptive field (network-input pixels)
    jump: int = 1        # cumulative stride (network-input pixels)


class MBV2(nn.Module):
    """Wrapper that owns a torchvision MobileNetV2 and exposes named layers."""

    def __init__(self, variant: str = "cifar", num_classes: int = 10,
                 input_size: int | None = None, width_mult: float = 1.0):
        super().__init__()
        assert variant in {"cifar", "imagenet"}
        self.variant = variant
        self.net = torchvision.models.mobilenet_v2(num_classes=num_classes, width_mult=width_mult)
        if variant == "cifar":
            self.net.features[0][0].stride = (1, 1)
            self.net.features[2].conv[1][0].stride = (1, 1)
            self.input_size = 32
            mean, std = CIFAR_MEAN, CIFAR_STD
        else:
            self.input_size = input_size or 128
            mean, std = IMAGENET_MEAN, IMAGENET_STD
        for m in self.net.modules():          # in-place activations break hooks
            if isinstance(m, nn.ReLU6):
                m.inplace = False
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))
        for p in self.parameters():
            p.requires_grad_(False)

    # ------------------------------------------------------------ loading
    @classmethod
    def imagenet_pretrained(cls, input_size: int = 128, weights: str = "IMAGENET1K_V1",
                            weights_path: str | None = None) -> "MBV2":
        """ImageNet weights for the trunk; a fresh zero head for CIFAR-10."""
        m = cls("imagenet", num_classes=10, input_size=input_size)
        if weights_path:
            sd = torch.load(weights_path, map_location="cpu")
        else:
            w = torchvision.models.MobileNet_V2_Weights[weights]
            sd = w.get_state_dict(progress=False)
        sd = {k: v for k, v in sd.items() if not k.startswith("classifier.")}
        missing, _ = m.net.load_state_dict(sd, strict=False)
        assert set(missing) == {"classifier.1.weight", "classifier.1.bias"}, missing
        m.net.classifier[1].weight.zero_()
        m.net.classifier[1].bias.zero_()
        return m

    @classmethod
    def random_init(cls, seed: int = 0) -> "MBV2":
        torch.manual_seed(seed)
        m = cls("cifar")
        for p in m.parameters():
            p.requires_grad_(False)
        return m

    # ------------------------------------------------------------ forward
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """x: uint8 (N,3,32,32) or float in [0,1]."""
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        if self.input_size != x.shape[-1]:
            x = F.interpolate(x, size=self.input_size, mode="bilinear", align_corners=False)
        return (x - self.mean) / self.std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.preprocess(x))

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Pooled 1280-d features (input of ``head``)."""
        f = self.net.features(self.preprocess(x))
        return F.adaptive_avg_pool2d(f, 1).flatten(1)

    # ------------------------------------------------------------ catalogue
    def modules_of(self, layer: str) -> dict[str, nn.Module]:
        """{'conv': Conv2d | Linear, 'bn': BatchNorm2d | None, 'act': module whose
        output is the layer's activation}."""
        f = self.net.features
        if layer == "stem":
            return {"conv": f[0][0], "bn": f[0][1], "act": f[0][2]}
        if layer == "last":
            return {"conv": f[18][0], "bn": f[18][1], "act": f[18][2]}
        if layer == "head":
            lin = self.net.classifier[1]
            return {"conv": lin, "bn": None, "act": lin}
        blk, part = _split(layer)
        seq = f[blk].conv
        if part == "out":
            return {"conv": None, "bn": None, "act": f[blk]}
        if blk == 1:
            table = {"dw": (seq[0][0], seq[0][1], seq[0][2]), "project": (seq[1], seq[2], seq[2])}
        else:
            table = {"expand": (seq[0][0], seq[0][1], seq[0][2]),
                     "dw": (seq[1][0], seq[1][1], seq[1][2]),
                     "project": (seq[2], seq[3], seq[3])}
        if part not in table:
            raise KeyError(f"block {blk} has no '{part}' layer")
        c, b, a = table[part]
        return {"conv": c, "bn": b, "act": a}

    def layers(self) -> list[LayerInfo]:
        out: list[LayerInfo] = []
        f = self.net.features

        def conv_info(name, kind, blk, conv, residual=False):
            out.append(LayerInfo(name, kind, blk, conv.in_channels, conv.out_channels,
                                 conv.stride[0], residual))

        conv_info("stem", "stem", 0, f[0][0])
        for k in range(1, 18):
            blk = f[k]
            if k > 1:
                conv_info(f"b{k}.expand", "expand", k, self.modules_of(f"b{k}.expand")["conv"])
            conv_info(f"b{k}.dw", "dw", k, self.modules_of(f"b{k}.dw")["conv"])
            conv_info(f"b{k}.project", "project", k, self.modules_of(f"b{k}.project")["conv"],
                      residual=bool(blk.use_res_connect))
        conv_info("last", "last", 18, f[18][0])
        lin = self.net.classifier[1]
        out.append(LayerInfo("head", "head", 19, lin.in_features, lin.out_features, 1, False))
        # receptive fields
        rf, jump = 1, 1
        for li in out:
            if li.kind == "head":
                li.rf, li.jump = self.input_size, jump
                continue
            k = 3 if li.kind in {"stem", "dw"} else 1
            rf = rf + (k - 1) * jump
            jump = jump * li.stride
            li.rf, li.jump = rf, jump
        return out

    def layer_names(self, include_out: bool = False) -> list[str]:
        names = [li.name for li in self.layers()]
        if include_out:
            names += [f"b{k}.out" for k in range(1, 18)]
        return names

    def layer_info(self, layer: str) -> LayerInfo:
        if layer.endswith(".out"):
            blk, _ = _split(layer)
            proj = self.layer_info(f"b{blk}.project")
            return LayerInfo(layer, "out", blk, proj.out_ch, proj.out_ch, 1,
                             proj.residual, proj.rf, proj.jump)
        for li in self.layers():
            if li.name == layer:
                return li
        raise KeyError(f"unknown layer '{layer}'")

    def channels(self, layer: str) -> int:
        return self.layer_info(layer).out_ch

    def named_tensors(self) -> dict[str, torch.Tensor]:
        """Agent-facing name -> tensor (shared storage, edit in place)."""
        out: dict[str, torch.Tensor] = {}
        for li in self.layers():
            m = self.modules_of(li.name)
            out[f"{li.name}.weight"] = m["conv"].weight
            if li.kind == "head":
                out["head.bias"] = m["conv"].bias
            if m["bn"] is not None:
                bn = m["bn"]
                for a in ("weight", "bias", "running_mean", "running_var"):
                    out[f"{li.name}.bn.{a}"] = getattr(bn, a)
        return out

    def n_params(self) -> int:
        return sum(p.numel() for p in self.net.parameters())

    def folded_kernel(self, layer: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Conv weight and bias with eval-mode BN folded in: y = W' * x + b'."""
        m = self.modules_of(layer)
        w = m["conv"].weight
        if m["bn"] is None:
            return w.clone(), (m["conv"].bias.clone() if m["conv"].bias is not None else torch.zeros(w.shape[0]))
        bn = m["bn"]
        s = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        return w * s.view(-1, *([1] * (w.dim() - 1))), bn.bias - bn.running_mean * s

    # ------------------------------------------------------------ structure edits
    def add_hidden_channel(self, block: int, expand_w: torch.Tensor, dw_w: torch.Tensor,
                           project_w: torch.Tensor, expand_bn: tuple = (1.0, 0.0, 0.0, 1.0),
                           dw_bn: tuple = (1.0, 0.0, 0.0, 1.0)) -> int:
        """Append one channel to the expansion of block ``block`` (>=2).

        expand_w: (in_ch,), dw_w: (3,3), project_w: (out_ch,). BN tuples are
        (gamma, beta, running_mean, running_var). With ``project_w`` = 0 the
        network's function is unchanged. Returns the new channel's index.
        """
        if block < 2 or block > 17:
            raise ValueError("hidden channels can be added to blocks 2..17 (block 1 has no expansion)")
        seq = self.net.features[block].conv
        e_conv, e_bn = seq[0][0], seq[0][1]
        d_conv, d_bn = seq[1][0], seq[1][1]
        p_conv = seq[2]
        n = e_conv.out_channels
        seq[0][0] = _grow_conv_out(e_conv, expand_w.view(1, -1, 1, 1))
        seq[0][1] = _grow_bn(e_bn, expand_bn)
        seq[1][0] = _grow_dw(d_conv, dw_w.view(1, 1, 3, 3))
        seq[1][1] = _grow_bn(d_bn, dw_bn)
        seq[2] = _grow_conv_in(p_conv, project_w.view(-1, 1, 1, 1))
        return n

    def remove_hidden_channel(self, block: int, channel: int) -> None:
        if block < 2 or block > 17:
            raise ValueError("hidden channels exist in blocks 2..17")
        seq = self.net.features[block].conv
        keep = [i for i in range(seq[0][0].out_channels) if i != channel]
        idx = torch.tensor(keep)
        seq[0][0] = _select_conv(seq[0][0], out_idx=idx)
        seq[0][1] = _select_bn(seq[0][1], idx)
        seq[1][0] = _select_conv(seq[1][0], out_idx=idx, dw=True)
        seq[1][1] = _select_bn(seq[1][1], idx)
        seq[2] = _select_conv(seq[2], in_idx=idx)

    def add_feature_channel(self, last_w: torch.Tensor, head_w: torch.Tensor,
                            bn: tuple = (1.0, 0.0, 0.0, 1.0)) -> int:
        """Append one channel to ``last`` (320 -> 1281) and its column in ``head``."""
        f18 = self.net.features[18]
        n = f18[0].out_channels
        f18[0] = _grow_conv_out(f18[0], last_w.view(1, -1, 1, 1))
        f18[1] = _grow_bn(f18[1], bn)
        lin = self.net.classifier[1]
        new = nn.Linear(lin.in_features + 1, lin.out_features)
        new.weight.data = torch.cat([lin.weight.data, head_w.view(-1, 1).to(lin.weight)], 1)
        new.bias.data = lin.bias.data.clone()
        self.net.classifier[1] = _frozen(new)
        return n


def _split(layer: str) -> tuple[int, str]:
    if not layer.startswith("b") or "." not in layer:
        raise KeyError(f"unknown layer '{layer}'")
    b, part = layer[1:].split(".", 1)
    blk = int(b)
    if not 1 <= blk <= 17:
        raise KeyError(f"block index must be 1..17, got {blk}")
    return blk, part


def _frozen(m: nn.Module) -> nn.Module:
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def _like_conv(c: nn.Conv2d, in_ch: int, out_ch: int, groups: int) -> nn.Conv2d:
    return nn.Conv2d(in_ch, out_ch, c.kernel_size, c.stride, c.padding, groups=groups,
                     bias=c.bias is not None).to(c.weight.device)


def _grow_conv_out(c: nn.Conv2d, w_new: torch.Tensor) -> nn.Conv2d:
    n = _like_conv(c, c.in_channels, c.out_channels + 1, c.groups)
    n.weight.data = torch.cat([c.weight.data, w_new.to(c.weight)], 0)
    return _frozen(n)


def _grow_conv_in(c: nn.Conv2d, w_new: torch.Tensor) -> nn.Conv2d:
    n = _like_conv(c, c.in_channels + 1, c.out_channels, c.groups)
    n.weight.data = torch.cat([c.weight.data, w_new.to(c.weight)], 1)
    return _frozen(n)


def _grow_dw(c: nn.Conv2d, w_new: torch.Tensor) -> nn.Conv2d:
    n = _like_conv(c, c.in_channels + 1, c.out_channels + 1, c.groups + 1)
    n.weight.data = torch.cat([c.weight.data, w_new.to(c.weight)], 0)
    return _frozen(n)


def _grow_bn(b: nn.BatchNorm2d, vals: tuple) -> nn.BatchNorm2d:
    g, be, mu, var = (float(v) for v in vals)
    n = nn.BatchNorm2d(b.num_features + 1, eps=b.eps, momentum=b.momentum).to(b.weight.device)
    n.weight.data = torch.cat([b.weight.data, b.weight.new_tensor([g])])
    n.bias.data = torch.cat([b.bias.data, b.bias.new_tensor([be])])
    n.running_mean = torch.cat([b.running_mean, b.running_mean.new_tensor([mu])])
    n.running_var = torch.cat([b.running_var, b.running_var.new_tensor([var])])
    n.num_batches_tracked = b.num_batches_tracked.clone()
    n.eval()
    return _frozen(n)


def _select_conv(c: nn.Conv2d, out_idx=None, in_idx=None, dw: bool = False) -> nn.Conv2d:
    w = c.weight.data
    if out_idx is not None:
        w = w[out_idx]
    if in_idx is not None:
        w = w[:, in_idx]
    if dw:
        n = _like_conv(c, w.shape[0], w.shape[0], w.shape[0])
    else:
        n = _like_conv(c, w.shape[1], w.shape[0], 1)
    n.weight.data = w.clone()
    return _frozen(n)


def _select_bn(b: nn.BatchNorm2d, idx) -> nn.BatchNorm2d:
    n = nn.BatchNorm2d(len(idx), eps=b.eps, momentum=b.momentum).to(b.weight.device)
    n.weight.data = b.weight.data[idx].clone()
    n.bias.data = b.bias.data[idx].clone()
    n.running_mean = b.running_mean[idx].clone()
    n.running_var = b.running_var[idx].clone()
    n.eval()
    return _frozen(n)
