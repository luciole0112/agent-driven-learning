"""Linear readout (the environment's probe_readout instrument) of saved models.

    python scripts/step0_readout.py runs/mbv2/R0c.pt runs/mbv2/baselines/R0c_es_stem_s0_b2000000.pt \
        --layers stem b3.out last --out runs/mbv2/step0_readout.json

Model 2's stage a (stem programming) is judged by the readout of the stem, since
the layers above stay random (docs/cifar_mbv2_design_v1.md, section 4.3).
Ridge on 2x2-pooled activations: fitted on the train pool (5,000) and scored on dev.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from adl.data.cifar10 import make_cifar_splits
from adl.env.budget import Budget
from adl.env.mbv2_env import MBV2Environment


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--layers", nargs="+", default=["stem", "b3.out", "last"])
    ap.add_argument("--out", default="runs/mbv2/step0_readout.json")
    a = ap.parse_args()
    sp = make_cifar_splits("data")
    out = Path(a.out)
    rep = json.loads(out.read_text()) if out.exists() else {}
    for path in a.models:
        m = torch.load(path, map_location="cuda", weights_only=False)
        env = MBV2Environment(m, sp, budget=Budget(), device="cuda")
        rep[path] = {}
        for layer in a.layers:
            txt = env.tool_probe_readout(layer).text
            rep[path][layer] = float(txt.rsplit(" ", 1)[1])
        print(path, rep[path], flush=True)
        out.write_text(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
