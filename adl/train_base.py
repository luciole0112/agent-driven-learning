"""Pre-train the base CNN on base shape classes with ordinary backprop.

This is the one place in the project where gradients touch the target
network, and it is allowed by the research rules (the base model is
"partially trained" on tasks unrelated to the new concept). The concept head
is never trained here and stays at zero.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from adl.data.shapes import SPLIT_SEEDS, make_dataset
from adl.models.cnn import EditableCNN


def train_base(n_train: int = 6000, n_val: int = 1000, epochs: int = 8,
               lr: float = 2e-3, batch: int = 128, seed: int = 0,
               device: str = "cpu", log=print) -> tuple[EditableCNN, dict]:
    torch.manual_seed(seed)
    tr = make_dataset(n_train, SPLIT_SEEDS["base_train"] + 100 * seed)
    va = make_dataset(n_val, SPLIT_SEEDS["base_val"] + 100 * seed)
    model = EditableCNN(seed=seed)
    trainable = [k for k in model.params if not k.startswith("head_concept")]
    for k in trainable:
        model.params[k] = model.params[k].to(device).requires_grad_(True)
    for k in model.params:
        model.params[k] = model.params[k].to(device)
    opt = torch.optim.Adam([model.params[k] for k in trainable], lr=lr)
    x = torch.from_numpy(tr["x"]).to(device)
    y = torch.from_numpy(tr["y_base"]).to(device)
    for ep in range(epochs):
        perm = torch.randperm(len(x), device=device)
        tot = 0.0
        for i in range(0, len(x), batch):
            idx = perm[i:i + batch]
            logits, _ = model(x[idx])
            loss = F.cross_entropy(logits, y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        acc = base_accuracy(model, va, device)
        log(f"epoch {ep + 1}: loss {tot / len(x):.4f} val_acc {acc:.4f}")
    for k in model.params:
        model.params[k] = model.params[k].detach().cpu()
    info = {"val_acc": base_accuracy(model, va, "cpu"), "seed": seed,
            "n_train": n_train, "epochs": epochs, "n_params": model.n_params()}
    return model, info


@torch.no_grad()
def base_accuracy(model: EditableCNN, data: dict, device: str = "cpu") -> float:
    x = torch.from_numpy(data["x"]).to(device)
    preds = []
    for i in range(0, len(x), 512):
        logits, _ = model(x[i:i + 512])
        preds.append(logits.argmax(1).cpu())
    return float((torch.cat(preds).numpy() == data["y_base"]).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/base/base_seed0.pt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    model, info = train_base(seed=args.seed, epochs=args.epochs, device=args.device)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out)
    out.with_suffix(".json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    np.set_printoptions(precision=3)
    main()
