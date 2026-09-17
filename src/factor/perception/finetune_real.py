"""Fine-tune an estimator on hindsight-labelled real moves. / 실기 사후 라벨로 추정기 미세조정.

Reads ``move_*.npz`` written by :class:`factor.real.recorder.MoveRecorder` and fine-tunes a
local-label checkpoint on the real frames (translation only), with a small learning rate and an
L2 pull towards the simulation weights so that what the twin taught is kept. Needs no simulator.

Usage::

    python -m factor.perception.finetune_real --checkpoint estimator.pt --data logs/factor/real_data \\
        --out estimator_real.pt

실기 기록(move_*.npz)으로 국소 라벨 추정기를 미세조정한다. 시뮬 가중치 쪽으로 L2 로 당겨 트윈에서
배운 것을 지킨다. 시뮬레이터 불필요.
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import torch

from factor.perception.model import EstimatorConfig, StateEstimator


def load_real(directory: str) -> dict[str, np.ndarray]:
    images, grips, labels = [], [], []
    for path in sorted(glob.glob(os.path.join(directory, "move_*.npz"))):
        d = np.load(path)
        keep = ~np.isnan(d["labels"]).any(axis=1)
        images.append(d["images"][keep])
        grips.append(d["grips"][keep])
        labels.append(d["labels"][keep])
    if not images:
        raise SystemExit(f"no labelled frames in {directory}")
    return {
        "images": np.concatenate(images),
        "grips": np.concatenate(grips),
        "labels": np.concatenate(labels),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--anchor_weight", type=float, default=1e-3, help="L2 pull towards the sim weights")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = EstimatorConfig(**ck["config"])
    if config.label != "local":
        raise SystemExit("hindsight labels are local corrections: use a --local_correction checkpoint")
    model = StateEstimator(config).to(device)
    model.load_state_dict(ck["state_dict"], strict=False)
    reference = {k: v.detach().clone() for k, v in model.state_dict().items()}

    data = load_real(args.data)
    x = torch.from_numpy(data["images"]).permute(0, 3, 1, 2).float() / 255.0
    g = torch.from_numpy(data["grips"]).float()
    y = torch.from_numpy(data["labels"]).float() / config.pos_scale  # normalised translation
    n = len(x)
    val = torch.arange(n) % 5 == 0
    print(f"[finetune] {n} labelled real frames ({int(val.sum())} validation)", flush=True)

    def evaluate() -> float:
        model.eval()
        with torch.no_grad():
            xb, gb, yb = x[val].to(device), g[val].to(device), y[val].to(device)
            out = model(xb[:, :3], xb[:, 3:] if xb.shape[1] > 3 else None, gb)
            return float(((out["residual"][:, :2] - yb) ** 2).mean().sqrt() * config.pos_scale * 1e3)

    print(f"[finetune] real val RMS before {evaluate():.2f} mm", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    train_idx = torch.nonzero(~val).squeeze(1)
    for ep in range(args.epochs):
        model.train()
        perm = train_idx[torch.randperm(len(train_idx))]
        for s in range(0, len(perm), 64):
            idx = perm[s : s + 64]
            xb, gb, yb = x[idx].to(device), g[idx].to(device), y[idx].to(device)
            out = model(xb[:, :3], xb[:, 3:] if xb.shape[1] > 3 else None, gb)
            logvar = out["logvar"][:, :2].clamp(-6.0, 4.0)
            loss = (0.5 * (torch.exp(-logvar) * (out["residual"][:, :2] - yb) ** 2 + logvar)).mean()
            pull = sum(((p - reference[k]) ** 2).sum() for k, p in model.named_parameters())
            loss = loss + args.anchor_weight * pull
            opt.zero_grad()
            loss.backward()
            opt.step()
        if ep % 5 == 4 or ep == args.epochs - 1:
            print(f"[finetune] epoch {ep}  real val RMS {evaluate():.2f} mm", flush=True)
    torch.save(
        {"config": config.__dict__, "state_dict": model.state_dict(), "finetuned_on": args.data}, args.out
    )
    print(f"[finetune] saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
