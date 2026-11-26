"""Our extension: does 'thinking longer' at test time help with deeper expressions?

The Looped Transformer shares one layer across all loops, so we can run it with
MORE (or fewer) loops at test time than it was trained with, without retraining.
We measure test accuracy for every depth at loops = 1 .. --max_loops.

  python loops.py --run base_looped            # model trained on all depths
  python loops.py --run depth4_looped          # trained on shallow only: does extra looping help unseen depths?
Writes figures/loops_<run>.csv and figures/loops_<run>.png
"""
import argparse
import json
import os
from collections import defaultdict

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import Vocab, ListOpsDataset, load_splits, make_loader
from models import build_model


@torch.no_grad()
def acc_by_depth(model, loader, items, device, loops):
    hit, tot = defaultdict(int), defaultdict(int)
    model.eval()
    for b in loader:
        p = model(b["ids"].to(device), b["lengths"], loops=loops).argmax(-1).cpu()
        for i, pred, y in zip(b["index"].tolist(), p.tolist(), b["targets"].tolist()):
            d = items[i]["depth"]
            hit[d] += int(pred == y)
            tot[d] += 1
    return {d: hit[d] / tot[d] for d in sorted(tot)}, sum(hit.values()) / sum(tot.values()), dict(tot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="base_looped")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--max_loops", type=int, default=12)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = os.path.join(args.runs, args.run)
    cfg = json.load(open(os.path.join(d, "config.json")))
    splits = load_splits()
    vocab = Vocab([s for s, _ in splits["train"]])
    model, _ = build_model("looped", len(vocab), vocab.pad_id, 0.0, cfg.get("pos", "learned"),
                           **{k: cfg["model_cfg"].get(k) for k in ("d_model", "loops")})
    model.load_state_dict(torch.load(os.path.join(d, "best.pt"), map_location="cpu"))
    model.to(device)
    test = ListOpsDataset(splits["test"], vocab)
    loader = make_loader(test, vocab, 256, shuffle=False)

    train_loops = cfg["model_cfg"]["loops"]
    rows, curves = ["loops,overall," + ",".join(f"depth{x}" for x in sorted({it['depth'] for it in test.items}))], {}
    for L in range(1, args.max_loops + 1):
        by_d, overall, counts = acc_by_depth(model, loader, test.items, device, L)
        curves[L] = by_d
        rows.append(f"{L},{overall:.4f}," + ",".join(f"{by_d[x]:.4f}" for x in sorted(by_d)))
        print(f"loops {L:2d}: overall {overall:.4f} | " + " ".join(f"d{x}:{v:.2f}" for x, v in by_d.items()))
    open(os.path.join(args.out, f"loops_{args.run}.csv"), "w").write("\n".join(rows) + "\n")

    fig, ax = plt.subplots(figsize=(6.5, 4))
    depths = sorted(next(iter(curves.values())))
    Ls = sorted(curves)
    for dep in depths:
        ax.plot(Ls, [curves[L][dep] for L in Ls], "o-", ms=3, label=f"depth {dep} (n={counts[dep]})")
    ax.axvline(train_loops, color="gray", ls="--")
    ax.text(train_loops + .1, 0.03, "loops used in training", color="gray", fontsize=8)
    ax.set(xlabel="loops at test time", ylabel="test accuracy", ylim=(0, 1),
           title=f"Thinking longer at test time ({args.run})")
    ax.grid(alpha=.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, f"loops_{args.run}.png"), dpi=200)


if __name__ == "__main__":
    main()
