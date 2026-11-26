"""Where does the Transformer look? Attention maps + a 'bracket focus' score.

For every closing bracket ']' (the moment a sub-list can be evaluated) we measure
how much of its attention lands on tokens INSIDE its own list (from the matching
'[OP' up to the ']'). A model that evaluates the nested structure should focus
inside the bracket; a model that guesses from surface patterns should not.
Random baseline = attention spread evenly over all earlier tokens.

  python attention.py --runs runs --out figures
Writes figures/attention_example_<model>.png and figures/bracket_focus.csv
"""
import argparse
import json
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import Vocab, load_splits, tokenize
from models import build_model

MODELS = {"transformer": "Transformer", "looped": "Looped TF (ours)"}


def spans(tokens):
    """(start, end) token positions of every bracketed list, end = position of ']'."""
    stack, out = [], []
    for i, t in enumerate(tokens):
        if t.startswith("["):
            stack.append(i)
        elif t == "]":
            out.append((stack.pop(), i))
    return out


def load(name, runs, vocab, device):
    d = os.path.join(runs, f"base_{name}")
    cfg = json.load(open(os.path.join(d, "config.json")))
    model, _ = build_model(name, len(vocab), vocab.pad_id, 0.0, cfg.get("pos", "learned"),
                           **{k: cfg["model_cfg"].get(k) for k in ("layers", "d_model", "loops")})
    model.load_state_dict(torch.load(os.path.join(d, "best.pt"), map_location="cpu"))
    return model.to(device).eval()


@torch.no_grad()
def maps_for(model, vocab, source, device):
    ids = torch.tensor([vocab.encode(source)], device=device)
    if hasattr(model, "_run"):
        _, maps = model.run_with_attention(ids)
    else:
        _, maps = model.run_with_attention(ids, model.encoder.layers)
    return [m[0].mean(0).cpu().numpy() for m in maps]   # per layer/loop: (T, T), heads averaged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--n", type=int, default=500, help="test expressions for the focus score")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits = load_splits()
    vocab = Vocab([s for s, _ in splits["train"]])
    # a medium-length, depth >= 2 example for the pictures
    example = next(s for s, _ in splits["test"]
                   if 14 <= len(tokenize(s)) <= 22 and sum(t.startswith("[") for t in tokenize(s)) >= 3)

    lines = ["model,layer_or_loop,bracket_focus,uniform_baseline"]
    for name, label in MODELS.items():
        if not os.path.exists(os.path.join(args.runs, f"base_{name}", "best.pt")):
            continue
        model = load(name, args.runs, vocab, device)
        # ---- picture
        toks = tokenize(example) + ["<eos>"]
        maps = maps_for(model, vocab, example, device)
        fig, axes = plt.subplots(1, len(maps), figsize=(4.2 * len(maps), 4.4))
        axes = np.atleast_1d(axes)
        for i, (ax, A) in enumerate(zip(axes, maps)):
            ax.imshow(A, cmap="viridis")
            ax.set_xticks(range(len(toks)), toks, rotation=90, fontsize=6)
            ax.set_yticks(range(len(toks)), toks, fontsize=6)
            ax.set_title(f"{label}: {'loop' if name == 'looped' else 'layer'} {i + 1}", fontsize=9)
        fig.supxlabel("attended token (key)", fontsize=8)
        fig.supylabel("current token (query)", fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, f"attention_example_{name}.png"), dpi=200)
        # ---- bracket focus score
        sums, base, cnt = None, 0.0, 0
        for s, _ in splits["test"][:args.n]:
            toks_s = tokenize(s)
            maps = maps_for(model, vocab, s, device)
            if sums is None:
                sums = np.zeros(len(maps))
            for a, e in spans(toks_s):
                for li, A in enumerate(maps):
                    sums[li] += A[e, a:e + 1].sum()
                base += (e - a + 1) / (e + 1)     # uniform causal attention
                cnt += 1
        for li in range(len(sums)):
            lines.append(f"{label},{li + 1},{sums[li] / cnt:.4f},{base / cnt:.4f}")
        print(f"{label}: bracket focus per layer {np.round(sums / cnt, 3)} vs uniform {base / cnt:.3f}")
    open(os.path.join(args.out, "bracket_focus.csv"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
