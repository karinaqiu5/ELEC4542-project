"""Mind-reading probes: does a model secretly compute the sub-answers?

The models are trained ONLY on the final answer. For every closing bracket ']'
we know (from the reference evaluator) the value of the list it closes, e.g.
'[SM 7 6 ]' -> 3. We freeze each trained model, take its hidden state at that
']' position and train a small linear classifier (logistic regression) to read
the sub-answer out of it. High probe accuracy = the value is linearly readable
inside the model = the model has computed it.

Control: the same probe on an UNTRAINED copy of each model (random weights).
If the untrained model also allows high accuracy, the probe is just memorising
the input, not reading the model's computation.

  python probe.py --runs runs --out figures            # probes every base_* run
Writes figures/probe.csv, figures/probe_by_level.png
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

from data import Vocab, load_splits, subexpr_values, tokenize
from models import build_model

NAMES = {"rnn": "RNN", "lstm": "LSTM", "transformer": "Transformer", "linear": "Linear Attn",
         "looped": "Looped TF (ours)"}
ORDER = ["rnn", "lstm", "transformer", "linear", "looped"]


@torch.no_grad()
def collect(model, examples, vocab, device, batch=256):
    """Hidden state at every ']' + its sub-answer, nesting level, and whether it is the final one."""
    model.eval()
    X, y, level, final = [], [], [], []
    for i in range(0, len(examples), batch):
        chunk = examples[i:i + batch]
        enc = [vocab.encode(s) for s, _ in chunk]
        T = max(map(len, enc))
        ids = torch.full((len(enc), T), vocab.pad_id, dtype=torch.long)
        for j, e in enumerate(enc):
            ids[j, :len(e)] = torch.tensor(e)
        lengths = torch.tensor([len(e) for e in enc])
        H = model.features(ids.to(device), lengths).float().cpu().numpy()
        for j, (s, _) in enumerate(chunk):
            subs = subexpr_values(s)
            for k, (pos, val, lev, _) in enumerate(subs):
                X.append(H[j, pos])
                y.append(val)
                level.append(lev)
                final.append(k == len(subs) - 1)
    return np.array(X), np.array(y), np.array(level), np.array(final)


def probe_model(name, state_path, cfg, splits, vocab, device, n_train, trained=True, seed=0):
    torch.manual_seed(seed)
    model, _ = build_model(name, len(vocab), vocab.pad_id, cfg.get("dropout", 0.1),
                           cfg.get("pos", "learned"), **{k: cfg["model_cfg"].get(k) for k in
                                                         ("layers", "d_model", "d_hidden", "loops")})
    if trained:
        model.load_state_dict(torch.load(state_path, map_location="cpu"))
    model.to(device)
    Xtr, ytr, _, _ = collect(model, splits["train"][:n_train], vocab, device)
    Xte, yte, lte, fte = collect(model, splits["test"], vocab, device)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit((Xtr - mu) / sd, ytr)
    pred = clf.predict((Xte - mu) / sd)
    ok = pred == yte
    res = {"all": ok.mean(), "inner": ok[~fte].mean(), "final": ok[fte].mean(), "n": len(yte)}
    for lev in sorted(set(lte)):
        res[f"level{lev}"] = ok[lte == lev].mean()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--n_train", type=int, default=8000, help="training expressions used to fit the probe")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits = load_splits()
    vocab = Vocab([s for s, _ in splits["train"]])

    # majority baseline for sub-answers
    vals = [v for s, _ in splits["test"] for _, v, _, _ in subexpr_values(s)]
    maj = np.bincount(vals).max() / len(vals)

    rows = []
    for m in ORDER:
        d = os.path.join(args.runs, f"base_{m}")
        if not os.path.exists(os.path.join(d, "best.pt")):
            continue
        cfg = json.load(open(os.path.join(d, "config.json")))
        for trained in (True, False):
            r = probe_model(m, os.path.join(d, "best.pt"), cfg, splits, vocab, device,
                            args.n_train, trained)
            r.update(model=NAMES[m], weights="trained" if trained else "untrained (control)")
            rows.append(r)
            print(f"{NAMES[m]:18s} {r['weights']:22s} all {r['all']:.3f} | inner {r['inner']:.3f} "
                  f"| final {r['final']:.3f}")

    levels = sorted({k for r in rows for k in r if k.startswith("level")}, key=lambda k: int(k[5:]))
    cols = ["model", "weights", "all", "inner", "final"] + levels + ["n"]
    with open(os.path.join(args.out, "probe.csv"), "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(c, "")) if c in ("model", "weights", "n")
                             else f"{r.get(c, float('nan')):.4f}" for c in cols) + "\n")
        f.write(f"majority-class sub-answer baseline,,{maj:.4f}\n")

    fig, ax = plt.subplots(figsize=(6.5, 4))
    for r in rows:
        if r["weights"] != "trained":
            continue
        xs = [int(k[5:]) for k in levels if k in r]
        ax.plot(xs, [r[f"level{x}"] for x in xs], "o-", label=r["model"])
    ctrl = [r for r in rows if r["weights"] != "trained"]
    if ctrl:
        best_ctrl = max(np.nanmean([r.get(k, np.nan) for k in levels]) for r in ctrl)
        ax.axhline(best_ctrl, color="gray", ls=":", label="best untrained control")
    ax.axhline(maj, color="black", ls="--", lw=.8, label="majority baseline")
    ax.set(xlabel="nesting level of the sub-expression (1 = outermost)",
           ylabel="probe accuracy (sub-answer)", title="Can we read sub-answers from the hidden state?")
    ax.grid(alpha=.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "probe_by_level.png"), dpi=200)
    print("saved", os.path.join(args.out, "probe.csv"))


if __name__ == "__main__":
    main()
