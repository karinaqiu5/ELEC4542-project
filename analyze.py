"""Make every figure and table for the report from the saved runs.

  python analyze.py --runs runs --out figures

Reads runs/*/{results,history}.json and test_predictions.csv.
Baseline runs are named base_<model>; depth runs depth<D>_<model>;
positional-encoding runs pos_<kind>_<model>.
"""
import argparse
import csv
import glob
import json
import os
from collections import Counter, defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import load_splits, tokenize, expr_length, expr_depth

NAMES = {"rnn": "RNN", "lstm": "LSTM", "transformer": "Transformer", "linear": "Linear Attn",
         "looped": "Looped TF (ours)"}
ORDER = ["rnn", "lstm", "transformer", "linear", "looped"]


def load_runs(root):
    runs = {}
    for d in sorted(glob.glob(os.path.join(root, "*"))):
        rp = os.path.join(d, "results.json")
        if not os.path.exists(rp):
            continue
        r = json.load(open(rp))
        r["history"] = json.load(open(os.path.join(d, "history.json")))
        pp = os.path.join(d, "test_predictions.csv")
        r["preds"] = list(csv.DictReader(open(pp))) if os.path.exists(pp) else []
        for p in r["preds"]:
            for k in ("length", "depth", "target", "pred"):
                p[k] = int(p[k])
        runs[os.path.basename(d)] = r
    return runs


def acc(rows):
    return sum(r["pred"] == r["target"] for r in rows) / len(rows) if rows else float("nan")


def curves(base, out):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for m in ORDER:
        if m not in base:
            continue
        h = base[m]["history"]
        step = [x["step"] for x in h]
        t = [x["train_time_s"] / 60 for x in h]
        ax[0].plot(step, [x["train_loss"] for x in h], label=NAMES[m])
        ax[1].plot(step, [x["val_acc"] for x in h], label=NAMES[m])
        ax[2].plot(t, [x["val_acc"] for x in h], label=NAMES[m])
    ax[0].set(xlabel="training step", ylabel="training loss", title="Training loss")
    ax[1].set(xlabel="training step", ylabel="validation accuracy", title="Val accuracy vs steps")
    ax[2].set(xlabel="training time (min, eval excluded)", ylabel="validation accuracy",
              title="Val accuracy vs time")
    for a in ax:
        a.grid(alpha=.3)
        a.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "curves.png"), dpi=200)


def main_table(base, majority, out):
    lines = ["model,params,best_val_acc,test_acc,best_epoch,train_time_min,steady_step_ms"]
    for m in ORDER:
        if m in base:
            r = base[m]
            lines.append(f"{NAMES[m]},{r['params']},{r['best_val_acc']:.4f},{r['test_acc']:.4f},"
                         f"{r['best_epoch']:.1f},{r['train_time_s'] / 60:.1f},{r['steady_step_ms']:.2f}")
    lines.append(f"Majority class,0,,{majority:.4f},,,")
    open(os.path.join(out, "main_table.csv"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def length_depth(base, train_lengths, out):
    edges = np.quantile(train_lengths, [0, .25, .5, .75, 1.0]).astype(int)
    edges[-1] += 1
    bins = [(edges[i], edges[i + 1]) for i in range(4)]
    any_preds = next(iter(base.values()))["preds"]
    depths = sorted({p["depth"] for p in any_preds})

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    report = ["# accuracy by length bin (test)"]
    xl = [f"{a}-{b - 1}" for a, b in bins]
    counts = [sum(a <= p["length"] < b for p in any_preds) for a, b in bins]
    report.append("bin," + ",".join(f"{x} (n={c})" for x, c in zip(xl, counts)))
    width = 0.16
    for i, m in enumerate(o for o in ORDER if o in base):
        P = base[m]["preds"]
        vals = [acc([p for p in P if a <= p["length"] < b]) for a, b in bins]
        ax[0].bar(np.arange(4) + i * width, vals, width, label=NAMES[m])
        report.append(NAMES[m] + "," + ",".join(f"{v:.3f}" for v in vals))
    ax[0].set_xticks(np.arange(4) + 2 * width)
    ax[0].set_xticklabels([f"{x}\n(n={c})" for x, c in zip(xl, counts)])
    ax[0].set(xlabel="length (tokens)", ylabel="test accuracy", title="Accuracy by length")

    report.append("# accuracy by depth (test)")
    dcounts = [sum(p["depth"] == d for p in any_preds) for d in depths]
    report.append("depth," + ",".join(f"{d} (n={c})" for d, c in zip(depths, dcounts)))
    for m in (o for o in ORDER if o in base):
        P = base[m]["preds"]
        vals = [acc([p for p in P if p["depth"] == d]) for d in depths]
        ax[1].plot(depths, vals, "o-", label=NAMES[m])
        report.append(NAMES[m] + "," + ",".join(f"{v:.3f}" for v in vals))
    ax[1].set_xticks(depths)
    ax[1].set_xticklabels([f"{d}\n(n={c})" for d, c in zip(depths, dcounts)])
    ax[1].set(xlabel="depth", ylabel="test accuracy", title="Accuracy by depth")
    for a in ax:
        a.grid(alpha=.3)
        a.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "length_depth.png"), dpi=200)

    # depth inside one length band, so depth is not just standing in for length
    lo, hi = bins[2]
    report.append(f"# accuracy by depth inside the length band {lo}-{hi - 1}")
    band = [p for p in any_preds if lo <= p["length"] < hi]
    bd = sorted({p["depth"] for p in band})
    report.append("depth," + ",".join(f"{d} (n={sum(p['depth'] == d for p in band)})" for d in bd))
    for m in (o for o in ORDER if o in base):
        P = [p for p in base[m]["preds"] if lo <= p["length"] < hi]
        report.append(NAMES[m] + "," + ",".join(f"{acc([p for p in P if p['depth'] == d]):.3f}" for d in bd))
    open(os.path.join(out, "length_depth.txt"), "w").write("\n".join(report) + "\n")
    print("\n".join(report))


def errors(base, out):
    lines = []
    for m in (o for o in ORDER if o in base):
        P = base[m]["preds"]
        wrong = [p for p in P if p["pred"] != p["target"]]
        lines.append(f"## {NAMES[m]}: {len(wrong)} wrong of {len(P)}")
        # which operator sits at the top of the wrong expressions
        top = Counter(tokenize(p["source"])[0] for p in wrong)
        top_all = Counter(tokenize(p["source"])[0] for p in P)
        lines.append("error rate by outer operator: " + ", ".join(
            f"{k} {top[k] / top_all[k]:.3f}" for k in sorted(top_all)))
        conf = Counter((p["target"], p["pred"]) for p in wrong).most_common(5)
        lines.append("most common (target -> pred): " + ", ".join(f"{t}->{q} x{c}" for (t, q), c in conf))
        for p in sorted(wrong, key=lambda p: p["length"])[:5]:
            lines.append(f"  target {p['target']} pred {p['pred']} depth {p['depth']}: {p['source']}")
    open(os.path.join(out, "errors.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def depth_experiment(runs, out):
    exp = {k: v for k, v in runs.items() if k.startswith("depth")}
    if not exp:
        return
    D = int(next(iter(exp)).split("_")[0][5:])
    lines = [f"# trained on depth <= {D}; tested on full test set",
             "model,in_dist_acc(depth<=D),ood_acc(depth>D),drop,full_baseline_ood"]
    fig, ax = plt.subplots(figsize=(6, 4))
    for m in ORDER:
        key = f"depth{D}_{m}"
        if key not in exp:
            continue
        P = exp[key]["preds"]
        ind = acc([p for p in P if p["depth"] <= D])
        ood = acc([p for p in P if p["depth"] > D])
        b = runs.get(f"base_{m}")
        base_ood = acc([p for p in b["preds"] if p["depth"] > D]) if b else float("nan")
        lines.append(f"{NAMES[m]},{ind:.4f},{ood:.4f},{ind - ood:.4f},{base_ood:.4f}")
        depths = sorted({p["depth"] for p in P})
        ax.plot(depths, [acc([p for p in P if p["depth"] == d]) for d in depths], "o-", label=NAMES[m])
    ax.axvline(D + .5, color="gray", ls="--")
    ax.text(D + .6, 0.05, "unseen depths", color="gray")
    ax.set(xlabel="test depth", ylabel="accuracy", title=f"Trained only on depth <= {D}")
    ax.grid(alpha=.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "depth_experiment.png"), dpi=200)
    open(os.path.join(out, "depth_experiment.csv"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def pos_ablation(runs, out):
    rows = [(k, v) for k, v in runs.items() if k.startswith("pos_")]
    if not rows:
        return
    lines = ["run,params,best_val_acc,test_acc"]
    for k, v in sorted(rows):
        lines.append(f"{k},{v['params']},{v['best_val_acc']:.4f},{v.get('test_acc', float('nan')):.4f}")
    open(os.path.join(out, "pos_ablation.csv"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    splits = load_splits()
    labels = Counter(t for _, t in splits["train"])
    maj = labels.most_common(1)[0][0]
    majority_test = sum(t == maj for _, t in splits["test"]) / len(splits["test"])
    train_lengths = [expr_length(tokenize(s)) for s, _ in splits["train"]]

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
    ax[0].bar(range(10), [labels[i] / len(splits["train"]) for i in range(10)])
    ax[0].set(xlabel="label", ylabel="fraction", title=f"Training labels (majority = {maj})")
    ax[0].set_xticks(range(10))
    dc = Counter(expr_depth(tokenize(s)) for s, _ in splits["train"])
    ax[1].bar(list(dc), list(dc.values()))
    ax[1].set(xlabel="depth", ylabel="count", title="Training depth distribution")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "data_stats.png"), dpi=200)
    print(f"majority class {maj}: test accuracy {majority_test:.4f}")

    runs = load_runs(args.runs)
    base = {k[5:]: v for k, v in runs.items() if k.startswith("base_")}
    if base:
        curves(base, args.out)
        main_table(base, majority_test, args.out)
        length_depth(base, train_lengths, args.out)
        errors(base, args.out)
    depth_experiment(runs, args.out)
    pos_ablation(runs, args.out)


if __name__ == "__main__":
    main()
