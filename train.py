"""Train one model on ListOps-64, select the best validation checkpoint, test it once.

Examples
  python train.py --model lstm --run_name base_lstm
  python train.py --model transformer --pos none --run_name ablate_tf_nopos
  python train.py --model lstm --train_max_depth 4 --run_name depth4_lstm   # controlled experiment
  python train.py --model looped --run_name base_looped                     # our extension

Outputs (runs/<run_name>/): config.json, history.json, results.json,
test_predictions.csv, best.pt
"""
import argparse
import csv
import json
import math
import os
import random
import statistics
import time

import numpy as np
import torch
from torch import nn

from data import Vocab, ListOpsDataset, load_splits, make_loader, expr_depth, tokenize
from models import build_model, count_params


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def evaluate(model, loader, device, collect=False):
    """Exact-match accuracy and mean loss over a whole split (sample-weighted)."""
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total, correct, loss_sum, preds = 0, 0, 0.0, []
    for b in loader:
        ids, lengths, y = b["ids"].to(device), b["lengths"], b["targets"].to(device)
        logits = model(ids, lengths)
        loss_sum += loss_fn(logits, y).item()
        p = logits.argmax(-1)
        correct += (p == y).sum().item()
        total += y.numel()
        if collect:
            preds.extend(zip(b["index"].tolist(), p.tolist()))
    model.train()
    return correct / total, loss_sum / total, preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["rnn", "lstm", "transformer", "linear", "looped"])
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_steps", type=int, default=500)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--eval_every", type=int, default=250, help="steps between validation checks")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pos", default="learned", choices=["learned", "sinusoidal", "none"])
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--d_model", type=int, default=None)
    ap.add_argument("--d_hidden", type=int, default=None)
    ap.add_argument("--loops", type=int, default=None, help="looped model: loops during training")
    ap.add_argument("--train_max_depth", type=int, default=None,
                    help="controlled experiment: train/validate only on depth <= this")
    ap.add_argument("--max_steps", type=int, default=None, help="optional cap (for quick tuning runs)")
    ap.add_argument("--no_test", action="store_true", help="tuning runs: skip the test split")
    ap.add_argument("--out_dir", default="runs")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = os.path.join(args.out_dir, args.run_name)
    os.makedirs(out, exist_ok=True)

    # ------------------------------------------------ data
    splits = load_splits()
    vocab = Vocab([s for s, _ in splits["train"]])
    train_ex, val_ex = splits["train"], splits["val"]
    if args.train_max_depth is not None:
        keep = lambda ex: [e for e in ex if expr_depth(tokenize(e[0])) <= args.train_max_depth]
        train_ex, val_ex = keep(train_ex), keep(val_ex)   # checkpoint chosen in-distribution
    train_ds = ListOpsDataset(train_ex, vocab)
    val_ds = ListOpsDataset(val_ex, vocab)
    test_ds = ListOpsDataset(splits["test"], vocab)
    train_loader = make_loader(train_ds, vocab, args.batch_size, shuffle=True, seed=args.seed)
    val_loader = make_loader(val_ds, vocab, 256, shuffle=False)
    test_loader = make_loader(test_ds, vocab, 256, shuffle=False)
    print(f"train {len(train_ds)} | val {len(val_ds)} | test {len(test_ds)} | vocab {len(vocab)} | device {device}")

    # ------------------------------------------------ model / optimiser
    set_seed(args.seed)
    overrides = {"layers": args.layers, "d_model": args.d_model, "d_hidden": args.d_hidden,
                 "loops": args.loops}
    model, cfg = build_model(args.model, len(vocab), vocab.pad_id, args.dropout, args.pos, **overrides)
    model.to(device)
    n_params = count_params(model)
    print(f"{args.model}: {n_params:,} parameters, config {cfg}")

    steps_per_epoch = len(train_loader)
    total_steps = args.max_steps or args.epochs * steps_per_epoch
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def lr_at(step):  # linear warm-up, then cosine decay to 0
        if step < args.warmup_steps:
            return (step + 1) / args.warmup_steps
        progress = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    loss_fn = nn.CrossEntropyLoss()

    config = {**vars(args), "model_cfg": cfg, "params": n_params, "vocab_size": len(vocab),
              "train_size": len(train_ds), "val_size": len(val_ds), "device": str(device),
              "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
              "torch": torch.__version__, "precision": "fp32"}
    json.dump(config, open(os.path.join(out, "config.json"), "w"), indent=2)

    # ------------------------------------------------ training
    history, step_times = [], []
    best = {"val_acc": -1.0, "step": None}
    train_time = eval_time = 0.0
    run_loss, run_n, step, done = 0.0, 0, 0, False
    model.train()
    while not done:
        for b in train_loader:
            sync(device)
            t0 = time.perf_counter()
            ids, lengths, y = b["ids"].to(device), b["lengths"], b["targets"].to(device)
            loss = loss_fn(model(ids, lengths), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            sched.step()
            sync(device)
            dt = time.perf_counter() - t0
            train_time += dt
            step_times.append(dt)
            run_loss += loss.item() * y.numel()
            run_n += y.numel()
            step += 1

            if step % args.eval_every == 0 or step == total_steps:
                t1 = time.perf_counter()
                val_acc, val_loss, _ = evaluate(model, val_loader, device)
                sync(device)
                eval_time += time.perf_counter() - t1
                history.append({"step": step, "epoch": step / steps_per_epoch,
                                "train_loss": run_loss / run_n, "val_loss": val_loss,
                                "val_acc": val_acc, "train_time_s": train_time})
                run_loss, run_n = 0.0, 0
                if val_acc > best["val_acc"]:          # strict '>' keeps the earliest on ties
                    best = {"val_acc": val_acc, "step": step}
                    torch.save(model.state_dict(), os.path.join(out, "best.pt"))
                print(f"step {step:6d} ep {step / steps_per_epoch:5.2f} | "
                      f"train loss {history[-1]['train_loss']:.4f} | val acc {val_acc:.4f} | "
                      f"train time {train_time:7.1f}s")
            if step >= total_steps:
                done = True
                break

    json.dump(history, open(os.path.join(out, "history.json"), "w"), indent=1)

    # ------------------------------------------------ test the selected checkpoint once
    results = {"model": args.model, "run_name": args.run_name, "params": n_params,
               "best_val_acc": best["val_acc"], "best_step": best["step"],
               "best_epoch": best["step"] / steps_per_epoch,
               "train_time_s": train_time, "val_time_s": eval_time,
               "steady_step_ms": 1000 * statistics.median(step_times[min(100, len(step_times) // 2):]),
               "total_steps": step}
    if not args.no_test:
        model.load_state_dict(torch.load(os.path.join(out, "best.pt"), map_location=device))
        test_acc, test_loss, preds = evaluate(model, test_loader, device, collect=True)
        results.update({"test_acc": test_acc, "test_loss": test_loss})
        with open(os.path.join(out, "test_predictions.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["index", "length", "depth", "target", "pred", "source"])
            for i, p in sorted(preds):
                it = test_ds.items[i]
                w.writerow([i, it["length"], it["depth"], it["target"], p, it["source"]])
        print(f"TEST accuracy {test_acc:.4f} (checkpoint from step {best['step']})")
    json.dump(results, open(os.path.join(out, "results.json"), "w"), indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
