"""ListOps-64 data pipeline: loading, tokenisation, length/depth, batching.

Run `python data.py` once to download the dataset and print basic statistics
(label distribution, majority-class baseline, length and depth distributions,
and a check that every Target matches a reference evaluator).
"""
import os
import statistics
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader

DATA_DIR = "data/listops-64"
PAD, UNK, EOS = "<pad>", "<unk>", "<eos>"
MAX_SOURCE_TOKENS = 64          # dataset limit; +1 position is reserved for EOS
NUM_CLASSES = 10


# ---------------------------------------------------------------- loading
def load_splits(data_dir=DATA_DIR):
    """Return a dict with 'train', 'val', 'test' -> list of (source, target)."""
    if os.environ.get("LISTOPS_FAKE"):          # offline smoke test only, never for results
        return fake_splits()
    from datasets import load_dataset, load_from_disk
    if os.path.isdir(data_dir):
        ds = load_from_disk(data_dir)
    else:
        ds = load_dataset("fengyang0317/listops-64")
        ds.save_to_disk(data_dir)

    def pick(*names):
        for n in names:
            if n in ds:
                return ds[n]
        raise KeyError(f"none of {names} in splits {list(ds.keys())}")

    out = {}
    for key, split in (("train", pick("train")),
                       ("val", pick("validation", "val", "dev")),
                       ("test", pick("test"))):
        cols = {c.lower(): c for c in split.column_names}
        src_col, tgt_col = cols["source"], cols["target"]
        out[key] = [(s, int(t)) for s, t in zip(split[src_col], split[tgt_col])]
    return out


# ---------------------------------------------------------------- tokens
def tokenize(source):
    """Split on whitespace; keeps operator tokens like '[MAX' and ']'."""
    return source.split()


def expr_length(tokens):
    """Number of source tokens (before EOS / padding)."""
    return len(tokens)


def expr_depth(tokens):
    """Maximum number of simultaneously open lists."""
    depth = best = 0
    for t in tokens:
        if t.startswith("["):
            depth += 1
            best = max(best, depth)
        elif t == "]":
            depth -= 1
    return best


class Vocab:
    """Shared vocabulary built from the training split only."""

    def __init__(self, train_sources):
        counts = Counter(t for s in train_sources for t in tokenize(s))
        self.itos = [PAD, UNK] + sorted(counts) + [EOS]
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        self.pad_id, self.unk_id, self.eos_id = 0, 1, self.stoi[EOS]

    def __len__(self):
        return len(self.itos)

    def encode(self, source):
        ids = [self.stoi.get(t, self.unk_id) for t in tokenize(source)]
        return ids + [self.eos_id]


# ---------------------------------------------------------------- reference evaluator (labels check / analysis only, never a model input)
def evaluate_expression(source):
    """Evaluate a ListOps expression; used only to sanity-check labels."""
    stack = [[]]
    ops = []
    for t in tokenize(source):
        if t.startswith("["):
            ops.append(t[1:])
            stack.append([])
        elif t == "]":
            args, op = stack.pop(), ops.pop()
            if op == "MAX":
                val = max(args)
            elif op == "MIN":
                val = min(args)
            elif op == "MED":
                val = int(statistics.median(args))
            elif op == "SM":
                val = sum(args) % 10
            else:
                raise ValueError(f"unknown operator {op}")
            stack[-1].append(val)
        elif t.isdigit():
            stack[-1].append(int(t))
        # any other token (e.g. parentheses in some ListOps variants) is ignored
    return stack[0][0]


def subexpr_values(source):
    """For every closing bracket: (token position, value of the list it closes,
    nesting level of that list (1 = outermost), operator). Used ONLY for the
    probing analysis after training; never given to the models."""
    stack, ops, out = [[]], [], []
    for pos, t in enumerate(tokenize(source)):
        if t.startswith("["):
            ops.append(t[1:])
            stack.append([])
        elif t == "]":
            level = len(ops)
            args, op = stack.pop(), ops.pop()
            val = {"MAX": max, "MIN": min, "SM": lambda a: sum(a) % 10,
                   "MED": lambda a: int(statistics.median(a))}[op](args)
            stack[-1].append(val)
            out.append((pos, val, level, op))
        elif t.isdigit():
            stack[-1].append(int(t))
    return out


def fake_splits(n_train=3000, n_val=300, n_test=300, seed=0):
    """Random ListOps-style expressions (<= 64 tokens) for offline smoke tests."""
    import random as _r
    rng = _r.Random(seed)

    def gen(depth):
        if depth == 0 or rng.random() < 0.3:
            return [str(rng.randint(0, 9))]
        toks = ["[" + rng.choice(["MAX", "MIN", "MED", "SM"])]
        for _ in range(rng.randint(2, 4)):
            toks += gen(depth - 1)
        return toks + ["]"]

    def make(n):
        out = []
        while len(out) < n:
            toks = gen(rng.randint(1, 5))
            if toks[0].startswith("[") and len(toks) <= MAX_SOURCE_TOKENS:
                s = " ".join(toks)
                out.append((s, evaluate_expression(s)))
        return out
    return {"train": make(n_train), "val": make(n_val), "test": make(n_test)}


# ---------------------------------------------------------------- torch dataset
class ListOpsDataset(Dataset):
    def __init__(self, examples, vocab):
        self.items = []
        for src, tgt in examples:
            toks = tokenize(src)
            if len(toks) > MAX_SOURCE_TOKENS:
                raise ValueError(f"expression longer than {MAX_SOURCE_TOKENS} tokens")
            self.items.append({
                "ids": vocab.encode(src),
                "target": tgt,
                "length": expr_length(toks),
                "depth": expr_depth(toks),
                "source": src,
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def make_collate(pad_id):
    def collate(batch):
        lengths = torch.tensor([len(b["ids"]) for b in batch])          # includes EOS
        ids = torch.full((len(batch), int(lengths.max())), pad_id, dtype=torch.long)
        for i, b in enumerate(batch):
            ids[i, :len(b["ids"])] = torch.tensor(b["ids"])
        return {
            "ids": ids,
            "lengths": lengths,
            "targets": torch.tensor([b["target"] for b in batch]),
            "index": torch.tensor([b.get("index", -1) for b in batch]),
        }
    return collate


def make_loader(dataset, vocab, batch_size, shuffle, seed=0):
    for i, item in enumerate(dataset.items):
        item["index"] = i
    gen = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=make_collate(vocab.pad_id), generator=gen,
                      num_workers=0)


# ---------------------------------------------------------------- stats
def describe(splits):
    train = splits["train"]
    labels = Counter(t for _, t in train)
    majority, maj_count = labels.most_common(1)[0]
    print("Split sizes:", {k: len(v) for k, v in splits.items()})
    print("Training label distribution:",
          {k: f"{v / len(train):.3f}" for k, v in sorted(labels.items())})
    for name in ("val", "test"):
        acc = sum(t == majority for _, t in splits[name]) / len(splits[name])
        print(f"Majority-class baseline (always predict {majority}) on {name}: {acc:.4f}")
    for name in ("train", "val", "test"):
        toks = [tokenize(s) for s, _ in splits[name]]
        lens = [expr_length(t) for t in toks]
        deps = [expr_depth(t) for t in toks]
        print(f"{name}: length min/median/max = {min(lens)}/{int(statistics.median(lens))}/{max(lens)}; "
              f"depth counts = {dict(sorted(Counter(deps).items()))}")
    bad = sum(evaluate_expression(s) != t for s, t in train)
    print(f"Label check on train: {len(train) - bad}/{len(train)} targets match the reference evaluator")


if __name__ == "__main__":
    describe(load_splits())
