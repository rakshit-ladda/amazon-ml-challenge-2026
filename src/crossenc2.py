"""Faster cross-encoder training/scoring (same model and objective as crossenc.py).

Speed-ups over crossenc.py:
  * length bucketing: pairs are grouped with others of similar length before
    padding, so most batches pad to ~40-60 tokens instead of 128;
  * tokenisation happens in DataLoader workers on pre-sorted chunks;
  * `score` can be limited to the uncertain band of a previous model
    (`--band lo hi` on a p column), since pairs already at p~0 or p~1 do not
    change their decision.

Usage:
  python crossenc2.py train <n_queries>                       (env CE_BASE/CE_INIT/CE_OUT/CE_EXTRA/CE_LR/CE_BS)
  python crossenc2.py score <pairs.parquet> <out.parquet> <shard> <nshards> [split]
"""
import os
import sys
import time

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from crossenc import train_pairs, with_text

W = "work"
BASE = os.environ.get("CE_BASE", "intfloat/multilingual-e5-small")
OUT = os.environ.get("CE_OUT", f"{W}/crossenc2")
MAXLEN = 128


class Bucketed(Dataset):
    """Yields whole batches: shuffled (train) or in order (score), then sorted by
    length inside windows of `window` batches so each batch pads to a similar length."""

    def __init__(self, a, b, y, bs, shuffle, window=50, seed=0):
        n = len(a)
        order = np.random.default_rng(seed).permutation(n) if shuffle else np.arange(n)
        lens = np.fromiter((len(x) + len(z) for x, z in zip(a, b)), dtype=np.int32, count=n)
        batches = []
        for s in range(0, n, bs * window):
            chunk = order[s:s + bs * window]
            chunk = chunk[np.argsort(lens[chunk], kind="stable")]
            batches += [chunk[i:i + bs] for i in range(0, len(chunk), bs)]
        if shuffle:
            np.random.default_rng(seed + 1).shuffle(batches)
        self.batches, self.a, self.b, self.y = batches, a, b, y

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, i):
        idx = self.batches[i]
        return idx, [self.a[j] for j in idx], [self.b[j] for j in idx], [self.y[j] for j in idx]


def make_collate(tok):
    def f(item):
        idx, a, b, y = item[0]
        enc = tok(a, b, truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt")
        return idx, enc, torch.tensor(y, dtype=torch.float32)
    return f


def train(n_queries):
    p = with_text(train_pairs(n_queries), "train").select("q_txt", "s_txt", "y")
    extra = os.environ.get("CE_EXTRA")
    if extra:
        x = pl.read_parquet(extra)
        n_extra = int(os.environ.get("CE_EXTRA_N", 0))  # optional subsample of pseudo pairs
        if n_extra and n_extra < x.height:
            x = x.sample(n_extra, seed=int(os.environ.get("CE_SEED", 0)))
        x = with_text(x, "test").select("q_txt", "s_txt", "y")
        p = pl.concat([p, x])
        print("added pseudo pairs", x.height, flush=True)
    print("train pairs", p.height, "pos rate", round(p["y"].mean(), 4), flush=True)
    init = os.environ.get("CE_INIT", BASE)
    bs = int(os.environ.get("CE_BS", 256))
    lr = float(os.environ.get("CE_LR", 4e-5))
    tok = AutoTokenizer.from_pretrained(init)
    model = AutoModelForSequenceClassification.from_pretrained(init, num_labels=1).cuda()
    ds = Bucketed(p["q_txt"].to_list(), p["s_txt"].to_list(), p["y"].to_list(), bs, shuffle=True,
                  seed=int(os.environ.get("CE_SEED", 0)))
    dl = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=make_collate(tok), num_workers=6, prefetch_factor=4)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=len(dl), pct_start=0.05)
    lossf = torch.nn.BCEWithLogitsLoss()
    model.train()
    t0, run, seen = time.time(), 0.0, 0
    for i, (_, enc, y) in enumerate(dl):
        enc = {k: v.cuda(non_blocking=True) for k, v in enc.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logit = model(**enc).logits.squeeze(-1)
        loss = lossf(logit.float(), y.cuda())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        run = 0.98 * run + 0.02 * loss.item()
        seen += len(y)
        if i % 1000 == 0:
            print(f"step {i}/{len(dl)} loss {run:.4f} {seen / (time.time() - t0):.0f} pairs/s", flush=True)
    print(f"done {len(dl)} steps {seen / (time.time() - t0):.0f} pairs/s", flush=True)
    model.save_pretrained(OUT)
    tok.save_pretrained(OUT)


@torch.no_grad()
def score(pairs_path, out, shard, nshards, split=None):
    """Score one shard of a (qid, s1id) parquet; writes (qid, s1id, logit) parquet."""
    p = pl.read_parquet(pairs_path).select("qid", "s1id")
    n = p.height
    lo, hi = n * shard // nshards, n * (shard + 1) // nshards
    split = split or ("test" if "test" in pairs_path else "train")
    p = with_text(p.slice(lo, hi - lo), split)
    tok = AutoTokenizer.from_pretrained(OUT)
    model = AutoModelForSequenceClassification.from_pretrained(OUT).cuda().to(torch.bfloat16).eval()
    ds = Bucketed(p["q_txt"].to_list(), p["s_txt"].to_list(), [0.0] * p.height, 1024, shuffle=False)
    dl = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=make_collate(tok), num_workers=6, prefetch_factor=4)
    res = np.empty(p.height, np.float32)
    t0 = time.time()
    for idx, enc, _ in dl:
        res[idx] = model(**{k: v.cuda(non_blocking=True) for k, v in enc.items()}).logits.squeeze(-1).float().cpu().numpy()
    print(f"scored {p.height} pairs {p.height / (time.time() - t0):.0f} pairs/s", flush=True)
    p.select("qid", "s1id").with_columns(logit=pl.Series(res)).write_parquet(out)


if __name__ == "__main__":
    if sys.argv[1] == "train":
        train(int(sys.argv[2]) if len(sys.argv) > 2 else 1_500_000)
    else:
        score(sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), sys.argv[6] if len(sys.argv) > 6 else None)
