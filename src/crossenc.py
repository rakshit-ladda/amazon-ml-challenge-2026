"""Cross-encoder reranker.

A transformer that reads both records of a candidate pair jointly and outputs a
match logit. It targets the hard negatives in this data: near-copies of a real
business with one real change (house number +/-1, an added real word, a different
legal form) versus true copies that only carry noise (typos, filler words,
missing address).

Base: intfloat/multilingual-e5-small (MIT, 118M params) with a 1-logit head.
Training pairs: bi-encoder top-3 neighbours of queries whose S1 is in fold A
(hash % 2 == 0), so fold B stays clean for validating the downstream model.

Usage:
  python crossenc.py train [n_queries]
  python crossenc.py score <pairs.parquet> <out.npy> <shard> <nshards>
"""
import os
import sys
import time

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

W = "work"
BASE = os.environ.get("CE_BASE", "intfloat/multilingual-e5-small")  # override for the larger variant
OUT = os.environ.get("CE_OUT", f"{W}/crossenc")
MAXLEN = 128


def records(split):
    """entity_id -> 'name | address' text for S1 and S2/S3 of a split."""
    cols = ["entity_id", "business_name", "business_address", "country"]
    df = pl.concat([pl.read_parquet(f"{W}/{split}_s{i}.parquet").select(cols) for i in (1, 2, 3)])
    if os.environ.get("CANON_FR") == "1":  # departement -> region for France (map learned by canon_fr.py)
        import json
        from canon_fr import canon_address
        amap = json.load(open(f"{W}/fr_admin_map.json"))
        df = df.with_columns(business_address=pl.when(pl.col("country") == "France").then(
            pl.col("business_address").map_elements(lambda a: canon_address(a, amap), return_dtype=pl.Utf8))
            .otherwise(pl.col("business_address")))
    return df.select("entity_id", txt=pl.col("business_name") + " | " + pl.col("business_address"))


def with_text(pairs, split):
    rec = records(split)
    return (pairs.join(rec.rename({"txt": "q_txt"}), left_on="qid", right_on="entity_id", how="left")
            .join(rec.rename({"txt": "s_txt"}), left_on="s1id", right_on="entity_id", how="left"))


def collate(tok):
    def f(batch):
        a, b, y = zip(*batch)
        enc = tok(list(a), list(b), truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor(y, dtype=torch.float32)
        return enc
    return f


def train_pairs(n_queries):
    """Bi-encoder top-3 pairs for fold-A queries (true S1 in fold A, or distractors
    whose top-1 is in fold A); pairs with a fold-B S1 are dropped."""
    gt = (pl.read_csv("dataset/train/train_ground_truth.tsv", separator="\t", infer_schema=False).fill_null("")
          .with_columns(qid=pl.col("matched_entity_ids").str.split(",")).explode("qid")
          .filter(pl.col("qid") != "").select(pl.col("source1_entity_id").alias("s1id"), "qid"))
    be = pl.read_parquet(os.environ.get("BE_PATH", f"{W}/train_be.parquet")).filter(pl.col("rk_be") < 3)  # any bi-encoder's candidates
    if os.environ.get("TRAIN_SPLIT") == "notE":
        # all training S1 except the held-out reranker set E (~85% of the data)
        from features import eval_s1
        keep = lambda c: ~eval_s1(pl.col(c))
    else:
        keep = lambda c: (pl.col(c).hash(7) % 2) == 0  # fold A (bi-encoder training half)
    fold_a = keep("s1id")
    top1 = be.filter(pl.col("rk_be") == 0).select("qid", top1=pl.col("s1id"))
    q = (top1.join(gt.rename({"s1id": "true_s1"}), on="qid", how="left")
         .filter(pl.when(pl.col("true_s1").is_null()).then(keep("top1"))
                 .otherwise(keep("true_s1"))).select("qid"))
    q = q.sample(min(n_queries, q.height), seed=int(os.environ.get("CE_QSEED", 0)))  # different query samples per run
    p = (be.join(q, on="qid", how="semi").filter(fold_a)
         .join(gt.with_columns(y=pl.lit(1.0)), on=["qid", "s1id"], how="left")
         .with_columns(pl.col("y").fill_null(0.0)).select("qid", "s1id", "y"))
    return p.sample(fraction=1.0, shuffle=True, seed=1)


def train(n_queries):
    p = with_text(train_pairs(n_queries), "train").select("q_txt", "s_txt", "y")
    extra = os.environ.get("CE_EXTRA")  # optional test-split pseudo-labelled pairs (qid, s1id, y)
    if extra:
        x = with_text(pl.read_parquet(extra), "test").select("q_txt", "s_txt", "y")
        p = pl.concat([p, x]).sample(fraction=1.0, shuffle=True, seed=2)
        print("added pseudo pairs", x.height, flush=True)
    print("train pairs", p.height, "pos rate", p["y"].mean(), flush=True)
    init = os.environ.get("CE_INIT", BASE)  # start from a trained cross-encoder when fine-tuning
    tok = AutoTokenizer.from_pretrained(init)
    model = AutoModelForSequenceClassification.from_pretrained(init, num_labels=1).cuda()
    data = list(zip(p["q_txt"].to_list(), p["s_txt"].to_list(), p["y"].to_list()))
    bs = int(os.environ.get("CE_BS", 256))
    dl = DataLoader(data, batch_size=bs, shuffle=False, collate_fn=collate(tok), num_workers=6)
    lr = float(os.environ.get("CE_LR", 4e-5))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = len(dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    lossf = torch.nn.BCEWithLogitsLoss()
    model.train()
    t0, run = time.time(), 0.0
    for i, b in enumerate(dl):
        b = {k: v.cuda(non_blocking=True) for k, v in b.items()}
        y = b.pop("labels")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logit = model(**b).logits.squeeze(-1)
        loss = lossf(logit.float(), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        run = 0.98 * run + 0.02 * loss.item()
        if i % 1000 == 0:
            print(f"step {i}/{steps} loss {run:.4f} {(i + 1) * bs / (time.time() - t0):.0f} pairs/s", flush=True)
    print(f"done {steps} steps {steps * bs / (time.time() - t0):.0f} pairs/s", flush=True)
    model.save_pretrained(OUT)
    tok.save_pretrained(OUT)


@torch.no_grad()
def score(pairs_path, out, shard, nshards, split=None):
    """Score one shard of a (qid, s1id) parquet; writes float32 logits .npy."""
    p = pl.read_parquet(pairs_path).select("qid", "s1id")
    n = p.height
    lo, hi = n * shard // nshards, n * (shard + 1) // nshards
    split = split or ("test" if "test" in pairs_path else "train")
    p = with_text(p.slice(lo, hi - lo), split)
    tok = AutoTokenizer.from_pretrained(OUT)
    model = AutoModelForSequenceClassification.from_pretrained(OUT).cuda().to(torch.bfloat16).eval()
    data = list(zip(p["q_txt"].to_list(), p["s_txt"].to_list(), [0.0] * p.height))
    dl = DataLoader(data, batch_size=1024, shuffle=False, collate_fn=collate(tok), num_workers=6)
    res, t0 = [], time.time()
    for i, b in enumerate(dl):
        b.pop("labels")
        res.append(model(**{k: v.cuda(non_blocking=True) for k, v in b.items()}).logits.squeeze(-1).float().cpu().numpy())
        if i % 500 == 0:
            print(f"{i * 1024}/{p.height} {(i + 1) * 1024 / (time.time() - t0):.0f} pairs/s", flush=True)
    np.save(out, np.concatenate(res))
    print("scored", out, flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "train":
        train(int(sys.argv[2]) if len(sys.argv) > 2 else 1_500_000)
    else:
        score(sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]))
