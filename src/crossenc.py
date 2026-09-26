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
import sys
import time

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

W = "work"
BASE = "Qwen/Qwen2.5-1.5B"
OUT = f"{W}/crossenc"
MAXLEN = 128


def records(split):
    """entity_id -> 'name | address' text for S1 and S2/S3 of a split."""
    cols = ["entity_id", "business_name", "business_address"]
    df = pl.concat([pl.read_parquet(f"{W}/{split}_s{i}.parquet").select(cols) for i in (1, 2, 3)])
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
    gt = (pl.read_csv("../dataset/train/train_ground_truth.tsv", separator="\t", infer_schema=False).fill_null("")
          .with_columns(qid=pl.col("matched_entity_ids").str.split(",")).explode("qid")
          .filter(pl.col("qid") != "").select(pl.col("source1_entity_id").alias("s1id"), "qid"))
    be = pl.read_parquet(f"{W}/train_be.parquet").filter(pl.col("rk_be") < 3)
    fold_a = (pl.col("s1id").hash(7) % 2) == 0
    top1 = be.filter(pl.col("rk_be") == 0).select("qid", top1=pl.col("s1id"))
    q = (top1.join(gt.rename({"s1id": "true_s1"}), on="qid", how="left")
         .filter(pl.when(pl.col("true_s1").is_null()).then((pl.col("top1").hash(7) % 2) == 0)
                 .otherwise((pl.col("true_s1").hash(7) % 2) == 0)).select("qid"))
    q = q.sample(min(n_queries, q.height), seed=0)
    p = (be.join(q, on="qid", how="semi").filter(fold_a)
         .join(gt.with_columns(y=pl.lit(1.0)), on=["qid", "s1id"], how="left")
         .with_columns(pl.col("y").fill_null(0.0)).select("qid", "s1id", "y"))
    return p.sample(fraction=1.0, shuffle=True, seed=1)


def train(n_queries):
    p = with_text(train_pairs(n_queries), "train")
    print("train pairs", p.height, "pos rate", p["y"].mean(), flush=True)
    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    
    # Load 1.5B LLM directly in BF16
    model = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=1, torch_dtype=torch.bfloat16).cuda()
    model.config.pad_token_id = tok.pad_token_id
    
    data = list(zip(p["q_txt"].to_list(), p["s_txt"].to_list(), p["y"].to_list()))
    
    # Gradient Accumulation
    BATCH_SIZE = 8
    GRAD_ACCUM = 32
    
    dl = DataLoader(data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate(tok), num_workers=6)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    
    total_steps = len(dl) // GRAD_ACCUM
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-5, total_steps=total_steps, pct_start=0.05)
    lossf = torch.nn.BCEWithLogitsLoss()
    model.train()
    
    t0, run = time.time(), 0.0
    opt.zero_grad(set_to_none=True)
    
    for i, b in enumerate(dl):
        b = {k: v.cuda(non_blocking=True) for k, v in b.items()}
        y = b.pop("labels")
        
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logit = model(**b).logits.squeeze(-1)
            loss = lossf(logit.float(), y) / GRAD_ACCUM
            
        loss.backward()
        
        if (i + 1) % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            
        run = 0.98 * run + 0.02 * (loss.item() * GRAD_ACCUM)
        if (i + 1) % (GRAD_ACCUM * 10) == 0:
            curr = (i + 1) // GRAD_ACCUM
            print(f"step {curr}/{total_steps} loss {run:.4f} {((i + 1) * BATCH_SIZE) / (time.time() - t0):.0f} pairs/s", flush=True)
            
    print(f"done {total_steps} steps {len(data) / (time.time() - t0):.0f} pairs/s", flush=True)
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
    
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        
    model = AutoModelForSequenceClassification.from_pretrained(OUT, torch_dtype=torch.bfloat16).cuda().eval()
    model.config.pad_token_id = tok.pad_token_id
    
    data = list(zip(p["q_txt"].to_list(), p["s_txt"].to_list(), [0.0] * p.height))
    # Reduced batch size for 1.5B LLM inference
    dl = DataLoader(data, batch_size=32, shuffle=False, collate_fn=collate(tok), num_workers=6)
    res, t0 = [], time.time()
    for i, b in enumerate(dl):
        b.pop("labels")
        res.append(model(**{k: v.cuda(non_blocking=True) for k, v in b.items()}).logits.squeeze(-1).float().cpu().numpy())
        if i % 100 == 0:
            print(f"{i * 32}/{p.height} {(i + 1) * 32 / (time.time() - t0):.0f} pairs/s", flush=True)
    np.save(out, np.concatenate(res))
    print("scored", out, flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "train":
        train(int(sys.argv[2]) if len(sys.argv) > 2 else 1_500_000)
    else:
        score(sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]))
