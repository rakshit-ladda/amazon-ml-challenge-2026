import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

"""Fine-tuned bi-encoder for dense candidate retrieval.

Base model: intfloat/multilingual-e5-small (MIT licence, 118M params) -- multilingual,
so it has seen Devanagari and French. Fine-tuned contrastively on (Source 1 record,
matching S2/S3 record) pairs from the training ground truth with in-batch negatives
(MultipleNegativesRankingLoss). Only S1 entities in fold A (hash % 2 == 0) are used for
training, so fold B stays clean for training/validating the downstream reranker.

Usage:
  python biencoder.py train  [n_pairs]
  python biencoder.py encode <split> <src> <shard> <nshards>   # one GPU slice per shard
  python biencoder.py search <split> [k]
"""
import os
import sys

import numpy as np
import polars as pl

W = "work"
BASE = "BAAI/bge-m3"
OUT = f"{W}/biencoder"
MAXLEN = 128


def text(df):
    """Model input: 'name | address'."""
    return (df["business_name"] + " | " + df["business_address"]).to_list()


def train(n_pairs):
    import torch
    from datasets import Dataset
    from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
    from sentence_transformers.sentence_transformer import losses
    from sentence_transformers.sentence_transformer.training_args import BatchSamplers

    gt = pl.read_csv("../dataset/train/train_ground_truth.tsv", separator="\t", infer_schema=False).fill_null("")
    links = (gt.rename({"source1_entity_id": "s1id"})
             .filter((pl.col("s1id").hash(7) % 2) == 0)
             .with_columns(qid=pl.col("matched_entity_ids").str.split(",")).explode("qid")
             .filter(pl.col("qid") != "").select("s1id", "qid"))
    s1 = pl.read_parquet(f"{W}/train_s1.parquet").select("entity_id", "business_name", "business_address", "country")
    q = pl.concat([pl.read_parquet(f"{W}/train_s{i}.parquet") for i in (2, 3)]).select(
        "entity_id", "business_name", "business_address")
    links = links.sample(min(n_pairs, links.height), seed=0)
    d = (links.join(s1, left_on="s1id", right_on="entity_id")
         .join(q, left_on="qid", right_on="entity_id", suffix="_q"))
    # shuffle within country so in-batch negatives are same-country (harder)
    d = d.sample(fraction=1.0, shuffle=True, seed=1).sort("country", maintain_order=True)
    a = ("query: " + d["business_name"] + " | " + d["business_address"]).to_list()
    b = ("query: " + d["business_name_q"] + " | " + d["business_address_q"]).to_list()
    print("train pairs", len(a), flush=True)

    model = SentenceTransformer(BASE, device="cuda")
    model.max_seq_length = MAXLEN
    ds = Dataset.from_dict({"anchor": a, "positive": b})
    args = SentenceTransformerTrainingArguments(
        output_dir=f"{W}/bi_ckpt", num_train_epochs=1, per_device_train_batch_size=32, gradient_accumulation_steps=16,
        learning_rate=5e-5, warmup_steps=200, bf16=True, logging_steps=200, save_strategy="no",
        batch_sampler=BatchSamplers.NO_DUPLICATES, dataloader_num_workers=4, report_to="none")
    trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=ds,
                                         loss=losses.MultipleNegativesRankingLoss(model))
    trainer.train()
    model.save(OUT)


def encode(split, src, shard, nshards):
    """Embed one shard of a source file; writes float16 .npy + ids."""
    import torch
    from sentence_transformers import SentenceTransformer
    df = pl.read_parquet(f"{W}/{split}_s{src}.parquet").select("entity_id", "business_name", "business_address")
    n = df.height
    lo, hi = n * shard // nshards, n * (shard + 1) // nshards
    df = df.slice(lo, hi - lo)
    model = SentenceTransformer(OUT, device="cuda")
    model.max_seq_length = MAXLEN
    model.half()
    e = model.encode(text(df), batch_size=128, normalize_embeddings=True, convert_to_numpy=True,
                     show_progress_bar=False).astype(np.float16)
    np.save(f"{W}/emb_{split}_s{src}_{shard}.npy", e)
    print("encoded", split, src, shard, e.shape, flush=True)


def load_emb(split, src):
    ids = pl.read_parquet(f"{W}/{split}_s{src}.parquet").select("entity_id", "country")
    parts = sorted([f for f in os.listdir(W) if f.startswith(f"emb_{split}_s{src}_")],
                   key=lambda f: int(f.rsplit("_", 1)[1][:-4]))
    return ids, np.concatenate([np.load(f"{W}/{f}") for f in parts])


def search(split, k):
    """Per country: top-k S1 neighbours (cosine) for every S2/S3 record, on GPU."""
    import torch
    s1ids, s1e = load_emb(split, 1)
    s1_all = s1ids["entity_id"].to_numpy()
    out = []
    for src in (2, 3):
        qids, qe = load_emb(split, src)
        q_all = qids["entity_id"].to_numpy()
        for c in s1ids["country"].unique().to_list():
            sidx = np.flatnonzero((s1ids["country"] == c).to_numpy())
            qidx = np.flatnonzero((qids["country"] == c).to_numpy())
            if len(qidx) == 0:
                continue
            S = torch.from_numpy(s1e[sidx]).cuda()
            QI, SI, SC = [], [], []
            for j in range(0, len(qidx), 2048):
                blk = qidx[j:j + 2048]
                sc, ix = torch.topk(torch.from_numpy(qe[blk]).cuda() @ S.T, k, dim=1)
                QI.append(np.repeat(blk, k)); SI.append(ix.cpu().numpy().ravel()); SC.append(sc.float().cpu().numpy().ravel())
            QI, SI, SC = np.concatenate(QI), np.concatenate(SI), np.concatenate(SC)
            out.append(pl.DataFrame({
                "qid": q_all[QI], "s1id": s1_all[sidx[SI]], "cos_be": SC,
                "rk_be": np.tile(np.arange(k, dtype=np.int16), len(QI) // k),
            }))
            del S
            print("searched", split, src, c, flush=True)
    pl.concat(out).write_parquet(f"{W}/{split}_be.parquet")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "train":
        train(int(sys.argv[2]) if len(sys.argv) > 2 else 2_000_000)
    elif cmd == "encode":
        encode(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
    elif cmd == "search":
        search(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 20)
