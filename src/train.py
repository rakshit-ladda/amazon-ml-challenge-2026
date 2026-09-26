"""Train the pairwise matcher and pick the decision threshold.

2-fold CV grouped by Source 1 entity gives out-of-fold probabilities for every
train pair. Decision rule: every S2/S3 record is assigned to its single most
probable S1 candidate if that probability clears a threshold t (each S2/S3
record matches at most one S1). t is chosen to maximise the challenge's macro
F0.5 over all train S1 entities. The final model is refit on all pairs."""
import sys

import lightgbm as lgb
import numpy as np
import polars as pl

from features import eval_s1
from metric import f05

W = "work"
FEATS = None  # set from the frame
PARAMS = dict(objective="binary", learning_rate=0.08, num_leaves=127, min_data_in_leaf=200,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              num_threads=16, verbose=-1)
ROUNDS = 300


def labels():
    """Ground truth as (qid, s1id) link frame and {s1: set(qids)} dict."""
    gt = (pl.read_csv("dataset/train/train_ground_truth.tsv", separator="\t", infer_schema=False)
          .fill_null("").rename({"source1_entity_id": "s1id"}))
    links = (gt.with_columns(qid=pl.col("matched_entity_ids").str.split(",")).explode("qid")
             .filter(pl.col("qid") != "").select("qid", "s1id").with_columns(y=pl.lit(1, pl.Int8)))
    return gt, links


def feature_cols(d):
    return [c for c in d.columns if c not in ("qid", "s1id", "y", "fold", "p")]


def assign(d, t):
    """Per query keep the argmax-probability S1 if p >= t. Returns (s1id, qid) frame."""
    best = d.sort("p", descending=True).unique("qid", keep="first")
    return best.filter(pl.col("p") >= t).select("s1id", "qid")


def score(pred, gt):
    """Macro F0.5 over every S1 in gt."""
    pm = {k: set(v) for k, v in pred.group_by("s1id").agg("qid").iter_rows()}
    tot = 0.0
    for s1, m in gt.iter_rows():
        tot += f05(pm.get(s1, set()), set(m.split(",")) if m else set())
    return tot / gt.height


if __name__ == "__main__":
    ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else ROUNDS
    gt, links = labels()
    d = (pl.read_parquet(f"{W}/train_feat.parquet").join(links, on=["qid", "s1id"], how="left")
         .with_columns(pl.col("y").fill_null(0), fold=(pl.col("qid").hash(3) % 2).cast(pl.Int8)))
    gt = gt.filter(eval_s1(pl.col("s1id")))  # score only the held-out reranker S1 set
    F = feature_cols(d)
    print("pairs", d.height, "pos", d["y"].sum(), "feats", len(F), flush=True)
    X = d.select(F).to_numpy().astype(np.float32)
    y = d["y"].to_numpy()
    fold = d["fold"].to_numpy()
    p = np.zeros(len(y), np.float32)
    sub = (d["qid"].hash(5) % 4 == 0).to_numpy()  # fit on 25% of queries for speed
    for k in (0, 1):
        tr = (fold != k) & sub
        m = lgb.train(PARAMS, lgb.Dataset(X[tr], y[tr], feature_name=F), ROUNDS)
        p[fold == k] = m.predict(X[fold == k], num_threads=16)
        print("fold", k, "done", flush=True)
    d = d.with_columns(p=pl.Series(p))
    d.select("qid", "s1id", "y", "p").write_parquet(f"{W}/train_oof.parquet")

    res = {t: score(assign(d, t), gt) for t in np.round(np.arange(0.2, 0.9, 0.05), 2)}
    for t, v in res.items():
        print(f"t={t:.2f}  macroF0.5={v:.4f}", flush=True)
    best_t = max(res, key=res.get)
    open(f"{W}/threshold.txt", "w").write(str(best_t))
    print("best t", best_t, res[best_t])

    m = lgb.train(PARAMS, lgb.Dataset(X[sub], y[sub], feature_name=F), ROUNDS)
    m.save_model(f"{W}/model.txt")
    imp = sorted(zip(m.feature_importance("gain"), F), reverse=True)
    print("top feats", [(f, int(g)) for g, f in imp[:15]])
