"""v3c: v3b with the density-dependent 'competitor count' features removed.

Why: s1_top1 / s1_top1_be count how many queries compete for the same S1. In
training they were computed on a subset of queries (the reranker's evaluation
subset), and test has ~1.9x more records per S1, so their test distribution is
far from training (val mean 3.45 vs test 6.18 for s1_top1_be). Every other part
of v3b is kept: same candidate pairs, same cross-encoder scores (cefr, cebfr),
same stage-2 features otherwise. Stage 1 is retrained without the counts and
its probability p1 recomputed for the stage-2 pairs; stage 2 is then refit.

  python v3c.py   -> work/train_oof2_v3c.parquet, work/test_pred_v3c.parquet, output_v3c/
"""
import json
import os

import lightgbm as lgb
import numpy as np
import polars as pl

from features import eval_s1
from predict import write_lists
from train import PARAMS, assign, labels, score

W = "work"
DROP = ["s1_top1", "s1_top1_be"]  # density-dependent: differ between train subset and full test


def lock_split(ids):
    """Lockbox: half of the held-out S1 set is never used to choose anything."""
    return (ids.hash(21) % 2) == 1


if __name__ == "__main__":
    gt, links = labels()
    gt_e = gt.filter(eval_s1(pl.col("s1id")))
    gt_tune, gt_lock = gt_e.filter(~lock_split(pl.col("s1id"))), gt_e.filter(lock_split(pl.col("s1id")))

    # ---- stage 1 without count features (same folds / subsample as train.py)
    f1 = (pl.read_parquet(f"{W}/train_feat.parquet").join(links, on=["qid", "s1id"], how="left")
          .with_columns(pl.col("y").fill_null(0), fold=(pl.col("qid").hash(3) % 2).cast(pl.Int8)))
    F1 = [c for c in f1.columns if c not in ["qid", "s1id", "y", "fold", "p"] + DROP]
    s2tr = pl.read_parquet(f"{W}/train_s2feat_v3b.parquet")
    need = f1.join(s2tr.select("qid", "s1id"), on=["qid", "s1id"], how="semi")  # only stage-2 pairs need p1
    sub = (f1["qid"].hash(5) % 4 == 0).to_numpy()
    X, y, fold = f1.select(F1).to_numpy().astype(np.float32), f1["y"].to_numpy(), f1["fold"].to_numpy()
    Xn, foldn = need.select(F1).to_numpy().astype(np.float32), need["fold"].to_numpy()
    p1 = np.zeros(need.height, np.float32)
    for k in (0, 1):
        m = lgb.train(PARAMS, lgb.Dataset(X[(fold != k) & sub], y[(fold != k) & sub], feature_name=F1), 300)
        p1[foldn == k] = m.predict(Xn[foldn == k], num_threads=16)
        print("stage-1 fold", k, flush=True)
    m1 = lgb.train(PARAMS, lgb.Dataset(X[sub], y[sub], feature_name=F1), 300)
    del X, f1
    te_pairs = pl.read_parquet(f"{W}/test_s2feat_v3b.parquet", columns=["qid", "s1id"])
    f1t = pl.scan_parquet(f"{W}/test_feat.parquet").join(te_pairs.lazy(), on=["qid", "s1id"], how="semi").collect()
    p1t = m1.predict(f1t.select(F1).to_numpy().astype(np.float32), num_threads=16)
    newp1 = {"train": need.select("qid", "s1id", p1n=pl.Series(p1)),
             "test": f1t.select("qid", "s1id", p1n=pl.Series(p1t.astype(np.float32)))}
    print("new p1 mean  val", float(p1.mean()), " test", float(p1t.mean()), flush=True)

    # ---- stage 2: v3b features with new p1-derived columns and no count features
    def s2(split):
        d = pl.read_parquet(f"{W}/{split}_s2feat_v3b.parquet").join(newp1[split], on=["qid", "s1id"], how="left")
        d = d.with_columns(p1=pl.col("p1n")).drop("p1n", *[c for c in DROP if c in d.columns])
        return d.with_columns(p1_gap=pl.col("p1") - pl.col("p1").max().over("qid"),
                              r1=pl.col("p1").rank(descending=True, method="ordinal").over("qid").cast(pl.Int8) - 1)

    tr = s2("train").join(links, on=["qid", "s1id"], how="left").with_columns(
        pl.col("y").fill_null(0), fold=(pl.col("qid").hash(3) % 2).cast(pl.Int8))
    F2 = [c for c in tr.columns if c not in ("qid", "s1id", "y", "fold", "p")]
    X2, y2, fold2 = tr.select(F2).to_numpy().astype(np.float32), tr["y"].to_numpy(), tr["fold"].to_numpy()
    prm = dict(PARAMS, learning_rate=0.05, num_leaves=63, min_data_in_leaf=100)
    p2 = np.zeros(len(y2), np.float32)
    for k in (0, 1):
        m = lgb.train(prm, lgb.Dataset(X2[fold2 != k], y2[fold2 != k], feature_name=F2), 500)
        p2[fold2 == k] = m.predict(X2[fold2 == k], num_threads=16)
    oof = tr.select("qid", "s1id", "y").with_columns(p=pl.Series(p2))
    oof.write_parquet(f"{W}/train_oof2_v3c.parquet")
    ref = pl.read_parquet(f"{W}/train_oof2_v3b.parquet")
    ts = np.round(np.arange(0.3, 0.71, 0.05), 2)
    tune = {t: score(assign(oof, t), gt_tune) for t in ts}
    bt = float(max(tune, key=tune.get))
    print(f"v3b  full-val {score(assign(ref, 0.5), gt_e):.4f}  lockbox {score(assign(ref, 0.5), gt_lock):.4f}  (t=0.50)")
    print(f"v3c  full-val {score(assign(oof, bt), gt_e):.4f}  lockbox {score(assign(oof, bt), gt_lock):.4f}  (t={bt}, chosen on tune half)")
    m2 = lgb.train(prm, lgb.Dataset(X2, y2, feature_name=F2), 500)
    m2.save_model(f"{W}/model2_v3c.txt")
    json.dump({"t": bt}, open(f"{W}/decision2_v3c.json", "w"))

    te = s2("test")
    pt = m2.predict(te.select(F2).to_numpy().astype(np.float32), num_threads=16)
    pred = te.select("qid", "s1id").with_columns(p=pl.Series(pt.astype(np.float32)))
    pred.write_parquet(f"{W}/test_pred_v3c.parquet")

    # ---- the confidence check: do val and test now look alike to the model?
    def dist(d, name):
        best = d.sort("p", descending=True).unique("qid", keep="first")["p"]
        print(f"{name}: best-candidate p  mean {best.mean():.4f}  share>=t {float((best >= bt).mean()):.4f}  "
              f"share in (0.05,0.95) {float(((best > 0.05) & (best < 0.95)).mean()):.4f}")
    dist(ref.select("qid", "s1id", "p"), "v3b val ")
    dist(pl.read_parquet(f"{W}/test_pred_v3b.parquet"), "v3b test")
    dist(oof.select("qid", "s1id", "p"), "v3c val ")
    dist(pred, "v3c test")
    os.makedirs("output_v3c", exist_ok=True)
    s1 = pl.read_parquet(f"{W}/test_s1.parquet").select("entity_id")
    write_lists(s1, pred.select("s1id", "qid"), "candidate_entity_ids", "output_v3c/candidate_pairs.tsv")
    write_lists(s1, assign(pred, bt), "matched_entity_ids", "output_v3c/matching_results.tsv")
