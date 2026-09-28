"""Cache v3c's stage-2 tables (work/{train,test}_s2feat_v3c.parquet) used by the final model.

Recreates v3c's stage-1 model without the density-dependent competitor-count features,
recomputes the stage-1 probability p1 for the stage-2 pairs, and stores the v3b stage-2
table with those columns replaced. Called from run_final.sh.
"""
import lightgbm as lgb
import numpy as np
import polars as pl

from train import PARAMS
from v3c import DROP

W = "work"


def build_frames(links):
    """Recreate v3c's stage-2 inputs (stage 1 without count features) and cache them."""
    f1 = (pl.read_parquet(f"{W}/train_feat.parquet").join(links, on=["qid", "s1id"], how="left")
          .with_columns(pl.col("y").fill_null(0), fold=(pl.col("qid").hash(3) % 2).cast(pl.Int8)))
    F1 = [c for c in f1.columns if c not in ["qid", "s1id", "y", "fold", "p"] + DROP]
    s2tr = pl.read_parquet(f"{W}/train_s2feat_v3b.parquet", columns=["qid", "s1id"])
    need = f1.join(s2tr, on=["qid", "s1id"], how="semi")
    sub = (f1["qid"].hash(5) % 4 == 0).to_numpy()
    X, y, fold = f1.select(F1).to_numpy().astype(np.float32), f1["y"].to_numpy(), f1["fold"].to_numpy()
    Xn, foldn = need.select(F1).to_numpy().astype(np.float32), need["fold"].to_numpy()
    p1 = np.zeros(need.height, np.float32)
    for k in (0, 1):
        m = lgb.train(PARAMS, lgb.Dataset(X[(fold != k) & sub], y[(fold != k) & sub], feature_name=F1), 300)
        p1[foldn == k] = m.predict(Xn[foldn == k], num_threads=16)
    m1 = lgb.train(PARAMS, lgb.Dataset(X[sub], y[sub], feature_name=F1), 300)
    del X, f1
    te_pairs = pl.read_parquet(f"{W}/test_s2feat_v3b.parquet", columns=["qid", "s1id"])
    f1t = pl.scan_parquet(f"{W}/test_feat.parquet").join(te_pairs.lazy(), on=["qid", "s1id"], how="semi").collect()
    p1t = m1.predict(f1t.select(F1).to_numpy().astype(np.float32), num_threads=16).astype(np.float32)
    newp1 = {"train": need.select("qid", "s1id", p1n=pl.Series(p1)), "test": f1t.select("qid", "s1id", p1n=pl.Series(p1t))}
    for split in ("train", "test"):
        d = pl.read_parquet(f"{W}/{split}_s2feat_v3b.parquet").join(newp1[split], on=["qid", "s1id"], how="left")
        d = d.with_columns(p1=pl.col("p1n")).drop("p1n", *[c for c in DROP if c in d.columns])
        d = d.with_columns(p1_gap=pl.col("p1") - pl.col("p1").max().over("qid"),
                           r1=pl.col("p1").rank(descending=True, method="ordinal").over("qid").cast(pl.Int8) - 1)
        d.write_parquet(f"{W}/{split}_s2feat_v3c.parquet")
        print("cached", split, d.shape, flush=True)
