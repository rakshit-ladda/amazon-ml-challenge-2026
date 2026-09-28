"""v3q: v3c + LLM cross-encoder scores (Qwen) as extra stage-2 features.

The Qwen models score only v3c's uncertain pairs (queries whose best v3c probability
is in (0.01, 0.99)); for every other pair the Qwen columns are missing, which
LightGBM handles natively. The band is defined the same way on validation and test,
so the feature means the same thing in both. Everything else is v3c's cached
stage-2 table. Weights/threshold chosen on the tune half; lockbox only reported.

  python v3q.py <tag> [<tag> ...]      # reads work/<tag>_{train,test}_band_*.parquet
"""
import glob
import json
import os
import sys

import lightgbm as lgb
import numpy as np
import polars as pl

from features import eval_s1
from predict import write_lists
from train import PARAMS, assign, labels, score
from v3c import lock_split

W = "work"


def add_llm(d, split, tags):
    for t in tags:
        parts = sorted(glob.glob(f"{W}/{t}_{split}_band_*.parquet"))
        s = pl.concat([pl.read_parquet(p) for p in parts]).rename({"logit": t})
        d = d.join(s, on=["qid", "s1id"], how="left").with_columns(
            (pl.col(t) - pl.col(t).max().over("qid")).alias(f"{t}_gap"))
        print(split, t, "scored pairs", s.height, flush=True)
    return d


if __name__ == "__main__":
    tags = sys.argv[1:]
    name = "_".join(tags)
    gt, links = labels()
    gt_e = gt.filter(eval_s1(pl.col("s1id")))
    gt_tune, gt_lock = gt_e.filter(~lock_split(pl.col("s1id"))), gt_e.filter(lock_split(pl.col("s1id")))
    tr = add_llm(pl.read_parquet(f"{W}/train_s2feat_v3c.parquet"), "train", tags)
    tr = tr.join(links, on=["qid", "s1id"], how="left").with_columns(
        pl.col("y").fill_null(0), fold=(pl.col("qid").hash(3) % 2).cast(pl.Int8))
    F = [c for c in tr.columns if c not in ("qid", "s1id", "y", "fold", "p")]
    X, y, fold = tr.select(F).to_numpy().astype(np.float32), tr["y"].to_numpy(), tr["fold"].to_numpy()
    prm = dict(PARAMS, learning_rate=0.05, num_leaves=63, min_data_in_leaf=100)
    p = np.zeros(len(y), np.float32)
    for k in (0, 1):
        m = lgb.train(prm, lgb.Dataset(X[fold != k], y[fold != k], feature_name=F), 500)
        p[fold == k] = m.predict(X[fold == k], num_threads=16)
    oof = tr.select("qid", "s1id").with_columns(p=pl.Series(p))
    tune = {t: score(assign(oof, t), gt_tune) for t in (0.35, 0.4, 0.45, 0.5, 0.55)}
    bt = float(max(tune, key=tune.get))
    print(f"v3c lockbox 0.9908 | v3q[{name}] lockbox {score(assign(oof, bt), gt_lock):.4f} "
          f"full val {score(assign(oof, bt), gt_e):.4f} (t={bt}, chosen on tune half)", flush=True)
    m = lgb.train(prm, lgb.Dataset(X, y, feature_name=F), 500)
    imp = sorted(zip(m.feature_importance("gain"), F), reverse=True)
    print("top feats", [(f, int(g)) for g, f in imp[:8]], flush=True)
    m.save_model(f"{W}/model2_v3q_{name}.txt")
    json.dump({"t": bt}, open(f"{W}/decision_v3q_{name}.json", "w"))
    te = add_llm(pl.read_parquet(f"{W}/test_s2feat_v3c.parquet"), "test", tags)
    pred = te.select("qid", "s1id").with_columns(
        p=pl.Series(m.predict(te.select(F).to_numpy().astype(np.float32), num_threads=16).astype(np.float32)))
    out = f"output_v3q_{name}"
    os.makedirs(out, exist_ok=True)
    s1 = pl.read_parquet(f"{W}/test_s1.parquet").select("entity_id")
    write_lists(s1, pred.select("s1id", "qid"), "candidate_entity_ids", f"{out}/candidate_pairs.tsv")
    write_lists(s1, assign(pred, bt), "matched_entity_ids", f"{out}/matching_results.tsv")
