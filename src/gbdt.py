"""v3e: GBDT ensemble on v3c's stage-2 features.

Same features, pairs and cross-encoder scores as v3c; three different learners
(LightGBM, XGBoost, CatBoost) trained with the same 2-fold split by query. Their
out-of-fold probabilities are blended with weights and a threshold chosen on the
tune half of the held-out S1 set; the lockbox half is only reported.

  python gbdt.py   -> work/{train,test}_s2feat_v3c.parquet (cached), output_v3e/
"""
import json
import os

import lightgbm as lgb
import numpy as np
import polars as pl

from features import eval_s1
from predict import write_lists
from train import PARAMS, assign, labels, score
from v3c import DROP, lock_split

W = "work"
GPU = os.environ.get("GBDT_GPU", "1") == "1"


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


def fit_lgb(Xa, ya, Xb):
    prm = dict(PARAMS, learning_rate=0.05, num_leaves=63, min_data_in_leaf=100)
    return lgb.train(prm, lgb.Dataset(Xa, ya), 500).predict(Xb, num_threads=16)


def fit_xgb(Xa, ya, Xb):
    import xgboost as xgb
    prm = dict(objective="binary:logistic", eta=0.05, max_depth=8, min_child_weight=20, subsample=0.8,
               colsample_bytree=0.8, tree_method="hist", device="cuda" if GPU else "cpu", nthread=16, eval_metric="logloss")
    m = xgb.train(prm, xgb.DMatrix(Xa, ya), 600)
    return m.predict(xgb.DMatrix(Xb))


def fit_cat(Xa, ya, Xb):
    from catboost import CatBoostClassifier
    m = CatBoostClassifier(iterations=800, learning_rate=0.08, depth=8, loss_function="Logloss", verbose=0,
                           task_type="GPU" if GPU else "CPU", thread_count=16, random_seed=0)
    m.fit(Xa, ya)
    return m.predict_proba(Xb)[:, 1]


LEARNERS = {"lgb": fit_lgb, "xgb": fit_xgb, "cat": fit_cat}

if __name__ == "__main__":
    gt, links = labels()
    gt_e = gt.filter(eval_s1(pl.col("s1id")))
    gt_tune, gt_lock = gt_e.filter(~lock_split(pl.col("s1id"))), gt_e.filter(lock_split(pl.col("s1id")))
    if not os.path.exists(f"{W}/test_s2feat_v3c.parquet"):
        build_frames(links)
    tr = (pl.read_parquet(f"{W}/train_s2feat_v3c.parquet").join(links, on=["qid", "s1id"], how="left")
          .with_columns(pl.col("y").fill_null(0), fold=(pl.col("qid").hash(3) % 2).cast(pl.Int8)))
    F = [c for c in tr.columns if c not in ("qid", "s1id", "y", "fold", "p")]
    X, y, fold = tr.select(F).to_numpy().astype(np.float32), tr["y"].to_numpy(), tr["fold"].to_numpy()
    base = tr.select("qid", "s1id")
    oof = {}
    for name, fn in LEARNERS.items():
        p = np.zeros(len(y), np.float32)
        for k in (0, 1):
            p[fold == k] = fn(X[fold != k], y[fold != k], X[fold == k])
        oof[name] = p
        d = base.with_columns(p=pl.Series(p))
        print(f"{name}: lockbox at t=0.45 {score(assign(d, 0.45), gt_lock):.4f}", flush=True)
    base.with_columns(**{k: pl.Series(v) for k, v in oof.items()}).write_parquet(f"{W}/train_oof_v3e.parquet")

    # weights + threshold on the tune half only
    grid = [(a, b, 1 - a - b) for a in np.arange(0, 1.01, 0.25) for b in np.arange(0, 1.01 - a, 0.25)]
    best = None
    for w in grid:
        p = w[0] * oof["lgb"] + w[1] * oof["xgb"] + w[2] * oof["cat"]
        d = base.with_columns(p=pl.Series(p.astype(np.float32)))
        for t in (0.4, 0.45, 0.5, 0.55):
            s = score(assign(d, t), gt_tune)
            if best is None or s > best[0]:
                best = (s, w, t)
    s, w, t = best
    p = w[0] * oof["lgb"] + w[1] * oof["xgb"] + w[2] * oof["cat"]
    d = base.with_columns(p=pl.Series(p.astype(np.float32)))
    print(f"blend weights lgb/xgb/cat = {tuple(round(x, 2) for x in w)}, t={t} (chosen on tune half)")
    print(f"v3c (LightGBM only) lockbox 0.9908 | v3e blend lockbox {score(assign(d, t), gt_lock):.4f} | full val {score(assign(d, t), gt_e):.4f}")
    json.dump({"w": [float(x) for x in w], "t": float(t)}, open(f"{W}/decision_v3e.json", "w"))

    te = pl.read_parquet(f"{W}/test_s2feat_v3c.parquet")
    Xt = te.select(F).to_numpy().astype(np.float32)
    pt = np.zeros(te.height, np.float64)
    for name, wt in zip(("lgb", "xgb", "cat"), w):
        if wt > 0:
            pt += wt * LEARNERS[name](X, y, Xt)
    pred = te.select("qid", "s1id").with_columns(p=pl.Series(pt.astype(np.float32)))
    pred.write_parquet(f"{W}/test_pred_v3e.parquet")
    os.makedirs("output_v3e", exist_ok=True)
    s1 = pl.read_parquet(f"{W}/test_s1.parquet").select("entity_id")
    write_lists(s1, pred.select("s1id", "qid"), "candidate_entity_ids", "output_v3e/candidate_pairs.tsv")
    write_lists(s1, assign(pred, t), "matched_entity_ids", "output_v3e/matching_results.tsv")
