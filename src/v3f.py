"""v3f: v3c with its two cross-encoder columns re-scored by cross-encoders trained on
~85% of the training businesses (everything except the held-out reranker set E)
instead of fold A (50%). Stage 1, candidates and all other stage-2 features are
v3c's cached tables; E stays unseen by every model, so the lockbox is honest.
Optional extra band-scored LLM tags (e.g. q15b) are added as in v3q.

  python v3f.py <ce_small_tag> <ce_base_tag> [band_tag ...]
    reads work/<tag>_{train,test}_s2_<i>.parquet (qid, s1id, logit) for the two CEs
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
from v3q import add_llm

W = "work"


def extra_feats(d, split):
    """Join optional feature tables named in EXTRA_FEATS (e.g. 'sib' -> work/{split}_sib.parquet)."""
    for f in [x for x in os.environ.get("EXTRA_FEATS", "").split(",") if x]:
        d = d.join(pl.read_parquet(f"{W}/{split}_{f}.parquet"), on=["qid", "s1id"], how="left")
    return d


def swap_ce(d, split, small, base):
    """Replace v3c's ce (e5-small) / ce2 (e5-base) columns and their per-query gap/rank.
    'keep' keeps v3c's original cross-encoder columns."""
    if small == "keep":
        return d
    d = d.drop([c for c in d.columns if c.startswith("ce")])
    for col, tag in (("ce", small), ("ce2", base)):
        s = pl.concat([pl.read_parquet(p) for p in sorted(glob.glob(f"{W}/{tag}_{split}_s2_*.parquet"))]).rename({"logit": col})
        d = d.join(s, on=["qid", "s1id"], how="left")
        assert d[col].null_count() == 0, (tag, split, d[col].null_count())
        d = d.with_columns((pl.col(col) - pl.col(col).max().over("qid")).alias(f"{col}_gap"),
                           pl.col(col).rank(descending=True).over("qid").cast(pl.Int8).alias(f"{col}_rank"))
    return d


if __name__ == "__main__":
    small, base, extra = sys.argv[1], sys.argv[2], sys.argv[3:]
    name = "_".join([small, base] + extra + [x for x in os.environ.get("EXTRA_FEATS", "").split(",") if x])
    gt, links = labels()
    gt_e = gt.filter(eval_s1(pl.col("s1id")))
    gt_tune, gt_lock = gt_e.filter(~lock_split(pl.col("s1id"))), gt_e.filter(lock_split(pl.col("s1id")))
    tr = extra_feats(swap_ce(pl.read_parquet(f"{W}/train_s2feat_v3c.parquet"), "train", small, base), "train")
    if extra:
        tr = add_llm(tr, "train", extra)
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
    oof.write_parquet(f"{W}/train_oof_v3f_{name}.parquet")
    tune = {t: score(assign(oof, t), gt_tune) for t in (0.35, 0.4, 0.45, 0.5, 0.55)}
    bt = float(max(tune, key=tune.get))
    print(f"v3c lockbox 0.9908 | v3f[{name}] lockbox {score(assign(oof, bt), gt_lock):.4f} "
          f"full val {score(assign(oof, bt), gt_e):.4f} (t={bt}, chosen on tune half)", flush=True)
    m = lgb.train(prm, lgb.Dataset(X, y, feature_name=F), 500)
    imp = sorted(zip(m.feature_importance("gain"), F), reverse=True)
    print("top feats", [(f, int(g)) for g, f in imp[:8]], flush=True)
    m.save_model(f"{W}/model2_v3f_{name}.txt")
    json.dump({"t": bt}, open(f"{W}/decision_v3f_{name}.json", "w"))
    te = extra_feats(swap_ce(pl.read_parquet(f"{W}/test_s2feat_v3c.parquet"), "test", small, base), "test")
    if extra:
        te = add_llm(te, "test", extra)
    pred = te.select("qid", "s1id").with_columns(
        p=pl.Series(m.predict(te.select(F).to_numpy().astype(np.float32), num_threads=16).astype(np.float32)))
    pred.write_parquet(f"{W}/test_pred_v3f_{name}.parquet")
    out = f"output_v3f_{name}"
    os.makedirs(out, exist_ok=True)
    s1 = pl.read_parquet(f"{W}/test_s1.parquet").select("entity_id")
    write_lists(s1, pred.select("s1id", "qid"), "candidate_entity_ids", f"{out}/candidate_pairs.tsv")
    write_lists(s1, assign(pred, bt), "matched_entity_ids", f"{out}/matching_results.tsv")
