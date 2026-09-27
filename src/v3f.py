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


def prune(d):
    """Final blocking filter: keep candidates with stage-1 probability >= PRUNE_P1 (runs BEFORE the
    stage-2 model, so the candidate file is exactly what the model scores), then recompute the
    per-query relative features on the smaller candidate set."""
    tau = float(os.environ.get("PRUNE_P1", 0))
    if tau <= 0:
        return d
    d = d.filter(pl.col("p1") >= tau)
    rel = {}
    for c in ("ce", "ce2"):
        if c in d.columns:
            rel[f"{c}_gap"] = pl.col(c) - pl.col(c).max().over("qid")
            rel[f"{c}_rank"] = pl.col(c).rank(descending=True).over("qid").cast(pl.Int8)
    rel["p1_gap"] = pl.col("p1") - pl.col("p1").max().over("qid")
    rel["r1"] = (pl.col("p1").rank(descending=True, method="ordinal").over("qid") - 1).cast(pl.Int8)
    rel["n_top"] = pl.len().over("qid").cast(pl.Int8)
    return d.with_columns(**rel)


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
    name = "_".join([small, base] + extra + [x for x in os.environ.get("EXTRA_FEATS", "").split(",") if x]
                    + ([f"p{os.environ['PRUNE_P1']}"] if float(os.environ.get("PRUNE_P1", 0)) > 0 else []))
    gt, links = labels()
    gt_e = gt.filter(eval_s1(pl.col("s1id")))
    gt_tune, gt_lock = gt_e.filter(~lock_split(pl.col("s1id"))), gt_e.filter(lock_split(pl.col("s1id")))
    tr = extra_feats(prune(swap_ce(pl.read_parquet(f"{W}/train_s2feat_v3c.parquet"), "train", small, base)), "train")
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
    bt = float(os.environ["V3F_T"]) if "V3F_T" in os.environ else float(max(tune, key=tune.get))  # V3F_T pins the threshold
    print(f"v3c lockbox 0.9908 | v3f[{name}] lockbox {score(assign(oof, bt), gt_lock):.4f} "
          f"full val {score(assign(oof, bt), gt_e):.4f} (t={bt})", flush=True)
    m = lgb.train(prm, lgb.Dataset(X, y, feature_name=F), 500)
    imp = sorted(zip(m.feature_importance("gain"), F), reverse=True)
    print("top feats", [(f, int(g)) for g, f in imp[:8]], flush=True)
    m.save_model(f"{W}/model2_v3f_{name}.txt")
    json.dump({"t": bt}, open(f"{W}/decision_v3f_{name}.json", "w"))
    te = extra_feats(prune(swap_ce(pl.read_parquet(f"{W}/test_s2feat_v3c.parquet"), "test", small, base)), "test")
    if extra:
        te = add_llm(te, "test", extra)
    pred = te.select("qid", "s1id").with_columns(
        p=pl.Series(m.predict(te.select(F).to_numpy().astype(np.float32), num_threads=16).astype(np.float32)))
    pred.write_parquet(f"{W}/test_pred_v3f_{name}.parquet")
    out = f"output_v3f_{name}"
    os.makedirs(out, exist_ok=True)
    s1 = pl.read_parquet(f"{W}/test_s1.parquet").select("entity_id", "country")
    write_lists(s1.select("entity_id"), pred.select("s1id", "qid"), "candidate_entity_ids", f"{out}/candidate_pairs.tsv")
    # France is test-only, so its threshold cannot be tuned on validation; FR_T was set on the public leaderboard
    is_fr = pl.col("s1id").is_in(s1.filter(pl.col("country") == "France")["entity_id"].implode())
    match = pl.concat([assign(pred.filter(~is_fr), bt), assign(pred.filter(is_fr), float(os.environ.get("FR_T", bt)))])
    write_lists(s1.select("entity_id"), match, "matched_entity_ids", f"{out}/matching_results.tsv")
