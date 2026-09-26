"""Stage 3: re-decide only the uncertain queries (cascade).

Queries whose best stage-2 probability is already near 0 or 1 keep the stage-2
decision. For the rest (about 5% of queries) we add scores from newer
cross-encoders and refit a small LightGBM on the held-out validation band, then
merge its probabilities back and choose the threshold on macro F0.5 over the
full validation set. Scoring only the band makes each new cross-encoder cheap
to evaluate.

  python stage3.py band                     # -> work/{train,test}_band.parquet
  python stage3.py fit  <ce_tag> [...]      # uses work/<tag>_{split}_band*.parquet logits
  python stage3.py predict <outdir> <ce_tag> [...]
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

W = "work"
BASE_TAG = os.environ.get("S3_BASE", "_v3b")  # stage-2 run whose features/probabilities we refine
LO, HI = 0.01, 0.99


def band():
    for split, src in (("train", f"{W}/train_oof2{BASE_TAG}.parquet"), ("test", f"{W}/test_pred{BASE_TAG}.parquet")):
        d = pl.read_parquet(src).select("qid", "s1id", "p")
        q = d.group_by("qid").agg(pl.col("p").max().alias("pm")).filter((pl.col("pm") > LO) & (pl.col("pm") < HI))
        b = d.join(q.select("qid"), on="qid", how="semi").rename({"p": "p2"})
        b.write_parquet(f"{W}/{split}_band.parquet")
        print(split, "band pairs", b.height, "queries", q.height)


def with_ce(split, tags):
    """Band pairs + stage-2 features + one logit column per cross-encoder tag."""
    b = pl.read_parquet(f"{W}/{split}_band.parquet")
    f2 = pl.scan_parquet(f"{W}/{split}_s2feat{BASE_TAG}.parquet").join(b.lazy().select("qid", "s1id"),
                                                                         on=["qid", "s1id"], how="semi").collect()
    b = b.join(f2, on=["qid", "s1id"], how="left")
    for t in tags:
        parts = sorted(glob.glob(f"{W}/{t}_{split}_band_*.parquet"))
        ce = pl.concat([pl.read_parquet(x) for x in parts]).rename({"logit": t})
        b = b.join(ce, on=["qid", "s1id"], how="left").with_columns(
            (pl.col(t) - pl.col(t).max().over("qid")).alias(f"{t}_gap"),
            pl.col(t).rank(descending=True).over("qid").cast(pl.Int8).alias(f"{t}_rank"))
        assert b[t].null_count() == 0, (t, b[t].null_count())
    return b


def fcols(d):
    return [c for c in d.columns if c not in ("qid", "s1id", "y", "fold", "p")]


def fit(tags):
    gt, links = labels()
    gt = gt.filter(eval_s1(pl.col("s1id")))
    b = (with_ce("train", tags).join(links, on=["qid", "s1id"], how="left")
         .with_columns(pl.col("y").fill_null(0), fold=(pl.col("qid").hash(9) % 2).cast(pl.Int8)))
    F = fcols(b)
    X, y, fold = b.select(F).to_numpy().astype(np.float32), b["y"].to_numpy(), b["fold"].to_numpy()
    prm = dict(PARAMS, learning_rate=0.03, num_leaves=31, min_data_in_leaf=50)
    p = np.zeros(len(y), np.float32)
    for k in (0, 1):
        m = lgb.train(prm, lgb.Dataset(X[fold != k], y[fold != k], feature_name=F), 600)
        p[fold == k] = m.predict(X[fold == k], num_threads=16)
    full = pl.read_parquet(f"{W}/train_oof2{BASE_TAG}.parquet").select("qid", "s1id", "p")
    base = {t: score(assign(full, t), gt) for t in (0.45, 0.5, 0.55)}
    print("stage-2 baseline:", {k: round(v, 4) for k, v in base.items()})
    new = full.join(b.select("qid", "s1id", p3=pl.Series(p)), on=["qid", "s1id"], how="left").with_columns(
        p=pl.coalesce("p3", "p")).drop("p3")
    res = {t: score(assign(new, t), gt) for t in np.round(np.arange(0.3, 0.71, 0.05), 2)}
    for t, v in res.items():
        print(f"stage-3 t={t:.2f} macroF0.5={v:.4f}", flush=True)
    bt = max(res, key=res.get)
    m = lgb.train(prm, lgb.Dataset(X, y, feature_name=F), 600)
    tag = "_".join(tags)
    m.save_model(f"{W}/model3_{tag}.txt")
    open(f"{W}/decision3_{tag}.json", "w").write(json.dumps({"t": float(bt), "score": res[bt]}))
    imp = sorted(zip(m.feature_importance("gain"), F), reverse=True)
    print("best t", bt, round(res[bt], 4), "top feats", [(f, int(g)) for g, f in imp[:10]])


def predict(outdir, tags):
    os.makedirs(outdir, exist_ok=True)
    tag = "_".join(tags)
    t = json.loads(open(f"{W}/decision3_{tag}.json").read())["t"]
    m = lgb.Booster(model_file=f"{W}/model3_{tag}.txt")
    b = with_ce("test", tags)
    b = b.select("qid", "s1id", p3=pl.Series(m.predict(b.select(m.feature_name()).to_numpy(), num_threads=16)))
    full = pl.read_parquet(f"{W}/test_pred{BASE_TAG}.parquet").select("qid", "s1id", "p")
    new = full.join(b, on=["qid", "s1id"], how="left").with_columns(p=pl.coalesce("p3", "p")).drop("p3")
    s1 = pl.read_parquet(f"{W}/test_s1.parquet").select("entity_id")
    write_lists(s1, new.select("s1id", "qid"), "candidate_entity_ids", f"{outdir}/candidate_pairs.tsv")
    write_lists(s1, assign(new, t), "matched_entity_ids", f"{outdir}/matching_results.tsv")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "band":
        band()
    elif cmd == "fit":
        fit(sys.argv[2:])
    else:
        predict(sys.argv[2], sys.argv[3:])
