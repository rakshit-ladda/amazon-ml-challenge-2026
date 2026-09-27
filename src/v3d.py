"""v3d: v3c, but France pairs are scored by a stage-2 model restricted to features
whose meaning carries over to France.

Adversarial validation (validation vs test, content features only) gave AUC 0.69
for US+India but 0.9995 for France. The France shift comes from features built on
US/India conventions: extra-token weights learned from US/India words (~73% of
French extra tokens are unknown vs 8% in validation), address-number counts (no
ZIP/PIN in French addresses), English/Indian legal-form flags, and runner-up
similarity (French names are more alike). The France model drops those and keeps
the cross-encoders, bi-encoder cosine, string similarities and house-number
equality. US and India keep the full v3c model.

  python v3d.py   -> output_v3d/, work/test_pred_v3d.parquet
"""
import json
import os

import lightgbm as lgb
import numpy as np
import polars as pl

from features import eval_s1
from predict import write_lists
from train import PARAMS, assign, labels, score
from v3c import lock_split

W = "work"
FR_DROP = ["p1", "p1_gap", "r1",  # stage-1 probability is itself built from the France-broken features
           "ex_q_unk", "ex_s_unk", "ex_q_wsum", "ex_s_wsum", "ex_q_wmin", "ex_s_wmin",
           "q_nnum", "s_nnum", "num_jac", "num_any", "num_first",
           "leg_priv_x", "leg_ltd_x", "l_tset", "cos_be_2nd", "cos_na_2nd"]

if __name__ == "__main__":
    gt, links = labels()
    gt_e = gt.filter(eval_s1(pl.col("s1id")))
    gt_tune, gt_lock = gt_e.filter(~lock_split(pl.col("s1id"))), gt_e.filter(lock_split(pl.col("s1id")))
    v3c_oof = pl.read_parquet(f"{W}/train_oof2_v3c.parquet")
    t_c = json.load(open(f"{W}/decision2_v3c.json"))["t"]

    import v3c as V
    tr = (pl.read_parquet(f"{W}/train_s2feat_v3b.parquet").drop([c for c in V.DROP])
          .join(links, on=["qid", "s1id"], how="left").with_columns(
              pl.col("y").fill_null(0), fold=(pl.col("qid").hash(3) % 2).cast(pl.Int8)))
    F_full = [c for c in tr.columns if c not in ("qid", "s1id", "y", "fold", "p")]
    F_fr = [c for c in F_full if c not in FR_DROP]
    print("France model features", len(F_fr), "of", len(F_full), "dropped", [c for c in F_full if c in FR_DROP])
    X, y, fold = tr.select(F_fr).to_numpy().astype(np.float32), tr["y"].to_numpy(), tr["fold"].to_numpy()
    prm = dict(PARAMS, learning_rate=0.05, num_leaves=63, min_data_in_leaf=100)
    p = np.zeros(len(y), np.float32)
    for k in (0, 1):
        m = lgb.train(prm, lgb.Dataset(X[fold != k], y[fold != k], feature_name=F_fr), 500)
        p[fold == k] = m.predict(X[fold == k], num_threads=16)
    oof = tr.select("qid", "s1id").with_columns(p=pl.Series(p))
    ts = np.round(np.arange(0.3, 0.71, 0.05), 2)
    tune = {t: score(assign(oof, t), gt_tune) for t in ts}
    t_fr = float(max(tune, key=tune.get))
    print(f"v3c full model    US/IN lockbox {score(assign(v3c_oof, t_c), gt_lock):.4f} (t={t_c})")
    print(f"France-safe model US/IN lockbox {score(assign(oof, t_fr), gt_lock):.4f} (t={t_fr})  <- cost of dropping France-broken features")
    m_fr = lgb.train(prm, lgb.Dataset(X, y, feature_name=F_fr), 500)
    m_fr.save_model(f"{W}/model2_v3d_fr.txt")

    te = pl.read_parquet(f"{W}/test_s2feat_v3b.parquet").drop([c for c in V.DROP])
    q = pl.concat([pl.read_parquet(f"{W}/test_s{i}.parquet").select(pl.col("entity_id").alias("qid"), "country") for i in (2, 3)])
    fr_q = q.filter(pl.col("country") == "France").select("qid")
    fr = te.join(fr_q, on="qid", how="semi")
    fr_pred = fr.select("qid", "s1id").with_columns(
        p=pl.Series(m_fr.predict(fr.select(F_fr).to_numpy().astype(np.float32), num_threads=16).astype(np.float32)))
    base = pl.read_parquet(f"{W}/test_pred_v3c.parquet")
    pred = pl.concat([base.join(fr_q, on="qid", how="anti"), fr_pred])  # every query's candidates share one country
    pred.write_parquet(f"{W}/test_pred_v3d.parquet")
    a = assign(base, t_c)
    b = pl.concat([assign(base.join(fr_q, on="qid", how="anti"), t_c), assign(fr_pred, t_fr)])
    sa, sb = set(zip(a["s1id"], a["qid"])), set(zip(b["s1id"], b["qid"]))
    print("test links v3c", len(sa), "v3d", len(sb), "added", len(sb - sa), "removed", len(sa - sb), "(changes are France-only)")
    os.makedirs("output_v3d", exist_ok=True)
    s1 = pl.read_parquet(f"{W}/test_s1.parquet").select("entity_id")
    write_lists(s1, pred.select("s1id", "qid"), "candidate_entity_ids", "output_v3d/candidate_pairs.tsv")
    write_lists(s1, b, "matched_entity_ids", "output_v3d/matching_results.tsv")
