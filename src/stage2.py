"""Stage 2: rerank each query's top-3 stage-1 candidates.

Adds (a) the cross-encoder logit, (b) house-number features that separate a typo
(11345 -> 91345) from a real change (159 -> 160), (c) name extra-token features:
tokens present in one name but not the other, weighted by how often that token is
noise in true pairs vs a real difference in hard negatives (weights learned on
fold A only), and (d) legal-form agreement. Then LightGBM + the same per-query
argmax assignment and macro-F0.5 threshold search as stage 1.

  python stage2.py pairs <split>      # choose top-3 pairs -> work/{split}_s2pairs.parquet
  python stage2.py tokw               # learn extra-token weights on fold A
  python stage2.py feats <split>      # hand features + CE -> work/{split}_s2feat.parquet
  python stage2.py fit                # CV, threshold, final model
  python stage2.py predict [outdir]   # write submission files
"""
import json
import math
import os
import sys
from collections import Counter
from multiprocessing import Pool

import lightgbm as lgb
import numpy as np
import polars as pl
from rapidfuzz.distance import Levenshtein

from features import eval_s1
from normalize import LEGAL, name_parts
from train import PARAMS, assign, labels, score

W = "work"
TOPN = 3
TAG = os.environ.get("S2_TAG", "")  # experiment tag for output files
CE_PREFIXES = os.environ.get("CE_PREFIXES", "ce").split(",")  # cross-encoder score sets -> ce, ce2, ...


def pairs(split):
    """Top-3 stage-1 candidates per query. For train, only queries relevant to the
    held-out set E (argmax S1 in E, or true S1 in E)."""
    src = f"{W}/train_oof.parquet" if split == "train" else f"{W}/test_pred.parquet"
    d = pl.read_parquet(src).select("qid", "s1id", "p").rename({"p": "p1"})
    d = d.sort("p1", descending=True).with_columns(r1=pl.int_range(pl.len()).over("qid").cast(pl.Int8))
    if split == "train":
        _, links = labels()
        best = d.filter(pl.col("r1") == 0).select("qid", best=pl.col("s1id"))
        rel = (best.join(links.select("qid", true=pl.col("s1id")), on="qid", how="left")
               .filter(eval_s1(pl.col("best")) | eval_s1(pl.col("true")).fill_null(False)).select("qid"))
        d = d.join(rel, on="qid", how="semi")
    d = d.filter(pl.col("r1") < TOPN)
    d.write_parquet(f"{W}/{split}_s2pairs.parquet")
    print(split, "stage-2 pairs", d.height, "queries", d["qid"].n_unique())


def _extras(a, b):
    """Tokens of a with no close match (lev ratio >= 0.8) in b."""
    out = []
    for t in a:
        if t in b:
            continue
        if not any(Levenshtein.normalized_similarity(t, u) >= 0.8 for u in b):
            out.append(t)
    return out


def _nums(s):
    return s.split() if s else []


def hand_rows(rows):
    """Per pair: extra tokens both ways, legal sets, and house-number relations."""
    out = []
    for qn, qc, sn, sc, qnum, snum in rows:
        q_core, q_leg = name_parts(qn, qc)
        s_core, s_leg = name_parts(sn, sc)
        ex_q, ex_s = _extras(q_core, s_core), _extras(s_core, q_core)
        qN, sN = _nums(qnum), _nums(snum)
        if qN and sN:
            a, b = qN[0], sN[0]
            lev = Levenshtein.distance(a, b)
            try:
                diff = abs(int(a) - int(b))
            except ValueError:
                diff = -1
            any_eq = int(bool(set(qN) & set(sN)))
            best_lev = min(Levenshtein.distance(x, y) for x in qN[:4] for y in sN[:4])
            num = (lev, min(diff, 10**6), int(a.lstrip("0") == b.lstrip("0")), any_eq, best_lev,
                   int(diff == 1), len(qN), len(sN))
        else:
            num = (-1, -1, -1, -1, -1, -1, len(qN), len(sN))
        ql, sl = set(q_leg), set(s_leg)
        out.append((" ".join(ex_q), " ".join(ex_s), len(ex_q), len(ex_s),
                    int(ql == sl), int(("private" in ql) != ("private" in sl)),
                    int(("limited" in ql) != ("limited" in sl)), len(ql ^ sl)) + num)
    return out


HAND_COLS = ["ex_q", "ex_s", "n_ex_q", "n_ex_s", "leg_eq", "leg_priv_x", "leg_ltd_x", "leg_symdiff",
             "hn_lev", "hn_diff", "hn_eq", "hn_any_eq", "hn_best_lev", "hn_off1", "q_nnum", "s_nnum"]


def hand(d, split):
    cols = ["entity_id", "business_name", "country", "nums"]
    s1 = pl.read_parquet(f"{W}/{split}_s1.parquet").select(cols)
    q = pl.concat([pl.read_parquet(f"{W}/{split}_s{i}.parquet").select(cols) for i in (2, 3)])
    j = (d.select("qid", "s1id").join(q, left_on="qid", right_on="entity_id", how="left")
         .join(s1, left_on="s1id", right_on="entity_id", how="left", suffix="_s"))
    rows = list(zip(j["business_name"], j["country"], j["business_name_s"], j["country_s"], j["nums"], j["nums_s"]))
    with Pool(16) as pool:
        res = [r for ch in pool.imap(hand_rows, (rows[i:i + 20000] for i in range(0, len(rows), 20000))) for r in ch]
    return pl.concat([d, pl.DataFrame(res, schema=HAND_COLS, orient="row")], how="horizontal")


def tokw():
    """Extra-token weights: log-odds that a token appearing as an 'extra' comes from
    a true pair vs a hard negative. Learned on fold-A bi-encoder top-3 pairs only."""
    from crossenc import train_pairs
    p = train_pairs(2_000_000).select("qid", "s1id", "y")
    h = hand(p, "train")
    pos, neg = Counter(), Counter()
    for ex_q, ex_s, y in zip(h["ex_q"], h["ex_s"], h["y"]):
        c = pos if y > 0.5 else neg
        for t in (ex_q + " " + ex_s).split():
            c[t] += 1
    npos, nneg = sum(pos.values()) + 1, sum(neg.values()) + 1
    w = {t: math.log((pos[t] + 1) / npos) - math.log((neg[t] + 1) / nneg)
         for t in set(pos) | set(neg) if pos[t] + neg[t] >= 20}
    pl.DataFrame({"tok": list(w), "w": list(w.values())}).write_parquet(f"{W}/tokw.parquet")
    print("token weights", len(w), "most noise-like", sorted(w, key=w.get)[-15:],
          "most real-change", sorted(w, key=w.get)[:15])


def feats(split):
    cache = f"{W}/{split}_s2hand.parquet"
    if os.path.exists(cache):
        d = pl.read_parquet(cache)
    else:
        d = hand(pl.read_parquet(f"{W}/{split}_s2pairs.parquet"), split)
        d.write_parquet(cache)
    if len(sys.argv) > 3 and sys.argv[3] == "handonly":
        return
    w = dict(pl.read_parquet(f"{W}/tokw.parquet").iter_rows())

    def agg(col):
        vals = [[w.get(t, 0.0) for t in s.split()] for s in d[col].to_list()]
        return (pl.Series([sum(v) for v in vals], dtype=pl.Float32),
                pl.Series([min(v) if v else 0.0 for v in vals], dtype=pl.Float32),
                pl.Series([sum(1 for t in s.split() if t not in w) for s in d[col].to_list()], dtype=pl.Int16))

    for col in ("ex_q", "ex_s"):
        s, m, u = agg(col)
        d = d.with_columns(**{f"{col}_wsum": s, f"{col}_wmin": m, f"{col}_unk": u})
    for j, pre in enumerate(CE_PREFIXES):
        col = "ce" if j == 0 else f"ce{j + 1}"
        ce = np.concatenate([np.load(f"{W}/{pre}_{split}_{i}.npy") for i in range(8)
                             if os.path.exists(f"{W}/{pre}_{split}_{i}.npy")])
        assert len(ce) == d.height, (pre, len(ce), d.height)
        d = d.with_columns(pl.Series(col, ce)).with_columns(
            (pl.col(col) - pl.col(col).max().over("qid")).alias(f"{col}_gap"),
            pl.col(col).rank(descending=True).over("qid").cast(pl.Int8).alias(f"{col}_rank"))
    d = d.with_columns(
        p1_gap=pl.col("p1") - pl.col("p1").max().over("qid"),
        n_top=pl.len().over("qid").cast(pl.Int8),
    ).drop("ex_q", "ex_s")
    f1 = pl.read_parquet(f"{W}/{split}_feat.parquet")
    d = d.join(f1, on=["qid", "s1id"], how="left")
    d.write_parquet(f"{W}/{split}_s2feat{TAG}.parquet")
    print(split, "stage-2 feats", d.shape)


def assign_ef(d, floor=0.15, a=1.0):
    """Expected-F0.5 decision per S1. Each query first picks its argmax S1; then per
    S1 the candidate queries (p >= floor) are sorted by p and the prefix size k that
    maximises E[F0.5] ~ 1.25*sum(top-k p) / (k + 0.25*a*sum(all p)) is kept, compared
    against predicting nothing (E[F] = prod(1 - p)). Returns (s1id, qid)."""
    best = d.sort("p", descending=True).unique("qid", keep="first").filter(pl.col("p") >= floor)
    g = (best.sort(["s1id", "p"], descending=[False, True])
         .with_columns(k=pl.int_range(1, pl.len() + 1).over("s1id"),
                       cum=pl.col("p").cum_sum().over("s1id"),
                       T=pl.col("p").sum().over("s1id") * a,
                       none=(1 - pl.col("p")).product().over("s1id"))
         .with_columns(ef=1.25 * pl.col("cum") / (pl.col("k") + 0.25 * pl.col("T"))))
    kbest = (g.group_by("s1id").agg(pl.col("k").get(pl.col("ef").arg_max()).alias("kb"),
                                    pl.col("ef").max().alias("efb"), pl.col("none").first()))
    kbest = kbest.filter(pl.col("efb") > pl.col("none"))
    return g.join(kbest, on="s1id").filter(pl.col("k") <= pl.col("kb")).select("s1id", "qid")


def fcols(d):
    return [c for c in d.columns if c not in ("qid", "s1id", "y", "fold", "p")]


def fit():
    gt, links = labels()
    gt = gt.filter(eval_s1(pl.col("s1id")))
    d = (pl.read_parquet(f"{W}/train_s2feat{TAG}.parquet").join(links, on=["qid", "s1id"], how="left")
         .with_columns(pl.col("y").fill_null(0), fold=(pl.col("qid").hash(3) % 2).cast(pl.Int8)))
    F = fcols(d)
    X, y, fold = d.select(F).to_numpy().astype(np.float32), d["y"].to_numpy(), d["fold"].to_numpy()
    p = np.zeros(len(y), np.float32)
    prm = dict(PARAMS, learning_rate=0.05, num_leaves=63, min_data_in_leaf=100)
    for k in (0, 1):
        m = lgb.train(prm, lgb.Dataset(X[fold != k], y[fold != k], feature_name=F), 500)
        p[fold == k] = m.predict(X[fold == k], num_threads=16)
    d = d.with_columns(p=pl.Series(p))
    base = {t: score(assign(d.with_columns(p=pl.col("p1")), t), gt) for t in (0.55, 0.6, 0.65)}
    print("stage-1 on same queries:", {k: round(v, 4) for k, v in base.items()})
    res = {t: score(assign(d, t), gt) for t in np.round(np.arange(0.3, 0.86, 0.05), 2)}
    for t, v in res.items():
        print(f"stage-2 t={t:.2f} macroF0.5={v:.4f}", flush=True)
    bt = max(res, key=res.get)
    open(f"{W}/threshold2{TAG}.txt", "w").write(str(bt))
    d.select("qid", "s1id", "y", "p").write_parquet(f"{W}/train_oof2{TAG}.parquet")
    dec = {"rule": "global", "t": float(bt), "score": res[bt]}
    for fl in (0.1, 0.2):
        for a in (1.0, 1.2, 1.4):
            v = score(assign_ef(d, fl, a), gt)
            print(f"expected-F decision floor={fl} a={a}: {v:.4f}", flush=True)
            if v > dec["score"]:
                dec = {"rule": "ef", "floor": fl, "a": a, "score": v}
    print("decision", dec)
    open(f"{W}/decision2{TAG}.json", "w").write(json.dumps(dec))
    m = lgb.train(prm, lgb.Dataset(X, y, feature_name=F), 500)
    m.save_model(f"{W}/model2{TAG}.txt")
    imp = sorted(zip(m.feature_importance("gain"), F), reverse=True)
    print("best t", bt, round(res[bt], 4), "top feats", [(f, int(g)) for g, f in imp[:15]])


def predict(outdir):
    from predict import write_lists
    os.makedirs(outdir, exist_ok=True)
    dec = json.loads(open(f"{W}/decision2{TAG}.json").read())
    m = lgb.Booster(model_file=f"{W}/model2{TAG}.txt")
    d = pl.read_parquet(f"{W}/test_s2feat{TAG}.parquet")
    d = d.with_columns(p=pl.Series(m.predict(d.select(m.feature_name()).to_numpy(), num_threads=16)))
    s1 = pl.read_parquet(f"{W}/test_s1.parquet").select("entity_id")
    write_lists(s1, d.select("s1id", "qid"), "candidate_entity_ids", f"{outdir}/candidate_pairs.tsv")
    match = assign(d, dec["t"]) if dec["rule"] == "global" else assign_ef(d, dec["floor"], dec["a"])
    write_lists(s1, match, "matched_entity_ids", f"{outdir}/matching_results.tsv")


if __name__ == "__main__":
    cmd = sys.argv[1]
    {"pairs": lambda: pairs(sys.argv[2]), "tokw": tokw, "feats": lambda: feats(sys.argv[2]),
     "fit": fit, "predict": lambda: predict(sys.argv[2] if len(sys.argv) > 2 else "output_v2")}[cmd]()
