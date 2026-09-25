"""Pair features for (query S2/S3 record, candidate S1 record).

String similarities come from rapidfuzz's multithreaded cpdist over the
normalised fields; context features describe how a pair ranks among all the
candidates of the same query (the matcher decides per query, since each S2/S3
record belongs to at most one S1 entity)."""
import os
import sys

import numpy as np
import polars as pl
from rapidfuzz import distance, fuzz, process

W = "work"


def sims(a, b, scorer, **kw):
    """Vectorised pairwise similarity, 0..1, workers=all cores."""
    return process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32, **kw) / (
        1.0 if scorer in (distance.JaroWinkler.normalized_similarity,) else 100.0)


def num_feats(qa, sa):
    """Digit-run overlap between two addresses: jaccard, any shared, first number equal."""
    jac, anyh, first = [], [], []
    for x, y in zip(qa, sa):
        xs, ys = x.split(), y.split()
        if not xs or not ys:
            jac.append(-1.0); anyh.append(-1); first.append(-1)
            continue
        X, Y = set(xs), set(ys)
        jac.append(len(X & Y) / len(X | Y)); anyh.append(int(bool(X & Y))); first.append(int(xs[0] == ys[0]))
    return np.array(jac, np.float32), np.array(anyh, np.int8), np.array(first, np.int8)


def eval_s1(ids: pl.Series) -> pl.Series:
    """Train S1 entities used to train/validate the reranker: fold B of the
    bi-encoder split (hash % 2 == 1), subsampled to ~30% of that fold."""
    return ((ids.hash(7) % 2) == 1) & ((ids.hash(11) % 10) < 3)


def candidates(split):
    """Union of sparse blockers and bi-encoder neighbours (if available)."""
    c = pl.read_parquet(f"{W}/{split}_cand.parquet")
    # keep only the sparse blockers' top-3 (bi-encoder carries recall; sparse adds a little)
    c = c.filter(pl.min_horizontal("rk_nm", "rk_ad", "rk_na") < 3)
    if os.path.exists(f"{W}/{split}_be.parquet"):
        be = pl.read_parquet(f"{W}/{split}_be.parquet").filter(pl.col("rk_be") < 10)
        c = c.join(be, on=["qid", "s1id"], how="full", coalesce=True)
    else:
        c = c.with_columns(cos_be=pl.lit(None, pl.Float32), rk_be=pl.lit(None, pl.Int16))
    if split == "train":
        gt = (pl.read_csv("dataset/train/train_ground_truth.tsv", separator="	", infer_schema=False)
              .fill_null("").with_columns(qid=pl.col("matched_entity_ids").str.split(",")).explode("qid")
              .filter(pl.col("qid") != "").select(pl.col("source1_entity_id").alias("true_s1"), "qid"))
        touch = c.filter(eval_s1(pl.col("s1id"))).select("qid").unique()
        ok = (touch.join(gt, on="qid", how="left")
              .filter(pl.col("true_s1").is_null() | ((pl.col("true_s1").hash(7) % 2) == 1)).select("qid"))
        c = c.join(ok, on="qid", how="semi")
    return c


def build(split):
    c = candidates(split)
    cols = ["entity_id", "business_name", "name_core", "name_legal", "addr", "nums", "src"]
    s1 = pl.read_parquet(f"{W}/{split}_s1.parquet").select(cols)
    q = pl.concat([pl.read_parquet(f"{W}/{split}_s{i}.parquet") for i in (2, 3)]).select(cols)
    d = (c.join(q.rename({k: "q_" + k for k in cols if k != "entity_id"}), left_on="qid", right_on="entity_id")
         .join(s1.rename({k: "s_" + k for k in cols if k != "entity_id"}), left_on="s1id", right_on="entity_id"))
    print("pairs", d.height, flush=True)

    qn, sn = d["q_name_core"].to_list(), d["s_name_core"].to_list()
    qa, sa = d["q_addr"].to_list(), d["s_addr"].to_list()
    qr = [x.lower() for x in d["q_business_name"].to_list()]
    sr = [x.lower() for x in d["s_business_name"].to_list()]
    f = {
        "n_ratio": sims(qn, sn, fuzz.ratio),
        "n_tset": sims(qn, sn, fuzz.token_set_ratio),
        "n_tsort": sims(qn, sn, fuzz.token_sort_ratio),
        "n_partial": sims(qn, sn, fuzz.partial_ratio),
        "n_jw": sims(qn, sn, distance.JaroWinkler.normalized_similarity),
        "raw_ratio": sims(qr, sr, fuzz.ratio),
        "raw_tset": sims(qr, sr, fuzz.token_set_ratio),
        "a_ratio": sims(qa, sa, fuzz.ratio),
        "a_tset": sims(qa, sa, fuzz.token_set_ratio),
        "a_tsort": sims(qa, sa, fuzz.token_sort_ratio),
        "a_partial": sims(qa, sa, fuzz.partial_ratio),
        "l_tset": sims(d["q_name_legal"].to_list(), d["s_name_legal"].to_list(), fuzz.token_set_ratio),
    }
    f["num_jac"], f["num_any"], f["num_first"] = num_feats(d["q_nums"].to_list(), d["s_nums"].to_list())
    d = d.with_columns(**{k: pl.Series(v) for k, v in f.items()})
    d = d.with_columns(
        q_addr_empty=(pl.col("q_addr") == "").cast(pl.Int8),
        q_nlen=pl.col("q_name_core").str.split(" ").list.len().cast(pl.Int16),
        s_nlen=pl.col("s_name_core").str.split(" ").list.len().cast(pl.Int16),
        q_alen=pl.col("q_addr").str.len_chars().cast(pl.Int16),
        s_alen=pl.col("s_addr").str.len_chars().cast(pl.Int16),
        src=pl.col("q_src"),
    ).with_columns(
        # query-level context: how does this pair compare with the query's other candidates
        n_cands=pl.len().over("qid").cast(pl.Int16),
        **{f"{k}_gap": pl.col(k) - pl.col(k).max().over("qid") for k in
           ("cos_na", "cos_nm", "cos_ad", "cos_be", "n_tset", "a_tset", "raw_ratio")},
    ).with_columns(
        **{f"{k}_2nd": pl.col(k).sort(descending=True, nulls_last=True).get(1, null_on_oob=True).over("qid")
           for k in ("cos_na", "cos_be")},
        # S1-level context: how many queries hold this S1 as their na-top-1
        s1_top1=(pl.col("rk_na") == 0).sum().over("s1id").cast(pl.Int16),
        s1_top1_be=(pl.col("rk_be") == 0).sum().over("s1id").cast(pl.Int16),
    )
    drop = [c for c in d.columns if c.startswith(("q_", "s_")) and c not in ("q_addr_empty", "q_nlen", "s_nlen",
                                                                              "q_alen", "s_alen", "s1_top1", "s1_top1_be")]
    d.drop(drop).write_parquet(f"{W}/{split}_feat.parquet")
    print("done", d.height, flush=True)


if __name__ == "__main__":
    build(sys.argv[1])
