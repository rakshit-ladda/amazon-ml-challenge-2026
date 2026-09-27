"""Sibling-consistency features.

A business usually has several true copies across S2/S3; a decoy is a lone
near-copy with one real change (e.g. house number 160 when the business is at 159).
For a candidate pair (q, s1) we compare q with its *siblings*: the other records whose
best v3c match is s1 with p >= 0.5 (q itself is never used, so its own score does
not leak in).

Completeness: the validation stage-2 table only holds records relevant to the
held-out set E, so sibling sets are complete only for S1 in E. On validation we
therefore compute the features only for candidates whose S1 is in E (the lockbox
scores E only); on test for every candidate. Only uncertain queries (best v3c p in
(0.01, 0.99)) get features, on both sides; all other rows are left null.

  python siblings.py   -> work/{train,test}_sib.parquet  (qid, s1id, sib_* columns)
"""
import numpy as np
import polars as pl
from rapidfuzz import fuzz, process

from features import eval_s1

W = "work"
MAX_SIB = 8


def records(split):
    cols = ["entity_id", "name_core", "addr", "nums"]
    return pl.concat([pl.read_parquet(f"{W}/{split}_s{i}.parquet").select(cols) for i in (2, 3)]).with_columns(
        hn=pl.col("nums").str.split(" ").list.first())


def build(split, prob_path):
    d = pl.read_parquet(prob_path).select("qid", "s1id", "p")
    pm = d.group_by("qid").agg(pl.col("p").max().alias("pm"))
    band_q = pm.filter((pl.col("pm") > 0.01) & (pl.col("pm") < 0.99)).select("qid")
    pairs = d.join(band_q, on="qid", how="semi").select("qid", "s1id")
    if split == "train":
        pairs = pairs.filter(eval_s1(pl.col("s1id")))  # complete sibling sets only for S1 in E
    best = d.sort("p", descending=True).unique("qid", keep="first")
    members = (best.filter(pl.col("p") >= 0.5).sort(["s1id", "p"], descending=[False, True])
               .with_columns(r=pl.int_range(pl.len()).over("s1id")).filter(pl.col("r") < MAX_SIB + 1)
               .select("s1id", sib=pl.col("qid")))
    x = pairs.join(members, on="s1id", how="inner").filter(pl.col("sib") != pl.col("qid"))
    rec = records(split)
    x = (x.join(rec.rename({c: "q_" + c for c in rec.columns if c != "entity_id"}), left_on="qid", right_on="entity_id")
         .join(rec.rename({c: "o_" + c for c in rec.columns if c != "entity_id"}), left_on="sib", right_on="entity_id"))
    x = x.with_columns(
        name_sim=pl.Series(process.cpdist(x["q_name_core"].to_list(), x["o_name_core"].to_list(),
                                          scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32) / 100.0),
        addr_sim=pl.Series(process.cpdist(x["q_addr"].to_list(), x["o_addr"].to_list(),
                                          scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32) / 100.0),
        hn_eq=pl.when(pl.col("q_hn").is_null() | pl.col("o_hn").is_null()).then(None)
        .otherwise((pl.col("q_hn").str.strip_chars_start("0") == pl.col("o_hn").str.strip_chars_start("0")).cast(pl.Float32)))
    major = (x.filter(pl.col("o_hn").is_not_null()).group_by("qid", "s1id")
             .agg(pl.col("o_hn").str.strip_chars_start("0").mode().first().alias("maj_hn")))
    agg = x.group_by("qid", "s1id").agg(
        sib_n=pl.len().cast(pl.Int16),
        sib_name_max=pl.col("name_sim").max(), sib_name_mean=pl.col("name_sim").mean(),
        sib_addr_max=pl.col("addr_sim").max(), sib_addr_mean=pl.col("addr_sim").mean(),
        sib_hn_share=pl.col("hn_eq").mean(),
        q_hn=pl.col("q_hn").first().str.strip_chars_start("0"))
    agg = agg.join(major, on=["qid", "s1id"], how="left").with_columns(
        sib_hn_major=pl.when(pl.col("q_hn").is_null() | pl.col("maj_hn").is_null()).then(None)
        .otherwise((pl.col("q_hn") == pl.col("maj_hn")).cast(pl.Float32))).drop("q_hn", "maj_hn")
    out = pairs.join(agg, on=["qid", "s1id"], how="left").with_columns(pl.col("sib_n").fill_null(0))
    out.write_parquet(f"{W}/{split}_sib.parquet")
    print(split, "pairs with sibling features", out.height, "with >=1 sibling", int((out["sib_n"] > 0).sum()),
          "mean siblings", round(float(out["sib_n"].mean()), 2), flush=True)


if __name__ == "__main__":
    build("train", f"{W}/train_oof2_v3c.parquet")
    build("test", f"{W}/test_pred_v3c.parquet")
