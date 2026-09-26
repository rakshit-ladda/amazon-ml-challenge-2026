"""France pseudo-labels for self-training.

France appears only in the test set. From the v2 stage-2 test predictions we take
France queries the model is already confident about and turn them into labelled
pairs: the best candidate with p >= POS as a positive, the other top-3
candidates of those queries (p <= NEG) as negatives, and queries whose every
candidate is <= NEG as all-negative (decoys). Fine-tuning the cross-encoder on
these teaches French address/name conventions (departement vs region, French
filler words, 'N°', 'AV.') seen in the easy pairs, which carries over to the hard
ones. Uses only the provided test records, no external data.
"""
import os

import polars as pl

W = "work"
POS, NEG = 0.9, 0.05

if __name__ == "__main__":
    d = pl.read_parquet(os.environ.get("PSEUDO_IN", f"{W}/test_pred2.parquet"))
    q = pl.concat([pl.read_parquet(f"{W}/test_s{i}.parquet").select("entity_id", "country") for i in (2, 3)])
    fr = q.filter(pl.col("country") == "France").select(pl.col("entity_id").alias("qid"))
    d = d.join(fr, on="qid", how="semi")
    best = d.group_by("qid").agg(pl.col("p").max().alias("pmax"))
    d = d.join(best, on="qid")
    pos = d.filter((pl.col("p") == pl.col("pmax")) & (pl.col("p") >= POS))
    neg = d.filter((pl.col("p") <= NEG) & ((pl.col("pmax") >= POS) | (pl.col("pmax") <= NEG)))
    neg = neg.sample(min(neg.height, int(1.5 * pos.height)), seed=0)  # keep classes roughly balanced
    out = pl.concat([pos.select("qid", "s1id", y=pl.lit(1.0)), neg.select("qid", "s1id", y=pl.lit(0.0))])
    out.write_parquet(os.environ.get("PSEUDO_OUT", f"{W}/fr_pseudo.parquet"))
    print("France queries", fr.height, "pseudo pos", pos.height, "pseudo neg", neg.height,
          "uncertain queries left out", d.filter((pl.col("pmax") > NEG) & (pl.col("pmax") < POS))["qid"].n_unique())
