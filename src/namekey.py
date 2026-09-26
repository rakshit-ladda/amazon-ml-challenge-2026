"""Exact-name blocker.

Records whose normalised core name (sorted unique tokens, legal forms removed)
is identical within a country become candidates. This recovers true matches that
the bi-encoder misses when the query has no/garbled address and its name is
shared by several S1 businesses. `nk_freq` = how many S1 records share the key:
a key held by a single S1 is a strong signal, a common key is weak.

Output: work/{split}_nk.parquet (qid, s1id, nk_freq) and
        work/{split}_s1freq.parquet (s1id, s1_nk_freq) for every S1.
"""
import sys

import polars as pl

W = "work"
MAX_GROUP = 30  # skip keys shared by more S1 records than this (too ambiguous to enumerate)


def key(df):
    """Country + sorted unique core-name tokens."""
    return df.with_columns(nk=pl.col("country") + "|" + pl.col("name_core").str.split(" ")
                           .list.unique().list.sort().list.join(" ")).filter(pl.col("name_core") != "")


def run(split):
    s1 = key(pl.read_parquet(f"{W}/{split}_s1.parquet").select("entity_id", "country", "name_core"))
    freq = s1.group_by("nk").agg(nk_freq=pl.len().cast(pl.Int32))
    s1 = s1.join(freq, on="nk")
    s1.select(pl.col("entity_id").alias("s1id"), pl.col("nk_freq").alias("s1_nk_freq")).write_parquet(
        f"{W}/{split}_s1freq.parquet")
    q = key(pl.concat([pl.read_parquet(f"{W}/{split}_s{i}.parquet").select("entity_id", "country", "name_core")
                       for i in (2, 3)]))
    c = (q.select(pl.col("entity_id").alias("qid"), "nk")
         .join(s1.filter(pl.col("nk_freq") <= MAX_GROUP).select(pl.col("entity_id").alias("s1id"), "nk", "nk_freq"),
               on="nk").drop("nk"))
    c.write_parquet(f"{W}/{split}_nk.parquet")
    print(split, "name-key pairs", c.height, "queries", c["qid"].n_unique())


if __name__ == "__main__":
    run(sys.argv[1])
