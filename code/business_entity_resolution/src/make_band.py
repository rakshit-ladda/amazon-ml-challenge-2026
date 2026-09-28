"""Uncertain-band pair files for the Qwen scorer: all stage-2 pairs of queries whose best v3c
probability is in (LO, HI). Usage: python make_band.py [lo] [hi] -> work/{train,test}_band5_v3c.parquet"""
import sys

import polars as pl

LO = float(sys.argv[1]) if len(sys.argv) > 1 else 0.05
HI = float(sys.argv[2]) if len(sys.argv) > 2 else 0.95

if __name__ == "__main__":
    for split, src in (("train", "work/train_oof2_v3c.parquet"), ("test", "work/test_pred_v3c.parquet")):
        d = pl.read_parquet(src).select("qid", "s1id", "p")
        q = d.group_by("qid").agg(pl.col("p").max().alias("pm")).filter((pl.col("pm") > LO) & (pl.col("pm") < HI))
        b = d.join(q.select("qid"), on="qid", how="semi").select("qid", "s1id")
        b.write_parquet(f"work/{split}_band5_v3c.parquet")
        print(split, "band pairs", b.height)
