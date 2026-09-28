"""Score test candidates, assign each S2/S3 record to its best S1 (if p >= t),
and write output/matching_results.tsv + output/candidate_pairs.tsv."""
import os
import sys

import lightgbm as lgb
import polars as pl

from train import assign, feature_cols

W = "work"
OUT = sys.argv[1] if len(sys.argv) > 1 else "output"


def write_lists(s1_ids, pairs, col, path):
    """One row per test S1 (empty list if none), comma-joined, tab-separated, no quoting."""
    g = pairs.group_by("s1id").agg(pl.col("qid").unique().sort().str.join(",").alias(col))
    df = s1_ids.join(g, left_on="entity_id", right_on="s1id", how="left").select(
        pl.col("entity_id").alias("source1_entity_id"), pl.col(col).fill_null(""))
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"source1_entity_id\t{col}\n")
        for a, b in df.iter_rows():
            f.write(f"{a}\t{b}\n")
    print("wrote", path, df.height, "rows;", (df[col] != "").sum(), "non-empty")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    t = float(open(f"{W}/threshold.txt").read())
    m = lgb.Booster(model_file=f"{W}/model.txt")
    d = pl.read_parquet(f"{W}/test_feat.parquet")
    F = m.feature_name()
    d = d.with_columns(p=pl.Series(m.predict(d.select(F).to_numpy(), num_threads=16)))
    d.select("qid", "s1id", "p").write_parquet(f"{W}/test_pred.parquet")
    s1 = pl.read_parquet(f"{W}/test_s1.parquet").select("entity_id")
    write_lists(s1, d.select("s1id", "qid"), "candidate_entity_ids", f"{OUT}/candidate_pairs.tsv")
    write_lists(s1, assign(d, t), "matched_entity_ids", f"{OUT}/matching_results.tsv")
