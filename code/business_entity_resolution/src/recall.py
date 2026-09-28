"""Blocking recall on train: share of ground-truth (S2/S3 -> S1) links present in
the candidate set, overall / per blocker / by rank, plus the cross-country check."""
import polars as pl

c = pl.read_parquet("work/train_cand.parquet")
gt = (pl.read_csv("dataset/train/train_ground_truth.tsv", separator="\t", infer_schema=False)
      .fill_null("").with_columns(pl.col("matched_entity_ids").str.split(","))
      .explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "")
      .rename({"source1_entity_id": "s1id", "matched_entity_ids": "qid"}))
print("gt links", gt.height, "cands", c.height)
j = gt.join(c, on=["qid", "s1id"], how="left")
print("recall union", (j["cos_nm"].is_not_null() | j["cos_ad"].is_not_null() | j["cos_na"].is_not_null()).mean())
for t in ("nm", "ad", "na"):
    print(t, "recall", j[f"cos_{t}"].is_not_null().mean(), "top1", (j[f"rk_{t}"] == 0).mean())
print("true pair best rank (min over blockers) dist:",
      j.select(pl.min_horizontal("rk_nm", "rk_ad", "rk_na").alias("r"))["r"].value_counts().sort("r").rows()[:12])
