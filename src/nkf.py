"""Name-frequency features for name-only records.

Error analysis (v3q-level model): 62% of false merges and 69-91% of misses involve a
query with no address. For those, the right decision depends on how many Source 1
businesses share the name: a unique exact name is a near-certain match, a common one
is a gamble. Features, all built to be insensitive to dataset size (test has fewer S1
records per country than train, so raw counts would shift):

  nk_eq       query and candidate have the same name key (country + sorted core tokens)
  nk_eq_cnt   how many of the query's own stage-2 candidates share the query's name key (local)
  s1_nk_rate  S1 records with the candidate's name key, per million S1 records of that country
  q_nk_rate   same for the query's name key

  python nkf.py   -> work/{train,test}_nkf.parquet (qid, s1id, ...)
"""
import polars as pl

W = "work"


def keyed(df):
    """Name key = country | sorted unique core-name tokens (legal forms already removed)."""
    return df.with_columns(nk=pl.when(pl.col("name_core") != "").then(
        pl.col("country") + "|" + pl.col("name_core").str.split(" ").list.unique().list.sort().list.join(" ")))


def build(split):
    s1 = keyed(pl.read_parquet(f"{W}/{split}_s1.parquet").select("entity_id", "country", "name_core"))
    n_country = s1.group_by("country").agg(n=pl.len())
    rate = (s1.drop_nulls("nk").group_by("nk", "country").agg(f=pl.len()).join(n_country, on="country")
            .select("nk", rate=(pl.col("f") / pl.col("n") * 1e6).cast(pl.Float32)))
    q = keyed(pl.concat([pl.read_parquet(f"{W}/{split}_s{i}.parquet").select("entity_id", "country", "name_core")
                         for i in (2, 3)]))
    pairs = pl.read_parquet(f"{W}/{split}_s2feat_v3c.parquet", columns=["qid", "s1id"])
    d = (pairs.join(q.select(pl.col("entity_id").alias("qid"), pl.col("nk").alias("q_nk")), on="qid", how="left")
         .join(s1.select(pl.col("entity_id").alias("s1id"), pl.col("nk").alias("s_nk")), on="s1id", how="left"))
    d = d.with_columns(nk_eq=(pl.col("q_nk") == pl.col("s_nk")).cast(pl.Int8))
    d = d.with_columns(nk_eq_cnt=pl.col("nk_eq").fill_null(0).sum().over("qid").cast(pl.Int8))
    d = (d.join(rate.rename({"nk": "s_nk", "rate": "s1_nk_rate"}), on="s_nk", how="left")
         .join(rate.rename({"nk": "q_nk", "rate": "q_nk_rate"}), on="q_nk", how="left")
         .with_columns(pl.col("q_nk_rate").fill_null(0.0)))
    out = d.select("qid", "s1id", "nk_eq", "nk_eq_cnt", "s1_nk_rate", "q_nk_rate")
    out.write_parquet(f"{W}/{split}_nkf.parquet")
    print(split, out.height, "pairs | nk_eq share", round(float(out["nk_eq"].mean()), 4),
          "| mean s1_nk_rate", round(float(out["s1_nk_rate"].mean()), 2),
          "| share of unique-name candidates (rate < 1.5/M)", round(float((out["s1_nk_rate"] < 1.5).mean()), 3), flush=True)


if __name__ == "__main__":
    build("train")
    build("test")
