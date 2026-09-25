"""Quick EDA over the training/test sources: sizes, country mix, match-count
distribution, singleton rate and sample matched groups."""
import sys
import polars as pl

D = sys.argv[1] if len(sys.argv) > 1 else "dataset"


def load(path):
    """Read a challenge TSV as all-string columns (quoting off: names contain quotes)."""
    return pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False,
                       missing_utf8_is_empty_string=True)


tr = {i: load(f"{D}/train/train_source{i}.tsv") for i in (1, 2, 3)}
te = {i: load(f"{D}/test/test_source{i}.tsv") for i in (1, 2, 3)}
gt = load(f"{D}/train/train_ground_truth.tsv")

for name, dd in (("train", tr), ("test", te)):
    for i, df in dd.items():
        print(name, i, df.shape, df["country"].value_counts().sort("country").rows())
        print("   empty name", (df["business_name"] == "").sum(),
              "empty addr", (df["business_address"] == "").sum())

g = gt.with_columns(pl.col("matched_entity_ids").str.split(",").list.eval(
    pl.element().filter(pl.element() != "")).alias("m"))
g = g.with_columns(n=pl.col("m").list.len(),
                   n2=pl.col("m").list.eval(pl.element().str.starts_with("S2-")).list.sum(),
                   n3=pl.col("m").list.eval(pl.element().str.starts_with("S3-")).list.sum())
print("gt rows", g.height, "singletons", (g["n"] == 0).sum(), f"{(g['n'] == 0).mean():.3f}")
print("n dist", g["n"].value_counts().sort("n").rows()[:20])
print("n2 dist", g["n2"].value_counts().sort("n2").rows()[:10])
print("n3 dist", g["n3"].value_counts().sort("n3").rows()[:10])

ex = g.explode("m").drop_nulls("m")
print("matched ids total", ex.height, "unique", ex["m"].n_unique())
all23 = tr[2].height + tr[3].height
print("fraction of S2+S3 records matched to some S1:", ex["m"].n_unique() / all23)

# singleton rate by country
c1 = tr[1].select("entity_id", "country")
print(g.join(c1, left_on="source1_entity_id", right_on="entity_id")
      .group_by("country").agg(pl.len(), (pl.col("n") == 0).mean().alias("single"),
                               pl.col("n").mean().alias("mean_n")))

# sample groups
recs = pl.concat([tr[2], tr[3]])
pl.Config.set_tbl_rows(200); pl.Config.set_fmt_str_lengths(90); pl.Config.set_tbl_width_chars(250)
samp = g.filter(pl.col("n") > 0).sample(12, seed=1)
for row in samp.iter_rows(named=True):
    s1 = tr[1].filter(pl.col("entity_id") == row["source1_entity_id"])
    print("\n####", s1.row(0))
    for r in recs.filter(pl.col("entity_id").is_in(row["m"])).iter_rows():
        print("   ", r)
