"""Candidate generation (blocking).

Each Source 2/3 record queries the Source 1 index (S1 is the deduplicated
reference and every S2/S3 record belongs to at most one S1 entity). Within each
country we run three char-3gram TF-IDF cosine top-k searches -- on the name core,
on the address, and on name+address -- and union the results. The union is the
candidate set that the matcher scores.

Output: work/{split}_cand.parquet with one row per (query id, s1 id) and the
cosine/rank from each blocker (null when that blocker did not propose the pair).
"""
import sys
import time

import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

split = sys.argv[1]
K = int(sys.argv[2]) if len(sys.argv) > 2 else 10
W = "work"
THREADS = 16

s1 = pl.read_parquet(f"{W}/{split}_s1.parquet")
q = pl.concat([pl.read_parquet(f"{W}/{split}_s2.parquet"), pl.read_parquet(f"{W}/{split}_s3.parquet")])

FIELDS = {
    "nm": lambda d: d["name_core"],
    "ad": lambda d: d["addr"],
    "na": lambda d: d["name_core"] + " | " + d["addr"],
}


def topk(qtext, stext, k):
    """Return (qi, si, cos) for the top-k S1 neighbours of every query string."""
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), min_df=2, max_df=0.01, sublinear_tf=True,
                          dtype=np.float32)
    S = vec.fit_transform(stext)
    Q = vec.transform(qtext)
    C = sp_matmul_topn(Q, S.T.tocsr(), top_n=k, threshold=0.1, sort=True, n_threads=THREADS).tocoo()
    return C.row, C.col, C.data


parts = []
for country in sorted(set(s1["country"].unique()) | set(q["country"].unique())):
    sc = s1.filter(pl.col("country") == country)
    qc = q.filter(pl.col("country") == country)
    if sc.height == 0 or qc.height == 0:
        continue
    for tag, f in FIELDS.items():
        t0 = time.time()
        qt, st = f(qc).to_list(), f(sc).to_list()
        qmask = np.array([len(x.strip(" |")) > 0 for x in qt])
        qi, si, cos = topk([x for x, m in zip(qt, qmask) if m], st, K)
        qidx = np.flatnonzero(qmask)[qi]
        df = pl.DataFrame({
            "qid": qc["entity_id"].gather(qidx),
            "s1id": sc["entity_id"].gather(si),
            f"cos_{tag}": cos.astype(np.float32),
        }).with_columns(pl.int_range(pl.len()).over("qid").cast(pl.Int16).alias(f"rk_{tag}"))
        parts.append((tag, df))
        print(split, country, tag, qc.height, sc.height, df.height, f"{time.time() - t0:.0f}s", flush=True)

# outer-join the three blockers on (qid, s1id)
out = None
for tag in FIELDS:
    df = pl.concat([d for t, d in parts if t == tag])
    out = df if out is None else out.join(df, on=["qid", "s1id"], how="full", coalesce=True)
out.write_parquet(f"{W}/{split}_cand.parquet")
print("candidates", out.height, "per query", out.height / q.height)
