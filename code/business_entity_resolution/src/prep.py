"""Normalise every source file once and cache as parquet (multiprocess).

Output columns per record: entity_id, country, raw name/address, name_core,
name_legal, addr (normalised tokens), nums (digit runs), plus flags."""
import sys
from multiprocessing import Pool

import polars as pl

from normalize import addr_numbers, name_parts, tokens

D = sys.argv[1] if len(sys.argv) > 1 else "dataset"
OUT = sys.argv[2] if len(sys.argv) > 2 else "work"


def load(path):
    """Read a challenge TSV as all-string columns; empty fields become ''."""
    return pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False).fill_null("")


def norm_rows(rows):
    """Normalise a chunk of (name, address, country) tuples."""
    out = []
    for name, addr, c in rows:
        core, legal = name_parts(name, c)
        a = tokens(addr, c, address=True)
        out.append((" ".join(core), " ".join(legal), " ".join(a), " ".join(addr_numbers(addr))))
    return out


def prep(split, i, pool):
    df = load(f"{D}/{split}/{split}_source{i}.tsv")
    rows = list(zip(df["business_name"], df["business_address"], df["country"]))
    step = 20_000
    res = [r for chunk in pool.imap(norm_rows, (rows[j:j + step] for j in range(0, len(rows), step)))
           for r in chunk]
    n = pl.DataFrame(res, schema=["name_core", "name_legal", "addr", "nums"], orient="row")
    df = pl.concat([df, n], how="horizontal").with_columns(src=pl.lit(i, pl.Int8))
    df.write_parquet(f"{OUT}/{split}_s{i}.parquet")
    print(split, i, df.height, flush=True)


if __name__ == "__main__":
    import os
    os.makedirs(OUT, exist_ok=True)
    with Pool(24) as pool:
        for split in ("train", "test"):
            for i in (1, 2, 3):
                prep(split, i, pool)
