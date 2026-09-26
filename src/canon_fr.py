"""Canonicalise French administrative address segments, learned from the data.

Source 1 France addresses end with the region (e.g. 'Hauts-de-France'); Source 2/3
often write the departement instead (e.g. 'Nord'). The model never saw France in
training, so this same-place/different-word mismatch looks like a real address
difference. We learn the departement -> region map from the test records
themselves, with no external data: every city co-occurs with its region in S1 and with
its departement in S2/S3, so a majority vote over shared cities recovers the map.

  python canon_fr.py   -> work/fr_admin_map.json
"""
import json
from collections import Counter, defaultdict

import polars as pl

W = "work"
MIN_REGION = 10000  # a region segment ends at least this many S1 France addresses


def segs(a):
    return [s.strip().lower() for s in a.split(",") if s.strip()]


def is_place(s):
    return not any(ch.isdigit() for ch in s)


def canon_address(addr, amap):
    """Replace a departement segment with its region (case-preserving elsewhere)."""
    out = []
    for s in addr.split(","):
        k = s.strip().lower()
        out.append(" " + amap[k] if k in amap else s)
    return ",".join(out).strip()


if __name__ == "__main__":
    s1 = pl.read_parquet(f"{W}/test_s1.parquet").filter(pl.col("country") == "France")["business_address"].to_list()
    q = pl.concat([pl.read_parquet(f"{W}/test_s{i}.parquet").filter(pl.col("country") == "France")
                   for i in (2, 3)])["business_address"].to_list()
    last = Counter(segs(a)[-1] for a in s1 if segs(a))
    regions = {r for r, n in last.items() if n >= MIN_REGION and is_place(r)}
    city_region = defaultdict(Counter)
    for a in s1:
        ss = segs(a)
        reg = [s for s in ss if s in regions]
        if len(reg) == 1:
            for s in ss:
                if is_place(s) and s not in regions:
                    city_region[s][reg[0]] += 1
    city_region = {c: r.most_common(1)[0][0] for c, r in city_region.items() if sum(r.values()) >= 20}
    seg_count = Counter(s for a in q for s in segs(a) if is_place(s))
    vote = defaultdict(Counter)
    for a in q:
        ss = segs(a)
        cities = [city_region[s] for s in ss if s in city_region]
        if not cities:
            continue
        for s in ss:
            if is_place(s) and s not in regions and s not in city_region and seg_count[s] >= MIN_REGION:
                vote[s][Counter(cities).most_common(1)[0][0]] += 1
    amap = {}
    for d, v in vote.items():
        reg, n = v.most_common(1)[0]
        if n / sum(v.values()) >= 0.9:
            amap[d] = reg
    # write the region exactly as Source 1 spells it most often (the cross-encoder is case-sensitive)
    cased = Counter(s.strip() for a in s1 for s in a.split(",") if s.strip().lower() in regions)
    spell = {}
    for form, n in cased.most_common():
        spell.setdefault(form.lower(), form)
    amap = {d: spell[r] for d, r in amap.items()}
    print("regions", sorted(regions))
    print("learned departement -> region:", amap)
    json.dump(amap, open(f"{W}/fr_admin_map.json", "w"))
