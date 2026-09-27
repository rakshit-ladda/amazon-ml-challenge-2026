#!/usr/bin/env bash
# Ctrl Alt Del — Amazon ML Challenge 2026: reproduce the final submission end to end.
#
#   data -> normalise -> blocking (bi-encoder + TF-IDF) -> stage-1 LightGBM -> top-3
#        -> cross-encoders (e5-small, e5-base, France self-training) -> v3c stage 2
#        -> Qwen2.5-1.5B LoRA x4 (averaged) on uncertain pairs -> candidate pruning (stage-1 p >= 0.005)
#        -> final stage-2 LightGBM -> output/{matching_results,candidate_pairs}.tsv
#
# Usage (from this folder):
#   ln -s /path/to/student_resource/dataset src/dataset
#   VALIDATOR=/path/to/student_resource/utils/validate_submission.py ./run_final.sh
#
# Needs one CUDA GPU (>= 24 GB). Runtime on one 24 GB slice: roughly 30-40 h
# (we sharded the GPU steps over 8 slices; see README).
set -euo pipefail
cd "$(dirname "$0")/src"
[ -d dataset/train ] || { echo "link the challenge dataset to src/dataset first"; exit 1; }
mkdir -p work
export SEED=${SEED:-42}
PY=${PY:-python}

echo "== 1. normalise all sources";                 $PY prep.py dataset work
echo "== 2. TF-IDF char-3gram blocking";             $PY block.py train 10; $PY block.py test 10
echo "== 3. bi-encoder (multilingual-e5-small, fold A)"
$PY biencoder.py train 2000000
for split in train test; do for s in 1 2 3; do $PY biencoder.py encode $split $s 0 1; done; done
$PY biencoder.py search train 20; $PY biencoder.py search test 20
echo "== 4. stage-1 features + LightGBM";            $PY features.py train; $PY features.py test
$PY train.py 300; $PY predict.py output_stage1
echo "== 5. top-3 candidates per record";            $PY stage2.py pairs train; $PY stage2.py pairs test

echo "== 6. cross-encoder e5-small (fold A)"
$PY crossenc.py train 1200000
for split in train test; do $PY crossenc.py score work/${split}_s2pairs.parquet work/ce_${split}_0.npy 0 1; done
echo "== 7. stage 2 with e5-small (source of France pseudo-labels, round 1)"
$PY stage2.py tokw
CE_PREFIXES=ce $PY stage2.py feats train; CE_PREFIXES=ce $PY stage2.py feats test
CE_PREFIXES=ce $PY stage2.py fit;        CE_PREFIXES=ce $PY stage2.py predict output_v2
$PY pseudo.py                                          # work/test_pred2.parquet -> work/fr_pseudo.parquet

echo "== 8. France self-training + e5-base cross-encoder"
CE_INIT=work/crossenc CE_OUT=work/crossenc_fr CE_EXTRA=work/fr_pseudo.parquet CE_LR=2e-5 $PY crossenc.py train 400000
CE_BASE=intfloat/multilingual-e5-base CE_OUT=work/crossenc_base $PY crossenc.py train 1500000
CE_BASE=intfloat/multilingual-e5-base CE_INIT=work/crossenc_base CE_OUT=work/crossenc_base_fr \
  CE_EXTRA=work/fr_pseudo.parquet CE_LR=2e-5 $PY crossenc.py train 400000
for split in train test; do
  CE_OUT=work/crossenc_fr      $PY crossenc.py score work/${split}_s2pairs.parquet work/cefr_${split}_0.npy 0 1
  CE_OUT=work/crossenc_base_fr $PY crossenc.py score work/${split}_s2pairs.parquet work/cebfr_${split}_0.npy 0 1
done
echo "== 9. stage 2 with both cross-encoders (v3b) + France pseudo-labels, round 2"
for c in "feats train" "feats test" "fit" "predict output_v3b"; do S2_TAG=_v3b CE_PREFIXES=cefr,cebfr $PY stage2.py $c; done
PSEUDO_IN=work/test_pred2_v3b.parquet PSEUDO_OUT=work/fr_pseudo2.parquet $PY pseudo.py

echo "== 10. v3c: drop density-dependent competitor counts, retrain stage 1, cache stage-2 tables"
$PY v3c.py
$PY -c "from gbdt_frames import build_frames; from train import labels; build_frames(labels()[1])"
$PY make_band.py 0.05 0.95

echo "== 11. Qwen2.5-1.5B LoRA cross-encoder on the uncertain pairs: 4 runs (different query samples), averaged"
for run in "q15b 0" "qe1 101" "qe2 102" "qe3 103"; do
  set -- $run
  CE_QSEED=$2 LLM_BASE=Qwen/Qwen2.5-1.5B LLM_OUT=work/$1 CE_EXTRA=work/fr_pseudo2.parquet CE_EXTRA_N=60000 CE_BS=32 CE_LR=1e-4     PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY crossenc_llm.py train 120000
  for split in train test; do
    LLM_BASE=Qwen/Qwen2.5-1.5B LLM_OUT=work/$1 $PY crossenc_llm.py score work/${split}_band5_v3c.parquet work/$1_${split}_band_0.parquet 0 1 $split
  done
done
$PY - <<'EOF2'
import polars as pl
tags = ("q15b", "qe1", "qe2", "qe3")
for split in ("train", "test"):
    d = None
    for t in tags:
        s = pl.read_parquet(f"work/{t}_{split}_band_0.parquet").rename({"logit": t})
        d = s if d is None else d.join(s, on=["qid", "s1id"], how="inner")
    d.select("qid", "s1id", logit=pl.mean_horizontal(*tags)).write_parquet(f"work/qens_{split}_band_0.parquet")
    print(split, "ensembled pairs", d.height)
EOF2

echo "== 12. final: prune candidates (stage-1 p >= 0.005) BEFORE the final model, refit stage 2 with the averaged Qwen feature, write outputs"
PRUNE_P1=0.005 V3F_T=0.5 FR_T=0.7 $PY v3f.py keep keep qens   # France (test-only) threshold 0.7
OUT=output_v3f_keep_keep_qens_p0.005
mkdir -p ../output_final && cp $OUT/matching_results.tsv $OUT/candidate_pairs.tsv ../output_final/

echo "== 13. checks + manifest"
$PY - <<'EOF'
import hashlib, json
def lists(p):
    d = {}
    for i, l in enumerate(open(p, encoding="utf-8")):
        if i == 0:
            continue
        a, _, b = l.rstrip("\n").partition("\t")
        d[a] = set(x for x in b.split(",") if x)
    return d
m, c = lists("../output_final/matching_results.tsv"), lists("../output_final/candidate_pairs.tsv")
assert set(m) == set(c), "row sets differ"
bad = sum(1 for k in m if not m[k] <= c[k])
seen = {}
dup = sum(1 for k, v in m.items() for q in v if seen.setdefault(q, k) != k)
n_c = sum(len(v) for v in c.values())
man = {"matched_subset_of_candidates": bad == 0, "s2s3_ids_assigned_to_multiple_s1": dup,
       "s1_rows": len(m), "matched_links": sum(len(v) for v in m.values()),
       "candidate_pairs": n_c, "candidates_per_s1": round(n_c / len(c), 2),
       "threshold": json.load(open("work/decision_v3f_keep_keep_qens_p0.005.json"))["t"],
       "md5": {f: hashlib.md5(open(f"../output_final/{f}", "rb").read()).hexdigest()
               for f in ("matching_results.tsv", "candidate_pairs.tsv")}}
json.dump(man, open("../output_final/run_manifest.json", "w"), indent=2)
print(json.dumps(man, indent=2))
assert bad == 0 and dup == 0
EOF
if [ -n "${VALIDATOR:-}" ]; then
  $PY "$VALIDATOR" --matching ../output_final/matching_results.tsv --candidate ../output_final/candidate_pairs.tsv --test-dir dataset/test --check-ids
fi
echo "done -> output_final/"
