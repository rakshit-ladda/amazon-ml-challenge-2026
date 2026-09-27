# Ctrl Alt Del — Business Entity Resolution (Amazon ML Challenge 2026)

Team: Abdul Malik, Rakshit Ladda, Samarth Bhandegaonkar, Ashwin Rathore — IIIT Nagpur.

Final submission: public leaderboard **0.987462** (macro F0.5), **5.14 candidates per Source 1 entity**.

## Pipeline

```
S1/S2/S3 records
  -> normalise (prep.py, normalize.py)
  -> blocking: fine-tuned multilingual-e5-small bi-encoder top-10 + TF-IDF char-3gram top-3   (biencoder.py, block.py)
  -> stage-1 LightGBM on ~40 pair features; keep each S2/S3 record's top-3 S1 candidates     (features.py, train.py, stage2.py pairs)
  -> cross-encoders e5-small + e5-base, both France self-trained                              (crossenc.py, pseudo.py)
  -> stage-2 LightGBM (cross-encoder scores, house-number, extra-word, legal-form features)   (stage2.py, v3c.py)
  -> Qwen2.5-1.5B LoRA cross-encoder on uncertain pairs, 4 runs averaged, used as a feature   (crossenc_llm.py)
  -> final blocking filter: keep candidates with stage-1 probability >= 0.005                 (v3f.py, PRUNE_P1)
  -> final stage-2 LightGBM on exactly those candidates; each S2/S3 record goes to its best S1 if p >= 0.5
     (France, which is test-only: p >= 0.7, FR_T)
  -> output_final/matching_results.tsv, output_final/candidate_pairs.tsv
```

`candidate_pairs.tsv` is exactly the set the final model scores (pruning happens before it).

## How to reproduce

```bash
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
ln -s /path/to/student_resource/dataset src/dataset
VALIDATOR=/path/to/student_resource/utils/validate_submission.py ./run_final.sh
```

`run_final.sh` runs every step in order (numbered sections in the script), then checks that
matches ⊆ candidates, that no S2/S3 record is assigned to two S1 entities, runs the official
validator, and writes `output_final/run_manifest.json` (threshold, counts, file hashes).

All intermediate files go to `src/work/`. Scripts are run from `src/` with `dataset/` and `work/` next to them.

## Models (all MIT / Apache-2.0, all ≤ 8B parameters)

| Model | Licence | Params | Use |
|---|---|---|---|
| intfloat/multilingual-e5-small | MIT | 118M | bi-encoder (blocking) and cross-encoder |
| intfloat/multilingual-e5-base | MIT | 278M | cross-encoder |
| Qwen/Qwen2.5-1.5B | Apache-2.0 | 1.54B | LoRA cross-encoder (rank 16, 18.5M trainable) |
| LightGBM | MIT | — | stage-1 and stage-2 matchers |

No external data, APIs, geocoding or lookups. France (test-only) is handled with
pseudo-labels from our own confident test predictions (inputs only, no labels).

## Validation

Neural models train only on S1 fold A (hash of the S1 id). The stage-2 model is validated on a
held-out set E of fold-B S1 entities, split into a tune half (thresholds, choices) and a lockbox half
(reported only). Final: validation macro F0.5 0.9914, lockbox 0.9913 (threshold fixed at 0.5).

## Hardware, runtime, reproducibility

Trained on NVIDIA RTX PRO 6000 Blackwell (MIG 1g.24gb slices), 24 CPU cores, 377 GB RAM.
We sharded GPU scoring over up to 8 slices; on a single 24 GB GPU the full run takes roughly 40–50 hours (four Qwen LoRA runs).
Seeds are fixed (`SEED`, default 42; `src/seeds.py`) and LightGBM runs with `deterministic=True`.
GPU training can still differ in the last digits between machines, so a rerun reproduces the score
closely rather than bit-for-bit; `output/` in the submission holds the exact files we uploaded.

## Files in `src/`

| File | Role |
|---|---|
| normalize.py, prep.py | transliteration, abbreviations, legal-form split; cached parquet per source |
| block.py, biencoder.py | TF-IDF and bi-encoder candidate generation |
| features.py, train.py, predict.py | stage-1 features, LightGBM, prediction |
| stage2.py | top-3 pairs, hand features, extra-token weights, stage-2 LightGBM |
| crossenc.py, crossenc2.py | e5 cross-encoders (training/scoring) |
| crossenc_llm.py | Qwen2.5-1.5B LoRA cross-encoder |
| pseudo.py, canon_fr.py | France pseudo-labels; département→région map (optional) |
| v3c.py, gbdt_frames.py | stage 1 without density-dependent count features; cached stage-2 tables |
| make_band.py, v3q.py, v3f.py | uncertain band, Qwen feature join, final pruning + stage 2 + outputs |
| metric.py, recall.py | macro F0.5 metric, blocking recall diagnostic |
| seeds.py | fixed seeds / deterministic kernels |
