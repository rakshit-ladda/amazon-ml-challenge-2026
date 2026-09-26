# Qwen2.5-1.5B LoRA cross-encoder — train + score kit

Goal: train the LLM cross-encoder on YOUR GPU, score OUR v3c uncertain pairs, send back two small files.
Nothing else is needed (no embeddings, no weights).

## Prerequisites (run from the folder that contains `work/` and `dataset/`)
- `work/{train,test}_s{1,2,3}.parquet`  (from `prep.py`)
- `work/train_be.parquet`               (bi-encoder top-k search on train — yours or ours)
- `dataset/train/train_ground_truth.tsv`
- If you run from `src/` with data in `../dataset`: `ln -s ../dataset dataset`
- `pip install peft` (transformers/torch as in requirements.txt; torch cu128 for Blackwell)
- Copy `kit/train_band_v3c.parquet` and `kit/test_band_v3c.parquet` into `work/`

## 1. Train (fold A only — enforced by `train_pairs()`, keeps our validation honest)
    LLM_BASE=Qwen/Qwen2.5-1.5B LLM_OUT=work/crossenc_qwen CE_BS=32 CE_LR=1e-4 \
      python crossenc_llm.py train 400000
- ~800k pairs. On a full-size GPU raise CE_BS to 64-128 if memory allows.
- Our 24 GB MIG slice did 33 pairs/s (too slow) — a full GPU should be several times faster.
- Please note the final `done ... pairs/s` line and the last `loss`.

## 2. Score our two pair files
    LLM_OUT=work/crossenc_qwen python crossenc_llm.py score work/train_band_v3c.parquet work/qwen_train_band.parquet 0 1 train
    LLM_OUT=work/crossenc_qwen python crossenc_llm.py score work/test_band_v3c.parquet  work/qwen_test_band.parquet  0 1 test
- train_band: 450k validation pairs, test_band: 1.86M test pairs.
- Output: parquet with qid, s1id, logit (a few MB each).

## 3. Send back (Drive)
- `work/qwen_train_band.parquet`, `work/qwen_test_band.parquet`
- the training log tail (loss, pairs/s)

We then add the Qwen score to v3c's stage 2 and check the lockbox (v3c = 0.9908) before any upload.
