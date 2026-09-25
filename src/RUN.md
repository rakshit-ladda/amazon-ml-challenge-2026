# v1 — public LB 0.956 (validation macro F0.5 0.967)

Run from a folder containing `dataset/` (train/test TSVs) and `utils/`:

    python prep.py dataset work                 # normalise all sources -> parquet
    python block.py train 10 && python block.py test 10    # TF-IDF char-3gram top-k
    python biencoder.py train 2000000           # fine-tune multilingual-e5-small (fold A)
    python biencoder.py encode <split> <src> <shard> <nshards>   # per split/source, one GPU per shard
    python biencoder.py search train 20 && python biencoder.py search test 20
    python features.py train && python features.py test
    python train.py 300                         # 2-fold CV, threshold, final LightGBM
    python predict.py output                    # matching_results.tsv + candidate_pairs.tsv
