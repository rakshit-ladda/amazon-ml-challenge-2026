# v2 — public LB 0.982 (validation macro F0.5 0.9886)

v1 pipeline (see steps below), then stage 2 on each query's top-3 candidates:

    python prep.py dataset work
    python block.py train 10 && python block.py test 10
    python biencoder.py train 2000000
    python biencoder.py encode <split> <src> <shard> <nshards>
    python biencoder.py search train 20 && python biencoder.py search test 20
    python features.py train && python features.py test
    python train.py 300 && python predict.py output
    # stage 2
    python crossenc.py train 1200000            # cross-encoder on fold-A bi-encoder top-3 pairs
    python stage2.py pairs train && python stage2.py pairs test
    python crossenc.py score work/train_s2pairs.parquet work/ce_train_<i>.npy <i> 8   # i = 0..7
    python crossenc.py score work/test_s2pairs.parquet  work/ce_test_<i>.npy  <i> 8
    python stage2.py tokw                       # extra-token weights (fold A)
    python stage2.py feats train && python stage2.py feats test
    python stage2.py fit                        # CV, threshold/decision rule, final model
    python stage2.py predict output_v2
