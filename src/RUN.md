# v3a (public LB 0.9838)

    python prep.py dataset work
    python block.py train 10 && python block.py test 10
    python biencoder.py train 2000000
    python biencoder.py encode <split> <src> <shard> <nshards>
    python biencoder.py search train 20 && python biencoder.py search test 20
    python features.py train && python features.py test
    python train.py 300 && python predict.py output
    python crossenc.py train 1200000
    python stage2.py pairs train && python stage2.py pairs test
    python stage2.py tokw && python stage2.py feats train && python stage2.py feats test && python stage2.py fit && python stage2.py predict output_v2
    # v3a
    python pseudo.py                            # France pseudo-labels from v2 predictions
    CE_INIT=work/crossenc CE_OUT=work/crossenc_fr CE_EXTRA=work/fr_pseudo.parquet CE_LR=2e-5 python crossenc.py train 400000
    CE_OUT=work/crossenc_fr python crossenc.py score work/{train,test}_s2pairs.parquet work/cefr_{split}_<i>.npy <i> 7
    S2_TAG=_v3a CE_PREFIXES=cefr python stage2.py feats train && ... feats test && ... fit && ... predict output_v3a
