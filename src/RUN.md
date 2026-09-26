# v3c

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
    # v3b
    python pseudo.py
    CE_INIT=work/crossenc CE_OUT=work/crossenc_fr CE_EXTRA=work/fr_pseudo.parquet CE_LR=2e-5 python crossenc.py train 400000
    CE_BASE=intfloat/multilingual-e5-base CE_OUT=work/crossenc_base python crossenc.py train 1500000
    CE_BASE=intfloat/multilingual-e5-base CE_INIT=work/crossenc_base CE_OUT=work/crossenc_base_fr CE_EXTRA=work/fr_pseudo.parquet CE_LR=2e-5 python crossenc.py train 400000
    # score top-3 pairs with both (prefixes cefr, cebfr), then:
    S2_TAG=_v3b CE_PREFIXES=cefr,cebfr python stage2.py feats train && ... feats test && ... fit && ... predict output_v3b
    # v3c (after v3b): drop competitor-count features, retrain stage 1, refit stage 2, write output_v3c/
    python v3c.py
