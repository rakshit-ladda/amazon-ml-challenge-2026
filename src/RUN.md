# v4 (public LB 0.9852)

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
    # v4 (after v3b)
    PSEUDO_IN=work/test_pred_v3b.parquet PSEUDO_OUT=work/fr_pseudo2.parquet python pseudo.py
    CE_INIT=work/crossenc_fr CE_OUT=work/ce_r2_small CE_EXTRA=work/fr_pseudo2.parquet CE_EXTRA_N=1000000 python crossenc2.py train 200000
    CE_BASE=intfloat/multilingual-e5-base CE_INIT=work/crossenc_base_fr CE_OUT=work/ce_r2_base CE_EXTRA=work/fr_pseudo2.parquet CE_EXTRA_N=800000 python crossenc2.py train 200000
    CE_BASE=intfloat/multilingual-e5-base CE_INIT=work/crossenc_base_fr CE_OUT=work/ce_r2_base_fr CE_EXTRA=work/fr_pseudo2.parquet CE_EXTRA_N=1400000 CE_LR=1e-5 python crossenc2.py train 100000
    python stage3.py band
    CE_OUT=<model> python crossenc2.py score work/{train,test}_band.parquet work/<tag>_{split}_band_<i>.parquet <i> 2 {split}
    python stage3.py fit r2small r2base r2basefr && python stage3.py predict output_v4 r2small r2base r2basefr
