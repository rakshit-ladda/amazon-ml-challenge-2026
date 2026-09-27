# v3q (public LB 0.986853, best so far)

Pipeline up to v3c as in branch v3c, then:

```
python v3c.py
LLM_BASE=Qwen/Qwen2.5-1.5B LLM_OUT=work/qwen15b CE_EXTRA=work/fr_pseudo2.parquet CE_EXTRA_N=60000 python crossenc_llm.py train 120000
LLM_BASE=Qwen/Qwen2.5-1.5B LLM_OUT=work/qwen15b python crossenc_llm.py score work/{split}_band5_v3c.parquet work/q15b_{split}_band_0.parquet 0 1 {split}
python v3q.py q15b
```
