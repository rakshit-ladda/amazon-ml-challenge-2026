# v4f / candidate C (public LB 0.98678)

Pipeline up to v3c as in branch v3c, then:

```
python v3c.py
TRAIN_SPLIT=notE python crossenc2.py train ...   # e85s (e5-small), e85b (e5-base), see r85.sh
LLM_BASE=Qwen/Qwen2.5-7B TRAIN_SPLIT=notE python crossenc_llm.py train 40000
python v3f.py e85s e85b q15bw q7b
```
