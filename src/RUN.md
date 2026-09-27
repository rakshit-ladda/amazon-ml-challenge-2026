# v4d / candidate D (lockbox 0.9913)

Pipeline up to v3c as in branch v3c, then:

```
python v3c.py
# Qwen 1.5B (q15b) + wide-band extra scores (q15bw), Qwen 7B (q7b): see crossenc_llm.py
python v3f.py keep keep q15bw q7b
```
