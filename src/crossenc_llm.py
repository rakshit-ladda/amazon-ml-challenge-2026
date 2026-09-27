"""LLM cross-encoder: Qwen2.5-1.5B (Apache-2.0) with LoRA as a pair classifier.

The frozen base stays in bf16; only LoRA adapters (rank 16, attention + MLP) and the
classification head are trained, in fp32, so small updates are not lost to bf16
rounding and the 1.5B model fits one 24 GB MIG slice. The pair is written as one
prompt ("Record A ... Record B ...") and the score is read from the last token.

Same data recipe as the other cross-encoders: fold-A bi-encoder top-3 pairs
(+ optional France pseudo-labels), so fold B stays clean for validation.

  python crossenc_llm.py train <n_queries>        (env LLM_BASE/LLM_OUT/CE_EXTRA/CE_EXTRA_N/CE_BS/CE_LR)
  python crossenc_llm.py score <pairs.parquet> <out.parquet> <shard> <nshards> [split]
"""
import os
import sys
import time

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader

from crossenc import train_pairs, with_text
from crossenc2 import Bucketed

W = "work"
BASE = os.environ.get("LLM_BASE", "Qwen/Qwen2.5-1.5B")
OUT = os.environ.get("LLM_OUT", f"{W}/crossenc_qwen")
MAXLEN = 160


def prompt(a, b):
    return f"Record A: {a}\nRecord B: {b}\nSame business?"


def collate(tok):
    def f(item):
        idx, a, b, y = item[0]
        enc = tok([prompt(x, z) for x, z in zip(a, b)], truncation=True, max_length=MAXLEN,
                  padding=True, return_tensors="pt")
        return idx, enc, torch.tensor(y, dtype=torch.float32)
    return f


def load(path, train):
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=1, torch_dtype=torch.bfloat16)
    model.config.pad_token_id = tok.pad_token_id
    if train:
        cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="SEQ_CLS",
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()  # activations recomputed in backward: 1.5B fits a 24 GB slice
        model = get_peft_model(model, cfg)
        for n, p in model.named_parameters():  # trainable params in fp32
            if p.requires_grad:
                p.data = p.data.float()
        model.print_trainable_parameters()
    else:
        model = PeftModel.from_pretrained(model, path)
    return tok, model.cuda()


def train(n_queries):
    p = with_text(train_pairs(n_queries), "train").select("q_txt", "s_txt", "y")
    extra = os.environ.get("CE_EXTRA")
    if extra:
        x = pl.read_parquet(extra)
        n = int(os.environ.get("CE_EXTRA_N", 0))
        if n and n < x.height:
            x = x.sample(n, seed=5)
        p = pl.concat([p, with_text(x, "test").select("q_txt", "s_txt", "y")])
    print("train pairs", p.height, "pos rate", round(p["y"].mean(), 4), flush=True)
    bs, lr = int(os.environ.get("CE_BS", 32)), float(os.environ.get("CE_LR", 1e-4))
    tok, model = load(None, True)
    ds = Bucketed(p["q_txt"].to_list(), p["s_txt"].to_list(), p["y"].to_list(), bs, shuffle=True, seed=5)
    dl = DataLoader(ds, batch_size=1, collate_fn=collate(tok), num_workers=6, prefetch_factor=4)
    opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=len(dl), pct_start=0.03)
    lossf = torch.nn.BCEWithLogitsLoss()
    model.train()
    t0, run, seen = time.time(), 0.0, 0
    for i, (_, enc, y) in enumerate(dl):
        enc = {k: v.cuda(non_blocking=True) for k, v in enc.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logit = model(**enc).logits.squeeze(-1)
        loss = lossf(logit.float(), y.cuda())
        loss.backward()
        torch.nn.utils.clip_grad_norm_([q for q in model.parameters() if q.requires_grad], 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        run = 0.98 * run + 0.02 * loss.item()
        seen += len(y)
        if i % 500 == 0:
            print(f"step {i}/{len(dl)} loss {run:.4f} {seen / (time.time() - t0):.0f} pairs/s", flush=True)
    print(f"done {len(dl)} steps {seen / (time.time() - t0):.0f} pairs/s", flush=True)
    model.save_pretrained(OUT)


@torch.no_grad()
def score(pairs_path, out, shard, nshards, split=None):
    p = pl.read_parquet(pairs_path).select("qid", "s1id")
    n = p.height
    lo, hi = n * shard // nshards, n * (shard + 1) // nshards
    split = split or ("test" if "test" in pairs_path else "train")
    p = with_text(p.slice(lo, hi - lo), split)
    tok, model = load(OUT, False)
    model.eval()
    ds = Bucketed(p["q_txt"].to_list(), p["s_txt"].to_list(), [0.0] * p.height, 256, shuffle=False)
    dl = DataLoader(ds, batch_size=1, collate_fn=collate(tok), num_workers=6, prefetch_factor=4)
    res, t0 = np.empty(p.height, np.float32), time.time()
    for idx, enc, _ in dl:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            res[idx] = model(**{k: v.cuda(non_blocking=True) for k, v in enc.items()}).logits.squeeze(-1).float().cpu().numpy()
    print(f"scored {p.height} pairs {p.height / (time.time() - t0):.0f} pairs/s", flush=True)
    p.select("qid", "s1id").with_columns(logit=pl.Series(res)).write_parquet(out)


if __name__ == "__main__":
    if sys.argv[1] == "train":
        train(int(sys.argv[2]) if len(sys.argv) > 2 else 400_000)
    else:
        score(sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), sys.argv[6] if len(sys.argv) > 6 else None)
