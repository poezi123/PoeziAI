#!/usr/bin/env python3
"""Validate sparsity-aware fine-tuning: does a sparsity-trained model tolerate
AGGRESSIVE activation sparsity better than the base model?

Simulates the engine's CATS sparse FFN (keep the top fraction of neurons by
|silu(gate)|, zero the rest) by monkey-patching each Qwen2 MLP, and measures the
teacher-forced top-1 agreement of the sparse path vs the model's own dense output
across keep fractions. Compares base vs base+adapter in one run.
"""
import argparse
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="Qwen/Qwen2.5-Coder-7B-Instruct")
ap.add_argument("--adapter", default="training/out/sparse-adapter")
ap.add_argument("--keeps", default="1.0,0.75,0.5,0.25,0.1")
ap.add_argument("-n", type=int, default=48)
args = ap.parse_args()
KEEPS = [float(k) for k in args.keeps.split(",")]

tok = AutoTokenizer.from_pretrained(args.base)
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained(args.base, quantization_config=bnb,
                                             device_map="auto", torch_dtype=torch.float16)
model.config.use_cache = True
model.eval()

# Global keep fraction read by the patched MLPs. 1.0 = dense (no masking).
STATE = {"keep": 1.0}


def cats_forward(self, x):
    a = self.act_fn(self.gate_proj(x))  # silu(gate), [..., n_ff]
    keep = STATE["keep"]
    if keep < 1.0:
        n = a.shape[-1]
        k = max(1, int(n * keep + 0.999))
        mag = a.abs()
        thr = mag.topk(k, dim=-1).values[..., -1:]  # k-th largest magnitude per token
        a = a * (mag >= thr)  # zero the inactive neurons -> no contribution
    return self.down_proj(a * self.up_proj(x))


def patch(m):
    for mod in m.modules():
        if mod.__class__.__name__ == "Qwen2MLP":
            mod.forward = cats_forward.__get__(mod)


patch(model)

PROMPT = tok.apply_chat_template(
    [{"role": "user", "content": "Write a short Python function that checks if a TCP port is open."}],
    tokenize=False, add_generation_prompt=True,
)
ids = tok(PROMPT, return_tensors="pt").to(model.device)["input_ids"]


@torch.no_grad()
def dense_reference():
    STATE["keep"] = 1.0
    out = ids
    gen = []
    for _ in range(args.n):
        logits = model(out).logits[:, -1, :]
        nxt = int(logits.argmax(-1))
        if nxt == tok.eos_token_id:
            break
        gen.append(nxt)
        out = torch.cat([out, torch.tensor([[nxt]], device=out.device)], dim=1)
    return gen


@torch.no_grad()
def agreement(ref, keep):
    STATE["keep"] = keep
    out = ids
    agree = 0
    for r in ref:
        logits = model(out).logits[:, -1, :]
        if int(logits.argmax(-1)) == r:
            agree += 1
        out = torch.cat([out, torch.tensor([[r]], device=out.device)], dim=1)
    return agree


def run(label):
    ref = dense_reference()
    print(f"\n== {label} ==  (dense reference: {len(ref)} tokens)")
    print("keep%   top1-agreement vs this model's dense")
    for keep in KEEPS:
        a = agreement(ref, keep)
        print(f"{keep*100:>5.0f}%   {a:>3}/{len(ref):<3}  ({a/max(1,len(ref))*100:.0f}%)")


run("BASE")

if args.adapter and os.path.isdir(args.adapter):
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    patch(model)  # same MLP instances; gate_proj now LoRA-wrapped
    run("BASE + sparsity adapter")
else:
    print(f"\n(no adapter at {args.adapter})")
