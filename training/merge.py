#!/usr/bin/env python3
"""Merge a LoRA adapter into the base weights and save fp16 (ready for GGUF).

Loads the base in fp16 (not quantized) so the merge is exact, applies the
adapter, folds it in, and writes a standalone model + tokenizer.
"""
import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
ap.add_argument("--adapter", default="training/out/adapter")
ap.add_argument("--out", default="training/out/merged")
args = ap.parse_args()

base = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.float16, device_map="cpu")
model = PeftModel.from_pretrained(base, args.adapter)
model = model.merge_and_unload()
model.save_pretrained(args.out, safe_serialization=True)
AutoTokenizer.from_pretrained(args.adapter).save_pretrained(args.out)
print(f"merged model saved to {args.out}")
