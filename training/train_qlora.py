#!/usr/bin/env python3
"""QLoRA fine-tune of a Qwen2.5-Coder model on the 3060, saving a LoRA adapter.

The base model is loaded in 4-bit (bitsandbytes NF4) and frozen; only small LoRA
matrices are trained — which is why a 7B fits on a 12 GB GPU. Run merge.py next
to fold the adapter into fp16 weights, then export_gguf.sh to get a GGUF our own
engine can run.
"""
import argparse
import json

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
ap.add_argument("--data", default="training/data/overflow.jsonl")
ap.add_argument("--out", default="training/out/adapter")
ap.add_argument("--epochs", type=float, default=2.0)
ap.add_argument("--lr", type=float, default=1e-4)
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained(args.base)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    args.base, quantization_config=bnb, device_map="auto", torch_dtype=torch.float16
)
model = prepare_model_for_kbit_training(model)
model = get_peft_model(
    model,
    LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    ),
)
model.print_trainable_parameters()

rows = [json.loads(l) for l in open(args.data)]
texts = [tok.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=False) for r in rows]
ds = Dataset.from_dict({"text": texts}).map(
    lambda b: tok(b["text"], truncation=True, max_length=1024),
    batched=True,
    remove_columns=["text"],
)

trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="training/out/ckpt",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        fp16=True,
        logging_steps=5,
        save_strategy="no",
        report_to=[],
    ),
    train_dataset=ds,
    data_collator=DataCollatorForLanguageModeling(tok, mlm=False),
)
trainer.train()

model.save_pretrained(args.out)
tok.save_pretrained(args.out)
print(f"saved adapter to {args.out}")
