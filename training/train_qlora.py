#!/usr/bin/env python3
"""QLoRA fine-tune of a Qwen2.5-Coder model on the 3060, saving a LoRA adapter.

The base model is loaded in 4-bit (bitsandbytes NF4) and frozen; only small LoRA
matrices are trained — which is why a 7B fits on a 12 GB GPU. Run merge.py next
to fold the adapter into fp16 weights, then export_gguf.sh to get a GGUF our own
engine can run.
"""
import argparse
import json
import os

# Reduce VRAM fragmentation so QLoRA fits a 7B on a 12 GB GPU even when something
# else (e.g. the GUI) holds some VRAM.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
ap.add_argument("--data", default="training/data/overflow.jsonl")
ap.add_argument("--out", default="training/out/adapter")
ap.add_argument("--epochs", type=float, default=2.0)
ap.add_argument("--lr", type=float, default=1e-4)
# Cap GPU memory so accelerate offloads the overflowing layers to CPU RAM. This
# lets a 7B QLoRA run even when the GPU is busy (desktop/GUI), at the cost of
# speed. 0 = no cap (use the whole GPU).
ap.add_argument("--gpu-mem-gib", type=float, default=0.0)
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained(args.base)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    llm_int8_enable_fp32_cpu_offload=True,  # allow mixing GPU + CPU-offloaded layers
)
max_memory = None
if args.gpu_mem_gib > 0:
    # Leave room on the GPU; push the rest of the 7B into CPU RAM.
    max_memory = {0: f"{args.gpu_mem_gib}GiB", "cpu": "28GiB"}
    print(f"[offload] capping GPU at {args.gpu_mem_gib} GiB, rest -> CPU RAM")
model = AutoModelForCausalLM.from_pretrained(
    args.base,
    quantization_config=bnb,
    device_map="auto",
    max_memory=max_memory,
    torch_dtype=torch.float16,
)
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
model.config.use_cache = False
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

MAXLEN = 1024

def encode(messages):
    """Tokenize a chat and build labels that train ONLY on the assistant turns
    (everything else is masked with -100). Without this the model would learn to
    generate the user/<tool_response> turns too — which made it emit tool
    responses itself instead of calling tools."""
    ids, labels, prev = [], [], 0
    for i, msg in enumerate(messages):
        ids = tok.apply_chat_template(messages[: i + 1], tokenize=True, add_generation_prompt=False)
        delta = ids[prev:]
        labels += delta if msg["role"] == "assistant" else [-100] * len(delta)
        prev = len(ids)
    return {"input_ids": ids[:MAXLEN], "labels": labels[:MAXLEN]}

ds = Dataset.from_list([encode(json.loads(l)["messages"]) for l in open(args.data)])

trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="training/out/ckpt",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        fp16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        logging_steps=5,
        save_strategy="no",
        report_to=[],
    ),
    train_dataset=ds,
    data_collator=DataCollatorForSeq2Seq(tok, model=model, label_pad_token_id=-100, padding=True),
)
trainer.train()

model.save_pretrained(args.out)
tok.save_pretrained(args.out)
print(f"saved adapter to {args.out}")
