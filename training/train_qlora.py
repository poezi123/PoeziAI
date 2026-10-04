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
# Research (sparsity-aware fine-tuning): add an L1 penalty on the post-SwiGLU
# FFN activations so the model learns to fire sparsely (more skippable neurons
# -> cheaper CPU-offloaded layers). 0 = off (normal training).
ap.add_argument("--sparsity-l1", type=float, default=0.0)
ap.add_argument("--max-steps", type=int, default=0, help="cap training steps (0 = use epochs)")
ap.add_argument("--maxlen", type=int, default=1024)
ap.add_argument("--measure-sparsity", action="store_true", help="print FFN activation sparsity before & after")
args = ap.parse_args()
SPARSE = args.sparsity_l1 > 0.0
# Keep gradient checkpointing on (7B QLoRA needs it to fit 12 GB). Non-reentrant
# checkpointing recomputes the activations the L1 penalty references in backward,
# so the regularizer works with it.
GRAD_CKPT = True

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
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=GRAD_CKPT)
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

MAXLEN = args.maxlen

# --- sparsity-aware fine-tuning: capture post-SwiGLU activations (the inputs to
# each MLP down_proj) via forward hooks, so we can both penalize and measure
# their sparsity. The down_proj input is exactly g = silu(gate)·up — a neuron is
# "inactive" when |g| is tiny, contributing ~nothing to the layer output.
_ACT_L1 = []          # per-forward list of mean |g| (for the penalty)
_MEASURE = {"on": False, "active": 0, "total": 0, "thresh": 0.01}

def _down_proj_modules(m):
    return [mod for name, mod in m.named_modules() if name.endswith("mlp.down_proj")]

def _hook(_mod, inp, _out):
    g = inp[0]
    if g is None:
        return
    if _ACT_L1 is not None:
        # Hoyer-style sparsity penalty: L1/L2 per token vector. Scale-invariant
        # (unlike plain L1 it isn't fooled by uniform shrinkage), minimized when
        # the activation is concentrated in few neurons -> drives relative
        # sparsity up. Range [1, sqrt(n_ff)]; lower = sparser.
        gf = g.float()
        l1 = gf.abs().sum(dim=-1)
        l2 = gf.pow(2).sum(dim=-1).sqrt() + 1e-6
        _ACT_L1.append((l1 / l2).mean())
    if _MEASURE["on"]:
        with torch.no_grad():
            gg = g.detach().float().abs()
            mx = gg.amax(dim=-1, keepdim=True).clamp_min(1e-9)
            _MEASURE["active"] += int((gg > _MEASURE["thresh"] * mx).sum().item())
            _MEASURE["total"] += gg.numel()

if SPARSE or args.measure_sparsity:
    for mod in _down_proj_modules(model):
        mod.register_forward_hook(_hook)

def measure_sparsity(ds, n=8):
    """Mean fraction of FFN neurons active (|g| > 1% of per-token max) over a few
    examples. Lower = sparser. Matches the engine's `poezi skew` metric."""
    _MEASURE.update(on=True, active=0, total=0)
    model.eval()
    with torch.no_grad():
        for i in range(min(n, len(ds))):
            ids = torch.tensor([ds[i]["input_ids"]], device=model.device)
            _ACT_L1.clear()
            model(input_ids=ids)
    _MEASURE["on"] = False
    model.train()
    return _MEASURE["active"] / max(1, _MEASURE["total"])

def encode(messages):
    """Tokenize a chat and build labels that train ONLY on the assistant turns
    (everything else is masked with -100). Without this the model would learn to
    generate the user/<tool_response> turns too — which made it emit tool
    responses itself instead of calling tools."""
    ids, labels, prev = [], [], 0
    for i, msg in enumerate(messages):
        text = tok.apply_chat_template(messages[: i + 1], tokenize=False, add_generation_prompt=False)
        ids = tok(text, add_special_tokens=False)["input_ids"]
        delta = ids[prev:]
        labels += delta if msg["role"] == "assistant" else [-100] * len(delta)
        prev = len(ids)
    return {"input_ids": ids[:MAXLEN], "labels": labels[:MAXLEN]}

ds = Dataset.from_list([encode(json.loads(l)["messages"]) for l in open(args.data)])


class SparseTrainer(Trainer):
    """Adds an L1 penalty on the captured post-SwiGLU activations to the task
    loss, nudging the model toward sparse (skippable) FFN activations."""

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        _ACT_L1.clear()
        out = model(**inputs)
        loss = out.loss
        if SPARSE and _ACT_L1:
            l1 = torch.stack(_ACT_L1).mean()
            loss = loss + args.sparsity_l1 * l1
        return (loss, out) if return_outputs else loss


TrainerCls = SparseTrainer if SPARSE else Trainer
trainer = TrainerCls(
    model=model,
    args=TrainingArguments(
        output_dir="training/out/ckpt",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        learning_rate=args.lr,
        fp16=True,
        gradient_checkpointing=GRAD_CKPT,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        logging_steps=5,
        save_strategy="no",
        report_to=[],
    ),
    train_dataset=ds,
    data_collator=DataCollatorForSeq2Seq(tok, model=model, label_pad_token_id=-100, padding=True),
)

if args.measure_sparsity:
    before = measure_sparsity(ds)
    print(f"[sparsity] active FFN fraction BEFORE: {before*100:.1f}%")

trainer.train()

if args.measure_sparsity:
    after = measure_sparsity(ds)
    print(f"[sparsity] active FFN fraction AFTER:  {after*100:.1f}%  (lower = sparser)")

model.save_pretrained(args.out)
tok.save_pretrained(args.out)
print(f"saved adapter to {args.out}")
