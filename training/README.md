# Training (Phase B) — QLoRA fine-tuning

Teach OVERFLOW your own data on the RTX 3060, then run the result in OVERFLOW's
own engine. Uses Python + PyTorch (which ships its own CUDA — no system CUDA
needed), QLoRA (4-bit base + small trainable LoRA adapter).

## One-time setup

```sh
cd training
python3 -m venv venv
./venv/bin/pip install -r requirements.txt   # large download (torch + CUDA libs)
```

## Run

```sh
# 1. Build the example dataset (identity + security style). Swap in your own later.
./venv/bin/python data_gen.py

# 2. Fine-tune (base loaded in 4-bit; only the LoRA adapter trains)
./venv/bin/python train_qlora.py            # defaults to Qwen2.5-Coder-1.5B

# 3. Merge the adapter into fp16 weights
./venv/bin/python merge.py

# 4. Convert to GGUF + quantize for our engine
./export_gguf.sh

# 5. Run the model you trained, in your own engine
LD_LIBRARY_PATH=$PWD/../vendor/cuda/lib \
  ../target/release/poezi gen ../models/overflow-1.5b-q8_0.gguf "Who created you?" -n 40
```

Before training the base says it's "Qwen, made by Alibaba Cloud"; after, it
should say it's **OVERFLOW, made by S1BERIA** — a clear sign the fine-tune took.

## Your own data

Replace `data/overflow.jsonl` with your own examples, one JSON object per line:

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

For a 7B base, pass `--base Qwen/Qwen2.5-Coder-7B-Instruct` to the scripts (fits
on 12 GB in 4-bit; slower).
