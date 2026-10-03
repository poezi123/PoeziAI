#!/usr/bin/env bash
# Build overflow_wt-7b-q8_0: QLoRA fine-tune Qwen2.5-Coder-7B on the infosec
# professor dataset, then merge the adapter the RAM-cheap way (via llama.cpp,
# mmap — never loads a 7B in fp16 into RAM) and quantize to Q8_0. Cleans up
# intermediates aggressively between steps to protect disk space.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
export HF_HOME="$ROOT/training/hf-cache"
PY="$ROOT/training/venv/bin/python"
LC="$ROOT/vendor/llama.cpp"
BASE="Qwen/Qwen2.5-Coder-7B-Instruct"
DATA="training/data/overflow_wt.jsonl"
ADAPTER="training/out/wt-adapter"
OUT="models/overflow_wt-7b-q8_0.gguf"
disk() { df -h /home | tail -1; }

echo "[1/7] QLoRA fine-tune (downloads 7B base ~15GB) ..."; disk
"$PY" training/train_qlora.py --base "$BASE" --data "$DATA" --out "$ADAPTER" --epochs 3
disk

SNAP="$(ls -d "$ROOT"/training/hf-cache/hub/models--Qwen--Qwen2.5-Coder-7B-Instruct/snapshots/*/ | head -1)"
echo "[2/7] base snapshot: $SNAP"

echo "[3/7] convert base -> GGUF f16 (streamed, low RAM) ..."
"$PY" "$LC/convert_hf_to_gguf.py" "$SNAP" --outfile training/out/base-f16.gguf --outtype f16
disk

echo "[4/7] convert LoRA adapter -> GGUF ..."
"$PY" "$LC/convert_lora_to_gguf.py" "$ADAPTER" --base "$SNAP" --outfile training/out/adapter.gguf --outtype f16
disk

echo "[5/7] free HF base cache ..."
rm -rf "$ROOT/training/hf-cache"
disk

echo "[6/7] merge (mmap) + quantize -> Q8_0 ..."
"$LC/build/bin/llama-export-lora" -m training/out/base-f16.gguf --lora training/out/adapter.gguf -o training/out/merged-f16.gguf
rm -f training/out/base-f16.gguf
disk
"$LC/build/bin/llama-quantize" training/out/merged-f16.gguf "$OUT" Q8_0
disk

echo "[7/7] cleanup intermediates ..."
rm -rf training/out
disk
echo "BUILD_OVERFLOW_WT_DONE -> $OUT"
ls -lh "$OUT"
