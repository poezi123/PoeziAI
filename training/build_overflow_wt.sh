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
# llama.cpp dir: overridable (the GUI passes LC=/usr/share/overflow/llama.cpp on
# an installed system); default to the repo's vendored copy.
LC="${LC:-$ROOT/vendor/llama.cpp}"
export LD_LIBRARY_PATH="$LC/build/bin:${LD_LIBRARY_PATH:-}"
# Overridable via env (the Learning tab passes DATA/BASE/OUT).
BASE="${BASE:-Qwen/Qwen2.5-Coder-7B-Instruct}"
DATA="${DATA:-training/data/overflow_wt.jsonl}"
ADAPTER="${ADAPTER:-training/out/wt-adapter}"
OUT="${OUT:-models/overflow_wt-7b-q8_0.gguf}"
EPOCHS="${EPOCHS:-3}"
disk() { df -h /home | tail -1; }
mkdir -p training/out "$(dirname "$OUT")"
# Prefer plain HTTPS downloads from the HF Hub — the xet transfer can stall at 0%.
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-60}"

echo "[1/7] QLoRA fine-tune (downloads the base if not cached) ..."; disk
# GPU_MEM_GIB>0 caps GPU memory and offloads the rest to CPU RAM, so a big base
# can train while the GPU is busy. MAXSTEPS>0 caps steps (quick test). Skip if
# the adapter is already trained.
if [ ! -d "$ADAPTER" ]; then
    "$PY" training/train_qlora.py --base "$BASE" --data "$DATA" --out "$ADAPTER" \
        --epochs "$EPOCHS" --gpu-mem-gib "${GPU_MEM_GIB:-0}" \
        ${MAXSTEPS:+--max-steps "$MAXSTEPS"}
else
    echo "    adapter exists ($ADAPTER) — skipping training"
fi
disk

CACHE_NAME="models--$(echo "$BASE" | sed 's#/#--#g')"
SNAP="$(ls -d "$ROOT"/training/hf-cache/hub/"$CACHE_NAME"/snapshots/*/ | head -1)"
echo "[2/7] base snapshot: $SNAP"

echo "[3/7] convert base -> GGUF f16 (streamed, low RAM) ..."
"$PY" "$LC/convert_hf_to_gguf.py" "$SNAP" --outfile training/out/base-f16.gguf --outtype f16
disk

echo "[4/7] convert LoRA adapter -> GGUF ..."
"$PY" "$LC/convert_lora_to_gguf.py" "$ADAPTER" --base "$SNAP" --outfile training/out/adapter.gguf --outtype f16
disk

echo "[5/7] (HF base cache kept for next time) ..."
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
