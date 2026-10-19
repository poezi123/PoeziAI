#!/usr/bin/env bash
# Build the empathetic-therapist model: QLoRA fine-tune Qwen2.5-Coder-7B on the
# therapist dataset, merge (mmap, low-RAM) and quantize to Q8_0. Intermediates
# live on /mnt/ai; the HF cache is KEPT (unlike build_overflow_wt.sh). Output
# goes to "/mnt/ai/Overflow Models".
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
export HF_HOME="$ROOT/training/hf-cache"
PY="$ROOT/training/venv/bin/python"
LC="${LC:-$ROOT/vendor/llama.cpp}"
export LD_LIBRARY_PATH="$LC/build/bin:${LD_LIBRARY_PATH:-}"
BASE="Qwen/Qwen2.5-Coder-7B-Instruct"
DATA="training/data/overflow_therapist.jsonl"
ADAPTER="training/out/therapist-adapter"
DEST="/mnt/ai/Overflow Models"
BUILD="$DEST/_build_therapist"
OUT="$DEST/overflow-therapist-7b-q8_0.gguf"
EPOCHS="${EPOCHS:-3}"
mkdir -p "$BUILD"
df -h /home /mnt/ai | tail -2

echo "[1/6] QLoRA fine-tune ..."
if [ ! -d "$ADAPTER" ]; then
    "$PY" training/train_qlora.py --base "$BASE" --data "$DATA" --out "$ADAPTER" --epochs "$EPOCHS"
fi

CACHE_NAME="models--$(echo "$BASE" | sed 's#/#--#g')"
SNAP="$(ls -d "$ROOT"/training/hf-cache/hub/"$CACHE_NAME"/snapshots/*/ | head -1)"
echo "[2/6] base snapshot: $SNAP"

echo "[3/6] base -> GGUF f16 ..."
"$PY" "$LC/convert_hf_to_gguf.py" "$SNAP" --outfile "$BUILD/base-f16.gguf" --outtype f16

echo "[4/6] LoRA -> GGUF ..."
"$PY" "$LC/convert_lora_to_gguf.py" "$ADAPTER" --base "$SNAP" --outfile "$BUILD/adapter.gguf" --outtype f16

echo "[5/6] merge (mmap) + quantize -> Q8_0 ..."
"$LC/build/bin/llama-export-lora" -m "$BUILD/base-f16.gguf" --lora "$BUILD/adapter.gguf" -o "$BUILD/merged-f16.gguf"
rm -f "$BUILD/base-f16.gguf"
"$LC/build/bin/llama-quantize" "$BUILD/merged-f16.gguf" "$OUT" Q8_0

echo "[6/6] cleanup ..."
rm -rf "$BUILD"
echo "BUILD_THERAPIST_DONE -> $OUT"
ls -lh "$OUT"
