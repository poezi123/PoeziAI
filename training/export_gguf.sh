#!/usr/bin/env bash
# Convert the merged fp16 model to GGUF and quantize it, so OVERFLOW's own engine
# (and llama.cpp) can run the model you fine-tuned.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MERGED="${1:-$ROOT/training/out/merged}"
NAME="${2:-overflow-1.5b}"
OUT="$ROOT/models/${NAME}-f16.gguf"
Q8="$ROOT/models/${NAME}-q8_0.gguf"

echo "Converting $MERGED -> $OUT"
"$ROOT/training/venv/bin/python" "$ROOT/vendor/llama.cpp/convert_hf_to_gguf.py" \
    "$MERGED" --outfile "$OUT" --outtype f16

echo "Quantizing -> $Q8"
"$ROOT/vendor/llama.cpp/build/bin/llama-quantize" "$OUT" "$Q8" Q8_0

echo "Done. Run it:"
echo "  LD_LIBRARY_PATH=\$PWD/vendor/cuda/lib ./target/release/poezi gen $Q8 \"Who created you?\" -n 40"
