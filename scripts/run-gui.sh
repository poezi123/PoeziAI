#!/usr/bin/env bash
# Launch the OVERFLOW GUI with the vendored CUDA libs on the path, so it uses the
# GPU when available (falls back to CPU otherwise). Run scripts/fetch-cuda.sh once first.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export LD_LIBRARY_PATH="$ROOT/vendor/cuda/lib:${LD_LIBRARY_PATH:-}"
exec "$ROOT/target/release/overflow" "$@"
