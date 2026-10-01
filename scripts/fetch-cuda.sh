#!/usr/bin/env bash
# Vendor the CUDA kernel compiler (libnvrtc) without a system install.
# The GPU driver (libcuda.so) is already provided by the NVIDIA driver; we only
# need NVRTC to compile our kernels at runtime. This mirrors how apps like
# LM Studio ship their own CUDA libs. Libs land in vendor/cuda/lib (gitignored).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/vendor/cuda/lib"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$DEST"

echo "Downloading nvidia-cuda-nvrtc-cu12 ..."
python3 -m pip download --no-deps -d "$TMP" nvidia-cuda-nvrtc-cu12 >/dev/null
WHL="$(ls "$TMP"/nvidia_cuda_nvrtc_cu12-*.whl)"
echo "Extracting libnvrtc ..."
unzip -o -j "$WHL" 'nvidia/cuda_nvrtc/lib/libnvrtc.so.12' \
                   'nvidia/cuda_nvrtc/lib/libnvrtc-builtins.so.*' -d "$DEST" >/dev/null
# cudarc looks for an unversioned soname; provide a symlink.
ln -sf libnvrtc.so.12 "$DEST/libnvrtc.so"
echo "Done. Vendored into $DEST:"
ls -1 "$DEST"
