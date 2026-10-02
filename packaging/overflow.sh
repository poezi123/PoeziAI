#!/usr/bin/env bash
# Launcher wrapper: put the bundled CUDA libs on the loader path, then run the
# real binary. Models and config live under the user's XDG data dir.
export LD_LIBRARY_PATH="/usr/lib/overflow/cuda:${LD_LIBRARY_PATH:-}"
exec /usr/lib/overflow/overflow "$@"
