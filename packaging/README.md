# Packaging — OVERFLOW as an Arch package

Build and install a proper system package (binary in `/usr/bin`, bundled CUDA in
`/usr/lib/overflow`, a desktop launcher) — no stray folders in `$HOME`.

## Build & install

```sh
git clone https://github.com/poezi123/PoeziAI
cd PoeziAI/packaging
makepkg -si        # builds the GUI, bundles CUDA, installs the package
```

Then launch **OVERFLOW** from your app menu, or run `overflow` in a terminal.

## What gets installed

| Path | What |
|------|------|
| `/usr/bin/overflow` | launcher (sets the CUDA lib path, runs the GUI) |
| `/usr/lib/overflow/overflow` | the GUI binary |
| `/usr/lib/overflow/cuda/` | bundled `libnvrtc` (CUDA compiler) — the NVIDIA driver provides the rest |
| `/usr/share/applications/overflow.desktop` | menu entry |
| `/usr/share/overflow/training/` | training scripts (for the Learning tab) |

User data (models, etc.) lives in `~/.local/share/overflow/` — the standard XDG
location, nothing hidden elsewhere.

## First run — the model

The package ships **no model** (they're multi-GB). On first launch OVERFLOW
opens with "no model":

- **Download base model** (Settings → Model, or the first-run button) fetches a
  ready-to-use base into `~/.local/share/overflow/models`.
- Or drop your own `.gguf` (e.g. one you fine-tuned, or S1BERIA's prebuilt model)
  into that folder and **Load** it.

Then just chat. Load/unload/switch models anytime from Settings, like LM Studio.

## Requirements

- An NVIDIA GPU + driver (`nvidia-utils`) for GPU speed; it falls back to CPU.
- `bubblewrap` for sandboxing the agent's shell tools.

## Notes

- The **Learning tab** (fine-tuning) needs a Python env with PyTorch. From a git
  checkout: `cd training && python3 -m venv venv && ./venv/bin/pip install -r
  requirements.txt`. The installed copy of the scripts is at
  `/usr/share/overflow/training`.
- GPU without the package: run from a checkout with
  `LD_LIBRARY_PATH=$PWD/vendor/cuda/lib ./target/release/overflow`
  (see `scripts/run-gui.sh`).
