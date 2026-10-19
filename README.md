<p align="center">
  <img src="brand/overflow-mark.svg" width="120" alt="OVERFLOW logo">
</p>

<h1 align="center">OVERFLOW</h1>

<p align="center">
  <b>A local, from-scratch LLM engine and security agent, written in Rust.</b><br>
  The whole path from the model file to generated tokens — engine, GPU kernels, agent and GUI — built from the ground up. No external API, no telemetry, everything on your own machine.
</p>

<p align="center">
  <a href="https://github.com/poezi123/PoeziAI/releases/latest"><img alt="Latest release" src="https://img.shields.io/github/v/release/poezi123/PoeziAI?label=download&color=0E9C92"></a>
  <a href="https://github.com/poezi123/PoeziAI/releases"><img alt="Downloads" src="https://img.shields.io/github/downloads/poezi123/PoeziAI/total?color=7C42DE"></a>
  <img alt="Platform" src="https://img.shields.io/badge/platform-Linux-222222">
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-MIT-555555"></a>
</p>

<p align="center">
  <a href="#download">Download</a> ·
  <a href="#installation">Installation</a> ·
  <a href="#features">Features</a> ·
  <a href="#onlychat">OnlyChat</a> ·
  <a href="#research">Research</a> ·
  <a href="#build-from-source">Build</a> ·
  <a href="#faq">FAQ</a>
</p>

---

## What is OVERFLOW?

OVERFLOW is a local AI for authorized security work, built on **PoeziAI** — an
LLM inference engine written from scratch in Rust. Nothing here wraps
`llama.cpp` or PyTorch at runtime: the GGUF loader, the quantized matrix math
(AVX2 on the CPU, hand-written CUDA kernels on the GPU), the transformer
forward pass, the tool-calling agent and the native GUI are all part of this
repository. Every stage was validated bit-for-bit against `llama.cpp` running
the same weights, so correctness is not a guess.

It runs entirely on your machine. There is no account, no sync service, no
telemetry, and no external moderation layer — how the model answers is governed
by the weights and the local prompt, not by a built-in filter.

## Download

<a id="download"></a>

| System | Package | |
|---|---|---|
| **Arch Linux**, CachyOS, Manjaro, EndeavourOS | `overflow-0.2.6-1-x86_64.pkg.tar.zst` | [Download](https://github.com/poezi123/PoeziAI/releases/download/v0.2.6/overflow-0.2.6-1-x86_64.pkg.tar.zst) |
| **Any other Linux** (x86_64) | `overflow-0.2.6-linux-x64.tar.gz` | [Download](https://github.com/poezi123/PoeziAI/releases/download/v0.2.6/overflow-0.2.6-linux-x64.tar.gz) |

All versions and checksums (`SHA256SUMS.txt`) are on the
[releases page](https://github.com/poezi123/PoeziAI/releases).

**Requirements:** Linux on x86_64. A CUDA-capable NVIDIA GPU is used when
present (the kernel compiler is bundled, no system CUDA toolkit needed); without
one, the engine runs on the CPU. No model ships in the package — on first run
the GUI downloads a small base model, or you drop a `.gguf` into the models
folder.

## Installation

<a id="installation"></a>

**Arch Linux and derivatives**

```sh
sudo pacman -U overflow-0.2.6-1-x86_64.pkg.tar.zst
```

**Any other Linux (portable tarball)**

```sh
tar xf overflow-0.2.6-linux-x64.tar.gz
cd overflow-0.2.6
./overflow
```

## Features

<a id="features"></a>

- **From-scratch engine.** GGUF loader, Q8_0 / Q4_K / Q6_K dequantization, the
  full Llama/Qwen forward pass — all in Rust.
- **CPU and GPU.** AVX2+FMA fused dequant-matmul on the CPU; hand-written NVRTC
  CUDA kernels on the GPU. Pick GPU, CPU, or a hybrid split.
- **Big model on a small card.** Whole layers are offloaded to the CPU so a
  model larger than your VRAM still runs at **full quality** — e.g. a 15 GB
  model on a 12 GB card, no quantization, just slower.
- **Lossless speed-up.** Prompt-lookup speculative decoding verifies drafts in
  one batched forward and emits **exactly** the greedy output — identical text,
  ~1.67× faster at temperature 0.
- **Security agent.** Reads and writes files, runs shell commands and uses the
  network, with Claude-style approval for anything outside the workspace and an
  optional `bubblewrap` sandbox.
- **OnlyChat.** A persistent, token-saving chat backed by a SQLite database —
  see below.
- **On-device fine-tuning.** The Learning tab drives a QLoRA pipeline on your
  own data.

## OnlyChat

<a id="onlychat"></a>

OnlyChat is a separate tab that keeps a conversation in a **database** instead
of in the context window. The model only ever sees the slice of history that
the chosen *context mode* feeds it, so a very long chat — including a large
ChatGPT export you want to continue locally — stays runnable at a bounded cost.

- **A chat database, chosen per chat.** When you create a chat you pick which
  SQLite database it lives in, or create a new file for it — different chats can
  live in different files. The AI reads from the file and writes each turn back.
- **Reads data that is already there.** Open an existing `.sqlite` and OnlyChat
  reads its own tables into chats so the AI can continue from that content; its
  tables are only read, never modified (OnlyChat keeps its own `oc_`-prefixed
  tables alongside).
- **Optional encryption.** A database can be encrypted at rest: the key is
  derived from a passphrase with **Argon2id** and every message is sealed with
  XChaCha20-Poly1305. The schema stays queryable; only the content is
  ciphertext.
- **Selectable context modes**, per chat:
  - *Ganzer Chat* — feed everything (exact, cost grows with length).
  - *Letzte N Runden* — a fixed window (constant cost).
  - *Zusammenfassung + Abruf* — a stored summary of the old part plus the
    passages that match the current question, plus the last turns (constant cost,
    keeps the gist of a huge chat).
- **Live token count.** The dev panel shows how many tokens the whole chat holds
  versus how many the chosen context mode actually feeds the model.
- **Manage chats.** A gear menu per chat renames or deletes it (deleting asks
  whether to remove the database file too).

## Research

<a id="research"></a>

A running question drives the project: *how do you spend less compute for the
same model size without losing quality?* The findings — including the honest
negative results — are written up as two reviews in [`docs/`](docs/) (in the
PoeziCode markup language). In short: the lossless speed-ups above are real and
shipped; large further gains on a dense model require giving something up
(sparsity, quantization), and that ceiling is quantified rather than oversold.

## Build from source

<a id="build-from-source"></a>

```sh
# engine + CLI
cargo build --release -p poezi-cli

# the GUI (binary: `overflow`)
cargo build --release -p overflow-gui
```

The Arch package is built from a checkout with:

```sh
cd packaging && makepkg -si
```

## FAQ

<a id="faq"></a>

**Does it send anything to the cloud?** No. There is no network call except the
optional first-run model download, which you trigger.

**Do I need the CUDA toolkit?** No. The kernel compiler (`libnvrtc`) is bundled;
you only need the NVIDIA driver. Without an NVIDIA GPU it runs on the CPU.

**Is this a wrapper around llama.cpp?** No. `llama.cpp` is used only as a
reference oracle in tests to prove the from-scratch engine produces identical
output.

**Is OnlyChat encryption on by default?** No. You choose per database; leave the
passphrase empty for an unencrypted file.

## License

MIT — see [LICENSE](LICENSE).
