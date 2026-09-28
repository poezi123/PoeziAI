# PoeziAI

A local, from-scratch LLM inference engine written in Rust — growing into a
security-focused coding agent. A learning project: understand how an LLM
actually runs by building the whole path from the model file to generated
tokens, then use it as a private assistant for authorized security work.

Everything runs locally. No external API, no telemetry.

## Status

Early scaffold — **Phase A0** (setup). See [PLAN.md](PLAN.md) for the full
roadmap and phases.

## Layout

```
crates/
  poezi-gguf     GGUF model file format loader
  poezi-tensor   tensor storage + math primitives
  poezi-model    transformer forward pass (Llama/Qwen)
  poezi-cli      terminal interface (binary: `poezi`)
```

## Build

```sh
cargo build
cargo run -p poezi-cli
```

## Approach

Every stage of the engine is validated against `llama.cpp` running the same
weights — matching output means our implementation is correct. Correctness
first (f32 on CPU), then quantization, then GPU (CUDA on an RTX 3060).

## Scope

Intended for authorized security work: your own systems, CTFs, and engagements
with permission.
