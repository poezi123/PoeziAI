# PoeziAI — Project Plan

A local, from-scratch LLM inference engine in Rust, evolving into a
security-focused coding agent. Built as a deep learning project by a
cybersecurity engineer. Everything runs locally — no external API, no
telemetry, no external moderation layer. The system prompt is ours.

**Scope of use:** authorized security work — own systems, CTFs, and
engagements with permission. The engine is a general dual-use tool; how it
behaves is governed by the model weights and our own prompts, not by any
built-in filter (there is none to add or remove).

## Target hardware (dev machine)

- CPU: 12 threads
- RAM: 16 GB
- GPU: NVIDIA RTX 3060, 12 GB VRAM (CUDA)
- Free disk on /home: ~82 GB (watch this — models + data add up fast)

## Guiding principles

1. **Test oracle first.** Every stage of our own engine is validated against
   `llama.cpp` running the same weights, same prompt, same sampler settings.
   Matching logits/output = our code is correct.
2. **Simple and correct before fast.** f32 on CPU first, then quantization,
   then GPU. Never optimize code we haven't proven correct.
3. **No unnecessary machinery.** Build what the project needs, nothing more.
4. **Everything local.** No data leaves the machine.

## Architecture (Cargo workspace)

| Crate          | Responsibility                                        |
|----------------|-------------------------------------------------------|
| `poezi-gguf`   | Parse the GGUF model file format (mmap, metadata)     |
| `poezi-tensor` | Tensor storage + math primitives (matmul, etc.)       |
| `poezi-model`  | Transformer forward pass (Llama/Qwen: RoPE, GQA, ...) |
| `poezi-cli`    | Terminal chat / generation interface                  |

Later crates (added when their phase starts): `poezi-agent` (tool-calling +
sandbox), `poezi-cuda` (GPU kernels).

---

## Phases

### Phase A — Own inference engine (current focus)

The learning core. Replace `llama.cpp` piece by piece behind the same behavior.

- [ ] **A0 — Setup.** Workspace, `.gitignore`, private repo. Build `llama.cpp`,
      pull a small model (Qwen2.5-Coder-1.5B, ~1 GB, Q8) and run it once. This
      is the reference.
- [ ] **A1 — GGUF loader.** Read the file format in Rust: header, metadata,
      tensor table, mmap the weights. No math yet — just load and print info.
- [ ] **A2 — Tokenizer.** BPE tokenizer matching the model's vocab. Round-trip
      test: encode→decode must equal input; compare token IDs vs. llama.cpp.
- [ ] **A3 — CPU forward pass (f32).** Full transformer: embeddings, RMSNorm,
      RoPE, grouped-query attention, SwiGLU MLP, KV-cache, final projection.
      Slow but correct. **Success = logits match llama.cpp within tolerance.**
- [ ] **A4 — Quantization (Q8/Q4).** Read and compute on 4-bit weights so a
      larger model fits in memory.
- [ ] **A5 — GPU (CUDA via `cudarc`).** Move matmul / attention onto the 3060.
      Kernels in CUDA C or PTX. This is where it gets fast.
- [ ] **A6 — CLI chat.** Terminal interface with sampling controls. First
      point where PoeziAI is a usable standalone tool.

### Phase D — Security agent

Turn the engine into something that does real work.

- [ ] Tool-calling loop: model requests a tool, we run it, feed back the result.
- [ ] **Sandbox (mandatory).** Tools run in an isolated container, never
      directly on the host.
- [ ] Starter tools: shell (sandboxed), file read/write, code execution.
- [ ] RAG over a local knowledge base (writeups, docs, man pages).
- [ ] Code → run → read error → fix loop.

### Phase B — Fine-tuning on our own data

Teach the model our domain and style.

- [ ] Dataset assembly (instruction → response format).
- [ ] QLoRA fine-tune of a 7B model on the 3060 (Python/PyTorch for training).
- [ ] Load the resulting LoRA adapter in our own engine.

### Phase C — Train a small model from scratch (optional learning exercise)

- [ ] nanoGPT-style, ~50–150M params, trained on the 3060. Won't be smart;
      the point is to have done every step once.

---

## Open questions / decisions

- Disk space: 82 GB free. May need a second drive once training data grows.
- Exact reference model version to pin for A0.

## Status log

- 2026-09-28: Project kicked off. Workspace scaffolded, repo created.
