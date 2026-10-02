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

## The real goal: intelligence-per-watt, in one domain

The aim is a system that punches above its hardware — strong reasoning, adapted
to a single RTX 3060, specialized for CS/security. Honest framing:

- A 7–14B model on this GPU will **not** out-think a frontier model in general.
  Raw intelligence scales with parameters × data × training compute, and that
  gap can't be closed on one consumer GPU. This is physics, not effort.
- But a **specialized system can beat a generalist on a narrow field**. On
  CS/security specifically, that is within reach.
- The lever we actually have: **speed doesn't matter to us.** Almost every
  "more from less" technique trades time for quality. We can spend it freely.

Where the intelligence is won (and where we can genuinely invent things — the
model weights of a 7B aren't where a single person adds value; the layers above
are):

1. **Test-time compute / reasoning** — think long: generate multiple lines of
   attack, self-check, discard, retry. Our biggest lever. (Phase E)
2. **Domain fine-tuning** — a 7B that only does CS/security, but does it well,
   trained on exploit-db, CVEs, writeups, our own reports. (Phase B)
3. **RAG + real tools** — knowledge need not live in the weights; look it up and
   run actual tools (nmap, gdb, debuggers) and read the results. (Phase D)
4. **Agent scaffolding** — plan → act → verify → correct. Good scaffolding
   around a mediocre model beats a strong model with none. (Phase D/E)

## Guiding principles

1. **Test oracle first.** Every stage of our own engine is validated against
   `llama.cpp` running the same weights, same prompt, same sampler settings.
   Matching logits/output = our code is correct.
2. **Simple and correct before fast.** f32 on CPU first, then quantization,
   then GPU. Never optimize code we haven't proven correct.
3. **No unnecessary machinery.** Build what the project needs, nothing more.
4. **Everything local.** No data leaves the machine.
5. **Spend time, not size.** When there's a choice, trade compute-time for
   quality — we don't optimize for latency, we optimize for how good the answer
   gets when allowed to think.
6. **Measure, don't guess.** "Smarter" only counts if a benchmark says so — see
   Phase E. Every reasoning idea is judged against a fixed CS/security eval set.

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

- [x] **A0 — Setup.** Workspace, `.gitignore`, private repo. Built `llama.cpp`
      (CPU-only, no CUDA toolkit installed yet), pulled Qwen2.5-Coder-1.5B-
      Instruct Q8 into `models/`, ran it once. This is the reference oracle.

      Reference invocation (deterministic — use for A1–A3 comparison):
      ```sh
      vendor/llama.cpp/build/bin/llama-cli \
        -m models/qwen2.5-coder-1.5b-instruct-q8_0.gguf \
        -p "<prompt>" -n <N> -st --temp 0
      ```
      `--temp 0` = greedy/deterministic. `llama.cpp` build: b1-680a036.
- [x] **A1 — GGUF loader.** Reads header, metadata, and tensor table in Rust;
      mmaps the weights; validates every tensor offset against the data blob.
      `poezi info <model.gguf>` prints a summary. Verified byte-for-byte against
      llama.cpp's `llama-gguf` reader (version, alignment, data offset, tensor
      count, per-tensor size/offset/dims all match).
- [x] **A2 — Tokenizer.** Byte-level BPE (GPT-2/Qwen2): byte<->unicode maps,
      Qwen2 pre-tokenizer regex (fancy-regex for the `(?!\S)` lookahead), greedy
      merges, special-token parsing. `poezi tokenize <model> <text>`. Token IDs
      match `llama-tokenize` on all 8 stress cases (digits, unicode, emoji,
      indentation, trailing whitespace, special tokens, operators); decode
      round-trips exactly incl. multibyte UTF-8.
- [x] **A3 — CPU forward pass (f32).** Full Qwen2 transformer: token embedding,
      RMSNorm, qkv (with Qwen2 bias), NEOX RoPE, grouped-query attention with a
      KV-cache, SwiGLU MLP, final norm, LM head. Q8_0 matmul kernel in
      poezi-tensor (multithreaded via std::thread::scope). `poezi gen <model>
      <prompt> -n N` greedy-generates. **Verified: identical output to
      llama-completion at temp 0** across factual + Python + Rust prompts
      (incl. 40-token generations).
- [x] **A4 — Quantization (Q4_K/Q5_K/Q6_K).** K-quant dequant kernels in
      poezi-tensor (super-blocks of 256 with packed 6-bit sub-scales), ported
      from ggml-quants.c. Generic `matmul(ty, ...)` dispatches by weight type;
      model carries each tensor's GgmlType (QTensor). Verified byte-identical to
      llama-completion on the Qwen2.5-Coder-1.5B Q4_K_M model (uses Q4_K + Q6_K),
      and Q8_0 path unchanged. ~3 tok/s on CPU — motivates A5 (GPU).
- [~] **A5 — GPU (CUDA via `cudarc`).** Move matmul / attention onto the 3060.
      - [x] **A5.0 — toolchain + first kernel.** Vendored CUDA (libnvrtc only, via
            `scripts/fetch-cuda.sh`) — no system install, no reboot; the driver's
            libcuda is already present. New `poezi-cuda` crate compiles a kernel at
            runtime (NVRTC) and runs it on the 3060. f32 matvec validated vs CPU
            (max err 1e-6). `poezi gpu` self-test passes. Run GPU paths with
            `LD_LIBRARY_PATH=vendor/cuda/lib`.
      - [x] **A5.1 — GPU Q8_0 matmul** (fp16 decoded in-kernel via bit ops, no
            CUDA headers). Matches CPU/poezi-tensor on random quants + varied fp16
            scales (~1e-5); `poezi gpu` checks f32 + Q8_0.
      - [x] **A5.2 — GPU Q4_K/Q6_K kernels** (256-value super-blocks, 6-bit
            sub-scales). Validated vs poezi-tensor CPU.
      - [x] **A5.3 — forward pass on GPU.** Weights uploaded to VRAM once at
            load; per-matmul only x up / y down. CPU fallback if no GPU/libs.
            7B Q4_K_M output matches llama.cpp; ~4.75 tok/s (vs ~0.5 CPU).
      - [x] **A5.4 — warp-per-row kernels.** One warp (32 lanes) per output row
            with shuffle reduction, replacing one-thread-per-row. ~44 tok/s at
            short context (was ~4.75) — a ~9x matmul speedup; output still matches
            llama.cpp. Next bottleneck: CPU attention grows with context length.
      - [ ] A5.5 — move attention/KV-cache to GPU + keep activations resident
            (removes CPU attention bottleneck at long context).
            KV-cache on GPU, run the whole model on the 3060 (the speed win).
- [x] **A6 — CLI chat.** Interactive REPL (`poezi chat <model>`) using Qwen2's
      ChatML template, a persistent KV-cache across turns, and sampling
      (temperature / top-k / top-p, greedy at temp 0; configurable --system,
      --seed). Verified: coherent code answers and correct multi-turn context
      recall. Done ahead of A4/A5 so PoeziAI is usable now. First point where
      PoeziAI is a usable standalone tool.

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

### Phase E — Reasoning & agent layer (the "more from less" core)

This is where the system becomes smarter than its size suggests. Everything
here is judged against a fixed eval set so "smarter" is measured, not felt.

- [ ] **Eval harness first.** A held-out set of CS/security tasks (CTF-style
      challenges, code fixes, exploit reasoning, log analysis) with checkable
      answers. Every technique below is a diff against this baseline.
- [ ] **Test-time compute.** Let the model think long: chain-of-thought,
      self-consistency (sample many, vote), best-of-N with a verifier,
      iterative refine (draft → critique → revise). Speed is free to spend.
- [ ] **Verifier loop.** For code/exploits, the ground truth is execution —
      run it in the sandbox, feed failures back, let the model fix. A cheap
      model + a real verifier often beats a bigger model alone.
- [ ] **Tree/graph search over reasoning.** Explore multiple solution paths,
      prune dead ends, backtrack. Where our own inventions can live.
- [ ] **Structured planning.** Decompose a task into sub-goals, solve, compose.
- [ ] Room to experiment: this phase is deliberately open-ended. Any new idea
      ships only if it moves the eval numbers.

### Phase F — Product: OVERFLOW (native GUI, branding, installer)

The shippable product. Name **OVERFLOW**, logo the **basilisk eye** mark (amber).

- [x] Branding: name + logo chosen; SVG assets in `brand/` (mark, mono, lockup).
- [x] Native GUI (`overflow-gui`, binary `overflow`): egui/eframe — pure Rust, no
      web. Dark terminal theme in the OVERFLOW palette, logo drawn in-code,
      streaming chat. Inference runs on a worker thread (owns the model, keeps the
      UI responsive); tokens streamed back UTF-8-safe. Builds + runs on the
      display. egui 0.36 API note: `App::ui(ui, frame)`, unified `Panel::top/bottom`,
      `Margin::symmetric(i8,i8)`, `Frame::NONE`.
- [x] Settings window (top-right gear): live hardware meters (CPU/RAM from /proc,
      GPU/VRAM from nvidia-smi on a 1 Hz monitor thread), tokens/sec, and the
      permission model — sandboxed by default; toggles for file read, file
      write, shell, network; plus a workspace folder (free access inside) with
      "ask before anything outside this folder" (Claude-Code-style approval),
      all enforced by the agent in Phase D. Hero reads
      "S1BERIA'S CYBERSECURITY AI"; top bar shows sandboxed/tools-enabled state.
- [ ] Polish the look live (fonts: embed JetBrains Mono; spacing; message bubbles).
- [ ] Settings UI extras (model picker, expose temp/top-k/top-p, edit system prompt).
- [ ] Bundle the model + ship an installer ("AI comes with the install"), like
      LM Studio bundling its own CUDA. GPU kernels (A5) bundled later too.

---

## Open questions / decisions

- Disk space: 82 GB free. May need a second drive once training data grows.
- Reference model pinned: Qwen2.5-Coder-1.5B-Instruct Q8 (in `models/`).
- CUDA toolkit (`nvcc`) not installed yet — needed for Phase A5 (GPU).

## Status log

- 2026-09-28: Project kicked off. Workspace scaffolded, repo created.
- 2026-09-28: Goal sharpened to intelligence-per-watt on CS/security; added
  Phase E (reasoning & agent layer) and the "spend time, not size" principle.
- 2026-09-28: A0 done. llama.cpp built (CPU), Qwen2.5-Coder-1.5B Q8 runs and
  emits correct Rust at temp 0. Reference oracle ready. Next: A1 GGUF loader.
- 2026-09-28: A1 done. poezi-gguf parses the format; `poezi info` output cross-
  checked against llama-gguf, exact match. Next: A2 tokenizer.
- 2026-09-28: A2 done. Byte-level BPE tokenizer in poezi-model; IDs match
  llama-tokenize on 8 stress cases, decode round-trips. Next: A3 forward pass.
- 2026-09-28: A3 done. Full f32 Qwen2 forward pass; `poezi gen` output is
  byte-identical to llama-completion at temp 0. The engine generates real text
  end to end. Reference for raw completion: llama-completion -no-cnv --temp 0.
  Next: A4 (Q4 quantization) or A6 (chat CLI) — user's choice.
- 2026-09-28: A6 done (chose it over A4). `poezi chat` REPL with ChatML,
  persistent KV-cache, and temp/top-k/top-p sampling. Multi-turn context works.
  PoeziAI is now a usable local chat tool. Remaining in A: A4 (Q4 quant, to fit a
  bigger model) and A5 (CUDA GPU). Then Phase D (security agent).
- 2026-10-01: A4 done. Q4_K/Q5_K/Q6_K dequant + generic matmul; byte-identical
  to llama.cpp on a Q4_K_M model. Any Q4_K_M/Q8_0 qwen2 GGUF now loads (incl. a
  7B once downloaded). CPU ~3 tok/s. Next: A5 (GPU) for speed, then Phase D.
- 2026-10-01: GPU deferred — no system CUDA (bundle our own later, like LM Studio);
  training won't need it either (PyTorch bundles CUDA). Started Phase F: product
  named OVERFLOW (basilisk-eye logo), native egui GUI with streaming chat on a
  worker thread. Builds + runs. Next: polish the GUI live, then settings + bundling.
- 2026-10-01: Settings window added (hardware meters, tokens/sec, permission model
  sandboxed-by-default), hero "S1BERIA'S CYBERSECURITY AI". Phase F GUI core done.
  Then: 7B model + GPU. User chose a bigger model next.
  user verifies the look live.
- 2026-10-01: Hardware panel now splits MODEL (this process, from /proc/self) vs
  SYSTEM. Added Qwen2.5-Coder-7B Q4_K_M — loads in our engine, generation
  byte-identical to llama.cpp (engine handles 3584-dim/4-kv config). GUI defaults
  to the 7B when present. 7B on CPU is slow (~0.5-1 tok/s) → strong case for A5 (GPU).
- 2026-10-01: A5.0 done. GPU reachable WITHOUT system CUDA: vendored libnvrtc
  (scripts/fetch-cuda.sh) + the driver's libcuda; poezi-cuda compiles kernels via
  NVRTC at runtime and runs on the RTX 3060. f32 matvec matches CPU (1e-6).
  `poezi gpu` passes. Next: GPU quantized matmul kernels, then wire into forward.
- 2026-10-01: A5.1 done. GPU Q8_0 matmul with in-kernel fp16 decode; validated vs
  poezi-tensor CPU (varied scales, ~1e-5). Next: A5.2 (Q4_K/Q6_K GPU kernels).
- 2026-10-01: A5.2+A5.3 done. GPU Q4_K/Q6_K kernels validated vs CPU; forward pass
  runs matmuls on the 3060 with resident VRAM weights. 7B output == llama.cpp,
  ~4.75 tok/s (~5-10x CPU). GUI shows compute backend; scripts/run-gui.sh sets the
  CUDA lib path. Next: A5.4 kernel optimization (keep activations on GPU, coalesced
  reads) for a bigger speedup.
- 2026-10-02: A5.4 done. Warp-per-row matmul kernels (shuffle reduction) -> ~44 tok/s
  short-context on the 7B (~9x over the naive kernel), output still == llama.cpp.
  Remaining bottleneck is CPU attention at long context (grows O(seq)); A5.5 = GPU
  attention + resident activations.
- 2026-10-02: A5.5 (GPU attention + resident KV cache) done and CORRECT, but it does
  not speed up end-to-end. Reliable 3-point benchmark (two disjoint 100-token ranges,
  both 270 ms/tok) shows cost is CONSTANT per token (not position-dependent) => the
  bottleneck is NOT attention but the ~225 synchronous CPU<->GPU round-trips per token
  (every matmul uploads x and downloads y). CORRECTION: the earlier "~44 tok/s" (A5.4)
  was a measurement artifact of the short-delta method; real throughput is ~3.7 tok/s.
  A5.4 warp kernels and A5.5 GPU attention are correct building blocks but optimize the
  wrong thing. THE fix = A5.6: GPU-resident forward (keep activations on-device, chain
  rmsnorm/rope/swiglu/residual kernels, download only the final logits => ~1 sync/token
  instead of ~225). Expected the real speedup.
