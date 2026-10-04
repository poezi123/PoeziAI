//! Qwen2 transformer forward pass (f32 on CPU).
//!
//! Decoder-only. For each token position we run:
//!
//! ```text
//!   x = embedding(token)
//!   for each layer:
//!       a = rmsnorm(x, attn_norm)
//!       q = Wq·a + bq ;  k = Wk·a + bk ;  v = Wv·a + bv     (Qwen2 has qkv bias)
//!       apply RoPE (NEOX) to q and k at this position
//!       append k, v to the KV cache
//!       for each query head h (grouped-query: 6 q-heads share 1 kv-head):
//!           scores_t = (q_h · k_t) / sqrt(head_dim)  for t in 0..=pos
//!           softmax(scores); out_h = Σ_t scores_t · v_t
//!       x += Wo · concat(out_h)
//!       f = rmsnorm(x, ffn_norm)
//!       x += Wdown · ( silu(Wgate·f) ⊙ (Wup·f) )            (SwiGLU)
//!   x = rmsnorm(x, output_norm)
//!   logits = Woutput · x
//! ```
//!
//! All matmuls use Q8_0 weights (see `poezi-tensor`); norms and biases are f32.

use std::sync::Arc;

use poezi_cuda::{GpuBackend, GpuConfig, GpuSession, GpuWeights};
use poezi_gguf::{GgmlType, GgufFile, MetaValue};
use poezi_tensor::{dequant_row, matmul, matmul_q8_0_masked, row_bytes};

/// A quantized weight: borrowed mmap bytes + its ggml type (CPU path).
struct QTensor<'a> {
    data: &'a [u8],
    ty: GgmlType,
}

#[derive(Debug)]
pub enum ModelError {
    MissingMeta(String),
    MissingTensor(String),
    Unsupported(String),
}

impl std::fmt::Display for ModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelError::MissingMeta(k) => write!(f, "missing metadata key {k:?}"),
            ModelError::MissingTensor(n) => write!(f, "missing tensor {n:?}"),
            ModelError::Unsupported(s) => write!(f, "unsupported: {s}"),
        }
    }
}

impl std::error::Error for ModelError {}

type Result<T> = std::result::Result<T, ModelError>;

#[derive(Debug, Clone)]
pub struct Config {
    pub n_layer: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub kv_dim: usize,
    pub n_ff: usize,
    pub vocab: usize,
    pub rope_theta: f32,
    pub rms_eps: f32,
}

/// Weights for one decoder layer. Q8_0 tensors stay as borrowed mmap bytes;
/// small f32 tensors (norms, biases) are copied out.
struct Layer<'a> {
    attn_norm: Vec<f32>,
    wq: QTensor<'a>,
    bq: Vec<f32>,
    wk: QTensor<'a>,
    bk: Vec<f32>,
    wv: QTensor<'a>,
    bv: Vec<f32>,
    wo: QTensor<'a>,
    ffn_norm: Vec<f32>,
    w_gate: QTensor<'a>,
    w_up: QTensor<'a>,
    w_down: QTensor<'a>,
}

pub struct Model<'a> {
    pub cfg: Config,
    tok_embd: QTensor<'a>, // [n_embd, vocab]
    out_norm: Vec<f32>,
    output: QTensor<'a>, // [n_embd, vocab]
    layers: Vec<Layer<'a>>,
    gpu: Option<Arc<GpuWeights>>,
    /// How many transformer layers are GPU-resident. Equals `n_layer` for a full
    /// GPU model; `< n_layer` means offloading (GPU prefix + CPU tail); `0` when
    /// there is no GPU at all.
    gpu_layers: usize,
    /// Fraction of FFN neurons to keep in the CPU path (activation sparsity). 1.0
    /// = full (default). `< 1.0` skips the up-projection of low-activation
    /// neurons (CATS-style), speeding the CPU/offload path near-losslessly.
    ffn_keep: f32,
}

/// Key/value cache. On CPU it grows per layer (kv_dim floats/position); on GPU
/// it lives in VRAM via `GpuAttn`, which also runs the attention kernel.
pub struct KvCache {
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    pub len: usize,
    gpu: Option<GpuSession>,
}

impl KvCache {
    /// CPU cache.
    pub fn new(n_layer: usize) -> Self {
        KvCache {
            k: vec![Vec::new(); n_layer],
            v: vec![Vec::new(); n_layer],
            len: 0,
            gpu: None,
        }
    }

    /// Rewind the cache to `len` positions (speculative decoding: drop the
    /// rejected tail after a partial accept). On GPU only the logical length
    /// moves — stale entries past `len` are overwritten by the next forward and
    /// never read (attention reads 0..=pos). On CPU the per-layer buffers shrink.
    pub fn truncate(&mut self, len: usize) {
        if len >= self.len {
            return;
        }
        if self.gpu.is_none() {
            for layer in self.k.iter_mut().chain(self.v.iter_mut()) {
                let per = if self.len > 0 { layer.len() / self.len } else { 0 };
                layer.truncate(len * per);
            }
        }
        self.len = len;
    }
}

/// Max sequence length the GPU KV cache preallocates (VRAM: n_layer*cap*kv_dim*2).
const GPU_KV_CAP: usize = 4096;

/// Per-layer FFN activation-sparsity measurement (see [`Model::probe_ffn_sparsity`]).
pub struct SparsityStats {
    /// Mean fraction of FFN neurons active (above threshold) per layer.
    pub per_layer_active: Vec<f64>,
    /// Mean active fraction across all layers.
    pub overall_active: f64,
    pub n_tokens: usize,
    pub rel_thresh: f32,
}

/// Internal accumulator threaded through the CPU forward while probing.
struct SparsityAccum {
    thresh: f32,
    active: Vec<u64>,
    total: Vec<u64>,
    /// Per-layer, per-neuron count of tokens the neuron was active for. Enabled
    /// for the skew probe; `None` for the plain sparsity probe.
    freq: Option<Vec<Vec<u32>>>,
    /// Per-layer boolean "hot" mask (a fixed candidate hot-set). When set, count
    /// how many active neurons fall inside it — used to test whether a hot-set
    /// learned on one corpus transfers to another.
    hot_mask: Option<Vec<Vec<bool>>>,
    hot_hits: Vec<u64>,
    hot_total: Vec<u64>,
}

/// Cross-corpus hot-set transfer result (see [`Model::probe_hotset_transfer`]).
pub struct TransferStats {
    pub hot_frac: f32,
    /// Coverage (share of activations landing in the fit-corpus hot-set) for each
    /// eval corpus, in the order passed.
    pub coverage: Vec<f64>,
}

/// Per-neuron activation skew (see [`Model::probe_neuron_skew`]). Answers: does a
/// fixed "hot" set of neurons capture most activations (power-law)? If so, a
/// static hot(VRAM)/cold(CPU) split is lossless AND fast.
pub struct SkewStats {
    /// Candidate hot-set fractions H (share of neurons kept resident).
    pub hot_fracs: Vec<f32>,
    /// For each H: fraction of all activation-events that land in the top-H
    /// most-frequently-active neurons, pooled over all layers.
    pub coverage_overall: Vec<f64>,
    /// For each H: per-layer coverage.
    pub coverage_per_layer: Vec<Vec<f64>>,
    pub thresh: f32,
    pub n_tokens: usize,
}

impl<'a> Model<'a> {
    pub fn load(g: &'a GgufFile) -> Result<Self> {
        let arch = g
            .meta("general.architecture")
            .and_then(MetaValue::as_str)
            .ok_or_else(|| ModelError::MissingMeta("general.architecture".into()))?
            .to_string();
        if arch != "qwen2" {
            return Err(ModelError::Unsupported(format!(
                "architecture {arch:?} (only qwen2 for now)"
            )));
        }
        let meta_u = |suffix: &str| -> Result<usize> {
            let key = format!("{arch}.{suffix}");
            g.meta(&key)
                .and_then(MetaValue::as_u64)
                .map(|v| v as usize)
                .ok_or(ModelError::MissingMeta(key))
        };
        let meta_f = |key: &str| -> Result<f32> {
            g.meta(key)
                .and_then(MetaValue::as_f32)
                .ok_or_else(|| ModelError::MissingMeta(key.into()))
        };

        let n_layer = meta_u("block_count")?;
        let n_embd = meta_u("embedding_length")?;
        let n_head = meta_u("attention.head_count")?;
        let n_head_kv = meta_u("attention.head_count_kv")?;
        let n_ff = meta_u("feed_forward_length")?;
        let rope_theta = meta_f(&format!("{arch}.rope.freq_base"))?;
        let rms_eps = meta_f(&format!("{arch}.attention.layer_norm_rms_epsilon"))?;
        let head_dim = n_embd / n_head;
        let kv_dim = n_head_kv * head_dim;

        // Small tensor (norm/bias) -> Vec<f32>, handling F32 or F16 storage.
        let f32_vec = move |name: &str| -> Result<Vec<f32>> {
            let info = g
                .tensor(name)
                .ok_or_else(|| ModelError::MissingTensor(name.into()))?;
            let bytes = g.tensor_data(info);
            let n = info.n_elements() as usize;
            let mut v = vec![0f32; n];
            dequant_row(info.ggml_type, bytes, &mut v);
            Ok(v)
        };
        // Quantized weight tensor -> borrowed mmap bytes + type.
        let qtensor = move |name: &str| -> Result<QTensor<'a>> {
            let info = g
                .tensor(name)
                .ok_or_else(|| ModelError::MissingTensor(name.into()))?;
            Ok(QTensor { data: g.tensor_data(info), ty: info.ggml_type })
        };

        let tok_embd = qtensor("token_embd.weight")?;
        // Tied embeddings (e.g. HF-converted Qwen2.5-Coder): if there is no
        // separate output projection, reuse the token embedding for the LM head.
        let output_name = if g.tensor("output.weight").is_some() {
            "output.weight"
        } else {
            "token_embd.weight"
        };
        let output = qtensor(output_name)?;
        let out_norm = f32_vec("output_norm.weight")?;

        // vocab = rows of the output projection.
        let out_info = g.tensor(output_name).unwrap();
        let vocab = out_info.dims[1] as usize;

        let mut layers = Vec::with_capacity(n_layer);
        for i in 0..n_layer {
            let p = format!("blk.{i}");
            layers.push(Layer {
                attn_norm: f32_vec(&format!("{p}.attn_norm.weight"))?,
                wq: qtensor(&format!("{p}.attn_q.weight"))?,
                bq: f32_vec(&format!("{p}.attn_q.bias"))?,
                wk: qtensor(&format!("{p}.attn_k.weight"))?,
                bk: f32_vec(&format!("{p}.attn_k.bias"))?,
                wv: qtensor(&format!("{p}.attn_v.weight"))?,
                bv: f32_vec(&format!("{p}.attn_v.bias"))?,
                wo: qtensor(&format!("{p}.attn_output.weight"))?,
                ffn_norm: f32_vec(&format!("{p}.ffn_norm.weight"))?,
                w_gate: qtensor(&format!("{p}.ffn_gate.weight"))?,
                w_up: qtensor(&format!("{p}.ffn_up.weight"))?,
                w_down: qtensor(&format!("{p}.ffn_down.weight"))?,
            });
        }

        let cfg = Config {
            n_layer,
            n_embd,
            n_head,
            n_head_kv,
            head_dim,
            kv_dim,
            n_ff,
            vocab,
            rope_theta,
            rms_eps,
        };

        // How many layers to keep on the GPU. OVERFLOW_GPU_LAYERS forces a count
        // (manual / testing); otherwise auto-size to free VRAM so a model bigger
        // than VRAM still loads (GPU prefix + CPU tail). Default when VRAM is
        // unknown: all layers (full GPU, current behavior).
        let n_gpu = match std::env::var("OVERFLOW_GPU_LAYERS").ok().and_then(|s| s.parse::<usize>().ok()) {
            Some(n) => n.min(n_layer),
            None => auto_gpu_layers(&cfg, &layers, &output),
        };

        // Build the resident GPU model (uploads the first `n_gpu` layers + the
        // final norm & output to VRAM). Falls back to CPU if GPU/libs absent.
        let gpu = GpuBackend::new()
            .ok()
            .map(Arc::new)
            .and_then(|be| build_gpu(&be, &cfg, &out_norm, &output, &layers, n_gpu).ok().map(Arc::new));
        let gpu_layers = if gpu.is_some() { n_gpu } else { 0 };

        // Activation-sparsity keep fraction for the CPU path (opt-in). Only a
        // value strictly between 0 and 1 enables the sparse FFN.
        let ffn_keep = std::env::var("OVERFLOW_FFN_KEEP")
            .ok()
            .and_then(|s| s.parse::<f32>().ok())
            .filter(|&k| k > 0.0 && k < 1.0)
            .unwrap_or(1.0);

        Ok(Model {
            cfg,
            tok_embd,
            out_norm,
            output,
            layers,
            gpu,
            gpu_layers,
            ffn_keep,
        })
    }

    /// Backend label: "GPU", "CPU", or "GPU+CPU N/M" when offloading.
    pub fn backend(&self) -> &'static str {
        match &self.gpu {
            Some(_) if self.gpu_layers == self.cfg.n_layer => "GPU",
            Some(_) => "GPU+CPU", // offloading (see gpu_layers() for the split)
            None => "CPU",
        }
    }

    /// Number of GPU-resident layers (for reporting the offload split).
    pub fn gpu_layers(&self) -> usize {
        self.gpu_layers
    }

    /// Offloading active: a GPU prefix plus a CPU tail.
    fn is_hybrid(&self) -> bool {
        self.gpu.is_some() && self.gpu_layers < self.cfg.n_layer
    }

    /// Create a KV cache matching the active backend (GPU session if available,
    /// else CPU). Falls back to CPU if the GPU session can't be allocated.
    pub fn new_cache(&self) -> KvCache {
        if let Some(w) = &self.gpu {
            if let Ok(sess) = w.new_session() {
                // Offloading: the GPU session holds KV for the GPU prefix layers;
                // the CPU tail layers need their own (host) KV vectors too.
                let (k, v) = if self.is_hybrid() {
                    (vec![Vec::new(); self.cfg.n_layer], vec![Vec::new(); self.cfg.n_layer])
                } else {
                    (Vec::new(), Vec::new())
                };
                return KvCache { k, v, len: 0, gpu: Some(sess) };
            }
        }
        KvCache::new(self.cfg.n_layer)
    }

    /// CPU matrix-vector product (the GPU path runs the whole forward on-device).
    fn matvec(&self, t: &QTensor, in_dim: usize, out_dim: usize, x: &[f32], y: &mut [f32]) {
        matmul(t.ty, t.data, in_dim, out_dim, x, y);
    }

    /// Run one token at position `cache.len`, update the cache, return logits.
    pub fn forward(&self, token: u32, cache: &mut KvCache) -> Vec<f32> {
        let cfg = &self.cfg;
        let pos = cache.len;

        // --- token embedding: dequantize row `token` of tok_embd ---
        let mut x = vec![0f32; cfg.n_embd];
        let rb = row_bytes(self.tok_embd.ty, cfg.n_embd);
        let row = &self.tok_embd.data[token as usize * rb..(token as usize + 1) * rb];
        dequant_row(self.tok_embd.ty, row, &mut x);

        let logits = if self.is_hybrid() {
            // Offloading: GPU prefix -> host hidden -> CPU tail layers -> GPU
            // norm+output. Only ~n_embd floats cross the PCIe boundary per side.
            // The GPU session and the CPU cache vectors live in the same KvCache,
            // so borrow `cache.gpu` in scoped blocks to interleave with the CPU
            // tail (which borrows cache.k/v).
            let w = self.gpu.as_ref().unwrap();
            let h = {
                let s = cache.gpu.as_mut().unwrap();
                w.forward_prefix(s, &x, pos).expect("gpu forward_prefix")
            };
            let h = self.forward_cpu_layers(h, pos, cache, self.gpu_layers..cfg.n_layer, None, None);
            let s = cache.gpu.as_mut().unwrap();
            w.finalize(s, &h).expect("gpu finalize")
        } else if let (Some(w), Some(s)) = (&self.gpu, cache.gpu.as_mut()) {
            // Fully GPU-resident forward; only the logits come back to the host.
            w.forward(s, &x, pos).expect("gpu forward")
        } else {
            self.forward_cpu(x, pos, cache, None, None)
        };
        cache.len += 1;
        logits
    }

    /// Verify `tokens` in ONE batched forward: they occupy positions
    /// `cache.len .. cache.len + tokens.len()`, and the returned logits[i] are
    /// the prediction *after* `tokens[i]` (i.e. for the next position). The GPU
    /// path is bit-identical to calling [`forward`] on each token in turn, so a
    /// caller can accept a verified prefix losslessly and then [`KvCache::truncate`]
    /// back. Without a GPU this falls back to exactly that token-by-token loop.
    pub fn forward_batch(&self, tokens: &[u32], cache: &mut KvCache) -> Vec<Vec<f32>> {
        if tokens.is_empty() {
            return Vec::new();
        }
        // Offloading has no batched kernel path (the GPU holds only a prefix);
        // fall back to token-by-token, which is still correct.
        if self.is_hybrid() {
            return tokens.iter().map(|&t| self.forward(t, cache)).collect();
        }
        if let (Some(w), Some(s)) = (&self.gpu, cache.gpu.as_mut()) {
            let cfg = &self.cfg;
            let base = cache.len;
            let rb = row_bytes(self.tok_embd.ty, cfg.n_embd);
            let mut xs = vec![0f32; tokens.len() * cfg.n_embd];
            for (t, &tok) in tokens.iter().enumerate() {
                let row = &self.tok_embd.data[tok as usize * rb..(tok as usize + 1) * rb];
                dequant_row(self.tok_embd.ty, row, &mut xs[t * cfg.n_embd..(t + 1) * cfg.n_embd]);
            }
            let logits = w.forward_batch(s, &xs, base, tokens.len()).expect("gpu forward_batch");
            cache.len += tokens.len();
            logits
        } else {
            tokens.iter().map(|&t| self.forward(t, cache)).collect()
        }
    }

    /// Run a CPU forward over `tokens`, threading the sparsity accumulator.
    fn run_corpus(&self, tokens: &[u32], acc: &mut SparsityAccum) {
        let cfg = &self.cfg;
        let mut cache = KvCache::new(cfg.n_layer);
        let rb = row_bytes(self.tok_embd.ty, cfg.n_embd);
        for (pos, &token) in tokens.iter().enumerate() {
            let mut x = vec![0f32; cfg.n_embd];
            let row = &self.tok_embd.data[token as usize * rb..(token as usize + 1) * rb];
            dequant_row(self.tok_embd.ty, row, &mut x);
            self.forward_cpu(x, pos, &mut cache, Some(acc), None);
            cache.len += 1;
        }
    }

    /// Research (domain-specialization hypothesis): learn a static "hot" neuron
    /// set from `fit` (e.g. security/code prompts), then measure what share of
    /// activations it covers on each corpus in `evals`. If a domain-fit hot-set
    /// covers domain eval prompts far better than generic text, a static
    /// per-domain VRAM placement is viable for our narrow agent where it fails
    /// for a generalist. `hot_frac` = share of neurons kept hot.
    pub fn probe_hotset_transfer(&self, fit: &[u32], evals: &[&[u32]], rel_thresh: f32, hot_frac: f32) -> TransferStats {
        let cfg = &self.cfg;
        // Phase 1: activation frequency on the fit corpus.
        let mut fit_acc = SparsityAccum {
            thresh: rel_thresh,
            active: vec![0u64; cfg.n_layer],
            total: vec![0u64; cfg.n_layer],
            freq: Some(vec![vec![0u32; cfg.n_ff]; cfg.n_layer]),
            hot_mask: None,
            hot_hits: Vec::new(),
            hot_total: Vec::new(),
        };
        self.run_corpus(fit, &mut fit_acc);
        let freq = fit_acc.freq.unwrap();
        // Hot mask per layer: the top hot_frac of neurons by fit frequency.
        let k = ((cfg.n_ff as f32 * hot_frac).ceil() as usize).clamp(1, cfg.n_ff);
        let mask: Vec<Vec<bool>> = freq
            .iter()
            .map(|lf| {
                let mut sorted = lf.clone();
                sorted.sort_unstable_by(|a, b| b.cmp(a));
                let thr = sorted[k - 1];
                lf.iter().map(|&f| f >= thr && f > 0).collect()
            })
            .collect();
        // Phase 2: coverage of that hot set on each eval corpus.
        let coverage = evals
            .iter()
            .map(|ev| {
                let mut acc = SparsityAccum {
                    thresh: rel_thresh,
                    active: vec![0u64; cfg.n_layer],
                    total: vec![0u64; cfg.n_layer],
                    freq: None,
                    hot_mask: Some(mask.clone()),
                    hot_hits: vec![0u64; cfg.n_layer],
                    hot_total: vec![0u64; cfg.n_layer],
                };
                self.run_corpus(ev, &mut acc);
                let hits: u64 = acc.hot_hits.iter().sum();
                let tot: u64 = acc.hot_total.iter().sum();
                if tot > 0 { hits as f64 / tot as f64 } else { 0.0 }
            })
            .collect();
        TransferStats { hot_frac, coverage }
    }

    /// Research: measure FFN activation sparsity on the real model by running a
    /// CPU forward over `tokens` and counting, per layer, how many post-SwiGLU
    /// neurons exceed `rel_thresh * max|activation|`. A low active fraction means
    /// most FFN weights barely contribute per token — the lever for PowerInfer-
    /// style hot/cold neuron offloading (big model, small VRAM, no quantization).
    /// Forces the CPU path regardless of GPU so it can read the activations.
    pub fn probe_ffn_sparsity(&self, tokens: &[u32], rel_thresh: f32) -> SparsityStats {
        let cfg = &self.cfg;
        let mut cache = KvCache::new(cfg.n_layer);
        let mut acc = SparsityAccum {
            thresh: rel_thresh,
            active: vec![0u64; cfg.n_layer],
            total: vec![0u64; cfg.n_layer],
            freq: None,
            hot_mask: None,
            hot_hits: Vec::new(),
            hot_total: Vec::new(),
        };
        let rb = row_bytes(self.tok_embd.ty, cfg.n_embd);
        for (pos, &token) in tokens.iter().enumerate() {
            let mut x = vec![0f32; cfg.n_embd];
            let row = &self.tok_embd.data[token as usize * rb..(token as usize + 1) * rb];
            dequant_row(self.tok_embd.ty, row, &mut x);
            self.forward_cpu(x, pos, &mut cache, Some(&mut acc), None);
            cache.len += 1;
        }
        let per_layer_active: Vec<f64> = acc
            .active
            .iter()
            .zip(&acc.total)
            .map(|(&a, &t)| if t > 0 { a as f64 / t as f64 } else { 0.0 })
            .collect();
        let tot_a: u64 = acc.active.iter().sum();
        let tot_t: u64 = acc.total.iter().sum();
        SparsityStats {
            overall_active: if tot_t > 0 { tot_a as f64 / tot_t as f64 } else { 0.0 },
            per_layer_active,
            n_tokens: tokens.len(),
            rel_thresh,
        }
    }

    /// Research: per-neuron activation skew. Run a CPU forward over `tokens`,
    /// count how often each FFN neuron is active, then for each hot-fraction H
    /// report what share of all activation-events the top-H most-frequent
    /// neurons capture. High coverage at small H ⇒ a static hot(VRAM)/cold(CPU)
    /// split is lossless and keeps most work on the fast path.
    pub fn probe_neuron_skew(&self, tokens: &[u32], rel_thresh: f32, hot_fracs: &[f32]) -> SkewStats {
        let cfg = &self.cfg;
        let mut cache = KvCache::new(cfg.n_layer);
        let mut acc = SparsityAccum {
            thresh: rel_thresh,
            active: vec![0u64; cfg.n_layer],
            total: vec![0u64; cfg.n_layer],
            freq: Some(vec![vec![0u32; cfg.n_ff]; cfg.n_layer]),
            hot_mask: None,
            hot_hits: Vec::new(),
            hot_total: Vec::new(),
        };
        let rb = row_bytes(self.tok_embd.ty, cfg.n_embd);
        for (pos, &token) in tokens.iter().enumerate() {
            let mut x = vec![0f32; cfg.n_embd];
            let row = &self.tok_embd.data[token as usize * rb..(token as usize + 1) * rb];
            dequant_row(self.tok_embd.ty, row, &mut x);
            self.forward_cpu(x, pos, &mut cache, Some(&mut acc), None);
            cache.len += 1;
        }
        let freq = acc.freq.unwrap();
        let mut coverage_per_layer = vec![Vec::with_capacity(cfg.n_layer); hot_fracs.len()];
        // Pooled totals across layers for the overall figure.
        let mut pooled_top = vec![0u64; hot_fracs.len()];
        let mut pooled_all = 0u64;
        for layer_freq in &freq {
            let mut sorted: Vec<u32> = layer_freq.clone();
            sorted.sort_unstable_by(|a, b| b.cmp(a)); // descending
            let total: u64 = sorted.iter().map(|&c| c as u64).sum();
            pooled_all += total;
            for (hi, &h) in hot_fracs.iter().enumerate() {
                let k = ((cfg.n_ff as f32 * h).ceil() as usize).clamp(0, cfg.n_ff);
                let top: u64 = sorted[..k].iter().map(|&c| c as u64).sum();
                pooled_top[hi] += top;
                let cov = if total > 0 { top as f64 / total as f64 } else { 0.0 };
                coverage_per_layer[hi].push(cov);
            }
        }
        let coverage_overall: Vec<f64> = pooled_top
            .iter()
            .map(|&t| if pooled_all > 0 { t as f64 / pooled_all as f64 } else { 0.0 })
            .collect();
        SkewStats {
            hot_fracs: hot_fracs.to_vec(),
            coverage_overall,
            coverage_per_layer,
            thresh: rel_thresh,
            n_tokens: tokens.len(),
        }
    }

    /// Research: teacher-forced top-1 agreement. Feed `prompt` then each token
    /// of `reference` (the dense greedy output), and count positions where the
    /// masked model's argmax still equals `reference[i]`. Unlike greedy decode
    /// this avoids the cascade (one flip derailing the rest), so it measures the
    /// true per-position quality impact of the FFN mask. `keep = None` returns
    /// `reference.len()` by construction.
    pub fn teacher_forced_agreement(&self, prompt: &[u32], reference: &[u32], keep: Option<f32>) -> usize {
        let cfg = &self.cfg;
        let mut cache = KvCache::new(cfg.n_layer);
        let rb = row_bytes(self.tok_embd.ty, cfg.n_embd);
        let embed = |token: u32| {
            let mut x = vec![0f32; cfg.n_embd];
            let row = &self.tok_embd.data[token as usize * rb..(token as usize + 1) * rb];
            dequant_row(self.tok_embd.ty, row, &mut x);
            x
        };
        let argmax = |v: &[f32]| {
            let mut best = 0usize;
            let mut bv = f32::NEG_INFINITY;
            for (i, &l) in v.iter().enumerate() {
                if l > bv {
                    bv = l;
                    best = i;
                }
            }
            best as u32
        };
        let mut last = Vec::new();
        for (pos, &t) in prompt.iter().enumerate() {
            last = self.forward_cpu(embed(t), pos, &mut cache, None, keep);
            cache.len += 1;
        }
        let mut agree = 0usize;
        for &r in reference {
            if !last.is_empty() && argmax(&last) == r {
                agree += 1;
            }
            let pos = cache.len;
            last = self.forward_cpu(embed(r), pos, &mut cache, None, keep);
            cache.len += 1;
        }
        agree
    }

    /// Research: greedy CPU decode with an optional oracle FFN sparsity mask
    /// (`keep` = fraction of neurons retained per layer, by |activation|).
    /// Returns the generated token ids. Used to measure the quality impact of
    /// dropping inactive FFN neurons vs the dense (`keep = None`) output.
    pub fn greedy_cpu(&self, prompt: &[u32], n: usize, keep: Option<f32>, eos: Option<u32>) -> Vec<u32> {
        let cfg = &self.cfg;
        let mut cache = KvCache::new(cfg.n_layer);
        let rb = row_bytes(self.tok_embd.ty, cfg.n_embd);
        let embed = |token: u32| {
            let mut x = vec![0f32; cfg.n_embd];
            let row = &self.tok_embd.data[token as usize * rb..(token as usize + 1) * rb];
            dequant_row(self.tok_embd.ty, row, &mut x);
            x
        };
        let mut last = Vec::new();
        for (pos, &t) in prompt.iter().enumerate() {
            last = self.forward_cpu(embed(t), pos, &mut cache, None, keep);
            cache.len += 1;
        }
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            let mut best = 0usize;
            let mut bv = f32::NEG_INFINITY;
            for (i, &l) in last.iter().enumerate() {
                if l > bv {
                    bv = l;
                    best = i;
                }
            }
            let nx = best as u32;
            if Some(nx) == eos {
                break;
            }
            out.push(nx);
            let pos = cache.len;
            last = self.forward_cpu(embed(nx), pos, &mut cache, None, keep);
            cache.len += 1;
        }
        out
    }

    /// CPU forward pass (fallback when no GPU is available). `probe`, when set,
    /// records FFN activation sparsity (research instrumentation; no effect on
    /// the output).
    fn forward_cpu(&self, x: Vec<f32>, pos: usize, cache: &mut KvCache, probe: Option<&mut SparsityAccum>, keep: Option<f32>) -> Vec<f32> {
        let cfg = &self.cfg;
        let x = self.forward_cpu_layers(x, pos, cache, 0..cfg.n_layer, probe, keep);
        let x = rmsnorm(&x, &self.out_norm, cfg.rms_eps);
        let mut logits = vec![0f32; cfg.vocab];
        self.matvec(&self.output, cfg.n_embd, cfg.vocab, &x, &mut logits);
        logits
    }

    /// Run the CPU transformer blocks for `range` on hidden state `x`, returning
    /// the new hidden state (no final norm/output). The offloading tail and the
    /// full CPU forward both go through here.
    fn forward_cpu_layers(&self, mut x: Vec<f32>, pos: usize, cache: &mut KvCache, range: std::ops::Range<usize>, mut probe: Option<&mut SparsityAccum>, keep: Option<f32>) -> Vec<f32> {
        let cfg = &self.cfg;
        for li in range {
            let layer = &self.layers[li];
            // ---- attention ----
            let a = rmsnorm(&x, &layer.attn_norm, cfg.rms_eps);

            let mut q = vec![0f32; cfg.n_embd];
            let mut k = vec![0f32; cfg.kv_dim];
            let mut v = vec![0f32; cfg.kv_dim];
            self.matvec(&layer.wq, cfg.n_embd, cfg.n_embd, &a, &mut q);
            self.matvec(&layer.wk, cfg.n_embd, cfg.kv_dim, &a, &mut k);
            self.matvec(&layer.wv, cfg.n_embd, cfg.kv_dim, &a, &mut v);
            add_bias(&mut q, &layer.bq);
            add_bias(&mut k, &layer.bk);
            add_bias(&mut v, &layer.bv);

            for h in 0..cfg.n_head {
                rope_neox(&mut q[h * cfg.head_dim..(h + 1) * cfg.head_dim], pos, cfg.rope_theta);
            }
            for h in 0..cfg.n_head_kv {
                rope_neox(&mut k[h * cfg.head_dim..(h + 1) * cfg.head_dim], pos, cfg.rope_theta);
            }

            cache.k[li].extend_from_slice(&k);
            cache.v[li].extend_from_slice(&v);

            let scale = 1.0 / (cfg.head_dim as f32).sqrt();
            let group = cfg.n_head / cfg.n_head_kv;
            let mut attn_out = vec![0f32; cfg.n_embd];
            for h in 0..cfg.n_head {
                let kvh = h / group;
                let qh = &q[h * cfg.head_dim..(h + 1) * cfg.head_dim];
                let mut scores = vec![0f32; pos + 1];
                for (t, score) in scores.iter_mut().enumerate() {
                    let koff = t * cfg.kv_dim + kvh * cfg.head_dim;
                    let kt = &cache.k[li][koff..koff + cfg.head_dim];
                    let mut dp = 0f32;
                    for i in 0..cfg.head_dim {
                        dp += qh[i] * kt[i];
                    }
                    *score = dp * scale;
                }
                softmax(&mut scores);
                let outh = &mut attn_out[h * cfg.head_dim..(h + 1) * cfg.head_dim];
                for (t, &w) in scores.iter().enumerate() {
                    let voff = t * cfg.kv_dim + kvh * cfg.head_dim;
                    let vt = &cache.v[li][voff..voff + cfg.head_dim];
                    for i in 0..cfg.head_dim {
                        outh[i] += w * vt[i];
                    }
                }
            }

            let mut o = vec![0f32; cfg.n_embd];
            self.matvec(&layer.wo, cfg.n_embd, cfg.n_embd, &attn_out, &mut o);
            for i in 0..cfg.n_embd {
                x[i] += o[i];
            }

            // ---- feed-forward (SwiGLU) ----
            let f = rmsnorm(&x, &layer.ffn_norm, cfg.rms_eps);
            let mut gate = vec![0f32; cfg.n_ff];
            let mut up = vec![0f32; cfg.n_ff];
            let sparse = self.ffn_keep < 1.0
                && layer.w_gate.ty == GgmlType::Q8_0
                && layer.w_up.ty == GgmlType::Q8_0;
            if sparse {
                // CATS-style sparse FFN: compute the gate fully, decide which
                // neurons are active from |silu(gate)|, then skip the up-projection
                // of the inactive ones (their contribution is ~0 anyway).
                self.matvec(&layer.w_gate, cfg.n_embd, cfg.n_ff, &f, &mut gate);
                for g in gate.iter_mut() {
                    *g = silu(*g);
                }
                let k = ((cfg.n_ff as f32 * self.ffn_keep).ceil() as usize).clamp(1, cfg.n_ff);
                let thr = kth_largest_abs(&gate, k);
                let active: Vec<bool> = gate.iter().map(|g| g.abs() >= thr).collect();
                matmul_q8_0_masked(layer.w_up.data, cfg.n_embd, cfg.n_ff, &f, &mut up, &active);
                for i in 0..cfg.n_ff {
                    gate[i] *= up[i]; // inactive up = 0 -> neuron contributes nothing
                }
            } else {
                self.matvec(&layer.w_gate, cfg.n_embd, cfg.n_ff, &f, &mut gate);
                self.matvec(&layer.w_up, cfg.n_embd, cfg.n_ff, &f, &mut up);
                for i in 0..cfg.n_ff {
                    gate[i] = silu(gate[i]) * up[i];
                }
            }
            // Research probe: how many of this layer's FFN neurons are "active"
            // (|post-SwiGLU| above a relative threshold)? Inactive neurons
            // contribute ~nothing to w_down — the basis for sparse/offloaded FFN.
            if let Some(acc) = probe.as_deref_mut() {
                let maxabs = gate.iter().fold(0f32, |m, &g| m.max(g.abs()));
                let thr = acc.thresh * maxabs;
                let mut active = 0u64;
                for (i, &g) in gate.iter().enumerate() {
                    if g.abs() > thr {
                        active += 1;
                        if let Some(freq) = acc.freq.as_mut() {
                            freq[li][i] += 1;
                        }
                        if let Some(mask) = acc.hot_mask.as_ref() {
                            acc.hot_total[li] += 1;
                            if mask[li][i] {
                                acc.hot_hits[li] += 1;
                            }
                        }
                    }
                }
                acc.active[li] += active;
                acc.total[li] += cfg.n_ff as u64;
            }
            // Oracle sparsity mask: keep only the top `keep` fraction of neurons
            // by |activation|, zero the rest before w_down. Upper bound on what a
            // perfect active-set predictor could achieve — tests quality impact.
            if let Some(keep) = keep {
                let k = ((cfg.n_ff as f32 * keep).ceil() as usize).clamp(1, cfg.n_ff);
                if k < cfg.n_ff {
                    let mut mags: Vec<f32> = gate.iter().map(|v| v.abs()).collect();
                    mags.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap());
                    let thr = mags[k - 1];
                    for g in gate.iter_mut() {
                        if g.abs() < thr {
                            *g = 0.0;
                        }
                    }
                }
            }
            let mut down = vec![0f32; cfg.n_embd];
            self.matvec(&layer.w_down, cfg.n_ff, cfg.n_embd, &gate, &mut down);
            for i in 0..cfg.n_embd {
                x[i] += down[i];
            }
        }
        x
    }
}

/// Build the resident GPU model: the first `n_gpu` transformer layers plus the
/// final norm and output projection go to VRAM. `n_gpu < layers.len()` means
/// offloading — the remaining layers run on the CPU.
fn build_gpu(
    be: &Arc<GpuBackend>,
    cfg: &Config,
    out_norm: &[f32],
    output: &QTensor,
    layers: &[Layer],
    n_gpu: usize,
) -> std::result::Result<GpuWeights, Box<dyn std::error::Error>> {
    let gcfg = GpuConfig {
        n_layer: n_gpu,
        n_embd: cfg.n_embd,
        n_head: cfg.n_head,
        n_head_kv: cfg.n_head_kv,
        head_dim: cfg.head_dim,
        kv_dim: cfg.kv_dim,
        n_ff: cfg.n_ff,
        vocab: cfg.vocab,
        rope_theta: cfg.rope_theta,
        rms_eps: cfg.rms_eps,
        kv_cap: GPU_KV_CAP,
    };
    let mut b = be.weights_builder(gcfg);
    b.out_norm(out_norm)?;
    b.output(output.data, output.ty)?;
    for l in &layers[..n_gpu] {
        b.layer(
            &l.attn_norm,
            (l.wq.data, l.wq.ty), &l.bq,
            (l.wk.data, l.wk.ty), &l.bk,
            (l.wv.data, l.wv.ty), &l.bv,
            (l.wo.data, l.wo.ty),
            &l.ffn_norm,
            (l.w_gate.data, l.w_gate.ty),
            (l.w_up.data, l.w_up.ty),
            (l.w_down.data, l.w_down.ty),
        )?;
    }
    b.build()
}

/// Choose how many layers fit in free VRAM (the rest run on CPU). Reads free
/// VRAM via nvidia-smi; returns `n_layer` (full GPU) when it can't tell, so the
/// default path is unchanged. Leaves headroom for the output projection, KV
/// cache, scratch buffers and the desktop.
fn auto_gpu_layers(cfg: &Config, layers: &[Layer], output: &QTensor) -> usize {
    let Some(free_mib) = free_vram_mib() else {
        return cfg.n_layer; // unknown -> assume it fits (previous behavior)
    };
    let free = free_mib * 1024 * 1024;
    // Per-layer VRAM: quantized weights + this layer's KV cache slice.
    let l = &layers[0];
    let per_layer_weight = (l.wq.data.len() + l.wk.data.len() + l.wv.data.len() + l.wo.data.len()
        + l.w_gate.data.len() + l.w_up.data.len() + l.w_down.data.len()) as u64;
    let per_layer_kv = (GPU_KV_CAP * cfg.kv_dim * 2 * 4) as u64;
    let per_layer = (per_layer_weight + per_layer_kv).max(1);
    // Fixed VRAM: output projection + norm + session scratch (vocab-sized logits
    // dominate). Plus 1 GiB headroom for the desktop and fragmentation.
    let fixed = output.data.len() as u64
        + (cfg.n_embd * 4) as u64
        + (cfg.vocab * 4 + cfg.n_ff * 2 * 4 + cfg.n_embd * 6 * 4) as u64
        + 1024 * 1024 * 1024;
    let budget = free.saturating_sub(fixed);
    ((budget / per_layer) as usize).min(cfg.n_layer)
}

/// Free VRAM in MiB via nvidia-smi, if available.
fn free_vram_mib() -> Option<u64> {
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.free", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    String::from_utf8_lossy(&out.stdout).lines().next()?.trim().parse::<u64>().ok()
}

fn rmsnorm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let ss: f32 = x.iter().map(|v| v * v).sum::<f32>() / n as f32;
    let scale = 1.0 / (ss + eps).sqrt();
    (0..n).map(|i| x[i] * scale * weight[i]).collect()
}

fn add_bias(x: &mut [f32], bias: &[f32]) {
    if bias.len() == x.len() {
        for i in 0..x.len() {
            x[i] += bias[i];
        }
    }
}

fn softmax(x: &mut [f32]) {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    for v in x.iter_mut() {
        *v /= sum;
    }
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// The `k`-th largest absolute value in `v` (1-indexed), via a partial select
/// (O(n), no full sort). Used as the activation-sparsity threshold: the top `k`
/// neurons by |activation| have |value| >= this.
fn kth_largest_abs(v: &[f32], k: usize) -> f32 {
    if k >= v.len() {
        return 0.0;
    }
    let mut mags: Vec<f32> = v.iter().map(|x| x.abs()).collect();
    // Put the k largest first; element k-1 is the smallest of those = threshold.
    mags.select_nth_unstable_by(k - 1, |a, b| b.partial_cmp(a).unwrap());
    mags[k - 1]
}

/// NEOX-style rotary position embedding on a single head vector, in place.
/// Pairs element `k` with element `k + head_dim/2`.
fn rope_neox(head: &mut [f32], pos: usize, theta: f32) {
    let d = head.len();
    let half = d / 2;
    for k in 0..half {
        let inv_freq = 1.0 / theta.powf((2 * k) as f32 / d as f32);
        let angle = pos as f32 * inv_freq;
        let (s, c) = angle.sin_cos();
        let x0 = head[k];
        let x1 = head[k + half];
        head[k] = x0 * c - x1 * s;
        head[k + half] = x0 * s + x1 * c;
    }
}
