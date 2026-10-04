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
use poezi_tensor::{dequant_row, matmul, row_bytes};

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

        // Build the resident GPU model (uploads all weights to VRAM). Falls back
        // to the CPU path if the GPU or libs are unavailable.
        let gpu = GpuBackend::new()
            .ok()
            .map(Arc::new)
            .and_then(|be| build_gpu(&be, &cfg, &out_norm, &output, &layers).ok().map(Arc::new));

        Ok(Model {
            cfg,
            tok_embd,
            out_norm,
            output,
            layers,
            gpu,
        })
    }

    /// "GPU" if a CUDA backend is active, else "CPU".
    pub fn backend(&self) -> &'static str {
        if self.gpu.is_some() {
            "GPU"
        } else {
            "CPU"
        }
    }

    /// Create a KV cache matching the active backend (GPU session if available,
    /// else CPU). Falls back to CPU if the GPU session can't be allocated.
    pub fn new_cache(&self) -> KvCache {
        if let Some(w) = &self.gpu {
            if let Ok(sess) = w.new_session() {
                return KvCache { k: Vec::new(), v: Vec::new(), len: 0, gpu: Some(sess) };
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

        let logits = if let (Some(w), Some(s)) = (&self.gpu, cache.gpu.as_mut()) {
            // Fully GPU-resident forward; only the logits come back to the host.
            w.forward(s, &x, pos).expect("gpu forward")
        } else {
            self.forward_cpu(x, pos, cache)
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

    /// CPU forward pass (fallback when no GPU is available).
    fn forward_cpu(&self, mut x: Vec<f32>, pos: usize, cache: &mut KvCache) -> Vec<f32> {
        let cfg = &self.cfg;
        for (li, layer) in self.layers.iter().enumerate() {
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
            self.matvec(&layer.w_gate, cfg.n_embd, cfg.n_ff, &f, &mut gate);
            self.matvec(&layer.w_up, cfg.n_embd, cfg.n_ff, &f, &mut up);
            for i in 0..cfg.n_ff {
                gate[i] = silu(gate[i]) * up[i];
            }
            let mut down = vec![0f32; cfg.n_embd];
            self.matvec(&layer.w_down, cfg.n_ff, cfg.n_embd, &gate, &mut down);
            for i in 0..cfg.n_embd {
                x[i] += down[i];
            }
        }

        let x = rmsnorm(&x, &self.out_norm, cfg.rms_eps);
        let mut logits = vec![0f32; cfg.vocab];
        self.matvec(&self.output, cfg.n_embd, cfg.vocab, &x, &mut logits);
        logits
    }
}

/// Build the resident GPU model by uploading all weights to VRAM.
fn build_gpu(
    be: &Arc<GpuBackend>,
    cfg: &Config,
    out_norm: &[f32],
    output: &QTensor,
    layers: &[Layer],
) -> std::result::Result<GpuWeights, Box<dyn std::error::Error>> {
    let gcfg = GpuConfig {
        n_layer: cfg.n_layer,
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
    for l in layers {
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
