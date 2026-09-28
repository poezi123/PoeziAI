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

use poezi_gguf::{GgufFile, MetaValue};
use poezi_tensor::{dequant_q8_0, matmul_q8_0};

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
    wq: &'a [u8],
    bq: Vec<f32>,
    wk: &'a [u8],
    bk: Vec<f32>,
    wv: &'a [u8],
    bv: Vec<f32>,
    wo: &'a [u8],
    ffn_norm: Vec<f32>,
    w_gate: &'a [u8],
    w_up: &'a [u8],
    w_down: &'a [u8],
}

pub struct Model<'a> {
    pub cfg: Config,
    tok_embd: &'a [u8], // Q8_0 [n_embd, vocab]
    out_norm: Vec<f32>,
    output: &'a [u8], // Q8_0 [n_embd, vocab]
    layers: Vec<Layer<'a>>,
}

/// Growable key/value cache, one flat buffer per layer (kv_dim floats/position).
pub struct KvCache {
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    pub len: usize,
}

impl KvCache {
    pub fn new(n_layer: usize) -> Self {
        KvCache {
            k: vec![Vec::new(); n_layer],
            v: vec![Vec::new(); n_layer],
            len: 0,
        }
    }
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

        // Small f32 tensor -> Vec<f32>.
        let f32_vec = |name: &str| -> Result<Vec<f32>> {
            let info = g
                .tensor(name)
                .ok_or_else(|| ModelError::MissingTensor(name.into()))?;
            let bytes = g.tensor_data(info);
            Ok(bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect())
        };
        // Q8_0 tensor -> borrowed mmap bytes.
        let q8 = |name: &str| -> Result<&'a [u8]> {
            let info = g
                .tensor(name)
                .ok_or_else(|| ModelError::MissingTensor(name.into()))?;
            Ok(g.tensor_data(info))
        };

        let tok_embd = q8("token_embd.weight")?;
        let output = q8("output.weight")?;
        let out_norm = f32_vec("output_norm.weight")?;

        // vocab = rows of the output projection.
        let out_info = g.tensor("output.weight").unwrap();
        let vocab = out_info.dims[1] as usize;

        let mut layers = Vec::with_capacity(n_layer);
        for i in 0..n_layer {
            let p = format!("blk.{i}");
            layers.push(Layer {
                attn_norm: f32_vec(&format!("{p}.attn_norm.weight"))?,
                wq: q8(&format!("{p}.attn_q.weight"))?,
                bq: f32_vec(&format!("{p}.attn_q.bias"))?,
                wk: q8(&format!("{p}.attn_k.weight"))?,
                bk: f32_vec(&format!("{p}.attn_k.bias"))?,
                wv: q8(&format!("{p}.attn_v.weight"))?,
                bv: f32_vec(&format!("{p}.attn_v.bias"))?,
                wo: q8(&format!("{p}.attn_output.weight"))?,
                ffn_norm: f32_vec(&format!("{p}.ffn_norm.weight"))?,
                w_gate: q8(&format!("{p}.ffn_gate.weight"))?,
                w_up: q8(&format!("{p}.ffn_up.weight"))?,
                w_down: q8(&format!("{p}.ffn_down.weight"))?,
            });
        }

        Ok(Model {
            cfg: Config {
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
            },
            tok_embd,
            out_norm,
            output,
            layers,
        })
    }

    /// Run one token at position `cache.len`, update the cache, return logits.
    pub fn forward(&self, token: u32, cache: &mut KvCache) -> Vec<f32> {
        let cfg = &self.cfg;
        let pos = cache.len;

        // --- token embedding: dequantize row `token` of tok_embd ---
        let mut x = vec![0f32; cfg.n_embd];
        let row_bytes = cfg.n_embd / 32 * 34;
        let row = &self.tok_embd[token as usize * row_bytes..(token as usize + 1) * row_bytes];
        dequant_q8_0(row, &mut x);

        for (li, layer) in self.layers.iter().enumerate() {
            // ---- attention ----
            let a = rmsnorm(&x, &layer.attn_norm, cfg.rms_eps);

            let mut q = vec![0f32; cfg.n_embd];
            let mut k = vec![0f32; cfg.kv_dim];
            let mut v = vec![0f32; cfg.kv_dim];
            matmul_q8_0(layer.wq, cfg.n_embd, cfg.n_embd, &a, &mut q);
            matmul_q8_0(layer.wk, cfg.n_embd, cfg.kv_dim, &a, &mut k);
            matmul_q8_0(layer.wv, cfg.n_embd, cfg.kv_dim, &a, &mut v);
            add_bias(&mut q, &layer.bq);
            add_bias(&mut k, &layer.bk);
            add_bias(&mut v, &layer.bv);

            // RoPE (NEOX) on each head of q and k.
            for h in 0..cfg.n_head {
                rope_neox(&mut q[h * cfg.head_dim..(h + 1) * cfg.head_dim], pos, cfg.rope_theta);
            }
            for h in 0..cfg.n_head_kv {
                rope_neox(&mut k[h * cfg.head_dim..(h + 1) * cfg.head_dim], pos, cfg.rope_theta);
            }

            // Append this position's k/v to the cache.
            cache.k[li].extend_from_slice(&k);
            cache.v[li].extend_from_slice(&v);

            // Attention output, per query head.
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
            matmul_q8_0(layer.wo, cfg.n_embd, cfg.n_embd, &attn_out, &mut o);
            for i in 0..cfg.n_embd {
                x[i] += o[i];
            }

            // ---- feed-forward (SwiGLU) ----
            let f = rmsnorm(&x, &layer.ffn_norm, cfg.rms_eps);
            let mut gate = vec![0f32; cfg.n_ff];
            let mut up = vec![0f32; cfg.n_ff];
            matmul_q8_0(layer.w_gate, cfg.n_embd, cfg.n_ff, &f, &mut gate);
            matmul_q8_0(layer.w_up, cfg.n_embd, cfg.n_ff, &f, &mut up);
            for i in 0..cfg.n_ff {
                gate[i] = silu(gate[i]) * up[i];
            }
            let mut down = vec![0f32; cfg.n_embd];
            matmul_q8_0(layer.w_down, cfg.n_ff, cfg.n_embd, &gate, &mut down);
            for i in 0..cfg.n_embd {
                x[i] += down[i];
            }
        }

        cache.len += 1;

        // ---- final norm + LM head ----
        let x = rmsnorm(&x, &self.out_norm, cfg.rms_eps);
        let mut logits = vec![0f32; cfg.vocab];
        matmul_q8_0(self.output, cfg.n_embd, cfg.vocab, &x, &mut logits);
        logits
    }
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
