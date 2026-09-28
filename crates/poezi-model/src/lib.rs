//! Transformer forward pass for Llama/Qwen-style decoder-only models.
//!
//! Wires the GGUF weights and tensor ops into the actual network: token
//! embeddings, N decoder layers (RMSNorm -> grouped-query attention with RoPE
//! -> RMSNorm -> SwiGLU MLP), a final norm, and the output projection to
//! vocab-sized logits. A KV-cache holds past keys/values so each new token is
//! O(seq) instead of recomputing the whole context.
//!
//! Phase A3 target: given a GGUF file + token IDs, produce next-token logits
//! that match llama.cpp within tolerance.

// TODO(A3): Config from GGUF metadata + forward pass.

pub mod tokenizer;
pub use tokenizer::{Tokenizer, TokenizerError};

/// Crate version string.
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}
