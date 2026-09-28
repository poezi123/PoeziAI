//! Tensor storage and math primitives.
//!
//! Starts as plain f32 on the CPU — correctness first. Quantized kernels (A4)
//! and GPU kernels (A5) slot in behind the same operations later.
//!
//! Phase A3 target: enough to run a transformer forward pass — matmul,
//! rmsnorm, softmax, elementwise ops, rope.

// TODO(A3): Tensor { data: Vec<f32>, shape: Vec<usize> } + core ops.

/// Placeholder so the crate compiles before the math lands.
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}
