//! GPU backend (A5) — first milestone.
//!
//! Proves the whole CUDA pipeline from our own Rust code: compile a kernel at
//! runtime with NVRTC, upload data, launch on the RTX 3060, read results back —
//! all against the vendored CUDA libs (see scripts/fetch-cuda.sh), with no
//! system CUDA install. The GPU driver (libcuda) is already on the machine; we
//! only ship the compiler (libnvrtc).
//!
//! This first kernel is a plain f32 matrix-vector product, validated against a
//! CPU reference. The quantized model kernels (Q8_0/Q4_K/Q6_K) build on this.

use std::error::Error;

use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;

/// CUDA C source for `y = W · x`, one thread per output row.
/// W is row-major `[out_dim, in_dim]`: row j is j*in_dim .. j*in_dim+in_dim.
const MATVEC_SRC: &str = r#"
extern "C" __global__ void matvec_f32(
    float* y, const float* w, const float* x, int in_dim, int out_dim)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= out_dim) return;
    const float* row = w + (size_t)j * in_dim;
    float acc = 0.0f;
    for (int i = 0; i < in_dim; ++i) acc += row[i] * x[i];
    y[j] = acc;
}
"#;

/// True if a CUDA device can be opened (driver present and a GPU available).
pub fn gpu_available() -> bool {
    CudaContext::new(0).is_ok()
}

/// Short description of GPU device 0, or an error.
pub fn device_name() -> Result<String, Box<dyn Error>> {
    let ctx = CudaContext::new(0)?;
    Ok(ctx.name()?)
}

/// `y = W · x` on the GPU. `w` is `[out_dim, in_dim]` row-major f32.
pub fn matvec_f32(w: &[f32], in_dim: usize, out_dim: usize, x: &[f32]) -> Result<Vec<f32>, Box<dyn Error>> {
    assert_eq!(w.len(), in_dim * out_dim);
    assert_eq!(x.len(), in_dim);

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    let module = ctx.load_module(compile_ptx(MATVEC_SRC)?)?;
    let func = module.load_function("matvec_f32")?;

    let w_dev = stream.clone_htod(w)?;
    let x_dev = stream.clone_htod(x)?;
    let mut y_dev = stream.alloc_zeros::<f32>(out_dim)?;

    let in_i = in_dim as i32;
    let out_i = out_dim as i32;
    let cfg = LaunchConfig::for_num_elems(out_dim as u32);
    let mut builder = stream.launch_builder(&func);
    builder.arg(&mut y_dev);
    builder.arg(&w_dev);
    builder.arg(&x_dev);
    builder.arg(&in_i);
    builder.arg(&out_i);
    unsafe { builder.launch(cfg) }?;

    Ok(stream.clone_dtoh(&y_dev)?)
}

/// Run the GPU matvec on deterministic data and return the max absolute error
/// versus a CPU reference. Used by the CLI/test to prove correctness.
pub fn self_test(in_dim: usize, out_dim: usize) -> Result<f32, Box<dyn Error>> {
    let w: Vec<f32> = (0..in_dim * out_dim).map(|k| ((k % 17) as f32 - 8.0) * 0.01).collect();
    let x: Vec<f32> = (0..in_dim).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();

    let gpu = matvec_f32(&w, in_dim, out_dim, &x)?;

    let mut max_err = 0f32;
    for j in 0..out_dim {
        let row = &w[j * in_dim..(j + 1) * in_dim];
        let cpu: f32 = row.iter().zip(&x).map(|(a, b)| a * b).sum();
        max_err = max_err.max((cpu - gpu[j]).abs());
    }
    Ok(max_err)
}

pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_matvec_matches_cpu() {
        if !gpu_available() {
            eprintln!("skipping: no CUDA device (run with LD_LIBRARY_PATH=vendor/cuda/lib)");
            return;
        }
        let err = self_test(512, 300).expect("gpu matvec");
        assert!(err < 1e-3, "max error too high: {err}");
    }
}
