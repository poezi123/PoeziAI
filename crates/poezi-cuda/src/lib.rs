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

/// CUDA C for `y = W · x` with Q8_0 weights, dequantized in-kernel. One thread
/// per output row. W is the raw Q8_0 byte stream: each 32-value block is a
/// 2-byte fp16 scale followed by 32 int8 quants (34 bytes). fp16 is decoded with
/// bit ops so NVRTC needs no CUDA headers.
const MATMUL_Q8_0_SRC: &str = r#"
__device__ __forceinline__ float h2f(unsigned short h) {
    unsigned int sign = (h >> 15) & 1u;
    unsigned int exp  = (h >> 10) & 0x1fu;
    unsigned int mant = h & 0x3ffu;
    float v;
    if (exp == 0u)          v = (float)mant * 5.9604645e-8f;      // 2^-24
    else if (exp == 0x1fu)  v = mant ? (0.0f/0.0f) : (1.0f/0.0f);
    else                    v = (1.0f + (float)mant / 1024.0f) * exp2f((float)((int)exp - 15));
    return sign ? -v : v;
}
extern "C" __global__ void matmul_q8_0(
    float* y, const unsigned char* w, const float* x, int in_dim, int out_dim)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= out_dim) return;
    int nb = in_dim / 32;
    long row = (long)j * nb * 34;
    float acc = 0.0f;
    for (int b = 0; b < nb; ++b) {
        long base = row + (long)b * 34;
        unsigned short h = (unsigned short)w[base] | ((unsigned short)w[base + 1] << 8);
        float d = h2f(h);
        int xo = b * 32;
        float blk = 0.0f;
        for (int i = 0; i < 32; ++i) {
            signed char q = (signed char)w[base + 2 + i];
            blk += (float)q * x[xo + i];
        }
        acc += d * blk;
    }
    y[j] = acc;
}
"#;

/// True if a CUDA device can be opened (driver present and a GPU available).
pub fn gpu_available() -> bool {
    CudaContext::new(0).is_ok()
}

/// `y = W · x` on the GPU where `w` is a Q8_0 byte stream for a `[out_dim, in_dim]`
/// weight (in_dim a multiple of 32). Dequantization happens in the kernel.
pub fn matmul_q8_0(
    w: &[u8],
    in_dim: usize,
    out_dim: usize,
    x: &[f32],
) -> Result<Vec<f32>, Box<dyn Error>> {
    assert_eq!(in_dim % 32, 0);
    assert_eq!(w.len(), out_dim * (in_dim / 32) * 34);
    assert_eq!(x.len(), in_dim);

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = ctx.load_module(compile_ptx(MATMUL_Q8_0_SRC)?)?;
    let func = module.load_function("matmul_q8_0")?;

    let w_dev = stream.clone_htod(w)?;
    let x_dev = stream.clone_htod(x)?;
    let mut y_dev = stream.alloc_zeros::<f32>(out_dim)?;

    let in_i = in_dim as i32;
    let out_i = out_dim as i32;
    let mut builder = stream.launch_builder(&func);
    builder.arg(&mut y_dev);
    builder.arg(&w_dev);
    builder.arg(&x_dev);
    builder.arg(&in_i);
    builder.arg(&out_i);
    unsafe { builder.launch(LaunchConfig::for_num_elems(out_dim as u32)) }?;

    Ok(stream.clone_dtoh(&y_dev)?)
}

/// Q8_0 GPU self-test with a fixed fp16 scale (0.03125): builds a byte stream,
/// runs it on the GPU, and compares to a trivial CPU reference. Returns max
/// absolute error.
pub fn self_test_q8(in_dim: usize, out_dim: usize) -> Result<f32, Box<dyn Error>> {
    let nb = in_dim / 32;
    let row_bytes = nb * 34;
    let d = 0.03125f32; // 2^-5, fp16 bits 0x2800 -> bytes 0x00,0x28
    let mut w = vec![0u8; out_dim * row_bytes];
    for j in 0..out_dim {
        for b in 0..nb {
            let base = j * row_bytes + b * 34;
            w[base] = 0x00;
            w[base + 1] = 0x28;
            for i in 0..32 {
                let q = ((j + b + i) % 251) as i32 - 125;
                w[base + 2 + i] = q as i8 as u8;
            }
        }
    }
    let x: Vec<f32> = (0..in_dim).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();

    let gpu = matmul_q8_0(&w, in_dim, out_dim, &x)?;

    let mut max_err = 0f32;
    for j in 0..out_dim {
        let mut acc = 0f32;
        for b in 0..nb {
            let base = j * row_bytes + b * 34;
            let mut blk = 0f32;
            for i in 0..32 {
                let q = w[base + 2 + i] as i8 as f32;
                blk += q * x[b * 32 + i];
            }
            acc += d * blk;
        }
        max_err = max_err.max((acc - gpu[j]).abs());
    }
    Ok(max_err)
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

    /// Rigorous Q8_0 check: random quants and varied (valid) fp16 scales,
    /// compared against the CPU kernel in poezi-tensor.
    #[test]
    fn gpu_q8_matches_cpu_tensor() {
        if !gpu_available() {
            return;
        }
        let (in_dim, out_dim) = (256usize, 200usize);
        let nb = in_dim / 32;
        let row = nb * 34;
        let mut s: u32 = 0x1234_5678;
        let mut rnd = || {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            s
        };
        let mut w = vec![0u8; out_dim * row];
        for j in 0..out_dim {
            for b in 0..nb {
                let base = j * row + b * 34;
                let exp = 8 + (rnd() % 11); // positive normal fp16, exp 8..18
                let mant = rnd() & 0x3ff;
                let h = ((exp << 10) | mant) as u16;
                w[base] = (h & 0xff) as u8;
                w[base + 1] = (h >> 8) as u8;
                for i in 0..32 {
                    w[base + 2 + i] = (rnd() & 0xff) as u8;
                }
            }
        }
        let x: Vec<f32> = (0..in_dim).map(|_| (rnd() % 1000) as f32 / 1000.0 - 0.5).collect();

        let gpu = matmul_q8_0(&w, in_dim, out_dim, &x).expect("gpu q8");
        let mut cpu = vec![0f32; out_dim];
        poezi_tensor::matmul(poezi_gguf::GgmlType::Q8_0, &w, in_dim, out_dim, &x, &mut cpu);

        let mut max_rel = 0f32;
        for j in 0..out_dim {
            let denom = cpu[j].abs().max(1.0);
            max_rel = max_rel.max((cpu[j] - gpu[j]).abs() / denom);
        }
        assert!(max_rel < 1e-3, "max relative error too high: {max_rel}");
    }
}
