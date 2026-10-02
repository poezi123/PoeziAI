//! GPU backend (A5): run the model's matmuls on the RTX 3060.
//!
//! Uses vendored CUDA (see scripts/fetch-cuda.sh): only libnvrtc is shipped; the
//! driver's libcuda is already present. Kernels are compiled at runtime with
//! NVRTC and run via the driver API (cudarc, dynamic-loading) — no system CUDA
//! install. fp16 is decoded in-kernel with bit ops so NVRTC needs no CUDA
//! headers.
//!
//! The expensive part of inference is the matmuls, so `GpuBackend` keeps the
//! quantized weights resident in VRAM and, per call, uploads only the small
//! input vector, launches the right dequant+matmul kernel, and reads the result
//! back. Everything else stays on the CPU. Kernels ported from the CPU paths in
//! poezi-tensor; validated against them.

use std::error::Error;

use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;
use poezi_gguf::GgmlType;

/// All quantized matmul kernels in one translation unit. Each is `y = W · x`
/// with one thread per output row; W is the raw ggml byte stream for its type.
const KERNELS: &str = r#"
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

// Sum a value across a 32-lane warp.
__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    return v;
}

// ---- Q8_0: 32-value blocks, 2-byte fp16 scale + 32 int8 (34 bytes) ----
// One warp per output row; lanes split the blocks, then warp-reduce.
extern "C" __global__ void matmul_q8_0(
    float* y, const unsigned char* w, const float* x, int in_dim, int out_dim)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int row = gid >> 5, lane = gid & 31;
    if (row >= out_dim) return;
    int nb = in_dim / 32;
    long rowbase = (long)row * nb * 34;
    float acc = 0.0f;
    for (int b = lane; b < nb; b += 32) {
        long base = rowbase + (long)b * 34;
        float d = h2f((unsigned short)w[base] | ((unsigned short)w[base + 1] << 8));
        int xo = b * 32;
        float blk = 0.0f;
        for (int i = 0; i < 32; ++i)
            blk += (float)((signed char)w[base + 2 + i]) * x[xo + i];
        acc += d * blk;
    }
    acc = warp_sum(acc);
    if (lane == 0) y[row] = acc;
}

// 6-bit scale/min unpack for the K-quants (ggml get_scale_min_k4).
__device__ __forceinline__ void get_sm(
    int j, const unsigned char* q, unsigned char* d, unsigned char* m)
{
    if (j < 4) { *d = q[j] & 63; *m = q[j + 4] & 63; }
    else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

// ---- Q4_K: 256-value super-blocks (d,dmin fp16; 12 packed scales; 128 qs) = 144 bytes ----
__device__ float dot_q4k(const unsigned char* sb, const float* x) {
    float d    = h2f((unsigned short)sb[0] | ((unsigned short)sb[1] << 8));
    float dmin = h2f((unsigned short)sb[2] | ((unsigned short)sb[3] << 8));
    const unsigned char* scales = sb + 4;
    const unsigned char* qs = sb + 16;
    float acc = 0.0f;
    int y = 0, is = 0, q = 0;
    for (int grp = 0; grp < 4; ++grp) {
        unsigned char sc, m;
        get_sm(is, scales, &sc, &m);     float d1 = d * sc, m1 = dmin * m;
        get_sm(is + 1, scales, &sc, &m); float d2 = d * sc, m2 = dmin * m;
        for (int l = 0; l < 32; ++l) { acc += (d1 * (float)(qs[q + l] & 0xF) - m1) * x[y]; ++y; }
        for (int l = 0; l < 32; ++l) { acc += (d2 * (float)(qs[q + l] >> 4)  - m2) * x[y]; ++y; }
        q += 32; is += 2;
    }
    return acc;
}
extern "C" __global__ void matmul_q4_k(
    float* y, const unsigned char* w, const float* x, int in_dim, int out_dim)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int row = gid >> 5, lane = gid & 31;
    if (row >= out_dim) return;
    int nb = in_dim / 256;
    long rowbase = (long)row * nb * 144;
    float acc = 0.0f;
    for (int b = lane; b < nb; b += 32) acc += dot_q4k(w + rowbase + (long)b * 144, x + b * 256);
    acc = warp_sum(acc);
    if (lane == 0) y[row] = acc;
}

// ---- Q6_K: 256-value super-blocks (128 ql; 64 qh; 16 int8 scales; fp16 d) = 210 bytes ----
__device__ float dot_q6k(const unsigned char* sb, const float* x) {
    const unsigned char* ql = sb;
    const unsigned char* qh = sb + 128;
    const signed char*   sc = (const signed char*)(sb + 192);
    float d = h2f((unsigned short)sb[208] | ((unsigned short)sb[209] << 8));
    float acc = 0.0f;
    for (int n = 0; n < 2; ++n) {
        int qlb = n * 64, qhb = n * 32, scb = n * 8, yb = n * 128;
        for (int l = 0; l < 32; ++l) {
            int is = l / 16;
            int q1 = (int)((ql[qlb + l]      & 0xF) | (((qh[qhb + l] >> 0) & 3) << 4)) - 32;
            int q2 = (int)((ql[qlb + l + 32] & 0xF) | (((qh[qhb + l] >> 2) & 3) << 4)) - 32;
            int q3 = (int)((ql[qlb + l]      >> 4)  | (((qh[qhb + l] >> 4) & 3) << 4)) - 32;
            int q4 = (int)((ql[qlb + l + 32] >> 4)  | (((qh[qhb + l] >> 6) & 3) << 4)) - 32;
            acc += d * (float)sc[scb + is + 0] * (float)q1 * x[yb + l];
            acc += d * (float)sc[scb + is + 2] * (float)q2 * x[yb + l + 32];
            acc += d * (float)sc[scb + is + 4] * (float)q3 * x[yb + l + 64];
            acc += d * (float)sc[scb + is + 6] * (float)q4 * x[yb + l + 96];
        }
    }
    return acc;
}
extern "C" __global__ void matmul_q6_k(
    float* y, const unsigned char* w, const float* x, int in_dim, int out_dim)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int row = gid >> 5, lane = gid & 31;
    if (row >= out_dim) return;
    int nb = in_dim / 256;
    long rowbase = (long)row * nb * 210;
    float acc = 0.0f;
    for (int b = lane; b < nb; b += 32) acc += dot_q6k(w + rowbase + (long)b * 210, x + b * 256);
    acc = warp_sum(acc);
    if (lane == 0) y[row] = acc;
}
"#;

/// A quantized weight resident in VRAM.
pub struct GpuWeight {
    buf: CudaSlice<u8>,
}

/// GPU context with the compiled matmul kernels; holds the device/stream alive.
pub struct GpuBackend {
    ctx: std::sync::Arc<CudaContext>,
    stream: std::sync::Arc<cudarc::driver::CudaStream>,
    f_q8: CudaFunction,
    f_q4k: CudaFunction,
    f_q6k: CudaFunction,
}

impl GpuBackend {
    /// Open the GPU and compile the kernels. Fails if no device / libs.
    pub fn new() -> Result<Self, Box<dyn Error>> {
        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();
        let module = ctx.load_module(compile_ptx(KERNELS)?)?;
        Ok(GpuBackend {
            f_q8: module.load_function("matmul_q8_0")?,
            f_q4k: module.load_function("matmul_q4_k")?,
            f_q6k: module.load_function("matmul_q6_k")?,
            ctx,
            stream,
        })
    }

    pub fn device_name(&self) -> String {
        self.ctx.name().unwrap_or_else(|_| "unknown".into())
    }

    /// Upload a quantized weight's bytes to VRAM (kept resident).
    pub fn upload(&self, bytes: &[u8]) -> Result<GpuWeight, Box<dyn Error>> {
        Ok(GpuWeight { buf: self.stream.clone_htod(bytes)? })
    }

    /// `y = W · x` using a resident weight. `ty` selects the kernel.
    pub fn matmul(
        &self,
        ty: GgmlType,
        w: &GpuWeight,
        in_dim: usize,
        out_dim: usize,
        x: &[f32],
        y: &mut [f32],
    ) -> Result<(), Box<dyn Error>> {
        let func = match ty {
            GgmlType::Q8_0 => &self.f_q8,
            GgmlType::Q4K => &self.f_q4k,
            GgmlType::Q6K => &self.f_q6k,
            other => return Err(format!("gpu matmul: unsupported type {other:?}").into()),
        };
        let x_dev = self.stream.clone_htod(x)?;
        let mut y_dev = self.stream.alloc_zeros::<f32>(out_dim)?;
        let in_i = in_dim as i32;
        let out_i = out_dim as i32;
        // One warp (32 threads) per output row; 256 threads = 8 warps per block.
        const THREADS: u32 = 256;
        let blocks = (out_dim as u32).div_ceil(THREADS / 32);
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b = self.stream.launch_builder(func);
        b.arg(&mut y_dev);
        b.arg(&w.buf);
        b.arg(&x_dev);
        b.arg(&in_i);
        b.arg(&out_i);
        unsafe { b.launch(cfg) }?;
        let out = self.stream.clone_dtoh(&y_dev)?;
        y.copy_from_slice(&out);
        Ok(())
    }

    /// Upload `w` and multiply in one shot (for tests / one-offs).
    pub fn matmul_bytes(
        &self,
        ty: GgmlType,
        w: &[u8],
        in_dim: usize,
        out_dim: usize,
        x: &[f32],
    ) -> Result<Vec<f32>, Box<dyn Error>> {
        let gw = self.upload(w)?;
        let mut y = vec![0f32; out_dim];
        self.matmul(ty, &gw, in_dim, out_dim, x, &mut y)?;
        Ok(y)
    }
}

/// True if a CUDA device can be opened.
pub fn gpu_available() -> bool {
    CudaContext::new(0).is_ok()
}

/// Device name of GPU 0.
pub fn device_name() -> Result<String, Box<dyn Error>> {
    Ok(CudaContext::new(0)?.name()?)
}

/// Q8_0 GPU self-test with a fixed fp16 scale (0.03125) vs a CPU reference.
pub fn self_test_q8(in_dim: usize, out_dim: usize) -> Result<f32, Box<dyn Error>> {
    let nb = in_dim / 32;
    let row_bytes = nb * 34;
    let d = 0.03125f32; // fp16 bits 0x2800 -> bytes 0x00,0x28
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

    let backend = GpuBackend::new()?;
    let gpu = backend.matmul_bytes(GgmlType::Q8_0, &w, in_dim, out_dim, &x)?;

    let mut max_err = 0f32;
    for j in 0..out_dim {
        let mut acc = 0f32;
        for b in 0..nb {
            let base = j * row_bytes + b * 34;
            let mut blk = 0f32;
            for i in 0..32 {
                blk += (w[base + 2 + i] as i8 as f32) * x[b * 32 + i];
            }
            acc += d * blk;
        }
        max_err = max_err.max((acc - gpu[j]).abs());
    }
    Ok(max_err)
}

pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
mod tests {
    use super::*;

    // Deterministic byte streams with valid, varied fp16 scales, compared to the
    // CPU kernels in poezi-tensor.
    fn rng() -> impl FnMut() -> u32 {
        let mut s: u32 = 0x1234_5678;
        move || {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            s
        }
    }
    fn fp16_bytes(r: &mut impl FnMut() -> u32) -> [u8; 2] {
        let exp = 8 + (r() % 11); // positive normal, exp 8..18
        let mant = r() & 0x3ff;
        let h = ((exp << 10) | mant) as u16;
        [(h & 0xff) as u8, (h >> 8) as u8]
    }

    fn check(ty: GgmlType, block_bytes: usize, block_elems: usize) {
        if !gpu_available() {
            return;
        }
        let (in_dim, out_dim) = (block_elems * 3, 200);
        let nb = in_dim / block_elems;
        let row = nb * block_bytes;
        let mut r = rng();
        let mut w = vec![0u8; out_dim * row];
        for byte in w.iter_mut() {
            *byte = (r() & 0xff) as u8;
        }
        // overwrite the fp16 scale fields with sane values per block
        for j in 0..out_dim {
            for b in 0..nb {
                let base = j * row + b * block_bytes;
                match ty {
                    GgmlType::Q4K => {
                        let d = fp16_bytes(&mut r);
                        let dm = fp16_bytes(&mut r);
                        w[base..base + 2].copy_from_slice(&d);
                        w[base + 2..base + 4].copy_from_slice(&dm);
                    }
                    GgmlType::Q6K => {
                        let d = fp16_bytes(&mut r);
                        w[base + 208..base + 210].copy_from_slice(&d);
                    }
                    _ => {}
                }
            }
        }
        let x: Vec<f32> = (0..in_dim).map(|_| (r() % 1000) as f32 / 1000.0 - 0.5).collect();

        let backend = GpuBackend::new().expect("gpu");
        let gpu = backend.matmul_bytes(ty, &w, in_dim, out_dim, &x).expect("gpu matmul");
        let mut cpu = vec![0f32; out_dim];
        poezi_tensor::matmul(ty, &w, in_dim, out_dim, &x, &mut cpu);

        let mut max_rel = 0f32;
        for j in 0..out_dim {
            let denom = cpu[j].abs().max(1.0);
            max_rel = max_rel.max((cpu[j] - gpu[j]).abs() / denom);
        }
        assert!(max_rel < 1e-3, "{ty:?}: max relative error {max_rel}");
    }

    #[test]
    fn gpu_q8_matches_cpu() {
        check(GgmlType::Q8_0, 34, 32);
    }
    #[test]
    fn gpu_q4k_matches_cpu() {
        check(GgmlType::Q4K, 144, 256);
    }
    #[test]
    fn gpu_q6k_matches_cpu() {
        check(GgmlType::Q6K, 210, 256);
    }
}
