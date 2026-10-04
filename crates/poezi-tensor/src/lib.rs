//! Tensor math primitives — f32 on the CPU, correctness first.
//!
//! For A3 we need exactly two low-level kernels beyond plain f32 elementwise
//! work: dequantizing Q8_0 blocks and multiplying a Q8_0 weight matrix by an
//! f32 vector. Everything else (rmsnorm, softmax, silu, rope) is simple f32 and
//! lives in `poezi-model` next to the config.
//!
//! Q8_0 layout: values are grouped into blocks of 32. Each block is a 2-byte
//! fp16 scale `d` followed by 32 signed int8 quants `q`; the dequantized value
//! is `d * q`. So a block is 34 bytes and encodes 32 f32 values.

use poezi_gguf::GgmlType;

/// Super-block size for the k-quants (Q4_K/Q5_K/Q6_K).
pub const QK_K: usize = 256;

/// Elements per Q8_0 block.
pub const QK8_0: usize = 32;
/// Bytes per Q8_0 block (fp16 scale + 32 int8).
pub const Q8_0_BLOCK_BYTES: usize = 34;

/// Convert an IEEE-754 half (fp16) bit pattern to f32.
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = (h >> 15) & 1;
    let exp = (h >> 10) & 0x1f;
    let mant = h & 0x3ff;
    let val = if exp == 0 {
        // zero or subnormal
        (mant as f32) * 2f32.powi(-24)
    } else if exp == 0x1f {
        if mant == 0 {
            f32::INFINITY
        } else {
            f32::NAN
        }
    } else {
        (1.0 + (mant as f32) / 1024.0) * 2f32.powi(exp as i32 - 15)
    };
    if sign == 1 {
        -val
    } else {
        val
    }
}

/// Dequantize a run of Q8_0 blocks into `out`. `out.len()` must be a multiple
/// of 32, and `src` must hold `out.len() / 32` blocks.
pub fn dequant_q8_0(src: &[u8], out: &mut [f32]) {
    assert_eq!(out.len() % QK8_0, 0, "Q8_0 length must be a multiple of 32");
    let nb = out.len() / QK8_0;
    for b in 0..nb {
        let base = b * Q8_0_BLOCK_BYTES;
        let d = f16_to_f32(u16::from_le_bytes([src[base], src[base + 1]]));
        for i in 0..QK8_0 {
            out[b * QK8_0 + i] = d * (src[base + 2 + i] as i8 as f32);
        }
    }
}

/// Dot product of one Q8_0-encoded weight row with an f32 vector `x`.
#[inline]
fn dot_q8_0(row: &[u8], x: &[f32], in_dim: usize) -> f32 {
    let nb = in_dim / QK8_0;
    let mut acc = 0f32;
    for b in 0..nb {
        let base = b * Q8_0_BLOCK_BYTES;
        let d = f16_to_f32(u16::from_le_bytes([row[base], row[base + 1]]));
        let xoff = b * QK8_0;
        let mut s = 0f32;
        for i in 0..QK8_0 {
            s += (row[base + 2 + i] as i8 as f32) * x[xoff + i];
        }
        acc += d * s;
    }
    acc
}

/// AVX2+FMA fused Q8_0 row·x dot. Per 32-value block: `s = Σ qᵢ·xᵢ` via 8-lane
/// FMA (int8→f32), then `acc += d·s`. Same block-factored arithmetic as
/// [`dot_q8_0`], just vectorized (results agree to rounding).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_q8_0_avx2(row: &[u8], x: &[f32], in_dim: usize) -> f32 { unsafe {
    use std::arch::x86_64::*;
    let nb = in_dim / QK8_0;
    let mut acc = 0f32;
    let qbase = row.as_ptr();
    let xbase = x.as_ptr();
    for b in 0..nb {
        let base = b * Q8_0_BLOCK_BYTES;
        let d = f16_to_f32(u16::from_le_bytes([row[base], row[base + 1]]));
        let qptr = qbase.add(base + 2);
        let xptr = xbase.add(b * QK8_0);
        let mut sum = _mm256_setzero_ps();
        let mut g = 0usize;
        while g < QK8_0 {
            // 8 int8 -> i32 -> f32, FMA with x
            let q8 = _mm_loadl_epi64(qptr.add(g) as *const __m128i);
            let qf = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q8));
            let xf = _mm256_loadu_ps(xptr.add(g));
            sum = _mm256_fmadd_ps(qf, xf, sum);
            g += 8;
        }
        // horizontal sum of the 8 lanes
        let lo = _mm256_castps256_ps128(sum);
        let hi = _mm256_extractf128_ps::<1>(sum);
        let s128 = _mm_add_ps(lo, hi);
        let s64 = _mm_add_ps(s128, _mm_movehl_ps(s128, s128));
        let s32 = _mm_add_ss(s64, _mm_shuffle_ps::<0x55>(s64, s64));
        acc += d * _mm_cvtss_f32(s32);
    }
    acc
}}

/// Pick the fastest available Q8_0 dot for this CPU.
#[inline]
fn dot_q8_0_best(row: &[u8], x: &[f32], in_dim: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
            return unsafe { dot_q8_0_avx2(row, x, in_dim) };
        }
    }
    dot_q8_0(row, x, in_dim)
}

/// Matrix-vector product `y = W · x` where `W` is a Q8_0 weight of shape
/// `[in_dim, out_dim]` in ggml layout (row `j` is one contiguous output's
/// weights). `y[j] = dot(row_j, x)`. Parallelized across output rows.
pub fn matmul_q8_0(w: &[u8], in_dim: usize, out_dim: usize, x: &[f32], y: &mut [f32]) {
    assert_eq!(in_dim % QK8_0, 0, "in_dim must be a multiple of 32");
    assert_eq!(x.len(), in_dim);
    assert_eq!(y.len(), out_dim);
    let row_bytes = in_dim / QK8_0 * Q8_0_BLOCK_BYTES;

    let nthreads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(out_dim.max(1));
    let chunk = out_dim.div_ceil(nthreads);

    std::thread::scope(|scope| {
        for (t, ychunk) in y.chunks_mut(chunk).enumerate() {
            let start = t * chunk;
            let w = &*w;
            let x = &*x;
            scope.spawn(move || {
                for (jj, yj) in ychunk.iter_mut().enumerate() {
                    let j = start + jj;
                    let row = &w[j * row_bytes..(j + 1) * row_bytes];
                    *yj = dot_q8_0_best(row, x, in_dim);
                }
            });
        }
    });
}

// ---------------------------------------------------------------------------
// K-quants (Q4_K / Q5_K / Q6_K): super-blocks of 256 values with 6-bit
// sub-block scales. Ported verbatim from ggml-quants.c dequantize_row_*.
// ---------------------------------------------------------------------------

/// Unpack a 6-bit scale and 6-bit min for sub-block `j` from the 12-byte packed
/// `scales` array (ggml's `get_scale_min_k4`).
#[inline]
fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        let d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        let m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
        (d, m)
    }
}

/// Dequantize one 256-value Q4_K super-block (144 bytes) into `out`.
fn dequant_q4_k_block(src: &[u8], out: &mut [f32]) {
    let d = f16_to_f32(u16::from_le_bytes([src[0], src[1]]));
    let dmin = f16_to_f32(u16::from_le_bytes([src[2], src[3]]));
    let scales = &src[4..16];
    let qs = &src[16..144];

    let mut y = 0;
    let mut is = 0;
    let mut q = 0;
    for _ in 0..4 {
        let (sc, m) = get_scale_min_k4(is, scales);
        let (d1, m1) = (d * sc as f32, dmin * m as f32);
        let (sc, m) = get_scale_min_k4(is + 1, scales);
        let (d2, m2) = (d * sc as f32, dmin * m as f32);
        for l in 0..32 {
            out[y] = d1 * (qs[q + l] & 0xF) as f32 - m1;
            y += 1;
        }
        for l in 0..32 {
            out[y] = d2 * (qs[q + l] >> 4) as f32 - m2;
            y += 1;
        }
        q += 32;
        is += 2;
    }
}

/// Dequantize one 256-value Q5_K super-block (176 bytes) into `out`.
fn dequant_q5_k_block(src: &[u8], out: &mut [f32]) {
    let d = f16_to_f32(u16::from_le_bytes([src[0], src[1]]));
    let dmin = f16_to_f32(u16::from_le_bytes([src[2], src[3]]));
    let scales = &src[4..16];
    let qh = &src[16..48];
    let qs = &src[48..176];

    let mut y = 0;
    let mut is = 0;
    let mut ql = 0;
    let mut u1: u8 = 1;
    let mut u2: u8 = 2;
    for _ in 0..4 {
        let (sc, m) = get_scale_min_k4(is, scales);
        let (d1, m1) = (d * sc as f32, dmin * m as f32);
        let (sc, m) = get_scale_min_k4(is + 1, scales);
        let (d2, m2) = (d * sc as f32, dmin * m as f32);
        for l in 0..32 {
            let hi = if qh[l] & u1 != 0 { 16 } else { 0 };
            out[y] = d1 * ((qs[ql + l] & 0xF) as i32 + hi) as f32 - m1;
            y += 1;
        }
        for l in 0..32 {
            let hi = if qh[l] & u2 != 0 { 16 } else { 0 };
            out[y] = d2 * ((qs[ql + l] >> 4) as i32 + hi) as f32 - m2;
            y += 1;
        }
        ql += 32;
        is += 2;
        u1 <<= 2;
        u2 <<= 2;
    }
}

/// Dequantize one 256-value Q6_K super-block (210 bytes) into `out`.
fn dequant_q6_k_block(src: &[u8], out: &mut [f32]) {
    let ql = &src[0..128];
    let qh = &src[128..192];
    let sc = &src[192..208]; // int8
    let d = f16_to_f32(u16::from_le_bytes([src[208], src[209]]));

    for n in 0..2 {
        let qlb = n * 64;
        let qhb = n * 32;
        let scb = n * 8;
        let yb = n * 128;
        for l in 0..32 {
            let is = l / 16;
            let q1 = ((ql[qlb + l] & 0xF) | (((qh[qhb + l] >> 0) & 3) << 4)) as i32 - 32;
            let q2 = ((ql[qlb + l + 32] & 0xF) | (((qh[qhb + l] >> 2) & 3) << 4)) as i32 - 32;
            let q3 = ((ql[qlb + l] >> 4) | (((qh[qhb + l] >> 4) & 3) << 4)) as i32 - 32;
            let q4 = ((ql[qlb + l + 32] >> 4) | (((qh[qhb + l] >> 6) & 3) << 4)) as i32 - 32;
            out[yb + l] = d * (sc[scb + is] as i8) as f32 * q1 as f32;
            out[yb + l + 32] = d * (sc[scb + is + 2] as i8) as f32 * q2 as f32;
            out[yb + l + 64] = d * (sc[scb + is + 4] as i8) as f32 * q3 as f32;
            out[yb + l + 96] = d * (sc[scb + is + 6] as i8) as f32 * q4 as f32;
        }
    }
}

/// Bytes occupied by a single weight row of `in_dim` elements of type `ty`.
pub fn row_bytes(ty: GgmlType, in_dim: usize) -> usize {
    let (elems, bytes) = ty.block();
    in_dim / elems as usize * bytes as usize
}

/// Dequantize one weight row (`src` is exactly its bytes) into `out`
/// (`out.len()` elements). Supports the types our models use.
pub fn dequant_row(ty: GgmlType, src: &[u8], out: &mut [f32]) {
    match ty {
        GgmlType::F32 => {
            for (o, c) in out.iter_mut().zip(src.chunks_exact(4)) {
                *o = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            }
        }
        GgmlType::F16 => {
            for (o, c) in out.iter_mut().zip(src.chunks_exact(2)) {
                *o = f16_to_f32(u16::from_le_bytes([c[0], c[1]]));
            }
        }
        GgmlType::Q8_0 => dequant_q8_0(src, out),
        GgmlType::Q4K => {
            for (b, chunk) in out.chunks_mut(QK_K).enumerate() {
                dequant_q4_k_block(&src[b * 144..b * 144 + 144], chunk);
            }
        }
        GgmlType::Q5K => {
            for (b, chunk) in out.chunks_mut(QK_K).enumerate() {
                dequant_q5_k_block(&src[b * 176..b * 176 + 176], chunk);
            }
        }
        GgmlType::Q6K => {
            for (b, chunk) in out.chunks_mut(QK_K).enumerate() {
                dequant_q6_k_block(&src[b * 210..b * 210 + 210], chunk);
            }
        }
        other => panic!("dequant_row: unsupported type {other:?}"),
    }
}

/// Generic matrix-vector product `y = W · x` for any supported weight type.
/// `W` has shape `[in_dim, out_dim]` in ggml layout. Each output row is
/// dequantized into a per-thread scratch buffer, then dotted with `x`.
pub fn matmul(ty: GgmlType, w: &[u8], in_dim: usize, out_dim: usize, x: &[f32], y: &mut [f32]) {
    assert_eq!(x.len(), in_dim);
    assert_eq!(y.len(), out_dim);
    let rb = row_bytes(ty, in_dim);

    let nthreads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(out_dim.max(1));
    let chunk = out_dim.div_ceil(nthreads);

    std::thread::scope(|scope| {
        for (t, ychunk) in y.chunks_mut(chunk).enumerate() {
            let start = t * chunk;
            let w = &*w;
            let x = &*x;
            scope.spawn(move || {
                // Q8_0 (our models) has a fused dequant+dot that never
                // materializes the row — avoids ~8x the weight traffic in a
                // scratch buffer. Other types still dequant into scratch.
                if ty == GgmlType::Q8_0 {
                    for (jj, yj) in ychunk.iter_mut().enumerate() {
                        let j = start + jj;
                        *yj = dot_q8_0_best(&w[j * rb..(j + 1) * rb], x, in_dim);
                    }
                } else {
                    let mut scratch = vec![0f32; in_dim];
                    for (jj, yj) in ychunk.iter_mut().enumerate() {
                        let j = start + jj;
                        dequant_row(ty, &w[j * rb..(j + 1) * rb], &mut scratch);
                        let mut acc = 0f32;
                        for i in 0..in_dim {
                            acc += scratch[i] * x[i];
                        }
                        *yj = acc;
                    }
                }
            });
        }
    });
}

/// Crate version string.
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_known_values() {
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert_eq!(f16_to_f32(0x3c00), 1.0); // 1.0 in fp16
        assert_eq!(f16_to_f32(0x4000), 2.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
    }

    #[test]
    fn q8_0_roundtrip_and_matmul() {
        // One block: scale 1.0 (0x3c00), quants 0..32 (but int8, so 0..32).
        let mut block = Vec::new();
        block.extend_from_slice(&0x3c00u16.to_le_bytes());
        for i in 0..32i8 {
            block.push(i as u8);
        }
        let mut out = [0f32; 32];
        dequant_q8_0(&block, &mut out);
        for i in 0..32 {
            assert_eq!(out[i], i as f32);
        }
        // matmul with x = all ones -> y[0] = sum(0..32) = 496.
        let x = [1f32; 32];
        let mut y = [0f32; 1];
        matmul_q8_0(&block, 32, 1, &x, &mut y);
        assert_eq!(y[0], (0..32).sum::<i32>() as f32);
    }

    #[test]
    fn q8_0_simd_matches_reference() {
        // Random-ish Q8_0 row + x; the SIMD/fused dot must match the
        // dequant-then-flat-dot reference to rounding.
        let in_dim = 32 * 40;
        let nb = in_dim / 32;
        let mut row = vec![0u8; nb * Q8_0_BLOCK_BYTES];
        let mut s: u32 = 0x1234_5678;
        let mut rng = || { s ^= s << 13; s ^= s >> 17; s ^= s << 5; s };
        for b in 0..nb {
            let base = b * Q8_0_BLOCK_BYTES;
            row[base..base + 2].copy_from_slice(&0x3a80u16.to_le_bytes()); // ~0.0156
            for i in 0..32 {
                row[base + 2 + i] = (rng() & 0xff) as u8;
            }
        }
        let x: Vec<f32> = (0..in_dim).map(|_| (rng() % 1000) as f32 / 1000.0 - 0.5).collect();
        let mut scratch = vec![0f32; in_dim];
        dequant_q8_0(&row, &mut scratch);
        let reference: f32 = scratch.iter().zip(&x).map(|(a, b)| a * b).sum();
        let got = dot_q8_0_best(&row, &x, in_dim);
        let rel = (got - reference).abs() / reference.abs().max(1e-3);
        assert!(rel < 1e-3, "SIMD dot rel err {rel} (got {got}, ref {reference})");
    }
}
