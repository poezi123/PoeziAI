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
                    *yj = dot_q8_0(row, x, in_dim);
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
}
