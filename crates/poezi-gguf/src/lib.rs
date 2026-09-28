//! GGUF model file format loader.
//!
//! GGUF is the on-disk format used by llama.cpp. Layout (all little-endian):
//!
//! ```text
//!   magic "GGUF" (4 bytes)
//!   version      (u32)          -- we expect 3
//!   tensor_count (i64)
//!   kv_count     (i64)          -- number of metadata key/value pairs
//!   metadata     kv_count * { key:string, type:u32, value }
//!   tensor_infos tensor_count * { name:string, n_dims:u32,
//!                                 dims:i64[n_dims], type:u32, offset:u64 }
//!   <padding to alignment>
//!   tensor data blob            -- each tensor at data_start + info.offset
//! ```
//!
//! Strings are a u64 byte-length followed by the raw UTF-8 bytes (no NUL).
//! Enums are stored as int32. `general.alignment` (u32) overrides the default
//! alignment of 32; the data blob starts at the first alignment boundary after
//! the tensor-info table.
//!
//! We memory-map the file so weights are paged in on demand by the OS instead
//! of being copied into RAM up front. Phase A1: parse everything, expose it,
//! validate tensor offsets. No math.

use std::collections::BTreeMap;
use std::fmt;
use std::path::Path;

use memmap2::Mmap;

const MAGIC: [u8; 4] = *b"GGUF";
const DEFAULT_ALIGNMENT: u64 = 32;

/// Errors that can occur while parsing a GGUF file.
#[derive(Debug)]
pub enum GgufError {
    Io(std::io::Error),
    /// The first four bytes were not "GGUF".
    BadMagic([u8; 4]),
    /// Version we don't support.
    UnsupportedVersion(u32),
    /// Ran off the end of the buffer while reading `what`.
    Truncated { what: &'static str, offset: usize },
    /// A metadata value type id we don't recognize.
    UnknownValueType(u32),
    /// A ggml tensor type id we don't recognize.
    UnknownTensorType(u32),
    /// A string field was not valid UTF-8.
    BadUtf8(usize),
    /// A tensor's data range falls outside the file.
    TensorOutOfBounds { name: String, offset: u64, size: u64, blob: u64 },
    /// A count/length that doesn't fit in usize on this platform.
    TooLarge(&'static str),
}

impl fmt::Display for GgufError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GgufError::Io(e) => write!(f, "io error: {e}"),
            GgufError::BadMagic(m) => write!(f, "not a GGUF file (magic = {m:?})"),
            GgufError::UnsupportedVersion(v) => write!(f, "unsupported GGUF version {v}"),
            GgufError::Truncated { what, offset } => {
                write!(f, "file truncated while reading {what} at offset {offset}")
            }
            GgufError::UnknownValueType(t) => write!(f, "unknown metadata value type {t}"),
            GgufError::UnknownTensorType(t) => write!(f, "unknown ggml tensor type {t}"),
            GgufError::BadUtf8(off) => write!(f, "invalid UTF-8 in string at offset {off}"),
            GgufError::TensorOutOfBounds { name, offset, size, blob } => write!(
                f,
                "tensor {name:?} range [{offset}, {offset}+{size}) exceeds data blob of {blob} bytes"
            ),
            GgufError::TooLarge(what) => write!(f, "{what} too large for this platform"),
        }
    }
}

impl std::error::Error for GgufError {}

impl From<std::io::Error> for GgufError {
    fn from(e: std::io::Error) -> Self {
        GgufError::Io(e)
    }
}

type Result<T> = std::result::Result<T, GgufError>;

/// Crate version string.
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// ggml tensor data types (subset relevant to the models we target).
///
/// Each variant knows its *block* layout: how many logical elements share one
/// encoded block, and how many bytes that block occupies. For unquantized types
/// the block is a single element. `n_bytes = n_elements / block_elems * block_bytes`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgmlType {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
    I8,
    I16,
    I32,
    I64,
    F64,
    Bf16,
}

impl GgmlType {
    fn from_id(id: u32) -> Result<Self> {
        use GgmlType::*;
        Ok(match id {
            0 => F32,
            1 => F16,
            2 => Q4_0,
            3 => Q4_1,
            6 => Q5_0,
            7 => Q5_1,
            8 => Q8_0,
            9 => Q8_1,
            10 => Q2K,
            11 => Q3K,
            12 => Q4K,
            13 => Q5K,
            14 => Q6K,
            15 => Q8K,
            24 => I8,
            25 => I16,
            26 => I32,
            27 => I64,
            28 => F64,
            30 => Bf16,
            other => return Err(GgufError::UnknownTensorType(other)),
        })
    }

    /// (elements per block, bytes per block). Verified against ggml-common.h.
    pub fn block(self) -> (u64, u64) {
        use GgmlType::*;
        match self {
            F32 => (1, 4),
            F16 => (1, 2),
            Bf16 => (1, 2),
            F64 => (1, 8),
            I8 => (1, 1),
            I16 => (1, 2),
            I32 => (1, 4),
            I64 => (1, 8),
            Q4_0 => (32, 18),
            Q4_1 => (32, 20),
            Q5_0 => (32, 22),
            Q5_1 => (32, 24),
            Q8_0 => (32, 34), // fp16 scale (2) + 32 * int8
            Q8_1 => (32, 36),
            Q2K => (256, 84),
            Q3K => (256, 110),
            Q4K => (256, 144),
            Q5K => (256, 176),
            Q6K => (256, 210),
            Q8K => (256, 292),
        }
    }

    /// Byte size of a tensor of this type holding `n_elements` values.
    pub fn n_bytes(self, n_elements: u64) -> u64 {
        let (elems, bytes) = self.block();
        // Tensor dims are always whole multiples of the block element count.
        (n_elements / elems) * bytes
    }
}

/// A metadata value. Arrays are homogeneous in GGUF but we store them as a
/// `Vec<MetaValue>` for simplicity.
#[derive(Debug, Clone, PartialEq)]
pub enum MetaValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    U64(u64),
    I64(i64),
    F64(f64),
    String(String),
    Array(Vec<MetaValue>),
}

impl MetaValue {
    /// Interpret the value as an unsigned integer, if it is one of the integer
    /// scalar types. Convenience for reading hyperparameters.
    pub fn as_u64(&self) -> Option<u64> {
        Some(match self {
            MetaValue::U8(v) => *v as u64,
            MetaValue::U16(v) => *v as u64,
            MetaValue::U32(v) => *v as u64,
            MetaValue::U64(v) => *v,
            MetaValue::I8(v) if *v >= 0 => *v as u64,
            MetaValue::I16(v) if *v >= 0 => *v as u64,
            MetaValue::I32(v) if *v >= 0 => *v as u64,
            MetaValue::I64(v) if *v >= 0 => *v as u64,
            _ => return None,
        })
    }

    pub fn as_f32(&self) -> Option<f32> {
        match self {
            MetaValue::F32(v) => Some(*v),
            MetaValue::F64(v) => Some(*v as f32),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            MetaValue::String(s) => Some(s.as_str()),
            _ => None,
        }
    }
}

/// Description of one tensor: name, shape, type, and where its bytes live.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    /// Dimensions in GGUF order (fastest-varying first).
    pub dims: Vec<u64>,
    pub ggml_type: GgmlType,
    /// Offset into the tensor data blob (not the file).
    pub offset: u64,
    /// Precomputed byte length of the tensor's data.
    pub n_bytes: u64,
}

impl TensorInfo {
    pub fn n_elements(&self) -> u64 {
        self.dims.iter().product()
    }
}

/// A parsed, memory-mapped GGUF file.
pub struct GgufFile {
    mmap: Mmap,
    pub version: u32,
    pub alignment: u64,
    /// Byte offset in the file where the tensor data blob begins.
    pub data_start: u64,
    pub metadata: BTreeMap<String, MetaValue>,
    pub tensors: Vec<TensorInfo>,
}

impl GgufFile {
    /// Open and fully parse a GGUF file (header, metadata, tensor table).
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let file = std::fs::File::open(path)?;
        // SAFETY: the file is opened read-only; we treat the mapping as an
        // immutable byte slice for the lifetime of `GgufFile`.
        let mmap = unsafe { Mmap::map(&file)? };

        let mut r = Reader::new(&mmap);

        let magic = r.read_bytes(4, "magic")?;
        if magic != MAGIC {
            let mut m = [0u8; 4];
            m.copy_from_slice(magic);
            return Err(GgufError::BadMagic(m));
        }

        let version = r.read_u32("version")?;
        if version != 3 {
            return Err(GgufError::UnsupportedVersion(version));
        }

        let tensor_count = r.read_i64("tensor_count")?;
        let kv_count = r.read_i64("kv_count")?;
        let tensor_count = usize_from(tensor_count, "tensor_count")?;
        let kv_count = usize_from(kv_count, "kv_count")?;

        // --- metadata ---
        let mut metadata = BTreeMap::new();
        for _ in 0..kv_count {
            let key = r.read_string("metadata key")?;
            let value = r.read_value()?;
            metadata.insert(key, value);
        }

        let alignment = metadata
            .get("general.alignment")
            .and_then(MetaValue::as_u64)
            .unwrap_or(DEFAULT_ALIGNMENT)
            .max(1);

        // --- tensor info table (offsets are relative to data_start) ---
        let mut raw_tensors = Vec::with_capacity(tensor_count);
        for _ in 0..tensor_count {
            let name = r.read_string("tensor name")?;
            let n_dims = r.read_u32("tensor n_dims")?;
            let mut dims = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                dims.push(usize_i64(r.read_i64("tensor dim")?, "tensor dim")?);
            }
            let ggml_type = GgmlType::from_id(r.read_u32("tensor type")?)?;
            let offset = r.read_u64("tensor offset")?;
            let n_elements: u64 = dims.iter().product();
            raw_tensors.push(TensorInfo {
                name,
                dims,
                ggml_type,
                offset,
                n_bytes: ggml_type.n_bytes(n_elements),
            });
        }

        // Data blob begins at the next alignment boundary after the header.
        let header_end = r.pos() as u64;
        let data_start = header_end.div_ceil(alignment) * alignment;

        // Validate every tensor lands inside the file.
        let blob_len = mmap.len() as u64 - data_start;
        for t in &raw_tensors {
            let end = t.offset.checked_add(t.n_bytes);
            if end.map_or(true, |e| e > blob_len) {
                return Err(GgufError::TensorOutOfBounds {
                    name: t.name.clone(),
                    offset: t.offset,
                    size: t.n_bytes,
                    blob: blob_len,
                });
            }
        }

        Ok(GgufFile {
            mmap,
            version,
            alignment,
            data_start,
            metadata,
            tensors: raw_tensors,
        })
    }

    /// Look up a metadata value by key.
    pub fn meta(&self, key: &str) -> Option<&MetaValue> {
        self.metadata.get(key)
    }

    /// Find a tensor by exact name.
    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// Raw bytes of a tensor's data, sliced out of the mmap.
    pub fn tensor_data(&self, info: &TensorInfo) -> &[u8] {
        let start = (self.data_start + info.offset) as usize;
        &self.mmap[start..start + info.n_bytes as usize]
    }

    /// Total size of the mapped file.
    pub fn file_size(&self) -> u64 {
        self.mmap.len() as u64
    }

    /// Human-readable summary for `poezi info`.
    pub fn summary(&self) -> String {
        use std::fmt::Write;
        let mut s = String::new();
        let _ = writeln!(s, "GGUF v{}  ({} bytes)", self.version, self.file_size());
        let _ = writeln!(
            s,
            "alignment: {}  data_start: {}  tensors: {}  metadata: {}",
            self.alignment,
            self.data_start,
            self.tensors.len(),
            self.metadata.len()
        );

        // A few interesting metadata keys, if present.
        let _ = writeln!(s, "\n-- key metadata --");
        for key in [
            "general.architecture",
            "general.name",
            "general.file_type",
        ] {
            if let Some(v) = self.metadata.get(key).and_then(MetaValue::as_str) {
                let _ = writeln!(s, "  {key} = {v}");
            }
        }
        // Architecture-prefixed hyperparameters (e.g. "qwen2.block_count").
        let arch = self
            .metadata
            .get("general.architecture")
            .and_then(MetaValue::as_str)
            .unwrap_or("");
        for suffix in [
            "block_count",
            "context_length",
            "embedding_length",
            "feed_forward_length",
            "attention.head_count",
            "attention.head_count_kv",
        ] {
            let key = format!("{arch}.{suffix}");
            if let Some(v) = self.metadata.get(&key).and_then(MetaValue::as_u64) {
                let _ = writeln!(s, "  {key} = {v}");
            }
        }

        // Tensor type histogram.
        let mut counts: BTreeMap<String, usize> = BTreeMap::new();
        for t in &self.tensors {
            *counts.entry(format!("{:?}", t.ggml_type)).or_default() += 1;
        }
        let _ = writeln!(s, "\n-- tensor types --");
        for (ty, n) in &counts {
            let _ = writeln!(s, "  {ty}: {n}");
        }

        // First handful of tensors as a sample.
        let _ = writeln!(s, "\n-- first tensors --");
        for t in self.tensors.iter().take(6) {
            let _ = writeln!(
                s,
                "  {:<28} {:?}\t{:?}\t{} bytes @ {}",
                t.name, t.dims, t.ggml_type, t.n_bytes, t.offset
            );
        }
        s
    }
}

// ---------------------------------------------------------------------------
// Little-endian cursor over the mapped bytes, with bounds checks everywhere.
// ---------------------------------------------------------------------------

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Reader { buf, pos: 0 }
    }

    fn pos(&self) -> usize {
        self.pos
    }

    fn read_bytes(&mut self, n: usize, what: &'static str) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or(GgufError::Truncated { what, offset: self.pos })?;
        if end > self.buf.len() {
            return Err(GgufError::Truncated { what, offset: self.pos });
        }
        let slice = &self.buf[self.pos..end];
        self.pos = end;
        Ok(slice)
    }

    fn read_u32(&mut self, what: &'static str) -> Result<u32> {
        let b = self.read_bytes(4, what)?;
        Ok(u32::from_le_bytes(b.try_into().unwrap()))
    }

    fn read_i32(&mut self, what: &'static str) -> Result<i32> {
        Ok(self.read_u32(what)? as i32)
    }

    fn read_u64(&mut self, what: &'static str) -> Result<u64> {
        let b = self.read_bytes(8, what)?;
        Ok(u64::from_le_bytes(b.try_into().unwrap()))
    }

    fn read_i64(&mut self, what: &'static str) -> Result<i64> {
        Ok(self.read_u64(what)? as i64)
    }

    fn read_string(&mut self, what: &'static str) -> Result<String> {
        let len = usize_from(self.read_i64(what)?, what)?;
        let start = self.pos;
        let bytes = self.read_bytes(len, what)?;
        String::from_utf8(bytes.to_vec()).map_err(|_| GgufError::BadUtf8(start))
    }

    /// Read a metadata value: a type id (u32) followed by its payload.
    fn read_value(&mut self) -> Result<MetaValue> {
        let ty = self.read_u32("value type")?;
        self.read_value_of(ty)
    }

    fn read_value_of(&mut self, ty: u32) -> Result<MetaValue> {
        Ok(match ty {
            0 => MetaValue::U8(self.read_bytes(1, "u8")?[0]),
            1 => MetaValue::I8(self.read_bytes(1, "i8")?[0] as i8),
            2 => MetaValue::U16(u16::from_le_bytes(self.read_bytes(2, "u16")?.try_into().unwrap())),
            3 => MetaValue::I16(i16::from_le_bytes(self.read_bytes(2, "i16")?.try_into().unwrap())),
            4 => MetaValue::U32(self.read_u32("u32")?),
            5 => MetaValue::I32(self.read_i32("i32")?),
            6 => MetaValue::F32(f32::from_le_bytes(self.read_bytes(4, "f32")?.try_into().unwrap())),
            7 => MetaValue::Bool(self.read_bytes(1, "bool")?[0] != 0),
            8 => MetaValue::String(self.read_string("string")?),
            9 => {
                // Array: element type (u32) + count (u64) + elements.
                let elem_ty = self.read_u32("array elem type")?;
                let count = usize_from(self.read_i64("array count")?, "array count")?;
                let mut items = Vec::with_capacity(count.min(1 << 20));
                for _ in 0..count {
                    items.push(self.read_value_of(elem_ty)?);
                }
                MetaValue::Array(items)
            }
            10 => MetaValue::U64(self.read_u64("u64")?),
            11 => MetaValue::I64(self.read_i64("i64")?),
            12 => MetaValue::F64(f64::from_le_bytes(self.read_bytes(8, "f64")?.try_into().unwrap())),
            other => return Err(GgufError::UnknownValueType(other)),
        })
    }
}

fn usize_from(v: i64, what: &'static str) -> Result<usize> {
    if v < 0 {
        return Err(GgufError::TooLarge(what));
    }
    usize::try_from(v as u64).map_err(|_| GgufError::TooLarge(what))
}

fn usize_i64(v: i64, what: &'static str) -> Result<u64> {
    if v < 0 {
        return Err(GgufError::TooLarge(what));
    }
    Ok(v as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_sizes() {
        assert_eq!(GgmlType::F32.n_bytes(10), 40);
        assert_eq!(GgmlType::F16.n_bytes(10), 20);
        // Q8_0: 64 elements = 2 blocks of 32 -> 2 * 34 bytes.
        assert_eq!(GgmlType::Q8_0.n_bytes(64), 68);
        // Q6_K: 512 elements = 2 super-blocks of 256 -> 2 * 210.
        assert_eq!(GgmlType::Q6K.n_bytes(512), 420);
    }

    #[test]
    fn reader_endianness() {
        let bytes = [0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00];
        let mut r = Reader::new(&bytes);
        assert_eq!(r.read_u32("a").unwrap(), 1);
        assert_eq!(r.read_u32("b").unwrap(), 2);
        assert!(r.read_u32("c").is_err()); // past the end
    }
}
