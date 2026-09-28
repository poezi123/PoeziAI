//! GGUF model file format loader.
//!
//! GGUF is the on-disk format used by llama.cpp: a header, a set of key/value
//! metadata entries (architecture, hyperparameters, tokenizer), and a table of
//! named tensors whose data follows in the file. We memory-map the file so the
//! OS pages weights in on demand instead of copying gigabytes into RAM.
//!
//! Phase A1 target: parse header + metadata + tensor table, expose them, and
//! print a summary. No math here.

// TODO(A1): implement GgufFile::open(path) -> parse magic "GGUF", version,
// tensor count, metadata kv count; then the metadata map and tensor infos.

/// Placeholder so the crate compiles before A1 lands.
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}
