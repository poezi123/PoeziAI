//! Byte-level BPE tokenizer (GPT-2 / Qwen2 style).
//!
//! Turns text into the token IDs the model consumes, and back. Three stages on
//! the way in:
//!
//! 1. **Pre-tokenization.** A regex splits the raw text into pieces (words,
//!    runs of punctuation, whitespace). Qwen2 uses a specific GPT-2-derived
//!    pattern with a `(?!\S)` lookahead, so we use a backtracking engine.
//! 2. **Byte encoding.** Each piece's UTF-8 *bytes* are mapped to a reversible
//!    set of printable Unicode chars (GPT-2's `bytes_to_unicode`). This is why
//!    the vocab shows `Ġ` for space and `Ċ` for newline: those are bytes 0x20
//!    and 0x0A remapped. Every one of the 256 byte values has a symbol, so any
//!    input is representable.
//! 3. **BPE merges.** Starting from single-byte symbols, repeatedly merge the
//!    adjacent pair with the best (lowest) merge rank until none apply. Each
//!    resulting symbol is a vocab entry with an ID.
//!
//! Special/control tokens (`<|im_start|>`, `<|im_end|>`, ...) are stored in the
//! vocab as their literal text and, when enabled, are matched atomically before
//! BPE so they map straight to their IDs.
//!
//! Validated against llama.cpp's `llama-tokenize`.

use std::collections::HashMap;

use fancy_regex::Regex;
use poezi_gguf::{GgufFile, MetaValue};

/// Qwen2 pre-tokenizer pattern, copied verbatim from llama.cpp
/// (LLAMA_VOCAB_PRE_TYPE_QWEN2).
const QWEN2_PRETOKENIZER: &str = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

#[derive(Debug)]
pub enum TokenizerError {
    MissingKey(&'static str),
    WrongType(&'static str),
    BadMerge(String),
    Regex(fancy_regex::Error),
}

impl std::fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TokenizerError::MissingKey(k) => write!(f, "tokenizer metadata missing key {k:?}"),
            TokenizerError::WrongType(k) => write!(f, "tokenizer metadata key {k:?} has wrong type"),
            TokenizerError::BadMerge(m) => write!(f, "malformed merge rule {m:?}"),
            TokenizerError::Regex(e) => write!(f, "regex error: {e}"),
        }
    }
}

impl std::error::Error for TokenizerError {}

type Result<T> = std::result::Result<T, TokenizerError>;

pub struct Tokenizer {
    /// id -> token surface string (byte-level encoded for normal tokens).
    id_to_token: Vec<String>,
    token_to_id: HashMap<String, u32>,
    /// (left, right) -> merge rank; lower rank = higher priority.
    merge_rank: HashMap<(String, String), u32>,
    /// Special tokens (text, id), sorted by descending text length so the
    /// longest match wins when scanning input.
    special: Vec<(String, u32)>,
    /// byte value -> its byte-level Unicode symbol.
    byte_encoder: [char; 256],
    /// byte-level Unicode symbol -> byte value.
    byte_decoder: HashMap<char, u8>,
    pub bos_id: Option<u32>,
    pub eos_id: Option<u32>,
    pub add_bos: bool,
    re: Regex,
}

impl Tokenizer {
    /// Build a tokenizer from a loaded GGUF file's metadata.
    pub fn from_gguf(f: &GgufFile) -> Result<Self> {
        let tokens = string_array(f, "tokenizer.ggml.tokens")?;
        let token_types = f
            .meta("tokenizer.ggml.token_type")
            .and_then(|v| match v {
                MetaValue::Array(a) => Some(a),
                _ => None,
            });

        let mut token_to_id = HashMap::with_capacity(tokens.len());
        for (i, t) in tokens.iter().enumerate() {
            token_to_id.insert(t.clone(), i as u32);
        }

        // Special tokens: token_type CONTROL(3) or USER_DEFINED(4).
        let mut special = Vec::new();
        if let Some(types) = token_types {
            for (i, ty) in types.iter().enumerate() {
                if let Some(t) = ty.as_u64() {
                    if (t == 3 || t == 4) && i < tokens.len() {
                        special.push((tokens[i].clone(), i as u32));
                    }
                }
            }
        }
        special.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

        // Merge rules: "left right" -> rank = position in the list.
        let merges = string_array(f, "tokenizer.ggml.merges")?;
        let mut merge_rank = HashMap::with_capacity(merges.len());
        for (rank, m) in merges.iter().enumerate() {
            let (a, b) = m
                .split_once(' ')
                .ok_or_else(|| TokenizerError::BadMerge(m.clone()))?;
            merge_rank.insert((a.to_string(), b.to_string()), rank as u32);
        }

        let (byte_encoder, byte_decoder) = build_byte_maps();

        let bos_id = f
            .meta("tokenizer.ggml.bos_token_id")
            .and_then(MetaValue::as_u64)
            .map(|v| v as u32);
        let eos_id = f
            .meta("tokenizer.ggml.eos_token_id")
            .and_then(MetaValue::as_u64)
            .map(|v| v as u32);
        let add_bos = match f.meta("tokenizer.ggml.add_bos_token") {
            Some(MetaValue::Bool(b)) => *b,
            _ => false,
        };

        let re = Regex::new(QWEN2_PRETOKENIZER).map_err(TokenizerError::Regex)?;

        Ok(Tokenizer {
            id_to_token: tokens,
            token_to_id,
            merge_rank,
            special,
            byte_encoder,
            byte_decoder,
            bos_id,
            eos_id,
            add_bos,
            re,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.id_to_token.len()
    }

    /// Encode text into token IDs.
    ///
    /// - `add_bos`: prepend the BOS token (only if the model defines one).
    /// - `parse_special`: recognize control tokens like `<|im_start|>` in the
    ///   input and emit their IDs atomically. With this off, such text is BPE'd
    ///   like any other characters.
    pub fn encode(&self, text: &str, add_bos: bool, parse_special: bool) -> Vec<u32> {
        let mut out = Vec::new();
        if add_bos {
            if let Some(b) = self.bos_id {
                out.push(b);
            }
        }
        if parse_special && !self.special.is_empty() {
            self.encode_with_special(text, &mut out);
        } else {
            self.encode_ordinary(text, &mut out);
        }
        out
    }

    /// Split on special tokens first, BPE the gaps.
    fn encode_with_special(&self, text: &str, out: &mut Vec<u32>) {
        let mut i = 0;
        let bytes = text.as_bytes();
        let mut segment_start = 0;
        while i < bytes.len() {
            let mut matched = None;
            for (s, id) in &self.special {
                if text[i..].starts_with(s.as_str()) {
                    matched = Some((s.len(), *id));
                    break;
                }
            }
            if let Some((len, id)) = matched {
                if segment_start < i {
                    self.encode_ordinary(&text[segment_start..i], out);
                }
                out.push(id);
                i += len;
                segment_start = i;
            } else {
                // advance one full char
                let ch_len = text[i..].chars().next().map_or(1, |c| c.len_utf8());
                i += ch_len;
            }
        }
        if segment_start < bytes.len() {
            self.encode_ordinary(&text[segment_start..], out);
        }
    }

    /// The plain byte-level BPE path (no special-token handling).
    fn encode_ordinary(&self, text: &str, out: &mut Vec<u32>) {
        // fancy-regex yields leftmost matches; the pattern's alternation covers
        // the whole input, so matches tile the text with no gaps.
        let mut search_from = 0;
        while search_from < text.len() {
            match self.re.find_from_pos(text, search_from) {
                Ok(Some(m)) => {
                    self.encode_piece(m.as_str(), out);
                    search_from = m.end().max(search_from + 1);
                }
                _ => break,
            }
        }
    }

    /// Byte-encode one pre-token, run BPE, emit IDs.
    fn encode_piece(&self, piece: &str, out: &mut Vec<u32>) {
        if piece.is_empty() {
            return;
        }
        // Map each raw byte to its byte-level symbol; each becomes one BPE unit.
        let mut symbols: Vec<String> = piece
            .as_bytes()
            .iter()
            .map(|&b| self.byte_encoder[b as usize].to_string())
            .collect();

        self.bpe(&mut symbols);

        for sym in symbols {
            if let Some(&id) = self.token_to_id.get(&sym) {
                out.push(id);
            }
            // A well-formed byte-level vocab contains every single-byte symbol,
            // so an unresolved symbol would indicate a corrupt vocab; we drop it
            // rather than panic.
        }
    }

    /// Greedy BPE: repeatedly merge every occurrence of the lowest-rank
    /// adjacent pair (GPT-2 semantics).
    fn bpe(&self, symbols: &mut Vec<String>) {
        if symbols.len() < 2 {
            return;
        }
        loop {
            // Find the best (lowest-rank) adjacent pair.
            let mut best: Option<(u32, usize)> = None;
            for i in 0..symbols.len() - 1 {
                let key = (symbols[i].clone(), symbols[i + 1].clone());
                if let Some(&rank) = self.merge_rank.get(&key) {
                    if best.map_or(true, |(r, _)| rank < r) {
                        best = Some((rank, i));
                    }
                }
            }
            let Some((_, _)) = best else { break };
            let (best_a, best_b) = {
                let (_, i) = best.unwrap();
                (symbols[i].clone(), symbols[i + 1].clone())
            };

            // Merge all non-overlapping occurrences of that pair in one pass.
            let mut merged = Vec::with_capacity(symbols.len());
            let mut i = 0;
            while i < symbols.len() {
                if i + 1 < symbols.len() && symbols[i] == best_a && symbols[i + 1] == best_b {
                    merged.push(format!("{best_a}{best_b}"));
                    i += 2;
                } else {
                    merged.push(symbols[i].clone());
                    i += 1;
                }
            }
            *symbols = merged;
            if symbols.len() < 2 {
                break;
            }
        }
    }

    /// Decode token IDs back into text.
    pub fn decode(&self, ids: &[u32]) -> String {
        // Collect the raw bytes across all tokens first, then interpret as
        // UTF-8 once — a multi-byte char can span two tokens.
        let mut bytes = Vec::new();
        for &id in ids {
            self.push_token_bytes(id, &mut bytes);
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Append the raw bytes a single token decodes to. Useful for streaming.
    pub fn push_token_bytes(&self, id: u32, out: &mut Vec<u8>) {
        let Some(tok) = self.id_to_token.get(id as usize) else {
            return;
        };
        for ch in tok.chars() {
            if let Some(&b) = self.byte_decoder.get(&ch) {
                out.push(b);
            }
        }
    }

    pub fn token_text(&self, id: u32) -> Option<&str> {
        self.id_to_token.get(id as usize).map(String::as_str)
    }
}

/// GPT-2's reversible byte<->unicode maps. Printable Latin bytes map to
/// themselves; the rest are shifted to a private range starting at 256.
fn build_byte_maps() -> ([char; 256], HashMap<char, u8>) {
    let mut bs: Vec<u32> = Vec::new();
    bs.extend(b'!' as u32..=b'~' as u32);
    bs.extend(0xA1u32..=0xAC);
    bs.extend(0xAEu32..=0xFF);
    let mut cs = bs.clone();
    let mut n = 0u32;
    for b in 0u32..256 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    let mut encoder = ['\0'; 256];
    let mut decoder = HashMap::with_capacity(256);
    for (b, c) in bs.iter().zip(cs.iter()) {
        let ch = char::from_u32(*c).expect("valid code point");
        encoder[*b as usize] = ch;
        decoder.insert(ch, *b as u8);
    }
    (encoder, decoder)
}

fn string_array(f: &GgufFile, key: &'static str) -> Result<Vec<String>> {
    match f.meta(key) {
        Some(MetaValue::Array(items)) => items
            .iter()
            .map(|v| match v {
                MetaValue::String(s) => Ok(s.clone()),
                _ => Err(TokenizerError::WrongType(key)),
            })
            .collect(),
        Some(_) => Err(TokenizerError::WrongType(key)),
        None => Err(TokenizerError::MissingKey(key)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_maps_roundtrip() {
        let (enc, dec) = build_byte_maps();
        for b in 0u8..=255 {
            let c = enc[b as usize];
            assert_eq!(dec.get(&c), Some(&b), "byte {b} did not round-trip");
        }
        // Space is byte 0x20 -> 'Ġ' (U+0120), a hallmark of this encoding.
        assert_eq!(enc[0x20], '\u{0120}');
    }
}
