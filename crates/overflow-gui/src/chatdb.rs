//! OnlyChat persistent store — a SQLite database of chats and messages.
//!
//! One database file can hold many chats. Message and summary text may be
//! encrypted at rest: a key is derived from a passphrase with Argon2id and each
//! value is sealed with XChaCha20-Poly1305 (random 24-byte nonce per value). An
//! unencrypted database stores the text as plain UTF-8 bytes. The schema (ids,
//! roles, timestamps) stays in the clear either way so the file is a normal,
//! queryable SQLite database; only the content is ciphertext.
//!
//! This is product plumbing, not the inference engine — it only reads and
//! writes rows. Context assembly for a turn lives in `assemble`.

use argon2::Argon2;
use chacha20poly1305::aead::{Aead, KeyInit};
use chacha20poly1305::{Key, XChaCha20Poly1305, XNonce};
use rand::RngCore;
use rusqlite::{params, Connection, OptionalExtension};
use std::path::Path;

/// Context-assembly strategy for a chat (how much history the model sees).
#[derive(Clone, Copy, PartialEq)]
pub enum CtxMode {
    /// Feed the whole chat verbatim. Exact, but cost grows with length.
    Full,
    /// Feed only the last N turns. Constant cost, forgets older context.
    LastN,
    /// Feed a stored summary of the old part + retrieved matches + the last N
    /// turns. Constant cost, keeps the gist of a very long chat.
    SummaryRetrieval,
}

impl CtxMode {
    pub fn from_i32(v: i32) -> Self {
        match v {
            0 => CtxMode::Full,
            2 => CtxMode::SummaryRetrieval,
            _ => CtxMode::LastN,
        }
    }
    pub fn to_i32(self) -> i32 {
        match self {
            CtxMode::Full => 0,
            CtxMode::LastN => 1,
            CtxMode::SummaryRetrieval => 2,
        }
    }
    pub fn label(self) -> &'static str {
        match self {
            CtxMode::Full => "Ganzer Chat",
            CtxMode::LastN => "Letzte N Runden",
            CtxMode::SummaryRetrieval => "Zusammenfassung + Abruf",
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
pub enum Role {
    User,
    Assistant,
}

impl Role {
    fn to_i32(self) -> i32 {
        match self {
            Role::User => 0,
            Role::Assistant => 1,
        }
    }
    fn from_i32(v: i32) -> Self {
        if v == 0 {
            Role::User
        } else {
            Role::Assistant
        }
    }
    pub fn is_user(self) -> bool {
        matches!(self, Role::User)
    }
}

pub struct ChatMsg {
    pub role: Role,
    pub text: String,
}

pub struct ChatMeta {
    pub id: i64,
    pub name: String,
    pub mode: CtxMode,
    pub n_msgs: i64,
}

/// Encryption state for the open database.
enum Crypto {
    Plain,
    Sealed(Box<XChaCha20Poly1305>),
}

impl Crypto {
    /// Store text -> bytes on disk. For sealed DBs the first 24 bytes are the
    /// nonce and the rest is the ciphertext; plain DBs store raw UTF-8.
    fn seal(&self, text: &str) -> Vec<u8> {
        match self {
            Crypto::Plain => text.as_bytes().to_vec(),
            Crypto::Sealed(c) => {
                let mut nonce = [0u8; 24];
                rand::thread_rng().fill_bytes(&mut nonce);
                let ct = c
                    .encrypt(XNonce::from_slice(&nonce), text.as_bytes())
                    .expect("xchacha encrypt");
                let mut out = Vec::with_capacity(24 + ct.len());
                out.extend_from_slice(&nonce);
                out.extend_from_slice(&ct);
                out
            }
        }
    }

    fn open(&self, blob: &[u8]) -> String {
        match self {
            Crypto::Plain => String::from_utf8_lossy(blob).into_owned(),
            Crypto::Sealed(c) => {
                if blob.len() < 24 {
                    return String::new();
                }
                let (nonce, ct) = blob.split_at(24);
                match c.decrypt(XNonce::from_slice(nonce), ct) {
                    Ok(pt) => String::from_utf8_lossy(&pt).into_owned(),
                    Err(_) => String::new(),
                }
            }
        }
    }
}

/// OWASP-ish Argon2id parameters (19 MiB, 2 passes). Stored in `meta` so a file
/// stays readable even if the defaults change later.
const ARGON_M_KIB: u32 = 19 * 1024;
const ARGON_T: u32 = 2;
const ARGON_P: u32 = 1;
const VERIFIER_PLAINTEXT: &[u8] = b"overflow-onlychat-v1";

fn derive_key(passphrase: &str, salt: &[u8]) -> Result<[u8; 32], String> {
    let params = argon2::Params::new(ARGON_M_KIB, ARGON_T, ARGON_P, Some(32))
        .map_err(|e| format!("argon2 params: {e}"))?;
    let a2 = Argon2::new(argon2::Algorithm::Argon2id, argon2::Version::V0x13, params);
    let mut key = [0u8; 32];
    a2.hash_password_into(passphrase.as_bytes(), salt, &mut key)
        .map_err(|e| format!("argon2: {e}"))?;
    Ok(key)
}

pub struct ChatDb {
    conn: Connection,
    crypto: Crypto,
}

impl ChatDb {
    /// True if the file at `path` is an encrypted OnlyChat database.
    pub fn is_encrypted(path: &Path) -> bool {
        Connection::open(path)
            .ok()
            .and_then(|c| {
                c.query_row("SELECT value FROM meta WHERE key='enc'", [], |r| {
                    r.get::<_, i64>(0)
                })
                .optional()
                .ok()
                .flatten()
            })
            .map(|v| v != 0)
            .unwrap_or(false)
    }

    /// True if the file already is an OnlyChat database (has our schema).
    pub fn is_chatdb(path: &Path) -> bool {
        Connection::open(path)
            .ok()
            .map(|c| {
                c.query_row(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='meta'",
                    [],
                    |_| Ok(()),
                )
                .optional()
                .ok()
                .flatten()
                .is_some()
            })
            .unwrap_or(false)
    }

    fn schema(conn: &Connection) -> Result<(), String> {
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value BLOB);
             CREATE TABLE IF NOT EXISTS chats (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 name TEXT NOT NULL,
                 created INTEGER NOT NULL,
                 mode INTEGER NOT NULL DEFAULT 1,
                 summary BLOB,
                 summary_upto INTEGER NOT NULL DEFAULT 0);
             CREATE TABLE IF NOT EXISTS messages (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 chat_id INTEGER NOT NULL,
                 role INTEGER NOT NULL,
                 ts INTEGER NOT NULL,
                 content BLOB NOT NULL);
             CREATE INDEX IF NOT EXISTS idx_msg_chat ON messages(chat_id, id);",
        )
        .map_err(|e| format!("schema: {e}"))
    }

    /// Create a new database. `passphrase = Some(..)` turns on encryption.
    pub fn create(path: &Path, passphrase: Option<&str>) -> Result<ChatDb, String> {
        if path.exists() {
            return Err("Datei existiert bereits".into());
        }
        let conn = Connection::open(path).map_err(|e| format!("open: {e}"))?;
        Self::schema(&conn)?;
        let crypto = match passphrase {
            None => {
                conn.execute("INSERT INTO meta(key,value) VALUES('enc', 0)", [])
                    .map_err(|e| e.to_string())?;
                Crypto::Plain
            }
            Some(pw) => {
                let mut salt = [0u8; 16];
                rand::thread_rng().fill_bytes(&mut salt);
                let key = derive_key(pw, &salt)?;
                let cipher = XChaCha20Poly1305::new(Key::from_slice(&key));
                // Verifier: a known plaintext sealed with the key, to detect a
                // wrong passphrase on open.
                let mut nonce = [0u8; 24];
                rand::thread_rng().fill_bytes(&mut nonce);
                let ct = cipher
                    .encrypt(XNonce::from_slice(&nonce), VERIFIER_PLAINTEXT)
                    .map_err(|e| format!("seal verifier: {e}"))?;
                let mut verifier = nonce.to_vec();
                verifier.extend_from_slice(&ct);
                conn.execute("INSERT INTO meta(key,value) VALUES('enc', 1)", [])
                    .map_err(|e| e.to_string())?;
                conn.execute("INSERT INTO meta(key,value) VALUES('salt', ?1)", params![salt.to_vec()])
                    .map_err(|e| e.to_string())?;
                conn.execute("INSERT INTO meta(key,value) VALUES('verifier', ?1)", params![verifier])
                    .map_err(|e| e.to_string())?;
                Crypto::Sealed(Box::new(cipher))
            }
        };
        Ok(ChatDb { conn, crypto })
    }

    /// Open an existing database. For an encrypted one a correct `passphrase`
    /// must be given; a wrong one is rejected via the stored verifier.
    pub fn open(path: &Path, passphrase: Option<&str>) -> Result<ChatDb, String> {
        let conn = Connection::open(path).map_err(|e| format!("open: {e}"))?;
        Self::schema(&conn)?;
        let enc: i64 = conn
            .query_row("SELECT value FROM meta WHERE key='enc'", [], |r| r.get(0))
            .optional()
            .map_err(|e| e.to_string())?
            .unwrap_or(0);
        if enc == 0 {
            return Ok(ChatDb { conn, crypto: Crypto::Plain });
        }
        let pw = passphrase.ok_or("Datenbank ist verschlüsselt — Passwort nötig")?;
        let salt: Vec<u8> = conn
            .query_row("SELECT value FROM meta WHERE key='salt'", [], |r| r.get(0))
            .map_err(|e| e.to_string())?;
        let verifier: Vec<u8> = conn
            .query_row("SELECT value FROM meta WHERE key='verifier'", [], |r| r.get(0))
            .map_err(|e| e.to_string())?;
        let key = derive_key(pw, &salt)?;
        let cipher = XChaCha20Poly1305::new(Key::from_slice(&key));
        if verifier.len() < 24 {
            return Err("Datenbank beschädigt (verifier)".into());
        }
        let (nonce, ct) = verifier.split_at(24);
        cipher
            .decrypt(XNonce::from_slice(nonce), ct)
            .ok()
            .filter(|pt| pt == VERIFIER_PLAINTEXT)
            .ok_or("Falsches Passwort")?;
        Ok(ChatDb { conn, crypto: Crypto::Sealed(Box::new(cipher)) })
    }

    pub fn list_chats(&self) -> Vec<ChatMeta> {
        let mut out = Vec::new();
        if let Ok(mut stmt) = self.conn.prepare(
            "SELECT c.id, c.name, c.mode, (SELECT COUNT(*) FROM messages m WHERE m.chat_id=c.id)
             FROM chats c ORDER BY c.id DESC",
        ) {
            let rows = stmt.query_map([], |r| {
                Ok(ChatMeta {
                    id: r.get(0)?,
                    name: r.get(1)?,
                    mode: CtxMode::from_i32(r.get(2)?),
                    n_msgs: r.get(3)?,
                })
            });
            if let Ok(rows) = rows {
                out.extend(rows.flatten());
            }
        }
        out
    }

    pub fn new_chat(&self, name: &str, mode: CtxMode) -> Result<i64, String> {
        self.conn
            .execute(
                "INSERT INTO chats(name, created, mode) VALUES(?1, ?2, ?3)",
                params![name, now(), mode.to_i32()],
            )
            .map_err(|e| e.to_string())?;
        Ok(self.conn.last_insert_rowid())
    }

    pub fn delete_chat(&self, id: i64) {
        let _ = self.conn.execute("DELETE FROM messages WHERE chat_id=?1", params![id]);
        let _ = self.conn.execute("DELETE FROM chats WHERE id=?1", params![id]);
    }

    pub fn set_mode(&self, id: i64, mode: CtxMode) {
        let _ = self
            .conn
            .execute("UPDATE chats SET mode=?1 WHERE id=?2", params![mode.to_i32(), id]);
    }

    pub fn append(&self, chat_id: i64, role: Role, text: &str) {
        let _ = self.conn.execute(
            "INSERT INTO messages(chat_id, role, ts, content) VALUES(?1, ?2, ?3, ?4)",
            params![chat_id, role.to_i32(), now(), self.crypto.seal(text)],
        );
    }

    pub fn messages(&self, chat_id: i64) -> Vec<ChatMsg> {
        let mut out = Vec::new();
        if let Ok(mut stmt) = self
            .conn
            .prepare("SELECT role, content FROM messages WHERE chat_id=?1 ORDER BY id")
        {
            let rows = stmt.query_map(params![chat_id], |r| {
                let role: i32 = r.get(0)?;
                let blob: Vec<u8> = r.get(1)?;
                Ok((role, blob))
            });
            if let Ok(rows) = rows {
                for (role, blob) in rows.flatten() {
                    out.push(ChatMsg { role: Role::from_i32(role), text: self.crypto.open(&blob) });
                }
            }
        }
        out
    }

    pub fn summary(&self, chat_id: i64) -> Option<(String, i64)> {
        self.conn
            .query_row(
                "SELECT summary, summary_upto FROM chats WHERE id=?1",
                params![chat_id],
                |r| {
                    let blob: Option<Vec<u8>> = r.get(0)?;
                    let upto: i64 = r.get(1)?;
                    Ok((blob, upto))
                },
            )
            .optional()
            .ok()
            .flatten()
            .and_then(|(blob, upto)| blob.map(|b| (self.crypto.open(&b), upto)))
    }

    pub fn set_summary(&self, chat_id: i64, text: &str, upto: i64) {
        let _ = self.conn.execute(
            "UPDATE chats SET summary=?1, summary_upto=?2 WHERE id=?3",
            params![self.crypto.seal(text), upto, chat_id],
        );
    }

    /// Import an OpenAI/ChatGPT `conversations.json` export: each conversation
    /// becomes a chat. Returns the number of chats imported.
    pub fn import_chatgpt(&self, json_path: &Path) -> Result<usize, String> {
        let bytes = std::fs::read(json_path).map_err(|e| format!("read: {e}"))?;
        let root: serde_json::Value =
            serde_json::from_slice(&bytes).map_err(|e| format!("json: {e}"))?;
        let convos = root.as_array().ok_or("erwartet: JSON-Array von Chats")?;
        let mut imported = 0usize;
        for convo in convos {
            let title = convo
                .get("title")
                .and_then(|t| t.as_str())
                .unwrap_or("Importierter Chat")
                .to_string();
            let turns = extract_turns(convo);
            if turns.is_empty() {
                continue;
            }
            let chat_id = self.new_chat(&title, CtxMode::SummaryRetrieval)?;
            for (role, text) in turns {
                self.append(chat_id, role, &text);
            }
            imported += 1;
        }
        Ok(imported)
    }
}

/// Pull an ordered user/assistant transcript out of one ChatGPT conversation
/// node map (ordered by `create_time`).
fn extract_turns(convo: &serde_json::Value) -> Vec<(Role, String)> {
    let mut nodes: Vec<(f64, Role, String)> = Vec::new();
    if let Some(map) = convo.get("mapping").and_then(|m| m.as_object()) {
        for node in map.values() {
            let Some(msg) = node.get("message") else { continue };
            let role = match msg.get("author").and_then(|a| a.get("role")).and_then(|r| r.as_str()) {
                Some("user") => Role::User,
                Some("assistant") => Role::Assistant,
                _ => continue,
            };
            let text = msg
                .get("content")
                .and_then(|c| c.get("parts"))
                .and_then(|p| p.as_array())
                .map(|parts| {
                    parts
                        .iter()
                        .filter_map(|p| p.as_str())
                        .collect::<Vec<_>>()
                        .join("\n")
                })
                .unwrap_or_default();
            if text.trim().is_empty() {
                continue;
            }
            let ts = msg.get("create_time").and_then(|t| t.as_f64()).unwrap_or(0.0);
            nodes.push((ts, role, text));
        }
    }
    nodes.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    nodes.into_iter().map(|(_, r, t)| (r, t)).collect()
}

fn now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// How many recent turns the windowed modes keep verbatim.
pub const LAST_N: usize = 8;

/// Build the turns to feed the model for the next answer, given the chat's
/// mode. `msgs` is the full stored history (oldest first); `query` is the new
/// user message (already appended to the store by the caller). Returns
/// `(prelude, turns)` where `prelude` is an optional system note (summary) and
/// `turns` are `(is_user, text)` pairs ending with the current question.
pub fn assemble(
    msgs: &[ChatMsg],
    mode: CtxMode,
    summary: Option<&str>,
) -> (Option<String>, Vec<(bool, String)>) {
    let to_turns = |slice: &[ChatMsg]| -> Vec<(bool, String)> {
        slice.iter().map(|m| (m.role.is_user(), m.text.clone())).collect()
    };
    match mode {
        CtxMode::Full => (None, to_turns(msgs)),
        CtxMode::LastN => {
            let start = msgs.len().saturating_sub(LAST_N * 2);
            (None, to_turns(&msgs[start..]))
        }
        CtxMode::SummaryRetrieval => {
            let start = msgs.len().saturating_sub(LAST_N * 2);
            let recent = &msgs[start..];
            // Retrieval over the older part: score by shared lowercased words
            // with the current question, keep the best few. Lightweight on
            // purpose — no embeddings, no extra model pass.
            let query = recent.last().map(|m| m.text.clone()).unwrap_or_default();
            let qwords: std::collections::HashSet<String> =
                words(&query).into_iter().filter(|w| w.len() > 3).collect();
            let mut scored: Vec<(usize, &ChatMsg)> = msgs[..start]
                .iter()
                .map(|m| {
                    let s = words(&m.text).into_iter().filter(|w| qwords.contains(w)).count();
                    (s, m)
                })
                .filter(|(s, _)| *s > 0)
                .collect();
            scored.sort_by(|a, b| b.0.cmp(&a.0));
            let mut turns: Vec<(bool, String)> = scored
                .into_iter()
                .take(3)
                .map(|(_, m)| (m.role.is_user(), format!("[früher] {}", m.text)))
                .collect();
            turns.extend(to_turns(recent));
            let prelude = summary.map(|s| {
                format!("Zusammenfassung des bisherigen Gesprächs (ältere Teile):\n{s}")
            });
            (prelude, turns)
        }
    }
}

fn words(s: &str) -> Vec<String> {
    s.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(|w| w.to_string())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("overflow-chatdb-test-{}-{}.sqlite", std::process::id(), name));
        let _ = std::fs::remove_file(&p);
        p
    }

    #[test]
    fn plain_roundtrip() {
        let path = tmp("plain");
        let db = ChatDb::create(&path, None).unwrap();
        let id = db.new_chat("c", CtxMode::LastN).unwrap();
        db.append(id, Role::User, "héllo — wörld 日本語");
        db.append(id, Role::Assistant, "antwort");
        drop(db);
        let db = ChatDb::open(&path, None).unwrap();
        let msgs = db.messages(id);
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].text, "héllo — wörld 日本語");
        assert!(msgs[1].role == Role::Assistant);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn encrypted_roundtrip_and_wrong_pass() {
        let path = tmp("enc");
        {
            let db = ChatDb::create(&path, Some("correct horse")).unwrap();
            let id = db.new_chat("c", CtxMode::Full).unwrap();
            db.append(id, Role::User, "geheim");
        }
        assert!(ChatDb::is_encrypted(&path));
        assert!(ChatDb::open(&path, Some("wrong")).is_err());
        assert!(ChatDb::open(&path, None).is_err());
        let db = ChatDb::open(&path, Some("correct horse")).unwrap();
        let chats = db.list_chats();
        assert_eq!(chats.len(), 1);
        assert_eq!(db.messages(chats[0].id)[0].text, "geheim");
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn import_chatgpt_export() {
        let path = tmp("imp");
        let db = ChatDb::create(&path, None).unwrap();
        let export = r#"[
          {"title":"T1","mapping":{
            "a":{"message":{"author":{"role":"user"},"content":{"parts":["frage"]},"create_time":1.0}},
            "b":{"message":{"author":{"role":"assistant"},"content":{"parts":["antwort"]},"create_time":2.0}},
            "sys":{"message":{"author":{"role":"system"},"content":{"parts":["x"]},"create_time":0.5}}
          }}
        ]"#;
        let jp = tmp("export").with_extension("json");
        std::fs::write(&jp, export).unwrap();
        let n = db.import_chatgpt(&jp).unwrap();
        assert_eq!(n, 1);
        let chats = db.list_chats();
        let msgs = db.messages(chats[0].id);
        assert_eq!(msgs.len(), 2); // system dropped
        assert_eq!(msgs[0].text, "frage");
        assert_eq!(msgs[1].text, "antwort");
        std::fs::remove_file(&path).ok();
        std::fs::remove_file(&jp).ok();
    }

    #[test]
    fn assemble_lastn_windows() {
        let msgs: Vec<ChatMsg> = (0..40)
            .map(|i| ChatMsg {
                role: if i % 2 == 0 { Role::User } else { Role::Assistant },
                text: format!("m{i}"),
            })
            .collect();
        let (prelude, turns) = assemble(&msgs, CtxMode::LastN, None);
        assert!(prelude.is_none());
        assert_eq!(turns.len(), LAST_N * 2);
        assert_eq!(turns.last().unwrap().1, "m39");
    }
}
