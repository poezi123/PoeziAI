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
            CtxMode::Full => "Whole chat",
            CtxMode::LastN => "Last few messages",
            CtxMode::SummaryRetrieval => "Summary + lookup",
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
                c.query_row("SELECT value FROM oc_meta WHERE key='enc'", [], |r| {
                    r.get::<_, i64>(0)
                })
                .optional()
                .ok()
                .flatten()
            })
            .map(|v| v != 0)
            .unwrap_or(false)
    }

    /// Create OnlyChat's own tables. They are namespaced with an `oc_` prefix so
    /// the database file may already contain unrelated tables (an imported or
    /// pre-existing SQLite file): we add our tables alongside whatever is there
    /// and never touch, rename or read the user's own tables. This is why
    /// pointing OnlyChat at an arbitrary `.sqlite` is lossless.
    fn schema(conn: &Connection) -> Result<(), String> {
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS oc_meta (key TEXT PRIMARY KEY, value BLOB);
             CREATE TABLE IF NOT EXISTS oc_chats (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 name TEXT NOT NULL,
                 created INTEGER NOT NULL,
                 mode INTEGER NOT NULL DEFAULT 1,
                 summary BLOB,
                 summary_upto INTEGER NOT NULL DEFAULT 0);
             CREATE TABLE IF NOT EXISTS oc_messages (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 chat_id INTEGER NOT NULL,
                 role INTEGER NOT NULL,
                 ts INTEGER NOT NULL,
                 content BLOB NOT NULL);
             CREATE INDEX IF NOT EXISTS idx_oc_msg_chat ON oc_messages(chat_id, id);",
        )
        .map_err(|e| format!("schema: {e}"))
    }

    /// Create a new database. `passphrase = Some(..)` turns on encryption.
    pub fn create(path: &Path, passphrase: Option<&str>) -> Result<ChatDb, String> {
        if path.exists() {
            return Err("A file with this name already exists".into());
        }
        let conn = Connection::open(path).map_err(|e| format!("open: {e}"))?;
        Self::schema(&conn)?;
        let crypto = match passphrase {
            None => {
                conn.execute("INSERT INTO oc_meta(key,value) VALUES('enc', 0)", [])
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
                conn.execute("INSERT INTO oc_meta(key,value) VALUES('enc', 1)", [])
                    .map_err(|e| e.to_string())?;
                conn.execute("INSERT INTO oc_meta(key,value) VALUES('salt', ?1)", params![salt.to_vec()])
                    .map_err(|e| e.to_string())?;
                conn.execute("INSERT INTO oc_meta(key,value) VALUES('verifier', ?1)", params![verifier])
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
            .query_row("SELECT value FROM oc_meta WHERE key='enc'", [], |r| r.get(0))
            .optional()
            .map_err(|e| e.to_string())?
            .unwrap_or(0);
        if enc == 0 {
            return Ok(ChatDb { conn, crypto: Crypto::Plain });
        }
        let pw = passphrase.ok_or("This database is locked — a password is needed")?;
        let salt: Vec<u8> = conn
            .query_row("SELECT value FROM oc_meta WHERE key='salt'", [], |r| r.get(0))
            .map_err(|e| e.to_string())?;
        let verifier: Vec<u8> = conn
            .query_row("SELECT value FROM oc_meta WHERE key='verifier'", [], |r| r.get(0))
            .map_err(|e| e.to_string())?;
        let key = derive_key(pw, &salt)?;
        let cipher = XChaCha20Poly1305::new(Key::from_slice(&key));
        if verifier.len() < 24 {
            return Err("This database file looks damaged".into());
        }
        let (nonce, ct) = verifier.split_at(24);
        cipher
            .decrypt(XNonce::from_slice(nonce), ct)
            .ok()
            .filter(|pt| pt == VERIFIER_PLAINTEXT)
            .ok_or("Wrong password")?;
        Ok(ChatDb { conn, crypto: Crypto::Sealed(Box::new(cipher)) })
    }

    pub fn list_chats(&self) -> Vec<ChatMeta> {
        let mut out = Vec::new();
        if let Ok(mut stmt) = self.conn.prepare(
            "SELECT c.id, c.name, c.mode, (SELECT COUNT(*) FROM oc_messages m WHERE m.chat_id=c.id)
             FROM oc_chats c ORDER BY c.id DESC",
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
                "INSERT INTO oc_chats(name, created, mode) VALUES(?1, ?2, ?3)",
                params![name, now(), mode.to_i32()],
            )
            .map_err(|e| e.to_string())?;
        Ok(self.conn.last_insert_rowid())
    }

    pub fn delete_chat(&self, id: i64) {
        let _ = self.conn.execute("DELETE FROM oc_messages WHERE chat_id=?1", params![id]);
        let _ = self.conn.execute("DELETE FROM oc_chats WHERE id=?1", params![id]);
    }

    pub fn set_mode(&self, id: i64, mode: CtxMode) {
        let _ = self
            .conn
            .execute("UPDATE oc_chats SET mode=?1 WHERE id=?2", params![mode.to_i32(), id]);
    }

    pub fn append(&self, chat_id: i64, role: Role, text: &str) {
        let _ = self.conn.execute(
            "INSERT INTO oc_messages(chat_id, role, ts, content) VALUES(?1, ?2, ?3, ?4)",
            params![chat_id, role.to_i32(), now(), self.crypto.seal(text)],
        );
    }

    pub fn messages(&self, chat_id: i64) -> Vec<ChatMsg> {
        let mut out = Vec::new();
        if let Ok(mut stmt) = self
            .conn
            .prepare("SELECT role, content FROM oc_messages WHERE chat_id=?1 ORDER BY id")
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
                "SELECT summary, summary_upto FROM oc_chats WHERE id=?1",
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
            "UPDATE oc_chats SET summary=?1, summary_upto=?2 WHERE id=?3",
            params![self.crypto.seal(text), upto, chat_id],
        );
    }

    /// Import an OpenAI/ChatGPT `conversations.json` export: each conversation
    /// becomes a chat. Returns the number of chats imported. (Kept for reuse;
    /// the GUI currently imports existing data via `import_foreign` instead.)
    #[allow(dead_code)]
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

    /// Whether this database is encrypted at rest.
    pub fn encrypted(&self) -> bool {
        matches!(self.crypto, Crypto::Sealed(_))
    }

    pub fn rename_chat(&self, id: i64, name: &str) {
        let _ = self
            .conn
            .execute("UPDATE oc_chats SET name=?1 WHERE id=?2", params![name, id]);
    }

    /// Read the database's own (non-`oc_`, non-`sqlite_`) tables and turn each
    /// into a chat, so the AI can see and continue from data that was already in
    /// the file. Heuristic and schema-agnostic: the most text-like column is the
    /// message content, a role/author/sender column (if any) sets the speaker
    /// else roles alternate, and rows are ordered by an id/time-like column or
    /// `rowid`. Foreign tables are only read, never modified. Returns how many
    /// tables yielded at least one message. (Kept for conversation-style files;
    /// structured memory DBs use `memory_context` instead.)
    #[allow(dead_code)]
    pub fn import_foreign(&self) -> Result<usize, String> {
        let tables: Vec<String> = {
            let mut stmt = self
                .conn
                .prepare(
                    "SELECT name FROM sqlite_master WHERE type='table' \
                     AND name NOT LIKE 'oc\\_%' ESCAPE '\\' \
                     AND name NOT LIKE 'sqlite\\_%' ESCAPE '\\'",
                )
                .map_err(|e| e.to_string())?;
            let rows = stmt
                .query_map([], |r| r.get::<_, String>(0))
                .map_err(|e| e.to_string())?;
            rows.flatten().collect()
        };
        let lc = |s: &str| s.to_lowercase();
        let quote = |c: &str| format!("\"{}\"", c.replace('"', "\"\""));
        let mut imported = 0usize;
        for table in tables {
            let cols: Vec<(String, String)> = {
                let mut stmt = match self.conn.prepare(&format!("PRAGMA table_info({})", quote(&table))) {
                    Ok(s) => s,
                    Err(_) => continue,
                };
                let rows = stmt.query_map([], |r| Ok((r.get::<_, String>(1)?, r.get::<_, String>(2)?)));
                match rows {
                    Ok(rows) => rows.flatten().collect(),
                    Err(_) => continue,
                }
            };
            if cols.is_empty() {
                continue;
            }
            let content_col = cols
                .iter()
                .map(|(n, _)| n.clone())
                .find(|n| {
                    ["content", "text", "message", "body", "msg", "value", "data", "prompt", "response"]
                        .contains(&lc(n).as_str())
                })
                .or_else(|| {
                    cols.iter()
                        .find(|(_, t)| lc(t).contains("text") || lc(t).contains("char") || lc(t).contains("clob"))
                        .map(|(n, _)| n.clone())
                })
                .or_else(|| cols.first().map(|(n, _)| n.clone()));
            let Some(content_col) = content_col else { continue };
            let role_col = cols.iter().map(|(n, _)| n.clone()).find(|n| {
                ["role", "author", "sender", "speaker", "is_user", "from", "who", "type"]
                    .contains(&lc(n).as_str())
            });
            let order_col = cols
                .iter()
                .map(|(n, _)| n.clone())
                .find(|n| {
                    [
                        "id", "created", "created_at", "timestamp", "ts", "time", "date",
                        "create_time", "idx", "seq", "position",
                    ]
                    .contains(&lc(n).as_str())
                })
                .unwrap_or_else(|| "rowid".to_string());

            let sql = match &role_col {
                Some(rc) => format!(
                    "SELECT {}, {} FROM {} ORDER BY {}",
                    quote(rc), quote(&content_col), quote(&table), quote(&order_col)
                ),
                None => format!(
                    "SELECT {} FROM {} ORDER BY {}",
                    quote(&content_col), quote(&table), quote(&order_col)
                ),
            };
            let has_role = role_col.is_some();
            let mut stmt = match self.conn.prepare(&sql) {
                Ok(s) => s,
                Err(_) => continue,
            };
            let rows = stmt.query_map([], |r| {
                if has_role {
                    Ok((Some(value_to_string(r.get_ref(0)?)), value_to_string(r.get_ref(1)?)))
                } else {
                    Ok((None, value_to_string(r.get_ref(0)?)))
                }
            });
            let Ok(rows) = rows else { continue };
            let collected: Vec<(Option<String>, String)> = rows.flatten().collect();

            let mut created: Option<i64> = None;
            let mut alt_user = true;
            let mut appended = 0usize;
            for (role_val, content) in collected {
                if content.trim().is_empty() {
                    continue;
                }
                let role = match role_val.as_deref().map(str::to_lowercase) {
                    Some(s) if s.contains("user") || s.contains("human") || s == "you" || s == "1" => {
                        Role::User
                    }
                    Some(s)
                        if s.contains("assist")
                            || s.contains("ai")
                            || s.contains("bot")
                            || s.contains("gpt")
                            || s.contains("system")
                            || s == "0" =>
                    {
                        Role::Assistant
                    }
                    _ => {
                        let r = if alt_user { Role::User } else { Role::Assistant };
                        alt_user = !alt_user;
                        r
                    }
                };
                let cid = match created {
                    Some(c) => c,
                    None => {
                        let c = self.new_chat(&format!("[DB] {table}"), CtxMode::SummaryRetrieval)?;
                        created = Some(c);
                        c
                    }
                };
                self.append(cid, role, &content);
                appended += 1;
            }
            if appended > 0 {
                imported += 1;
            }
        }
        Ok(imported)
    }
}

impl ChatDb {
    /// Build a knowledge/context block from a structured "memory" database (the
    /// kind with facts/chunks/memory_packs/projects/… tables, optionally with
    /// FTS5 indexes). Curated rows plus query-relevant chunks, char-budgeted so
    /// it never blows the context window. Returns None if the file has none of
    /// these tables (then there is nothing to inject). Read-only.
    pub fn memory_context(&self, query: &str, budget: usize) -> Option<String> {
        let table_exists = |t: &str| -> bool {
            self.conn
                .query_row(
                    "SELECT 1 FROM sqlite_master WHERE type IN ('table','view') AND name=?1",
                    params![t],
                    |_| Ok(()),
                )
                .optional()
                .ok()
                .flatten()
                .is_some()
        };
        // Collect rows from a query into "- col1 col2 …" lines, ignoring errors.
        let rows = |sql: &str, ncols: usize| -> Vec<String> {
            let mut out = Vec::new();
            if let Ok(mut stmt) = self.conn.prepare(sql) {
                if let Ok(mut r) = stmt.query([]) {
                    while let Ok(Some(row)) = r.next() {
                        let parts: Vec<String> = (0..ncols)
                            .filter_map(|i| row.get_ref(i).ok().map(value_to_string))
                            .filter(|s| !s.trim().is_empty())
                            .collect();
                        if !parts.is_empty() {
                            out.push(format!("- {}", parts.join(" — ")));
                        }
                    }
                }
            }
            out
        };

        let mut sections: Vec<(String, Vec<String>)> = Vec::new();
        if table_exists("memory_packs") {
            sections.push(("Memory-Packs".into(), rows("SELECT name, content FROM memory_packs", 2)));
        }
        if table_exists("preferences") {
            sections.push(("Präferenzen".into(), rows("SELECT key, value FROM preferences", 2)));
        }
        if table_exists("facts") {
            sections.push((
                "Fakten".into(),
                rows("SELECT subject, predicate, value FROM facts WHERE COALESCE(is_current,1)=1 ORDER BY COALESCE(priority,0) DESC LIMIT 200", 3),
            ));
        }
        if table_exists("projects") {
            sections.push(("Projekte".into(), rows("SELECT name, status, description FROM projects", 3)));
        }
        if table_exists("skills") {
            sections.push(("Skills".into(), rows("SELECT category, name FROM skills", 2)));
        }
        if table_exists("entities") {
            sections.push(("Entitäten".into(), rows("SELECT name, kind, description FROM entities", 3)));
        }
        if table_exists("events") {
            sections.push((
                "Ereignisse".into(),
                rows("SELECT event_date, title, description FROM events ORDER BY event_date DESC LIMIT 25", 3),
            ));
        }

        // Query-relevant document chunks via FTS5 if available.
        if table_exists("chunks_fts") || table_exists("chunks") {
            let terms: Vec<String> = query
                .to_lowercase()
                .split(|c: char| !c.is_alphanumeric())
                .filter(|w| w.len() > 3)
                .take(12)
                .map(|w| w.to_string())
                .collect();
            if !terms.is_empty() && table_exists("chunks_fts") {
                let match_expr = terms.join(" OR ").replace('\'', " ");
                let sql = format!(
                    "SELECT body FROM chunks_fts WHERE chunks_fts MATCH '{match_expr}' LIMIT 6"
                );
                let hits = rows(&sql, 1);
                if !hits.is_empty() {
                    sections.push(("Passende Notizen".into(), hits));
                }
            }
        }

        if sections.iter().all(|(_, v)| v.is_empty()) {
            return None;
        }

        let mut out = String::from(
            "Gespeichertes Wissen über den Nutzer und seine Projekte (aus der Datenbank). \
             Nutze es, um die Fragen des Nutzers konkret zu beantworten:\n",
        );
        'outer: for (title, lines) in sections {
            if lines.is_empty() {
                continue;
            }
            out.push_str(&format!("\n[{title}]\n"));
            for l in lines {
                if out.len() + l.len() + 1 > budget {
                    out.push_str("…\n");
                    break 'outer;
                }
                out.push_str(&l);
                out.push('\n');
            }
        }
        Some(out)
    }
}

/// Render a SQLite value as display text (for reading foreign tables).
fn value_to_string(v: rusqlite::types::ValueRef) -> String {
    use rusqlite::types::ValueRef;
    match v {
        ValueRef::Null => String::new(),
        ValueRef::Integer(i) => i.to_string(),
        ValueRef::Real(f) => f.to_string(),
        ValueRef::Text(t) => String::from_utf8_lossy(t).into_owned(),
        ValueRef::Blob(_) => String::new(),
    }
}

/// Pull an ordered user/assistant transcript out of one ChatGPT conversation
/// node map (ordered by `create_time`).
#[allow(dead_code)]
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
    fn coexists_with_foreign_schema() {
        // A pre-existing SQLite file with its own, differently-shaped `messages`
        // and `chats` tables (the reported crash: no column `chat_id`). Opening
        // it must succeed, leave the foreign data untouched, and still work.
        let path = tmp("foreign");
        {
            let c = Connection::open(&path).unwrap();
            c.execute_batch(
                "CREATE TABLE messages (id INTEGER PRIMARY KEY, body TEXT);
                 INSERT INTO messages(body) VALUES('users own row');
                 CREATE TABLE chats (foo TEXT);
                 INSERT INTO chats(foo) VALUES('bar');",
            )
            .unwrap();
        }
        // Open via OnlyChat: adds oc_* tables alongside, no collision.
        let db = ChatDb::open(&path, None).unwrap();
        let id = db.new_chat("neu", CtxMode::LastN).unwrap();
        db.append(id, Role::User, "hallo");
        assert_eq!(db.messages(id)[0].text, "hallo");
        drop(db);
        // The user's original tables and rows are intact and unmodified.
        let c = Connection::open(&path).unwrap();
        let body: String =
            c.query_row("SELECT body FROM messages", [], |r| r.get(0)).unwrap();
        assert_eq!(body, "users own row");
        let foo: String = c.query_row("SELECT foo FROM chats", [], |r| r.get(0)).unwrap();
        assert_eq!(foo, "bar");
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn import_foreign_reads_user_table() {
        let path = tmp("fimport");
        {
            let c = Connection::open(&path).unwrap();
            c.execute_batch(
                "CREATE TABLE conversation (id INTEGER PRIMARY KEY, role TEXT, content TEXT);
                 INSERT INTO conversation(role,content) VALUES('user','frage eins');
                 INSERT INTO conversation(role,content) VALUES('assistant','antwort eins');",
            )
            .unwrap();
        }
        let db = ChatDb::open(&path, None).unwrap();
        let n = db.import_foreign().unwrap();
        assert_eq!(n, 1);
        let chats = db.list_chats();
        assert_eq!(chats.len(), 1);
        let msgs = db.messages(chats[0].id);
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].text, "frage eins");
        assert!(msgs[0].role == Role::User);
        assert!(msgs[1].role == Role::Assistant);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn memory_context_from_structured_db() {
        let path = tmp("mem");
        {
            let c = Connection::open(&path).unwrap();
            c.execute_batch(
                "CREATE TABLE facts (subject TEXT, predicate TEXT, value TEXT, is_current INTEGER, priority INTEGER);
                 INSERT INTO facts VALUES('Leon','lebt in','Österreich',1,5);
                 INSERT INTO facts VALUES('Leon','Projekt','OVERFLOW',1,9);
                 CREATE TABLE memory_packs (name TEXT, content TEXT);
                 INSERT INTO memory_packs VALUES('Profil','Leon baut eine lokale KI namens OVERFLOW.');",
            )
            .unwrap();
        }
        let db = ChatDb::open(&path, None).unwrap();
        let ctx = db.memory_context("Was ist OVERFLOW?", 8000).expect("some context");
        assert!(ctx.contains("OVERFLOW"));
        assert!(ctx.contains("Fakten") && ctx.contains("Memory-Packs"));
        // A plain db with no knowledge tables yields None.
        let p2 = tmp("mem_none");
        let db2 = ChatDb::create(&p2, None).unwrap();
        assert!(db2.memory_context("x", 8000).is_none());
        std::fs::remove_file(&path).ok();
        std::fs::remove_file(&p2).ok();
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
