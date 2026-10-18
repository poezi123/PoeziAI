//! OVERFLOW — a local AI, native desktop GUI (egui/eframe).
//!
//! The window is a thin client: it renders the transcript and forwards messages
//! to the inference worker thread, streaming tokens back into the view. No web
//! stack, pure Rust. The whole look is drawn from the OVERFLOW palette.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod chatdb;
mod hw;
mod trainer;
mod worker;

use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use eframe::egui;
use egui::{Color32, FontFamily, Pos2, Sense, Stroke, Vec2};
use poezi_model::agent::GenCfg;

use worker::{FromWorker, Handle, ToWorker};

#[derive(PartialEq, Clone, Copy)]
enum Tab {
    Chat,
    OnlyChat,
    Learning,
    Docs,
}

/// Where streamed tokens from the worker should land. OnlyChat shares the same
/// worker/model as Chat, so a turn tags its destination to route the stream.
#[derive(PartialEq, Clone, Copy)]
enum Stream {
    Chat,
    OnlyChat,
}

/// OnlyChat tab state: an open chat database (optional), its chat list, the
/// active chat's transcript, and the new-chat / open-database inputs.
/// One open chat database (its file path + handle). Many may be open at once;
/// each chat belongs to exactly one, chosen when the chat is created.
struct OpenDb {
    path: String,
    db: chatdb::ChatDb,
}

/// Which database a new chat should go into.
#[derive(Clone, PartialEq)]
enum DbPick {
    /// An already-open database, by index into `dbs`.
    Existing(usize),
    /// A brand-new database file (path + optional encryption).
    NewFile,
}

struct OnlyChatState {
    /// All currently open databases. A chat lives in exactly one of them.
    dbs: Vec<OpenDb>,
    status: String,
    /// Flattened chat list across all open dbs: (db index, chat meta).
    chats: Vec<(usize, chatdb::ChatMeta)>,
    /// Active chat as (db index, chat id).
    active: Option<(usize, i64)>,
    active_mode: chatdb::CtxMode,
    msgs: Vec<chatdb::ChatMsg>,
    input: String,
    stream_buf: String,
    generating: bool,
    summarizing: bool,
    /// Token counts for the active chat: whole chat vs. what the context mode
    /// actually feeds the model (shown in the dev panel).
    tok_full: Option<usize>,
    tok_ctx: Option<usize>,
    /// Chat being renamed: (db index, chat id, name buffer).
    rename: Option<(usize, i64, String)>,
    /// Chat pending delete confirmation: (db index, chat id).
    confirm_delete: Option<(usize, i64)>,
    // new-chat form
    show_new: bool,
    new_name: String,
    new_mode: chatdb::CtxMode,
    new_db_pick: DbPick,
    new_file_path: String,
    new_file_encrypt: bool,
    new_file_pass: String,
    // open-existing-database form
    show_open: bool,
    open_path: String,
    open_pass: String,
}

impl Default for OnlyChatState {
    fn default() -> Self {
        OnlyChatState {
            dbs: Vec::new(),
            status: String::new(),
            chats: Vec::new(),
            active: None,
            active_mode: chatdb::CtxMode::LastN,
            msgs: Vec::new(),
            input: String::new(),
            stream_buf: String::new(),
            generating: false,
            summarizing: false,
            tok_full: None,
            tok_ctx: None,
            rename: None,
            confirm_delete: None,
            show_new: false,
            new_name: String::new(),
            new_mode: chatdb::CtxMode::SummaryRetrieval,
            new_db_pick: DbPick::NewFile,
            new_file_path: default_db_path(),
            new_file_encrypt: false,
            new_file_pass: String::new(),
            show_open: false,
            open_path: default_db_path(),
            open_pass: String::new(),
        }
    }
}

/// Drop any `<think>…</think>` block from a model answer before it is stored or
/// shown in OnlyChat (the reasoning is internal, the stored turn is the reply).
fn strip_think(s: &str) -> String {
    let mut out = String::new();
    let mut rest = s;
    while let Some(start) = rest.find("<think>") {
        out.push_str(&rest[..start]);
        if let Some(end) = rest[start..].find("</think>") {
            rest = &rest[start + end + "</think>".len()..];
        } else {
            rest = "";
            break;
        }
    }
    out.push_str(rest);
    out.trim().to_string()
}

/// Default location for a new OnlyChat database (under the data dir).
fn default_db_path() -> String {
    data_models_dir()
        .parent()
        .map(|p| p.join("chats.sqlite"))
        .unwrap_or_else(|| PathBuf::from("chats.sqlite"))
        .display()
        .to_string()
}

/// Short display name for a database path (its file name).
fn db_name(path: &str) -> String {
    std::path::Path::new(path)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or(path)
        .to_string()
}

/// State for the Learning tab (drives the Python QLoRA pipeline).
/// Pre-made datasets you can pick in the Learning tab: (label, jsonl path, generator script).
/// The first is the default — OVERFLOW's strong infosec-professor set.
const DATASETS: &[(&str, &str, &str)] = &[
    ("OVERFLOW_WT · Infosec professor + offensive", "training/data/overflow_wt.jsonl", "training/data_gen_wt.py"),
    ("Identity + security basics (small)", "training/data/overflow.jsonl", "training/data_gen.py"),
];

struct LearnState {
    base: String,
    data: String,
    gen_script: String,
    epochs: String,
    loss: Option<f32>,
    step: Option<(u32, u32)>,
    status: String,
    /// Base label of the running task (e.g. "download 7B"), without the live
    /// percentage that gets appended to `status`.
    task: String,
    /// Download progress in percent, when the task reports one.
    pct: Option<f32>,
}

impl Default for LearnState {
    fn default() -> Self {
        LearnState {
            base: "Qwen/Qwen2.5-Coder-7B-Instruct".into(),
            data: DATASETS[0].1.into(),
            gen_script: DATASETS[0].2.into(),
            epochs: "3".into(),
            loss: None,
            step: None,
            status: "idle".into(),
            task: String::new(),
            pct: None,
        }
    }
}

// --- OVERFLOW palette ---
const BG: Color32 = Color32::from_rgb(0x07, 0x0b, 0x0d);
const PANEL: Color32 = Color32::from_rgb(0x0d, 0x13, 0x16);
const SURFACE: Color32 = Color32::from_rgb(0x06, 0x0a, 0x0b);
const LINE: Color32 = Color32::from_rgb(0x1c, 0x2a, 0x2b);
const FG: Color32 = Color32::from_rgb(0xd7, 0xe0, 0xdd);
const MUTED: Color32 = Color32::from_rgb(0x6f, 0x84, 0x81);
const AMBER: Color32 = Color32::from_rgb(0xf5, 0xb9, 0x4d);
const GREEN: Color32 = Color32::from_rgb(0x4d, 0xf0, 0xa0);
const RED: Color32 = Color32::from_rgb(0xff, 0x5a, 0x5a);
const CYAN: Color32 = Color32::from_rgb(0x49, 0xd6, 0xe6);

// Qwen2.5-Coder GGUF downloads (our engine is Qwen2-architecture).
const QWEN_1_5B_URL: &str = "https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF/resolve/main/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";
const QWEN_7B_URL: &str = "https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF/resolve/main/qwen2.5-coder-7b-instruct-q4_k_m.gguf";

/// Split assistant text into (thinking, answer) around a `<think>…</think>` block.
fn split_think(s: &str) -> (Option<&str>, &str) {
    if let Some(start) = s.find("<think>") {
        let after = &s[start + 7..];
        if let Some(end) = after.find("</think>") {
            return (Some(&after[..end]), &after[end + 8..]);
        }
        return (Some(after), ""); // still thinking (no closing tag yet)
    }
    (None, s)
}

/// Last path segment of a URL (before any query), for a download filename.
fn filename_from_url(url: &str) -> String {
    url.split('?')
        .next()
        .unwrap_or(url)
        .rsplit('/')
        .next()
        .filter(|s| s.ends_with(".gguf") && !s.is_empty())
        .unwrap_or("download.gguf")
        .to_string()
}

/// The math behind what the engine does — shown in the dev panel.
const FORMULAS: [&str; 6] = [
    "softmax   pᵢ = e^(zᵢ/T) / Σⱼ e^(zⱼ/T)",
    "RMSNorm   y = x·w / √(mean(x²)+ε)",
    "RoPE      rotate (xₖ, xₖ₊d/2) by θ·pos",
    "attention softmax(Q·Kᵀ/√d)·V",
    "SwiGLU    SiLU(x·Wg) ⊙ (x·Wu)",
    "loss      L = −Σ y·log p   (cross-entropy)",
];

use std::path::{Path, PathBuf};

/// Standard per-user data dir for models: ./models in a dev checkout, else
/// $XDG_DATA_HOME/overflow/models (defaulting to ~/.local/share). No stray dirs.
fn data_models_dir() -> PathBuf {
    if Path::new("models").is_dir() {
        return PathBuf::from("models");
    }
    let base = std::env::var_os("XDG_DATA_HOME")
        .map(PathBuf::from)
        .filter(|p| !p.as_os_str().is_empty())
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local/share")))
        .unwrap_or_else(|| PathBuf::from("."));
    base.join("overflow/models")
}

/// All .gguf models found in the dev ./models dir, the XDG data dir, and the
/// system dir where the Arch package installs the bundled model.
fn list_gguf() -> Vec<PathBuf> {
    let mut dirs = vec![
        PathBuf::from("models"),
        data_models_dir(),
        PathBuf::from("/usr/share/overflow/models"),
    ];
    dirs.dedup();
    let mut out = Vec::new();
    for d in dirs {
        if let Ok(rd) = std::fs::read_dir(&d) {
            for e in rd.flatten() {
                let p = e.path();
                if p.extension().map(|x| x == "gguf").unwrap_or(false) {
                    out.push(p);
                }
            }
        }
    }
    out.sort();
    out.dedup();
    out
}

/// Model to auto-load at startup: an explicit arg wins; else the best available.
fn default_model_path() -> Option<PathBuf> {
    if let Some(arg) = std::env::args().nth(1) {
        return Some(arg.into());
    }
    let prefs = [
        "overflow_wt-7b-q8_0.gguf",
        "qwen2.5-coder-7b-instruct-q4_k_m.gguf",
        "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
        "qwen2.5-coder-1.5b-instruct-q8_0.gguf",
    ];
    let all = list_gguf();
    for name in prefs {
        if let Some(p) = all.iter().find(|p| p.file_name().map(|n| n == name).unwrap_or(false)) {
            return Some(p.clone());
        }
    }
    all.into_iter().next()
}

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([920.0, 660.0])
            .with_min_inner_size([520.0, 420.0])
            .with_title("OVERFLOW"),
        ..Default::default()
    };
    eframe::run_native(
        "OVERFLOW",
        options,
        Box::new(|cc| {
            install_theme(&cc.egui_ctx);
            Ok(Box::new(OverflowApp::new(default_model_path())))
        }),
    )
}

enum Status {
    NoModel,
    Loading,
    Ready,
    Generating,
    Error(String),
}

enum Role {
    User,
    Assistant,
    Tool,
}

struct Msg {
    role: Role,
    text: String,
}

/// What the AI is allowed to do on the machine. OVERFLOW is a tool for security
/// work, so the default is UNRESTRICTED: it may read/write files, run shell
/// commands and use the network out of the box — but still asks before touching
/// anything outside the workspace folder (Claude-style). Flip any of these off
/// in Settings to tighten it, or turn on `sandbox` to isolate it fully.
struct Permissions {
    sandbox: bool,
    fs_read: bool,
    fs_write: bool,
    shell: bool,
    network: bool,
    /// Folder the AI may act within freely.
    workspace: String,
    /// Ask for confirmation before anything outside the workspace folder.
    confirm_outside: bool,
}

impl Default for Permissions {
    fn default() -> Self {
        let workspace = std::env::current_dir()
            .ok()
            .map(|p| p.display().to_string())
            .or_else(|| std::env::var("HOME").ok())
            .unwrap_or_else(|| ".".into());
        Permissions {
            sandbox: false,
            fs_read: true,
            fs_write: true,
            shell: true,
            network: true,
            workspace,
            confirm_outside: true,
        }
    }
}

struct OverflowApp {
    handle: Option<Handle>,
    current_model: Option<String>,
    status: Status,
    transcript: Vec<Msg>,
    input: String,
    settings_open: bool,
    perms: Permissions,
    tok_per_s: f32,
    backend: String,
    pending_approval: Option<(String, bool)>,
    hw: Arc<Mutex<hw::HwStats>>,
    tab: Tab,
    dev: bool,
    console: Vec<String>,
    /// Raw model output of the current turn (tokens as generated, including
    /// <think>/<tool_call> tags) — shown in the dev panel.
    raw_stream: String,
    learn: LearnState,
    trainer: Option<trainer::Trainer>,
    dl_url: String,
    gen: GenCfg,
    show_thinking: bool,
    only: OnlyChatState,
    stream: Stream,
}

impl OverflowApp {
    fn new(model_path: Option<PathBuf>) -> Self {
        let (handle, current_model, status) = match model_path {
            Some(p) => (Some(worker::spawn(p.clone())), Some(p.display().to_string()), Status::Loading),
            None => (None, None, Status::NoModel),
        };
        OverflowApp {
            handle,
            current_model,
            status,
            transcript: Vec::new(),
            input: String::new(),
            settings_open: false,
            perms: Permissions::default(),
            tok_per_s: 0.0,
            backend: "…".to_string(),
            pending_approval: None,
            hw: hw::spawn(),
            tab: Tab::Chat,
            dev: false,
            console: Vec::new(),
            raw_stream: String::new(),
            learn: LearnState::default(),
            trainer: None,
            dl_url: String::new(),
            gen: GenCfg::default(),
            show_thinking: true,
            only: OnlyChatState::default(),
            stream: Stream::Chat,
        }
    }

    /// Load a model (respawns the worker). Frees any previous one first.
    fn load_model(&mut self, path: PathBuf) {
        self.handle = None; // drop old worker -> frees its model/VRAM
        self.log(format!("loading model {}", path.display()));
        self.handle = Some(worker::spawn(path.clone()));
        self.current_model = Some(path.display().to_string());
        self.status = Status::Loading;
        self.backend = "…".into();
        self.transcript.clear();
        self.tab = Tab::Chat;
    }

    /// Unload the current model and free its memory.
    fn unload_model(&mut self) {
        self.handle = None;
        self.current_model = None;
        self.status = Status::NoModel;
        self.backend = "…".into();
        self.log("model unloaded");
    }

    /// Download a model (by URL) into the data dir, streamed via curl.
    fn download_model(&mut self, label: &str, url: &str, filename: &str) {
        let dir = data_models_dir();
        let _ = std::fs::create_dir_all(&dir);
        let out = dir.join(filename);
        self.start(
            label,
            "curl",
            vec![
                "-L".into(),
                "--fail".into(),
                "--progress-bar".into(), // emit a parseable progress meter (redrawn with \r)
                "-o".into(),
                out.display().to_string(),
                url.to_string(),
            ],
        );
    }

    /// Default base model (Qwen2.5-Coder 1.5B) — used by the first-run button.
    fn download_base(&mut self) {
        self.download_model("download base 1.5B", QWEN_1_5B_URL, "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf");
    }

    fn log(&mut self, line: impl Into<String>) {
        self.console.push(line.into());
        let n = self.console.len();
        if n > 800 {
            self.console.drain(..n - 800);
        }
    }

    /// Project root (where the training scripts expect to run).
    fn cwd() -> std::path::PathBuf {
        std::env::current_dir().unwrap_or_else(|_| ".".into())
    }

    fn venv_python() -> std::path::PathBuf {
        Self::cwd().join("training/venv/bin/python")
    }

    fn start(&mut self, label: &str, program: &str, args: Vec<String>) {
        if self.trainer.is_some() {
            self.log("a task is already running");
            return;
        }
        let envs = vec![(
            "HF_HOME".to_string(),
            Self::cwd().join("training/hf-cache").display().to_string(),
        )];
        self.log(format!("$ {label}"));
        self.learn.status = format!("running: {label}");
        self.learn.task = label.to_string();
        self.learn.pct = None;
        self.learn.loss = None;
        self.learn.step = None;
        self.trainer = Some(trainer::run(program, args, envs, Self::cwd()));
    }

    fn drain_trainer(&mut self) {
        let mut done: Option<bool> = None;
        if let Some(t) = &self.trainer {
            while let Ok(msg) = t.rx.try_recv() {
                match msg {
                    trainer::TrainerMsg::Line(l) => {
                        if trainer::is_progress(&l) {
                            // A download/progress meter: update the percentage and
                            // collapse consecutive meter lines into one that
                            // redraws in place, exactly like a terminal.
                            if let Some(pct) = trainer::parse_percent(&l) {
                                self.learn.pct = Some(pct);
                                self.learn.status = format!("running: {} — {pct:.0}%", self.learn.task);
                            }
                            if self.console.last().map(|p| trainer::is_progress(p)).unwrap_or(false) {
                                if let Some(last) = self.console.last_mut() {
                                    *last = l;
                                }
                            } else {
                                self.console.push(l);
                            }
                        } else {
                            if let Some(loss) = trainer::parse_loss(&l) {
                                self.learn.loss = Some(loss);
                            }
                            if let Some(p) = trainer::parse_progress(&l) {
                                self.learn.step = Some(p);
                            }
                            self.console.push(l);
                        }
                    }
                    trainer::TrainerMsg::Done(ok) => done = Some(ok),
                }
            }
        }
        let n = self.console.len();
        if n > 800 {
            self.console.drain(..n - 800);
        }
        if let Some(ok) = done {
            self.learn.status = if ok { "done ✓".into() } else { "failed ✗".into() };
            self.log(if ok { "task finished" } else { "task failed" });
            self.trainer = None;
        }
    }

    fn send_current_input(&mut self) {
        let text = self.input.trim().to_string();
        if text.is_empty() {
            return;
        }
        let Some(handle) = &self.handle else {
            self.log("no model loaded — load one in settings");
            return;
        };
        let perms = worker::perms(
            self.perms.sandbox,
            self.perms.fs_read,
            self.perms.fs_write,
            self.perms.shell,
            self.perms.network,
            &self.perms.workspace,
            self.perms.confirm_outside,
        );
        handle.cancel.store(false, Ordering::Relaxed);
        let _ = handle.tx.send(ToWorker::User { text: text.clone(), perms, gen: self.gen.clone() });
        self.raw_stream.clear();
        self.transcript.push(Msg { role: Role::User, text });
        self.transcript.push(Msg { role: Role::Assistant, text: String::new() });
        self.status = Status::Generating;
        self.input.clear();
    }

    // ---- OnlyChat helpers -------------------------------------------------

    /// Reload the chat list and the active chat's transcript from the DB.
    /// Rebuild the flattened chat list from all open databases and reload the
    /// active chat's transcript, then refresh its token counts.
    fn only_reload(&mut self) {
        let mut chats = Vec::new();
        for (i, od) in self.only.dbs.iter().enumerate() {
            for meta in od.db.list_chats() {
                chats.push((i, meta));
            }
        }
        self.only.chats = chats;
        // Drop the active selection if its chat no longer exists.
        if let Some((di, id)) = self.only.active {
            if !self.only.chats.iter().any(|(i, m)| *i == di && m.id == id) {
                self.only.active = None;
            }
        }
        if let Some((di, id)) = self.only.active {
            self.only.msgs = self.only.dbs[di].db.messages(id);
            if let Some((_, meta)) = self.only.chats.iter().find(|(i, m)| *i == di && m.id == id) {
                self.only.active_mode = meta.mode;
            }
        } else {
            self.only.msgs.clear();
            self.only.tok_full = None;
            self.only.tok_ctx = None;
        }
        self.only_request_tokens();
    }

    /// Add a database to the open set (deduped by path). `create` makes a new
    /// file; otherwise an existing file is opened. Returns its index in `dbs`.
    fn only_add_db(
        &mut self,
        raw_path: &str,
        pass: Option<String>,
        create: bool,
        encrypt: bool,
    ) -> Option<usize> {
        let path = PathBuf::from(raw_path.trim());
        if raw_path.trim().is_empty() {
            self.only.status = "Pfad fehlt".into();
            return None;
        }
        // Already open? reuse it.
        if let Some(i) = self.only.dbs.iter().position(|d| d.path == path.display().to_string()) {
            return Some(i);
        }
        let result = if create {
            if let Some(parent) = path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            let pass = (encrypt).then(|| pass.clone().unwrap_or_default());
            if encrypt && pass.as_deref().map(|p| p.is_empty()).unwrap_or(true) {
                self.only.status = "Für Verschlüsselung ein Passwort setzen".into();
                return None;
            }
            chatdb::ChatDb::create(&path, pass.as_deref())
        } else {
            if !path.exists() {
                self.only.status = "Datei nicht gefunden".into();
                return None;
            }
            if chatdb::ChatDb::is_encrypted(&path) && pass.as_deref().map(|p| p.is_empty()).unwrap_or(true) {
                self.only.status = "Datenbank ist verschlüsselt — Passwort eingeben".into();
                return None;
            }
            chatdb::ChatDb::open(&path, pass.as_deref())
        };
        match result {
            Ok(db) => {
                // Opening an existing file with data but no OnlyChat chats yet:
                // read its own tables into chats so the AI sees that content.
                // (Creating a fresh file has nothing to import.)
                if !create && db.list_chats().is_empty() {
                    let _ = db.import_foreign();
                }
                self.only.dbs.push(OpenDb { path: path.display().to_string(), db });
                self.only.status = format!("{}: {}", if create { "erstellt" } else { "geöffnet" }, path.display());
                self.only_reload();
                Some(self.only.dbs.len() - 1)
            }
            Err(e) => {
                self.only.status = e;
                None
            }
        }
    }

    fn only_open_existing(&mut self) {
        let path = self.only.open_path.clone();
        let pass = (!self.only.open_pass.is_empty()).then(|| self.only.open_pass.clone());
        if self.only_add_db(&path, pass, false, false).is_some() {
            self.only.show_open = false;
            self.only.open_pass.clear();
        }
    }

    fn only_new_chat(&mut self) {
        let name = {
            let n = self.only.new_name.trim();
            if n.is_empty() { "Neuer Chat".to_string() } else { n.to_string() }
        };
        let mode = self.only.new_mode;
        // Resolve the target database from the picker.
        let di = match self.only.new_db_pick.clone() {
            DbPick::Existing(i) if i < self.only.dbs.len() => i,
            _ => {
                let path = self.only.new_file_path.clone();
                let pass = (!self.only.new_file_pass.is_empty()).then(|| self.only.new_file_pass.clone());
                let encrypt = self.only.new_file_encrypt;
                match self.only_add_db(&path, pass, true, encrypt) {
                    Some(i) => i,
                    None => return,
                }
            }
        };
        match self.only.dbs[di].db.new_chat(&name, mode) {
            Ok(id) => {
                self.only.active = Some((di, id));
                self.only.new_name.clear();
                self.only.new_file_pass.clear();
                self.only.show_new = false;
                self.only_reload();
            }
            Err(e) => self.only.status = e,
        }
    }

    fn only_rename_chat(&mut self, di: usize, id: i64, name: &str) {
        if di < self.only.dbs.len() {
            self.only.dbs[di].db.rename_chat(id, name);
        }
        self.only_reload();
    }

    /// Delete a chat. With `delete_file`, the whole database file is removed
    /// (closing it first) — which also removes any other chats it holds.
    fn only_delete_chat(&mut self, di: usize, id: i64, delete_file: bool) {
        if di >= self.only.dbs.len() {
            return;
        }
        if delete_file {
            let path = self.only.dbs[di].path.clone();
            self.only.dbs.remove(di); // drop closes the connection
            let _ = std::fs::remove_file(&path);
            self.only.active = None;
        } else {
            self.only.dbs[di].db.delete_chat(id);
            if self.only.active == Some((di, id)) {
                self.only.active = None;
            }
        }
        self.only_reload();
    }

    /// Ask the worker to count tokens for the active chat: the whole history vs.
    /// what the current context mode would actually feed the model.
    fn only_request_tokens(&mut self) {
        let (Some(handle), Some((di, id))) = (&self.handle, self.only.active) else { return };
        let full: String = self.only.msgs.iter().map(|m| m.text.clone()).collect::<Vec<_>>().join("\n");
        let summary = self.only.dbs[di].db.summary(id).map(|(s, _)| s);
        let (prelude, turns) = chatdb::assemble(&self.only.msgs, self.only.active_mode, summary.as_deref());
        let mut ctx = prelude.unwrap_or_default();
        for (_, t) in &turns {
            ctx.push('\n');
            ctx.push_str(t);
        }
        let _ = handle.tx.send(ToWorker::CountTokens { full, ctx });
    }

    fn only_send(&mut self) {
        let text = self.only.input.trim().to_string();
        if text.is_empty() || self.only.generating {
            return;
        }
        let Some(handle) = &self.handle else {
            self.only.status = "kein Modell geladen".into();
            return;
        };
        let Some((di, id)) = self.only.active else {
            self.only.status = "keinen Chat ausgewählt".into();
            return;
        };
        let db = &self.only.dbs[di].db;
        // Persist the user turn, then assemble the context for the model.
        db.append(id, chatdb::Role::User, &text);
        self.only.msgs = db.messages(id);
        let mode = self.only.active_mode;
        db.set_mode(id, mode);
        let summary = db.summary(id).map(|(s, _)| s);
        let (prelude, turns) = chatdb::assemble(&self.only.msgs, mode, summary.as_deref());
        self.stream = Stream::OnlyChat;
        self.only.stream_buf.clear();
        self.only.generating = true;
        handle.cancel.store(false, Ordering::Relaxed);
        let _ = handle.tx.send(ToWorker::ChatOnly { turns, prelude, gen: self.gen.clone() });
        self.only.input.clear();
        self.status = Status::Generating;
        self.only_request_tokens();
    }

    fn only_summarize(&mut self) {
        if self.only.summarizing || self.only.generating {
            return;
        }
        let Some(handle) = &self.handle else { return };
        if self.only.active.is_none() {
            return;
        }
        // Summarize everything older than the verbatim window.
        let cut = self.only.msgs.len().saturating_sub(chatdb::LAST_N * 2);
        if cut == 0 {
            self.only.status = "noch nichts zu kürzen".into();
            return;
        }
        let mut body = String::new();
        for m in &self.only.msgs[..cut] {
            let who = if m.role.is_user() { "User" } else { "AI" };
            body.push_str(&format!("{who}: {}\n", m.text));
        }
        self.only.summarizing = true;
        self.stream = Stream::Chat; // summary is silent; keep Chat routing idle
        let _ = handle.tx.send(ToWorker::Summarize { text: body, gen: self.gen.clone() });
        self.only.status = "fasse zusammen …".into();
    }

    fn drain_events(&mut self) {
        let mut events = Vec::new();
        if let Some(handle) = &self.handle {
            while let Ok(ev) = handle.rx.try_recv() {
                events.push(ev);
            }
        }
        for ev in events {
            match ev {
                FromWorker::Ready => self.status = Status::Ready,
                FromWorker::Backend(b) => {
                    self.log(format!("[backend] {b}"));
                    self.backend = b;
                }
                FromWorker::Token(s) => {
                    if self.stream == Stream::OnlyChat {
                        self.only.stream_buf.push_str(&s);
                    } else {
                        self.raw_stream.push_str(&s);
                        if let Some(last) = self.transcript.last_mut() {
                            if matches!(last.role, Role::Assistant) {
                                last.text.push_str(&s);
                            }
                        }
                    }
                    self.status = Status::Generating;
                }
                FromWorker::Stats { tok_per_s } => self.tok_per_s = tok_per_s,
                FromWorker::ToolCall(desc) => {
                    self.log(format!("[tool] {desc}"));
                    if let Some(last) = self.transcript.last_mut() {
                        last.role = Role::Tool;
                        last.text = format!("⚙ {desc}");
                    }
                }
                FromWorker::ToolResult(r) => {
                    self.log(format!("[result] {}", r.replace('\n', " ")));
                    self.transcript.push(Msg { role: Role::Tool, text: format!("→ {r}") });
                    self.transcript.push(Msg { role: Role::Assistant, text: String::new() });
                }
                FromWorker::Approval { desc, outside } => {
                    self.log(format!("[approval needed] {desc}{}", if outside { " (outside)" } else { "" }));
                    self.pending_approval = Some((desc, outside));
                }
                FromWorker::Summary(s) => {
                    let s = s.trim().to_string();
                    if let Some((di, id)) = self.only.active {
                        let upto = self.only.msgs.len().saturating_sub(chatdb::LAST_N * 2) as i64;
                        self.only.dbs[di].db.set_summary(id, &s, upto);
                        self.only.status = "Zusammenfassung gespeichert".into();
                    }
                    self.only.summarizing = false;
                    self.only_request_tokens();
                }
                FromWorker::OnlyChatTokens { full, ctx } => {
                    self.only.tok_full = Some(full);
                    self.only.tok_ctx = Some(ctx);
                }
                FromWorker::Done => {
                    self.status = Status::Ready;
                    if self.stream == Stream::OnlyChat && self.only.generating {
                        let text = std::mem::take(&mut self.only.stream_buf);
                        let text = strip_think(&text);
                        if let Some((di, id)) = self.only.active {
                            if !text.trim().is_empty() {
                                self.only.dbs[di].db.append(id, chatdb::Role::Assistant, text.trim());
                            }
                        }
                        self.only.generating = false;
                        self.only_reload();
                    }
                }
                FromWorker::Error(e) => {
                    self.log(format!("[error] {e}"));
                    self.status = Status::Error(e);
                }
            }
        }
    }

    /// The right-side settings panel: models, hardware, permissions, advanced.
    fn show_settings(&mut self, ui: &mut egui::Ui) {
        if !self.settings_open {
            return;
        }
        egui::Panel::right("settings")
            .resizable(false)
            .frame(
                egui::Frame::NONE
                    .fill(PANEL)
                    .inner_margin(egui::Margin::same(14))
                    .stroke(Stroke::new(1.0, LINE)),
            )
            .show(ui, |ui| {
                ui.set_min_width(340.0);
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("SETTINGS").color(AMBER).size(14.0).strong());
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.button(egui::RichText::new("✕").color(MUTED)).clicked() {
                            self.settings_open = false;
                        }
                    });
                });
                ui.add_space(6.0);
                egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {

                // ---- model management (load / unload / download) ----
                section(ui, "MODEL");
                let current = self.current_model.clone();
                let cur_name = current
                    .as_deref()
                    .and_then(|p| p.rsplit('/').next())
                    .unwrap_or("none");
                ui.label(egui::RichText::new(format!("loaded: {cur_name}")).color(if current.is_some() { GREEN } else { MUTED }).size(12.0));
                let busy = self.trainer.is_some();
                let models = list_gguf();
                if models.is_empty() {
                    ui.label(egui::RichText::new("no models found").color(MUTED).size(11.0));
                }
                let mut to_load: Option<PathBuf> = None;
                for m in &models {
                    let name = m.file_name().and_then(|n| n.to_str()).unwrap_or("?");
                    let is_cur = current.as_deref() == Some(m.display().to_string().as_str());
                    if ui.selectable_label(is_cur, egui::RichText::new(name).size(11.5).color(if is_cur { AMBER } else { FG })).clicked() {
                        to_load = Some(m.clone());
                    }
                }
                if let Some(p) = to_load {
                    self.load_model(p);
                }
                ui.add_space(4.0);
                if ui.add_enabled(self.handle.is_some(), egui::Button::new("Unload (free memory)")).clicked() {
                    self.unload_model();
                }
                ui.add_space(6.0);
                ui.label(egui::RichText::new("Download a Qwen2.5-Coder base:").color(MUTED).size(11.0));
                ui.horizontal(|ui| {
                    if ui.add_enabled(!busy, egui::Button::new("1.5B (~1.1 GB)")).clicked() {
                        self.download_model("download 1.5B", QWEN_1_5B_URL, "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf");
                    }
                    if ui.add_enabled(!busy, egui::Button::new("7B (~4.4 GB)")).clicked() {
                        self.download_model("download 7B", QWEN_7B_URL, "qwen2.5-coder-7b-instruct-q4_k_m.gguf");
                    }
                });
                ui.add_space(6.0);
                ui.label(egui::RichText::new("…or your own model (GGUF URL):").color(MUTED).size(11.0));
                ui.horizontal(|ui| {
                    ui.add(egui::TextEdit::singleline(&mut self.dl_url).hint_text("https://…/model.gguf").desired_width(210.0));
                    let ok = !busy && self.dl_url.trim().ends_with(".gguf");
                    if ui.add_enabled(ok, egui::Button::new("Download")).clicked() {
                        let url = self.dl_url.trim().to_string();
                        let name = filename_from_url(&url);
                        self.download_model("download (custom)", &url, &name);
                    }
                });

                ui.add_space(12.0);
                let hw = self.hw.lock().map(|g| *g).unwrap_or_default();

                section(ui, "MODEL — THIS PROCESS");
                meter(ui, "CPU", hw.proc_cpu_pct / 100.0, format!("{:.0}%", hw.proc_cpu_pct), GREEN);
                let rssf = if hw.mem_total_gb > 0.0 { hw.proc_rss_gb / hw.mem_total_gb } else { 0.0 };
                meter(ui, "RAM", rssf, format!("{:.2} GB", hw.proc_rss_gb), AMBER);
                let bcol = if self.backend == "GPU" { GREEN } else { MUTED };
                ui.label(
                    egui::RichText::new(format!("compute: {}", self.backend))
                        .color(bcol)
                        .size(11.0),
                );

                ui.add_space(12.0);
                section(ui, "SYSTEM");
                meter(ui, "CPU", hw.cpu_pct / 100.0, format!("{:.0}%", hw.cpu_pct), GREEN);
                let memf = if hw.mem_total_gb > 0.0 { hw.mem_used_gb / hw.mem_total_gb } else { 0.0 };
                meter(ui, "RAM", memf, format!("{:.1} / {:.1} GB", hw.mem_used_gb, hw.mem_total_gb), AMBER);
                match (hw.gpu_pct, hw.vram_used_mb, hw.vram_total_mb) {
                    (Some(g), Some(vu), Some(vt)) => {
                        meter(ui, "GPU", g / 100.0, format!("{g:.0}%"), GREEN);
                        let vf = if vt > 0.0 { vu / vt } else { 0.0 };
                        meter(ui, "VRAM", vf, format!("{vu:.0} / {vt:.0} MB"), AMBER);
                    }
                    _ => {
                        ui.label(egui::RichText::new("GPU   n/a (nvidia-smi)").color(MUTED).size(12.0));
                    }
                }

                ui.add_space(12.0);
                section(ui, "GENERATION");
                ui.label(
                    egui::RichText::new(format!("{:.1} tokens / sec", self.tok_per_s))
                        .color(GREEN)
                        .size(13.0),
                );

                ui.add_space(12.0);
                section(ui, "PERMISSIONS");
                ui.checkbox(&mut self.perms.sandbox, "Sandbox mode — no system access");
                let tools = !self.perms.sandbox;
                ui.add_enabled_ui(tools, |ui| {
                    ui.checkbox(&mut self.perms.fs_read, "Read files");
                    ui.checkbox(&mut self.perms.fs_write, "Create / modify files");
                    ui.checkbox(&mut self.perms.shell, "Run shell commands");
                    ui.checkbox(&mut self.perms.network, "Network access");

                    ui.add_space(8.0);
                    ui.label(
                        egui::RichText::new("Workspace folder — free access inside:")
                            .color(MUTED)
                            .size(11.0),
                    );
                    ui.add(
                        egui::TextEdit::singleline(&mut self.perms.workspace)
                            .desired_width(f32::INFINITY)
                            .hint_text("/path/to/folder"),
                    );
                    ui.checkbox(
                        &mut self.perms.confirm_outside,
                        "Ask before anything outside this folder",
                    );
                });
                ui.add_space(6.0);
                ui.label(
                    egui::RichText::new(
                        "Inside the workspace the agent acts freely; outside, it asks first — \
                         like Claude Code. Enforced with the agent (Phase D).",
                    )
                    .color(MUTED)
                    .size(11.0),
                );

                // ---- where the model files live (absolute paths) ----
                ui.add_space(12.0);
                section(ui, "MODEL FILES");
                ui.label(egui::RichText::new("Put or drop .gguf models in these folders:").color(MUTED).size(11.0));
                let cwd = std::env::current_dir().unwrap_or_else(|_| ".".into());
                for d in [PathBuf::from("models"), data_models_dir(), PathBuf::from("/usr/share/overflow/models")] {
                    let abs = if d.is_absolute() { d.clone() } else { cwd.join(&d) };
                    let abs = abs.canonicalize().unwrap_or(abs);
                    let exists = abs.is_dir();
                    ui.label(
                        egui::RichText::new(format!("{} {}", if exists { "•" } else { "◦" }, abs.display()))
                            .color(if exists { FG } else { MUTED })
                            .size(11.0),
                    );
                }
                ui.label(
                    egui::RichText::new("qwen2.5-coder-* = base models · overflow-* = your fine-tunes")
                        .color(MUTED)
                        .size(10.5),
                );

                // ---- advanced settings (collapsed by default, at the bottom) ----
                ui.add_space(14.0);
                egui::CollapsingHeader::new(egui::RichText::new("ADVANCED").color(AMBER).size(12.0).strong())
                    .id_salt("advanced")
                    .show(ui, |ui| {
                        ui.checkbox(&mut self.gen.thinking, "Thinking mode (reason before answering)");
                        ui.checkbox(&mut self.show_thinking, "Show thinking in chat");
                        ui.horizontal(|ui| {
                            ui.label(egui::RichText::new("effort").color(MUTED).size(12.0));
                            ui.selectable_value(&mut self.gen.effort, 0, "low");
                            ui.selectable_value(&mut self.gen.effort, 1, "med");
                            ui.selectable_value(&mut self.gen.effort, 2, "high");
                        });
                        ui.add_space(4.0);
                        ui.add(egui::Slider::new(&mut self.gen.temp, 0.0..=1.5).text("temperature"));
                        ui.add(egui::Slider::new(&mut self.gen.top_p, 0.1..=1.0).text("top-p"));
                        ui.add(egui::Slider::new(&mut self.gen.top_k, 1..=200).text("top-k"));
                        ui.add(egui::Slider::new(&mut self.gen.max_tokens, 32..=4096).text("max tokens"));
                        ui.add(egui::Slider::new(&mut self.gen.repeat_penalty, 1.0..=2.0).text("repeat penalty"));
                        ui.add(egui::Slider::new(&mut self.gen.repeat_last_n, 16..=512).text("repeat window"));
                        ui.label(egui::RichText::new("temp 0 = deterministic. repeat penalty >1 stops loops.").color(MUTED).size(10.5));
                    });
                }); // end scroll area
            });
    }

    /// Right-hand developer panel: live metrics, the math, and a raw console.
    fn dev_panel(&mut self, ui: &mut egui::Ui) {
        egui::Panel::right("dev")
            .frame(egui::Frame::NONE.fill(SURFACE).inner_margin(egui::Margin::same(12)).stroke(Stroke::new(1.0, LINE)))
            .show(ui, |ui| {
                ui.set_min_width(340.0);
                section(ui, "DEV · METRICS");
                ui.label(egui::RichText::new(format!("backend   {}", self.backend)).color(GREEN).size(12.0));
                ui.label(egui::RichText::new(format!("speed     {:.1} tok/s", self.tok_per_s)).color(FG).size(12.0));
                if self.tab == Tab::OnlyChat {
                    if let Some(full) = self.only.tok_full {
                        ui.label(egui::RichText::new(format!("chat      {full} Token")).color(CYAN).size(12.0));
                    }
                    if let Some(ctx) = self.only.tok_ctx {
                        ui.label(egui::RichText::new(format!("context   {ctx} Token (Modus: {})", self.only.active_mode.label())).color(CYAN).size(12.0));
                    }
                }
                if let Some(l) = self.learn.loss {
                    ui.label(egui::RichText::new(format!("loss      {l:.4}")).color(AMBER).size(12.0));
                }
                if let Some((a, b)) = self.learn.step {
                    ui.label(egui::RichText::new(format!("step      {a}/{b}")).color(MUTED).size(12.0));
                }

                ui.add_space(10.0);
                section(ui, "DEV · RAW TOKENS");
                egui::ScrollArea::vertical()
                    .stick_to_bottom(true)
                    .max_height(180.0)
                    .auto_shrink([false, false])
                    .id_salt("raw_tokens")
                    .show(ui, |ui| {
                        let raw = if self.raw_stream.is_empty() { "(nothing generated yet)" } else { &self.raw_stream };
                        ui.label(egui::RichText::new(raw).color(GREEN).size(10.5).monospace());
                    });

                ui.add_space(10.0);
                section(ui, "DEV · MATH");
                for f in FORMULAS {
                    ui.label(egui::RichText::new(f).color(CYAN).size(11.5));
                }

                ui.add_space(10.0);
                section(ui, "DEV · CONSOLE");
                egui::ScrollArea::vertical()
                    .stick_to_bottom(true)
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        for line in &self.console {
                            ui.label(egui::RichText::new(line).color(MUTED).size(10.5));
                        }
                    });
            });
    }

    /// Chat tab: approval dialog, composer, and the transcript.
    fn chat_body(&mut self, ui: &mut egui::Ui) {
        let can_send = matches!(self.status, Status::Ready) && self.pending_approval.is_none();

        if let Some((desc, outside)) = self.pending_approval.clone() {
            egui::Panel::bottom("approval")
                .frame(egui::Frame::NONE.fill(PANEL).inner_margin(egui::Margin::symmetric(16, 12)).stroke(Stroke::new(1.0, RED)))
                .show(ui, |ui| {
                    let head = if outside {
                        format!("⚠ allow tool (OUTSIDE workspace): {desc}")
                    } else {
                        format!("allow tool: {desc}")
                    };
                    ui.label(egui::RichText::new(head).color(if outside { RED } else { FG }).strong());
                    ui.add_space(6.0);
                    ui.horizontal(|ui| {
                        if ui.button(egui::RichText::new("Allow once").color(GREEN)).clicked() {
                            if let Some(h) = &self.handle { let _ = h.tx.send(ToWorker::Approve(true)); }
                            self.pending_approval = None;
                        }
                        if ui.button(egui::RichText::new("Deny").color(RED)).clicked() {
                            if let Some(h) = &self.handle { let _ = h.tx.send(ToWorker::Approve(false)); }
                            self.pending_approval = None;
                        }
                    });
                });
        }

        egui::Panel::bottom("composer")
            .frame(egui::Frame::NONE.fill(BG).inner_margin(egui::Margin::symmetric(16, 12)))
            .show(ui, |ui| {
                let generating = matches!(self.status, Status::Generating);
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new(">").color(AMBER).strong());
                    if generating {
                        if ui.button(egui::RichText::new("■ Stop").color(RED)).clicked() {
                            if let Some(h) = &self.handle {
                                h.cancel.store(true, Ordering::Relaxed);
                            }
                            self.log("stop requested");
                        }
                    }
                    let edit = egui::TextEdit::singleline(&mut self.input)
                        .hint_text("type a message, Enter to send")
                        .desired_width(f32::INFINITY);
                    let resp = ui.add_enabled(can_send, edit);
                    let entered = resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
                    if entered && can_send {
                        self.send_current_input();
                        resp.request_focus();
                    }
                });
            });

        egui::CentralPanel::default()
            .frame(egui::Frame::NONE.fill(BG).inner_margin(egui::Margin::symmetric(16, 12)))
            .show(ui, |ui| {
                if self.transcript.is_empty() {
                    ui.add_space(24.0);
                    let no_model = self.current_model.is_none();
                    let busy = self.trainer.is_some();
                    let mut do_dl = false;
                    ui.vertical_centered(|ui| {
                        ui.label(egui::RichText::new("OVERFLOW").color(AMBER).size(28.0).strong());
                        ui.label(egui::RichText::new("S1BERIA'S CYBERSECURITY AI").color(FG).size(14.0).strong());
                        ui.label(egui::RichText::new("runs entirely on this machine").color(MUTED).size(12.0));
                        if no_model {
                            ui.add_space(16.0);
                            ui.label(egui::RichText::new("No model loaded.").color(RED).size(13.0));
                            ui.label(egui::RichText::new("Download one to begin, or drop a .gguf in the models folder and load it in Settings.").color(MUTED).size(11.5));
                            ui.add_space(8.0);
                            if ui.add_enabled(!busy, egui::Button::new(egui::RichText::new("Download base model (1.5B)").color(GREEN))).clicked() {
                                do_dl = true;
                            }
                            if busy {
                                ui.label(egui::RichText::new(format!("{} — see dev console", self.learn.status)).color(AMBER).size(11.0));
                            }
                        }
                    });
                    if do_dl {
                        self.download_base();
                    }
                }
                egui::ScrollArea::vertical()
                    .stick_to_bottom(true)
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        let generating = matches!(self.status, Status::Generating);
                        let last = self.transcript.len().saturating_sub(1);
                        for (i, msg) in self.transcript.iter().enumerate() {
                            let active = generating && i == last;
                            if msg.text.is_empty() && !active {
                                continue;
                            }
                            match msg.role {
                                Role::Tool => {
                                    ui.add_space(6.0);
                                    ui.label(egui::RichText::new(&msg.text).color(AMBER).size(12.5));
                                }
                                Role::User => {
                                    ui.add_space(10.0);
                                    ui.label(egui::RichText::new("you ▸").color(AMBER).size(12.0).strong());
                                    ui.label(egui::RichText::new(&msg.text).color(FG));
                                }
                                Role::Assistant => {
                                    ui.add_space(10.0);
                                    ui.label(egui::RichText::new("overflow ▸").color(GREEN).size(12.0).strong());
                                    let (think, answer) = split_think(&msg.text);
                                    if self.show_thinking {
                                        if let Some(t) = think {
                                            egui::CollapsingHeader::new(
                                                egui::RichText::new("💭 thinking").color(CYAN).size(11.5),
                                            )
                                            .id_salt(i)
                                            .default_open(active && answer.trim().is_empty())
                                            .show(ui, |ui| {
                                                ui.label(egui::RichText::new(t.trim()).color(MUTED).size(11.5));
                                            });
                                        }
                                    }
                                    let mut body = answer.to_string();
                                    if active {
                                        body.push('▌');
                                    }
                                    if !body.trim().is_empty() || !active {
                                        ui.label(egui::RichText::new(body).color(FG));
                                    }
                                }
                            }
                        }
                        ui.add_space(8.0);
                    });
            });
    }

    /// OnlyChat tab: a persistent, token-saving chat over a SQLite store. The
    /// model only ever sees the slice of history the chosen context mode feeds
    /// it, so a huge imported chat stays runnable at a bounded cost.
    fn onlychat_body(&mut self, ui: &mut egui::Ui) {
        // ---- left: databases + chat list + actions ----
        let mut select: Option<(usize, i64)> = None;
        let mut menu_rename: Option<(usize, i64, String)> = None;
        let mut menu_delete: Option<(usize, i64)> = None;
        let mut new_chat = false;
        let mut open_db = false;
        let mut close_db: Option<usize> = None;
        let mut do_rename: Option<(usize, i64, String)> = None;
        let mut cancel_rename = false;
        let mut do_delete: Option<(usize, i64, bool)> = None;
        let mut cancel_delete = false;
        egui::Panel::left("onlychat_list")
            .resizable(false)
            .frame(egui::Frame::NONE.fill(PANEL).inner_margin(egui::Margin::same(12)).stroke(Stroke::new(1.0, LINE)))
            .show(ui, |ui| {
                ui.set_min_width(260.0);
                ui.label(egui::RichText::new("ONLYCHAT").color(AMBER).size(13.0).strong());
                ui.add_space(6.0);
                ui.horizontal(|ui| {
                    if ui.button(egui::RichText::new("+ Neuer Chat").color(GREEN).size(12.0)).clicked() {
                        self.only.show_new = !self.only.show_new;
                        self.only.show_open = false;
                    }
                    if ui.button(egui::RichText::new("DB öffnen").color(CYAN).size(12.0)).clicked() {
                        self.only.show_open = !self.only.show_open;
                        self.only.show_new = false;
                    }
                });

                // -- open an existing database (so its chats show up) --
                if self.only.show_open {
                    egui::Frame::NONE.fill(SURFACE).inner_margin(egui::Margin::same(8)).stroke(Stroke::new(1.0, LINE)).show(ui, |ui| {
                        ui.label(egui::RichText::new("vorhandene .sqlite öffnen").color(FG).size(11.5));
                        ui.add(egui::TextEdit::singleline(&mut self.only.open_path).desired_width(f32::INFINITY).hint_text("…/memory.sqlite3"));
                        ui.add(egui::TextEdit::singleline(&mut self.only.open_pass).password(true).desired_width(f32::INFINITY).hint_text("Passwort (falls verschlüsselt)"));
                        if ui.button(egui::RichText::new("öffnen").color(GREEN).size(11.5)).clicked() {
                            open_db = true;
                        }
                    });
                }

                // -- new chat: name, mode, and WHICH database it goes into --
                if self.only.show_new {
                    egui::Frame::NONE.fill(SURFACE).inner_margin(egui::Margin::same(8)).stroke(Stroke::new(1.0, LINE)).show(ui, |ui| {
                        ui.add(egui::TextEdit::singleline(&mut self.only.new_name).desired_width(f32::INFINITY).hint_text("Name"));
                        ui.horizontal(|ui| {
                            ui.label(egui::RichText::new("Modus").color(MUTED).size(11.0));
                            egui::ComboBox::from_id_salt("new_mode")
                                .selected_text(self.only.new_mode.label())
                                .show_ui(ui, |ui| {
                                    for m in [chatdb::CtxMode::Full, chatdb::CtxMode::LastN, chatdb::CtxMode::SummaryRetrieval] {
                                        ui.selectable_value(&mut self.only.new_mode, m, m.label());
                                    }
                                });
                        });
                        ui.horizontal(|ui| {
                            ui.label(egui::RichText::new("DB").color(MUTED).size(11.0));
                            let sel_label = match &self.only.new_db_pick {
                                DbPick::Existing(i) => self.only.dbs.get(*i).map(|d| db_name(&d.path)).unwrap_or_else(|| "?".into()),
                                DbPick::NewFile => "＋ neue Datei…".to_string(),
                            };
                            egui::ComboBox::from_id_salt("new_db_pick")
                                .selected_text(sel_label)
                                .show_ui(ui, |ui| {
                                    for (i, od) in self.only.dbs.iter().enumerate() {
                                        ui.selectable_value(&mut self.only.new_db_pick, DbPick::Existing(i), db_name(&od.path));
                                    }
                                    ui.selectable_value(&mut self.only.new_db_pick, DbPick::NewFile, "＋ neue Datei…");
                                });
                        });
                        if self.only.new_db_pick == DbPick::NewFile {
                            ui.add(egui::TextEdit::singleline(&mut self.only.new_file_path).desired_width(f32::INFINITY).hint_text("…/chats.sqlite"));
                            ui.horizontal(|ui| {
                                ui.checkbox(&mut self.only.new_file_encrypt, egui::RichText::new("verschlüsseln").color(FG).size(11.0));
                                if self.only.new_file_encrypt {
                                    ui.add(egui::TextEdit::singleline(&mut self.only.new_file_pass).password(true).desired_width(140.0).hint_text("Passwort"));
                                }
                            });
                        }
                        if ui.button(egui::RichText::new("anlegen").color(AMBER).size(12.0)).clicked() {
                            new_chat = true;
                        }
                    });
                }

                ui.add_space(6.0);
                ui.separator();
                // -- chat list across all open databases --
                egui::ScrollArea::vertical().auto_shrink([false, false]).max_height(ui.available_height() - 150.0).show(ui, |ui| {
                    if self.only.chats.is_empty() {
                        ui.label(egui::RichText::new("noch keine Chats — lege einen an oder öffne eine DB").color(MUTED).size(11.0));
                    }
                    let multi = self.only.dbs.len() > 1;
                    for (di, meta) in &self.only.chats {
                        let is_cur = self.only.active == Some((*di, meta.id));
                        ui.horizontal(|ui| {
                            let label = format!("{}  ({})", meta.name, meta.n_msgs);
                            if ui.selectable_label(is_cur, egui::RichText::new(label).size(12.0).color(if is_cur { AMBER } else { FG })).clicked() {
                                select = Some((*di, meta.id));
                            }
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                // Gear menu: rename / delete (and more).
                                ui.menu_button(egui::RichText::new("⚙").color(MUTED).size(12.0), |ui| {
                                    if ui.button("Umbenennen").clicked() {
                                        menu_rename = Some((*di, meta.id, meta.name.clone()));
                                    }
                                    if ui.button(egui::RichText::new("Löschen").color(RED)).clicked() {
                                        menu_delete = Some((*di, meta.id));
                                    }
                                });
                            });
                        });
                        if multi {
                            ui.label(egui::RichText::new(format!("   {}", db_name(&self.only.dbs[*di].path))).color(MUTED).size(9.5));
                        }
                    }
                });
                if !self.only.dbs.is_empty() {
                    ui.add_space(6.0);
                    section(ui, "OFFENE DBs");
                    for (i, od) in self.only.dbs.iter().enumerate() {
                        ui.horizontal(|ui| {
                            let enc = if od.db.encrypted() { "🔒 " } else { "" };
                            ui.label(egui::RichText::new(format!("{enc}{}", db_name(&od.path))).color(FG).size(10.5));
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                if ui.button(egui::RichText::new("schließen").color(MUTED).size(9.5)).clicked() {
                                    close_db = Some(i);
                                }
                            });
                        });
                    }
                }
                if !self.only.status.is_empty() {
                    ui.add_space(6.0);
                    ui.label(egui::RichText::new(&self.only.status).color(MUTED).size(11.0));
                }
            });

        // ---- rename dialog ----
        if self.only.rename.is_some() {
            egui::Panel::bottom("oc_rename")
                .frame(egui::Frame::NONE.fill(PANEL).inner_margin(egui::Margin::symmetric(16, 12)).stroke(Stroke::new(1.0, LINE)))
                .show(ui, |ui| {
                    ui.label(egui::RichText::new("Chat umbenennen").color(AMBER).strong());
                    ui.add_space(4.0);
                    let mut save = false;
                    if let Some((_, _, buf)) = self.only.rename.as_mut() {
                        let resp = ui.add(egui::TextEdit::singleline(buf).desired_width(320.0));
                        if resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                            save = true;
                        }
                    }
                    ui.horizontal(|ui| {
                        if ui.button(egui::RichText::new("Speichern").color(GREEN)).clicked() {
                            save = true;
                        }
                        if ui.button(egui::RichText::new("Abbrechen").color(MUTED)).clicked() {
                            cancel_rename = true;
                        }
                    });
                    if save {
                        if let Some((d, i, b)) = &self.only.rename {
                            if !b.trim().is_empty() {
                                do_rename = Some((*d, *i, b.trim().to_string()));
                            }
                        }
                    }
                });
        }

        // ---- delete confirmation ----
        if let Some((di, id)) = self.only.confirm_delete {
            let fname = self.only.dbs.get(di).map(|d| db_name(&d.path)).unwrap_or_default();
            let others = self.only.chats.iter().filter(|(i, _)| *i == di).count().saturating_sub(1);
            egui::Panel::bottom("oc_delete")
                .frame(egui::Frame::NONE.fill(PANEL).inner_margin(egui::Margin::symmetric(16, 12)).stroke(Stroke::new(1.0, RED)))
                .show(ui, |ui| {
                    ui.label(egui::RichText::new("Chat löschen").color(RED).strong());
                    ui.label(egui::RichText::new(format!("Datenbankdatei: {fname}")).color(MUTED).size(11.5));
                    if others > 0 {
                        ui.label(egui::RichText::new(format!("Achtung: die Datei enthält noch {others} weitere Chat(s) — beim Mitlöschen gehen die auch verloren.")).color(AMBER).size(11.0));
                    }
                    ui.add_space(4.0);
                    ui.horizontal(|ui| {
                        if ui.button(egui::RichText::new("Nur diesen Chat").color(FG)).clicked() {
                            do_delete = Some((di, id, false));
                        }
                        if ui.button(egui::RichText::new("Chat + Datenbankdatei").color(RED)).clicked() {
                            do_delete = Some((di, id, true));
                        }
                        if ui.button(egui::RichText::new("Abbrechen").color(MUTED)).clicked() {
                            cancel_delete = true;
                        }
                    });
                });
        }

        // ---- bottom composer ----
        let mut send = false;
        let can_send = self.handle.is_some() && !self.only.generating && self.only.active.is_some();
        egui::Panel::bottom("onlychat_composer")
            .frame(egui::Frame::NONE.fill(BG).inner_margin(egui::Margin::symmetric(16, 12)))
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new(">").color(AMBER).strong());
                    if self.only.generating {
                        if ui.button(egui::RichText::new("■ Stop").color(RED)).clicked() {
                            if let Some(h) = &self.handle {
                                h.cancel.store(true, Ordering::Relaxed);
                            }
                        }
                    }
                    let edit = egui::TextEdit::singleline(&mut self.only.input)
                        .hint_text("Nachricht — Enter zum Senden")
                        .desired_width(f32::INFINITY);
                    let resp = ui.add_enabled(can_send, edit);
                    let entered = resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
                    if entered && can_send {
                        send = true;
                        resp.request_focus();
                    }
                });
            });

        // ---- central: active transcript + per-chat context mode ----
        let mut set_mode: Option<chatdb::CtxMode> = None;
        let mut summarize = false;
        egui::CentralPanel::default()
            .frame(egui::Frame::NONE.fill(BG).inner_margin(egui::Margin::symmetric(16, 12)))
            .show(ui, |ui| {
                if self.only.active.is_none() {
                    ui.add_space(24.0);
                    ui.vertical_centered(|ui| {
                        ui.label(egui::RichText::new("Wähle links einen Chat oder lege einen neuen an.").color(MUTED).size(13.0));
                    });
                    return;
                }
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("Kontext-Modus:").color(MUTED).size(11.5));
                    let mut mode = self.only.active_mode;
                    egui::ComboBox::from_id_salt("active_mode")
                        .selected_text(mode.label())
                        .show_ui(ui, |ui| {
                            for m in [chatdb::CtxMode::Full, chatdb::CtxMode::LastN, chatdb::CtxMode::SummaryRetrieval] {
                                ui.selectable_value(&mut mode, m, m.label());
                            }
                        });
                    if mode != self.only.active_mode {
                        set_mode = Some(mode);
                    }
                    if self.only.active_mode == chatdb::CtxMode::SummaryRetrieval {
                        if ui.add_enabled(!self.only.summarizing, egui::Button::new(egui::RichText::new("Älteres zusammenfassen").color(CYAN).size(11.5))).clicked() {
                            summarize = true;
                        }
                    }
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if let (Some(full), Some(ctx)) = (self.only.tok_full, self.only.tok_ctx) {
                            ui.label(egui::RichText::new(format!("Kontext {ctx} / {full} Token")).color(MUTED).size(11.0));
                        }
                    });
                });
                ui.separator();
                egui::ScrollArea::vertical().stick_to_bottom(true).auto_shrink([false, false]).show(ui, |ui| {
                    for m in &self.only.msgs {
                        if m.role.is_user() {
                            ui.add_space(10.0);
                            ui.label(egui::RichText::new("you ▸").color(AMBER).size(12.0).strong());
                            ui.label(egui::RichText::new(&m.text).color(FG));
                        } else {
                            ui.add_space(10.0);
                            ui.label(egui::RichText::new("overflow ▸").color(GREEN).size(12.0).strong());
                            ui.label(egui::RichText::new(&m.text).color(FG));
                        }
                    }
                    if self.only.generating {
                        ui.add_space(10.0);
                        ui.label(egui::RichText::new("overflow ▸").color(GREEN).size(12.0).strong());
                        let mut body = strip_think(&self.only.stream_buf);
                        body.push('▌');
                        ui.label(egui::RichText::new(body).color(FG));
                    }
                    ui.add_space(8.0);
                });
            });

        // ---- apply collected actions (outside the UI closures) ----
        if let Some(di) = close_db {
            if di < self.only.dbs.len() {
                self.only.dbs.remove(di);
            }
            self.only.active = None; // indices shift; simplest is to reselect
            self.only_reload();
        }
        if let Some(r) = menu_rename {
            self.only.rename = Some(r);
        }
        if let Some(d) = menu_delete {
            self.only.confirm_delete = Some(d);
        }
        if let Some((di, id, name)) = do_rename {
            self.only_rename_chat(di, id, &name);
            self.only.rename = None;
        }
        if cancel_rename {
            self.only.rename = None;
        }
        if let Some((di, id, file)) = do_delete {
            self.only_delete_chat(di, id, file);
            self.only.confirm_delete = None;
        }
        if cancel_delete {
            self.only.confirm_delete = None;
        }
        if let Some(sel) = select {
            self.only.active = Some(sel);
            self.only_reload();
        }
        if open_db {
            self.only_open_existing();
        }
        if new_chat {
            self.only_new_chat();
        }
        if let Some(mode) = set_mode {
            self.only.active_mode = mode;
            if let Some((di, id)) = self.only.active {
                self.only.dbs[di].db.set_mode(id, mode);
            }
            self.only_reload();
        }
        if summarize {
            self.only_summarize();
        }
        if send {
            self.only_send();
        }
    }

    /// Learning tab: drive the QLoRA pipeline (Python) from the GUI.
    fn learning_body(&mut self, ui: &mut egui::Ui) {
        egui::CentralPanel::default()
            .frame(egui::Frame::NONE.fill(BG).inner_margin(egui::Margin::symmetric(16, 12)))
            .show(ui, |ui| {
                ui.label(egui::RichText::new("LEARNING").color(AMBER).size(20.0).strong());
                ui.label(
                    egui::RichText::new("Fine-tune a model on your own data (QLoRA on the GPU), then load it here.")
                        .color(MUTED)
                        .size(12.0),
                );
                ui.add_space(4.0);
                let have_venv = Self::venv_python().exists();
                if !have_venv {
                    ui.label(egui::RichText::new("Python env not set up yet — click 'Set up env' (large download).").color(RED).size(12.0));
                }

                ui.add_space(10.0);
                section(ui, "CONFIG");
                egui::Grid::new("learncfg").num_columns(2).spacing([10.0, 6.0]).show(ui, |ui| {
                    ui.label(egui::RichText::new("base model").color(MUTED).size(12.0));
                    ui.add(egui::TextEdit::singleline(&mut self.learn.base).desired_width(380.0));
                    ui.end_row();
                    ui.label(egui::RichText::new("dataset").color(MUTED).size(12.0));
                    let cur_label = DATASETS.iter().find(|d| d.1 == self.learn.data.as_str()).map(|d| d.0).unwrap_or("custom");
                    egui::ComboBox::from_id_salt("dataset_pick").width(380.0).selected_text(cur_label).show_ui(ui, |ui| {
                        for (label, path, gen) in DATASETS {
                            if ui.selectable_label(self.learn.data.as_str() == *path, *label).clicked() {
                                self.learn.data = (*path).into();
                                self.learn.gen_script = (*gen).into();
                            }
                        }
                    });
                    ui.end_row();
                    ui.label(egui::RichText::new("dataset path").color(MUTED).size(12.0));
                    ui.add(egui::TextEdit::singleline(&mut self.learn.data).desired_width(380.0));
                    ui.end_row();
                    ui.label(egui::RichText::new("epochs").color(MUTED).size(12.0));
                    ui.add(egui::TextEdit::singleline(&mut self.learn.epochs).desired_width(80.0));
                    ui.end_row();
                });

                ui.add_space(10.0);
                section(ui, "PIPELINE");
                let busy = self.trainer.is_some();
                let py = Self::venv_python().display().to_string();
                ui.horizontal_wrapped(|ui| {
                    if ui.add_enabled(!busy, egui::Button::new("1 · Generate dataset")).clicked() {
                        let gen = self.learn.gen_script.clone();
                        self.start("generate dataset", &py, vec![gen]);
                    }
                    if ui.add_enabled(!busy && have_venv, egui::Button::new("2 · Train + build GGUF")).clicked() {
                        // RAM/disk-safe pipeline: QLoRA -> mmap merge -> Q8. Removing any
                        // stale adapter forces a fresh train on the selected dataset.
                        let (b, d, e) = (self.learn.base.clone(), self.learn.data.clone(), self.learn.epochs.clone());
                        let cmd = format!(
                            "rm -rf training/out && env BASE='{b}' DATA='{d}' EPOCHS='{e}' bash training/build_overflow_wt.sh"
                        );
                        self.start("train + build", "sh", vec!["-c".into(), cmd]);
                    }
                    if ui.add_enabled(!busy, egui::Button::new("3 · Load trained model")).clicked() {
                        self.load_model("models/overflow_wt-7b-q8_0.gguf".into());
                    }
                });
                ui.add_space(4.0);
                if ui.add_enabled(!busy, egui::Button::new("Set up env (venv + deps)")).clicked() {
                    let cmd = "python3 -m venv training/venv && training/venv/bin/pip install -r training/requirements.txt sentencepiece protobuf".to_string();
                    self.start("setup env", "sh", vec!["-c".into(), cmd]);
                }

                ui.add_space(12.0);
                section(ui, "PROGRESS");
                ui.label(egui::RichText::new(format!("status: {}", self.learn.status)).color(if busy { AMBER } else { GREEN }).size(13.0));
                if let Some(l) = self.learn.loss {
                    ui.label(egui::RichText::new(format!("loss: {l:.4}")).color(AMBER).size(13.0));
                }
                if let Some((a, b)) = self.learn.step {
                    ui.add(egui::ProgressBar::new(a as f32 / b as f32).desired_width(400.0).fill(AMBER).text(format!("{a}/{b}")));
                }
                ui.add_space(8.0);
                ui.label(egui::RichText::new("(toggle 'dev' for the full console)").color(MUTED).size(11.0));
                egui::ScrollArea::vertical().stick_to_bottom(true).auto_shrink([false, false]).show(ui, |ui| {
                    for line in self.console.iter().rev().take(14).collect::<Vec<_>>().into_iter().rev() {
                        ui.label(egui::RichText::new(line).color(MUTED).size(10.5));
                    }
                });
            });
    }
}

/// A small uppercase section heading with a rule.
fn section(ui: &mut egui::Ui, title: &str) {
    ui.label(egui::RichText::new(title).color(AMBER).size(11.0).strong());
    ui.separator();
}

/// A labeled progress meter: `NAME [=====      ] value`.
fn meter(ui: &mut egui::Ui, label: &str, frac: f32, text: String, fill: Color32) {
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new(format!("{label:<5}")).color(MUTED).size(12.0));
        ui.add(
            egui::ProgressBar::new(frac.clamp(0.0, 1.0))
                .desired_width(210.0)
                .fill(fill)
                .text(egui::RichText::new(text).size(11.0).color(BG)),
        );
    });
}

impl eframe::App for OverflowApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.drain_events();
        self.drain_trainer();

        egui::Panel::top("titlebar")
            .frame(egui::Frame::NONE.fill(PANEL).inner_margin(egui::Margin::symmetric(16, 12)))
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    draw_logo(ui, 26.0);
                    ui.add_space(10.0);
                    ui.label(egui::RichText::new("OVERFLOW").color(FG).size(20.0).strong());
                    ui.add_space(14.0);
                    if ui.selectable_label(self.tab == Tab::Chat, egui::RichText::new("Chat").size(13.0)).clicked() {
                        self.tab = Tab::Chat;
                    }
                    if ui.selectable_label(self.tab == Tab::OnlyChat, egui::RichText::new("OnlyChat").size(13.0)).clicked() {
                        self.tab = Tab::OnlyChat;
                    }
                    if ui.selectable_label(self.tab == Tab::Learning, egui::RichText::new("Learning").size(13.0)).clicked() {
                        self.tab = Tab::Learning;
                    }
                    if ui.selectable_label(self.tab == Tab::Docs, egui::RichText::new("Docs").size(13.0)).clicked() {
                        self.tab = Tab::Docs;
                    }
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.button(egui::RichText::new("[ settings ]").color(FG).size(12.0)).clicked() {
                            self.settings_open = !self.settings_open;
                        }
                        ui.add_space(10.0);
                        if ui
                            .selectable_label(self.dev, egui::RichText::new("dev").color(if self.dev { CYAN } else { MUTED }).size(12.0))
                            .clicked()
                        {
                            self.dev = !self.dev;
                        }
                        ui.add_space(12.0);
                        let (plabel, pcol) = if self.perms.sandbox { ("sandboxed", GREEN) } else { ("tools enabled", RED) };
                        ui.label(egui::RichText::new(plabel).color(pcol).size(12.0));
                        ui.add_space(12.0);
                        let (dot, label, col) = match &self.status {
                            Status::NoModel => ("\u{25cb}", "no model".to_string(), MUTED),
                            Status::Loading => ("\u{25cf}", "loading\u{2026}".to_string(), AMBER),
                            Status::Ready => ("\u{25cf}", "ready".to_string(), GREEN),
                            Status::Generating => ("\u{25cf}", "thinking\u{2026}".to_string(), AMBER),
                            Status::Error(e) => ("\u{25cf}", format!("error: {e}"), RED),
                        };
                        ui.label(egui::RichText::new(label).color(MUTED).size(12.0));
                        ui.add_space(6.0);
                        ui.label(egui::RichText::new(dot).color(col).size(12.0));
                    });
                });
            });

        if self.dev {
            self.dev_panel(ui);
        }
        self.show_settings(ui); // right-side panel (before the central area)
        match self.tab {
            Tab::Chat => self.chat_body(ui),
            Tab::OnlyChat => self.onlychat_body(ui),
            Tab::Learning => self.learning_body(ui),
            Tab::Docs => docs_body(ui),
        }

        let ctx = ui.ctx().clone();
        ctx.request_repaint_after(Duration::from_millis(1000));
        if matches!(self.status, Status::Generating | Status::Loading) || self.trainer.is_some() {
            ctx.request_repaint();
        }
    }
}

/// A documentation section: amber heading + body lines.
fn doc_section(ui: &mut egui::Ui, title: &str, lines: &[&str]) {
    ui.add_space(14.0);
    ui.label(egui::RichText::new(title).color(AMBER).size(15.0).strong());
    ui.add_space(4.0);
    for l in lines {
        if l.is_empty() {
            ui.add_space(4.0);
        } else if let Some(h) = l.strip_prefix("# ") {
            ui.label(egui::RichText::new(h).color(CYAN).size(12.5).strong());
        } else {
            ui.label(egui::RichText::new(*l).color(FG).size(12.5));
        }
    }
}

/// Docs tab: how to prompt, how to teach it, tools, and models.
fn docs_body(ui: &mut egui::Ui) {
    egui::CentralPanel::default()
        .frame(egui::Frame::NONE.fill(BG).inner_margin(egui::Margin::symmetric(20, 14)))
        .show(ui, |ui| {
            ui.label(egui::RichText::new("DOCS").color(AMBER).size(20.0).strong());
            ui.label(egui::RichText::new("OVERFLOW — S1BERIA's local cybersecurity AI").color(MUTED).size(12.0));
            egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                doc_section(ui, "Getting started", &[
                    "Load a model (Settings -> Model), then type in the Chat tab.",
                    "Everything runs locally on your machine — nothing is sent anywhere.",
                    "Tabs: Chat (talk + agent), Learning (fine-tune), Docs (this).",
                    "Toggle 'dev' (top bar) for live metrics, the math, and a raw console.",
                ]);
                doc_section(ui, "Prompting tips", &[
                    "# Be specific about the goal and the context.",
                    "Say the target, the constraints, and the output you want.",
                    "Weak:  \"scan the network\"",
                    "Strong: \"nmap a SYN scan of 10.0.0.0/24, top 100 ports, greppable output\".",
                    "",
                    "# Give it a role / standing context via the system prompt.",
                    "Settings -> permissions has a system prompt; state you're doing authorized",
                    "security work so it answers directly without hedging.",
                    "",
                    "# Ask for steps, then execution.",
                    "\"First outline the approach, then run the commands\" — lets you review.",
                    "",
                    "# Let it use tools.",
                    "In the agent, ask it to read files or run commands; it calls tools itself.",
                    "Keep requests concrete: name files, hosts, and exact commands when you can.",
                    "",
                    "# Bigger model = better answers.",
                    "The 1.5B is fast but shallow; the 7B reasons far better (slower).",
                ]);
                doc_section(ui, "Teaching it (fine-tuning)", &[
                    "The Learning tab trains the model on YOUR data (QLoRA on the GPU).",
                    "",
                    "# 1. Make a dataset.",
                    "One JSON object per line, chat format:",
                    "  {\"messages\":[{\"role\":\"user\",\"content\":\"Q\"},{\"role\":\"assistant\",\"content\":\"A\"}]}",
                    "Put good, varied examples — the data is what it learns.",
                    "",
                    "# 2. Train.",
                    "Learning tab: set base, dataset, epochs -> Train. Watch the loss fall.",
                    "",
                    "# 3. Export + load.",
                    "Merge + export GGUF, then Load trained model. Done — it's now your model.",
                    "",
                    "# Tips.",
                    "50-500 focused examples already shift behavior. More epochs = stronger",
                    "but can overfit. Keep a few 'test' questions to check before/after.",
                ]);
                doc_section(ui, "Tools, permissions & sandbox", &[
                    "The agent can read/write files, list dirs, and run shell commands.",
                    "Default is sandboxed (no access). Enable tools in Settings.",
                    "Workspace folder = free access inside; outside it asks first.",
                    "Shell runs inside bubblewrap: workspace-only, no network unless granted.",
                    "You approve anything sensitive with an in-app Allow/Deny dialog.",
                ]);
                doc_section(ui, "The overflow_wt model", &[
                    "overflow_wt-7b-q8_0 is OVERFLOW's flagship model: Qwen2.5-Coder-7B",
                    "fine-tuned (QLoRA) to act like an information-security professor —",
                    "it explains infosec clearly across crypto, network/web security, binary",
                    "exploitation, privilege escalation, malware, forensics and incident response.",
                    "It ships with the installed package and is loaded by default.",
                    "'wt' = walkthrough: it teaches the reasoning, not just the answer.",
                    "For authorized security work and learning.",
                ]);
                doc_section(ui, "Models", &[
                    "Settings -> Model: load, unload, or download a model (like LM Studio).",
                    "Models live in ~/.local/share/overflow/models (or ./models in a dev build);",
                    "the package also installs to /usr/share/overflow/models.",
                    "Download a base model to get started, or load your own fine-tuned GGUF.",
                    "Unload frees the memory/VRAM; load a different one anytime.",
                    "The engine runs Qwen2-architecture GGUFs.",
                ]);
                ui.add_space(16.0);
            });
        });
}

/// Apply the OVERFLOW dark terminal theme: palette + all-monospace text.
fn install_theme(ctx: &egui::Context) {
    let mut v = egui::Visuals::dark();
    v.panel_fill = BG;
    v.window_fill = BG;
    v.extreme_bg_color = SURFACE;
    v.override_text_color = Some(FG);
    v.hyperlink_color = AMBER;
    v.selection.bg_fill = AMBER.linear_multiply(0.35);
    v.selection.stroke = Stroke::new(1.0, AMBER);
    v.widgets.noninteractive.bg_stroke = Stroke::new(1.0, LINE);
    v.widgets.inactive.bg_fill = PANEL;
    v.widgets.inactive.weak_bg_fill = PANEL;
    v.widgets.hovered.bg_fill = Color32::from_rgb(0x1a, 0x24, 0x26);
    v.widgets.active.bg_fill = Color32::from_rgb(0x22, 0x2e, 0x30);
    ctx.set_visuals(v);

    ctx.all_styles_mut(|style| {
        for (_, font) in style.text_styles.iter_mut() {
            font.family = FontFamily::Monospace;
        }
    });
}

/// Draw the OVERFLOW mark — a basilisk eye: a hex scope framing a reptilian
/// slit pupil — from the 64x64 design scaled to `size`, in amber.
fn draw_logo(ui: &mut egui::Ui, size: f32) {
    let (resp, painter) = ui.allocate_painter(Vec2::splat(size), Sense::hover());
    let o = resp.rect.min;
    let s = size / 64.0;
    let p = |x: f32, y: f32| Pos2::new(o.x + x * s, o.y + y * s);

    let hex = vec![
        p(32.0, 6.0), p(54.0, 19.0), p(54.0, 45.0), p(32.0, 58.0), p(10.0, 45.0), p(10.0, 19.0),
    ];
    painter.add(egui::Shape::closed_line(hex, Stroke::new(2.4 * s, AMBER)));
    let pupil = vec![p(32.0, 20.0), p(36.0, 32.0), p(32.0, 44.0), p(28.0, 32.0)];
    painter.add(egui::Shape::convex_polygon(pupil, AMBER, Stroke::NONE));
}
