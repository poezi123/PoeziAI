//! OVERFLOW — a local AI, native desktop GUI (egui/eframe).
//!
//! The window is a thin client: it renders the transcript and forwards messages
//! to the inference worker thread, streaming tokens back into the view. No web
//! stack, pure Rust. The whole look is drawn from the OVERFLOW palette.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod chatdb;
mod hw;
mod server;
mod trainer;
mod worker;

use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use eframe::egui;
use egui::{Color32, FontFamily, Pos2, Sense, Stroke, Vec2};
use poezi_gguf::GgufFile;
use poezi_model::agent::GenCfg;
use poezi_model::{LoadOpts, MemEstimate};

use worker::{FromWorker, Handle, ToWorker};

#[derive(PartialEq, Clone, Copy)]
enum Tab {
    Chat,
    OnlyChat,
    Learning,
    Server,
    Docs,
}

/// Server-mode state: the running HTTPS server (if any) and the shared bridge
/// to the model worker that it forwards requests through.
struct ServerState {
    handle: Option<server::ServerHandle>,
    bridge: server::Bridge,
    /// Permissions + generation settings shared with the running server, so the
    /// phone's gear panel can read and change them live.
    cfg: server::SharedCfg,
    status: String,
}

impl Default for ServerState {
    fn default() -> Self {
        let p = Permissions::default();
        let perms = worker::perms(p.sandbox, p.fs_read, p.fs_write, p.shell, p.network, &p.workspace, p.confirm_outside);
        ServerState {
            handle: None,
            bridge: std::sync::Arc::new(std::sync::Mutex::new(None)),
            cfg: std::sync::Arc::new(std::sync::Mutex::new(server::SrvCfg { perms, gen: GenCfg::default() })),
            status: String::new(),
        }
    }
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
    // Some fine-tunes wrap the reply in <answer>…</answer>; show just the text.
    out.replace("<answer>", "").replace("</answer>", "").trim().to_string()
}

/// Drop `<answer>…</answer>` wrapper tags a fine-tune may emit, keeping the text
/// inside. Returns the cleaned reply, trimmed.
fn strip_answer_tags(s: &str) -> String {
    s.replace("<answer>", "").replace("</answer>", "").trim().to_string()
}

/// Percent-encode a query string for a URL.
fn url_encode(s: &str) -> String {
    let mut out = String::new();
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => out.push(b as char),
            b' ' => out.push('+'),
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

/// Very small HTML→text: drop tags/scripts, decode a few entities, collapse
/// whitespace. Good enough to feed search snippets to the model.
///
/// Walks the string by chars (never by raw byte offsets) so non-ASCII text in
/// the page can't land us on a non-char boundary — tag names are ASCII, so the
/// script/style checks use a case-insensitive compare on the remaining slice.
fn html_to_text(html: &str) -> String {
    let mut out = String::new();
    let mut in_tag = false;
    let mut skip_depth = 0i32;
    for (i, ch) in html.char_indices() {
        if ch == '<' {
            let head = &html[i..];
            let ci = |p: &str| {
                head.len() >= p.len() && head.as_bytes()[..p.len()].eq_ignore_ascii_case(p.as_bytes())
            };
            if ci("<script") || ci("<style") {
                skip_depth += 1;
            } else if ci("</script") || ci("</style") {
                skip_depth = (skip_depth - 1).max(0);
            }
            in_tag = true;
        } else if ch == '>' {
            in_tag = false;
            out.push(' ');
        } else if !in_tag && skip_depth == 0 {
            out.push(ch);
        }
    }
    let out = out
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#x27;", "'")
        .replace("&#39;", "'")
        .replace("&nbsp;", " ");
    out.split_whitespace().collect::<Vec<_>>().join(" ")
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

/// OnlyChat system focus: tells the model it works against a saved database.
const ONLYCHAT_FOCUS: &str = "Du bist OVERFLOW im OnlyChat-Modus. Du arbeitest mit einer gespeicherten \
Datenbank als Gedächtnis: nutze das unten gegebene gespeicherte Wissen und den bisherigen \
Gesprächsverlauf aktiv, um konkret und konsistent zu antworten. Erfinde keine Fakten dazu; wenn etwas \
nicht in den Daten steht, sage es.";

/// Combine the OnlyChat focus instruction, retrieved database knowledge, and an
/// optional conversation summary into the system prelude.
fn compose_prelude(memory: Option<String>, summary: Option<String>) -> String {
    let mut out = String::from(ONLYCHAT_FOCUS);
    if let Some(m) = memory {
        out.push_str("\n\n");
        out.push_str(&m);
    }
    if let Some(s) = summary {
        out.push_str("\n\n");
        out.push_str(&s);
    }
    out
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
    ("CTF · binary exploitation walkthroughs", "training/data/overflow_ctf.jsonl", "training/data_gen_ctf.py"),
    ("Web security · vulns + secure coding", "training/data/overflow_websec.jsonl", "training/data_gen_websec.py"),
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
    /// Output GGUF filename of the last train+build (what "Load" loads).
    out_name: String,
}

impl Default for LearnState {
    fn default() -> Self {
        let base = "Qwen/Qwen2.5-Coder-1.5B-Instruct";
        LearnState {
            base: base.into(),
            data: DATASETS[0].1.into(),
            gen_script: DATASETS[0].2.into(),
            epochs: "3".into(),
            loss: None,
            step: None,
            status: "idle".into(),
            task: String::new(),
            pct: None,
            out_name: model_out_name(base),
        }
    }
}

/// Fine-tune bases offered in the Learning tab (HF repo ids). Smaller trains and
/// downloads faster; the default is a small one so it finishes on a laptop.
const BASES: &[&str] = &[
    "Qwen/Qwen2.5-Coder-0.5B-Instruct",
    "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    "Qwen/Qwen2.5-Coder-3B-Instruct",
    "Qwen/Qwen2.5-Coder-7B-Instruct",
];

/// Output GGUF filename derived from a base repo id, e.g.
/// "Qwen/Qwen2.5-Coder-1.5B-Instruct" -> "overflow-qwen2.5-coder-1.5b-instruct-q8_0.gguf".
fn model_out_name(base: &str) -> String {
    let tail = base.rsplit('/').next().unwrap_or(base).to_lowercase();
    format!("overflow-{tail}-q8_0.gguf")
}

/// HF bases already downloaded into the local cache, as repo ids
/// (e.g. "Qwen/Qwen2.5-Coder-7B-Instruct"). Picking one of these trains with
/// no download — the QLoRA pipeline reuses the cached snapshot. `hub` is the
/// `training/hf-cache/hub` dir. GGUF files in the model folder are NOT listed:
/// they are inference-only and cannot be fine-tuned (QLoRA needs the HF base).
fn installed_bases(hub: &std::path::Path) -> Vec<String> {
    let mut out = Vec::new();
    if let Ok(rd) = std::fs::read_dir(hub) {
        for e in rd.flatten() {
            let name = e.file_name().to_string_lossy().into_owned();
            let Some(rest) = name.strip_prefix("models--") else { continue };
            // needs a real (non-empty) snapshot to be usable
            let has_snap = std::fs::read_dir(e.path().join("snapshots"))
                .map(|r| r.flatten().any(|s| s.path().is_dir()))
                .unwrap_or(false);
            if has_snap {
                out.push(rest.replace("--", "/"));
            }
        }
    }
    out.sort();
    out
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

/// Selectable models to download: (label, url, filename, ~GB needed to load).
/// All single-file Qwen2.5-Coder GGUFs the engine can load. The bigger ones
/// ("schlauer") exceed 12 GB VRAM and run via whole-layer CPU offloading at full
/// quality. The last field drives the per-model hardware-fit badge.
const MODELS: &[(&str, &str, &str, f32)] = &[
    (
        "Qwen2.5-Coder 0.5B · Q4 (tiny)",
        "https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF/resolve/main/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf",
        "qwen2.5-coder-0.5b-instruct-q4_k_m.gguf",
        1.0,
    ),
    (
        "Qwen2.5-Coder 1.5B · Q4 (fast)",
        QWEN_1_5B_URL,
        "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
        2.0,
    ),
    (
        "Qwen2.5-Coder 3B · Q4",
        "https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct-GGUF/resolve/main/qwen2.5-coder-3b-instruct-q4_k_m.gguf",
        "qwen2.5-coder-3b-instruct-q4_k_m.gguf",
        3.2,
    ),
    (
        "Qwen2.5-Coder 7B · Q4",
        "https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF/resolve/main/qwen2.5-coder-7b-instruct-q4_k_m.gguf",
        "qwen2.5-coder-7b-instruct-q4_k_m.gguf",
        6.0,
    ),
    (
        "Qwen2.5-Coder 14B · Q4 (smartest)",
        "https://huggingface.co/bartowski/Qwen2.5-Coder-14B-Instruct-GGUF/resolve/main/Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf",
        "qwen2.5-coder-14b-instruct-q4_k_m.gguf",
        10.5,
    ),
];

/// Hardware-fit verdict for a model of `need_gb`, given VRAM and total RAM.
/// Reflects the offloading research: a model bigger than VRAM still runs at full
/// quality as long as it fits in VRAM + RAM (GPU prefix + CPU tail).
fn model_fit(need_gb: f32, vram_gb: Option<f32>, ram_gb: f32) -> (String, Color32) {
    match vram_gb {
        Some(v) if need_gb <= v * 0.9 => ("✓ fits fully in the GPU — nice and fast".into(), GREEN),
        Some(v) if need_gb <= v + ram_gb * 0.8 => (
            format!("↯ runs split (GPU {v:.0} GB + computer memory) — full quality, slower"),
            CYAN,
        ),
        Some(_) => ("✗ too big for this computer".into(), RED),
        None => {
            if need_gb <= ram_gb * 0.8 {
                ("↯ runs on the processor — full quality, slower".into(), CYAN)
            } else {
                ("✗ too big for this computer's memory".into(), RED)
            }
        }
    }
}

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
/// The user's XDG data dir (`$XDG_DATA_HOME` or `~/.local/share`).
fn xdg_data_dir() -> PathBuf {
    std::env::var_os("XDG_DATA_HOME")
        .map(PathBuf::from)
        .filter(|p| !p.as_os_str().is_empty())
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local/share")))
        .unwrap_or_else(|| PathBuf::from("."))
}

fn data_models_dir() -> PathBuf {
    if Path::new("models").is_dir() {
        return PathBuf::from("models");
    }
    xdg_data_dir().join("overflow/models")
}

/// All .gguf models found in the dev ./models dir, the XDG data dir, and the
/// system dir where the Arch package installs the bundled model.
fn list_gguf() -> Vec<PathBuf> {
    let mut dirs = vec![
        PathBuf::from("models"),
        data_models_dir(),
        PathBuf::from("/usr/share/overflow/models"),
        PathBuf::from("/mnt/ai/Overflow Models"),
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

/// Token-limit presets offered in the load panel (KV-cache capacity). More
/// tokens = longer memory, but more VRAM/RAM.
const CTX_CHOICES: [usize; 6] = [1024, 2048, 4096, 8192, 16384, 32768];

/// Human-readable bytes as GB (1 GB = 2^30).
fn gb(bytes: u64) -> String {
    format!("{:.1} GB", bytes as f64 / 1_073_741_824.0)
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
    dl_sel: usize,
    gen: GenCfg,
    show_thinking: bool,
    /// Detected GPU name (from nvidia-smi), shown in Details; None if no GPU.
    gpu_name: Option<String>,
    /// A model chosen in the list but not loaded yet (shown in the load panel).
    pending_model: Option<PathBuf>,
    /// VRAM estimate for `pending_model`, read from its GGUF header.
    mem: Option<MemEstimate>,
    /// Chosen token limit (KV-cache capacity) for the next load.
    n_ctx: usize,
    gen_tokens: u32,
    /// When the current model load started (for load progress + ETA).
    load_start: Option<Instant>,
    only: OnlyChatState,
    stream: Stream,
    server: ServerState,
    /// Pending internet-research fetch (result arrives on this channel).
    research_rx: Option<std::sync::mpsc::Receiver<String>>,
    research_query: String,
}

impl OverflowApp {
    fn new(model_path: Option<PathBuf>) -> Self {
        let gpu_name = hw::gpu_name();
        // No auto-load: a model given on the command line is pre-selected in the
        // load panel, but the user picks the token limit and device, then loads.
        let (pending_model, mem) = match &model_path {
            Some(p) => (Some(p.clone()), GgufFile::open(p).ok().and_then(|g| poezi_model::estimate_mem(&g))),
            None => (None, None),
        };
        let mut app = OverflowApp {
            handle: None,
            current_model: None,
            status: Status::NoModel,
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
            load_start: None,
            learn: LearnState::default(),
            trainer: None,
            dl_url: String::new(),
            dl_sel: 1,
            gen: GenCfg::default(),
            show_thinking: true,
            gpu_name,
            pending_model,
            mem,
            n_ctx: 4096,
            gen_tokens: 0,
            only: OnlyChatState::default(),
            stream: Stream::Chat,
            server: ServerState::default(),
            research_rx: None,
            research_query: String::new(),
        };
        if app.pending_model.is_some() {
            app.log("pick a token limit and a device below, then press Load");
        }
        app.sync_server_bridge();
        app
    }

    /// Point the server bridge at the current worker (or None), so server-mode
    /// requests reach whatever model is loaded now.
    fn sync_server_bridge(&self) {
        if let Ok(mut b) = self.server.bridge.lock() {
            *b = self.handle.as_ref().map(|h| h.tx.clone());
        }
    }

    fn toggle_server(&mut self) {
        if let Some(h) = self.server.handle.take() {
            h.stop.store(true, std::sync::atomic::Ordering::Relaxed);
            self.server.status = "Server stopped".into();
            return;
        }
        self.sync_server_bridge();
        // Seed the server's shared settings from the GUI's current permissions
        // and generation knobs; the phone can change them afterwards.
        if let Ok(mut c) = self.server.cfg.lock() {
            c.perms = worker::perms(
                self.perms.sandbox,
                self.perms.fs_read,
                self.perms.fs_write,
                self.perms.shell,
                self.perms.network,
                &self.perms.workspace,
                self.perms.confirm_outside,
            );
            c.gen = self.gen.clone();
        }
        match server::start(self.server.bridge.clone(), self.server.cfg.clone(), self.hw.clone()) {
            Ok(h) => {
                // The URL (with its secret token) is shown separately as a
                // selectable field, not folded into the status line.
                self.server.status = "running".into();
                self.server.handle = Some(h);
            }
            Err(e) => self.server.status = format!("Error: {e}"),
        }
    }

    /// Load a model with explicit options (respawns the worker). Frees any
    /// previous one first.
    fn load_model(&mut self, path: PathBuf, opts: LoadOpts) {
        self.handle = None; // drop old worker -> frees its model/VRAM
        self.load_start = Some(Instant::now());
        let where_ = match opts.gpu_layers {
            Some(0) => "CPU".to_string(),
            Some(n) => format!("GPU ({n} layers)"),
            None => "GPU (auto)".to_string(),
        };
        self.log(format!("loading {} · {} tokens · {}", path.display(), opts.n_ctx, where_));
        self.handle = Some(worker::spawn(path.clone(), opts));
        self.current_model = Some(path.display().to_string());
        self.pending_model = None;
        self.status = Status::Loading;
        self.backend = "…".into();
        self.transcript.clear();
        self.sync_server_bridge();
        self.tab = Tab::Chat;
    }

    /// Select a model (does not load it): compute its memory estimate and show
    /// the load panel so the user can pick a token limit and device.
    fn select_model(&mut self, path: PathBuf) {
        self.mem = GgufFile::open(&path).ok().and_then(|g| poezi_model::estimate_mem(&g));
        self.pending_model = Some(path);
    }

    /// Turn the current choices (pending model, token limit, device) into load
    /// options and start loading.
    fn load_pending(&mut self, gpu_layers: Option<usize>) {
        if let Some(p) = self.pending_model.clone() {
            self.load_model(p, LoadOpts { n_ctx: self.n_ctx, gpu_layers });
        }
    }

    /// The load panel: pick a token limit, see what it costs in VRAM (and how
    /// much spills to RAM), then load on GPU, split, or CPU.
    fn load_panel(&mut self, ui: &mut egui::Ui) {
        let busy = self.trainer.is_some();
        let has_gpu = self.gpu_name.is_some();
        ui.add_space(8.0);
        let name = self
            .pending_model
            .as_ref()
            .and_then(|p| p.file_name().and_then(|n| n.to_str()))
            .unwrap_or("model")
            .to_string();
        ui.label(egui::RichText::new(format!("Set up: {name}")).color(AMBER).size(12.0).strong());

        // `Some(gpu_layers)` means "start loading with this split"; collected here
        // and acted on after the UI so we don't borrow self inside the widgets.
        let mut action: Option<Option<usize>> = None;

        let Some(mem) = self.mem else {
            ui.label(egui::RichText::new("Can't read this file's size — you can still load it.").color(MUTED).size(10.5));
            ui.horizontal(|ui| {
                if has_gpu && ui.add_enabled(!busy, egui::Button::new("Load on GPU")).clicked() {
                    action = Some(None);
                }
                if ui.add_enabled(!busy, egui::Button::new("Load on CPU")).clicked() {
                    action = Some(Some(0));
                }
            });
            if let Some(a) = action {
                self.load_pending(a);
            }
            return;
        };

        // Token limit: one selectable row per choice, each showing its cost.
        ui.label(egui::RichText::new("Token limit — how much the AI can keep in mind at once:").color(MUTED).size(11.0));
        let mut chosen = self.n_ctx;
        for &c in &CTX_CHOICES {
            let lbl = format!("{}k tokens", c / 1024);
            let need = mem.vram_full(c);
            let (txt, col) = match (mem.fits_full(c), mem.gpu_layers_fit(c)) {
                (Some(true), _) => (format!("{lbl}  ·  {} VRAM  ✓ fits on GPU", gb(need)), GREEN),
                (Some(false), Some(0)) => {
                    (format!("{lbl}  ·  needs {} — won't fit, runs on CPU", gb(need)), RED)
                }
                (Some(false), Some(k)) => (
                    format!("{lbl}  ·  needs {} → {k}/{} on GPU, rest in RAM (slower)", gb(need), mem.n_layer),
                    AMBER,
                ),
                _ => (format!("{lbl}  ·  about {} VRAM", gb(need)), MUTED),
            };
            ui.selectable_value(&mut chosen, c, egui::RichText::new(txt).color(col).size(11.0));
        }
        self.n_ctx = chosen;

        let n_ctx = self.n_ctx;
        let fits = mem.fits_full(n_ctx);
        let fit_layers = mem.gpu_layers_fit(n_ctx);
        if let Some(free) = mem.free_vram {
            ui.label(egui::RichText::new(format!("Free VRAM now: {}", gb(free))).color(MUTED).size(10.5));
        }

        ui.add_space(4.0);
        // Device buttons adapt to whether the model fits.
        match (has_gpu, fits, fit_layers) {
            (false, _, _) => {
                ui.label(egui::RichText::new("No NVIDIA GPU found — this runs on the CPU.").color(MUTED).size(10.5));
                if ui.add_enabled(!busy, egui::Button::new(egui::RichText::new("Load on CPU").color(GREEN))).clicked() {
                    action = Some(Some(0));
                }
            }
            (true, Some(true), _) | (true, None, _) => {
                ui.horizontal(|ui| {
                    if ui.add_enabled(!busy, egui::Button::new(egui::RichText::new("Load on GPU").color(GREEN))).clicked() {
                        action = Some(Some(mem.n_layer));
                    }
                    if ui.add_enabled(!busy, egui::Button::new("CPU instead")).clicked() {
                        action = Some(Some(0));
                    }
                });
            }
            (true, Some(false), Some(k)) => {
                ui.label(
                    egui::RichText::new(format!(
                        "This doesn't fully fit in VRAM at {n_ctx} tokens. Put the rest in RAM, or pick a smaller token limit."
                    ))
                    .color(AMBER)
                    .size(10.5),
                );
                ui.horizontal(|ui| {
                    if k > 0
                        && ui
                            .add_enabled(!busy, egui::Button::new(egui::RichText::new(format!("GPU + RAM ({k}/{} on GPU)", mem.n_layer)).color(AMBER)))
                            .clicked()
                    {
                        action = Some(Some(k));
                    }
                    if ui.add_enabled(!busy, egui::Button::new(egui::RichText::new("CPU only").color(GREEN))).clicked() {
                        action = Some(Some(0));
                    }
                });
            }
            (true, Some(false), None) => {
                if ui.add_enabled(!busy, egui::Button::new("Load on CPU")).clicked() {
                    action = Some(Some(0));
                }
            }
        }

        if let Some(a) = action {
            self.load_pending(a);
        }
    }

    /// Unload the current model and free its memory.
    fn unload_model(&mut self) {
        self.handle = None;
        self.sync_server_bridge();
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
                "--".into(), // stop option parsing: a URL can't act as a curl flag
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

    /// Working root for the Learning pipeline: where `training/…` resolves and
    /// where the venv / data / outputs are written. In a dev checkout that is
    /// the repo (training/ sits next to us). In the installed package the
    /// scripts live read-only under /usr/share/overflow, so we use a writable
    /// data dir and seed the training scripts into it (so `training/…` commands
    /// both exist and can write).
    fn cwd() -> std::path::PathBuf {
        if PathBuf::from("training/build_overflow_wt.sh").exists() {
            return std::env::current_dir().unwrap_or_else(|_| ".".into());
        }
        let base = xdg_data_dir().join("overflow");
        let dest = base.join("training");
        let _ = std::fs::create_dir_all(&dest);
        let src = PathBuf::from("/usr/share/overflow/training");
        if src.join("build_overflow_wt.sh").exists() {
            if let Ok(rd) = std::fs::read_dir(&src) {
                for e in rd.flatten() {
                    let p = e.path();
                    if p.is_file() {
                        if let Some(n) = p.file_name() {
                            let _ = std::fs::copy(&p, dest.join(n));
                        }
                    }
                }
            }
        }
        base
    }

    /// Where the llama.cpp GGUF-export tools live (convert scripts + binaries):
    /// the repo's vendored copy in dev, or the system copy in the package.
    fn llama_dir() -> Option<PathBuf> {
        for d in [Self::cwd().join("vendor/llama.cpp"), PathBuf::from("/usr/share/overflow/llama.cpp")] {
            if d.join("convert_hf_to_gguf.py").exists() {
                return Some(d);
            }
        }
        None
    }

    fn venv_python() -> std::path::PathBuf {
        Self::cwd().join("training/venv/bin/python")
    }

    fn start(&mut self, label: &str, program: &str, args: Vec<String>) {
        self.start_env(label, program, args, vec![]);
    }

    /// Like `start`, but with extra environment variables for the child. Config
    /// (dataset path, base, …) is passed this way instead of interpolated into a
    /// `sh -c` string, so a value with a quote or space can't break out into the
    /// shell — there is no shell.
    fn start_env(&mut self, label: &str, program: &str, args: Vec<String>, extra: Vec<(String, String)>) {
        if self.trainer.is_some() {
            self.log("a task is already running");
            return;
        }
        let mut envs = vec![
            ("HF_HOME".to_string(), Self::cwd().join("training/hf-cache").display().to_string()),
            // The xet transfer can hang at 0% on some networks — use plain HTTPS.
            ("HF_HUB_DISABLE_XET".to_string(), "1".to_string()),
            ("HF_HUB_DOWNLOAD_TIMEOUT".to_string(), "60".to_string()),
        ];
        envs.extend(extra);
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

    /// Start an internet-research fetch for the current input. Requires network
    /// access to be enabled; otherwise it errors and tells the user to turn it
    /// on first. The result is fed to the model as context when it arrives.
    fn start_research(&mut self) {
        let q = self.input.trim().to_string();
        if q.is_empty() || self.research_rx.is_some() {
            return;
        }
        if !self.perms.network {
            self.log("Web search: internet is OFF — please turn it on in Settings first.");
            self.status = Status::Error("Internet is off — turn it on in Settings".into());
            return;
        }
        let (tx, rx) = std::sync::mpsc::channel();
        self.research_rx = Some(rx);
        self.research_query = q.clone();
        self.log(format!("🌐 Searching the web: {q}"));
        std::thread::spawn(move || {
            let url = format!("https://lite.duckduckgo.com/lite/?q={}", url_encode(&q));
            let out = std::process::Command::new("curl")
                .args(["-sL", "--max-time", "20", "-A", "Mozilla/5.0 (OVERFLOW)", &url])
                .output();
            let text = match out {
                Ok(o) if o.status.success() => {
                    let t = html_to_text(&String::from_utf8_lossy(&o.stdout));
                    // Drop DuckDuckGo's region/time-filter nav so the model sees
                    // the actual hits: they start right after "Past Year".
                    let t = match t.find("Past Year") {
                        Some(p) => t[p + "Past Year".len()..].trim().to_string(),
                        None => t,
                    };
                    let t: String = t.chars().take(2500).collect();
                    if t.trim().is_empty() {
                        "(web search returned nothing usable)".to_string()
                    } else {
                        t
                    }
                }
                Ok(o) => format!("(web search failed: curl exit {:?})", o.status.code()),
                Err(e) => format!("(web search failed: {e} — is curl installed?)"),
            };
            let _ = tx.send(text);
        });
    }

    /// Poll the research channel; when the fetch finishes, feed results + the
    /// original question to the model as a normal turn.
    fn drain_research(&mut self) {
        let done = self.research_rx.as_ref().and_then(|rx| rx.try_recv().ok());
        if let Some(results) = done {
            self.research_rx = None;
            let q = std::mem::take(&mut self.research_query);
            self.input = format!(
                "Use these web search results (they may be incomplete or wrong) to answer my question. Say what you based your answer on.\n\n[Search results]\n{results}\n\n[Question]\n{q}"
            );
            self.send_current_input();
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
        self.gen_tokens = 0;
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
            self.only.status = "Please enter a file path".into();
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
                self.only.status = "To encrypt it, please set a password".into();
                return None;
            }
            chatdb::ChatDb::create(&path, pass.as_deref())
        } else {
            if !path.exists() {
                self.only.status = "File not found".into();
                return None;
            }
            if chatdb::ChatDb::is_encrypted(&path) && pass.as_deref().map(|p| p.is_empty()).unwrap_or(true) {
                self.only.status = "This database is locked — please enter the password".into();
                return None;
            }
            chatdb::ChatDb::open(&path, pass.as_deref())
        };
        match result {
            Ok(db) => {
                self.only.dbs.push(OpenDb { path: path.display().to_string(), db });
                self.only.status = format!("{}: {}", if create { "created" } else { "opened" }, path.display());
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
            if n.is_empty() { "New chat".to_string() } else { n.to_string() }
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
    /// Search every open database for knowledge relevant to `query` and merge
    /// the hits, so it doesn't matter which DB the active chat lives in. The
    /// char budget is split across the open DBs.
    fn memory_over_dbs(dbs: &[OpenDb], query: &str) -> Option<String> {
        let budget = (8000 / dbs.len().max(1)).max(3000);
        let parts: Vec<String> = dbs.iter().filter_map(|od| od.db.memory_context(query, budget)).collect();
        if parts.is_empty() { None } else { Some(parts.join("\n")) }
    }

    fn only_request_tokens(&mut self) {
        let (Some(handle), Some((di, id))) = (&self.handle, self.only.active) else { return };
        let full: String = self.only.msgs.iter().map(|m| m.text.clone()).collect::<Vec<_>>().join("\n");
        let db = &self.only.dbs[di].db;
        let summary = db.summary(id).map(|(s, _)| s);
        let query = self.only.msgs.last().map(|m| m.text.clone()).unwrap_or_default();
        let (asm, turns) = chatdb::assemble(&self.only.msgs, self.only.active_mode, summary.as_deref());
        let mut ctx = compose_prelude(Self::memory_over_dbs(&self.only.dbs, &query), asm);
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
            self.only.status = "No model loaded".into();
            return;
        };
        let Some((di, id)) = self.only.active else {
            self.only.status = "No chat picked".into();
            return;
        };
        let db = &self.only.dbs[di].db;
        // Persist the user turn, then assemble the context for the model.
        db.append(id, chatdb::Role::User, &text);
        self.only.msgs = db.messages(id);
        let mode = self.only.active_mode;
        db.set_mode(id, mode);
        let summary = db.summary(id).map(|(s, _)| s);
        let (asm, turns) = chatdb::assemble(&self.only.msgs, mode, summary.as_deref());
        let mem = Self::memory_over_dbs(&self.only.dbs, &text);
        let prelude = Some(compose_prelude(mem, asm));
        self.stream = Stream::OnlyChat;
        self.only.stream_buf.clear();
        self.only.generating = true;
        self.gen_tokens = 0;
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
            self.only.status = "nothing old enough to shorten yet".into();
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
        self.only.status = "making a summary …".into();
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
                FromWorker::Ready => {
                    if let Some(t0) = self.load_start.take() {
                        self.log(format!("[load] done in {:.1}s", t0.elapsed().as_secs_f32()));
                    }
                    self.status = Status::Ready;
                }
                FromWorker::LoadProgress { done, total } => {
                    // Show how much is loaded and roughly how long is left.
                    let gb = |b: u64| b as f32 / 1_073_741_824.0;
                    let pct = if total > 0 { done as f32 / total as f32 * 100.0 } else { 0.0 };
                    let eta = self.load_start.map(|t0| {
                        let el = t0.elapsed().as_secs_f32();
                        if done > 0 { el * (total.saturating_sub(done)) as f32 / done as f32 } else { 0.0 }
                    });
                    let eta_s = match eta {
                        Some(s) if s >= 1.0 => format!(", ~{s:.0}s left"),
                        _ => String::new(),
                    };
                    self.log(format!("[load] {pct:.0}%  ({:.1}/{:.1} GB){eta_s}", gb(done), gb(total)));
                }
                FromWorker::Backend(b) => {
                    self.log(format!("[backend] {b}"));
                    self.backend = b;
                }
                FromWorker::Note(n) => {
                    self.log(format!("[note] {n}"));
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
                FromWorker::Stats { tok_per_s, produced } => {
                    self.tok_per_s = tok_per_s;
                    self.gen_tokens = produced;
                }
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
                        self.only.status = "Summary saved".into();
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
                    ui.label(egui::RichText::new("Settings").color(AMBER).size(14.0).strong());
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.button(egui::RichText::new("✕").color(MUTED)).clicked() {
                            self.settings_open = false;
                        }
                    });
                });
                ui.add_space(6.0);
                egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {

                // ---- model management (load / unload / download) ----
                section(ui, "Model");
                let current = self.current_model.clone();
                let cur_name = current
                    .as_deref()
                    .and_then(|p| p.rsplit('/').next())
                    .unwrap_or("none");
                ui.label(egui::RichText::new(format!("in use: {cur_name}")).color(if current.is_some() { GREEN } else { MUTED }).size(12.0));
                let busy = self.trainer.is_some();
                let models = list_gguf();
                if models.is_empty() {
                    ui.label(egui::RichText::new("No models yet").color(MUTED).size(11.0));
                }
                let mut to_select: Option<PathBuf> = None;
                for m in &models {
                    let name = m.file_name().and_then(|n| n.to_str()).unwrap_or("?");
                    let md = m.display().to_string();
                    let is_cur = current.as_deref() == Some(md.as_str());
                    let is_pending = self.pending_model.as_ref().map(|p| p.display().to_string()) == Some(md);
                    let col = if is_cur { GREEN } else if is_pending { AMBER } else { FG };
                    let mark = if is_cur { "● " } else { "" };
                    if ui.selectable_label(is_pending, egui::RichText::new(format!("{mark}{name}")).size(11.5).color(col)).clicked() {
                        to_select = Some(m.clone());
                    }
                }
                if let Some(p) = to_select {
                    self.select_model(p);
                }

                // ---- load panel: token limit + fit + device (shown once a model
                // is picked but not yet loaded) ----
                if self.pending_model.is_some() {
                    self.load_panel(ui);
                }
                ui.add_space(4.0);
                if ui.add_enabled(self.handle.is_some(), egui::Button::new("Unload model (free up memory)")).clicked() {
                    self.unload_model();
                }
                ui.add_space(6.0);
                ui.label(egui::RichText::new("Get a model (Qwen2.5-Coder):").color(MUTED).size(11.0));
                // Hardware for the fit badge (VRAM + total RAM).
                let (vram_gb, ram_gb) = {
                    let hw = self.hw.lock().map(|g| *g).unwrap_or_default();
                    (hw.vram_total_mb.map(|m| m / 1024.0), hw.mem_total_gb)
                };
                let sel = self.dl_sel.min(MODELS.len() - 1);
                egui::ComboBox::from_id_salt("dl_model")
                    .selected_text(MODELS[sel].0)
                    .width(300.0)
                    .show_ui(ui, |ui| {
                        for (i, m) in MODELS.iter().enumerate() {
                            let (_, col) = model_fit(m.3, vram_gb, ram_gb);
                            // Colored dot = fit; label stays readable.
                            let item = egui::RichText::new(format!("● {}  (~{:.0} GB)", m.0, m.3));
                            ui.selectable_value(&mut self.dl_sel, i, item.color(col));
                        }
                    });
                // Verdict line for the selected model.
                let (verdict, vcol) = model_fit(MODELS[sel].3, vram_gb, ram_gb);
                ui.label(egui::RichText::new(verdict).color(vcol).size(11.0));
                if ui.add_enabled(!busy, egui::Button::new(egui::RichText::new("Download").color(GREEN))).clicked() {
                    let (label, url, file, _) = MODELS[sel];
                    self.download_model(label, url, file);
                }
                ui.label(egui::RichText::new("Green = fits fully in the GPU · Cyan = still runs at full quality thanks to our offload work (just slower) · Red = too big.").color(MUTED).size(10.0));
                ui.add_space(6.0);
                ui.label(egui::RichText::new("…or your own model (GGUF link):").color(MUTED).size(11.0));
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

                section(ui, "This app right now");
                meter(ui, "CPU", hw.proc_cpu_pct / 100.0, format!("{:.0}%", hw.proc_cpu_pct), GREEN);
                let rssf = if hw.mem_total_gb > 0.0 { hw.proc_rss_gb / hw.mem_total_gb } else { 0.0 };
                meter(ui, "RAM", rssf, format!("{:.2} GB", hw.proc_rss_gb), AMBER);
                let bcol = if self.backend.starts_with("GPU") { GREEN } else { MUTED };
                ui.label(
                    egui::RichText::new(format!("running on: {}", self.backend))
                        .color(bcol)
                        .size(11.0),
                );

                ui.add_space(12.0);
                section(ui, "Your computer");
                meter(ui, "CPU", hw.cpu_pct / 100.0, format!("{:.0}%", hw.cpu_pct), GREEN);
                let memf = if hw.mem_total_gb > 0.0 { hw.mem_used_gb / hw.mem_total_gb } else { 0.0 };
                meter(ui, "RAM", memf, format!("{:.1} / {:.1} GB", hw.mem_used_gb, hw.mem_total_gb), AMBER);
                if let Some(name) = &self.gpu_name {
                    ui.label(egui::RichText::new(name).color(FG).size(11.0));
                }
                match (hw.gpu_pct, hw.vram_used_mb, hw.vram_total_mb) {
                    (Some(g), Some(vu), Some(vt)) => {
                        meter(ui, "GPU", g / 100.0, format!("{g:.0}%"), GREEN);
                        let vf = if vt > 0.0 { vu / vt } else { 0.0 };
                        meter(ui, "VRAM", vf, format!("{vu:.0} / {vt:.0} MB"), AMBER);
                    }
                    _ => {
                        ui.label(egui::RichText::new("GPU   not found (nvidia-smi)").color(MUTED).size(12.0));
                    }
                }

                ui.add_space(12.0);
                section(ui, "Speed");
                ui.label(
                    egui::RichText::new(format!("{:.1} words per second (tokens)", self.tok_per_s))
                        .color(GREEN)
                        .size(13.0),
                );

                ui.add_space(12.0);
                section(ui, "What the AI is allowed to do");
                ui.checkbox(&mut self.perms.sandbox, "Safe mode — the AI can't touch your computer");
                let tools = !self.perms.sandbox;
                ui.add_enabled_ui(tools, |ui| {
                    ui.checkbox(&mut self.perms.fs_read, "Read files");
                    ui.checkbox(&mut self.perms.fs_write, "Create and change files");
                    ui.checkbox(&mut self.perms.shell, "Run commands");
                    ui.checkbox(&mut self.perms.network, "Use the internet");

                    ui.add_space(8.0);
                    ui.label(
                        egui::RichText::new("Work folder — the AI can do anything inside it:")
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
                        "Ask me first before touching anything outside this folder",
                    );
                });
                ui.add_space(6.0);
                ui.label(
                    egui::RichText::new(
                        "Inside the work folder the AI acts on its own; outside it, it asks you \
                         first — just like Claude Code.",
                    )
                    .color(MUTED)
                    .size(11.0),
                );

                // ---- where the model files live (absolute paths) ----
                ui.add_space(12.0);
                section(ui, "Where models live");
                ui.label(egui::RichText::new("Put your .gguf models in one of these folders:").color(MUTED).size(11.0));
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
                    egui::RichText::new("qwen2.5-coder-* = base models · overflow-* = models you trained")
                        .color(MUTED)
                        .size(10.5),
                );

                // ---- advanced settings (collapsed by default, at the bottom) ----
                ui.add_space(14.0);
                egui::CollapsingHeader::new(egui::RichText::new("More settings").color(AMBER).size(12.0).strong())
                    .id_salt("advanced")
                    .show(ui, |ui| {
                        ui.checkbox(&mut self.gen.thinking, "Thinking mode (think first, then answer)");
                        ui.checkbox(&mut self.show_thinking, "Show the thinking in the chat");
                        ui.horizontal(|ui| {
                            ui.label(egui::RichText::new("how hard it thinks").color(MUTED).size(12.0));
                            ui.selectable_value(&mut self.gen.effort, 0, "low");
                            ui.selectable_value(&mut self.gen.effort, 1, "medium");
                            ui.selectable_value(&mut self.gen.effort, 2, "high");
                        });
                        ui.add_space(4.0);
                        // Quick presets.
                        ui.horizontal(|ui| {
                            ui.label(egui::RichText::new("quick set").color(MUTED).size(12.0));
                            if ui.button("Always same answer").clicked() {
                                self.gen.temp = 0.0;
                            }
                            if ui.button("Careful").clicked() {
                                self.gen.temp = 0.3;
                                self.gen.top_p = 0.9;
                            }
                            if ui.button("Creative").clicked() {
                                self.gen.temp = 0.9;
                                self.gen.top_p = 0.95;
                            }
                        });
                        ui.add(egui::Slider::new(&mut self.gen.temp, 0.0..=1.5).text("randomness (temperature)"));
                        ui.add(egui::Slider::new(&mut self.gen.top_p, 0.1..=1.0).text("top-p"));
                        ui.add(egui::Slider::new(&mut self.gen.top_k, 1..=200).text("top-k"));
                        ui.add(egui::Slider::new(&mut self.gen.max_tokens, 32..=8192).text("longest answer (tokens)"));
                        ui.add(egui::Slider::new(&mut self.gen.repeat_penalty, 1.0..=2.0).text("stop repeating"));
                        ui.add(egui::Slider::new(&mut self.gen.repeat_last_n, 16..=1024).text("how far back to check"));
                        ui.horizontal(|ui| {
                            if ui.button("Back to default").clicked() {
                                self.gen = GenCfg::default();
                            }
                        });
                        ui.label(egui::RichText::new("Randomness 0 = always the same answer (and the lossless speed-up is on). \"Stop repeating\" above 1 breaks loops. \"Longest answer\" caps how much it writes.").color(MUTED).size(10.5));
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
                section(ui, "Live info");
                let bcol = if self.backend.starts_with("GPU") { GREEN } else { AMBER };
                ui.label(egui::RichText::new(format!("runs on   {}", self.backend)).color(bcol).size(12.0));
                // What hardware OVERFLOW detected at startup.
                ui.label(
                    egui::RichText::new(format!(
                        "hardware  {}",
                        self.gpu_name.as_deref().unwrap_or("no NVIDIA GPU (CPU only)")
                    ))
                    .color(MUTED)
                    .size(11.0),
                );
                let generating = matches!(self.status, Status::Generating);
                // Live speed + token count while the model is producing output.
                let spd = egui::RichText::new(format!("speed     {:.1} tok/s", self.tok_per_s))
                    .color(if generating { GREEN } else { FG }).size(12.0);
                ui.label(spd);
                ui.label(egui::RichText::new(format!("tokens    {}{}", self.gen_tokens, if generating { " ▸" } else { "" }))
                    .color(if generating { AMBER } else { MUTED }).size(12.0));
                if self.tab == Tab::OnlyChat {
                    if let Some(full) = self.only.tok_full {
                        ui.label(egui::RichText::new(format!("whole chat   {full} tokens")).color(CYAN).size(12.0));
                    }
                    if let Some(ctx) = self.only.tok_ctx {
                        ui.label(egui::RichText::new(format!("sent to AI   {ctx} tokens (mode: {})", self.only.active_mode.label())).color(CYAN).size(12.0));
                    }
                }
                if let Some(l) = self.learn.loss {
                    ui.label(egui::RichText::new(format!("loss      {l:.4}")).color(AMBER).size(12.0));
                }
                if let Some((a, b)) = self.learn.step {
                    ui.label(egui::RichText::new(format!("step      {a}/{b}")).color(MUTED).size(12.0));
                }

                ui.add_space(10.0);
                section(ui, "Raw output");
                egui::ScrollArea::vertical()
                    .stick_to_bottom(true)
                    .max_height(180.0)
                    .auto_shrink([false, false])
                    .id_salt("raw_tokens")
                    .show(ui, |ui| {
                        let raw = if self.raw_stream.is_empty() { "(nothing yet)" } else { &self.raw_stream };
                        ui.label(egui::RichText::new(raw).color(GREEN).size(10.5).monospace());
                    });

                ui.add_space(10.0);
                section(ui, "The math behind it");
                for f in FORMULAS {
                    ui.label(egui::RichText::new(f).color(CYAN).size(11.5));
                }

                ui.add_space(10.0);
                section(ui, "Log");
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
                        format!("⚠ Let the AI do this OUTSIDE your work folder: {desc}")
                    } else {
                        format!("Let the AI do this: {desc}")
                    };
                    ui.label(egui::RichText::new(head).color(if outside { RED } else { FG }).strong());
                    ui.add_space(6.0);
                    ui.horizontal(|ui| {
                        if ui.button(egui::RichText::new("Yes, once").color(GREEN)).clicked() {
                            if let Some(h) = &self.handle { let _ = h.tx.send(ToWorker::Approve(true)); }
                            self.pending_approval = None;
                        }
                        if ui.button(egui::RichText::new("No").color(RED)).clicked() {
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
                let researching = self.research_rx.is_some();
                let mut do_research = false;
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
                    // Internet research: fetch web results for the query first.
                    let net = self.perms.network;
                    let rcol = if net { CYAN } else { MUTED };
                    let rlabel = if researching { "🌐…" } else { "🌐" };
                    if ui.add_enabled(can_send && !researching, egui::Button::new(egui::RichText::new(rlabel).color(rcol)))
                        .on_hover_text(if net { "Search the web for this question" } else { "Internet is off — turn it on in Settings first" })
                        .clicked()
                    {
                        do_research = true;
                    }
                    let edit = egui::TextEdit::singleline(&mut self.input)
                        .hint_text("Type a message, press Enter to send")
                        .desired_width(f32::INFINITY);
                    let resp = ui.add_enabled(can_send && !researching, edit);
                    let entered = resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
                    if entered && can_send {
                        self.send_current_input();
                        resp.request_focus();
                    }
                });
                if do_research {
                    self.start_research();
                }
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
                        ui.label(egui::RichText::new("runs fully on your own computer").color(MUTED).size(12.0));
                        if no_model {
                            ui.add_space(16.0);
                            ui.label(egui::RichText::new("No model loaded yet.").color(RED).size(13.0));
                            ui.label(egui::RichText::new("Download one to start, or put a .gguf file in the models folder and load it in Settings.").color(MUTED).size(11.5));
                            ui.add_space(8.0);
                            if ui.add_enabled(!busy, egui::Button::new(egui::RichText::new("Download a starter model (1.5B)").color(GREEN))).clicked() {
                                do_dl = true;
                            }
                            if busy {
                                ui.label(egui::RichText::new(format!("{} — see the Log under 'details'", self.learn.status)).color(AMBER).size(11.0));
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
                                    let mut body = strip_answer_tags(answer);
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
                    if ui.button(egui::RichText::new("+ New chat").color(GREEN).size(12.0)).clicked() {
                        self.only.show_new = !self.only.show_new;
                        self.only.show_open = false;
                    }
                    if ui.button(egui::RichText::new("Open a database").color(CYAN).size(12.0)).clicked() {
                        self.only.show_open = !self.only.show_open;
                        self.only.show_new = false;
                    }
                });

                // -- open an existing database (so its chats show up) --
                if self.only.show_open {
                    egui::Frame::NONE.fill(SURFACE).inner_margin(egui::Margin::same(8)).stroke(Stroke::new(1.0, LINE)).show(ui, |ui| {
                        ui.label(egui::RichText::new("Open an existing .sqlite file").color(FG).size(11.5));
                        ui.add(egui::TextEdit::singleline(&mut self.only.open_path).desired_width(f32::INFINITY).hint_text("…/memory.sqlite3"));
                        ui.add(egui::TextEdit::singleline(&mut self.only.open_pass).password(true).desired_width(f32::INFINITY).hint_text("Password (if it's locked)"));
                        if ui.button(egui::RichText::new("Open").color(GREEN).size(11.5)).clicked() {
                            open_db = true;
                        }
                    });
                }

                // -- new chat: name, mode, and WHICH database it goes into --
                if self.only.show_new {
                    egui::Frame::NONE.fill(SURFACE).inner_margin(egui::Margin::same(8)).stroke(Stroke::new(1.0, LINE)).show(ui, |ui| {
                        ui.add(egui::TextEdit::singleline(&mut self.only.new_name).desired_width(f32::INFINITY).hint_text("Name"));
                        ui.horizontal(|ui| {
                            ui.label(egui::RichText::new("Mode").color(MUTED).size(11.0));
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
                                DbPick::NewFile => "＋ new file…".to_string(),
                            };
                            egui::ComboBox::from_id_salt("new_db_pick")
                                .selected_text(sel_label)
                                .show_ui(ui, |ui| {
                                    for (i, od) in self.only.dbs.iter().enumerate() {
                                        ui.selectable_value(&mut self.only.new_db_pick, DbPick::Existing(i), db_name(&od.path));
                                    }
                                    ui.selectable_value(&mut self.only.new_db_pick, DbPick::NewFile, "＋ new file…");
                                });
                        });
                        if self.only.new_db_pick == DbPick::NewFile {
                            ui.add(egui::TextEdit::singleline(&mut self.only.new_file_path).desired_width(f32::INFINITY).hint_text("…/chats.sqlite"));
                            ui.horizontal(|ui| {
                                ui.checkbox(&mut self.only.new_file_encrypt, egui::RichText::new("Lock it (encrypt)").color(FG).size(11.0));
                                if self.only.new_file_encrypt {
                                    ui.add(egui::TextEdit::singleline(&mut self.only.new_file_pass).password(true).desired_width(140.0).hint_text("Password"));
                                }
                            });
                        }
                        if ui.button(egui::RichText::new("Create").color(AMBER).size(12.0)).clicked() {
                            new_chat = true;
                        }
                    });
                }

                ui.add_space(6.0);
                ui.separator();
                // -- chat list across all open databases --
                egui::ScrollArea::vertical().auto_shrink([false, false]).max_height(ui.available_height() - 150.0).show(ui, |ui| {
                    if self.only.chats.is_empty() {
                        ui.label(egui::RichText::new("No chats yet — create one or open a database").color(MUTED).size(11.0));
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
                                    if ui.button("Rename").clicked() {
                                        menu_rename = Some((*di, meta.id, meta.name.clone()));
                                    }
                                    if ui.button(egui::RichText::new("Delete").color(RED)).clicked() {
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
                    section(ui, "Open databases");
                    for (i, od) in self.only.dbs.iter().enumerate() {
                        ui.horizontal(|ui| {
                            let enc = if od.db.encrypted() { "🔒 " } else { "" };
                            ui.label(egui::RichText::new(format!("{enc}{}", db_name(&od.path))).color(FG).size(10.5));
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                if ui.button(egui::RichText::new("Close").color(MUTED).size(9.5)).clicked() {
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
                    ui.label(egui::RichText::new("Rename chat").color(AMBER).strong());
                    ui.add_space(4.0);
                    let mut save = false;
                    if let Some((_, _, buf)) = self.only.rename.as_mut() {
                        let resp = ui.add(egui::TextEdit::singleline(buf).desired_width(320.0));
                        if resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                            save = true;
                        }
                    }
                    ui.horizontal(|ui| {
                        if ui.button(egui::RichText::new("Save").color(GREEN)).clicked() {
                            save = true;
                        }
                        if ui.button(egui::RichText::new("Cancel").color(MUTED)).clicked() {
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
                    ui.label(egui::RichText::new("Delete chat").color(RED).strong());
                    ui.label(egui::RichText::new(format!("Database file: {fname}")).color(MUTED).size(11.5));
                    if others > 0 {
                        ui.label(egui::RichText::new(format!("Careful: this file still holds {others} more chat(s) — deleting the file loses those too.")).color(AMBER).size(11.0));
                    }
                    ui.add_space(4.0);
                    ui.horizontal(|ui| {
                        if ui.button(egui::RichText::new("Just this chat").color(FG)).clicked() {
                            do_delete = Some((di, id, false));
                        }
                        if ui.button(egui::RichText::new("Chat + the database file").color(RED)).clicked() {
                            do_delete = Some((di, id, true));
                        }
                        if ui.button(egui::RichText::new("Cancel").color(MUTED)).clicked() {
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
                        .hint_text("Type a message, press Enter to send")
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
                        ui.label(egui::RichText::new("Pick a chat on the left, or create a new one.").color(MUTED).size(13.0));
                    });
                    return;
                }
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("How much it reads:").color(MUTED).size(11.5));
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
                        if ui.add_enabled(!self.only.summarizing, egui::Button::new(egui::RichText::new("Summarize older messages").color(CYAN).size(11.5))).clicked() {
                            summarize = true;
                        }
                    }
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if let (Some(full), Some(ctx)) = (self.only.tok_full, self.only.tok_ctx) {
                            ui.label(egui::RichText::new(format!("sent to AI: {ctx} / {full} tokens")).color(MUTED).size(11.0));
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

    /// Server tab: expose a mobile chat web UI over HTTPS (reach it via your
    /// Tailscale IP from anywhere while the PC is on).
    fn server_body(&mut self, ui: &mut egui::Ui) {
        let mut toggle = false;
        egui::CentralPanel::default()
            .frame(egui::Frame::NONE.fill(BG).inner_margin(egui::Margin::symmetric(20, 16)))
            .show(ui, |ui| {
                ui.label(egui::RichText::new("SERVER").color(AMBER).size(20.0).strong());
                ui.label(egui::RichText::new("Chat with the AI from anywhere — a secure (HTTPS) web page on your Tailscale address.").color(MUTED).size(12.0));
                ui.add_space(14.0);
                let running = self.server.handle.is_some();
                egui::Frame::NONE.fill(PANEL).inner_margin(egui::Margin::same(16)).stroke(Stroke::new(1.0, LINE)).show(ui, |ui| {
                    ui.set_max_width(620.0);
                    let dot = if running { GREEN } else { MUTED };
                    ui.horizontal(|ui| {
                        // pulsing status dot while running
                        let t = ui.input(|i| i.time);
                        let a = if running { 0.5 + 0.5 * (t * 2.0).sin() as f32 } else { 1.0 };
                        ui.label(egui::RichText::new("●").color(dot.gamma_multiply(a)).size(14.0));
                        ui.label(egui::RichText::new(if running { "Server is running" } else { "Server is off" }).color(if running { GREEN } else { FG }).strong());
                    });
                    ui.add_space(8.0);
                    if self.handle.is_none() {
                        ui.label(egui::RichText::new("Load a model first (Settings → Model).").color(RED).size(12.0));
                    }
                    let label = if running { "Stop the server" } else { "Start the server" };
                    let col = if running { RED } else { GREEN };
                    if ui.add_enabled(self.handle.is_some() || running, egui::Button::new(egui::RichText::new(label).color(col).strong())).clicked() {
                        toggle = true;
                    }
                    if !self.server.status.is_empty() {
                        ui.add_space(8.0);
                        ui.label(egui::RichText::new(&self.server.status).color(if running { GREEN } else { MUTED }).size(12.5).monospace());
                    }
                    if running {
                        if let Some(h) = self.server.handle.as_ref() {
                            ui.add_space(8.0);
                            ui.label(egui::RichText::new("Address for your phone (tap to select / copy):").color(MUTED).size(11.0));
                            ui.add(egui::Label::new(egui::RichText::new(&h.url).color(CYAN).size(12.0).monospace()).selectable(true).wrap());
                        }
                        ui.add_space(6.0);
                        ui.label(egui::RichText::new("Open this address on your phone (same Tailscale network). The first time, accept the certificate warning (it's self-signed).").color(MUTED).size(11.0));
                        ui.add_space(4.0);
                        ui.label(egui::RichText::new("⚠ The part after #k= is an access key — anyone with the link can chat with your model. Don't share it. It's new every time you start the server.").color(AMBER).size(10.5));
                    }
                    ui.add_space(6.0);
                    ui.label(egui::RichText::new("Note: port 443 may need extra rights — if so, the server uses 8443 by itself.").color(MUTED).size(10.5));
                });
            });
        if toggle {
            self.toggle_server();
        }
    }

    /// Learning tab: drive the QLoRA pipeline (Python) from the GUI.
    fn learning_body(&mut self, ui: &mut egui::Ui) {
        egui::CentralPanel::default()
            .frame(egui::Frame::NONE.fill(BG).inner_margin(egui::Margin::symmetric(16, 12)))
            .show(ui, |ui| {
                ui.label(egui::RichText::new("LEARNING").color(AMBER).size(20.0).strong());
                ui.label(
                    egui::RichText::new("Teach a model with your own examples, then load it here. It learns on your GPU.")
                        .color(MUTED)
                        .size(12.0),
                );
                ui.add_space(4.0);
                let have_venv = Self::venv_python().exists();
                if !have_venv {
                    ui.label(egui::RichText::new("The setup isn't ready yet — click 'Set things up' first (big download).").color(RED).size(12.0));
                }

                ui.add_space(10.0);
                section(ui, "Setup");
                egui::Grid::new("learncfg").num_columns(2).spacing([10.0, 6.0]).show(ui, |ui| {
                    ui.label(egui::RichText::new("start from").color(MUTED).size(12.0));
                    ui.horizontal(|ui| {
                        let installed = installed_bases(&Self::cwd().join("training/hf-cache/hub"));
                        egui::ComboBox::from_id_salt("base_pick").width(300.0).selected_text(self.learn.base.clone()).show_ui(ui, |ui| {
                            // Already-installed bases first — these train with no download.
                            if !installed.is_empty() {
                                ui.label(egui::RichText::new("installed · no download").color(GREEN).size(10.5));
                                for b in &installed {
                                    if ui.selectable_label(&self.learn.base == b, format!("✓ {b}")).clicked() {
                                        self.learn.base = b.clone();
                                        self.learn.out_name = model_out_name(b);
                                    }
                                }
                                ui.separator();
                                ui.label(egui::RichText::new("download").color(MUTED).size(10.5));
                            }
                            for b in BASES {
                                if installed.iter().any(|i| i == *b) {
                                    continue; // already shown above
                                }
                                if ui.selectable_label(self.learn.base == *b, *b).clicked() {
                                    self.learn.base = (*b).into();
                                    self.learn.out_name = model_out_name(b);
                                }
                            }
                        });
                        if installed.iter().any(|i| i == &self.learn.base) {
                            ui.label(egui::RichText::new("installed").color(GREEN).size(11.0));
                        }
                    });
                    ui.end_row();
                    ui.label(egui::RichText::new("examples").color(MUTED).size(12.0));
                    let cur_label = DATASETS.iter().find(|d| d.1 == self.learn.data.as_str()).map(|d| d.0).unwrap_or("your own");
                    egui::ComboBox::from_id_salt("dataset_pick").width(380.0).selected_text(cur_label).show_ui(ui, |ui| {
                        for (label, path, gen) in DATASETS {
                            if ui.selectable_label(self.learn.data.as_str() == *path, *label).clicked() {
                                self.learn.data = (*path).into();
                                self.learn.gen_script = (*gen).into();
                            }
                        }
                    });
                    ui.end_row();
                    ui.label(egui::RichText::new("file path").color(MUTED).size(12.0));
                    ui.add(egui::TextEdit::singleline(&mut self.learn.data).desired_width(380.0));
                    ui.end_row();
                    ui.label(egui::RichText::new("rounds (epochs)").color(MUTED).size(12.0));
                    ui.add(egui::TextEdit::singleline(&mut self.learn.epochs).desired_width(80.0));
                    ui.end_row();
                });
                ui.add_space(2.0);
                ui.label(egui::RichText::new(
                    "Models you already downloaded need no download. The .gguf files in your model folder can't be trained — teaching always starts from a base model."
                ).color(MUTED).size(10.5));

                ui.add_space(10.0);
                section(ui, "Steps");
                let busy = self.trainer.is_some();
                let py = Self::venv_python().display().to_string();
                ui.horizontal_wrapped(|ui| {
                    if ui.add_enabled(!busy, egui::Button::new("1 · Make the examples")).clicked() {
                        let gen = self.learn.gen_script.clone();
                        self.start("generate dataset", &py, vec![gen]);
                    }
                    if ui.add_enabled(!busy && have_venv, egui::Button::new("2 · Teach + build the model")).clicked() {
                        // RAM/disk-safe pipeline: QLoRA -> mmap merge -> Q8. Removing any
                        // stale adapter forces a fresh train on the selected dataset.
                        let (b, d, e) = (self.learn.base.clone(), self.learn.data.clone(), self.learn.epochs.clone());
                        let lc = Self::llama_dir().map(|p| p.display().to_string()).unwrap_or_default();
                        let out = model_out_name(&b);
                        self.learn.out_name = out.clone();
                        // Force a fresh train on the selected dataset (done in
                        // Rust, not via the shell). Config goes in as env vars.
                        let _ = std::fs::remove_dir_all(Self::cwd().join("training/out"));
                        let envs = vec![
                            ("LC".to_string(), lc),
                            ("BASE".to_string(), b),
                            ("DATA".to_string(), d),
                            ("EPOCHS".to_string(), e),
                            ("OUT".to_string(), format!("models/{out}")),
                        ];
                        self.start_env("train + build", "bash", vec!["training/build_overflow_wt.sh".into()], envs);
                    }
                    if ui.add_enabled(!busy, egui::Button::new("3 · Load the new model")).clicked() {
                        // Select it and jump to Settings so the user picks the
                        // token limit and device in the load panel.
                        self.select_model(Self::cwd().join("models").join(&self.learn.out_name));
                        self.settings_open = true;
                    }
                });
                ui.add_space(4.0);
                if ui.add_enabled(!busy, egui::Button::new("Set things up (one time)")).clicked() {
                    let cmd = "python3 -m venv training/venv && training/venv/bin/pip install -r training/requirements.txt sentencepiece protobuf".to_string();
                    self.start("setup env", "sh", vec!["-c".into(), cmd]);
                }

                ui.add_space(12.0);
                section(ui, "Progress");
                ui.label(egui::RichText::new(format!("status: {}", self.learn.status)).color(if busy { AMBER } else { GREEN }).size(13.0));
                if let Some(l) = self.learn.loss {
                    ui.label(egui::RichText::new(format!("error score (lower is better): {l:.4}")).color(AMBER).size(13.0));
                }
                if let Some((a, b)) = self.learn.step {
                    ui.add(egui::ProgressBar::new(a as f32 / b as f32).desired_width(400.0).fill(AMBER).text(format!("{a}/{b}")));
                }
                ui.add_space(8.0);
                ui.label(egui::RichText::new("(turn on 'details' at the top for the full log)").color(MUTED).size(11.0));
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
        self.drain_research();

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
                    if ui.selectable_label(self.tab == Tab::Server, egui::RichText::new("Server").size(13.0)).clicked() {
                        self.tab = Tab::Server;
                    }
                    if ui.selectable_label(self.tab == Tab::Docs, egui::RichText::new("Help").size(13.0)).clicked() {
                        self.tab = Tab::Docs;
                    }
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.button(egui::RichText::new("[ settings ]").color(FG).size(12.0)).clicked() {
                            self.settings_open = !self.settings_open;
                        }
                        ui.add_space(10.0);
                        if ui
                            .selectable_label(self.dev, egui::RichText::new("details").color(if self.dev { CYAN } else { MUTED }).size(12.0))
                            .clicked()
                        {
                            self.dev = !self.dev;
                        }
                        ui.add_space(12.0);
                        let (plabel, pcol) = if self.perms.sandbox { ("safe mode", GREEN) } else { ("tools on", RED) };
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
                        // Pulse the dot while the model is busy.
                        let pulsing = matches!(self.status, Status::Generating | Status::Loading);
                        let a = if pulsing {
                            let t = ui.input(|i| i.time);
                            0.4 + 0.6 * (0.5 + 0.5 * (t * 3.0).sin() as f32)
                        } else {
                            1.0
                        };
                        ui.label(egui::RichText::new(dot).color(col.gamma_multiply(a)).size(12.0));
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
            Tab::Server => self.server_body(ui),
            Tab::Docs => docs_body(ui),
        }

        let ctx = ui.ctx().clone();
        ctx.request_repaint_after(Duration::from_millis(1000));
        if matches!(self.status, Status::Generating | Status::Loading)
            || self.trainer.is_some()
            || self.research_rx.is_some()
            || self.only.generating
            || self.server.handle.is_some()
        {
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
            ui.label(egui::RichText::new("HELP").color(AMBER).size(20.0).strong());
            ui.label(egui::RichText::new("OVERFLOW — S1BERIA's local cybersecurity AI").color(MUTED).size(12.0));
            egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                doc_section(ui, "First steps", &[
                    "Load a model (Settings -> Model), then type in the Chat tab.",
                    "Everything runs on your own computer. Nothing leaves it, unless you ask",
                    "for a web search or download a model yourself.",
                    "Tabs: Chat (talk + let it do tasks), OnlyChat (chat that uses a database),",
                    "Learning (teach a model), Server (use it from your phone), Help (this).",
                    "Turn on 'details' (top bar) to see speed, word count, the math, and a log.",
                ]);
                doc_section(ui, "OnlyChat (chat that uses a database)", &[
                    "OnlyChat keeps each chat in a database file, not in the AI's memory,",
                    "so even very long chats stay cheap and fast.",
                    "",
                    "# You pick the database for each chat.",
                    "When you make a chat, you choose which .sqlite file it goes in, or make a",
                    "new one. Use 'Open a database' to open a file you already have.",
                    "",
                    "# It can read a memory database.",
                    "Open a database full of your notes (facts, projects, ...) and OnlyChat finds",
                    "the parts that matter and hands them to the AI. So it really knows your",
                    "saved data. It only reads your tables — it never changes them.",
                    "",
                    "# How much it reads (per chat).",
                    "Whole chat / Last few messages / Summary + lookup — the last two keep the",
                    "cost low. Turn on 'details' to see how much is sent to the AI.",
                    "",
                    "# Locking (encryption).",
                    "A database can be locked with a password, so the text on disk is unreadable",
                    "without it.",
                    "",
                    "# Managing chats.",
                    "The gear (⚙) on a chat renames or deletes it (and can delete the whole file).",
                ]);
                doc_section(ui, "Searching the web", &[
                    "The 🌐 button next to the chat box searches the web for your question",
                    "and gives the results to the AI to use.",
                    "You must turn the internet ON first (Settings -> what the AI may do);",
                    "if it's off, it tells you to switch it on. Results can be wrong — treat",
                    "them as hints, not facts.",
                ]);
                doc_section(ui, "Using it from your phone (Server)", &[
                    "The Server tab opens a small, safe (HTTPS) chat page from this computer.",
                    "Load a model, open the Server tab, press 'Start the server'.",
                    "Open the address it shows on your phone (same Tailscale network).",
                    "The first time, accept the certificate warning (it's self-signed), then chat.",
                    "Keep the link private — it holds an access key. It uses the same one model.",
                ]);
                doc_section(ui, "Tips for asking", &[
                    "# Say clearly what you want and the situation.",
                    "Name the target, the limits, and the output you want.",
                    "Weak:  \"scan the network\"",
                    "Better: \"nmap a SYN scan of 10.0.0.0/24, top 100 ports, greppable output\".",
                    "",
                    "# Give it a role up front.",
                    "Settings has a system prompt; say you're doing authorized security work,",
                    "so it answers straight without holding back.",
                    "",
                    "# Ask for a plan first, then the doing.",
                    "\"First outline the approach, then run the commands\" — so you can check it.",
                    "",
                    "# Let it use its tools.",
                    "Ask it to read files or run commands; it does that by itself.",
                    "Be concrete: name the files, hosts, and exact commands when you can.",
                    "",
                    "# A bigger model gives better answers.",
                    "The 1.5B is fast but simple; the 7B thinks much better (but slower).",
                ]);
                doc_section(ui, "Teaching a model", &[
                    "The Learning tab teaches a model with YOUR examples (on the GPU).",
                    "",
                    "# 1. Make the examples.",
                    "One line per example, in chat form:",
                    "  {\"messages\":[{\"role\":\"user\",\"content\":\"Q\"},{\"role\":\"assistant\",\"content\":\"A\"}]}",
                    "Give good, varied examples — the AI learns from exactly this.",
                    "",
                    "# 2. Teach.",
                    "Learning tab: choose what to start from, the examples, the rounds -> Teach.",
                    "Watch the error score fall. Models you already have show as 'installed -",
                    "no download'; pick one to skip the download.",
                    "",
                    "# 3. Load it.",
                    "The app builds the new model and you press 'Load the new model'. Done —",
                    "it's yours now.",
                    "",
                    "# Tips.",
                    "50-500 good examples already change how it answers. More rounds = stronger,",
                    "but too many can make it memorize. Keep a few test questions to compare.",
                ]);
                doc_section(ui, "What the AI may do (tools & safety)", &[
                    "The AI can read and write files, list folders, and run commands for you.",
                    "By default it's in safe mode (no access). Turn things on in Settings.",
                    "Work folder = it works freely inside; outside it, it asks you first.",
                    "Commands run in a locked box (bubblewrap): work folder only, no internet",
                    "unless you allow it. You say Yes or No to anything risky, right in the app.",
                ]);
                doc_section(ui, "The overflow_wt model", &[
                    "overflow_wt-7b-q8_0 is OVERFLOW's main model: Qwen2.5-Coder-7B, taught to",
                    "act like an information-security teacher. It explains security clearly —",
                    "crypto, network and web security, binary exploitation, gaining access,",
                    "malware, forensics and handling incidents.",
                    "It comes with the installed package and loads by default.",
                    "'wt' = walkthrough: it teaches the thinking, not just the answer.",
                    "For authorized security work and learning.",
                ]);
                doc_section(ui, "Models", &[
                    "Settings -> Model: pick a model and Download, or load your own .gguf file.",
                    "Sizes go from 0.5B (tiny and fast) to 14B (smartest).",
                    "Bigger = smarter but needs more memory. Thanks to our offload work, even a",
                    "14B still runs on a 12 GB card at full quality — just slower.",
                    "Models live in ~/.local/share/overflow/models (or ./models in a dev build);",
                    "the package also installs to /usr/share/overflow/models, and OVERFLOW also",
                    "looks in /mnt/ai/Overflow Models.",
                    "'Unload' frees the memory; you can load another one anytime.",
                ]);
                doc_section(ui, "Ready-made examples (Learning tab)", &[
                    "A few example sets you can teach from, right in the Learning tab:",
                    "security teacher, name + basics, CTF / binary exploitation, web security.",
                    "Or bring your own file (chat form). Teach -> build -> load.",
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
