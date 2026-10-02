//! OVERFLOW — a local AI, native desktop GUI (egui/eframe).
//!
//! The window is a thin client: it renders the transcript and forwards messages
//! to the inference worker thread, streaming tokens back into the view. No web
//! stack, pure Rust. The whole look is drawn from the OVERFLOW palette.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod hw;
mod trainer;
mod worker;

use std::sync::{Arc, Mutex};
use std::time::Duration;

use eframe::egui;
use egui::{Color32, FontFamily, Pos2, Sense, Stroke, Vec2};

use worker::{FromWorker, Handle, ToWorker};

#[derive(PartialEq, Clone, Copy)]
enum Tab {
    Chat,
    Learning,
}

/// State for the Learning tab (drives the Python QLoRA pipeline).
struct LearnState {
    base: String,
    data: String,
    epochs: String,
    loss: Option<f32>,
    step: Option<(u32, u32)>,
    status: String,
}

impl Default for LearnState {
    fn default() -> Self {
        LearnState {
            base: "Qwen/Qwen2.5-Coder-1.5B-Instruct".into(),
            data: "training/data/overflow.jsonl".into(),
            epochs: "3".into(),
            loss: None,
            step: None,
            status: "idle".into(),
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

/// The math behind what the engine does — shown in the dev panel.
const FORMULAS: [&str; 6] = [
    "softmax   pᵢ = e^(zᵢ/T) / Σⱼ e^(zⱼ/T)",
    "RMSNorm   y = x·w / √(mean(x²)+ε)",
    "RoPE      rotate (xₖ, xₖ₊d/2) by θ·pos",
    "attention softmax(Q·Kᵀ/√d)·V",
    "SwiGLU    SiLU(x·Wg) ⊙ (x·Wu)",
    "loss      L = −Σ y·log p   (cross-entropy)",
];

/// Model to load: an explicit path argument wins; otherwise the first of these
/// that exists (prefer the smartest available).
fn default_model_path() -> std::path::PathBuf {
    if let Some(arg) = std::env::args().nth(1) {
        return arg.into();
    }
    let prefs = [
        "models/qwen2.5-coder-7b-instruct-q4_k_m.gguf",
        "models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
        "models/qwen2.5-coder-1.5b-instruct-q8_0.gguf",
    ];
    prefs
        .iter()
        .find(|p| std::path::Path::new(p).exists())
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| prefs[0].into())
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

/// What the AI is allowed to do on the machine. Default is fully sandboxed:
/// the chat can neither touch the filesystem nor run anything. Enforcement of
/// these lands with the agent/tools (Phase D); they define what it may do.
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
            sandbox: true,
            fs_read: false,
            fs_write: false,
            shell: false,
            network: false,
            workspace,
            confirm_outside: true,
        }
    }
}

struct OverflowApp {
    handle: Handle,
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
    learn: LearnState,
    trainer: Option<trainer::Trainer>,
}

impl OverflowApp {
    fn new(model_path: std::path::PathBuf) -> Self {
        OverflowApp {
            handle: worker::spawn(model_path),
            status: Status::Loading,
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
            learn: LearnState::default(),
            trainer: None,
        }
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
                        if let Some(loss) = trainer::parse_loss(&l) {
                            self.learn.loss = Some(loss);
                        }
                        if let Some(p) = trainer::parse_progress(&l) {
                            self.learn.step = Some(p);
                        }
                        self.console.push(l);
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

    /// Reload the chat model (e.g. after training produced a new GGUF).
    fn load_model(&mut self, path: std::path::PathBuf) {
        self.log(format!("loading model {}", path.display()));
        self.handle = worker::spawn(path);
        self.status = Status::Loading;
        self.backend = "…".into();
        self.transcript.clear();
        self.tab = Tab::Chat;
    }

    fn send_current_input(&mut self) {
        let text = self.input.trim().to_string();
        if text.is_empty() {
            return;
        }
        self.transcript.push(Msg { role: Role::User, text: text.clone() });
        self.transcript.push(Msg { role: Role::Assistant, text: String::new() });
        let perms = worker::perms(
            self.perms.sandbox,
            self.perms.fs_read,
            self.perms.fs_write,
            self.perms.shell,
            self.perms.network,
            &self.perms.workspace,
            self.perms.confirm_outside,
        );
        let _ = self.handle.tx.send(ToWorker::User { text, perms });
        self.status = Status::Generating;
        self.input.clear();
    }

    fn drain_events(&mut self) {
        while let Ok(ev) = self.handle.rx.try_recv() {
            match ev {
                FromWorker::Ready => self.status = Status::Ready,
                FromWorker::Backend(b) => {
                    self.log(format!("[backend] {b}"));
                    self.backend = b;
                }
                FromWorker::Token(s) => {
                    if let Some(last) = self.transcript.last_mut() {
                        if matches!(last.role, Role::Assistant) {
                            last.text.push_str(&s);
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
                FromWorker::Done => self.status = Status::Ready,
                FromWorker::Error(e) => {
                    self.log(format!("[error] {e}"));
                    self.status = Status::Error(e);
                }
            }
        }
    }

    /// The top-right settings window: live hardware, generation speed, and the
    /// permission model (sandboxed by default).
    fn show_settings(&mut self, ctx: &egui::Context) {
        let mut open = self.settings_open;
        egui::Window::new("settings")
            .open(&mut open)
            .anchor(egui::Align2::RIGHT_TOP, egui::vec2(-12.0, 52.0))
            .collapsible(false)
            .resizable(false)
            .frame(
                egui::Frame::NONE
                    .fill(PANEL)
                    .inner_margin(egui::Margin::same(16))
                    .stroke(Stroke::new(1.0, LINE)),
            )
            .show(ctx, |ui| {
                ui.set_width(300.0);

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
            });
        self.settings_open = open;
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
                if let Some(l) = self.learn.loss {
                    ui.label(egui::RichText::new(format!("loss      {l:.4}")).color(AMBER).size(12.0));
                }
                if let Some((a, b)) = self.learn.step {
                    ui.label(egui::RichText::new(format!("step      {a}/{b}")).color(MUTED).size(12.0));
                }

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
                            let _ = self.handle.tx.send(ToWorker::Approve(true));
                            self.pending_approval = None;
                        }
                        if ui.button(egui::RichText::new("Deny").color(RED)).clicked() {
                            let _ = self.handle.tx.send(ToWorker::Approve(false));
                            self.pending_approval = None;
                        }
                    });
                });
        }

        egui::Panel::bottom("composer")
            .frame(egui::Frame::NONE.fill(BG).inner_margin(egui::Margin::symmetric(16, 12)))
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new(">").color(AMBER).strong());
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
                    ui.vertical_centered(|ui| {
                        ui.label(egui::RichText::new("OVERFLOW").color(AMBER).size(28.0).strong());
                        ui.label(egui::RichText::new("S1BERIA'S CYBERSECURITY AI").color(FG).size(14.0).strong());
                        ui.label(egui::RichText::new("runs entirely on this machine").color(MUTED).size(12.0));
                    });
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
                                _ => {
                                    let (who, col) = match msg.role {
                                        Role::User => ("you", AMBER),
                                        _ => ("overflow", GREEN),
                                    };
                                    ui.add_space(10.0);
                                    ui.label(egui::RichText::new(format!("{who} ▸")).color(col).size(12.0).strong());
                                    let mut body = msg.text.clone();
                                    if active && matches!(msg.role, Role::Assistant) {
                                        body.push('▌');
                                    }
                                    ui.label(egui::RichText::new(body).color(FG));
                                }
                            }
                        }
                        ui.add_space(8.0);
                    });
            });
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
                        self.start("generate dataset", &py, vec!["training/data_gen.py".into()]);
                    }
                    if ui.add_enabled(!busy && have_venv, egui::Button::new("2 · Train (QLoRA)")).clicked() {
                        let (b, d, e) = (self.learn.base.clone(), self.learn.data.clone(), self.learn.epochs.clone());
                        self.start("train", &py, vec![
                            "training/train_qlora.py".into(), "--base".into(), b,
                            "--data".into(), d, "--epochs".into(), e,
                        ]);
                    }
                    if ui.add_enabled(!busy && have_venv, egui::Button::new("3 · Merge + export GGUF")).clicked() {
                        let b = self.learn.base.clone();
                        let cmd = format!("{py} training/merge.py --base {b} && training/export_gguf.sh");
                        self.start("merge + export", "sh", vec!["-c".into(), cmd]);
                    }
                    if ui.add_enabled(!busy, egui::Button::new("4 · Load trained model")).clicked() {
                        self.load_model("models/overflow-1.5b-q8_0.gguf".into());
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
                    if ui.selectable_label(self.tab == Tab::Learning, egui::RichText::new("Learning").size(13.0)).clicked() {
                        self.tab = Tab::Learning;
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
        match self.tab {
            Tab::Chat => self.chat_body(ui),
            Tab::Learning => self.learning_body(ui),
        }

        let ctx = ui.ctx().clone();
        self.show_settings(&ctx);
        ctx.request_repaint_after(Duration::from_millis(1000));
        if matches!(self.status, Status::Generating | Status::Loading) || self.trainer.is_some() {
            ctx.request_repaint();
        }
    }
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
