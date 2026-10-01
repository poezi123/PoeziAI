//! OVERFLOW — a local AI, native desktop GUI (egui/eframe).
//!
//! The window is a thin client: it renders the transcript and forwards messages
//! to the inference worker thread, streaming tokens back into the view. No web
//! stack, pure Rust. The whole look is drawn from the OVERFLOW palette.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod hw;
mod worker;

use std::sync::{Arc, Mutex};
use std::time::Duration;

use eframe::egui;
use egui::{Color32, FontFamily, Pos2, Sense, Stroke, Vec2};

use worker::{FromWorker, Handle, ToWorker};

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

fn default_model_path() -> std::path::PathBuf {
    std::env::args()
        .nth(1)
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| "models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf".into())
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
    hw: Arc<Mutex<hw::HwStats>>,
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
            hw: hw::spawn(),
        }
    }

    fn send_current_input(&mut self) {
        let text = self.input.trim().to_string();
        if text.is_empty() {
            return;
        }
        self.transcript.push(Msg { role: Role::User, text: text.clone() });
        self.transcript.push(Msg { role: Role::Assistant, text: String::new() });
        let _ = self.handle.tx.send(ToWorker::User(text));
        self.status = Status::Generating;
        self.input.clear();
    }

    fn drain_events(&mut self) {
        while let Ok(ev) = self.handle.rx.try_recv() {
            match ev {
                FromWorker::Ready => self.status = Status::Ready,
                FromWorker::Token(s) => {
                    if let Some(last) = self.transcript.last_mut() {
                        if matches!(last.role, Role::Assistant) {
                            last.text.push_str(&s);
                        }
                    }
                    self.status = Status::Generating;
                }
                FromWorker::Stats { tok_per_s } => self.tok_per_s = tok_per_s,
                FromWorker::Done => self.status = Status::Ready,
                FromWorker::Error(e) => self.status = Status::Error(e),
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

                section(ui, "HARDWARE");
                let hw = self.hw.lock().map(|g| *g).unwrap_or_default();
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

        egui::Panel::top("titlebar")
            .frame(egui::Frame::NONE.fill(PANEL).inner_margin(egui::Margin::symmetric(16, 12)))
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    draw_logo(ui, 26.0);
                    ui.add_space(10.0);
                    ui.label(egui::RichText::new("OVERFLOW").color(FG).size(20.0).strong());
                    ui.add_space(8.0);
                    ui.label(egui::RichText::new("local engine").color(MUTED).size(12.0));
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        // settings button (top-right)
                        if ui
                            .button(egui::RichText::new("[ settings ]").color(FG).size(12.0))
                            .clicked()
                        {
                            self.settings_open = !self.settings_open;
                        }
                        ui.add_space(12.0);
                        // security posture
                        let (plabel, pcol) = if self.perms.sandbox {
                            ("sandboxed", GREEN)
                        } else {
                            ("tools enabled", RED)
                        };
                        ui.label(egui::RichText::new(plabel).color(pcol).size(12.0));
                        ui.add_space(12.0);
                        // model status
                        let (dot, label, col) = match &self.status {
                            Status::Loading => ("●", "loading model…".to_string(), AMBER),
                            Status::Ready => ("●", "ready".to_string(), GREEN),
                            Status::Generating => ("●", "thinking…".to_string(), AMBER),
                            Status::Error(e) => ("●", format!("error: {e}"), RED),
                        };
                        ui.label(egui::RichText::new(label).color(MUTED).size(12.0));
                        ui.add_space(6.0);
                        ui.label(egui::RichText::new(dot).color(col).size(12.0));
                    });
                });
            });

        let can_send = matches!(self.status, Status::Ready);

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
                        ui.label(
                            egui::RichText::new("S1BERIA'S CYBERSECURITY AI")
                                .color(FG)
                                .size(14.0)
                                .strong(),
                        );
                        ui.label(
                            egui::RichText::new("runs entirely on this machine")
                                .color(MUTED)
                                .size(12.0),
                        );
                    });
                }
                egui::ScrollArea::vertical()
                    .stick_to_bottom(true)
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        let generating = matches!(self.status, Status::Generating);
                        let last = self.transcript.len().saturating_sub(1);
                        for (i, msg) in self.transcript.iter().enumerate() {
                            let (who, col) = match msg.role {
                                Role::User => ("you", AMBER),
                                Role::Assistant => ("overflow", GREEN),
                            };
                            ui.add_space(10.0);
                            ui.label(egui::RichText::new(format!("{who} ▸")).color(col).size(12.0).strong());
                            let mut body = msg.text.clone();
                            if generating && i == last && matches!(msg.role, Role::Assistant) {
                                body.push('▌');
                            }
                            ui.label(egui::RichText::new(body).color(FG));
                        }
                        ui.add_space(8.0);
                    });
            });

        let ctx = ui.ctx().clone();
        self.show_settings(&ctx);

        // Keep hardware stats ticking even while idle.
        ctx.request_repaint_after(Duration::from_millis(1000));
        if matches!(self.status, Status::Generating | Status::Loading) {
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

    // hex frame
    let hex = vec![
        p(32.0, 6.0),
        p(54.0, 19.0),
        p(54.0, 45.0),
        p(32.0, 58.0),
        p(10.0, 45.0),
        p(10.0, 19.0),
    ];
    painter.add(egui::Shape::closed_line(hex, Stroke::new(2.4 * s, AMBER)));

    // vertical slit pupil (a vesica, as a filled polygon)
    let pupil = vec![p(32.0, 20.0), p(36.0, 32.0), p(32.0, 44.0), p(28.0, 32.0)];
    painter.add(egui::Shape::convex_polygon(pupil, AMBER, Stroke::NONE));
}
