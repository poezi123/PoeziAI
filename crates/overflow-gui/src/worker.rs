//! Inference worker thread — runs the tool-calling agent for the GUI.
//!
//! Owns the model + KV cache (sidesteps the mmap borrow) and drives
//! `poezi_model::agent`. Tool activity and tokens stream back to the UI; when a
//! tool needs manual approval the worker asks the UI and blocks until it answers.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::Arc;
use std::thread;
use std::time::Instant;

use poezi_gguf::GgufFile;
use poezi_model::agent::{self, Event, GenCfg, Host, Perms, ToolCall};
use poezi_model::{LoadOpts, Model, Tokenizer};

/// UI -> worker.
pub enum ToWorker {
    User { text: String, perms: Perms, gen: GenCfg },
    Approve(bool),
    /// OnlyChat: a plain chat completion over an assembled context (no tools,
    /// fresh KV cache). `turns` is (is_user, text) oldest-first ending with the
    /// new user message; `prelude` is an optional summary note.
    ChatOnly { turns: Vec<(bool, String)>, prelude: Option<String>, gen: GenCfg },
    /// OnlyChat: summarize the given text silently; replies with `Summary`.
    Summarize { text: String, gen: GenCfg },
    /// OnlyChat: count tokens in the whole chat and in the context a turn would
    /// actually feed; replies with `OnlyChatTokens`.
    CountTokens { full: String, ctx: String },
    /// Server mode: a chat turn from the web UI. When `perms` grant tools the
    /// agent runs with them; otherwise it's a plain chat. The full reply is sent
    /// back on `reply` (no GUI streaming).
    ServerChat { turns: Vec<(bool, String)>, perms: Perms, gen: GenCfg, reply: Sender<String> },
}

/// worker -> UI.
pub enum FromWorker {
    Ready,
    Backend(String),
    /// A one-off informational line for the Log (e.g. why the GPU was skipped).
    Note(String),
    Token(String),
    ToolCall(String),
    ToolResult(String),
    Approval { desc: String, outside: bool },
    Stats { tok_per_s: f32, produced: u32 },
    /// Model loading: bytes streamed into VRAM so far, out of the total.
    LoadProgress { done: u64, total: u64 },
    /// OnlyChat: a summary produced by `Summarize`, to be stored by the UI.
    Summary(String),
    /// OnlyChat: (whole-chat tokens, context tokens) for the active chat.
    OnlyChatTokens { full: usize, ctx: usize },
    Done,
    Error(String),
}

pub struct Handle {
    pub tx: Sender<ToWorker>,
    pub rx: Receiver<FromWorker>,
    pub cancel: Arc<AtomicBool>,
}

pub fn spawn(model_path: PathBuf, opts: LoadOpts) -> Handle {
    let (tx_in, rx_in) = std::sync::mpsc::channel();
    let (tx_out, rx_out) = std::sync::mpsc::channel();
    let cancel = Arc::new(AtomicBool::new(false));
    let cancel_worker = Arc::clone(&cancel);
    thread::spawn(move || {
        if let Err(e) = run(model_path, opts, &rx_in, &tx_out, &cancel_worker) {
            let _ = tx_out.send(FromWorker::Error(e));
        }
    });
    Handle { tx: tx_in, rx: rx_out, cancel }
}

fn run(
    path: PathBuf,
    opts: LoadOpts,
    rx_in: &Receiver<ToWorker>,
    tx_out: &Sender<FromWorker>,
    cancel: &AtomicBool,
) -> Result<(), String> {
    let gguf = GgufFile::open(&path).map_err(|e| format!("open model: {e}"))?;
    let tok = Tokenizer::from_gguf(&gguf).map_err(|e| format!("tokenizer: {e}"))?;
    let model = Model::load_with_opts(&gguf, opts, &mut |p| {
        let _ = tx_out.send(FromWorker::LoadProgress { done: p.done_bytes, total: p.total_bytes });
    })
    .map_err(|e| format!("load model: {e}"))?;
    let mut cache = model.new_cache();
    let mut first_turn = true;

    // Friendly backend label with the layer split so the UI can show exactly
    // what was chosen (all on GPU, a GPU/CPU split, or pure CPU).
    let label = match model.backend() {
        "GPU" => format!("GPU ({} layers)", model.n_layers()),
        "GPU+CPU" => format!("GPU+CPU ({}/{} layers on GPU)", model.gpu_layers(), model.n_layers()),
        _ => "CPU".to_string(),
    };
    let _ = tx_out.send(FromWorker::Backend(label));
    // If a GPU was wanted but failed, tell the user why it is on the CPU.
    if let Some(e) = model.gpu_error() {
        let _ = tx_out.send(FromWorker::Note(format!("GPU not available — {e}; running on CPU")));
    }
    let _ = tx_out.send(FromWorker::Ready);

    while let Ok(msg) = rx_in.recv() {
        match msg {
            ToWorker::User { text, perms, gen } => {
                cancel.store(false, Ordering::Relaxed); // clear any prior stop
                let mut host = GuiHost { tx: tx_out, rx: rx_in, start: None, n: 0 };
                // Up to 20 tool rounds so the agent can investigate autonomously
                // without the user re-prompting after each step.
                agent::run(&model, &tok, &mut cache, first_turn, &text, &perms, &gen, cancel, &mut host, 20);
                first_turn = false;
                let _ = tx_out.send(FromWorker::Done);
            }
            ToWorker::ChatOnly { turns, prelude, gen } => {
                cancel.store(false, Ordering::Relaxed);
                // Fresh cache: OnlyChat rebuilds the context from the store each
                // turn, so nothing from a previous turn must linger.
                let mut chat_cache = model.new_cache();
                let mut host = GuiHost { tx: tx_out, rx: rx_in, start: None, n: 0 };
                agent::chat_plain(
                    &model, &tok, &mut chat_cache, &turns, prelude.as_deref(), &gen, cancel, &mut host,
                );
                let _ = tx_out.send(FromWorker::Done);
            }
            ToWorker::Summarize { text, gen } => {
                cancel.store(false, Ordering::Relaxed);
                let mut chat_cache = model.new_cache();
                let mut silent = SilentHost;
                let prompt = format!(
                    "Fasse das folgende Gespräch in wenigen Sätzen zusammen. Nenne die wichtigsten \
                     Fakten, Entscheidungen und offenen Punkte, damit man später daran anknüpfen kann. \
                     Keine Einleitung, nur die Zusammenfassung.\n\n{text}"
                );
                let mut g = gen.clone();
                g.thinking = false;
                g.max_tokens = g.max_tokens.min(400);
                let summary = agent::chat_plain(
                    &model, &tok, &mut chat_cache, &[(true, prompt)], None, &g, cancel, &mut silent,
                );
                let _ = tx_out.send(FromWorker::Summary(summary));
                let _ = tx_out.send(FromWorker::Done);
            }
            ToWorker::CountTokens { full, ctx } => {
                // encode(text, add_bos=false, special=true) mirrors how the
                // agent feeds text, so the count matches what the model sees.
                let full = tok.encode(&full, false, true).len();
                let ctx = tok.encode(&ctx, false, true).len();
                let _ = tx_out.send(FromWorker::OnlyChatTokens { full, ctx });
            }
            ToWorker::ServerChat { turns, perms, gen, reply } => {
                cancel.store(false, Ordering::Relaxed);
                let tools = !perms.sandbox
                    && (perms.fs_read || perms.fs_write || perms.shell || perms.network);
                let mut c = model.new_cache();
                if tools {
                    // Run the full tool-using agent over the conversation. The
                    // earlier turns are fed as context, then the newest question.
                    let (last_user, history) = match turns.split_last() {
                        Some((l, h)) => (l.1.clone(), h),
                        None => (String::new(), &[][..]),
                    };
                    let mut query = String::new();
                    for (is_user, t) in history {
                        query.push_str(if *is_user { "User: " } else { "Assistant: " });
                        query.push_str(t);
                        query.push('\n');
                    }
                    query.push_str(&last_user);
                    // The phone set these permissions deliberately, so tool calls
                    // they allow are auto-approved (no interactive prompt here).
                    let mut host = ServerCollect { out: String::new() };
                    agent::run(&model, &tok, &mut c, true, &query, &perms, &gen, cancel, &mut host, 20);
                    let _ = reply.send(host.out);
                } else {
                    let mut silent = SilentHost;
                    let out = agent::chat_plain(&model, &tok, &mut c, &turns, None, &gen, cancel, &mut silent);
                    let _ = reply.send(out);
                }
            }
            ToWorker::Approve(_) => {}
        }
    }
    Ok(())
}

/// Server-mode host: collects the visible answer text (token stream) and
/// auto-approves tool calls the phone's permissions allow.
struct ServerCollect {
    out: String,
}
impl Host for ServerCollect {
    fn approve(&mut self, _call: &ToolCall, _outside: bool) -> bool {
        true
    }
    fn event(&mut self, ev: Event) {
        if let Event::Token(s) = ev {
            self.out.push_str(&s);
        }
    }
}

/// Discards all events — used when generating a summary we don't want streamed
/// into the visible transcript.
struct SilentHost;
impl Host for SilentHost {
    fn approve(&mut self, _call: &ToolCall, _outside: bool) -> bool {
        false
    }
    fn event(&mut self, _ev: Event) {}
}

/// Bridges the agent core to the GUI channels.
struct GuiHost<'a> {
    tx: &'a Sender<FromWorker>,
    rx: &'a Receiver<ToWorker>,
    start: Option<Instant>,
    n: u32,
}

impl Host for GuiHost<'_> {
    fn approve(&mut self, call: &ToolCall, outside: bool) -> bool {
        let desc = format!("{}({})", call.name, agent::compact(&call.args));
        let _ = self.tx.send(FromWorker::Approval { desc, outside });
        loop {
            match self.rx.recv() {
                Ok(ToWorker::Approve(b)) => return b,
                Ok(_) => continue,
                Err(_) => return false,
            }
        }
    }

    fn event(&mut self, ev: Event) {
        match ev {
            Event::Token(s) => {
                let t0 = *self.start.get_or_insert_with(Instant::now);
                self.n += 1;
                let secs = t0.elapsed().as_secs_f32();
                if secs > 0.0 {
                    let _ = self.tx.send(FromWorker::Stats { tok_per_s: self.n as f32 / secs, produced: self.n });
                }
                let _ = self.tx.send(FromWorker::Token(s.to_string()));
            }
            Event::ToolCall(c) => {
                let _ = self
                    .tx
                    .send(FromWorker::ToolCall(format!("{}({})", c.name, agent::compact(&c.args))));
            }
            Event::ToolResult(r) => {
                let _ = self.tx.send(FromWorker::ToolResult(r.chars().take(400).collect()));
            }
        }
    }
}

/// Build agent permissions from the GUI settings.
#[allow(clippy::too_many_arguments)]
pub fn perms(
    sandbox: bool,
    fs_read: bool,
    fs_write: bool,
    shell: bool,
    network: bool,
    workspace: &str,
    confirm_outside: bool,
) -> Perms {
    Perms {
        sandbox,
        fs_read,
        fs_write,
        shell,
        network,
        workspace: PathBuf::from(workspace),
        confirm_outside,
    }
}
