//! Runs the training scripts as subprocesses and streams their output to the UI.
//!
//! The GUI doesn't train in Rust — it drives the proven Python/QLoRA pipeline in
//! `training/` and shows progress live. stdout and stderr are read on background
//! threads and sent line by line over a channel.

use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::mpsc::{channel, Receiver};
use std::thread;

pub enum TrainerMsg {
    Line(String),
    Done(bool),
}

pub struct Trainer {
    pub rx: Receiver<TrainerMsg>,
}

/// Spawn `program args…` in `cwd` with extra env vars, streaming output.
pub fn run(
    program: &str,
    args: Vec<String>,
    envs: Vec<(String, String)>,
    cwd: PathBuf,
) -> Trainer {
    let (tx, rx) = channel();
    let program = program.to_string();
    thread::spawn(move || {
        let mut cmd = Command::new(&program);
        cmd.args(&args)
            .current_dir(&cwd)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        for (k, v) in &envs {
            cmd.env(k, v);
        }
        match cmd.spawn() {
            Ok(mut child) => {
                let out = child.stdout.take();
                let err = child.stderr.take();
                let tx2 = tx.clone();
                let h1 = out.map(|o| {
                    let tx = tx2.clone();
                    thread::spawn(move || {
                        for l in BufReader::new(o).lines().map_while(Result::ok) {
                            let _ = tx.send(TrainerMsg::Line(l));
                        }
                    })
                });
                let tx3 = tx.clone();
                let h2 = err.map(|e| {
                    let tx = tx3.clone();
                    thread::spawn(move || {
                        for l in BufReader::new(e).lines().map_while(Result::ok) {
                            let _ = tx.send(TrainerMsg::Line(l));
                        }
                    })
                });
                let status = child.wait();
                if let Some(h) = h1 {
                    let _ = h.join();
                }
                if let Some(h) = h2 {
                    let _ = h.join();
                }
                let _ = tx.send(TrainerMsg::Done(status.map(|s| s.success()).unwrap_or(false)));
            }
            Err(e) => {
                let _ = tx.send(TrainerMsg::Line(format!("spawn error: {e}")));
                let _ = tx.send(TrainerMsg::Done(false));
            }
        }
    });
    Trainer { rx }
}

/// Pull `{'loss': 0.42, ...}` out of a transformers log line.
pub fn parse_loss(line: &str) -> Option<f32> {
    let i = line.find("'loss':")?;
    let rest = &line[i + 7..];
    let rest = rest.trim_start().trim_start_matches(['\'', '"']);
    let end = rest.find(['\'', '"', ',', '}']).unwrap_or(rest.len());
    rest[..end].trim().parse().ok()
}

/// Pull `(step, total)` out of a tqdm-style ` 12/30 ` fragment.
pub fn parse_progress(line: &str) -> Option<(u32, u32)> {
    // find the last " a/b " pattern
    let mut best = None;
    for part in line.split_whitespace() {
        if let Some((a, b)) = part.split_once('/') {
            if let (Ok(a), Ok(b)) = (a.parse::<u32>(), b.parse::<u32>()) {
                if b > 0 {
                    best = Some((a, b));
                }
            }
        }
    }
    best
}
