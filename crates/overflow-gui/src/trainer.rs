//! Runs the training scripts as subprocesses and streams their output to the UI.
//!
//! The GUI doesn't train in Rust — it drives the proven Python/QLoRA pipeline in
//! `training/` and shows progress live. stdout and stderr are read on background
//! threads and sent line by line over a channel.

use std::io::{BufReader, Read};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::mpsc::{channel, Receiver, Sender};
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
                    thread::spawn(move || stream_lines(o, &tx))
                });
                let tx3 = tx.clone();
                let h2 = err.map(|e| {
                    let tx = tx3.clone();
                    thread::spawn(move || stream_lines(e, &tx))
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

/// Read a child stream and emit one `Line` per terminal line, breaking on BOTH
/// `\n` and `\r`. Tools like `curl` redraw their progress meter in place with
/// `\r` (no newline), so a plain line reader would show nothing until the end —
/// splitting on `\r` too lets each progress update reach the UI live.
fn stream_lines(reader: impl Read, tx: &Sender<TrainerMsg>) {
    let mut br = BufReader::new(reader);
    let mut buf: Vec<u8> = Vec::new();
    let mut byte = [0u8; 1];
    loop {
        match br.read(&mut byte) {
            Ok(0) => break,
            Ok(_) => {
                let b = byte[0];
                if b == b'\n' || b == b'\r' {
                    if !buf.is_empty() {
                        let _ = tx.send(TrainerMsg::Line(String::from_utf8_lossy(&buf).into_owned()));
                        buf.clear();
                    }
                } else {
                    buf.push(b);
                }
            }
            Err(_) => break,
        }
    }
    if !buf.is_empty() {
        let _ = tx.send(TrainerMsg::Line(String::from_utf8_lossy(&buf).into_owned()));
    }
}

/// True if a line looks like a download/progress meter (curl `-#` bar or its
/// default table) — used to collapse repeated updates into one in-place line.
pub fn is_progress(line: &str) -> bool {
    line.contains('%') && (line.contains('#') || line.contains("Dload") || line.contains("Xferd"))
}

/// Pull a percentage (e.g. `38.2%`) out of a progress line.
pub fn parse_percent(line: &str) -> Option<f32> {
    let i = line.rfind('%')?;
    let head = &line[..i];
    let start = head
        .rfind(|c: char| !(c.is_ascii_digit() || c == '.'))
        .map(|p| p + 1)
        .unwrap_or(0);
    head[start..].parse().ok()
}

/// Pull `{'loss': 0.42, ...}` out of a transformers log line.
pub fn parse_loss(line: &str) -> Option<f32> {
    let i = line.find("'loss':")?;
    let rest = &line[i + 7..];
    let rest = rest.trim_start().trim_start_matches(['\'', '"']);
    let end = rest.find(['\'', '"', ',', '}']).unwrap_or(rest.len());
    rest[..end].trim().parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percent_from_curl_bar() {
        assert_eq!(parse_percent("######                    38.2%"), Some(38.2));
        assert_eq!(parse_percent("#################### 100.0%"), Some(100.0));
        assert_eq!(parse_percent("no percent here"), None);
    }

    #[test]
    fn progress_detection() {
        assert!(is_progress("####   12.0%"));
        assert!(is_progress("  % Total    Dload  Upload   5.0%"));
        assert!(!is_progress("{'loss': 0.2, 'epoch': 1.0}"));
    }
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
