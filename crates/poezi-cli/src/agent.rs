//! CLI frontend for the tool-calling agent (core lives in `poezi_model::agent`).

use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::process::ExitCode;

use poezi_gguf::GgufFile;
use poezi_model::agent::{self, Event, Host, Perms, ToolCall};
use poezi_model::{Model, Tokenizer};

pub struct AgentOpts {
    pub workspace: PathBuf,
    pub auto_approve: bool,
    pub network: bool,
    pub max_rounds: usize,
}

/// Terminal host: approve via stdin, show tool activity on stderr, buffer the
/// visible assistant text so only the final answer is printed to stdout.
struct CliHost {
    auto: bool,
    buf: String,
}

impl Host for CliHost {
    fn approve(&mut self, call: &ToolCall, outside: bool) -> bool {
        if self.auto {
            return true;
        }
        let warn = if outside { "  ⚠ OUTSIDE workspace" } else { "" };
        eprint!("  allow {} {}{warn}? [y/N] ", call.name, agent::compact(&call.args));
        let _ = std::io::stderr().flush();
        let mut line = String::new();
        if std::io::stdin().lock().read_line(&mut line).is_err() {
            return false;
        }
        matches!(line.trim(), "y" | "Y" | "yes")
    }

    fn event(&mut self, ev: Event) {
        match ev {
            Event::Token(s) => self.buf.push_str(s),
            Event::ToolCall(c) => {
                eprintln!("\n  ⚙ tool: {}({})", c.name, agent::compact(&c.args));
                self.buf.clear(); // the streamed text was the tool call
            }
            Event::ToolResult(r) => {
                let preview: String = r.chars().take(200).collect();
                eprintln!("  → {}{}", preview.replace('\n', " "), if r.len() > 200 { " …" } else { "" });
            }
        }
    }
}

pub fn run(path: &str, query: &str, opts: AgentOpts) -> ExitCode {
    let gguf = match GgufFile::open(path) {
        Ok(f) => f,
        Err(e) => return fail(&format!("open model: {e}")),
    };
    let tok = match Tokenizer::from_gguf(&gguf) {
        Ok(t) => t,
        Err(e) => return fail(&format!("tokenizer: {e}")),
    };
    let model = match Model::load(&gguf) {
        Ok(m) => m,
        Err(e) => return fail(&format!("load model: {e}")),
    };
    eprintln!("OVERFLOW agent — backend {}, workspace {}", model.backend(), opts.workspace.display());

    // CLI runs with full capabilities granted; approval still asked unless --yes.
    let perms = Perms {
        sandbox: false,
        fs_read: true,
        fs_write: true,
        shell: true,
        network: opts.network,
        workspace: opts.workspace.clone(),
        confirm_outside: true,
    };
    let mut host = CliHost { auto: opts.auto_approve, buf: String::new() };
    let mut cache = model.new_cache();
    agent::run(&model, &tok, &mut cache, true, query, &perms, &mut host, opts.max_rounds);

    println!("{}", host.buf.trim());
    ExitCode::SUCCESS
}

fn fail(msg: &str) -> ExitCode {
    eprintln!("error: {msg}");
    ExitCode::FAILURE
}
