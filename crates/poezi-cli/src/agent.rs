//! Security agent (Phase D): the model calls real tools.
//!
//! Uses Qwen2.5's native tool-calling format — tools are declared in the system
//! prompt inside `<tools>…</tools>`, the model emits
//! `<tool_call>{"name":…,"arguments":…}</tool_call>`, we run the tool and feed
//! back a `<tool_response>`. Everything runs through the permission model:
//! actions inside the workspace folder are allowed; anything outside (or a shell
//! command) asks first — Claude-Code style.
//!
//! Sandbox note: this enforces workspace path-confinement + interactive
//! approval. OS-level isolation (bubblewrap/container) is future work.

use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use poezi_gguf::GgufFile;
use poezi_model::{KvCache, Model, Tokenizer};
use serde_json::Value;

pub struct AgentOpts {
    pub workspace: PathBuf,
    pub auto_approve: bool,
    pub max_rounds: usize,
}

const TOOLS_JSON: &str = r#"{"type": "function", "function": {"name": "read_file", "description": "Read a UTF-8 text file and return its contents.", "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "File path"}}, "required": ["path"]}}}
{"type": "function", "function": {"name": "write_file", "description": "Create or overwrite a text file.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}}
{"type": "function", "function": {"name": "list_dir", "description": "List the entries of a directory.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}
{"type": "function", "function": {"name": "run_shell", "description": "Run a shell command and return its stdout+stderr.", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}"#;

fn system_prompt() -> String {
    format!(
        "You are OVERFLOW, S1BERIA's local cybersecurity AI. You run entirely on the \
         user's machine. Be direct and technical. Use tools to inspect and act on the \
         system; when a tool result answers the question, reply in plain text.\n\n\
         # Tools\n\nYou may call one or more functions to assist with the user query.\n\n\
         You are provided with function signatures within <tools></tools> XML tags:\n\
         <tools>\n{TOOLS_JSON}\n</tools>\n\n\
         For each function call, return a json object with function name and arguments \
         within <tool_call></tool_call> XML tags:\n\
         <tool_call>\n{{\"name\": <function-name>, \"arguments\": <args-json-object>}}\n</tool_call>"
    )
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

    let mut cache = model.new_cache();
    let mut last = Vec::new();
    let feed = |text: &str, cache: &mut KvCache, last: &mut Vec<f32>| {
        for id in tok.encode(text, false, true) {
            *last = model.forward(id, cache);
        }
    };

    // First turn: system (with tools) + user.
    let opening = format!(
        "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
        system_prompt(),
        query
    );
    feed(&opening, &mut cache, &mut last);

    for _ in 0..opts.max_rounds {
        let out = generate(&model, &tok, &mut cache, &mut last);
        match parse_tool_call(&out) {
            None => {
                // Final answer.
                println!("{}", out.trim());
                return ExitCode::SUCCESS;
            }
            Some((name, args)) => {
                eprintln!("\n  ⚙ tool: {name}({})", compact(&args));
                let result = match approve_and_run(&name, &args, &opts) {
                    Ok(r) => r,
                    Err(denied) => denied,
                };
                let preview: String = result.chars().take(200).collect();
                eprintln!("  → {}{}", preview.replace('\n', " "), if result.len() > 200 { " …" } else { "" });

                let resp = format!(
                    "<|im_end|>\n<|im_start|>user\n<tool_response>\n{}\n</tool_response><|im_end|>\n<|im_start|>assistant\n",
                    result
                );
                feed(&resp, &mut cache, &mut last);
            }
        }
    }
    eprintln!("(stopped: max tool rounds reached)");
    ExitCode::SUCCESS
}

/// Greedy-generate the assistant turn until EOS; return decoded text.
fn generate(model: &Model, tok: &Tokenizer, cache: &mut KvCache, last: &mut Vec<f32>) -> String {
    let mut bytes = Vec::new();
    for _ in 0..1024 {
        let next = argmax(last) as u32;
        if Some(next) == tok.eos_id {
            break;
        }
        tok.push_token_bytes(next, &mut bytes);
        *last = model.forward(next, cache);
    }
    String::from_utf8_lossy(&bytes).into_owned()
}

fn argmax(v: &[f32]) -> usize {
    let mut best = 0;
    let mut bv = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > bv {
            bv = x;
            best = i;
        }
    }
    best
}

const TOOL_NAMES: [&str; 4] = ["read_file", "write_file", "list_dir", "run_shell"];

/// Find a tool call in the model's output. Accepts the official
/// `<tool_call>…</tool_call>` form, a markdown ```json fence, or a bare JSON
/// object — whatever the model emits — and only treats it as a call if the name
/// is one of our tools.
fn parse_tool_call(text: &str) -> Option<(String, Value)> {
    // Prefer the region inside explicit tags, else scan the whole text.
    let region = match (text.find("<tool_call>"), text.find("</tool_call>")) {
        (Some(s), Some(e)) if e > s => &text[s + "<tool_call>".len()..e],
        _ => text,
    };
    let json = first_json_object(region)?;
    let v: Value = serde_json::from_str(&json).ok()?;
    let name = v.get("name")?.as_str()?.to_string();
    if !TOOL_NAMES.contains(&name.as_str()) {
        return None;
    }
    let args = v.get("arguments").cloned().unwrap_or(Value::Null);
    Some((name, args))
}

/// First balanced `{…}` substring (brace-matching, string-aware).
fn first_json_object(s: &str) -> Option<String> {
    let start = s.find('{')?;
    let mut depth = 0i32;
    let mut in_str = false;
    let mut esc = false;
    for (i, c) in s[start..].char_indices() {
        if in_str {
            if esc {
                esc = false;
            } else if c == '\\' {
                esc = true;
            } else if c == '"' {
                in_str = false;
            }
            continue;
        }
        match c {
            '"' => in_str = true,
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(s[start..start + i + 1].to_string());
                }
            }
            _ => {}
        }
    }
    None
}

fn compact(v: &Value) -> String {
    let s = v.to_string();
    if s.len() > 120 {
        format!("{}…", &s[..120])
    } else {
        s
    }
}

/// Check permissions (ask if needed) and run the tool. Err holds a denial text
/// to feed back to the model.
fn approve_and_run(name: &str, args: &Value, opts: &AgentOpts) -> Result<String, String> {
    let ws = opts.workspace.canonicalize().unwrap_or_else(|_| opts.workspace.clone());

    let path_arg = args.get("path").and_then(|p| p.as_str());
    let outside = path_arg
        .map(|p| {
            let abs = resolve(&ws, p);
            !abs.starts_with(&ws)
        })
        .unwrap_or(false);

    // Shell and anything outside the workspace always need approval.
    let needs_ask = name == "run_shell" || outside || matches!(name, "write_file");
    if needs_ask && !opts.auto_approve && !ask(name, args, outside) {
        return Err("Denied by user.".to_string());
    }

    match name {
        "read_file" => {
            let p = resolve(&ws, path_arg.ok_or("missing path")?);
            std::fs::read_to_string(&p)
                .map(|c| c.chars().take(8000).collect())
                .map_err(|e| format!("read_file error: {e}"))
        }
        "write_file" => {
            let p = resolve(&ws, path_arg.ok_or("missing path")?);
            let content = args.get("content").and_then(|c| c.as_str()).unwrap_or("");
            std::fs::write(&p, content)
                .map(|_| format!("wrote {} bytes to {}", content.len(), p.display()))
                .map_err(|e| format!("write_file error: {e}"))
        }
        "list_dir" => {
            let p = resolve(&ws, path_arg.unwrap_or("."));
            let mut out = String::new();
            let rd = std::fs::read_dir(&p).map_err(|e| format!("list_dir error: {e}"))?;
            for entry in rd.flatten() {
                let ft = entry.file_type().ok();
                let slash = if ft.map(|t| t.is_dir()).unwrap_or(false) { "/" } else { "" };
                out.push_str(&format!("{}{}\n", entry.file_name().to_string_lossy(), slash));
            }
            Ok(out)
        }
        "run_shell" => {
            let cmd = args.get("command").and_then(|c| c.as_str()).ok_or("missing command")?;
            let o = std::process::Command::new("sh")
                .arg("-c")
                .arg(cmd)
                .current_dir(&ws)
                .output()
                .map_err(|e| format!("run_shell error: {e}"))?;
            let mut r = String::from_utf8_lossy(&o.stdout).into_owned();
            r.push_str(&String::from_utf8_lossy(&o.stderr));
            Ok(r.chars().take(8000).collect())
        }
        other => Err(format!("unknown tool: {other}")),
    }
}

/// Resolve a possibly-relative path against the workspace.
fn resolve(ws: &Path, p: &str) -> PathBuf {
    let pb = PathBuf::from(p);
    let joined = if pb.is_absolute() { pb } else { ws.join(pb) };
    joined.canonicalize().unwrap_or(joined)
}

/// Interactive approval prompt (stderr/stdin).
fn ask(name: &str, args: &Value, outside: bool) -> bool {
    let warn = if outside { "  ⚠ OUTSIDE workspace" } else { "" };
    eprint!("  allow {name} {}{warn}? [y/N] ", compact(args));
    let _ = std::io::stderr().flush();
    let mut line = String::new();
    if std::io::stdin().lock().read_line(&mut line).is_err() {
        return false;
    }
    matches!(line.trim(), "y" | "Y" | "yes")
}

fn fail(msg: &str) -> ExitCode {
    eprintln!("error: {msg}");
    ExitCode::FAILURE
}
