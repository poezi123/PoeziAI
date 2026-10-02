//! Reusable tool-calling agent core (Phase D).
//!
//! Drives the model through a tool-use loop using Qwen2's native format, shared
//! by the CLI and the GUI. The frontend supplies a [`Host`]: it decides manual
//! approvals and receives streamed events (assistant tokens, tool calls, tool
//! results). Permissions are enforced here — actions covered by a granted
//! capability and inside the workspace run automatically; everything else asks
//! the host first. Sandbox mode blocks all tools.
//!
//! Sandbox is path-confinement + approval; OS-level isolation is future work.

use std::path::{Path, PathBuf};

use serde_json::Value;

use crate::{KvCache, Model, Tokenizer};

/// What the agent is allowed to do (mirrors the GUI's permission settings).
#[derive(Clone)]
pub struct Perms {
    pub sandbox: bool,
    pub fs_read: bool,
    pub fs_write: bool,
    pub shell: bool,
    pub network: bool,
    pub workspace: PathBuf,
    pub confirm_outside: bool,
}

pub struct ToolCall {
    pub name: String,
    pub args: Value,
}

/// Streamed to the host as the agent works.
pub enum Event<'a> {
    /// A chunk of the assistant's visible text (final answer or its thinking).
    Token(&'a str),
    /// The assistant requested a tool (the preceding tokens were its call).
    ToolCall(&'a ToolCall),
    /// The tool's result that will be fed back to the model.
    ToolResult(&'a str),
}

/// The frontend (CLI/GUI) implements this.
pub trait Host {
    /// Approve a tool the permissions didn't auto-allow. `outside` = the path is
    /// outside the workspace. Return true to run it.
    fn approve(&mut self, call: &ToolCall, outside: bool) -> bool;
    /// Receive a progress event.
    fn event(&mut self, ev: Event);
}

const TOOL_NAMES: [&str; 4] = ["read_file", "write_file", "list_dir", "run_shell"];

const TOOLS_JSON: &str = r#"{"type": "function", "function": {"name": "read_file", "description": "Read a UTF-8 text file and return its contents.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}
{"type": "function", "function": {"name": "write_file", "description": "Create or overwrite a text file.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}}
{"type": "function", "function": {"name": "list_dir", "description": "List the entries of a directory.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}
{"type": "function", "function": {"name": "run_shell", "description": "Run a shell command and return stdout+stderr.", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}"#;

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

/// Run one user query to completion (possibly several tool rounds).
pub fn run(
    model: &Model,
    tok: &Tokenizer,
    cache: &mut KvCache,
    first_turn: bool,
    query: &str,
    perms: &Perms,
    host: &mut dyn Host,
    max_rounds: usize,
) {
    let mut last = Vec::new();
    let feed = |text: &str, cache: &mut KvCache, last: &mut Vec<f32>| {
        for id in tok.encode(text, false, true) {
            *last = model.forward(id, cache);
        }
    };

    let opening = if first_turn {
        format!(
            "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            system_prompt(),
            query
        )
    } else {
        format!("<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n", query)
    };
    feed(&opening, cache, &mut last);

    for _ in 0..max_rounds {
        // Generate the assistant turn, streaming decoded text to the host.
        let mut raw = Vec::new();
        let mut pending = Vec::new();
        loop {
            let next = argmax(&last) as u32;
            if Some(next) == tok.eos_id {
                break;
            }
            tok.push_token_bytes(next, &mut raw);
            tok.push_token_bytes(next, &mut pending);
            if let Ok(s) = std::str::from_utf8(&pending) {
                host.event(Event::Token(s));
                pending.clear();
            } else if let Some(valid) = valid_prefix(&pending) {
                host.event(Event::Token(&valid));
                let n = valid.len();
                pending.drain(..n);
            }
            last = model.forward(next, cache);
        }

        let out = String::from_utf8_lossy(&raw).into_owned();
        let Some(call) = parse_tool_call(&out) else {
            return; // final answer already streamed
        };
        host.event(Event::ToolCall(&call));

        let (auto, outside) = decide(&call, perms);
        let result = if perms.sandbox {
            "Sandbox mode is on: tool use is disabled. Enable permissions in settings.".to_string()
        } else if auto || host.approve(&call, outside) {
            execute(&call, perms)
        } else {
            "Denied by user.".to_string()
        };
        host.event(Event::ToolResult(&result));

        let resp = format!(
            "<|im_end|>\n<|im_start|>user\n<tool_response>\n{}\n</tool_response><|im_end|>\n<|im_start|>assistant\n",
            result
        );
        feed(&resp, cache, &mut last);
    }
}

/// (auto_allowed, path_is_outside_workspace).
fn decide(call: &ToolCall, perms: &Perms) -> (bool, bool) {
    let cap = match call.name.as_str() {
        "read_file" | "list_dir" => perms.fs_read,
        "write_file" => perms.fs_write,
        "run_shell" => perms.shell,
        _ => false,
    };
    let ws = perms.workspace.canonicalize().unwrap_or_else(|_| perms.workspace.clone());
    let outside = match call.name.as_str() {
        "run_shell" => true, // always treat shell as sensitive
        _ => call
            .args
            .get("path")
            .and_then(|p| p.as_str())
            .map(|p| !resolve(&ws, p).starts_with(&ws))
            .unwrap_or(false),
    };
    let auto = cap && !(outside && perms.confirm_outside);
    (auto, outside)
}

fn execute(call: &ToolCall, perms: &Perms) -> String {
    let ws = perms.workspace.canonicalize().unwrap_or_else(|_| perms.workspace.clone());
    let path = call.args.get("path").and_then(|p| p.as_str());
    match call.name.as_str() {
        "read_file" => match path {
            Some(p) => std::fs::read_to_string(resolve(&ws, p))
                .map(|c| c.chars().take(8000).collect())
                .unwrap_or_else(|e| format!("read_file error: {e}")),
            None => "read_file error: missing path".into(),
        },
        "write_file" => match path {
            Some(p) => {
                let content = call.args.get("content").and_then(|c| c.as_str()).unwrap_or("");
                let rp = resolve(&ws, p);
                std::fs::write(&rp, content)
                    .map(|_| format!("wrote {} bytes to {}", content.len(), rp.display()))
                    .unwrap_or_else(|e| format!("write_file error: {e}"))
            }
            None => "write_file error: missing path".into(),
        },
        "list_dir" => {
            let p = resolve(&ws, path.unwrap_or("."));
            match std::fs::read_dir(&p) {
                Ok(rd) => {
                    let mut out = String::new();
                    for e in rd.flatten() {
                        let dir = e.file_type().map(|t| t.is_dir()).unwrap_or(false);
                        out.push_str(&e.file_name().to_string_lossy());
                        out.push_str(if dir { "/\n" } else { "\n" });
                    }
                    out
                }
                Err(e) => format!("list_dir error: {e}"),
            }
        }
        "run_shell" => match call.args.get("command").and_then(|c| c.as_str()) {
            Some(cmd) => match sandboxed_shell(cmd, &ws, perms.network) {
                Ok((o, note)) => {
                    let mut r = String::from_utf8_lossy(&o.stdout).into_owned();
                    r.push_str(&String::from_utf8_lossy(&o.stderr));
                    let body: String = r.chars().take(8000).collect();
                    format!("{note}{body}")
                }
                Err(e) => format!("run_shell error: {e}"),
            },
            None => "run_shell error: missing command".into(),
        },
        other => format!("unknown tool: {other}"),
    }
}

/// Is bubblewrap available for OS-level sandboxing?
fn bwrap_available() -> bool {
    use std::sync::OnceLock;
    static OK: OnceLock<bool> = OnceLock::new();
    *OK.get_or_init(|| {
        std::process::Command::new("bwrap")
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    })
}

/// Run a shell command confined to the workspace via bubblewrap: read-only
/// system, workspace read-write, isolated /tmp and PIDs, no network unless
/// granted. Falls back to an unsandboxed run (with a warning note) if bwrap is
/// missing. Returns (output, note-prefix-for-the-model).
fn sandboxed_shell(cmd: &str, ws: &Path, network: bool) -> std::io::Result<(std::process::Output, String)> {
    use std::process::Command;
    if bwrap_available() {
        let mut c = Command::new("bwrap");
        c.args([
            "--ro-bind", "/usr", "/usr",
            "--symlink", "usr/bin", "/bin",
            "--symlink", "usr/lib", "/lib",
            "--symlink", "usr/lib", "/lib64",
            "--symlink", "usr/bin", "/sbin",
            "--ro-bind", "/etc", "/etc",
            "--proc", "/proc",
            "--dev", "/dev",
            "--tmpfs", "/tmp",
        ]);
        c.arg("--bind").arg(ws).arg(ws);
        c.arg("--chdir").arg(ws);
        c.arg("--unshare-all");
        if network {
            c.arg("--share-net");
        }
        c.arg("--die-with-parent");
        c.arg("sh").arg("-c").arg(cmd);
        let note = format!(
            "[sandboxed: workspace-only, net {}]\n",
            if network { "on" } else { "off" }
        );
        Ok((c.output()?, note))
    } else {
        let o = Command::new("sh").arg("-c").arg(cmd).current_dir(ws).output()?;
        Ok((o, "[UNSANDBOXED: bubblewrap not installed]\n".to_string()))
    }
}

fn resolve(ws: &Path, p: &str) -> PathBuf {
    let pb = PathBuf::from(p);
    let joined = if pb.is_absolute() { pb } else { ws.join(pb) };
    joined.canonicalize().unwrap_or(joined)
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

fn valid_prefix(bytes: &[u8]) -> Option<String> {
    match std::str::from_utf8(bytes) {
        Ok(s) => Some(s.to_string()),
        Err(e) if e.valid_up_to() > 0 => {
            Some(std::str::from_utf8(&bytes[..e.valid_up_to()]).unwrap().to_string())
        }
        _ => None,
    }
}

/// Find a tool call in the output (tags, ```json fence, or bare JSON); only
/// fires for a known tool name.
pub fn parse_tool_call(text: &str) -> Option<ToolCall> {
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
    Some(ToolCall { name, args: v.get("arguments").cloned().unwrap_or(Value::Null) })
}

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

/// Render a value compactly for display.
pub fn compact(v: &Value) -> String {
    let s = v.to_string();
    if s.len() > 140 {
        format!("{}…", &s[..140])
    } else {
        s
    }
}
