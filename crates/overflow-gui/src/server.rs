//! OVERFLOW server mode — a tiny HTTPS server that exposes a mobile-friendly
//! chat web UI, so you can talk to the model from your phone over Tailscale
//! while the PC is on. No web framework: a hand-rolled HTTP/1.1 handler over a
//! rustls TLS stream with a self-signed certificate (so the phone sees a
//! one-time certificate warning, then a normal HTTPS connection).
//!
//! Inference is NOT run here; each request is forwarded to the existing model
//! worker through a shared sender (the `Bridge`), so the one loaded model is
//! reused and requests are serialized with the GUI's own.
//!
//! Security posture (single-user, on your own tailnet):
//!   * Binds the Tailscale IP only — never `0.0.0.0` — so a hostile network the
//!     laptop happens to join (café Wi-Fi) cannot reach it. No tailnet → binds
//!     localhost only (fail safe: no accidental remote exposure).
//!   * A random 144-bit token is minted per run and required on `/api/chat`.
//!     It travels in the URL fragment the GUI shows you (`…/#k=<token>`); the
//!     fragment is never sent to a server by a browser and is not readable
//!     cross-origin, so another tailnet device or a DNS-rebinding page in your
//!     browser cannot drive the model without the link.
//!   * No permissive CORS; the page is same-origin so none is needed.
//!   * Request body is capped and in-flight connections are limited (DoS).

use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{channel, Sender};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use poezi_model::agent::{GenCfg, Perms};
use rand::RngCore;
use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer};
use rustls::ServerConfig;

use crate::hw::HwStats;
use crate::worker::ToWorker;

/// Shared access to the current worker's inbox (None while no model is loaded).
pub type Bridge = Arc<Mutex<Option<Sender<ToWorker>>>>;

/// Settings the phone can see and change: the agent permissions and the
/// generation knobs. Shared with the GUI so a change from either side sticks.
pub struct SrvCfg {
    pub perms: Perms,
    pub gen: GenCfg,
}
pub type SharedCfg = Arc<Mutex<SrvCfg>>;

/// Live hardware snapshot the phone reads to show PC load.
pub type SharedHw = Arc<Mutex<HwStats>>;

/// Largest request body we accept (the chat JSON). Anything bigger is refused
/// before it is buffered, so a bogus Content-Length can't exhaust memory.
const MAX_BODY: usize = 512 * 1024;
/// Largest header block we buffer while waiting for the blank line.
const MAX_HEADERS: usize = 64 * 1024;
/// Most connections handled at once; extra ones are dropped (flood guard).
const MAX_INFLIGHT: usize = 16;

pub struct ServerHandle {
    pub stop: Arc<AtomicBool>,
    pub url: String,
}

/// Start the HTTPS server. Binds the Tailscale IP (fallback: localhost) on port
/// 443, then 8443 if 443 needs privileges. A self-signed cert covers that IP.
pub fn start(bridge: Bridge, cfg: SharedCfg, hw: SharedHw) -> Result<ServerHandle, String> {
    // Install the ring crypto provider once (ignore error if already set).
    let _ = rustls::crypto::ring::default_provider().install_default();
    let ip = tailscale_ip().unwrap_or_else(|| "127.0.0.1".to_string());
    let tls = Arc::new(tls_config(&ip)?);
    let (listener, port) = bind(&ip)?;
    listener.set_nonblocking(true).map_err(|e| e.to_string())?;
    let stop = Arc::new(AtomicBool::new(false));
    let stop2 = stop.clone();
    let token = Arc::new(gen_token());
    // The token rides in the URL fragment: browsers never send a fragment to
    // the server and other origins can't read it, so it stays a shared secret
    // between whoever holds the link and this process.
    let url = format!("https://{ip}:{port}/#k={token}");
    let inflight = Arc::new(AtomicUsize::new(0));
    thread::spawn(move || loop {
        if stop2.load(Ordering::Relaxed) {
            break;
        }
        match listener.accept() {
            Ok((stream, _addr)) => {
                if inflight.load(Ordering::Relaxed) >= MAX_INFLIGHT {
                    continue; // drop: too many connections in flight
                }
                inflight.fetch_add(1, Ordering::Relaxed);
                let bridge = bridge.clone();
                let cfg = cfg.clone();
                let hw = hw.clone();
                let tls = tls.clone();
                let token = token.clone();
                let inflight2 = inflight.clone();
                thread::spawn(move || {
                    let mut tcp = stream;
                    let _ = tcp.set_read_timeout(Some(Duration::from_secs(30)));
                    let _ = tcp.set_write_timeout(Some(Duration::from_secs(30)));
                    if let Ok(mut conn) = rustls::ServerConnection::new(tls) {
                        let mut s = rustls::Stream::new(&mut conn, &mut tcp);
                        let _ = handle_conn(&mut s, &bridge, &cfg, &hw, &token);
                    }
                    inflight2.fetch_sub(1, Ordering::Relaxed);
                });
            }
            Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                thread::sleep(Duration::from_millis(120));
            }
            Err(_) => thread::sleep(Duration::from_millis(200)),
        }
    });
    Ok(ServerHandle { stop, url })
}

/// A fresh 144-bit random token, lowercase hex (36 chars).
fn gen_token() -> String {
    let mut b = [0u8; 18];
    rand::thread_rng().fill_bytes(&mut b);
    b.iter().map(|x| format!("{x:02x}")).collect()
}

/// Constant-time byte compare, so token checking can't be timed.
fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

/// A rustls config with a freshly generated self-signed cert covering the given
/// IP, `localhost` and `overflow` (SANs).
fn tls_config(ip: &str) -> Result<ServerConfig, String> {
    let mut sans = vec!["localhost".to_string(), "overflow".to_string()];
    if ip.contains('.') {
        sans.push(ip.to_string());
    }
    let cert = rcgen::generate_simple_self_signed(sans).map_err(|e| e.to_string())?;
    let cert_der = CertificateDer::from(cert.cert.der().to_vec());
    let key_der = PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(cert.key_pair.serialize_der()));
    ServerConfig::builder()
        .with_no_client_auth()
        .with_single_cert(vec![cert_der], key_der)
        .map_err(|e| e.to_string())
}

/// Bind the given address (the Tailscale IP, or localhost when there is none)
/// on 443, else 8443. Binding a specific address — never `0.0.0.0` — keeps the
/// listener off every other network interface.
fn bind(ip: &str) -> Result<(TcpListener, u16), String> {
    for port in [443u16, 8443] {
        if let Ok(l) = TcpListener::bind((ip, port)) {
            return Ok((l, port));
        }
    }
    Err(format!("Konnte {ip} auf Port 443/8443 nicht binden (belegt/blockiert)"))
}

/// This machine's Tailscale IPv4, via the `tailscale` CLI if present.
fn tailscale_ip() -> Option<String> {
    let out = std::process::Command::new("tailscale").args(["ip", "-4"]).output().ok()?;
    let s = String::from_utf8_lossy(&out.stdout);
    s.lines().map(str::trim).find(|l| l.contains('.') && !l.is_empty()).map(|s| s.to_string())
}

fn handle_conn<S: Read + Write>(
    stream: &mut S,
    bridge: &Bridge,
    cfg: &SharedCfg,
    hw: &SharedHw,
    token: &str,
) -> std::io::Result<()> {
    let mut buf = Vec::new();
    let mut tmp = [0u8; 4096];
    // Read until headers end.
    let header_end = loop {
        let n = stream.read(&mut tmp)?;
        if n == 0 {
            return Ok(());
        }
        buf.extend_from_slice(&tmp[..n]);
        if let Some(p) = find_subslice(&buf, b"\r\n\r\n") {
            break p;
        }
        if buf.len() > MAX_HEADERS {
            return respond(stream, 431, "text/plain", b"headers too large");
        }
    };
    let head = String::from_utf8_lossy(&buf[..header_end]).to_string();
    let mut lines = head.lines();
    let req_line = lines.next().unwrap_or("");
    let mut parts = req_line.split_whitespace();
    let method = parts.next().unwrap_or("");
    let path = parts.next().unwrap_or("/");
    let content_len = head
        .lines()
        .find_map(|l| l.to_ascii_lowercase().strip_prefix("content-length:").map(|v| v.trim().parse::<usize>().ok()))
        .flatten()
        .unwrap_or(0);
    // Refuse an oversized body before buffering a single byte of it.
    if content_len > MAX_BODY {
        return respond(stream, 413, "text/plain", b"payload too large");
    }
    // Bearer token from the Authorization header (the page reads it from the
    // URL fragment and sends it here).
    let bearer = head
        .lines()
        .find_map(|l| l.to_ascii_lowercase().strip_prefix("authorization:").map(|v| v.trim().to_string()))
        .and_then(|v| v.strip_prefix("bearer ").map(|t| t.trim().to_string()));
    // Read the remaining body, never past the cap.
    let mut body = buf[header_end + 4..].to_vec();
    body.truncate(content_len);
    while body.len() < content_len {
        let n = stream.read(&mut tmp)?;
        if n == 0 {
            break;
        }
        body.extend_from_slice(&tmp[..n]);
        if body.len() > MAX_BODY {
            return respond(stream, 413, "text/plain", b"payload too large");
        }
    }

    // Every API route is token-gated (constant-time). The page itself (GET /)
    // is public; it carries no secret and can't do anything without the token.
    let authed = bearer.as_deref().map(|t| ct_eq(t.as_bytes(), token.as_bytes())).unwrap_or(false);
    let deny = |stream: &mut S| -> std::io::Result<()> {
        let msg = serde_json::json!({
            "error": "Not allowed — open the address shown in OVERFLOW (the one with #k=...)."
        })
        .to_string();
        respond(stream, 401, "application/json", msg.as_bytes())
    };

    match (method, path) {
        ("GET", "/") | ("GET", "/index.html") => respond(stream, 200, "text/html; charset=utf-8", PAGE.as_bytes()),
        ("POST", "/api/chat") => {
            if !authed {
                return deny(stream);
            }
            let reply = run_chat(&body, bridge, cfg);
            let json = serde_json::json!({ "reply": reply }).to_string();
            respond(stream, 200, "application/json", json.as_bytes())
        }
        ("GET", "/api/stats") => {
            if !authed {
                return deny(stream);
            }
            respond(stream, 200, "application/json", stats_json(hw).as_bytes())
        }
        ("GET", "/api/settings") => {
            if !authed {
                return deny(stream);
            }
            respond(stream, 200, "application/json", settings_json(cfg).as_bytes())
        }
        ("POST", "/api/settings") => {
            if !authed {
                return deny(stream);
            }
            apply_settings(&body, cfg);
            respond(stream, 200, "application/json", settings_json(cfg).as_bytes())
        }
        _ => respond(stream, 404, "text/plain", b"not found"),
    }
}

/// Current permissions + generation settings as JSON (for the phone's panel).
fn settings_json(cfg: &SharedCfg) -> String {
    let g = cfg.lock().unwrap();
    serde_json::json!({
        "perms": {
            "sandbox": g.perms.sandbox,
            "fs_read": g.perms.fs_read,
            "fs_write": g.perms.fs_write,
            "shell": g.perms.shell,
            "network": g.perms.network,
            "confirm_outside": g.perms.confirm_outside,
            "workspace": g.perms.workspace.display().to_string(),
        },
        "gen": {
            "thinking": g.gen.thinking,
            "effort": g.gen.effort,
            "temp": g.gen.temp,
            "top_p": g.gen.top_p,
            "top_k": g.gen.top_k,
            "max_tokens": g.gen.max_tokens,
            "repeat_penalty": g.gen.repeat_penalty,
            "repeat_last_n": g.gen.repeat_last_n,
        }
    })
    .to_string()
}

/// Apply a settings update from the phone (any missing field is left as-is).
fn apply_settings(body: &[u8], cfg: &SharedCfg) {
    let Ok(v) = serde_json::from_slice::<serde_json::Value>(body) else { return };
    let mut g = cfg.lock().unwrap();
    if let Some(p) = v.get("perms") {
        let b = |k: &str, cur: bool| p.get(k).and_then(|x| x.as_bool()).unwrap_or(cur);
        g.perms.sandbox = b("sandbox", g.perms.sandbox);
        g.perms.fs_read = b("fs_read", g.perms.fs_read);
        g.perms.fs_write = b("fs_write", g.perms.fs_write);
        g.perms.shell = b("shell", g.perms.shell);
        g.perms.network = b("network", g.perms.network);
        g.perms.confirm_outside = b("confirm_outside", g.perms.confirm_outside);
        if let Some(w) = p.get("workspace").and_then(|x| x.as_str()) {
            if !w.trim().is_empty() {
                g.perms.workspace = std::path::PathBuf::from(w);
            }
        }
    }
    if let Some(gn) = v.get("gen") {
        if let Some(x) = gn.get("thinking").and_then(|x| x.as_bool()) { g.gen.thinking = x; }
        if let Some(x) = gn.get("effort").and_then(|x| x.as_u64()) { g.gen.effort = x as u8; }
        if let Some(x) = gn.get("temp").and_then(|x| x.as_f64()) { g.gen.temp = x as f32; }
        if let Some(x) = gn.get("top_p").and_then(|x| x.as_f64()) { g.gen.top_p = x as f32; }
        if let Some(x) = gn.get("top_k").and_then(|x| x.as_u64()) { g.gen.top_k = x as usize; }
        if let Some(x) = gn.get("max_tokens").and_then(|x| x.as_u64()) { g.gen.max_tokens = x as usize; }
        if let Some(x) = gn.get("repeat_penalty").and_then(|x| x.as_f64()) { g.gen.repeat_penalty = x as f32; }
        if let Some(x) = gn.get("repeat_last_n").and_then(|x| x.as_u64()) { g.gen.repeat_last_n = x as usize; }
    }
}

/// Live PC load as JSON.
fn stats_json(hw: &SharedHw) -> String {
    let h = hw.lock().map(|g| *g).unwrap_or_default();
    serde_json::json!({
        "cpu": h.cpu_pct,
        "mem_used": h.mem_used_gb,
        "mem_total": h.mem_total_gb,
        "gpu": h.gpu_pct,
        "vram_used": h.vram_used_mb,
        "vram_total": h.vram_total_mb,
    })
    .to_string()
}

/// Parse {messages:[{role,content}]}, run one turn through the worker, return text.
fn run_chat(body: &[u8], bridge: &Bridge, cfg: &SharedCfg) -> String {
    let v: serde_json::Value = match serde_json::from_slice(body) {
        Ok(v) => v,
        Err(_) => return "Error: bad request".to_string(),
    };
    let mut turns: Vec<(bool, String)> = Vec::new();
    if let Some(arr) = v.get("messages").and_then(|m| m.as_array()) {
        for m in arr {
            let role = m.get("role").and_then(|r| r.as_str()).unwrap_or("user");
            let content = m.get("content").and_then(|c| c.as_str()).unwrap_or("").to_string();
            if content.trim().is_empty() {
                continue;
            }
            turns.push((role != "assistant", content));
        }
    }
    if turns.is_empty() {
        return String::new();
    }
    let tx = { bridge.lock().ok().and_then(|g| g.clone()) };
    let Some(tx) = tx else {
        return "No model loaded — load one in OVERFLOW.".to_string();
    };
    let (perms, gen) = {
        let g = cfg.lock().unwrap();
        (g.perms.clone(), g.gen.clone())
    };
    let (rtx, rrx) = channel::<String>();
    if tx.send(ToWorker::ServerChat { turns, perms, gen, reply: rtx }).is_err() {
        return "Can't reach the model.".to_string();
    }
    match rrx.recv_timeout(Duration::from_secs(600)) {
        Ok(s) => strip_think(&s),
        Err(_) => "Timed out.".to_string(),
    }
}

fn strip_think(s: &str) -> String {
    let mut out = String::new();
    let mut rest = s;
    while let Some(a) = rest.find("<think>") {
        out.push_str(&rest[..a]);
        if let Some(b) = rest[a..].find("</think>") {
            rest = &rest[a + b + "</think>".len()..];
        } else {
            rest = "";
            break;
        }
    }
    out.push_str(rest);
    out.trim().to_string()
}

fn respond<S: Write>(stream: &mut S, code: u16, ctype: &str, body: &[u8]) -> std::io::Result<()> {
    let status = match code {
        200 => "200 OK",
        204 => "204 No Content",
        401 => "401 Unauthorized",
        404 => "404 Not Found",
        413 => "413 Payload Too Large",
        431 => "431 Request Header Fields Too Large",
        _ => "500 Internal Server Error",
    };
    // No CORS header on purpose: the page is served from this same origin, so
    // it needs none, and withholding it keeps other origins from reading
    // responses in a victim's browser.
    let header = format!(
        "HTTP/1.1 {status}\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\nX-Content-Type-Options: nosniff\r\nReferrer-Policy: no-referrer\r\nConnection: close\r\n\r\n",
        body.len()
    );
    stream.write_all(header.as_bytes())?;
    stream.write_all(body)?;
    stream.flush()
}

fn find_subslice(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

/// The mobile chat page served at `/`. It reads the token from its own URL
/// fragment (`#k=...`) and sends it as a Bearer header; the fragment itself is
/// never transmitted to the server by the browser.
const PAGE: &str = r#"<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>OVERFLOW</title>
<style>
  :root{--bg:#070b0d;--panel:#0d1316;--fg:#d7e0dd;--muted:#6f8481;--amber:#f5b94d;--green:#4df0a0;--line:#1c2a2b}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--fg);font-family:system-ui,sans-serif;display:flex;flex-direction:column;height:100dvh}
  header{padding:12px 16px;border-bottom:1px solid var(--line);color:var(--amber);font-weight:700;letter-spacing:1px;display:flex;justify-content:space-between;align-items:center}
  #gear{background:none;border:0;color:var(--amber);font-size:22px;padding:0 4px;cursor:pointer}
  #log{flex:1;overflow-y:auto;padding:16px;display:flex;flex-direction:column;gap:12px}
  .msg{max-width:85%;padding:10px 12px;border-radius:12px;white-space:pre-wrap;line-height:1.4}
  .you{align-self:flex-end;background:#15303a;border:1px solid var(--line)}
  .ai{align-self:flex-start;background:var(--panel);border:1px solid var(--line)}
  .who{font-size:11px;color:var(--muted);margin-bottom:3px}
  form{display:flex;gap:8px;padding:12px;border-top:1px solid var(--line)}
  #t{flex:1;padding:12px;border-radius:10px;border:1px solid var(--line);background:var(--panel);color:var(--fg);font-size:16px}
  button{padding:12px 16px;border-radius:10px;border:0;background:var(--amber);color:#0b0b0b;font-weight:700}
  button:disabled{opacity:.5}
  #panel{position:fixed;inset:0;background:rgba(0,0,0,.6);display:none;z-index:9}
  #panel.open{display:block}
  #sheet{position:absolute;right:0;top:0;bottom:0;width:min(420px,92vw);background:var(--bg);border-left:1px solid var(--line);padding:16px;overflow-y:auto}
  #sheet h2{color:var(--amber);font-size:14px;margin:18px 0 8px;letter-spacing:.5px}
  #sheet h2:first-child{margin-top:0}
  .row{display:flex;justify-content:space-between;align-items:center;gap:10px;padding:6px 0;font-size:14px}
  .row label{color:var(--fg)}
  .hint{color:var(--muted);font-size:12px;margin:2px 0 6px}
  .bar{height:7px;border-radius:4px;background:var(--panel);overflow:hidden;flex:1;margin-left:10px}
  .bar>span{display:block;height:100%;background:var(--green)}
  .row input[type=range]{flex:1}
  .row .val{min-width:46px;text-align:right;color:var(--amber);font-variant-numeric:tabular-nums}
  #sheet input[type=text]{width:100%;padding:8px;border-radius:8px;border:1px solid var(--line);background:var(--panel);color:var(--fg);font-size:14px}
  #sheet select{background:var(--panel);color:var(--fg);border:1px solid var(--line);border-radius:8px;padding:5px}
  #save{width:100%;margin-top:16px}
  .close{background:none;border:1px solid var(--line);color:var(--muted);float:right;padding:6px 10px;font-weight:400}
</style></head><body>
<header><span>OVERFLOW · remote</span><button id="gear" title="Settings">&#9881;</button></header>
<div id="log"></div>
<form id="f"><input id="t" placeholder="Message…" autocomplete="off"><button id="b">Send</button></form>

<div id="panel"><div id="sheet">
  <button class="close" id="close">close</button>
  <h2>PC load</h2>
  <div class="row"><label>CPU</label><div class="bar"><span id="bcpu"></span></div><div class="val" id="vcpu">–</div></div>
  <div class="row"><label>RAM</label><div class="bar"><span id="bram"></span></div><div class="val" id="vram">–</div></div>
  <div class="row"><label>GPU</label><div class="bar"><span id="bgpu"></span></div><div class="val" id="vgpu">–</div></div>
  <div class="row"><label>VRAM</label><div class="bar"><span id="bvr"></span></div><div class="val" id="vvr">–</div></div>

  <h2>Permissions</h2>
  <div class="hint">Tools only work when safe mode is OFF. These let the AI act on this PC &mdash; be careful.</div>
  <div class="row"><label>Safe mode (no tools)</label><input type="checkbox" id="p_sandbox"></div>
  <div class="row"><label>Read files</label><input type="checkbox" id="p_fs_read"></div>
  <div class="row"><label>Write files</label><input type="checkbox" id="p_fs_write"></div>
  <div class="row"><label>Run commands</label><input type="checkbox" id="p_shell"></div>
  <div class="row"><label>Internet</label><input type="checkbox" id="p_network"></div>
  <div class="row"><label>Confirm outside workspace</label><input type="checkbox" id="p_confirm"></div>
  <div class="hint">Working folder:</div>
  <input type="text" id="p_ws" placeholder="/home/you/project">

  <h2>More settings</h2>
  <div class="row"><label>Thinking mode</label><input type="checkbox" id="g_think"></div>
  <div class="row"><label>How hard it thinks</label><select id="g_effort"><option value="0">low</option><option value="1">medium</option><option value="2">high</option></select></div>
  <div class="row"><label>Randomness</label><input type="range" id="g_temp" min="0" max="1.5" step="0.05"><div class="val" id="g_temp_v"></div></div>
  <div class="row"><label>top-p</label><input type="range" id="g_topp" min="0.1" max="1" step="0.05"><div class="val" id="g_topp_v"></div></div>
  <div class="row"><label>top-k</label><input type="range" id="g_topk" min="1" max="200" step="1"><div class="val" id="g_topk_v"></div></div>
  <div class="row"><label>Longest answer</label><input type="range" id="g_max" min="32" max="8192" step="32"><div class="val" id="g_max_v"></div></div>
  <div class="row"><label>Stop repeating</label><input type="range" id="g_rep" min="1" max="2" step="0.05"><div class="val" id="g_rep_v"></div></div>
  <div class="row"><label>How far back</label><input type="range" id="g_rln" min="16" max="1024" step="16"><div class="val" id="g_rln_v"></div></div>

  <button id="save">Save settings</button>
</div></div>

<script>
const log=document.getElementById('log'),f=document.getElementById('f'),t=document.getElementById('t'),b=document.getElementById('b');
const msgs=[];
const TOKEN=(new URLSearchParams(location.hash.slice(1)).get('k'))||'';
const H={'content-type':'application/json','authorization':'Bearer '+TOKEN};
const $=id=>document.getElementById(id);
function add(role,text){const d=document.createElement('div');d.className='msg '+(role==='user'?'you':'ai');d.innerHTML='<div class="who">'+(role==='user'?'you':'overflow')+'</div>';d.appendChild(document.createTextNode(text));log.appendChild(d);log.scrollTop=log.scrollHeight;return d;}
if(!TOKEN){add('assistant','No access key in the link. Open the address that OVERFLOW shows in the Server tab (…/#k=…).');b.disabled=true;t.disabled=true;$('gear').disabled=true;}
f.onsubmit=async e=>{e.preventDefault();const text=t.value.trim();if(!text)return;t.value='';add('user',text);msgs.push({role:'user',content:text});b.disabled=true;const d=add('assistant','…');try{const r=await fetch('/api/chat',{method:'POST',headers:H,body:JSON.stringify({messages:msgs})});const j=await r.json();d.removeChild(d.lastChild);d.appendChild(document.createTextNode(j.reply||'(empty)'));msgs.push({role:'assistant',content:j.reply||''});}catch(err){d.appendChild(document.createTextNode(' Error: '+err));}b.disabled=false;log.scrollTop=log.scrollHeight;};

// ---- settings panel ----
const panel=$('panel');let statsTimer=null;
function setBar(bar,val,txt){const v=Math.max(0,Math.min(100,val||0));$(bar).style.width=v+'%';}
async function pollStats(){try{const r=await fetch('/api/stats',{headers:H});const s=await r.json();
  $('bcpu').style.width=(s.cpu||0)+'%';$('vcpu').textContent=Math.round(s.cpu||0)+'%';
  const mp=s.mem_total? (s.mem_used/s.mem_total*100):0;$('bram').style.width=mp+'%';$('vram').textContent=(s.mem_used||0).toFixed(1)+'/'+(s.mem_total||0).toFixed(1)+' GB';
  if(s.gpu==null){$('vgpu').textContent='n/a';}else{$('bgpu').style.width=s.gpu+'%';$('vgpu').textContent=Math.round(s.gpu)+'%';}
  if(s.vram_total==null){$('vvr').textContent='n/a';}else{$('bvr').style.width=(s.vram_used/s.vram_total*100)+'%';$('vvr').textContent=Math.round(s.vram_used)+'/'+Math.round(s.vram_total)+' MB';}
}catch(e){}}
function bindRange(id){const el=$(id),out=$(id+'_v');const upd=()=>out.textContent=el.value;el.oninput=upd;return upd;}
const ranges=['g_temp','g_topp','g_topk','g_max','g_rep','g_rln'].map(bindRange);
async function loadSettings(){try{const r=await fetch('/api/settings',{headers:H});const j=await r.json();const p=j.perms,g=j.gen;
  $('p_sandbox').checked=p.sandbox;$('p_fs_read').checked=p.fs_read;$('p_fs_write').checked=p.fs_write;$('p_shell').checked=p.shell;$('p_network').checked=p.network;$('p_confirm').checked=p.confirm_outside;$('p_ws').value=p.workspace||'';
  $('g_think').checked=g.thinking;$('g_effort').value=g.effort;$('g_temp').value=g.temp;$('g_topp').value=g.top_p;$('g_topk').value=g.top_k;$('g_max').value=g.max_tokens;$('g_rep').value=g.repeat_penalty;$('g_rln').value=g.repeat_last_n;
  ranges.forEach(u=>u());
}catch(e){}}
$('gear').onclick=()=>{panel.classList.add('open');loadSettings();pollStats();statsTimer=setInterval(pollStats,2000);};
function closePanel(){panel.classList.remove('open');if(statsTimer){clearInterval(statsTimer);statsTimer=null;}}
$('close').onclick=closePanel;
panel.onclick=e=>{if(e.target===panel)closePanel();};
$('save').onclick=async()=>{const body={perms:{sandbox:$('p_sandbox').checked,fs_read:$('p_fs_read').checked,fs_write:$('p_fs_write').checked,shell:$('p_shell').checked,network:$('p_network').checked,confirm_outside:$('p_confirm').checked,workspace:$('p_ws').value},gen:{thinking:$('g_think').checked,effort:+$('g_effort').value,temp:+$('g_temp').value,top_p:+$('g_topp').value,top_k:+$('g_topk').value,max_tokens:+$('g_max').value,repeat_penalty:+$('g_rep').value,repeat_last_n:+$('g_rln').value}};
  $('save').textContent='Saving…';try{await fetch('/api/settings',{method:'POST',headers:H,body:JSON.stringify(body)});$('save').textContent='Saved ✓';setTimeout(()=>$('save').textContent='Save settings',1200);}catch(e){$('save').textContent='Error';}};
</script></body></html>"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_is_unique_and_sized() {
        let a = gen_token();
        let b = gen_token();
        assert_eq!(a.len(), 36);
        assert!(a.chars().all(|c| c.is_ascii_hexdigit()));
        assert_ne!(a, b);
    }

    #[test]
    fn ct_eq_matches_only_equal() {
        assert!(ct_eq(b"abc", b"abc"));
        assert!(!ct_eq(b"abc", b"abd"));
        assert!(!ct_eq(b"abc", b"ab"));
        assert!(!ct_eq(b"", b"x"));
    }
}
