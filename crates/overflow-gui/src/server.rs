//! OVERFLOW server mode — a tiny HTTPS server that exposes a mobile-friendly
//! chat web UI, so you can talk to the model from your phone over Tailscale
//! while the PC is on. No web framework: a hand-rolled HTTP/1.1 handler over a
//! rustls TLS stream with a self-signed certificate (so the phone sees a
//! one-time certificate warning, then a normal HTTPS connection).
//!
//! Inference is NOT run here; each request is forwarded to the existing model
//! worker through a shared sender (the `Bridge`), so the one loaded model is
//! reused and requests are serialized with the GUI's own.

use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Sender};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use poezi_model::agent::GenCfg;
use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer};
use rustls::ServerConfig;

use crate::worker::ToWorker;

/// Shared access to the current worker's inbox (None while no model is loaded).
pub type Bridge = Arc<Mutex<Option<Sender<ToWorker>>>>;

pub struct ServerHandle {
    pub stop: Arc<AtomicBool>,
    pub url: String,
}

/// Start the HTTPS server. Tries port 443, falls back to 8443 (443 needs
/// privileges). A self-signed cert is generated for the Tailscale IP.
pub fn start(bridge: Bridge, gen: GenCfg) -> Result<ServerHandle, String> {
    // Install the ring crypto provider once (ignore error if already set).
    let _ = rustls::crypto::ring::default_provider().install_default();
    let ip = tailscale_ip().unwrap_or_else(|| "127.0.0.1".to_string());
    let tls = Arc::new(tls_config(&ip)?);
    let (listener, port) = bind()?;
    listener.set_nonblocking(true).map_err(|e| e.to_string())?;
    let stop = Arc::new(AtomicBool::new(false));
    let stop2 = stop.clone();
    let url = format!("https://{ip}:{port}");
    thread::spawn(move || loop {
        if stop2.load(Ordering::Relaxed) {
            break;
        }
        match listener.accept() {
            Ok((stream, _addr)) => {
                let bridge = bridge.clone();
                let gen = gen.clone();
                let tls = tls.clone();
                thread::spawn(move || {
                    let mut tcp = stream;
                    let _ = tcp.set_read_timeout(Some(Duration::from_secs(30)));
                    if let Ok(mut conn) = rustls::ServerConnection::new(tls) {
                        let mut s = rustls::Stream::new(&mut conn, &mut tcp);
                        let _ = handle_conn(&mut s, &bridge, &gen);
                    }
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

fn bind() -> Result<(TcpListener, u16), String> {
    for port in [443u16, 8443] {
        if let Ok(l) = TcpListener::bind(("0.0.0.0", port)) {
            return Ok((l, port));
        }
    }
    Err("Port 443 und 8443 belegt/blockiert".to_string())
}

/// This machine's Tailscale IPv4, via the `tailscale` CLI if present.
fn tailscale_ip() -> Option<String> {
    let out = std::process::Command::new("tailscale").args(["ip", "-4"]).output().ok()?;
    let s = String::from_utf8_lossy(&out.stdout);
    s.lines().map(str::trim).find(|l| l.contains('.') && !l.is_empty()).map(|s| s.to_string())
}

fn handle_conn<S: Read + Write>(stream: &mut S, bridge: &Bridge, gen: &GenCfg) -> std::io::Result<()> {
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
        if buf.len() > 1 << 20 {
            return Ok(());
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
    // Read the remaining body.
    let mut body = buf[header_end + 4..].to_vec();
    while body.len() < content_len {
        let n = stream.read(&mut tmp)?;
        if n == 0 {
            break;
        }
        body.extend_from_slice(&tmp[..n]);
    }

    match (method, path) {
        ("GET", "/") | ("GET", "/index.html") => respond(stream, 200, "text/html; charset=utf-8", PAGE.as_bytes()),
        ("OPTIONS", _) => respond(stream, 204, "text/plain", b""),
        ("POST", "/api/chat") => {
            let reply = run_chat(&body, bridge, gen);
            let json = serde_json::json!({ "reply": reply }).to_string();
            respond(stream, 200, "application/json", json.as_bytes())
        }
        _ => respond(stream, 404, "text/plain", b"not found"),
    }
}

/// Parse {messages:[{role,content}]}, run one turn through the worker, return text.
fn run_chat(body: &[u8], bridge: &Bridge, gen: &GenCfg) -> String {
    let v: serde_json::Value = match serde_json::from_slice(body) {
        Ok(v) => v,
        Err(_) => return "Fehler: ungültige Anfrage".to_string(),
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
        return "Kein Modell geladen — lade in OVERFLOW ein Modell.".to_string();
    };
    let (rtx, rrx) = channel::<String>();
    if tx.send(ToWorker::ServerChat { turns, gen: gen.clone(), reply: rtx }).is_err() {
        return "Modell nicht erreichbar.".to_string();
    }
    match rrx.recv_timeout(Duration::from_secs(600)) {
        Ok(s) => strip_think(&s),
        Err(_) => "Zeitüberschreitung.".to_string(),
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
        404 => "404 Not Found",
        _ => "500 Internal Server Error",
    };
    let header = format!(
        "HTTP/1.1 {status}\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Headers: content-type\r\nConnection: close\r\n\r\n",
        body.len()
    );
    stream.write_all(header.as_bytes())?;
    stream.write_all(body)?;
    stream.flush()
}

fn find_subslice(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

/// The mobile chat page served at `/`.
const PAGE: &str = r#"<!doctype html>
<html lang="de"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>OVERFLOW</title>
<style>
  :root{--bg:#070b0d;--panel:#0d1316;--fg:#d7e0dd;--muted:#6f8481;--amber:#f5b94d;--green:#4df0a0;--line:#1c2a2b}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--fg);font-family:system-ui,sans-serif;display:flex;flex-direction:column;height:100dvh}
  header{padding:12px 16px;border-bottom:1px solid var(--line);color:var(--amber);font-weight:700;letter-spacing:1px}
  #log{flex:1;overflow-y:auto;padding:16px;display:flex;flex-direction:column;gap:12px}
  .msg{max-width:85%;padding:10px 12px;border-radius:12px;white-space:pre-wrap;line-height:1.4}
  .you{align-self:flex-end;background:#15303a;border:1px solid var(--line)}
  .ai{align-self:flex-start;background:var(--panel);border:1px solid var(--line)}
  .who{font-size:11px;color:var(--muted);margin-bottom:3px}
  form{display:flex;gap:8px;padding:12px;border-top:1px solid var(--line)}
  input{flex:1;padding:12px;border-radius:10px;border:1px solid var(--line);background:var(--panel);color:var(--fg);font-size:16px}
  button{padding:12px 16px;border-radius:10px;border:0;background:var(--amber);color:#0b0b0b;font-weight:700}
  button:disabled{opacity:.5}
</style></head><body>
<header>OVERFLOW · remote</header>
<div id="log"></div>
<form id="f"><input id="t" placeholder="Nachricht…" autocomplete="off"><button id="b">Senden</button></form>
<script>
const log=document.getElementById('log'),f=document.getElementById('f'),t=document.getElementById('t'),b=document.getElementById('b');
const msgs=[];
function add(role,text){const d=document.createElement('div');d.className='msg '+(role==='user'?'you':'ai');d.innerHTML='<div class="who">'+(role==='user'?'du':'overflow')+'</div>';d.appendChild(document.createTextNode(text));log.appendChild(d);log.scrollTop=log.scrollHeight;return d;}
f.onsubmit=async e=>{e.preventDefault();const text=t.value.trim();if(!text)return;t.value='';add('user',text);msgs.push({role:'user',content:text});b.disabled=true;const d=add('assistant','…');try{const r=await fetch('/api/chat',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({messages:msgs})});const j=await r.json();d.removeChild(d.lastChild);d.appendChild(document.createTextNode(j.reply||'(leer)'));msgs.push({role:'assistant',content:j.reply||''});}catch(err){d.appendChild(document.createTextNode(' Fehler: '+err));}b.disabled=false;log.scrollTop=log.scrollHeight;};
</script></body></html>"#;
