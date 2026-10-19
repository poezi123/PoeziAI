#!/usr/bin/env python3
"""Generate a web-security fine-tuning dataset for OVERFLOW.

Concise, practical answers on web vulnerabilities and their fixes — for
authorized testing and secure coding. Same JSONL chat format as the other
generators.
"""
import json
import random
from pathlib import Path

random.seed(2)

QA = [
    ("How do I test for SQL injection?",
     "Try breaking the query context: a single quote, then `' OR '1'='1`, then order-by/union probing to count columns. Prefer parameterized queries as the fix. Use `sqlmap -u URL --batch` to confirm and enumerate, only against targets you're authorized to test."),
    ("What is stored vs reflected XSS?",
     "Reflected XSS echoes attacker input straight back in the response (needs a crafted link); stored XSS is saved server-side (e.g. a comment) and hits every viewer. Both run attacker JS in the victim's origin. Fix: context-aware output encoding and a strict Content-Security-Policy."),
    ("How does CSRF work and how do I prevent it?",
     "The browser auto-sends cookies, so a malicious site can trigger a state-changing request as the logged-in victim. Prevent with per-request anti-CSRF tokens, `SameSite=Lax/Strict` cookies, and requiring a custom header for state-changing calls."),
    ("What is SSRF and why is it dangerous?",
     "Server-Side Request Forgery: the app fetches a URL you control, so you make it reach internal services (169.254.169.254 cloud metadata, localhost admin panels). Fix: allowlist destinations, resolve+validate the IP, block link-local/private ranges, and disable unused URL schemes."),
    ("How do I find IDOR?",
     "Change an object identifier you shouldn't own — `/api/orders/1001` → `1002` — and see if you get someone else's data. Fix: check authorization on the object per request, not just authentication; prefer unguessable IDs but never rely on that alone."),
    ("What's the right way to store passwords?",
     "A slow, salted password hash: Argon2id (preferred), scrypt, or bcrypt — never fast hashes like MD5/SHA-256 alone. Each password gets a unique random salt; the KDF cost is tuned so a single verify takes tens of milliseconds."),
    ("How do JWTs get misused?",
     "Common bugs: accepting `alg: none`, confusing HS256/RS256 so the public key is used as an HMAC secret, no expiry, and no revocation. Validate the algorithm server-side, verify signature and `exp`/`aud`/`iss`, and keep secrets out of the token."),
    ("What is path traversal?",
     "User input like `../../etc/passwd` escapes the intended directory. Fix: canonicalize the path and verify it stays within the allowed base dir; don't pass user input to file APIs directly, and reject `..` and absolute paths."),
    ("How do I set a secure cookie?",
     "`Set-Cookie: session=...; Secure; HttpOnly; SameSite=Lax; Path=/`. `HttpOnly` blocks JS access (XSS theft), `Secure` keeps it HTTPS-only, `SameSite` limits cross-site sending. Scope `Path`/`Domain` tightly and keep sessions short."),
    ("What does a good CSP look like?",
     "Start strict and loosen as needed: `default-src 'self'; script-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'`. Avoid `unsafe-inline`; use nonces/hashes for needed inline scripts. CSP is defense-in-depth, not a substitute for output encoding."),
    ("How do I test file-upload security?",
     "Check whether the server trusts the client-supplied filename/content-type, stores uploads in web root, or runs them. Try double extensions, null bytes, SVG/HTML with script, and polyglots. Fix: validate type by content, store outside web root, randomize names, serve with a safe content-type."),
    ("What is open redirect and why does it matter?",
     "A `?next=` parameter that redirects anywhere lets attackers craft trusted-looking phishing links and can chain into OAuth token theft. Fix: allowlist redirect targets or only permit relative paths."),
]

SYS = "You are OVERFLOW, a local cybersecurity AI. Give concise, practical web-security answers for authorized testing and secure coding."

def ex(u, a):
    return {"messages": [
        {"role": "system", "content": SYS},
        {"role": "user", "content": u},
        {"role": "assistant", "content": a},
    ]}

rows = []
wraps = ["{q}", "{q}", "On a pentest, {q}", "Secure coding: {q}", "Quick question — {q}"]
for _ in range(3):
    for q, a in QA:
        w = random.choice(wraps).format(q=q[0].lower() + q[1:])
        rows.append(ex(w, a))

random.shuffle(rows)
out = Path(__file__).parent / "data" / "overflow_websec.jsonl"
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(f"wrote {len(rows)} examples to {out}")
