#!/usr/bin/env python3
"""Generate a CTF / binary-exploitation fine-tuning dataset for OVERFLOW.

Short, concrete walkthroughs in chat format — the kind of reasoning a CTF player
wants: what the bug is, how to find it, and how to drive it to a shell/flag. For
authorized security practice and CTFs. Same JSONL chat format as the other
generators, so it drops straight into the Learning tab.
"""
import json
import random
from pathlib import Path

random.seed(1)

QA = [
    ("How do I find the offset to RIP in a stack buffer overflow?",
     "Generate a cyclic pattern (`pwndbg` `cyclic 200` or pwntools `cyclic(200)`), feed it as input, let it crash, then read the value in RIP (or the saved return address) and `cyclic -l <value>` to get the exact offset."),
    ("What is a ret2libc attack and when do I use it?",
     "When the stack is non-executable (NX) you can't run shellcode on it, so instead you overwrite the return address to jump into existing libc code — classically `system(\"/bin/sh\")`. You need libc's base (leak it) and the offsets of `system` and a `\"/bin/sh\"` string; align the stack to 16 bytes before the call on modern glibc."),
    ("How do I leak a libc address to defeat ASLR?",
     "Use a bug that prints memory (format string, or an overflow that lets you call `puts`/`write` on a GOT entry). Call `puts(puts@got)` via a small ROP chain, read the leaked address, subtract the known offset of `puts` in your libc to get the base, then compute every other symbol from that base."),
    ("What's a ROP chain?",
     "Return-Oriented Programming: you chain together short instruction sequences ending in `ret` ('gadgets') already present in the binary/libc to perform arbitrary operations without injecting code. Find gadgets with `ROPgadget` or `ropper`, then stack the addresses (and the data they pop) as your payload."),
    ("How do I exploit a format string vulnerability?",
     "If user input reaches `printf(user)` directly, `%p` leaks stack values and `%n` writes. Find your input's argument index with a few `%p`s, then use `%<n>$p` to leak and `%<n>$n` (with width specifiers to control the written value) to overwrite a GOT entry or return address."),
    ("What is a GOT overwrite?",
     "The Global Offset Table holds resolved addresses of libc functions. If you get an arbitrary write, overwrite e.g. `printf@got` with `system`, so the next `printf(\"/bin/sh\")` actually calls `system(\"/bin/sh\")`. Works when Full RELRO is off."),
    ("How do I check binary protections before exploiting?",
     "`checksec --file=./bin` (or pwntools `ELF('./bin')`). Look at NX (no stack execution), PIE (base randomized), RELRO (GOT writable?), and stack canary. These decide your technique: canary → leak it first; PIE → leak a code address; NX → ROP/ret2libc."),
    ("How do I bypass a stack canary?",
     "You can't guess it, so leak it: a format-string read, an off-by-one that prints past a string, or a child process that keeps the same canary after fork. Read the canary, then include its exact value in your overflow so the check passes before you reach the saved RIP."),
    ("Give me a minimal pwntools solve skeleton.",
     "```python\nfrom pwn import *\nelf = context.binary = ELF('./chall')\nio = process('./chall')  # or remote('host', 1337)\nio.sendlineafter(b'> ', b'A'*offset + p64(gadget))\nio.interactive()\n```\nSet `context.log_level='debug'` while developing."),
    ("How do I approach an unknown CTF binary?",
     "`file` and `checksec` it, run `strings`, open it in Ghidra/IDA for the `main` logic, run it to see the interface, then look for the classic bugs: unbounded `gets`/`read`, format strings, integer/size mismatches, use-after-free. Form a hypothesis, test it with a crash, then build the exploit."),
    ("What is use-after-free and how is it exploited in CTFs?",
     "A freed chunk is still referenced. Free an object, allocate a new one of the same size so the allocator hands back that memory, and now the dangling pointer reads/writes your controlled data — often used to hijack a function pointer or vtable. Heap-note challenges are the usual setting."),
    ("How does tcache poisoning work?",
     "In modern glibc, freeing a small chunk pushes it onto a per-size tcache singly-linked list. If you can overwrite a freed chunk's `fd`, the next two allocations return first the real chunk and then your arbitrary pointer — giving an arbitrary-write primitive. On glibc ≥2.32 account for the pointer 'safe-linking' XOR."),
    ("How do I find a '/bin/sh' string for ret2libc?",
     "It already exists in libc. With pwntools: `next(libc.search(b'/bin/sh'))` after setting `libc.address` to the leaked base. In gdb: `search-pattern /bin/sh`."),
]

SYS = "You are OVERFLOW, a local cybersecurity AI for authorized CTF and exploitation practice. Answer concisely and technically."

def ex(u, a):
    return {"messages": [
        {"role": "system", "content": SYS},
        {"role": "user", "content": u},
        {"role": "assistant", "content": a},
    ]}

rows = []
wraps = ["{q}", "{q}", "For a CTF: {q}", "In pwn, {q}", "Quick: {q}"]
for _ in range(3):
    for q, a in QA:
        w = random.choice(wraps).format(q=q[0].lower() + q[1:])
        rows.append(ex(w, a))

random.shuffle(rows)
out = Path(__file__).parent / "data" / "overflow_ctf.jsonl"
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(f"wrote {len(rows)} examples to {out}")
