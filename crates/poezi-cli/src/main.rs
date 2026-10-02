//! PoeziAI command-line entry point.
//!
//! Grows into the chat / generation interface (A6). For now:
//!   poezi                 -- print versions
//!   poezi info <model>    -- parse a GGUF file and print a summary (A1)

use std::process::ExitCode;

use poezi_gguf::MetaValue;

mod chat;

/// Read the value after a `--flag` on the command line, if present.
fn flag_val<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .map(String::as_str)
}

use std::io::Write;

/// Greedy (argmax) generation — the deterministic path we compare against
/// llama.cpp at temp 0.
fn generate(path: &str, prompt: &str, n_predict: usize) -> ExitCode {
    let gguf = match poezi_gguf::GgufFile::open(path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };
    let tok = match poezi_model::Tokenizer::from_gguf(&gguf) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };
    let model = match poezi_model::Model::load(&gguf) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };

    eprintln!("backend: {}", model.backend());
    let ids = tok.encode(prompt, tok.add_bos, true);
    eprintln!("prompt: {} tokens", ids.len());

    let mut cache = model.new_cache();
    let mut last_logits = Vec::new();
    for &id in &ids {
        last_logits = model.forward(id, &mut cache);
    }

    let mut generated: Vec<u32> = Vec::new();
    let mut out = std::io::stdout();
    for _ in 0..n_predict {
        let next = argmax(&last_logits) as u32;
        if Some(next) == tok.eos_id {
            break;
        }
        generated.push(next);
        // Stream bytes as they decode.
        let mut bytes = Vec::new();
        tok.push_token_bytes(next, &mut bytes);
        let _ = out.write_all(&bytes);
        let _ = out.flush();
        last_logits = model.forward(next, &mut cache);
    }
    println!();
    eprintln!("\ngenerated {} tokens", generated.len());
    let ids_str: Vec<String> = generated.iter().map(|i| i.to_string()).collect();
    eprintln!("ids: [{}]", ids_str.join(", "));
    ExitCode::SUCCESS
}

fn argmax(v: &[f32]) -> usize {
    let mut best = 0;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > best_val {
            best_val = x;
            best = i;
        }
    }
    best
}

/// Print a metadata value; arrays are summarized (count + a window of items
/// starting at `start`).
fn print_meta(v: &MetaValue, start: usize) {
    match v {
        MetaValue::Array(items) => {
            println!("array of {} items", items.len());
            for (off, it) in items.iter().skip(start).take(16).enumerate() {
                println!("  [{}] {it:?}", start + off);
            }
        }
        other => println!("{other:?}"),
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("info") => {
            let Some(path) = args.get(1) else {
                eprintln!("usage: poezi info <model.gguf>");
                return ExitCode::FAILURE;
            };
            match poezi_gguf::GgufFile::open(path) {
                Ok(f) => {
                    print!("{}", f.summary());
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    eprintln!("error: {e}");
                    ExitCode::FAILURE
                }
            }
        }
        Some("gpu") => {
            match poezi_cuda::device_name() {
                Ok(name) => {
                    println!("GPU: {name}");
                    match poezi_cuda::self_test_q8(4096, 4096) {
                        Ok(err) => {
                            println!("Q8_0 matmul (4096x4096): max error {err:.2e} — OK");
                            ExitCode::SUCCESS
                        }
                        Err(e) => {
                            eprintln!("Q8_0 kernel error: {e}");
                            ExitCode::FAILURE
                        }
                    }
                }
                Err(e) => {
                    eprintln!("no CUDA device: {e}");
                    eprintln!("(run with LD_LIBRARY_PATH=vendor/cuda/lib after scripts/fetch-cuda.sh)");
                    ExitCode::FAILURE
                }
            }
        }
        Some("chat") => {
            let Some(path) = args.get(1) else {
                eprintln!("usage: poezi chat <model.gguf> [--system S] [--temp T] [--top-k K] [--top-p P] [--seed N]");
                return ExitCode::FAILURE;
            };
            let seed = flag_val(&args, "--seed")
                .and_then(|s| s.parse().ok())
                .unwrap_or_else(|| {
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_nanos() as u64)
                        .unwrap_or(0x9E3779B9)
                });
            let opts = chat::ChatOpts {
                system: flag_val(&args, "--system")
                    .unwrap_or("You are PoeziAI, a helpful and knowledgeable assistant.")
                    .to_string(),
                temp: flag_val(&args, "--temp").and_then(|s| s.parse().ok()).unwrap_or(0.7),
                top_k: flag_val(&args, "--top-k").and_then(|s| s.parse().ok()).unwrap_or(40),
                top_p: flag_val(&args, "--top-p").and_then(|s| s.parse().ok()).unwrap_or(0.95),
                seed,
                max_tokens: flag_val(&args, "-n").and_then(|s| s.parse().ok()).unwrap_or(512),
            };
            chat::run(path, opts)
        }
        Some("gen") => {
            let (Some(path), Some(prompt)) = (args.get(1), args.get(2)) else {
                eprintln!("usage: poezi gen <model.gguf> <prompt> [-n N]");
                return ExitCode::FAILURE;
            };
            let n_predict: usize = args
                .iter()
                .position(|a| a == "-n")
                .and_then(|i| args.get(i + 1))
                .and_then(|s| s.parse().ok())
                .unwrap_or(32);
            generate(path, prompt, n_predict)
        }
        Some("tokenize") => {
            let (Some(path), Some(text)) = (args.get(1), args.get(2)) else {
                eprintln!("usage: poezi tokenize <model.gguf> <text> [--ids]");
                return ExitCode::FAILURE;
            };
            let ids_only = args.iter().any(|a| a == "--ids");
            match poezi_gguf::GgufFile::open(path) {
                Ok(f) => match poezi_model::Tokenizer::from_gguf(&f) {
                    Ok(tok) => {
                        let ids = tok.encode(text, tok.add_bos, true);
                        if ids_only {
                            let strs: Vec<String> = ids.iter().map(|i| i.to_string()).collect();
                            println!("{}", strs.join(" "));
                        } else {
                            println!("{} tokens", ids.len());
                            for id in &ids {
                                let t = tok.token_text(*id).unwrap_or("<?>");
                                println!("  {id:>6}  {t:?}");
                            }
                            let round = tok.decode(&ids);
                            println!("decode: {round:?}");
                            println!("roundtrip_ok: {}", round == *text);
                        }
                        ExitCode::SUCCESS
                    }
                    Err(e) => {
                        eprintln!("error: {e}");
                        ExitCode::FAILURE
                    }
                },
                Err(e) => {
                    eprintln!("error: {e}");
                    ExitCode::FAILURE
                }
            }
        }
        Some("meta") => {
            let (Some(path), Some(key)) = (args.get(1), args.get(2)) else {
                eprintln!("usage: poezi meta <model.gguf> <key>");
                return ExitCode::FAILURE;
            };
            let start: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(0);
            match poezi_gguf::GgufFile::open(path) {
                Ok(f) => match f.meta(key) {
                    Some(v) => {
                        print_meta(v, start);
                        ExitCode::SUCCESS
                    }
                    None => {
                        eprintln!("no such key: {key}");
                        ExitCode::FAILURE
                    }
                },
                Err(e) => {
                    eprintln!("error: {e}");
                    ExitCode::FAILURE
                }
            }
        }
        Some(other) => {
            eprintln!("unknown command: {other}");
            eprintln!("usage: poezi [info <model.gguf>]");
            ExitCode::FAILURE
        }
        None => {
            println!("PoeziAI v{}", env!("CARGO_PKG_VERSION"));
            println!("  gguf   : {}", poezi_gguf::version());
            println!("  tensor : {}", poezi_tensor::version());
            println!("  model  : {}", poezi_model::version());
            println!("\nTry: poezi info <model.gguf>   (see PLAN.md for the roadmap)");
            ExitCode::SUCCESS
        }
    }
}
