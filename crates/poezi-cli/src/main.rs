//! PoeziAI command-line entry point.
//!
//! Grows into the chat / generation interface (A6). For now:
//!   poezi                 -- print versions
//!   poezi info <model>    -- parse a GGUF file and print a summary (A1)

use std::process::ExitCode;

use poezi_gguf::MetaValue;

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
