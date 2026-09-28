//! PoeziAI command-line entry point.
//!
//! Grows into the chat / generation interface (A6). For now:
//!   poezi                 -- print versions
//!   poezi info <model>    -- parse a GGUF file and print a summary (A1)

use std::process::ExitCode;

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
