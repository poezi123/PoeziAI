//! PoeziAI command-line entry point.
//!
//! Grows into the chat / generation interface (A6). For now it just proves the
//! workspace builds and links all crates together.

fn main() {
    println!("PoeziAI v{}", env!("CARGO_PKG_VERSION"));
    println!("  gguf   : {}", poezi_gguf::version());
    println!("  tensor : {}", poezi_tensor::version());
    println!("  model  : {}", poezi_model::version());
    println!("\nScaffold only — see PLAN.md for the roadmap (currently: Phase A0).");
}
