#![cfg(feature = "ai-candle")]

use anyhow::{Context, Result};
use std::path::PathBuf;

use browerai_ai_core::CandleCodeLlm;

fn main() -> Result<()> {
    run_codegen_demo()
}

#[cfg(feature = "ai-candle")]
fn run_codegen_demo() -> Result<()> {
    env_logger::init();

    // Throttle CPU threads by default to avoid pegging all cores.
    // You can override by exporting RAYON_NUM_THREADS / OMP_NUM_THREADS / CANDLE_NUM_THREADS / THREADS_HINT.
    let threads_hint: usize = std::env::var("THREADS_HINT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2);
    if std::env::var("RAYON_NUM_THREADS").is_err() {
        std::env::set_var("RAYON_NUM_THREADS", threads_hint.to_string());
    }
    if std::env::var("OMP_NUM_THREADS").is_err() {
        std::env::set_var("OMP_NUM_THREADS", threads_hint.to_string());
    }
    if std::env::var("CANDLE_NUM_THREADS").is_err() {
        std::env::set_var("CANDLE_NUM_THREADS", threads_hint.to_string());
    }

    // Default paths (downloaded by scripts/download_qwen2_5_coder_gguf.sh)
    let model_path =
        PathBuf::from("models/local/qwen2_5_coder_7b_gguf/qwen2.5-coder-7b-instruct-q5_k_m.gguf");
    let tokenizer_path = PathBuf::from("models/local/qwen2_5_coder_7b_gguf/tokenizer.json");

    let prefer_gpu = std::env::var("PREFER_GPU")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(true);

    log::info!("🚀 Loading Qwen2.5-Coder-7B (GGUF) via Candle ...");
    let mut llm = CandleCodeLlm::new(&model_path, &tokenizer_path, prefer_gpu)
        .with_context(|| "Failed to build CandleCodeLlm")?;

    // A short JS/HTML-aware coding prompt
    let prompt = r#"You are a concise JS/HTML/CSS assistant. Given a task, return a minimal code snippet.
Task: Write a vanilla JS function `debounce(fn, delay)` and show a usage example that logs scroll events."#;

    // Allow shortening generation to keep CPU load reasonable.
    let max_new_tokens: usize = std::env::var("MAX_NEW_TOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(80);

    let output = llm.generate(
        prompt,
        max_new_tokens,
        /* temperature */ 0.7,
        /* top_k */ Some(40),
        /* top_p */ Some(0.9),
        /* repeat_penalty */ 1.05,
        /* repeat_last_n */ 48,
    )?;

    println!("\n=== Model Output ===\n{}", output.trim());
    Ok(())
}
