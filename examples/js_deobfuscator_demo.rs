//! JS Deobfuscator Demo
//!
//! Demonstrates using the Seq2Seq JS deobfuscator model to transform
//! minified/obfuscated JavaScript code.
//!
//! Run with:
//! ```bash
//! cargo run --example js_deobfuscator_demo --features ai
//! ```

use browerai::ai::{integration::JsDeobfuscatorIntegration, InferenceEngine};
use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    env_logger::init();

    println!("=== BrowerAI JS Deobfuscator Demo ===\n");

    // Initialize inference engine
    let engine = InferenceEngine::new()?;
    println!("✓ Inference engine initialized");

    // Load JS deobfuscator model
    let model_path = std::env::current_dir()
        .unwrap()
        .join("models/local/js_deobfuscator_v1.onnx");

    println!("Looking for model at: {:?}", model_path);

    if !model_path.exists() {
        println!("✗ Model not found at {:?}", model_path);
        println!("\nTo use this demo, train the model first:");
        println!("  cd training");
        println!("  python scripts/train_seq2seq_deobfuscator.py");
        println!("  cp models/js_deobfuscator_v1.onnx* ../models/local/");
        return Ok(());
    }

    let mut integration = JsDeobfuscatorIntegration::new(&engine, Some(&model_path), None)?;
    println!("✓ JS deobfuscator model loaded");
    println!("  Enabled: {}\n", integration.is_enabled());

    // Test cases
    let test_cases = vec![
        ("Simple minified", "var a=function(){return 42;}"),
        ("Arrow function", "const b=(x)=>x*2;"),
        ("Obfuscated loop", "for(let i=0;i<10;i++){console.log(i)}"),
    ];

    for (name, obfuscated) in test_cases {
        println!("--- Test: {} ---", name);
        println!("Input:  {}", obfuscated);

        match integration.deobfuscate(obfuscated) {
            Ok(deobfuscated) => {
                println!("Output: {}", deobfuscated);
            }
            Err(e) => {
                println!("Error: {}", e);
                println!("Note: Model may need more training data for better results");
            }
        }
        println!();
    }

    println!("=== Demo Complete ===");
    println!("\nModel Info:");
    println!("  Architecture: Seq2Seq (BiLSTM Encoder + LSTM Decoder)");
    println!("  Parameters: ~2.2M");
    println!("  Vocabulary: 160 tokens (JS keywords + operators + variables)");
    println!("  Max sequence: 60 tokens");
    println!("\nNext Steps:");
    println!("  1. Collect more real JS samples (training/scripts/crawl_js_assets.py)");
    println!(
        "  2. Generate more obfuscation pairs (training/scripts/generate_obfuscation_pairs.py)"
    );
    println!("  3. Retrain with larger dataset for better quality");

    Ok(())
}
