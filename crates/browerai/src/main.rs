use anyhow::{anyhow, Result};
#[cfg(feature = "ai")]
use std::path::PathBuf;

#[cfg(feature = "ai")]
use browerai_ai_core::performance_monitor::PerformanceMonitor;
#[cfg(feature = "ai")]
use browerai_ai_core::{AiReporter, AiRuntime, FeedbackPipeline, InferenceEngine, ModelManager};
use browerai_css_parser::CssParser;
use browerai_html_parser::HtmlParser;
use browerai_js_parser::JsParser;
#[cfg(feature = "ml")]
use browerai_ml::MlSession;
use browerai_renderer_core::RenderEngine;

fn main() -> Result<()> {
    // Initialize logger
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();

    log::info!("╔════════════════════════════════════════════════════════════════╗");
    log::info!("║          BrowerAI - AI-Powered Self-Learning Browser         ║");
    log::info!("╚════════════════════════════════════════════════════════════════╝");

    // Parse command line arguments
    let args: Vec<String> = std::env::args().collect();
    let mode = if args.len() > 1 {
        args[1].as_str()
    } else {
        "demo"
    };

    match mode {
        "--ai-report" => {
            // AI report mode
            run_ai_report()?;
        }
        "--learn" => {
            // Learning mode: visit real websites
            let urls = if args.len() > 2 {
                args[2..].iter().map(|s| s.as_str()).collect()
            } else {
                // Default test websites
                vec!["https://example.com", "https://httpbin.org/html"]
            };
            run_learning_mode(&urls)?;
        }
        "--export-feedback" => {
            // Export feedback data
            let output = if args.len() > 2 {
                &args[2]
            } else {
                "./feedback_data.json"
            };
            run_export_feedback(output)?;
        }
        _ => {
            // Demo mode
            run_demo_mode()?;
        }
    }

    Ok(())
}

/// AI report mode
#[cfg(feature = "ai")]
fn run_ai_report() -> Result<()> {
    log::info!("🔍 Generating AI system report...\n");

    let model_dir = PathBuf::from("./models/local");
    let mut model_manager = ModelManager::new(model_dir)?;

    // Try to load model configuration
    let config_path = PathBuf::from("./models/model_config.toml");
    if config_path.exists() {
        model_manager.load_config(&config_path)?;
        log::info!("✅ Model configuration loaded");
    } else {
        log::warn!(
            "⚠️  Model configuration file not found: {}",
            config_path.display()
        );
    }

    let perf_monitor = PerformanceMonitor::new(true);
    let inference_engine = InferenceEngine::with_monitor(perf_monitor.clone())?;
    let runtime = AiRuntime::with_models(inference_engine, model_manager);

    let reporter = AiReporter::new(runtime, perf_monitor);
    let report = reporter.generate_full_report();

    println!("{}", report);

    Ok(())
}

#[cfg(not(feature = "ai"))]
fn run_ai_report() -> Result<()> {
    Err(anyhow!(
        "AI feature is disabled. Rebuild with --features ai to generate reports."
    ))
}

/// Learning mode: visit real websites
#[cfg(feature = "ml")]
fn run_learning_mode(urls: &[&str]) -> Result<()> {
    log::info!("🎓 Entering learning mode (Rust/tch-rs)...\n");

    // Replace old WebsiteLearner with Rust ML session using tch-rs
    let session = MlSession::new()?;
    let output = session.smoke_test()?;

    log::info!(
        "🌐 URLs provided (placeholder, not crawled here): {:?}",
        urls
    );
    log::info!(
        "✅ ML smoke test succeeded. Output shape: {:?}",
        output.size()
    );
    log::info!("💡 Next: wire real data pipeline to tch models (training/inference)");

    Ok(())
}

#[cfg(not(feature = "ml"))]
fn run_learning_mode(urls: &[&str]) -> Result<()> {
    log::info!("🌐 URLs provided: {:?}", urls);
    Err(anyhow!(
        "ML feature is disabled. Rebuild with --features ml to enable learning mode with tch-rs."
    ))
}

/// Export feedback data
#[cfg(feature = "ai")]
fn run_export_feedback(output: &str) -> Result<()> {
    log::info!("💾 Exporting feedback data to: {}", output);

    let perf_monitor = PerformanceMonitor::new(true);
    let inference_engine = InferenceEngine::with_monitor(perf_monitor)?;
    let runtime = AiRuntime::new(inference_engine);

    let json = runtime.feedback().export_training_samples()?;
    std::fs::write(output, json)?;

    log::info!("✅ Export completed!");
    Ok(())
}

#[cfg(not(feature = "ai"))]
fn run_export_feedback(_output: &str) -> Result<()> {
    Err(anyhow!(
        "AI feature is disabled. Rebuild with --features ai to export feedback."
    ))
}

/// Demo mode
fn run_demo_mode() -> Result<()> {
    log::info!("🎬 Demo Mode\n");
    log::info!("Hint: Run with the following options:");
    log::info!("  --ai-report          Generate AI system report");
    log::info!("  --learn [urls...]    Visit real websites and learn");
    log::info!("  --export-feedback    Export feedback data\n");

    // Initialize parsers (baseline, AI disabled by default)
    let html_parser = HtmlParser::new();
    let css_parser = CssParser::new();
    let js_parser = JsParser::new();

    // Initialize render engine
    let mut render_engine = RenderEngine::new();

    // Example: Parse HTML
    let sample_html = r#"
        <!DOCTYPE html>
        <html>
            <head>
                <title>BrowerAI Test Page</title>
            </head>
            <body>
                <h1>Welcome to BrowerAI</h1>
                <p>This is an AI-powered self-learning browser that can automatically parse and render web content.</p>
                <div>
                    <h2>Core Features</h2>
                    <ul>
                        <li>AI-driven HTML/CSS/JS parsing</li>
                        <li>Online learning and model optimization</li>
                        <li>Performance monitoring and feedback collection</li>
                    </ul>
                </div>
            </body>
        </html>
    "#;

    log::info!("🔍 Parsing HTML document...");
    let dom = html_parser.parse(sample_html)?;
    let text = html_parser.extract_text(&dom);
    log::info!(
        "📝 Extracted text content ({} characters):\n{}",
        text.trim().len(),
        text.trim()
    );

    // Example: Parse CSS
    let sample_css = r#"
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }
        h1 {
            color: #333;
            font-size: 24px;
        }
        h2 {
            color: #666;
        }
    "#;

    log::info!("\n🎨 Parsing CSS...");
    let css_rules = css_parser.parse(sample_css)?;
    log::info!("✅ Parsed {} CSS rules", css_rules.len());

    // Example: Parse JavaScript
    let sample_js = r#"
        function greet(name) {
            return "Hello, " + name + "!";
        }
        
        const result = greet("BrowerAI");
        console.log(result);
        
        // Calculate Fibonacci sequence
        function fibonacci(n) {
            if (n <= 1) return n;
            return fibonacci(n - 1) + fibonacci(n - 2);
        }
    "#;

    log::info!("\n⚙️  Parsing JavaScript...");
    let js_ast = js_parser.parse(sample_js)?;
    log::info!("✅ Parsed {} JavaScript statements", js_ast.statement_count);

    // Example: Rendering
    log::info!("\n🖼️  Rendering HTML + CSS...");
    let render_tree = render_engine.render(&dom, &css_rules)?;
    log::info!(
        "✅ Created render tree with {} nodes",
        render_tree.nodes.len()
    );

    log::info!("\n✅ Demo completed!");
    log::info!("📖 Next step: Run 'cargo run --bin browerai -- --learn' to start learning from real websites");

    Ok(())
}
