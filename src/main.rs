mod ai;
mod dom;
mod learning;
mod network;
mod parser;
mod plugins;
mod renderer;

use anyhow::Result;
use std::path::PathBuf;

use ai::{AiReporter, AiRuntime, FeedbackPipeline, InferenceEngine, ModelManager};
use ai::performance_monitor::PerformanceMonitor;
use learning::WebsiteLearner;

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
                vec![
                    "https://example.com",
                    "https://httpbin.org/html",
                ]
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
        log::warn!("⚠️  Model configuration file not found: {}", config_path.display());
    }

    let perf_monitor = PerformanceMonitor::new(true);
    let inference_engine = InferenceEngine::with_monitor(perf_monitor.clone())?;
    let runtime = AiRuntime::with_models(inference_engine, model_manager);

    let reporter = AiReporter::new(runtime, perf_monitor);
    let report = reporter.generate_full_report();
    
    println!("{}", report);

    Ok(())
}

/// Learning mode: visit real websites
fn run_learning_mode(urls: &[&str]) -> Result<()> {
    log::info!("🎓 Entering learning mode...\n");

    // Initialize AI runtime
    let model_dir = PathBuf::from("./models/local");
    let mut model_manager = ModelManager::new(model_dir)?;
    
    let config_path = PathBuf::from("./models/model_config.toml");
    if config_path.exists() {
        model_manager.load_config(&config_path)?;
    }

    let perf_monitor = PerformanceMonitor::new(true);
    let inference_engine = InferenceEngine::with_monitor(perf_monitor)?;
    let runtime = AiRuntime::with_models(inference_engine, model_manager);

    // Create website learner
    let learner = WebsiteLearner::new(runtime.clone())?;

    // Batch visit websites
    log::info!("🌐 Starting batch visit of {} websites...\n", urls.len());
    let reports = learner.batch_visit(urls);

    // Generate learning report
    log::info!("\n{}", "═".repeat(64));
    log::info!("📊 Learning Report Summary");
    log::info!("{}", "═".repeat(64));
    
    for report in &reports {
        log::info!("\n{}", report.format());
    }

    // Output feedback statistics
    log::info!("\n{}", runtime.feedback().generate_summary());

    // Auto-export feedback data
    // Use current directory to save feedback file
    let timestamp = chrono::Local::now().format("%Y%m%d_%H%M%S");
    let feedback_file = format!("feedback_{}.json", timestamp);
    
    // Use training/data directory if exists, otherwise use current directory
    let feedback_path = if std::path::Path::new("./training/data").exists() {
        format!("./training/data/feedback_{}.json", timestamp)
    } else {
        feedback_file.clone()
    };
    learner.export_feedback(&feedback_path)?;

    log::info!("\n✅ Learning completed! Next steps:");
    log::info!("  1. View feedback data: {}", feedback_file);
    log::info!("  2. Run 'cargo run --bin browerai -- --ai-report' to check AI status");
    log::info!("  3. Train models using feedback data (see training/QUICKSTART.md)");

    Ok(())
}

/// Export feedback data
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

/// Demo mode
fn run_demo_mode() -> Result<()> {
    use parser::{CssParser, HtmlParser, JsParser};
    use renderer::RenderEngine;

    log::info!("🎬 Demo Mode\n");
    log::info!("Hint: Run with the following options:");
    log::info!("  --ai-report          Generate AI system report");
    log::info!("  --learn [urls...]    Visit real websites and learn");
    log::info!("  --export-feedback    Export feedback data\n");

    // Initialize AI runtime
    let model_dir = PathBuf::from("./models/local");
    let model_manager = ModelManager::new(model_dir)?;
    let perf_monitor = PerformanceMonitor::new(true);
    let inference_engine = InferenceEngine::with_monitor(perf_monitor)?;
    let runtime = AiRuntime::with_models(inference_engine, model_manager);

    // Initialize parsers (with AI runtime)
    let html_parser = HtmlParser::with_ai_runtime(runtime.clone());
    let css_parser = CssParser::with_ai_runtime(runtime.clone());
    let js_parser = JsParser::with_ai_runtime(runtime.clone());

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
    log::info!("📝 Extracted text content ({} characters):\n{}", text.trim().len(), text.trim());

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
    log::info!("✅ Created render tree with {} nodes", render_tree.nodes.len());

    // Display feedback statistics
    log::info!("\n{}", runtime.feedback().generate_summary());

    log::info!("\n✅ Demo completed!");
    log::info!("📖 Next step: Run 'cargo run --bin browerai -- --learn' to start learning from real websites");

    Ok(())
}
