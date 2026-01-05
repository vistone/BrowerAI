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
    // 初始化日志
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();

    log::info!("╔════════════════════════════════════════════════════════════════╗");
    log::info!("║          BrowerAI - AI自主学习浏览器                          ║");
    log::info!("║          AI-Powered Self-Learning Browser                     ║");
    log::info!("╚════════════════════════════════════════════════════════════════╝");

    // 解析命令行参数
    let args: Vec<String> = std::env::args().collect();
    let mode = if args.len() > 1 {
        args[1].as_str()
    } else {
        "demo"
    };

    match mode {
        "--ai-report" => {
            // AI 报告模式
            run_ai_report()?;
        }
        "--learn" => {
            // 学习模式：访问真实网站
            let urls = if args.len() > 2 {
                args[2..].iter().map(|s| s.as_str()).collect()
            } else {
                // 默认测试网站
                vec![
                    "https://example.com",
                    "https://httpbin.org/html",
                ]
            };
            run_learning_mode(&urls)?;
        }
        "--export-feedback" => {
            // 导出反馈数据
            let output = if args.len() > 2 {
                &args[2]
            } else {
                "./feedback_data.json"
            };
            run_export_feedback(output)?;
        }
        _ => {
            // 演示模式
            run_demo_mode()?;
        }
    }

    Ok(())
}

/// AI 报告模式
fn run_ai_report() -> Result<()> {
    log::info!("🔍 生成 AI 系统报告...\n");

    let model_dir = PathBuf::from("./models/local");
    let mut model_manager = ModelManager::new(model_dir)?;
    
    // 尝试加载模型配置
    let config_path = PathBuf::from("./models/model_config.toml");
    if config_path.exists() {
        model_manager.load_config(&config_path)?;
        log::info!("✅ 已加载模型配置");
    } else {
        log::warn!("⚠️  模型配置文件不存在: {}", config_path.display());
    }

    let perf_monitor = PerformanceMonitor::new(true);
    let inference_engine = InferenceEngine::with_monitor(perf_monitor.clone())?;
    let runtime = AiRuntime::with_models(inference_engine, model_manager);

    let reporter = AiReporter::new(runtime, perf_monitor);
    let report = reporter.generate_full_report();
    
    println!("{}", report);

    Ok(())
}

/// 学习模式：访问真实网站
fn run_learning_mode(urls: &[&str]) -> Result<()> {
    log::info!("🎓 进入学习模式...\n");

    // 初始化 AI 运行时
    let model_dir = PathBuf::from("./models/local");
    let mut model_manager = ModelManager::new(model_dir)?;
    
    let config_path = PathBuf::from("./models/model_config.toml");
    if config_path.exists() {
        model_manager.load_config(&config_path)?;
    }

    let perf_monitor = PerformanceMonitor::new(true);
    let inference_engine = InferenceEngine::with_monitor(perf_monitor)?;
    let runtime = AiRuntime::with_models(inference_engine, model_manager);

    // 创建网站学习器
    let learner = WebsiteLearner::new(runtime.clone())?;

    // 批量访问网站
    log::info!("🌐 开始批量访问 {} 个网站...\n", urls.len());
    let reports = learner.batch_visit(urls);

    // 生成学习报告
    log::info!("\n{}", "═".repeat(64));
    log::info!("📊 学习报告摘要");
    log::info!("{}", "═".repeat(64));
    
    for report in &reports {
        log::info!("\n{}", report.format());
    }

    // 输出反馈统计
    log::info!("\n{}", runtime.feedback().generate_summary());

    // 自动导出反馈数据
    // 使用当前目录保存反馈文件
    let timestamp = chrono::Local::now().format("%Y%m%d_%H%M%S");
    let feedback_file = format!("feedback_{}.json", timestamp);
    
    // 如果training/data目录存在则使用，否则使用当前目录
    let feedback_path = if std::path::Path::new("./training/data").exists() {
        format!("./training/data/feedback_{}.json", timestamp)
    } else {
        feedback_file.clone()
    };
    learner.export_feedback(&feedback_path)?;

    log::info!("\n✅ 学习完成！下一步:");
    log::info!("  1. 查看反馈数据: {}", feedback_file);
    log::info!("  2. 运行 'cargo run --bin browerai -- --ai-report' 查看 AI 状态");
    log::info!("  3. 使用反馈数据训练模型（参考 training/QUICKSTART.md）");

    Ok(())
}

/// 导出反馈数据
fn run_export_feedback(output: &str) -> Result<()> {
    log::info!("💾 导出反馈数据到: {}", output);
    
    let perf_monitor = PerformanceMonitor::new(true);
    let inference_engine = InferenceEngine::with_monitor(perf_monitor)?;
    let runtime = AiRuntime::new(inference_engine);

    let json = runtime.feedback().export_training_samples()?;
    std::fs::write(output, json)?;

    log::info!("✅ 导出完成！");
    Ok(())
}

/// 演示模式
fn run_demo_mode() -> Result<()> {
    use parser::{CssParser, HtmlParser, JsParser};
    use renderer::RenderEngine;

    log::info!("🎬 演示模式\n");
    log::info!("提示：使用以下参数运行：");
    log::info!("  --ai-report          生成 AI 系统报告");
    log::info!("  --learn [urls...]    访问真实网站并学习");
    log::info!("  --export-feedback    导出反馈数据\n");

    // 初始化 AI 运行时
    let model_dir = PathBuf::from("./models/local");
    let model_manager = ModelManager::new(model_dir)?;
    let perf_monitor = PerformanceMonitor::new(true);
    let inference_engine = InferenceEngine::with_monitor(perf_monitor)?;
    let runtime = AiRuntime::with_models(inference_engine, model_manager);

    // 初始化解析器（使用 AI 运行时）
    let html_parser = HtmlParser::with_ai_runtime(runtime.clone());
    let css_parser = CssParser::with_ai_runtime(runtime.clone());
    let js_parser = JsParser::with_ai_runtime(runtime.clone());

    // 初始化渲染引擎
    let mut render_engine = RenderEngine::new();

    // 示例：解析 HTML
    let sample_html = r#"
        <!DOCTYPE html>
        <html>
            <head>
                <title>BrowerAI 测试页面</title>
            </head>
            <body>
                <h1>欢迎使用 BrowerAI</h1>
                <p>这是一个具有 AI 自主学习能力的浏览器，可以自动解析和渲染网页内容。</p>
                <div>
                    <h2>核心特性</h2>
                    <ul>
                        <li>AI 驱动的 HTML/CSS/JS 解析</li>
                        <li>在线学习和模型优化</li>
                        <li>性能监控和反馈收集</li>
                    </ul>
                </div>
            </body>
        </html>
    "#;

    log::info!("🔍 解析 HTML 文档...");
    let dom = html_parser.parse(sample_html)?;
    let text = html_parser.extract_text(&dom);
    log::info!("📝 提取的文本内容 ({} 字符):\n{}", text.trim().len(), text.trim());

    // 示例：解析 CSS
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

    log::info!("\n🎨 解析 CSS...");
    let css_rules = css_parser.parse(sample_css)?;
    log::info!("✅ 解析了 {} 条 CSS 规则", css_rules.len());

    // 示例：解析 JavaScript
    let sample_js = r#"
        function greet(name) {
            return "你好, " + name + "!";
        }
        
        const result = greet("BrowerAI");
        console.log(result);
        
        // 计算斐波那契数列
        function fibonacci(n) {
            if (n <= 1) return n;
            return fibonacci(n - 1) + fibonacci(n - 2);
        }
    "#;

    log::info!("\n⚙️  解析 JavaScript...");
    let js_ast = js_parser.parse(sample_js)?;
    log::info!("✅ 解析了 {} 条 JavaScript 语句", js_ast.statement_count);

    // 示例：渲染
    log::info!("\n🖼️  渲染 HTML + CSS...");
    let render_tree = render_engine.render(&dom, &css_rules)?;
    log::info!("✅ 创建了包含 {} 个节点的渲染树", render_tree.nodes.len());

    // 显示反馈统计
    log::info!("\n{}", runtime.feedback().generate_summary());

    log::info!("\n✅ 演示完成！");
    log::info!("📖 下一步：运行 'cargo run --bin browerai -- --learn' 开始学习真实网站");

    Ok(())
}
