// 双渲染模式示例
// 演示原始渲染 vs AI再生成渲染

use anyhow::Result;
use browerai::network::HttpClient;
use browerai::renderer::{RenderEngine, WebsiteRegenerator};

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();

    println!("=== BrowerAI 双渲染模式演示 ===\n");

    // 测试URL
    let url = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "https://example.com".to_string());

    println!("📥 Fetching: {}", url);

    // 1. 获取网站内容
    let client = HttpClient::new();
    let response = client.get(&url).await?;
    let html = response.body;

    println!("✅ Fetched {} bytes\n", html.len());

    // 2. 原始渲染
    println!("🎨 Original Rendering:");
    println!("{}", "=".repeat(50));

    let engine = RenderEngine::default();
    let original_render = engine.render_html(&html)?;

    println!("DOM Nodes: {}", original_render.node_count);
    println!("Layout Time: {:?}", original_render.layout_time);
    println!("Paint Time: {:?}", original_render.paint_time);
    println!();

    // 3. AI再生成
    println!("🤖 AI Regeneration:");
    println!("{}", "=".repeat(50));

    // 检查模型是否存在
    let model_path = "models/local/website_generator_v1.onnx";
    let config_path = "models/local/website_generator_v1_config.json";

    if !std::path::Path::new(model_path).exists() {
        println!("⚠️  Model not found: {}", model_path);
        println!("Please train and export the model first:");
        println!("  cd training");
        println!("  python3 scripts/train_paired_website_generator.py");
        println!("  python3 scripts/export_to_onnx.py --checkpoint checkpoints/paired_generator/epoch_30.pt");
        return Ok(());
    }

    let regenerator = WebsiteRegenerator::new(model_path, config_path)?;
    let regenerated = regenerator.regenerate_from_html(&html)?;

    println!("✅ Regeneration complete");
    println!("Original HTML: {} bytes", html.len());
    println!("Regenerated HTML: {} bytes", regenerated.html.len());
    println!("Regenerated CSS: {} bytes", regenerated.css.len());
    println!("Regenerated JS: {} bytes", regenerated.js.len());
    println!();

    // 4. AI版本渲染
    println!("🎨 AI-Regenerated Rendering:");
    println!("{}", "=".repeat(50));

    let ai_render = engine.render_html(&regenerated.html)?;

    println!("DOM Nodes: {}", ai_render.node_count);
    println!("Layout Time: {:?}", ai_render.layout_time);
    println!("Paint Time: {:?}", ai_render.paint_time);
    println!();

    // 5. 对比
    println!("📊 Comparison:");
    println!("{}", "=".repeat(50));
    println!(
        "Size Reduction: {:.1}%",
        (1.0 - regenerated.html.len() as f64 / html.len() as f64) * 100.0
    );
    println!(
        "Node Reduction: {:.1}%",
        (1.0 - ai_render.node_count as f64 / original_render.node_count as f64) * 100.0
    );

    // 6. 显示代码片段对比
    println!("\n📝 Code Snippets:");
    println!("{}", "=".repeat(50));

    println!("Original (first 200 chars):");
    println!("{}", &html[..200.min(html.len())]);
    println!();

    println!("AI-Regenerated (first 200 chars):");
    println!("{}", &regenerated.html[..200.min(regenerated.html.len())]);

    Ok(())
}
