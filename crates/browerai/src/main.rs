//! BrowerAI - 真正的AI驱动浏览器
//! 核心：双沙盒架构 - 标准渲染 + AI 学习 → 保功能、换体验

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

// 核心模块
use browerai_dual_sandbox::{
    DualSandboxEngine,
    ComponentExtractor, JsUnderstander,
};

/// BrowerAI - AI驱动的智能浏览器
#[derive(Parser)]
#[command(name = "browerai")]
#[command(about = "BrowerAI: 双沙盒架构 - 真正的AI学习", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// 学习网站意图 + 生成新站点（保功能、换体验）
    Learn {
        /// 要学习的网站URL
        url: String,

        /// 输出目录
        #[arg(short, long, default_value = "output")]
        output: PathBuf,

        /// 生成体验变体数量
        #[arg(short = 'n', long, default_value = "3")]
        variants: usize,
    },

    /// 显示版本信息
    Version,
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Learn { url, output, variants } => {
            learn_and_generate(&url, &output, variants).await?;
        }
        Commands::Version => {
            println!("BrowerAI v0.2.0 (双沙盒架构)");
            println!("真正的AI驱动浏览器 - 标准渲染 + AI学习");
        }
    }

    Ok(())
}

/// 双沙盒流水线：真正的AI学习 → 理解组件 → 重组生成
async fn learn_and_generate(url: &str, output_dir: &PathBuf, variant_count: usize) -> Result<()> {
    let start = Instant::now();

    log::info!("╔══════════════════════════════════════════════════════════════╗");
    log::info!("║  BrowerAI - 真正的AI学习：理解组件 + 重组生成                ║");
    log::info!("╚══════════════════════════════════════════════════════════════╝");
    log::info!("🎯 目标网站: {}", url);

    fs::create_dir_all(output_dir)?;

    // 使用双沙盒引擎处理网站
    log::info!("\n[双沙盒处理] 启动双沙盒引擎...");
    let engine = DualSandboxEngine::new()?;
    let result = engine.process_website(url).await?;
    
    // 额外：提取组件和理解JS，保存到本地
    log::info!("\n[保存学习结果] 将理解的内容保存到本地...");
    save_learning_results(&result, output_dir).await?;

    // 输出沙盒1结果
    log::info!("\n[沙盒1 - 标准渲染]");
    log::info!("   ✓ HTML: {} 字节", result.original.html.len());
    log::info!("   ✓ CSS 文件: {} 个", result.original.css_resources.len());
    log::info!("   ✓ JS 文件: {} 个", result.original.js_resources.len());

    // 输出沙盒2结果
    log::info!("\n[沙盒2 - AI 学习]");
    log::info!("   ✓ 网站意图: {:?}", result.learned.intent.primary_type);
    log::info!("   ✓ 颜色: {} 种", 
        result.learned.styles.colors.primary_colors.len() +
        result.learned.styles.colors.background_colors.len() +
        result.learned.styles.colors.text_colors.len()
    );
    log::info!("   ✓ 字体: {} 种", result.learned.styles.typography.font_families.len());
    log::info!("   ✓ 功能点: {} 个", result.learned.functions.user_functions.len());

    // 输出生成结果
    log::info!("\n[重组生成 - AI生成的新网站]");
    if let Some(ref generated) = result.generated {
        log::info!("   ✓ 生成HTML: {} 字节", generated.html.len());
        log::info!("   ✓ 生成CSS: {} 字节", generated.css.len());
        log::info!("   ✓ 生成JS: {} 字节", generated.js.len());
        log::info!("   ✓ 使用组件: {}", generated.metadata.components_used.join(", "));
        log::info!("   ✓ 实现功能: {}", generated.metadata.features_implemented.join(", "));
        
        // 保存AI生成的网站
        let generated_dir = output_dir.join("ai_generated");
        fs::create_dir_all(&generated_dir)?;
        fs::write(generated_dir.join("index.html"), &generated.html)?;
        fs::write(generated_dir.join("styles.css"), &generated.css)?;
        fs::write(generated_dir.join("script.js"), &generated.js)?;
        log::info!("   ✓ 已保存到: {}", generated_dir.display());
    }

    // 生成体验变体
    log::info!("\n[生成引擎 - 体验变体]");
    
    for (idx, variant) in result.variants.iter().enumerate().take(variant_count) {
        let variant_dir = output_dir.join(format!("variant_{}", idx + 1));
        fs::create_dir_all(&variant_dir)?;

        // 保存AI生成的变体（全新结构，不是注入样式）
        fs::write(variant_dir.join("index.html"), &variant.html)?;
        fs::write(variant_dir.join("styles.css"), &variant.css)?;
        fs::write(variant_dir.join("script.js"), &variant.js)?;

        log::info!(
            "   ✓ 变体 {}: {} - HTML={}字节, CSS={}字节, JS={}字节",
            idx + 1,
            variant.name,
            variant.html.len(),
            variant.css.len(),
            variant.js.len(),
        );
    }

    // 生成报告
    let report = serde_json::json!({
        "url": url,
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "dual_sandbox": {
            "html_bytes": result.original.html.len(),
            "css_files": result.original.css_resources.len(),
            "js_files": result.original.js_resources.len(),
            "colors_learned": result.learned.styles.colors.primary_colors.len(),
            "fonts_learned": result.learned.styles.typography.font_families.len(),
            "functions_learned": result.learned.functions.user_functions.len(),
        },
        "ai_generation": result.generated.as_ref().map(|g| serde_json::json!({
            "html_bytes": g.html.len(),
            "css_bytes": g.css.len(),
            "js_bytes": g.js.len(),
            "components_used": g.metadata.components_used,
            "features_implemented": g.metadata.features_implemented,
        })),
        "variants_generated": result.variants.len().min(variant_count),
    });
    fs::write(
        output_dir.join("dual_sandbox_report.json"),
        serde_json::to_string_pretty(&report)?,
    )?;

    let total = start.elapsed();
    log::info!(
        "\n✅ 双沙盒处理完成，总耗时 {:.2}s，变体 {} 个",
        total.as_secs_f64(),
        result.variants.len().min(variant_count),
    );
    log::info!("📦 输出目录: {}", output_dir.canonicalize()?.display());
    Ok(())
}

/// 将样式注入 HTML
#[allow(dead_code)]
fn inject_styles(html: &str, css: &str) -> String {
    let style_tag = format!("<style>\n{}\n</style>\n", css);
    
    if let Some(head_end) = html.find("</head>") {
        let mut result = html.to_string();
        result.insert_str(head_end, &style_tag);
        result
    } else if let Some(body_start) = html.find("<body") {
        let mut result = html.to_string();
        result.insert_str(body_start, &style_tag);
        result
    } else {
        format!("<!DOCTYPE html>\n<html>\n<head>\n{}</head>\n<body>\n{}\n</body>\n</html>", style_tag, html)
    }
}

/// 保存学习结果到本地
async fn save_learning_results(
    result: &browerai_dual_sandbox::ProcessedWebsite,
    output_dir: &Path,
) -> Result<()> {
    let learning_dir = output_dir.join("learning_results");
    fs::create_dir_all(&learning_dir)?;
    
    // 1. 保存原始资源
    let raw_dir = learning_dir.join("raw");
    fs::create_dir_all(&raw_dir)?;
    fs::write(raw_dir.join("original.html"), &result.original.html)?;
    
    // 保存CSS
    let css_dir = raw_dir.join("css");
    fs::create_dir_all(&css_dir)?;
    for (i, css) in result.original.css_resources.iter().enumerate() {
        fs::write(css_dir.join(format!("{}.css", i)), &css.content)?;
    }
    
    // 保存JS
    let js_dir = raw_dir.join("js");
    fs::create_dir_all(&js_dir)?;
    for (i, js) in result.original.js_resources.iter().enumerate() {
        fs::write(js_dir.join(format!("{}.js", i)), &js.content)?;
    }
    
    // 2. 提取并保存组件
    log::info!("   🔧 提取UI组件...");
    let component_extractor = ComponentExtractor::new();
    let all_css = result.original.css_resources.iter()
        .map(|c| c.content.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    let components = component_extractor.extract(&result.original.html, &all_css);
    
    let components_json = serde_json::json!({
        "summary": {
            "buttons": components.buttons.len(),
            "forms": components.forms.len(),
            "navigations": components.navigations.len(),
            "cards": components.cards.len(),
            "layouts": components.layouts.len(),
            "others": components.others.len(),
        },
        "buttons": components.buttons,
        "forms": components.forms,
        "navigations": components.navigations,
        "cards": components.cards,
        "layouts": components.layouts,
        "others": components.others,
    });
    fs::write(
        learning_dir.join("components.json"),
        serde_json::to_string_pretty(&components_json)?,
    )?;
    log::info!("   ✓ 组件已保存到: learning_results/components.json");
    
    // 3. 理解并保存JS意图
    log::info!("   📜 解析JS功能意图...");
    let js_understander = JsUnderstander::new();
    let all_js = result.original.js_resources.iter()
        .map(|j| j.content.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    let intents = js_understander.understand(&all_js);
    
    let intents_json = serde_json::json!({
        "summary": {
            "interactions": intents.interactions.len(),
            "data_flows": intents.data_flows.len(),
            "state_management": intents.state_management.len(),
            "api_intents": intents.api_intents.len(),
            "animations": intents.animations.len(),
        },
        "interactions": intents.interactions,
        "data_flows": intents.data_flows,
        "state_management": intents.state_management,
        "api_intents": intents.api_intents,
        "animations": intents.animations,
    });
    fs::write(
        learning_dir.join("intents.json"),
        serde_json::to_string_pretty(&intents_json)?,
    )?;
    log::info!("   ✓ 意图已保存到: learning_results/intents.json");
    
    // 4. 保存样式系统
    let styles_json = serde_json::json!({
        "colors": {
            "primary": result.learned.styles.colors.primary_colors,
            "secondary": result.learned.styles.colors.secondary_colors,
            "background": result.learned.styles.colors.background_colors,
            "text": result.learned.styles.colors.text_colors,
            "accent": result.learned.styles.colors.accent_colors,
        },
        "typography": {
            "fonts": result.learned.styles.typography.font_families,
            "sizes": result.learned.styles.typography.font_sizes,
        },
    });
    fs::write(
        learning_dir.join("styles.json"),
        serde_json::to_string_pretty(&styles_json)?,
    )?;
    log::info!("   ✓ 样式已保存到: learning_results/styles.json");
    
    // 5. 创建可视化报告
    let report_html = generate_learning_report(&components, &intents);
    fs::write(learning_dir.join("report.html"), report_html)?;
    log::info!("   ✓ 可视化报告: learning_results/report.html");
    
    log::info!("   📁 所有学习结果保存在: {}", learning_dir.display());
    Ok(())
}

/// 生成学习报告HTML
fn generate_learning_report(
    components: &browerai_dual_sandbox::ComponentLibrary,
    intents: &browerai_dual_sandbox::FunctionIntents,
) -> String {
    format!(r#"<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>AI学习结果报告</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; background: #f5f5f5; }}
        h1 {{ color: #333; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        .section {{ background: white; border-radius: 8px; padding: 20px; margin: 20px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .stat {{ display: inline-block; background: #3498db; color: white; padding: 10px 20px; border-radius: 20px; margin: 5px; }}
        .component {{ background: #f8f9fa; border-left: 4px solid #2ecc71; padding: 15px; margin: 10px 0; border-radius: 4px; }}
        .intent {{ background: #f8f9fa; border-left: 4px solid #e74c3c; padding: 15px; margin: 10px 0; border-radius: 4px; }}
        pre {{ background: #2c3e50; color: #ecf0f1; padding: 15px; border-radius: 4px; overflow-x: auto; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
    </style>
</head>
<body>
    <h1>🧠 AI学习结果报告</h1>
    
    <div class="section">
        <h2>📊 统计概览</h2>
        <div>
            <span class="stat">按钮组件: {}</span>
            <span class="stat">表单组件: {}</span>
            <span class="stat">导航组件: {}</span>
            <span class="stat">卡片组件: {}</span>
            <span class="stat">布局组件: {}</span>
        </div>
        <div style="margin-top: 15px;">
            <span class="stat">交互意图: {}</span>
            <span class="stat">数据流: {}</span>
            <span class="stat">状态管理: {}</span>
            <span class="stat">API意图: {}</span>
        </div>
    </div>
    
    <div class="grid">
        <div class="section">
            <h2>🔧 提取的组件</h2>
            <p>查看 <code>components.json</code> 获取完整数据</p>
        </div>
        
        <div class="section">
            <h2>📜 理解的功能意图</h2>
            <p>查看 <code>intents.json</code> 获取完整数据</p>
        </div>
    </div>
    
    <div class="section">
        <h2>📁 文件说明</h2>
        <ul>
            <li><code>raw/</code> - 原始HTML/CSS/JS资源</li>
            <li><code>components.json</code> - UI组件库（按钮、表单、导航等）</li>
            <li><code>intents.json</code> - 功能意图（交互、数据流、API等）</li>
            <li><code>styles.json</code> - 样式系统（颜色、字体等）</li>
            <li><code>report.html</code> - 本报告</li>
        </ul>
    </div>
    
    <div class="section">
        <h2>💡 如何使用</h2>
        <p>这些JSON文件包含了AI从网站学习到的所有知识：</p>
        <ol>
            <li><strong>components.json</strong> - 包含UI组件的结构、样式和行为</li>
            <li><strong>intents.json</strong> - 包含功能意图，可以重新实现为任何框架</li>
            <li><strong>styles.json</strong> - 包含颜色、字体等设计系统</li>
        </ol>
        <p>你可以基于这些数据：</p>
        <ul>
            <li>生成全新的网站（不同技术栈）</li>
            <li>创建设计系统文档</li>
            <li>分析网站架构模式</li>
            <li>迁移到新的前端框架</li>
        </ul>
    </div>
</body>
</html>"#,
        components.buttons.len(),
        components.forms.len(),
        components.navigations.len(),
        components.cards.len(),
        components.layouts.len(),
        intents.interactions.len(),
        intents.data_flows.len(),
        intents.state_management.len(),
        intents.api_intents.len(),
    )
}
