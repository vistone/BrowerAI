//! 双沙盒处理网站示例

use browerai_dual_sandbox::DualSandboxEngine;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    env_logger::init();

    // 创建双沙盒引擎
    let engine = DualSandboxEngine::new()?;

    // 处理网站
    let url = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "https://www.rust-lang.org".to_string());

    println!("处理网站: {}", url);
    println!();

    let result = engine.process_website(&url).await?;

    // 输出结果
    println!("\n=== 处理结果 ===");
    println!();

    // 原始渲染信息
    println!("[沙盒1 - 标准渲染]");
    println!("  HTML 大小: {} 字节", result.original.html.len());
    println!("  CSS 文件: {} 个", result.original.css_resources.len());
    println!("  JS 文件: {} 个", result.original.js_resources.len());
    println!("  DOM 节点: {} 个", result.original.dom_tree.node_count());
    println!();

    // 学习结果
    println!("[沙盒2 - AI 学习]");
    println!("  网站意图: {:?}", result.learned.intent.primary_type);
    println!("  置信度: {:.0}%", result.learned.intent.confidence * 100.0);
    println!("  核心功能: {:?}", result.learned.intent.core_features);
    println!(
        "  颜色数量: {} 种",
        result.learned.styles.colors.primary_colors.len()
            + result.learned.styles.colors.background_colors.len()
            + result.learned.styles.colors.text_colors.len()
    );
    println!(
        "  字体数量: {} 种",
        result.learned.styles.typography.font_families.len()
    );
    println!(
        "  功能点: {} 个",
        result.learned.functions.user_functions.len()
    );
    println!();

    // 生成的变体
    println!("[生成引擎 - 体验变体]");
    for (i, variant) in result.variants.iter().enumerate() {
        println!("  变体 {}: {}", i + 1, variant.name);
        println!(
            "    主色: {:?}",
            variant.styles.colors.primary_colors.first().map(|c| &c.hex)
        );
        println!(
            "    字体: {:?}",
            variant
                .styles
                .typography
                .font_families
                .first()
                .map(|f| &f.name)
        );
        println!("    功能映射: {} 个", variant.function_mappings.len());
    }

    Ok(())
}
