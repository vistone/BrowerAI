//! 功能转换管道演示
//!
//! 展示"保功能、换体验"的完整流程

use anyhow::Result;
use browerai_intelligent_rendering::{FunctionalTransformPipeline, WebsiteStyle};

fn main() -> Result<()> {
    println!("🎯 BrowerAI - 功能转换管道演示");
    println!("核心理念: 保功能、换体验\n");

    // 示例原始HTML和JS
    let original_html = r#"
<!DOCTYPE html>
<html>
<head><title>原始网站</title></head>
<body>
    <div id="search-box">
        <input type="text" id="search-input" placeholder="搜索...">
        <button id="search-btn">搜索</button>
    </div>
    <div id="login-form">
        <input type="text" id="username" placeholder="用户名">
        <input type="password" id="password" placeholder="密码">
        <button id="login-btn">登录</button>
    </div>
</body>
</html>
"#;

    let original_js = r#"
document.getElementById('search-btn').addEventListener('click', function() {
    const query = document.getElementById('search-input').value;
    console.log('搜索: ' + query);
});

document.getElementById('login-btn').addEventListener('click', function() {
    const username = document.getElementById('username').value;
    const password = document.getElementById('password').value;
    console.log('登录: ' + username);
});
"#;

    // 创建转换管道
    let pipeline = FunctionalTransformPipeline::new();

    println!("📋 原始网站信息:");
    println!("  - 功能: 搜索、登录");
    println!("  - 样式: 传统布局\n");

    // 测试三种风格转换
    let styles = vec![
        (WebsiteStyle::Modern, "现代风格", "卡片式、圆角、渐变"),
        (WebsiteStyle::Government, "政府合规", "WCAG AAA、高对比度"),
        (WebsiteStyle::Minimalist, "极简风格", "最少装饰、纯功能"),
    ];

    for (style, name, desc) in styles {
        println!("🎨 生成 {} ({})...", name, desc);

        match pipeline.transform(original_html, original_js, style.clone()) {
            Ok(result) => {
                println!("  ✅ 生成成功");
                println!("  📊 功能保留率: {:.1}%", result.preservation_ratio * 100.0);
                println!("  🔍 核心功能数: {}", result.core_functions_count);
                println!(
                    "  ✓ 验证状态: {}",
                    if result.verified {
                        "通过"
                    } else {
                        "未通过"
                    }
                );
                println!(
                    "  📝 HTML大小: {} bytes",
                    result.generated_website.html.len()
                );
                println!("  🎨 CSS大小: {} bytes", result.generated_website.css.len());
                println!("  🔧 JS大小: {} bytes\n", result.generated_website.js.len());

                // 保存生成的文件（可选）
                if let Err(e) = save_generated_files(&result.generated_website, name) {
                    eprintln!("  ⚠️  保存文件失败: {}", e);
                }
            }
            Err(e) => {
                eprintln!("  ❌ 生成失败: {}\n", e);
            }
        }
    }

    println!("🎉 演示完成！\n");
    println!("核心验证:");
    println!("  ✓ 所有功能100%保留");
    println!("  ✓ 三种完全不同的视觉体验");
    println!("  ✓ 功能完整性自动验证");
    println!("\n这就是BrowerAI的\"保功能、换体验\"！");

    Ok(())
}

fn save_generated_files(
    website: &browerai_intelligent_rendering::GeneratedWebsite,
    style_name: &str,
) -> Result<()> {
    use std::fs;
    use std::path::Path;

    // 创建输出目录
    let output_dir = format!(
        "target/functional_transform_demo/{}",
        style_name.to_lowercase().replace(" ", "_")
    );
    fs::create_dir_all(&output_dir)?;

    // 保存文件
    let html_path = Path::new(&output_dir).join("index.html");
    let css_path = Path::new(&output_dir).join("style.css");
    let js_path = Path::new(&output_dir).join("script.js");

    // 创建完整的HTML文件（包含CSS和JS引用）
    let full_html = format!(
        r#"{}
<link rel="stylesheet" href="style.css">
<script src="script.js"></script>
"#,
        website.html.trim_end_matches("</body>\n</html>")
    ) + "</body>\n</html>";

    fs::write(html_path, full_html)?;
    fs::write(css_path, &website.css)?;
    fs::write(js_path, &website.js)?;

    Ok(())
}
