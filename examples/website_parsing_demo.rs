//! BrowerAI 网站解析演示
//! 
//! 演示如何使用 BrowerAI 解析真实网站
//! 
//! 用法: ./website_parsing_demo <URL>
//! 示例: ./website_parsing_demo https://www.rust-lang.org

use anyhow::Result;
use browerai::prelude::*;
use browerai::network::HttpClient;
use std::env;

fn main() -> Result<()> {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info")
    ).init();

    // 获取命令行参数
    let args: Vec<String> = env::args().collect();
    let url = if args.len() > 1 {
        &args[1]
    } else {
        "https://www.rust-lang.org"
    };

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  BrowerAI 网站解析演示                                       ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    println!("\n🎯 目标网站: {}", url);

    // 1. 创建HTTP客户端并获取内容
    println!("\n[1/4] 获取网站内容...");
    let client = HttpClient::new();
    let response = client.get(url)?;
    let html = response.text()?;
    println!("   ✓ 获取成功: {} bytes", html.len());
    println!("   ✓ HTTP状态: 200 OK");

    // 2. 解析HTML
    println!("\n[2/4] 解析HTML...");
    let html_parser = HtmlParser::new();
    let doc = html_parser.parse(&html)?;
    println!("   ✓ HTML解析成功");
    println!("   ✓ 文本节点数: {}", doc.text_node_count());

    // 3. 提取脚本
    println!("\n[3/4] 提取脚本...");
    let scripts = html_parser.extract_scripts(&doc);
    println!("   ✓ 发现 {} 个脚本", scripts.len());
    for (i, script) in scripts.iter().take(3).enumerate() {
        let preview = if script.len() > 80 {
            format!("{}...", &script[..80])
        } else {
            script.clone()
        };
        println!("   [{}] {}", i + 1, preview.replace('\n', " "));
    }
    if scripts.len() > 3 {
        println!("   ... 还有 {} 个脚本", scripts.len() - 3);
    }

    // 4. 提取资源
    println!("\n[4/4] 提取资源...");
    let resources = html_parser.extract_resources(&doc);
    println!("   ✓ 发现 {} 个资源", resources.len());
    for (i, resource) in resources.iter().take(10).enumerate() {
        println!("   [{}] {}", i + 1, resource);
    }
    if resources.len() > 10 {
        println!("   ... 还有 {} 个资源", resources.len() - 10);
    }

    println!("\n✅ 网站解析完成!");
    println!("\n📊 总结:");
    println!("  ├─ 目标URL: {}", url);
    println!("  ├─ HTML大小: {} bytes", html.len());
    println!("  ├─ 文本节点: {}", doc.text_node_count());
    println!("  ├─ 脚本数量: {}", scripts.len());
    println!("  └─ 资源数量: {}", resources.len());

    Ok(())
}
