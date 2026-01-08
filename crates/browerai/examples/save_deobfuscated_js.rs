//! 反混淆JS并保存到文件
//!
//! 演示如何：
//! 1. 从URL获取混淆的JS
//! 2. 反混淆处理
//! 3. 保存到新文件
//!
//! 运行：cargo run --example save_deobfuscated_js

use anyhow::{Context, Result};
use browerai::learning::{DeobfuscationStrategy, JsDeobfuscator, WebsiteDeobfuscationVerifier};
use std::fs;
use std::path::Path;

fn main() -> Result<()> {
    env_logger::init();

    println!("🚀 BrowerAI - JavaScript 反混淆工具\n");

    // 示例1: 从字符串反混淆
    deobfuscate_from_string()?;

    // 示例2: 从URL获取并反混淆
    deobfuscate_from_url()?;

    // 示例3: 批量处理文件
    batch_deobfuscate()?;

    Ok(())
}

/// 示例1: 从字符串反混淆并保存
fn deobfuscate_from_string() -> Result<()> {
    println!("📝 示例1: 从字符串反混淆\n");

    let obfuscated_code = r#"
var _0x1a2b=['Hello','World','log'];
(function(_0x4c2d,_0x12eb){
    var _0x31c4=function(_0x2a1f){
        while(--_0x2a1f){
            _0x4c2d['push'](_0x4c2d['shift']());
        }
    };
    _0x31c4(++_0x12eb);
}(_0x1a2b,0x123));
var _0x4b5c=function(_0x4c2d,_0x12eb){
    _0x4c2d=_0x4c2d-0x0;
    var _0x31c4=_0x1a2b[_0x4c2d];
    return _0x31c4;
};
console[_0x4b5c('0x2')](_0x4b5c('0x0'),_0x4b5c('0x1'));
"#;

    // 创建反混淆器
    let deobfuscator = JsDeobfuscator::new();

    // 反混淆
    println!("⚙️  正在反混淆...");
    let result = deobfuscator
        .deobfuscate(obfuscated_code, DeobfuscationStrategy::Comprehensive)
        .context("反混淆失败")?;

    // 打印统计信息
    println!("✅ 反混淆完成!");
    println!("   原始代码: {} 字节", obfuscated_code.len());
    println!("   新代码:   {} 字节", result.code.len());
    println!(
        "   可读性提升: {:.2}% → {:.2}%",
        result.improvement.readability_before * 100.0,
        result.improvement.readability_after * 100.0
    );

    // 保存到文件
    let output_path = "output/deobfuscated_example1.js";
    save_js_to_file(&result.code, output_path)?;

    println!("💾 已保存到: {}\n", output_path);

    // 显示预览
    println!("📄 新代码预览 (前200字符):");
    println!("---");
    println!("{}", &result.code.chars().take(200).collect::<String>());
    println!("---\n");

    Ok(())
}

/// 示例2: 从真实URL获取并反混淆
fn deobfuscate_from_url() -> Result<()> {
    println!("🌐 示例2: 从真实网站获取并反混淆\n");

    // 小型库，速度快
    let url = "https://cdn.jsdelivr.net/npm/dayjs@1.11.10/dayjs.min.js";

    println!("📡 正在从 {} 下载...", url);

    let mut verifier = WebsiteDeobfuscationVerifier::new();
    let result = verifier
        .verify_website(url, None)
        .map_err(|e| anyhow::anyhow!("获取或反混淆失败: {}", e))?;

    println!("✅ 处理完成!");
    println!("   原始大小:     {} 字节", result.original_size);
    println!("   反混淆后:     {} 字节", result.deobfuscated_size);
    println!(
        "   可读性改进:   {:.2}%",
        result.readability_improvement * 100.0
    );
    println!("   处理时间:     {} 毫秒", result.processing_time_ms);
    println!("   检测到技术:   {:?}", result.obfuscation_techniques);

    // 保存原始和反混淆版本
    let base_name = "output/dayjs";
    save_js_to_file(
        &result.original_code,
        &format!("{}_original.min.js", base_name),
    )?;
    save_js_to_file(
        &result.deobfuscated_code,
        &format!("{}_deobfuscated.js", base_name),
    )?;

    println!("💾 已保存:");
    println!("   - {}_original.min.js (原始混淆版)", base_name);
    println!("   - {}_deobfuscated.js (反混淆版)\n", base_name);

    Ok(())
}

/// 示例3: 批量处理多个文件
fn batch_deobfuscate() -> Result<()> {
    println!("📦 示例3: 批量处理\n");

    let test_cases = vec![
        (
            "简单函数",
            r#"
function a(b,c){return b+c;}
var d=a(1,2);console.log(d);
"#,
        ),
        (
            "字符串混淆",
            r#"
var _0x=['test','message'];
function log(){console.log(_0x[0],_0x[1]);}
log();
"#,
        ),
        (
            "表达式混淆",
            r#"
var x=!![];var y=![];
if(x&&!y){console.log('true');}
"#,
        ),
    ];

    let deobfuscator = JsDeobfuscator::new();

    for (i, (name, code)) in test_cases.iter().enumerate() {
        println!("  [{}/{}] 处理: {}", i + 1, test_cases.len(), name);

        let result = deobfuscator.deobfuscate(code, DeobfuscationStrategy::Comprehensive)?;

        let output_path = format!("output/batch_{}.js", i + 1);
        save_js_to_file(&result.code, &output_path)?;

        println!(
            "      ✓ {} 字节 → {} 字节, 已保存到 {}",
            code.len(),
            result.code.len(),
            output_path
        );
    }

    println!("\n✅ 批量处理完成!\n");

    Ok(())
}

/// 保存JS代码到文件
fn save_js_to_file(code: &str, path: &str) -> Result<()> {
    // 确保输出目录存在
    if let Some(parent) = Path::new(path).parent() {
        fs::create_dir_all(parent).context(format!("创建目录失败: {:?}", parent))?;
    }

    // 添加文件头注释
    let header = format!(
        "// 由 BrowerAI 反混淆生成\n// 生成时间: {}\n// 原始路径: {}\n\n",
        chrono::Local::now().format("%Y-%m-%d %H:%M:%S"),
        path
    );

    let content = format!("{}{}", header, code);

    // 写入文件
    fs::write(path, content).context(format!("写入文件失败: {}", path))?;

    Ok(())
}
