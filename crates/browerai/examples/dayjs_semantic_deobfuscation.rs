//! 对真实 Day.js 库进行语义化反混淆
//!
//! 运行：cargo run --example dayjs_semantic_deobfuscation

use anyhow::Result;
use browerai::learning::{SemanticRenamer, WebsiteDeobfuscationVerifier};
use std::fs;

fn main() -> Result<()> {
    env_logger::init();

    println!("🚀 Day.js 语义化反混淆\n");

    let url = "https://cdn.jsdelivr.net/npm/dayjs@1.11.10/dayjs.min.js";

    println!("📡 正在从 {} 下载...", url);

    // 步骤1: 获取并基础反混淆
    let mut verifier = WebsiteDeobfuscationVerifier::new();
    let result = verifier
        .verify_website(url, None)
        .map_err(|e| anyhow::anyhow!("获取失败: {}", e))?;

    println!("✅ 下载完成: {} 字节\n", result.original_size);

    // 步骤2: 语义化重命名
    println!("🧠 正在进行语义分析和重命名...");
    let mut semantic_renamer = SemanticRenamer::new();
    let semantic_code = semantic_renamer.analyze_and_rename(&result.deobfuscated_code);

    println!("✅ 语义分析完成\n");

    // 保存所有版本
    fs::create_dir_all("output/dayjs_analysis")?;

    // 1. 原始混淆版
    fs::write(
        "output/dayjs_analysis/1_original.min.js",
        &result.original_code,
    )?;

    // 2. 基础反混淆版
    fs::write(
        "output/dayjs_analysis/2_basic_deobfuscated.js",
        &result.deobfuscated_code,
    )?;

    // 3. 语义化版本
    fs::write("output/dayjs_analysis/3_semantic.js", &semantic_code)?;

    // 4. 重命名映射表
    let mut rename_report = String::from("# Day.js 语义重命名报告\n\n");
    rename_report.push_str(&format!("## 统计信息\n\n"));
    rename_report.push_str(&format!("- 原始大小: {} 字节\n", result.original_size));
    rename_report.push_str(&format!(
        "- 基础反混淆: {} 字节\n",
        result.deobfuscated_size
    ));
    rename_report.push_str(&format!("- 语义化版本: {} 字节\n", semantic_code.len()));
    rename_report.push_str(&format!(
        "- 语义重命名数量: {} 个\n",
        semantic_renamer.get_rename_map().len()
    ));
    rename_report.push_str(&format!("- 处理时间: {} ms\n\n", result.processing_time_ms));

    rename_report.push_str("## 重命名映射表\n\n");
    rename_report.push_str("| 原变量名 | 语义化名称 | 说明 |\n");
    rename_report.push_str("|----------|------------|------|\n");

    let mut renames: Vec<_> = semantic_renamer.get_rename_map().iter().collect();
    renames.sort_by_key(|(k, _)| k.to_string());

    for (old_name, new_name) in &renames {
        let description = match new_name.as_str() {
            name if name.contains("MILLISECONDS") => "时间常量",
            name if name.contains("UNIT_") => "单位常量",
            name if name.contains("INVALID") => "错误信息",
            _ => "其他",
        };
        rename_report.push_str(&format!(
            "| `{}` | `{}` | {} |\n",
            old_name, new_name, description
        ));
    }

    fs::write("output/dayjs_analysis/4_rename_report.md", &rename_report)?;

    // 打印结果
    println!("📊 处理完成！文件已保存到 output/dayjs_analysis/\n");
    println!("文件列表:");
    println!(
        "  1_original.min.js        - 原始混淆版 ({} 字节)",
        result.original_size
    );
    println!(
        "  2_basic_deobfuscated.js  - 基础反混淆 ({} 字节)",
        result.deobfuscated_size
    );
    println!(
        "  3_semantic.js            - 语义化版本 ({} 字节)",
        semantic_code.len()
    );
    println!("  4_rename_report.md       - 重命名报告\n");

    println!("🎯 语义化改进:");
    println!(
        "  重命名变量: {} 个",
        semantic_renamer.get_rename_map().len()
    );

    println!("\n前5个重命名示例:");
    for (i, (old_name, new_name)) in renames.iter().take(5).enumerate() {
        println!("  {}. {} → {}", i + 1, old_name, new_name);
    }

    if renames.len() > 5 {
        println!("  ... 还有 {} 个重命名", renames.len() - 5);
    }

    println!("\n💡 使用建议:");
    println!("  1. 查看重命名报告了解所有变更");
    println!("  2. 对比 1_original 和 3_semantic 看整体效果");
    println!("  3. 语义化名称便于代码审计和理解");

    Ok(())
}
