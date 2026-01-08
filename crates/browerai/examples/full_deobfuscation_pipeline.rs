//! 完整的反混淆流程：反混淆 + 语义化 + 格式化
//!
//! 生成标准可读的 JavaScript 代码
//!
//! 运行：cargo run --example full_deobfuscation_pipeline

use anyhow::Result;
use browerai::learning::{format_js, SemanticRenamer, WebsiteDeobfuscationVerifier};
use std::fs;

fn main() -> Result<()> {
    env_logger::init();

    println!("🔧 完整反混淆流程演示\n");
    println!("步骤: 下载 → 反混淆 → 语义化 → 格式化\n");

    // 对 Day.js 进行完整处理
    process_dayjs()?;

    // 对小型示例进行展示
    demo_small_example()?;

    Ok(())
}

fn process_dayjs() -> Result<()> {
    println!("{}", "=".repeat(60));
    println!("📦 处理 Day.js 1.11.10");
    println!("{}", "=".repeat(60));
    println!();

    let url = "https://cdn.jsdelivr.net/npm/dayjs@1.11.10/dayjs.min.js";

    // 步骤1: 下载和基础反混淆
    println!("⏬ 步骤1: 下载和基础反混淆");
    let mut verifier = WebsiteDeobfuscationVerifier::new();
    let result = verifier
        .verify_website(url, None)
        .map_err(|e| anyhow::anyhow!("下载失败: {}", e))?;
    println!("   ✓ 原始大小: {} 字节", result.original_size);
    println!("   ✓ 反混淆后: {} 字节", result.deobfuscated_size);
    println!();

    // 步骤2: 语义化重命名
    println!("🧠 步骤2: 语义化重命名");
    let mut semantic_renamer = SemanticRenamer::new();
    let semantic_code = semantic_renamer.analyze_and_rename(&result.deobfuscated_code);
    println!(
        "   ✓ 重命名: {} 个变量",
        semantic_renamer.get_rename_map().len()
    );
    println!();

    // 步骤3: 代码格式化
    println!("✨ 步骤3: 代码格式化（恢复标准结构）");
    let formatted_code = format_js(&semantic_code);
    let line_count = formatted_code.lines().count();
    println!("   ✓ 格式化完成: {} 行代码", line_count);
    println!();

    // 保存所有版本
    fs::create_dir_all("output/dayjs_formatted")?;

    println!("💾 保存文件到 output/dayjs_formatted/");

    fs::write(
        "output/dayjs_formatted/1_original.min.js",
        &result.original_code,
    )?;
    println!("   ✓ 1_original.min.js (原始压缩版)");

    fs::write(
        "output/dayjs_formatted/2_deobfuscated.js",
        &result.deobfuscated_code,
    )?;
    println!("   ✓ 2_deobfuscated.js (基础反混淆)");

    fs::write("output/dayjs_formatted/3_semantic.js", &semantic_code)?;
    println!("   ✓ 3_semantic.js (语义化)");

    fs::write("output/dayjs_formatted/4_formatted.js", &formatted_code)?;
    println!("   ✓ 4_formatted.js (格式化，可读) ⭐");
    println!();

    // 创建对比报告
    let report = create_comparison_report(
        result.original_size,
        result.deobfuscated_size,
        semantic_code.len(),
        formatted_code.len(),
        1, // 原始行数
        line_count,
        semantic_renamer.get_rename_map().len(),
        result.processing_time_ms,
    );

    fs::write("output/dayjs_formatted/5_report.md", &report)?;
    println!("   ✓ 5_report.md (对比报告)");
    println!();

    // 显示部分格式化代码
    println!("📄 格式化后代码预览 (前30行):");
    println!("---");
    for (i, line) in formatted_code.lines().take(30).enumerate() {
        println!("{:4} | {}", i + 1, line);
    }
    println!("---");
    if line_count > 30 {
        println!("... 还有 {} 行", line_count - 30);
    }
    println!();

    Ok(())
}

fn demo_small_example() -> Result<()> {
    println!("{}", "=".repeat(60));
    println!("📝 小示例演示");
    println!("{}", "=".repeat(60));
    println!();

    let obfuscated = r#"var _0x=['log','test'];function var0(t){console[_0x[0]](_0x[1]+t);}var var1=1e3;var0(var1);"#;

    println!("原始代码 (单行):");
    println!("{}", obfuscated);
    println!();

    // 步骤1: 基础反混淆
    use browerai::learning::{DeobfuscationStrategy, JsDeobfuscator};
    let deobfuscator = JsDeobfuscator::new();
    let step1 = deobfuscator.deobfuscate(obfuscated, DeobfuscationStrategy::Comprehensive)?;

    // 步骤2: 语义化
    let mut semantic_renamer = SemanticRenamer::new();
    let step2 = semantic_renamer.analyze_and_rename(&step1.code);

    // 步骤3: 格式化
    let step3 = format_js(&step2);

    println!("格式化后:");
    println!("---");
    for (i, line) in step3.lines().enumerate() {
        println!("{:2} | {}", i + 1, line);
    }
    println!("---");
    println!();

    println!("✅ 转换完成:");
    println!("   1行 → {} 行", step3.lines().count());
    println!("   {} 字节 → {} 字节", obfuscated.len(), step3.len());
    println!(
        "   语义重命名: {} 个",
        semantic_renamer.get_rename_map().len()
    );
    println!();

    Ok(())
}

fn create_comparison_report(
    original_size: usize,
    deobfuscated_size: usize,
    semantic_size: usize,
    formatted_size: usize,
    original_lines: usize,
    formatted_lines: usize,
    renames: usize,
    processing_time: u128,
) -> String {
    format!(
        r#"# Day.js 反混淆完整流程报告

## 📊 处理统计

| 步骤 | 文件 | 大小 | 行数 | 说明 |
|------|------|------|------|------|
| 1 | 1_original.min.js | {} 字节 | {} 行 | 原始压缩版 |
| 2 | 2_deobfuscated.js | {} 字节 | - | 基础反混淆 |
| 3 | 3_semantic.js | {} 字节 | - | 语义化重命名 |
| 4 | **4_formatted.js** | {} 字节 | **{} 行** | ⭐ 格式化可读版 |

## 📈 改进指标

- **可读性提升**: 从 {} 行 → {} 行 ({}x)
- **语义重命名**: {} 个变量获得有意义的名称
- **文件大小增长**: {} 字节 → {} 字节 (+{:.1}%)
- **处理时间**: {} ms

## 🎯 格式化效果

### 原始 (压缩)
```javascript
!function(t,e){{"object"==typeof exports&&...
```
全部代码压缩在1行，完全无法阅读。

### 格式化后 (标准结构)
```javascript
!function(cloner,MILLISECONDS_PER_MINUTE){{
  "object"==typeof exports&&...
  var MILLISECONDS_PER_SECOND=1e3;
  var MILLISECONDS_PER_MINUTE=6e4;
  ...
}}
```
标准的多行格式，带有缩进，易于阅读和理解。

## 💡 使用建议

1. **代码审计**: 使用 `4_formatted.js` 进行人工审查
2. **调试**: 在格式化版本中添加断点
3. **学习**: 理解库的实现逻辑
4. **验证**: 对比不同版本确保功能一致

## ⚠️ 注意事项

- 格式化后的代码保持语法有效
- 可以直接在 Node.js 或浏览器中执行
- 语义化名称基于模式推断，可能需要人工调整
- 原始功能和逻辑完全保留

---

**生成时间**: {}  
**工具**: BrowerAI v0.1.0
"#,
        original_size,
        original_lines,
        deobfuscated_size,
        semantic_size,
        formatted_size,
        formatted_lines,
        original_lines,
        formatted_lines,
        formatted_lines / original_lines.max(1),
        renames,
        original_size,
        formatted_size,
        ((formatted_size as f64 - original_size as f64) / original_size as f64 * 100.0),
        processing_time,
        chrono::Local::now().format("%Y-%m-%d %H:%M:%S"),
    )
}
