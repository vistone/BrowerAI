//! 语义化反混淆示例
//!
//! 演示如何基于函数行为和语义推断有意义的变量名
//!
//! 运行：cargo run --example semantic_deobfuscation_demo

use anyhow::Result;
use browerai::learning::{DeobfuscationStrategy, JsDeobfuscator, SemanticRenamer};

fn main() -> Result<()> {
    env_logger::init();

    println!("🧠 BrowerAI - 语义化反混淆演示\n");
    println!("基于函数行为智能推断变量名\n");

    // 示例1: Day.js 风格的时间常量
    demo_time_constants()?;

    // 示例2: 函数行为推断
    demo_function_behavior()?;

    // 示例3: 完整的反混淆流程
    demo_full_pipeline()?;

    Ok(())
}

/// 示例1: 时间常量的语义推断
fn demo_time_constants() -> Result<()> {
    println!("📊 示例1: 时间常量语义推断\n");

    let obfuscated_code = r#"
var var0=1e3;
var var1=6e4;
var var2=36e5;
var var3="millisecond";
var var4="second";
var var5="minute";
var var6="hour";
var var7="Invalid Date";

function calculate(time) {
    return time * var0;
}
"#;

    println!("原始混淆代码:");
    println!("{}", obfuscated_code);

    // 第一步：基础反混淆
    let deobfuscator = JsDeobfuscator::new();
    let basic_result =
        deobfuscator.deobfuscate(obfuscated_code, DeobfuscationStrategy::Comprehensive)?;

    // 第二步：语义重命名
    let mut semantic_renamer = SemanticRenamer::new();
    let semantic_result = semantic_renamer.analyze_and_rename(&basic_result.code);

    println!("\n✨ 语义化重命名后:");
    println!("{}", semantic_result);

    println!("\n📋 重命名映射表:");
    for (old_name, new_name) in semantic_renamer.get_rename_map() {
        println!("  {} → {}", old_name, new_name);
    }

    println!("\n✅ 效果对比:");
    println!("  var0=1e3 → MILLISECONDS_PER_SECOND=1e3");
    println!("  var1=6e4 → MILLISECONDS_PER_MINUTE=6e4");
    println!("  var3=\"millisecond\" → UNIT_MILLISECOND=\"millisecond\"");
    println!();

    Ok(())
}

/// 示例2: 函数行为推断
fn demo_function_behavior() -> Result<()> {
    println!("🔍 示例2: 函数行为推断\n");

    let obfuscated_code = r#"
function var10(date) {
    return date.format("YYYY-MM-DD");
}

function var11(str) {
    return new Date(str);
}

function var12(obj) {
    return obj.clone();
}

function var13(value) {
    if (!value) return false;
    return validate(value);
}
"#;

    println!("原始代码:");
    println!("{}", obfuscated_code);

    let mut semantic_renamer = SemanticRenamer::new();
    let result = semantic_renamer.analyze_and_rename(obfuscated_code);

    println!("\n✨ 推断后:");
    println!("{}", result);

    println!("\n💡 推断逻辑:");
    println!("  包含 'format' → formatter函数");
    println!("  包含 'new Date' → dateCreator函数");
    println!("  包含 'clone' → cloner函数");
    println!("  包含 'validate' → validator函数");
    println!();

    Ok(())
}

/// 示例3: 完整反混淆流程
fn demo_full_pipeline() -> Result<()> {
    println!("🚀 示例3: 完整的语义化反混淆流程\n");

    // 模拟真实的混淆代码
    let heavily_obfuscated = r#"
var _0x=['time','format','parse'];
var var0=1e3;
var var1=6e4;
var var2="millisecond";

function var10(t) {
    return t.format(_0x[1]);
}

function var11(t) {
    var var20 = t * var0;
    return new Date(var20);
}

var var30 = {
    unit: var2,
    convert: function(val) {
        return val * var0;
    }
};
"#;

    println!("重度混淆代码:");
    println!("{}", heavily_obfuscated);

    // 步骤1: 基础反混淆
    println!("\n⚙️  步骤1: 基础反混淆（字符串数组、表达式简化）");
    let deobfuscator = JsDeobfuscator::new();
    let step1 =
        deobfuscator.deobfuscate(heavily_obfuscated, DeobfuscationStrategy::Comprehensive)?;

    println!(
        "可读性: {:.1}% → {:.1}%",
        step1.improvement.readability_before * 100.0,
        step1.improvement.readability_after * 100.0
    );

    // 步骤2: 语义重命名
    println!("\n🧠 步骤2: 语义化重命名（基于行为推断）");
    let mut semantic_renamer = SemanticRenamer::new();
    let step2 = semantic_renamer.analyze_and_rename(&step1.code);

    println!("\n✨ 最终结果:");
    println!("{}", step2);

    println!("\n📊 改进统计:");
    println!("  应用的转换步骤: {:?}", step1.steps);
    println!(
        "  语义重命名数量: {}",
        semantic_renamer.get_rename_map().len()
    );
    println!(
        "  总体可读性提升: {:.1}%",
        (step1.improvement.readability_after - step1.improvement.readability_before) * 100.0
    );

    println!("\n🎯 关键改进:");
    for (old_name, new_name) in semantic_renamer.get_rename_map() {
        println!("  {} → {}", old_name, new_name);
    }

    println!("\n💡 小结:");
    println!("  ✅ 变量名从 var0/var1 变为 MILLISECONDS_PER_SECOND");
    println!("  ✅ 函数名从 var10/var11 变为 formatter/dateCreator");
    println!("  ✅ 字符串常量获得语义化名称");
    println!("  ✅ 基于上下文和行为的智能推断");
    println!();

    Ok(())
}
