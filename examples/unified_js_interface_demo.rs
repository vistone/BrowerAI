//! 统一 JS 接口演示
//! 
//! 展示如何使用 UnifiedJsInterface 来处理 JS 执行和分析，
//! 结合 V8、SWC、Boa 三个引擎的优势。
//! 
//! 运行：cargo run --example unified_js_interface_demo --features ai

use browerai::prelude::*;

fn main() {
    env_logger::builder()
        .filter_level(log::LevelFilter::Debug)
        .try_init()
        .ok();

    println!("=== 统一 JS 接口演示 ===\n");

    // 创建统一接口
    let mut unified_interface = UnifiedJsInterface::new();

    // 示例 1: 执行简单 JS 代码
    println!("示例 1: 执行 JS 代码");
    let js_code = "console.log('Hello from Unified Interface'); 2 + 2";
    match unified_interface.execute_for_render(js_code) {
        Ok(result) => {
            println!("✓ 执行成功");
            println!("  输出: {}", result.output);
            println!("  引擎: {}", result.engine);
            println!("  成功: {}", result.success);
        }
        Err(e) => println!("✗ 执行失败: {}", e),
    }
    println!();

    // 示例 2: 解析 ES6 模块
    println!("示例 2: 解析 ES6 模块");
    let module_code = r#"
        import { Component } from 'react';
        export class App extends Component {
            render() {
                return <div>Hello</div>;
            }
        }
    "#;
    match unified_interface.parse_for_analysis(module_code) {
        Ok(result) => {
            println!("✓ 解析成功");
            println!("  有效: {}", result.is_valid);
            println!("  是模块: {}", result.is_module);
            println!("  是 TS/JSX: {}", result.is_typescript_jsx);
            println!("  语句数: {}", result.statement_count);
            println!("  引擎: {}", result.engine);
        }
        Err(e) => println!("✗ 解析失败: {}", e),
    }
    println!();

    // 示例 3: 快速验证 JS 语法
    println!("示例 3: 快速验证 JS 语法");
    let test_cases = vec![
        ("valid", "const x = 1; x + 2;"),
        ("invalid", "const x = 1 x + 2;"),
        ("module", "import x from 'y'; export {x};"),
    ];

    for (name, code) in test_cases {
        match unified_interface.quick_validate(code) {
            Ok(is_valid) => {
                println!("✓ {}: {}", name, if is_valid { "有效" } else { "无效" });
            }
            Err(e) => println!("✗ {}: {}", name, e),
        }
    }
    println!();

    // 示例 4: 展示引擎多样性
    println!("示例 4: 不同类型代码的引擎选择");
    let code_samples = vec![
        (
            "同步脚本",
            "const obj = { x: 1 }; console.log(obj.x);",
        ),
        (
            "TypeScript",
            "interface User { name: string; } const u: User = { name: 'Alice' };",
        ),
        (
            "JSX",
            "const element = <div>Hello</div>;",
        ),
    ];

    for (desc, code) in code_samples {
        match unified_interface.parse_for_analysis(code) {
            Ok(result) => {
                println!(
                    "✓ {}: 使用 {} 引擎",
                    desc, result.engine
                );
            }
            Err(e) => println!("✗ {}: {}", desc, e),
        }
    }

    println!("\n=== 演示完成 ===");
}
