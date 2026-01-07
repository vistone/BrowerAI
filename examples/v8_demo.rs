//! Example: Using V8 JavaScript Engine
//! 
//! This example demonstrates how to use the V8 JavaScript engine
//! for maximum performance and ES2024+ compatibility.
//! 
//! To run: cargo run --example v8_demo --features v8

#![cfg(feature = "v8")]

use anyhow::Result;
use browerai::js_v8::V8JsParser;

fn main() -> Result<()> {
    env_logger::init();

    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║          BrowerAI - V8 JavaScript Engine Demo                ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    // Create V8 parser
    println!("🚀 Initializing V8 JavaScript engine...");
    let mut parser = V8JsParser::new()?;
    println!("✅ V8 initialized successfully!\n");

    // Example 1: Parse modern JavaScript
    println!("=== Example 1: Parse Modern JavaScript ===");
    let modern_js = r#"
        const greet = (name) => `Hello, ${name}!`;
        const result = greet("BrowerAI");
    "#;
    
    let ast = parser.parse(modern_js)?;
    println!("✅ Parsed {} bytes of JavaScript", ast.source_length);
    println!("   Valid: {}, Compiled: {}\n", ast.is_valid, ast.compiled);

    // Example 2: Execute JavaScript
    println!("=== Example 2: Execute JavaScript ===");
    let result = parser.execute("2 + 2 * 10")?;
    println!("   2 + 2 * 10 = {}\n", result);

    // Example 3: ES2024 Features
    println!("=== Example 3: ES2024+ Features ===");
    
    // Async/await
    let async_js = r#"
        async function fetchData() {
            const data = await Promise.resolve({name: "BrowerAI", version: "1.0"});
            return data;
        }
    "#;
    let ast = parser.parse(async_js)?;
    println!("✅ Async/await syntax parsed successfully");

    // Optional chaining
    let optional_chain_js = "const value = obj?.nested?.property ?? 'default';";
    parser.parse(optional_chain_js)?;
    println!("✅ Optional chaining parsed successfully");

    // Template literals
    let template_js = "const msg = `Value: ${x}, Count: ${y}`;";
    parser.parse(template_js)?;
    println!("✅ Template literals parsed successfully\n");

    // Example 4: Class syntax
    println!("=== Example 4: Class Syntax ===");
    let class_js = r#"
        class Calculator {
            constructor() {
                this.result = 0;
            }
            
            add(x, y) {
                return x + y;
            }
            
            static version() {
                return "1.0.0";
            }
        }
    "#;
    parser.parse(class_js)?;
    println!("✅ ES6 class syntax parsed successfully\n");

    // Example 5: Arrow functions and destructuring
    println!("=== Example 5: Arrow Functions & Destructuring ===");
    let result = parser.execute("((x, y) => x * y)(6, 7)")?;
    println!("   (6 * 7) via arrow function = {}", result);
    
    let destructure_js = "const {a, b} = {a: 1, b: 2}; const [x, y] = [3, 4];";
    parser.parse(destructure_js)?;
    println!("✅ Destructuring parsed successfully\n");

    // Example 6: Performance comparison info
    println!("=== V8 vs Boa Comparison ===");
    println!("V8 Advantages:");
    println!("  ✓ Full ES2024+ support");
    println!("  ✓ Maximum runtime performance");
    println!("  ✓ Industry-standard compatibility");
    println!("  ✓ JIT compilation for hot code");
    println!("\nBoa Advantages:");
    println!("  ✓ Pure Rust (no C++ dependencies)");
    println!("  ✓ Faster compilation times");
    println!("  ✓ Smaller binary size");
    println!("  ✓ Good for simple scripts\n");

    println!("✅ All V8 demos completed successfully!");
    println!("\n💡 Tip: Use V8 for production workloads requiring maximum");
    println!("   JavaScript compatibility and performance.\n");

    Ok(())
}
