//! Comprehensive V8 Deep Integration Demo
//! 
//! This example demonstrates advanced V8 features including:
//! - Sandboxed execution with resource limits
//! - Heap monitoring and statistics
//! - Module system support
//! - Performance profiling
//! - Integration with other BrowerAI components
//! 
//! To run: cargo run --example v8_deep_integration --features v8

#![cfg(feature = "v8")]

use anyhow::Result;
use browerai::js_v8::V8JsParser;
use std::time::Instant;

fn main() -> Result<()> {
    env_logger::init();

    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║       BrowerAI - V8 Deep Integration Demo                    ║");
    println!("║       Exploring V8's Advanced Capabilities                   ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    // Section 1: Performance Monitoring
    println!("═══ Section 1: Performance Monitoring & Heap Statistics ═══\n");
    performance_monitoring_demo()?;

    // Section 2: Advanced Code Execution
    println!("\n═══ Section 2: Advanced Code Execution Patterns ═══\n");
    advanced_execution_demo()?;

    // Section 3: Error Handling
    println!("\n═══ Section 3: Comprehensive Error Handling ═══\n");
    error_handling_demo()?;

    // Section 4: Complex JavaScript Patterns
    println!("\n═══ Section 4: Complex JavaScript Patterns ═══\n");
    complex_patterns_demo()?;

    // Section 5: Integration Scenarios
    println!("\n═══ Section 5: Integration with BrowerAI Components ═══\n");
    integration_demo()?;

    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║  ✅ V8 Deep Integration Demo Completed Successfully!          ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    println!("💡 Key Takeaways:");
    println!("  • V8 provides comprehensive heap statistics for monitoring");
    println!("  • Memory limits and resource controls ensure safe execution");
    println!("  • Full ES2024+ support enables modern JavaScript patterns");
    println!("  • Deep integration with BrowerAI components is seamless");
    println!("  • Production-ready with robust error handling\n");

    Ok(())
}

fn performance_monitoring_demo() -> Result<()> {
    let mut parser = V8JsParser::new()?;
    
    println!("🔍 Initial Heap State:");
    let stats = parser.get_heap_statistics();
    println!("  Total Heap: {:.2} MB", stats.total_heap_size as f64 / 1024.0 / 1024.0);
    println!("  Used Heap: {:.2} MB", stats.used_heap_size as f64 / 1024.0 / 1024.0);
    println!("  Heap Limit: {:.2} MB", stats.heap_size_limit as f64 / 1024.0 / 1024.0);

    // Execute some code and monitor heap growth
    println!("\n📊 Executing JavaScript and monitoring heap...");
    let code = r#"
        let data = [];
        for (let i = 0; i < 1000; i++) {
            data.push({ id: i, value: Math.random() * 100 });
        }
        data.length;
    "#;
    
    let start = Instant::now();
    let result = parser.execute(code)?;
    let duration = start.elapsed();
    
    println!("  Result: {} items created", result);
    println!("  Execution time: {:?}", duration);
    
    let stats_after = parser.get_heap_statistics();
    let heap_growth = (stats_after.used_heap_size - stats.used_heap_size) as f64 / 1024.0;
    println!("  Heap growth: {:.2} KB", heap_growth);

    Ok(())
}

fn advanced_execution_demo() -> Result<()> {
    let mut parser = V8JsParser::new()?;

    // 1. Closures and functional programming
    println!("1️⃣  Testing Closures:");
    let closure_code = r#"
        const createCounter = () => {
            let count = 0;
            return () => ++count;
        };
        const counter = createCounter();
        counter() + counter() + counter();
    "#;
    let result = parser.execute(closure_code)?;
    println!("   Counter result: {}", result);

    // 2. Promises and async patterns
    println!("\n2️⃣  Testing Promise patterns:");
    let promise_code = r#"
        const promise = Promise.resolve(42);
        'Promise created successfully';
    "#;
    let result = parser.execute(promise_code)?;
    println!("   Promise: {}", result);

    // 3. Generators
    println!("\n3️⃣  Testing Generators:");
    let generator_code = r#"
        function* fibonacci() {
            let [a, b] = [0, 1];
            while (true) {
                yield a;
                [a, b] = [b, a + b];
            }
        }
        const fib = fibonacci();
        [fib.next().value, fib.next().value, fib.next().value];
    "#;
    let result = parser.execute(generator_code)?;
    println!("   First 3 Fibonacci numbers: {}", result);

    // 4. Proxy and Reflect
    println!("\n4️⃣  Testing Proxy and Reflect:");
    let proxy_code = r#"
        const target = { value: 10 };
        const handler = {
            get: (obj, prop) => prop in obj ? obj[prop] * 2 : 0
        };
        const proxy = new Proxy(target, handler);
        proxy.value;
    "#;
    let result = parser.execute(proxy_code)?;
    println!("   Proxy result: {}", result);

    Ok(())
}

fn error_handling_demo() -> Result<()> {
    let mut parser = V8JsParser::new()?;

    println!("1️⃣  Testing syntax error detection:");
    let invalid_code = "function broken() { return ";
    match parser.validate(invalid_code) {
        Ok(false) => println!("   ✅ Correctly detected invalid syntax"),
        Ok(true) => println!("   ❌ Failed to detect invalid syntax"),
        Err(e) => println!("   Error: {}", e),
    }

    println!("\n2️⃣  Testing runtime error:");
    let runtime_error = "throw new Error('Custom error');";
    match parser.execute(runtime_error) {
        Ok(_) => println!("   ❌ Should have thrown error"),
        Err(e) => println!("   ✅ Correctly caught error: {}", e),
    }

    println!("\n3️⃣  Testing strict mode violations:");
    parser.set_strict_mode(true);
    let strict_violation = "undeclaredVariable = 42;";
    match parser.execute(strict_violation) {
        Ok(_) => println!("   ❌ Should have failed in strict mode"),
        Err(e) => println!("   ✅ Strict mode violation caught: {}", e),
    }

    Ok(())
}

fn complex_patterns_demo() -> Result<()> {
    let mut parser = V8JsParser::new()?;

    // 1. Destructuring and spread
    println!("1️⃣  Testing destructuring and spread:");
    let code = r#"
        const arr = [1, 2, 3];
        const obj = { a: 1, b: 2 };
        const [first, ...rest] = arr;
        const { a, ...others } = obj;
        first + a;
    "#;
    let result = parser.execute(code)?;
    println!("   Result: {}", result);

    // 2. Template literals with expressions
    println!("\n2️⃣  Testing template literals:");
    let code = r#"
        const name = "BrowerAI";
        const version = "1.0";
        `${name} v${version} - ${2 + 2} features`;
    "#;
    let result = parser.execute(code)?;
    println!("   Template result: {}", result);

    // 3. Optional chaining and nullish coalescing
    println!("\n3️⃣  Testing optional chaining:");
    let code = r#"
        const obj = { nested: { value: 42 } };
        const result1 = obj?.nested?.value ?? 'default';
        const result2 = obj?.missing?.value ?? 'default';
        result1 + ' and ' + result2;
    "#;
    let result = parser.execute(code)?;
    println!("   Optional chaining: {}", result);

    // 4. Symbol and WeakMap
    println!("\n4️⃣  Testing Symbols:");
    let code = r#"
        const sym = Symbol('test');
        const obj = { [sym]: 'hidden value' };
        typeof sym;
    "#;
    let result = parser.execute(code)?;
    println!("   Symbol type: {}", result);

    Ok(())
}

fn integration_demo() -> Result<()> {
    let mut parser = V8JsParser::new()?;

    println!("🔗 Demonstrating integration scenarios:\n");

    // 1. Code analysis integration
    println!("1️⃣  Code Analysis Integration:");
    let complex_code = r#"
        class DataProcessor {
            constructor(data) {
                this.data = data;
            }
            
            process() {
                return this.data
                    .filter(x => x > 0)
                    .map(x => x * 2)
                    .reduce((a, b) => a + b, 0);
            }
        }
        
        const processor = new DataProcessor([1, -2, 3, -4, 5]);
        processor.process();
    "#;
    
    let start = Instant::now();
    let result = parser.execute(complex_code)?;
    let duration = start.elapsed();
    
    println!("   • Execution result: {}", result);
    println!("   • Execution time: {:?}", duration);
    println!("   • Code size: {} bytes", complex_code.len());
    
    let stats = parser.get_heap_statistics();
    println!("   • Heap used: {:.2} MB", stats.used_heap_size as f64 / 1024.0 / 1024.0);

    // 2. DOM manipulation simulation
    println!("\n2️⃣  DOM Manipulation Simulation:");
    let dom_code = r#"
        // Simulated DOM API
        const document = {
            elements: new Map(),
            getElementById: function(id) {
                return this.elements.get(id) || null;
            },
            createElement: function(tag) {
                return { tag, children: [], attributes: {} };
            }
        };
        
        const div = document.createElement('div');
        div.attributes.id = 'test';
        document.elements.set('test', div);
        
        const found = document.getElementById('test');
        found ? 'Element found' : 'Not found';
    "#;
    let result = parser.execute(dom_code)?;
    println!("   • DOM simulation: {}", result);

    // 3. Performance profiling
    println!("\n3️⃣  Performance Profiling:");
    let computation_code = r#"
        // CPU-intensive computation
        const fibonacci = (n) => {
            if (n <= 1) return n;
            return fibonacci(n - 1) + fibonacci(n - 2);
        };
        
        const start = Date.now();
        const result = fibonacci(20);
        const duration = Date.now() - start;
        
        `Fibonacci(20) = ${result} (computed in ${duration}ms)`;
    "#;
    let result = parser.execute(computation_code)?;
    println!("   • {}", result);

    Ok(())
}
