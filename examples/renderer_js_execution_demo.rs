/// Demonstration of the RenderEngine with JavaScript execution capability
/// 
/// This example shows how the renderer now integrates with the hybrid JS orchestrator
/// to execute <script> tags during the rendering process.

use browerai_html_parser::HtmlParser;
use browerai_renderer_core::RenderEngine;

fn main() -> anyhow::Result<()> {
    env_logger::init();
    
    println!("=== Renderer JavaScript Execution Demo ===\n");
    
    // Example 1: Basic script execution
    println!("Example 1: Basic Script Execution");
    println!("----------------------------------");
    
    let html = r#"
        <!DOCTYPE html>
        <html>
        <head>
            <title>Test Page</title>
            <script>
                console.log('Initializing page...');
                var config = { theme: 'dark', version: '1.0' };
            </script>
        </head>
        <body>
            <h1>Welcome</h1>
            <div id="content">
                <p>This page contains executable JavaScript</p>
            </div>
            <script>
                function greet(name) {
                    console.log('Hello, ' + name + '!');
                    return name;
                }
                
                var result = greet('BrowerAI');
                console.log('Result:', result);
            </script>
        </body>
        </html>
    "#;
    
    let parser = HtmlParser::new();
    let dom = parser.parse(html)?;
    
    let mut engine = RenderEngine::with_viewport(1024.0, 768.0);
    
    #[cfg(feature = "ai")]
    {
        println!("✓ AI feature enabled - scripts will be executed");
        println!("✓ Using hybrid JS orchestrator (V8 + SWC + Boa)");
    }
    
    #[cfg(not(feature = "ai"))]
    {
        println!("✗ AI feature disabled - scripts will be skipped");
        println!("  (Run with: cargo run --example renderer_js_execution_demo --features ai)");
    }
    
    println!("\nRendering HTML with {} script tags...", 
             html.matches("<script>").count());
    
    let render_tree = engine.render(&dom, &[])?;
    
    println!("✓ Render completed successfully");
    println!("✓ Generated {} render nodes", render_tree.nodes.len());
    
    // Example 2: Multiple scripts with dependencies
    println!("\n\nExample 2: Multiple Scripts with Dependencies");
    println!("----------------------------------------------");
    
    let html2 = r#"
        <!DOCTYPE html>
        <html>
        <body>
            <script>
                // Library code
                var MathLib = {
                    add: function(a, b) { return a + b; },
                    multiply: function(a, b) { return a * b; }
                };
            </script>
            
            <script>
                // Application code using library
                var x = MathLib.add(10, 20);
                var y = MathLib.multiply(x, 2);
                console.log('Calculation result: x=' + x + ', y=' + y);
            </script>
            
            <h1>Calculator Result</h1>
        </body>
        </html>
    "#;
    
    let dom2 = parser.parse(html2)?;
    let mut engine2 = RenderEngine::new();
    
    println!("Rendering HTML with dependent scripts...");
    let render_tree2 = engine2.render(&dom2, &[])?;
    
    println!("✓ Render completed successfully");
    println!("✓ Generated {} render nodes", render_tree2.nodes.len());
    
    // Example 3: Error handling
    println!("\n\nExample 3: Error Handling");
    println!("-------------------------");
    
    let html3 = r#"
        <!DOCTYPE html>
        <html>
        <body>
            <script>
                console.log('This script runs fine');
                var x = 42;
            </script>
            
            <script>
                // This might have issues depending on engine
                invalid syntax here;
            </script>
            
            <script>
                console.log('This script should still run');
            </script>
            
            <h1>Error Recovery Test</h1>
        </body>
        </html>
    "#;
    
    let dom3 = parser.parse(html3)?;
    let mut engine3 = RenderEngine::new();
    
    println!("Rendering HTML with potentially problematic scripts...");
    let result = engine3.render(&dom3, &[]);
    
    match result {
        Ok(render_tree3) => {
            println!("✓ Render completed (errors in scripts are logged but don't fail render)");
            println!("✓ Generated {} render nodes", render_tree3.nodes.len());
        }
        Err(e) => {
            println!("✗ Render failed: {}", e);
        }
    }
    
    // Summary
    println!("\n\n=== Integration Summary ===");
    println!("✓ RenderEngine now executes <script> tags during render()");
    println!("✓ Scripts are extracted in document order");
    println!("✓ Execution happens before layout calculation");
    println!("✓ Script errors are logged but don't block rendering");
    println!("✓ Feature flag 'ai' controls script execution");
    println!("✓ Graceful fallback when AI feature is disabled");
    
    #[cfg(feature = "ai")]
    {
        println!("\n=== Orchestrator Details ===");
        println!("Engine Policy: Controlled by BROWERAI_RENDER_JS_POLICY");
        println!("Available Policies:");
        println!("  - performance: Use V8 for maximum speed");
        println!("  - secure: Use Boa for sandboxed execution");
        println!("  - balanced: Automatic selection (default)");
    }
    
    Ok(())
}
