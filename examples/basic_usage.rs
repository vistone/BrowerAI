use anyhow::Result;
use browerai::parser::{CssParser, HtmlParser, JsParser};
use browerai::renderer::RenderEngine;

fn main() -> Result<()> {
    env_logger::init();

    // Example 1: Parse HTML
    println!("=== HTML Parsing Example ===");
    let html_parser = HtmlParser::new();
    let html = r#"
        <html>
            <head><title>Example Page</title></head>
            <body>
                <h1>Hello, BrowerAI!</h1>
                <p>This is a test page.</p>
                <div class="container">
                    <ul>
                        <li>Item 1</li>
                        <li>Item 2</li>
                        <li>Item 3</li>
                    </ul>
                </div>
            </body>
        </html>
    "#;

    let dom = html_parser.parse(html)?;
    let text = html_parser.extract_text(&dom);
    println!("Extracted text: {}", text);

    // Example 2: Parse CSS
    println!("\n=== CSS Parsing Example ===");
    let css_parser = CssParser::new();
    let css = r#"
        body {
            font-family: Arial, sans-serif;
            background-color: #f0f0f0;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        h1 {
            color: #333;
            font-size: 2em;
        }
    "#;

    let rules = css_parser.parse(css)?;
    println!("Parsed {} CSS rules", rules.len());

    // Example 3: Parse JavaScript
    println!("\n=== JavaScript Parsing Example ===");
    let js_parser = JsParser::new();
    let js = r#"
        function calculate(a, b) {
            return a + b;
        }
        
        const result = calculate(5, 3);
        console.log("Result:", result);
    "#;

    let ast = js_parser.parse(js)?;
    println!("Parsed {} tokens", ast.tokens.len());
    println!(
        "Sample tokens: {:?}",
        &ast.tokens[..ast.tokens.len().min(10)]
    );

    // Example 4: Validate JavaScript
    let valid = js_parser.validate(js)?;
    println!("JavaScript is valid: {}", valid);

    // Example 5: Rendering
    println!("\n=== Rendering Example ===");
    let mut render_engine = RenderEngine::new();
    let render_tree = render_engine.render(&dom, &rules)?;
    println!("Created render tree with {} nodes", render_tree.nodes.len());

    println!("\n=== All examples completed successfully! ===");

    Ok(())
}
