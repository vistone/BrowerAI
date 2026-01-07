# Getting Started

Welcome to the BrowerAI Getting Started guide! This section will help you install BrowerAI, set up your development environment, and create your first project.

## Installation

### Prerequisites

BrowerAI requires:
- Rust 1.70 or later
- Cargo (comes with Rust)

### Installing Rust

If you don't have Rust installed, install it using rustup:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### Adding BrowerAI to Your Project

Add BrowerAI to your `Cargo.toml`:

```toml
[dependencies]
browerai = "0.1"
```

### Optional Features

BrowerAI has several optional features:

```toml
[dependencies]
browerai = { version = "0.1", features = ["ai", "v8"] }
```

**Available features:**
- `ai` - Enable AI-powered parsing and rendering
- `ml` - Enable machine learning features (requires LibTorch)
- `v8` - Enable V8 JavaScript engine (requires C++ compiler)

## Quick Start

Let's parse some HTML:

```rust
use browerai::HtmlParser;

fn main() -> anyhow::Result<()> {
    let parser = HtmlParser::new();
    let html = "<h1>Hello, BrowerAI!</h1>";
    
    let result = parser.parse(html)?;
    println!("Parsed: {:?}", result);
    
    let text = parser.extract_text(html)?;
    println!("Text: {}", text);
    
    Ok(())
}
```

## Your First Complete Example

Here's a complete example that parses HTML, CSS, and JavaScript:

```rust
use browerai::{HtmlParser, CssParser, JsParser};
use anyhow::Result;

fn main() -> Result<()> {
    // Parse HTML
    let html_parser = HtmlParser::new();
    let html = r#"
        <!DOCTYPE html>
        <html>
            <head>
                <title>My Page</title>
            </head>
            <body>
                <h1 class="title">Welcome</h1>
                <p id="content">This is a test page.</p>
            </body>
        </html>
    "#;
    
    let dom = html_parser.parse(html)?;
    println!("Parsed HTML successfully!");
    
    // Parse CSS
    let mut css_parser = CssParser::new();
    let css = r#"
        .title {
            color: blue;
            font-size: 24px;
        }
        #content {
            margin: 10px;
        }
    "#;
    
    let styles = css_parser.parse(css)?;
    println!("Parsed CSS successfully!");
    
    // Parse JavaScript
    let mut js_parser = JsParser::new();
    let js = r#"
        function greet(name) {
            return `Hello, ${name}!`;
        }
        console.log(greet("World"));
    "#;
    
    let ast = js_parser.parse(js)?;
    println!("Parsed JavaScript successfully!");
    
    Ok(())
}
```

## Next Steps

- [HTML Parsing Guide](../user-guide/html-parsing.md) - Learn about HTML parsing features
- [CSS Parsing Guide](../user-guide/css-parsing.md) - Learn about CSS parsing
- [JavaScript Support](../user-guide/javascript.md) - Learn about JavaScript engines
- [AI Features](../user-guide/ai-features.md) - Explore AI-powered capabilities

## Getting Help

- [GitHub Issues](https://github.com/vistone/BrowerAI/issues) - Report bugs or request features
- [Discussions](https://github.com/vistone/BrowerAI/discussions) - Ask questions
- [Examples](https://github.com/vistone/BrowerAI/tree/main/examples) - See more examples
