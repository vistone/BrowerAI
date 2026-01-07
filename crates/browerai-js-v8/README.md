# browerai-js-v8

V8 JavaScript engine integration for BrowerAI.

## Overview

This crate provides integration with Google's V8 JavaScript engine, the same engine that powers Chrome and Node.js. It offers:

- **Maximum Compatibility**: Supports all modern JavaScript features (ES2024+)
- **High Performance**: Industry-leading JavaScript execution speed
- **Production Ready**: Battle-tested engine used by billions of users

## Features

- Parse JavaScript with V8
- Execute JavaScript code
- Validate JavaScript syntax
- Full ES2024+ support including:
  - Async/await
  - ES Modules
  - Classes and decorators
  - Optional chaining
  - Nullish coalescing
  - And more...

## Usage

```rust
use browerai_js_v8::V8JsParser;
use anyhow::Result;

fn main() -> Result<()> {
    let mut parser = V8JsParser::new()?;
    
    // Parse JavaScript
    let ast = parser.parse("const x = 42;")?;
    println!("Parsed successfully!");
    
    // Execute JavaScript
    let result = parser.execute("1 + 2")?;
    println!("Result: {}", result); // Output: Result: 3
    
    Ok(())
}
```

## Comparison with Boa

| Feature | Boa (Pure Rust) | V8 |
|---------|-----------------|-----|
| Dependencies | Pure Rust, no C++ | C++ runtime required |
| Compilation | Fast | Slower (C++ build) |
| Runtime Performance | Good | Excellent |
| ES2024 Support | Partial | Complete |
| Binary Size | Smaller | Larger |
| Use Case | Embedded, simple scripts | Complex apps, Node.js compat |

## When to Use V8

Use V8 when you need:
- Maximum JavaScript compatibility
- Best runtime performance
- Support for complex modern frameworks
- Node.js API compatibility

Use Boa when you need:
- Pure Rust dependency chain
- Fast compilation times
- Smaller binary size
- Simple script execution

## Requirements

V8 requires a C++ compiler and may take longer to compile on first build due to downloading and building the V8 library.

## License

MIT
