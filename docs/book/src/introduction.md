# BrowerAI - AI-Powered Browser Engine

Welcome to BrowerAI! An experimental AI-powered browser with machine learning integration.

## Key Features

- 🚀 Dual JavaScript Engines (Boa & V8)
- 🧠 AI-Enhanced Parsing
- 🏗️ Modern 19-Crate Architecture
- ✅ Production-Grade Quality

## Quick Example

```rust
use browerai::prelude::*;

fn main() -> anyhow::Result<()> {
    let parser = HtmlParser::new();
    let dom = parser.parse("<html><body>Hello!</body></html>")?;
    Ok(())
}
```

See [Installation](./getting-started/installation.md) to get started!
