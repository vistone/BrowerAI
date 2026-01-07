# Architecture Overview

BrowerAI is built using a modular workspace architecture with 19 specialized crates. This design promotes code reuse, clear separation of concerns, and enables independent development of components.

## Workspace Structure

```
browerai/
├── crates/
│   ├── browerai/                # Main library crate
│   ├── browerai-core/           # Core traits and types
│   ├── browerai-dom/            # DOM implementation
│   ├── browerai-html-parser/    # HTML5 parser
│   ├── browerai-css-parser/     # CSS parser
│   ├── browerai-js-parser/      # JavaScript parser (Boa)
│   ├── browerai-js-analyzer/    # JavaScript analysis
│   ├── browerai-js-v8/          # V8 JavaScript engine
│   ├── browerai-ai-core/        # AI core infrastructure
│   ├── browerai-ai-integration/ # AI integration
│   ├── browerai-ml/             # Machine learning toolkit
│   ├── browerai-renderer-core/  # Core rendering
│   ├── browerai-renderer-predictive/ # Predictive rendering
│   ├── browerai-intelligent-rendering/ # AI-powered rendering
│   ├── browerai-learning/       # Learning system
│   ├── browerai-network/        # HTTP client & caching
│   ├── browerai-devtools/       # Developer tools
│   ├── browerai-testing/        # Testing utilities
│   └── browerai-plugins/        # Plugin system
├── examples/                    # Usage examples
├── tests/                       # Integration tests
├── docs/                        # Documentation
└── fuzz/                        # Fuzz testing
```

## Core Components

### 1. Parsing Pipeline

The parsing pipeline processes HTML, CSS, and JavaScript:

- **HTML Parser**: Uses html5ever for standards-compliant parsing
- **CSS Parser**: Uses cssparser for CSS3 support
- **JavaScript Parser**: Boa (default) or V8 (optional)

### 2. DOM Module

W3C-compliant DOM API with:
- Element nodes, text nodes, attributes
- Event handling system
- JavaScript sandbox integration
- Mutation observers

### 3. Rendering Pipeline

Three-stage rendering:
1. **Layout**: Calculate positions and sizes
2. **Paint**: Generate visual output
3. **Optimize**: AI-powered optimizations (optional)

### 4. AI Integration

- ONNX Runtime and Candle ML support
- Model hot-reloading
- GPU acceleration (optional)
- Parser enhancement and rendering optimization

### 5. JavaScript Engines

**Boa (Default)**: Pure Rust, fast builds, ES2024 partial support  
**V8 (Optional)**: Industry-standard, full ES2024+, maximum performance

## Feature Flags

```toml
[features]
default = []
ai = ["browerai-ai-core", "browerai-ai-integration"]
ml = ["browerai-ml"]  # Requires LibTorch
v8 = ["browerai-js-v8"]  # Requires C++ compiler
```

## Next Steps

- [Parser Details](parser-details.md) - Deep dive into parsers
- [Rendering System](rendering-system.md) - Learn about rendering
- [AI Integration](ai-integration.md) - Understand AI features
