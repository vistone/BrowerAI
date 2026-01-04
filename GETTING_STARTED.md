# Getting Started with BrowerAI

This guide will help you get started with BrowerAI and understand how to work with the technology stack.

## Quick Start

### Prerequisites

1. **Rust Installation**
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   source $HOME/.cargo/env
   ```

2. **Clone and Build**
   ```bash
   git clone https://github.com/vistone/BrowerAI.git
   cd BrowerAI
   cargo build --release
   ```

3. **Run the Example**
   ```bash
   cargo run
   ```

## Technology Stack Overview

### 1. HTML Parsing (html5ever)

BrowerAI uses `html5ever`, a high-performance HTML5 parser written in Rust. It follows the HTML5 parsing specification and can handle malformed HTML.

**Key Features:**
- Standards-compliant HTML5 parsing
- Streaming parser for memory efficiency
- Produces a Document Object Model (DOM) tree

**Example Usage:**
```rust
use browerai::HtmlParser;

let parser = HtmlParser::new();
let html = "<html><body><h1>Hello!</h1></body></html>";
let dom = parser.parse(html)?;
let text = parser.extract_text(&dom);
println!("Text content: {}", text);
```

**Learning Resources:**
- [html5ever documentation](https://docs.rs/html5ever/)
- [HTML5 Living Standard](https://html.spec.whatwg.org/)

### 2. CSS Parsing (cssparser)

The CSS parser handles Cascading Style Sheets, allowing BrowerAI to understand styles and layout rules.

**Key Features:**
- CSS tokenization
- Selector parsing
- Property value parsing

**Example Usage:**
```rust
use browerai::CssParser;

let parser = CssParser::new();
let css = "body { color: red; font-size: 14px; }";
let rules = parser.parse(css)?;
println!("Parsed {} rules", rules.len());
```

**Learning Resources:**
- [cssparser documentation](https://docs.rs/cssparser/)
- [CSS Specification](https://www.w3.org/Style/CSS/)

### 3. JavaScript Parsing

BrowerAI provides basic JavaScript tokenization and validation capabilities.

**Example Usage:**
```rust
use browerai::JsParser;

let parser = JsParser::new();
let js = "function hello() { return 'world'; }";
let ast = parser.parse(js)?;
let is_valid = parser.validate(js)?;
println!("Valid JS: {}, Tokens: {}", is_valid, ast.tokens.len());
```

### 4. AI Integration (ONNX Runtime)

BrowerAI uses ONNX Runtime for ML model inference. This allows the browser to use trained models for intelligent parsing and rendering.

**Key Components:**
- **InferenceEngine**: Manages ONNX Runtime and model execution
- **ModelManager**: Organizes and loads models from the local library

**Example Usage:**
```rust
use browerai::{InferenceEngine, ModelManager};
use std::path::PathBuf;

// Initialize model manager
let model_dir = PathBuf::from("./models/local");
let mut manager = ModelManager::new(model_dir)?;

// Initialize inference engine (requires 'ai' feature)
// cargo build --features ai
let engine = InferenceEngine::new()?;
```

## Building with AI Features

By default, the ONNX Runtime dependency is optional. To build with full AI capabilities:

```bash
cargo build --features ai --release
```

**Why Optional?**
- ONNX Runtime requires downloading native libraries
- Not all users may need AI features immediately
- Allows for lighter builds during development

## Creating AI Models for BrowerAI

### Model Training Pipeline

1. **Data Collection**
   - Collect HTML/CSS/JS samples
   - Label data for supervised learning
   - Create training/validation/test splits

2. **Model Design**
   - Choose appropriate architecture (Transformer, LSTM, etc.)
   - Define input/output formats
   - Consider model size vs. accuracy trade-offs

3. **Training**
   - Use PyTorch, TensorFlow, or other frameworks
   - Train on your dataset
   - Monitor metrics and adjust hyperparameters

4. **Export to ONNX**
   ```python
   import torch
   import torch.onnx
   
   # Load your trained model
   model = YourModel()
   model.load_state_dict(torch.load('model.pth'))
   model.eval()
   
   # Create dummy input matching your model's expected input
   dummy_input = torch.randn(1, 128)
   
   # Export
   torch.onnx.export(
       model,
       dummy_input,
       "your_model.onnx",
       export_params=True,
       opset_version=15,
       input_names=['input'],
       output_names=['output']
   )
   ```

5. **Add to BrowerAI**
   - Place `.onnx` file in `models/local/`
   - Update `models/model_config.toml`
   - Use in your parser/renderer code

### Example Model Configuration

```toml
[[models]]
name = "html_structure_predictor"
model_type = "HtmlParser"
path = "html_structure_v1.onnx"
description = "Predicts HTML structure and suggests fixes"
version = "1.0.0"

[[models]]
name = "css_optimizer"
model_type = "CssParser"
path = "css_opt_v1.onnx"
description = "Optimizes CSS rules and removes duplicates"
version = "1.0.0"
```

## Development Workflow

### 1. Local Development

```bash
# Build in debug mode
cargo build

# Run with logging
RUST_LOG=debug cargo run

# Run tests
cargo test

# Run a specific example
cargo run --example basic_usage
```

### 2. Testing Your Changes

```bash
# Run unit tests
cargo test

# Run with different features
cargo test --features ai

# Run specific test
cargo test test_parse_simple_html
```

### 3. Code Quality

```bash
# Check code without building
cargo check

# Format code
cargo fmt

# Lint code
cargo clippy
```

## Project Structure

```
BrowerAI/
├── src/
│   ├── ai/                    # AI/ML components
│   │   ├── model_manager.rs  # Model library management
│   │   └── inference.rs      # ONNX Runtime wrapper
│   ├── parser/               # Content parsers
│   │   ├── html.rs          # HTML parser
│   │   ├── css.rs           # CSS parser
│   │   └── js.rs            # JavaScript parser
│   ├── renderer/            # Rendering engine
│   │   └── engine.rs        # Render tree builder
│   ├── lib.rs               # Library exports
│   └── main.rs              # CLI application
├── models/                   # Model storage
│   ├── local/               # ONNX model files
│   └── model_config.toml    # Model configuration
├── examples/                # Example code
│   └── basic_usage.rs       # Basic usage examples
├── Cargo.toml               # Rust dependencies
└── README.md                # Project documentation
```

## Learning Path

### Week 1-2: Foundation
- [ ] Learn Rust basics
- [ ] Understand HTML5 specification
- [ ] Study CSS parsing fundamentals
- [ ] Explore JavaScript syntax

### Week 3-4: Parsers
- [ ] Deep dive into html5ever
- [ ] Work with cssparser
- [ ] Implement custom parser features
- [ ] Write comprehensive tests

### Week 5-6: AI Integration
- [ ] Learn ONNX fundamentals
- [ ] Study ML model architectures
- [ ] Collect training data
- [ ] Train initial models

### Week 7-8: Integration
- [ ] Integrate models with parsers
- [ ] Test end-to-end workflows
- [ ] Optimize performance
- [ ] Document findings

## Common Tasks

### Adding a New Parser Feature

1. Identify the feature requirements
2. Add implementation in the appropriate parser file
3. Write tests for the new feature
4. Update documentation
5. Run tests: `cargo test`

### Training a New Model

1. Prepare your dataset
2. Design model architecture
3. Train and validate
4. Export to ONNX format
5. Add to `models/local/`
6. Update `model_config.toml`
7. Integrate with parser/renderer

### Debugging

```bash
# Enable all logging
RUST_LOG=trace cargo run

# Enable specific module logging
RUST_LOG=browerai::parser=debug cargo run

# Run with backtrace
RUST_BACKTRACE=1 cargo run
```

## Next Steps

1. **Explore the codebase**: Read through the source files
2. **Run examples**: Try `cargo run --example basic_usage`
3. **Modify and experiment**: Change the sample HTML/CSS/JS
4. **Add features**: Start with small enhancements
5. **Train models**: Collect data and train your first model

## Getting Help

- Read the [README.md](README.md) for overview
- Check [models/README.md](models/README.md) for model details
- Review test files for usage examples
- Open issues on GitHub for questions

## Contributing

We welcome contributions! Areas to focus on:

1. **Parser Enhancements**: Improve HTML/CSS/JS parsing
2. **Model Training**: Create and share trained models
3. **Performance**: Optimize critical paths
4. **Documentation**: Improve guides and examples
5. **Testing**: Add more comprehensive tests

Happy coding! 🚀
