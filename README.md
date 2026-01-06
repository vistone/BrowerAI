# BrowerAI

🤖 **AI-Powered Self-Learning Browser** | **AI驱动的自主学习浏览器**

An experimental browser that uses machine learning to autonomously parse and render HTML/CSS/JS.

---

## 🚀 Quick Start

```bash
# Build (without ML toolkit - requires LibTorch download)
cargo build --release

# Build with ML toolkit (requires LibTorch)
cargo build --release --features ml

# Run demo
cargo run --bin browerai

# Test
cargo test --workspace
```

## 📚 Documentation

**All documentation is in `docs/` directory:**

- **[Architecture](docs/ARCHITECTURE.md)** - System design & components
- **[Getting Started](docs/en/GETTING_STARTED.md)** - Setup guide
- **[Testing](docs/COMPREHENSIVE_TESTING.md)** - Test framework
- **[Training](docs/en/ONNX_TRAINING_GUIDE.md)** - ML pipeline
- **[Full Index](docs/INDEX.md)** - Complete docs

## 📊 Project Status

**Phase 3 Week 3**: ✅ Complete
- 459+ tests passing
- Workspace architecture with 18 specialized crates
- Build system fixed (ML toolkit is now optional)
- Code quality improvements applied
- All clippy warnings addressed

See [docs/phases/PHASE3_WEEK3_COMPLETION_REPORT.md](docs/phases/PHASE3_WEEK3_COMPLETION_REPORT.md)

## 🏗️ Project Structure

```
BrowerAI/
├── crates/               # Modular workspace crates
│   ├── browerai/         # Main binary and library
│   ├── browerai-core/    # Core types and traits
│   ├── browerai-dom/     # Document Object Model
│   ├── browerai-html-parser/   # HTML parsing
│   ├── browerai-css-parser/    # CSS parsing
│   ├── browerai-js-parser/     # JavaScript parsing
│   ├── browerai-js-analyzer/   # JS deep analysis
│   ├── browerai-ai-core/       # AI runtime (optional)
│   ├── browerai-ai-integration/  # AI integration
│   ├── browerai-ml/      # ML toolkit (optional, requires LibTorch)
│   ├── browerai-renderer-*  # Rendering engines
│   ├── browerai-intelligent-rendering/  # AI-powered rendering
│   ├── browerai-learning/     # Learning system
│   ├── browerai-network/      # HTTP client & crawler
│   ├── browerai-devtools/     # Developer tools
│   ├── browerai-testing/      # Testing utilities
│   └── browerai-plugins/      # Plugin system
├── docs/                # 📚 Documentation
├── examples/            # Example programs
├── tests/               # Integration test suites
└── training/            # Python ML training pipeline
```

## 🔧 Development

```bash
# Format code
cargo fmt --all

# Check for issues
cargo clippy --workspace

# Run specific crate tests
cargo test -p browerai-js-analyzer

# Run integration tests
cargo test --workspace --tests

# Build documentation
cargo doc --workspace --open
```

## ✨ Features

- **Modular Architecture**: 18 specialized crates for maintainability
- **Optional AI/ML**: Build without torch dependencies for faster compilation
- **Pure Rust Parsers**: HTML5ever, cssparser, Boa (no V8 dependency)
- **Advanced JS Analysis**: Scope, dataflow, control flow, and call graph analysis
- **Intelligent Rendering**: AI-powered layout and rendering optimization
- **Learning System**: Feedback collection and model improvement
- **Plugin System**: Extensible architecture
- **Developer Tools**: Built-in profiling and debugging

## 🎯 Build Features

- `ai` - Enable ONNX-based AI features (default: disabled)
- `ai-candle` - Enable Candle-based GGUF LLMs
- `ml` - Enable PyTorch-based ML toolkit (requires LibTorch download)

## 🧪 Testing

Current test status: **459+ tests passing**

```bash
# All tests
cargo test --workspace

# Library tests only
cargo test --workspace --lib

# Integration tests
cargo test --workspace --tests

# Specific test suite
cargo test --test phase3_week3_enhanced_call_graph_tests
```

## 📈 Recent Improvements

- ✅ Fixed critical build issue (made ML toolkit optional)
- ✅ Applied clippy auto-fixes (improved code quality)
- ✅ Fixed all test compilation errors
- ✅ Improved workspace architecture
- ✅ Enhanced error handling patterns
- ✅ Code formatting standardization

---

**Last Updated**: January 6, 2026 | **Status**: ✅ Active Development
