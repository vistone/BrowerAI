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

**Week 6 Learning**: ✅ Complete
- ✨ 100% Real Data Learning System
- 🎯 5,491 real code files collected
- 🔀 12 obfuscation techniques applied
- 🚀 GPU acceleration with CUDA
- 📊 50 epochs training completed

See [docs/phases/PHASE3_WEEK3_COMPLETION_REPORT.md](docs/phases/PHASE3_WEEK3_COMPLETION_REPORT.md)

See [WEEK6_COMPLETION_SUMMARY.md](WEEK6_COMPLETION_SUMMARY.md)

## 🏗️ Project Structure

```
BrowerAI/
├── crates/               # Modular workspace crates
│   ├── browerai/         # Main binary and library
│   ├── browerai-core/    # Core types and traits
│   ├── browerai-dom/     # Document Object Model
│   ├── browerai-html-parser/   # HTML parsing
│   ├── browerai-css-parser/    # CSS parsing
│   ├── browerai-js-parser/     # JavaScript parsing (Boa)
│   ├── browerai-js-v8/         # JavaScript V8 engine (optional)
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

## 🎓 Week 6 - Real Data Learning System

**Run the complete real data learning pipeline:**

```bash
# Quick start - 完整流程
just learn-real

# Custom parameters - 自定义参数
just learn-real-techniques 6       # 6种混淆技术
just learn-real-epochs 100         # 100个epochs
just learn-real-batch 64           # 批大小64

# Full customization - 全参数自定义
just learn-real-custom crates 6 100 64

# View results - 查看结果
just learn-results

# Help
just learn-help
```

**What happens:**
1. 📂 Collects 5,000+ real code files from project
2. 🔀 Applies 12 obfuscation techniques
3. 🧠 Extracts 48-dimensional features
4. 🚀 GPU-accelerated training (50 epochs)
5. 📊 Outputs training history & results

**Output location:** `data/real_codes/`
- `raw_codes.jsonl` - Original code samples
- `obfuscated_samples.jsonl` - Obfuscated samples
- `training_history.json` - Training metrics

See [WEEK6_COMPLETION_SUMMARY.md](WEEK6_COMPLETION_SUMMARY.md) for details

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
- `v8` - Enable V8 JavaScript engine (Google's V8, full ES2024+ support)

### JavaScript Engine Options

BrowerAI supports two JavaScript engines:

**Boa (Default - Pure Rust)**
- Pure Rust implementation
- Fast compilation
- Smaller binary size
- Good ES6+ support
- Best for: Embedded systems, simple scripts

**V8 (Optional - High Performance)**
- Google's V8 engine (Chrome/Node.js)
- Full ES2024+ compatibility
- Maximum runtime performance
- Industry-standard
- Best for: Complex apps, production workloads

```bash
# Use default Boa engine
cargo build

# Use V8 engine for maximum compatibility
cargo build --features v8

# Run V8 demo
cargo run --example v8_demo --features v8
```

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
