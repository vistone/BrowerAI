# BrowerAI

🤖 **AI-Powered Self-Learning Browser** | **AI驱动的自主学习浏览器**

An experimental browser that uses machine learning to autonomously parse and render HTML/CSS/JS.

---

## 🚀 Quick Start

```bash
# Build
cargo build --release

# Run
cargo run

# Test
cargo test
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
- 459 tests passing
- Scope & Data Flow Analysis done
- JavaScript deobfuscation enhanced
- Step 4 Rust integration testing complete

See [docs/phases/PHASE3_WEEK3_COMPLETION_REPORT.md](docs/phases/PHASE3_WEEK3_COMPLETION_REPORT.md)

## 🏗️ Project Structure

```
BrowerAI/
├── src/                 # Rust browser engine
├── training/            # ML model training
├── docs/                # 📚 Documentation
├── examples/            # Example programs
├── tests/               # Test suites
├── models/              # ONNX models
└── Cargo.toml
```

## 🔧 Development

```bash
# Format code
cargo fmt --all

# Run tests
cargo test --lib
cargo test --test '*'

# Run specific test suite
cargo test parser::js_analyzer
```

---

**Last Updated**: January 6, 2026 | **Status**: ✅ Active Development
