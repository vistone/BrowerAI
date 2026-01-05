# BrowerAI - English Documentation

🤖 **AI-Powered Self-Learning Browser** - An experimental browser using machine learning for autonomous parsing and rendering

[中文文档](../zh-CN/README.md) | **English** | [Main README](../../README.md)

## Overview

BrowerAI is an experimental browser project that uses AI-powered autonomous learning to parse and render web content. Unlike traditional browsers with hard-coded rules, BrowerAI continuously learns from visiting real websites, using machine learning models to understand and process HTML, CSS, and JavaScript.

**Core Concept**: Browser as Teacher - Each website visit is a learning opportunity, forming a complete cycle of "Visit → Parse → Feedback → Train → Deploy".

## ✨ Key Features

### 🎓 Autonomous Learning System
- **Real Website Visiting**: Automatically visit and learn from real website structures and content
- **Feedback Collection Pipeline**: Record all parsing, rendering, and performance data for training
- **Learning Loop**: Complete automated workflow from visiting to model training
- **Batch Learning**: Support parallel website visits for data collection

### 🧠 AI-Enhanced Engine
- **AI HTML Parsing**: ML models assist in understanding HTML structure and complexity
- **Intelligent CSS Optimization**: AI-generated CSS optimization suggestions
- **JS Code Analysis**: ML-driven JavaScript pattern recognition
- **Adaptive Rendering**: AI-optimized rendering engine

### 📊 Monitoring and Reporting
- **AI System Reports**: Comprehensive model health status and performance monitoring
- **Performance Metrics**: Real-time tracking of inference time and success rates
- **Feedback Statistics**: Detailed event type distribution and trends
- **Training Data Export**: JSON format for model training

### 🔄 Continuous Improvement
- **Model Version Management**: Semantic versioning and lifecycle management
- **A/B Testing Framework**: Built-in experiment system for comparing model versions
- **Online Learning**: Support for incremental learning and model fine-tuning
- **Self-Optimization**: Automatic parameter adjustment based on historical data

## 🚀 Quick Start

### Demo AI Integration
```bash
cargo run
```

### View AI System Status
```bash
cargo run -- --ai-report
```

### Learn from Real Websites
```bash
# Single website
cargo run -- --learn https://example.com

# Multiple websites
cargo run -- --learn https://example.com https://httpbin.org/html https://www.w3.org
```

See [Quick Reference](QUICKREF.md) for complete command reference.

## 🎯 Learning Workflow

```
1. Visit Website → 2. Collect Feedback → 3. Train Model → 4. Deploy Update → 5. Revisit
     ↓                   ↓                   ↓                ↓                ↓
  HTTP GET          JSON Export         ONNX Training     Load Model      Performance Boost
```

**Complete Workflow**:
```bash
# 1. Collect data
cargo run -- --learn https://example.com https://httpbin.org/html

# 2. View feedback
cat training/data/feedback_*.json | jq '.'

# 3. Train model (Python)
cd training && python scripts/train_html_parser_v2.py

# 4. Deploy model
cp training/models/*.onnx models/local/

# 5. Test results
cargo build --features ai && cargo run -- --ai-report
```

## 📚 Documentation

- **[Quick Reference](QUICKREF.md)** - Quick reference and common commands
- **[Getting Started](GETTING_STARTED.md)** - Project tutorial
- **[Learning Guide](LEARNING_GUIDE.md)** - Learning and tuning detailed guide
- **[AI Implementation](AI_LEARNING_IMPLEMENTATION.md)** - Technical implementation report
- **[Training Guide](../../training/README.md)** - Model training quick start
- **[Contributing](CONTRIBUTING.md)** - Contribution guidelines

## 🏗️ Architecture

```
BrowerAI/
├── src/                              # Rust core
│   ├── ai/                           # AI/ML core system
│   ├── parser/                       # Content parsers (AI-enhanced)
│   ├── renderer/                     # Rendering engine
│   ├── learning/                     # Learning system
│   ├── network/                      # Network layer
│   └── main.rs                       # CLI entry
├── models/                           # Model library
│   ├── model_config.toml             # Model configuration
│   └── local/                        # Local ONNX models
├── training/                         # Training pipeline
│   ├── data/                         # Feedback data
│   ├── scripts/                      # Python training scripts
│   └── models/                       # Trained ONNX models
└── examples/                         # Example code
```

## Technology Stack

### Core Technologies
- **Rust**: Primary programming language for performance and safety
- **ONNX Runtime**: ML inference engine (via `ort` crate)
- **html5ever**: HTML parsing foundation
- **cssparser**: CSS parsing utilities
- **selectors**: CSS selector matching

### AI/ML Stack
- **ONNX**: Open Neural Network Exchange format for models
- **ort**: Rust bindings for ONNX Runtime

## Getting Started

### Prerequisites

- Rust 1.70 or later
- Cargo (comes with Rust)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/vistone/BrowerAI.git
cd BrowerAI
```

2. Build the project:
```bash
cargo build --release
```

3. Run the application:
```bash
cargo run
```

### Running Tests

```bash
cargo test
```

## Model Library

BrowerAI uses a local model library stored in `models/local/`. The system supports the following model types:

- **HtmlParser**: Models for HTML structure understanding
- **CssParser**: Models for CSS rule optimization
- **JsParser**: Models for JavaScript analysis
- **LayoutOptimizer**: Models for layout calculations
- **RenderingOptimizer**: Models for rendering optimizations

### Adding Models

1. Place your ONNX model files in `models/local/`
2. Create or update the model configuration (see `models/model_config.toml`)
3. The model manager will automatically load and manage your models

Example model configuration:
```toml
[[models]]
name = "html_parser_v1"
model_type = "HtmlParser"
path = "html_parser_v1.onnx"
description = "Base HTML parsing model"
version = "1.0.0"
```

## Development Roadmap

### Phase 1: Foundation ✅ Complete
- [x] Project structure setup
- [x] Basic HTML/CSS/JS parsers
- [x] ONNX Runtime integration
- [x] Model management system
- [x] Initial model training pipeline

### Phase 2: AI Enhancement ✅ Complete
- [x] Train HTML parsing models
- [x] Train CSS optimization models
- [x] Train JavaScript analysis models
- [x] Implement model inference in parsers

### Phase 3: Rendering ✅ Complete
- [x] AI-powered layout engine
- [x] Intelligent rendering optimizations
- [x] Performance profiling and tuning

### Phase 4: Advanced Features ✅ Complete
- [x] Real-time learning and adaptation
- [x] Model fine-tuning based on usage
- [x] Multi-model ensemble approaches

### Phase 5: Learning & Adaptation ✅ Complete
- [x] Feedback collection system
- [x] Online learning pipeline
- [x] Model versioning
- [x] A/B testing framework
- [x] Self-optimization
- [x] User personalization

## Learning Resources

### HTML
- [HTML5 Specification](https://html.spec.whatwg.org/)
- [html5ever Documentation](https://docs.rs/html5ever/)

### CSS
- [CSS Specification](https://www.w3.org/Style/CSS/)
- [cssparser Documentation](https://docs.rs/cssparser/)

### JavaScript
- [ECMAScript Specification](https://tc39.es/ecma262/)

### ONNX and ML
- [ONNX Documentation](https://onnx.ai/)
- [ort Crate Documentation](https://docs.rs/ort/)
- [ONNX Runtime](https://onnxruntime.ai/)

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

MIT License - see [LICENSE](../../LICENSE) file for details

## Acknowledgments

- **pykeio/ort**: Excellent Rust bindings for ONNX Runtime
- **html5ever**: Robust HTML5 parser
- **cssparser**: CSS parsing utilities from Servo project

## Future Vision

BrowerAI aims to create a browser where:
- AI models continuously learn from web content patterns
- Parsing and rendering are optimized through machine learning
- The browser adapts to new web technologies autonomously
- Performance improves over time through reinforcement learning

This is an experimental project pushing the boundaries of what's possible with AI in web browsing technology.
