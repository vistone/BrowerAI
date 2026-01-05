# BrowerAI

🤖 **AI-Powered Self-Learning Browser** - 基于浏览器技术的 AI 自主学习系统

## Overview

BrowerAI 是一个实验性浏览器项目，使用 AI 自主学习来解析和渲染网页内容。与传统浏览器使用硬编码规则不同，BrowerAI 通过访问真实网站不断学习，使用机器学习模型理解和处理 HTML、CSS 和 JavaScript。

**核心理念**：浏览器作为教师 (Browser as Teacher) - 每次访问网站都是一次学习机会，形成"访问 → 解析 → 反馈 → 训练 → 部署"的完整闭环。

## ✨ 核心特性

### 🎓 自主学习系统
- **真实网站访问**: 自动访问并学习真实网站的结构和内容
- **反馈收集管道**: 记录所有解析、渲染、性能数据用于训练
- **学习闭环**: 从访问到模型训练的完整自动化流程
- **批量学习**: 支持并行访问多个网站收集数据

### 🧠 AI 增强引擎
- **AI HTML 解析**: ML 模型辅助理解 HTML 结构和复杂度
- **智能 CSS 优化**: AI 生成 CSS 优化建议
- **JS 代码分析**: ML 驱动的 JavaScript 模式识别
- **自适应渲染**: AI 优化的渲染引擎

### 📊 监控与报告
- **AI 系统报告**: 全面的模型健康状态和性能监控
- **性能指标**: 实时追踪推理时间、成功率
- **反馈统计**: 详细的事件类型分布和趋势
- **训练数据导出**: JSON 格式用于模型训练

### 🔄 持续改进
- **模型版本管理**: 语义化版本控制和生命周期管理
- **A/B 测试框架**: 内置实验系统对比模型版本
- **在线学习**: 支持增量学习和模型微调
- **自我优化**: 基于历史数据自动调整参数

## 🚀 快速开始

### 演示 AI 集成
```bash
cargo run
```

### 查看 AI 系统状态
```bash
cargo run -- --ai-report
```

### 访问真实网站学习
```bash
# 单个网站
cargo run -- --learn https://example.com

# 多个网站
cargo run -- --learn https://example.com https://httpbin.org/html https://www.w3.org
```

**学习输出示例**：
```
🌐 开始批量访问 2 个网站...
    
📍 [1/2] 访问: https://example.com
  ✅ 获取成功，大小: 513 bytes，耗时: 0.05s
  ✅ HTML 解析成功，耗时: 0.44ms
  📝 提取文本内容: 285 字符
  ✅ 渲染完成，节点数: 19
✅ 访问完成！总耗时: 53.02ms，反馈事件数: 2

📊 学习报告摘要
════════════════════════════════════════
网站: https://example.com
HTML 大小: 513 bytes | CSS 规则: 7 | 渲染节点: 19

【反馈管道统计】
  总事件数: 2
  HTML 解析事件: 1
  CSS 解析事件: 1

💾 反馈数据已导出到: ./training/data/feedback_20260104_103839.json
```

查看 [QUICKREF.md](QUICKREF.md) 获取完整命令参考。

## 🎯 学习工作流

```
1. 访问网站 → 2. 收集反馈 → 3. 训练模型 → 4. 部署更新 → 5. 再次访问
    ↓              ↓              ↓              ↓              ↓
  HTTP GET     JSON 导出      ONNX 训练      模型加载      性能提升
```

**完整流程**：
```bash
# 1. 收集数据
cargo run -- --learn https://example.com https://httpbin.org/html

# 2. 查看反馈
cat training/data/feedback_*.json | jq '.'

# 3. 训练模型（Python）
cd training && python scripts/train_html_parser_v2.py

# 4. 部署模型
cp training/models/*.onnx models/local/

# 5. 测试效果
cargo build --features ai && cargo run -- --ai-report
```

## 📚 文档

- **[QUICKREF.md](QUICKREF.md)** - 快速参考和常用命令
- **[LEARNING_GUIDE.md](LEARNING_GUIDE.md)** - 学习与调优详细指南
- **[AI_LEARNING_IMPLEMENTATION.md](AI_LEARNING_IMPLEMENTATION.md)** - 技术实现报告
- **[GETTING_STARTED.md](GETTING_STARTED.md)** - 项目入门教程
- **[training/QUICKSTART.md](training/QUICKSTART.md)** - 模型训练快速开始

## 🏗️ Architecture

```
BrowerAI/
├── src/
│   ├── ai/                          # AI/ML 核心系统
│   │   ├── runtime.rs               # AI 运行时（集成所有 AI 组件）
│   │   ├── inference.rs             # ONNX 推理引擎
│   │   ├── model_manager.rs         # 模型库管理
│   │   ├── feedback_pipeline.rs     # 反馈事件收集 ⭐ NEW
│   │   ├── reporter.rs              # AI 状态报告 ⭐ NEW
│   │   ├── performance_monitor.rs   # 性能监控
│   │   └── hot_reload.rs            # 模型热重载
│   ├── parser/                      # 内容解析器（AI 增强）
│   │   ├── html.rs                  # HTML 解析 + AI 验证
│   │   ├── css.rs                   # CSS 解析 + AI 优化
│   │   └── js.rs                    # JavaScript 解析 + AI 分析
│   ├── renderer/                    # 渲染引擎
│   │   ├── engine.rs                # AI 优化的渲染
│   │   ├── layout.rs                # 布局计算
│   │   └── paint.rs                 # 绘制操作
│   ├── learning/                    # 学习系统 ⭐ NEW
│   │   ├── website_learner.rs       # 真实网站访问学习器
│   │   ├── feedback.rs              # 用户反馈收集
│   │   ├── online_learning.rs       # 在线学习
│   │   ├── versioning.rs            # 模型版本管理
│   │   ├── ab_testing.rs            # A/B 测试框架
│   │   ├── personalization.rs       # 用户个性化
│   │   └── optimization.rs          # 自我优化
│   ├── network/                     # 网络层
│   │   ├── http.rs                  # HTTP 客户端
│   │   └── cache.rs                 # 资源缓存
│   └── main.rs                      # CLI 入口（4 种模式）⭐ NEW
├── models/
│   ├── model_config.toml            # 模型配置文件
│   └── local/                       # 本地 ONNX 模型存储
├── training/                        # 训练管道
│   ├── data/                        # 反馈数据（自动生成）⭐ NEW
│   │   └── feedback_*.json
│   ├── scripts/                     # Python 训练脚本
│   │   ├── train_html_parser_v2.py
│   │   ├── train_css_parser.py
│   │   └── train_js_parser.py
│   └── models/                      # 训练输出的 ONNX 模型
└── examples/                        # 示例代码
    └── basic_usage.rs
```

**数据流**:
```
真实网站 URLs
    ↓ (HTTP GET)
WebsiteLearner
    ↓ (HTML 字符串)
HtmlParser (AI) → FeedbackPipeline
    ↓ (DOM tree)
CssParser (AI) → FeedbackPipeline
    ↓ (CSS rules)
JsParser (AI) → FeedbackPipeline
    ↓ (AST)
RenderEngine → FeedbackPipeline
    ↓ (渲染结果)
JSON 导出 (training/data/feedback_*.json)
    ↓
Python 训练脚本
    ↓
ONNX 模型 (training/models/*.onnx)
    ↓
部署到 models/local/
    ↓
下次访问使用新模型 ♻️
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
- **ort**: Rust bindings for ONNX Runtime (https://github.com/pykeio/ort)

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

To quickly focus on the technology stack:

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

## Documentation

- **[Implementation Guide](IMPLEMENTATION_GUIDE.md)** - Comprehensive guide covering all implementations
- **[Roadmap](ROADMAP.md)** - Development roadmap and progress tracking
- **[Getting Started](GETTING_STARTED.md)** - Quick start guide for developers
- **[Contributing](CONTRIBUTING.md)** - Contribution guidelines

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see LICENSE file for details

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