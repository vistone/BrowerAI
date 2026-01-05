# BrowerAI

🤖 **AI-Powered Self-Learning Browser** | **AI驱动的自主学习浏览器**

[English Documentation](docs/en/README.md) | [中文文档](docs/zh-CN/README.md)

---

## English

BrowerAI is an experimental browser project that uses AI-powered autonomous learning to parse and render web content. Unlike traditional browsers with hard-coded rules, BrowerAI continuously learns from visiting real websites.

### Quick Start

```bash
# Demo AI integration
cargo run

# View AI system status
cargo run -- --ai-report

# Learn from real websites
cargo run -- --learn https://example.com
```

### Key Features

- 🎓 **Autonomous Learning**: Learn from real websites automatically
- 🧠 **AI-Enhanced Parsing**: ML-powered HTML/CSS/JS parsing
- 🔨 **Code Generation**: Intelligent HTML/CSS/JS code generation with templates
- 🔓 **JS Deobfuscation**: Advanced multi-technique JavaScript deobfuscation
- 🔄 **Continuous Learning**: Automated learn-infer-generate loop
- 📊 **Performance Monitoring**: Real-time inference metrics
- 🎯 **Multi-Strategy**: Progressive and adaptive processing

### Documentation

- [Full Documentation](docs/en/README.md)
- [Enhancement Guide](docs/ENHANCEMENTS.md) - NEW!
- [Optimization Summary](docs/OPTIMIZATION_SUMMARY.md) - NEW!
- [Intelligent Rendering Architecture](docs/INTELLIGENT_RENDERING_ARCHITECTURE.md) - NEW!
- [Real Network Testing Guide](docs/REAL_NETWORK_TESTING.md) - **NEW!**
- [Comprehensive Testing](docs/COMPREHENSIVE_TESTING.md) - NEW!
- [Real Test Results](docs/REAL_TEST_RESULTS.md) - NEW!
- [Quick Reference](docs/en/QUICKREF.md)
- [Getting Started](docs/en/GETTING_STARTED.md)
- [Training Guide](training/README.md)
- [Model Zoo](models/MODEL_ZOO.md) - NEW!

### Technology Stack

- **Rust** - Core language
- **ONNX Runtime** - ML inference
- **html5ever, cssparser** - Parsing foundations

### License

MIT License - see [LICENSE](LICENSE)

---

## 中文

BrowerAI 是一个实验性浏览器项目，使用 AI 自主学习来解析和渲染网页内容。与传统浏览器使用硬编码规则不同，BrowerAI 通过访问真实网站不断学习。

### 快速开始

```bash
# 演示 AI 集成
cargo run

# 查看 AI 系统状态  
cargo run -- --ai-report

# 访问真实网站学习
cargo run -- --learn https://example.com
```

### 核心特性

- 🎓 **自主学习系统**: 自动从真实网站学习
- 🧠 **AI 增强解析**: ML 驱动的 HTML/CSS/JS 解析
- 🔨 **代码生成**: 智能 HTML/CSS/JS 代码生成，支持模板
- 🔓 **JS 去混淆**: 高级多技术 JavaScript 去混淆
- 🔄 **持续学习**: 自动化学习-推理-生成循环
- 📊 **性能监控**: 实时推理指标
- 🎯 **多策略**: 渐进式和自适应处理

### 文档

- [完整文档](docs/zh-CN/README.md)
- [增强功能指南](docs/ENHANCEMENTS.md) - 新增!
- [优化总结](docs/OPTIMIZATION_SUMMARY.md) - 新增!
- [智能渲染架构](docs/INTELLIGENT_RENDERING_ARCHITECTURE.md) - 新增!
- [真实网络测试指南](docs/REAL_NETWORK_TESTING.md) - **新增!**
- [全面测试文档](docs/COMPREHENSIVE_TESTING.md) - 新增!
- [真实测试结果](docs/REAL_TEST_RESULTS.md) - 新增!
- [快速参考](docs/zh-CN/QUICKREF.md)
- [入门指南](docs/zh-CN/GETTING_STARTED.md)
- [训练指南](training/README.md)
- [模型库](models/MODEL_ZOO.md) - 新增!

### 技术栈

- **Rust** - 核心语言
- **ONNX Runtime** - ML 推理
- **html5ever, cssparser** - 解析基础

### 许可证

MIT 许可证 - 参见 [LICENSE](LICENSE)

---

**Status**: ✅ All Phases Complete | 所有阶段已完成

[![Rust](https://img.shields.io/badge/rust-1.70%2B-orange.svg)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
