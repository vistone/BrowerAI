# BrowerAI - 中文文档

🤖 **AI驱动的自主学习浏览器** - 使用机器学习进行自主解析和渲染的实验性浏览器

**中文** | [English](../en/README.md) | [主README](../../README.md)

## 概述

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

查看 [快速参考](QUICKREF.md) 获取完整命令参考。

## 🎯 学习工作流

```
1. 访问网站 → 2. 收集反馈 → 3. 训练模型 → 4. 部署更新 → 5. 再次访问
    ↓              ↓              ↓              ↓              ↓
  HTTP GET     JSON 导出      ONNX 训练      模型加载      性能提升
```

**完整流程**:
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

- **[快速参考](QUICKREF.md)** - 快速参考和常用命令
- **[入门指南](GETTING_STARTED.md)** - 项目入门教程
- **[学习指南](LEARNING_GUIDE.md)** - 学习与调优详细指南
- **[AI实现](AI_LEARNING_IMPLEMENTATION.md)** - 技术实现报告
- **[训练指南](../../training/README.md)** - 模型训练快速开始
- **[贡献指南](CONTRIBUTING.md)** - 贡献指南

## 🏗️ 架构

```
BrowerAI/
├── src/                              # Rust 核心
│   ├── ai/                           # AI/ML 核心系统
│   ├── parser/                       # 内容解析器（AI 增强）
│   ├── renderer/                     # 渲染引擎
│   ├── learning/                     # 学习系统
│   ├── network/                      # 网络层
│   └── main.rs                       # CLI 入口
├── models/                           # 模型库
│   ├── model_config.toml             # 模型配置
│   └── local/                        # 本地 ONNX 模型
├── training/                         # 训练管道
│   ├── data/                         # 反馈数据
│   ├── scripts/                      # Python 训练脚本
│   └── models/                       # 训练输出的 ONNX 模型
└── examples/                         # 示例代码
```

## 技术栈

### 核心技术
- **Rust**: 主要编程语言，提供性能和安全性
- **ONNX Runtime**: ML 推理引擎（通过 `ort` crate）
- **html5ever**: HTML 解析基础
- **cssparser**: CSS 解析工具
- **selectors**: CSS 选择器匹配

### AI/ML 技术栈
- **ONNX**: 开放神经网络交换格式
- **ort**: ONNX Runtime 的 Rust 绑定

## 快速开始

### 前置要求

- Rust 1.70 或更高版本
- Cargo（随 Rust 一起提供）

### 安装

1. 克隆仓库：
```bash
git clone https://github.com/vistone/BrowerAI.git
cd BrowerAI
```

2. 构建项目：
```bash
cargo build --release
```

3. 运行应用：
```bash
cargo run
```

### 运行测试

```bash
cargo test
```

## 模型库

BrowerAI 使用存储在 `models/local/` 中的本地模型库。系统支持以下模型类型：

- **HtmlParser**: HTML 结构理解模型
- **CssParser**: CSS 规则优化模型
- **JsParser**: JavaScript 分析模型
- **LayoutOptimizer**: 布局计算模型
- **RenderingOptimizer**: 渲染优化模型

### 添加模型

1. 将 ONNX 模型文件放入 `models/local/`
2. 创建或更新模型配置（参见 `models/model_config.toml`）
3. 模型管理器将自动加载和管理您的模型

模型配置示例：
```toml
[[models]]
name = "html_parser_v1"
model_type = "HtmlParser"
path = "html_parser_v1.onnx"
description = "基础 HTML 解析模型"
version = "1.0.0"
```

## 开发路线图

### 第一阶段：基础 ✅ 已完成
- [x] 项目结构设置
- [x] 基本 HTML/CSS/JS 解析器
- [x] ONNX Runtime 集成
- [x] 模型管理系统
- [x] 初始模型训练管道

### 第二阶段：AI 增强 ✅ 已完成
- [x] 训练 HTML 解析模型
- [x] 训练 CSS 优化模型
- [x] 训练 JavaScript 分析模型
- [x] 在解析器中实现模型推理

### 第三阶段：渲染 ✅ 已完成
- [x] AI 驱动的布局引擎
- [x] 智能渲染优化
- [x] 性能分析和调优

### 第四阶段：高级功能 ✅ 已完成
- [x] 实时学习和适应
- [x] 基于使用的模型微调
- [x] 多模型集成方法

### 第五阶段：学习与适应 ✅ 已完成
- [x] 反馈收集系统
- [x] 在线学习管道
- [x] 模型版本管理
- [x] A/B 测试框架
- [x] 自我优化
- [x] 用户个性化

## 学习资源

### HTML
- [HTML5 规范](https://html.spec.whatwg.org/)
- [html5ever 文档](https://docs.rs/html5ever/)

### CSS
- [CSS 规范](https://www.w3.org/Style/CSS/)
- [cssparser 文档](https://docs.rs/cssparser/)

### JavaScript
- [ECMAScript 规范](https://tc39.es/ecma262/)

### ONNX 和 ML
- [ONNX 文档](https://onnx.ai/)
- [ort Crate 文档](https://docs.rs/ort/)
- [ONNX Runtime](https://onnxruntime.ai/)

## 贡献

欢迎贡献！详情请参见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 许可证

MIT 许可证 - 详见 [LICENSE](../../LICENSE) 文件

## 致谢

- **pykeio/ort**: 优秀的 ONNX Runtime Rust 绑定
- **html5ever**: 强大的 HTML5 解析器
- **cssparser**: 来自 Servo 项目的 CSS 解析工具

## 未来愿景

BrowerAI 旨在创建一个浏览器，使得：
- AI 模型持续从网页内容模式中学习
- 通过机器学习优化解析和渲染
- 浏览器自主适应新的网络技术
- 通过强化学习不断提高性能

这是一个实验性项目，探索 AI 在网络浏览技术中的可能性边界。
