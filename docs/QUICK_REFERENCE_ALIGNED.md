# BrowerAI 快速参考 - 代码对齐版

**版本**: 1.0  
**日期**: 2026-02-17  
**用途**: 快速查阅关键代码位置和设计理念

---

## 🎯 核心口号

```
保功能、换体验 | Preserve Functionality, Change Experience
```

**代码体现**: `crates/browerai-intelligent-rendering/src/functional_transform.rs:23-38`

---

## 📁 关键代码位置速查

### 核心流程

| 功能 | 文件路径 | 关键函数/结构体 |
|------|----------|-----------------|
| 完整流水线 | `crates/browerai/src/main.rs:177-357` | `learn_and_generate()` |
| 功能转换管道 | `crates/browerai-intelligent-rendering/src/functional_transform.rs:23-38` | `FunctionalTransformPipeline` |
| 智能推理 | `crates/browerai-intelligent-rendering/src/reasoning.rs:87-106` | `IntelligentReasoning::reason()` |
| 7阶段分析 | `crates/browerai-js-analyzer/src/analysis_pipeline.rs:63-132` | `AnalysisPipeline::analyze()` |
| 真实网站学习 | `crates/browerai-learning/src/real_website_learner.rs:63-139` | `RealWebsiteLearner::learn_website()` |
| ONNX 推理 | `crates/browerai-ai-core/src/inference.rs` | `InferenceEngine::run()` |
| 热重载 | `crates/browerai-ai-core/src/hot_reload.rs` | `HotReloadManager::watch_and_reload()` |

### 反混淆策略

| 策略 | 文件路径 | 关键函数 |
|------|----------|----------|
| 字符串数组展开 | `crates/browerai-deobfuscation/src/enhanced_deobfuscation.rs:171-195` | `detect_string_array()` |
| 代理函数移除 | `crates/browerai-deobfuscation/src/enhanced_deobfuscation.rs:122-131` | `remove_proxy_functions()` |
| 控制流还原 | `crates/browerai-deobfuscation/src/control_flow_graph.rs` | `unflatten_control_flow()` |
| 常量折叠 | `crates/browerai-deobfuscation/src/enhanced_deobfuscation.rs:138-143` | `fold_constants()` |

---

## 🔢 关键数字

| 数字 | 含义 | 验证 |
|------|------|------|
| **27** | 模块化 crates 数量 | `Cargo.toml` workspace |
| **7** | JS 分析阶段数 | `analysis_pipeline.rs` |
| **18** | 反混淆策略数 | `browerai-deobfuscation/src/` |
| **17,542** | 真实训练样本数 | `training/scripts/` |
| **48** | 特征维度数 | `extract_features.py` |
| **50** | 训练 epochs | `training/config/` |
| **98.49%** | 模型验证准确率 | 训练日志 |
| **80%** | 功能保留阈值 | `model_orchestrator.rs:428` |
| **0.31s** | 增量编译时间 | 实测 |
| **700+** | 测试用例数 | `cargo test` |

---

## 🏗️ 架构分层速查

```
应用层
├── browerai-api-server (REST API)
├── browerai (CLI)
└── frontend (React)

学习层
├── browerai-learning
│   ├── real_website_learner.rs
│   ├── workflow_extractor.rs
│   └── learning_quality.rs
└── browerai-deobfuscation
    ├── enhanced_deobfuscation.rs
    ├── control_flow_graph.rs
    └── symbolic_executor.rs

AI 增强层
├── browerai-ai-core
│   ├── inference.rs (ONNX)
│   ├── hot_reload.rs
│   └── model_loader.rs
└── browerai-ai-integration

业务逻辑层
├── browerai-intelligent-rendering
│   ├── functional_transform.rs
│   ├── reasoning.rs
│   └── generation.rs
└── browerai-renderer-*

解析层
├── browerai-html-parser (html5ever)
├── browerai-css-parser (cssparser)
├── browerai-js-parser (Boa)
└── browerai-js-analyzer
    ├── scope_analyzer.rs      # Stage 1
    ├── swc_extractor.rs       # Stage 2
    ├── dataflow_analyzer.rs   # Stage 3
    ├── controlflow_analyzer.rs # Stage 4
    ├── enhanced_call_graph.rs # Stage 5
    ├── loop_analyzer.rs       # Stage 6
    └── analysis_pipeline.rs   # Stage 7

核心层
├── browerai-core (types, error, traits)
├── browerai-dom (DOM, Web APIs)
├── browerai-cache (multi-layer)
├── browerai-db (PostgreSQL)
└── browerai-metrics (Prometheus)
```

---

## 📝 常用命令

### 构建

```bash
# 完整构建（无 AI）
cargo build --release --workspace --exclude browerai-ml --exclude browerai-js-v8

# 带 AI 构建
cargo build --release --features ai

# 带 V8 构建
cargo build --release --features v8
```

### 测试

```bash
# 全部测试
cargo test --workspace --exclude browerai-ml --exclude browerai-js-v8

# 单 crate 测试
cargo test -p browerai-js-analyzer

# 特定测试
cargo test --test phase3_week3_enhanced_call_graph_tests
```

### 运行

```bash
# 学习并生成
cargo run --bin browerai -- learn https://example.com output/ 3

# API 服务器
cargo run --bin browerai-api-server

# 反混淆演示
cargo run --example enhanced_js_deobfuscation_demo
```

---

## 🎨 三种生成风格

| 风格 | 代码位置 | 特点 |
|------|----------|------|
| Modern | `crates/browerai-intelligent-rendering/src/lib.rs:14` | 卡片式、圆角、渐变 |
| Government | `crates/browerai-intelligent-rendering/src/lib.rs:16` | WCAG AAA、高对比度 |
| Minimalist | `crates/browerai-intelligent-rendering/src/lib.rs:18` | 最少装饰、纯功能 |

---

## 🔌 Feature Flags

| Feature | 依赖 | 用途 |
|---------|------|------|
| `ai` | `browerai-ai-core`, `ort` | 启用 ONNX 推理 |
| `ml` | `browerai-ml`, `tch` | 启用 ML 训练（需 LibTorch） |
| `v8` | `browerai-js-v8` | 启用 V8 引擎 |
| `onnx` | `ort` | 启用 ONNX Runtime |

---

## 📊 性能指标

| 指标 | 数值 | 测试条件 |
|------|------|----------|
| 增量编译 | 0.31s | 修改单 crate |
| 全量编译 | 1m 59s | Release 模式 |
| 测试通过率 | 100% | 700+ 测试 |
| ONNX 推理 | 35ms | fast_enhanced.onnx |
| 缓存加速比 | 53.77x | 多层缓存 |
| 内存占用 | <200MB | 完整配置 |

---

## 🔗 依赖关系速查

```
browerai (主入口)
├── browerai-core
├── browerai-learning
│   ├── browerai-js-analyzer
│   ├── browerai-deobfuscation
│   └── browerai-network
├── browerai-intelligent-rendering
│   ├── browerai-js-analyzer
│   ├── browerai-renderer-core
│   └── browerai-ai-core
└── browerai-api-server
    ├── browerai-learning
    └── browerai-intelligent-rendering
```

---

## 🐛 调试技巧

```bash
# 详细日志
RUST_LOG=debug cargo test test_name -- --nocapture

# 单步测试
cargo test --test phase3_week3_enhanced_call_graph_tests::test_recursive_chain_detection

# 性能分析
cargo bench

# 检查代码
cargo clippy --workspace --exclude browerai-ml --exclude browerai-js-v8
```

---

## 📚 相关文档

| 文档 | 路径 | 用途 |
|------|------|------|
| 核心设计哲学 | `docs/CORE_DESIGN_PHILOSOPHY_ALIGNED.md` | 设计理念 |
| 架构设计 | `docs/ARCHITECTURE_CODE_ALIGNED.md` | 架构详情 |
| 设计决策 | `docs/DESIGN_DECISIONS_ALIGNED.md` | 决策记录 |
| 学习路径 | `docs/LEARNING_PATH.md` | 学习指南 |
| 项目结构 | `PROJECT_STRUCTURE.md` | 项目概览 |

---

**最后更新**: 2026-02-17  
**代码对齐状态**: ✅ 已验证
