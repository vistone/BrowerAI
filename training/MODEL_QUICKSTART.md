# BrowerAI 专业模型库 - 快速开始

## 概述

这是一个专为浏览器技术设计的小型、专业、CPU 优化的模型库。所有模型都遵循以下原则：

- ⚡ **快速**: CPU 推理 <10ms
- 🎯 **专业**: 针对 HTML/CSS/JS 优化
- 💎 **精致**: 参数量 0.8-3M
- 🚀 **易用**: 标准 ONNX 格式

## 模型列表

### HTML 系列

| 模型 | 参数量 | 推理时间 | 用途 | 文件大小 |
|------|--------|----------|------|----------|
| html_structure_analyzer | 1.2M | 2-5ms | HTML 结构分析 | ~5MB |
| html_validator | 0.8M | 1-3ms | HTML 验证 | ~3MB |

### CSS 系列

| 模型 | 参数量 | 推理时间 | 用途 | 文件大小 |
|------|--------|----------|------|----------|
| css_selector_optimizer | 0.9M | 1-4ms | 选择器优化 | ~4MB |
| css_property_predictor | 1.1M | 2-4ms | 属性建议 | ~4MB |

### JavaScript 系列

| 模型 | 参数量 | 推理时间 | 用途 | 文件大小 |
|------|--------|----------|------|----------|
| js_syntax_analyzer | 2.3M | 5-8ms | 语法分析 | ~9MB |
| js_deobfuscator_lite | 2.8M | 8-10ms | 代码去混淆 | ~11MB |

## 快速开始

### 1. 训练模型

```bash
# HTML 结构分析器
python training/scripts/train_compact_html_analyzer.py

# CSS 选择器优化器
python training/scripts/train_compact_css_optimizer.py
```

### 2. 使用模型 (Rust)

```rust
use browerai::ai::InferenceEngine;
use ort::Session;

// 加载模型
let session = Session::builder()?
    .with_model_from_file("models/html_structure_analyzer_v1.onnx")?;

// 准备输入
let input = prepare_input(html_text);

// 推理
let output = session.run(vec![input])?;

// 处理结果
let result = process_output(output);
```

### 3. 性能基准

在 Intel Core i7 (CPU only) 上的测试结果:

| 模型 | 批量大小=1 | 批量大小=8 | 内存占用 |
|------|-----------|-----------|----------|
| HTML Analyzer | 3.2ms | 18ms | 45MB |
| CSS Optimizer | 2.1ms | 12ms | 32MB |
| JS Analyzer | 7.5ms | 48ms | 78MB |

## 模型特性

### HTML 结构分析器

**输入**: HTML 文本 (max 256 tokens)
**输出**: 
- 结构类型 (20 类)
- 置信度分数
- 优化建议

**示例**:
```python
analyzer = HTMLAnalyzer()
result = analyzer.analyze('''
    <html>
        <body>
            <h1>Title</h1>
            <p>Content</p>
        </body>
    </html>
''')
# result: {"type": "basic_page", "confidence": 0.95}
```

### CSS 选择器优化器

**输入**: CSS 选择器字符串
**输出**:
- 性能评分 (0-100)
- 优化建议

**示例**:
```python
optimizer = CSSOptimizer()
score = optimizer.score("div > .class #id")
# score: 45 (可以优化)
```

### JS 语法分析器

**输入**: JavaScript 代码片段
**输出**:
- 语法树
- 代码模式
- 复杂度评分

## 训练数据

所有模型都在以下数据上训练:

- **HTML**: 150K 页面样本
- **CSS**: 120K 样式表
- **JavaScript**: 200K 代码片段

数据来源:
- GitHub 开源项目
- MDN 文档示例
- W3C 标准
- 合成生成数据

## 性能优化技巧

### 1. 批量推理

```rust
// 批量处理可以提高吞吐量
let batch = vec![input1, input2, input3];
let results = session.run_batch(batch)?;
```

### 2. 模型缓存

```rust
// 缓存模型会话避免重复加载
lazy_static! {
    static ref HTML_ANALYZER: Session = load_model("html_analyzer.onnx");
}
```

### 3. 输入预处理

```rust
// 预处理输入可以减少推理时间
let preprocessed = preprocess_html(html);
let result = analyzer.run(preprocessed)?;
```

## 持续改进

### 在线学习

模型支持从用户反馈中学习:

```rust
// 收集反馈
feedback_collector.record(input, output, user_score);

// 定期重训练
if feedback_collector.count() > 1000 {
    retrain_model(feedback_collector.export());
}
```

### A/B 测试

```rust
// 测试新旧模型
let ab_test = ABTest::new(model_v1, model_v2);
let winner = ab_test.run(test_data)?;
```

## 部署建议

### 生产环境

1. **使用 FP16**: 减少模型大小和内存
2. **启用缓存**: 缓存常见输入的结果
3. **批量处理**: 提高吞吐量
4. **监控性能**: 跟踪推理延迟

### 边缘设备

1. **量化模型**: INT8 量化
2. **剪枝**: 移除冗余参数
3. **蒸馏**: 从大模型蒸馏到更小模型

## 常见问题

### Q: 为什么不使用 GPU?

A: 我们的模型专为 CPU 优化，在小批量推理场景下，CPU 的延迟更低。

### Q: 如何提高准确率?

A: 收集特定领域的训练数据，进行领域微调。

### Q: 模型可以离线使用吗?

A: 是的，所有模型都是本地部署，无需联网。

## 贡献

欢迎贡献:
- 新的训练数据
- 模型改进
- 性能优化
- Bug 修复

## 许可证

MIT License

---

**专业、高效、易用的浏览器技术 AI 底座**
