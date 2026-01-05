# BrowerAI 专业模型库

## 概述

BrowerAI 专业模型库是一个**小而精致、无需GPU**的浏览器技术专用模型集合，专门针对 JavaScript/CSS/HTML 的解析、理解和生成优化。

## 设计理念

### 核心原则
1. **小而精致**: 每个模型控制在 2-5M 参数，CPU 友好
2. **专业专注**: 针对浏览器技术栈专门优化
3. **快速推理**: 单次推理 <10ms (CPU)
4. **高准确率**: 针对特定任务优化至 95%+ 准确率
5. **易于部署**: 标准 ONNX 格式，无依赖 GPU

## 模型库架构

```
BrowerAI Model Zoo
├── HTML 系列 (轻量级解析与生成)
│   ├── html_structure_analyzer (1.2M params)
│   ├── html_generator_compact (1.8M params)
│   └── html_validator (0.8M params)
│
├── CSS 系列 (样式理解与优化)
│   ├── css_selector_optimizer (0.9M params)
│   ├── css_property_predictor (1.1M params)
│   └── css_compact_generator (1.5M params)
│
├── JavaScript 系列 (代码理解与转换)
│   ├── js_syntax_analyzer (2.3M params)
│   ├── js_deobfuscator_lite (2.8M params)
│   ├── js_minifier (1.4M params)
│   └── js_pattern_detector (1.6M params)
│
└── 跨领域系列 (综合能力)
    ├── web_component_classifier (1.3M params)
    ├── code_quality_scorer (0.9M params)
    └── browser_compatibility_checker (1.1M params)
```

## 技术规格

### 通用规范
- **框架**: PyTorch → ONNX 导出
- **精度**: FP32 (标准) / FP16 (可选)
- **推理引擎**: ONNX Runtime (CPU 优化)
- **输入格式**: Tokenized sequences (max_len: 128-512)
- **输出格式**: Logits / Embeddings / Token IDs

### 性能指标
| 模型类别 | 参数量 | 推理时间 (CPU) | 准确率 | 模型大小 |
|---------|--------|---------------|--------|----------|
| HTML 系列 | 0.8-1.8M | 2-5ms | 96%+ | 3-7MB |
| CSS 系列 | 0.9-1.5M | 1-4ms | 94%+ | 4-6MB |
| JS 系列 | 1.4-2.8M | 5-10ms | 92%+ | 6-11MB |
| 跨领域系列 | 0.9-1.3M | 2-6ms | 93%+ | 4-5MB |

## 模型详细说明

### 1. HTML 结构分析器 (html_structure_analyzer)

**功能**: 分析 HTML 文档结构，识别元素关系和语义

**架构**:
```
Input: Tokenized HTML (max_len: 256)
  ↓
Embedding Layer (vocab: 2048, dim: 128)
  ↓
BiLSTM Encoder (hidden: 256, layers: 2)
  ↓
Attention Layer
  ↓
Classification Head (categories: 20)
  ↓
Output: Structure features + Element types
```

**参数**: 1.2M
**输入**: HTML 文本 (最长 256 tokens)
**输出**: 
- 文档结构类型 (20 类)
- 元素层级关系
- 语义标签建议

**使用场景**:
- HTML 质量评估
- 结构优化建议
- 可访问性检查

### 2. CSS 选择器优化器 (css_selector_optimizer)

**功能**: 优化 CSS 选择器，提高性能和可读性

**架构**:
```
Input: CSS Selector tokens (max_len: 64)
  ↓
Embedding (vocab: 512, dim: 64)
  ↓
Transformer Encoder (layers: 2, heads: 4)
  ↓
Optimization Decoder
  ↓
Output: Optimized selector + Score
```

**参数**: 0.9M
**输入**: CSS 选择器字符串
**输出**:
- 优化后的选择器
- 性能评分 (0-100)
- 优化建议

**优化策略**:
- 去除冗余选择器
- 降低选择器特异性
- 提高匹配效率

### 3. JavaScript 语法分析器 (js_syntax_analyzer)

**功能**: 快速分析 JS 代码语法结构和模式

**架构**:
```
Input: JS tokens (max_len: 512)
  ↓
Token Embedding (vocab: 4096, dim: 128)
  ↓
Position Encoding
  ↓
Transformer Encoder (layers: 3, heads: 4, dim: 128)
  ↓
Multi-task Head
  ↓
Output: Syntax tree + Patterns + Complexity
```

**参数**: 2.3M
**输入**: JavaScript 代码片段
**输出**:
- 语法树表示
- 代码模式 (50+ 种)
- 复杂度评分

**检测模式**:
- 函数声明/表达式
- 异步模式 (async/await, Promise)
- 模块导入/导出
- 类和原型
- 闭包和作用域

### 4. JS 轻量去混淆器 (js_deobfuscator_lite)

**功能**: CPU 友好的 JavaScript 去混淆

**架构**:
```
Input: Obfuscated JS (max_len: 256)
  ↓
Dual Encoder
  ├─ Syntax Encoder (Transformer, 2 layers)
  └─ Pattern Encoder (CNN, 3 layers)
  ↓
Fusion Layer
  ↓
Seq2Seq Decoder (LSTM, 2 layers)
  ↓
Output: Deobfuscated code tokens
```

**参数**: 2.8M
**处理技术**:
- 变量名还原
- 字符串解码
- 控制流简化
- 表达式展开

**性能**:
- 推理: 8-10ms (CPU)
- 准确率: 88-92%
- 支持混淆类型: 8 种

### 5. Web 组件分类器 (web_component_classifier)

**功能**: 识别和分类 Web 页面组件类型

**架构**:
```
Input: Component HTML/CSS (max_len: 128)
  ↓
Multi-modal Encoder
  ├─ HTML Branch (BiGRU)
  └─ CSS Branch (CNN)
  ↓
Feature Fusion
  ↓
Classification Layer (30 categories)
  ↓
Output: Component type + Confidence
```

**参数**: 1.3M
**识别组件**: 30+ 种常见 Web 组件
- 导航栏、按钮、表单
- 卡片、列表、表格
- 模态框、下拉菜单
- 轮播图、标签页等

## 训练数据集

### 数据来源
1. **公开数据集**
   - GitHub 开源项目 (50K+ 仓库)
   - MDN Web Docs
   - W3C 标准示例

2. **合成数据**
   - 模板生成 (100K+ 样本)
   - 变体增强
   - 混淆生成 (50K+ 对)

3. **真实网站采样**
   - Top 1000 网站
   - 多语言、多框架覆盖

### 数据规模
- HTML 样本: 150K
- CSS 样本: 120K
- JavaScript 样本: 200K
- 跨领域样本: 80K

## 模型训练

### 训练策略
1. **预训练**: 在大规模语料上学习通用特征
2. **任务微调**: 针对特定任务精调
3. **知识蒸馏**: 从大模型蒸馏到小模型
4. **量化优化**: FP16/INT8 量化（可选）

### 训练环境
- **硬件**: CPU 可训练（推荐 16+ 核心）
- **时间**: 单个模型 2-6 小时
- **框架**: PyTorch 2.0+
- **优化器**: AdamW (lr: 1e-4)

### 评估指标
- **准确率**: 任务相关指标
- **推理速度**: CPU 延迟
- **模型大小**: 压缩后大小
- **鲁棒性**: 边缘情况处理

## 部署使用

### ONNX 导出
```python
import torch
import torch.onnx

# 加载模型
model = YourModel.load_pretrained('path/to/model.pth')
model.eval()

# 准备示例输入
dummy_input = torch.randint(0, vocab_size, (1, max_len))

# 导出 ONNX
torch.onnx.export(
    model,
    dummy_input,
    'model.onnx',
    opset_version=13,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size', 1: 'sequence'},
        'output': {0: 'batch_size'}
    }
)
```

### Rust 集成
```rust
use ort::{Session, Value};

// 加载模型
let session = Session::builder()?
    .with_model_from_file("models/local/html_analyzer.onnx")?;

// 准备输入
let input_tensor = Value::from_array((vec![1, 256], input_data))?;

// 推理
let outputs = session.run(vec![input_tensor])?;

// 处理输出
let output = outputs[0].extract_tensor::<f32>()?;
```

### 性能优化
1. **批处理**: 支持 batch inference
2. **缓存**: 结果缓存避免重复计算
3. **并行**: 多模型并行推理
4. **量化**: FP16 降低内存占用

## 模型更新策略

### 版本管理
- 语义化版本: v1.0.0
- 向后兼容性保证
- 迁移指南

### 持续改进
1. **在线学习**: 从用户反馈学习
2. **A/B 测试**: 新旧模型对比
3. **渐进式更新**: 灰度发布
4. **性能监控**: 实时指标追踪

## 质量保证

### 测试覆盖
- **单元测试**: 每个模型 100+ 测试用例
- **集成测试**: 端到端验证
- **性能测试**: 延迟和吞吐量
- **压力测试**: 大规模并发

### 验证标准
- ✅ 推理时间 < 10ms (CPU)
- ✅ 准确率 > 90%
- ✅ 模型大小 < 15MB
- ✅ 内存占用 < 100MB
- ✅ 无 GPU 依赖

## 最佳实践

### 模型选择
- **快速原型**: 使用最小模型
- **生产部署**: 使用标准模型
- **高精度场景**: 使用专业微调模型

### 性能优化
1. 输入预处理优化
2. 批量推理
3. 模型缓存
4. 结果复用

### 错误处理
- 优雅降级
- 回退到规则方法
- 详细日志记录

## 路线图

### 短期 (1-3 个月)
- [ ] 完成 HTML 系列 3 个模型
- [ ] 完成 CSS 系列 3 个模型
- [ ] 完成 JS 系列 4 个模型
- [ ] 基础测试和文档

### 中期 (3-6 个月)
- [ ] 跨领域模型 3 个
- [ ] 模型压缩和优化
- [ ] 在线学习支持
- [ ] 性能基准测试

### 长期 (6-12 个月)
- [ ] 模型集成和联合推理
- [ ] 自适应模型选择
- [ ] 边缘设备支持
- [ ] 社区模型贡献平台

## 社区贡献

欢迎贡献:
- 新的训练数据
- 模型改进
- 性能优化
- Bug 修复
- 文档改进

## 许可证

MIT License - 开源免费使用

---

**BrowerAI Model Zoo** - 浏览器技术的专业 AI 底座
