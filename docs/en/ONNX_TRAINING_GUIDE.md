# BrowerAI ONNX 模型库和训练指南

## 📦 需要的模型库

BrowerAI 需要两层模型库架构：

### 1. 训练端（Python）

**位置**: `training/`

**目的**: 使用 PyTorch 训练模型并导出为 ONNX 格式

**核心库**:
- **PyTorch** (⭐⭐⭐⭐⭐ 推荐): 训练深度学习模型
- **ONNX**: 模型格式标准
- **ONNXRuntime**: Python 端模型验证

**为什么选 PyTorch**:
- ✅ 最成熟的 Python ML 框架
- ✅ ONNX 导出简单 (`torch.onnx.export`)
- ✅ 生态系统丰富（预训练模型、工具）
- ✅ 社区支持最好
- ✅ 调试友好

**替代方案**:
- TensorFlow + tf2onnx（也不错，但生态略弱）
- scikit-learn + skl2onnx（适合传统 ML）

### 2. 推理端（Rust）

**位置**: `src/ai/`, `models/local/`

**目的**: 加载 ONNX 模型并进行高速推理

**核心库**:
- **ort** (⭐⭐⭐⭐⭐ 当前使用): ONNX Runtime for Rust
  - GitHub: https://github.com/pykeio/ort
  - 文档: https://docs.rs/ort/

**为什么选 ort**:
- ✅ 官方 ONNX Runtime 的 Rust 绑定
- ✅ Microsoft 支持，稳定可靠
- ✅ CPU/GPU 加速
- ✅ API 友好，类型安全
- ✅ 活跃维护

**替代方案**:
- `tract` (Pure Rust ML 推理): 无需 C++ 依赖，但模型支持有限

## 🎯 模型类型设计

### 1. HTML 复杂度预测器

**任务**: 回归（输出 0.0-1.0 复杂度评分）

**输入特征** (100 维):
```
- 标签数量（归一化）
- 嵌套深度
- 文本长度
- 表格/表单数量
- 多媒体元素数量
- class/id 使用情况
- 语义标签占比
- ...
```

**输出**: `complexity: f32` (0.0-1.0)

**训练脚本**: `training/scripts/train_html_complexity.py`

### 2. CSS 优化建议生成器

**任务**: 多标签分类（5 个优化类别）

**输入特征** (80 维):
```
- 规则数量
- 选择器复杂度
- 重复属性数量
- 未使用选择器数量
- 颜色格式统计
- ...
```

**输出**: `suggestions: [f32; 5]`
- [0]: 合并重复规则
- [1]: 简化选择器
- [2]: 删除未使用
- [3]: 优化颜色值
- [4]: 压缩属性

**训练脚本**: `training/scripts/train_css_optimizer.py`

### 3. JS 模式识别器（未来）

**任务**: 多分类（检测代码模式）

**输出**: `patterns: Vec<String>`
- "event_driven"
- "promise_chain"
- "async_await"
- "callback_hell"
- ...

## 🛠️ ONNX 工具链推荐

### Python 端

```bash
# 核心依赖
pip install torch onnx onnxruntime

# 工具链
pip install onnx-simplifier  # 模型优化
pip install netron           # 可视化
pip install onnxoptimizer    # 额外优化
```

**常用命令**:

```bash
# 1. 验证 ONNX 模型
python -c "import onnx; onnx.checker.check_model('model.onnx')"

# 2. 简化模型（减小体积）
python -m onnxsim model.onnx model_simplified.onnx

# 3. 可视化模型结构
netron model.onnx  # 打开浏览器

# 4. 查看模型信息
python -c "
import onnx
model = onnx.load('model.onnx')
print('输入:', [(i.name, i.type) for i in model.graph.input])
print('输出:', [(o.name, o.type) for o in model.graph.output])
"
```

### Rust 端

```toml
# Cargo.toml
[dependencies]
ort = { version = "2.0.0-rc.10", optional = true }

[features]
ai = ["ort"]
```

**使用示例** (已在 `src/ai/inference.rs`):

```rust
use ort::{Session, Value};

// 加载模型
let session = Session::builder()?
    .with_optimization_level(GraphOptimizationLevel::Level3)?
    .commit_from_file("models/local/model.onnx")?;

// 推理
let input = ndarray::Array::from_shape_vec((1, 100), features)?;
let outputs = session.run(ort::inputs!["features" => input.view()]?)?;
let result: f32 = outputs["complexity"].try_extract_scalar()?;
```

## 📚 完整训练流程

### 步骤 1: 环境准备

```bash
cd training
./setup_env.sh  # 自动安装依赖
```

### 步骤 2: 数据收集

```bash
cd ..
cargo run -- --learn https://example.com https://github.com
```

反馈数据保存到 `training/data/feedback_*.json`

### 步骤 3: 训练模型

```bash
cd training/scripts

# HTML 复杂度
python train_html_complexity.py \
    --data ../data/feedback_*.json \
    --epochs 100 \
    --output ../models/html_complexity_v1.onnx

# CSS 优化
python train_css_optimizer.py \
    --data ../data/feedback_*.json \
    --epochs 100 \
    --output ../models/css_optimizer_v1.onnx
```

### 步骤 4: 验证模型

```bash
python validate_model.py ../models/html_complexity_v1.onnx --benchmark
```

输出：
```
✅ ONNX 格式验证通过
✅ 平均推理时间: 0.234 ms
✅ 吞吐量: 4273.5 次/秒
```

### 步骤 5: 部署模型

```bash
cd ../..

# 复制模型
cp training/models/*.onnx models/local/

# 更新配置
cat >> models/model_config.toml << EOF
[[models]]
name = "html_complexity_v1"
model_type = "HtmlParser"
path = "html_complexity_v1.onnx"
version = "1.0.0"
enabled = true
