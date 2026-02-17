# models/onnx_exports/

ONNX模型导出目录，用于Rust端推理集成。

## 导出的模型

1. **selector_embedding.onnx** - 选择器嵌入模型（2.83M参数）
   - 输入：`[batch, seq_len]` int64 token序列
   - 输出：`[batch, seq_len, 128]` 嵌入向量

2. **property_predictor.onnx** - 属性预测模型（2.66M参数）
   - 输入：`[batch, 10, 128]` 嵌入序列
   - 输出：`[batch, 50]` 属性概率（sigmoid）

3. **color_model.onnx** - 颜色学习模型（4.40M参数）
   - 输入：`[batch, 3, 32, 32]` RGB图像
   - 输出：`[batch, num_classes]` 颜色分类

4. **complete_model.onnx** - 完整页面模型（1.65M参数）
   - 输入：`[batch, seq_len, 256]` 特征序列
   - 输出：`[batch, output_dim]` 统一表示

5. **finetuned_model.onnx** - 微调模型（0.27M参数）
   - 输入：`[batch, 512]` 特征向量
   - 输出：`[batch, 512]` LoRA增强表示

## 导出方法

```bash
cd /home/stone/BrowerAI
python training/models/export_to_onnx.py --output models/onnx_exports/
```

## Rust集成

在 `models/model_config.toml` 中注册ONNX模型：

```toml
[[models]]
name = "selector_embedding_v2"
model_type = "SelectorEmbedding"
path = "onnx_exports/selector_embedding.onnx"
version = "2.0.0"
format = "onnx"

[[models]]
name = "property_predictor_v2"
model_type = "PropertyPredictor"
path = "onnx_exports/property_predictor.onnx"
version = "2.0.0"
format = "onnx"
```

加载示例（Rust）：

```rust
use browerai::ai::InferenceEngine;

let engine = InferenceEngine::from_config("models/model_config.toml")?;
let selector_tokens = vec![1, 5, 10, 3]; // tokenized selector
let embeddings = engine.infer("selector_embedding_v2", &selector_tokens)?;
```
