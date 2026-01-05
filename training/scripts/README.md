# BrowerAI 训练脚本

本目录包含用于训练 ONNX 模型的 Python 脚本。

## 📦 安装依赖

```bash
cd training
pip install -r requirements.txt
```

## 🎓 训练脚本

### 1. HTML 复杂度预测模型

训练模型预测 HTML 文档的复杂度（0.0-1.0）：

```bash
python scripts/train_html_complexity.py \
    --data ../data/feedback_*.json \
    --epochs 100 \
    --batch-size 32 \
    --lr 0.001 \
    --output ../models/html_complexity_v1.onnx
```

**参数说明**:
- `--data`: 反馈数据文件模式（支持通配符）
- `--epochs`: 训练轮数（默认 100）
- `--batch-size`: 批次大小（默认 32）
- `--lr`: 学习率（默认 0.001）
- `--val-split`: 验证集比例（默认 0.2）
- `--output`: ONNX 输出路径

**输出**:
- `../models/html_complexity_v1.onnx` - ONNX 模型
- `../models/html_complexity_best.pth` - PyTorch 最佳权重

### 2. CSS 优化建议模型

训练模型生成 CSS 优化建议（多标签分类）：

```bash
python scripts/train_css_optimizer.py \
    --data ../data/feedback_*.json \
    --epochs 100 \
    --output ../models/css_optimizer_v1.onnx
```

**优化建议类别**:
- 合并重复规则
- 简化选择器
- 删除未使用选择器
- 优化颜色值
- 压缩属性

### 3. 其他模型

参考现有脚本模板创建：
- `train_js_analyzer.py` - JS 代码模式识别
- `train_layout_optimizer.py` - 布局优化
- `train_render_optimizer.py` - 渲染优化

## 🧪 验证模型

使用验证脚本测试训练好的模型：

```bash
# 基础验证
python scripts/validate_model.py ../models/html_complexity_v1.onnx

# 包含性能测试
python scripts/validate_model.py ../models/html_complexity_v1.onnx --benchmark --runs 1000
```

**验证内容**:
- ✅ ONNX 格式正确性
- ✅ 输入/输出形状
- ✅ 推理功能
- ⚡ 性能基准（推理时间）

## 📊 数据准备

### 收集训练数据

首先运行 BrowerAI 收集反馈数据：

```bash
cd ../..
cargo run --bin browerai -- --learn https://example.com https://github.com
```

反馈数据会自动保存到 `training/data/feedback_*.json`。

### 数据格式

反馈数据是 JSON 数组：

```json
[
  {
    "type": "html_parsing",
    "timestamp": "2026-01-04T10:38:39Z",
    "success": true,
    "ai_used": true,
    "complexity": 0.5,
    "error": null
  },
  {
    "type": "css_parsing",
    "timestamp": "2026-01-04T10:38:39Z",
    "success": true,
    "ai_used": true,
    "rule_count": 7,
    "error": null
  }
]
```

### 推荐数据量

| 模型类型 | 最少样本 | 推荐样本 | 说明 |
|---------|---------|---------|------|
| HTML 复杂度 | 100 | 1,000+ | 访问 10+ 网站 |
| CSS 优化 | 50 | 500+ | 访问 5+ 有 CSS 的网站 |
| JS 分析 | 50 | 500+ | 访问 5+ 有 JS 的网站 |

## 🚀 部署模型

### 1. 复制模型到部署目录

```bash
cp models/html_complexity_v1.onnx ../../models/local/
cp models/css_optimizer_v1.onnx ../../models/local/
```

### 2. 更新模型配置

编辑 `../../models/model_config.toml`:

```toml
[[models]]
name = "html_complexity_v1"
model_type = "HtmlParser"
path = "html_complexity_v1.onnx"
version = "1.0.0"
enabled = true

[[models]]
name = "css_optimizer_v1"
model_type = "CssParser"
path = "css_optimizer_v1.onnx"
version = "1.0.0"
enabled = true
```

### 3. 重新编译启用 AI 特性

```bash
cd ../..
cargo build --release --features ai
```

### 4. 测试效果

```bash
# 查看 AI 状态
cargo run --release -- --ai-report

# 测试真实网站
cargo run --release -- --learn https://example.com
```

## 📈 训练技巧

### 数据不足时

如果训练数据 < 100 样本，建议：

1. **收集更多数据**
   ```bash
   # 批量访问网站
   cargo run -- --learn \
       https://example.com \
       https://github.com \
       https://rust-lang.org \
       https://developer.mozilla.org
   ```

2. **使用预训练模型微调**（未来支持）

3. **数据增强**
   - 添加噪声
   - 特征随机遮挡
   - 时间序列扰动

### 优化训练

**过拟合**:
- 增加 Dropout 概率
- 减少模型层数
- 增加数据量
- 使用正则化

**欠拟合**:
- 增加模型容量
- 降低学习率
- 增加训练轮数
- 检查特征质量

**训练慢**:
- 使用 GPU (`--device cuda`)
- 增加批次大小
- 减少模型参数
- 使用混合精度训练

## 🔧 高级用法

### 自定义特征提取

编辑 `extract_html_features()` 函数：

```python
def extract_html_features(event: dict) -> Tuple[List[float], float]:
    features = []
    
    # 添加自定义特征
    features.append(calculate_dom_depth(event))
    features.append(count_semantic_tags(event))
    features.append(estimate_interactivity(event))
    
    # ... 更多特征
    
    return features, label
```

### 自定义模型架构

```python
class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 例如：Transformer 编码器
        self.encoder = nn.TransformerEncoder(...)
        self.fc = nn.Linear(...)
    
    def forward(self, x):
        x = self.encoder(x)
        return self.fc(x)
```

### 导出优化模型

```bash
# 简化 ONNX 模型（减小体积）
python -m onnxsim model.onnx model_simplified.onnx

# 可视化模型结构
pip install netron
netron model.onnx
```

## 📚 参考资源

- [PyTorch ONNX Export](https://pytorch.org/docs/stable/onnx.html)
- [ONNX Runtime](https://onnxruntime.ai/)
- [ort Rust 库](https://docs.rs/ort/)
- [ONNX Model Zoo](https://github.com/onnx/models)

## 🐛 常见问题

**Q: ModuleNotFoundError: No module named 'torch'**  
A: 安装依赖 `pip install -r requirements.txt`

**Q: 训练数据不足**  
A: 运行 `cargo run -- --learn` 收集更多网站数据

**Q: ONNX 导出失败**  
A: 检查模型是否包含不支持的操作，使用 `opset_version=14`

**Q: Rust 端加载模型失败**  
A: 确保编译时启用了 `--features ai`，检查模型路径和配置

**Q: 推理速度慢**  
A: 
- 使用 `--release` 编译
- 简化模型（减少参数）
- 使用 ONNX 优化工具

---

需要帮助？查看 [LEARNING_GUIDE.md](../../LEARNING_GUIDE.md) 或提交 Issue。
