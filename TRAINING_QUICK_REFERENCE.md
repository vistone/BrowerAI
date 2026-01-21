# Training 目录快速参考

## 📊 一览表

```
training/                      总计 116 MB
├── data/                      105 MB    ⭐ 训练数据
├── models/                    8.8 MB    ⭐ 模型文件
├── features/                  1.5 MB    特征数据
├── logs/                      672 KB    日志
├── scripts/                   396 KB    脚本 (3,507 行)
├── core/                      372 KB    模块 (3,116 行)
├── semantic_learning/         172 KB    语义学习
├── venv/                      36 KB     虚拟环境
└── configs/                   20 KB     配置
```

---

## 🎯 核心功能模块

### 1. 数据 (core/data/)
```python
from core.data.tokenizers import Tokenizer
from core.data.website_dataset import WebsiteDataset

# 编码: 字符级 (229 字符表)
tokenizer = Tokenizer()
tokens = tokenizer.encode(html_code)

# 数据集: PyTorch 兼容
dataset = WebsiteDataset(
    data_file='data/website_paired.jsonl',
    tokenizer=tokenizer
)
```

### 2. 模型 (core/models/)
```python
from core.models.website_learner import WebsiteLearner

# 架构: Transformer Encoder-Decoder
model = WebsiteLearner(
    vocab_size=229,
    d_model=256,
    nhead=8,
    num_layers=3
)

# 输入: 原始网站代码
# 输出: 简化优化版本
output = model(input_tokens)
```

### 3. 训练 (core/trainers/)
```python
from core.trainers.trainer import Trainer

trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=30,
    batch_size=2,
    lr=0.001
)
trainer.train()
```

---

## 📁 数据集结构

### 输入: 原始网站代码
```json
{
  "html": "<html>...</html>",
  "css": "body { ... }",
  "js": "function(...) { ... }",
  "url": "https://example.com"
}
```

### 输出: 简化版本
```json
{
  "html_original": "...",
  "html_simplified": "...",  // 压缩、优化
  "css_original": "...",
  "css_simplified": "...",
  "js_original": "...",
  "js_simplified": "..."     // 解混淆、简化
}
```

---

## 🚀 快速开始 (5 步)

### 1️⃣ 环境准备
```bash
cd training
pip install -r requirements.txt
```

### 2️⃣ 查看数据
```bash
ls -lh data/websites/
# 142 行: 1000_sites.jsonl (大规模)
# 13 行:  quick_train.jsonl (快速)
```

### 3️⃣ 生成配对数据
```bash
python scripts/create_simplified_dataset.py \
  --input data/website_complete.jsonl \
  --output data/website_paired.jsonl
```

### 4️⃣ 训练模型
```bash
python scripts/train_paired_website_generator.py
# 输出: checkpoints/paired_generator/epoch_*.pt
```

### 5️⃣ 导出 ONNX
```bash
python scripts/export_to_onnx.py \
  --checkpoint checkpoints/paired_generator/epoch_30.pt \
  --output ../models/local/website_generator_v1.onnx
```

---

## 📋 脚本使用场景

| 脚本 | 用途 | 输入 | 输出 |
|------|------|------|------|
| `batch_crawl_websites.py` | 爬取新网站 | URLs | websites/*.jsonl |
| `extract_website_complete.py` | 完整网站提取 | 原始数据 | website_complete.jsonl |
| `create_simplified_dataset.py` | 生成简化版 | website_complete.jsonl | website_paired.jsonl |
| `train_paired_website_generator.py` | 训练模型 | website_paired.jsonl | checkpoints/*.pt |
| `export_to_onnx.py` | 导出模型 | checkpoints/*.pt | *.onnx |
| `extract_features.py` | 特征提取 | 代码 | 特征向量 |
| `dataset_manager.py` | 数据管理 | 数据文件 | 统计/验证 |

---

## 🔍 关键文件位置

```
training/
├── scripts/
│   ├── train_paired_website_generator.py  ⭐ 主训练脚本
│   └── export_to_onnx.py                  ⭐ ONNX 导出
├── core/
│   ├── models/
│   │   ├── website_learner.py             ⭐ 主模型
│   │   └── transformer.py
│   ├── data/
│   │   ├── tokenizers.py                  字符编码
│   │   └── website_dataset.py             数据集
│   └── trainers/
│       └── trainer.py                     训练器
├── data/
│   ├── websites/1000_sites.jsonl          ⭐ 大数据集
│   ├── website_complete.jsonl             完整网站
│   └── website_paired.jsonl               配对数据
├── models/
│   └── *.onnx.data                        ONNX 模型
├── configs/
│   └── website_learner.yaml               ⭐ 训练配置
└── README.md, QUICKSTART.md               文档
```

---

## ⚙️ 配置参数

### 模型配置 (website_learner.yaml)
```yaml
# 架构
model:
  d_model: 256           # 嵌入维度
  nhead: 8               # 注意力头数
  num_layers: 3          # 编码器/解码器层数
  vocab_size: 229        # 字符表大小

# 训练
training:
  batch_size: 2          # 批大小
  epochs: 30             # 训练轮数
  learning_rate: 0.001   # 学习率
  
# 数据
data:
  max_length: 5000       # 最大序列长度
  validation_split: 0.1  # 验证集比例
```

---

## 📊 性能指标

### 预期效果 (训练后)
- **BLEU 分数**: 0.70+ (相似度)
- **语法正确**: 95%+
- **语义保留**: 99%+
- **压缩率**: 72.95% (代码量)

### 训练时间
- **数据规模**: 13-142 网站
- **训练时间**: 2-3 小时 (GPU)
- **收敛周期**: 20-30 epochs

---

## 🔗 集成到 Rust

### 1. 导出模型
```bash
python scripts/export_to_onnx.py \
  --checkpoint checkpoints/paired_generator/epoch_30.pt \
  --output ../models/local/website_generator.onnx
```

### 2. Rust 中使用
```rust
use browerai_ai_core::inference::InferenceEngine;

let engine = InferenceEngine::new(
    "models/local/website_generator.onnx"
)?;

let input = tokenizer.encode(original_html)?;
let output = engine.infer(&input)?;
let simplified_html = tokenizer.decode(&output)?;
```

---

## ⚠️ 常见问题

### Q: 数据文件太大怎么办?
**A**: 数据已在 .gitignore 中排除，只需本地使用
```bash
# 数据不会被提交
git status | grep "websites/"  # 应该显示: 排除的文件
```

### Q: 如何加载已训练模型?
**A**: 
```python
from core.models.website_learner import WebsiteLearner
model = WebsiteLearner.load('checkpoints/paired_generator/epoch_30.pt')
```

### Q: 模型怎样部署到生产?
**A**: 导出为 ONNX，集成到 Rust 核心
```bash
python scripts/export_to_onnx.py --checkpoint ... --output ...
# 然后在 Rust 中使用 InferenceEngine 加载
```

### Q: 如何添加新的网站数据?
**A**:
```bash
# 方法1: 直接爬取
python scripts/batch_crawl_websites.py --urls-file urls.txt --output ...

# 方法2: 添加到现有数据
cat new_sites.jsonl >> data/websites/all_sites.jsonl
```

---

## 📚 文档导航

| 文件 | 内容 | 适合人群 |
|------|------|---------|
| `README.md` | 项目概览、核心思想、目录结构 | 所有人 |
| `QUICKSTART.md` | 分步教程、完整示例 | 初学者 |
| `WEBSITE_GENERATION_PLAN.md` | 设计细节、技术方案 | 开发者 |
| `TRAINING_DIRECTORY_ANALYSIS.md` | 详细分析、模块说明 | 架构师 |
| `core/models/website_learner.py` | 模型实现 | ML 工程师 |
| `scripts/train_paired_website_generator.py` | 训练实现 | 数据科学家 |

---

## 🎓 学习路径

1. **新手**: README → QUICKSTART
2. **开发者**: scripts/ → core/data/ → core/models/
3. **高级**: WEBSITE_GENERATION_PLAN → trainer 实现 → 自定义扩展
4. **研究员**: 论文分析 → 架构改进 → 性能优化

---

## 📞 快速命令参考

```bash
# 环境
cd training && pip install -r requirements.txt

# 数据
python scripts/extract_website_complete.py --input ... --output ...
python scripts/create_simplified_dataset.py --input ... --output ...

# 训练
python scripts/train_paired_website_generator.py

# 评估
python scripts/count_parameters.py  # 模型大小
python scripts/extract_features.py  # 特征提析

# 导出
python scripts/export_to_onnx.py --checkpoint ... --output ...

# 管理
python scripts/dataset_manager.py --action validate
```

---

**最后更新**: 2026-01-22
**版本**: 1.0
**维护者**: BrowerAI Team
