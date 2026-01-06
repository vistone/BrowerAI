# Quick Start - Website Regeneration Training

快速开始训练网站再生成模型

## 目标

训练一个模型：
- **输入**：原始网站代码（HTML+CSS+JS）
- **输出**：简化优化版本（功能相同，代码更简洁）
- **用途**：BrowerAI双渲染模式

## 步骤

### 1. 环境准备

```bash
cd /workspaces/BrowerAI/training

# 安装依赖
pip install torch torchvision torchaudio beautifulsoup4 cssutils

# 或使用requirements.txt
pip install -r requirements.txt
```

### 2. 准备数据

#### 方法A：使用现有数据（推荐）
```bash
# 已有139个完整网站数据
ls -lh data/website_complete.jsonl

# 生成配对数据（原始→简化）
python scripts/create_simplified_dataset.py \
  --input data/website_complete.jsonl \
  --output data/website_paired.jsonl
```

输出：
```
INFO: Loaded 139 websites
INFO: Processed 139/139 websites
✅ 简化数据集创建完成:
  - 网站数量: 139
  - 原始代码总量: 1203.2 KB
  - 简化代码总量: 877.7 KB
  - 平均压缩率: 72.95%
```

#### 方法B：爬取新数据（可选）
```bash
# 爬取更多网站
python scripts/batch_crawl_websites.py \
  --urls-file data/urls.txt \
  --output data/websites/new_sites.jsonl \
  --max-workers 10

# 提取完整网站
python scripts/extract_website_complete.py \
  --input data/websites/new_sites.jsonl \
  --output data/website_complete_new.jsonl
```

### 3. 训练模型

```bash
# 配对训练（原始→简化）
python scripts/train_paired_website_generator.py
```

参数：
- **数据**：`data/website_paired.jsonl`（139对）
- **模型**：Transformer Encoder-Decoder
- **vocab**：229字符
- **架构**：d_model=256, nhead=8, layers=3
- **训练**：30 epochs, batch_size=2
- **时间**：约2-3小时

输出日志示例：
```
INFO: Loading paired websites from data/website_paired.jsonl
INFO: Loaded 139 website pairs
INFO: Vocab size: 229
INFO: Model: vocab=229, d_model=256, layers=3, device=cpu
INFO: Starting training (原始→简化)...

INFO: Epoch 1/30, Batch 10, Loss: 4.5136
INFO: Epoch 1/30, Batch 20, Loss: 4.1401
INFO: Epoch 1/30 - Avg Loss: 4.2134
INFO: Saved checkpoint: epoch_1.pt

INFO: Epoch 10/30 - Avg Loss: 3.0245
INFO: Epoch 20/30 - Avg Loss: 2.1156
INFO: Epoch 30/30 - Avg Loss: 1.5234
✅ Training completed!
```

检查点保存在：`checkpoints/paired_generator/epoch_*.pt`

### 4. 监控训练

#### 查看实时日志
```bash
tail -f logs/paired_training_*.log
```

#### 查看检查点
```bash
ls -lh checkpoints/paired_generator/
# epoch_1.pt, epoch_2.pt, ..., epoch_30.pt
```

#### 检查进程
```bash
ps aux | grep train_paired
```

### 5. 导出ONNX

训练完成后导出为ONNX格式（用于Rust）：

```bash
python scripts/export_to_onnx.py \
  --checkpoint checkpoints/paired_generator/epoch_30.pt \
  --output ../models/local/website_generator_v1.onnx \
  --vocab-size 229 \
  --seq-len 1024
```

输出：
```
INFO: Loading checkpoint from checkpoints/paired_generator/epoch_30.pt
INFO: Model loaded, vocab_size=229, d_model=256, layers=3
INFO: Exporting to ONNX...
✅ ONNX模型已导出到: ../models/local/website_generator_v1.onnx
✅ 配置文件已保存到: ../models/local/website_generator_v1_config.json

模型信息:
  - 输入1: src (网站源代码序列) - shape: [batch, src_len]
  - 输入2: tgt (目标代码序列起始) - shape: [batch, tgt_len]
  - 输出: logits (字符概率分布) - shape: [batch, tgt_len, 229]
```

### 6. 测试集成

在Rust中测试双渲染：

```bash
cd /workspaces/BrowerAI

# 更新模型配置
cat >> models/model_config.toml << 'EOF'

[[models]]
name = "website_generator_v1"
model_type = "WebsiteGenerator"
path = "website_generator_v1.onnx"
version = "1.0.0"
description = "Website code regeneration (original -> simplified)"
EOF

# 运行双渲染示例
cargo run --example dual_rendering_demo https://example.com
```

预期输出：
```
📥 Fetching: https://example.com
✅ Fetched 1256 bytes

🎨 Original Rendering:
DOM Nodes: 245
Layout Time: 12ms

🤖 AI Regeneration:
✅ Regeneration complete
Original HTML: 1256 bytes
Regenerated HTML: 892 bytes (29% reduction)

🎨 AI-Regenerated Rendering:
DOM Nodes: 178 (27% reduction)

📊 Comparison:
Size Reduction: 29.0%
Node Reduction: 27.3%
```

## 故障排除

### 问题1：训练中断
```bash
# 从最后一个checkpoint继续（手动修改代码加载checkpoint）
python scripts/train_paired_website_generator.py
```

### 问题2：内存不足
```bash
# 减小batch_size（编辑train_paired_website_generator.py）
batch_size = 1  # 原来是2
```

### 问题3：Loss不下降
- 检查数据质量
- 增加训练epochs
- 调整learning_rate

### 问题4：ONNX导出失败
```bash
# 确保PyTorch版本
pip install torch==2.1.0

# 检查checkpoint是否完整
python -c "import torch; print(torch.load('checkpoints/paired_generator/epoch_30.pt').keys())"
```

## 数据格式

### website_complete.jsonl
```json
{
  "website_id": "example_com",
  "url": "https://example.com",
  "original": {
    "html": "<!DOCTYPE html>...",
    "css": ".container{...}",
    "js": "function init(){...}"
  },
  "metadata": {
    "dom_depth": 15,
    "element_count": 120
  }
}
```

### website_paired.jsonl
```json
{
  "url": "https://example.com",
  "original": "<html><head><style>.long-class{...}",
  "simplified": "<html><head><style>.c1{...}",
  "original_len": 5230,
  "simplified_len": 3821,
  "compression_ratio": 0.73
}
```

## 下一步

- [WEBSITE_GENERATION_PLAN.md](WEBSITE_GENERATION_PLAN.md) - 详细设计
- [../docs/NEXT_STEP_OPTIMIZATION.md](../docs/NEXT_STEP_OPTIMIZATION.md) - 优化报告
- [../src/renderer/ai_regeneration.rs](../src/renderer/ai_regeneration.rs) - Rust集成代码
