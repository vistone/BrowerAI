# BrowerAI Training Pipeline

AI 网站再生成训练：输入原始网站代码，输出等价的简化版本

## 🎯 核心思想

### 整体网站学习
不学习孤立的技术点（JS/HTML/CSS分开），而是将完整网站（HTML+CSS+JS）作为一个整体来学习。

> "学习应该是整个网站的思想去学习，而不是单独的某个技术层面学习"

### 配对训练模式
- **输入**：原始网站代码（冗余、未优化）
- **输出**：简化版本（压缩、优化、功能相同）
- **用途**：双渲染模式 - 原始 vs AI优化对比

## 📁 目录结构

```
training/
├── README.md                       # 本文件
├── QUICKSTART.md                   # 快速开始
├── WEBSITE_GENERATION_PLAN.md      # 设计文档
├── requirements.txt                # 依赖
│
├── data/                           # 训练数据
│   ├── website_complete.jsonl     # 完整网站（139个）
│   └── website_paired.jsonl       # 配对数据（原始→简化）
│
├── scripts/                        # 核心脚本
│   ├── extract_website_complete.py          # 提取完整网站
│   ├── create_simplified_dataset.py         # 生成简化数据
│   ├── train_paired_website_generator.py    # ★ 配对训练
│   └── export_to_onnx.py                    # 导出ONNX
│
├── checkpoints/paired_generator/   # 检查点
└── logs/                           # 日志
```

## 🚀 快速开始

```bash
# 1) 安装依赖
pip install -r requirements.txt

# 2) 生成配对数据（原始→简化）
python scripts/create_simplified_dataset.py \
  --input data/website_complete.jsonl \
  --output data/website_paired.jsonl

# 3) 训练模型（约 30 epochs）
python scripts/train_paired_website_generator.py

# 4) 导出 ONNX 供 Rust 使用
python scripts/export_to_onnx.py \
  --checkpoint checkpoints/paired_generator/epoch_30.pt \
  --output ../models/local/website_generator_v1.onnx
```

## 📊 数据统计

- **网站数**：139个
- **原始代码**：1203 KB
- **简化代码**：878 KB
- **压缩率**：73%

## 📖 详细文档

- [QUICKSTART.md](QUICKSTART.md) - 详细步骤
- [WEBSITE_GENERATION_PLAN.md](WEBSITE_GENERATION_PLAN.md) - 设计文档
- [../docs/NEXT_STEP_OPTIMIZATION.md](../docs/NEXT_STEP_OPTIMIZATION.md) - 实施报告

## 🎓 设计理念

### 从错误到正确
1. ❌ 框架分类（React/Vue）
2. ❌ 单独技术（JS/HTML/CSS分开）
3. ❌ 自编码器（输入=输出）
4. ✅ **配对生成（原始→简化）**

### 为什么这样？
- 学习"整个网站的思想"，不是孤立技术
- 输入完整网站，输出优化版本
- 功能相同，代码更简洁
- 用于双渲染对比

## 🔧 技术栈

- **模型**：Transformer Encoder-Decoder
- **vocab_size**：229（字符级）
- **架构**：d_model=256, nhead=8, layers=3
- **训练**：30 epochs, batch_size=2
- **输出**：ONNX（用于Rust集成）
