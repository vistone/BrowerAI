# BrowerAI Training Pipeline

AI网站再生成训练流程 - 学习整体网站，输入原始代码输出简化版本

## 🎯 核心思想

### 整体网站学习
**不学习孤立的技术点**（JS/HTML/CSS分开），而是将**完整网站**（HTML+CSS+JS）作为一个整体来学习。

> "学习应该是整个网站的思想去学习，而不是单独的某个技术层面学习"

### 配对训练模式
- **输入**：原始网站代码（冗余、未优化、有tracking代码）
- **输出**：简化版本（压缩class名、去除冗余、功能相同）
- **用途**：双渲染模式 - 用户可对比原始 vs AI优化版本

## 📁 目录结构

```
training/
├── README.md                          # 本文件
├── QUICKSTART.md                      # 快速开始指南
├── WEBSITE_GENERATION_PLAN.md         # 详细设计文档
├── requirements.txt                   # Python依赖
│
├── data/                              # 训练数据
│   ├── websites/1000_sites.jsonl     # 爬取的原始网站
│   ├── website_complete.jsonl        # 完整网站（139个）
│   └── website_paired.jsonl          # 配对数据（原始→简化）
│
├── scripts/                           # 核心脚本
│   ├── batch_crawl_websites.py               # 爬取网站
│   ├── extract_website_complete.py           # 提取完整网站
│   ├── create_simplified_dataset.py          # 生成简化版本
│   ├── train_paired_website_generator.py     # ★ 配对训练
│   └── export_to_onnx.py                     # 导出ONNX
│
├── checkpoints/paired_generator/      # 训练检查点
├── logs/                              # 训练日志
└── _archived_old_approach/            # 归档的旧代码
```

## 🚀 快速开始

### 1. 安装依赖
\`\`\`bash
pip install -r requirements.txt
\`\`\`

### 2. 生成配对数据
\`\`\`bash
python scripts/create_simplified_dataset.py \
  --input data/website_complete.jsonl \
  --output data/website_paired.jsonl
\`\`\`

### 3. 训练模型
\`\`\`bash
python scripts/train_paired_website_generator.py
\`\`\`

### 4. 导出ONNX
\`\`\`bash
python scripts/export_to_onnx.py \
  --checkpoint checkpoints/paired_generator/epoch_30.pt \
  --output ../models/local/website_generator_v1.onnx
\`\`\`

## 📖 详细文档

- [QUICKSTART.md](QUICKSTART.md) - 快速开始
- [WEBSITE_GENERATION_PLAN.md](WEBSITE_GENERATION_PLAN.md) - 设计文档
- [../docs/NEXT_STEP_OPTIMIZATION.md](../docs/NEXT_STEP_OPTIMIZATION.md) - 实施报告

## 🎓 设计理念

### 从错误中学习
1. ❌ 框架分类（React/Vue）→ 不是需求
2. ❌ 单独技术（JS/HTML/CSS分开）→ 割裂整体
3. ❌ 自编码器（输入=输出）→ 没学到简化
4. ✅ **配对生成（原始→简化）→ 正确方向！**

### 实际应用
双渲染模式：用户可切换查看原始 vs AI优化版本
