# 🧠 真实 1000+ URL 学习训练报告

## 概要
- 数据来源: `data/website_training_1000_real.jsonl` (132 个真实爬取样本)
- 训练脚本: `training/large_scale_website_trainer.py`
- 训练配置: epochs=30, batch_size=8
- 模型输出: `checkpoints/website_generator_1000_real_v2/checkpoint_epoch_30.pt`
- 生成网站: 50 个 (`generated_websites_1000_real_v2/`)

## 训练日志摘要
```
Epoch 25: Val Loss=0.0611
Epoch 27: Val Loss=0.0386
Epoch 30: Val Loss=0.0400 ← 最终
```

说明:
- 真实数据样本规模较小 (132)，训练损失迅速归零，验证损失在 0.038~0.061 之间波动，整体稳定。

## 生成与评估
- 生成数量: 50
- 评估报告: `generated_websites_1000_real_v2/evaluation_report.json`
- 质量指标:
  - HTML 平均质量: 100%
  - CSS 平均质量: 100%
  - JS 平均质量: 100%
  - 总体平均质量: 100%
  - 多样性评分: 25%

## 结论与后续建议
- 真实性: 生成结果质量高，但多样性受限 (25%)，原因可能是样本数量偏少与模板趋同。
- 建议:
  1. 扩大真实爬取样本至 ≥ 300（重跑 `training/crawl_1000_websites_fixed.py`）
  2. 在生成阶段提升随机性 (temperature/top-k)
  3. 引入更多类别 (news/portfolio/social) 增加结构差异
  4. 结合模板生成与真实爬取数据混合训练 (提高泛化)

## 文件索引
- 训练数据: `data/website_training_1000_real.jsonl` (132 行)
- 模型: `checkpoints/website_generator_1000_real_v2/checkpoint_epoch_30.pt`
- 生成网站目录: `generated_websites_1000_real_v2/`
- 评估报告: `generated_websites_1000_real_v2/evaluation_report.json`

## 运行复现
```bash
# 训练
python3 training/large_scale_website_trainer.py \
  --data-file data/website_training_1000_real.jsonl \
  --epochs 30 --batch-size 8 \
  --output-dir checkpoints/website_generator_1000_real_v2

# 生成与评估
python3 training/evaluate_generated_websites.py \
  --model-path checkpoints/website_generator_1000_real_v2/checkpoint_epoch_30.pt \
  --num-websites 50 \
  --output-dir generated_websites_1000_real_v2
```
