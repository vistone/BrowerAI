# 大规模网站学习 - 完整工作流

本指南描述如何爬取1000+网站、训练模型、并进行推理生成。

## 🎯 目标

1. 爬取1000+个真实网站数据
2. 训练完整的网站理解模型
3. 保存模型权重到本地
4. 使用模型进行推理和生成

---

## 📋 准备工作

### 1. 检查数据文件

```bash
cd /workspaces/BrowerAI/training

# 查看URL列表
head data/large_urls.txt
wc -l data/large_urls.txt  # 应显示约1000行
```

### 2. 安装依赖（如需要）

```bash
pip install tqdm aiohttp beautifulsoup4 torch
```

---

## 🕷️ 第一步: 大规模爬取

### 方案A: 完整爬取（推荐）

爬取1000个网站，每个网站2层深度，每站最多5个页面：

```bash
python scripts/batch_crawl_websites.py \
  --urls-file data/large_urls.txt \
  --output-dir data/websites/large_scale \
  --batch-size 50 \
  --depth 2 \
  --max-pages 5 \
  --output data/websites/large_train.jsonl
```

**预计结果**:
- 网站数量: ~1000 个
- 页面总数: ~3000-4000 页
- 时间消耗: 6-10 小时（取决于网络）
- 存储空间: ~2-5 GB

**特性**:
- ✅ 断点续传（中断后可恢复）
- ✅ 批次处理（每50个网站一批）
- ✅ 错误重试
- ✅ 进度日志保存

### 方案B: 快速测试（100个网站）

如果只想快速测试：

```bash
# 创建测试URL列表
head -100 data/large_urls.txt > data/test_100_urls.txt

# 爬取100个网站
python scripts/batch_crawl_websites.py \
  --urls-file data/test_100_urls.txt \
  --output-dir data/websites/test_100 \
  --batch-size 20 \
  --depth 2 \
  --max-pages 5 \
  --output data/websites/test_100_train.jsonl
```

**预计结果**:
- 网站数量: ~100 个
- 页面总数: ~300-400 页
- 时间消耗: 30-60 分钟
- 存储空间: ~200-500 MB

### 方案C: 恢复中断的爬取

如果爬取中断，可以继续：

```bash
# 系统会自动从 data/websites/large_scale/crawl_progress.json 恢复
python scripts/batch_crawl_websites.py \
  --urls-file data/large_urls.txt \
  --output-dir data/websites/large_scale \
  --batch-size 50 \
  --depth 2 \
  --max-pages 5 \
  --output data/websites/large_train.jsonl
```

### 方案D: 只合并已有批次

如果已经爬取了部分批次，只想合并：

```bash
python scripts/batch_crawl_websites.py \
  --output-dir data/websites/large_scale \
  --output data/websites/large_train.jsonl \
  --merge
```

---

## 🔍 检查爬取结果

```bash
# 统计网站数量
wc -l data/websites/large_train.jsonl

# 查看第一个网站
head -1 data/websites/large_train.jsonl | python -m json.tool | head -50

# 统计详细信息
cat data/websites/large_train.jsonl | python -c "
import json
import sys

total_sites = 0
total_pages = 0
frameworks = {}

for line in sys.stdin:
    data = json.loads(line)
    total_sites += 1
    total_pages += data.get('depth', 1)
    fw = data.get('metadata', {}).get('framework', 'Unknown')
    frameworks[fw] = frameworks.get(fw, 0) + 1

print(f'网站总数: {total_sites}')
print(f'页面总数: {total_pages}')
print(f'平均深度: {total_pages/total_sites:.1f}')
print(f'\n框架分布:')
for fw, count in sorted(frameworks.items(), key=lambda x: -x[1]):
    print(f'  {fw}: {count}')
"
```

---

## 🤖 第二步: 训练模型

### 完整训练（推荐）

使用所有数据训练50个epoch：

```bash
python scripts/train_large_scale.py \
  --data-file data/websites/large_train.jsonl \
  --checkpoint-dir checkpoints/large_scale \
  --epochs 50 \
  --batch-size 8 \
  --learning-rate 1e-4
```

**训练配置**:
- Epoch数: 50
- Batch size: 8
- 学习率: 0.0001
- 优化器: AdamW
- 调度器: CosineAnnealingLR

**预计时间**:
- CPU训练: 20-40 小时
- GPU训练: 2-5 小时

**检查点保存**:
- `checkpoints/large_scale/latest_checkpoint.pt` - 最新检查点
- `checkpoints/large_scale/best_model.pt` - 最佳模型
- `checkpoints/large_scale/checkpoint_epoch_N.pt` - 定期保存
- `checkpoints/large_scale/training_history.json` - 训练历史
- `checkpoints/large_scale/website_learner.onnx` - ONNX模型

### 快速训练（测试）

使用较少的epoch快速测试：

```bash
python scripts/train_large_scale.py \
  --data-file data/websites/test_100_train.jsonl \
  --checkpoint-dir checkpoints/test_100 \
  --epochs 10 \
  --batch-size 4 \
  --learning-rate 1e-4
```

### 恢复训练

如果训练中断，添加 `--resume` 从检查点继续：

```bash
python scripts/train_large_scale.py \
  --data-file data/websites/large_train.jsonl \
  --checkpoint-dir checkpoints/large_scale \
  --epochs 50 \
  --batch-size 8 \
  --learning-rate 1e-4 \
  --resume
```

---

## 📊 监控训练

### 查看训练日志

```bash
# 实时查看
tail -f checkpoints/large_scale/*.log

# 查看训练历史
cat checkpoints/large_scale/training_history.json | python -m json.tool
```

### 可视化训练曲线

```python
import json
import matplotlib.pyplot as plt

with open('checkpoints/large_scale/training_history.json') as f:
    history = json.load(f)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Loss')

plt.subplot(1, 2, 2)
plt.plot(history['train_acc'], label='Train Acc')
plt.plot(history['val_acc'], label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.title('Training Accuracy')

plt.tight_layout()
plt.savefig('training_curves.png')
```

---

## 🎯 第三步: 推理与生成

### 单个网站推理

```bash
python scripts/inference_website.py \
  --model checkpoints/large_scale/best_model.pt \
  --mode single \
  --url "https://example.com" \
  --html "<html>...</html>" \
  --css "body {...}" \
  --js "console.log('hello')"
```

### 从文件推理

```bash
python scripts/inference_website.py \
  --model checkpoints/large_scale/best_model.pt \
  --mode single \
  --input data/websites/depth_test.jsonl
```

### 批量推理

对所有网站进行推理：

```bash
python scripts/inference_website.py \
  --model checkpoints/large_scale/best_model.pt \
  --mode batch \
  --input data/websites/large_train.jsonl \
  --output results/inference_results.json \
  --max-samples 1000
```

**输出文件包含**:
- 每个网站的分类预测
- 框架识别结果
- 风格嵌入向量
- 相似度计算数据

### 查看推理结果

```bash
# 查看结果
cat results/inference_results.json | python -m json.tool | head -100

# 统计分类准确率
cat results/inference_results.json | python -c "
import json
import sys

data = json.load(sys.stdin)
results = data['results']

print(f'总推理数量: {len(results)}')
print(f'\n分类分布:')
categories = {}
for r in results:
    cat = r['category']
    conf = r['category_confidence']
    categories[cat] = categories.get(cat, 0) + 1

for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
    print(f'  {cat}: {count}')
"
```

---

## 📦 模型文件说明

训练完成后，`checkpoints/large_scale/` 包含：

| 文件 | 大小 | 说明 |
|------|------|------|
| `best_model.pt` | ~200MB | 最佳模型（验证损失最低） |
| `latest_checkpoint.pt` | ~200MB | 最新检查点 |
| `website_learner.onnx` | ~200MB | ONNX格式（用于部署） |
| `training_history.json` | ~10KB | 训练历史 |
| `checkpoint_epoch_*.pt` | ~200MB | 定期保存的检查点 |

### 使用ONNX模型

```python
import onnxruntime as ort
import numpy as np

# 加载ONNX模型
session = ort.InferenceSession('checkpoints/large_scale/website_learner.onnx')

# 准备输入
html_ids = np.random.randint(0, 10000, (1, 2048), dtype=np.int64)
css_ids = np.random.randint(0, 10000, (1, 1024), dtype=np.int64)
js_ids = np.random.randint(0, 10000, (1, 2048), dtype=np.int64)
url_features = np.random.randn(1, 128).astype(np.float32)

# 推理
outputs = session.run(None, {
    'html_ids': html_ids,
    'css_ids': css_ids,
    'js_ids': js_ids,
    'url_features': url_features
})

print('Category logits:', outputs[0])
print('Framework logits:', outputs[1])
print('Style embedding:', outputs[2])
```

---

## 🔄 完整工作流示例

### 示例1: 完整流程（1000网站）

```bash
#!/bin/bash
# 完整工作流

echo "==== 第1步: 爬取1000个网站 ===="
python scripts/batch_crawl_websites.py \
  --urls-file data/large_urls.txt \
  --output-dir data/websites/large_scale \
  --batch-size 50 \
  --depth 2 \
  --max-pages 5 \
  --output data/websites/large_train.jsonl

echo "==== 第2步: 训练模型 ===="
python scripts/train_large_scale.py \
  --data-file data/websites/large_train.jsonl \
  --checkpoint-dir checkpoints/large_scale \
  --epochs 50 \
  --batch-size 8 \
  --learning-rate 1e-4

echo "==== 第3步: 批量推理 ===="
python scripts/inference_website.py \
  --model checkpoints/large_scale/best_model.pt \
  --mode batch \
  --input data/websites/large_train.jsonl \
  --output results/inference_results.json

echo "==== 完成! ===="
```

### 示例2: 快速测试流程（100网站）

```bash
#!/bin/bash
# 快速测试工作流

# 准备测试数据
head -100 data/large_urls.txt > data/test_100_urls.txt

# 爬取
python scripts/batch_crawl_websites.py \
  --urls-file data/test_100_urls.txt \
  --output-dir data/websites/test_100 \
  --batch-size 20 \
  --depth 2 \
  --max-pages 5 \
  --output data/websites/test_100_train.jsonl

# 训练
python scripts/train_large_scale.py \
  --data-file data/websites/test_100_train.jsonl \
  --checkpoint-dir checkpoints/test_100 \
  --epochs 10 \
  --batch-size 4

# 推理
python scripts/inference_website.py \
  --model checkpoints/test_100/best_model.pt \
  --mode batch \
  --input data/websites/test_100_train.jsonl \
  --output results/test_100_results.json

echo "测试完成!"
```

---

## ⚠️ 注意事项

### 爬取阶段
1. **尊重robots.txt**: 爬虫会遵守网站的robots协议
2. **速率限制**: 批次间有30秒休息，避免被封IP
3. **错误处理**: 失败的网站会被记录但不会中断流程
4. **断点续传**: 使用 `crawl_progress.json` 跟踪进度

### 训练阶段
1. **内存需求**: 建议至少16GB RAM
2. **GPU加速**: 有GPU会快很多（20x+）
3. **检查点保存**: 每5个epoch自动保存
4. **过拟合**: 监控验证集损失，必要时早停

### 推理阶段
1. **批量处理**: 大规模推理时使用batch模式
2. **结果缓存**: 推理结果保存为JSON可重复使用
3. **ONNX部署**: 生产环境推荐使用ONNX格式

---

## 🎉 预期结果

完成整个流程后，您将拥有：

1. **数据集**: 1000个网站，~3000-4000个页面
2. **模型**: 训练好的网站理解模型（~200MB）
3. **推理结果**: 完整的网站分类和分析
4. **嵌入向量**: 可用于相似度搜索和推荐

**模型能力**:
- 🏷️  网站分类（10个类别）
- 🎨 框架识别（8种主流框架）
- 📊 风格分析（嵌入向量）
- 🔍 相似网站推荐

---

## 📚 相关文档

- [DEPTH_QUICKREF.md](DEPTH_QUICKREF.md) - 深度爬取快速参考
- [HOLISTIC_LEARNING_GUIDE.md](HOLISTIC_LEARNING_GUIDE.md) - 整体学习指南
- [README.md](README.md) - 主文档

---

**祝训练顺利！🚀**
