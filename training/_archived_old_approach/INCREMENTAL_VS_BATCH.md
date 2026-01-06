# 增量学习 vs 批量学习 对比

## 🔄 两种学习模式对比

### 批量学习模式（原方案）

```
Step 1: 爬取所有网站 (2-3小时)
  ├── 网站1 → 保存到文件
  ├── 网站2 → 保存到文件
  ├── ...
  └── 网站1000 → 保存到文件
  结果: 生成 1000_sites.jsonl (3-5 GB)

Step 2: 加载所有数据 (需要8-16 GB RAM)
  └── 一次性训练所有网站 (10-20小时)
  
Step 3: 保存模型
```

**缺点：**
- ❌ 存储占用: 3-5 GB中间数据
- ❌ 内存需求: 8-16 GB
- ❌ 等待时间长: 必须等所有爬取完成
- ❌ 中断风险: 中断后数据可能不完整

---

### 增量学习模式（新方案）✨

```
For 每个网站:
  ├── Step 1: 爬取网站 (~5秒)
  ├── Step 2: 立即学习 (~1秒)  
  └── Step 3: 更新模型 (保存检查点)
  
无需中间数据存储！
```

**优点：**
- ✅ **零中间存储**: 不保存JSONL文件
- ✅ **内存友好**: 只处理当前网站 (~100MB)
- ✅ **实时更新**: 模型随时可用
- ✅ **中断安全**: 每N个网站自动保存
- ✅ **可恢复**: 中断后从断点继续

---

## 📊 详细对比

| 特性 | 批量学习 | 增量学习 ⭐ |
|------|---------|-----------|
| **中间数据** | 3-5 GB | 0 GB |
| **内存需求** | 8-16 GB | 1-2 GB |
| **总时间** | 15-25 小时 | 3-5 小时 |
| **首个模型可用时间** | 15小时后 | 10分钟后 |
| **中断恢复** | 需要重新爬取 | 从断点继续 |
| **存储空间** | 需要大量空间 | 只需模型文件 |

---

## 🚀 使用增量学习

### 方式1: 一键运行

```bash
cd /workspaces/BrowerAI/training
./run_incremental_learning.sh
```

### 方式2: 自定义参数

```bash
cd /workspaces/BrowerAI/training

python scripts/incremental_learning.py \
  --urls-file data/large_urls.txt \
  --checkpoint-dir checkpoints/incremental \
  --depth 2 \
  --max-pages 5 \
  --learning-rate 1e-4 \
  --save-frequency 10
```

**参数说明:**
- `--save-frequency 10`: 每10个网站保存一次检查点
- `--learning-rate 1e-4`: 学习率（较小值更稳定）
- `--depth 2`: 爬取深度
- `--max-pages 5`: 每站最多页面数

---

## 🔧 工作原理

### 增量学习流程

```python
for website in websites:
    # 1️⃣ 爬取
    website_data = await crawler.crawl_website(url)
    
    # 2️⃣ 提取特征
    text = extract_text(website_data)
    tokens = tokenizer.encode(text)
    
    # 3️⃣ 训练一步
    loss = model.train_step(tokens, labels)
    
    # 4️⃣ 更新模型
    optimizer.step()
    
    # 5️⃣ 定期保存 (每10个网站)
    if idx % 10 == 0:
        save_checkpoint()
```

**关键特性:**
- **在线学习**: 无需存储所有数据
- **增量更新**: 模型持续改进
- **检查点机制**: 自动保存进度

---

## 💾 存储占用对比

### 批量学习存储

```
data/websites/
├── 1000_sites.jsonl        3.5 GB  ← 中间数据
│
checkpoints/
├── epoch_10.pt            200 MB
├── epoch_20.pt            200 MB
├── best_model.pt          200 MB
└── final_model.onnx       100 MB
                           --------
总计:                      4.2 GB
```

### 增量学习存储 ✨

```
checkpoints/incremental/
├── latest.pt              200 MB  ← 随时可用
└── final_model.pt         200 MB
                           --------
总计:                      400 MB  (节省3.8 GB!)
```

---

## ⚡ 性能对比

### 时间线对比

**批量学习:**
```
0小时    2小时                  15小时                 25小时
  |--------|----------------------|---------------------|
  开始    爬取完成              训练中...            完成
                                                     ↑ 模型可用
```

**增量学习:**
```
0小时    10分钟     1小时          3小时
  |--------|----------|------------|
  开始    首次保存   持续学习...   完成
          ↑ 模型可用  ↑ 持续改进    ↑ 最终模型
```

---

## 🛡️ 中断恢复

### 批量学习中断

```bash
# 爬取进行到60% 时中断
❌ 问题: 需要重新爬取所有网站
❌ 浪费: 已爬取60%的数据无法使用
```

### 增量学习中断 ✨

```bash
# 训练到第500个网站时中断
✅ 已保存: checkpoints/incremental/latest.pt
✅ 状态: 已训练500个网站

# 恢复训练
./run_incremental_learning.sh
# 自动从第501个网站继续！
```

---

## 📈 实时监控

增量学习支持实时查看进度：

```bash
# 终端1: 运行学习
./run_incremental_learning.sh

# 终端2: 监控进度
watch -n 5 'ls -lh checkpoints/incremental/latest.pt'

# 终端3: 查看日志
tail -f logs/incremental_*.log
```

---

## 🎯 推荐使用场景

### 使用增量学习（推荐）✨

- ✅ 大规模网站学习 (>100个)
- ✅ 存储空间有限
- ✅ 需要实时模型
- ✅ 长时间训练任务
- ✅ 不稳定网络环境

### 使用批量学习

- ⚠️ 需要数据分析
- ⚠️ 需要多次实验
- ⚠️ 小规模数据 (<50个网站)

---

## 🔄 从批量迁移到增量

如果已有JSONL文件，可以转换：

```bash
# 使用已有数据进行增量学习
python scripts/incremental_learning.py \
  --urls-file <(jq -r '.url + "," + .category' data/websites/1000_sites.jsonl) \
  --checkpoint-dir checkpoints/from_batch
```

---

## 💡 最佳实践

### 1. 选择合适的保存频率

```bash
# 快速网站 (响应快)
--save-frequency 20  # 每20个保存

# 慢速网站 (响应慢)
--save-frequency 5   # 每5个保存

# 推荐
--save-frequency 10  # 平衡
```

### 2. 调整学习率

```bash
# 快速学习 (可能不稳定)
--learning-rate 1e-3

# 稳定学习 (推荐)
--learning-rate 1e-4

# 精细调整
--learning-rate 5e-5
```

### 3. 监控训练质量

```bash
# 查看训练历史
python -c "
import torch
ckpt = torch.load('checkpoints/incremental/latest.pt')
history = ckpt['training_history']
avg_loss = sum(h['loss'] for h in history[-100:]) / 100
print(f'最近100个网站平均损失: {avg_loss:.4f}')
"
```

---

## 🆚 总结

**简单来说:**

**批量学习** = 先存后学（占空间，等待久）
```
爬取所有 → 保存文件 (3-5GB) → 加载训练 → 完成
```

**增量学习** = 边爬边学（省空间，实时可用）✨
```
爬一个 → 学一个 → 保存模型 → 继续...
```

**结论:** 对于1000个网站，**强烈推荐使用增量学习模式！**

---

## 📚 相关文档

- [增量学习快速开始](QUICKSTART_INCREMENTAL.md)
- [模型检查点管理](CHECKPOINT_GUIDE.md)
- [性能调优指南](PERFORMANCE_TUNING.md)
