# Week 6 快速启动 - 5 步性能优化计划

**日期**: 2026-01-31 开始  
**持续**: 7 天 (到 2026-02-07)  
**目标**: 准确率 41.67% → 75%+ | 并发 1K → 5K+ RPS

---

## ⚡ 5 步执行计划

### 步骤 1️⃣  数据诊断与增强 (Day 1-2)

**命令**:
```bash
# 运行诊断脚本
python3 training/scripts/week6_diagnostics.py

# 输出分析结果
# - 框架覆盖差距
# - 特征重要性排序
# - 数据不均衡统计
```

**采集新样本**:
```bash
# 扩展框架覆盖 (6 → 12)
python3 training/scripts/collect_framework_samples.py \
  --frameworks ember,backbone,alpine,htmx,nextjs,nuxt \
  --samples-per-framework 100 \
  --output data/week6_samples/

# 扩展混淆检测 (4 → 8)
python3 training/scripts/generate_obfuscation_samples.py \
  --techniques property-encryption,function-wrapping,regex-obfuscation,array-obfuscation \
  --samples 50 \
  --output data/week6_obfuscation/
```

**预期**: +600 样本, 框架覆盖翻倍

---

### 步骤 2️⃣  模型改进训练 (Day 3-4)

**启动改进的混合模型 v3**:
```bash
# 训练新模型
python3 training/pipelines/train_hybrid_model_v3.py \
  --input-features important_features_only \
  --hidden-dims 256,128,64 \
  --dropout 0.3 \
  --epochs 100 \
  --batch-size 32 \
  --loss-function weighted_bce \
  --k-fold 5 \
  --output models/hybrid_v3.onnx

# 预期准确率: 70-80%
```

**多模型集成**:
```bash
# 集成三个模型的预测
python3 training/pipelines/ensemble_models.py \
  --models neural_net,random_forest,gradient_boosting \
  --weights 0.4,0.3,0.3 \
  --output models/ensemble_v3.onnx

# 预期准确率: 78-82%
```

**预期**: 准确率从 41.67% → 75%+

---

### 步骤 3️⃣  规则权重学习 (Day 4)

**优化规则权重**:
```bash
python3 training/scripts/optimize_rule_weights.py \
  --init-weights data/current_rule_weights.json \
  --training-samples data/week6_samples/ \
  --epochs 100 \
  --learning-rate 0.01 \
  --output models/learned_rule_weights.json
```

**预期**: 规则准确率 35% → 55-65%

---

### 步骤 4️⃣  性能优化实施 (Day 5)

**API 并行化改进**:
```bash
# 使用线程池处理推理
cargo build --release --features parallel-inference

# 启动 API (自动并行处理)
./target/release/browerai-api-server

# 压力测试
ab -n 10000 -c 100 http://localhost:8080/health

# 预期: 1,000 RPS → 5,000+ RPS
```

**缓存策略优化**:
```bash
# 启用多级缓存
CACHE_L1_SIZE=10000 \
CACHE_L2_ENABLED=true \
./target/release/browerai-api-server

# 预期: 缓存命中 53.77% → 85%+
```

---

### 步骤 5️⃣  验证与报告 (Day 6-7)

**最终验证**:
```bash
# 运行完整性能测试
python3 testing/performance_benchmark.py \
  --duration 60 \
  --concurrency 5000 \
  --model hybrid_v3

# 输出:
# ✅ 准确率: 75.5%
# ✅ 平均延迟: 8.2ms
# ✅ 缓存命中: 85.3%
# ✅ 并发处理: 5,241 RPS

# 生成性能报告
python3 training/scripts/generate_week6_report.py
```

---

## 📊 成功验证表

| 指标 | 当前 | Week 6 目标 | 验证命令 | 状态 |
|------|------|-----------|---------|------|
| **模型准确率** | 41.67% | ≥75% | `python3 evaluate_model.py` | ⏳ |
| **并发吞吐** | 1,000 RPS | ≥5,000 RPS | `ab -n 10000 -c 100` | ⏳ |
| **缓存命中** | 53.77% | ≥85% | `curl http://localhost:8080/metrics` | ⏳ |
| **框架覆盖** | 6 个 | ≥12 个 | `grep frameworks data/samples.json` | ⏳ |
| **混淆检测** | 4 种 | ≥8 种 | `grep obfuscation data/samples.json` | ⏳ |

---

## 🔧 故障排查

**问题 1: 模型训练超时**
```bash
# 解决：使用预设超参数快速训练
python3 training/pipelines/train_hybrid_model_v3.py \
  --fast-mode \
  --epochs 50
```

**问题 2: 采样速度慢**
```bash
# 解决：启用并行采样
python3 training/scripts/collect_framework_samples.py \
  --parallel 8 \
  --timeout 30
```

**问题 3: API 内存溢出**
```bash
# 解决：增加内存限制或启用流式处理
./target/release/browerai-api-server \
  --max-batch-size 128 \
  --stream-large-responses
```

---

## 📈 关键检查点

**Day 2 检查**:
```bash
# 验证样本采集进度
ls -lh data/week6_samples/ | tail -1
# 预期: 600+ 样本

# 检查样本质量
python3 -c "import json; s=json.load(open('data/week6_samples/summary.json')); print(f\"框架覆盖: {len(s['frameworks'])} 个\")"
```

**Day 4 检查**:
```bash
# 验证模型训练进度
tail -20 logs/training.log
# 预期: 准确率 70%+, 损失 <0.3

# 查看最佳模型
cat models/best_model_metrics.json
```

**Day 6 检查**:
```bash
# 验证性能指标
curl http://localhost:8080/metrics | grep detect_latency
# 预期: p95 <20ms, p99 <50ms

# 检查缓存效率
curl http://localhost:8080/metrics | grep cache_hit
# 预期: 命中率 >80%
```

---

## 🎯 决策树

```
开始 Week 6
    ↓
[采集新样本] ← 成功？ → 否 → 增加采集超时/并发
    ↓ 是
[训练模型 v3] ← 准确率 <70%？ → 是 → 调整超参数/特征
    ↓ 否 (≥70%)
[集成模型] ← 准确率 <75%？ → 是 → 尝试其他集成方式
    ↓ 否 (≥75%)
[优化规则权重] ← 改进 <5%？ → 是 → 尝试贝叶斯优化
    ↓ 否
[性能优化] ← 吞吐 <5K RPS？ → 是 → 增加线程/批大小
    ↓ 否
[生成报告] ← 所有 P1 达成？ → 否 → 回到诊断阶段
    ↓ 是
完成 Week 6 ✅
```

---

## 📝 日志记录

每天执行后记录进度：

```bash
# Day 1-2: 数据增强完成
echo "✅ Day 2: $(date) - 样本数: $(find data/week6_samples -type f | wc -l)" >> WEEK6_PROGRESS.log

# Day 3-4: 模型训练完成
echo "✅ Day 4: $(date) - 模型准确率: $(cat models/best_metrics.json | grep accuracy)" >> WEEK6_PROGRESS.log

# Day 5: 性能优化完成
echo "✅ Day 5: $(date) - 吞吐: $(ab -n 1000 -c 100 http://localhost:8080/health 2>&1 | grep 'Requests per second')" >> WEEK6_PROGRESS.log

# Day 6-7: 验证与报告
echo "✅ Day 7: $(date) - Week 6 完成" >> WEEK6_PROGRESS.log
```

---

## 🚀 立即开始

```bash
# 1. 切换到项目目录
cd /home/stone/BrowerAI

# 2. 激活虚拟环境
source .venv/bin/activate

# 3. 运行诊断
python3 training/scripts/week6_diagnostics.py

# 4. 开始采样
python3 training/scripts/collect_framework_samples.py \
  --frameworks ember,backbone,alpine,htmx,nextjs,nuxt \
  --samples-per-framework 100 \
  --output data/week6_samples/

# 5. 监控进度
watch -n 5 'ls -lh data/week6_samples/ | tail -3'
```

---

**生成**: 2026-01-31 20:30 UTC  
**版本**: Week 6 v1.0  
**责任人**: BrowerAI Team  
**预计完成**: 2026-02-07

