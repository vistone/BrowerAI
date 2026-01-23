# Week 5 执行计划 - 真实网站数据驱动学习

**启动日期**: 2026-01-31  
**当前阶段**: Week 5 - PostgreSQL 持久化 & 大规模学习  
**目标**: 将 1000+ 真实网站学习系统投入生产

---

## 📋 任务总览

### 🎯 核心目标 (Week 5)

1. **PostgreSQL 数据持久化** ⏳ 正在进行
   - 模型检测结果存储
   - 学习反馈记录
   - 性能指标追踪

2. **1000+ 真实网站学习** ✅ 基础完成
   - 已收集 59.57 MB 真实代码数据
   - 框架检测基础能力建立
   - 混淆检测初步实现

3. **模型性能优化** ⏳ 待优化
   - 框架检测准确率: 基础 → 85%+
   - 混淆检测准确率: 基础 → 80%+
   - 代码恢复率: 基础 → 85%+

4. **生产部署** ⏳ 待进行
   - Docker 容器化
   - API 服务启动
   - 监控告警集成

---

## ⚡ 立即执行的任务 (下一步)

### 任务 1: 生成大规模训练数据集 (1-2 小时)

**文件**: `training/scaleable_data_generator.py`  
**目标**: 生成 10,000+ 混淆/未混淆代码对

```bash
# 运行大规模数据生成器
cd /home/stone/BrowerAI
python training/scaleable_data_generator.py --output real_data/training_pairs_large.jsonl
```

**输出**: 
- `real_data/training_pairs_large.jsonl` - 大规模训练对集合
- `real_data/statistics.json` - 数据统计报告

---

### 任务 2: 训练框架检测模型 (优化版本) (2-3 小时)

**目标**: 提高框架检测准确率到 85%+

```bash
# 步骤 1: 使用生产级训练器
python training/trainers/production_trainer.py \
  --mode detect \
  --data real_data/training_pairs_large.jsonl \
  --epochs 50 \
  --batch-size 64 \
  --output models/local/framework_detector_v2.pt

# 步骤 2: GPU 加速训练 (如果可用)
python training/trainers/enhanced_gpu_trainer.py \
  --model framework_detector \
  --learning-rate 0.001 \
  --dropout 0.3
```

**期望成果**:
- ✅ 框架检测准确率: 85%+
- ✅ 支持 24 个框架
- ✅ 推理延迟: <50ms

---

### 任务 3: PostgreSQL 集成 (1-2 小时)

**目标**: 建立完整的数据持久化层

```bash
# 步骤 1: 初始化 PostgreSQL
sudo systemctl start postgresql
createdb browerai

# 步骤 2: 运行迁移脚本
psql browerai -f crates/browerai-db/migrations/001_init.sql

# 步骤 3: 验证连接
python -c "
from crates.browerai_persistent_layer import PersistentStorage
storage = PersistentStorage()
print('✅ PostgreSQL 连接成功')
"
```

**数据表**:
- `detections` - 检测结果存储
- `learning_feedback` - 学习反馈
- `performance_metrics` - 性能指标
- `model_versions` - 模型版本管理

---

### 任务 4: 混淆检测优化 (2 小时)

**目标**: 混淆检测准确率 80%+

```bash
# 步骤 1: 训练混淆检测模型
python training/trainers/obfuscation_trainer.py \
  --data real_data/obfuscation_samples.jsonl \
  --techniques "variable_rename,control_flow,string_encoding,dead_code" \
  --epochs 100

# 步骤 2: 性能测试
python training/validation/test_obfuscation_detection.py \
  --model models/local/obfuscation_detector_v2.pt \
  --test-samples 1000
```

---

### 任务 5: API 服务启动 (0.5 小时)

**目标**: 启动生产 API 服务

```bash
# 步骤 1: 编译 API Server
cargo build --release \
  --package browerai-api-server \
  --features "postgresql,redis"

# 步骤 2: 启动服务
./target/release/browerai-api-server \
  --host 0.0.0.0 \
  --port 8080 \
  --db-url postgresql://localhost/browerai

# 步骤 3: 验证健康检查
curl http://localhost:8080/health
```

---

## 📊 优先级清单

### 🔴 高优先级 (今天完成)

- [ ] 运行大规模数据生成器
- [ ] 启动框架检测模型训练
- [ ] PostgreSQL 初始化和验证
- [ ] API 健康检查

### 🟡 中优先级 (本周完成)

- [ ] 混淆检测模型优化
- [ ] 性能基准测试
- [ ] Docker 容器化
- [ ] 监控面板配置

### 🟢 低优先级 (本周末完成)

- [ ] 文档完善
- [ ] 部署脚本优化
- [ ] 性能报告生成

---

## 🚀 快速命令参考

### 启动完整学习管道

```bash
cd /home/stone/BrowerAI

# 1. 生成数据
echo "📊 生成训练数据..."
python training/scaleable_data_generator.py

# 2. 训练模型
echo "🤖 训练框架检测模型..."
python training/trainers/production_trainer.py --mode detect

# 3. 测试性能
echo "⚡ 测试模型性能..."
python training/validation/test_framework_detection.py

# 4. 启动 API
echo "🚀 启动 API 服务..."
cargo build --release --package browerai-api-server
./target/release/browerai-api-server
```

### 监控学习进度

```bash
# 查看实时进度
tail -f LEARNING_PROGRESS.md

# 查看训练日志
tail -f logs/training_latest.log

# 测试 API 端点
curl -X POST http://localhost:8080/detect \
  -H "Content-Type: application/json" \
  -d '{"code": "function test() { return 42; }"}'
```

---

## ✅ 完成标准

### Week 5 成功标准

| 指标 | 目标 | 状态 |
|-----|------|------|
| 数据规模 | 10,000+ 训练对 | ⏳ |
| 框架检测准确率 | 85%+ | ⏳ |
| 混淆检测准确率 | 80%+ | ⏳ |
| 代码恢复率 | 85%+ | ⏳ |
| API 延迟 | <100ms | ⏳ |
| 并发吞吐量 | 100+ req/s | ⏳ |
| PostgreSQL 集成 | 完全 | ⏳ |
| Docker 就绪 | 可部署 | ⏳ |

---

## 📞 需要帮助？

### 常见问题

**Q: 数据生成耗时过长？**  
A: 使用 `--sample-limit 1000` 快速测试

**Q: 显存不足？**  
A: 降低 batch-size: `--batch-size 32`

**Q: PostgreSQL 连接失败？**  
A: 检查 `.env` 中的数据库配置

---

**下一步**: 立即执行任务 1 - 数据生成器

```bash
cd /home/stone/BrowerAI && python training/scaleable_data_generator.py
```
