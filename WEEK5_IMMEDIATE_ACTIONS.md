# 🚀 Week 5 立即行动计划

**当前状态**: Week 5 进行中，混合模型 v2 已完成  
**待处理文件**: 3,281 个（需要清理）  
**当前分支**: week5-postgresql-persistence

---

## ⚡ 立即执行 (优先级排序)

### 1️⃣ 清理工作目录 (15 分钟)

```bash
cd /home/stone/BrowerAI

# 选项 A: 放弃所有未提交的更改 (保留代码)
git clean -fd  # 删除未跟踪文件
git reset --hard HEAD  # 重置到最后一个 commit

# 选项 B: 提交当前工作 (推荐)
git add -A
git commit -m "chore: cleanup Week 5 intermediates"
git push origin week5-postgresql-persistence
```

---

### 2️⃣ PostgreSQL 持久化集成 (1 小时)

**目标**: 建立检测结果数据库存储

```bash
#  步骤 1: 初始化 PostgreSQL
sudo systemctl start postgresql
createdb browerai

# 步骤 2: 运行迁移
psql browerai << 'EOF'
CREATE TABLE detections (
    id SERIAL PRIMARY KEY,
    code_hash VARCHAR(64),
    detected_framework VARCHAR(50),
    confidence FLOAT,
    techniques JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE learning_metrics (
    id SERIAL PRIMARY KEY,
    model_version VARCHAR(50),
    accuracy FLOAT,
    precision FLOAT,
    recall FLOAT,
    timestamp TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_code_hash ON detections(code_hash);
CREATE INDEX idx_created ON detections(created_at);
EOF

# 步骤 3: 验证连接
python3 << 'EOF'
import psycopg2
conn = psycopg2.connect("dbname=browerai user=stone")
print("✅ PostgreSQL 连接成功")
conn.close()
EOF
```

---

### 3️⃣ Rust API 服务启动 (1 小时)

**目标**: 启动生产 API 服务，集成 PostgreSQL

```bash
# 步骤 1: 编译 API Server
cargo build --release \
  --package browerai-api-server \
  --features "postgresql"

# 步骤 2: 配置环境变量
cat > .env.production << 'EOF'
DATABASE_URL=postgresql://stone:password@localhost/browerai
API_PORT=8080
API_HOST=0.0.0.0
MAX_CONNECTIONS=20
REDIS_URL=redis://localhost:6379
EOF

# 步骤 3: 启动服务
./target/release/browerai-api-server

# 步骤 4: 测试端点 (新终端)
curl -X GET http://localhost:8080/health
curl -X POST http://localhost:8080/api/v1/detect \
  -H "Content-Type: application/json" \
  -d '{
    "code": "const React = require(\"react\");"
  }'
```

---

### 4️⃣ 模型性能评估 (1 小时)

**目标**: 评估当前模型在真实数据上的性能

```bash
# 步骤 1: 运行性能测试
python3 training/pipelines/final_production_system.py \
  --mode evaluate \
  --model-dir models/local \
  --test-data data/website_training_mixed_v2.jsonl \
  --output evaluation_report.json

# 步骤 2: 生成报告
python3 << 'EOF'
import json

with open('evaluation_report.json') as f:
    metrics = json.load(f)

print("\n📊 模型评估报告")
print("=" * 60)
for model, scores in metrics.items():
    print(f"\n{model}:")
    print(f"  准确率: {scores.get('accuracy', 0):.2%}")
    print(f"  精确率: {scores.get('precision', 0):.2%}")
    print(f"  召回率: {scores.get('recall', 0):.2%}")
    print(f"  F1分数: {scores.get('f1', 0):.2%}")
EOF
```

---

### 5️⃣ 生产部署检查清单 (0.5 小时)

```bash
# 检查所有必需组件
echo "🔍 生产部署检查..."

# ✅ 数据库
psql browerai -c "SELECT 1;" && echo "✅ PostgreSQL 运行中"

# ✅ API 服务
curl -s http://localhost:8080/health | grep -q healthy && echo "✅ API 服务运行中"

# ✅ 模型文件
ls -lh models/local/*.pt models/local/*.onnx 2>/dev/null | wc -l | xargs echo "✅ 模型文件数量:"

# ✅ 日志目录
mkdir -p logs && echo "✅ 日志目录准备完毕"

# ✅ 配置文件
test -f .env.production && echo "✅ 生产配置已设置"

echo "\n🎉 部署检查完成！"
```

---

## 📊 当前成果统计

| 指标 | 数值 | 状态 |
|-----|------|------|
| 真实网站数据 | 1,000+ | ✅ |
| 混合模型版本 | v2 | ✅ |
| 框架检测能力 | 8+ | ✅ |
| PostgreSQL | 待集成 | ⏳ |
| API 服务 | 待启动 | ⏳ |
| 生产就绪 | 70% | 🟡 |

---

## 🎯 Week 5 完成标准

- [ ] PostgreSQL 数据持久化完全集成
- [ ] API 服务正式上线
- [ ] 模型准确率评估完成
- [ ] 性能基准测试通过
- [ ] 生产部署文档完成
- [ ] Docker 镜像构建成功

---

## 📞 故障排查

**PostgreSQL 连接失败**:
```bash
# 检查服务
sudo systemctl status postgresql

# 创建用户 (如需要)
sudo -u postgres createuser stone
sudo -u postgres createdb browerai -O stone
```

**API 启动失败**:
```bash
# 检查端口
netstat -tlnp | grep 8080

# 检查日志
cat logs/api_server.log
```

**模型加载失败**:
```bash
# 验证模型文件
file models/local/*.pt
ls -lh models/local/

# 检查版本
python3 -c "import torch; print(torch.__version__)"
```

---

## ⏭️ 执行顺序

1. **立即**: 清理工作目录 (`git clean` / `git reset`)
2. **第一小时**: PostgreSQL 初始化
3. **第二小时**: API 服务启动和测试
4. **第三小时**: 模型评估
5. **第四小时**: 生产部署检查和文档

**预计总时间**: 4 小时

---

**下一步**: 开始任务 1 - 清理工作目录

```bash
cd /home/stone/BrowerAI
git status | head -20  # 检查状态
git add -A
git commit -m "Week 5: PostgreSQL integration and API server startup"
```
