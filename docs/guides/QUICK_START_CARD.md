# 🚀 WEEK 5 快速启动卡

**当前状态**: 实网站学习完成, 生产部署最后冲刺  
**完成度**: 60% → 目标 100%  
**预计耗时**: 2 小时

---

## ⚡ 四步完成 Week 5

### STEP 1️⃣: PostgreSQL (30 分钟)

```bash
# 启动数据库
sudo systemctl start postgresql

# 创建数据库
createdb browerai

# 创建表
psql browerai << 'EOF'
CREATE TABLE detections (
  id SERIAL PRIMARY KEY,
  code_hash VARCHAR(64) UNIQUE,
  framework VARCHAR(50),
  confidence FLOAT,
  techniques JSONB,
  timestamp TIMESTAMP DEFAULT NOW()
);

CREATE TABLE metrics (
  id SERIAL PRIMARY KEY,
  metric_name VARCHAR(100),
  value FLOAT,
  timestamp TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_hash ON detections(code_hash);
CREATE INDEX idx_time ON detections(timestamp);
EOF

# 验证
psql browerai -c "SELECT 1;" && echo "✅ PostgreSQL OK"
```

---

### STEP 2️⃣: API 服务 (30 分钟)

```bash
# 编译
cargo build --release --package browerai-api-server

# 配置
cat > .env.production << 'EOF'
DATABASE_URL=postgresql://stone@localhost/browerai
API_HOST=0.0.0.0
API_PORT=8080
EOF

# 启动 (后台运行)
./target/release/browerai-api-server &

# 等待 3 秒
sleep 3

# 测试
curl http://localhost:8080/health

# 预期输出: {"status":"healthy"}
```

---

### STEP 3️⃣: 模型评估 (30 分钟)

```bash
# 运行评估
python3 training/pipelines/final_production_system.py \
  --mode evaluate \
  --model-dir models/local \
  --output evaluation_result.json

# 查看结果
cat evaluation_result.json | python3 -m json.tool | head -50
```

---

### STEP 4️⃣: 验证检查 (15 分钟)

```bash
echo "🔍 验证生产就绪..."

# ✅ PostgreSQL
psql browerai -c "SELECT COUNT(*) FROM detections;" && echo "✅ DB OK"

# ✅ API
curl -s http://localhost:8080/health | grep -q healthy && echo "✅ API OK"

# ✅ 模型
ls -lh models/local/*.pt 2>/dev/null | wc -l | xargs echo "✅ Models:"

# ✅ 日志
test -d logs && echo "✅ Logs OK"

# ✅ 配置
test -f .env.production && echo "✅ Config OK"

echo "✨ 检查完成!"
```

---

## 📊 成功标志

| 组件 | 检查命令 | 成功标志 |
|-----|---------|--------|
| PostgreSQL | `psql browerai -c "SELECT 1;"` | ✅ 无错误 |
| API | `curl http://localhost:8080/health` | ✅ 200 OK |
| 模型 | `ls models/local/*.pt \| wc -l` | ✅ 3 个文件 |
| 数据库 | `psql browerai -c "TABLE detections;"` | ✅ 创建成功 |

---

## 🎯 测试 API

```bash
# 框架检测
curl -X POST http://localhost:8080/api/v1/detect \
  -H "Content-Type: application/json" \
  -d '{
    "code": "const React = require(\"react\");"
  }'

# 期望输出:
# {"framework":"React","confidence":0.92,"techniques":[...]}

# 混淆检测
curl -X POST http://localhost:8080/api/v1/obfuscate \
  -H "Content-Type: application/json" \
  -d '{
    "code": "function test(){return 42;}"
  }'

# 期望输出:
# {"obfuscated":true,"techniques":["control_flow"],...}
```

---

## 📝 快速日志检查

```bash
# 实时监控 API 日志
tail -f logs/api_server.log

# 监控数据库活动
watch -n 1 "psql browerai -c \"SELECT COUNT(*) FROM detections;\""

# 监控系统资源
watch -n 1 "free -h && echo && ps aux | grep browerai"
```

---

## 🚨 故障排除

| 错误 | 原因 | 解决 |
|-----|-----|------|
| `psql: 找不到...` | PostgreSQL 未启动 | `sudo systemctl start postgresql` |
| `Connection refused` | API 未启动 | `./target/release/browerai-api-server` |
| `模型不存在` | 路径错误 | 检查 `models/local/` 目录 |
| `权限被拒绝` | 权限问题 | `sudo chown -R stone:stone .` |

---

## ✨ 预期成果

完成 4 步后:

✅ PostgreSQL 运行中  
✅ API 服务响应请求  
✅ 框架检测可用  
✅ 混淆检测可用  
✅ 所有数据持久化  

**系统状态**: 🟢 生产就绪

---

## 📞 需要帮助?

```bash
# 查看详细文档
cat WEEK5_IMMEDIATE_ACTIONS.md    # 完整行动计划
cat WEEK5_PROGRESS_REPORT.md      # 进度报告

# 查看日志
tail -50 logs/api_server.log      # API 日志
tail -50 logs/training.log        # 训练日志

# 查看状态
git status                        # 代码状态
cargo build --package browerai-api-server 2>&1 | tail -20  # 编译状态
```

---

## ⏱️ 时间表

| 时间 | 任务 | 预计 |
|-----|-----|------|
| 现在 | 阅读此卡 | 2 分钟 |
| 第 1 步 | PostgreSQL | 30 分钟 |
| 第 2 步 | API 服务 | 30 分钟 |
| 第 3 步 | 模型评估 | 30 分钟 |
| 第 4 步 | 验证检查 | 15 分钟 |
| **总计** | | **105 分钟** |

---

## 🎊 完成后

```bash
# 提交工作
git add -A
git commit -m "✨ Week 5 Complete: PostgreSQL + API + Evaluation"
git push origin week5-postgresql-persistence

# 更新进度
echo "✅ Week 5 COMPLETE - Production Ready 100%" >> WEEK5_PROGRESS_REPORT.md

# 准备 Week 6
echo "Ready for Week 6: GPU Acceleration & Distributed Inference"
```

---

**现在就开始! 👇**

```bash
cd /home/stone/BrowerAI
sudo systemctl start postgresql
```
