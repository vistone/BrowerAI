# Week 8 Quick Start Guide - 5分钟快速开始

## 📋 Pre-Flight Checklist

### 必要文件验证
```bash
cd /home/stone/BrowerAI

# 检查 Week 8 文件
ls -la WEEK8_*.md          # 应该有多个文件
ls -la training/http_client.py
ls -la training/real_http_integration_tests.py
ls -la training/api_server.py  # Week 6 created
```

### 依赖检查
```bash
# 激活虚拟环境
source venv_test/bin/activate

# 验证关键包
python -c "import flask, numpy, requests, pydantic; print('✅ All deps OK')"
```

---

## 🚀 3-Step Quick Start (5 minutes)

### Step 1: 启动 API 服务器 (Terminal 1)
```bash
cd /home/stone/BrowerAI
source venv_test/bin/activate
cd training
python api_server.py
```

**Expected Output**:
```
[2024-01-XX 12:34:56] INFO: Starting Flask API server...
 * Running on http://127.0.0.1:5000
```

✅ **等待看到 "Running on http://127.0.0.1:5000"**

---

### Step 2: 在另一个终端运行测试 (Terminal 2)
```bash
cd /home/stone/BrowerAI
source venv_test/bin/activate
cd training
python real_http_integration_tests.py
```

**Expected Output**:
```
Starting Real HTTP Integration Tests...
┌─────────────────────────────────────┐
│  Testing Real HTTP Communication    │
└─────────────────────────────────────┘

[1/8] Test Health Check
  ✓ PASSED - Duration: 2.45ms
  
[2/8] Test Feature Generation
  ✓ PASSED - Duration: 3.12ms
  
... (continues for 8 tests)
```

✅ **等待看到 "8/8 tests passed"**

---

### Step 3: 查看性能报告
```bash
# 测试完成后，会自动生成报告
cat week8_test_results.json | python -m json.tool
```

---

## 📊 Success Criteria

### Phase A (Real HTTP Communication)
```
✅ All 8 tests passing
✅ Latency: < 5ms per request
✅ No timeout errors
✅ No connection errors
✅ Retry logic working
✅ Server health verified
```

### Performance Baseline
```
✅ Average Latency: 2-5ms (vs simulation 2.4ms)
✅ Throughput: 450+ RPS
✅ Memory: < 60MB
✅ CPU: 11-23%
```

---

## 🔧 Troubleshooting

### Issue: "Connection refused on 127.0.0.1:5000"
**Solution**: 确保 API 服务器正在运行（Step 1）
```bash
# 检查端口
lsof -i :5000
# 如果有进程，杀死它
kill -9 <PID>
# 重新启动 API 服务器
python training/api_server.py
```

### Issue: "TimeoutError: Read timed out"
**Reason**: API 服务器响应缓慢
**Solution**: 
```bash
# 检查 API 服务器日志
# 增加日志级别
export FLASK_ENV=development
python training/api_server.py
```

### Issue: "ModuleNotFoundError"
**Solution**: 确保虚拟环境激活
```bash
source venv_test/bin/activate
python -c "import flask, requests; print('OK')"
```

---

## 📈 What's Happening

### Real HTTP Client (`http_client.py`)
- 替换 HTTP 模拟为真实请求
- 添加 5 秒超时
- 实现重试逻辑（最多 3 次，指数退避）
- 连接池 (10 连接)
- 自动错误处理

### Integration Tests (`real_http_integration_tests.py`)
8 个测试场景：
1. **Health Check** - 验证服务器连接
2. **Feature Generation** - 生成 48D 特征向量
3. **Feedback Processing** - 处理质量反馈
4. **Learning Loop** - 3 次迭代学习循环
5. **Throughput** - 10 个连续请求
6. **Latency Consistency** - 5 个请求（方差 < 10ms）
7. **Network Resilience** - 超时和重试测试
8. **Error Handling** - 错误场景验证

---

## 🎯 Next Steps After Phase A

### Phase B: Stress Testing (3 hours)
```bash
# 创建压力测试框架
python training/stress_test.py --load 50 --duration 60
```

### Phase C: Docker (2 hours)
```bash
# 构建和测试容器
docker build -f Dockerfile.api -t browerai-api .
docker-compose up
```

### Phase D: Kubernetes (4 hours)
```bash
# 部署到本地集群
kubectl apply -f k8s/
kubectl get pods
```

---

## 📊 Real-time Monitoring (Optional)

### 在第三个终端运行性能监控
```bash
cd /home/stone/BrowerAI
source venv_test/bin/activate
python training/performance_monitor.py
```

**Monitors**:
- 实时延迟（毫秒）
- 请求吞吐量（RPS）
- 内存使用量
- CPU 使用率
- 错误率

---

## 📝 Configuration

### 修改 HTTP 客户端配置 (if needed)

编辑 `training/real_http_integration_tests.py`:
```python
# 第 1 行，更改配置
config = HttpClientConfig(
    base_url="http://127.0.0.1:5000",      # API 地址
    timeout=5.0,                            # 超时秒数
    max_retries=3,                          # 最多重试次数
    backoff_factor=0.5,                     # 退避系数
    pool_connections=10,                    # 连接池大小
    pool_maxsize=10                         # 最大池大小
)
```

---

## ✅ Phase A Completion Checklist

- [ ] API 服务器启动成功
- [ ] 所有 8 个测试通过
- [ ] 延迟 < 5ms
- [ ] 没有超时错误
- [ ] 没有连接错误
- [ ] 重试逻辑工作正常
- [ ] 性能报告生成
- [ ] 文档已读

---

## 🚀 You're Ready!

**Current Status**: ✅ Framework Complete
**Next Action**: Run the 3 steps above
**Estimated Time**: 5-10 minutes
**Success Rate**: Should be 100% (8/8 tests)

```bash
# TL;DR - Just run these three commands:

# Terminal 1
cd /home/stone/BrowerAI && source venv_test/bin/activate && cd training && python api_server.py

# Terminal 2
cd /home/stone/BrowerAI && source venv_test/bin/activate && cd training && python real_http_integration_tests.py
```

**Let's start! 🎯**
