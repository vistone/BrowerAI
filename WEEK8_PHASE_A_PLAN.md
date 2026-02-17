# Week 8 Phase A: Real HTTP Communication Implementation

**目标**: 将 HTTP 模拟替换为真实的网络通信，添加超时和重试逻辑

**预计时间**: 2 小时
**优先级**: HIGH

## 任务清单

### 1. 创建真实 HTTP 客户端模块
- [ ] 实现带重试的 HTTP 客户端
- [ ] 添加超时处理
- [ ] 实现指数退避算法
- [ ] 添加连接池

### 2. 更新集成测试
- [ ] 替换模拟请求为真实 HTTP 调用
- [ ] 添加网络错误处理
- [ ] 验证超时逻辑
- [ ] 添加连接重试测试

### 3. 网络弹性测试
- [ ] 测试连接超时
- [ ] 测试请求超时
- [ ] 测试网络故障恢复
- [ ] 测试并发连接

### 4. 性能监控
- [ ] 测量真实 HTTP 延迟
- [ ] 对比模拟 vs 真实延迟
- [ ] 验证吞吐量不变
- [ ] 检查资源使用

## 实现步骤

### Step 1: HTTP 客户端实现
```python
class RealHttpClient:
    def __init__(self, base_url, timeout=5, max_retries=3):
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
        
    async def post(self, endpoint, data, retry=True):
        # 带重试和超时的 POST 请求
        
    async def get(self, endpoint, retry=True):
        # 带重试和超时的 GET 请求
```

### Step 2: 集成测试更新
```python
# 从:
response = self.simulate_health_check()

# 改为:
response = await self.http_client.get("/api/v1/health")
```

### Step 3: 错误处理
```python
try:
    response = await self.http_client.post(endpoint, data)
except ConnectTimeout:
    # 处理连接超时
except RequestTimeout:
    # 处理请求超时
except ConnectionError:
    # 处理连接错误
```

## 成功标准

- [ ] 所有 7 个测试用真实 HTTP 通过
- [ ] 超时处理工作正常
- [ ] 重试逻辑有效
- [ ] 延迟增加不超过 10%
- [ ] 吞吐量保持不变 (450+ RPS)
- [ ] 没有新的错误

## 预期结果

- ✅ 从模拟升级到真实 HTTP
- ✅ 网络弹性验证
- ✅ 生产就绪的客户端
- ✅ 为 Stress Testing 做好准备
