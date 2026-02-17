# Week 8 Phase C - Docker 容器化计划

## 阶段概述

**目标**: 将 Week 6-7 验证的 API Server 容器化，实现生产级别的部署。

**时间**: 2 小时
**优先级**: HIGH
**前置条件**: Phase A + Phase B ✅

---

## 1. 实现计划

### 1.1 Docker 镜像优化

**目标**: 创建小型、安全、高效的生产镜像

**要求**:
- ✅ 多阶段构建 (Multi-stage build)
- ✅ 使用 slim Python 基础镜像 (python:3.11-slim)
- ✅ 非 root 用户运行
- ✅ 健康检查
- ✅ 资源限制
- ✅ 镜像大小 < 1GB

**文件**:
- `Dockerfile.python-api`: Python API 优化构建
- `.dockerignore`: 排除不必要文件

**预期镜像大小**:
- 基础镜像: 130MB
- 依赖: 300-400MB
- 应用代码: 5MB
- **总计**: ~500-600MB (比 Rust ~2GB 小 70%)

---

### 1.2 Docker Compose 编排

**目标**: 定义完整的容器栈

**服务**:
1. **api-server**: Python Flask 应用
   - 端口: 5000
   - 工作进程: 4 (gunicorn)
   - 线程: 2 per worker
   - CPU 限制: 2.0 核心
   - 内存限制: 1GB

2. **nginx**: 反向代理 (可选)
   - 端口: 80/443
   - 负载均衡
   - SSL/TLS 支持

3. **prometheus**: 指标收集
   - 端口: 9090
   - 数据保留: 30 天

4. **grafana**: 可视化仪表板
   - 端口: 3000
   - 预配置的数据源

**文件**: `docker-compose.python-api.yml`

---

### 1.3 容器测试策略

#### 测试 1: 构建验证
- ✅ 镜像构建成功
- ✅ 镜像大小验证
- ✅ 镜像扫描 (漏洞)
- ✅ 基础镜像信息

#### 测试 2: 容器启动
- ✅ 容器启动成功
- ✅ 健康检查通过
- ✅ 端口映射正确
- ✅ 日志输出正常

#### 测试 3: API 功能测试
- ✅ 所有端点可访问
- ✅ 请求/响应正确
- ✅ 错误处理完善
- ✅ 响应时间符合预期

#### 测试 4: 容器内压力测试
- ✅ 10 并发 (100 请求)
- ✅ 25 并发 (250 请求)
- ✅ 50 并发 (500 请求)
- ✅ 100 并发 (500 请求)

#### 测试 5: 性能对比
- Host vs Container 延迟差异 < 10%
- 吞吐量损失 < 5%
- 内存占用 (容器) vs (Host) 对比
- CPU 使用率对比

#### 测试 6: 资源监控
- ✅ CPU 使用率 < 50% (正常负载)
- ✅ 内存使用率 < 80% (极限负载)
- ✅ 网络 I/O 监控
- ✅ 磁盘 I/O 监控

---

## 2. 成功标准

| 指标 | 目标 | 验证方法 |
|------|------|--------|
| 镜像构建 | 成功，无警告 | Docker build log |
| 镜像大小 | < 1GB | `docker images` |
| 容器启动 | < 3秒 | 启动时间测量 |
| 健康检查 | 100% 通过 | curl /health |
| API 可用性 | 100% | 所有端点测试 |
| 压力测试 | 100% 成功率 | 4 个负载等级 |
| 性能对比 | Host: 100%, Container: > 95% | RPS 对比 |
| 资源效率 | CPU < 50%, Mem < 1GB | htop / docker stats |
| 安全性 | 非 root, 无已知漏洞 | docker scan |
| 监控集成 | Prometheus + Grafana 可用 | Web UI 访问 |

---

## 3. 实现时间表

```
[ 0 - 15分钟 ] 创建 Dockerfile 和 docker-compose
[ 15 - 30分钟 ] 构建镜像和初步测试
[ 30 - 45分钟 ] 容器启动和功能验证
[ 45 - 75分钟 ] 容器内压力测试 (4 个等级)
[ 75 - 105分钟 ] 性能分析和对比
[ 105 - 120分钟 ] 文档和报告
```

---

## 4. 预期结果

### 4.1 镜像信息
```
Repository: browerai-api:latest
Tag: latest
Size: ~550MB
Base: python:3.11-slim
Build Time: ~3-5分钟
Layers: 25-30 (optimized)
```

### 4.2 容器性能
- **启动时间**: 1-3 秒
- **首次响应**: < 100ms
- **吞吐量**: 150+ RPS (容器内)
- **延迟**: 2-10ms (容器内)
- **内存**: 150-200MB (运行中)
- **CPU**: 15-40% (正常负载)

### 4.3 性能对比 (Host vs Container)
```
指标              Host        Container    损失      状态
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RPS (50 并发)     164 RPS     155-160 RPS  < 3%     ✅
延迟 (平均)       3.61ms      3.8-4.2ms   < 15%     ✅
内存              46MB        180-200MB   本地化   ✅
CPU               30%         35-40%      +5-10%   ✅
启动时间          瞬间        1-3秒       可接受   ✅
```

---

## 5. 监控和可观测性

### 5.1 内置健康检查
```bash
GET /health
Response: { "status": "healthy", "timestamp": "..." }
```

### 5.2 Prometheus 指标
- `http_requests_total`: 请求总数
- `http_request_duration_seconds`: 请求延迟
- `http_requests_in_progress`: 进行中的请求
- `process_resident_memory_bytes`: 内存使用

### 5.3 Grafana 仪表板
- 实时请求速率
- 延迟分布 (P50, P95, P99)
- 错误率
- 资源使用趋势
- 容器性能

---

## 6. 故障排查

| 问题 | 原因 | 解决方案 |
|------|------|--------|
| 构建失败 | 依赖错误 | 检查 requirements.txt |
| 启动失败 | 健康检查超时 | 增加 start_period |
| 高延迟 | CPU/内存不足 | 增加资源限制 |
| 连接错误 | 网络配置 | 检查 docker-compose 网络 |
| 性能下降 | I/O 瓶颈 | 检查磁盘和网络 |

---

## 7. 交付物

✅ 代码:
- `Dockerfile.python-api`: 生产就绪的 Docker 构建
- `docker-compose.python-api.yml`: 完整的容器编排
- `.dockerignore`: 优化的构建上下文
- `run_phase_c_tests.sh`: 自动化测试脚本

✅ 文档:
- `WEEK8_PHASE_C_PLAN.md`: 本文档
- `WEEK8_PHASE_C_EXECUTION_REPORT.md`: 执行结果

✅ 结果:
- Docker 镜像 (browerai-api:latest)
- 压力测试结果 JSON
- 性能对比报告
- Grafana 仪表板截图

---

## 8. 关键决策

1. **多阶段构建**: 将镜像大小从 2GB 减少到 550MB
2. **Gunicorn + Gthread**: 支持并发和异步操作
3. **非 root 用户**: 提高安全性
4. **Resource Limits**: 防止容器消耗过多资源
5. **健康检查**: 自动重启失败的容器
6. **日志驱动**: JSON-file with rotation (防止磁盘撑满)

---

## 9. 后续步骤

✅ Phase C: Docker 容器化 (本阶段)
⏳ Phase D: Kubernetes 部署
  - 部署配置 (Deployment, Service, Ingress)
  - 自动扩展 (HPA)
  - 负载均衡
  - 蓝绿部署

⏳ Phase E: CI/CD 集成
  - GitHub Actions
  - 自动构建和推送镜像
  - 自动化测试
  - 自动部署

---

## 10. 相关资源

- Docker 文档: https://docs.docker.com/
- Docker Compose: https://docs.docker.com/compose/
- Best Practices: https://docs.docker.com/develop/dev-best-practices/
- Gunicorn 配置: https://gunicorn.org/
- Prometheus: https://prometheus.io/docs/
- Grafana: https://grafana.com/docs/

---

**文档更新**: 2026-02-01
**版本**: 1.0.0
**状态**: 待执行
