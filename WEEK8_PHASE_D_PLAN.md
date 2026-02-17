# Week 8 Phase D - Kubernetes 部署计划

## 阶段概述

**目标**: 将容器化的 API Server 部署到 Kubernetes 集群，实现生产级高可用部署。

**时间**: 4-6 小时  
**优先级**: HIGH  
**前置条件**: Phase A + B + C ✅

---

## 1. 部署架构

### 1.1 Kubernetes 资源

```
┌─────────────────────────────────────────────────────┐
│           Kubernetes Cluster (Minikube)             │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ┌──────────────────────────────────────────────┐  │
│  │  Namespace: browerai                         │  │
│  ├──────────────────────────────────────────────┤  │
│  │                                              │  │
│  │  Deployment (browerai-api)                  │  │
│  │  ├─ Replica 1: Pod (5000)                   │  │
│  │  ├─ Replica 2: Pod (5000)                   │  │
│  │  └─ Replica 3: Pod (5000)                   │  │
│  │                                              │  │
│  │  Service (ClusterIP)                        │  │
│  │  └─ :5000 → Pod:5000 (负载均衡)            │  │
│  │                                              │  │
│  │  HPA (Horizontal Pod Autoscaler)            │  │
│  │  ├─ Min Replicas: 2                         │  │
│  │  ├─ Max Replicas: 10                        │  │
│  │  ├─ CPU Target: 70%                         │  │
│  │  └─ Memory Target: 80%                      │  │
│  │                                              │  │
│  │  Ingress (browerai.localhost)               │  │
│  │  └─ Route /api → Service:5000               │  │
│  │                                              │  │
│  │  PDB (Pod Disruption Budget)                │  │
│  │  └─ Min Available: 2                        │  │
│  │                                              │  │
│  │  RBAC (ServiceAccount + Role)               │  │
│  │  NetworkPolicy (Egress/Ingress)             │  │
│  │                                              │  │
│  └──────────────────────────────────────────────┘  │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### 1.2 资源清单文件

```
k8s/
├── deployment.yaml       # Deployment + Service + HPA + RBAC
├── ingress.yaml         # Ingress + LoadBalancer Service
├── monitoring.yaml      # ResourceQuota + LimitRange + Monitoring
├── storage.yaml         # (可选) PersistentVolume 配置
└── kustomization.yaml   # (可选) Kustomize 编排
```

---

## 2. 部署步骤

### Step 1: 环境准备 (15 分钟)
- ✅ 检查 kubectl 和 Minikube
- ✅ 启动 Minikube 集群
- ✅ 加载 Docker 镜像到 Minikube
- ✅ 验证集群连接

### Step 2: 创建 K8s 资源 (20 分钟)
- ✅ 创建 namespace
- ✅ 应用 ConfigMap 和 Secret
- ✅ 应用 Deployment
- ✅ 应用 Service
- ✅ 应用 Ingress
- ✅ 应用 HPA

### Step 3: 验证部署 (15 分钟)
- ✅ 检查 Pod 状态
- ✅ 检查 Service 连接
- ✅ 验证 HPA 就绪
- ✅ 测试 API 端点

### Step 4: 蓝绿部署验证 (20 分钟)
- ✅ 部署蓝色版本 (当前)
- ✅ 部署绿色版本 (新版本)
- ✅ 流量切换
- ✅ 回滚测试

### Step 5: 压力测试 (20 分钟)
- ✅ 10 并发请求
- ✅ 25 并发请求
- ✅ 50 并发请求
- ✅ 100 并发请求
- ✅ HPA 自动扩展验证

### Step 6: 性能验证 (15 分钟)
- ✅ 对比 Host vs Docker vs K8s
- ✅ 延迟分析
- ✅ 吞吐量对比
- ✅ 资源使用监控

### Step 7: 报告生成 (10 分钟)
- ✅ 生成执行报告
- ✅ 性能对比表
- ✅ 问题汇总

---

## 3. 关键配置说明

### 3.1 Deployment 配置
```yaml
副本数: 3 (初始)
更新策略: RollingUpdate (滚动更新)
  maxSurge: 1 (最多增加 1 个)
  maxUnavailable: 0 (保证可用)

资源请求:
  CPU: 500m (最小)
  Memory: 256Mi (最小)

资源限制:
  CPU: 2000m (最多)
  Memory: 1Gi (最多)

健康检查:
  Liveness: /health (30s)
  Readiness: /health (10s)
  Startup: /health (300s max)

优雅关闭:
  终止宽限期: 30s
  Pre-stop hook: sleep 15s
```

### 3.2 HPA 配置
```yaml
最小副本: 2
最大副本: 10

扩展指标:
  CPU: > 70% 时扩展
  Memory: > 80% 时扩展

扩展行为:
  向上: 1 分钟内增加 100% 或 2 个 Pod
  向下: 5 分钟稳定后逐步减少 50%
```

### 3.3 RBAC 配置
```yaml
ServiceAccount: browerai-api
Role: 只能访问 ConfigMap 和 Secret
RoleBinding: 连接 ServiceAccount 和 Role
```

### 3.4 网络策略
```yaml
Ingress: 允许来自 Ingress Controller 和同 Namespace 的流量
Egress: 允许 DNS、数据库、外部 HTTP/HTTPS
```

---

## 4. 成功标准

| 指标 | 目标 | 验证方法 |
|------|------|--------|
| Deployment | 成功 | kubectl get deployment |
| Pod 就绪 | 3/3 ready | kubectl get pods |
| Service | 可访问 | kubectl get service |
| HPA | 就绪 | kubectl get hpa |
| Ingress | 分配 IP | kubectl get ingress |
| 健康检查 | 100% | curl /health |
| API 可用 | 100% | 所有端点测试 |
| 自动扩展 | 正常工作 | 高负载下观察副本增加 |
| 蓝绿部署 | 零停机 | 切换过程中无请求失败 |
| 性能对标 | Host: 100% | 性能 ≥ 95% |

---

## 5. 时间表

```
[ 0 - 15分钟 ] 环境准备 (kubectl, Minikube, 镜像)
[ 15 - 35分钟 ] 创建 K8s 资源
[ 35 - 50分钟 ] 验证部署
[ 50 - 70分钟 ] 蓝绿部署测试
[ 70 - 90分钟 ] 压力测试 (4 个负载等级)
[ 90 - 105分钟 ] 性能验证和分析
[ 105 - 115分钟 ] 报告生成
```

---

## 6. 预期结果

### 6.1 部署信息
```
Namespace: browerai
Deployment: browerai-api-deployment
  Replicas: 3
  Ready: 3/3
  Available: 3/3

Service: browerai-api-service
  Type: ClusterIP
  Port: 5000
  Selector: app=browerai-api

Ingress: browerai-api-ingress
  Host: browerai.localhost
  Path: /
  Service: browerai-api-service:5000

HPA: browerai-api-hpa
  Min: 2, Max: 10
  Current: 3
  Targets: CPU 70%, Memory 80%

PDB: browerai-api-pdb
  Min Available: 2
```

### 6.2 性能指标 (预期)
```
吞吐量: 155-160 RPS (保留 95%+)
延迟: 3.9-4.2ms (保留 110%)
成功率: 100%
P95: 8-9ms
P99: 13-15ms

资源使用 (50 并发):
  CPU: 35-40%
  内存: 200-250MB
```

### 6.3 HPA 验证 (预期)
```
正常负载 (50 并发):
  副本数: 3
  CPU: 35-40%
  内存: 80-90%

高负载 (100 并发):
  副本数: 5-7 (自动扩展)
  CPU: 40-50%
  内存: 60-70% (分散)
```

### 6.4 蓝绿部署验证
```
切换过程:
  0s: 绿色版本部署完成
  0s: 流量 100% → 蓝色
  30s: 流量切换 → 绿色
  60s: 蓝色版本下线
  结果: 零停机, 0 个请求失败
```

---

## 7. 故障排查

| 问题 | 原因 | 解决方案 |
|------|------|--------|
| Pod CrashLoopBackOff | 健康检查失败 | 检查日志: kubectl logs |
| Pending Pod | 资源不足 | 增加 node 或降低请求 |
| 无法访问 Service | 网络策略 | 检查 NetworkPolicy |
| HPA 不工作 | 指标不可用 | 检查 metrics-server |
| 蓝绿部署失败 | 版本冲突 | 检查镜像和标签 |

---

## 8. 生产检查清单

### 安全性
- [ ] RBAC 配置完整
- [ ] NetworkPolicy 启用
- [ ] 非 root 用户运行
- [ ] Secret 使用加密存储
- [ ] Pod 安全策略配置

### 可靠性
- [ ] 健康检查配置
- [ ] Pod 反亲和性设置
- [ ] PDB 最小可用性
- [ ] 优雅关闭配置
- [ ] 重试和超时策略

### 可观测性
- [ ] 日志聚合配置
- [ ] Prometheus 监控
- [ ] 告警规则定义
- [ ] Grafana 仪表板
- [ ] 性能指标导出

### 可扩展性
- [ ] HPA 配置正确
- [ ] 资源配额设置
- [ ] LimitRange 定义
- [ ] 扩展行为配置
- [ ] 节点容量规划

---

## 9. 下一步

✅ Phase D: Kubernetes 部署 (本阶段)
⏳ Phase E: CI/CD 集成
  - GitHub Actions 工作流
  - 自动化构建和推送
  - 自动化部署流程

⏳ Phase F: 生产上线
  - 云平台部署 (AWS/GCP/Azure)
  - 域名和 SSL/TLS
  - 自动备份和恢复
  - 灾难恢复计划

---

## 10. 参考资源

- K8s 文档: https://kubernetes.io/docs/
- Kubectl: https://kubernetes.io/docs/reference/kubectl/
- Minikube: https://minikube.sigs.k8s.io/
- HPA: https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/
- Ingress: https://kubernetes.io/docs/concepts/services-networking/ingress/
- RBAC: https://kubernetes.io/docs/reference/access-authn-authz/rbac/

---

**文档更新**: 2026-02-01  
**版本**: 1.0.0  
**状态**: 待执行
