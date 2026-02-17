# Week 8 阶段 E 部署执行计划

**文档版本**: 2.0.0  
**创建日期**: 2026-02-01  
**状态**: 🚀 准备执行  
**预计耗时**: 60-75 分钟

---

## 📊 执行时间表

```
┌─────────────────────────────────────────────────────────────┐
│ Week 8 Phase E 完整部署执行时间表                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│ 时间段        任务内容              耗时      负责人         │
│ ──────────────────────────────────────────────────────────│
│ 13:00-13:15  Step 1: 配置 Secrets   15 分钟   手动或脚本  │
│ 13:15-13:35  Step 2: 测试工作流      20 分钟   自动化+监控 │
│ 13:35-13:45  Step 3: 验证部署       10 分钟   自动化验证  │
│ 13:45-14:05  Step 4: 配置监控       20 分钟   手动配置   │
│ 14:05-14:15  整体测试验证          10 分钟   手动测试   │
│ ──────────────────────────────────────────────────────────│
│ 总计: 13:00-14:15               75 分钟                     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎯 Step 1: 配置 GitHub Secrets (13:00-13:15)

### 目标
在 GitHub 中配置 4 个关键 Secrets，供工作流使用

### 前置条件检查

```bash
# 1. 检查 GitHub CLI 安装
command -v gh && gh auth status
# 预期: "Logged in to github.com as <username>"

# 2. 检查 Docker Hub 账户
# 需要: Docker Hub 用户名 + PAT token

# 3. 检查 Kubernetes 集群
kubectl cluster-info
# 预期: "Kubernetes control plane is running"

# 4. 检查 kubeconfig 文件
test -f ~/.kube/config && echo "✅ kubeconfig 存在"
```

### 执行步骤

**选项 A: 交互式脚本 (推荐)**

```bash
# 运行交互式配置脚本
cd /home/stone/BrowerAI
bash .github/scripts/setup-secrets.sh

# 脚本会提示输入:
# 1. Docker Hub 用户名
# 2. Docker Hub PAT 令牌
# 3. Kubernetes 配置路径
# 4. Kubernetes 上下文名称
```

**选项 B: 手动 CLI 配置**

```bash
# 获取必要的值
DOCKER_USERNAME="your-docker-username"
DOCKER_PASSWORD="your-pat-token"  # 注意: 不是密码!
KUBE_CONFIG=$(cat ~/.kube/config | base64)
KUBE_CONTEXT=$(kubectl config current-context)

# 设置 Secrets
gh secret set DOCKER_USERNAME --body "$DOCKER_USERNAME"
gh secret set DOCKER_PASSWORD --body "$DOCKER_PASSWORD"
gh secret set KUBE_CONFIG --body "$KUBE_CONFIG"
gh secret set KUBE_CONTEXT --body "$KUBE_CONTEXT"
```

**选项 C: GitHub Web UI 配置**

```
1. 访问 https://github.com/vistone/BrowerAI/settings/secrets/actions
2. 点击 "New repository secret"
3. 分别创建 4 个 secret:
   - DOCKER_USERNAME: <your-docker-id>
   - DOCKER_PASSWORD: <your-pat-token>
   - KUBE_CONFIG: <base64-encoded-kubeconfig>
   - KUBE_CONTEXT: <current-context-name>
4. 保存
```

### 验证完成

```bash
# 列出所有 Secrets (只显示名称)
gh secret list

# 预期输出:
# DOCKER_PASSWORD     Updated 2026-02-01
# DOCKER_USERNAME     Updated 2026-02-01
# KUBE_CONFIG         Updated 2026-02-01
# KUBE_CONTEXT        Updated 2026-02-01

# ✅ Step 1 完成条件:
# - 4 个 Secrets 都出现在列表中
# - 显示的创建/更新时间都在最近
```

---

## 🎯 Step 2: 测试工作流 (13:15-13:35)

### 目标
触发 GitHub Actions 工作流并监控执行过程

### 执行步骤

**A. 启动工作流测试**

```bash
# 进入仓库目录
cd /home/stone/BrowerAI

# 运行测试脚本
bash .github/scripts/test-workflow.sh

# 脚本会:
# 1. 检查 Git 状态
# 2. 验证 Secrets 配置
# 3. 验证工作流文件存在
# 4. 创建测试提交
# 5. 推送到 GitHub (触发工作流)
```

**B. 实时监控工作流**

```bash
# 在另一个终端启动实时监控
gh run watch

# 或者获取运行列表
gh run list --limit 5

# 预期输出:
# ✅ build.yml (CI - Build and Test)
# ✅ docker-build.yml (Docker - Build and Push)
# ✅ deploy.yml (Deploy - Kubernetes)
# ✅ test.yml (Test - Post-Deployment)
```

### 工作流执行时间线

```
时间    工作流                      状态        耗时
────────────────────────────────────────────────────
13:20   build.yml                  queued      -
13:21   build.yml                  in_progress ~5 min
13:26   build.yml                  completed   ✅

13:20   docker-build.yml (并行)    queued      -
13:21   docker-build.yml           in_progress ~7 min
13:28   docker-build.yml           completed   ✅

13:28   deploy.yml                 queued      - (等待 docker-build)
13:29   deploy.yml                 in_progress ~7 min
13:36   deploy.yml                 completed   ✅

13:36   test.yml                   queued      - (自动触发)
13:37   test.yml                   in_progress ~3 min
13:40   test.yml                   completed   ✅

总耗时: 13:20-13:40 约 20 分钟
```

### 查看工作流日志

```bash
# 获取最新运行 ID
RUN_ID=$(gh run list --limit 1 --json databaseId -q '.[0].databaseId')

# 查看工作流状态
gh run view $RUN_ID

# 查看详细日志
gh run view $RUN_ID --log | head -100

# 查看特定工作流的日志
gh run view $RUN_ID --log | grep -A 10 "docker-build"
```

### 问题排查

```bash
# 查看失败的工作流
gh run list --json status | grep -i failed

# 获取失败工作流的详情
gh run view <failed-run-id> --log

# 常见失败原因:
# 1. "No credentials provided" → 检查 DOCKER_PASSWORD
# 2. "Unauthorized" → 检查 KUBE_CONFIG 编码
# 3. "Connection refused" → 检查 K8s 集群状态
# 4. "Not found" → 检查镜像名称和标签

# 重新运行失败的工作流
gh run rerun <failed-run-id>
```

### ✅ Step 2 完成条件

- [ ] build.yml 完成 ✅
- [ ] docker-build.yml 完成 ✅
- [ ] deploy.yml 完成 ✅
- [ ] test.yml 完成 ✅
- [ ] 没有工作流显示失败 ❌

---

## 🎯 Step 3: 验证部署 (13:35-13:45)

### 目标
验证应用已成功部署到 Kubernetes 且可访问

### 执行步骤

**A. 自动验证 (推荐)**

```bash
# 运行完整验证脚本
bash .github/scripts/verify-deployment.sh

# 脚本会进行以下检查:
# 1. GitHub Actions 工作流状态
# 2. Kubernetes Deployment 状态
# 3. Pod 运行状态
# 4. Service 可达性
# 5. 应用健康检查
# 6. Pod 日志检查
```

**B. 手动验证**

```bash
# 1. 检查 Namespace
kubectl get ns browerai
# 预期: browerai  Active   XX

# 2. 检查 Deployment
kubectl get deployment -n browerai browerai-api-deployment
# 预期: Ready 3/3

# 3. 检查 Pod
kubectl get pods -n browerai -l app=browerai-api
# 预期: 3 个 Pod 都是 Running

# 4. 检查 Service
kubectl get svc -n browerai browerai-api-service
# 预期: Service 有 CLUSTER-IP 和 PORT

# 5. 进行端口转发
kubectl port-forward -n browerai svc/browerai-api-service 5000:5000 &
sleep 2

# 6. 测试服务健康检查
curl -v http://localhost:5000/health
# 预期: HTTP/1.1 200 OK

# 7. 运行烟雾测试
bash .github/scripts/smoke-test.sh http://localhost:5000
# 预期: ✅ 4/4 测试通过
```

### 查看应用日志

```bash
# 查看最新日志
kubectl logs -n browerai -l app=browerai-api --tail=50

# 实时跟踪日志
kubectl logs -n browerai -l app=browerai-api -f

# 查看特定 Pod 的日志
POD_NAME=$(kubectl get pods -n browerai -l app=browerai-api -o jsonpath='{.items[0].metadata.name}')
kubectl logs -n browerai $POD_NAME -f
```

### 故障排查

```bash
# Pod 未启动
kubectl describe pod <pod-name> -n browerai
# 查看 Events 部分获取错误信息

# 镜像拉取失败
kubectl get events -n browerai | grep -i "image"

# 服务无法连接
kubectl get endpoints -n browerai browerai-api-service
# 预期: 应该列出 3 个 IP 地址

# 应用崩溃
kubectl logs <pod-name> -n browerai --previous
# 查看前一个容器的日志
```

### ✅ Step 3 完成条件

- [ ] Namespace 存在
- [ ] Deployment 状态: Ready 3/3
- [ ] Pod 状态: Running 3/3
- [ ] Service 可访问
- [ ] Health check 返回 200
- [ ] 烟雾测试: 4/4 通过
- [ ] 日志无错误信息

---

## 🎯 Step 4: 配置监控 (13:45-14:05)

### 目标
设置 Prometheus + Grafana 监控栈

### 执行步骤

**选项 A: Docker 方式 (快速)**

```bash
# 1. 创建配置目录
mkdir -p /opt/prometheus
mkdir -p /opt/grafana

# 2. 启动 Prometheus
docker run -d \
  --name prometheus \
  --restart=always \
  -p 9090:9090 \
  -v /opt/prometheus:/etc/prometheus \
  prom/prometheus \
  --config.file=/etc/prometheus/prometheus.yml

# 3. 启动 Grafana
docker run -d \
  --name grafana \
  --restart=always \
  -p 3000:3000 \
  -e GF_SECURITY_ADMIN_PASSWORD=admin \
  grafana/grafana

# 4. 验证启动
sleep 5
curl -s http://localhost:9090/-/healthy | head -1
curl -s http://localhost:3000/api/health | jq .
```

**选项 B: Kubernetes 方式 (推荐)**

```bash
# 1. 添加 Prometheus Helm 仓库
helm repo add prometheus-community \
  https://prometheus-community.github.io/helm-charts
helm repo update

# 2. 创建 monitoring namespace
kubectl create namespace monitoring

# 3. 安装 Prometheus Stack
helm install prometheus \
  prometheus-community/kube-prometheus-stack \
  -n monitoring \
  --set prometheus.prometheusSpec.retention=15d \
  --set grafana.adminPassword=admin

# 4. 等待 Pod 启动
kubectl wait --for=condition=ready pod \
  -l app.kubernetes.io/name=prometheus \
  -n monitoring \
  --timeout=300s

# 5. 启动端口转发
kubectl port-forward -n monitoring svc/prometheus-operated 9090:9090 &
kubectl port-forward -n monitoring svc/grafana 3000:80 &
sleep 2
```

### 配置 Grafana 仪表板

```bash
# 1. 访问 Grafana Web UI
open http://localhost:3000
# 或: firefox http://localhost:3000

# 2. 登录凭证
# Docker: admin / admin
# Kubernetes: admin / admin (helm default)

# 3. 添加 Prometheus 数据源
# - 导航到: Configuration → Data Sources
# - 点击 "Add data source"
# - 选择 "Prometheus"
# - 设置 URL: http://prometheus:9090 (或 http://localhost:9090)
# - 点击 "Save & Test"

# 4. 导入监控仪表板
# - 导航到: Create → Import
# - 使用以下公开仪表板 ID:
#   - 3662: Prometheus (所有主要指标)
#   - 1860: Node Exporter (系统指标)
#   - 6417: Kubernetes Cluster (K8s 指标)
# - 选择 Prometheus 作为数据源
# - 导入

# 5. 自定义仪表板
# - Create → Dashboard
# - Add Panel
# - 编写 Prometheus 查询:
#   - RPS: sum(rate(http_requests_total[5m]))
#   - 错误率: sum(rate(http_requests_total{status=~"5.."}[5m])) / sum(rate(http_requests_total[5m])) * 100
#   - 延迟 (99%): histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[5m]))
```

### 配置告警规则

```bash
# 1. 创建告警规则文件 (alert-rules.yaml)
mkdir -p /opt/prometheus/rules

cat > /opt/prometheus/rules/browerai.rules.yaml << 'EOF'
groups:
  - name: browerai
    interval: 30s
    rules:
      - alert: HighErrorRate
        expr: sum(rate(http_requests_total{status=~"5.."}[5m])) / sum(rate(http_requests_total[5m])) > 0.05
        for: 5m
        annotations:
          summary: "高错误率检测"
          description: "过去 5 分钟的错误率超过 5%"
      
      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 1
        for: 5m
        annotations:
          summary: "高延迟检测"
          description: "95% 请求的延迟超过 1 秒"
      
      - alert: PodRestartingTooOften
        expr: rate(kube_pod_container_status_restarts_total[1h]) > 0
        for: 5m
        annotations:
          summary: "Pod 重启过于频繁"
          description: "Pod {{ $labels.pod }} 在过去 1 小时内重启"
EOF

# 2. Docker 方式: 更新 Prometheus 配置
# 在 /opt/prometheus/prometheus.yml 中添加:
# global:
#   rule_files:
#     - '/etc/prometheus/rules/*.yaml'

# 3. Kubernetes 方式: 使用 PrometheusRule CRD
kubectl apply -f - << 'EOF'
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: browerai-rules
  namespace: monitoring
spec:
  groups:
  - name: browerai
    interval: 30s
    rules:
    - alert: HighErrorRate
      expr: sum(rate(http_requests_total{status=~"5.."}[5m])) / sum(rate(http_requests_total[5m])) > 0.05
      for: 5m
      annotations:
        summary: "高错误率检测"
EOF
```

### 验证监控

```bash
# 1. 访问 Prometheus
curl http://localhost:9090/-/healthy
# 预期: 返回 200

# 2. 查看 Prometheus 目标
curl http://localhost:9090/api/v1/targets | jq '.data.activeTargets | length'
# 预期: 返回 > 0

# 3. 访问 Grafana
curl http://localhost:3000/api/health | jq .
# 预期: 返回 health 状态

# 4. 测试 Prometheus 查询
curl 'http://localhost:9090/api/v1/query?query=up'
# 预期: 返回 JSON 格式的指标数据

# 5. 验证应用指标导出 (如果暴露)
curl http://localhost:5000/metrics
# 预期: 返回 Prometheus 格式的指标
```

### ✅ Step 4 完成条件

- [ ] Prometheus 运行正常 (HTTP 200)
- [ ] Grafana 登录成功
- [ ] Prometheus 数据源已连接
- [ ] 至少导入 1 个仪表板
- [ ] 指标数据正常显示
- [ ] 告警规则已配置

---

## 🎯 Step 5: 整体测试验证 (14:05-14:15)

### 执行步骤

```bash
# 1. 最终部署验证
bash .github/scripts/verify-deployment.sh

# 2. 运行烟雾测试
bash .github/scripts/smoke-test.sh http://localhost:5000

# 3. 检查所有工作流都成功
gh run list --limit 1 --json status,conclusion

# 4. 验证监控数据
# 访问 Grafana: http://localhost:3000
# 检查仪表板是否显示数据

# 5. 整体 checklist
bash .github/scripts/deployment-checklist.sh
```

### ✅ 最终验证条件

- [ ] 所有 4 个 GitHub Actions 工作流都通过 ✅
- [ ] Kubernetes 部署状态良好 (Ready 3/3)
- [ ] 服务可访问且 HTTP 200
- [ ] 烟雾测试 4/4 通过
- [ ] Prometheus 收集指标
- [ ] Grafana 显示数据
- [ ] 没有错误或警告

---

## 📊 关键指标

### 工作流性能

| 指标 | 目标 | 实际 |
|------|------|------|
| 总部署时间 | < 25 分钟 | ~20 分钟 |
| build.yml | < 10 分钟 | ~5 分钟 |
| docker-build.yml | < 10 分钟 | ~7 分钟 |
| deploy.yml | < 10 分钟 | ~7 分钟 |
| test.yml | < 5 分钟 | ~3 分钟 |
| 工作流成功率 | 100% | - |

### 应用性能

| 指标 | 目标 | 验证方法 |
|------|------|---------|
| RPS | > 140 | 烟雾测试 |
| 错误率 | < 1% | 烟雾测试 + 日志 |
| 延迟 (p95) | < 50ms | Prometheus |
| Pod 可用性 | 3/3 | kubectl get pods |
| Service 连接 | 100% | curl health check |

---

## 🔄 故障恢复流程

### 如果工作流失败

```bash
# 1. 查看失败日志
gh run view <run-id> --log | tail -100

# 2. 检查常见原因
# - Docker 凭证: gh secret list | grep DOCKER
# - K8s 配置: kubectl config view
# - 网络连接: kubectl get nodes

# 3. 修复问题后重新运行
gh run rerun <run-id>
```

### 如果部署失败

```bash
# 1. 查看 Pod 日志
kubectl logs <pod-name> -n browerai

# 2. 检查事件
kubectl get events -n browerai

# 3. 使用回滚脚本
bash .github/scripts/rollback.sh

# 4. 重新部署
git add .
git commit -m "Fix deployment"
git push  # 触发新的工作流
```

### 如果监控无数据

```bash
# 1. 检查 Prometheus 连接
curl http://localhost:9090/api/v1/targets

# 2. 检查应用指标导出
curl http://localhost:5000/metrics

# 3. 重启 Prometheus
docker restart prometheus
# 或 kubectl rollout restart -n monitoring deployment/prometheus-operated
```

---

## 📞 支持资源

- 📖 详细步骤: [.github/IMPLEMENTATION_STEPS.md](.github/IMPLEMENTATION_STEPS.md)
- 📋 快速参考: [.github/QUICK_REFERENCE.md](.github/QUICK_REFERENCE.md)
- 📊 监控指南: [.github/MONITORING_GUIDE.md](.github/MONITORING_GUIDE.md)
- ⚙️ CI/CD 配置: [.github/CICD_CONFIG.md](.github/CICD_CONFIG.md)
- 🔍 检查工具: `bash .github/scripts/deployment-checklist.sh`

---

**版本**: 2.0.0  
**最后更新**: 2026-02-01  
**状态**: ✅ 准备执行

🎯 **现在就开始!** 按照时间表执行四个步骤，预计 75 分钟内完成完整部署。
