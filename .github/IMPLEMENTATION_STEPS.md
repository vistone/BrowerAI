# Week 8 Phase E+ 实施步骤指南

**文档版本**: 1.0.0  
**创建日期**: 2026-02-01  
**目的**: 指导完整的 GitHub Secrets 配置、工作流测试和部署验证

---

## 🎯 四个关键步骤

### Step 1️⃣: 配置 GitHub Secrets (15-20 分钟)

#### 前置条件
- [ ] Docker Hub 账户 ([注册](https://hub.docker.com/))
- [ ] Kubernetes 集群 (Minikube 或云平台)
- [ ] GitHub CLI 或 Web 访问权限

#### 执行步骤

**A. 获取 Docker Hub 凭证**

```bash
# 1. 访问 Docker Hub
# https://hub.docker.com/settings/security

# 2. 创建 Personal Access Token (PAT)
# - 点击 "New Access Token"
# - 设置名称: "GitHub Actions"
# - 选择权限: "Read & Write"
# - 生成并复制 token

DOCKER_USERNAME="your-docker-username"
DOCKER_PASSWORD="your-pat-token"
```

**B. 获取 Kubernetes 配置**

```bash
# 1. 对于 Minikube
export KUBE_CONFIG=$(cat ~/.kube/config | base64)
export KUBE_CONTEXT=$(kubectl config current-context)

# 验证
echo $KUBE_CONFIG    # 应输出长的 base64 字符串
echo $KUBE_CONTEXT   # 应输出 "minikube"

# 2. 对于云平台 (EKS/GKE/AKS)
# 先从云平台获取 kubeconfig
# 然后执行同样的 base64 编码
```

**C. 配置 GitHub Secrets**

```bash
# 使用 GitHub CLI (推荐)
bash .github/scripts/setup-secrets.sh

# 或手动配置
gh secret set DOCKER_USERNAME --body "$DOCKER_USERNAME"
gh secret set DOCKER_PASSWORD --body "$DOCKER_PASSWORD"
gh secret set KUBE_CONFIG --body "$KUBE_CONFIG"
gh secret set KUBE_CONTEXT --body "$KUBE_CONTEXT"

# 验证
gh secret list
```

**D. 验证配置**

```bash
# 检查所有 secrets 已配置
gh secret list | grep -E "DOCKER_USERNAME|DOCKER_PASSWORD|KUBE_CONFIG|KUBE_CONTEXT"

# 预期输出:
# DOCKER_PASSWORD     Updated 2026-02-01
# DOCKER_USERNAME     Updated 2026-02-01
# KUBE_CONFIG         Updated 2026-02-01
# KUBE_CONTEXT        Updated 2026-02-01
```

---

### Step 2️⃣: 测试工作流 (5-10 分钟)

#### 执行步骤

**A. 启动工作流测试**

```bash
# 1. 确保在 Git 仓库根目录
cd /home/stone/BrowerAI

# 2. 运行测试脚本
bash .github/scripts/test-workflow.sh

# 3. 脚本会:
#    - 检查 GitHub Secrets 配置
#    - 检查工作流文件存在
#    - 创建测试提交
#    - 推送到 GitHub
```

**B. 监控工作流运行**

```bash
# 实时监控最新运行
gh run watch

# 列出所有运行
gh run list

# 查看特定运行的日志
gh run view <run-id> --log

# 查看最新运行的日志
gh run view --log
```

**C. 预期的工作流执行顺序**

```
时间     工作流名称                  状态       耗时
────────────────────────────────────────────────────
13:00   CI - Build and Test        ⏳ 运行中   ~5 min
14:00   Docker - Build and Push    ⏳ 运行中   ~7 min (并行)
15:00   Deploy - Kubernetes        ⏳ 运行中   ~7 min
15:15   Test - Post-Deployment     ⏳ 运行中   ~3 min
        
完成时间: ~15-20 分钟
总耗时:   13:00-15:20
```

**D. 监控命令**

```bash
# GitHub CLI 方式
gh run watch                    # 实时监控
gh run list --limit 10          # 最近 10 个运行
gh run view <id> --json status  # 查看状态

# Web UI 方式
# https://github.com/vistone/BrowerAI/actions
```

---

### Step 3️⃣: 验证自动部署 (10-15 分钟)

#### 前置条件
- [ ] Step 2 工作流全部成功 (✅ 绿色)
- [ ] K8s 集群正在运行
- [ ] kubectl 已配置

#### 执行步骤

**A. 检查部署状态**

```bash
# 运行验证脚本
bash .github/scripts/verify-deployment.sh

# 脚本会检查:
# ✅ GitHub Actions 工作流状态
# ✅ Kubernetes Deployment
# ✅ Pod 状态
# ✅ Service 可达性
# ✅ Pod 日志
```

**B. 手动验证**

```bash
# 1. 检查 Namespace
kubectl get ns browerai

# 2. 检查 Deployment
kubectl get deployment -n browerai browerai-api-deployment
kubectl describe deployment browerai-api-deployment -n browerai

# 3. 检查 Pod 状态
kubectl get pods -n browerai -l app=browerai-api
kubectl describe pod <pod-name> -n browerai

# 4. 查看日志
kubectl logs -n browerai -l app=browerai-api --tail=50 -f

# 5. 测试服务
# 启用端口转发
kubectl port-forward -n browerai svc/browerai-api-service 5000:5000

# 在另一个终端测试
curl http://localhost:5000/health
```

**C. 运行烟雾测试**

```bash
# 启动端口转发 (如上)
# 然后运行:
bash .github/scripts/smoke-test.sh http://localhost:5000

# 预期输出:
# ✅ 4 项 API 测试全部通过
# ✅ 响应时间正常 (< 50ms)
# ✅ Service 状态: 🟢 Healthy
```

**D. 常见问题排查**

| 问题 | 排查步骤 |
|------|---------|
| Pod 未启动 | `kubectl describe pod <name> -n browerai` |
| 镜像拉取失败 | `kubectl get events -n browerai` |
| Service 不可达 | `kubectl get svc -n browerai` |
| 日志中有错误 | `kubectl logs <pod> -n browerai` |
| K8s 连接失败 | `kubectl config view` |

---

### Step 4️⃣: 配置监控 (20-30 分钟)

#### 前置条件
- [ ] Step 3 部署验证通过
- [ ] Docker 已安装 (可选，使用 Helm 则不需)

#### 执行步骤

**A. 快速启动 Prometheus (Docker)**

```bash
# 1. 创建配置目录
mkdir -p /opt/prometheus

# 2. 创建 Prometheus 配置 (参考 MONITORING_GUIDE.md)
# 可从 .github/MONITORING_GUIDE.md 复制

# 3. 启动 Prometheus
docker run -d \
  --name prometheus \
  -p 9090:9090 \
  -v /opt/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml \
  prom/prometheus \
  --config.file=/etc/prometheus/prometheus.yml

# 4. 验证
curl http://localhost:9090/-/healthy
```

**B. 启动 Grafana (Docker)**

```bash
# 1. 启动 Grafana
docker run -d \
  --name grafana \
  -p 3000:3000 \
  -e GF_SECURITY_ADMIN_PASSWORD=admin \
  grafana/grafana

# 2. 访问 Web UI
# http://localhost:3000
# 用户: admin
# 密码: admin

# 3. 添加 Prometheus 数据源
# Configuration → Data Sources → Add
# - Type: Prometheus
# - URL: http://host.docker.internal:9090 (Mac/Windows)
#        或 http://localhost:9090 (Linux)
```

**C. 创建监控仪表盘**

```
1. 登录 Grafana (http://localhost:3000)
2. Create → Dashboard → Add panel
3. 选择 Prometheus 作为数据源
4. 输入查询:
   - RPS: sum(rate(http_requests_total[5m]))
   - 错误率: (sum(rate(http_requests_total{status=~"5.."}[5m])) / sum(rate(http_requests_total[5m]))) * 100
   - 延迟: histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[5m]))
5. 保存仪表盘
```

**D. Kubernetes 方式 (Helm)**

```bash
# 1. 添加 Helm 仓库
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# 2. 安装 Prometheus Stack
helm install prometheus prometheus-community/kube-prometheus-stack \
  -n monitoring \
  --create-namespace

# 3. 等待 Pod 就绪
kubectl wait --for=condition=ready pod \
  -l app.kubernetes.io/name=prometheus \
  -n monitoring \
  --timeout=300s

# 4. 端口转发
kubectl port-forward -n monitoring svc/prometheus-operated 9090:9090
kubectl port-forward -n monitoring svc/grafana 3000:80

# 5. 访问
# Prometheus: http://localhost:9090
# Grafana: http://localhost:3000 (admin:prom-operator)
```

**E. 验证监控**

```bash
# 1. 检查指标收集
curl http://localhost:9090/api/v1/query?query=up | jq .

# 2. Grafana 查询测试
# 在 Explore 中测试查询:
# - http_requests_total
# - http_request_duration_seconds_bucket
# - container_memory_usage_bytes

# 3. 配置告警规则
# 参考 .github/MONITORING_GUIDE.md 中的告警规则
```

---

## 📋 完整检查清单

### ✅ Step 1: GitHub Secrets

- [ ] DOCKER_USERNAME 已设置
- [ ] DOCKER_PASSWORD 已设置 (PAT，非密码)
- [ ] KUBE_CONFIG 已设置 (base64 编码)
- [ ] KUBE_CONTEXT 已设置
- [ ] 运行 `gh secret list` 验证全部显示

### ✅ Step 2: 工作流测试

- [ ] build.yml 运行成功 (✅ Lint + 测试)
- [ ] docker-build.yml 运行成功 (✅ 镜像推送)
- [ ] deploy.yml 运行成功 (✅ K8s 部署)
- [ ] test.yml 运行成功 (✅ 烟雾测试)
- [ ] 总耗时 15-20 分钟
- [ ] 访问 GitHub Actions 页面验证

### ✅ Step 3: 部署验证

- [ ] Namespace browerai 存在
- [ ] Deployment 状态: Ready 3/3
- [ ] Pod 状态: Running 3/3
- [ ] Service 可访问
- [ ] 健康检查返回 200
- [ ] 烟雾测试全部通过 (4/4)
- [ ] 日志无错误信息

### ✅ Step 4: 监控配置

- [ ] Prometheus 运行正常 (http://localhost:9090)
- [ ] Grafana 运行正常 (http://localhost:3000)
- [ ] Prometheus 数据源已添加
- [ ] 至少创建 1 个监控仪表盘
- [ ] 指标数据正常显示
- [ ] 告警规则已配置

---

## 🚀 快速命令参考

```bash
# === Step 1: 配置 Secrets ===
bash .github/scripts/setup-secrets.sh
gh secret list

# === Step 2: 测试工作流 ===
bash .github/scripts/test-workflow.sh
gh run watch
gh run view --log

# === Step 3: 验证部署 ===
bash .github/scripts/verify-deployment.sh
kubectl port-forward -n browerai svc/browerai-api-service 5000:5000
bash .github/scripts/smoke-test.sh http://localhost:5000

# === Step 4: 配置监控 ===
# Docker 方式
docker run -d --name prometheus -p 9090:9090 prom/prometheus
docker run -d --name grafana -p 3000:3000 grafana/grafana

# Kubernetes 方式
helm install prometheus prometheus-community/kube-prometheus-stack -n monitoring --create-namespace

# === 监控访问 ===
curl http://localhost:9090/-/healthy         # Prometheus
curl http://localhost:3000/api/health        # Grafana
```

---

## 📞 故障排查

### 工作流失败

```bash
# 1. 查看详细日志
gh run view <run-id> --log

# 2. 常见原因
# - Docker credentials 错误: 检查 DOCKER_PASSWORD 是否是 PAT
# - K8s 认证失败: 检查 KUBE_CONFIG 是否正确编码
# - 镜像不存在: 等待 docker-build.yml 完成

# 3. 重新运行
gh run rerun <run-id>
```

### 部署失败

```bash
# 1. 查看 Pod 描述
kubectl describe pod <pod-name> -n browerai

# 2. 查看 Pod 日志
kubectl logs <pod-name> -n browerai

# 3. 查看事件
kubectl get events -n browerai

# 4. 检查镜像
docker pull <image-name>:latest

# 5. 手动部署
kubectl apply -f k8s/deployment.yaml
```

### 监控无数据

```bash
# 1. 检查 Prometheus 指标
curl http://localhost:9090/api/v1/targets

# 2. 检查应用是否导出指标
curl http://localhost:5000/metrics

# 3. 重启 Prometheus
docker restart prometheus

# 4. 检查 Grafana 数据源
# Settings → Data Sources → Test
```

---

## 📊 预期结果

### 完成后的状态

```
✅ GitHub Actions 工作流
   - 每次推送代码自动构建和部署
   - 所有 4 个工作流都在运行
   - 平均耗时 15-20 分钟

✅ Kubernetes 部署
   - 3 个 Pod 正在运行
   - Service 可访问
   - HPA 已配置 (2-10 Pod)

✅ 监控告警
   - Prometheus 收集指标
   - Grafana 展示仪表盘
   - 告警规则已激活
   - 可实时监控应用性能

✅ 自动化流程
   - 代码 → 构建 → 镜像 → 部署 → 验证 (全自动)
   - 失败自动通知
   - 支持一键回滚
```

---

**文档版本**: 1.0.0  
**创建日期**: 2026-02-01  
**状态**: ✅ 准备执行

---

## 下一步

完成上述四个步骤后:

1. ✅ 系统完全生产就绪
2. ✅ 所有 CI/CD 流程自动化
3. ✅ 监控告警完整配置
4. ✅ 可支持高并发部署

**建议**:
- 保留这份指南供日后参考
- 定期检查监控仪表盘
- 每周查看工作流日志
- 记录常见问题和解决方案
