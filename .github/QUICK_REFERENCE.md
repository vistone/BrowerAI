# Week 8 完整部署参考卡片

## 🎯 四大步骤快速参考

### Step 1️⃣: 配置 GitHub Secrets (15 分钟)

```bash
# 准备凭证
export DOCKER_USERNAME="your-docker-id"
export DOCKER_PASSWORD="your-pat-token"  # 非密码!
export KUBE_CONFIG="$(cat ~/.kube/config | base64)"
export KUBE_CONTEXT="$(kubectl config current-context)"

# 方式 A: 交互式配置
bash .github/scripts/setup-secrets.sh

# 方式 B: CLI 配置
gh secret set DOCKER_USERNAME --body "$DOCKER_USERNAME"
gh secret set DOCKER_PASSWORD --body "$DOCKER_PASSWORD"
gh secret set KUBE_CONFIG --body "$KUBE_CONFIG"
gh secret set KUBE_CONTEXT --body "$KUBE_CONTEXT"

# 验证
gh secret list | grep -E "DOCKER|KUBE"
```

✅ **成功指标**: 输出显示 4 个 secret

---

### Step 2️⃣: 测试工作流 (5 分钟)

```bash
# 启动工作流测试
bash .github/scripts/test-workflow.sh

# 监控工作流进度
gh run watch              # 实时看板
gh run list --limit 5     # 历史列表
gh run view --log         # 查看日志
```

✅ **预期结果**:
- build.yml ✅ (5 分钟)
- docker-build.yml ✅ (7 分钟)
- deploy.yml ✅ (7 分钟)
- test.yml ✅ (3 分钟)
- **总耗时: 15-20 分钟**

---

### Step 3️⃣: 验证部署 (10 分钟)

```bash
# 自动验证
bash .github/scripts/verify-deployment.sh

# 或手动验证
# 1. 检查部署
kubectl get deployment -n browerai
kubectl get pods -n browerai -l app=browerai-api

# 2. 端口转发
kubectl port-forward -n browerai svc/browerai-api-service 5000:5000

# 3. 测试服务
curl http://localhost:5000/health

# 4. 烟雾测试
bash .github/scripts/smoke-test.sh http://localhost:5000

# 5. 查看日志
kubectl logs -n browerai -l app=browerai-api --tail=50 -f
```

✅ **成功指标**:
- Pod Status: Running 3/3
- HTTP Health: 200 OK
- Smoke Tests: 4/4 通过

---

### Step 4️⃣: 配置监控 (20 分钟)

#### Docker 启动 (简单)

```bash
# Prometheus
docker run -d --name prometheus -p 9090:9090 prom/prometheus

# Grafana
docker run -d --name grafana -p 3000:3000 \
  -e GF_SECURITY_ADMIN_PASSWORD=admin \
  grafana/grafana

# 访问
# Prometheus: http://localhost:9090
# Grafana: http://localhost:3000 (admin:admin)
```

#### Kubernetes 启动 (推荐)

```bash
# 添加仓库
helm repo add prometheus-community \
  https://prometheus-community.github.io/helm-charts
helm repo update

# 安装
helm install prometheus \
  prometheus-community/kube-prometheus-stack \
  -n monitoring --create-namespace

# 端口转发
kubectl port-forward -n monitoring svc/prometheus-operated 9090:9090
kubectl port-forward -n monitoring svc/grafana 3000:80

# 访问
# Prometheus: http://localhost:9090
# Grafana: http://localhost:3000 (admin:prom-operator)
```

#### 配置 Grafana 数据源和仪表板

```
1. 登录 Grafana (http://localhost:3000)
2. Configuration → Data Sources
3. 添加 Prometheus:
   - URL: http://localhost:9090
4. 测试并保存
5. Create → Dashboard → Import
6. 导入 ID: 3662, 1860, 6417 (Prometheus 仪表板)
```

✅ **成功指标**:
- Prometheus 指标可查询
- Grafana 看板可访问
- 应用指标显示正常

---

## 📊 实时监控命令

### GitHub Actions

```bash
# 实时监控
gh run watch

# 列出最近运行
gh run list --limit 10

# 查看特定运行的详情
gh run view <run-id>

# 查看工作流日志
gh run view <run-id> --log

# 重新运行
gh run rerun <run-id>
```

### Kubernetes 部署

```bash
# 实时监控 Pod
watch kubectl get pods -n browerai

# Pod 详细信息
kubectl describe pod <pod-name> -n browerai

# Pod 日志
kubectl logs <pod-name> -n browerai -f

# 所有事件
kubectl get events -n browerai

# 部署滚动状态
kubectl rollout status deployment/browerai-api-deployment -n browerai
```

### 监控指标

```bash
# Prometheus 查询
curl 'http://localhost:9090/api/v1/query?query=http_requests_total'

# 应用指标端点 (如果暴露)
curl http://localhost:5000/metrics

# 检查监控目标
curl http://localhost:9090/api/v1/targets
```

---

## 🔍 常见问题排查

### GitHub Secrets 问题

```bash
# 问题: Secret 未通过 CI
# 解决:
gh secret list  # 确认存在
gh secret set KEY --body "value"  # 重新设置

# 问题: Docker 认证失败
# 原因: DOCKER_PASSWORD 是密码而非 PAT
# 解决:
# 1. 在 Docker Hub 创建 PAT
# 2. 使用 PAT 而非密码
gh secret set DOCKER_PASSWORD --body "your-pat-token"
```

### 工作流失败

```bash
# 查看详细错误日志
gh run view <run-id> --log

# 常见原因:
# 1. Docker 凭证错误 → 检查 DOCKER_PASSWORD
# 2. K8s 认证失败 → 检查 KUBE_CONFIG 是否正确编码
# 3. 镜像拉取失败 → 等待 docker-build.yml 完成
# 4. 权限不足 → 检查 RBAC 配置

# 重新运行
gh run rerun <run-id>
```

### 部署失败

```bash
# 查看 Pod 日志
kubectl logs <pod-name> -n browerai

# 查看 Pod 描述
kubectl describe pod <pod-name> -n browerai

# 查看事件
kubectl get events -n browerai --sort-by='.lastTimestamp'

# 检查 Deployment
kubectl describe deployment browerai-api-deployment -n browerai

# 手动应用配置
kubectl apply -f k8s/deployment.yaml
```

### 服务不可达

```bash
# 检查 Service
kubectl get svc -n browerai
kubectl describe svc browerai-api-service -n browerai

# 检查端口转发
kubectl port-forward -n browerai svc/browerai-api-service 5000:5000

# 测试连接
curl -v http://localhost:5000/health

# 查看 Endpoints
kubectl get endpoints -n browerai
```

### 监控无数据

```bash
# Prometheus 健康检查
curl http://localhost:9090/-/healthy

# 查看 Prometheus 目标
curl http://localhost:9090/api/v1/targets

# Grafana 健康检查
curl http://localhost:3000/api/health

# 检查应用是否导出指标
curl http://localhost:5000/metrics

# 重启 Prometheus
docker restart prometheus
# 或
kubectl rollout restart deployment/prometheus-operated -n monitoring
```

---

## 📋 检查清单

### 部署前

- [ ] GitHub Secrets 已配置 (4/4)
- [ ] 工作流文件存在 (5/5)
- [ ] K8s 集群可访问
- [ ] Docker Hub 账户有效
- [ ] 本地 kubeconfig 正确

### 部署中

- [ ] 工作流 build.yml 运行
- [ ] 工作流 docker-build.yml 运行
- [ ] 工作流 deploy.yml 运行
- [ ] 工作流 test.yml 运行
- [ ] 没有工作流失败

### 部署后

- [ ] Pod Status: Running 3/3
- [ ] Service 可访问
- [ ] Health check: 200 OK
- [ ] 烟雾测试: 4/4 通过
- [ ] 日志无错误

### 监控

- [ ] Prometheus 运行
- [ ] Grafana 运行
- [ ] 指标可查询
- [ ] 仪表板显示数据
- [ ] 告警规则有效

---

## ⚡ 一键命令组合

### 完整部署流程

```bash
# 1. 配置 Secrets
bash .github/scripts/setup-secrets.sh

# 2. 测试工作流
bash .github/scripts/test-workflow.sh
gh run watch  # 等待完成 (15-20 min)

# 3. 验证部署
bash .github/scripts/verify-deployment.sh

# 4. 启动监控
docker run -d --name prometheus -p 9090:9090 prom/prometheus
docker run -d --name grafana -p 3000:3000 -e GF_SECURITY_ADMIN_PASSWORD=admin grafana/grafana

# 完成!
echo "🎉 系统已部署并监控就绪"
```

### 健康检查脚本

```bash
#!/bin/bash
echo "=== GitHub Actions ==="
gh run list --limit 1 --json status

echo "=== Kubernetes ==="
kubectl get pods -n browerai

echo "=== Service ==="
kubectl get svc -n browerai

echo "=== Monitoring ==="
curl -s http://localhost:9090/-/healthy | head -1
curl -s http://localhost:3000/api/health | jq .
```

---

## 📞 获取帮助

### 自动检查工具

```bash
# 交互式检查清单
bash .github/scripts/deployment-checklist.sh
```

### 文档参考

- 详细步骤: `.github/IMPLEMENTATION_STEPS.md`
- 监控指南: `.github/MONITORING_GUIDE.md`
- CI/CD 配置: `.github/CICD_CONFIG.md`
- Phase E 计划: `WEEK8_PHASE_E_PLAN.md`

### 常用命令速查

| 任务 | 命令 |
|------|------|
| 查看工作流 | `gh run list` |
| 监控工作流 | `gh run watch` |
| 看工作流日志 | `gh run view --log` |
| 检查 Pod | `kubectl get pods -n browerai` |
| 查看日志 | `kubectl logs <pod> -n browerai -f` |
| 进行端口转发 | `kubectl port-forward -n browerai svc/browerai-api-service 5000:5000` |
| 烟雾测试 | `bash .github/scripts/smoke-test.sh http://localhost:5000` |
| Prometheus | `curl http://localhost:9090` |
| Grafana | `curl http://localhost:3000` |

---

**创建日期**: 2026-02-01  
**状态**: ✅ 生产就绪  
**版本**: 1.0.0

💡 **提示**: 将此文件加入书签以便快速参考!
