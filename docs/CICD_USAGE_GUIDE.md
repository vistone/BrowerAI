# 🚀 BrowerAI CI/CD Pipeline 使用指南

## 📋 概览

本项目已配置完整的 CI/CD 流程，包括：
- ✅ 自动化构建和测试
- ✅ Docker 镜像构建和推送
- ✅ Kubernetes 自动部署
- ✅ 回滚机制
- ✅ 安全扫描

## 🔧 配置的 Workflows

### 1. `complete-cicd.yml` - 完整 CI/CD 流程
**触发条件:**
- Push 到 `main` 或 `week5-postgresql-persistence` 分支
- 创建 tag (如 `v1.0.0`)
- 手动触发

**流程步骤:**
```
1. build-and-test → Rust 代码构建和测试
2. python-check → Python 环境检查
3. build-docker → Docker 镜像构建
4. security-scan → 安全漏洞扫描
5. push-docker → 推送镜像到 Docker Hub
6. deploy-k8s → 部署到 Kubernetes
7. post-deploy-test → 部署后测试
8. create-release → 创建 GitHub Release (仅 tag)
9. notify → 发送通知
```

### 2. `rollback-deployment.yml` - 回滚部署
**触发条件:** 手动触发

**使用方法:**
1. 前往 GitHub Actions 页面
2. 选择 "Rollback Deployment" workflow
3. 点击 "Run workflow"
4. 填写参数:
   - Environment: staging/production
   - Revision: 回滚版本数 (0=上一个版本)
   - Reason: 回滚原因

## 🔑 必需的 GitHub Secrets

在 GitHub 仓库设置中添加以下 Secrets:

### Docker Hub
```
DOCKER_USERNAME = your-dockerhub-username
DOCKER_PASSWORD = your-dockerhub-token
```

### Kubernetes (可选)
```
KUBE_CONFIG = base64-encoded-kubeconfig
KUBE_CONTEXT = kubernetes-context-name
```

### API Endpoint (可选)
```
API_ENDPOINT = https://your-api-endpoint.com
```

## 📝 使用示例

### 场景 1: 开发分支推送
```bash
git checkout week5-postgresql-persistence
git add .
git commit -m "feat: add new feature"
git push origin week5-postgresql-persistence
```
**结果:** 触发 CI/CD，执行到 security-scan，不会自动部署

### 场景 2: 合并到主分支
```bash
git checkout main
git merge week5-postgresql-persistence
git push origin main
```
**结果:** 完整 CI/CD 流程，包括部署到 staging 环境

### 场景 3: 发布新版本
```bash
git checkout main
git tag v1.0.0
git push origin v1.0.0
```
**结果:** 完整 CI/CD + GitHub Release 创建

### 场景 4: 手动部署特定环境
1. 前往 Actions 页面
2. 选择 "Complete CI/CD Pipeline"
3. 点击 "Run workflow"
4. 选择分支和部署环境
5. 点击 "Run workflow"

### 场景 5: 紧急回滚
1. 前往 Actions 页面
2. 选择 "Rollback Deployment"
3. 点击 "Run workflow"
4. 填写:
   - Environment: production
   - Revision: 0 (回滚到上一个版本)
   - Reason: "Critical bug in production"
5. 点击 "Run workflow"

## 🐳 本地 Docker 测试

### 构建镜像
```bash
docker build -f Dockerfile.api -t browerai-api:local .
```

### 运行容器
```bash
docker run -p 3000:3000 browerai-api:local
```

### 测试健康检查
```bash
curl http://localhost:3000/api/health
```

## ☸️ 本地 Kubernetes 测试 (Minikube)

### 启动 Minikube
```bash
minikube start
```

### 应用配置
```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/browerai-api.yaml
```

### 查看状态
```bash
kubectl get pods -n browerai
kubectl get svc -n browerai
```

### 访问服务
```bash
minikube service browerai-api-service -n browerai
```

## 📊 监控和日志

### 查看 Workflow 运行状态
```
https://github.com/vistone/BrowerAI/actions
```

### 查看部署日志
```bash
kubectl logs -f deployment/browerai-api-deployment -n browerai
```

### 查看部署历史
```bash
kubectl rollout history deployment/browerai-api-deployment -n browerai
```

## 🔍 故障排查

### CI/CD 失败
1. 检查 Actions 日志
2. 查看具体失败的 job
3. 检查错误信息
4. 修复后重新推送

### 部署失败
```bash
# 查看 pod 状态
kubectl get pods -n browerai

# 查看 pod 日志
kubectl logs <pod-name> -n browerai

# 查看事件
kubectl get events -n browerai --sort-by='.lastTimestamp'
```

### Docker 镜像问题
```bash
# 拉取镜像测试
docker pull <your-username>/browerai-api:latest

# 本地运行测试
docker run -p 3000:3000 <your-username>/browerai-api:latest
```

## 🎯 最佳实践

### 1. 分支策略
- `main`: 生产环境代码
- `week5-postgresql-persistence`: 开发分支
- feature branches: 功能开发

### 2. 提交信息格式
```
feat: add new feature
fix: fix bug
docs: update documentation
test: add tests
refactor: refactor code
```

### 3. 版本标签
- `v1.0.0`: 主版本发布
- `v1.0.1`: 补丁版本
- `v1.1.0`: 次版本发布

### 4. 部署流程
1. 开发分支测试
2. 合并到 main
3. 自动部署到 staging
4. 测试验证
5. 创建 tag 发布
6. 自动部署到 production (如果配置)

## 📞 支持

遇到问题?
1. 查看 Actions 日志
2. 查看本文档的故障排查部分
3. 提交 Issue 到 GitHub

---

**Last Updated:** 2026-02-02  
**Version:** Week 8 Phase E
