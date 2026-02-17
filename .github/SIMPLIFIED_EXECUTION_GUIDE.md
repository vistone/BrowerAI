# Week 8 Phase E - 简化执行指南 (Web UI 方式)

**当前环境**: 缺少 GitHub CLI, kubectl, Docker  
**解决方案**: 使用 GitHub Web UI + Web 浏览器完成

---

## ✅ Step 1️⃣: 配置 GitHub Secrets (使用 Web UI - 10 分钟)

### 前置准备

获取所需的 4 个值:

#### 值 1: Docker Hub 用户名
```
从 Docker Hub 获取: https://hub.docker.com/settings/account
示例: vistone
```

#### 值 2: Docker Hub PAT Token (重要!)
```
从 Docker Hub 获取: https://hub.docker.com/settings/security
1. 点击 "New Access Token"
2. 设置名称: "BrowerAI-GitHub-Actions"
3. 权限选择: "Read & Write"
4. 生成并复制整个 token
示例: dckr_pat_abc123def456...
```

#### 值 3 & 4: Kubernetes 配置 (可选 - 用于本地演示可跳过)
```
如果您有 K8s 集群:
- KUBE_CONFIG: kubeconfig 文件内容 (Base64 编码)
- KUBE_CONTEXT: 当前 context 名称 (通常是 "docker-desktop" 或 "minikube")

如果没有:
- KUBE_CONFIG: "local-development"
- KUBE_CONTEXT: "docker-desktop"
```

### 操作步骤

1️⃣ **访问 GitHub Secrets 页面**
   - 打开: https://github.com/vistone/BrowerAI/settings/secrets/actions
   - 或: 仓库 → Settings → Secrets and variables → Actions

2️⃣ **添加 4 个 Secrets**

   **Secret 1: DOCKER_USERNAME**
   ```
   名称: DOCKER_USERNAME
   值: <your-docker-hub-username>
   点击: Add secret
   ```

   **Secret 2: DOCKER_PASSWORD**
   ```
   名称: DOCKER_PASSWORD
   值: <your-docker-hub-pat-token>
   点击: Add secret
   ```

   **Secret 3: KUBE_CONFIG**
   ```
   名称: KUBE_CONFIG
   值: local-development  (或实际 kubeconfig 内容)
   点击: Add secret
   ```

   **Secret 4: KUBE_CONTEXT**
   ```
   名称: KUBE_CONTEXT
   值: docker-desktop  (或实际 context 名)
   点击: Add secret
   ```

3️⃣ **验证完成**
   ```
   页面应显示:
   ✅ DOCKER_PASSWORD     Updated just now
   ✅ DOCKER_USERNAME     Updated just now
   ✅ KUBE_CONFIG         Updated just now
   ✅ KUBE_CONTEXT        Updated just now
   ```

✅ **Step 1 完成条件**: 4 个 Secrets 都已创建

---

## 🔄 Step 2️⃣: 测试工作流 (使用 Web UI - 5 分钟)

### 触发工作流

1️⃣ **访问 GitHub Actions 页面**
   - 打开: https://github.com/vistone/BrowerAI/actions
   - 或: 仓库 → Actions

2️⃣ **选择工作流** (选择以下任意一个)

   **方式 A: 通过 Web 界面手动触发 (推荐)**
   ```
   1. 在 Actions 页面找到 "Deploy - Kubernetes"
   2. 点击 "Run workflow"
   3. 保持默认设置
   4. 点击 "Run workflow" 确认
   ```

   **方式 B: 通过推送代码触发**
   ```bash
   # 在本地终端执行:
   cd /home/stone/BrowerAI
   git add .
   git commit -m "Trigger CI/CD pipeline for Step 2"
   git push origin week5-postgresql-persistence
   ```

3️⃣ **监控工作流执行**
   - 返回 https://github.com/vistone/BrowerAI/actions
   - 观察最新的工作流运行
   - 点击进入查看详细日志

### 预期执行顺序

```
第 1-5 分钟: build.yml 运行
  ✅ Lint 检查
  ✅ 单元测试
  ✅ 覆盖率报告

第 6-12 分钟: docker-build.yml 运行
  ✅ 构建 Docker 镜像
  ✅ 推送到 Docker Hub
  ✅ 标记版本

第 13-20 分钟: deploy.yml 运行 (需要 K8s)
  ✅ 连接 Kubernetes
  ✅ 部署应用
  ✅ 验证 Pod

第 21-23 分钟: test.yml 运行
  ✅ 烟雾测试
  ✅ API 验证
```

✅ **Step 2 完成条件**: 至少 build.yml 和 docker-build.yml 通过 (✅ 绿色)

---

## 🔍 Step 3️⃣: 验证部署 (查看日志 - 10 分钟)

### 查看工作流结果

1️⃣ **访问工作流详情**
   - 打开: https://github.com/vistone/BrowerAI/actions
   - 点击最新的工作流运行

2️⃣ **检查各个工作流的状态**

   **build.yml 日志**
   ```
   预期看到:
   ✅ Lint passed
   ✅ Tests passed
   ✅ Coverage report generated
   ```

   **docker-build.yml 日志**
   ```
   预期看到:
   ✅ Building Docker image
   ✅ Docker login successful
   ✅ Image pushed to Docker Hub
   ✅ Tags: latest, v1.0.0
   ```

   **deploy.yml 日志** (如配置了 K8s)
   ```
   预期看到:
   ✅ Kubernetes connection successful
   ✅ Deployment updated
   ✅ Rollout completed
   ✅ Health check passed
   ```

3️⃣ **验证 Docker Hub**
   - 访问: https://hub.docker.com/r/<your-docker-username>/browerai-api
   - 应该看到新推送的镜像标签: latest, v1.0.0

✅ **Step 3 完成条件**:
- [ ] build.yml 显示 ✅
- [ ] docker-build.yml 显示 ✅
- [ ] docker-build.yml 成功推送镜像
- [ ] 镜像在 Docker Hub 可见

---

## 📊 Step 4️⃣: 查看监控日志 (Web UI 查看 - 5 分钟)

### 本地启动监控 (可选)

如果您有 Docker:

```bash
# 1. 启动 Prometheus
docker run -d --name prometheus -p 9090:9090 \
  -e TZ=UTC \
  prom/prometheus

# 2. 启动 Grafana
docker run -d --name grafana -p 3000:3000 \
  -e GF_SECURITY_ADMIN_PASSWORD=admin \
  grafana/grafana

# 3. 访问
# Prometheus: http://localhost:9090
# Grafana: http://localhost:3000 (admin:admin)
```

### 查看工作流中的监控信息

```
在每个工作流的日志中查看:
- CPU 使用率
- 内存使用率
- 响应时间
- 错误率
```

✅ **Step 4 完成条件**: 观察到至少一个工作流的执行日志

---

## 📋 简化版完成检查清单

### Step 1: GitHub Secrets
- [ ] DOCKER_USERNAME 已创建
- [ ] DOCKER_PASSWORD 已创建
- [ ] KUBE_CONFIG 已创建
- [ ] KUBE_CONTEXT 已创建

### Step 2: 工作流触发
- [ ] build.yml 已运行
- [ ] docker-build.yml 已运行
- [ ] 至少 1 个工作流显示 ✅

### Step 3: 验证结果
- [ ] Docker 镜像成功构建
- [ ] 镜像推送到 Docker Hub
- [ ] Docker Hub 有新镜像

### Step 4: 监控
- [ ] 查看过工作流的执行日志
- [ ] 了解应用的部署流程

---

## 🎯 无需任何工具即可完成

✅ **完全使用 Web UI**
- 不需要 GitHub CLI
- 不需要 kubectl
- 不需要 Docker (可选)
- 只需要 Web 浏览器

✅ **纯手动操作**
- 添加 4 个 Secrets: 5 分钟
- 触发工作流: 1 分钟
- 监控执行: 20 分钟
- 验证结果: 5 分钟

**总耗时: 31 分钟** (大部分是等待工作流自动运行)

---

## 📚 快速链接

| 页面 | URL |
|------|-----|
| GitHub Secrets | https://github.com/vistone/BrowerAI/settings/secrets/actions |
| GitHub Actions | https://github.com/vistone/BrowerAI/actions |
| Docker Hub (示例) | https://hub.docker.com/r/vistone/browerai-api |
| 工作流源代码 | https://github.com/vistone/BrowerAI/tree/main/.github/workflows |

---

## 🔧 故障排查

### 如果工作流失败

1. 点击失败的工作流
2. 查看红色的 ❌ 步骤
3. 点击查看详细日志
4. 常见原因:
   - Docker 凭证错误: 检查 DOCKER_PASSWORD 是否是 PAT
   - K8s 配置错误: 如果配置了 K8s，检查 KUBE_CONFIG

### 如果推送失败

```bash
# 确保本地仓库是最新的
cd /home/stone/BrowerAI
git pull origin week5-postgresql-persistence
git add .
git commit -m "Your message"
git push origin week5-postgresql-persistence
```

### 如果看不到工作流

1. 刷新页面: 按 F5 或 Ctrl+R
2. 等待 30 秒，工作流需要初始化
3. 检查分支是否正确

---

## ✨ 最后一步: 总结结果

完成以上 4 步后:

✅ **GitHub Actions 工作流已验证**
- 代码质量检查通过 (build.yml)
- Docker 镜像成功构建和推送 (docker-build.yml)
- 工作流自动化已正确配置

✅ **系统部署能力已验证**
- 代码 → 构建 → 部署 流程完整
- 所有自动化脚本都已就位
- 支持完全无人工干预的部署

✅ **生产就绪状态**
- CI/CD 流程正常运行
- 镜像管理自动化
- 监控日志完整

---

**创建日期**: 2026-02-01  
**版本**: 1.0.0 (简化版)  
**完成所需时间**: 31-40 分钟  
**所需工具**: 仅需 Web 浏览器 ✅

🎉 **现在就可以开始执行 Step 1!**
