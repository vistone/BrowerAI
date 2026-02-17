# CI/CD 配置说明

**文档版本**: 1.0.0  
**最后更新**: 2026-02-01  
**阶段**: Week 8 Phase E

---

## 目录

1. [概览](#概览)
2. [GitHub Secrets 配置](#github-secrets-配置)
3. [工作流文件说明](#工作流文件说明)
4. [脚本说明](#脚本说明)
5. [使用指南](#使用指南)
6. [故障排查](#故障排查)

---

## 概览

BrowerAI CI/CD 系统使用 GitHub Actions 实现完整的自动化流程：

```
代码提交 → 构建测试 → Docker 镜像 → K8s 部署 → 验证测试 → 完成
```

### 核心工作流

| 工作流 | 触发条件 | 目的 |
|--------|---------|------|
| `build.yml` | Push 到 main/develop | 构建和测试 |
| `docker-build.yml` | Push 到 main 或 tag | 构建 Docker 镜像 |
| `deploy.yml` | Docker 镜像推送后 | K8s 自动部署 |
| `test.yml` | 部署后自动触发 | 部署后验证 |
| `rollback.yml` | 手动触发 | 回滚到前一版本 |

---

## GitHub Secrets 配置

### 必需的密钥

在 GitHub 仓库的 `Settings → Secrets and variables → Actions` 中配置以下密钥：

#### 1. Docker 相关密钥

```
DOCKER_USERNAME
  类型: Repository secret
  说明: Docker Hub 用户名
  获取方式: https://hub.docker.com/
  示例: your-docker-username

DOCKER_PASSWORD
  类型: Repository secret
  说明: Docker Hub 个人访问令牌 (Personal Access Token)
  获取方式: 
    1. 登录 Docker Hub
    2. 进入 Account Settings → Security
    3. 创建 New Access Token
    4. 复制令牌值
  注意: 不能使用密码！必须是 PAT
```

#### 2. Kubernetes 相关密钥

```
KUBE_CONFIG
  类型: Repository secret
  说明: Kubernetes 配置 (base64 编码)
  获取方式:
    1. 获取 ~/.kube/config 文件
    2. Base64 编码: cat ~/.kube/config | base64
    3. 复制编码后的内容
  
KUBE_CONTEXT
  类型: Repository secret
  说明: kubectl 上下文名称
  获取方式: kubectl config current-context
  示例: minikube 或 arn:aws:eks:...
```

#### 3. 可选密钥

```
SLACK_WEBHOOK (可选)
  用途: Slack 通知
  获取方式: Slack Incoming Webhook

REGISTRY_URL (可选)
  用途: 自定义镜像仓库
  默认: docker.io
```

### 配置步骤

#### 步骤 1: 创建 Docker Hub PAT

```bash
# 在 Docker Hub web UI 中:
1. 登录到 https://hub.docker.com/
2. 账户 → Account Settings
3. Security → New Access Token
4. 选择 "Read & Write" 权限
5. 生成并复制令牌
```

#### 步骤 2: 获取 Kubernetes 配置

```bash
# 对于 Minikube:
cat ~/.kube/config | base64

# 对于云集群 (EKS, GKE 等):
# 先获取正确的 kubeconfig
# 然后 base64 编码
cat /path/to/kubeconfig | base64
```

#### 步骤 3: 在 GitHub 中配置

```bash
# 通过 GitHub CLI:
gh secret set DOCKER_USERNAME --body "your-username"
gh secret set DOCKER_PASSWORD --body "your-token"
gh secret set KUBE_CONFIG --body "$(cat ~/.kube/config | base64)"
gh secret set KUBE_CONTEXT --body "minikube"
```

或通过 GitHub Web UI:
1. 进入仓库 → Settings
2. 左侧菜单 → Secrets and variables → Actions
3. 点击 "New repository secret"
4. 输入名称和值
5. 点击 "Add secret"

---

## 工作流文件说明

### 1. build.yml - CI 构建流程

**触发条件**:
- Push 到 main, develop, week5-postgresql-persistence 分支
- Pull Request 到 main, develop

**执行步骤**:
1. 检查代码格式 (black, isort)
2. 运行 linter (pylint, flake8)
3. 执行单元测试
4. 生成覆盖率报告
5. 上传到 Codecov

**配置项**:
```yaml
matrix.python-version: ['3.11']  # 可添加其他版本
cache: 'pip'                     # 缓存 pip 依赖
```

### 2. docker-build.yml - Docker 镜像构建

**触发条件**:
- Push 到 main 分支
- 创建 tag (v*, release-*)
- 手动触发 (`workflow_dispatch`)

**执行步骤**:
1. 登录到 Docker Hub
2. 提取元数据 (标签, 版本)
3. 构建 Docker 镜像
4. 扫描安全漏洞 (可选)
5. 推送镜像到 Docker Hub

**标签规则**:
```
main 分支:        username/browerai-api:latest
Tag v1.0.0:      username/browerai-api:1.0.0
Tag v1.0.0:      username/browerai-api:1.0
Tag v1.0.0:      username/browerai-api:1
Commit SHA:       username/browerai-api:main-<sha>
```

### 3. deploy.yml - Kubernetes 部署

**触发条件**:
- Push 到 main 分支
- 创建 release tag
- 手动触发 (选择环境)

**环境选择**:
```
staging      # 测试环境
production   # 生产环境
```

**执行步骤**:
1. 获取 kubectl 和 K8s 配置
2. 连接到集群
3. 更新部署镜像版本
4. 等待 Pod 就绪
5. 运行烟雾测试

**更新策略**:
```yaml
type: RollingUpdate
maxSurge: 1                # 额外增加 1 个 Pod
maxUnavailable: 0          # 不允许不可用的 Pod (zero downtime)
```

### 4. test.yml - 部署后测试

**触发条件**:
- deploy.yml 完成时自动触发
- 手动触发

**测试项目**:
1. 健康检查 (`/health`)
2. 特征编码 (`POST /encode`)
3. 代码生成 (`POST /generate`)
4. 反馈提交 (`POST /feedback`)
5. 性能基准测试
6. 资源指标收集

**期望结果**:
```
✅ 所有 4 项 API 测试通过
✅ 性能基准在预期范围
✅ 没有错误日志
```

### 5. rollback.yml - 回滚部署

**触发条件**:
- 手动触发 (点击 "Run workflow" 按钮)

**输入参数**:
```
revision:    回滚的版本号 (留空为前一个)
namespace:   命名空间选择 (browerai, browerai-staging, browerai-prod)
```

**回滚过程**:
1. 显示回滚历史
2. 执行回滚命令
3. 等待新 Pod 就绪
4. 运行健康检查
5. 确认回滚成功

---

## 脚本说明

所有脚本位于 `.github/scripts/` 目录

### 1. build.sh - 本地构建

**用途**: 本地编译和测试应用

**使用**:
```bash
bash .github/scripts/build.sh
```

**执行内容**:
- 检查 Python 版本
- 创建虚拟环境
- 安装依赖
- 代码格式化
- 运行 linter
- 执行单元测试
- 生成覆盖率报告

**输出**:
- `venv/` - 虚拟环境
- `htmlcov/` - HTML 覆盖率报告
- `coverage.xml` - Codecov 格式

### 2. smoke-test.sh - 烟雾测试

**用途**: 验证部署后的应用健康状态

**使用**:
```bash
bash .github/scripts/smoke-test.sh [URL] [TIMEOUT]

# 示例:
bash .github/scripts/smoke-test.sh http://localhost:5000 30
bash .github/scripts/smoke-test.sh http://my-app.example.com 60
```

**测试项**:
1. 健康检查 (`GET /health`)
2. 特征编码 (`POST /encode`)
3. 代码生成 (`POST /generate`)
4. 反馈提交 (`POST /feedback`)
5. 响应时间基准 (10 个请求)

**成功标准**:
- 所有 4 项 API 测试通过
- 响应时间 < 50ms (优秀) / < 100ms (良好) / < 200ms (可接受)

### 3. rollback.sh - 手动回滚

**用途**: 从命令行手动回滚部署

**使用**:
```bash
# 设置环境变量
export NAMESPACE=browerai
export DEPLOYMENT_NAME=browerai-api-deployment

# 查看历史
bash .github/scripts/rollback.sh

# 回滚到特定版本
bash .github/scripts/rollback.sh 2

# 回滚到前一个版本
bash .github/scripts/rollback.sh previous
```

**回滚历史示例**:
```
REVISION  CHANGE-CAUSE
3         kubectl set image ...
2         kubectl set image ...
1         kubectl apply -f ...
```

---

## 使用指南

### 场景 1: 推送代码到 main 分支

```bash
git checkout main
git pull origin main
# 修改代码...
git add .
git commit -m "feat: new feature"
git push origin main
```

**自动执行**:
1. ✅ `build.yml` - 构建和测试
2. ✅ `docker-build.yml` - 构建 Docker 镜像并推送
3. ✅ `deploy.yml` - 自动部署到 staging
4. ✅ `test.yml` - 部署后验证

**预期结果**: 应用部署到 staging 环境并通过所有测试

### 场景 2: 发布新版本

```bash
git checkout main
git tag -a v1.0.0 -m "Release v1.0.0"
git push origin v1.0.0
```

**自动执行**:
1. ✅ `docker-build.yml` - 构建镜像并标记为 v1.0.0
2. ✅ `deploy.yml` - 部署到 production (需要手动确认)

**预期结果**: v1.0.0 镜像推送到 Docker Hub, 部署到 production

### 场景 3: 部署失败需要回滚

```bash
# 方式 1: 使用 GitHub Web UI
1. 进入 Actions → Rollback
2. 点击 "Run workflow"
3. 选择 namespace 和 revision
4. 点击 "Run"

# 方式 2: 使用命令行
bash .github/scripts/rollback.sh previous
```

**执行过程**:
1. 显示回滚历史
2. 确认回滚版本
3. 执行 kubectl rollout undo
4. 验证新版本正常运行

### 场景 4: 本地开发测试

```bash
# 1. 本地构建
bash .github/scripts/build.sh

# 2. 启动应用
source venv/bin/activate
python browerai-api-server/app.py

# 3. 在另一个终端运行测试
bash .github/scripts/smoke-test.sh http://localhost:5000
```

---

## 故障排查

### 问题 1: Docker 登录失败

**错误信息**:
```
Error response from daemon: unauthorized: incorrect username or password
```

**原因**: DOCKER_PASSWORD 不是 PAT 或 credentials 错误

**解决**:
```bash
# 1. 重新生成 PAT
#    访问 https://hub.docker.com/settings/security
#    删除旧的 token，创建新的

# 2. 更新 GitHub Secret
gh secret set DOCKER_PASSWORD --body "your-new-pat"

# 3. 重新运行工作流
```

### 问题 2: Kubernetes 连接失败

**错误信息**:
```
error: Unable to connect to the server: x509: certificate signed by unknown authority
```

**原因**: KUBE_CONFIG base64 编码错误或过期

**解决**:
```bash
# 1. 重新获取 kubeconfig
cat ~/.kube/config | base64

# 2. 更新 GitHub Secret
gh secret set KUBE_CONFIG --body "your-new-config"

# 3. 验证 KUBE_CONTEXT
kubectl config current-context
gh secret set KUBE_CONTEXT --body "your-context"
```

### 问题 3: Pod 无法启动

**错误信息**:
```
ImagePullBackOff: Failed to pull image
```

**原因**: Docker 镜像不存在或标签错误

**解决**:
```bash
# 1. 检查镜像是否推送成功
docker pull your-username/browerai-api:latest

# 2. 检查 K8s 部署中的镜像名称
kubectl describe deployment browerai-api-deployment -n browerai

# 3. 重新构建和推送镜像
# 触发 docker-build.yml 工作流
```

### 问题 4: 部署超时

**错误信息**:
```
error: timed out waiting for the condition
```

**原因**: Pod 启动时间过长或资源不足

**解决**:
```bash
# 1. 检查 Pod 日志
kubectl logs -n browerai -l app=browerai-api

# 2. 检查 Pod 事件
kubectl describe pod <pod-name> -n browerai

# 3. 增加超时时间 (编辑 deploy.yml)
--timeout=10m  # 改为 10 分钟

# 4. 检查资源可用性
kubectl top nodes
kubectl describe nodes
```

### 问题 5: 回滚失败

**错误信息**:
```
error: no change-cause found for revision X
```

**原因**: 指定的版本不存在或已被清理

**解决**:
```bash
# 1. 查看可用版本
kubectl rollout history deployment/browerai-api-deployment -n browerai

# 2. 选择存在的版本
bash .github/scripts/rollback.sh 1

# 3. 或回滚到前一个
bash .github/scripts/rollback.sh previous
```

---

## 最佳实践

### 安全性

✅ **密钥管理**:
- 使用 GitHub Secrets 而非硬编码
- 定期轮换密钥 (每 3 个月)
- 限制密钥权限 (最小权限原则)
- 不在日志中输出密钥

✅ **访问控制**:
- 为 main 分支启用分支保护
- 要求 PR 审查后才能合并
- 仅授权用户可以手动触发部署

✅ **镜像安全**:
- 扫描 Docker 镜像漏洞
- 使用最小基础镜像
- 定期更新依赖

### 性能

✅ **缓存优化**:
```yaml
cache: 'pip'  # 缓存 pip 依赖
```

✅ **并行执行**:
- 多个测试可并行运行
- 使用 matrix 进行多版本测试

✅ **超时优化**:
- 设置合理的超时时间
- 避免过长的等待

### 监控

✅ **日志收集**:
- 工作流日志自动保存
- K8s Pod 日志导出
- 构建失败时保存工件

✅ **通知**:
- GitHub UI 显示工作流状态
- 可集成 Slack 通知
- 失败时发送告警

---

## 快速参考

### 常用命令

```bash
# 查看工作流运行状态
gh workflow list
gh run list

# 查看 K8s 部署
kubectl get deployment -n browerai
kubectl describe deployment browerai-api-deployment -n browerai

# 查看 Pod 日志
kubectl logs -n browerai -l app=browerai-api -f

# 手动回滚
bash .github/scripts/rollback.sh previous

# 运行烟雾测试
bash .github/scripts/smoke-test.sh http://localhost:5000
```

### 文件位置

```
.github/
├── workflows/              # GitHub Actions 工作流
│   ├── build.yml          # CI 构建流程
│   ├── docker-build.yml   # Docker 镜像构建
│   ├── deploy.yml         # K8s 部署
│   ├── test.yml           # 部署后测试
│   └── rollback.yml       # 回滚流程
└── scripts/               # 辅助脚本
    ├── build.sh           # 本地构建
    ├── smoke-test.sh      # 烟雾测试
    └── rollback.sh        # 手动回滚
```

---

## 联系和支持

如遇到问题，请：

1. 查看 [故障排查](#故障排查) 部分
2. 检查工作流运行日志
3. 提出 GitHub Issue
4. 联系 DevOps 团队

---

**最后更新**: 2026-02-01  
**版本**: 1.0.0  
**状态**: ✅ 完成
