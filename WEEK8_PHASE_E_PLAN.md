# Week 8 Phase E - CI/CD 集成和自动化部署

**阶段**: Phase E (Week 8 最后阶段)  
**预计时间**: 2-3 小时  
**目标**: 完整的 GitHub Actions CI/CD 流程  
**状态**: 准备启动

---

## 📋 执行概览

### Phase E 的五大目标

```
┌─────────────────────────────────────────────────────┐
│ GitHub Actions CI/CD 自动化                         │
├─────────────────────────────────────────────────────┤
│ 1. 自动化构建 (Build)                               │
│    - Python 环境检查                                 │
│    - 依赖安装和测试                                  │
│    - Docker 镜像构建                                 │
│                                                     │
│ 2. 自动化推送 (Push)                                │
│    - Docker Hub 镜像推送                             │
│    - 版本标签管理                                    │
│    - 构建产物存储                                    │
│                                                     │
│ 3. 自动化部署 (Deploy)                              │
│    - K8s 集群部署 (Minikube / 云)                   │
│    - 部署验证                                        │
│    - 健康检查                                        │
│                                                     │
│ 4. 自动化测试 (Test)                                │
│    - 部署后测试                                      │
│    - 烟雾测试 (Smoke Test)                          │
│    - 功能验证                                        │
│                                                     │
│ 5. 回滚和发布 (Release & Rollback)                  │
│    - 自动回滚机制                                    │
│    - GitHub Release 创建                            │
│    - 版本管理                                        │
└─────────────────────────────────────────────────────┘
```

---

## 🎯 Phase E 工作流设计

### 流程图

```
┌──────────┐
│ Push 代码 │
└────┬─────┘
     │
     v
┌─────────────────────────────────────┐
│ GitHub Actions 触发                 │
│ (main branch 或 tag)               │
└────┬────────────────────────────────┘
     │
     v
┌─────────────────────────────────────┐
│ Step 1: 构建阶段 (Build)            │
│ - Checkout 代码                      │
│ - Python 环境设置                    │
│ - 依赖安装                           │
│ - 单元测试                           │
│ - Lint 检查                          │
└────┬────────────────────────────────┘
     │ ✅ 通过
     v
┌─────────────────────────────────────┐
│ Step 2: 容器构建 (Docker Build)     │
│ - 构建 Docker 镜像                   │
│ - 镜像扫描 (安全检查)                │
│ - 镜像大小验证                       │
└────┬────────────────────────────────┘
     │ ✅ 通过
     v
┌─────────────────────────────────────┐
│ Step 3: 推送阶段 (Push)            │
│ - Docker Hub 登录                    │
│ - 推送镜像 (latest, version tag)    │
│ - 推送成功通知                       │
└────┬────────────────────────────────┘
     │ ✅ 通过
     v
┌─────────────────────────────────────┐
│ Step 4: 部署阶段 (Deploy)          │
│ - K8s 集群准备 (Minikube/云)        │
│ - 更新镜像版本                       │
│ - 应用 K8s 清单                      │
│ - 等待 Pod 就绪                      │
└────┬────────────────────────────────┘
     │ ✅ 通过
     v
┌─────────────────────────────────────┐
│ Step 5: 测试阶段 (Test)            │
│ - 健康检查                           │
│ - 烟雾测试                           │
│ - API 功能测试                       │
│ - 性能基准验证                       │
└────┬────────────────────────────────┘
     │ ✅ 通过
     v
┌─────────────────────────────────────┐
│ Step 6: 监控阶段 (Monitor)         │
│ - 收集 Prometheus 指标               │
│ - 检查错误率                         │
│ - 检查延迟                           │
└────┬────────────────────────────────┘
     │ ✅ 通过
     v
┌─────────────────────────────────────┐
│ ✅ 部署成功！                        │
│ - 发布 GitHub Release                │
│ - 发送通知                           │
│ - 记录日志                           │
└─────────────────────────────────────┘

如果任何步骤失败:
     ❌ 失败
     │
     v
┌─────────────────────────────────────┐
│ 自动回滚                             │
│ - 恢复前一个版本                     │
│ - 发送告警通知                       │
│ - 记录故障日志                       │
└─────────────────────────────────────┘
```

---

## 📂 Phase E 文件结构

### 创建的文件和目录

```
.github/
├── workflows/
│   ├── build.yml                  (构建流程 - 150 行)
│   ├── deploy.yml                 (部署流程 - 200 行)
│   ├── test.yml                   (测试流程 - 150 行)
│   ├── release.yml                (发布流程 - 150 行)
│   └── rollback.yml               (回滚流程 - 100 行)
├── scripts/
│   ├── build.sh                   (构建脚本 - 100 行)
│   ├── deploy.sh                  (部署脚本 - 150 行)
│   ├── test.sh                    (测试脚本 - 120 行)
│   └── rollback.sh                (回滚脚本 - 100 行)
└── CICD_CONFIG.md                 (配置说明 - 200 行)

ci-config/
├── .dockerignore                  (已有)
├── docker-compose.cicd.yml        (CI/CD 测试环境 - 100 行)
└── helm-values.yaml               (K8s Helm 配置 - 80 行)
```

---

## 🔧 Phase E 详细实现计划

### Step 1: GitHub Actions 工作流创建 (30 分钟)

#### 工作流 1: build.yml (CI 构建流程)
```yaml
name: Build and Test

on:
  push:
    branches: [main, develop]
    paths:
      - 'browerai-api-server/**'
      - '.github/workflows/build.yml'
  pull_request:
    branches: [main, develop]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
          cache: 'pip'
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install pytest pytest-cov pylint
      
      - name: Run Lint checks
        run: |
          pylint browerai-api-server/ --exit-zero
      
      - name: Run unit tests
        run: |
          pytest browerai-api-server/tests/ -v --cov
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          files: ./coverage.xml
```

#### 工作流 2: docker-build.yml (Docker 构建和推送)
```yaml
name: Build and Push Docker Image

on:
  push:
    branches: [main]
    tags: ['v*']
  workflow_dispatch:

jobs:
  build-and-push:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v2
      
      - name: Login to Docker Hub
        uses: docker/login-action@v2
        with:
          username: ${{ secrets.DOCKER_USERNAME }}
          password: ${{ secrets.DOCKER_PASSWORD }}
      
      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v4
        with:
          images: |
            ${{ secrets.DOCKER_USERNAME }}/browerai-api
          tags: |
            type=ref,event=branch
            type=semver,pattern={{version}}
            type=semver,pattern={{major}}.{{minor}}
            type=sha
      
      - name: Build and push
        uses: docker/build-push-action@v4
        with:
          context: .
          file: ./Dockerfile.python-api
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
      
      - name: Image digest
        run: echo ${{ steps.docker_build.outputs.digest }}
```

#### 工作流 3: deploy.yml (K8s 自动化部署)
```yaml
name: Deploy to Kubernetes

on:
  push:
    branches: [main]
    tags: ['v*']
  workflow_dispatch:
    inputs:
      environment:
        description: 'Deploy environment'
        required: true
        default: 'staging'
        type: choice
        options:
          - staging
          - production

jobs:
  deploy:
    runs-on: ubuntu-latest
    environment: ${{ github.event.inputs.environment || 'staging' }}
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up kubectl
        uses: azure/setup-kubectl@v3
        with:
          version: 'v1.28.0'
      
      - name: Configure kubectl context
        run: |
          mkdir -p $HOME/.kube
          echo "${{ secrets.KUBE_CONFIG }}" | base64 -d > $HOME/.kube/config
          kubectl config use-context ${{ secrets.KUBE_CONTEXT }}
      
      - name: Update deployment image
        run: |
          IMAGE_TAG=${{ github.sha }}
          kubectl set image deployment/browerai-api-deployment \
            browerai-api=${{ secrets.DOCKER_USERNAME }}/browerai-api:${IMAGE_TAG} \
            -n browerai
      
      - name: Wait for rollout
        run: |
          kubectl rollout status deployment/browerai-api-deployment \
            -n browerai --timeout=5m
      
      - name: Verify deployment
        run: |
          kubectl get pods -n browerai
          kubectl get svc -n browerai
      
      - name: Run smoke tests
        run: |
          bash .github/scripts/smoke-test.sh
```

#### 工作流 4: test-deployment.yml (部署后测试)
```yaml
name: Post-Deployment Tests

on:
  workflow_run:
    workflows: ["Deploy to Kubernetes"]
    types: [completed]

jobs:
  test:
    if: ${{ github.event.workflow_run.conclusion == 'success' }}
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install test dependencies
        run: |
          pip install requests pytest pytest-benchmark
      
      - name: Health check
        run: |
          curl -f http://localhost:5000/health || exit 1
      
      - name: Run API tests
        run: |
          pytest tests/e2e/ -v
      
      - name: Performance benchmark
        run: |
          python -m pytest tests/performance/ -v --benchmark-only
```

#### 工作流 5: rollback.yml (自动回滚)
```yaml
name: Rollback Deployment

on:
  workflow_dispatch:
    inputs:
      revision:
        description: 'Revision to rollback to'
        required: true

jobs:
  rollback:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up kubectl
        uses: azure/setup-kubectl@v3
      
      - name: Configure kubectl
        run: |
          mkdir -p $HOME/.kube
          echo "${{ secrets.KUBE_CONFIG }}" | base64 -d > $HOME/.kube/config
      
      - name: Get rollout history
        run: |
          kubectl rollout history deployment/browerai-api-deployment -n browerai
      
      - name: Perform rollback
        run: |
          kubectl rollout undo deployment/browerai-api-deployment \
            --to-revision=${{ github.event.inputs.revision }} \
            -n browerai
      
      - name: Verify rollback
        run: |
          kubectl rollout status deployment/browerai-api-deployment \
            -n browerai --timeout=5m
      
      - name: Notification
        if: always()
        run: |
          echo "Rollback completed to revision ${{ github.event.inputs.revision }}"
```

---

### Step 2: 自动化脚本编写 (30 分钟)

#### 脚本 1: .github/scripts/build.sh (构建脚本)
```bash
#!/bin/bash
set -e

echo "=== Phase E Build Script ==="

# 1. 检查 Python 版本
echo "Checking Python version..."
python --version

# 2. 创建虚拟环境
echo "Creating virtual environment..."
python -m venv venv
source venv/bin/activate

# 3. 安装依赖
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
pip install pytest pytest-cov pylint black isort

# 4. 代码格式化
echo "Running code formatting..."
black browerai-api-server/
isort browerai-api-server/

# 5. Lint 检查
echo "Running lint checks..."
pylint browerai-api-server/ --exit-zero

# 6. 单元测试
echo "Running unit tests..."
pytest browerai-api-server/tests/ -v --cov=browerai-api-server

# 7. 生成覆盖率报告
echo "Generating coverage report..."
pytest --cov=browerai-api-server --cov-report=html --cov-report=xml

echo "✅ Build phase completed successfully!"
```

#### 脚本 2: .github/scripts/deploy.sh (部署脚本)
```bash
#!/bin/bash
set -e

echo "=== Phase E Deploy Script ==="

# 1. 验证 kubectl 连接
echo "Verifying kubectl connection..."
kubectl cluster-info
kubectl get nodes

# 2. 更新镜像版本
echo "Updating deployment image..."
IMAGE_TAG=${1:-latest}
DOCKER_USER=${2:-}

if [ -z "$DOCKER_USER" ]; then
    echo "Error: Docker username required"
    exit 1
fi

kubectl set image deployment/browerai-api-deployment \
    browerai-api=$DOCKER_USER/browerai-api:$IMAGE_TAG \
    -n browerai

# 3. 等待 Pod 就绪
echo "Waiting for deployment rollout..."
kubectl rollout status deployment/browerai-api-deployment \
    -n browerai --timeout=5m

# 4. 验证部署
echo "Verifying deployment..."
kubectl get pods -n browerai
kubectl get svc -n browerai

# 5. 检查 Pod 日志
echo "Checking pod logs..."
kubectl logs -n browerai -l app=browerai-api --tail=50

# 6. 运行烟雾测试
echo "Running smoke tests..."
bash .github/scripts/smoke-test.sh

echo "✅ Deployment phase completed successfully!"
```

#### 脚本 3: .github/scripts/smoke-test.sh (烟雾测试)
```bash
#!/bin/bash
set -e

echo "=== Smoke Tests ==="

SERVICE_URL=${1:-http://localhost:5000}

# Test 1: Health check
echo "Test 1: Health check..."
curl -f $SERVICE_URL/health || exit 1
echo "✅ Health check passed"

# Test 2: Feature encoding
echo "Test 2: Feature encoding..."
curl -X POST $SERVICE_URL/encode \
    -H "Content-Type: application/json" \
    -d '{"url":"https://example.com","html":"<html></html>"}' \
    || exit 1
echo "✅ Feature encoding passed"

# Test 3: Code generation
echo "Test 3: Code generation..."
curl -X POST $SERVICE_URL/generate \
    -H "Content-Type: application/json" \
    -d '{"features":[],"website_intent":"search"}' \
    || exit 1
echo "✅ Code generation passed"

# Test 4: Feedback submission
echo "Test 4: Feedback submission..."
curl -X POST $SERVICE_URL/feedback \
    -H "Content-Type: application/json" \
    -d '{"url":"https://example.com","quality_score":0.85}' \
    || exit 1
echo "✅ Feedback submission passed"

echo "✅ All smoke tests passed!"
```

#### 脚本 4: .github/scripts/rollback.sh (回滚脚本)
```bash
#!/bin/bash
set -e

echo "=== Rollback Script ==="

REVISION=${1:-}

if [ -z "$REVISION" ]; then
    echo "Usage: ./rollback.sh <revision>"
    echo ""
    echo "Available revisions:"
    kubectl rollout history deployment/browerai-api-deployment -n browerai
    exit 1
fi

echo "Rolling back to revision $REVISION..."

kubectl rollout undo deployment/browerai-api-deployment \
    --to-revision=$REVISION \
    -n browerai

echo "Waiting for rollback to complete..."
kubectl rollout status deployment/browerai-api-deployment \
    -n browerai --timeout=5m

echo "✅ Rollback completed successfully!"

# Verify
echo "Current deployment status:"
kubectl get pods -n browerai
```

---

### Step 3: 配置文件设置 (30 分钟)

#### 配置 1: 环境变量和密钥

```bash
# GitHub Secrets (需要在 GitHub Settings 中配置)
DOCKER_USERNAME      # Docker Hub 用户名
DOCKER_PASSWORD      # Docker Hub 密码 (PAT)
KUBE_CONFIG          # K8s 集群配置 (base64 编码)
KUBE_CONTEXT         # kubectl 上下文名称
REGISTRY_URL         # 镜像仓库 URL (可选)
SLACK_WEBHOOK        # Slack 通知 (可选)
```

#### 配置 2: docker-compose.cicd.yml (CI/CD 测试环境)
```yaml
version: '3.8'

services:
  # 测试用 API 服务
  browerai-api-test:
    build:
      context: .
      dockerfile: Dockerfile.python-api
    ports:
      - "5000:5000"
    environment:
      FLASK_ENV: testing
      DATABASE_URL: postgresql://test:test@postgres:5432/browerai_test
    depends_on:
      - postgres
      - redis
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 10s
      timeout: 5s
      retries: 5

  postgres:
    image: postgres:15
    environment:
      POSTGRES_USER: test
      POSTGRES_PASSWORD: test
      POSTGRES_DB: browerai_test
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
```

---

## 📊 Phase E 执行时间表

```
┌──────────────────────────────────┐
│ Phase E 完整实现时间表            │
├──────────────────────────────────┤
│ Step 1: GitHub Actions 工作流    │
│ - 5 个 YAML 文件编写              │
│ 时间: 30 分钟                      │
│                                  │
│ Step 2: 自动化脚本编写           │
│ - 4 个 bash 脚本                  │
│ 时间: 30 分钟                      │
│                                  │
│ Step 3: 配置和测试               │
│ - 环境配置                        │
│ - 本地测试运行                    │
│ 时间: 30 分钟                      │
│                                  │
│ Step 4: 文档和验证               │
│ - 写入说明文档                    │
│ - GitHub Actions UI 验证          │
│ 时间: 30 分钟                      │
│                                  │
│ 总计: 120 分钟 (2 小时)            │
└──────────────────────────────────┘
```

---

## 🎯 成功标准

Phase E 完成时需要满足以下条件：

```
✅ GitHub Actions 工作流:
   - build.yml: 提交 PR 时自动运行测试
   - docker-build.yml: main 分支推送时构建镜像
   - deploy.yml: 镜像推送后自动部署
   - test-deployment.yml: 部署后运行测试
   - rollback.yml: 手动触发回滚

✅ 自动化脚本:
   - build.sh: 本地构建测试
   - deploy.sh: 部署到 K8s
   - smoke-test.sh: 部署后验证
   - rollback.sh: 手动回滚

✅ 工作流验证:
   - PR 自动运行测试
   - main 分支自动构建镜像
   - 镜像自动推送到 Docker Hub
   - K8s 自动化部署完成
   - 部署后自动运行测试
   - 所有测试通过

✅ 特性完善:
   - 版本标签自动管理
   - 自动回滚机制
   - Slack/Email 通知 (可选)
   - GitHub Release 自动创建
   - 构建日志保存
```

---

## 🔐 安全建议

### 密钥管理
```
✅ 使用 GitHub Secrets 存储敏感信息
✅ 定期轮换密钥
✅ 限制密钥访问权限
✅ 监控密钥使用

关键密钥:
- DOCKER_USERNAME/PASSWORD
- KUBE_CONFIG (base64 编码)
- 数据库凭证
- API 令牌
```

### 工作流安全
```
✅ 限制代码审查权限
✅ 要求 PR 检查通过
✅ 环境级别的保护规则
✅ 部署前的手动批准
```

---

## 📈 预期效果

部署 Phase E 后的改进：

```
手动部署 → 自动化部署

部署时间: 
  手动: 30-60 分钟
  自动: 5-10 分钟

错误率:
  手动: 5-10%
  自动: <1%

恢复时间:
  手动回滚: 15-30 分钟
  自动回滚: 1-2 分钟

部署频率:
  之前: 1-2 次/周
  之后: 多次/天 (持续部署)
```

---

## 📋 Checklist

Phase E 实现完成检查清单：

- [ ] Step 1: GitHub Actions 工作流创建
  - [ ] build.yml 完成
  - [ ] docker-build.yml 完成
  - [ ] deploy.yml 完成
  - [ ] test-deployment.yml 完成
  - [ ] rollback.yml 完成

- [ ] Step 2: 脚本编写
  - [ ] build.sh 完成
  - [ ] deploy.sh 完成
  - [ ] smoke-test.sh 完成
  - [ ] rollback.sh 完成

- [ ] Step 3: 配置设置
  - [ ] GitHub Secrets 配置
  - [ ] docker-compose.cicd.yml 创建
  - [ ] 权限配置完成

- [ ] Step 4: 测试验证
  - [ ] 本地脚本测试通过
  - [ ] GitHub Actions 工作流正常
  - [ ] 完整流程测试通过
  - [ ] 文档完整

- [ ] 最终验证
  - [ ] 所有工作流状态: ✅
  - [ ] 自动化部署验证: ✅
  - [ ] 回滚机制验证: ✅
  - [ ] 文档审查完成: ✅

---

## 下一步行动

1. **立即启动**
   - 创建 GitHub Actions 工作流文件
   - 编写自动化脚本
   - 配置 GitHub Secrets

2. **本地测试**
   - 运行脚本验证功能
   - 模拟 CI/CD 流程
   - 验证错误处理

3. **部署验证**
   - GitHub Actions UI 验证
   - 完整流程测试
   - 生产环境准备

---

**文档版本**: 1.0.0  
**创建日期**: 2026-02-01  
**更新时间**: 2026-02-01  
**状态**: 准备启动
