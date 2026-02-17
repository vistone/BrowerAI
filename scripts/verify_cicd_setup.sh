#!/bin/bash

# Week 8 Phase E - CI/CD验证脚本
# 验证完整的CI/CD流程配置

set -e

echo "================================================"
echo "🔍 Week 8 Phase E - CI/CD 流程验证"
echo "================================================"
echo ""

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

success_count=0
fail_count=0

# 检查函数
check_file() {
    local file=$1
    local description=$2
    
    if [ -f "$file" ]; then
        echo -e "${GREEN}✅${NC} $description: $file"
        ((success_count++))
        return 0
    else
        echo -e "${RED}❌${NC} $description: $file (缺失)"
        ((fail_count++))
        return 1
    fi
}

check_dir() {
    local dir=$1
    local description=$2
    
    if [ -d "$dir" ]; then
        echo -e "${GREEN}✅${NC} $description: $dir"
        ((success_count++))
        return 0
    else
        echo -e "${RED}❌${NC} $description: $dir (缺失)"
        ((fail_count++))
        return 1
    fi
}

# 1. 检查 GitHub Actions Workflows
echo "📋 1. 检查 GitHub Actions Workflows"
echo "----------------------------------------"
check_file ".github/workflows/complete-cicd.yml" "完整CI/CD流程"
check_file ".github/workflows/rollback-deployment.yml" "回滚部署流程"
check_file ".github/workflows/comprehensive-ci.yml" "综合CI测试"
check_file ".github/workflows/docker-build.yml" "Docker构建"
check_file ".github/workflows/deploy.yml" "K8s部署"
echo ""

# 2. 检查 Dockerfile
echo "🐳 2. 检查 Dockerfile"
echo "----------------------------------------"
check_file "Dockerfile.api" "API服务器Dockerfile"
check_file "Dockerfile.api-server" "API服务器Dockerfile (备选)"
check_file "Dockerfile.prod" "生产环境Dockerfile"
echo ""

# 3. 检查 Kubernetes 配置
echo "☸️  3. 检查 Kubernetes 配置"
echo "----------------------------------------"
check_dir "k8s" "K8s配置目录"
check_file "k8s/deployment.yaml" "Deployment配置"
check_file "k8s/browerai-api.yaml" "API服务配置"
check_file "k8s/browerai-ingress.yaml" "Ingress配置"
check_file "k8s/monitoring.yaml" "监控配置"
echo ""

# 4. 检查 Docker Compose
echo "🐋 4. 检查 Docker Compose"
echo "----------------------------------------"
check_file "docker-compose.yml" "开发Docker Compose"
check_file "config/docker-compose.api.yml" "API Docker Compose"
check_file "config/docker-compose.monitoring.yml" "监控Docker Compose"
echo ""

# 5. 检查构建文件
echo "🔧 5. 检查构建文件"
echo "----------------------------------------"
check_file "Cargo.toml" "Rust工作区配置"
check_file "requirements.txt" "Python依赖"
check_file "Justfile" "Just命令文件"
echo ""

# 6. 检查脚本
echo "📜 6. 检查脚本文件"
echo "----------------------------------------"
check_dir "scripts" "脚本目录"
check_file "scripts/real_system_integration_test.sh" "集成测试脚本"
if [ -d "scripts" ]; then
    echo "   脚本列表:"
    ls -1 scripts/*.sh 2>/dev/null | while read script; do
        echo "   - $(basename $script)"
    done
fi
echo ""

# 7. 验证 workflow 语法
echo "✔️  7. 验证 Workflow 语法"
echo "----------------------------------------"
for workflow in .github/workflows/*.yml; do
    if [ -f "$workflow" ]; then
        # 基本YAML语法检查
        if grep -q "^name:" "$workflow" && grep -q "^on:" "$workflow" && grep -q "^jobs:" "$workflow"; then
            echo -e "${GREEN}✅${NC} $(basename $workflow) - 基本语法正确"
            ((success_count++))
        else
            echo -e "${RED}❌${NC} $(basename $workflow) - 语法可能有误"
            ((fail_count++))
        fi
    fi
done
echo ""

# 8. 检查必需的密钥文档
echo "🔑 8. 检查密钥和配置文档"
echo "----------------------------------------"
check_file ".github/CICD_CONFIG.md" "CI/CD配置文档"
check_file ".github/DEPLOYMENT_TIMELINE.md" "部署时间线"
check_file ".github/IMPLEMENTATION_STEPS.md" "实施步骤"
echo ""

# 9. 检查 CI/CD 流程完整性
echo "🔄 9. 检查 CI/CD 流程完整性"
echo "----------------------------------------"

check_workflow_job() {
    local workflow=$1
    local job=$2
    
    if grep -q "  $job:" "$workflow"; then
        echo -e "${GREEN}✅${NC} $workflow 包含 $job"
        return 0
    else
        echo -e "${YELLOW}⚠️${NC}  $workflow 缺少 $job"
        return 1
    fi
}

MAIN_WORKFLOW=".github/workflows/complete-cicd.yml"
if [ -f "$MAIN_WORKFLOW" ]; then
    check_workflow_job "$MAIN_WORKFLOW" "build-and-test"
    check_workflow_job "$MAIN_WORKFLOW" "build-docker"
    check_workflow_job "$MAIN_WORKFLOW" "push-docker"
    check_workflow_job "$MAIN_WORKFLOW" "deploy-k8s"
    check_workflow_job "$MAIN_WORKFLOW" "post-deploy-test"
fi
echo ""

# 10. Docker 镜像验证
echo "🖼️  10. 检查 Docker 镜像配置"
echo "----------------------------------------"
if [ -f "Dockerfile.api" ]; then
    if grep -q "FROM rust" Dockerfile.api; then
        echo -e "${GREEN}✅${NC} Dockerfile.api 使用 Rust base image"
        ((success_count++))
    else
        echo -e "${YELLOW}⚠️${NC}  Dockerfile.api 可能不包含 Rust"
    fi
    
    if grep -q "EXPOSE" Dockerfile.api; then
        PORT=$(grep "EXPOSE" Dockerfile.api | awk '{print $2}')
        echo -e "${GREEN}✅${NC} Dockerfile.api 暴露端口: $PORT"
        ((success_count++))
    else
        echo -e "${YELLOW}⚠️${NC}  Dockerfile.api 未指定暴露端口"
    fi
fi
echo ""

# 11. K8s配置验证
echo "⚙️  11. 检查 K8s 配置详情"
echo "----------------------------------------"
if [ -f "k8s/deployment.yaml" ]; then
    if grep -q "kind: Deployment" k8s/deployment.yaml; then
        echo -e "${GREEN}✅${NC} deployment.yaml 类型正确"
        ((success_count++))
    fi
    
    if grep -q "replicas:" k8s/deployment.yaml; then
        REPLICAS=$(grep "replicas:" k8s/deployment.yaml | head -1 | awk '{print $2}')
        echo -e "${GREEN}✅${NC} 副本数配置: $REPLICAS"
        ((success_count++))
    fi
    
    if grep -q "image:" k8s/deployment.yaml; then
        IMAGE=$(grep "image:" k8s/deployment.yaml | head -1 | awk '{print $2}')
        echo -e "${GREEN}✅${NC} 镜像配置: $IMAGE"
        ((success_count++))
    fi
fi
echo ""

# 12. 监控配置检查
echo "📊 12. 检查监控配置"
echo "----------------------------------------"
check_file "config/prometheus.yml" "Prometheus配置"
check_file "config/alertmanager.yml" "Alertmanager配置"
check_file "config/alert_rules.yml" "告警规则"
if [ -d "grafana/provisioning" ]; then
    echo -e "${GREEN}✅${NC} Grafana provisioning 目录存在"
    ((success_count++))
fi
echo ""

# 13. 环境变量检查
echo "🌍 13. 环境变量配置"
echo "----------------------------------------"
echo "需要在 GitHub Secrets 中配置以下密钥:"
echo "  - DOCKER_USERNAME (Docker Hub用户名)"
echo "  - DOCKER_PASSWORD (Docker Hub密码/Token)"
echo "  - KUBE_CONFIG (K8s集群配置, base64编码)"
echo "  - KUBE_CONTEXT (K8s上下文名称)"
echo "  - API_ENDPOINT (API服务器端点, 可选)"
echo ""

# 总结
echo "================================================"
echo "📈 验证总结"
echo "================================================"
echo -e "${GREEN}成功:${NC} $success_count"
echo -e "${RED}失败:${NC} $fail_count"
echo ""

if [ $fail_count -eq 0 ]; then
    echo -e "${GREEN}✅ CI/CD 配置验证通过!${NC}"
    echo ""
    echo "下一步操作:"
    echo "1. 推送代码到 GitHub"
    echo "2. 在 GitHub 仓库设置中添加必要的 Secrets"
    echo "3. 创建 tag 触发完整的 CI/CD 流程: git tag v1.0.0 && git push --tags"
    echo "4. 查看 Actions 页面监控流程执行"
    exit 0
else
    echo -e "${YELLOW}⚠️  CI/CD 配置存在问题，请修复后再试${NC}"
    exit 1
fi
