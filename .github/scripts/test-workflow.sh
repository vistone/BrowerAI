#!/bin/bash
# 工作流测试脚本 - 测试 GitHub Actions 工作流

set -e

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🧪 GitHub Actions 工作流测试"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 颜色代码
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 配置
REPO="${1:-.}"  # 默认当前目录
BRANCH="${2:-main}"

echo -e "${BLUE}📋 工作流测试步骤${NC}"
echo ""

# Step 1: 检查 Git 状态
echo -e "${BLUE}Step 1: 检查 Git 状态${NC}"
echo "────────────────────────────────────"
cd "$REPO"

if ! git status > /dev/null 2>&1; then
    echo "❌ 不在 Git 仓库中"
    exit 1
fi

echo "✅ Git 仓库存在"
echo ""

# Step 2: 检查远程仓库
echo -e "${BLUE}Step 2: 检查远程仓库${NC}"
echo "────────────────────────────────────"

if ! git remote get-url origin > /dev/null 2>&1; then
    echo "❌ 未配置远程仓库"
    exit 1
fi

REMOTE_URL=$(git remote get-url origin)
echo "远程仓库: $REMOTE_URL"
echo "✅ 远程仓库已配置"
echo ""

# Step 3: 检查分支
echo -e "${BLUE}Step 3: 检查分支${NC}"
echo "────────────────────────────────────"

CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
echo "当前分支: $CURRENT_BRANCH"
echo "目标分支: $BRANCH"
echo ""

# Step 4: 检查 GitHub Secrets
echo -e "${BLUE}Step 4: 检查 GitHub Secrets${NC}"
echo "────────────────────────────────────"

if ! command -v gh &> /dev/null; then
    echo -e "${YELLOW}⚠️  GitHub CLI 未安装${NC}"
    echo "请访问 https://cli.github.com/ 安装"
    echo ""
else
    echo "检查必需的 Secrets..."
    
    REQUIRED_SECRETS=("DOCKER_USERNAME" "DOCKER_PASSWORD" "KUBE_CONFIG" "KUBE_CONTEXT")
    MISSING_SECRETS=()
    
    for secret in "${REQUIRED_SECRETS[@]}"; do
        if gh secret list | grep -q "^$secret"; then
            echo "  ✅ $secret 已配置"
        else
            echo "  ❌ $secret 未配置"
            MISSING_SECRETS+=("$secret")
        fi
    done
    
    if [ ${#MISSING_SECRETS[@]} -gt 0 ]; then
        echo ""
        echo -e "${YELLOW}⚠️  缺少以下 Secrets:${NC}"
        for secret in "${MISSING_SECRETS[@]}"; do
            echo "  - $secret"
        done
        echo ""
        echo "请运行: bash .github/scripts/setup-secrets.sh"
        exit 1
    fi
    
    echo -e "${GREEN}✅ 所有 Secrets 已配置${NC}"
    echo ""
fi

# Step 5: 检查工作流文件
echo -e "${BLUE}Step 5: 检查工作流文件${NC}"
echo "────────────────────────────────────"

WORKFLOW_FILES=(
    ".github/workflows/build.yml"
    ".github/workflows/docker-build.yml"
    ".github/workflows/deploy.yml"
    ".github/workflows/test.yml"
    ".github/workflows/rollback.yml"
)

for workflow in "${WORKFLOW_FILES[@]}"; do
    if [ -f "$workflow" ]; then
        echo "✅ $workflow"
    else
        echo "❌ $workflow 不存在"
    fi
done
echo ""

# Step 6: 创建测试文件
echo -e "${BLUE}Step 6: 创建测试提交${NC}"
echo "────────────────────────────────────"

TEST_FILE=".github/workflows/test-trigger.txt"
echo "This file triggers the workflow tests at $(date)" > "$TEST_FILE"

echo "创建测试文件: $TEST_FILE"
echo ""

# Step 7: 提交和推送
echo -e "${BLUE}Step 7: 提交和推送到 GitHub${NC}"
echo "────────────────────────────────────"

if [ "$CURRENT_BRANCH" != "$BRANCH" ]; then
    echo -e "${YELLOW}⚠️  当前分支 ($CURRENT_BRANCH) 与目标分支 ($BRANCH) 不同${NC}"
    read -p "是否切换到 $BRANCH 分支? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git checkout "$BRANCH" || {
            echo "创建新分支..."
            git checkout -b "$BRANCH"
        }
    else
        echo "使用当前分支: $CURRENT_BRANCH"
        BRANCH=$CURRENT_BRANCH
    fi
fi

echo "添加文件..."
git add "$TEST_FILE"

echo "提交..."
git commit -m "test: trigger workflow at $(date)" || echo "无需提交 (文件未变化)"

echo "推送到 GitHub ($BRANCH)..."
git push origin "$BRANCH" || echo "推送失败，请检查权限"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ 工作流测试启动!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "接下来的步骤:"
echo "1. 访问: https://github.com/vistone/BrowerAI/actions"
echo "2. 查看工作流运行状态"
echo "3. 观察以下工作流的执行:"
echo "   - CI - Build and Test"
echo "   - Docker - Build and Push Image"
echo "   - Deploy - Kubernetes Deployment"
echo "   - Test - Post-Deployment Verification"
echo ""
echo "预期耗时: 15-25 分钟"
echo ""
echo "监控命令:"
if command -v gh &> /dev/null; then
    echo "  gh run list                    # 列出所有运行"
    echo "  gh run view --log              # 查看最新运行的日志"
    echo "  gh run watch                   # 实时监控最新运行"
fi
echo ""
