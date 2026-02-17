#!/bin/bash
# GitHub Secrets 配置指南和自动化脚本

set -e

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔐 GitHub Secrets 配置向导"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 颜色代码
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 配置项检查
echo -e "${BLUE}📋 必需的 GitHub Secrets 清单${NC}"
echo ""
echo "1. DOCKER_USERNAME"
echo "   说明: Docker Hub 用户名"
echo "   获取: https://hub.docker.com/ → Account"
echo ""
echo "2. DOCKER_PASSWORD"  
echo "   说明: Docker Hub 个人访问令牌 (PAT)"
echo "   获取: https://hub.docker.com/settings/security → New Access Token"
echo "   注意: 必须是 PAT，不能是密码!"
echo ""
echo "3. KUBE_CONFIG"
echo "   说明: Kubernetes 配置 (base64 编码)"
echo "   获取: cat ~/.kube/config | base64"
echo "   用于: K8s 集群认证"
echo ""
echo "4. KUBE_CONTEXT"
echo "   说明: kubectl 上下文名称"
echo "   获取: kubectl config current-context"
echo "   示例: minikube 或 arn:aws:eks:region:account:cluster/name"
echo ""

# 提示用户操作
echo -e "${YELLOW}⚠️  请按以下步骤配置:${NC}"
echo ""
echo "方式 1: 使用 GitHub CLI (推荐)"
echo "────────────────────────────────────────"
echo ""
echo "# 1. 首先安装 GitHub CLI"
echo "# https://cli.github.com/"
echo ""
echo "# 2. 认证到 GitHub"
echo "gh auth login"
echo ""
echo "# 3. 获取必需的值"
echo ""
echo "# Docker Hub:"
echo "export DOCKER_USERNAME=\"your-username\""
echo "export DOCKER_PASSWORD=\"your-pat-token\""
echo ""
echo "# Kubernetes:"
echo "export KUBE_CONFIG=\$(cat ~/.kube/config | base64)"
echo "export KUBE_CONTEXT=\$(kubectl config current-context)"
echo ""
echo "# 4. 配置 Secrets"
echo "gh secret set DOCKER_USERNAME --body \"\$DOCKER_USERNAME\""
echo "gh secret set DOCKER_PASSWORD --body \"\$DOCKER_PASSWORD\""
echo "gh secret set KUBE_CONFIG --body \"\$KUBE_CONFIG\""
echo "gh secret set KUBE_CONTEXT --body \"\$KUBE_CONTEXT\""
echo ""
echo "# 5. 验证配置"
echo "gh secret list"
echo ""
echo "───────────────────────────────────────────"
echo ""
echo "方式 2: 使用 GitHub Web UI"
echo "───────────────────────────────────────────"
echo ""
echo "1. 进入仓库 → Settings"
echo "2. 左侧菜单 → Secrets and variables → Actions"
echo "3. 点击 'New repository secret'"
echo "4. 输入以下 4 个 secret:"
echo ""
echo "   Secret 1:"
echo "   Name: DOCKER_USERNAME"
echo "   Value: your-docker-username"
echo ""
echo "   Secret 2:"
echo "   Name: DOCKER_PASSWORD"
echo "   Value: your-docker-hub-pat"
echo ""
echo "   Secret 3:"
echo "   Name: KUBE_CONFIG"
echo "   Value: (base64 encoded kubeconfig)"
echo ""
echo "   Secret 4:"
echo "   Name: KUBE_CONTEXT"
echo "   Value: minikube (或你的 K8s 上下文)"
echo ""
echo "5. 点击 'Add secret'"
echo ""
echo "───────────────────────────────────────────"
echo ""

# 交互式配置
read -p "是否现在配置这些 Secrets? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "开始配置..."
    echo ""
    
    # 检查 GitHub CLI
    if ! command -v gh &> /dev/null; then
        echo -e "${RED}❌ GitHub CLI 未安装${NC}"
        echo "请访问 https://cli.github.com/ 安装"
        exit 1
    fi
    
    # 检查认证
    if ! gh auth status > /dev/null 2>&1; then
        echo -e "${RED}❌ GitHub CLI 未认证${NC}"
        echo "请运行: gh auth login"
        exit 1
    fi
    
    echo -e "${GREEN}✅ GitHub CLI 已认证${NC}"
    echo ""
    
    # 获取 Docker 用户名
    read -p "输入 Docker Hub 用户名: " docker_username
    
    # 获取 Docker PAT
    echo ""
    echo "访问 https://hub.docker.com/settings/security 获取 Personal Access Token"
    read -sp "输入 Docker Hub PAT (不会显示): " docker_password
    echo ""
    
    # 获取 K8s 配置
    echo ""
    if [ -f ~/.kube/config ]; then
        kube_config=$(cat ~/.kube/config | base64 -w 0)
        echo -e "${GREEN}✅ 找到 kubeconfig${NC}"
    else
        echo -e "${RED}❌ 未找到 ~/.kube/config${NC}"
        read -p "输入 kubeconfig 路径: " kube_path
        kube_config=$(cat "$kube_path" | base64 -w 0)
    fi
    
    # 获取 K8s 上下文
    if command -v kubectl &> /dev/null; then
        kube_context=$(kubectl config current-context)
        echo -e "${GREEN}✅ 当前 K8s 上下文: $kube_context${NC}"
    else
        echo -e "${YELLOW}⚠️  kubectl 未找到，手动输入${NC}"
        read -p "输入 K8s 上下文名称: " kube_context
    fi
    
    echo ""
    echo "配置以下 Secrets:"
    echo "  DOCKER_USERNAME: $docker_username"
    echo "  DOCKER_PASSWORD: ****"
    echo "  KUBE_CONFIG: (base64, 长度: ${#kube_config})"
    echo "  KUBE_CONTEXT: $kube_context"
    echo ""
    
    read -p "确认无误? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo ""
        echo "正在配置..."
        
        gh secret set DOCKER_USERNAME --body "$docker_username"
        echo "✅ DOCKER_USERNAME 已设置"
        
        gh secret set DOCKER_PASSWORD --body "$docker_password"
        echo "✅ DOCKER_PASSWORD 已设置"
        
        gh secret set KUBE_CONFIG --body "$kube_config"
        echo "✅ KUBE_CONFIG 已设置"
        
        gh secret set KUBE_CONTEXT --body "$kube_context"
        echo "✅ KUBE_CONTEXT 已设置"
        
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "✅ GitHub Secrets 配置完成!"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""
        echo "验证配置:"
        gh secret list
    else
        echo "已取消配置"
        exit 0
    fi
else
    echo ""
    echo "请按照上述说明手动配置 GitHub Secrets"
    echo ""
    echo "配置完成后，运行以下命令验证:"
    echo "  gh secret list"
    echo ""
    exit 0
fi
