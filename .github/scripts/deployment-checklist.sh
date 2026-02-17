#!/bin/bash

# BrowerAI 完整部署检查清单
# 文件: .github/scripts/deployment-checklist.sh
# 目的: 交互式检查部署的每个步骤

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 计数器
TOTAL_CHECKS=0
PASSED_CHECKS=0
FAILED_CHECKS=0

# 工具函数
print_header() {
    echo -e "\n${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}\n"
}

check_item() {
    local description=$1
    local command=$2
    ((TOTAL_CHECKS++))
    
    echo -n "检查: $description ... "
    if eval "$command" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ 通过${NC}"
        ((PASSED_CHECKS++))
        return 0
    else
        echo -e "${RED}❌ 失败${NC}"
        ((FAILED_CHECKS++))
        return 1
    fi
}

show_result() {
    echo -e "\n${BLUE}══════════════════════════════════════════════════════════${NC}"
    echo -e "检查结果摘要"
    echo -e "${BLUE}══════════════════════════════════════════════════════════${NC}"
    echo -e "总检查数: $TOTAL_CHECKS"
    echo -e "通过: ${GREEN}$PASSED_CHECKS${NC}"
    echo -e "失败: ${RED}$FAILED_CHECKS${NC}"
    
    if [ $FAILED_CHECKS -eq 0 ]; then
        echo -e "\n${GREEN}🎉 所有检查通过! 系统可以部署${NC}\n"
        return 0
    else
        echo -e "\n${RED}⚠️  存在 $FAILED_CHECKS 项失败. 请修复后重试${NC}\n"
        return 1
    fi
}

# === Step 1: GitHub Secrets 检查 ===
check_secrets() {
    print_header "Step 1️⃣: GitHub Secrets 配置检查"
    
    echo "检查 GitHub Secrets 是否已配置..."
    
    check_item "DOCKER_USERNAME secret 存在" \
        "gh secret list | grep -q DOCKER_USERNAME"
    
    check_item "DOCKER_PASSWORD secret 存在" \
        "gh secret list | grep -q DOCKER_PASSWORD"
    
    check_item "KUBE_CONFIG secret 存在" \
        "gh secret list | grep -q KUBE_CONFIG"
    
    check_item "KUBE_CONTEXT secret 存在" \
        "gh secret list | grep -q KUBE_CONTEXT"
    
    echo -e "\n${YELLOW}📝 GitHub Secrets 配置指南:${NC}"
    echo "使用以下命令配置 secrets:"
    echo "  bash .github/scripts/setup-secrets.sh"
    echo "或手动配置:"
    echo "  gh secret set DOCKER_USERNAME --body 'your-username'"
    echo "  gh secret set DOCKER_PASSWORD --body 'your-pat-token'"
    echo "  gh secret set KUBE_CONFIG --body \"$(cat ~/.kube/config | base64)\""
    echo "  gh secret set KUBE_CONTEXT --body \"$(kubectl config current-context)\""
}

# === Step 2: 工作流文件检查 ===
check_workflows() {
    print_header "Step 2️⃣: GitHub Actions 工作流文件检查"
    
    echo "检查所需的工作流文件..."
    
    check_item "build.yml 存在" \
        "test -f .github/workflows/build.yml"
    
    check_item "docker-build.yml 存在" \
        "test -f .github/workflows/docker-build.yml"
    
    check_item "deploy.yml 存在" \
        "test -f .github/workflows/deploy.yml"
    
    check_item "test.yml 存在" \
        "test -f .github/workflows/test.yml"
    
    check_item "rollback.yml 存在" \
        "test -f .github/workflows/rollback.yml"
    
    echo -e "\n${YELLOW}📝 工作流文件检查完成${NC}"
}

# === Step 3: Kubernetes 集群检查 ===
check_kubernetes() {
    print_header "Step 3️⃣: Kubernetes 集群检查"
    
    echo "检查 Kubernetes 环境..."
    
    check_item "kubectl 已安装" \
        "command -v kubectl > /dev/null"
    
    check_item "kubectl 可连接到集群" \
        "kubectl cluster-info > /dev/null"
    
    check_item "namespace 'browerai' 存在" \
        "kubectl get ns browerai > /dev/null 2>&1"
    
    if kubectl get ns browerai > /dev/null 2>&1; then
        check_item "Deployment 'browerai-api-deployment' 存在" \
            "kubectl get deployment browerai-api-deployment -n browerai > /dev/null 2>&1"
        
        check_item "Pod 正在运行" \
            "kubectl get pods -n browerai -l app=browerai-api | grep Running"
    fi
    
    echo -e "\n${YELLOW}📝 Kubernetes 配置:${NC}"
    echo "创建 namespace (如果不存在):"
    echo "  kubectl create namespace browerai"
    echo ""
    echo "查看部署状态:"
    echo "  kubectl get deployment -n browerai"
    echo "  kubectl get pods -n browerai"
}

# === Step 4: Docker 环境检查 ===
check_docker() {
    print_header "Step 4️⃣: Docker 环境检查"
    
    echo "检查 Docker 环境..."
    
    check_item "Docker 已安装" \
        "command -v docker > /dev/null"
    
    check_item "Docker daemon 正在运行" \
        "docker ps > /dev/null"
    
    echo -e "\n${YELLOW}📝 Docker Hub 认证:${NC}"
    echo "登录 Docker Hub:"
    echo "  docker login"
    echo "或使用 DOCKER_PASSWORD (PAT token)"
}

# === Step 5: 应用部署检查 ===
check_application() {
    print_header "Step 5️⃣: 应用部署检查"
    
    if ! kubectl get ns browerai > /dev/null 2>&1; then
        echo -e "${YELLOW}⚠️  Namespace 'browerai' 不存在, 跳过应用检查${NC}"
        return
    fi
    
    echo "检查应用部署状态..."
    
    READY_PODS=$(kubectl get pods -n browerai -l app=browerai-api \
        --no-headers 2>/dev/null | grep Running | wc -l)
    
    if [ "$READY_PODS" -gt 0 ]; then
        check_item "Pod 正在运行 ($READY_PODS 个)" \
            "kubectl get pods -n browerai -l app=browerai-api | grep Running"
        
        check_item "Service 存在" \
            "kubectl get svc browerai-api-service -n browerai > /dev/null 2>&1"
        
        check_item "HPA 配置" \
            "kubectl get hpa -n browerai | grep browerai-api"
    else
        echo -e "${YELLOW}⚠️  没有运行中的 Pod, 跳过应用检查${NC}"
    fi
    
    echo -e "\n${YELLOW}📝 部署应用:${NC}"
    echo "  kubectl apply -f k8s/namespace.yaml"
    echo "  kubectl apply -f k8s/deployment.yaml"
    echo "  kubectl apply -f k8s/service.yaml"
}

# === Step 6: 监控系统检查 ===
check_monitoring() {
    print_header "Step 6️⃣: 监控系统检查"
    
    echo "检查监控系统..."
    
    # Prometheus
    if check_item "Prometheus 运行" \
        "curl -s http://localhost:9090/-/healthy > /dev/null"; then
        :
    fi
    
    # Grafana
    if check_item "Grafana 运行" \
        "curl -s http://localhost:3000/api/health > /dev/null"; then
        :
    fi
    
    # K8s Prometheus (如果使用 Helm)
    if kubectl get ns monitoring > /dev/null 2>&1; then
        check_item "Prometheus (K8s) 运行" \
            "kubectl get pods -n monitoring | grep prometheus"
    fi
    
    echo -e "\n${YELLOW}📝 启动监控:${NC}"
    echo "Docker 方式:"
    echo "  docker run -d --name prometheus -p 9090:9090 prom/prometheus"
    echo "  docker run -d --name grafana -p 3000:3000 grafana/grafana"
    echo ""
    echo "Kubernetes 方式 (推荐):"
    echo "  helm repo add prometheus-community https://prometheus-community.github.io/helm-charts"
    echo "  helm install prometheus prometheus-community/kube-prometheus-stack -n monitoring --create-namespace"
}

# === Step 7: 工作流运行检查 ===
check_workflow_runs() {
    print_header "Step 7️⃣: GitHub Actions 工作流运行检查"
    
    echo "检查最近的工作流运行..."
    
    if ! command -v gh > /dev/null; then
        echo -e "${YELLOW}⚠️  GitHub CLI 未安装${NC}"
        return
    fi
    
    # 获取最近的运行
    LATEST_RUN=$(gh run list --limit 1 --json status,name,createdAt,number \
        2>/dev/null || echo "")
    
    if [ -z "$LATEST_RUN" ]; then
        echo -e "${YELLOW}⚠️  没有工作流运行记录${NC}"
        echo ""
        echo "触发第一次运行:"
        echo "  git add ."
        echo "  git commit -m 'Trigger CI/CD pipeline'"
        echo "  git push"
        return
    fi
    
    check_item "存在工作流运行" \
        "gh run list --limit 1 | grep -q '.'"
    
    echo -e "\n${YELLOW}📝 最近的工作流运行:${NC}"
    gh run list --limit 5 --json status,name,createdAt,number \
        --template '{{range .}}[{{.number}}] {{.status}} - {{.name}} ({{.createdAt}}){{"\n"}}{{end}}'
    
    echo ""
    echo "监控工作流:"
    echo "  gh run watch"
    echo "  gh run view <run-id> --log"
}

# === Step 8: 集成测试 ===
check_integration_tests() {
    print_header "Step 8️⃣: 集成测试检查"
    
    echo "检查测试脚本..."
    
    check_item "build.sh 存在" \
        "test -f .github/scripts/build.sh"
    
    check_item "smoke-test.sh 存在" \
        "test -f .github/scripts/smoke-test.sh"
    
    check_item "verify-deployment.sh 存在" \
        "test -f .github/scripts/verify-deployment.sh"
    
    check_item "setup-secrets.sh 存在" \
        "test -f .github/scripts/setup-secrets.sh"
    
    echo -e "\n${YELLOW}📝 运行测试:${NC}"
    echo "本地构建和测试:"
    echo "  bash .github/scripts/build.sh"
    echo ""
    echo "设置 Secrets:"
    echo "  bash .github/scripts/setup-secrets.sh"
    echo ""
    echo "部署后验证:"
    echo "  bash .github/scripts/verify-deployment.sh"
    echo ""
    echo "烟雾测试 (需要应用运行):"
    echo "  bash .github/scripts/smoke-test.sh http://localhost:5000"
}

# === 主菜单 ===
show_menu() {
    print_header "BrowerAI 完整部署检查清单"
    
    echo "选择要检查的内容:"
    echo ""
    echo "  1️⃣  GitHub Secrets 配置"
    echo "  2️⃣  工作流文件"
    echo "  3️⃣  Kubernetes 集群"
    echo "  4️⃣  Docker 环境"
    echo "  5️⃣  应用部署"
    echo "  6️⃣  监控系统"
    echo "  7️⃣  工作流运行"
    echo "  8️⃣  集成测试"
    echo "  a  🔍 检查全部"
    echo "  q  退出"
    echo ""
}

# === 主程序 ===
main() {
    while true; do
        TOTAL_CHECKS=0
        PASSED_CHECKS=0
        FAILED_CHECKS=0
        
        show_menu
        read -p "请选择 [1-8/a/q]: " choice
        
        case $choice in
            1)
                check_secrets
                show_result || true
                ;;
            2)
                check_workflows
                show_result || true
                ;;
            3)
                check_kubernetes
                show_result || true
                ;;
            4)
                check_docker
                show_result || true
                ;;
            5)
                check_application
                show_result || true
                ;;
            6)
                check_monitoring
                show_result || true
                ;;
            7)
                check_workflow_runs
                show_result || true
                ;;
            8)
                check_integration_tests
                show_result || true
                ;;
            a|A)
                check_secrets
                PASS1=$PASSED_CHECKS
                FAIL1=$FAILED_CHECKS
                
                check_workflows
                PASS2=$PASSED_CHECKS
                FAIL2=$FAILED_CHECKS
                
                check_kubernetes
                PASS3=$PASSED_CHECKS
                FAIL3=$FAILED_CHECKS
                
                check_docker
                PASS4=$PASSED_CHECKS
                FAIL4=$FAILED_CHECKS
                
                check_application
                PASS5=$PASSED_CHECKS
                FAIL5=$FAILED_CHECKS
                
                check_monitoring
                PASS6=$PASSED_CHECKS
                FAIL6=$FAILED_CHECKS
                
                check_workflow_runs
                PASS7=$PASSED_CHECKS
                FAIL7=$FAILED_CHECKS
                
                check_integration_tests
                
                show_result || true
                ;;
            q|Q)
                echo -e "${BLUE}再见! 🚀${NC}"
                exit 0
                ;;
            *)
                echo -e "${RED}无效选择${NC}"
                ;;
        esac
        
        read -p "按 Enter 继续..."
    done
}

main "$@"
