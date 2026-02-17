#!/bin/bash
# 部署验证脚本 - 验证自动部署是否成功

set -e

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ 自动部署验证"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 颜色代码
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

NAMESPACE="${1:-browerai}"

echo -e "${BLUE}📋 部署验证清单${NC}"
echo ""

# 检查 1: 工作流运行状态
echo -e "${BLUE}检查 1: GitHub Actions 工作流运行${NC}"
echo "────────────────────────────────────"

if command -v gh &> /dev/null; then
    echo "最近的工作流运行:"
    gh run list --limit 5 2>/dev/null || echo "⚠️  无法获取工作流列表"
    echo ""
    
    # 获取最新运行的状态
    LATEST_RUN=$(gh run list --limit 1 --json status,name,updatedAt --template '{{range .}}{{.name}}: {{.status}} ({{.updatedAt}}){{end}}' 2>/dev/null || echo "")
    if [ -n "$LATEST_RUN" ]; then
        echo "最新运行: $LATEST_RUN"
    fi
else
    echo -e "${YELLOW}⚠️  GitHub CLI 未安装，无法检查工作流${NC}"
fi
echo ""

# 检查 2: Kubernetes 部署
echo -e "${BLUE}检查 2: Kubernetes 部署状态${NC}"
echo "────────────────────────────────────"

if ! command -v kubectl &> /dev/null; then
    echo -e "${RED}❌ kubectl 未安装${NC}"
    echo "请访问 https://kubernetes.io/docs/tasks/tools/ 安装"
    echo ""
else
    echo "检查集群连接..."
    if kubectl cluster-info > /dev/null 2>&1; then
        echo -e "${GREEN}✅ 集群连接正常${NC}"
        echo ""
        
        echo "检查 Namespace: $NAMESPACE"
        if kubectl get ns "$NAMESPACE" > /dev/null 2>&1; then
            echo -e "${GREEN}✅ Namespace 存在${NC}"
            echo ""
            
            # 检查 Deployment
            echo "Deployment 状态:"
            if kubectl get deployment -n "$NAMESPACE" -l app=browerai-api > /dev/null 2>&1; then
                kubectl get deployment -n "$NAMESPACE" -l app=browerai-api -o wide
                echo ""
                
                # 检查 Pod
                echo "Pod 状态:"
                kubectl get pods -n "$NAMESPACE" -l app=browerai-api -o wide
                echo ""
                
                # 检查 Service
                echo "Service 状态:"
                kubectl get svc -n "$NAMESPACE" -l app=browerai-api
                echo ""
                
                # 检查 HPA
                echo "HPA 状态:"
                kubectl get hpa -n "$NAMESPACE" 2>/dev/null || echo "HPA 未创建"
                echo ""
                
                # 检查最近的事件
                echo "最近事件 (最后 10 个):"
                kubectl get events -n "$NAMESPACE" --sort-by='.lastTimestamp' | tail -10
                echo ""
            else
                echo -e "${RED}❌ Deployment 不存在${NC}"
                echo "请检查部署是否成功"
            fi
        else
            echo -e "${RED}❌ Namespace 不存在${NC}"
            echo "可能的原因:"
            echo "  1. K8s 部署尚未启动"
            echo "  2. K8s 配置有误"
            echo "  3. 集群不可用"
        fi
    else
        echo -e "${RED}❌ 无法连接到 Kubernetes 集群${NC}"
        echo "请检查 kubeconfig 配置"
    fi
fi
echo ""

# 检查 3: Docker 镜像
echo -e "${BLUE}检查 3: Docker 镜像${NC}"
echo "────────────────────────────────────"

if command -v docker &> /dev/null; then
    DOCKER_USER=${DOCKER_USERNAME:-$(git config user.name)}
    
    echo "检查镜像: $DOCKER_USER/browerai-api"
    
    if docker pull "$DOCKER_USER/browerai-api:latest" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ 镜像存在${NC}"
        echo ""
        
        echo "镜像详情:"
        docker images "$DOCKER_USER/browerai-api" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"
    else
        echo -e "${YELLOW}⚠️  镜像不存在或无法拉取${NC}"
        echo "可能的原因:"
        echo "  1. 镜像尚未推送"
        echo "  2. Docker Hub 认证失败"
        echo "  3. 镜像名称不正确"
    fi
else
    echo -e "${YELLOW}⚠️  Docker 未安装${NC}"
fi
echo ""

# 检查 4: 服务可达性
echo -e "${BLUE}检查 4: 服务可达性${NC}"
echo "────────────────────────────────────"

if command -v kubectl &> /dev/null && [ -n "$NAMESPACE" ]; then
    echo "启动端口转发..."
    
    # 检查是否已有端口转发
    if netstat -tuln 2>/dev/null | grep -q ":5000 "; then
        echo "端口 5000 已被占用"
    else
        # 启动端口转发
        kubectl port-forward -n "$NAMESPACE" svc/browerai-api-service 5000:5000 > /dev/null 2>&1 &
        PORT_FORWARD_PID=$!
        sleep 2
        
        echo "端口转发 PID: $PORT_FORWARD_PID"
        echo ""
    fi
    
    # 测试服务
    echo "测试服务可达性..."
    if curl -s http://localhost:5000/health > /dev/null 2>&1; then
        echo -e "${GREEN}✅ 服务可达 (/health)${NC}"
        
        HEALTH=$(curl -s http://localhost:5000/health)
        echo "响应: $HEALTH"
    else
        echo -e "${YELLOW}⚠️  服务不可达${NC}"
        echo "可能的原因:"
        echo "  1. Pod 尚未启动"
        echo "  2. 应用启动失败"
        echo "  3. 网络连接问题"
    fi
    
    # 杀死端口转发
    if [ -n "$PORT_FORWARD_PID" ]; then
        kill $PORT_FORWARD_PID 2>/dev/null || true
    fi
else
    echo -e "${YELLOW}⚠️  无法进行服务测试${NC}"
fi
echo ""

# 检查 5: 日志
echo -e "${BLUE}检查 5: Pod 日志${NC}"
echo "────────────────────────────────────"

if command -v kubectl &> /dev/null; then
    echo "最近的 Pod 日志 (最后 20 行):"
    echo ""
    
    if kubectl get pods -n "$NAMESPACE" -l app=browerai-api > /dev/null 2>&1; then
        kubectl logs -n "$NAMESPACE" -l app=browerai-api --tail=20 --timestamps=true 2>/dev/null || echo "无日志"
    else
        echo "⚠️  未找到 Pod"
    fi
else
    echo -e "${YELLOW}⚠️  kubectl 未安装${NC}"
fi
echo ""

# 总结
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ 部署验证完成"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "后续步骤:"
echo "1. 确认所有检查都通过"
echo "2. 如有失败，查看日志确定原因"
echo "3. 使用 smoke-test.sh 运行功能验证"
echo "4. 配置监控 (Prometheus + Grafana)"
echo ""
echo "命令参考:"
echo "  kubectl get pods -n $NAMESPACE"
echo "  kubectl logs -n $NAMESPACE -l app=browerai-api -f"
echo "  kubectl describe pod <pod-name> -n $NAMESPACE"
echo "  kubectl port-forward -n $NAMESPACE svc/browerai-api-service 5000:5000"
echo ""
