#!/bin/bash

# 快速验证CI/CD配置
# Week 8 Phase E

echo "🚀 CI/CD配置快速验证"
echo ""

PASS=0
FAIL=0

check() {
    if [ -f "$1" ] || [ -d "$1" ]; then
        echo "✅ $2"
        ((PASS++))
    else
        echo "❌ $2 (缺失: $1)"
        ((FAIL++))
    fi
}

# 核心文件检查
check ".github/workflows/complete-cicd.yml" "完整CI/CD流程"
check ".github/workflows/rollback-deployment.yml" "回滚机制"
check "Dockerfile.api" "Docker配置"
check "k8s/deployment.yaml" "K8s部署配置"
check "Cargo.toml" "Rust项目配置"

echo ""
echo "📊 结果: $PASS 个检查通过, $FAIL 个失败"
echo ""

if [ $FAIL -eq 0 ]; then
    echo "✅ CI/CD配置完整"
    echo ""
    echo "📖 查看使用指南: docs/CICD_USAGE_GUIDE.md"
    echo ""
    echo "🎯 下一步:"
    echo "1. 配置GitHub Secrets (DOCKER_USERNAME, DOCKER_PASSWORD)"
    echo "2. 推送代码到GitHub触发CI/CD"
    echo "3. 或创建tag发布: git tag v1.0.0 && git push --tags"
    exit 0
else
    echo "⚠️ 有配置缺失，请检查"
    exit 1
fi
