#!/bin/bash

# 全面的项目测试套件

set -e

echo "================================================"
echo "🚀 BrowerAI 全面测试套件"
echo "================================================"
echo ""

TEST_RESULTS=()

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 测试函数
run_test() {
    local test_name=$1
    local command=$2
    
    echo -e "${BLUE}📋 $test_name${NC}"
    if eval "$command" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ 通过${NC}"
        TEST_RESULTS+=("✅ $test_name")
    else
        echo -e "${RED}❌ 失败${NC}"
        TEST_RESULTS+=("❌ $test_name")
    fi
    echo ""
}

# ==================== 1. Rust编译测试 ====================
echo -e "${YELLOW}═══ 第1部分: Rust编译测试 ═══${NC}"
echo ""

run_test "API服务器编译" "cd /home/stone/BrowerAI && cargo build --release -p browerai-api-server"
run_test "主库编译" "cd /home/stone/BrowerAI && cargo build --release --lib -p browerai"
run_test "HTML解析器编译" "cd /home/stone/BrowerAI && cargo build --release -p browerai-html-parser"
run_test "CSS解析器编译" "cd /home/stone/BrowerAI && cargo build --release -p browerai-css-parser"

# ==================== 2. 前端测试 ====================
echo -e "${YELLOW}═══ 第2部分: 前端测试 ═══${NC}"
echo ""

run_test "前端依赖检查" "test -f /home/stone/BrowerAI/crates/browerai-webclient/node_modules/react/package.json"
run_test "TypeScript编译" "cd /home/stone/BrowerAI/crates/browerai-webclient && npm run type-check"

# ==================== 3. 文件和目录结构测试 ====================
echo -e "${YELLOW}═══ 第3部分: 结构验证 ═══${NC}"
echo ""

run_test "Docker Compose存在" "test -f /home/stone/BrowerAI/docker-compose.yml"
run_test "K8s部署清单" "test -f /home/stone/BrowerAI/k8s/deployment.yaml"
run_test "CI/CD主流程" "test -f /home/stone/BrowerAI/.github/workflows/complete-cicd.yml"
run_test "回滚流程" "test -f /home/stone/BrowerAI/.github/workflows/rollback-deployment.yml"

# ==================== 4. 文档检查 ====================
echo -e "${YELLOW}═══ 第4部分: 文档完整性 ═══${NC}"
echo ""

run_test "CI/CD使用指南" "test -f /home/stone/BrowerAI/docs/CICD_USAGE_GUIDE.md"
run_test "项目最终状态报告" "test -f /home/stone/BrowerAI/PROJECT_FINAL_STATUS.md"
run_test "Phase E完成报告" "test -f /home/stone/BrowerAI/WEEK8_PHASE_E_COMPLETION_REPORT.md"

# ==================== 5. 脚本测试 ====================
echo -e "${YELLOW}═══ 第5部分: 测试脚本 ═══${NC}"
echo ""

run_test "简单API测试脚本" "test -x /home/stone/BrowerAI/scripts/simple_api_test.sh"
run_test "快速CI/CD检查" "test -x /home/stone/BrowerAI/scripts/quick_cicd_check.sh"

# ==================== 6. 配置文件验证 ====================
echo -e "${YELLOW}═══ 第6部分: 配置文件 ═══${NC}"
echo ""

run_test "Cargo工作区" "test -f /home/stone/BrowerAI/Cargo.toml"
run_test "环境配置" "test -f /home/stone/BrowerAI/.env.example || test -f /home/stone/BrowerAI/config/prometheus.yml"
run_test "Dockerfile" "test -f /home/stone/BrowerAI/Dockerfile.api"

# ==================== 7. 依赖检查 ====================
echo -e "${YELLOW}═══ 第7部分: 依赖检查 ═══${NC}"
echo ""

# 检查关键命令
check_command() {
    if command -v $1 &> /dev/null; then
        echo -e "${GREEN}✅ $1 已安装${NC}"
        TEST_RESULTS+=("✅ $1 已安装")
    else
        echo -e "${YELLOW}⚠️ $1 未安装${NC}"
        TEST_RESULTS+=("⚠️ $1 未安装")
    fi
}

check_command "cargo"
check_command "npm"
check_command "docker"
check_command "git"

echo ""

# ==================== 总结 ====================
echo -e "${YELLOW}═══════════════════════════════════════${NC}"
echo -e "${BLUE}📊 测试结果总结${NC}"
echo -e "${YELLOW}═══════════════════════════════════════${NC}"
echo ""

pass=0
fail=0

for result in "${TEST_RESULTS[@]}"; do
    if [[ $result == ✅* ]]; then
        echo -e "${GREEN}$result${NC}"
        ((pass++))
    elif [[ $result == ❌* ]]; then
        echo -e "${RED}$result${NC}"
        ((fail++))
    else
        echo "$result"
    fi
done

echo ""
echo -e "${BLUE}通过: $pass${NC}"
if [ $fail -gt 0 ]; then
    echo -e "${RED}失败: $fail${NC}"
else
    echo -e "${GREEN}失败: 0${NC}"
fi

echo ""

if [ $fail -eq 0 ]; then
    echo -e "${GREEN}✅ 所有测试通过！${NC}"
    echo -e "${GREEN}项目已就绪提交到GitHub${NC}"
    exit 0
else
    echo -e "${YELLOW}⚠️ 有$fail个测试失败，请检查${NC}"
    exit 1
fi
