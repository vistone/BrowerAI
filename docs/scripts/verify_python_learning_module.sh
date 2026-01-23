#!/bin/bash
# Python 学习模块快速验证脚本

set -e

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║     Python 学习模块系统验证                                    ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}[1/5] 检查 Python 版本${NC}"
python3 --version
echo ""

echo -e "${BLUE}[2/5] 验证模块导入${NC}"
python3 << 'EOF'
try:
    from training.core.models import WebsiteGenerator, CodeEncoder, CodeDecoder, WebsiteIntentClassifier
    print("✅ 所有模块导入成功")
    print(f"   - WebsiteGenerator: {WebsiteGenerator}")
    print(f"   - CodeEncoder: {CodeEncoder}")
    print(f"   - CodeDecoder: {CodeDecoder}")
    print(f"   - WebsiteIntentClassifier: {WebsiteIntentClassifier}")
except Exception as e:
    print(f"❌ 导入失败: {e}")
    exit(1)
EOF
echo ""

echo -e "${BLUE}[3/5] 运行单元测试${NC}"
if python3 training/test_website_generator.py > /tmp/unit_test.log 2>&1; then
    echo "✅ 单元测试通过"
    grep "✅" /tmp/unit_test.log | head -5
else
    echo "❌ 单元测试失败"
    cat /tmp/unit_test.log
    exit 1
fi
echo ""

echo -e "${BLUE}[4/5] 运行演示脚本${NC}"
if timeout 60 python3 training/demo_website_generator.py > /tmp/demo.log 2>&1; then
    echo "✅ 演示脚本执行成功"
    grep "✅" /tmp/demo.log | head -5
else
    echo "❌ 演示脚本失败"
    tail -20 /tmp/demo.log
    exit 1
fi
echo ""

echo -e "${BLUE}[5/5] 运行集成测试${NC}"
if timeout 120 python3 training/integration_tests.py > /tmp/integration.log 2>&1; then
    echo "✅ 集成测试通过"
    grep "✅" /tmp/integration.log | head -5
else
    echo "❌ 集成测试失败"
    tail -20 /tmp/integration.log
    exit 1
fi
echo ""

echo "╔════════════════════════════════════════════════════════════════╗"
echo -e "║     ${GREEN}✅ 所有验证通过！系统已准备就绪${NC}                              ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

echo "下一步："
echo "1. 准备网站配对数据: data/website_paired.jsonl"
echo "   格式: {\"original\": \"...\", \"simplified\": \"...\", \"intent\": {...}}"
echo ""
echo "2. 开始训练:"
echo "   python3 training/scripts/train_paired_website_generator.py"
echo ""
echo "3. 监控损失:"
echo "   - HTML 代码损失"
echo "   - CSS 代码损失"
echo "   - JS 代码损失"
echo "   - 意图分类损失"
echo ""

echo "文档参考:"
echo "- 快速参考: PYTHON_LEARNING_MODULE_QUICK_REFERENCE.md"
echo "- 修复报告: PYTHON_LEARNING_MODULE_FIX_REPORT.md"
echo "- 项目状态: PYTHON_LEARNING_MODULE_FINAL_STATUS.md"
echo ""
