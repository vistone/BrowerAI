#!/bin/bash
# CSS深度学习完整演示脚本

set -e  # 出错停止

echo ""
echo "╔════════════════════════════════════════════════════════╗"
echo "║   BrowerAI CSS深度学习系统 - 完整演示                  ║"
echo "║   CSS Enhanced Learning System - Full Demo             ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""

# 配置
PROJECT_DIR="/home/stone/BrowerAI"
PYTHON="python"

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 进入项目目录
cd "$PROJECT_DIR"

# 步骤1: 显示项目信息
echo -e "${BLUE}【步骤1】项目信息${NC}"
echo "════════════════════════════════════════════════════════"
echo "项目路径: $PROJECT_DIR"
echo "Python版本: $($PYTHON --version)"
echo ""

# 检查虚拟环境
if [ -d ".venv" ]; then
    echo -e "${GREEN}✓ 虚拟环境已存在${NC}"
    source .venv/bin/activate
else
    echo -e "${YELLOW}⚠ 虚拟环境不存在，请先创建${NC}"
    exit 1
fi

# 步骤2: 显示模型信息
echo ""
echo -e "${BLUE}【步骤2】模型检查${NC}"
echo "════════════════════════════════════════════════════════"

if [ -f "checkpoints/css_style_model.pt" ]; then
    echo -e "${GREEN}✓ CSS学习模型已存在${NC}"
    ls -lh checkpoints/css_style_model.pt
else
    echo -e "${YELLOW}⚠ CSS学习模型不存在${NC}"
fi

if [ -f "checkpoints/website_understanding_model.pt" ]; then
    echo -e "${GREEN}✓ ML模型已存在${NC}"
    ls -lh checkpoints/website_understanding_model.pt
else
    echo -e "${YELLOW}⚠ ML模型不存在${NC}"
fi

echo ""

# 步骤3: 显示文档
echo -e "${BLUE}【步骤3】项目文档${NC}"
echo "════════════════════════════════════════════════════════"

DOCS=(
    "CSS_LEARNING_INDEX.md"
    "CSS_PROJECT_SUMMARY.md"
    "CSS_LEARNING_QUICK_REF.md"
    "CSS_LEARNING_SUMMARY.md"
    "CSS_IMPROVEMENT_GUIDE.md"
    "CSS_LEARNING_COMPLETION_REPORT.md"
)

echo "可用文档:"
for doc in "${DOCS[@]}"; do
    if [ -f "$doc" ]; then
        size=$(wc -l < "$doc")
        echo -e "  ${GREEN}✓${NC} $doc ($size行)"
    fi
done

echo ""

# 步骤4: 运行选择菜单
echo -e "${BLUE}【步骤4】选择操作${NC}"
echo "════════════════════════════════════════════════════════"
echo ""
echo "请选择操作:"
echo "  1) 运行完整ML流程（包含CSS学习）"
echo "  2) 仅训练CSS学习模型"
echo "  3) 查看生成的网站"
echo "  4) 查看项目文档"
echo "  5) 显示验证结果"
echo "  6) 退出"
echo ""
read -p "请输入选择 (1-6): " choice

case $choice in
    1)
        echo ""
        echo -e "${BLUE}【执行】完整ML流程${NC}"
        echo "════════════════════════════════════════════════════════"
        $PYTHON training/run_css_enhanced_pipeline.py
        ;;
    2)
        echo ""
        echo -e "${BLUE}【执行】CSS学习模型训练${NC}"
        echo "════════════════════════════════════════════════════════"
        $PYTHON training/css_deep_learner.py
        ;;
    3)
        echo ""
        echo -e "${BLUE}【查看】生成的网站${NC}"
        echo "════════════════════════════════════════════════════════"
        if [ -d "generated_websites_enhanced" ]; then
            echo "生成的网站目录结构:"
            find generated_websites_enhanced -type f | head -20
            echo ""
            echo "网站位置:"
            for intent in ecommerce saas news; do
                if [ -f "generated_websites_enhanced/$intent/index.html" ]; then
                    echo -e "  ${GREEN}✓${NC} generated_websites_enhanced/$intent/index.html"
                fi
            done
        else
            echo -e "${YELLOW}未找到生成的网站，请先运行完整流程${NC}"
        fi
        ;;
    4)
        echo ""
        echo -e "${BLUE}【查看】项目文档${NC}"
        echo "════════════════════════════════════════════════════════"
        echo ""
        echo "推荐阅读顺序:"
        echo "  1. CSS_LEARNING_INDEX.md (总索引)"
        echo "  2. CSS_PROJECT_SUMMARY.md (项目总结)"
        echo "  3. CSS_LEARNING_QUICK_REF.md (快速参考)"
        echo ""
        echo "选择要查看的文档:"
        echo "  1) CSS_LEARNING_INDEX.md"
        echo "  2) CSS_PROJECT_SUMMARY.md"
        echo "  3) CSS_LEARNING_QUICK_REF.md"
        echo "  4) CSS_LEARNING_SUMMARY.md"
        echo "  5) CSS_IMPROVEMENT_GUIDE.md"
        echo ""
        read -p "请选择 (1-5): " doc_choice
        
        case $doc_choice in
            1) less CSS_LEARNING_INDEX.md ;;
            2) less CSS_PROJECT_SUMMARY.md ;;
            3) less CSS_LEARNING_QUICK_REF.md ;;
            4) less CSS_LEARNING_SUMMARY.md ;;
            5) less CSS_IMPROVEMENT_GUIDE.md ;;
        esac
        ;;
    5)
        echo ""
        echo -e "${BLUE}【显示】最近的验证结果${NC}"
        echo "════════════════════════════════════════════════════════"
        echo ""
        echo "验证统计 (来自最后一次运行):"
        echo ""
        echo "电商网站 (Ecommerce):"
        echo "  总体评分: 70.6/100 (C级)"
        echo "  - DOM相似度: 72.3% ✓"
        echo "  - JS功能一致: 70.0% ✓"
        echo "  - CSS样式一致: 51.5% ⚠️"
        echo "  - 网络请求匹配: 100.0% ✓"
        echo ""
        echo "SaaS平台:"
        echo "  总体评分: 69.0/100 (C级)"
        echo "  - DOM相似度: 77.2% ✓"
        echo "  - JS功能一致: 62.0% ○"
        echo "  - CSS样式一致: 49.0% ⚠️"
        echo "  - 网络请求匹配: 100.0% ✓"
        echo ""
        echo "新闻网站 (News):"
        echo "  总体评分: 52.6/100 (D级)"
        echo "  - DOM相似度: 75.2% ✓"
        echo "  - JS功能一致: 48.3% ○"
        echo "  - CSS样式一致: 32.3% ⚠️"
        echo "  - 网络请求匹配: 50.0% ○"
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "平均评分: 64.1/100"
        echo "CSS改进目标: 44% → 75%+"
        echo "参考: CSS_IMPROVEMENT_GUIDE.md"
        ;;
    6)
        echo "退出演示"
        exit 0
        ;;
    *)
        echo -e "${YELLOW}无效选择${NC}"
        exit 1
        ;;
esac

echo ""
echo "╔════════════════════════════════════════════════════════╗"
echo "║              演示完成！感谢使用                        ║"
echo "║         更多信息请查看 CSS_LEARNING_INDEX.md           ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""
