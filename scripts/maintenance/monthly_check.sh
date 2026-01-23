#!/bin/bash
# scripts/maintenance/monthly_check.sh
# 月度项目结构清理检查

set -e

echo "📋 BrowerAI 月度项目维护检查"
echo "=============================="
echo ""
echo "检查时间: $(date)"
echo ""

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 1. 检查文档结构
echo -e "${YELLOW}1️⃣  文档结构检查${NC}"
echo "   docs/ 根目录 .md 文件数（应仅有 README.md）:"
doc_count=$(ls -1 docs/*.md 2>/dev/null | wc -l)
echo "   📄 ${doc_count} 个"
if [ "$doc_count" -eq 1 ]; then
    echo -e "   ${GREEN}✅ 正常${NC}"
else
    echo -e "   ${RED}⚠️  异常，请检查${NC}"
fi

echo ""
echo "   docs/archived/ 文档统计:"
archived_count=$(find docs/archived -name "*.md" -type f | wc -l)
echo "   📄 ${archived_count} 个文档"
if [ "$archived_count" -gt 15 ]; then
    echo -e "   ${YELLOW}💡 建议: 文档数量较多，考虑是否有可以删除的${NC}"
fi

echo ""
echo "   docs/maintenance/ 文件统计:"
maintenance_count=$(find docs/maintenance -name "*.md" -type f | wc -l)
echo "   📄 ${maintenance_count} 个文档"

# 2. 检查 Python 模块结构
echo ""
echo -e "${YELLOW}2️⃣  Python 模块结构检查${NC}"
echo "   training/ 模块总数:"
module_count=$(ls -d training/*/ 2>/dev/null | grep -v __pycache__ | wc -l)
echo "   📦 ${module_count} 个模块"

echo ""
echo "   检查各模块的 __init__.py:"
missing_init=0
for dir in training/detectors training/crawlers training/trainers training/obfuscation training/pipelines training/generators training/evaluation training/optimization training/onnx training/metrics training/services training/utils training/scripts/data_tools training/scripts/export; do
    if [ -d "$dir" ]; then
        if [ ! -f "$dir/__init__.py" ]; then
            echo -e "   ${RED}✗ $dir 缺少 __init__.py${NC}"
            ((missing_init++))
        fi
    fi
done
if [ $missing_init -eq 0 ]; then
    echo -e "   ${GREEN}✅ 所有模块都有 __init__.py${NC}"
else
    echo -e "   ${RED}⚠️  找到 $missing_init 个缺失的 __init__.py${NC}"
fi

# 3. 检查训练脚本
echo ""
echo -e "${YELLOW}3️⃣  训练脚本统计${NC}"
root_py=$(ls -1 training/*.py 2>/dev/null | wc -l)
echo "   training/ 根目录 .py 文件:"
echo "   📄 ${root_py} 个"
if [ "$root_py" -eq 1 ]; then
    echo -e "   ${GREEN}✅ 正常（仅 __init__.py）${NC}"
else
    echo -e "   ${RED}⚠️  异常，应仅有 __init__.py${NC}"
fi

echo ""
echo "   training/scripts/legacy/ 脚本统计:"
legacy_count=$(ls -1 training/scripts/legacy/*.py 2>/dev/null | wc -l)
echo "   📄 ${legacy_count} 个脚本"
if [ "$legacy_count" -gt 0 ]; then
    echo -e "   ${YELLOW}💡 建议: 检查这些脚本是否真正需要保留${NC}"
fi

# 4. 检查测试结构
echo ""
echo -e "${YELLOW}4️⃣  测试分类检查${NC}"
test_categories=$(ls -d tests/*/ 2>/dev/null | wc -l)
echo "   测试分类数: ${test_categories} 个"
echo ""
echo "   各分类测试数量:"
for dir in tests/*/; do
    dir_name=$(basename "$dir")
    file_count=$(find "$dir" -name "*tests.rs" -o -name "test_*.rs" | wc -l)
    printf "   ├─ %-20s: %2d 个文件\n" "$dir_name" "$file_count"
done

# 5. 检查文件大小
echo ""
echo -e "${YELLOW}5️⃣  项目体积统计${NC}"
echo "   主要目录大小:"
du -sh training/ docs/ tests/ crates/ 2>/dev/null | sort -h | sed 's/^/   ├─ /'

# 6. 检查 Python 语法
echo ""
echo -e "${YELLOW}6️⃣  Python 语法检查${NC}"
py_errors=$(python3 -m py_compile training/**/*.py 2>&1 | wc -l)
if [ "$py_errors" -eq 0 ]; then
    echo -e "   ${GREEN}✅ 所有 Python 文件语法正确${NC}"
else
    echo -e "   ${YELLOW}⚠️  发现 $py_errors 个语法问题${NC}"
fi

# 7. 生成建议
echo ""
echo -e "${YELLOW}7️⃣  维护建议${NC}"
echo ""

recommendations=0

if [ "$doc_count" -ne 1 ]; then
    echo -e "   ${RED}❌ docs/ 根目录有非 README.md 的文件，需要清理${NC}"
    ((recommendations++))
fi

if [ "$archived_count" -gt 15 ]; then
    echo -e "   ${YELLOW}💡 docs/archived/ 文档较多（${archived_count}个），可考虑清理${NC}"
    ((recommendations++))
fi

if [ "$legacy_count" -gt 0 ]; then
    echo -e "   ${YELLOW}💡 training/scripts/legacy/ 有 ${legacy_count} 个脚本，确认是否真正需要${NC}"
    ((recommendations++))
fi

if [ "$missing_init" -gt 0 ]; then
    echo -e "   ${RED}❌ 有 ${missing_init} 个模块缺少 __init__.py${NC}"
    ((recommendations++))
fi

if [ "$recommendations" -eq 0 ]; then
    echo -e "   ${GREEN}✅ 项目结构良好，无特殊建议${NC}"
fi

# 8. 生成报告
echo ""
echo -e "${YELLOW}📊 维护报告摘要${NC}"
echo "   ├─ 文档：${doc_count} 个根文件 + ${archived_count} 个归档文件"
echo "   ├─ Python：${module_count} 个模块，${legacy_count} 个遗留脚本"
echo "   ├─ 测试：${test_categories} 个分类"
echo "   └─ 结构状态：$([ $recommendations -eq 0 ] && echo '✅ 良好' || echo '⚠️  需注意')"

echo ""
echo "✅ 月度检查完成！"
echo ""
echo "📝 详细说明请查看: docs/maintenance/MAINTENANCE_GUIDE.md"
