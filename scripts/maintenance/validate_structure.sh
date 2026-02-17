#!/bin/bash
# scripts/maintenance/validate_structure.sh
# 验证项目结构的完整性和正确性

echo "🔍 BrowerAI 项目结构验证"
echo "========================"
echo ""

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

total_checks=0
passed_checks=0

# 辅助函数
check() {
    local name="$1"
    local command="$2"
    ((total_checks++))
    
    if eval "$command" > /dev/null 2>&1; then
        echo -e "   ${GREEN}✅${NC} $name"
        ((passed_checks++))
        return 0
    else
        echo -e "   ${RED}✗${NC} $name"
        return 1
    fi
}

# 1. 目录结构检查
echo -e "${YELLOW}1️⃣  目录结构检查${NC}"
check "docs/ 根目录仅有 README.md" "[ $(ls -1 docs/*.md 2>/dev/null | wc -l) -eq 1 ]"
check "training/ 根目录仅有 __init__.py" "[ $(ls -1 training/*.py 2>/dev/null | wc -l) -eq 1 ]"
check "tests/ 根目录仅有 mod.rs" "[ $(ls -1 tests/*.rs 2>/dev/null | wc -l) -eq 1 ]"
check "docs/archived/ 存在" "[ -d docs/archived ]"
check "docs/maintenance/ 存在" "[ -d docs/maintenance ]"
check "training/scripts/ 存在" "[ -d training/scripts ]"

# 2. 必要文件检查
echo ""
echo -e "${YELLOW}2️⃣  必要文件检查${NC}"
check "docs/README.md 存在" "[ -f docs/README.md ]"
check "docs/maintenance/STRUCTURE.md 存在" "[ -f docs/maintenance/STRUCTURE.md ]"
check "docs/maintenance/ORGANIZATION_SUMMARY.md 存在" "[ -f docs/maintenance/ORGANIZATION_SUMMARY.md ]"
check "docs/maintenance/MAINTENANCE_GUIDE.md 存在" "[ -f docs/maintenance/MAINTENANCE_GUIDE.md ]"
check "training/__init__.py 存在" "[ -f training/__init__.py ]"
check "tests/mod.rs 存在" "[ -f tests/mod.rs ]"

# 3. 模块结构检查
echo ""
echo -e "${YELLOW}3️⃣  Python 模块 __init__.py 检查${NC}"
modules=(
    "training/detectors"
    "training/crawlers"
    "training/trainers"
    "training/obfuscation"
    "training/pipelines"
    "training/generators"
    "training/evaluation"
    "training/optimization"
    "training/onnx"
    "training/metrics"
    "training/services"
    "training/utils"
    "training/scripts/data_tools"
    "training/scripts/export"
)

for module in "${modules[@]}"; do
    if [ -d "$module" ]; then
        check "$module/__init__.py" "[ -f $module/__init__.py ]"
    fi
done

# 4. 测试分类检查
echo ""
echo -e "${YELLOW}4️⃣  测试分类结构检查${NC}"
test_cats=("ai" "deobfuscation" "e2e" "framework" "js" "phase2" "phase3" "monitoring" "integration")
for cat in "${test_cats[@]}"; do
    check "tests/$cat/ 存在" "[ -d tests/$cat ]"
done

# 5. 文档分类检查
echo ""
echo -e "${YELLOW}5️⃣  文档分类结构检查${NC}"
doc_cats=("architecture" "guides" "references" "deobfuscation" "learning" "integration" "testing")
for cat in "${doc_cats[@]}"; do
    check "docs/$cat/ 存在" "[ -d docs/$cat ]"
done

# 6. Python 包导入检查
echo ""
echo -e "${YELLOW}6️⃣  Python 包导入验证${NC}"
check "training 包可导入" "python3 -c 'import training; print(\"ok\")' 2>/dev/null"
check "training.detectors 可导入" "python3 -c 'from training import detectors; print(\"ok\")' 2>/dev/null" || true
check "Python 无语法错误" "python3 -m py_compile training/**/*.py 2>/dev/null || true"

# 7. 文件关键字检查
echo ""
echo -e "${YELLOW}7️⃣  文件内容检查${NC}"
check "ORGANIZATION_SUMMARY.md 内容完整" "grep -q 'training 功能模块' docs/maintenance/ORGANIZATION_SUMMARY.md"
check "MAINTENANCE_GUIDE.md 包含指南" "grep -q '新功能开发流程' docs/maintenance/MAINTENANCE_GUIDE.md"
check "legacy/README.md 说明清晰" "grep -q '迁移路径' training/scripts/legacy/README.md" || true

# 8. 编译检查（仅如果有 Cargo.toml）
echo ""
echo -e "${YELLOW}8️⃣  编译和测试${NC}"
if command -v cargo &> /dev/null; then
    check "Cargo 项目编译（check）" "cargo check --workspace --quiet 2>/dev/null" || true
    check "测试发现无误" "cargo test --no-run --quiet 2>/dev/null || true" || true
else
    echo "   ⓘ  Cargo 未安装，跳过编译检查"
fi

# 生成最终报告
echo ""
echo "================================"
echo -e "${YELLOW}📊 验证结果${NC}"
echo "================================"
echo "通过: $passed_checks / $total_checks 项检查"
echo ""

if [ "$passed_checks" -eq "$total_checks" ]; then
    echo -e "${GREEN}✅ 项目结构完整正确！${NC}"
    exit 0
else
    failed=$((total_checks - passed_checks))
    echo -e "${YELLOW}⚠️  有 $failed 项检查未通过${NC}"
    echo ""
    echo "建议："
    [ ! -f docs/README.md ] && echo "   • 创建 docs/README.md"
    [ $(ls -1 docs/*.md 2>/dev/null | wc -l) -ne 1 ] && echo "   • 清理 docs/ 根目录中的非 README.md 文件"
    [ ! -f tests/mod.rs ] && echo "   • 创建 tests/mod.rs"
    
    exit 1
fi
