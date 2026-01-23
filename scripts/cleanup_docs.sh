#!/bin/bash
# 文档清理脚本 - 删除过期的markdown文件

cd /home/stone/BrowerAI

echo "=== 文档清理开始 ==="
echo ""

# 需要保留的文件
KEEP_FILES=(
  "README.md"
  "CONTRIBUTING.md"
  "CHANGELOG.md"
  "Cargo.toml"
)

# 需要删除的文件模式
DELETE_PATTERNS=(
  "BULK_CSS_*"
  "CSS_LEARNING_*"
  "CSS_*"
  "DUAL_SANDBOX_*"
  "COMPLETE_*_GUIDE.md"
  "ANALYSIS_*"
  "COMPREHENSIVE_*"
  "LARGE_SCALE_*"
  "INTENT_OPTIMIZATION_*"
  "DATA_QUALITY_*"
  "ECOMMERCE_*"
  "EXPERIMENT_*"
  "IMPLEMENTATION_*"
  "INFERENCE_*"
  "COMPILATION_*"
  "COMPARATIVE_*"
  "CRATES_IO_*"
  "DEPLOYMENT_GUIDE.md"
  "DESIGN_*"
  "DOCUMENTATION_*"
  "EXECUTION_*"
  "FINAL_*"
  "FOLLOWUP_*"
  "GETTING_STARTED.md"
  "GLOBAL_DEPLOYMENT_*"
  "*_COMPLETE.md"
  "*_SUMMARY.md"
  "*_REPORT.md"
  "*_CHECKLIST.md"
  "*_CERTIFICATE.txt"
  "*_SUMMARY_*.md"
  "*_GUIDE.md"
  "*_QUICKSTART.md"
)

echo "扫描需要删除的文件..."
DELETE_COUNT=0

for pattern in "${DELETE_PATTERNS[@]}"; do
  for file in $pattern 2>/dev/null; do
    if [ -f "$file" ] && [ "$file" != "README.md" ] && [ "$file" != "CONTRIBUTING.md" ] && [ "$file" != "CHANGELOG.md" ]; then
      echo "删除: $file"
      rm -f "$file"
      ((DELETE_COUNT++))
    fi
  done
done

echo ""
echo "✅ 清理完成: 删除了 $DELETE_COUNT 个文件"
echo ""
echo "保留的文件:"
ls -1 *.md 2>/dev/null | head -20

echo ""
echo "剩余markdown文件数:"
find . -maxdepth 1 -name "*.md" -type f | wc -l
