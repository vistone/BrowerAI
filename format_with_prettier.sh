#!/bin/bash
input_file="$1"
output_file="$2"

# 尝试用 prettier 格式化
if command -v npx &> /dev/null; then
  npx prettier --write --parser babel "$input_file" 2>/dev/null && \
  cp "$input_file" "$output_file" && echo "✓ prettier 格式化成功" && exit 0
fi

# 如果 prettier 不可用，尝试用简单的方法
echo "! prettier 不可用，使用简单的换行格式化"
exit 1
