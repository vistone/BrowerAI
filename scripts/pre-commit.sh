#!/usr/bin/env bash
# Pre-commit hook for BrowerAI
set -euo pipefail

echo "Running pre-commit checks..."

# 基本检查：确保没有意外添加的大文件
if git diff --cached --name-only | grep -qE '\.(onnx|pt|pth|pkl)$'; then
    echo "❌ 错误：检测到模型文件，这些文件不应提交到 git"
    echo "请使用 git reset HEAD <file> 取消暂存这些文件"
    exit 1
fi

echo "✓ Pre-commit 检查通过"
exit 0
