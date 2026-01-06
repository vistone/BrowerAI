#!/bin/bash
# 一键导出PyTorch模型到ONNX

set -e

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║         PyTorch → ONNX 模型导出                               ║"
echo "║         Export trained model for Rust integration              ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# 配置
CHECKPOINT="${1:-checkpoints/incremental/latest.pt}"
OUTPUT="${2:-../models/local/website_learner_v1.onnx}"

echo "📋 配置:"
echo "  - Checkpoint: $CHECKPOINT"
echo "  - 输出: $OUTPUT"
echo ""

if [ ! -f "$CHECKPOINT" ]; then
    echo "❌ 错误: Checkpoint文件不存在: $CHECKPOINT"
    echo "   请先运行增量学习生成模型"
    exit 1
fi

echo "🚀 开始导出..."
echo ""

python3 scripts/export_to_onnx.py \
    --checkpoint "$CHECKPOINT" \
    --output "$OUTPUT" \
    --vocab-size 10000 \
    --max-seq-length 512

echo ""
echo "✅ 导出完成!"
echo ""
echo "📦 下一步:"
echo "  1. 检查输出文件: $OUTPUT"
echo "  2. 更新 ../models/model_config.toml"
echo "  3. 在Rust代码中使用此模型"
echo ""
