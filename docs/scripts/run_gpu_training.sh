#!/bin/bash
# GPU框架检测模型训练启动脚本

PROJECT_DIR="/home/stone/BrowerAI"
cd "$PROJECT_DIR"

echo "=================================="
echo "🚀 GPU框架检测模型训练"
echo "=================================="

# 等待混淆数据准备
echo ""
echo "⏳ 等待混淆数据准备..."

while [ ! -f "real_data/obfuscated_code/training_pairs.jsonl" ]; do
    echo "  等待中... $(date)"
    sleep 5
done

PAIR_COUNT=$(wc -l < real_data/obfuscated_code/training_pairs.jsonl)
echo "✅ 混淆数据准备完成! ($PAIR_COUNT 对)"

# 验证数据大小
PAIR_SIZE=$(du -sh real_data/obfuscated_code | cut -f1)
echo "📊 数据大小: $PAIR_SIZE"

# 启动GPU训练
echo ""
echo "🤖 启动PyTorch GPU训练..."
echo "========================================"

python3 training/gpu_framework_detector.py \
    --epochs 10 \
    --batch-size 32 \
    --learning-rate 0.001 \
    --device cuda \
    --save-model models/local/framework_detector.onnx

# 检查训练结果
if [ $? -eq 0 ]; then
    echo ""
    echo "✨ 训练成功！"
    if [ -f "models/local/framework_detector.onnx" ]; then
        ONNX_SIZE=$(du -sh models/local/framework_detector.onnx | cut -f1)
        echo "✅ 模型已保存: models/local/framework_detector.onnx ($ONNX_SIZE)"
    fi
else
    echo "❌ 训练失败"
    exit 1
fi
