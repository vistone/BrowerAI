#!/bin/bash
# JS反混淆模型训练 + 导出脚本

set -e

cd "$(dirname "$0")"

echo "=========================================="
echo "JS反混淆模型训练 (Seq2Seq)"
echo "=========================================="
echo ""
echo "任务: 混淆JS → 清晰JS"
echo "数据: data/obfuscation_pairs.jsonl"
echo "模型: Encoder-Decoder LSTM"
echo "输出格式: ONNX (用于Rust集成)"
echo ""

# 检查数据
if [ ! -f "data/obfuscation_pairs.jsonl" ]; then
    echo "❌ 数据文件不存在: data/obfuscation_pairs.jsonl"
    echo "请先准备数据"
    exit 1
fi

DATA_COUNT=$(wc -l < data/obfuscation_pairs.jsonl)
echo "✅ 数据文件: $DATA_COUNT 个混淆/反混淆对"
echo ""

# 训练
echo "开始训练..."
python3 scripts/train_js_deobfuscator.py

# 导出ONNX
echo ""
echo "导出ONNX模型..."
python3 scripts/export_js_deobfuscator.py

echo ""
echo "=========================================="
echo "✅ 完成!"
echo "=========================================="
echo ""
echo "模型已保存到: models/local/js_deobfuscator_v1.onnx"
echo ""
echo "下一步集成到Rust:"
echo "1. 更新 models/model_config.toml"
echo "2. 在 src/parser/js.rs 中启用AI反混淆"
echo "3. 测试: cargo test --features ai"
