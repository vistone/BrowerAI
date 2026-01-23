#!/bin/bash
# 自动执行完整管道: 等待数据 → 验证 → 训练 → 评估

set -e

echo "🚀 BrowerAI 完整学习管道"
echo "================================"
echo ""

# 步骤1: 等待数据生成
echo "⏳ 第1步: 等待实时数据生成..."
timeout=1800
interval=10
elapsed=0

while [ $elapsed -lt $timeout ]; do
    if [ -f real_data/obfuscated_code/training_pairs.jsonl ]; then
        pairs=$(wc -l < real_data/obfuscated_code/training_pairs.jsonl)
        if [ $pairs -gt 100 ]; then
            echo "✅ 数据已生成: $pairs 个训练对"
            break
        fi
    fi
    sleep $interval
    elapsed=$((elapsed + interval))
    echo -n "."
done

if [ ! -f real_data/obfuscated_code/training_pairs.jsonl ]; then
    echo ""
    echo "❌ 数据生成失败或超时"
    exit 1
fi

pairs=$(wc -l < real_data/obfuscated_code/training_pairs.jsonl)
echo ""
echo "📊 数据验证:"
python3 << 'EOF'
import json
from collections import Counter
import os

data_file = 'real_data/obfuscated_code/training_pairs.jsonl'
pairs = []

with open(data_file, 'r') as f:
    for line in f:
        try:
            pairs.append(json.loads(line))
        except:
            pass

print(f"   总条数: {len(pairs)}")
print(f"   文件大小: {os.path.getsize(data_file) / 1024 / 1024:.1f}MB")

if pairs:
    # 检查数据完整性
    required_fields = ['original', 'obfuscated', 'obfuscator']
    valid_count = sum(1 for p in pairs if all(f in p for f in required_fields))
    print(f"   有效条数: {valid_count}")
    
    # 统计混淆器
    obfuscators = Counter([p.get('obfuscator', 'unknown') for p in pairs])
    print(f"   混淆器分布:")
    for obfuscator, count in obfuscators.most_common():
        print(f"      {obfuscator}: {count}")
    
    # 平均大小
    avg_original = sum(len(p['original']) for p in pairs) // len(pairs)
    avg_obfuscated = sum(len(p['obfuscated']) for p in pairs) // len(pairs)
    print(f"   平均大小:")
    print(f"      原始代码: {avg_original} bytes")
    print(f"      混淆代码: {avg_obfuscated} bytes")
    
    print()
    if len(pairs) >= 1000:
        print("✅ 数据质量满足训练要求 (≥1000条)")
    else:
        print(f"⚠️  数据量有限 ({len(pairs)} < 1000)")
EOF

echo ""
echo "✅ 第1步完成: 数据准备就绪"
echo ""

# 步骤2: 模型训练
echo "⏳ 第2步: 启动增强版GPU训练..."
echo ""

python3 training/enhanced_gpu_trainer.py

echo ""
echo "✅ 第2步完成: 模型训练"
echo ""

# 步骤3: 评估和结果
echo "📊 最终结果:"
if [ -f models/local/best_framework_detector.pt ]; then
    echo "✅ 最佳模型已保存: models/local/best_framework_detector.pt"
fi

if [ -f models/local/framework_detector_enhanced.pt ]; then
    echo "✅ 最终模型已保存: models/local/framework_detector_enhanced.pt"
fi

if [ -f models/local/framework_detector_enhanced.onnx ]; then
    echo "✅ ONNX模型已保存: models/local/framework_detector_enhanced.onnx"
fi

echo ""
echo "🎉 完整管道执行完成!"
echo ""
echo "下一步:"
echo "  1. 评估模型: python3 training/evaluate_model.py"
echo "  2. 使用模型: cargo run --release -- --ai"
echo "  3. 部署模型: cargo build --features ai --release"
