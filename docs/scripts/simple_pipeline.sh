#!/bin/bash
# 简化的快速管道 - 使用纯Python混淆器

echo "🚀 BrowerAI 简化学习管道"
echo "================================"
echo ""

# 清理环境
echo "清理旧数据..."
rm -f real_data/obfuscated_code/training_pairs.jsonl
mkdir -p real_data/obfuscated_code

# 步骤1: 纯Python混淆
echo "⏳ 步骤1: Python代码混淆 (无依赖)..."
echo ""

python3 training/python_code_obfuscator.py

if [ ! -f real_data/obfuscated_code/training_pairs.jsonl ]; then
    echo ""
    echo "❌ 数据生成失败!"
    exit 1
fi

pairs=$(wc -l < real_data/obfuscated_code/training_pairs.jsonl)
echo ""
echo "✅ 数据生成完成: $pairs 个训练对"
echo ""

# 步骤2: 验证数据
echo "📊 步骤2: 数据验证..."
python3 << 'EOF'
import json

pairs = []
with open('real_data/obfuscated_code/training_pairs.jsonl') as f:
    for line in f:
        try:
            pairs.append(json.loads(line))
        except:
            pass

if pairs:
    print(f"   有效条数: {len(pairs)}")
    
    obfuscators = {}
    for p in pairs:
        obf = p.get('obfuscator', 'unknown')
        obfuscators[obf] = obfuscators.get(obf, 0) + 1
    
    print(f"   混淆器分布:")
    for obf, cnt in obfuscators.items():
        print(f"      {obf}: {cnt}")
    
    print()
    if len(pairs) >= 1000:
        print("✅ 数据质量满足训练要求 (≥1000条)")
    else:
        print(f"⚠️  数据量有限 ({len(pairs)} < 1000)，但可以开始训练")
else:
    print("❌ 未生成任何训练对!")
    exit 1
EOF

echo ""

# 步骤3: GPU模型训练
if [ "$pairs" -gt 0 ]; then
    echo "🚀 步骤3: 启动GPU模型训练..."
    echo ""
    
    python3 training/enhanced_gpu_trainer.py
    
    echo ""
    echo "="*70
    echo "✅ 完整管道执行完成!"
    echo "="*70
    echo ""
    
    echo "生成的模型:"
    ls -lh models/local/*.pt models/local/*.onnx 2>/dev/null | awk '{print "   " $9 " (" $5 ")"}'
else
    echo "⚠️  跳过训练: 没有足够的数据"
fi

echo ""
echo "下一步: python3 training/evaluate_model.py"
