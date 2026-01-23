#!/bin/bash
# 启动高性能数据生成 + 自动训练管道

echo "🚀 BrowerAI 高性能学习管道 v2"
echo "================================"
echo ""

# 清理环境
echo "清理环境..."
rm -f real_data/obfuscated_code/training_pairs.jsonl
mkdir -p real_data/obfuscated_code

# 步骤1: 启动高性能混淆器(4线程)
echo "⏳ 步骤1: 启动高性能NPM混淆器 (4线程)..."
echo ""

timeout 2400 python3 training/fast_npm_obfuscator.py

if [ ! -f real_data/obfuscated_code/training_pairs.jsonl ]; then
    echo ""
    echo "❌ 数据生成失败!"
    exit 1
fi

pairs=$(wc -l < real_data/obfuscated_code/training_pairs.jsonl)
echo ""
echo "✅ 数据生成完成: $pairs 个训练对"
echo ""

# 步骤2: 数据验证
echo "📊 步骤2: 数据验证..."
python3 << 'EOF'
import json
from collections import Counter

pairs = []
with open('real_data/obfuscated_code/training_pairs.jsonl') as f:
    for line in f:
        try:
            pairs.append(json.loads(line))
        except:
            pass

print(f"   有效条数: {len(pairs)}")

if pairs:
    obfuscators = Counter([p.get('obfuscator') for p in pairs])
    print(f"   混淆器分布:")
    for obf, cnt in obfuscators.most_common():
        print(f"      {obf}: {cnt}")
    
    packages = Counter([p.get('package') for p in pairs])
    print(f"   前5个包:")
    for pkg, cnt in packages.most_common(5):
        print(f"      {pkg}: {cnt}")
    
    print()
    if len(pairs) >= 1000:
        print("✅ 数据质量满足训练要求!")
    else:
        print(f"⚠️  数据量有限 ({len(pairs)} < 1000)")
EOF

echo ""

# 步骤3: 模型训练
echo "🚀 步骤3: 启动GPU模型训练 (20 epochs)..."
echo ""

python3 training/enhanced_gpu_trainer.py

echo ""
echo "="*70
echo "✅ 完整管道执行完成!"
echo "="*70
echo ""
echo "生成的模型:"
ls -lh models/local/*.pt models/local/*.onnx 2>/dev/null | awk '{print "   " $9 " (" $5 ")"}'
echo ""
echo "下一步: python3 training/evaluate_model.py"
