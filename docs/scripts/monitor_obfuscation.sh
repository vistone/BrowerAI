#!/bin/bash
# 监控npm_code_obfuscator进度

echo "⏳ 监控数据生成进度 (超时: 30分钟)..."
echo ""

count=0
max_iterations=180  # 30分钟 × 60秒 / 10秒 = 180次

while [ $count -lt $max_iterations ]; do
    if [ -f real_data/obfuscated_code/training_pairs.jsonl ]; then
        pairs=$(wc -l < real_data/obfuscated_code/training_pairs.jsonl)
        if [ $pairs -gt 0 ]; then
            echo "✅ 数据生成完成!"
            echo ""
            echo "📊 统计信息:"
            echo "   训练对数: $pairs"
            
            # 显示包的分布
            echo ""
            echo "📦 包分布 (前10):"
            python3 << 'EOF'
import json
from collections import Counter

pairs = []
with open('real_data/obfuscated_code/training_pairs.jsonl', 'r') as f:
    for line in f:
        try:
            pairs.append(json.loads(line))
        except:
            pass

if pairs:
    packages = Counter([p.get('package', 'unknown') for p in pairs])
    for pkg, cnt in packages.most_common(10):
        print(f"     {pkg}: {cnt}")
    
    print()
    print(f"   总大小: {sum(p.get('obfuscated_size', 0) for p in pairs) / 1024 / 1024:.1f}MB")
EOF
            
            exit 0
        fi
    fi
    
    # 显示进度
    echo -n "."
    sleep 10
    count=$((count + 1))
    
    # 每30个点换行
    if [ $((count % 30)) -eq 0 ]; then
        echo " $((count * 10))秒"
    fi
done

echo ""
echo "❌ 超时! 混淆器未完成 (超过30分钟)"
echo ""
echo "可能原因:"
echo "- Terser/UglifyJS处理缓慢"
echo "- 磁盘I/O瓶颈"
echo "- 内存压力"
echo ""
echo "检查日志:"
tail -50 npm_obfuscation.log
