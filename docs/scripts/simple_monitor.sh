#!/bin/bash
# 简单的实时进度监控

while true; do
    clear
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📊 BrowerAI 学习进度"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "⏱️  时间: $(date '+%H:%M:%S')"
    echo ""
    
    # 数据生成
    if [ -f real_data/obfuscated_code/training_pairs.jsonl ]; then
        pairs=$(wc -l < real_data/obfuscated_code/training_pairs.jsonl)
        size=$(du -h real_data/obfuscated_code/training_pairs.jsonl | awk '{print $1}')
        echo "✅ 数据生成: $pairs 行 ($size)"
    else
        echo "⏳ 数据生成: 准备中..."
    fi
    
    # 模型
    if [ -f models/local/framework_detector_enhanced.pt ]; then
        size=$(du -h models/local/framework_detector_enhanced.pt | awk '{print $1}')
        echo "✅ 模型训练: 完成 ($size)"
    else
        echo "⏳ 模型训练: 准备中..."
    fi
    
    echo ""
    echo "最近日志:"
    if [ -f fast_pipeline.log ]; then
        tail -5 fast_pipeline.log | sed 's/^/  /'
    fi
    
    echo ""
    echo "按 Ctrl+C 停止 | 每10秒刷新"
    sleep 10
done
