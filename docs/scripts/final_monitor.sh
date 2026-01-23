#!/bin/bash
# 最终进度监控 - 简洁版

echo ""
echo "🎯 BrowerAI 学习管道 - 最终监控"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 函数: 显示进度条
show_progress() {
    local current=$1
    local total=$2
    local width=30
    
    if [ $total -eq 0 ]; then
        local percent=0
    else
        local percent=$((current * 100 / total))
    fi
    
    local filled=$((percent * width / 100))
    local empty=$((width - filled))
    
    printf "%3d%% [" $percent
    printf "%${filled}s" | tr ' ' '█'
    printf "%${empty}s" | tr ' ' '░'
    echo "]"
}

# 检查日志
if [ ! -f simple_pipeline.log ]; then
    echo "❌ 日志文件不存在"
    exit 1
fi

# 提取最新状态
latest_line=$(tail -1 simple_pipeline.log)

# 检查混淆进度
if echo "$latest_line" | grep -q "已处理"; then
    # 提取数字
    processed=$(echo "$latest_line" | grep -o '\b[0-9]\+/2000' | head -1 | cut -d/ -f1)
    generated=$(echo "$latest_line" | grep -o '生成 [0-9]\+ 个' | grep -o '[0-9]\+')
    
    echo "📊 数据混淆阶段:"
    echo "   已处理文件: $processed/2000"
    show_progress "$processed" "2000"
    echo "   已生成对数: $generated"
    echo ""
    
    if [ "$processed" -ge 2000 ]; then
        echo "✅ 混淆完成! 等待验证和训练..."
    fi
fi

# 检查是否开始训练
if tail simple_pipeline.log | grep -q "Epoch"; then
    echo ""
    echo "🤖 模型训练阶段:"
    epoch=$(tail simple_pipeline.log | grep "Epoch" | tail -1 | grep -o 'Epoch [0-9]\+' | grep -o '[0-9]\+')
    echo "   当前轮次: $epoch/20"
    show_progress "$epoch" "20"
fi

# 检查最终模型
echo ""
echo "💾 生成的文件:"
if [ -f models/local/best_framework_detector.pt ]; then
    size=$(du -h models/local/best_framework_detector.pt | awk '{print $1}')
    echo "   ✅ best_framework_detector.pt ($size)"
fi

if [ -f models/local/framework_detector_enhanced.pt ]; then
    size=$(du -h models/local/framework_detector_enhanced.pt | awk '{print $1}')
    echo "   ✅ framework_detector_enhanced.pt ($size)"
fi

if [ -f real_data/obfuscated_code/training_pairs.jsonl ]; then
    size=$(du -h real_data/obfuscated_code/training_pairs.jsonl | awk '{print $1}')
    lines=$(wc -l < real_data/obfuscated_code/training_pairs.jsonl)
    echo "   ✅ training_pairs.jsonl ($size, $lines行)"
fi

echo ""
echo "⏱️  时间: $(date '+%H:%M:%S')"
echo ""
echo "后续: tail -f simple_pipeline.log | grep 'Epoch\\|Loss\\|准确率'"
