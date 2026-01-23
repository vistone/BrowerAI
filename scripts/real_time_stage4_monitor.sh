#!/bin/bash

# 实时监控阶段4训练进度

clear

while true; do
    echo "====== 阶段4 实时监控 ($(date '+%Y-%m-%d %H:%M:%S')) ======"
    echo ""
    
    # 检查进程
    if ps aux | grep -q "enhanced_gpu_trainer.py" && ! ps aux | grep -q "[g]rep"; then
        echo "✅ 阶段4进程运行中 (PID: $(ps aux | grep enhanced_gpu_trainer | grep -v grep | awk '{print $2}'))"
    else
        echo "⚠️  阶段4进程已结束"
    fi
    
    echo ""
    echo "📊 GPU 使用情况:"
    nvidia-smi --query-gpu=index,name,driver_version,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu --format=csv,noheader 2>/dev/null | while IFS=',' read -r idx name driver gpu_util mem_util mem_used mem_total temp; do
        echo "   GPU$idx: GPU利用率=${gpu_util} 内存=${mem_used}/${mem_total} 温度=${temp}"
    done
    
    echo ""
    echo "📈 训练进度:"
    log_file="/home/stone/BrowerAI/stage4_enhanced_training.log"
    if [ -f "$log_file" ]; then
        line_count=$(wc -l < "$log_file")
        echo "   日志行数: $line_count"
        
        # 显示最后5行日志
        echo ""
        echo "   最后的训练输出:"
        tail -5 "$log_file" | sed 's/^/     /'
    fi
    
    echo ""
    echo "💾 模型文件检查:"
    if [ -f "/home/stone/BrowerAI/models/local/framework_detector_enhanced.pt" ]; then
        size=$(ls -lh /home/stone/BrowerAI/models/local/framework_detector_enhanced.pt | awk '{print $5}')
        echo "   ✅ framework_detector_enhanced.pt ($size)"
    fi
    
    echo ""
    echo "按 Ctrl+C 退出监控"
    echo "========================================"
    echo ""
    
    sleep 15
done
