#!/bin/bash

# 自动监控阶段4，完成后自动启动阶段5

set -e

WORK_DIR="/home/stone/BrowerAI"
VENV_PYTHON="$WORK_DIR/.venv/bin/python"
LOG_FILE="$WORK_DIR/stage4_enhanced_training.log"
MODEL_DIR="$WORK_DIR/models/local"

log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 📊 $1"
}

log_success() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ $1"
}

log_warning() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ⚠️  $1"
}

# 启动实时监控进程
start_monitoring() {
    log_info "启动实时监控..."
    
    (
        while true; do
            if ! ps aux | grep -q "enhanced_gpu_trainer.py" || [ ! -f "$LOG_FILE" ]; then
                break
            fi
            
            # 显示GPU使用率
            if command -v nvidia-smi &> /dev/null; then
                echo ""
                echo "GPU 状态 $(date '+%H:%M:%S')"
                nvidia-smi --query-gpu=name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu --format=csv,noheader
            fi
            
            # 显示最后的日志行数
            line_count=$(wc -l < "$LOG_FILE" 2>/dev/null || echo "0")
            epoch_info=$(tail -1 "$LOG_FILE" 2>/dev/null | grep -o "Epoch.*" || echo "运行中...")
            echo "日志行数: $line_count | $epoch_info"
            
            sleep 30
        done
    ) &
    
    MONITOR_PID=$!
    log_success "监控进程启动 (PID: $MONITOR_PID)"
}

# 检查阶段4完成
check_stage4_complete() {
    if ! ps aux | grep -q "enhanced_gpu_trainer.py"; then
        # 进程结束，检查是否成功
        if [ -f "$LOG_FILE" ] && tail -1 "$LOG_FILE" | grep -q "已保存\|ONNX"; then
            return 0
        fi
    fi
    return 1
}

# 启动阶段5
launch_stage5() {
    log_success "阶段4完成！启动阶段5..."
    
    # 查找模型文件
    model_file=""
    if [ -f "$MODEL_DIR/framework_detector_enhanced.pt" ]; then
        model_file="$MODEL_DIR/framework_detector_enhanced.pt"
    elif [ -f "$MODEL_DIR/enhanced_final.pt" ]; then
        model_file="$MODEL_DIR/enhanced_final.pt"
    elif [ -f "$MODEL_DIR/comprehensive_best.pt" ]; then
        model_file="$MODEL_DIR/comprehensive_best.pt"
    fi
    
    if [ -z "$model_file" ]; then
        log_warning "未找到模型文件"
        return 1
    fi
    
    log_info "使用模型: $model_file"
    
    # 启动ONNX转换
    if [ -f "$WORK_DIR/training/convert_to_onnx.py" ]; then
        log_info "启动ONNX模型转换..."
        nohup "$VENV_PYTHON" "$WORK_DIR/training/convert_to_onnx.py" \
            --model "$model_file" \
            > "$WORK_DIR/stage5_onnx_export.log" 2>&1 &
        
        STAGE5_PID=$!
        log_success "阶段5已启动 (PID: $STAGE5_PID)"
        echo "$STAGE5_PID" > "$WORK_DIR/.stage5_pid"
    else
        log_warning "convert_to_onnx.py 不存在"
        return 1
    fi
}

# 主函数
main() {
    log_info "====== 自动监控和切换系统启动 ======"
    log_info "监控文件: $LOG_FILE"
    
    # 启动监控
    start_monitoring
    
    # 等待阶段4完成
    log_info "等待阶段4完成..."
    check_count=0
    while true; do
        if check_stage4_complete; then
            log_success "阶段4完成！"
            break
        fi
        
        check_count=$((check_count + 1))
        if [ $((check_count % 12)) -eq 0 ]; then  # 每6分钟输出一次
            log_info "阶段4运行中 (第 $((check_count * 5)) 秒)"
        fi
        
        sleep 5
    done
    
    # 启动阶段5
    if launch_stage5; then
        log_success "阶段5启动成功"
    else
        log_warning "阶段5启动失败"
    fi
    
    log_info "====== 监控完成 ======"
}

main "$@"
