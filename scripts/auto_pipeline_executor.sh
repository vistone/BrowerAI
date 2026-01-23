#!/bin/bash

# 全自动管道执行脚本：监控阶段2，自动启动阶段4和阶段5

set -e

WORK_DIR="/home/stone/BrowerAI"
VENV_PYTHON="$WORK_DIR/.venv/bin/python"
LOG_DIR="$WORK_DIR"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] ✅${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] ⚠️${NC} $1"
}

log_error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ❌${NC} $1"
}

# 函数：检查阶段2是否完成
check_stage2_complete() {
    local output_file="$WORK_DIR/real_data/obfuscated_code/augmented_training_pairs.jsonl"
    local log_file="$WORK_DIR/data_augmentation.log"
    
    if [ ! -f "$output_file" ]; then
        return 1
    fi
    
    # 检查日志是否显示完成（最后一行通常包含总数统计）
    if tail -1 "$log_file" 2>/dev/null | grep -q "3175"; then
        return 0
    fi
    
    # 备选检查：如果输出文件大小稳定（10秒内没有变化）
    local current_size=$(wc -l < "$output_file" 2>/dev/null || echo 0)
    sleep 5
    local new_size=$(wc -l < "$output_file" 2>/dev/null || echo 0)
    
    if [ "$current_size" -eq "$new_size" ] && [ "$current_size" -gt 5000 ]; then
        return 0
    fi
    
    return 1
}

# 函数：启动阶段4增强训练
launch_stage4() {
    log_info "🚀 启动阶段4：增强GPU训练..."
    
    cd "$WORK_DIR"
    
    # 检查模型文件
    if [ ! -f "$WORK_DIR/models/local/large_scale_best.pt" ]; then
        log_error "找不到阶段3模型文件: large_scale_best.pt"
        return 1
    fi
    
    # 检查数据文件
    if [ ! -f "$WORK_DIR/real_data/obfuscated_code/augmented_training_pairs.jsonl" ]; then
        log_error "找不到阶段2输出: augmented_training_pairs.jsonl"
        return 1
    fi
    
    local output_file="$WORK_DIR/real_data/obfuscated_code/augmented_training_pairs.jsonl"
    local line_count=$(wc -l < "$output_file")
    log_success "数据准备完毕: $line_count 个训练对"
    
    # 启动阶段4（后台运行）
    if [ -f "$WORK_DIR/training/enhanced_gpu_trainer.py" ]; then
        log_info "启动增强训练脚本..."
        nohup "$VENV_PYTHON" "$WORK_DIR/training/enhanced_gpu_trainer.py" \
            --epochs 30 \
            --batch-size 64 \
            --device cuda \
            --augmentation strong \
            --data-file "$output_file" \
            > "$LOG_DIR/stage4_enhanced_training.log" 2>&1 &
        
        local pid=$!
        log_success "阶段4已启动 (PID: $pid)"
        echo "$pid" > "$WORK_DIR/.stage4_pid"
        
        sleep 3
        if ps -p $pid > /dev/null 2>&1; then
            log_success "进程确认运行中"
            return 0
        else
            log_error "进程启动失败，检查日志: $LOG_DIR/stage4_enhanced_training.log"
            return 1
        fi
    else
        log_warning "enhanced_gpu_trainer.py 不存在，跳过阶段4"
        return 1
    fi
}

# 函数：启动阶段5 ONNX导出
launch_stage5() {
    log_info "🚀 启动阶段5：ONNX模型导出..."
    
    cd "$WORK_DIR"
    
    # 等待阶段4完成（检查模型文件）
    log_info "等待阶段4完成，检查模型输出..."
    
    local max_wait=300  # 5分钟超时
    local elapsed=0
    
    while [ $elapsed -lt $max_wait ]; do
        if [ -f "$WORK_DIR/models/local/comprehensive_best.pt" ] || \
           [ -f "$WORK_DIR/models/local/enhanced_final.pt" ]; then
            log_success "阶段4模型文件已生成，开始ONNX导出..."
            break
        fi
        
        elapsed=$((elapsed + 10))
        echo -n "."
        sleep 10
    done
    
    # 启动ONNX转换
    local model_file=""
    if [ -f "$WORK_DIR/models/local/comprehensive_best.pt" ]; then
        model_file="$WORK_DIR/models/local/comprehensive_best.pt"
    elif [ -f "$WORK_DIR/models/local/enhanced_final.pt" ]; then
        model_file="$WORK_DIR/models/local/enhanced_final.pt"
    fi
    
    if [ -z "$model_file" ]; then
        log_error "找不到可用的模型文件进行ONNX转换"
        return 1
    fi
    
    log_info "使用模型: $model_file"
    
    if [ -f "$WORK_DIR/training/convert_to_onnx.py" ]; then
        nohup "$VENV_PYTHON" "$WORK_DIR/training/convert_to_onnx.py" \
            --model "$model_file" \
            > "$LOG_DIR/stage5_onnx_export.log" 2>&1 &
        
        local pid=$!
        log_success "阶段5已启动 (PID: $pid)"
        echo "$pid" > "$WORK_DIR/.stage5_pid"
    else
        log_warning "convert_to_onnx.py 不存在"
        return 1
    fi
}

# 主监控循环
main() {
    log_info "====== 全自动管道执行系统启动 ======"
    log_info "工作目录: $WORK_DIR"
    log_info "Python: $VENV_PYTHON"
    
    # 检查阶段2状态
    log_info "检查阶段2状态..."
    if check_stage2_complete; then
        log_success "阶段2已完成 ✅"
        
        # 启动阶段4
        if launch_stage4; then
            log_success "阶段4启动成功"
            
            # 启动实时监控
            log_info "启动阶段4实时监控..."
            nohup watch -n 30 "tail -20 $LOG_DIR/stage4_enhanced_training.log; echo '---'; nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader" > /dev/null 2>&1 &
            
            # 自动启动阶段5（在后台持续检查）
            log_info "等待阶段4完成后自动启动阶段5..."
            (
                while true; do
                    if [ -f "$WORK_DIR/models/local/comprehensive_best.pt" ] || \
                       [ -f "$WORK_DIR/models/local/enhanced_final.pt" ]; then
                        sleep 10  # 确保模型完全保存
                        launch_stage5
                        break
                    fi
                    sleep 30
                done
            ) &
            
            log_success "自动执行流程已就绪，后台监控中..."
        else
            log_error "阶段4启动失败"
            exit 1
        fi
    else
        log_warning "阶段2仍在运行，将持续监控..."
        
        # 持续监控阶段2
        while ! check_stage2_complete; do
            local current_lines=$(wc -l < "$WORK_DIR/real_data/obfuscated_code/augmented_training_pairs.jsonl" 2>/dev/null || echo "0")
            log_info "阶段2进度: $current_lines 行数据已生成"
            sleep 10
        done
        
        log_success "阶段2完成！启动阶段4..."
        launch_stage4
    fi
    
    log_info "====== 执行流程已启动，系统进入监控模式 ======"
}

# 执行主函数
main "$@"
