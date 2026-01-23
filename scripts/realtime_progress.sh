#!/bin/bash

# 全面学习进度总览脚本
# 实时监控所有5个阶段的进度

YELLOW='\033[1;33m'
GREEN='\033[1;32m'
BLUE='\033[1;34m'
RED='\033[1;31m'
CYAN='\033[1;36m'
NC='\033[0m'

clear

echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}   🎓 全面学习系统 - 实时进度总览${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo ""
echo "🕐 更新时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# ============================================================
# 阶段1：扩展NPM包采集
# ============================================================
echo -e "${CYAN}阶段1️⃣  - 数据扩展（NPM包采集）${NC}"
echo "─────────────────────────────────────────────────────────"

if pgrep -f "extended_npm_crawler.py" > /dev/null; then
    echo -e "${GREEN}✅ 正在运行${NC}"
    
    if [ -f "extended_npm_collection.log" ]; then
        COLLECTED=$(grep -c "downloaded" extended_npm_collection.log 2>/dev/null || echo "0")
        FAILED=$(grep -c "failed\|error" extended_npm_collection.log 2>/dev/null || echo "0")
        
        echo "   已采集: ${GREEN}${COLLECTED}个包${NC}"
        echo "   失败: ${RED}${FAILED}个${NC}"
        
        # 显示最新进度
        LATEST=$(tail -5 extended_npm_collection.log 2>/dev/null | grep -E "Downloaded|Extracting" | tail -1)
        if [ ! -z "$LATEST" ]; then
            echo "   最新: $LATEST"
        fi
    fi
else
    if [ -f "extended_npm_collection.log" ] && [ -s "extended_npm_collection.log" ]; then
        echo -e "${GREEN}✅ 已完成${NC}"
        COLLECTED=$(grep -c "downloaded" extended_npm_collection.log 2>/dev/null || echo "0")
        echo "   采集包数: ${GREEN}${COLLECTED}个${NC}"
    else
        echo -e "${YELLOW}⏳ 待执行${NC}"
        echo "   命令: python3 training/extended_npm_crawler.py"
    fi
fi
echo ""

# ============================================================
# 阶段2：数据增强
# ============================================================
echo -e "${CYAN}阶段2️⃣  - 数据增强（3倍混淆）${NC}"
echo "─────────────────────────────────────────────────────────"

if pgrep -f "fast_npm_obfuscator.py" > /dev/null; then
    echo -e "${GREEN}✅ 正在运行${NC}"
    
    if [ -f "data_augmentation.log" ]; then
        PROCESSED=$(grep -oP "Processed \K\d+" data_augmentation.log 2>/dev/null | tail -1 || echo "0")
        echo "   已处理: ${GREEN}${PROCESSED}个样本${NC}"
    fi
else
    if [ -f "real_data/obfuscated_code/augmented_training_pairs.jsonl" ] && [ -s "real_data/obfuscated_code/augmented_training_pairs.jsonl" ]; then
        AUGMENTED=$(wc -l < real_data/obfuscated_code/augmented_training_pairs.jsonl 2>/dev/null || echo "0")
        echo -e "${GREEN}✅ 已完成${NC}"
        echo "   增强样本: ${GREEN}${AUGMENTED}个${NC}"
    else
        echo -e "${YELLOW}⏳ 待执行${NC}"
        echo "   依赖: 阶段1完成"
    fi
fi
echo ""

# ============================================================
# 阶段3：大规模训练（50轮）
# ============================================================
echo -e "${CYAN}阶段3️⃣  - 大规模训练（50轮）${NC}"
echo "─────────────────────────────────────────────────────────"

if pgrep -f "large_scale_trainer.py" > /dev/null; then
    echo -e "${GREEN}✅ 正在运行${NC}"
    
    if [ -f "large_scale_training_50epochs.log" ]; then
        # 尝试提取最新的epoch信息
        CURRENT_EPOCH=$(grep -oP "Epoch \K\d+" large_scale_training_50epochs.log 2>/dev/null | tail -1 || echo "0")
        
        if [ ! -z "$CURRENT_EPOCH" ] && [ "$CURRENT_EPOCH" -gt 0 ]; then
            PERCENT=$((CURRENT_EPOCH * 100 / 50))
            echo "   进度: ${YELLOW}${CURRENT_EPOCH}/50 轮 (${PERCENT}%)${NC}"
            
            # 显示最新准确率
            LATEST_ACC=$(grep "准确率\|Accuracy" large_scale_training_50epochs.log 2>/dev/null | tail -1)
            if [ ! -z "$LATEST_ACC" ]; then
                echo "   最新: $LATEST_ACC"
            fi
        else
            echo "   状态: 初始化完成，训练中..."
        fi
    fi
    
    # GPU状态
    GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    if [ ! -z "$GPU_UTIL" ]; then
        echo "   GPU: ${CYAN}${GPU_UTIL}% 使用率${NC} | ${CYAN}${GPU_MEM}MB 显存${NC}"
    fi
else
    echo -e "${YELLOW}⏳ 进程未运行（可能已完成或出错）${NC}"
    
    if [ -f "large_scale_training_50epochs.log" ]; then
        FINAL_EPOCH=$(grep -oP "Epoch \K\d+" large_scale_training_50epochs.log 2>/dev/null | tail -1)
        if [ ! -z "$FINAL_EPOCH" ]; then
            if [ "$FINAL_EPOCH" -eq 50 ]; then
                echo -e "   ${GREEN}✅ 已完成 50 轮训练${NC}"
            else
                echo "   最后完成: Epoch $FINAL_EPOCH"
            fi
        fi
    fi
fi
echo ""

# ============================================================
# 阶段4：增强训练（30轮）
# ============================================================
echo -e "${CYAN}阶段4️⃣  - 增强训练（30轮）${NC}"
echo "─────────────────────────────────────────────────────────"

if pgrep -f "enhanced_gpu_trainer.py" > /dev/null; then
    echo -e "${GREEN}✅ 正在运行${NC}"
    
    if [ -f "enhanced_training_30epochs.log" ]; then
        CURRENT_EPOCH=$(grep -oP "Epoch \K\d+" enhanced_training_30epochs.log 2>/dev/null | tail -1 || echo "0")
        
        if [ ! -z "$CURRENT_EPOCH" ] && [ "$CURRENT_EPOCH" -gt 0 ]; then
            PERCENT=$((CURRENT_EPOCH * 100 / 30))
            echo "   进度: ${YELLOW}${CURRENT_EPOCH}/30 轮 (${PERCENT}%)${NC}"
        fi
    fi
else
    if [ -f "enhanced_training_30epochs.log" ] && [ -s "enhanced_training_30epochs.log" ]; then
        FINAL_EPOCH=$(grep -oP "Epoch \K\d+" enhanced_training_30epochs.log 2>/dev/null | tail -1)
        if [ ! -z "$FINAL_EPOCH" ] && [ "$FINAL_EPOCH" -eq 30 ]; then
            echo -e "${GREEN}✅ 已完成${NC}"
        else
            echo -e "${YELLOW}⏳ 进程结束（未完成30轮）${NC}"
        fi
    else
        echo -e "${YELLOW}⏳ 待执行${NC}"
        echo "   依赖: 阶段3完成"
    fi
fi
echo ""

# ============================================================
# 阶段5：模型导出与验证
# ============================================================
echo -e "${CYAN}阶段5️⃣  - 模型导出与验证${NC}"
echo "─────────────────────────────────────────────────────────"

if [ -f "models/local/comprehensive_framework_detector.onnx" ]; then
    ONNX_SIZE=$(du -h models/local/comprehensive_framework_detector.onnx 2>/dev/null | cut -f1)
    echo -e "${GREEN}✅ ONNX模型已导出${NC}"
    echo "   文件: comprehensive_framework_detector.onnx"
    echo "   大小: ${ONNX_SIZE}"
else
    echo -e "${YELLOW}⏳ 待执行${NC}"
    echo "   依赖: 阶段4完成"
fi
echo ""

# ============================================================
# 总体进度评估
# ============================================================
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}   📊 总体进度评估${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"

COMPLETED=0
TOTAL=5

# 计算已完成的阶段
[ -f "extended_npm_collection.log" ] && [ -s "extended_npm_collection.log" ] && ! pgrep -f "extended_npm_crawler.py" > /dev/null && COMPLETED=$((COMPLETED + 1))
[ -f "real_data/obfuscated_code/augmented_training_pairs.jsonl" ] && [ -s "real_data/obfuscated_code/augmented_training_pairs.jsonl" ] && COMPLETED=$((COMPLETED + 1))
[ -f "large_scale_training_50epochs.log" ] && grep -q "Epoch 50" large_scale_training_50epochs.log && COMPLETED=$((COMPLETED + 1))
[ -f "enhanced_training_30epochs.log" ] && grep -q "Epoch 30" enhanced_training_30epochs.log && COMPLETED=$((COMPLETED + 1))
[ -f "models/local/comprehensive_framework_detector.onnx" ] && COMPLETED=$((COMPLETED + 1))

PERCENT=$((COMPLETED * 100 / TOTAL))

echo ""
echo -e "完成阶段: ${GREEN}${COMPLETED}/${TOTAL}${NC} (${PERCENT}%)"
echo ""

if [ $COMPLETED -eq $TOTAL ]; then
    echo -e "${GREEN}✅ 全面学习完成！${NC}"
elif [ $COMPLETED -eq 0 ]; then
    echo -e "${YELLOW}⏳ 准备启动${NC}"
elif [ $COMPLETED -lt 3 ]; then
    echo -e "${BLUE}🔄 数据准备阶段${NC}"
else
    echo -e "${BLUE}🔄 模型训练阶段${NC}"
fi

echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
