#!/bin/bash

# 全面学习进度监控脚本
# 用于跟踪多阶段训练的执行情况

set -e

YELLOW='\033[1;33m'
GREEN='\033[1;32m'
BLUE='\033[1;34m'
RED='\033[1;31m'
NC='\033[0m'

echo -e "${BLUE}══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}   🎓 全面学习进度监控系统${NC}"
echo -e "${BLUE}══════════════════════════════════════════════════════════════${NC}"
echo ""

# 阶段1：数据扩展
echo -e "${YELLOW}阶段1️⃣ - 数据扩展${NC}"
if [ -f "real_data/npm_packages_extended" ]; then
    PACKAGE_COUNT=$(find real_data/npm_packages_extended -name "package.json" | wc -l)
    echo -e "  ✅ 扩展NPM包：${GREEN}${PACKAGE_COUNT}个${NC}"
else
    echo -e "  ⏳ 扩展NPM包：${YELLOW}待执行${NC}"
    echo -e "  💡 命令：${BLUE}python3 training/extended_npm_crawler.py${NC}"
fi

# 当前数据统计
if [ -f "real_data/obfuscated_code/training_pairs.jsonl" ]; then
    CURRENT_SAMPLES=$(wc -l < real_data/obfuscated_code/training_pairs.jsonl)
    echo -e "  📊 当前样本数：${GREEN}${CURRENT_SAMPLES}${NC}"
else
    echo -e "  ❌ 训练数据不存在"
fi
echo ""

# 阶段2：数据增强
echo -e "${YELLOW}阶段2️⃣ - 数据增强${NC}"
if [ -f "real_data/obfuscated_code/augmented_training_pairs.jsonl" ]; then
    AUGMENTED_SAMPLES=$(wc -l < real_data/obfuscated_code/augmented_training_pairs.jsonl)
    echo -e "  ✅ 增强样本：${GREEN}${AUGMENTED_SAMPLES}个${NC}"
    MULTIPLIER=$((AUGMENTED_SAMPLES / CURRENT_SAMPLES))
    echo -e "  📈 增强倍数：${GREEN}${MULTIPLIER}x${NC}"
else
    echo -e "  ⏳ 数据增强：${YELLOW}待执行${NC}"
    echo -e "  💡 命令：${BLUE}python3 training/fast_npm_obfuscator.py --methods all --multiplier 3${NC}"
fi
echo ""

# 阶段3：大规模训练（50轮）
echo -e "${YELLOW}阶段3️⃣ - 大规模训练（50轮）${NC}"
if [ -f "large_scale_training_50epochs.log" ]; then
    CURRENT_EPOCH=$(grep -oP "Epoch \K\d+" large_scale_training_50epochs.log | tail -1 || echo "0")
    TOTAL_EPOCHS=50
    
    if [ -z "$CURRENT_EPOCH" ]; then
        CURRENT_EPOCH=0
    fi
    
    if [ "$CURRENT_EPOCH" -eq "$TOTAL_EPOCHS" ]; then
        echo -e "  ✅ 训练完成：${GREEN}${CURRENT_EPOCH}/${TOTAL_EPOCHS} 轮${NC}"
        
        # 提取最终准确率
        FINAL_ACC=$(grep "准确率" large_scale_training_50epochs.log | tail -1 | grep -oP "\d+\.\d+%" || echo "N/A")
        echo -e "  🎯 最终准确率：${GREEN}${FINAL_ACC}${NC}"
    elif [ "$CURRENT_EPOCH" -gt 0 ]; then
        PERCENT=$((CURRENT_EPOCH * 100 / TOTAL_EPOCHS))
        echo -e "  🔄 训练进行中：${YELLOW}${CURRENT_EPOCH}/${TOTAL_EPOCHS} 轮 (${PERCENT}%)${NC}"
        
        # 提取当前准确率和损失
        CURRENT_ACC=$(grep "准确率" large_scale_training_50epochs.log | tail -1 | grep -oP "\d+\.\d+%" || echo "N/A")
        CURRENT_LOSS=$(grep "Loss" large_scale_training_50epochs.log | tail -1 | grep -oP "Loss: \K[\d\.]+" || echo "N/A")
        
        echo -e "  📊 当前准确率：${BLUE}${CURRENT_ACC}${NC}"
        echo -e "  📉 当前损失：${BLUE}${CURRENT_LOSS}${NC}"
        
        # 最近3轮的准确率趋势
        echo -e "  📈 最近趋势："
        grep "准确率" large_scale_training_50epochs.log | tail -3 | while read -r line; do
            echo -e "     ${line}"
        done
    else
        echo -e "  🚀 训练启动中：${BLUE}初始化完成${NC}"
    fi
    
    # 检查GPU使用情况
    if command -v nvidia-smi &> /dev/null; then
        GPU_USAGE=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)
        GPU_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
        echo -e "  🖥️  GPU使用率：${BLUE}${GPU_USAGE}%${NC}"
        echo -e "  💾 GPU显存：${BLUE}${GPU_MEM}MB${NC}"
    fi
else
    echo -e "  ⏳ 大规模训练：${YELLOW}待启动${NC}"
    echo -e "  💡 命令：${BLUE}python3 training/large_scale_trainer.py --epochs 50 --batch-size 64 --device cuda${NC}"
fi
echo ""

# 阶段4：增强训练（30轮）
echo -e "${YELLOW}阶段4️⃣ - 增强训练（30轮）${NC}"
if [ -f "enhanced_training_30epochs.log" ]; then
    ENHANCED_EPOCH=$(grep -oP "Epoch \K\d+" enhanced_training_30epochs.log | tail -1 || echo "0")
    ENHANCED_TOTAL=30
    
    if [ "$ENHANCED_EPOCH" -eq "$ENHANCED_TOTAL" ]; then
        echo -e "  ✅ 增强训练完成：${GREEN}${ENHANCED_EPOCH}/${ENHANCED_TOTAL} 轮${NC}"
    elif [ "$ENHANCED_EPOCH" -gt 0 ]; then
        echo -e "  🔄 增强训练中：${YELLOW}${ENHANCED_EPOCH}/${ENHANCED_TOTAL} 轮${NC}"
    else
        echo -e "  🚀 增强训练启动中${NC}"
    fi
else
    echo -e "  ⏳ 增强训练：${YELLOW}待执行${NC}"
    echo -e "  💡 依赖：阶段3完成后执行"
    echo -e "  💡 命令：${BLUE}python3 training/enhanced_gpu_trainer.py --epochs 30 --augmentation strong --device cuda${NC}"
fi
echo ""

# 阶段5：模型导出与验证
echo -e "${YELLOW}阶段5️⃣ - 模型导出与验证${NC}"
if [ -f "models/local/comprehensive_framework_detector.onnx" ]; then
    ONNX_SIZE=$(du -h models/local/comprehensive_framework_detector.onnx | cut -f1)
    echo -e "  ✅ ONNX模型：${GREEN}${ONNX_SIZE}${NC}"
    
    # 检查Rust集成测试
    if cargo test --test ai_integration_tests &> /dev/null; then
        echo -e "  ✅ Rust集成测试：${GREEN}通过${NC}"
    else
        echo -e "  ⚠️  Rust集成测试：${YELLOW}需要更新${NC}"
    fi
else
    echo -e "  ⏳ ONNX导出：${YELLOW}待执行${NC}"
    echo -e "  💡 依赖：阶段4完成后执行"
    echo -e "  💡 命令：${BLUE}python3 training/convert_to_onnx.py --model models/local/comprehensive_best.pt${NC}"
fi
echo ""

# 总体进度评估
echo -e "${BLUE}══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}   📊 总体进度评估${NC}"
echo -e "${BLUE}══════════════════════════════════════════════════════════════${NC}"

COMPLETED_STAGES=0
TOTAL_STAGES=5

# 计算完成的阶段
[ -d "real_data/npm_packages_extended" ] && COMPLETED_STAGES=$((COMPLETED_STAGES + 1))
[ -f "real_data/obfuscated_code/augmented_training_pairs.jsonl" ] && COMPLETED_STAGES=$((COMPLETED_STAGES + 1))
[ -f "large_scale_training_50epochs.log" ] && grep -q "Epoch 50" large_scale_training_50epochs.log && COMPLETED_STAGES=$((COMPLETED_STAGES + 1))
[ -f "enhanced_training_30epochs.log" ] && grep -q "Epoch 30" enhanced_training_30epochs.log && COMPLETED_STAGES=$((COMPLETED_STAGES + 1))
[ -f "models/local/comprehensive_framework_detector.onnx" ] && COMPLETED_STAGES=$((COMPLETED_STAGES + 1))

OVERALL_PERCENT=$((COMPLETED_STAGES * 100 / TOTAL_STAGES))

echo -e "完成阶段：${GREEN}${COMPLETED_STAGES}/${TOTAL_STAGES}${NC} (${OVERALL_PERCENT}%)"

if [ $COMPLETED_STAGES -eq $TOTAL_STAGES ]; then
    echo -e "状态：${GREEN}✅ 全面学习完成！${NC}"
elif [ $COMPLETED_STAGES -eq 0 ]; then
    echo -e "状态：${YELLOW}⏳ 准备启动${NC}"
else
    echo -e "状态：${BLUE}🔄 进行中${NC}"
fi

echo ""

# 下一步建议
echo -e "${YELLOW}💡 下一步行动：${NC}"
if [ ! -d "real_data/npm_packages_extended" ]; then
    echo -e "  1️⃣ 运行扩展NPM采集器：${BLUE}python3 training/extended_npm_crawler.py${NC}"
elif [ ! -f "real_data/obfuscated_code/augmented_training_pairs.jsonl" ]; then
    echo -e "  2️⃣ 执行数据增强：${BLUE}python3 training/fast_npm_obfuscator.py --methods all --multiplier 3${NC}"
elif [ ! -f "large_scale_training_50epochs.log" ] || ! grep -q "Epoch 50" large_scale_training_50epochs.log; then
    echo -e "  3️⃣ 等待大规模训练完成（当前${CURRENT_EPOCH:-0}/50轮）"
    echo -e "     监控命令：${BLUE}tail -f large_scale_training_50epochs.log${NC}"
elif [ ! -f "enhanced_training_30epochs.log" ] || ! grep -q "Epoch 30" enhanced_training_30epochs.log; then
    echo -e "  4️⃣ 启动增强训练：${BLUE}python3 training/enhanced_gpu_trainer.py --epochs 30 --device cuda${NC}"
elif [ ! -f "models/local/comprehensive_framework_detector.onnx" ]; then
    echo -e "  5️⃣ 导出ONNX模型：${BLUE}python3 training/convert_to_onnx.py${NC}"
else
    echo -e "  ✅ 所有阶段完成！可以运行：${BLUE}cargo test --test ai_integration_tests${NC}"
fi

echo ""
echo -e "${BLUE}══════════════════════════════════════════════════════════════${NC}"
