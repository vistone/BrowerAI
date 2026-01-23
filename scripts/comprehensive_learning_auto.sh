#!/bin/bash

# 全面学习自动化执行脚本
# 依次执行5个阶段的完整训练流程

set -e

YELLOW='\033[1;33m'
GREEN='\033[1;32m'
BLUE='\033[1;34m'
RED='\033[1;31m'
NC='\033[0m'

# 日志函数
log_stage() {
    echo ""
    echo -e "${BLUE}══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}   $1${NC}"
    echo -e "${BLUE}══════════════════════════════════════════════════════════════${NC}"
    echo ""
}

log_info() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

# 检查是否有训练正在运行
check_training() {
    if pgrep -f "large_scale_trainer.py" > /dev/null; then
        return 0  # 训练正在运行
    else
        return 1  # 没有训练
    fi
}

# 主流程
log_stage "🎓 全面学习自动化流程"

echo -e "${YELLOW}说明：${NC}"
echo "• 此脚本将自动执行5个阶段的完整训练"
echo "• 阶段3（50轮训练）当前正在后台运行"
echo "• 我们先执行阶段1和2，为阶段4做准备"
echo ""

read -p "是否继续？[y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "已取消"
    exit 0
fi

# ============================================================
# 阶段1：扩展NPM数据采集
# ============================================================
log_stage "阶段1️⃣ - 扩展NPM数据采集（45+包）"

if [ -d "real_data/npm_packages_extended" ] && [ "$(ls -A real_data/npm_packages_extended 2>/dev/null)" ]; then
    log_warning "扩展NPM包目录已存在，跳过采集"
else
    log_info "开始采集45+个扩展NPM包..."
    log_info "目标：Vue, Angular, Preact, SolidJS, 状态管理, 构建工具, UI库, 测试框架"
    
    if [ -f "training/extended_npm_crawler.py" ]; then
        python3 training/extended_npm_crawler.py 2>&1 | tee extended_npm_collection.log
        log_info "扩展NPM包采集完成"
        
        # 统计采集结果
        if [ -d "real_data/npm_packages_extended" ]; then
            PACKAGE_COUNT=$(find real_data/npm_packages_extended -name "package.json" 2>/dev/null | wc -l)
            TOTAL_SIZE=$(du -sh real_data/npm_packages_extended 2>/dev/null | cut -f1)
            log_info "采集结果：${PACKAGE_COUNT}个包，总大小：${TOTAL_SIZE}"
        fi
    else
        log_error "extended_npm_crawler.py 不存在，请先创建"
        exit 1
    fi
fi

# ============================================================
# 阶段2：数据增强（3倍混淆）
# ============================================================
log_stage "阶段2️⃣ - 数据增强（3倍混淆）"

CURRENT_SAMPLES=$(wc -l < real_data/obfuscated_code/training_pairs.jsonl 2>/dev/null || echo "0")
log_info "当前样本数：${CURRENT_SAMPLES}"

if [ -f "real_data/obfuscated_code/augmented_training_pairs.jsonl" ]; then
    AUGMENTED_SAMPLES=$(wc -l < real_data/obfuscated_code/augmented_training_pairs.jsonl)
    log_warning "增强数据已存在（${AUGMENTED_SAMPLES}个样本），跳过"
else
    log_info "开始3倍数据增强..."
    log_info "方法：变量重命名 + 控制流混淆 + 字符串加密 + 死代码注入"
    
    if [ -f "training/fast_npm_obfuscator.py" ]; then
        python3 training/fast_npm_obfuscator.py --methods all --multiplier 3 --output real_data/obfuscated_code/augmented_training_pairs.jsonl 2>&1 | tee data_augmentation.log
        
        AUGMENTED_SAMPLES=$(wc -l < real_data/obfuscated_code/augmented_training_pairs.jsonl)
        MULTIPLIER=$((AUGMENTED_SAMPLES / CURRENT_SAMPLES))
        log_info "数据增强完成：${CURRENT_SAMPLES} → ${AUGMENTED_SAMPLES} (${MULTIPLIER}x)"
    else
        log_warning "fast_npm_obfuscator.py 不存在，跳过数据增强"
    fi
fi

# ============================================================
# 阶段3：大规模训练（50轮）- 状态检查
# ============================================================
log_stage "阶段3️⃣ - 大规模训练（50轮）- 状态检查"

if check_training; then
    log_info "50轮训练正在后台运行"
    
    if [ -f "large_scale_training_50epochs.log" ]; then
        CURRENT_EPOCH=$(grep -oP "Epoch \K\d+" large_scale_training_50epochs.log 2>/dev/null | tail -1 || echo "0")
        if [ -z "$CURRENT_EPOCH" ]; then
            CURRENT_EPOCH=0
        fi
        
        PERCENT=$((CURRENT_EPOCH * 100 / 50))
        log_info "训练进度：${CURRENT_EPOCH}/50 轮 (${PERCENT}%)"
        
        # 显示最近进度
        echo ""
        echo -e "${YELLOW}最近训练日志：${NC}"
        tail -20 large_scale_training_50epochs.log | grep -E "Epoch|Loss|准确率" || echo "暂无数据"
        echo ""
        
        # 估算剩余时间
        if [ "$CURRENT_EPOCH" -gt 0 ]; then
            # 假设每轮5-10分钟
            REMAINING_EPOCHS=$((50 - CURRENT_EPOCH))
            EST_MIN=$((REMAINING_EPOCHS * 5))
            EST_MAX=$((REMAINING_EPOCHS * 10))
            log_info "预计剩余时间：${EST_MIN}-${EST_MAX}分钟"
        fi
    fi
    
    log_warning "等待阶段3完成后，可以执行阶段4和5"
    log_info "监控命令：tail -f large_scale_training_50epochs.log"
else
    log_warning "50轮训练未运行，建议重新启动"
    log_info "启动命令：python3 training/large_scale_trainer.py --epochs 50 --batch-size 64 --device cuda"
fi

# ============================================================
# 阶段4：增强训练（30轮）- 准备
# ============================================================
log_stage "阶段4️⃣ - 增强训练（30轮）- 准备"

log_info "依赖：等待阶段3完成"
log_info "数据源：augmented_training_pairs.jsonl（增强后数据）"
log_info "启动命令："
echo -e "${BLUE}  python3 training/enhanced_gpu_trainer.py \\${NC}"
echo -e "${BLUE}    --epochs 30 \\${NC}"
echo -e "${BLUE}    --augmentation strong \\${NC}"
echo -e "${BLUE}    --device cuda \\${NC}"
echo -e "${BLUE}    --data-file real_data/obfuscated_code/augmented_training_pairs.jsonl \\${NC}"
echo -e "${BLUE}    2>&1 | tee enhanced_training_30epochs.log${NC}"

# ============================================================
# 阶段5：模型导出与验证 - 准备
# ============================================================
log_stage "阶段5️⃣ - 模型导出与验证 - 准备"

log_info "依赖：等待阶段4完成"
log_info "导出命令："
echo -e "${BLUE}  python3 training/convert_to_onnx.py \\${NC}"
echo -e "${BLUE}    --model models/local/comprehensive_best.pt \\${NC}"
echo -e "${BLUE}    --output models/local/comprehensive_framework_detector.onnx \\${NC}"
echo -e "${BLUE}    --opset 14${NC}"

echo ""
log_info "验证命令："
echo -e "${BLUE}  cargo test --test ai_integration_tests${NC}"

# ============================================================
# 总结
# ============================================================
log_stage "📊 全面学习流程总结"

echo -e "${GREEN}✅ 已完成：${NC}"
echo "  • 阶段1：扩展NPM包采集（或已存在）"
echo "  • 阶段2：数据增强（或已存在）"
echo ""

echo -e "${YELLOW}🔄 进行中：${NC}"
echo "  • 阶段3：50轮大规模训练（后台运行）"
echo ""

echo -e "${BLUE}⏳ 待执行：${NC}"
echo "  • 阶段4：30轮增强训练（等待阶段3完成）"
echo "  • 阶段5：ONNX导出与验证（等待阶段4完成）"
echo ""

echo -e "${YELLOW}💡 建议操作：${NC}"
echo "  1. 监控训练进度："
echo -e "     ${BLUE}watch -n 30 ./comprehensive_learning_monitor.sh${NC}"
echo ""
echo "  2. 查看实时训练日志："
echo -e "     ${BLUE}tail -f large_scale_training_50epochs.log${NC}"
echo ""
echo "  3. 检查GPU使用情况："
echo -e "     ${BLUE}watch -n 5 nvidia-smi${NC}"
echo ""
echo "  4. 阶段3完成后，手动执行阶段4："
echo -e "     ${BLUE}python3 training/enhanced_gpu_trainer.py --epochs 30 --augmentation strong --device cuda${NC}"
echo ""

log_stage "🎓 全面学习流程配置完成"

echo -e "${GREEN}总体目标：${NC}"
echo "  • 数据量：17K → 100K+ 样本（6倍扩展）"
echo "  • 框架数：21 → 65+ 包（3倍增加）"
echo "  • 训练轮数：3 → 80 轮（50+30，27倍深度）"
echo "  • 泛化能力：基础训练 → 强数据增强 + 深度学习"
echo ""

echo -e "${YELLOW}预计总耗时：${NC}"
echo "  • 阶段1：30分钟（NPM采集）"
echo "  • 阶段2：1-2小时（数据增强）"
echo "  • 阶段3：4-8小时（50轮训练）"
echo "  • 阶段4：2-4小时（30轮增强训练）"
echo "  • 阶段5：10分钟（导出验证）"
echo "  • 总计：8-15小时"
echo ""

log_info "全面学习流程已启动，请耐心等待训练完成"
