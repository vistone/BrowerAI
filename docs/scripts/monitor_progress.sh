#!/bin/bash
# 实时进度监控仪表板

PROJECT_DIR="/home/stone/BrowerAI"
cd "$PROJECT_DIR"

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

clear

while true; do
    # 清屏并显示标题
    clear
    echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║${NC}     🌍 真实数据学习系统 - 进度监控仪表板                       ${BLUE}║${NC}"
    echo -e "${BLUE}║${NC}     更新时间: $(date '+%Y-%m-%d %H:%M:%S')                        ${BLUE}║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
    
    echo ""
    
    # 1. GitHub爬虫状态
    echo -e "${BLUE}1. GitHub 框架爬虫${NC}"
    if [ -d "real_data/github_frameworks" ]; then
        GH_FILES=$(find real_data/github_frameworks -type f | wc -l)
        GH_SIZE=$(du -sh real_data/github_frameworks 2>/dev/null | cut -f1)
        echo -e "   ${GREEN}✅ 已完成${NC}  - 文件: $GH_FILES 个  大小: $GH_SIZE"
    else
        echo -e "   ${RED}❌ 未开始${NC}"
    fi
    
    # 2. NPM爬虫状态
    echo ""
    echo -e "${BLUE}2. NPM 包爬虫${NC}"
    if [ -d "real_data/npm_packages" ]; then
        NPM_FILES=$(find real_data/npm_packages -type f | wc -l)
        NPM_SIZE=$(du -sh real_data/npm_packages 2>/dev/null | cut -f1)
        echo -e "   ${GREEN}✅ 已完成${NC}  - 文件: $NPM_FILES 个  大小: $NPM_SIZE"
    else
        echo -e "   ${RED}❌ 未开始${NC}"
    fi
    
    # 3. 代码混淆状态
    echo ""
    echo -e "${BLUE}3. 代码混淆生成${NC}"
    if pgrep -f "real_code_obfuscator.py" > /dev/null 2>&1; then
        echo -e "   ${YELLOW}⏳ 运行中${NC}"
        PAIR_COUNT=$(wc -l < real_data/obfuscated_code/training_pairs.jsonl 2>/dev/null || echo 0)
        if [ "$PAIR_COUNT" -gt 0 ]; then
            OBF_SIZE=$(du -sh real_data/obfuscated_code 2>/dev/null | cut -f1)
            echo -e "      已生成: $PAIR_COUNT 对 ($OBF_SIZE)"
        fi
    elif [ -f "real_data/obfuscated_code/training_pairs.jsonl" ]; then
        PAIR_COUNT=$(wc -l < real_data/obfuscated_code/training_pairs.jsonl)
        OBF_SIZE=$(du -sh real_data/obfuscated_code 2>/dev/null | cut -f1)
        echo -e "   ${GREEN}✅ 已完成${NC}  - 混淆对: $PAIR_COUNT 对  大小: $OBF_SIZE"
    else
        echo -e "   ${RED}❌ 未开始${NC}"
    fi
    
    # 4. GPU训练状态
    echo ""
    echo -e "${BLUE}4. GPU 框架检测训练${NC}"
    if pgrep -f "gpu_framework_detector.py" > /dev/null 2>&1; then
        echo -e "   ${YELLOW}⏳ 训练中${NC}"
        if [ -f "gpu_train.log" ]; then
            EPOCH=$(grep -o "Epoch [0-9]*" gpu_train.log 2>/dev/null | tail -1)
            LOSS=$(grep -o "Loss: [0-9.]*" gpu_train.log 2>/dev/null | tail -1)
            if [ ! -z "$EPOCH" ]; then
                echo -e "      $EPOCH - $LOSS"
            fi
        fi
    elif [ -f "models/local/framework_detector.onnx" ]; then
        MODEL_SIZE=$(du -sh models/local/framework_detector.onnx | cut -f1)
        echo -e "   ${GREEN}✅ 已完成${NC}  - 模型: $MODEL_SIZE"
    else
        if [ -f "real_data/obfuscated_code/training_pairs.jsonl" ]; then
            echo -e "   ${YELLOW}⏳ 等待启动${NC}"
        else
            echo -e "   ${RED}❌ 等待数据${NC}"
        fi
    fi
    
    # 总体进度
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    TOTAL_SIZE=0
    if [ -d "real_data/github_frameworks" ]; then
        GH_BYTES=$(du -sb real_data/github_frameworks 2>/dev/null | cut -f1)
        TOTAL_SIZE=$((TOTAL_SIZE + GH_BYTES))
    fi
    if [ -d "real_data/npm_packages" ]; then
        NPM_BYTES=$(du -sb real_data/npm_packages 2>/dev/null | cut -f1)
        TOTAL_SIZE=$((TOTAL_SIZE + NPM_BYTES))
    fi
    if [ -d "real_data/obfuscated_code" ]; then
        OBF_BYTES=$(du -sb real_data/obfuscated_code 2>/dev/null | cut -f1)
        TOTAL_SIZE=$((TOTAL_SIZE + OBF_BYTES))
    fi
    
    TOTAL_MB=$((TOTAL_SIZE / 1024 / 1024))
    echo -e "   ${BLUE}📊 总数据量: ${TOTAL_MB} MB${NC}"
    
    # 刷新间隔
    echo ""
    echo -e "按 Ctrl+C 停止监控 (10秒后自动刷新)..."
    sleep 10
done
