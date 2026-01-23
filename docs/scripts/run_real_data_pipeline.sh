#!/bin/bash
# 真实数据学习执行主脚本

set -e

PROJECT_DIR="/home/stone/BrowerAI"
cd "$PROJECT_DIR"

echo "=========================================="
echo "🌍 真实数据学习系统执行"
echo "=========================================="

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 进度显示函数
show_progress() {
    echo -e "${GREEN}✅${NC} $1"
}

show_warning() {
    echo -e "${YELLOW}⏳${NC} $1"
}

show_error() {
    echo -e "${RED}❌${NC} $1"
}

# 等待进程完成
wait_for_process() {
    local name=$1
    local log_file=$2
    
    echo -e "\n${YELLOW}等待 $name 完成...${NC}"
    
    while true; do
        if ! pgrep -f "$name" > /dev/null 2>&1; then
            show_progress "$name 完成"
            break
        fi
        
        # 显示进度（最后5行）
        if [ -f "$log_file" ]; then
            echo -ne "\\033[1A\\033[K"  # 清空上一行
            tail -3 "$log_file" | head -1
        fi
        
        sleep 2
    done
}

# 检查GitHub爬虫
echo ""
echo "1️⃣  检查GitHub框架爬虫..."
if [ -d "real_data/github_frameworks" ] && [ "$(find real_data/github_frameworks -type f | wc -l)" -gt 20 ]; then
    show_progress "GitHub框架已爬取 ($(find real_data/github_frameworks -type f | wc -l) 个文件)"
else
    show_warning "GitHub框架爬虫..."
    timeout 600 python3 training/github_framework_crawler.py > github.log 2>&1 || true
    show_progress "GitHub框架爬虫完成"
fi

# 检查NPM爬虫
echo ""
echo "2️⃣  检查NPM包爬虫..."
if [ -d "real_data/npm_packages" ] && [ "$(find real_data/npm_packages -type f | wc -l)" -gt 35 ]; then
    show_progress "NPM包已爬取 ($(find real_data/npm_packages -type f | wc -l) 个文件)"
else
    show_warning "NPM包爬虫..."
    timeout 600 python3 training/npm_package_crawler.py > npm.log 2>&1 &
    NPM_PID=$!
fi

# 检查代码混淆
echo ""
echo "3️⃣  检查代码混淆..."
if [ -f "real_data/obfuscated_code/training_pairs.jsonl" ]; then
    PAIR_COUNT=$(wc -l < real_data/obfuscated_code/training_pairs.jsonl)
    show_progress "代码混淆对已生成 ($PAIR_COUNT 对)"
else
    show_warning "生成混淆代码对..."
    timeout 600 python3 training/real_code_obfuscator.py > obfuscator.log 2>&1 &
    OBF_PID=$!
fi

# 等待NPM和混淆任务完成
if [ ! -z "$NPM_PID" ]; then
    wait_for_process "npm_package_crawler" "npm.log"
fi

if [ ! -z "$OBF_PID" ]; then
    wait_for_process "real_code_obfuscator" "obfuscator.log"
fi

# 统计数据
echo ""
echo "=========================================="
echo "📊 数据统计"
echo "=========================================="

if [ -d "real_data/github_frameworks" ]; then
    GITHUB_FILES=$(find real_data/github_frameworks -type f | wc -l)
    GITHUB_SIZE=$(du -sh real_data/github_frameworks | cut -f1)
    echo "GitHub源代码: $GITHUB_FILES 个文件 ($GITHUB_SIZE)"
fi

if [ -d "real_data/npm_packages" ]; then
    NPM_FILES=$(find real_data/npm_packages -type f | wc -l)
    NPM_SIZE=$(du -sh real_data/npm_packages | cut -f1)
    echo "NPM包: $NPM_FILES 个文件 ($NPM_SIZE)"
fi

if [ -f "real_data/obfuscated_code/training_pairs.jsonl" ]; then
    PAIR_COUNT=$(wc -l < real_data/obfuscated_code/training_pairs.jsonl)
    PAIR_SIZE=$(du -sh real_data/obfuscated_code | cut -f1)
    echo "混淆对: $PAIR_COUNT 对 ($PAIR_SIZE)"
fi

# 启动GPU训练
echo ""
echo "4️⃣  启动GPU框架检测训练..."

if [ -f "real_data/obfuscated_code/training_pairs.jsonl" ]; then
    show_warning "启动PyTorch训练..."
    timeout 3600 python3 training/gpu_framework_detector.py > gpu_train.log 2>&1 &
    GPU_PID=$!
    
    echo "GPU训练进程 PID: $GPU_PID"
    echo "日志文件: gpu_train.log"
    echo ""
    echo "实时监控命令:"
    echo "  tail -f gpu_train.log"
    echo ""
    
    wait_for_process "gpu_framework_detector" "gpu_train.log"
else
    show_error "没有混淆对数据，跳过GPU训练"
fi

# 最终总结
echo ""
echo "=========================================="
echo "✨ 执行完成！"
echo "=========================================="
echo ""
echo "生成的模型:"
if [ -f "models/local/framework_detector.onnx" ]; then
    ONNX_SIZE=$(du -sh models/local/framework_detector.onnx | cut -f1)
    show_progress "ONNX模型: $ONNX_SIZE"
fi

echo ""
echo "下一步:"
echo "1. 测试模型推理: cargo run --example gpu_inference_demo"
echo "2. 集成到浏览器: cargo build --features ai"
echo "3. 部署: cargo build --release"
