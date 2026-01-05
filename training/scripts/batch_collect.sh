#!/bin/bash
# 批量收集训练数据脚本
# 用法: ./batch_collect.sh [batch_size] [delay_seconds]

set -e

BATCH_SIZE=${1:-10}  # 每批处理的网站数量
DELAY=${2:-2}        # 每批之间的延迟（秒）

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$SCRIPT_DIR/../.."
WEBSITE_LIST="$SCRIPT_DIR/../data/website_list.txt"
BINARY="$PROJECT_ROOT/target/release/browerai"

echo "🚀 BrowerAI 批量数据收集工具"
echo "=================================="
echo "项目根目录: $PROJECT_ROOT"
echo "网站列表: $WEBSITE_LIST"
echo "每批网站数: $BATCH_SIZE"
echo "批次延迟: ${DELAY}秒"
echo ""

# 检查二进制文件是否存在
if [ ! -f "$BINARY" ]; then
    echo "❌ 未找到编译好的程序: $BINARY"
    echo "请先运行: cargo build --release --features ai"
    exit 1
fi

# 检查网站列表是否存在
if [ ! -f "$WEBSITE_LIST" ]; then
    echo "❌ 未找到网站列表: $WEBSITE_LIST"
    exit 1
fi

# 切换到项目根目录（重要！确保路径正确）
cd "$PROJECT_ROOT"
echo "✅ 工作目录: $(pwd)"
echo ""

# 读取网站列表（过滤注释和空行）
readarray -t URLS < <(grep -v '^#' "$WEBSITE_LIST" | grep -v '^[[:space:]]*$')
TOTAL_URLS=${#URLS[@]}

echo "📊 统计信息:"
echo "   总网站数: $TOTAL_URLS"
echo "   预计批次: $(( ($TOTAL_URLS + $BATCH_SIZE - 1) / $BATCH_SIZE ))"
echo "   预计耗时: ~$(( $TOTAL_URLS * 3 / 60 )) 分钟"
echo ""

read -p "按 Enter 开始收集，或 Ctrl+C 取消..."
echo ""

# 分批处理
BATCH_NUM=1
PROCESSED=0
FAILED=0
START_TIME=$(date +%s)

for ((i=0; i<$TOTAL_URLS; i+=$BATCH_SIZE)); do
    BATCH_URLS=("${URLS[@]:$i:$BATCH_SIZE}")
    BATCH_COUNT=${#BATCH_URLS[@]}
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📦 批次 $BATCH_NUM/$(( ($TOTAL_URLS + $BATCH_SIZE - 1) / $BATCH_SIZE ))"
    echo "   处理 $BATCH_COUNT 个网站 (${PROCESSED}/${TOTAL_URLS})"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # 执行学习命令
    if "$BINARY" --learn "${BATCH_URLS[@]}" 2>&1 | tee -a batch_collect.log; then
        PROCESSED=$((PROCESSED + BATCH_COUNT))
        echo "✅ 批次 $BATCH_NUM 完成"
    else
        FAILED=$((FAILED + BATCH_COUNT))
        echo "⚠️  批次 $BATCH_NUM 部分失败"
    fi
    
    BATCH_NUM=$((BATCH_NUM + 1))
    
    # 批次间延迟（最后一批不需要）
    if [ $i -lt $((TOTAL_URLS - BATCH_SIZE)) ]; then
        echo ""
        echo "⏳ 等待 ${DELAY} 秒后继续..."
        sleep $DELAY
        echo ""
    fi
done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "════════════════════════════════════════"
echo "🎉 数据收集完成！"
echo "════════════════════════════════════════"
echo "📊 统计:"
echo "   总网站数: $TOTAL_URLS"
echo "   成功处理: $PROCESSED"
echo "   失败: $FAILED"
echo "   总耗时: $((ELAPSED / 60)) 分 $((ELAPSED % 60)) 秒"
echo ""
echo "📁 反馈数据已保存到: $PROJECT_ROOT/training/data/feedback_*.json"
echo ""
echo "🎓 下一步:"
echo "   1. 查看反馈数据: ls -lh training/data/feedback_*.json"
echo "   2. 训练模型: cd training/scripts && python train_html_complexity.py --epochs 100"
echo "   3. 查看日志: tail -100 batch_collect.log"
echo ""
