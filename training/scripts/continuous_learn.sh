#!/bin/bash
# 持续学习脚本 - 不间断学习指定时间

set -e

DURATION=${1:-3600}  # 默认1小时(3600秒)
DELAY=${2:-0}        # 网站之间延迟(秒)

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$SCRIPT_DIR/../.."
WEBSITE_LIST="$SCRIPT_DIR/../data/website_list.txt"
BINARY="$PROJECT_ROOT/target/release/browerai"

echo "🚀 BrowerAI 持续学习模式"
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                    不间断学习进行中...                      ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "⏱️  设置:"
echo "   运行时间: ${DURATION} 秒 (~$(( $DURATION / 60 )) 分钟)"
echo "   网站延迟: ${DELAY} 秒"
echo ""

if [ ! -f "$BINARY" ]; then
    echo "❌ 未找到编译程序: $BINARY"
    exit 1
fi

if [ ! -f "$WEBSITE_LIST" ]; then
    echo "❌ 未找到网站列表: $WEBSITE_LIST"
    exit 1
fi

cd "$PROJECT_ROOT"

# 读取网站列表
readarray -t URLS < <(grep -v '^#' "$WEBSITE_LIST" | grep -v '^[[:space:]]*$')
TOTAL_URLS=${#URLS[@]}

echo "📊 网站数: $TOTAL_URLS"
echo ""

START_TIME=$(date +%s)
END_TIME=$((START_TIME + DURATION))
ROUND=1
TOTAL_WEBSITES=0
TOTAL_EVENTS=0
SUCCESSFUL=0
FAILED=0

echo "🎯 开始持续学习..."
echo ""

while [ $(date +%s) -lt $END_TIME ]; do
    ELAPSED=$(($(date +%s) - START_TIME))
    REMAINING=$((DURATION - ELAPSED))
    PERCENT=$((ELAPSED * 100 / DURATION))
    
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║ 🔄 轮次 $ROUND | ⏳ ${ELAPSED}秒/${DURATION}秒 (${PERCENT}%)"
    echo "╚════════════════════════════════════════════════════════════╝"
    
    # 随机打乱网站列表顺序，增加多样性
    SHUFFLED=($(printf '%s\n' "${URLS[@]}" | shuf))
    
    # 选择本轮要访问的网站数(根据剩余时间动态调整)
    # 假设每个网站平均1秒，选择足够填充剩余时间的网站数
    REMAINING_WEBSITES=$(($REMAINING / 2))
    if [ $REMAINING_WEBSITES -gt $TOTAL_URLS ]; then
        REMAINING_WEBSITES=$TOTAL_URLS
    fi
    if [ $REMAINING_WEBSITES -lt 1 ]; then
        REMAINING_WEBSITES=1
    fi
    
    echo "   本轮网站数: $REMAINING_WEBSITES (共 $TOTAL_URLS)"
    echo ""
    
    for ((i=0; i<$REMAINING_WEBSITES && $(date +%s) -lt $END_TIME; i++)); do
        URL="${SHUFFLED[$i]}"
        ELAPSED=$(($(date +%s) - START_TIME))
        REMAINING=$((DURATION - ELAPSED))
        
        echo "   [$((i+1))/$REMAINING_WEBSITES] 访问: $URL"
        
        # 执行学习,并捕获输出
        if OUTPUT=$("$BINARY" --learn "$URL" 2>&1); then
            TOTAL_WEBSITES=$((TOTAL_WEBSITES + 1))
            SUCCESSFUL=$((SUCCESSFUL + 1))
            
            # 统计反馈事件数
            FEEDBACK=$(ls -t training/data/feedback_*.json 2>/dev/null | head -1)
            if [ -n "$FEEDBACK" ]; then
                EVENT_COUNT=$(python3 -c "import json; print(len(json.load(open('$FEEDBACK'))))" 2>/dev/null || echo "0")
                TOTAL_EVENTS=$((TOTAL_EVENTS + EVENT_COUNT))
            fi
        else
            FAILED=$((FAILED + 1))
        fi
        
        # 网站间延迟
        if [ $DELAY -gt 0 ] && [ $((i+1)) -lt $REMAINING_WEBSITES ]; then
            sleep $DELAY
        fi
    done
    
    ROUND=$((ROUND + 1))
    echo ""
done

FINAL_TIME=$(date +%s)
TOTAL_ELAPSED=$((FINAL_TIME - START_TIME))

# 最终统计
echo ""
echo "════════════════════════════════════════════════════════════"
echo "🎉 学习完成！"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "📊 最终统计:"
echo "   ├─ 运行时间: ${TOTAL_ELAPSED}秒 (~$(( $TOTAL_ELAPSED / 60 ))分钟)"
echo "   ├─ 总轮数: $((ROUND - 1))"
echo "   ├─ 访问网站: $TOTAL_WEBSITES"
echo "   ├─ 成功: $SUCCESSFUL"
echo "   ├─ 失败: $FAILED"
echo "   └─ 收集事件: ~$TOTAL_EVENTS"
echo ""

# 统计最终数据
echo "📈 最终数据统计:"
python3 << 'EOF'
import json, glob
total = 0
html = css = js = 0

for f in sorted(glob.glob('training/data/feedback_*.json')):
    try:
        data = json.load(open(f))
        if isinstance(data, list):
            total += len(data)
            html += sum(1 for e in data if e.get('type') == 'html_parsing')
            css += sum(1 for e in data if e.get('type') == 'css_parsing')
            js += sum(1 for e in data if e.get('type') == 'js_parsing')
    except: pass

print(f'   ├─ 总反馈事件: {total}')
print(f'   ├─ HTML 样本: {html}')
print(f'   ├─ CSS 样本: {css}')
print(f'   └─ JS 样本: {js}')
print(f'')
print(f'📁 反馈文件: {len(glob.glob("training/data/feedback_*.json"))} 个')

print(f'')
print(f'🎓 建议下一步:')
print(f'   1. 训练改进的模型: python train_html_complexity.py --epochs 100')
print(f'   2. 查看 AI 报告: cargo run --features ai -- --ai-report')
print(f'   3. 继续收集更多数据: ./batch_collect.sh')
EOF

echo ""
echo "✅ 日志已保存: continuous_learn.log"
echo ""
