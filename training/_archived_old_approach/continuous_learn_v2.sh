#!/bin/bash
# 高效持续学习脚本 - 优化版本

DURATION=${1:-3600}  # 1小时
BATCH_SIZE=${2:-5}   # 每批网站数

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$SCRIPT_DIR/../.."
WEBSITE_LIST="$SCRIPT_DIR/../data/website_list.txt"
BINARY="$PROJECT_ROOT/target/release/browerai"

cd "$PROJECT_ROOT"

# 读取网站列表
readarray -t URLS < <(grep -v '^#' "$WEBSITE_LIST" | grep -v '^[[:space:]]*$')
TOTAL_URLS=${#URLS[@]}

START_TIME=$(date +%s)
END_TIME=$((START_TIME + DURATION))
ROUND=1

echo "🚀 BrowerAI 持续学习模式 - 开始时间: $(date +'%H:%M:%S')"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "⏱️  目标: 运行 $((DURATION/60)) 分钟"
echo "📊 网站库: $TOTAL_URLS 个网站"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

while [ $(date +%s) -lt $END_TIME ]; do
    ELAPSED=$(($(date +%s) - START_TIME))
    REMAINING=$((DURATION - ELAPSED))
    PERCENT=$((ELAPSED * 100 / DURATION))
    
    # 计算进度条
    FILLED=$((PERCENT / 5))
    EMPTY=$((20 - FILLED))
    PROGRESS_BAR=$(printf '█%.0s' $(seq 1 $FILLED))$(printf '░%.0s' $(seq 1 $EMPTY))
    
    echo "⏳ [$PROGRESS_BAR] ${PERCENT}% | $(printf '%02d' $((REMAINING/60))):$(printf '%02d' $((REMAINING%60))) 剩余"
    
    # 打乱网站列表
    SHUFFLED=($(printf '%s\n' "${URLS[@]}" | shuf | head -$BATCH_SIZE))
    
    for URL in "${SHUFFLED[@]}"; do
        if [ $(date +%s) -ge $END_TIME ]; then
            break
        fi
        
        # 快速执行学习(后台)
        "$BINARY" --learn "$URL" > /dev/null 2>&1 &
        PIDS="$PIDS $!"
        
        # 限制并发数
        if [ $(echo "$PIDS" | wc -w) -ge 3 ]; then
            wait -n
            PIDS=$(echo "$PIDS" | tr ' ' '\n' | grep -v "^$" | head -2 | tr '\n' ' ')
        fi
    done
    
    ROUND=$((ROUND + 1))
    sleep 2
done

# 等待所有后台任务完成
wait

FINAL_TIME=$(date +%s)
TOTAL_ELAPSED=$((FINAL_TIME - START_TIME))

# 统计结果
echo ""
echo "════════════════════════════════════════════════════════════"
echo "✅ 学习完成！完成时间: $(date +'%H:%M:%S')"
echo "════════════════════════════════════════════════════════════"

python3 << 'EOF'
import json, glob, os
total = html = css = js = 0
files = sorted(glob.glob('training/data/feedback_*.json'))

for f in files:
    try:
        data = json.load(open(f))
        if isinstance(data, list):
            total += len(data)
            html += sum(1 for e in data if e.get('type') == 'html_parsing')
            css += sum(1 for e in data if e.get('type') == 'css_parsing')
            js += sum(1 for e in data if e.get('type') == 'js_parsing')
    except: pass

print(f'')
print(f'📊 收集数据统计:')
print(f'   ├─ 总反馈事件: {total}')
print(f'   ├─ HTML 样本: {html}')
print(f'   ├─ CSS 样本: {css}')
print(f'   ├─ JS 样本: {js}')
print(f'   └─ 数据文件: {len(files)} 个')
print(f'')

if html > 0:
    growth = ((html - 36) / 36 * 100) if html > 36 else 0
    print(f'🚀 增长情况:')
    print(f'   初始 HTML 样本: 36')
    print(f'   当前 HTML 样本: {html}')
    print(f'   增长幅度: {growth:+.0f}%')
    print(f'')

print(f'💾 最近生成的文件:')
if files:
    for f in files[-3:]:
        size = os.path.getsize(f)
        print(f'   - {os.path.basename(f)} ({size} bytes)')
EOF

echo ""
