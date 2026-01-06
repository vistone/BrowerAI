#!/bin/bash
# 实时监控增量学习进度

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║         增量学习进度监控                                       ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# 检查进程
PID=$(ps aux | grep incremental_learning.py | grep -v grep | awk '{print $2}')
if [ -z "$PID" ]; then
    echo "❌ 学习进程未运行"
    echo ""
    echo "启动命令: ./run_incremental_learning.sh"
    exit 1
fi

echo "✅ 进程运行中 (PID: $PID)"
echo ""

# 显示当前进度
LATEST_LOG=$(ls -t logs/incremental_*.log 2>/dev/null | head -1)
if [ -f "$LATEST_LOG" ]; then
    CURRENT=$(grep -o '\[[0-9]\+/977\]' "$LATEST_LOG" | tail -1 | grep -o '[0-9]\+' | head -1)
    if [ ! -z "$CURRENT" ]; then
        PERCENT=$(echo "scale=1; $CURRENT * 100 / 977" | bc)
        REMAINING=$((977 - CURRENT))
        echo "📊 当前进度: $CURRENT/977 ($PERCENT%)"
        echo "   剩余: $REMAINING 个网站"
        echo ""
    fi
    
    # 显示最近损失
    echo "📈 最近10次学习记录:"
    grep "📈 损失" "$LATEST_LOG" | tail -10 | while read line; do
        echo "   $line"
    done
    echo ""
    
    # Checkpoint信息
    if [ -f "checkpoints/incremental/latest.pt" ]; then
        SIZE=$(ls -lh checkpoints/incremental/latest.pt | awk '{print $5}')
        MTIME=$(stat -c %y checkpoints/incremental/latest.pt 2>/dev/null || stat -f "%Sm" checkpoints/incremental/latest.pt)
        echo "💾 Checkpoint: $SIZE (最后保存: $(date -r checkpoints/incremental/latest.pt '+%H:%M:%S' 2>/dev/null || echo '未知'))"
        echo ""
    fi
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📝 实时日志 (按 Ctrl+C 退出):"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 实时跟踪日志
tail -f "$LATEST_LOG" 2>/dev/null | grep --line-buffered -E "(\[|损失|保存|爬取完成)"
