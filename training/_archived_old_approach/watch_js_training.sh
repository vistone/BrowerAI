#!/bin/bash
# 监控JS反混淆模型训练进度

LATEST_LOG=$(ls -t logs/js_deobfuscator_train_*.log 2>/dev/null | head -1)

if [ -z "$LATEST_LOG" ]; then
    echo "❌ 未找到训练日志文件"
    exit 1
fi

echo "=========================================="
echo "JS反混淆模型训练监控"
echo "=========================================="
echo "日志文件: $LATEST_LOG"
echo ""

# 显示最新10个epoch的loss
echo "📈 最近10个epochs:"
grep "Epoch" "$LATEST_LOG" | tail -10
echo ""

# 检查是否完成
if grep -q "Training completed!" "$LATEST_LOG"; then
    echo "✅ 训练已完成!"
    
    # 检查ONNX导出
    if grep -q "ONNX model exported successfully" "$LATEST_LOG"; then
        echo "✅ ONNX模型已导出"
        
        ONNX_FILE="../models/local/js_deobfuscator_v1.onnx"
        if [ -f "$ONNX_FILE" ]; then
            SIZE=$(ls -lh "$ONNX_FILE" | awk '{print $5}')
            echo "   文件: $ONNX_FILE ($SIZE)"
        fi
    else
        echo "⚠️  ONNX导出状态未知"
    fi
else
    echo "🔄 训练进行中..."
    echo ""
    echo "📝 实时日志 (按Ctrl+C停止):"
    tail -f "$LATEST_LOG"
fi
