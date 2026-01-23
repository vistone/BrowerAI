#!/bin/bash
# 实时监控仪表板

echo ""
echo "╔════════════════════════════════════════════════════════╗"
echo "║     BrowerAI 实时学习进度监控                          ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""

check_npm_obfuscator() {
    if [ -f real_data/obfuscated_code/training_pairs.jsonl ]; then
        lines=$(wc -l < real_data/obfuscated_code/training_pairs.jsonl)
        size=$(du -h real_data/obfuscated_code/training_pairs.jsonl | cut -f1)
        echo "✅ NPM混淆器"
        echo "   状态: 完成"
        echo "   训练对: $lines"
        echo "   文件大小: $size"
        return 0
    else
        echo "⏳ NPM混淆器"
        echo "   状态: 处理中..."
        if [ -f npm_obfuscation.log ]; then
            lines=$(wc -l < npm_obfuscation.log)
            echo "   日志行数: $lines"
        fi
        return 1
    fi
}

check_training() {
    if [ -f complete_pipeline.log ]; then
        lines=$(tail -5 complete_pipeline.log)
        if echo "$lines" | grep -q "Epoch"; then
            echo "🔄 模型训练"
            echo "   状态: 进行中"
            tail -3 complete_pipeline.log | grep -E "Epoch|Loss|准确率" || true
            return 0
        elif echo "$lines" | grep -q "✅"; then
            echo "✅ 模型训练"
            echo "   状态: 完成"
            return 0
        fi
    fi
    echo "⏳ 模型训练"
    echo "   状态: 等待数据..."
    return 1
}

check_models() {
    echo "💾 生成的模型:"
    if [ -f models/local/best_framework_detector.pt ]; then
        size=$(du -h models/local/best_framework_detector.pt | cut -f1)
        echo "   ✅ best_framework_detector.pt ($size)"
    fi
    if [ -f models/local/framework_detector_enhanced.pt ]; then
        size=$(du -h models/local/framework_detector_enhanced.pt | cut -f1)
        echo "   ✅ framework_detector_enhanced.pt ($size)"
    fi
    if [ -f models/local/framework_detector_enhanced.onnx ]; then
        size=$(du -h models/local/framework_detector_enhanced.onnx | cut -f1)
        echo "   ✅ framework_detector_enhanced.onnx ($size)"
    fi
}

while true; do
    clear
    
    echo "╔════════════════════════════════════════════════════════╗"
    echo "║     BrowerAI 实时学习进度监控                          ║"
    echo "╚════════════════════════════════════════════════════════╝"
    echo ""
    echo "🕐 更新时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📊 管道进度"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    check_npm_obfuscator
    echo ""
    check_training
    echo ""
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    check_models
    echo ""
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📂 数据统计"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "   NPM包数据: $(find real_data/npm_packages -name '*.js' -o -name '*.ts' 2>/dev/null | wc -l) 个文件"
    echo "   GitHub数据: $(find real_data/github_frameworks -type f 2>/dev/null | wc -l) 个文件"
    echo ""
    
    if [ -f complete_pipeline.log ]; then
        if grep -q "第2步完成" complete_pipeline.log; then
            echo "✅ 完整管道已完成!"
            break
        fi
    fi
    
    echo "按 Ctrl+C 退出监控"
    echo ""
    sleep 5
done

echo ""
echo "╔════════════════════════════════════════════════════════╗"
echo "║           🎉 学习过程全部完成!                        ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""
echo "下一步:"
echo "  1. 评估模型: python3 training/evaluate_model.py"
echo "  2. 查看日志: tail -100 complete_pipeline.log | grep -A5 '准确率'"
echo "  3. 部署使用: cargo build --features ai --release"
