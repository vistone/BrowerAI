#!/bin/bash
# 1000网站学习 - 一键完整脚本
# 使用高并发加速，总时间约 4-6 小时

set -e  # 遇到错误立即退出

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║         BrowerAI - 1000网站大规模学习 一键脚本                ║"
echo "║                 High Concurrency Version                       ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# 配置
URLS_FILE="data/large_urls.txt"
OUTPUT_DATA="data/websites/1000_sites.jsonl"
CHECKPOINT_DIR="checkpoints/large_1000"
RESULTS_FILE="results/1000_inference.json"
CONCURRENCY=20  # 🔥 并发数（可调整：10-50）

# 创建目录
mkdir -p data/websites checkpoints results logs

echo "📋 配置信息:"
echo "  - URL列表: $URLS_FILE"
echo "  - 输出数据: $OUTPUT_DATA"
echo "  - 并发数: $CONCURRENCY 🔥"
echo "  - 检查点: $CHECKPOINT_DIR"
echo ""

# ==========================================
# 步骤1: 高并发爬取网站
# ==========================================
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  步骤 1/3: 爬取1000个网站（预计2-3小时）                      ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

if [ -f "$OUTPUT_DATA" ]; then
    EXISTING_COUNT=$(wc -l < "$OUTPUT_DATA")
    echo "⚠️  发现已有数据文件: $EXISTING_COUNT 个网站"
    read -p "是否覆盖？(y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "跳过爬取步骤，使用现有数据"
    else
        rm -f "$OUTPUT_DATA"
        echo "开始爬取..."
        python scripts/prepare_website_data.py \
          --urls-file "$URLS_FILE" \
          --output "$OUTPUT_DATA" \
          --depth 2 \
          --max-pages 5 \
          --concurrency $CONCURRENCY \
          2>&1 | tee logs/crawl_$(date +%Y%m%d_%H%M%S).log
    fi
else
    echo "开始爬取..."
    python scripts/prepare_website_data.py \
      --urls-file "$URLS_FILE" \
      --output "$OUTPUT_DATA" \
      --depth 2 \
      --max-pages 5 \
      --concurrency $CONCURRENCY \
      2>&1 | tee logs/crawl_$(date +%Y%m%d_%H%M%S).log
fi

# 验证数据
if [ ! -f "$OUTPUT_DATA" ]; then
    echo "❌ 错误: 爬取失败，未找到输出文件"
    exit 1
fi

SITE_COUNT=$(wc -l < "$OUTPUT_DATA")
echo ""
echo "✅ 步骤1完成！成功爬取 $SITE_COUNT 个网站"
echo ""

# 数据统计
echo "📊 数据统计:"
cat "$OUTPUT_DATA" | python3 -c "
import json, sys
sites = [json.loads(l) for l in sys.stdin]
total_pages = sum(s.get('depth', 1) for s in sites)
frameworks = {}
categories = {}

for s in sites:
    fw = s.get('metadata', {}).get('framework', 'Unknown')
    cat = s.get('category', 'unknown')
    frameworks[fw] = frameworks.get(fw, 0) + 1
    categories[cat] = categories.get(cat, 0) + 1

print(f'  网站总数: {len(sites)}')
print(f'  页面总数: {total_pages}')
print(f'  平均深度: {total_pages/len(sites):.1f} 页/站')
print(f'\\n  Top 5 框架:')
for fw, count in sorted(frameworks.items(), key=lambda x: -x[1])[:5]:
    print(f'    {fw}: {count}')
print(f'\\n  Top 5 分类:')
for cat, count in sorted(categories.items(), key=lambda x: -x[1])[:5]:
    print(f'    {cat}: {count}')
"
echo ""

# ==========================================
# 步骤2: 训练模型
# ==========================================
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  步骤 2/3: 训练模型（预计2-40小时，取决于硬件）              ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

read -p "是否开始训练？这可能需要很长时间 (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "开始训练..."
    python scripts/train_large_scale.py \
      --data-file "$OUTPUT_DATA" \
      --checkpoint-dir "$CHECKPOINT_DIR" \
      --epochs 50 \
      --batch-size 8 \
      --learning-rate 1e-4 \
      2>&1 | tee logs/train_$(date +%Y%m%d_%H%M%S).log
    
    echo ""
    echo "✅ 步骤2完成！模型已保存"
    ls -lh "$CHECKPOINT_DIR"/best_model.pt
    echo ""
else
    echo "跳过训练步骤"
    echo ""
fi

# ==========================================
# 步骤3: 推理生成
# ==========================================
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  步骤 3/3: 推理生成（预计10-30分钟）                          ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

if [ ! -f "$CHECKPOINT_DIR/best_model.pt" ]; then
    echo "⚠️  未找到训练好的模型，跳过推理步骤"
else
    read -p "是否进行批量推理？(y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "开始推理..."
        python scripts/inference_website.py \
          --model "$CHECKPOINT_DIR/best_model.pt" \
          --mode batch \
          --input "$OUTPUT_DATA" \
          --output "$RESULTS_FILE" \
          --max-samples 1000 \
          2>&1 | tee logs/inference_$(date +%Y%m%d_%H%M%S).log
        
        echo ""
        echo "✅ 步骤3完成！推理结果已保存"
        ls -lh "$RESULTS_FILE"
        echo ""
    else
        echo "跳过推理步骤"
        echo ""
    fi
fi

# ==========================================
# 总结
# ==========================================
echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                        🎉 全部完成！                           ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "📁 生成的文件:"
echo "  - 数据集: $OUTPUT_DATA"
[ -f "$CHECKPOINT_DIR/best_model.pt" ] && echo "  - 模型: $CHECKPOINT_DIR/best_model.pt"
[ -f "$RESULTS_FILE" ] && echo "  - 推理结果: $RESULTS_FILE"
echo ""
echo "📊 快速查看:"
echo "  查看数据: head -1 $OUTPUT_DATA | python -m json.tool"
[ -f "$CHECKPOINT_DIR/best_model.pt" ] && echo "  查看模型: ls -lh $CHECKPOINT_DIR/"
[ -f "$RESULTS_FILE" ] && echo "  查看结果: cat $RESULTS_FILE | python -m json.tool | head -50"
echo ""
echo "✨ 恭喜！您已经完成了1000个网站的学习！"
echo ""
