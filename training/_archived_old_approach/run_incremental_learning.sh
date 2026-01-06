#!/bin/bash
# 增量学习 - 一键脚本
# 边爬边学，无需保存大量中间数据

set -e

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║         增量学习模式 - 爬一个学一个                           ║"
echo "║         Incremental Learning: Crawl → Learn → Update           ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# 配置
URLS_FILE="data/large_urls.txt"
CHECKPOINT_DIR="checkpoints/incremental"
DEPTH=2
MAX_PAGES=5
LEARNING_RATE=1e-4
SAVE_FREQ=10  # 每10个网站保存一次

echo "📋 配置:"
echo "  - URL列表: $URLS_FILE"
echo "  - 检查点目录: $CHECKPOINT_DIR"
echo "  - 深度: $DEPTH, 页面数: $MAX_PAGES"
echo "  - 学习率: $LEARNING_RATE"
echo "  - 保存频率: 每 $SAVE_FREQ 个网站"
echo ""

# 创建目录
mkdir -p "$CHECKPOINT_DIR" logs

echo "🚀 开始增量学习..."
echo ""

# 运行增量学习
python scripts/incremental_learning.py \
  --urls-file "$URLS_FILE" \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --depth $DEPTH \
  --max-pages $MAX_PAGES \
  --learning-rate $LEARNING_RATE \
  --save-frequency $SAVE_FREQ \
  2>&1 | tee logs/incremental_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                        🎉 完成！                               ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "📁 生成的文件:"
echo "  - 模型: $CHECKPOINT_DIR/latest.pt"
echo "  - 最终模型: $CHECKPOINT_DIR/final_model.pt"
echo ""
echo "💡 优势:"
echo "  ✅ 无需保存大量中间数据 (节省3-5GB)"
echo "  ✅ 实时更新模型 (随时可用)"
echo "  ✅ 内存友好 (只处理当前网站)"
echo "  ✅ 中断安全 (模型已保存)"
echo ""
