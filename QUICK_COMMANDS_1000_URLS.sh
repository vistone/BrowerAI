#!/bin/bash
# 快速命令参考 - 从 1000+ URL 库生成网站

echo "🚀 从 1000+ URL 库生成网站 - 快速命令"
echo "========================================"
echo ""

# 查看生成的网站
echo "1️⃣ 查看已生成的 100 个网站:"
echo "   ls -lh generated_websites_1000_library/ | head -20"
echo ""

# 打开第一个网站
echo "2️⃣ 在浏览器打开第一个网站:"
echo "   cd generated_websites_1000_library/website_1/"
echo "   python3 -m http.server 8000"
echo "   # 然后在浏览器访问 http://localhost:8000"
echo ""

# 查看评估报告
echo "3️⃣ 查看生成网站的评估报告:"
echo "   cat generated_websites_1000_library/evaluation_report.json | python3 -m json.tool"
echo ""

# 查看源 URL 库
echo "4️⃣ 查看 1000+ URL 库:"
echo "   wc -l training/data/large_urls.txt"
echo "   head -20 training/data/large_urls.txt"
echo ""

# 查看训练数据
echo "5️⃣ 查看训练数据样本:"
echo "   head -1 data/website_training_1000_generated.jsonl | python3 -m json.tool | head -30"
echo ""

# 扩展到更多网站
echo "6️⃣ 扩展 - 生成 500 个网站 (需要修改脚本中的 limit 参数):"
echo "   # 编辑 training/generate_from_1000_urls.py，将 limit=200 改为 limit=500"
echo "   python3 training/generate_from_1000_urls.py"
echo "   python3 training/large_scale_website_trainer.py \\"
echo "       --data-file data/website_training_1000_generated.jsonl \\"
echo "       --epochs 50 \\"
echo "       --batch-size 8 \\"
echo "       --output-dir checkpoints/website_generator_1000_library_v2"
echo "   python3 training/evaluate_generated_websites.py \\"
echo "       --model-path checkpoints/website_generator_1000_library_v2/checkpoint_epoch_50.pt \\"
echo "       --num-websites 500"
echo ""

# 统计生成的网站
echo "7️⃣ 统计生成的网站:"
echo "   find generated_websites_1000_library -name 'index.html' | wc -l"
echo ""

# 检查生成网站的多样性
echo "8️⃣ 查看不同网站的 HTML 头部 (检查多样性):"
echo "   for i in 1 5 10 15 20; do"
echo "     echo \"=== website_\$i ===\""
echo "     head -15 generated_websites_1000_library/website_\$i/index.html | tail -5"
echo "   done"
echo ""

# 查看统计信息
echo "9️⃣ 查看训练统计:"
echo "   cat training_1000_log.txt | tail -20"
echo ""

# 验证代码质量
echo "🔟 验证所有网站代码质量:"
echo "   cat generated_websites_1000_library/evaluation_report.json | python3 -c \\"
echo "       \"import sys, json; data = json.load(sys.stdin); \\"
echo "       print(f'总网站数: {len(data[\\\"websites\\\"])}'); \\"
echo "       print(f'HTML 平均质量: {data[\\\"average_html_quality\\\"]:.1%}'); \\"
echo "       print(f'CSS 平均质量: {data[\\\"average_css_quality\\\"]:.1%}'); \\"
echo "       print(f'JS 平均质量: {data[\\\"average_js_quality\\\"]:.1%}')\""
echo ""

echo "📊 文件位置总结:"
echo "  • 1000+ URL 库: training/data/large_urls.txt (1,018 个 URLs)"
echo "  • 数据生成脚本: training/generate_from_1000_urls.py"
echo "  • 训练数据: data/website_training_1000_generated.jsonl (200 个样本)"
echo "  • 训练模型: checkpoints/website_generator_1000_library_v1/"
echo "  • 生成网站: generated_websites_1000_library/ (100 个网站)"
echo "  • 完整报告: LEARNING_FROM_1000_URLS_REPORT.md"
echo ""

echo "✅ 系统已准备好用 1000+ URL 库训练和生成网站!"
