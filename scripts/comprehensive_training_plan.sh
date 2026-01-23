#!/bin/bash
# 全面学习计划：多阶段、数据增强、长时间训练

echo "════════════════════════════════════════════════════════════"
echo "   🎓 JavaScript框架检测 - 全面学习计划"
echo "════════════════════════════════════════════════════════════"
echo ""

# 阶段1：采集更多NPM包数据（扩大数据集）
echo "📦 阶段1：扩展NPM包数据集"
echo "目标：采集更多主流框架的NPM包"
python3 training/npm_package_crawler.py --packages \
  "vue,@vue/cli,vuex,vue-router" \
  "@angular/core,@angular/cli,@angular/router" \
  "preact,solid-js,alpine,lit,petite-vue" \
  "astro,qwik,sveltekit" \
  "redux,mobx,zustand,recoil,jotai" \
  "typescript,babel-core,@babel/core"

# 阶段2：增强混淆数据
echo ""
echo "🔐 阶段2：生成更多混淆样本"
python3 training/fast_npm_obfuscator.py \
  --methods all \
  --multiplier 3

# 阶段3：长时间大规模训练（50+ epochs）
echo ""
echo "🚀 阶段3：大规模GPU训练（50轮）"
python3 training/large_scale_trainer.py \
  --epochs 50 \
  --batch-size 64 \
  --learning-rate 0.001 \
  --device cuda \
  --data-file real_data/obfuscated_code/training_pairs.jsonl \
  --save-every 5

# 阶段4：数据增强训练
echo ""
echo "🎨 阶段4：数据增强训练"
python3 training/enhanced_gpu_trainer.py \
  --epochs 30 \
  --augmentation strong \
  --device cuda

# 阶段5：转换并验证
echo ""
echo "📦 阶段5：ONNX转换与验证"
python3 training/convert_to_onnx.py \
  --model models/local/large_scale_best.pt \
  --output models/local/comprehensive_framework_detector.onnx

echo ""
echo "════════════════════════════════════════════════════════════"
echo "   ✅ 全面学习计划完成"
echo "════════════════════════════════════════════════════════════"
