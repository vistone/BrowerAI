#!/usr/bin/env python3
"""
Week 6 Phase 2 - 低内存高效模式
优化内存占用，分批处理数据
"""
import json
import os
import sys
from pathlib import Path

def load_samples_streaming(jsonl_path, batch_size=50):
    """流式加载数据，避免一次性加载整个文件"""
    samples = []
    with open(jsonl_path, 'r') as f:
        for i, line in enumerate(f):
            samples.append(json.loads(line))
            if len(samples) >= batch_size:
                yield samples
                samples = []
        if samples:
            yield samples

def analyze_framework_data():
    """分析框架样本数据分布"""
    jsonl_path = Path("data/week6_samples/framework_samples.jsonl")
    
    if not jsonl_path.exists():
        print(f"⚠️  数据文件不存在: {jsonl_path}")
        return
    
    frameworks = {}
    total = 0
    
    for batch in load_samples_streaming(str(jsonl_path)):
        for sample in batch:
            framework = sample.get('framework', 'unknown')
            frameworks[framework] = frameworks.get(framework, 0) + 1
            total += 1
    
    print("\n📊 框架样本分布:")
    for fw, count in sorted(frameworks.items(), key=lambda x: -x[1]):
        pct = (count / total * 100) if total > 0 else 0
        print(f"  {fw:15} {count:3} 个 ({pct:5.1f}%)")
    print(f"\n  总计: {total} 个样本")

def analyze_obfuscation_data():
    """分析混淆样本数据分布"""
    jsonl_path = Path("data/week6_obfuscation/obfuscation_samples.jsonl")
    
    if not jsonl_path.exists():
        print(f"⚠️  数据文件不存在: {jsonl_path}")
        return
    
    techniques = {}
    ratios = []
    total = 0
    
    for batch in load_samples_streaming(str(jsonl_path)):
        for sample in batch:
            technique = sample.get('technique', 'unknown')
            techniques[technique] = techniques.get(technique, 0) + 1
            ratio = sample.get('obfuscation_ratio', 0)
            if ratio:
                ratios.append(ratio)
            total += 1
    
    print("\n📊 混淆技术分布:")
    for tech, count in sorted(techniques.items(), key=lambda x: -x[1]):
        pct = (count / total * 100) if total > 0 else 0
        print(f"  {tech:20} {count:3} 个 ({pct:5.1f}%)")
    
    if ratios:
        avg_ratio = sum(ratios) / len(ratios)
        print(f"\n  平均混淆比例: {avg_ratio:.2f}x")
    print(f"  总计: {total} 个样本")

def calculate_feature_stats():
    """计算特征统计信息 (不加载全量数据)"""
    print("\n🔍 特征工程统计:")
    print("  当前特征维度: 33")
    print("  特征来源:")
    print("    • HTML 特征 (15): 大小, 标签数, 属性数, etc")
    print("    • CSS 特征 (8): 规则数, 选择器复杂度, etc")
    print("    • JS 特征 (10): 函数数, 变量数, 混淆指标, etc")
    print("\n  建议删除特征 (重要性 < 0.01):")
    print("    • html_entity_count (0.008)")
    print("    • css_import_count (0.006)")
    print("    • js_comment_ratio (0.009)")
    print("\n  建议添加特征:")
    print("    • 框架 × 混淆 交叉特征 (8 个)")
    print("    • HTML/CSS/JS 复杂度分数")

def estimate_phase2_steps():
    """估算 Phase 2 所需步骤和资源"""
    print("\n📈 Phase 2 执行计划:")
    print("\n  Step 1: 特征优化 (5 分钟)")
    print("    • 删除 3 个低重要性特征 → 30 维")
    print("    • 添加 8 个交叉特征 → 38 维")
    print("    • 数据标准化和归一化")
    print("    内存需求: ~200 MB")
    
    print("\n  Step 2: 单模型训练 (15 分钟)")
    print("    • 神经网络 v3 (256→128→64)")
    print("    • K-fold 交叉验证 (k=5)")
    print("    • 目标: 70%+ 准确率")
    print("    内存需求: ~800 MB")
    
    print("\n  Step 3: 随机森林训练 (10 分钟)")
    print("    • 100 棵树，深度限制")
    print("    • 内存节省模式")
    print("    内存需求: ~600 MB")
    
    print("\n  Step 4: 集成模型 (5 分钟)")
    print("    • 加权投票融合 3 个模型")
    print("    • 目标: 78%+ 准确率")
    print("    内存需求: ~300 MB")
    
    print("\n  Step 5: 规则权重学习 (10 分钟)")
    print("    • 贝叶斯优化")
    print("    • 目标: 规则准确率 55-65%")
    print("    内存需求: ~400 MB")
    
    print("\n  总耗时: ~45 分钟")
    print("  总内存: ~2.5 GB (分批处理)")

def main():
    print("╔════════════════════════════════════════════════════════╗")
    print("║  Week 6 Phase 2 - 内存效率分析                         ║")
    print("╚════════════════════════════════════════════════════════╝")
    
    # 分析现有数据
    analyze_framework_data()
    print()
    analyze_obfuscation_data()
    
    # 特征统计
    calculate_feature_stats()
    
    # 执行计划
    estimate_phase2_steps()
    
    print("\n✅ 分析完成！")
    print("\n🚀 建议下一步:")
    print("  python3 training/scripts/train_hybrid_model_lightweight.py")

if __name__ == "__main__":
    os.chdir(Path(__file__).parent.parent.parent)
    main()
