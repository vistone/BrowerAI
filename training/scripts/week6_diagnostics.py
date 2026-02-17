#!/usr/bin/env python3
"""
Week 6 数据诊断与模型分析
目标：识别准确率低的原因并制定优化策略
"""

import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

class Week6ModelDiagnostics:
    """Week 6 模型诊断工具"""
    
    def __init__(self, data_dir='data/learning_system'):
        self.data_dir = Path(data_dir)
        self.results = {}
    
    def load_training_data(self):
        """加载训练数据"""
        logger.info("📥 加载训练数据...")
        
        try:
            # 加载样本数据
            samples_file = self.data_dir / 'samples.json'
            if samples_file.exists():
                with open(samples_file) as f:
                    samples = json.load(f)
                logger.info(f"✅ 加载 {len(samples)} 个样本")
                
                # 统计框架分布
                frameworks = {}
                for sample in samples:
                    fw = sample.get('detected_framework', 'unknown')
                    frameworks[fw] = frameworks.get(fw, 0) + 1
                
                logger.info("📊 框架分布:")
                for fw, count in sorted(frameworks.items(), key=lambda x: -x[1]):
                    percentage = (count / len(samples)) * 100
                    logger.info(f"  {fw}: {count} ({percentage:.1f}%)")
                
                return samples
        except Exception as e:
            logger.warning(f"⚠️  无法加载样本: {e}")
            return None
    
    def analyze_feature_importance(self, samples):
        """分析特征重要性"""
        logger.info("\n🔍 特征重要性分析...")
        
        if not samples or len(samples) < 10:
            logger.warning("⚠️  样本数不足，跳过特征分析")
            return
        
        # 提取特征和标签
        X, y = [], []
        for sample in samples:
            features = [
                len(sample.get('html', '')),
                sample.get('detected_framework', '').count('react'),
                sample.get('detected_framework', '').count('vue'),
                sample.get('detected_framework', '').count('angular'),
                len(str(sample.get('metadata', {})).split(',')),
            ]
            X.append(features)
            y.append(1 if sample.get('framework') else 0)
        
        if len(set(y)) < 2:
            logger.warning("⚠️  标签不均衡，跳过分析")
            return
        
        # 训练特征重要性模型
        try:
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X, y)
            
            feature_names = ['html_size', 'react_count', 'vue_count', 'angular_count', 'metadata_fields']
            importances = rf.feature_importances_
            
            feature_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances,
                'percentage': (importances / importances.sum()) * 100
            }).sort_values('importance', ascending=False)
            
            logger.info("📈 特征重要性排序:")
            for _, row in feature_df.iterrows():
                bar = '█' * int(row['percentage'] / 5)
                logger.info(f"  {row['feature']:20} {row['percentage']:6.2f}% {bar}")
            
            # 识别低重要性特征
            low_importance = feature_df[feature_df['importance'] < 0.05]
            if len(low_importance) > 0:
                logger.warning(f"⚠️  建议删除低重要性特征:")
                for _, row in low_importance.iterrows():
                    logger.warning(f"    - {row['feature']}: {row['importance']:.4f}")
            
            self.results['feature_importance'] = feature_df.to_dict(orient='records')
            
        except Exception as e:
            logger.error(f"❌ 特征分析失败: {e}")
    
    def analyze_accuracy_bottlenecks(self, samples):
        """分析准确率低的原因"""
        logger.info("\n🔴 准确率诊断 (当前: 41.67%)...")
        
        if not samples:
            return
        
        correct = 0
        incorrect = 0
        framework_accuracy = {}
        
        for sample in samples:
            fw = sample.get('detected_framework', 'unknown')
            is_correct = sample.get('framework') == fw
            
            if is_correct:
                correct += 1
            else:
                incorrect += 1
            
            if fw not in framework_accuracy:
                framework_accuracy[fw] = {'correct': 0, 'total': 0}
            
            framework_accuracy[fw]['total'] += 1
            if is_correct:
                framework_accuracy[fw]['correct'] += 1
        
        total_accuracy = (correct / (correct + incorrect)) * 100 if (correct + incorrect) > 0 else 0
        logger.info(f"整体准确率: {total_accuracy:.2f}%")
        
        logger.info("\n📊 框架级准确率:")
        for fw, stats in sorted(framework_accuracy.items(), key=lambda x: -x[1]['correct']):
            acc = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
            status = "✅" if acc > 70 else "⚠️ " if acc > 50 else "❌"
            logger.info(f"  {status} {fw:15} {acc:6.2f}% ({stats['correct']}/{stats['total']})")
        
        # 找出准确率最低的框架
        worst_frameworks = sorted(
            framework_accuracy.items(),
            key=lambda x: (x[1]['correct'] / x[1]['total']) if x[1]['total'] > 0 else 0
        )[:3]
        
        if worst_frameworks:
            logger.warning("\n🎯 需要改进的框架 (准确率最低):")
            for fw, stats in worst_frameworks:
                acc = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
                logger.warning(f"  {fw}: {acc:.2f}% (建议增加样本)")
        
        self.results['accuracy_by_framework'] = {
            fw: {
                'accuracy': (stats['correct'] / stats['total']) if stats['total'] > 0 else 0,
                'correct': stats['correct'],
                'total': stats['total']
            }
            for fw, stats in framework_accuracy.items()
        }
    
    def analyze_data_imbalance(self, samples):
        """分析数据不均衡"""
        logger.info("\n⚖️  数据均衡性分析...")
        
        if not samples:
            return
        
        # 框架分布
        fw_distribution = {}
        for sample in samples:
            fw = sample.get('detected_framework', 'unknown')
            fw_distribution[fw] = fw_distribution.get(fw, 0) + 1
        
        total = len(samples)
        min_count = min(fw_distribution.values()) if fw_distribution else 0
        max_count = max(fw_distribution.values()) if fw_distribution else 0
        
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        logger.info(f"框架分布不均衡比例: {imbalance_ratio:.2f}:1")
        
        if imbalance_ratio > 3:
            logger.warning(f"⚠️  数据严重不均衡！建议:")
            logger.warning("  1. 对多数类进行欠采样")
            logger.warning("  2. 对少数类进行过采样")
            logger.warning("  3. 使用加权损失函数")
            logger.warning("  4. 采集更多少数类样本")
        
        self.results['data_imbalance'] = {
            'imbalance_ratio': imbalance_ratio,
            'distribution': fw_distribution
        }
    
    def generate_recommendations(self):
        """生成优化建议"""
        logger.info("\n💡 Week 6 优化建议:")
        logger.info("\n🔴 关键优化 (P1 - 影响准确率):")
        logger.info("  1. 采集更多框架样本 (特别是准确率<50% 的)")
        logger.info("  2. 处理数据不均衡问题 (使用过采样/欠采样)")
        logger.info("  3. 特征工程优化 (删除低重要性特征)")
        logger.info("  4. 使用加权损失函数 (处理类别不均衡)")
        
        logger.info("\n🟡 中等优化 (P2 - 改进准确率):")
        logger.info("  1. 超参数调优 (学习率、层数、Dropout 率)")
        logger.info("  2. 数据增强 (mixup、cutout 等)")
        logger.info("  3. 集成学习 (投票、堆叠、袋装)")
        logger.info("  4. 特征交互 (多项式特征、交叉特征)")
        
        logger.info("\n🟢 其他优化 (P3 - 提升性能):")
        logger.info("  1. 模型压缩 (知识蒸馏、剪枝、量化)")
        logger.info("  2. 推理优化 (批处理、缓存、异步)")
        logger.info("  3. 规则融合 (学习规则权重)")
        logger.info("  4. 多模态学习 (HTML + JS + CSS)")
    
    def run_diagnostics(self):
        """运行完整诊断"""
        logger.info("="*80)
        logger.info("🔍 Week 6 - BrowerAI 模型诊断系统")
        logger.info("="*80)
        
        samples = self.load_training_data()
        
        if samples:
            self.analyze_feature_importance(samples)
            self.analyze_accuracy_bottlenecks(samples)
            self.analyze_data_imbalance(samples)
        
        self.generate_recommendations()
        
        logger.info("\n" + "="*80)
        logger.info("📋 诊断完成！")
        logger.info("="*80)
        
        return self.results

def main():
    """主函数"""
    diagnostics = Week6ModelDiagnostics()
    diagnostics.run_diagnostics()

if __name__ == '__main__':
    main()
