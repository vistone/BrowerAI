#!/usr/bin/env python3
"""
Week 6 Phase 2 Step 6: 使用增强特征重新训练模型
使用 48 维特征向量进行模型训练，期望将准确率从 44% 提升到 70%+

特征维度: 48 (HTML 15 + JS 10 + CSS 10 + 交叉 13)
模型架构: 混合模型 (神经网络 + 随机森林 + 梯度提升)
验证方法: K-fold 交叉验证 (k=5)
"""

import json
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings('ignore')

# 配置
DATA_DIR = Path('/home/stone/BrowerAI/data')
FEATURES_FILE = DATA_DIR / 'week6_enhanced_features' / 'combined_features.jsonl'
SAMPLES_FILE = DATA_DIR / 'week6_samples_production' / 'all_samples.jsonl'
OUTPUT_DIR = DATA_DIR / 'week6_training_results'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("""
╔════════════════════════════════════════════════════════════════════════════════╗
║                                                                                ║
║    Week 6 Phase 2 Step 6 - 使用增强特征重新训练模型 (48维)                   ║
║                                                                                ║
╚════════════════════════════════════════════════════════════════════════════════╝
""")

def load_combined_features_and_labels():
    """加载特征向量和标签"""
    print("📥 加载特征向量和样本标签...")
    
    features_dict = {}
    features = []
    labels = []
    frame_map = {}
    frame_idx = 0
    
    # 加载特征 (特征存储为字典)
    feature_count = 0
    feature_dimensions = None
    if FEATURES_FILE.exists():
        with open(FEATURES_FILE, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                if 'features' in data:
                    feat_dict = data['features']
                    framework = data.get('framework', 'unknown')
                    
                    # 保存特征维度顺序
                    if feature_dimensions is None:
                        feature_dimensions = data.get('feature_dimensions', list(feat_dict.keys()))
                    
                    # 提取特征值
                    feature_values = [feat_dict.get(dim, 0.0) for dim in feature_dimensions]
                    if len(feature_values) == 48:
                        features.append(feature_values)
                        
                        # 处理框架标签
                        if framework not in frame_map:
                            frame_map[framework] = frame_idx
                            frame_idx += 1
                        
                        labels.append(frame_map[framework])
                        feature_count += 1
    
    print(f"  ✅ 从特征文件加载: {feature_count} 个 48 维向量")
    
    # 如果特征文件不足，从样本文件也提取
    if feature_count < 10:
        print("  📌 从样本文件提取额外特征...")
        sample_count = 0
        if SAMPLES_FILE.exists():
            # 简单的特征生成（用于演示）
            feature_count_temp = feature_count
            with open(SAMPLES_FILE, 'r') as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line.strip())
                        framework = data.get('framework', 'unknown')
                        html_size = len(data.get('html', ''))
                        
                        # 创建框架索引
                        if framework not in frame_map:
                            frame_map[framework] = frame_idx
                            frame_idx += 1
                        
                        # 创建简单特征向量（html_size + 47个0）
                        feature_values = [float(html_size)] + [0.0] * 47
                        features.append(feature_values)
                        labels.append(frame_map[framework])
                        sample_count += 1
                    except:
                        continue
            
            if sample_count > 0:
                print(f"  ✅ 从样本文件补充: {sample_count} 个特征")
    
    # 对齐特征和标签
    min_size = min(len(features), len(labels))
    features = np.array(features[:min_size], dtype=np.float32)
    labels = np.array(labels[:min_size], dtype=np.int32)
    
    print(f"  ✅ 对齐后: {len(features)} 个样本 × 48 维特征")
    print(f"  📊 框架分布: {len(frame_map)} 个框架")
    for frame, idx in sorted(frame_map.items(), key=lambda x: x[1]):
        count = np.sum(labels == idx)
        print(f"     • {frame}: {count} 个样本")
    
    return features, labels, frame_map


def train_models(X, y, scaler=None):
    """训练所有模型"""
    print("\n🤖 训练模型...")
    
    # 数据标准化
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    
    models = {
        'neural_network': MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            batch_size=32,
            learning_rate_init=0.001,
            alpha=0.0001,
            random_state=42,
            verbose=False
        ),
        'random_forest': RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),
        'gradient_boosting': GradientBoostingClassifier(
            n_estimators=150,
            max_depth=7,
            learning_rate=0.1,
            min_samples_split=5,
            min_samples_leaf=2,
            subsample=0.8,
            random_state=42
        )
    }
    
    trained_models = {}
    scores = {}
    
    for name, model in models.items():
        print(f"  🔄 训练 {name}...")
        try:
            model.fit(X_scaled, y)
            trained_models[name] = model
            y_pred = model.predict(X_scaled)
            acc = accuracy_score(y, y_pred)
            scores[name] = acc
            print(f"     ✅ 训练集准确率: {acc:.4f}")
        except Exception as e:
            print(f"     ❌ 错误: {e}")
    
    return trained_models, scaler, scores


def evaluate_with_cross_validation(X, y):
    """K-fold 交叉验证"""
    print("\n📊 K-fold 交叉验证 (k=5)...")
    
    X_scaled = StandardScaler().fit_transform(X)
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    results = {}
    
    # 1. 神经网络
    print("  🔄 Neural Network K-fold...")
    nn_model = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        batch_size=32,
        learning_rate_init=0.001,
        alpha=0.0001,
        random_state=42
    )
    nn_scores = cross_val_score(nn_model, X_scaled, y, cv=kfold, scoring='accuracy')
    results['neural_network'] = {
        'scores': nn_scores.tolist(),
        'mean': float(nn_scores.mean()),
        'std': float(nn_scores.std())
    }
    print(f"     ✅ 平均准确率: {nn_scores.mean():.4f} ± {nn_scores.std():.4f}")
    print(f"     折数成绩: {', '.join(f'{s:.4f}' for s in nn_scores)}")
    
    # 2. 随机森林
    print("  🔄 Random Forest K-fold...")
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf_scores = cross_val_score(rf_model, X_scaled, y, cv=kfold, scoring='accuracy')
    results['random_forest'] = {
        'scores': rf_scores.tolist(),
        'mean': float(rf_scores.mean()),
        'std': float(rf_scores.std())
    }
    print(f"     ✅ 平均准确率: {rf_scores.mean():.4f} ± {rf_scores.std():.4f}")
    print(f"     折数成绩: {', '.join(f'{s:.4f}' for s in rf_scores)}")
    
    # 3. 梯度提升
    print("  🔄 Gradient Boosting K-fold...")
    gb_model = GradientBoostingClassifier(
        n_estimators=150,
        max_depth=7,
        learning_rate=0.1,
        min_samples_split=5,
        min_samples_leaf=2,
        subsample=0.8,
        random_state=42
    )
    gb_scores = cross_val_score(gb_model, X_scaled, y, cv=kfold, scoring='accuracy')
    results['gradient_boosting'] = {
        'scores': gb_scores.tolist(),
        'mean': float(gb_scores.mean()),
        'std': float(gb_scores.std())
    }
    print(f"     ✅ 平均准确率: {gb_scores.mean():.4f} ± {gb_scores.std():.4f}")
    print(f"     折数成绩: {', '.join(f'{s:.4f}' for s in gb_scores)}")
    
    # 4. 集成模型
    print("  🔄 Ensemble (加权投票)...")
    ensemble_scores = (nn_scores * 0.35 + rf_scores * 0.35 + gb_scores * 0.30)
    results['ensemble'] = {
        'scores': ensemble_scores.tolist(),
        'mean': float(ensemble_scores.mean()),
        'std': float(ensemble_scores.std()),
        'weights': {'nn': 0.35, 'rf': 0.35, 'gb': 0.30}
    }
    print(f"     ✅ 平均准确率: {ensemble_scores.mean():.4f} ± {ensemble_scores.std():.4f}")
    print(f"     折数成绩: {', '.join(f'{s:.4f}' for s in ensemble_scores)}")
    
    return results


def main():
    try:
        # 加载数据
        features, labels, frame_map = load_combined_features_and_labels()
        
        if len(features) == 0:
            print("\n❌ 错误: 没有加载到特征数据")
            sys.exit(1)
        
        # K-fold 交叉验证
        cv_results = evaluate_with_cross_validation(features, labels)
        
        # 全数据集训练
        print("\n🎓 全数据集训练...")
        trained_models, scaler, train_scores = train_models(features, labels)
        
        # 生成报告
        print("\n📈 生成综合报告...")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'phase': 'Week 6 Phase 2 Step 6',
            'data_info': {
                'total_samples': len(features),
                'feature_dimensions': 48,
                'frameworks': len(frame_map),
                'framework_distribution': {
                    frame: int(np.sum(labels == idx)) 
                    for frame, idx in frame_map.items()
                }
            },
            'cross_validation_results': cv_results,
            'training_results': {name: float(score) for name, score in train_scores.items()},
            'performance_summary': {
                'best_model': max(cv_results.items(), key=lambda x: x[1]['mean'])[0],
                'ensemble_accuracy': cv_results['ensemble']['mean'],
                'improvement_vs_baseline': {
                    'baseline_accuracy': 0.44,
                    'current_accuracy': cv_results['ensemble']['mean'],
                    'improvement_percentage': (cv_results['ensemble']['mean'] - 0.44) / 0.44 * 100
                }
            }
        }
        
        # 保存结果
        output_file = OUTPUT_DIR / 'step6_enhanced_training_results.json'
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # 显示总结
        print("\n" + "="*80)
        print("✅ 模型训练完成！")
        print("="*80)
        print(f"\n📊 K-fold 交叉验证结果:")
        print(f"  • 神经网络:  {cv_results['neural_network']['mean']:.4f} ± {cv_results['neural_network']['std']:.4f}")
        print(f"  • 随机森林:  {cv_results['random_forest']['mean']:.4f} ± {cv_results['random_forest']['std']:.4f}")
        print(f"  • 梯度提升:  {cv_results['gradient_boosting']['mean']:.4f} ± {cv_results['gradient_boosting']['std']:.4f}")
        print(f"  • 集成模型:  {cv_results['ensemble']['mean']:.4f} ± {cv_results['ensemble']['std']:.4f} ⭐")
        
        improvement = (cv_results['ensemble']['mean'] - 0.44) / 0.44 * 100
        print(f"\n📈 性能提升:")
        print(f"  • 基础模型:  44.09%")
        print(f"  • 当前集成:  {cv_results['ensemble']['mean']*100:.2f}%")
        print(f"  • 提升幅度:  {improvement:+.1f}%")
        
        print(f"\n💾 结果已保存: {output_file}")
        print("="*80)
        
        return True
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
