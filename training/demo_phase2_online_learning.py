#!/usr/bin/env python3
"""
Phase 2: Complete Online Learning Demonstration
完整的在线学习演示 - 展示从反馈到模型更新的完整流程

演示场景:
1. 单网页学习反馈
2. 批量反馈处理 (模拟多个网页)
3. 10轮迭代学习
4. 学习曲线分析
5. 模型拟合度评估
"""

import sys
sys.path.insert(0, '/home/stone/BrowerAI/training')

import numpy as np
from phase2_online_learning import OnlineLearningSystem
import json
from datetime import datetime


def demo_single_feedback():
    """演示场景 1: 处理单个网页的学习反馈"""
    print("\n" + "="*70)
    print("Demo 1: Single Website Learning Feedback")
    print("="*70)
    
    system = OnlineLearningSystem(feature_dim=48, latent_dim=256, learning_rate=0.001)
    
    # 模拟单个网页反馈
    print("\n[Scenario] Processing feedback for Blog Website")
    print("  HTML: 2,456 chars")
    print("  CSS selectors: 45")
    print("  JavaScript functions: 12")
    
    # 模拟特征提取
    original_features = np.array([
        # HTML metrics (10)
        0.8, 0.7, 0.6, 0.5, 0.4,  # 深度, 节点数, 嵌套等
        0.3, 0.4, 0.5, 0.6, 0.7,
        # CSS metrics (8)
        0.6, 0.7, 0.5, 0.4, 0.3, 0.2, 0.4, 0.5,
        # JS metrics (10)
        0.5, 0.6, 0.7, 0.4, 0.3, 0.8, 0.5, 0.4, 0.3, 0.2,
        # Page structure (8)
        0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.8,
        # Design style (7)
        0.6, 0.7, 0.5, 0.4, 0.3, 0.6, 0.7,
        # Complexity (5)
        0.4, 0.3, 0.2, 0.1, 0.5,
    ])
    
    # 生成的特征与原始特征略有偏差
    generated_features = original_features + np.random.randn(48) * 0.05
    
    # 模拟潜在向量
    latent_vector = np.random.randn(256)
    
    # 初始权重
    weights = np.random.randn(48, 256) * 0.01
    
    # 质量评分 (用户评价: 80% 质量)
    quality_score = 0.80
    
    print(f"  Quality score provided by user: {quality_score:.1%}")
    
    # 处理反馈
    result = system.process_feedback(
        original_features, generated_features, latent_vector,
        weights, quality_score, session_id="blog_001"
    )
    
    lr = result['learning_record']
    print(f"\n✓ Feedback processed successfully")
    print(f"  Reconstruction loss: {lr['loss']['reconstruction_loss']:.4f}")
    print(f"  Quality loss: {lr['loss']['quality_loss']:.4f}")
    print(f"  Regularization loss: {lr['loss']['regularization_loss']:.4f}")
    print(f"  Total loss: {lr['loss']['total_loss']:.4f}")
    print(f"  Gradient norm: {lr['gradient_info']['gradient_norm']:.4f}")
    print(f"  Weight change: {result['weight_change_norm']:.4f}")
    print(f"  Processing time: {lr['processing_time_ms']:.2f}ms")


def demo_batch_feedback():
    """演示场景 2: 批量处理多个网页的反馈"""
    print("\n" + "="*70)
    print("Demo 2: Batch Feedback Processing (3 Websites)")
    print("="*70)
    
    system = OnlineLearningSystem(feature_dim=48, latent_dim=256, learning_rate=0.001)
    weights = np.random.randn(48, 256) * 0.01
    
    websites = [
        {"name": "Tech Blog", "quality": 0.85, "chars": 3241},
        {"name": "E-commerce Store", "quality": 0.72, "chars": 5123},
        {"name": "Portfolio Site", "quality": 0.90, "chars": 1843},
    ]
    
    print(f"\nProcessing {len(websites)} websites:")
    
    total_loss = 0.0
    for i, site in enumerate(websites, 1):
        original = np.random.rand(48)
        generated = original + np.random.randn(48) * 0.08
        latent = np.random.randn(256)
        
        result = system.process_feedback(
            original, generated, latent, weights,
            site['quality'], session_id=f"batch_{i}"
        )
        
        loss = result['learning_record']['loss']['total_loss']
        total_loss += loss
        weights = result['updated_weights']
        
        print(f"  [{i}] {site['name']:<20} Quality: {site['quality']:.0%}  Loss: {loss:.4f}")
    
    avg_loss = total_loss / len(websites)
    print(f"\n✓ Batch processing complete")
    print(f"  Average loss: {avg_loss:.4f}")
    print(f"  Total system updates: {system.total_updates}")


def demo_iterative_learning():
    """演示场景 3: 多轮迭代学习 (模拟学习改进过程)"""
    print("\n" + "="*70)
    print("Demo 3: 10-Round Iterative Learning")
    print("="*70)
    
    system = OnlineLearningSystem(feature_dim=48, latent_dim=256, learning_rate=0.001)
    weights = np.random.randn(48, 256) * 0.01
    
    np.random.seed(42)
    print("\nRound | Avg Loss | Quality Score | Weight Change | Gradient Norm")
    print("-" * 70)
    
    for round_num in range(1, 11):
        # 随机生成反馈
        num_feedbacks = 3
        round_loss = 0.0
        round_grad_norm = 0.0
        
        for j in range(num_feedbacks):
            original = np.random.rand(48)
            generated = original + np.random.randn(48) * (0.12 - round_num * 0.005)
            latent = np.random.randn(256)
            # 质量逐轮改进
            quality = 0.65 + (round_num - 1) * 0.02 + np.random.rand() * 0.05
            
            result = system.process_feedback(
                original, generated, latent, weights,
                quality, session_id=f"iter_{round_num}_{j}"
            )
            
            round_loss += result['learning_record']['loss']['total_loss']
            round_grad_norm += result['learning_record']['gradient_info']['gradient_norm']
            weights = result['updated_weights']
        
        avg_loss = round_loss / num_feedbacks
        avg_grad = round_grad_norm / num_feedbacks
        
        summary = system.get_learning_summary()
        weight_change = np.linalg.norm(weights - np.random.randn(48, 256) * 0.01)
        
        print(f"{round_num:2d}   | {avg_loss:8.4f} | {quality:13.1%} | {weight_change:13.4f} | {avg_grad:11.4f}")
    
    # 最终统计
    print("-" * 70)
    final_summary = system.get_learning_summary()
    print(f"\n✓ Iterative learning complete")
    print(f"  Total iterations: {final_summary['total_updates']}")
    print(f"  Final loss: {final_summary['latest_loss']:.4f}")
    print(f"  Loss trend: {final_summary['loss_trend']}")
    print(f"  Optimizer timesteps: {final_summary['optimizer_timesteps']}")


def demo_learning_curves():
    """演示场景 4: 学习曲线分析"""
    print("\n" + "="*70)
    print("Demo 4: Learning Curves Analysis")
    print("="*70)
    
    system = OnlineLearningSystem(feature_dim=48, latent_dim=256, learning_rate=0.001)
    weights = np.random.randn(48, 256) * 0.01
    
    np.random.seed(42)
    print("\nGenerating learning data (50 iterations)...")
    
    for i in range(50):
        original = np.random.rand(48)
        # 生成误差逐渐减小
        noise_level = max(0.01, 0.15 - i * 0.002)
        generated = original + np.random.randn(48) * noise_level
        latent = np.random.randn(256)
        quality = 0.6 + (i / 50) * 0.35 + np.random.rand() * 0.05
        
        result = system.process_feedback(
            original, generated, latent, weights, quality, session_id=f"curve_{i}"
        )
        weights = result['updated_weights']
    
    # 分析学习曲线
    loss_history = list(system.loss_history)
    grad_history = list(system.gradient_norm_history)
    
    print("\n✓ Learning curves generated")
    
    # 分段分析
    segments = {
        'Early (0-10)': loss_history[:10],
        'Mid (10-30)': loss_history[10:30],
        'Late (30-50)': loss_history[30:50],
    }
    
    print("\nLoss Statistics by Phase:")
    print("  Phase      | Mean Loss | Std Dev | Min  | Max")
    print("  " + "-" * 47)
    for phase, losses in segments.items():
        if losses:
            mean = np.mean(losses)
            std = np.std(losses)
            min_l = np.min(losses)
            max_l = np.max(losses)
            print(f"  {phase:<10} | {mean:9.4f} | {std:7.4f} | {min_l:4.4f} | {max_l:4.4f}")
    
    print("\nGradient Norm Statistics:")
    print(f"  Mean: {np.mean(grad_history):.4f}")
    print(f"  Std:  {np.std(grad_history):.4f}")
    print(f"  Min:  {np.min(grad_history):.4f}")
    print(f"  Max:  {np.max(grad_history):.4f}")
    
    # 收敛检查
    if len(loss_history) > 20:
        early_avg = np.mean(loss_history[:10])
        late_avg = np.mean(loss_history[-10:])
        improvement = (early_avg - late_avg) / early_avg * 100
        print(f"\nConvergence Analysis:")
        print(f"  Early average loss: {early_avg:.4f}")
        print(f"  Late average loss:  {late_avg:.4f}")
        print(f"  Improvement: {improvement:.1f}%")


def demo_model_quality():
    """演示场景 5: 模型拟合度评估"""
    print("\n" + "="*70)
    print("Demo 5: Model Fitting Quality Assessment")
    print("="*70)
    
    system = OnlineLearningSystem(feature_dim=48, latent_dim=256, learning_rate=0.001)
    weights = np.random.randn(48, 256) * 0.01
    
    np.random.seed(42)
    print("\nTraining model on synthetic website dataset...")
    
    # 训练数据
    num_samples = 30
    for i in range(num_samples):
        original = np.random.rand(48)
        generated = original + np.random.randn(48) * 0.08
        latent = np.random.randn(256)
        quality = np.clip(0.7 + np.random.randn() * 0.15, 0.5, 0.95)
        
        system.process_feedback(original, generated, latent, weights, quality, session_id=f"model_{i}")
        weights = weights + np.random.randn(48, 256) * 0.0001
    
    # 评估指标
    print(f"\n✓ Training complete")
    
    summary = system.get_learning_summary()
    detailed = system.get_detailed_metrics()
    
    print("\nModel Fitting Metrics:")
    print(f"  Total training samples: {num_samples}")
    print(f"  Total parameter updates: {summary['total_updates']}")
    print(f"  Final loss: {summary['latest_loss']:.4f}")
    print(f"  Average loss: {summary['average_loss']:.4f}")
    print(f"  Loss std deviation: {np.std(list(system.loss_history)):.4f}")
    
    print("\nComponentwise Loss Analysis:")
    print(f"  Reconstruction loss: {detailed['loss_reconstruction']:.4f}")
    print(f"  Quality loss: {detailed['loss_quality']:.4f}")
    
    print("\nOptimizer Statistics:")
    print(f"  Learning rate: {summary['learning_rate']:.6f}")
    print(f"  Optimizer timesteps: {summary['optimizer_timesteps']}")
    print(f"  Average gradient norm: {summary['average_gradient_norm']:.4f}")
    print(f"  Max gradient norm: {summary['max_gradient_norm']:.4f}")
    
    # 质量评估
    loss_improvement = ((list(system.loss_history)[0] - list(system.loss_history)[-1]) / 
                        max(list(system.loss_history)[0], 1e-8) * 100)
    
    print("\nModel Quality Assessment:")
    if loss_improvement > 30:
        status = "✓ Excellent convergence"
    elif loss_improvement > 10:
        status = "✓ Good convergence"
    else:
        status = "⚠ Moderate convergence"
    
    print(f"  {status}")
    print(f"  Loss improvement: {loss_improvement:.1f}%")


def main():
    """运行所有演示"""
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + " "*15 + "Phase 2: Online Learning System" + " "*23 + "█")
    print("█" + " "*18 + "Complete Demonstration Suite" + " "*25 + "█")
    print("█" + " "*68 + "█")
    print("█"*70)
    
    demo_single_feedback()
    demo_batch_feedback()
    demo_iterative_learning()
    demo_learning_curves()
    demo_model_quality()
    
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + " "*20 + "✅ All demonstrations complete!" + " "*17 + "█")
    print("█" + " "*68 + "█")
    print("█"*70 + "\n")


if __name__ == '__main__':
    main()
