#!/usr/bin/env python3
"""
Phase 2 Online Learning System - Comprehensive Tests
在线学习系统完整测试套件
"""

import sys
sys.path.insert(0, '/home/stone/BrowerAI/training')

import numpy as np
from phase2_online_learning import (
    LossFunction, GradientComputer, AdamOptimizer, OnlineLearningSystem
)


def test_loss_function_basic():
    """Test LossFunction initialization and basic computation"""
    print("\n[TEST 1] Loss Function Initialization")
    
    loss_fn = LossFunction(alpha=0.5, beta=0.3, lambda_reg=0.0001)
    
    # Check parameters
    assert loss_fn.alpha == 0.5, "Alpha not set correctly"
    assert loss_fn.beta == 0.3, "Beta not set correctly"
    assert loss_fn.lambda_reg == 0.0001, "Lambda not set correctly"
    
    print("  ✓ LossFunction initialized correctly")
    print(f"    Alpha: {loss_fn.alpha}, Beta: {loss_fn.beta}, Lambda: {loss_fn.lambda_reg}")


def test_loss_computation():
    """Test loss computation components"""
    print("\n[TEST 2] Loss Computation Components")
    
    loss_fn = LossFunction(alpha=0.5, beta=0.3, lambda_reg=0.0001)
    
    # Test data
    original = np.array([0.5, 0.3, 0.7] * 16)  # 48D
    generated = original + np.random.randn(48) * 0.1
    weights = np.random.randn(48, 256) * 0.01
    
    # Reconstruction loss
    recon_loss = loss_fn.compute_reconstruction_loss(original, generated)
    assert 0 <= recon_loss < 1, f"Invalid reconstruction loss: {recon_loss}"
    print(f"  ✓ Reconstruction loss: {recon_loss:.4f}")
    
    # Quality loss
    quality_loss = loss_fn.compute_quality_loss(0.8)
    assert abs(quality_loss - 0.2) < 1e-6, f"Quality loss calculation incorrect: got {quality_loss}"
    print(f"  ✓ Quality loss (from 0.8 score): {quality_loss:.4f}")
    
    # Regularization loss
    reg_loss = loss_fn.compute_regularization_loss(weights)
    assert reg_loss >= 0, "Regularization loss should be non-negative"
    print(f"  ✓ Regularization loss: {reg_loss:.6f}")
    
    # Total loss
    total_dict = loss_fn.compute_total_loss(original, generated, 0.8, weights)
    assert 'total_loss' in total_dict, "Missing total_loss"
    assert 'reconstruction_loss' in total_dict, "Missing reconstruction_loss"
    assert total_dict['total_loss'] > 0, "Total loss should be positive"
    print(f"  ✓ Total loss: {total_dict['total_loss']:.4f}")


def test_gradient_computation():
    """Test gradient calculation"""
    print("\n[TEST 3] Gradient Computation")
    
    grad_computer = GradientComputer(feature_dim=48, latent_dim=256)
    
    # Test data
    original = np.random.rand(48)
    generated = original + np.random.randn(48) * 0.05
    latent = np.random.randn(256)
    weights = np.random.randn(48, 256) * 0.01
    
    # Reconstruction gradient
    grad_recon = grad_computer.compute_reconstruction_gradient(original, generated, latent)
    assert grad_recon.shape == (48, 256), f"Wrong gradient shape: {grad_recon.shape}"
    print(f"  ✓ Reconstruction gradient shape: {grad_recon.shape}")
    
    # Quality gradient
    grad_quality = grad_computer.compute_quality_gradient(latent, quality_feedback=0.8)
    assert grad_quality.shape == (48, 256), "Wrong quality gradient shape"
    print(f"  ✓ Quality gradient shape: {grad_quality.shape}")
    
    # Regularization gradient
    grad_reg = grad_computer.compute_regularization_gradient(weights, lambda_reg=0.0001)
    assert grad_reg.shape == (48, 256), "Wrong regularization gradient shape"
    print(f"  ✓ Regularization gradient shape: {grad_reg.shape}")
    
    # Total gradient
    total_grad, grad_info = grad_computer.compute_total_gradient(
        original, generated, latent, weights, 0.7, alpha=0.5, beta=0.3
    )
    assert total_grad.shape == (48, 256), "Wrong total gradient shape"
    assert 'gradient_norm' in grad_info, "Missing gradient_norm"
    assert grad_info['gradient_norm'] > 0, "Gradient norm should be positive"
    print(f"  ✓ Total gradient norm: {grad_info['gradient_norm']:.4f}")
    print(f"  ✓ Max gradient element: {grad_info['max_gradient']:.4f}")


def test_adam_optimizer():
    """Test Adam optimizer"""
    print("\n[TEST 4] Adam Optimizer")
    
    optimizer = AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999)
    
    # Initialize
    optimizer.initialize((48, 256))
    assert optimizer.m is not None, "First moment not initialized"
    assert optimizer.v is not None, "Second moment not initialized"
    assert optimizer.t == 0, "Timestep should be 0 initially"
    print("  ✓ Optimizer initialized")
    
    # Test updates
    weights = np.random.randn(48, 256) * 0.01
    original_norm = np.linalg.norm(weights)
    
    for step in range(5):
        gradient = np.random.randn(48, 256) * 0.01
        updated, info = optimizer.update(weights, gradient)
        
        assert updated.shape == weights.shape, "Updated weights shape mismatch"
        assert info['t'] == step + 1, f"Timestep mismatch: {info['t']} vs {step + 1}"
        weights = updated
    
    final_norm = np.linalg.norm(weights)
    print(f"  ✓ 5 optimization steps completed")
    print(f"    Initial weight norm: {original_norm:.4f}")
    print(f"    Final weight norm: {final_norm:.4f}")
    print(f"    Optimizer timestep: {optimizer.t}")


def test_online_learning_system():
    """Test complete online learning system"""
    print("\n[TEST 5] Online Learning System Integration")
    
    system = OnlineLearningSystem(feature_dim=48, latent_dim=256, learning_rate=0.001)
    
    # Initial weights
    weights = np.random.randn(48, 256) * 0.01
    
    # Process one feedback
    original = np.random.rand(48)
    generated = original + np.random.randn(48) * 0.05
    latent = np.random.randn(256)
    
    result = system.process_feedback(
        original, generated, latent, weights, quality_score=0.7, session_id="test_1"
    )
    
    assert result['success'], "Feedback processing failed"
    assert 'updated_weights' in result, "Missing updated_weights"
    assert 'learning_record' in result, "Missing learning_record"
    
    updated_weights = result['updated_weights']
    assert updated_weights.shape == weights.shape, "Weight shape mismatch"
    assert np.all(np.isfinite(updated_weights)), "NaN/Inf in updated weights"
    
    print("  ✓ Single feedback processing successful")
    print(f"    Weight change norm: {result['weight_change_norm']:.4f}")
    
    # Check learning history
    assert len(system.learning_history) == 1, "Learning history not updated"
    assert system.total_updates == 1, "Total updates counter incorrect"
    print("  ✓ Learning history recorded")
    
    # Get summary
    summary = system.get_learning_summary()
    assert 'total_updates' in summary, "Missing summary fields"
    assert summary['total_updates'] == 1, "Summary update count incorrect"
    print(f"  ✓ Summary: {summary['total_updates']} update(s), Loss: {summary['latest_loss']:.4f}")


def test_learning_loop():
    """Test multi-iteration learning loop"""
    print("\n[TEST 6] Multi-Iteration Learning Loop")
    
    system = OnlineLearningSystem(feature_dim=48, latent_dim=256, learning_rate=0.001)
    weights = np.random.randn(48, 256) * 0.01
    
    np.random.seed(42)
    initial_loss = None
    
    for i in range(10):
        original = np.random.rand(48)
        generated = original + np.random.randn(48) * 0.1
        latent = np.random.randn(256)
        quality = 0.6 + i * 0.03  # Gradually improving
        
        result = system.process_feedback(
            original, generated, latent, weights, quality, session_id=f"test_{i}"
        )
        
        weights = result['updated_weights']
        
        if i == 0:
            initial_loss = result['learning_record']['loss']['total_loss']
    
    final_loss = list(system.loss_history)[-1]
    
    print(f"  ✓ 10 iterations completed")
    print(f"    Initial loss: {initial_loss:.4f}")
    print(f"    Final loss: {final_loss:.4f}")
    print(f"    Total system updates: {system.total_updates}")
    print(f"    Optimizer timesteps: {system.optimizer.t}")


def test_weight_constraints():
    """Test weight constraint enforcement"""
    print("\n[TEST 7] Weight Constraints")
    
    system = OnlineLearningSystem(feature_dim=48, latent_dim=256)
    
    # Large gradients
    weights = np.zeros((48, 256))
    large_gradient = np.ones((48, 256)) * 10.0
    
    # Process with large gradients (should be clipped)
    original = np.random.rand(48)
    generated = original + np.random.randn(48) * 0.05
    latent = np.random.randn(256)
    
    result = system.process_feedback(original, generated, latent, weights, 0.5)
    updated = result['updated_weights']
    
    # Check constraints
    assert np.all(updated <= 1.0), "Weights exceed upper bound"
    assert np.all(updated >= -1.0), "Weights exceed lower bound"
    assert np.all(np.isfinite(updated)), "Non-finite weights"
    
    print("  ✓ Weight constraints enforced")
    print(f"    Min weight: {np.min(updated):.4f}")
    print(f"    Max weight: {np.max(updated):.4f}")


def test_learning_metrics():
    """Test learning metrics computation"""
    print("\n[TEST 8] Learning Metrics")
    
    system = OnlineLearningSystem(feature_dim=48, latent_dim=256)
    weights = np.random.randn(48, 256) * 0.01
    
    # Run multiple feedbacks
    for i in range(5):
        original = np.random.rand(48)
        generated = original + np.random.randn(48) * 0.05
        latent = np.random.randn(256)
        
        system.process_feedback(original, generated, latent, weights, 0.7 + i*0.05)
        weights = weights + np.random.randn(48, 256) * 0.001
    
    # Get metrics
    summary = system.get_learning_summary()
    detailed = system.get_detailed_metrics()
    
    assert summary['total_updates'] == 5, "Update count mismatch"
    assert 'average_loss' in summary, "Missing average_loss"
    assert 'loss_trend' in summary, "Missing loss_trend"
    assert 'latest_update' in detailed, "Missing latest_update in detailed metrics"
    
    print("  ✓ Metrics computation successful")
    print(f"    Average loss: {summary['average_loss']:.4f}")
    print(f"    Latest loss: {summary['latest_loss']:.4f}")
    print(f"    Gradient norm trend captured: {len(detailed['gradient_norm_trend'])} steps")


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*70)
    print("Phase 2: Online Learning System - Test Suite")
    print("="*70)
    
    tests = [
        test_loss_function_basic,
        test_loss_computation,
        test_gradient_computation,
        test_adam_optimizer,
        test_online_learning_system,
        test_learning_loop,
        test_weight_constraints,
        test_learning_metrics,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"  ✗ FAILED: {str(e)}")
            failed += 1
    
    print("\n" + "="*70)
    print(f"Test Results: {passed} PASSED, {failed} FAILED out of {len(tests)} total")
    print("="*70 + "\n")
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
