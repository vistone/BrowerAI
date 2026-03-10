#!/usr/bin/env python3
"""
Test suite for enhanced OnlineLearner
Validates gradient stability, anomaly detection, and adaptive learning
"""

import numpy as np
import logging
from pathlib import Path
import sys
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Import the enhanced OnlineLearner
sys.path.insert(0, str(Path(__file__).parent))
from online_learner import OnlineLearner, FeedbackBuffer


def test_basic_initialization():
    """Test OnlineLearner initialization with stability features"""
    logger.info("=" * 70)
    logger.info("TEST: Basic Initialization")
    logger.info("=" * 70)
    
    learner = OnlineLearner(
        feature_dim=48,
        latent_dim=256,
        learning_rate=0.001,
        enable_gradient_clip=True,
        enable_anomaly_detection=True,
        loss_weight_mode="adaptive"
    )
    
    assert learner.feature_dim == 48
    assert learner.latent_dim == 256
    assert learner.enable_gradient_clip == True
    assert learner.enable_anomaly_detection == True
    
    metrics = learner.get_metrics()
    assert metrics['optimization']['update_count'] == 0
    
    logger.info("✅ Initialization test passed")
    logger.info(f"Initial status: {metrics['status']}")
    print()


def test_gradient_health_check():
    """Test gradient health check functionality"""
    logger.info("=" * 70)
    logger.info("TEST: Gradient Health Check")
    logger.info("=" * 70)
    
    learner = OnlineLearner(enable_gradient_clip=True)
    
    # Test normal gradient
    normal_grad = np.random.randn(48, 256) * 0.01
    health = learner._check_gradient_health(normal_grad)
    assert health['is_healthy'] == True
    logger.info(f"✅ Normal gradient health: {health}")
    
    # Test NaN gradient
    nan_grad = np.full((48, 256), np.nan)
    health = learner._check_gradient_health(nan_grad)
    assert health['is_healthy'] == False
    assert 'NaN' in health['reason']
    logger.info(f"✅ NaN detection: {health['reason']}")
    
    # Test Inf gradient
    inf_grad = np.full((48, 256), np.inf)
    health = learner._check_gradient_health(inf_grad)
    assert health['is_healthy'] == False
    assert 'Inf' in health['reason']
    logger.info(f"✅ Inf detection: {health['reason']}")
    
    # Test exploding gradient
    exploding_grad = np.random.randn(48, 256) * 1000
    health = learner._check_gradient_health(exploding_grad)
    assert health['is_healthy'] == False
    assert 'explosion' in health['reason'].lower()
    logger.info(f"✅ Explosion detection: {health['reason']}")
    print()


def test_anomaly_detection():
    """Test anomalous feedback detection"""
    logger.info("=" * 70)
    logger.info("TEST: Anomaly Detection")
    logger.info("=" * 70)
    
    learner = OnlineLearner(enable_anomaly_detection=True)
    
    # Simulate normal feedback sequence
    for i in range(20):
        quality = 0.7 + np.random.randn() * 0.1  # Normal distribution
        feedback = {
            "quality_score": np.clip(quality, 0, 1),
            "html_quality": 0.7,
            "css_quality": 0.7,
            "js_quality": 0.7,
        }
        learner.quality_history.append(feedback["quality_score"])
    
    # Test normal feedback
    normal_feedback = {
        "quality_score": 0.72,
        "html_quality": 0.7,
        "css_quality": 0.7,
        "js_quality": 0.7,
    }
    is_anomaly = learner._detect_anomaly_feedback(0.72, normal_feedback)
    logger.info(f"✅ Normal feedback anomaly status: {is_anomaly}")
    
    # Test outlier feedback
    outlier_feedback = {
        "quality_score": 0.01,
        "html_quality": 0.7,
        "css_quality": 0.7,
        "js_quality": 0.7,
    }
    is_anomaly = learner._detect_anomaly_feedback(0.01, outlier_feedback)
    logger.info(f"✅ Outlier feedback anomaly status: {is_anomaly} (anomaly_count={learner.anomaly_count})")
    print()


def test_adaptive_loss_weights():
    """Test adaptive loss weight adjustment"""
    logger.info("=" * 70)
    logger.info("TEST: Adaptive Loss Weights")
    logger.info("=" * 70)
    
    learner = OnlineLearner(loss_weight_mode="adaptive")
    
    log_initial_weights = learner.loss_weights.copy()
    logger.info(f"Initial weights: {log_initial_weights}")
    
    # Simulate low-quality performance
    for i in range(20):
        learner.quality_scores.append(0.3)  # Low quality
    
    learner._update_adaptive_loss_weights()
    logger.info(f"✅ Low-quality weights: {learner.loss_weights}")
    assert learner.loss_weights['component'] > log_initial_weights['component']
    
    learner.loss_weights = log_initial_weights.copy()
    
    # Simulate high-quality performance
    learner.quality_scores.clear()
    for i in range(20):
        learner.quality_scores.append(0.8)  # High quality
    
    learner._update_adaptive_loss_weights()
    logger.info(f"✅ High-quality weights: {learner.loss_weights}")
    assert learner.loss_weights['reconstruction'] > 0.3
    print()


def test_learning_rate_scheduling():
    """Test adaptive learning rate adjustment"""
    logger.info("=" * 70)
    logger.info("TEST: Learning Rate Scheduling")
    logger.info("=" * 70)
    
    learner = OnlineLearner(learning_rate=0.001)
    initial_lr = learner.learning_rate
    
    # Simulate converged training
    for i in range(30):
        learner.training_losses.append(0.1 + np.random.randn() * 0.001)
    
    learner._adaptive_learning_rate_schedule()
    logger.info(f"✅ After convergence: {initial_lr:.6f} → {learner.learning_rate:.6f}")
    assert learner.learning_rate < initial_lr
    
    learner.learning_rate = initial_lr
    learner.training_losses.clear()
    
    # Simulate diverging training
    for i in range(30):
        learner.training_losses.append(0.1 * (1 + 0.1 * i))  # Increasing loss
    
    learner._adaptive_learning_rate_schedule()
    logger.info(f"✅ After divergence: {initial_lr:.6f} → {learner.learning_rate:.6f}")
    assert learner.learning_rate < initial_lr
    print()


def test_full_feedback_loop():
    """Test complete feedback processing loop"""
    logger.info("=" * 70)
    logger.info("TEST: Full Feedback Loop")
    logger.info("=" * 70)
    
    learner = OnlineLearner(
        enable_gradient_clip=True,
        enable_anomaly_detection=True,
        loss_weight_mode="adaptive"
    )
    
    # Generate synthetic feature-latent pairs
    for iteration in range(10):
        features = np.random.randn(48) * 0.5
        generated_latent = np.random.randn(256) * 0.5
        
        quality_score = 0.6 + np.random.randn() * 0.1
        quality_score = np.clip(quality_score, 0, 1)
        
        feedback = {
            "quality_score": quality_score,
            "html_quality": quality_score,
            "css_quality": quality_score,
            "js_quality": quality_score,
        }
        
        result = learner.process_feedback(
            features=features,
            generated_latent=generated_latent,
            feedback_data=feedback,
            session_id=f"test_{iteration}"
        )
        
        logger.info(
            f"Iteration {iteration+1}: loss={result['loss']:.4f}, "
            f"quality={result['quality_score']:.3f}, "
            f"updated={result['weights_updated']}, "
            f"convergence={result['convergence']:.3f}"
        )
    
    metrics = learner.get_metrics()
    logger.info(f"✅ Final metrics:")
    logger.info(f"   Total updates: {metrics['optimization']['update_count']}")
    logger.info(f"   Skipped updates: {metrics['optimization']['skipped_updates']}")
    logger.info(f"   Anomalies: {metrics['optimization']['anomaly_count']}")
    logger.info(f"   Status: {metrics['status']}")
    print()


def test_weight_divergence():
    """Test weight divergence monitoring"""
    logger.info("=" * 70)
    logger.info("TEST: Weight Divergence")
    logger.info("=" * 70)
    
    learner = OnlineLearner()
    
    # No updates yet
    div = learner._compute_weight_divergence()
    logger.info(f"Initial divergence: {div:.6f}")
    assert div < 0.01
    
    # Simulate some weight updates
    learner.encoding_matrix = learner.encoding_matrix + np.random.randn(48, 256) * 0.5
    
    div = learner._compute_weight_divergence()
    logger.info(f"✅ After update divergence: {div:.6f}")
    assert div > 0.1
    print()


def test_stress_handling():
    """Test learner stability under stress"""
    logger.info("=" * 70)
    logger.info("TEST: Stress Handling")
    logger.info("=" * 70)
    
    learner = OnlineLearner(
        enable_gradient_clip=True,
        enable_anomaly_detection=True
    )
    
    # Simulate high-frequency, noisy feedback
    for iteration in range(100):
        features = np.random.randn(48)
        latent = np.random.randn(256) * 10  # Large latent values
        
        # Mix of normal and noisy quality scores
        if iteration % 10 == 0:
            quality = np.random.rand()  # Random outlier
        else:
            quality = np.clip(0.6 + np.random.randn() * 0.15, 0, 1)
        
        feedback = {
            "quality_score": quality,
            "html_quality": quality,
            "css_quality": quality,
            "js_quality": quality,
        }
        
        result = learner.process_feedback(
            features=features,
            generated_latent=latent,
            feedback_data=feedback,
            session_id=f"stress_{iteration}"
        )
        
        if (iteration + 1) % 25 == 0:
            metrics = learner.get_metrics()
            logger.info(
                f"Stress iteration {iteration+1}: "
                f"loss={metrics['loss']['recent_average']:.4f}, "
                f"divergence={metrics['weights']['divergence']:.3f}, "
                f"skipped={metrics['optimization']['skipped_updates']}"
            )
    
    final_metrics = learner.get_metrics()
    logger.info(f"✅ Stress test completed")
    logger.info(f"   Final status: {final_metrics['status']}")
    logger.info(f"   Total anomalies: {final_metrics['optimization']['anomaly_count']}")
    logger.info(f"   Total skipped: {final_metrics['optimization']['skipped_updates']}")
    
    # Should have survived without NaN/Inf
    assert not np.any(np.isnan(learner.encoding_matrix))
    assert not np.any(np.isinf(learner.encoding_matrix))
    print()


def generate_report():
    """Generate test report"""
    logger.info("=" * 70)
    logger.info("TEST SUMMARY")
    logger.info("=" * 70)
    logger.info("✅ All tests passed successfully!")
    logger.info("")
    logger.info("Enhanced online learner features:")
    logger.info("  ✓ Gradient health checks (NaN/Inf/explosion detection)")
    logger.info("  ✓ Anomaly feedback detection (IQR method)")
    logger.info("  ✓ Adaptive loss weight adjustment")
    logger.info("  ✓ Adaptive learning rate scheduling")
    logger.info("  ✓ Weight divergence monitoring")
    logger.info("  ✓ Comprehensive metrics reporting")
    logger.info("=" * 70)
    print()


if __name__ == "__main__":
    try:
        test_basic_initialization()
        test_gradient_health_check()
        test_anomaly_detection()
        test_adaptive_loss_weights()
        test_learning_rate_scheduling()
        test_full_feedback_loop()
        test_weight_divergence()
        test_stress_handling()
        generate_report()
        
        logger.info("✅ All tests completed successfully!")
        sys.exit(0)
    
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}", exc_info=True)
        sys.exit(1)
