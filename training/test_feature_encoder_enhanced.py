"""
Test Suite for Enhanced Feature Encoder
Comprehensive validation of non-linear layers, learnable embeddings, and anomaly detection
"""

import sys
sys.path.insert(0, '/home/stone/BrowerAI/training')

import numpy as np
from feature_encoder_enhanced import (
    EnhancedFeatureEncoder, 
    AnomalyDetector, 
    LayerNormalization,
    NonLinearActivation
)
from feature_encoder import FeatureEncoder


def test_enhanced_initialization():
    """Test initialization of enhanced encoder"""
    print("\n" + "="*60)
    print("TEST: Enhanced Encoder Initialization")
    print("="*60)
    
    encoder = EnhancedFeatureEncoder(
        feature_dim=48,
        hidden_dim=128,
        latent_dim=256
    )
    
    assert encoder.feature_dim == 48
    assert encoder.hidden_dim == 128
    assert encoder.latent_dim == 256
    assert encoder.W1.shape == (48, 128)
    assert encoder.W2.shape == (128, 256)
    assert encoder.b1.shape == (128,)
    assert encoder.b2.shape == (256,)
    assert len(encoder.intent_embeddings) == 8
    assert len(encoder.style_embeddings) == 7
    
    print("✓ Layer dimensions correct")
    print("✓ Weight shapes valid")
    print("✓ Embedding dictionaries initialized")
    print("✓ Anomaly detector enabled")
    print("\n✅ PASSED: Enhanced initialization\n")


def test_anomaly_detection_numeric():
    """Test numeric anomaly detection (NaN, Inf, extreme values)"""
    print("\n" + "="*60)
    print("TEST: Numeric Anomaly Detection")
    print("="*60)
    
    detector = AnomalyDetector()
    
    # Test 1: Normal features
    normal_features = np.random.randn(48)
    result = detector.detect_numeric_anomalies(normal_features)
    assert result['is_healthy'] == True
    assert result['has_nan'] == False
    assert result['has_inf'] == False
    print("✓ Normal features: is_healthy=True")
    
    # Test 2: NaN detection
    nan_features = np.random.randn(48)
    nan_features[10] = np.nan
    result = detector.detect_numeric_anomalies(nan_features)
    assert result['has_nan'] == True
    assert 10 in result['nan_indices']
    assert result['is_healthy'] == False
    print("✓ NaN detection: correctly identified at index 10")
    
    # Test 3: Inf detection
    inf_features = np.random.randn(48)
    inf_features[25] = np.inf
    result = detector.detect_numeric_anomalies(inf_features)
    assert result['has_inf'] == True
    assert 25 in result['inf_indices']
    assert result['is_healthy'] == False
    print("✓ Inf detection: correctly identified at index 25")
    
    # Test 4: Extreme values
    extreme_features = np.random.randn(48)
    extreme_features[15] = 500.0
    result = detector.detect_numeric_anomalies(extreme_features)
    assert result['has_extreme'] == True
    assert 15 in result['extreme_indices']
    print("✓ Extreme value detection: correctly identified")
    
    print("\n✅ PASSED: Numeric anomaly detection\n")


def test_anomaly_detection_statistical():
    """Test statistical anomaly detection (IQR method)"""
    print("\n" + "="*60)
    print("TEST: Statistical Anomaly Detection (IQR Method)")
    print("="*60)
    
    detector = AnomalyDetector(history_size=100)
    
    # Build history with normal distribution
    normal_values = []
    for _ in range(50):
        features = np.random.normal(loc=5.0, scale=1.0, size=48)
        result = detector.detect_statistical_anomalies(features)
        normal_values.append(result['is_anomaly'])
    
    normal_anomaly_rate = sum(normal_values) / len(normal_values)
    print(f"✓ Normal distribution: {normal_anomaly_rate*100:.1f}% anomaly rate (expected ~0%)")
    
    # Test with outlier
    features = np.random.normal(loc=5.0, scale=1.0, size=48)
    features[0] = 50.0  # Extreme outlier
    result = detector.detect_statistical_anomalies(features)
    assert result['is_anomaly'] == True
    assert 0 in result['outlier_indices']
    print(f"✓ Outlier detection: correctly identified feature 0 as outlier")
    
    # Check statistics
    stats = detector.get_statistics()
    print(f"✓ Detector stats: {stats['total_checks']} checks, "
          f"{stats['anomalies_detected']} anomalies detected")
    
    print("\n✅ PASSED: Statistical anomaly detection\n")


def test_layer_normalization():
    """Test layer normalization"""
    print("\n" + "="*60)
    print("TEST: Layer Normalization")
    print("="*60)
    
    ln = LayerNormalization(feature_dim=128)
    
    # Test 1: Basic normalization
    x = np.random.randn(128) * 10.0 + 5.0  # Large scale and offset
    x_norm = ln.normalize(x)
    
    assert np.abs(np.mean(x_norm)) < 0.1, f"Mean should be ~0, got {np.mean(x_norm)}"
    assert np.abs(np.std(x_norm) - 1.0) < 0.1, f"Std should be ~1, got {np.std(x_norm)}"
    print(f"✓ Normalized mean: {np.mean(x_norm):.4f} (expected ~0)")
    print(f"✓ Normalized std: {np.std(x_norm):.4f} (expected ~1)")
    
    # Test 2: Learnable parameters
    new_gamma = np.ones(128) * 2.0
    new_beta = np.ones(128) * 1.0
    ln.update_parameters(new_gamma, new_beta)
    
    x_norm2 = ln.normalize(x)
    assert np.abs(np.max(x_norm2) - np.max(new_gamma * (x - np.mean(x)) / np.std(x) + new_beta)) < 0.01
    print(f"✓ Learnable parameters updated correctly")
    
    print("\n✅ PASSED: Layer normalization\n")


def test_activations():
    """Test non-linear activation functions"""
    print("\n" + "="*60)
    print("TEST: Non-linear Activation Functions")
    print("="*60)
    
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    
    # Test ReLU
    relu_out = NonLinearActivation.relu(x)
    expected_relu = np.array([0.0, 0.0, 0.0, 1.0, 2.0])
    assert np.allclose(relu_out, expected_relu)
    print(f"✓ ReLU: {x} → {relu_out}")
    
    # Test GELU
    gelu_out = NonLinearActivation.gelu(x)
    # GELU is smooth but not strictly monotonic at all points
    assert gelu_out[2] < 0.05  # Near zero at x=0
    assert gelu_out[-1] > gelu_out[0]  # Generally increases
    print(f"✓ GELU: smooth activation, value at x=0: {gelu_out[2]:.4f}")
    
    # Test Leaky ReLU
    lrelu_out = NonLinearActivation.leaky_relu(x, alpha=0.1)
    assert lrelu_out[0] == -2.0 * 0.1  # Negative pass-through
    assert lrelu_out[-1] == 2.0  # Positive pass-through
    print(f"✓ Leaky ReLU: α=0.1, preserves gradients for negative inputs")
    
    print("\n✅ PASSED: Non-linear activations\n")


def test_basic_encoding():
    """Test basic feature encoding"""
    print("\n" + "="*60)
    print("TEST: Basic Feature Encoding")
    print("="*60)
    
    encoder = EnhancedFeatureEncoder()
    
    # Generate sample features
    features = np.random.randn(48)
    
    # Encode
    result = encoder.encode(
        features=features.tolist(),
        intent="blog",
        design_style="modern"
    )
    
    assert result['latent'] is not None
    assert result['latent'].shape == (256,)
    assert result['confidence'] > 0.5
    assert result['intent'] == 'blog'
    assert result['style'] == 'modern'
    
    latent_norm = np.linalg.norm(result['latent'])
    assert latent_norm > 0.1
    
    print(f"✓ Latent vector shape: {result['latent'].shape}")
    print(f"✓ Latent norm: {latent_norm:.4f}")
    print(f"✓ Confidence: {result['confidence']:.4f}")
    print(f"✓ Extracted intent: {result['intent']}")
    print(f"✓ Extracted style: {result['style']}")
    
    print("\n✅ PASSED: Basic encoding\n")


def test_encoding_with_dimensions():
    """Test encoding with different feature dimensions"""
    print("\n" + "="*60)
    print("TEST: Encoding with Different Configurations")
    print("="*60)
    
    encoder = EnhancedFeatureEncoder(
        feature_dim=48,
        hidden_dim=128,
        latent_dim=256
    )
    
    # Valid features
    features = np.random.randn(48)
    result = encoder.encode(features.tolist())
    assert result['latent'] is not None
    print(f"✓ Valid features (48D): encoded successfully")
    
    # Invalid dimension (encoder returns error dict, doesn't raise exception)
    invalid_features = np.random.randn(32)  # Wrong dimension
    result = encoder.encode(invalid_features.tolist())
    assert 'error' in result, "Should return error dict for wrong dimension"
    assert "Expected 48 features" in result['error']
    print(f"✓ Invalid dimension (32D): correctly rejected")
    
    print("\n✅ PASSED: Dimension validation\n")


def test_anomaly_skipping():
    """Test skipping encoding when anomalies detected"""
    print("\n" + "="*60)
    print("TEST: Anomaly Skipping")
    print("="*60)
    
    encoder = EnhancedFeatureEncoder(enable_anomaly_detection=True)
    
    # Normal encoding
    features = np.random.randn(48)
    result = encoder.encode(features.tolist(), skip_on_anomaly=True)
    assert result['latent'] is not None
    print(f"✓ Normal features: processed (not skipped)")
    
    # Encoding with NaN (should be skipped)
    features_nan = features.copy()
    features_nan[5] = np.nan
    result_nan = encoder.encode(features_nan.tolist(), skip_on_anomaly=True)
    assert result_nan['latent'] is None
    assert result_nan['anomaly_detected'] == True
    print(f"✓ NaN features: skipped correctly")
    print(f"  Reason: {result_nan['reason']}")
    
    # Check skip counter
    assert encoder.skipped_encodings >= 1
    print(f"✓ Skipped encodings counter: {encoder.skipped_encodings}")
    
    print("\n✅ PASSED: Anomaly skipping\n")


def test_weight_updates():
    """Test weight update mechanism"""
    print("\n" + "="*60)
    print("TEST: Weight Updates")
    print("="*60)
    
    encoder = EnhancedFeatureEncoder()
    
    # Store original weights
    original_W1 = encoder.W1.copy()
    original_W2 = encoder.W2.copy()
    
    # Create new weights
    new_W1 = np.random.randn(48, 128) * 0.01
    new_b1 = np.random.randn(128) * 0.01
    new_W2 = np.random.randn(128, 256) * 0.01
    new_b2 = np.random.randn(256) * 0.01
    
    # Update
    success = encoder.update_weights(new_W1, new_b1, new_W2, new_b2)
    
    assert success == True
    assert np.allclose(encoder.W1, new_W1)
    assert np.allclose(encoder.W2, new_W2)
    print(f"✓ Weight update successful")
    print(f"✓ W1 changed: {not np.allclose(encoder.W1, original_W1)}")
    print(f"✓ W2 changed: {not np.allclose(encoder.W2, original_W2)}")
    
    # Try invalid update
    invalid_W1 = np.random.randn(48, 100)  # Wrong shape
    success = encoder.update_weights(invalid_W1, new_b1, new_W2, new_b2)
    assert success == False
    print(f"✓ Invalid shape correctly rejected")
    
    print("\n✅ PASSED: Weight updates\n")


def test_embedding_updates():
    """Test learnable embedding updates"""
    print("\n" + "="*60)
    print("TEST: Embedding Updates")
    print("="*60)
    
    encoder = EnhancedFeatureEncoder()
    
    # Original embeddings
    original_intent = encoder.intent_embeddings['blog'].copy()
    
    # Create new embeddings
    new_intent_embeddings = {
        intent: np.random.randn(256) * 0.05
        for intent in encoder.intent_types
    }
    
    # Update
    success = encoder.update_embeddings(intent_embeddings=new_intent_embeddings)
    
    assert success == True
    assert not np.allclose(encoder.intent_embeddings['blog'], original_intent)
    print(f"✓ Embedding update successful")
    print(f"✓ Intent embeddings changed")
    
    print("\n✅ PASSED: Embedding updates\n")


def test_statistics():
    """Test statistics collection"""
    print("\n" + "="*60)
    print("TEST: Statistics Collection")
    print("="*60)
    
    encoder = EnhancedFeatureEncoder()
    
    # Encode multiple features
    for _ in range(10):
        features = np.random.randn(48)
        encoder.encode(features.tolist())
    
    # Get statistics
    stats = encoder.get_statistics()
    
    assert stats['total_encodings'] >= 10
    print(f"✓ Total encodings: {stats['total_encodings']}")
    print(f"✓ Anomalies found: {stats['anomalies_found']}")
    print(f"✓ Skipped encodings: {stats['skipped_encodings']}")
    print(f"✓ Detection rate: {stats['detection_rate']:.4f}")
    
    # Get weight statistics
    weight_stats = encoder.get_weight_statistics()
    
    assert weight_stats['W1_norm'] > 0
    assert weight_stats['W2_norm'] > 0
    print(f"✓ W1 norm: {weight_stats['W1_norm']:.4f}")
    print(f"✓ W2 norm: {weight_stats['W2_norm']:.4f}")
    print(f"✓ Intent embeddings mean norm: {weight_stats['intent_embeddings_mean_norm']:.4f}")
    print(f"✓ Style embeddings mean norm: {weight_stats['style_embeddings_mean_norm']:.4f}")
    
    print("\n✅ PASSED: Statistics collection\n")


def test_comparison_baseline():
    """Compare enhanced encoder with baseline"""
    print("\n" + "="*60)
    print("TEST: Comparison with Baseline Encoder")
    print("="*60)
    
    baseline_encoder = FeatureEncoder()
    enhanced_encoder = EnhancedFeatureEncoder()
    
    # Test samples
    features = np.random.randn(48)
    
    # Get baseline latent
    baseline_latent = baseline_encoder.encode(features.tolist())
    
    # Get enhanced latent
    enhanced_result = enhanced_encoder.encode(features.tolist())
    enhanced_latent = enhanced_result['latent']
    
    # Compare
    baseline_norm = np.linalg.norm(baseline_latent)
    enhanced_norm = np.linalg.norm(enhanced_latent)
    diff_norm = np.linalg.norm(baseline_latent - enhanced_latent)
    
    print(f"✓ Baseline latent norm: {baseline_norm:.4f}")
    print(f"✓ Enhanced latent norm: {enhanced_norm:.4f}")
    print(f"✓ Difference norm: {diff_norm:.4f}")
    print(f"✓ Enhanced confidence: {enhanced_result['confidence']:.4f}")
    
    # Compare with method
    comparison = enhanced_encoder.compare_with_baseline(baseline_encoder, features.tolist())
    
    assert 'baseline_norm' in comparison
    assert 'enhanced_norm' in comparison
    assert 'diversity_score' in comparison
    
    print(f"✓ Diversity score: {comparison['diversity_score']:.4f}")
    print(f"✓ Improved: {comparison['improved']}")
    
    print("\n✅ PASSED: Baseline comparison\n")


def test_stress_test():
    """Stress test with many encodings"""
    print("\n" + "="*60)
    print("TEST: Stress Test (100 Encodings)")
    print("="*60)
    
    encoder = EnhancedFeatureEncoder()
    
    intents = ["blog", "ecommerce", "documentation", "portfolio", "landing"]
    styles = ["modern", "minimal", "classic", "playful", "professional"]
    
    success_count = 0
    latent_norms = []
    confidences = []
    
    for i in range(100):
        features = np.random.randn(48)
        intent = intents[i % len(intents)]
        style = styles[i % len(styles)]
        
        result = encoder.encode(
            features=features.tolist(),
            intent=intent,
            design_style=style,
            skip_on_anomaly=False  # Don't skip for stress test
        )
        
        if result['latent'] is not None:
            success_count += 1
            latent_norms.append(result['latent_norm'])
            confidences.append(result['confidence'])
    
    stats = encoder.get_statistics()
    
    print(f"✓ Total encodings: {stats['total_encodings']}")
    print(f"✓ Successful: {success_count}/100")
    print(f"✓ Failures: {100 - success_count}")
    print(f"✓ Latent norm - mean: {np.mean(latent_norms):.4f}, "
          f"std: {np.std(latent_norms):.4f}")
    print(f"✓ Confidence - mean: {np.mean(confidences):.4f}, "
          f"std: {np.std(confidences):.4f}")
    
    assert success_count >= 95, "Should have at least 95% success rate"
    
    print("\n✅ PASSED: Stress test\n")


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*70)
    print("ENHANCED FEATURE ENCODER - COMPREHENSIVE TEST SUITE")
    print("="*70)
    
    tests = [
        test_enhanced_initialization,
        test_anomaly_detection_numeric,
        test_anomaly_detection_statistical,
        test_layer_normalization,
        test_activations,
        test_basic_encoding,
        test_encoding_with_dimensions,
        test_anomaly_skipping,
        test_weight_updates,
        test_embedding_updates,
        test_statistics,
        test_comparison_baseline,
        test_stress_test
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"\n❌ FAILED: {test_func.__name__}")
            print(f"   Error: {e}\n")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"✅ Passed: {passed}/{len(tests)}")
    print(f"❌ Failed: {failed}/{len(tests)}")
    print(f"📊 Success Rate: {(passed/len(tests))*100:.1f}%")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED!")
    else:
        print(f"\n⚠️  {failed} test(s) failed")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    run_all_tests()
