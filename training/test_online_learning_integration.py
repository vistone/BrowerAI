"""
Test Suite for Online Learning Integration - P1 #2
Tests feature extraction, encoding, framework detection, and complete pipeline
"""

import sys
sys.path.insert(0, '/home/stone/BrowerAI/training')

import numpy as np
from online_learning_integration import OnlineLearningIntegration


def test_basic_initialization():
    """Test basic initialization"""
    print("\n" + "="*60)
    print("TEST: Basic Initialization")
    print("="*60)
    
    integration = OnlineLearningIntegration(
        feature_dim=48,
        latent_dim=256,
        learning_rate=0.001
    )
    
    assert integration.feature_dim == 48
    assert integration.latent_dim == 256
    assert integration.learning_rate == 0.001
    assert integration.metrics['total_processed'] == 0
    
    print("✓ Feature dimension: 48")
    print("✓ Latent dimension: 256")
    print("✓ Learning rate: 0.001")
    print("✓ Initial metrics: zero")
    print("\n✅ PASSED: Basic initialization\n")


def test_feature_extraction():
    """Test 48D feature extraction"""
    print("\n" + "="*60)
    print("TEST: Feature Extraction")
    print("="*60)
    
    integration = OnlineLearningIntegration()
    
    website_data = {
        "html": "<html><body><div>Test</div></body></html>",
        "css": [{"type": "inline", "size": 500}],
        "scripts": [{"type": "inline", "size": 1000}],
        "detected_frameworks": {"React": 0.8},
        "metadata": {"html_size": 2000},
        "success": True
    }
    
    features = integration._extract_features(website_data)
    
    assert len(features) == 48, f"Expected 48 features, got {len(features)}"
    assert features.dtype == np.float32
    assert np.all(np.isfinite(features)), "Features contain NaN or Inf"
    assert np.all(features >= 0), "Some features are negative"
    
    print(f"✓ Feature dimension: {len(features)}")
    print(f"✓ Feature dtype: {features.dtype}")
    print(f"✓ Feature range: [{features.min():.4f}, {features.max():.4f}]")
    print(f"✓ Feature mean: {features.mean():.4f}")
    print("\n✅ PASSED: Feature extraction\n")


def test_simple_encoding():
    """Test 48D -> 256D encoding"""
    print("\n" + "="*60)
    print("TEST: Simple Encoding")
    print("="*60)
    
    integration = OnlineLearningIntegration()
    
    features = np.random.randn(48).astype(np.float32)
    latent = integration._simple_encoding(features)
    
    assert latent.shape == (256,), f"Expected shape (256,), got {latent.shape}"
    assert np.all(np.isfinite(latent)), "Latent vector contains NaN or Inf"
    
    # Check normalization
    norm = np.linalg.norm(latent)
    print(f"✓ Input dimension: 48")
    print(f"✓ Output dimension: 256")
    print(f"✓ Latent norm: {norm:.4f}")
    print("\n✅ PASSED: Simple encoding\n")


def test_code_validation():
    """Test code generation validation"""
    print("\n" + "="*60)
    print("TEST: Code Validation")
    print("="*60)
    
    integration = OnlineLearningIntegration()
    
    # Test empty result
    score1 = integration._validate_generated_code({})
    assert score1 == 0.0
    print("✓ Empty result: score = 0.0")
    
    # Test partial result (HTML only)
    score2 = integration._validate_generated_code({"html": "<html></html>"})
    assert 0.0 < score2 <= 1.0
    print(f"✓ HTML only: score = {score2:.2f}")
    
    # Test complete result
    score3 = integration._validate_generated_code({
        "html": "<html></html>",
        "css": "body { }",
        "js": "console.log('test');"
    })
    assert score3 > 0.5
    print(f"✓ Complete result: score = {score3:.2f}")
    
    print("\n✅ PASSED: Code validation\n")


def test_single_website_processing():
    """Test processing a single website"""
    print("\n" + "="*60)
    print("TEST: Single Website Processing")
    print("="*60)
    
    integration = OnlineLearningIntegration()
    
    website_data = {
        "html": "<html><head><title>Test</title></head><body><div>Content</div></body></html>",
        "css": [{"type": "inline", "size": 1000}],
        "scripts": [{"type": "external", "size": 5000}],
        "detected_frameworks": {"React": 0.9, "Vue": 0.3},
        "metadata": {"html_size": 5000, "response_time_ms": 100},
        "title": "Test Site",
        "description": "A test site",
        "success": True
    }
    
    result = integration.process_website(website_data, session_id="test_1")
    
    assert result["success"], "Processing failed"
    assert "framework" in result
    assert "quality_score" in result
    assert "processing_time_ms" in result
    assert result["quality_score"] >= 0.0
    assert result["quality_score"] <= 1.0
    
    print(f"✓ Framework: {result['framework']}")
    print(f"✓ Quality score: {result['quality_score']:.3f}")
    print(f"✓ Processing time: {result['processing_time_ms']:.2f}ms")
    print(f"✓ Metrics updated:")
    print(f"  - total_processed: {integration.metrics['total_processed']}")
    print(f"  - average_latency_ms: {integration.metrics['average_latency_ms']:.2f}")
    
    print("\n✅ PASSED: Single website processing\n")


def test_batch_processing():
    """Test batch processing multiple websites"""
    print("\n" + "="*60)
    print("TEST: Batch Processing")
    print("="*60)
    
    integration = OnlineLearningIntegration()
    
    websites = [
        {
            "html": "<html><body><div>Site 1</div></body></html>",
            "css": [],
            "scripts": [],
            "detected_frameworks": {"React": 0.8},
            "metadata": {},
            "success": True
        },
        {
            "html": "<html><body><div>Site 2</div></body></html>",
            "css": [{"type": "inline", "size": 500}],
            "scripts": [],
            "detected_frameworks": {"Vue": 0.7},
            "metadata": {},
            "success": True
        },
        {
            "html": "<html><body><div>Site 3</div></body></html>",
            "css": [],
            "scripts": [{"type": "inline", "size": 1000}],
            "detected_frameworks": {"Angular": 0.6},
            "metadata": {},
            "success": True
        }
    ]
    
    result = integration.batch_process(websites, session_id="batch_test_1")
    
    assert result["total"] == 3
    assert result["successful"] == 3
    assert result["failed"] == 0
    assert result["success_rate"] == 1.0
    assert "frameworks" in result
    
    print(f"✓ Total websites: {result['total']}")
    print(f"✓ Successful: {result['successful']}")
    print(f"✓ Failed: {result['failed']}")
    print(f"✓ Success rate: {result['success_rate']:.1%}")
    print(f"✓ Framework distribution: {result['frameworks']}")
    
    # Check session was logged
    assert len(integration.sessions) == 1
    print(f"✓ Session logged: {integration.sessions[0]['session_id']}")
    
    print("\n✅ PASSED: Batch processing\n")


def test_caching():
    """Test feature caching mechanism"""
    print("\n" + "="*60)
    print("TEST: Feature Caching")
    print("="*60)
    
    integration = OnlineLearningIntegration()
    
    website_data = {
        "html": "<html></html>",
        "css": [],
        "scripts": [],
        "metadata": {},
        "success": True
    }
    
    # First processing
    result1 = integration.process_website(website_data, "cache_test_1")
    cache_hit_1 = integration.metrics['cache_hit_count']
    
    # Second processing (same data)
    result2 = integration.process_website(website_data, "cache_test_2")
    cache_hit_2 = integration.metrics['cache_hit_count']
    
    assert cache_hit_2 > cache_hit_1, "Cache not working"
    assert result1["cached"] == False, "First call should not be cached"
    assert result2["cached"] == True, "Second call should be cached"
    
    print(f"✓ First processing: cached = {result1['cached']}")
    print(f"✓ Second processing: cached = {result2['cached']}")
    print(f"✓ Cache hits: {cache_hit_2}")
    print(f"✓ Cache size: {len(integration.feature_cache)}")
    
    print("\n✅ PASSED: Feature caching\n")


def test_system_status():
    """Test system status retrieval"""
    print("\n" + "="*60)
    print("TEST: System Status")
    print("="*60)
    
    integration = OnlineLearningIntegration()
    
    status = integration.get_system_status()
    
    assert "components" in status
    assert "metrics" in status
    assert "cache_size" in status
    assert "sessions_count" in status
    
    components = status["components"]
    print(f"✓ OnlineLearner: {components['online_learner']}")
    print(f"✓ FeatureEncoder: {components['feature_encoder']}")
    print(f"✓ FrameworkDetector: {components['framework_detector']}")
    print(f"✓ CodeGenerator: {components['code_generator']}")
    print(f"✓ Cache size: {status['cache_size']}")
    print(f"✓ Sessions: {status['sessions_count']}")
    
    print("\n✅ PASSED: System status\n")


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("🧪 P1 #2: Online Learning Integration Test Suite")
    print("="*60)
    
    tests = [
        test_basic_initialization,
        test_feature_extraction,
        test_simple_encoding,
        test_code_validation,
        test_single_website_processing,
        test_batch_processing,
        test_caching,
        test_system_status
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"\n❌ FAILED: {test.__name__}")
            print(f"Error: {e}\n")
            failed += 1
        except Exception as e:
            print(f"\n❌ ERROR: {test.__name__}")
            print(f"Exception: {e}\n")
            failed += 1
    
    # Summary
    print("="*60)
    print(f"✅ PASSED: {passed}/{len(tests)}")
    print(f"❌ FAILED: {failed}/{len(tests)}")
    print("="*60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
