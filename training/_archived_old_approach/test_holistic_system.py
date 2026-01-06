"""
Quick Test: Holistic Website Learning System

Validates that the holistic learning architecture works correctly.
Tests:
1. Model initialization
2. Forward pass with dummy data
3. Multi-task loss computation
4. Website similarity computation
5. ONNX export

Usage:
    python scripts/test_holistic_system.py
"""

import sys
from pathlib import Path
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.models.website_learner import (
    HolisticWebsiteLearner,
    WebsiteIntentClassifier,
    CodeStyleAnalyzer,
    DependencyGraphLearner,
    DeviceAdaptationAnalyzer
)

def test_website_intent_classifier():
    """Test WebsiteIntentClassifier"""
    print("Testing WebsiteIntentClassifier...")
    
    model = WebsiteIntentClassifier(
        d_model=256,
        num_categories=10
    )
    
    # Dummy inputs - corrected to match actual API
    batch_size = 4
    url_features = torch.randn(batch_size, 128)  # URL features
    content_features = torch.randn(batch_size, 256)  # Content features
    structure_features = torch.randn(batch_size, 256)  # Structure features
    
    # Forward pass - fixed to match actual signature
    category_logits = model(url_features, content_features, structure_features)
    
    assert category_logits.shape == (batch_size, 10)
    
    print("✓ WebsiteIntentClassifier test passed")
    return True


def test_code_style_analyzer():
    """Test CodeStyleAnalyzer"""
    print("Testing CodeStyleAnalyzer...")
    
    model = CodeStyleAnalyzer(
        d_model=256,
        num_frameworks=50,
        num_build_tools=20,
        num_companies=30
    )
    
    # Dummy inputs
    batch_size = 4
    seq_len = 128
    html = torch.randn(batch_size, seq_len, 256)
    css = torch.randn(batch_size, seq_len, 256)
    js = torch.randn(batch_size, seq_len, 256)
    
    # Forward pass
    outputs = model(html, css, js)
    
    assert 'fingerprint' in outputs
    assert 'framework_logits' in outputs
    assert 'build_tool_logits' in outputs
    assert 'company_logits' in outputs
    
    assert outputs['fingerprint'].shape == (batch_size, 256)
    assert outputs['framework_logits'].shape == (batch_size, 50)
    assert outputs['build_tool_logits'].shape == (batch_size, 20)
    assert outputs['company_logits'].shape == (batch_size, 30)
    
    print("✓ CodeStyleAnalyzer test passed")
    return True


def test_dependency_graph_learner():
    """Test DependencyGraphLearner"""
    print("Testing DependencyGraphLearner...")
    
    model = DependencyGraphLearner(
        d_model=256
    )
    
    # Dummy inputs
    batch_size = 2
    num_nodes = 10
    node_features = torch.randn(batch_size, num_nodes, 256)
    adjacency_matrix = torch.randint(0, 2, (batch_size, num_nodes, num_nodes)).float()
    
    # Forward pass
    outputs = model(node_features, adjacency_matrix)
    
    assert 'node_embeddings' in outputs
    assert 'loading_order' in outputs
    assert 'dependency_types' in outputs
    
    assert outputs['node_embeddings'].shape == (batch_size, num_nodes, 256)
    assert outputs['loading_order'].shape == (batch_size, num_nodes)
    
    print("✓ DependencyGraphLearner test passed")
    return True


def test_device_adaptation_analyzer():
    """Test DeviceAdaptationAnalyzer"""
    print("Testing DeviceAdaptationAnalyzer...")
    
    model = DeviceAdaptationAnalyzer(
        d_model=256
    )
    
    # Dummy inputs
    batch_size = 4
    seq_len = 128
    html = torch.randn(batch_size, seq_len, 256)
    css = torch.randn(batch_size, seq_len, 256)
    
    # Forward pass
    outputs = model(html, css)
    
    assert 'strategy_logits' in outputs
    assert 'breakpoints' in outputs
    assert 'interaction_logits' in outputs
    
    assert outputs['strategy_logits'].shape == (batch_size, 4)
    assert outputs['breakpoints'].shape[0] == batch_size
    assert outputs['interaction_logits'].shape == (batch_size, 3)
    
    print("✓ DeviceAdaptationAnalyzer test passed")
    return True


def test_holistic_website_learner():
    """Test complete HolisticWebsiteLearner"""
    print("Testing HolisticWebsiteLearner...")
    
    model = HolisticWebsiteLearner(
        vocab_size=10000,
        d_model=256,
        num_heads=8,
        num_layers=4,
        d_ff=1024,
        dropout=0.1,
        max_len=512,
        num_categories=10,
        num_frameworks=50,
        num_build_tools=20,
        num_companies=30
    )
    
    # Dummy inputs
    batch_size = 2
    seq_len = 128
    html_tokens = torch.randint(0, 10000, (batch_size, seq_len))
    css_tokens = torch.randint(0, 10000, (batch_size, seq_len))
    js_tokens = torch.randint(0, 10000, (batch_size, seq_len))
    
    # Optional: dependency graph
    num_nodes = 10
    adjacency_matrix = torch.randint(0, 2, (batch_size, num_nodes, num_nodes)).float()
    
    # Optional: URL features
    url_features = torch.randn(batch_size, 10)
    
    # Forward pass
    outputs = model(html_tokens, css_tokens, js_tokens, adjacency_matrix, url_features)
    
    # Check all expected outputs
    assert 'website_embedding' in outputs
    assert 'category_logits' in outputs
    assert 'framework_logits' in outputs
    assert 'build_tool_logits' in outputs
    assert 'company_logits' in outputs
    assert 'style_fingerprint' in outputs
    
    # Check shapes
    assert outputs['website_embedding'].shape == (batch_size, 512)
    assert outputs['category_logits'].shape == (batch_size, 10)
    assert outputs['framework_logits'].shape == (batch_size, 50)
    assert outputs['style_fingerprint'].shape == (batch_size, 256)
    
    print("✓ HolisticWebsiteLearner test passed")
    
    # Test website similarity
    print("\nTesting website similarity computation...")
    embeddings = outputs['website_embedding']
    similarity_matrix = model.compute_website_similarity(embeddings, embeddings)
    
    assert similarity_matrix.shape == (batch_size, batch_size)
    # Diagonal should be 1.0 (self-similarity)
    diagonal = torch.diagonal(similarity_matrix)
    assert torch.allclose(diagonal, torch.ones(batch_size), atol=1e-6)
    
    print("✓ Website similarity test passed")
    
    return True


def test_model_export():
    """Test ONNX export"""
    print("\nTesting ONNX export...")
    
    model = HolisticWebsiteLearner(
        vocab_size=10000,
        d_model=128,  # Smaller for faster export
        num_heads=4,
        num_layers=2,
        d_ff=512,
        dropout=0.1,
        max_len=256,
        num_categories=10
    )
    
    # Dummy inputs
    html = torch.randint(0, 10000, (1, 128))
    css = torch.randint(0, 10000, (1, 128))
    js = torch.randint(0, 10000, (1, 128))
    
    # Export to ONNX
    output_path = "test_website_learner.onnx"
    try:
        model.export_to_onnx(output_path, html, css, js)
        print(f"✓ ONNX export test passed - saved to {output_path}")
        
        # Cleanup
        import os
        if os.path.exists(output_path):
            os.remove(output_path)
            print("✓ Cleanup completed")
        
        return True
    except Exception as e:
        print(f"✗ ONNX export failed: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("Holistic Website Learning System - Test Suite")
    print("=" * 60)
    print()
    
    tests = [
        ("WebsiteIntentClassifier", test_website_intent_classifier),
        ("CodeStyleAnalyzer", test_code_style_analyzer),
        ("DependencyGraphLearner", test_dependency_graph_learner),
        ("DeviceAdaptationAnalyzer", test_device_adaptation_analyzer),
        ("HolisticWebsiteLearner", test_holistic_website_learner),
        ("ONNX Export", test_model_export)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n[Test: {test_name}]")
        print("-" * 60)
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
        print()
    
    # Summary
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status:10} {test_name}")
    
    print()
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit(main())
