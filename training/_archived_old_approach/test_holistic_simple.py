"""
Quick Test: Holistic Website Learning System - Simplified

Validates basic model loading and structure.

Usage:
    python scripts/test_holistic_simple.py
"""

import sys
from pathlib import Path
import torch
import yaml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.models.website_learner import (
    HolisticWebsiteLearner,
    WebsiteIntentClassifier,
    CodeStyleAnalyzer,
    DependencyGraphLearner,
    DeviceAdaptationAnalyzer
)

def test_models_initialize():
    """Test that all models can be initialized"""
    print("Testing model initialization...")
    
    try:
        # Test WebsiteIntentClassifier
        intent_classifier = WebsiteIntentClassifier(d_model=256, num_categories=10)
        print("✓ WebsiteIntentClassifier initialized")
        
        # Test CodeStyleAnalyzer
        style_analyzer = CodeStyleAnalyzer(d_model=256)
        print("✓ CodeStyleAnalyzer initialized")
        
        # Test DependencyGraphLearner
        dep_learner = DependencyGraphLearner(d_model=256)
        print("✓ DependencyGraphLearner initialized")
        
        # Test DeviceAdaptationAnalyzer
        device_analyzer = DeviceAdaptationAnalyzer(d_model=256)
        print("✓ DeviceAdaptationAnalyzer initialized")
        
        # Test HolisticWebsiteLearner with config
        config = {
            "vocab_size": 10000,
            "d_model": 256,
            "num_heads": 8,
            "num_layers": 4,
            "d_ff": 1024,
            "dropout": 0.1,
            "max_len": 512
        }
        holistic_learner = HolisticWebsiteLearner(config)
        print("✓ HolisticWebsiteLearner initialized")
        
        return True
    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_loading():
    """Test that YAML config can be loaded"""
    print("\nTesting configuration loading...")
    
    try:
        config_path = Path(__file__).parent.parent / "configs" / "website_learner.yaml"
        
        if not config_path.exists():
            print(f"⚠ Config file not found at {config_path}")
            return False
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"✓ Configuration loaded from {config_path}")
        print(f"  - Model d_model: {config['model']['d_model']}")
        print(f"  - Training epochs: {config['training']['epochs']}")
        print(f"  - Categories: {config['model']['num_categories']}")
        
        return True
    except Exception as e:
        print(f"✗ Config loading failed: {e}")
        return False


def test_parameter_count():
    """Test parameter counting"""
    print("\nTesting parameter count...")
    
    try:
        config = {
            "vocab_size": 10000,
            "d_model": 256,
            "num_heads": 8,
            "num_layers": 4,
            "d_ff": 1024,
            "dropout": 0.1,
            "max_len": 512
        }
        
        model = HolisticWebsiteLearner(config)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"✓ Total parameters: {total_params:,}")
        print(f"✓ Trainable parameters: {trainable_params:,}")
        print(f"✓ Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB")
        
        return True
    except Exception as e:
        print(f"✗ Parameter counting failed: {e}")
        return False


def test_model_structure():
    """Test model has expected components"""
    print("\nTesting model structure...")
    
    try:
        config = {
            "vocab_size": 10000,
            "d_model": 256,
            "num_heads": 8,
            "num_layers": 4,
            "d_ff": 1024,
            "dropout": 0.1,
            "max_len": 512
        }
        
        model = HolisticWebsiteLearner(config)
        
        # Check for expected components
        expected_components = [
            'shared_embedding',
            'html_encoder',
            'css_encoder',
            'js_encoder',
            'intent_classifier',
            'style_analyzer',
            'dependency_learner'
        ]
        
        for component in expected_components:
            if hasattr(model, component):
                print(f"✓ Found component: {component}")
            else:
                print(f"✗ Missing component: {component}")
                return False
        
        return True
    except Exception as e:
        print(f"✗ Structure check failed: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("Holistic Website Learning System - Simple Test Suite")
    print("=" * 60)
    print()
    
    tests = [
        ("Model Initialization", test_models_initialize),
        ("Configuration Loading", test_config_loading),
        ("Parameter Count", test_parameter_count),
        ("Model Structure", test_model_structure)
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
    
    # Summary
    print("\n" + "=" * 60)
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
        print("\nNext steps:")
        print("1. Prepare website data: python scripts/prepare_website_data.py")
        print("2. Train the model: python scripts/train_holistic_website.py --config configs/website_learner.yaml")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit(main())
