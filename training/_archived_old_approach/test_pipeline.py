#!/usr/bin/env python3
"""
End-to-end test of the training pipeline
"""
import json
from pathlib import Path
import sys

def test_feature_extraction():
    """Test feature extraction output"""
    features_path = Path('features/extracted_features.jsonl')
    if not features_path.exists():
        print("❌ Features file not found")
        return False
    
    # Load and validate
    with open(features_path, 'r') as f:
        lines = f.readlines()
    
    if len(lines) < 10:
        print(f"❌ Too few features: {len(lines)}")
        return False
    
    # Validate first feature
    feat = json.loads(lines[0])
    required_keys = ['event_type', 'url_features']
    if not all(k in feat for k in required_keys):
        print(f"❌ Missing required keys in features")
        return False
    
    print(f"✅ Feature extraction: {len(lines)} features")
    return True

def test_classifier():
    """Test classifier model"""
    model_path = Path('models/site_classifier.pkl')
    if not model_path.exists():
        print("❌ Classifier model not found")
        return False
    
    size = model_path.stat().st_size
    if size < 100:
        print(f"❌ Classifier too small: {size} bytes")
        return False
    
    print(f"✅ Classifier trained: {size} bytes")
    return True

def test_theme_generator():
    """Test theme generator output"""
    theme_path = Path('models/sample_theme_recommendations.json')
    if not theme_path.exists():
        print("❌ Theme recommendations not found")
        return False
    
    with open(theme_path, 'r') as f:
        data = json.load(f)
    
    if 'alternative_themes' not in data:
        print("❌ Missing alternative_themes in output")
        return False
    
    themes = data['alternative_themes']
    if len(themes) < 3:
        print(f"❌ Too few themes: {len(themes)}")
        return False
    
    # Validate first theme
    theme = themes[0]
    required = ['scheme_name', 'colors', 'generated_css']
    if not all(k in theme for k in required):
        print(f"❌ Missing required keys in theme")
        return False
    
    print(f"✅ Theme generator: {len(themes)} themes")
    return True

def test_obfuscation_analysis():
    """Test obfuscation analysis"""
    analysis_path = Path('data/obfuscation_analysis.json')
    if not analysis_path.exists():
        print("❌ Obfuscation analysis not found")
        return False
    
    with open(analysis_path, 'r') as f:
        data = json.load(f)
    
    if len(data) < 5:
        print(f"❌ Too few analyzed events: {len(data)}")
        return False
    
    print(f"✅ Obfuscation analysis: {len(data)} events")
    return True

def test_data_collection():
    """Test data collection"""
    data_dir = Path('data')
    feedback_files = list(data_dir.glob('feedback_*.json'))
    
    if len(feedback_files) < 10:
        print(f"❌ Too few feedback files: {len(feedback_files)}")
        return False
    
    # Check a recent file has content
    recent = sorted(feedback_files)[-1]
    with open(recent, 'r') as f:
        events = json.load(f)
    
    if len(events) < 1:
        print(f"❌ Empty feedback file: {recent}")
        return False
    
    # Check for content field
    has_content = any('content' in e for e in events)
    if not has_content:
        print(f"❌ No content field in events")
        return False
    
    print(f"✅ Data collection: {len(feedback_files)} files, {len(events)} events in latest")
    return True

def main():
    """Run all tests"""
    print("🧪 Running BrowerAI Training Pipeline Tests\n")
    
    tests = [
        ("Data Collection", test_data_collection),
        ("Feature Extraction", test_feature_extraction),
        ("Site Classifier", test_classifier),
        ("Theme Generator", test_theme_generator),
        ("Obfuscation Analysis", test_obfuscation_analysis),
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\n📋 Testing: {name}")
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*60)
    print("📊 Test Summary")
    print("="*60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}")
    
    print(f"\n🎯 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1

if __name__ == '__main__':
    sys.exit(main())
