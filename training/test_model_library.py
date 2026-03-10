#!/usr/bin/env python3
"""
Model Library Comprehensive Test Suite
验证模型库的所有功能组件
"""

import sys
sys.path.insert(0, '/home/stone/BrowerAI/training')

import numpy as np
from model_library import (
    ModelLibrary, ModelLibraryConfig, FeatureExtractor,
    LatentEncoder, CodeGenerationModel, QualityValidator,
    LearningTracker
)


def test_feature_extractor():
    """测试特征提取器"""
    print("\n" + "="*70)
    print("TEST 1: Feature Extractor (48D特征提取)")
    print("="*70)
    
    extractor = FeatureExtractor()
    
    # 测试网站数据
    website_data = {
        'html': '<html><head><title>Test</title></head><body><div><p>Hello</p></div></body></html>',
        'css': 'body { margin: 0; } .container { display: flex; }',
        'scripts': 'let x = 1; function test() { return x; }',
    }
    
    features = extractor.extract(website_data)
    
    # 验证
    assert features.shape == (48,), f"Shape mismatch: {features.shape}"
    assert features.dtype == np.float32, f"Type mismatch: {features.dtype}"
    assert np.all(features >= 0) and np.all(features <= 1), "Features not in [0, 1] range"
    assert not np.any(np.isnan(features)), "Features contain NaN"
    assert not np.any(np.isinf(features)), "Features contain Inf"
    
    print(f"✅ Features extracted successfully")
    print(f"   Shape: {features.shape}")
    print(f"   Range: [{features.min():.3f}, {features.max():.3f}]")
    print(f"   Mean: {features.mean():.3f}, Std: {features.std():.3f}")
    print(f"   HTML metrics [0-9]: {features[0:10]}")
    print(f"   CSS metrics [10-17]: {features[10:18]}")
    print(f"   JS metrics [18-27]: {features[18:28]}")


def test_latent_encoder():
    """测试潜在编码器"""
    print("\n" + "="*70)
    print("TEST 2: Latent Encoder (48D → 256D编码)")
    print("="*70)
    
    encoder = LatentEncoder(feature_dim=48, latent_dim=256)
    
    # 创建测试特征
    features = np.random.rand(48).astype(np.float32)
    
    # 编码
    latent = encoder.encode(features, intent='blog', style='modern')
    
    # 验证
    assert latent.shape == (256,), f"Shape mismatch: {latent.shape}"
    assert latent.dtype == np.float32, f"Type mismatch: {latent.dtype}"
    assert not np.any(np.isnan(latent)), "Latent contains NaN"
    assert not np.any(np.isinf(latent)), "Latent contains Inf"
    
    # 解码测试
    decoded = encoder.decode(latent)
    assert decoded.shape == (48,), f"Decoded shape mismatch"
    
    print(f"✅ Encoding successful")
    print(f"   Input shape: (48,)")
    print(f"   Latent shape: {latent.shape}")
    print(f"   Latent norm: {np.linalg.norm(latent):.3f}")
    print(f"   Decoded shape: {decoded.shape}")
    print(f"   Encoding count: {encoder.encoding_count}")


def test_code_generation():
    """测试代码生成"""
    print("\n" + "="*70)
    print("TEST 3: Code Generation (256D → HTML/CSS/JS)")
    print("="*70)
    
    generator = CodeGenerationModel(latent_dim=256)
    
    # 创建潜在向量
    latent = np.random.randn(256).astype(np.float32)
    
    # 生成代码
    code = generator.generate(latent, intent='ecommerce')
    
    # 验证
    assert 'html' in code, "Missing HTML"
    assert 'css' in code, "Missing CSS"
    assert 'javascript' in code, "Missing JavaScript"
    assert code['generation_id'] > 0, "Generation ID not set"
    
    print(f"✅ Code generation successful")
    print(f"   HTML length: {len(code['html'])} chars")
    print(f"   CSS length: {len(code['css'])} chars")
    print(f"   JavaScript length: {len(code['javascript'])} chars")
    print(f"   Generation ID: {code['generation_id']}")
    print(f"   Intent: {code['intent']}")
    print(f"   HTML preview: {code['html'][:50]}...")


def test_quality_validator():
    """测试质量验证器"""
    print("\n" + "="*70)
    print("TEST 4: Quality Validator (代码质量评估)")
    print("="*70)
    
    validator = QualityValidator()
    
    # 测试代码
    test_code = {
        'html': '<!DOCTYPE html><html><head><title>Test</title></head><body><h1>Hello</h1></body></html>',
        'css': 'body { font-family: Arial; } h1 { color: #333; }',
        'javascript': 'function test() { console.log("test"); };'
    }
    
    scores = validator.validate(test_code)
    
    # 验证
    assert 'html_quality' in scores, "Missing HTML quality"
    assert 'css_quality' in scores, "Missing CSS quality"
    assert 'js_quality' in scores, "Missing JS quality"
    assert 'overall_quality' in scores, "Missing overall quality"
    assert 0 <= scores['overall_quality'] <= 1, "Quality score out of range"
    
    print(f"✅ Quality validation successful")
    print(f"   HTML quality: {scores['html_quality']:.3f}")
    print(f"   CSS quality: {scores['css_quality']:.3f}")
    print(f"   JS quality: {scores['js_quality']:.3f}")
    print(f"   Overall quality: {scores['overall_quality']:.3f}")
    print(f"   Validation count: {validator.validation_count}")


def test_learning_tracker():
    """测试学习追踪器"""
    print("\n" + "="*70)
    print("TEST 5: Learning Tracker (学习指标追踪)")
    print("="*70)
    
    tracker = LearningTracker()
    
    # 模拟学习过程
    for i in range(10):
        loss = 0.5 - (i * 0.04)  # Loss decreasing
        quality = 0.5 + (i * 0.04)  # Quality improving
        tracker.log_sample(loss, quality, framework='react')
        tracker.log_learning_update(gradient_norm=0.01, learning_rate=0.001)
        tracker.log_processing_time(5.0 + np.random.rand() * 2.0)
    
    summary = tracker.get_summary()
    
    # 验证
    assert summary['total_samples'] == 10, "Sample count mismatch"
    assert summary['learning_iterations'] == 10, "Iteration count mismatch"
    assert summary['average_loss'] > 0, "Loss should be positive"
    assert summary['average_quality'] > 0, "Quality should be positive"
    
    print(f"✅ Learning tracking successful")
    print(f"   Total samples: {summary['total_samples']}")
    print(f"   Learning iterations: {summary['learning_iterations']}")
    print(f"   Average loss: {summary['average_loss']:.4f}")
    print(f"   Average quality: {summary['average_quality']:.3f}")
    print(f"   Average processing time: {summary['average_processing_time_ms']:.2f}ms")
    print(f"   Framework distribution: {summary['framework_distribution']}")
    print(f"   Elapsed time: {summary['elapsed_seconds']:.2f}s")


def test_complete_pipeline():
    """测试完整的模型库管道"""
    print("\n" + "="*70)
    print("TEST 6: Complete Model Library Pipeline (完整管道)")
    print("="*70)
    
    config = ModelLibraryConfig()
    library = ModelLibrary(config=config)
    
    # 创建测试网站
    websites = [
        {
            'html': '<html><body><header><nav></nav></header><main><article></article></main></body></html>',
            'css': 'body { margin: 0; } header { background: #333; }',
            'scripts': 'document.addEventListener("DOMContentLoaded", () => {});',
            'framework': 'react',
            'intent': 'blog',
            'style': 'modern',
        },
        {
            'html': '<html><body><div><form><input type="text"><button>Submit</button></form></div></body></html>',
            'css': 'form { display: flex; } input { border: 1px solid #ccc; }',
            'scripts': 'const form = document.querySelector("form"); form.addEventListener("submit", (e) => {});',
            'framework': 'vue',
            'intent': 'ecommerce',
            'style': 'minimal',
        },
    ]
    
    # 处理网站
    results = []
    for website in websites:
        result = library.process_website(website)
        results.append(result)
    
    # 验证
    assert len(results) == 2, "Result count mismatch"
    for result in results:
        assert 'features' in result, "Missing features"
        assert 'latent' in result, "Missing latent"
        assert 'generated_code' in result, "Missing generated code"
        assert 'quality_scores' in result, "Missing quality scores"
        assert result['status'] == 'success', "Processing failed"
    
    status = library.get_model_status()
    
    print(f"✅ Complete pipeline successful")
    print(f"   Websites processed: 2")
    print(f"   Total samples: {status['learning_tracker']['total_samples']}")
    print(f"   Feature extractions: {status['feature_extractor']['extractions']}")
    print(f"   Latent encodings: {status['latent_encoder']['encodings']}")
    print(f"   Code generations: {status['code_generator']['generations']}")
    print(f"   Quality validations: {status['quality_validator']['validations']}")
    print(f"   Average quality: {status['learning_tracker']['average_quality']:.3f}")


def test_batch_processing():
    """测试批量处理"""
    print("\n" + "="*70)
    print("TEST 7: Batch Processing (批量处理3个网站)")
    print("="*70)
    
    library = ModelLibrary()
    
    # 创建多个网站
    websites = []
    for i in range(3):
        website = {
            'html': f'<html><body><h1>Website {i}</h1></body></html>',
            'css': f'h1 {{ color: color_{i}; }}',
            'scripts': f'console.log("Website {i}");',
            'framework': ['react', 'vue', 'angular'][i],
            'intent': 'blog',
        }
        websites.append(website)
    
    # 批量处理
    batch_result = library.batch_process(websites)
    
    # 验证
    assert batch_result['total_processed'] == 3, "Total count mismatch"
    assert batch_result['successful'] == 3, "Success count mismatch"
    assert batch_result['failed'] == 0, "Error count should be 0"
    
    print(f"✅ Batch processing successful")
    print(f"   Total processed: {batch_result['total_processed']}")
    print(f"   Successful: {batch_result['successful']}")
    print(f"   Failed: {batch_result['failed']}")
    print(f"   Processing times (ms):")
    for i, result in enumerate(batch_result['results']):
        print(f"     Website {i}: {result['processing_time_ms']:.2f}ms")
    print(f"   Summary:")
    summary = batch_result['summary']
    print(f"     Average quality: {summary['average_quality']:.3f}")
    print(f"     Average loss: {summary['average_loss']:.4f}")


def test_model_persistence():
    """测试模型持久化"""
    print("\n" + "="*70)
    print("TEST 8: Model Persistence (模型保存和加载)")
    print("="*70)
    
    import tempfile
    import os
    
    # 创建临时文件
    fd, temp_path = tempfile.mkstemp(suffix='.pkl')
    os.close(fd)
    
    try:
        # 保存模型
        library1 = ModelLibrary()
        original_encoding_matrix = library1.latent_encoder.weight_matrix.copy()
        library1.save_model(temp_path)
        
        # 加载模型
        library2 = ModelLibrary()
        library2.load_model(temp_path)
        
        # 验证
        loaded_encoding_matrix = library2.latent_encoder.weight_matrix
        assert np.allclose(original_encoding_matrix, loaded_encoding_matrix), "Weight matrix mismatch"
        
        print(f"✅ Model persistence successful")
        print(f"   Saved to: {temp_path}")
        print(f"   Weight matrix shape: {original_encoding_matrix.shape}")
        print(f"   Weight matrix verified: ✓")
    
    finally:
        # 清理
        if os.path.exists(temp_path):
            os.remove(temp_path)


def run_all_tests():
    """执行所有测试"""
    print("\n" + "="*70)
    print("🧠 BrowerAI Model Library - Complete Test Suite")
    print("="*70)
    
    tests = [
        ("Feature Extractor", test_feature_extractor),
        ("Latent Encoder", test_latent_encoder),
        ("Code Generation", test_code_generation),
        ("Quality Validator", test_quality_validator),
        ("Learning Tracker", test_learning_tracker),
        ("Complete Pipeline", test_complete_pipeline),
        ("Batch Processing", test_batch_processing),
        ("Model Persistence", test_model_persistence),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"\n❌ {name} FAILED: {e}")
            import traceback
            traceback.print_exc()
    
    # 总结
    print("\n" + "="*70)
    print("📊 Test Summary")
    print("="*70)
    print(f"✅ Passed: {passed}/{len(tests)}")
    print(f"❌ Failed: {failed}/{len(tests)}")
    if failed == 0:
        print("\n🎉 All tests passed!")
    print("="*70 + "\n")


if __name__ == '__main__':
    run_all_tests()
