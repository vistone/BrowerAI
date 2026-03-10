"""
Comprehensive Test Suite for Enhanced Framework Detector
验证性能评估、多源集成、动态规则加载等功能
"""

import sys
sys.path.insert(0, '/home/stone/BrowerAI/training')

import json
import os
import tempfile
from pathlib import Path
from framework_detector_enhanced import (
    EnsembleFrameworkDetector,
    RuleBasedDetector,
    FrameworkPattern,
    DetectionMethod,
    PerformanceMetrics
)


def test_rule_detector_initialization():
    """测试规则检测器初始化"""
    print("\n" + "="*60)
    print("TEST: Rule Detector Initialization")
    print("="*60)
    
    detector = RuleBasedDetector()
    
    assert len(detector.frameworks) >= 8, "Should have at least 8 frameworks"
    assert "react" in detector.frameworks
    assert "vue" in detector.frameworks
    assert "angular" in detector.frameworks
    
    print(f"✓ Initialized {len(detector.frameworks)} frameworks")
    print(f"✓ Default frameworks: {list(detector.frameworks.keys())[:5]}...")
    
    for fw in ["react", "vue", "nextjs"]:
        patterns = detector.frameworks[fw].patterns
        print(f"✓ {fw}: {len(patterns)} detection patterns")
    
    print("\n✅ PASSED: Rule detector initialization\n")


def test_react_detection():
    """测试React框架检测"""
    print("\n" + "="*60)
    print("TEST: React Framework Detection")
    print("="*60)
    
    detector = RuleBasedDetector()
    
    # React代码示例
    react_code = """
    import React, { useState, useEffect } from 'react';
    
    export function Counter() {
        const [count, setCount] = useState(0);
        
        useEffect(() => {
            console.log('Count:', count);
        }, [count]);
        
        return (
            <>
                <button onClick={() => setCount(c => c + 1)}>
                    Increment: {count}
                </button>
            </>
        );
    }
    """
    
    results = detector.detect(react_code)
    
    assert "react" in results, "Should detect React"
    confidence = results["react"][0]
    assert confidence > 0.3, f"Confidence should be > 0.3, got {confidence}"
    
    matched_patterns = results["react"][1]
    assert len(matched_patterns) > 0, "Should find matched patterns"
    
    print(f"✓ Detected framework: react")
    print(f"✓ Confidence: {confidence:.4f}")
    print(f"✓ Matched patterns: {len(matched_patterns)}")
    print(f"  Pattern samples: {matched_patterns[:3]}")
    
    print("\n✅ PASSED: React detection\n")


def test_vue_detection():
    """测试Vue框架检测"""
    print("\n" + "="*60)
    print("TEST: Vue Framework Detection")
    print("="*60)
    
    detector = RuleBasedDetector()
    
    vue_code = """
    import { defineComponent, ref, reactive } from 'vue';
    
    export default defineComponent({
        name: 'Counter',
        template: `
            <div>
                <p>Count: {{ count }}</p>
                <button @click="increment">Increment</button>
            </div>
        `,
        setup() {
            const count = ref(0);
            
            return {
                count,
                increment: () => count.value++
            };
        }
    });
    """
    
    results = detector.detect(vue_code)
    
    assert "vue" in results, "Should detect Vue"
    confidence = results["vue"][0]
    assert confidence > 0.3, f"Confidence should be > 0.3, got {confidence}"
    
    print(f"✓ Detected framework: vue")
    print(f"✓ Confidence: {confidence:.4f}")
    print(f"✓ Matched patterns: {len(results['vue'][1])}")
    
    print("\n✅ PASSED: Vue detection\n")


def test_nextjs_detection():
    """测试Next.js框架检测"""
    print("\n" + "="*60)
    print("TEST: Next.js Framework Detection")
    print("="*60)
    
    detector = RuleBasedDetector()
    
    nextjs_code = """
    import { GetServerSideProps } from 'next';
    import Image from 'next/image';
    import { useRouter } from 'next/router';
    
    export default function Page({ data }) {
        const router = useRouter();
        
        return (
            <div>
                <Image src="/image.png" />
                <button onClick={() => router.push('/')}>Home</button>
            </div>
        );
    }
    
    export const getServerSideProps: GetServerSideProps = async (context) => {
        return { props: { data: [] } };
    };
    """
    
    results = detector.detect(nextjs_code)
    
    assert "nextjs" in results, "Should detect Next.js"
    confidence = results["nextjs"][0]
    assert confidence > 0.5, f"Confidence should be > 0.5, got {confidence}"
    
    print(f"✓ Detected framework: nextjs")
    print(f"✓ Confidence: {confidence:.4f}")
    print(f"✓ Priority: {detector.frameworks['nextjs'].priority}")
    
    print("\n✅ PASSED: Next.js detection\n")


def test_dynamic_rule_loading():
    """测试动态规则加载"""
    print("\n" + "="*60)
    print("TEST: Dynamic Rule Loading")
    print("="*60)
    
    detector = RuleBasedDetector()
    
    # 创建临时规则文件
    temp_rules = {
        "custom_framework": {
            "patterns": [
                r"custom_import",
                r"CustomClass",
                r"useCustom"
            ],
            "weight": 1.5,
            "min_matches": 1,
            "priority": 110
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(temp_rules, f)
        temp_file = f.name
    
    try:
        # 加载规则
        success = detector.load_rules_from_file(temp_file)
        assert success, "Should successfully load rules"
        assert "custom_framework" in detector.frameworks
        
        custom_pattern = detector.frameworks["custom_framework"]
        assert custom_pattern.weight == 1.5
        assert custom_pattern.priority == 110
        
        print("✓ Rules loaded from file")
        print(f"✓ Custom framework added: {custom_pattern.name}")
        print(f"✓ Weight: {custom_pattern.weight}")
        print(f"✓ Priority: {custom_pattern.priority}")
        
        # 测试检测
        test_code = "const x = useCustom();"
        results = detector.detect(test_code)
        
        if "custom_framework" in results:
            print(f"✓ Custom framework detected with confidence: {results['custom_framework'][0]:.4f}")
        
        # 保存规则
        save_temp = tempfile.NamedTemporaryFile(suffix='.json', delete=False).name
        save_success = detector.save_rules_to_file(save_temp)
        assert save_success, "Should successfully save rules"
        
        # 验证保存的文件
        with open(save_temp, 'r') as f:
            saved_rules = json.load(f)
        
        assert "custom_framework" in saved_rules
        print("✓ Rules saved to file")
        
        os.unlink(save_temp)
        
    finally:
        os.unlink(temp_file)
    
    print("\n✅ PASSED: Dynamic rule loading\n")


def test_ensemble_detector():
    """测试集合检测器"""
    print("\n" + "="*60)
    print("TEST: Ensemble Framework Detector")
    print("="*60)
    
    detector = EnsembleFrameworkDetector()
    
    # 单个检测
    react_code = "import React, { useState } from 'react';"
    result = detector.detect(react_code)
    
    assert result.framework == "react"
    assert result.confidence > 0
    assert result.method == DetectionMethod.RULE_BASED
    
    print(f"✓ Single detection: {result.framework}")
    print(f"✓ Confidence: {result.confidence:.4f}")
    print(f"✓ Method: {result.method.value}")
    print(f"✓ Timestamp: {result.timestamp}")
    
    print("\n✅ PASSED: Ensemble detector\n")


def test_batch_detection():
    """测试批量检测"""
    print("\n" + "="*60)
    print("TEST: Batch Detection & Metrics")
    print("="*60)
    
    detector = EnsembleFrameworkDetector()
    
    test_cases = [
        {
            "code": "import React from 'react';",
            "expected_framework": "react"
        },
        {
            "code": "import Vue from 'vue';",
            "expected_framework": "vue"
        },
        {
            "code": "import { Component } from '@angular/core';",
            "expected_framework": "angular"
        },
        {
            "code": "import { createApp } from 'vue';",
            "expected_framework": "vue"
        },
    ]
    
    # 批量检测
    results = detector.batch_detect(test_cases)
    
    assert len(results) == len(test_cases)
    
    print(f"✓ Batch detection completed: {len(results)} samples")
    
    # 检查指标
    metrics = detector.get_metrics()
    
    print(f"✓ Total detections: {metrics.total_detections}")
    print(f"✓ Correct detections: {metrics.correct_detections}")
    print(f"✓ Accuracy: {metrics.calculate_accuracy():.2%}")
    print(f"✓ Average confidence: {metrics.avg_confidence:.4f}")
    
    print("\n✅ PASSED: Batch detection\n")


def test_source_weighting():
    """测试源权重设置"""
    print("\n" + "="*60)
    print("TEST: Source Weighting")
    print("="*60)
    
    detector = EnsembleFrameworkDetector()
    
    # 原始权重
    original_weights = detector.confidence_weights.copy()
    print(f"✓ Original weights: {original_weights}")
    
    # 设置新权重
    new_weights = {
        'rule_based': 0.8,
        'ai_based': 0.2
    }
    detector.set_source_weights(new_weights)
    
    # 验证权重已更新
    print(f"✓ Updated weights: {detector.confidence_weights}")
    assert abs(detector.confidence_weights['rule_based'] - 0.8) < 0.01
    
    print("\n✅ PASSED: Source weighting\n")


def test_performance_metrics():
    """测试性能指标计算"""
    print("\n" + "="*60)
    print("TEST: Performance Metrics Calculation")
    print("="*60)
    
    metrics = PerformanceMetrics()
    
    # 模拟检测结果
    metrics.total_detections = 100
    metrics.correct_detections = 85
    metrics.avg_confidence = 0.75
    
    # 为React框架添加统计
    metrics.framework_stats['react']['true_positive'] = 40
    metrics.framework_stats['react']['false_positive'] = 5
    metrics.framework_stats['react']['false_negative'] = 10
    
    # 计算指标
    accuracy = metrics.calculate_accuracy()
    precision = metrics.calculate_framework_precision('react')
    recall = metrics.calculate_framework_recall('react')
    f1 = metrics.calculate_f1_score('react')
    
    print(f"✓ Accuracy: {accuracy:.2%}")
    assert accuracy == 0.85
    
    print(f"✓ Precision (react): {precision:.4f}")
    print(f"✓ Recall (react): {recall:.4f}")
    print(f"✓ F1 Score (react): {f1:.4f}")
    
    assert precision == 40 / (40 + 5)
    assert recall == 40 / (40 + 10)
    
    print("\n✅ PASSED: Performance metrics\n")


def test_detection_history():
    """测试检测历史追踪"""
    print("\n" + "="*60)
    print("TEST: Detection History Tracking")
    print("="*60)
    
    detector = EnsembleFrameworkDetector()
    
    # 执行多次检测
    codes = [
        "import React from 'react';",
        "import Vue from 'vue';",
        "import { Component } from '@angular/core';",
    ]
    
    for code in codes:
        detector.detect(code)
    
    # 获取历史
    history = detector.get_detection_history(limit=10)
    
    assert len(history) == 3
    assert history[0]['framework'] == 'react'
    assert history[1]['framework'] == 'vue'
    assert history[2]['framework'] == 'angular'
    
    print(f"✓ Recorded {len(history)} detections in history")
    
    for i, record in enumerate(history):
        print(f"  {i+1}. {record['framework']} (confidence: {record['confidence']:.4f})")
    
    print("\n✅ PASSED: Detection history\n")


def test_trend_analysis():
    """测试趋势分析"""
    print("\n" + "="*60)
    print("TEST: Trend Analysis")
    print("="*60)
    
    detector = EnsembleFrameworkDetector()
    
    # 执行多次检测
    codes = [
        "import React from 'react';",
        "import React from 'react';",
        "import React from 'react';",
        "import Vue from 'vue';",
        "import { Component } from '@angular/core';",
    ]
    
    for code in codes:
        detector.detect(code)
    
    # 分析趋势
    trends = detector.analyze_detection_trends()
    
    print(f"✓ Trend analysis completed")
    
    for framework, percentage in trends.items():
        print(f"  {framework}: {percentage:.1%}")
    
    assert trends.get('react', 0) == 0.6  # 3/5
    
    print("\n✅ PASSED: Trend analysis\n")


def test_metrics_export_import():
    """测试指标导出和导入"""
    print("\n" + "="*60)
    print("TEST: Metrics Export & Import")
    print("="*60)
    
    detector = EnsembleFrameworkDetector()
    
    # 执行一些检测
    test_cases = [
        {"code": "import React from 'react';", "expected_framework": "react"},
        {"code": "import Vue from 'vue';", "expected_framework": "vue"},
    ]
    detector.batch_detect(test_cases)
    
    # 导出指标
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_file = f.name
    
    try:
        export_success = detector.export_metrics_to_file(temp_file)
        assert export_success, "Should successfully export metrics"
        
        # 验证导出文件
        with open(temp_file, 'r') as f:
            exported_data = json.load(f)
        
        assert 'timestamp' in exported_data
        assert 'summary' in exported_data
        assert 'trends' in exported_data
        
        print(f"✓ Metrics exported successfully")
        print(f"  - Total detections: {exported_data['summary']['total_detections']}")
        print(f"  - Accuracy: {exported_data['summary']['accuracy']:.2%}")
        
        # 导入指标
        import_success = detector.import_metrics_from_file(temp_file)
        assert import_success, "Should successfully import metrics"
        
        print(f"✓ Metrics imported successfully")
        
    finally:
        os.unlink(temp_file)
    
    print("\n✅ PASSED: Metrics export/import\n")


def test_stress_test():
    """应力测试"""
    print("\n" + "="*60)
    print("TEST: Stress Test (500 detections)")
    print("="*60)
    
    detector = EnsembleFrameworkDetector()
    
    test_codes = [
        ("import React from 'react';", "react"),
        ("import Vue from 'vue';", "vue"),
        ("import { Component } from '@angular/core';", "angular"),
    ]
    
    # 执行500次检测
    detection_count = 0
    for _ in range(167):  # 3 x 167 = 501
        for code, expected in test_codes:
            result = detector.detect(code)
            detector._update_metrics(result, expected)
            detection_count += 1
    
    # 获取指标
    summary = detector.get_metrics_summary()
    
    print(f"✓ Completed {detection_count} detections")
    print(f"✓ Accuracy: {summary['accuracy']:.2%}")
    print(f"✓ Correct: {summary['correct_detections']}/{summary['total_detections']}")
    print(f"✓ Average confidence: {summary['avg_confidence']:.4f}")
    
    # 分析趋势
    trends = detector.analyze_detection_trends()
    for fw, pct in trends.items():
        print(f"  {fw}: {pct:.1%}")
    
    print("\n✅ PASSED: Stress test\n")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*70)
    print("ENHANCED FRAMEWORK DETECTOR - COMPREHENSIVE TEST SUITE")
    print("="*70)
    
    tests = [
        test_rule_detector_initialization,
        test_react_detection,
        test_vue_detection,
        test_nextjs_detection,
        test_dynamic_rule_loading,
        test_ensemble_detector,
        test_batch_detection,
        test_source_weighting,
        test_performance_metrics,
        test_detection_history,
        test_trend_analysis,
        test_metrics_export_import,
        test_stress_test,
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
    
    # 总结
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
