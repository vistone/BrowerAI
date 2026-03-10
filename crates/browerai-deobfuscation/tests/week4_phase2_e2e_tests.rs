/// Week 4 Phase 2: E2E 集成测试
///
/// 验证完整的检测流程：代码 → 特征提取 → 推理 → 结果
#[cfg(test)]
mod week4_phase2_e2e_tests {
    use browerai_deobfuscation::{FeatureExtractor, OnnxObfuscationDetector};

    // 测试样本定义
    const CONTROL_FLOW_SAMPLE: &str = r#"
var state = 0;
while (true) {
    switch(state) {
        case 0: console.log('a'); state = 1; break;
        case 1: console.log('b'); state = 2; break;
        case 2: console.log('c'); return;
    }
}
"#;

    const STRING_ENCODING_SAMPLE: &str = r#"
var secret = "\x48\x65\x6c\x6c\x6f";
var pass = "\x70\x61\x73\x73\x77\x6f\x72\x64";
console.log(secret);
"#;

    /// 测试 1: 基础检测流程
    #[test]
    fn test_basic_detection_flow() {
        let model_path = "../../models/local/week3_obfuscation_detector.onnx";

        // 步骤 1: 初始化检测器
        let detector = OnnxObfuscationDetector::new(model_path);
        assert!(
            detector.is_ok(),
            "Failed to create detector: {:?}",
            detector.err()
        );
        let detector = detector.unwrap();

        // 步骤 2: 提交简单代码样本
        let code = "function test() { return 42; }";
        let result = detector.detect(code);
        assert!(result.is_ok(), "Detection failed: {:?}", result.err());
        let result = result.unwrap();

        // 步骤 3: 验证结果格式
        assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
        assert_eq!(result.features.len(), 33, "Feature dimension mismatch");
        assert_eq!(result.scores.len(), 8, "Score dimension mismatch");

        // 步骤 4: 检查复杂度指标
        assert!(result.complexity_metrics.code_length > 0);

        println!("✅ Basic flow test passed");
        println!("   Detected: {:?}", result.technique);
        println!("   Confidence: {:.2}%", result.confidence * 100.0);
    }

    /// 测试 2: 多个样本序列处理
    #[test]
    fn test_multiple_samples_sequential() {
        let model_path = "../../models/local/week3_obfuscation_detector.onnx";
        let detector = OnnxObfuscationDetector::new(model_path).unwrap();

        let test_codes = [
            "var a = 1;",
            "function f() { return 2; }",
            "console.log('test');",
            "var x = [1,2,3];",
            "if (true) { ok(); }",
        ];

        let mut results = Vec::new();
        for (i, code) in test_codes.iter().enumerate() {
            let result = detector.detect(code);
            assert!(result.is_ok(), "Sample {} failed", i);
            results.push(result.unwrap());
        }

        // 验证结果独立性
        assert_eq!(results.len(), test_codes.len());

        // 检查每个结果
        for (i, result) in results.iter().enumerate() {
            assert_eq!(result.features.len(), 33, "Sample {} feature dim", i);
            assert!(result.confidence >= 0.0, "Sample {} confidence", i);
        }

        println!(
            "✅ Sequential processing test passed ({} samples)",
            test_codes.len()
        );
    }

    /// 测试 3: 缓存性能测试
    #[test]
    fn test_cache_hit_performance() {
        use std::time::Instant;

        let model_path = "../../models/local/week3_obfuscation_detector.onnx";
        let detector = OnnxObfuscationDetector::new(model_path).unwrap();

        let code = "function cached() { return 42; }";

        // 第一次：冷缓存
        let start = Instant::now();
        let result1 = detector.detect(code).unwrap();
        let cold_time = start.elapsed();

        // 第二次：热缓存
        let start = Instant::now();
        let result2 = detector.detect(code).unwrap();
        let hot_time = start.elapsed();

        // 第三次：确认缓存
        let start = Instant::now();
        let result3 = detector.detect(code).unwrap();
        let hot_time2 = start.elapsed();

        // 验证结果一致性
        assert_eq!(result1.technique, result2.technique);
        assert_eq!(result2.technique, result3.technique);
        assert_eq!(result1.confidence, result2.confidence);

        // 缓存统计
        let cache_size = detector.cache_stats();
        assert_eq!(cache_size, 1, "Cache should have 1 entry");

        println!("✅ Cache test passed");
        println!("   Cold: {:?}", cold_time);
        println!("   Hot 1: {:?}", hot_time);
        println!("   Hot 2: {:?}", hot_time2);
        println!("   Cache entries: {}", cache_size);

        // 缓存应该加速（虽然模拟推理很快，但缓存更快）
        if hot_time < cold_time {
            let speedup = cold_time.as_micros() as f32 / hot_time.as_micros() as f32;
            println!("   Speedup: {:.2}x", speedup);
        }
    }

    /// 测试 4: 控制流扁平化检测
    #[test]
    fn test_control_flow_detection() {
        let model_path = "../../models/local/week3_obfuscation_detector.onnx";
        let detector = OnnxObfuscationDetector::new(model_path).unwrap();

        let samples = [CONTROL_FLOW_SAMPLE, STRING_ENCODING_SAMPLE];

        for (i, code) in samples.iter().enumerate() {
            let result = detector.detect(code);
            assert!(result.is_ok(), "Sample {} failed", i);

            let result = result.unwrap();

            // 验证特征维度
            assert_eq!(result.features.len(), 33);

            // 验证置信度范围
            assert!(result.confidence >= 0.0 && result.confidence <= 1.0);

            println!(
                "Sample {}: {:?} ({:.1}%)",
                i,
                result.technique,
                result.confidence * 100.0
            );
        }

        println!("✅ Control flow detection test passed");
    }

    /// 测试 5: 字符串编码检测 (简化)
    #[test]
    fn test_string_encoding_detection() {
        let model_path = "../../models/local/week3_obfuscation_detector.onnx";
        let detector = OnnxObfuscationDetector::new(model_path).unwrap();

        let result = detector.detect(STRING_ENCODING_SAMPLE).unwrap();

        assert_eq!(result.features.len(), 33);
        assert!(result.confidence >= 0.0);

        println!("✅ String encoding detection test passed");
    }

    /// 测试 6: 批量样本测试
    #[test]
    fn test_all_samples_batch() {
        let model_path = "../../models/local/week3_obfuscation_detector.onnx";
        let detector = OnnxObfuscationDetector::new(model_path).unwrap();

        let test_samples = [
            "var a = 1;",
            "function f() { return 2; }",
            "console.log('test');",
            CONTROL_FLOW_SAMPLE,
            STRING_ENCODING_SAMPLE,
        ];

        let mut success_count = 0;

        for (i, code) in test_samples.iter().enumerate() {
            match detector.detect(code) {
                Ok(result) => {
                    success_count += 1;

                    // 验证基本约束
                    assert_eq!(result.features.len(), 33);
                    assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
                }
                Err(e) => {
                    eprintln!("Sample {} failed: {}", i, e);
                }
            }
        }

        println!("✅ Batch test completed");
        println!("   Total samples: {}", test_samples.len());
        println!("   Successful: {}", success_count);
        println!(
            "   Success rate: {:.1}%",
            (success_count as f32 / test_samples.len() as f32) * 100.0
        );

        // 应该全部成功
        assert_eq!(success_count, test_samples.len());
    }

    /// 测试 7: 特征提取器独立测试
    #[test]
    fn test_feature_extractor_standalone() {
        let extractor = FeatureExtractor::new();

        let test_codes = [
            "var x = 1;",
            "function f() {}",
            "console.log('test');",
            "if (true) { ok(); }",
        ];

        for (i, code) in test_codes.iter().enumerate() {
            let features = extractor.extract_features(code);
            assert!(features.is_ok(), "Feature extraction {} failed", i);

            let features = features.unwrap();
            assert_eq!(features.len(), 33, "Feature dimension mismatch at {}", i);

            // 验证特征值有效性
            for (j, &val) in features.iter().enumerate() {
                assert!(val.is_finite(), "Feature {}[{}] is not finite", i, j);
            }
        }

        println!(
            "✅ Feature extractor test passed ({} codes)",
            test_codes.len()
        );
    }

    /// 测试 8: 空代码和边界条件
    #[test]
    fn test_edge_cases() {
        let model_path = "../../models/local/week3_obfuscation_detector.onnx";
        let detector = OnnxObfuscationDetector::new(model_path).unwrap();

        // 边界 1: 空代码
        let result = detector.detect("");
        if let Ok(r) = result {
            println!(
                "Empty code: {:?} ({:.1}%)",
                r.technique,
                r.confidence * 100.0
            );
        } else {
            println!("Empty code rejected (expected)");
        }

        // 边界 2: 极短代码
        let result = detector.detect("x");
        assert!(result.is_ok(), "Single char failed");

        // 边界 3: 长代码
        let long_code = "var x = 1;\n".repeat(100);
        let result = detector.detect(&long_code);
        assert!(result.is_ok(), "Long code failed");

        // 边界 4: 特殊字符
        let special = "α = 'ñ'; /* 中文 */";
        let result = detector.detect(special);
        assert!(result.is_ok(), "Special chars failed");

        println!("✅ Edge cases test passed");
    }

    /// 测试 9: 清空缓存功能
    #[test]
    fn test_cache_clear() {
        let model_path = "../../models/local/week3_obfuscation_detector.onnx";
        let detector = OnnxObfuscationDetector::new(model_path).unwrap();

        // 填充缓存
        for i in 0..10 {
            let code = format!("var x{} = {};", i, i);
            detector.detect(&code).unwrap();
        }

        // 检查缓存大小
        let size_before = detector.cache_stats();
        assert_eq!(size_before, 10, "Cache should have 10 entries");

        // 清空缓存
        detector.clear_cache();

        // 验证清空
        let size_after = detector.cache_stats();
        assert_eq!(size_after, 0, "Cache should be empty");

        println!("✅ Cache clear test passed");
        println!("   Before: {} entries", size_before);
        println!("   After: {} entries", size_after);
    }

    /// 测试 10: 结果结构完整性
    #[test]
    fn test_result_structure() {
        let model_path = "../../models/local/week3_obfuscation_detector.onnx";
        let detector = OnnxObfuscationDetector::new(model_path).unwrap();

        let code = "function test() { return 42; }";
        let result = detector.detect(code).unwrap();

        // 检查所有字段
        assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
        assert_eq!(result.features.len(), 33);
        assert_eq!(result.scores.len(), 8);
        assert_eq!(result.recovery_guidance.len(), 1024);

        // 复杂度指标
        assert!(result.complexity_metrics.code_length > 0);
        // string_count is usize, always >= 0, so check just passes

        println!("✅ Result structure test passed");
        println!("   Technique: {:?}", result.technique);
        println!("   Confidence: {:.2}%", result.confidence * 100.0);
        println!("   Features: {} dims", result.features.len());
        println!("   Scores: {} dims", result.scores.len());
        println!("   Recovery: {} dims", result.recovery_guidance.len());
        println!("   Code length: {}", result.complexity_metrics.code_length);
    }
}
