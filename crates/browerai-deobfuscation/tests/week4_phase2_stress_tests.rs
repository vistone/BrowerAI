/// Week 4 Phase 2: 压力测试与稳定性验证

#[cfg(test)]
mod week4_phase2_stress_tests {
    use browerai_deobfuscation::OnnxObfuscationDetector;
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    /// 测试 1: 高并发请求
    #[test]
    fn stress_concurrent_requests() {
        let model_path = "../../models/local/week3_obfuscation_detector.onnx";
        let detector = Arc::new(OnnxObfuscationDetector::new(model_path).unwrap());
        
        let num_threads = 10;
        let requests_per_thread = 100;
        
        println!("=== Concurrent Stress Test ===");
        println!("Threads: {}", num_threads);
        println!("Requests per thread: {}", requests_per_thread);
        println!("Total requests: {}", num_threads * requests_per_thread);
        
        let mut handles = vec![];
        
        for thread_id in 0..num_threads {
            let detector_clone = Arc::clone(&detector);
            let handle = thread::spawn(move || {
                let mut success = 0;
                let mut failures = 0;
                
                for i in 0..requests_per_thread {
                    let code = format!("function f{}_{}() {{ return {}; }}", 
                                     thread_id, i, i);
                    match detector_clone.detect(&code) {
                        Ok(result) => {
                            // 验证结果有效性
                            assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
                            assert_eq!(result.features.len(), 33);
                            success += 1;
                        }
                        Err(e) => {
                            eprintln!("Thread {} request {} failed: {}", thread_id, i, e);
                            failures += 1;
                        }
                    }
                }
                
                (success, failures)
            });
            handles.push(handle);
        }
        
        let mut total_success = 0;
        let mut total_failures = 0;
        
        for (i, handle) in handles.into_iter().enumerate() {
            match handle.join() {
                Ok((success, failures)) => {
                    println!("Thread {}: {} success, {} failures", i, success, failures);
                    total_success += success;
                    total_failures += failures;
                }
                Err(_) => {
                    eprintln!("Thread {} panicked!", i);
                    total_failures += requests_per_thread;
                }
            }
        }
        
        let total_requests = num_threads * requests_per_thread;
        let success_rate = (total_success as f64 / total_requests as f64) * 100.0;
        
        println!("\nResults:");
        println!("  Total: {}", total_requests);
        println!("  Success: {}", total_success);
        println!("  Failures: {}", total_failures);
        println!("  Success rate: {:.1}%", success_rate);
        println!("  Cache size: {}", detector.cache_stats());
        
        // 目标: 100% 成功率
        assert_eq!(total_failures, 0, "Some requests failed");
        assert_eq!(total_success, total_requests, "Not all requests succeeded");
        
        println!("✅ Concurrent stress test passed (1000 requests, 0 failures)");
    }

    /// 测试 2: 连续长时间运行
    #[test]
    fn stress_sustained_load() {
        let model_path = "../../models/local/week3_obfuscation_detector.onnx";
        let detector = OnnxObfuscationDetector::new(model_path).unwrap();
        
        let duration_secs = 5; // 5秒持续负载
        let start = std::time::Instant::now();
        let mut count = 0;
        
        println!("=== Sustained Load Test ===");
        println!("Duration: {}s", duration_secs);
        
        while start.elapsed() < Duration::from_secs(duration_secs) {
            let code = format!("var x{} = {};", count, count);
            detector.detect(&code).unwrap();
            count += 1;
        }
        
        let elapsed = start.elapsed();
        let rate = count as f64 / elapsed.as_secs_f64();
        
        println!("Processed: {} requests", count);
        println!("Time: {:.2}s", elapsed.as_secs_f64());
        println!("Rate: {:.2} req/sec", rate);
        println!("Cache size: {}", detector.cache_stats());
        
        // 应该至少处理几百个请求
        assert!(count > 100, "Too few requests processed: {}", count);
        
        println!("✅ Sustained load test passed");
    }

    /// 测试 3: 边界输入稳定性
    #[test]
    fn stress_boundary_inputs() {
        let model_path = "../../models/local/week3_obfuscation_detector.onnx";
        let detector = OnnxObfuscationDetector::new(model_path).unwrap();
        
        println!("=== Boundary Input Test ===");
        
        let long_string = "a".repeat(10000);
        let test_cases = vec![
            ("", "empty"),
            ("x", "single char"),
            (long_string.as_str(), "very long"),
            ("function(){}", "minimal function"),
            ("/* comment */", "only comment"),
            (";;;;;;;;;;;;", "only semicolons"),
            ("{{{{{}}}}}", "only braces"),
            ("const α = 'ñ'; /* 中文 */", "unicode"),
            ("\t\n\r ", "whitespace"),
            ("eval(eval(eval('x')))", "nested eval"),
        ];
        
        let mut handled = 0;
        let mut rejected = 0;
        
        for (code, description) in test_cases {
            match detector.detect(code) {
                Ok(result) => {
                    println!("  ✓ {}: {:?} ({:.1}%)", 
                             description, result.technique, result.confidence * 100.0);
                    handled += 1;
                }
                Err(e) => {
                    println!("  ✗ {}: {}", description, e);
                    rejected += 1;
                }
            }
        }
        
        println!("\nResults:");
        println!("  Handled: {}", handled);
        println!("  Rejected: {}", rejected);
        
        // 大部分应该能处理
        assert!(handled >= 7, "Too many rejections: {}", rejected);
        
        println!("✅ Boundary input test passed");
    }

    /// 测试 4: 内存泄漏检测
    #[test]
    fn stress_memory_leak_check() {
        let model_path = "../../models/local/week3_obfuscation_detector.onnx";
        let detector = OnnxObfuscationDetector::new(model_path).unwrap();
        
        println!("=== Memory Leak Check ===");
        
        // 多轮循环，每轮清空缓存
        for round in 0..5 {
            detector.clear_cache();
            
            // 填充缓存
            for i in 0..200 {
                let code = format!("var x{}_{}= {};", round, i, i);
                detector.detect(&code).unwrap();
            }
            
            let cache_size = detector.cache_stats();
            println!("Round {}: {} cache entries", round, cache_size);
            
            // 验证缓存清空后重建
            assert!(cache_size >= 150, "Cache not properly filled in round {}", round);
        }
        
        // 最终清空
        detector.clear_cache();
        assert_eq!(detector.cache_stats(), 0, "Cache not properly cleared");
        
        println!("✅ No obvious memory leaks detected");
    }

    /// 测试 5: 错误恢复能力
    #[test]
    fn stress_error_recovery() {
        let model_path = "../../models/local/week3_obfuscation_detector.onnx";
        let detector = OnnxObfuscationDetector::new(model_path).unwrap();
        
        println!("=== Error Recovery Test ===");
        
        // 交替正常和异常输入
        let mut success_count = 0;
        let mut error_count = 0;
        
        for i in 0..50 {
            let code = if i % 5 == 0 {
                // 每5个一个异常情况
                ""
            } else {
                "function f() { return 42; }"
            };
            
            match detector.detect(code) {
                Ok(_) => success_count += 1,
                Err(_) => error_count += 1,
            }
        }
        
        println!("Success: {}", success_count);
        println!("Errors: {}", error_count);
        
        // 应该能恢复并继续处理
        assert!(success_count >= 40, "Too many failures after errors");
        
        println!("✅ Error recovery test passed");
    }

    /// 测试 6: 随机输入鲁棒性
    #[test]
    fn stress_random_inputs() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let model_path = "../../models/local/week3_obfuscation_detector.onnx";
        let detector = OnnxObfuscationDetector::new(model_path).unwrap();
        
        println!("=== Random Input Test ===");
        
        let templates = vec![
            "var x = {};",
            "function f() {{ return {}; }}",
            "if ({}) {{ ok(); }}",
            "for (var i=0; i<{}; i++) {{}}",
            "console.log('{}');",
        ];
        
        let mut processed = 0;
        
        for i in 0..100 {
            // 使用哈希生成"随机"值
            let mut hasher = DefaultHasher::new();
            i.hash(&mut hasher);
            let rand_val = hasher.finish() % 1000;
            
            let template = &templates[i % templates.len()];
            let code = template.replace("{}", &rand_val.to_string());
            
            match detector.detect(&code) {
                Ok(result) => {
                    assert_eq!(result.features.len(), 33);
                    processed += 1;
                }
                Err(e) => {
                    eprintln!("Failed on: {} (error: {})", code, e);
                }
            }
        }
        
        println!("Processed: {}/100", processed);
        assert!(processed >= 95, "Too many random input failures");
        
        println!("✅ Random input test passed");
    }

    /// 测试 7: 缓存竞争条件
    #[test]
    fn stress_cache_race_conditions() {
        let model_path = "../../models/local/week3_obfuscation_detector.onnx";
        let detector = Arc::new(OnnxObfuscationDetector::new(model_path).unwrap());
        
        println!("=== Cache Race Condition Test ===");
        
        // 多个线程同时访问相同代码（缓存竞争）
        let shared_code = "function shared() { return 42; }";
        
        let num_threads = 20;
        let mut handles = vec![];
        
        for _i in 0..num_threads {
            let detector_clone = Arc::clone(&detector);
            let code = shared_code.to_string();
            
            let handle = thread::spawn(move || {
                // 每个线程检测100次相同代码
                for _ in 0..100 {
                    detector_clone.detect(&code).unwrap();
                }
            });
            handles.push(handle);
        }
        
        for handle in handles {
            handle.join().unwrap();
        }
        
        // 缓存应该只有1个条目（相同代码）
        let cache_size = detector.cache_stats();
        println!("Cache size: {} (expected: 1)", cache_size);
        assert_eq!(cache_size, 1, "Cache corruption detected");
        
        println!("✅ No race conditions detected");
    }

    /// 测试 8: 综合压力测试
    #[test]
    fn stress_comprehensive() {
        let model_path = "../../models/local/week3_obfuscation_detector.onnx";
        let detector = Arc::new(OnnxObfuscationDetector::new(model_path).unwrap());
        
        println!("\n=== Comprehensive Stress Test ===");
        
        let num_threads = 5;
        let requests_per_thread = 200;
        let total = num_threads * requests_per_thread;
        
        println!("Configuration:");
        println!("  Threads: {}", num_threads);
        println!("  Requests/thread: {}", requests_per_thread);
        println!("  Total: {}", total);
        
        let start = std::time::Instant::now();
        let mut handles = vec![];
        
        for _thread_id in 0..num_threads {
            let detector_clone = Arc::clone(&detector);
            let handle = thread::spawn(move || {
                let mut stats = (0, 0); // (success, failure)
                
                for i in 0..requests_per_thread {
                    // 混合不同类型的输入
                    let code = match i % 4 {
                        0 => format!("var x{} = {};", i, i),
                        1 => format!("function f{}() {{ return {}; }}", i, i),
                        2 => "if (true) { ok(); }".to_string(),
                        _ => "console.log('test');".to_string(),
                    };
                    
                    match detector_clone.detect(&code) {
                        Ok(r) => {
                            assert_eq!(r.features.len(), 33);
                            stats.0 += 1;
                        }
                        Err(_) => stats.1 += 1,
                    }
                }
                
                stats
            });
            handles.push(handle);
        }
        
        let mut total_success = 0;
        let mut total_failure = 0;
        
        for handle in handles {
            let (s, f) = handle.join().unwrap();
            total_success += s;
            total_failure += f;
        }
        
        let elapsed = start.elapsed();
        let throughput = total as f64 / elapsed.as_secs_f64();
        
        println!("\nResults:");
        println!("  Total requests: {}", total);
        println!("  Success: {}", total_success);
        println!("  Failures: {}", total_failure);
        println!("  Success rate: {:.1}%", (total_success as f64 / total as f64) * 100.0);
        println!("  Time: {:.2}s", elapsed.as_secs_f64());
        println!("  Throughput: {:.2} req/sec", throughput);
        println!("  Cache size: {}", detector.cache_stats());
        
        // 验证
        assert_eq!(total_failure, 0, "Had failures");
        assert_eq!(total_success, total, "Not all succeeded");
        
        println!("\n✅ Comprehensive stress test passed");
        println!("   {} concurrent requests, 0 failures, {:.2} req/sec", 
                 total, throughput);
    }
}
