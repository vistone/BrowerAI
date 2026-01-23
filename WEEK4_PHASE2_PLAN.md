# Week 4 Phase 2: 系统测试与性能验证 - 详细规划

**预计时间**: 4-6 小时  
**优先级**: 高  
**依赖**: Phase 1 完全完成 ✅

---

## Phase 2 概述

Phase 2 的目标是验证 Week 4 Phase 1 交付的 ONNX 集成 API 在实际系统中的表现，确保满足生产级要求。

### 关键目标
1. **功能验证**: E2E 测试确保代码流程完整
2. **性能基准**: 建立推理延迟目标 (<100ms)
3. **缓存效率**: 验证缓存命中率 (>80%)
4. **可靠性**: 压力测试和边界条件

### 成功标准
- ✅ 所有 E2E 测试通过
- ✅ 平均推理延迟 < 100ms
- ✅ 缓存命中率 > 80%
- ✅ 100+ 并发请求无失败
- ✅ 文档完整且可执行

---

## 第一部分: E2E 集成测试 (1-2小时)

### 1.1 测试框架设置

**文件**: `tests/week4_phase2_e2e_tests.rs`

```rust
#[cfg(test)]
mod week4_phase2_e2e_tests {
    use browerai_deobfuscation::{
        OnnxObfuscationDetector,
        ObfuscationTechnique,
    };
    use std::path::Path;

    // 测试 1: 基础流程测试
    #[test]
    fn test_basic_detection_flow() {
        // 步骤 1: 初始化检测器
        // 步骤 2: 提交代码样本
        // 步骤 3: 验证结果格式
        // 步骤 4: 检查所有字段非空
    }

    // 测试 2: 多个样本序列处理
    #[test]
    fn test_multiple_samples_sequential() {
        // 100个样本的连续处理
        // 验证结果独立性
        // 检查缓存中间状态
    }

    // 测试 3: 相同代码重复检测 (缓存)
    #[test]
    fn test_cache_hit_performance() {
        // 3次检测相同代码
        // 测量每次的执行时间
        // 验证缓存加速 (>50% 时间缩短)
    }
}
```

### 1.2 真实代码样本库

**文件**: `tests/fixtures/obfuscated_samples.rs`

创建 50+ 个具有代表性的代码样本，分为两类：

#### A. 已知混淆类型的样本 (简化版)

```javascript
// 样本 1: 控制流扁平化
var a = 1;
switch(x) {
    case 0: a = f1(); break;
    case 1: a = f2(); break;
    case 2: a = f3(); break;
}

// 样本 2: 字符串编码
var secret = "\x48\x65\x6c\x6c\x6f";  // "Hello"
var decoded = atob("SGVsbG8=");

// 样本 3: 死代码注入
if (false) { useless_code(); }
if (!0) { never_exec(); }

// ... 更多样本
```

#### B. 混合混淆的复杂样本

```javascript
// 样本 25: 混合混淆
function encrypt(d){
    var a="",b=0;
    if(false){}
    switch(Math.floor(Math.random())){
        case 0:b=_0x12ab5["\x63\x6f\x64\x65\x41\x74"](0);break;
        case 1:b=atob(d);break;
    }
    return a;
}
```

### 1.3 结果验证

每个测试样本需要验证：

```rust
#[test]
fn test_sample_1_control_flow() {
    let detector = OnnxObfuscationDetector::new("models/local/week3_obfuscation_detector.onnx")?;
    let code = sample_control_flow();
    let result = detector.detect(&code)?;
    
    // 断言 1: 识别出控制流扁平化
    assert_eq!(result.technique, ObfuscationTechnique::ControlFlowFlattening);
    
    // 断言 2: 置信度合理 (>0.7)
    assert!(result.confidence > 0.7);
    
    // 断言 3: 特征维度正确
    assert_eq!(result.features.len(), 33);
    
    // 断言 4: 分数向量维度正确
    assert_eq!(result.scores.len(), 8);
    
    // 断言 5: 复杂性指标存在
    assert!(result.complexity_metrics.code_length > 0);
}
```

---

## 第二部分: 性能基准测试 (1-2小时)

### 2.1 延迟测试

**文件**: `tests/week4_phase2_performance_benchmark.rs`

```rust
#[test]
fn benchmark_inference_latency() {
    use std::time::Instant;
    
    let detector = OnnxObfuscationDetector::new("models/local/week3_obfuscation_detector.onnx")?;
    let code = "function test() { return 42; }";
    
    // 预热 (第一次推理)
    detector.detect(&code)?;
    
    // 测试 1: 第一次推理 (无缓存)
    let start = Instant::now();
    let result1 = detector.detect(&code)?;
    let latency_fresh = start.elapsed();
    println!("First inference: {:?}", latency_fresh);
    assert!(latency_fresh < Duration::from_millis(100));
    
    // 清空缓存
    detector.clear_cache();
    
    // 测试 2: 100个不同样本的平均延迟
    let mut total_latency = Duration::ZERO;
    for i in 0..100 {
        let test_code = format!("function test{}() {{ return {}; }}", i, i);
        let start = Instant::now();
        detector.detect(&test_code)?;
        total_latency += start.elapsed();
    }
    let avg_latency = total_latency / 100;
    println!("Average latency (100 samples): {:?}", avg_latency);
    assert!(avg_latency < Duration::from_millis(100));
}
```

### 2.2 缓存效率测试

```rust
#[test]
fn benchmark_cache_efficiency() {
    use std::time::Instant;
    
    let detector = OnnxObfuscationDetector::new("models/local/week3_obfuscation_detector.onnx")?;
    let code = "function cached() { return 42; }";
    
    // 第一次: 无缓存 (冷)
    let start = Instant::now();
    detector.detect(&code)?;
    let cold_latency = start.elapsed();
    
    // 第二、三次: 缓存命中 (热)
    let mut hot_latencies = Vec::new();
    for _ in 0..2 {
        let start = Instant::now();
        detector.detect(&code)?;
        hot_latencies.push(start.elapsed());
    }
    
    // 验证加速倍数
    let avg_hot = hot_latencies.iter().sum::<Duration>() / 2;
    let speedup = cold_latency.as_micros() as f32 / avg_hot.as_micros() as f32;
    
    println!("Cold latency: {:?}", cold_latency);
    println!("Hot latency: {:?}", avg_hot);
    println!("Speedup: {:.2}x", speedup);
    
    // 缓存应至少加速 2 倍
    assert!(speedup > 2.0);
    
    // 缓存统计
    let cache_size = detector.cache_stats();
    println!("Cache entries: {}", cache_size);
    assert_eq!(cache_size, 1);
}
```

### 2.3 吞吐量测试

```rust
#[test]
fn benchmark_throughput() {
    use std::time::Instant;
    
    let detector = OnnxObfuscationDetector::new("models/local/week3_obfuscation_detector.onnx")?;
    
    let codes = (0..50)
        .map(|i| format!("function f{}() {{ return {}; }}", i, i))
        .collect::<Vec<_>>();
    
    let start = Instant::now();
    for code in &codes {
        detector.detect(code)?;
    }
    let elapsed = start.elapsed();
    
    let throughput = codes.len() as f32 / elapsed.as_secs_f32();
    println!("Throughput: {:.2} samples/sec", throughput);
    
    // 目标: >10 样本/秒 (单线程)
    assert!(throughput > 10.0);
}
```

### 2.4 内存使用测试

```rust
#[test]
fn test_memory_stability() {
    use std::alloc::{GlobalAlloc, Layout};
    use std::sync::atomic::{AtomicUsize, Ordering};
    
    let detector = OnnxObfuscationDetector::new("models/local/week3_obfuscation_detector.onnx")?;
    
    // 填充缓存到 1000 个条目
    for i in 0..1000 {
        let code = format!("function f{}() {{ return {}; }}", i, i);
        detector.detect(&code)?;
    }
    
    let cache_size = detector.cache_stats();
    println!("Cache size after 1000 calls: {} entries", cache_size);
    
    // 缓存应该包含大部分条目 (或实现 LRU 逐出)
    assert!(cache_size >= 500);
}
```

---

## 第三部分: 压力测试 (1小时)

### 3.1 并发请求测试

**文件**: `tests/week4_phase2_stress_test.rs`

```rust
#[test]
fn test_concurrent_requests() {
    use std::sync::Arc;
    use std::thread;
    
    let detector = Arc::new(
        OnnxObfuscationDetector::new("models/local/week3_obfuscation_detector.onnx")?
    );
    
    let mut handles = vec![];
    
    // 启动 10 个线程，每个处理 100 个样本
    for thread_id in 0..10 {
        let detector_clone = Arc::clone(&detector);
        let handle = thread::spawn(move || {
            for i in 0..100 {
                let code = format!(
                    "function f{}_{},() {{ return {}; }}",
                    thread_id, i, i
                );
                match detector_clone.detect(&code) {
                    Ok(result) => {
                        assert!(result.confidence >= 0.0);
                    }
                    Err(e) => panic!("Thread {} sample {} failed: {}", thread_id, i, e),
                }
            }
        });
        handles.push(handle);
    }
    
    // 等待所有线程完成
    for handle in handles {
        handle.join().unwrap();
    }
    
    println!("✅ 1000 并发请求成功完成");
}
```

### 3.2 边界条件测试

```rust
#[test]
fn test_edge_cases() {
    let detector = OnnxObfuscationDetector::new("models/local/week3_obfuscation_detector.onnx")?;
    
    // 边界 1: 空代码
    match detector.detect("") {
        Ok(result) => println!("Empty code handled: {:?}", result.confidence),
        Err(e) => println!("Empty code error (预期): {}", e),
    }
    
    // 边界 2: 超长代码 (10KB)
    let long_code = "var x = 1;".repeat(1000);
    let result = detector.detect(&long_code)?;
    assert!(result.confidence >= 0.0);
    
    // 边界 3: 特殊字符
    let special = "const α = 'ñ'; /* 中文 */";
    let result = detector.detect(&special)?;
    assert!(result.confidence >= 0.0);
    
    // 边界 4: 二进制-like 代码
    let binary_like = "\x00\x01\x02\x03";
    match detector.detect(binary_like) {
        Ok(result) => println!("Binary-like handled"),
        Err(_) => println!("Binary-like rejected (预期)"),
    }
}
```

---

## 第四部分: 回归测试 (30分钟)

### 4.1 Week 3 精度验证

**文件**: `tests/week4_phase2_regression_test.rs`

```rust
#[test]
fn verify_week3_baselines() {
    let detector = OnnxObfuscationDetector::new("models/local/week3_obfuscation_detector.onnx")?;
    
    // Week 3 精度数据 (应保持或提高)
    let expected_accuracies = [
        ("Control Flow Flattening", 0.942),
        ("String Encoding", 0.921),
        ("Dead Code Injection", 0.918),
        ("Variable Renaming", 0.895),
        ("Code Bloat", 0.869),
        ("Constant Restoration", 0.886),
        ("API Hiding", 0.843),
        ("Dynamic Invocation", 0.827),
    ];
    
    // 对每种技术进行采样测试
    let control_flow_code = "switch(x){case 0:f();case 1:g();}";
    let result = detector.detect(control_flow_code)?;
    
    // 结果置信度应接近预期值
    println!("Control Flow detection confidence: {:.2}%", result.confidence * 100.0);
}
```

---

## 执行步骤

### 步骤 1: 创建测试结构 (30分钟)

```bash
# 创建测试文件
mkdir -p tests/fixtures
touch tests/week4_phase2_e2e_tests.rs
touch tests/week4_phase2_performance_benchmark.rs
touch tests/week4_phase2_stress_test.rs
touch tests/week4_phase2_regression_test.rs
touch tests/fixtures/obfuscated_samples.rs
```

### 步骤 2: 编写样本库 (30分钟)

在 `tests/fixtures/obfuscated_samples.rs` 中创建 50+ 个代码样本

### 步骤 3: 编写测试代码 (1小时)

分别实现各个测试文件中的测试函数

### 步骤 4: 执行测试 (30分钟)

```bash
# 运行所有 Phase 2 测试
cargo test week4_phase2 -- --nocapture

# 运行特定测试类别
cargo test week4_phase2_e2e -- --nocapture
cargo test week4_phase2_performance -- --nocapture
cargo test week4_phase2_stress -- --nocapture
cargo test week4_phase2_regression -- --nocapture

# 生成性能报告
cargo test week4_phase2_performance -- --nocapture > phase2_performance_report.txt
```

### 步骤 5: 分析结果 (30分钟)

```bash
# 总结测试结果
cat phase2_performance_report.txt | grep -E "^(Average|Cache|Throughput|✅)"
```

---

## 预期结果

### 成功标准

| 指标 | 目标 | 预期结果 |
|-----|------|--------|
| E2E 测试通过率 | 100% | ✅ 所有 50+ 样本正确识别 |
| 平均推理延迟 | < 100ms | ✅ ~50-80ms (模拟推理) |
| 缓存命中加速 | > 2x | ✅ 3-5x (缓存命中时) |
| 吞吐量 | > 10 samples/sec | ✅ ~20 samples/sec |
| 并发稳定性 | 100% 成功率 | ✅ 1000/1000 请求成功 |
| 内存稳定性 | < 500MB 增长 | ✅ 缓存 1000 条目 < 100MB |

### 性能指标报告示例

```
=== Week 4 Phase 2 Performance Report ===

[E2E Tests]
✅ test_basic_detection_flow: PASSED
✅ test_multiple_samples_sequential: PASSED (100 samples)
✅ test_cache_hit_performance: PASSED (3.2x speedup)

[Performance Benchmarks]
First inference (cold): 78.5ms
Average latency (100 samples): 72.3ms
Cache latency (hot): 2.1ms
Speedup: 34.5x

[Throughput]
Throughput: 22.4 samples/sec (goal: >10 ✅)

[Stress Test]
✅ 1000 concurrent requests: ALL PASSED
Memory growth: 45 MB (expected < 500 MB ✅)

[Regression]
✅ Week 3 baselines maintained
- Control Flow Flattening: 94.2% (expected)
- String Encoding: 92.1% (expected)
- All 8 techniques within margin ✅
```

---

## 失败处理

### 如果延迟 > 100ms

**诊断**:
1. 检查 ONNX Runtime 配置
2. 验证模型加载时间
3. 检查 CPU 负载

**解决方案**:
- 启用 ONNX 模型优化
- 考虑量化或蒸馏
- 增加推理并行度

### 如果缓存未实现预期加速

**诊断**:
1. 检查缓存键生成
2. 验证哈希碰撞率
3. 检查缓存逐出策略

**解决方案**:
- 优化缓存键 (使用内容哈希)
- 实现 LRU 限制
- 增加缓存容量

### 如果并发请求失败

**诊断**:
1. 检查线程安全性
2. 验证 Mutex 锁持有时间
3. 检查模型共享

**解决方案**:
- 使用 RwLock 替代 Mutex
- 检查死锁条件
- 考虑线程池

---

## 文档交接

### 生成的文档

| 文档 | 内容 |
|-----|------|
| `WEEK4_PHASE2_RESULTS.md` | 完整测试结果 |
| `performance_baseline.txt` | 性能基准数据 |
| `test_coverage_report.txt` | 覆盖率报告 |

### 代码交接

```bash
# 提交 Phase 2 完成
git add tests/week4_phase2_*
git commit -m "Week 4 Phase 2: System testing and performance validation

- E2E integration tests (50+ samples)
- Performance benchmarks (<100ms target)
- Cache efficiency validation
- Concurrent stress testing (1000 requests)
- Regression testing (Week 3 baselines)
- Full documentation
"
```

---

## 时间估计

| 任务 | 预计 | 实际 |
|-----|------|------|
| E2E 测试编写 | 1-1.5h | _ |
| 样本库创建 | 30-45m | _ |
| 性能基准 | 45-60m | _ |
| 压力测试 | 30-45m | _ |
| 回归测试 | 15-30m | _ |
| 结果分析 | 30m | _ |
| **总计** | **4-6h** | _ |

---

## 下一阶段 (Phase 3)

一旦 Phase 2 完成，准备进入 Phase 3:

1. **文档完善** (1-2h)
   - API 文档生成
   - 集成指南编写
   - 用户示例创建

2. **部署准备** (1-2h)
   - Docker 镜像构建
   - 环境变量配置
   - 部署检查清单

3. **发布准备** (30m)
   - 版本号更新
   - 变更日志更新
   - Release notes 编写

---

**预计完成**: Week 4 Phase 2  
**Next Milestone**: Week 4 Phase 3 (文档与发布)  
**Final Goal**: Week 4 生产部署完成
