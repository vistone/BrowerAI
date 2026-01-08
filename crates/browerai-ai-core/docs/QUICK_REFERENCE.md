# AI-Core 快速参考卡

## 模块一览

| 模块 | 用途 | 关键类型 |
|------|------|---------|
| `model_provider` | Trait抽象和注册表 | `ModelProvider`, `ModelProviderRegistry` |
| `onnx_provider` | ONNX Runtime实现 | `OnnxModelProvider`, `OnnxModel` |
| `advanced_metrics` | 性能指标收集 | `MetricsAggregator`, `InferenceMetrics` |
| `resilience` | 故障隔离和恢复 | `CircuitBreaker`, `RetryPolicy` |
| `inference` | 推理引擎核心 | `InferenceEngine` |
| `runtime` | 统一运行时 | `AiRuntime` |
| `config` | 配置管理 | `AiConfig`, `FallbackTracker` |
| `gpu_support` | GPU加速 | `GpuConfig`, `GpuProvider` |

---

## 常见用法模式

### 1. 基础模型加载

```rust
use browerai_ai_core::*;
use std::sync::Arc;

// 创建注册表
let registry = ModelProviderRegistry::new();

// 注册提供者
registry.register(Arc::new(OnnxModelProvider::new()))?;

// 加载模型
let config = ModelLoadConfig::new("model.onnx".into())
    .with_gpu(true);
let model = registry.load_model(&config)?;
```

### 2. 运行推理

```rust
// 单个推理
let output = model.infer(&input_vec, &[1, 3])?;

// 批推理
let inputs = vec![
    (input1, vec![1, 3]),
    (input2, vec![1, 3]),
];
let outputs = model.infer_batch(&inputs)?;
```

### 3. 性能监控

```rust
let aggregator = MetricsAggregator::new(1000);

// 记录指标
for _ in 0..100 {
    aggregator.record(InferenceMetrics {
        model_name: "test".into(),
        inference_time: Duration::from_millis(50),
        input_size: 1024,
        output_size: 1024,
        memory_peak_mb: 256,
        cache_hit: false,
        success: true,
        timestamp: Instant::now(),
    });
}

// 获取统计
let snapshot = aggregator.snapshot();
println!("成功率: {:.1}%", snapshot.success_rate * 100.0);
println!("P95延迟: {:.2}ms", snapshot.p95_latency_ms.unwrap_or(0.0));
```

### 4. 高可用推理

```rust
let cb = CircuitBreaker::new(CircuitBreakerConfig::default());
let retry = RetryPolicy::new(RetryConfig::default());

if cb.allow_request() {
    match retry.execute(|| model.infer(&input, &shape)) {
        Ok(output) => {
            cb.record_success();
            output
        }
        Err(e) => {
            cb.record_failure();
            // 使用降级
            fallback_model.infer(&input, &shape)?
        }
    }
}
```

---

## 关键API

### ModelProvider

```rust
pub trait ModelProvider: Send + Sync {
    fn load_model(&self, config: &ModelLoadConfig) -> ModelResult<Arc<dyn Model>>;
    fn validate_model(&self, path: &Path) -> ModelResult<ModelMetadata>;
    fn info(&self) -> ProviderInfo;
    fn can_load(&self, path: &Path) -> bool;
}
```

### Model

```rust
pub trait Model: Send + Sync {
    fn infer(&self, input: &[f32], shape: &[i64]) -> ModelResult<Vec<f32>>;
    fn infer_batch(&self, inputs: &[(Vec<f32>, Vec<i64>)]) -> ModelResult<Vec<Vec<f32>>>;
    fn metadata(&self) -> &ModelMetadata;
    fn warmup(&self) -> ModelResult<()>;
    fn health_check(&self) -> ModelResult<()>;
}
```

### MetricsAggregator

```rust
pub struct MetricsAggregator {
    pub fn new(max_history: usize) -> Self;
    pub fn record(&self, metric: InferenceMetrics);
    pub fn snapshot(&self) -> MetricsSnapshot;
    pub fn clear(&self);
}
```

### CircuitBreaker

```rust
pub struct CircuitBreaker {
    pub fn new(config: CircuitBreakerConfig) -> Self;
    pub fn allow_request(&self) -> bool;
    pub fn record_success(&self);
    pub fn record_failure(&self);
    pub fn current_state(&self) -> CircuitState;
    pub fn reset(&self);
}
```

---

## 配置示例

### 模型加载

```rust
let config = ModelLoadConfig::new("model.onnx".into())
    .with_gpu(true)
    .with_warmup(true)
    .with_validation(true)
    .with_option("precision", "fp16");
```

### 断路器

```rust
let cb_config = CircuitBreakerConfig {
    failure_threshold: 0.5,        // 50% 触发
    request_window: 10,             // 最后10个请求
    timeout_duration: Duration::from_secs(30),
    enable_recovery: true,
};
```

### 重试策略

```rust
let retry_config = RetryConfig {
    max_attempts: 3,
    initial_backoff: Duration::from_millis(100),
    max_backoff: Duration::from_secs(10),
    backoff_multiplier: 2.0,
};
```

---

## 特征启用

```toml
# Cargo.toml

[dependencies]
browerai-ai-core = { version = "0.2", features = ["onnx"] }

# 或

[dependencies.browerai-ai-core]
version = "0.2"
features = ["onnx", "candle"]
```

## 编译命令

```bash
# 无AI功能
cargo build -p browerai-ai-core

# 启用ONNX
cargo build -p browerai-ai-core --features onnx

# 启用Candle (GGUF)
cargo build -p browerai-ai-core --features candle

# 所有特性
cargo build -p browerai-ai-core --all-features
```

---

## 测试命令

```bash
# 单元测试
cargo test -p browerai-ai-core --lib

# 集成测试
cargo test --test integration_tests

# 特定测试
cargo test test_circuit_breaker_resilience

# 显示输出
cargo test -- --nocapture

# 性能基准
cargo bench -p browerai-ai-core
```

---

## 错误处理

```rust
use browerai_ai_core::*;

match model.infer(&input, &shape) {
    Ok(output) => println!("Success: {:?}", output),
    Err(e) => {
        eprintln!("Inference failed: {}", e);
        // 使用降级或重试
    }
}
```

## 日志调试

```bash
# 启用日志
RUST_LOG=debug cargo run

# 仅ai-core日志
RUST_LOG=browerai_ai_core=debug cargo run

# 包含backtrace
RUST_BACKTRACE=1 cargo run
```

---

## 扩展点

### 添加新提供者

1. 实现 `ModelProvider` trait
2. 实现 `Model` trait  
3. 调用 `registry.register(provider)`

### 添加监控回调

```rust
pub struct MyCallback;

impl InferenceCallback for MyCallback {
    fn on_post_inference(&self, metrics: &InferenceMetrics) {
        // 自定义处理
    }
    
    fn on_inference_failed(&self, model: &str, error: &str) {
        // 错误处理
    }
}
```

---

## 性能建议

| 场景 | 建议 |
|------|------|
| 冷启动 | 启用 `with_warmup(true)` |
| 高吞吐量 | 使用批推理 `infer_batch()` |
| GPU环境 | 启用 `with_gpu(true)` |
| 实时应用 | 配置 CircuitBreaker |
| 长时间运行 | 启用监控和指标收集 |

---

## 故障排除

| 问题 | 解决方案 |
|------|---------|
| 模型加载失败 | 检查路径、格式、ONNX特性 |
| 推理超时 | 增加 CircuitBreaker 窗口或使用重试 |
| 内存溢出 | 减小批大小或启用量化 |
| 性能下降 | 检查 MetricsSnapshot 中的 P95延迟 |
| GPU未使用 | 确认 `with_gpu(true)` 且驱动正常 |

---

## 版本兼容性

- **Rust**: 1.70+
- **ONNX Runtime**: 1.16+
- **Candle**: 0.3+

---

## 相关资源

- 📖 [Developer Guide](./DEVELOPER_GUIDE.md)
- 📋 [Enhancement Plan](./ENHANCEMENT_PLAN.md)
- 🏗️ [Implementation Summary](./IMPLEMENTATION_SUMMARY.md)
- 🔗 [Main Architecture](../../docs/ARCHITECTURE.md)

---

**最后更新**: 2026-01-07  
**版本**: 0.2.0
