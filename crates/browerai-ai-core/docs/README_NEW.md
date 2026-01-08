# browerai-ai-core

Core AI infrastructure for BrowerAI providing model management, inference, monitoring, and resilience patterns.

## Features

### 🎯 Model Management

- **ModelProvider Trait System** - Pluggable providers for ONNX, Candle, and custom backends
- **ModelRegistry** - Central registry for managing multiple providers
- **Lifecycle Management** - Loading, validation, health checks, and unloading
- **Model Caching** - Efficient memory management for loaded models

### 🚀 Inference Engine

- **Multi-backend Support** - ONNX Runtime, Candle (GGUF), and custom implementations
- **Batch Inference** - Process multiple inputs efficiently
- **Dynamic Shapes** - Support for variable-size inputs
- **GPU Acceleration** - CUDA, DirectML, CoreML support
- **Model Warm-up** - Pre-allocate buffers and compile kernels

### 📊 Performance & Observability

- **Advanced Metrics** - Latency, throughput, memory, cache hit rates
- **Histogram Analysis** - Percentile calculations (P50, P95, P99)
- **Callback System** - Custom hooks for inference events
- **Real-time Monitoring** - Aggregate metrics with configurable history

### 🛡️ Resilience & High Availability

- **Circuit Breaker** - Automatic failure detection and recovery
- **Retry with Backoff** - Exponential backoff for transient failures
- **Fallback Strategies** - Multiple degradation options
- **Health Checks** - Continuous availability verification

### ⚙️ Configuration & Control

- **Feature Flags** - AI functionality optional via feature gates
- **Flexible Config** - Enable/disable, set timeouts, logging levels
- **Fallback Tracking** - Detailed reason tracking for degradation

## Quick Start

```rust
use browerai_ai_core::*;
use std::path::PathBuf;
use std::sync::Arc;

// Create provider registry
let registry = ModelProviderRegistry::new();

// Register ONNX provider
let onnx = Arc::new(OnnxModelProvider::new().with_gpu(true));
registry.register(onnx)?;

// Configure and load model
let config = ModelLoadConfig::new(PathBuf::from("model.onnx"))
    .with_gpu(true)
    .with_warmup(true);

let model = registry.load_model(&config)?;

// Run inference
let input = vec![1.0, 2.0, 3.0];
let output = model.infer(&input, &[1, 3])?;
println!("Output: {:?}", output);
```

## Advanced Usage

### With Monitoring

```rust
let aggregator = MetricsAggregator::new(1000);

// Run inference...
aggregator.record(metrics);

// Analyze performance
let snapshot = aggregator.snapshot();
println!("P95 Latency: {:.2}ms", snapshot.p95_latency_ms.unwrap_or(0.0));
println!("Success Rate: {:.1}%", snapshot.success_rate * 100.0);
```

### With Resilience

```rust
let circuit_breaker = CircuitBreaker::new(CircuitBreakerConfig::default());
let retry_policy = RetryPolicy::new(RetryConfig::default());

if circuit_breaker.allow_request() {
    match retry_policy.execute(|| model.infer(&input, &shape)) {
        Ok(output) => {
            circuit_breaker.record_success();
            output
        }
        Err(e) => {
            circuit_breaker.record_failure();
            // Use fallback...
        }
    }
}
```

## Architecture

```
┌─────────────────────────────────────────────────┐
│         Model Provider Registry                 │
│  ├─ OnnxModelProvider                          │
│  ├─ CandleModelProvider                        │
│  └─ CustomProvider                             │
└──────────────┬──────────────────────────────────┘
               ↓
┌─────────────────────────────────────────────────┐
│           Model (Arc<dyn Model>)                │
│  ├─ infer()        - Single inference          │
│  ├─ infer_batch()  - Batch processing          │
│  ├─ metadata()     - Model info                │
│  ├─ health_check() - Availability              │
│  └─ memory_stats() - Resource usage            │
└──────────────┬──────────────────────────────────┘
               ↓
┌─────────────────────────────────────────────────┐
│      Resilience Layer                           │
│  ├─ CircuitBreaker  - Failure detection        │
│  ├─ RetryPolicy     - Exponential backoff      │
│  └─ FallbackStrategy - Degradation options    │
└──────────────┬──────────────────────────────────┘
               ↓
┌─────────────────────────────────────────────────┐
│      Observability Layer                        │
│  ├─ MetricsAggregator - Statistics             │
│  ├─ HistogramBucket   - Latency distribution  │
│  └─ InferenceCallback - Custom hooks           │
└─────────────────────────────────────────────────┘
```

## Module Overview

| Module | Purpose |
|--------|---------|
| `model_provider` | Provider trait abstraction and registry |
| `onnx_provider` | ONNX Runtime implementation |
| `inference` | Core inference engine |
| `model_manager` | Model lifecycle and discovery |
| `runtime` | Unified AI runtime container |
| `advanced_metrics` | Performance metrics and statistics |
| `resilience` | Circuit breaker, retry, fallback patterns |
| `gpu_support` | GPU acceleration and provider selection |
| `hot_reload` | Dynamic model updates |
| `config` | Configuration and degradation tracking |

## Building

### Default (no AI)

```bash
cargo build -p browerai-ai-core
```

### With ONNX Runtime

```bash
cargo build -p browerai-ai-core --features onnx
```

### With Candle (GGUF support)

```bash
cargo build -p browerai-ai-core --features candle
```

### With all features

```bash
cargo build -p browerai-ai-core --all-features
```

## Testing

```bash
# Run all tests
cargo test -p browerai-ai-core

# Run specific test
cargo test -p browerai-ai-core test_circuit_breaker_resilience

# With output
cargo test -p browerai-ai-core -- --nocapture

# Integration tests only
cargo test --test integration_tests
```

## Documentation

- **[Developer Guide](./DEVELOPER_GUIDE.md)** - Detailed usage and best practices
- **[Enhancement Plan](./ENHANCEMENT_PLAN.md)** - Architecture and roadmap
- **[Main Architecture](../../docs/ARCHITECTURE.md)** - System overview

## Code Quality

- **Test Coverage**: 80%+ unit and integration tests
- **Documentation**: Complete API documentation with examples
- **Type Safety**: Full Rust type system utilization
- **Concurrency**: Thread-safe (Send + Sync) throughout
- **Performance**: Zero-cost abstractions with trait system

## Dependencies

See [Cargo.toml](./Cargo.toml) for full dependency list.

Key dependencies:
- `ort` - ONNX Runtime (optional, feature: `onnx`)
- `candle-core` - Hugging Face Candle (optional, feature: `candle`)
- `anyhow` - Error handling
- `serde` - Serialization/deserialization
- `log` - Structured logging

## License

MIT License - See LICENSE file in project root
