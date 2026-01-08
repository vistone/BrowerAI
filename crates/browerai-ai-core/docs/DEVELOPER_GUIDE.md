# BrowerAI AI-Core Developer Guide

## Overview

`browerai-ai-core` 是 BrowerAI 的核心 AI 基础设施包，提供以下关键功能：

- **模型管理**: 生命周期管理、版本控制、健康检查
- **推理引擎**: ONNX/Candle 后端支持、批处理、GPU 加速
- **可观测性**: 指标收集、性能监控、分布式追踪
- **高可用性**: 断路器、重试策略、降级处理

## 架构概览

### 模块组织

```
src/
├── model_provider.rs      # 提供者 trait 定义和注册表
├── onnx_provider.rs       # ONNX 运行时实现
├── inference.rs           # 推理引擎核心
├── model_manager.rs       # 模型生命周期管理
├── runtime.rs             # 统一运行时容器
├── advanced_metrics.rs    # 性能指标收集
├── resilience.rs          # 容错和弹性模式
├── gpu_support.rs         # GPU 加速支持
├── hot_reload.rs          # 模型热更新
├── config.rs              # 配置和降级追踪
└── ...其他支持模块
```

### 数据流

```
用户代码
    ↓
ModelProviderRegistry (查找合适的提供者)
    ↓
ModelProvider::load_model (加载模型)
    ↓
Model trait (推理接口)
    ↓
MetricsAggregator (收集指标)
    ↓
CircuitBreaker (故障检测)
```

## 快速开始

### 基本使用

```rust
use browerai_ai_core::*;
use std::path::PathBuf;
use std::sync::Arc;

// 1. 创建提供者注册表
let registry = ModelProviderRegistry::new();

// 2. 注册 ONNX 提供者
let onnx_provider = Arc::new(OnnxModelProvider::new().with_gpu(true));
registry.register(onnx_provider)?;

// 3. 配置模型加载
let config = ModelLoadConfig::new(PathBuf::from("model.onnx"))
    .with_gpu(true)
    .with_warmup(true)
    .with_validation(true);

// 4. 加载模型
let model = registry.load_model(&config)?;

// 5. 运行推理
let input = vec![1.0, 2.0, 3.0];
let output = model.infer(&input, &[1, 3])?;

println!("Output: {:?}", output);
```

### 带监控的推理

```rust
use browerai_ai_core::*;
use std::sync::Arc;

// 创建指标聚合器
let aggregator = MetricsAggregator::new(1000);

// 定义回调
struct LoggingCallback;

impl InferenceCallback for LoggingCallback {
    fn on_post_inference(&self, metrics: &InferenceMetrics) {
        println!(
            "Model: {}, Latency: {:.2}ms, Success: {}",
            metrics.model_name,
            metrics.latency_ms(),
            metrics.success
        );
    }

    fn on_inference_failed(&self, model: &str, error: &str) {
        eprintln!("Inference failed for {}: {}", model, error);
    }
}

// 使用回调
let callback = Arc::new(LoggingCallback);
// ... 传递给推理处理代码
```

### 高可用推理

```rust
use browerai_ai_core::*;

// 配置断路器
let cb_config = CircuitBreakerConfig {
    failure_threshold: 0.5,       // 50% 失败率触发
    request_window: 10,            // 跟踪最后 10 个请求
    timeout_duration: Duration::from_secs(30),
    enable_recovery: true,
};

// 配置重试
let retry_config = RetryConfig {
    max_attempts: 3,
    initial_backoff: Duration::from_millis(100),
    max_backoff: Duration::from_secs(10),
    backoff_multiplier: 2.0,
};

// 创建弹性推理包装
let resilient = ResilientInference::new(
    cb_config,
    retry_config,
    FallbackStrategy::AlternativeModel("fallback_model.onnx".to_string()),
);

// 使用断路器
if resilient.circuit_breaker().allow_request() {
    // 执行推理
    resilient.circuit_breaker().record_success();
} else {
    // 触发降级
    println!("Circuit breaker open, using fallback");
}
```

## 核心概念

### ModelProvider Trait

`ModelProvider` 是一个可插拔的工厂 trait，负责加载特定格式的模型：

```rust
pub trait ModelProvider: Send + Sync {
    fn load_model(&self, config: &ModelLoadConfig) -> ModelResult<Arc<dyn Model>>;
    fn validate_model(&self, path: &Path) -> ModelResult<ModelMetadata>;
    fn info(&self) -> ProviderInfo;
    fn can_load(&self, path: &Path) -> bool;
}
```

**关键特性：**
- 线程安全 (Send + Sync)
- 模式匹配 (can_load)
- 验证支持 (validate_model)
- 自描述 (info)

### Model Trait

`Model` trait 定义推理接口，由提供者创建：

```rust
pub trait Model: Send + Sync {
    fn infer(&self, input: &[f32], shape: &[i64]) -> ModelResult<Vec<f32>>;
    fn infer_batch(&self, inputs: &[(Vec<f32>, Vec<i64>)]) -> ModelResult<Vec<Vec<f32>>>;
    fn metadata(&self) -> &ModelMetadata;
    fn warmup(&self) -> ModelResult<()>;
    fn health_check(&self) -> ModelResult<()>;
    fn memory_stats(&self) -> Option<MemoryStats>;
}
```

### CircuitBreaker

断路器使用有限状态机实现故障隔离：

```
[Closed] --失败率高--> [Open] --超时--> [HalfOpen]
  ↑                                        |
  +----------成功率高----------------------+
```

## 最佳实践

### 1. 模型验证

始终验证模型，特别是从不信任来源加载时：

```rust
// ✅ 好的做法
let config = ModelLoadConfig::new(path)
    .with_validation(true);  // 默认启用

// ❌ 避免
let config = ModelLoadConfig::new(path);
config.validate = false;  // 直接禁用验证
```

### 2. 模型预热

在生产环境中，预热模型以获得稳定的性能：

```rust
// ✅ 推荐
let config = ModelLoadConfig::new(path)
    .with_warmup(true);

// 预热会：
// - 分配必要的内存缓冲区
// - 编译GPU内核
// - 初始化运行时
```

### 3. 监控和指标

使用 MetricsAggregator 跟踪性能：

```rust
let aggregator = MetricsAggregator::new(1000);

for _ in 0..100 {
    // 运行推理...
    aggregator.record(metrics);
}

let snapshot = aggregator.snapshot();
println!("P95 Latency: {:.2}ms", snapshot.p95_latency_ms.unwrap_or(0.0));
println!("Success Rate: {:.2}%", snapshot.success_rate * 100.0);
```

### 4. 优雅降级

实现降级策略应对模型故障：

```rust
match resilient.circuit_breaker().allow_request() {
    true => {
        // 尝试主模型
        match model.infer(&input, &shape) {
            Ok(output) => {
                resilient.circuit_breaker().record_success();
                output
            }
            Err(_) => {
                resilient.circuit_breaker().record_failure();
                // 使用备用策略
                fallback_model.infer(&input, &shape)?
            }
        }
    }
    false => {
        // 直接使用备用
        fallback_model.infer(&input, &shape)?
    }
}
```

## 扩展 AI-Core

### 添加新的提供者

1. **实现 ModelProvider trait：**

```rust
pub struct CustomModelProvider;

impl ModelProvider for CustomModelProvider {
    fn load_model(&self, config: &ModelLoadConfig) -> ModelResult<Arc<dyn Model>> {
        // 实现加载逻辑
    }

    fn validate_model(&self, path: &Path) -> ModelResult<ModelMetadata> {
        // 实现验证逻辑
    }

    fn info(&self) -> ProviderInfo {
        ProviderInfo {
            name: "Custom Provider".to_string(),
            version: "1.0.0".to_string(),
            supported_formats: vec!["custom".to_string()],
            capabilities: ProviderCapabilities {
                gpu: false,
                quantization: false,
                dynamic_shapes: true,
                batch_inference: true,
                max_model_size_mb: Some(2048),
            },
        }
    }

    fn can_load(&self, path: &Path) -> bool {
        path.extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext == "custom")
            .unwrap_or(false)
    }
}
```

2. **实现 Model trait：**

```rust
pub struct CustomModel {
    metadata: ModelMetadata,
}

impl Model for CustomModel {
    fn infer(&self, input: &[f32], _shape: &[i64]) -> ModelResult<Vec<f32>> {
        // 实现推理
        Ok(input.to_vec())  // 示例：回显输入
    }

    fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    fn health_check(&self) -> ModelResult<()> {
        Ok(())
    }
}
```

3. **注册提供者：**

```rust
let registry = ModelProviderRegistry::new();
registry.register(Arc::new(CustomModelProvider))?;
```

### 添加自定义指标收集

```rust
pub struct CustomMetricsCollector {
    // 实现细节
}

impl InferenceCallback for CustomMetricsCollector {
    fn on_post_inference(&self, metrics: &InferenceMetrics) {
        // 自定义处理逻辑
        // 例如：发送到远程监控系统
    }
}
```

## 测试

### 单元测试

```bash
# 运行所有测试
cargo test -p browerai-ai-core

# 运行特定模块的测试
cargo test -p browerai-ai-core model_provider::

# 显示测试输出
cargo test -p browerai-ai-core -- --nocapture
```

### 集成测试

```bash
# 运行集成测试
cargo test --test integration_tests

# 特定测试
cargo test --test integration_tests test_circuit_breaker_resilience
```

### 性能基准

```bash
cargo bench -p browerai-ai-core
```

## 常见问题

### Q: ONNX 特性何时必需？
**A:** 当使用 OnnxModelProvider 或想要 ONNX Runtime 支持时。编译时指定 `--features ai` 或在 Cargo.toml 中添加 `onnx` 特性。

### Q: 如何处理大型模型？
**A:** 
1. 使用 ModelLoadConfig 的 `with_option` 方法传递提供者特定选项
2. 启用量化以减小模型大小
3. 考虑模型分割或异构推理

### Q: 推理性能低下怎么办？
**A:**
1. 启用模型预热 (`.with_warmup(true)`)
2. 使用 GPU 加速 (`.with_gpu(true)`)
3. 检查 MetricsSnapshot 中的 P95 延迟
4. 启用批推理以提高吞吐量

### Q: 可以动态切换模型吗？
**A:** 可以，使用 HotReloadManager 进行模型热更新：

```rust
let hot_reload = HotReloadManager::new(model_dir);
hot_reload.watch_for_updates()?;
```

## 性能优化指南

### 内存优化
- 使用 ModelCache 缓存加载的模型
- 启用模型量化
- 监控 MemoryStats

### 延迟优化
- 预热模型
- 使用批推理
- 启用 GPU 加速
- 调整 CircuitBreaker 窗口大小

### 吞吐量优化
- 启用批处理
- 使用 infer_batch API
- 根据硬件调整批大小
- 使用 HistogramBucket 分析分布

## 故障排除

### 模型加载失败
1. 验证文件存在: `std::path::Path::exists()`
2. 检查格式支持: `provider.can_load(path)`
3. 查看详细错误信息: `validate_model(path)`

### 推理失败
1. 检查输入形状是否匹配 metadata
2. 验证数据类型 (浮点vs整数)
3. 检查模型健康状态: `model.health_check()`

### 性能问题
1. 检查指标: `aggregator.snapshot()`
2. 验证 GPU 使用: `gpu_support` 模块
3. 分析断路器状态: `circuit_breaker.current_state()`

## 相关资源

- [架构文档](../../docs/ARCHITECTURE.md)
- [性能监控指南](../../docs/TESTING_STRATEGY.md)
- [最佳实践](./ENHANCEMENT_PLAN.md)

## 版本历史

- **v0.2.0** (当前): 新增 ModelProvider trait、advanced_metrics、resilience
- **v0.1.0**: 初始版本，基础 ONNX 和模型管理

## 许可证

MIT License - 详见项目根目录的 LICENSE 文件
