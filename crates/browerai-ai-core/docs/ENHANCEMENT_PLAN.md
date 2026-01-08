# BrowerAI AI-Core 增强计划

## 一、现状分析

### 1.1 核心模块评估

#### ✅ 已有的良好设计
- **模块划分清晰**: 
  - `model_manager.rs` - 模型生命周期管理
  - `inference.rs` - ONNX推理引擎
  - `runtime.rs` - 统一运行时容器
  - `config.rs` - 配置和降级跟踪
  - `gpu_support.rs` - GPU加速支持
  - `performance_monitor.rs` - 性能监控

- **特性门控设计**: AI功能通过feature flags可选
- **错误处理**: 使用anyhow的Result模式
- **降级策略**: FallbackTracker记录失败原因

#### ⚠️ 需要改进的地方

1. **缺少trait抽象和扩展性**
   - 模型加载器硬编码ONNX
   - 没有通用的ModelProvider trait
   - 推理引擎没有可插拔设计

2. **性能监控不完整**
   - PerformanceMonitor是stub实现
   - 没有实际的指标收集
   - 缺少内存、GPU、吞吐量监控

3. **并发和线程安全不充分**
   - ModelManager没有实现Send+Sync
   - 没有并发推理支持
   - 缺少model预热和缓存

4. **测试覆盖不足**
   - 大多数函数没有单元测试
   - 没有集成测试
   - 缺少性能基准测试

5. **文档和示例缺乏**
   - 模块级别文档不完整
   - 没有使用示例
   - 缺少最佳实践指南

6. **类型安全和编码规范**
   - 某些部分缺少#[derive]属性
   - 没有consistent的error types
   - 缺少validation logic

### 1.2 代码质量指标

| 指标 | 现状 | 目标 |
|------|------|------|
| 测试覆盖率 | ~30% | 80%+ |
| 文档完整度 | 50% | 100% |
| 并发支持 | 基础 | 完整 |
| 性能监控 | 占位符 | 完全实现 |

---

## 二、学习优秀开源项目

### 2.1 参考项目模式

#### A. Hugging Face Transformers
**学习内容:**
- 模型注册表Registry模式
- Hub集成设计
- 加载器工厂模式（Factory）

**应用:**
```rust
pub struct ModelRegistry {
    models: HashMap<String, Box<dyn ModelLoader>>,
}

pub trait ModelProvider: Send + Sync {
    fn load(&self, path: &Path) -> Result<DynModel>;
    fn info(&self) -> ModelInfo;
}
```

#### B. ONNX Runtime
**学习内容:**
- Session管理生命周期
- ExecutionProvider抽象
- 输入输出序列化

**应用:**
- 完善InferenceEngine的序列化
- 添加batch inference支持
- 更好的provider选择逻辑

#### C. Ray/Anyscale
**学习内容:**
- 分布式推理架构
- 模型服务化
- 动态扩缩容

**应用:**
- 可选的分布式推理支持
- 模型服务接口（HTTP/RPC）
- 负载均衡

#### D. MLflow
**学习内容:**
- 实验跟踪
- 模型注册和版本管理
- 性能指标记录

**应用:**
```rust
pub struct ModelMetrics {
    accuracy: f64,
    latency_ms: u64,
    throughput: u32,
    memory_peak_mb: u64,
}

pub trait MetricsBackend: Send + Sync {
    fn record(&self, metrics: ModelMetrics);
}
```

#### E. PyTorch Lightning
**学习内容:**
- 训练流程的标准化
- Callback系统
- Logger抽象

**应用:**
```rust
pub trait InferenceCallback: Send + Sync {
    fn on_pre_inference(&self, input: &InputData);
    fn on_post_inference(&self, output: &OutputData, latency: Duration);
}
```

---

## 三、全局增强规划

### 3.1 架构改进

```
┌─────────────────────────────────────────────────────────┐
│              Enhanced AI-Core Architecture              │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────────────────────────────────────────────┐  │
│  │  Model Management Layer                          │  │
│  │  ├─ ModelRegistry (type-safe registration)      │  │
│  │  ├─ ModelProvider (trait-based abstraction)     │  │
│  │  ├─ VersionedModel (semantic versioning)       │  │
│  │  └─ ModelCache (memory-aware caching)          │  │
│  └──────────────────────────────────────────────────┘  │
│                    ↓                                    │
│  ┌──────────────────────────────────────────────────┐  │
│  │  Inference Engine Layer                          │  │
│  │  ├─ Pluggable backends (ONNX, Candle, etc)     │  │
│  │  ├─ Batch inference (async/concurrent)         │  │
│  │  ├─ Dynamic shape support                      │  │
│  │  └─ Warm-up and pre-allocation                │  │
│  └──────────────────────────────────────────────────┘  │
│                    ↓                                    │
│  ┌──────────────────────────────────────────────────┐  │
│  │  Performance & Observability Layer               │  │
│  │  ├─ Real metrics collection                     │  │
│  │  ├─ Distributed tracing                        │  │
│  │  ├─ Custom callbacks/hooks                     │  │
│  │  └─ Export to external systems                │  │
│  └──────────────────────────────────────────────────┘  │
│                    ↓                                    │
│  ┌──────────────────────────────────────────────────┐  │
│  │  Resilience & Control Layer                      │  │
│  │  ├─ Advanced fallback strategies                │  │
│  │  ├─ Circuit breaker pattern                    │  │
│  │  ├─ Rate limiting and quotas                   │  │
│  │  └─ Health checks and recovery                 │  │
│  └──────────────────────────────────────────────────┘  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 3.2 关键改进项

#### Phase 1: 基础结构完善
- [ ] 定义ModelProvider trait体系
- [ ] 实现ModelRegistry注册表
- [ ] 重构ModelManager利用trait
- [ ] 添加ModelCache组件
- [ ] 完善类型安全的错误类型

#### Phase 2: 推理引擎增强
- [ ] 支持多backend（ONNX、Candle、Custom）
- [ ] 实现batch推理API
- [ ] 添加动态shape支持
- [ ] 实现模型预热机制
- [ ] GPU内存管理

#### Phase 3: 性能可观测性
- [ ] 真实的指标收集（不是stub）
- [ ] 分布式追踪支持（OpenTelemetry）
- [ ] Callback/Hook系统
- [ ] 性能报告生成
- [ ] 实时监控dashboard支持

#### Phase 4: 高可用和容错
- [ ] Circuit breaker实现
- [ ] 高级降级策略（阈值、黑名单）
- [ ] 模型预热和自检
- [ ] 分布式推理支持
- [ ] 优雅降级

#### Phase 5: 开发者体验
- [ ] 完整的单元测试（80%+覆盖）
- [ ] 集成和E2E测试
- [ ] 详细的文档和示例
- [ ] 性能基准测试套件
- [ ] Debug工具和诊断命令

---

## 四、具体技术方案

### 4.1 Model Provider Trait体系

```rust
/// Base trait for all model providers
pub trait ModelProvider: Send + Sync {
    fn load(&self, config: &ModelConfig) -> Result<Arc<dyn Model>>;
    fn validate(&self, model_path: &Path) -> Result<()>;
    fn info(&self) -> ProviderInfo;
}

/// Dynamic trait for inference
pub trait Model: Send + Sync {
    fn infer(&self, input: &Tensor) -> Result<Tensor>;
    fn infer_batch(&self, inputs: &[Tensor]) -> Result<Vec<Tensor>>;
    fn get_metadata(&self) -> ModelMetadata;
    fn warmup(&self) -> Result<()>;
}

pub enum ProviderInfo {
    Onnx { versions: Vec<String> },
    Candle { backends: Vec<String> },
    Custom { name: String, version: String },
}
```

### 4.2 性能监控实现

```rust
pub struct InferenceMetrics {
    pub model_name: String,
    pub inference_time: Duration,
    pub input_size: usize,
    pub output_size: usize,
    pub memory_peak: u64,
    pub cache_hit: bool,
    pub success: bool,
    pub timestamp: Instant,
}

pub trait MetricsCollector: Send + Sync {
    fn collect(&self, metrics: InferenceMetrics);
    fn export(&self) -> MetricsSnapshot;
    fn flush(&self) -> Result<()>;
}

pub struct AdvancedPerformanceMonitor {
    collectors: Vec<Arc<dyn MetricsCollector>>,
    histogram: HistogramBucket,
    aggregator: Arc<MetricsAggregator>,
}
```

### 4.3 高可用模式

```rust
pub struct ResilientInferenceEngine {
    engine: InferenceEngine,
    circuit_breaker: CircuitBreaker,
    fallback_strategy: FallbackStrategy,
    health_checker: HealthChecker,
}

pub enum FallbackStrategy {
    /// Immediate fallback on any error
    Immediate,
    /// Wait for timeout then fallback
    Timeout(Duration),
    /// Retry with exponential backoff
    Retry { max_attempts: u32, backoff: Duration },
    /// Multi-model ensemble
    Ensemble(Vec<ModelConfig>),
}

pub struct CircuitBreaker {
    failure_threshold: f64,
    recovery_timeout: Duration,
    state: Arc<RwLock<CircuitState>>,
}
```

---

## 五、实现路线图

### Week 1: 架构重构
- [ ] 实现ModelProvider trait和实现
- [ ] 重构ModelManager
- [ ] 添加ModelRegistry和ModelCache
- [ ] Unit tests基础设施

### Week 2: 推理增强
- [ ] 支持批推理
- [ ] 动态shape支持
- [ ] Candle后端集成
- [ ] 模型预热机制

### Week 3: 可观测性
- [ ] 真实指标收集
- [ ] OpenTelemetry集成
- [ ] Callback系统
- [ ] 性能报告

### Week 4: 高可用
- [ ] Circuit breaker实现
- [ ] 高级降级策略
- [ ] 健康检查
- [ ] 集成测试

### Week 5: 文档和完善
- [ ] 完整文档编写
- [ ] 示例代码
- [ ] 基准测试
- [ ] 开发者指南

---

## 六、代码质量标准

### 6.1 核心原则

1. **类型安全**: 充分利用Rust类型系统
2. **并发友好**: 所有共享数据必须Send+Sync
3. **零成本抽象**: Trait使用不产生运行时开销
4. **文档驱动**: 所有public API必须有docs注释
5. **测试优先**: 新feature必须有相应测试

### 6.2 编码规范

```rust
// Good: 完整的公共API文档
/// Loads a model from the specified path with automatic validation.
///
/// # Arguments
/// * `path` - Path to the model file
/// * `config` - Provider configuration
///
/// # Returns
/// Returns a loaded model ready for inference, or an error if loading fails.
///
/// # Examples
/// ```ignore
/// let model = provider.load(&path, &config)?;
/// let output = model.infer(&input)?;
/// ```
pub fn load(&self, path: &Path, config: &Config) -> Result<Arc<dyn Model>> {
    // Implementation
}

// Good: 通用error类型
#[derive(thiserror::Error, Debug)]
pub enum ModelError {
    #[error("Failed to load model from {path}: {reason}")]
    LoadFailed { path: PathBuf, reason: String },
    
    #[error("Model validation failed: {0}")]
    ValidationFailed(String),
}

// Good: 单元测试
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_model_load_success() {
        // Test implementation
    }
}
```

---

## 七、Success Metrics

| 指标 | 当前 | 6周后 | 方式 |
|------|------|-------|------|
| 单元测试覆盖 | 30% | 85% | cargo tarpaulin |
| 文档完整度 | 50% | 100% | doc注释 + guides |
| 并发支持 | 基础 | 完全 | Sync + async/await |
| 性能监控 | Stub | 完整 | 真实metrics收集 |
| API稳定性 | 不稳定 | 稳定 | 语义版本控制 |
| 编译警告 | 20+ | 0 | clippy clean |

---

## 八、后续步骤

1. ✅ 完成架构设计文档 (本文)
2. → 实现ModelProvider trait体系
3. → 重构现有模块
4. → 添加测试和文档
5. → 性能基准测试
6. → 发布v1.0版本
