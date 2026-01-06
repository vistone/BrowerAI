# Phase 3 Week 3 - API 参考指南

## 快速导航 (Quick Navigation)

- [EnhancedCallGraphAnalyzer](#enhancedcallgraphanalyzer)
- [LoopAnalyzer](#loopanalyzer)
- [OptimizedAnalyzer](#optimizedanalyzer)
- [AnalysisPipeline](#analysispipeline)

---

## EnhancedCallGraphAnalyzer

### 概述

高级调用图分析器，支持上下文敏感的函数关系分析、递归检测、深度计算和热路径识别。

### 导入

```rust
use browerai::parser::js_analyzer::EnhancedCallGraphAnalyzer;
```

### 主要类型

#### CallContext

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CallContext {
    Global,      // 全局作用域调用
    Local,       // 本地作用域调用
    Method,      // 方法调用 (obj.method())
}
```

#### CallNode

```rust
pub struct CallNode {
    pub name: Arc<str>,              // 函数名称
    pub context: CallContext,        // 调用上下文
    pub call_count: usize,           // 被调用次数
    pub complexity: usize,           // 复杂度评分
}
```

#### CallEdge

```rust
pub struct CallEdge {
    pub from: Arc<str>,              // 源函数
    pub to: Arc<str>,                // 目标函数
    pub context_type: CallContext,   // 上下文类型
    pub frequency: usize,            // 调用频率
}
```

#### EnhancedCallGraph

```rust
pub struct EnhancedCallGraph {
    pub nodes: HashMap<Arc<str>, CallNode>,
    pub edges: Vec<CallEdge>,
    pub recursive_chains: Vec<Vec<Arc<str>>>,
    pub depths: HashMap<Arc<str>, usize>,
}

impl EnhancedCallGraph {
    pub fn edge_count(&self) -> usize
    pub fn node_count(&self) -> usize
}
```

### API 方法

#### new()

创建新的分析器实例。

```rust
let mut analyzer = EnhancedCallGraphAnalyzer::new();
```

#### analyze()

执行完整的调用图分析。

**签名**:
```rust
pub fn analyze(
    &mut self,
    ast: &ExtractedAst,
    scope_tree: &ScopeTree,
    data_flow: &DataFlowGraph,
    cfg: &ControlFlowInfo,
) -> Result<EnhancedCallGraph>
```

**参数**:
- `ast`: 提取的 AST
- `scope_tree`: 作用域树
- `data_flow`: 数据流图
- `cfg`: 控制流信息

**返回**: 
- `Ok(EnhancedCallGraph)`: 包含所有分析结果
- `Err(anyhow::Error)`: 分析失败

**示例**:
```rust
let graph = analyzer.analyze(&ast, &scope, &df, &cfg)?;
println!("节点数: {}", graph.node_count());
println!("边数: {}", graph.edge_count());
```

#### detect_recursive_chains()

检测所有递归链。

**签名**:
```rust
pub fn detect_recursive_chains(
    &self,
    graph: &EnhancedCallGraph,
) -> Vec<Vec<Arc<str>>>
```

**返回**: 递归链列表，每个链是函数名的向量

**时间复杂度**: O(V + E)

**示例**:
```rust
let chains = analyzer.detect_recursive_chains(&graph);
for chain in chains {
    println!("递归链: {}", chain.join(" -> "));
}
```

#### calculate_depths()

计算从起点函数到所有其他函数的深度。

**签名**:
```rust
pub fn calculate_depths(
    &self,
    graph: &EnhancedCallGraph,
    start_func: &str,
) -> Result<HashMap<Arc<str>, usize>>
```

**参数**:
- `graph`: 调用图
- `start_func`: 起点函数名

**返回**: 函数名到深度的映射

**时间复杂度**: O(V + E)

**示例**:
```rust
let depths = analyzer.calculate_depths(&graph, "main")?;
for (func, depth) in &depths {
    println!("{}: 深度 {}", func, depth);
}
```

#### identify_hot_paths()

识别高频率的调用路径。

**签名**:
```rust
pub fn identify_hot_paths(
    &self,
    graph: &EnhancedCallGraph,
) -> Vec<String>
```

**返回**: 排序的热路径字符串列表

**说明**: 路径按频率降序排列

**示例**:
```rust
let hot_paths = analyzer.identify_hot_paths(&graph);
for (idx, path) in hot_paths.iter().take(5).enumerate() {
    println!("热路径 #{}: {}", idx + 1, path);
}
```

---

## LoopAnalyzer

### 概述

高级循环分析器，支持循环类型识别、归纳变量检测、迭代估计和复杂度评分。

### 导入

```rust
use browerai::parser::js_analyzer::LoopAnalyzer;
```

### 主要类型

#### LoopType

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoopType {
    For,              // for (let i = 0; i < 10; i++)
    While,            // while (condition)
    DoWhile,          // do { ... } while (condition)
    IteratorLoop,     // for (let x of array) 或 for (let x in obj)
}
```

#### UpdatePattern

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpdatePattern {
    Increment(i32),    // i++, i+=k
    Decrement(i32),    // i--, i-=k
    Multiply(i32),     // i*=k
    Divide(i32),       // i/=k
    Complex,           // 其他复杂更新
}
```

#### IterationEstimate

```rust
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IterationEstimate {
    Fixed(usize),             // 固定迭代次数
    Bounded(usize),           // 有界但可变
    Unbounded,                // 无界
    RuntimeDependent,         // 运行时依赖
}
```

#### InductionVariable

```rust
pub struct InductionVariable {
    pub name: Arc<str>,
    pub initial_value: Option<i32>,
    pub update_pattern: UpdatePattern,
    pub final_value: Option<i32>,
}
```

#### LoopAnalysis

```rust
pub struct LoopAnalysis {
    pub loop_type: LoopType,
    pub induction_variables: Vec<InductionVariable>,
    pub iteration_estimate: IterationEstimate,
    pub complexity_score: usize,        // 1-100
    pub nesting_depth: usize,
    pub is_infinite: bool,
}
```

### API 方法

#### new()

创建新的循环分析器。

```rust
let mut analyzer = LoopAnalyzer::new();
```

#### analyze()

分析所有循环。

**签名**:
```rust
pub fn analyze(
    &mut self,
    ast: &ExtractedAst,
    scope_tree: &ScopeTree,
    data_flow: &DataFlowGraph,
    cfg: &ControlFlowInfo,
) -> Result<Vec<LoopAnalysis>>
```

**返回**: 循环分析结果列表

**示例**:
```rust
let analyses = analyzer.analyze(&ast, &scope, &df, &cfg)?;
for (idx, analysis) in analyses.iter().enumerate() {
    println!("循环 #{}", idx);
    println!("  类型: {:?}", analysis.loop_type);
    println!("  迭代: {:?}", analysis.iteration_estimate);
    println!("  复杂度: {}", analysis.complexity_score);
}
```

#### detect_induction_variables()

从循环头和锁存块检测归纳变量。

**签名**:
```rust
pub fn detect_induction_variables(
    &self,
    loop_info: &LoopInfo,
) -> Vec<InductionVariable>
```

**返回**: 归纳变量列表

**示例**:
```rust
let induction_vars = analyzer.detect_induction_variables(&loop_info);
for var in induction_vars {
    println!("变量: {}", var.name);
    println!("  初值: {:?}", var.initial_value);
    println!("  更新: {:?}", var.update_pattern);
}
```

---

## OptimizedAnalyzer

### 概述

提供缓存、增量分析和性能监控的优化分析器。内部使用 Mutex 实现线程安全。

### 导入

```rust
use browerai::parser::js_analyzer::{OptimizedAnalyzer, AnalysisCache, PerformanceMetrics};
```

### 主要类型

#### AnalysisCache

```rust
pub struct AnalysisCache {
    // 私有字段
}

impl AnalysisCache {
    pub fn new() -> Self
    pub fn with_capacity(capacity: usize) -> Self
    pub fn get(&mut self, key: &str, input_hash: u64) -> Option<Arc<str>>
    pub fn put(&mut self, key: String, data: Arc<str>, input_hash: u64)
    pub fn clear(&mut self)
    pub fn stats(&self) -> CacheStats
}
```

#### CacheStats

```rust
pub struct CacheStats {
    pub size: usize,          // 缓存中的条目数
    pub hits: usize,          // 命中次数
    pub misses: usize,        // 未命中次数
}
```

#### IncrementalAnalyzer

```rust
pub struct IncrementalAnalyzer {
    // 私有字段
}

impl IncrementalAnalyzer {
    pub fn mark_analyzed(&mut self, func_name: &str, input_hash: u64)
    pub fn needs_analysis(&self, func_name: &str, input_hash: u64) -> bool
    pub fn add_dependency(&mut self, func_a: &str, func_b: &str)
    pub fn get_affected_functions(&self, changed_func: &str) -> Vec<String>
    pub fn invalidate_transitive(&mut self, func_name: &str)
    pub fn get_dirty_functions(&self) -> Vec<String>
}
```

#### PerformanceMetrics

```rust
pub struct PerformanceMetrics {
    pub total_time_ms: f64,
    pub analysis_count: usize,
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub peak_memory_bytes: usize,
}

impl PerformanceMetrics {
    pub fn avg_time_ms(&self) -> f64
    pub fn cache_hit_rate(&self) -> f64  // 0.0 - 1.0
}
```

#### OptimizedAnalyzer

```rust
pub struct OptimizedAnalyzer {
    // 私有字段: Arc<Mutex<AnalysisCache>>, 等
}

impl OptimizedAnalyzer {
    pub fn new() -> Self
    pub fn cache(&self, key: &str, input_hash: u64) -> Option<Arc<str>>
    pub fn cache_put(&mut self, key: String, data: Arc<str>, input_hash: u64)
    pub fn cache_clear(&mut self)
    pub fn cache_stats(&self) -> CacheStats
    pub fn metrics(&self) -> PerformanceMetrics
    pub fn record_cache_hit(&mut self)
    pub fn record_cache_miss(&mut self)
    pub fn record_analysis(&mut self, time_ms: f64)
    pub fn reset(&mut self)
}
```

### API 方法

#### new()

创建新的优化分析器。

```rust
let mut optimizer = OptimizedAnalyzer::new();
```

#### cache() - 查询缓存

查询缓存中的值。

**签名**:
```rust
pub fn cache(&self, key: &str, input_hash: u64) -> Option<Arc<str>>
```

**参数**:
- `key`: 缓存键
- `input_hash`: 输入数据的哈希值（用于验证）

**返回**: 缓存的数据，如果存在且哈希匹配

**示例**:
```rust
if let Some(cached) = optimizer.cache("analysis_foo", hash) {
    println!("缓存命中: {}", cached);
} else {
    println!("缓存未命中");
}
```

#### cache_put() - 写入缓存

将分析结果存入缓存。

**签名**:
```rust
pub fn cache_put(&mut self, key: String, data: Arc<str>, input_hash: u64)
```

**参数**:
- `key`: 缓存键
- `data`: 要缓存的数据
- `input_hash`: 输入的哈希值

**示例**:
```rust
let result = Arc::from("analysis_result_data");
optimizer.cache_put("key".to_string(), result, input_hash);
```

#### cache_stats() - 缓存统计

获取缓存统计信息。

**签名**:
```rust
pub fn cache_stats(&self) -> CacheStats
```

**返回**:
```rust
CacheStats {
    size: 42,
    hits: 100,
    misses: 25,
}
```

**示例**:
```rust
let stats = optimizer.cache_stats();
println!("缓存大小: {}", stats.size);
println!("命中率: {:.1}%", (stats.hits as f64 / (stats.hits + stats.misses) as f64) * 100.0);
```

#### metrics() - 性能指标

获取性能指标。

**签名**:
```rust
pub fn metrics(&self) -> PerformanceMetrics
```

**示例**:
```rust
let metrics = optimizer.metrics();
println!("总分析数: {}", metrics.analysis_count);
println!("平均耗时: {:.2}ms", metrics.avg_time_ms());
println!("缓存命中率: {:.1}%", metrics.cache_hit_rate() * 100.0);
```

#### record_cache_hit() / record_cache_miss()

记录缓存命中/未命中。

**签名**:
```rust
pub fn record_cache_hit(&mut self)
pub fn record_cache_miss(&mut self)
```

**使用场景**: 在 analyze() 方法中检查缓存时调用

**示例**:
```rust
if cached {
    optimizer.record_cache_hit();
} else {
    optimizer.record_cache_miss();
    // 执行分析...
    optimizer.record_analysis(elapsed_ms);
}
```

#### reset()

重置所有状态。

**签名**:
```rust
pub fn reset(&mut self)
```

**效果**:
- 清空缓存
- 重置增量分析器
- 重置性能指标

---

## AnalysisPipeline

### 概述

完整的分析管道，协调所有分析器（AST、作用域、数据流、CFG、循环、调用图）的执行，包含自动缓存和性能监控。

### 导入

```rust
use browerai::parser::js_analyzer::{AnalysisPipeline, FullAnalysisResult, PipelineStats};
```

### 主要类型

#### FullAnalysisResult

```rust
pub struct FullAnalysisResult {
    pub cached: bool,              // 是否来自缓存
    pub time_ms: f64,              // 分析耗时 (毫秒)
    pub ast_valid: bool,           // AST 有效性
    pub scope_count: usize,        // 作用域数量
    pub dataflow_nodes: usize,     // 数据流节点数
    pub cfg_nodes: usize,          // 控制流图节点数
    pub loop_count: usize,         // 循环数量
    pub call_edges: usize,         // 调用边数量
}
```

#### PipelineStats

```rust
pub struct PipelineStats {
    pub total_analyses: usize,     // 总分析次数
    pub cache_hit_rate: f64,       // 缓存命中率 (0.0-1.0)
    pub avg_time_ms: f64,          // 平均耗时 (毫秒)
    pub cache_size: usize,         // 缓存中的条目数
}
```

#### AnalysisPipeline

```rust
pub struct AnalysisPipeline {
    // 私有字段
}

impl AnalysisPipeline {
    pub fn new() -> Self
    pub fn analyze(&mut self, source: &str) -> Result<FullAnalysisResult>
    pub fn metrics(&self) -> PerformanceMetrics
    pub fn reset(&mut self)
    pub fn stats(&mut self) -> PipelineStats
}

impl Default for AnalysisPipeline {
    fn default() -> Self { Self::new() }
}
```

### API 方法

#### new()

创建新的分析管道。

```rust
let mut pipeline = AnalysisPipeline::new();
```

#### analyze()

执行完整的分析管道。

**签名**:
```rust
pub fn analyze(&mut self, source: &str) -> Result<FullAnalysisResult>
```

**参数**:
- `source`: JavaScript 源代码

**返回**:
- `Ok(FullAnalysisResult)`: 包含所有分析结果
- `Err(anyhow::Error)`: 分析失败

**分析流程**:
1. 计算源代码哈希
2. 查询缓存
3. (如果未命中) 执行 7 阶段分析:
   - 提取 AST
   - 分析作用域
   - 分析数据流
   - 分析控制流
   - 分析循环
   - 分析调用图
4. 记录性能指标
5. 存储结果到缓存
6. 返回结果

**时间复杂度**:
- 首次分析: O(n)，其中 n 是代码大小
- 缓存命中: O(1)

**示例**:
```rust
let code = r#"
    function quicksort(arr) {
        if (arr.length <= 1) return arr;
        const pivot = arr[0];
        const left = arr.filter(x => x < pivot);
        const right = arr.filter(x => x >= pivot);
        return [...quicksort(left), pivot, ...quicksort(right)];
    }
"#;

let result = pipeline.analyze(code)?;

println!("分析结果:");
println!("  缓存命中: {}", result.cached);
println!("  耗时: {:.2}ms", result.time_ms);
println!("  AST 有效: {}", result.ast_valid);
println!("  作用域数: {}", result.scope_count);
println!("  CFG 节点: {}", result.cfg_nodes);
println!("  循环数: {}", result.loop_count);
println!("  调用边: {}", result.call_edges);
```

#### metrics()

获取当前的性能指标。

**签名**:
```rust
pub fn metrics(&self) -> PerformanceMetrics
```

**返回**: 累积的性能指标

**示例**:
```rust
let metrics = pipeline.metrics();
println!("总分析次数: {}", metrics.analysis_count);
println!("总耗时: {:.2}ms", metrics.total_time_ms);
println!("平均耗时: {:.2}ms", metrics.avg_time_ms());
```

#### stats()

获取管道统计信息。

**签名**:
```rust
pub fn stats(&mut self) -> PipelineStats
```

**返回**: 包含分析和缓存统计的 PipelineStats

**示例**:
```rust
let stats = pipeline.stats();
println!("总分析: {}", stats.total_analyses);
println!("缓存命中率: {:.1}%", stats.cache_hit_rate * 100.0);
println!("平均耗时: {:.2}ms", stats.avg_time_ms);
println!("缓存大小: {}", stats.cache_size);
```

#### reset()

重置管道的所有状态。

**签名**:
```rust
pub fn reset(&mut self)
```

**效果**:
- 清空缓存
- 重置所有分析器
- 重置性能指标

**示例**:
```rust
pipeline.reset();
let stats = pipeline.stats();
assert_eq!(stats.total_analyses, 0);
```

---

## 工具函数 (Utility Functions)

### hash_string()

计算字符串的确定性哈希。

**签名**:
```rust
pub fn hash_string(s: &str) -> u64
```

**特性**:
- 确定性: 相同输入产生相同输出
- 快速: O(n)，其中 n 是字符串长度
- 用于缓存验证

**示例**:
```rust
use browerai::parser::js_analyzer::performance_optimizer::hash_string;

let hash1 = hash_string("let x = 42;");
let hash2 = hash_string("let x = 42;");
assert_eq!(hash1, hash2);
```

---

## 错误处理 (Error Handling)

所有分析器使用 `anyhow::Result<T>` 进行错误处理。

```rust
use anyhow::{Context, Result};

fn safe_analysis() -> Result<FullAnalysisResult> {
    let mut pipeline = AnalysisPipeline::new();
    let code = std::fs::read_to_string("main.js")
        .context("Failed to read source file")?;
    pipeline.analyze(&code)
        .context("Analysis failed")
}
```

### 常见错误

| 错误 | 原因 | 解决方案 |
|------|------|--------|
| 语法错误 | 代码包含无效的 JavaScript | 验证源代码 |
| 空指针 | 分析器状态不一致 | 使用 default() 或 new() |
| 缓存溢出 | 太多分析被缓存 | 调用 reset() 或增加缓存大小 |

---

## 性能调优 (Performance Tuning)

### 缓存配置

```rust
// 使用更大的缓存
let mut analyzer = OptimizedAnalyzer::new();
// 注意: 当前使用固定大小 100，可通过修改 DEFAULT_CACHE_SIZE 调整
```

### 指标监控

```rust
let mut pipeline = AnalysisPipeline::new();

// 多次分析
for code in code_samples {
    pipeline.analyze(code)?;
}

// 检查效率
let stats = pipeline.stats();
if stats.cache_hit_rate < 0.5 {
    println!("警告: 缓存命中率低，考虑增加缓存大小");
}
```

### 最佳实践

1. **重用管道实例** - 避免重复创建，利用缓存
2. **监控指标** - 定期检查性能数据
3. **合理的缓存大小** - 平衡内存和性能
4. **批量处理** - 分析多个文件时重用相同实例

---

## 完整示例 (Full Example)

```rust
use browerai::parser::js_analyzer::{
    AnalysisPipeline,
    EnhancedCallGraphAnalyzer,
    LoopAnalyzer,
};
use anyhow::Result;

fn analyze_code_samples() -> Result<()> {
    let samples = vec![
        r#"function fibonacci(n) { if (n <= 1) return n; return fibonacci(n-1) + fibonacci(n-2); }"#,
        r#"function loop_test() { for (let i = 0; i < 10; i++) { console.log(i); } }"#,
        r#"function main() { fibonacci(5); loop_test(); }"#,
    ];

    let mut pipeline = AnalysisPipeline::new();

    for (idx, code) in samples.iter().enumerate() {
        println!("\n样本 #{}:", idx + 1);
        
        let result = pipeline.analyze(code)?;
        
        println!("  缓存命中: {}", result.cached);
        println!("  耗时: {:.2}ms", result.time_ms);
        println!("  作用域: {}", result.scope_count);
        println!("  循环: {}", result.loop_count);
        println!("  调用边: {}", result.call_edges);
    }

    // 显示累计统计
    let stats = pipeline.stats();
    println!("\n累计统计:");
    println!("  总分析: {}", stats.total_analyses);
    println!("  缓存命中率: {:.1}%", stats.cache_hit_rate * 100.0);
    println!("  平均耗时: {:.2}ms", stats.avg_time_ms);
    println!("  缓存大小: {}", stats.cache_size);

    Ok(())
}
```

---

**文档版本**: 1.0  
**最后更新**: 2024  
**相关文件**: [PHASE3_WEEK3_COMPLETION_REPORT.md](./PHASE3_WEEK3_COMPLETION_REPORT.md)
