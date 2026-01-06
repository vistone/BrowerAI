# Phase 3 Week 3 - 集成指南

## 概述 (Overview)

本指南指导开发者如何将 Phase 3 Week 3 的新模块集成到现有项目中。

---

## 1. 依赖导入 (Importing Dependencies)

### 基础导入

```rust
// 导入主要类型
use browerai::parser::js_analyzer::{
    AnalysisPipeline,
    FullAnalysisResult,
    PipelineStats,
};

// 导入个别分析器
use browerai::parser::js_analyzer::{
    EnhancedCallGraphAnalyzer,
    LoopAnalyzer,
    OptimizedAnalyzer,
};

// 导入辅助类型
use browerai::parser::js_analyzer::{
    AnalysisCache,
    PerformanceMetrics,
    IncrementalAnalyzer,
};
```

### Cargo.toml 配置

```toml
[dependencies]
browerai = { path = "./browerai", features = ["ai"] }
anyhow = "1.0"
log = "0.4"
env_logger = "0.10"  # 可选：用于日志
```

---

## 2. 快速启动 (Quick Start)

### 最小示例

```rust
use browerai::parser::js_analyzer::AnalysisPipeline;

fn main() -> anyhow::Result<()> {
    let mut pipeline = AnalysisPipeline::new();
    
    let code = "function test() { return 42; }";
    let result = pipeline.analyze(code)?;
    
    println!("分析成功: {} 个作用域", result.scope_count);
    Ok(())
}
```

### 启用日志

```rust
fn main() -> anyhow::Result<()> {
    env_logger::init();  // 初始化日志
    
    log::info!("开始分析");
    let mut pipeline = AnalysisPipeline::new();
    let result = pipeline.analyze("let x = 1;")?;
    log::debug!("分析结果: {:?}", result);
    
    Ok(())
}
```

---

## 3. 集成场景 (Integration Scenarios)

### 场景 A: IDE 集成 (LSP Server)

```rust
use browerai::parser::js_analyzer::AnalysisPipeline;

/// LSP 文档分析服务
pub struct AnalysisService {
    pipeline: AnalysisPipeline,
}

impl AnalysisService {
    pub fn new() -> Self {
        Self {
            pipeline: AnalysisPipeline::new(),
        }
    }

    /// 分析文档（当文件改变时调用）
    pub fn analyze_document(&mut self, uri: &str, content: &str) -> anyhow::Result<()> {
        match self.pipeline.analyze(content) {
            Ok(result) => {
                log::info!("分析 {}: {} 个作用域", uri, result.scope_count);
                self.publish_diagnostics(uri, &result);
                Ok(())
            }
            Err(e) => {
                log::error!("分析失败 {}: {}", uri, e);
                Err(e)
            }
        }
    }

    fn publish_diagnostics(&self, uri: &str, result: &FullAnalysisResult) {
        // 发送诊断信息到客户端
        log::debug!("诊断信息 {}: AST有效={}, 循环数={}", 
            uri, result.ast_valid, result.loop_count);
    }
}
```

**使用**:
```rust
let mut service = AnalysisService::new();
service.analyze_document("file:///src/main.js", "let x = 1;")?;
```

### 场景 B: 静态分析工具

```rust
use browerai::parser::js_analyzer::{AnalysisPipeline, OptimizedAnalyzer};
use std::path::Path;

/// 代码质量分析工具
pub struct CodeQualityChecker {
    pipeline: AnalysisPipeline,
    optimizer: OptimizedAnalyzer,
}

impl CodeQualityChecker {
    pub fn new() -> Self {
        Self {
            pipeline: AnalysisPipeline::new(),
            optimizer: OptimizedAnalyzer::new(),
        }
    }

    /// 检查单个文件
    pub fn check_file(&mut self, path: &Path) -> anyhow::Result<CheckResult> {
        let content = std::fs::read_to_string(path)?;
        let analysis = self.pipeline.analyze(&content)?;

        let quality_score = self.calculate_score(&analysis);
        
        Ok(CheckResult {
            file: path.to_string_lossy().to_string(),
            quality_score,
            scope_count: analysis.scope_count,
            loop_count: analysis.loop_count,
            complexity: analysis.call_edges as i32,
        })
    }

    fn calculate_score(&self, result: &FullAnalysisResult) -> f64 {
        let mut score = 100.0;
        
        // 根据循环数扣分
        score -= (result.loop_count as f64) * 2.0;
        
        // 根据调用边数扣分
        score -= (result.call_edges as f64) * 1.0;
        
        score.max(0.0)
    }

    /// 批量检查目录
    pub fn check_directory(&mut self, dir: &Path) -> anyhow::Result<Vec<CheckResult>> {
        let mut results = Vec::new();
        
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            
            if path.extension().map_or(false, |e| e == "js") {
                match self.check_file(&path) {
                    Ok(result) => results.push(result),
                    Err(e) => log::warn!("检查失败 {}: {}", path.display(), e),
                }
            }
        }
        
        Ok(results)
    }
}

#[derive(Debug)]
pub struct CheckResult {
    pub file: String,
    pub quality_score: f64,
    pub scope_count: usize,
    pub loop_count: usize,
    pub complexity: i32,
}
```

**使用**:
```rust
let mut checker = CodeQualityChecker::new();
let results = checker.check_directory(Path::new("src/"))?;

for result in results {
    println!("{}: 质量分数 {:.1}", result.file, result.quality_score);
}
```

### 场景 C: 性能分析工具

```rust
use browerai::parser::js_analyzer::AnalysisPipeline;

/// 性能分析器
pub struct PerformanceAnalyzer {
    pipeline: AnalysisPipeline,
}

impl PerformanceAnalyzer {
    pub fn new() -> Self {
        Self {
            pipeline: AnalysisPipeline::new(),
        }
    }

    /// 分析代码复杂度
    pub fn analyze_complexity(&mut self, code: &str) -> anyhow::Result<ComplexityReport> {
        let result = self.pipeline.analyze(code)?;
        
        let cyclomatic_complexity = self.estimate_cyclomatic_complexity(&result);
        let cognitive_complexity = self.estimate_cognitive_complexity(&result);
        
        Ok(ComplexityReport {
            cyclomatic_complexity,
            cognitive_complexity,
            loop_count: result.loop_count,
            call_graph_size: result.call_edges,
            nesting_depth: 3,  // 简化示例
        })
    }

    fn estimate_cyclomatic_complexity(&self, result: &FullAnalysisResult) -> i32 {
        // CC = cfg_nodes - edges + 2
        (result.cfg_nodes as i32 - result.call_edges as i32 + 2).max(1)
    }

    fn estimate_cognitive_complexity(&self, result: &FullAnalysisResult) -> i32 {
        // 嵌套 + 决策
        let nesting = result.loop_count as i32;
        let decisions = result.cfg_nodes as i32 / 2;
        nesting + decisions
    }

    /// 性能基准测试
    pub fn benchmark(&mut self, code: &str, iterations: usize) -> anyhow::Result<BenchmarkResult> {
        let mut times = Vec::new();
        
        for _ in 0..iterations {
            let result = self.pipeline.analyze(code)?;
            times.push(result.time_ms);
        }
        
        let avg_time = times.iter().sum::<f64>() / times.len() as f64;
        let min_time = times.iter().copied().fold(f64::INFINITY, f64::min);
        let max_time = times.iter().copied().fold(0.0, f64::max);
        
        Ok(BenchmarkResult {
            avg_time_ms: avg_time,
            min_time_ms: min_time,
            max_time_ms: max_time,
            iterations,
        })
    }
}

#[derive(Debug)]
pub struct ComplexityReport {
    pub cyclomatic_complexity: i32,
    pub cognitive_complexity: i32,
    pub loop_count: usize,
    pub call_graph_size: usize,
    pub nesting_depth: i32,
}

#[derive(Debug)]
pub struct BenchmarkResult {
    pub avg_time_ms: f64,
    pub min_time_ms: f64,
    pub max_time_ms: f64,
    pub iterations: usize,
}
```

**使用**:
```rust
let mut analyzer = PerformanceAnalyzer::new();

// 分析复杂度
let code = "function test(arr) { for (let i = 0; i < arr.length; i++) { ... } }";
let report = analyzer.analyze_complexity(code)?;
println!("圈复杂度: {}", report.cyclomatic_complexity);

// 性能基准
let benchmark = analyzer.benchmark(code, 100)?;
println!("平均耗时: {:.2}ms", benchmark.avg_time_ms);
```

### 场景 D: 缓存管理

```rust
use browerai::parser::js_analyzer::OptimizedAnalyzer;

/// 缓存管理器
pub struct CacheManager {
    optimizer: OptimizedAnalyzer,
}

impl CacheManager {
    pub fn new() -> Self {
        Self {
            optimizer: OptimizedAnalyzer::new(),
        }
    }

    /// 预热缓存
    pub fn warm_up_cache(&mut self, code_samples: &[&str]) -> anyhow::Result<()> {
        for (idx, code) in code_samples.iter().enumerate() {
            let hash = browerai::parser::js_analyzer::performance_optimizer::hash_string(code);
            let key = format!("sample_{}", idx);
            let data = std::sync::Arc::from(code.to_string());
            self.optimizer.cache_put(key, data, hash);
        }
        
        let stats = self.optimizer.cache_stats();
        log::info!("缓存预热完成: {} 条记录", stats.size);
        Ok(())
    }

    /// 获取缓存统计
    pub fn get_stats(&self) -> CacheStats {
        self.optimizer.cache_stats()
    }

    /// 清空缓存
    pub fn clear(&mut self) {
        self.optimizer.cache_clear();
        log::info!("缓存已清空");
    }
}

use browerai::parser::js_analyzer::performance_optimizer::CacheStats;
```

---

## 4. 错误处理 (Error Handling)

### 基本错误处理

```rust
use anyhow::{Context, Result};

fn safe_analysis(code: &str) -> Result<()> {
    let mut pipeline = AnalysisPipeline::new();
    
    pipeline.analyze(code)
        .context("JavaScript 分析失败")?;
    
    Ok(())
}
```

### 高级错误处理

```rust
use anyhow::{anyhow, Context, Result};

fn analyze_with_recovery(code: &str) -> Result<()> {
    let mut pipeline = AnalysisPipeline::new();
    
    match pipeline.analyze(code) {
        Ok(result) => {
            if !result.ast_valid {
                return Err(anyhow!("AST 无效"));
            }
            log::info!("分析成功");
            Ok(())
        }
        Err(e) => {
            log::error!("分析失败: {}", e);
            
            // 尝试恢复
            pipeline.reset();
            
            // 可选: 重试或使用备用策略
            Err(e).context("无法恢复错误")
        }
    }
}
```

### 错误分类

| 错误类型 | 原因 | 处理 |
|---------|------|------|
| 语法错误 | JavaScript 代码无效 | 验证源码，返回诊断 |
| 分析失败 | 内部错误 | 记录日志，重置管道 |
| 超时 | 分析耗时过长 | 取消分析，使用缓存 |
| 内存 | 内存不足 | 清空缓存，垃圾回收 |

---

## 5. 性能优化 (Performance Optimization)

### 缓存策略

```rust
fn optimize_with_caching() -> anyhow::Result<()> {
    let mut pipeline = AnalysisPipeline::new();
    
    let code_samples = vec![
        "let x = 1;",
        "let y = 2;",
        "let x = 1;",  // 重复，应该命中缓存
    ];
    
    for code in code_samples {
        let result = pipeline.analyze(code)?;
        if result.cached {
            log::info!("缓存命中，节省 {:.2}ms", result.time_ms);
        } else {
            log::info!("新分析，耗时 {:.2}ms", result.time_ms);
        }
    }
    
    let stats = pipeline.stats();
    log::info!("最终缓存命中率: {:.1}%", stats.cache_hit_rate * 100.0);
    
    Ok(())
}
```

### 增量分析

```rust
fn incremental_analysis_example() -> anyhow::Result<()> {
    use browerai::parser::js_analyzer::OptimizedAnalyzer;
    
    let mut optimizer = OptimizedAnalyzer::new();
    
    // 第一次分析
    let code1 = "function foo() { return 42; }";
    let hash1 = browerai::parser::js_analyzer::performance_optimizer::hash_string(code1);
    
    // 如果有依赖信息
    // optimizer.incremental().mark_analyzed("foo", hash1);
    
    // 代码改变
    let code2 = "function foo() { return 43; }  function bar() { return foo(); }";
    let hash2 = browerai::parser::js_analyzer::performance_optimizer::hash_string(code2);
    
    // 只需要分析被修改函数及其依赖
    // let affected = optimizer.incremental().get_affected_functions("foo");
    
    Ok(())
}
```

### 批量处理

```rust
fn batch_analysis(files: &[(&str, &str)]) -> anyhow::Result<Vec<AnalysisResult>> {
    let mut pipeline = AnalysisPipeline::new();
    let mut results = Vec::new();
    
    for (filename, content) in files {
        log::info!("分析: {}", filename);
        let result = pipeline.analyze(content)?;
        results.push(AnalysisResult {
            filename: filename.to_string(),
            result,
        });
    }
    
    // 显示总体统计
    let stats = pipeline.stats();
    log::info!("批量分析完成: {}", stats.total_analyses);
    log::info!("平均耗时: {:.2}ms", stats.avg_time_ms);
    log::info!("缓存命中率: {:.1}%", stats.cache_hit_rate * 100.0);
    
    Ok(results)
}

#[derive(Debug)]
struct AnalysisResult {
    filename: String,
    result: FullAnalysisResult,
}
```

---

## 6. 测试集成 (Testing Integration)

### 单元测试

```rust
#[cfg(test)]
mod tests {
    use browerai::parser::js_analyzer::AnalysisPipeline;

    #[test]
    fn test_analysis_pipeline() {
        let mut pipeline = AnalysisPipeline::new();
        let code = "function test() { return 42; }";
        
        let result = pipeline.analyze(code);
        assert!(result.is_ok());
        
        let r = result.unwrap();
        assert!(r.ast_valid);
        assert_eq!(r.loop_count, 0);
    }

    #[test]
    fn test_caching() {
        let mut pipeline = AnalysisPipeline::new();
        let code = "let x = 1;";
        
        // 第一次分析
        let r1 = pipeline.analyze(code).unwrap();
        assert!(!r1.cached);
        
        // 第二次应该命中缓存
        let r2 = pipeline.analyze(code).unwrap();
        assert!(r2.cached);
    }
}
```

### 集成测试

```rust
#[cfg(test)]
mod integration_tests {
    use browerai::parser::js_analyzer::{
        AnalysisPipeline,
        EnhancedCallGraphAnalyzer,
        LoopAnalyzer,
    };

    #[test]
    fn test_full_pipeline() -> anyhow::Result<()> {
        let mut pipeline = AnalysisPipeline::new();
        let code = include_str!("fixtures/complex.js");
        
        let result = pipeline.analyze(code)?;
        
        assert!(result.ast_valid);
        assert!(result.scope_count > 0);
        assert!(result.call_edges > 0);
        
        Ok(())
    }
}
```

---

## 7. 部署清单 (Deployment Checklist)

### 集成前检查

- [ ] 依赖版本正确
- [ ] 导入路径正确
- [ ] 错误处理就位
- [ ] 日志配置完毕
- [ ] 测试通过

### 运行时检查

- [ ] 缓存工作正常
- [ ] 性能指标被收集
- [ ] 日志级别适当
- [ ] 内存使用合理
- [ ] CPU 使用在预期范围内

### 监控指标

```rust
fn monitor_performance(pipeline: &AnalysisPipeline) {
    let metrics = pipeline.metrics();
    
    // 关键指标
    assert!(metrics.avg_time_ms() < 100.0, "分析耗时过长");
    assert!(metrics.cache_hit_rate() > 0.7, "缓存命中率过低");
    
    // 可选警告
    if metrics.cache_hit_rate() < 0.5 {
        log::warn!("缓存命中率低于 50%");
    }
}
```

---

## 8. 常见问题 (FAQ)

### Q1: 如何重置分析器状态？

```rust
pipeline.reset();
```

### Q2: 缓存的默认大小是多少？

答: 100 条记录。可通过修改 `DEFAULT_CACHE_SIZE` 调整。

### Q3: 支持并行分析吗？

答: 当前使用 Mutex 提供线程安全，支持多线程访问。可使用 Rayon 进行并行处理。

### Q4: 如何调试性能问题？

```rust
env::set_var("RUST_LOG", "debug");
env_logger::init();
// 现在所有 log::debug!() 调用会显示
```

### Q5: 支持增量分析吗？

答: 提供了 IncrementalAnalyzer，可追踪依赖和脏数据。

---

## 9. 最佳实践总结 (Best Practices)

### DO ✅

- ✅ 重用 AnalysisPipeline 实例以利用缓存
- ✅ 使用 `context()` 添加有意义的错误信息
- ✅ 监控性能指标
- ✅ 定期调用 `reset()` 清理旧数据
- ✅ 使用日志追踪问题

### DON'T ❌

- ❌ 为每个分析创建新的 AnalysisPipeline
- ❌ 忽视错误返回值
- ❌ 在紧密循环中创建分析器
- ❌ 不管理缓存大小
- ❌ 使用 unwrap() 在生产代码中

---

**版本**: 1.0  
**最后更新**: 2024  
**相关文档**: 
- [API 参考](./PHASE3_WEEK3_API_GUIDE.md)
- [完成报告](./PHASE3_WEEK3_COMPLETION_REPORT.md)
