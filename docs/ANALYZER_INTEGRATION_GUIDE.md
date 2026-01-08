# JS Analyzer 管线集成指南

## 概述

本文档说明如何在 BrowerAI 的 JS 分析管线中集成混合 JS 编排器，以支持更精准的代码分析。

## 集成点

### 分析管线现状

当前的 `js_analyzer` 提供了深度分析能力：

```
JavaScript Code
    ↓
ScopeAnalyzer (词汇作用域)
    ↓
DataflowAnalyzer (变量流)
    ↓
ControlflowAnalyzer (控制流)
    ↓
CallGraphAnalyzer (调用关系)
    ↓
分析结果
```

### 混合编排器的价值

当前只能用启发式方法。混合编排器提供的能力：

1. **精准的 AST**（SWC）
   - 完整的 TypeScript 和 JSX 支持
   - 准确的作用域信息
   - 详细的语法树

2. **实时执行**（V8/Boa）
   - 静态分析无法获得的运行时信息
   - 动态属性和计算值
   - 模块加载和初始化效果

3. **智能选择**（策略）
   - 根据代码类型选择合适的分析方法
   - 平衡精度和速度

## 集成方案

### 方案 A：混合分析（推荐）

在分析过程中同时使用静态和动态分析：

```rust
pub struct HybridJsAnalyzer {
    // 静态分析部分
    scope_analyzer: ScopeAnalyzer,
    dataflow_analyzer: DataflowAnalyzer,
    callgraph_analyzer: CallGraphAnalyzer,
    
    // 动态分析部分（新增）
    ast_provider: AnalysisJsAstProvider,
    orchestrator: Option<HybridJsOrchestrator>,
}

#[derive(Debug, Clone)]
pub struct HybridAnalysisResult {
    // 静态分析结果
    pub scope_info: ScopeInfo,
    pub dataflow: DataflowGraph,
    pub call_graph: CallGraph,
    
    // 动态分析结果（新增）
    pub is_module: bool,
    pub is_typescript_jsx: bool,
    pub runtime_properties: HashMap<String, Value>,
    pub detected_frameworks: Vec<String>,
    
    // 融合结果
    pub confidence: f32,
    pub mixed_insights: Vec<String>,
}

impl HybridJsAnalyzer {
    pub fn new() -> Self {
        Self {
            scope_analyzer: ScopeAnalyzer::new(),
            dataflow_analyzer: DataflowAnalyzer::new(),
            callgraph_analyzer: CallGraphAnalyzer::new(),
            ast_provider: AnalysisJsAstProvider::new(),
            orchestrator: None,
            #[cfg(feature = "ai")]
            {
                orchestrator: Some(HybridJsOrchestrator::with_policy(
                    OrchestrationPolicy::Balanced
                ));
            }
        }
    }
    
    pub fn analyze(&mut self, source: &str) -> Result<HybridAnalysisResult> {
        // 第一步：静态分析（总是执行）
        let static_result = self.analyze_static(source)?;
        
        // 第二步：AST 特征检测
        let ast_info = self.ast_provider.parse_and_analyze(source)?;
        
        // 第三步：可选的动态分析
        let dynamic_result = self.analyze_dynamic(source)?;
        
        // 第四步：融合结果
        let combined = self.combine_results(
            static_result,
            ast_info,
            dynamic_result
        )?;
        
        Ok(combined)
    }
    
    fn analyze_static(&mut self, source: &str) -> Result<StaticAnalysisResult> {
        let scope = self.scope_analyzer.analyze(source)?;
        let dataflow = self.dataflow_analyzer.analyze(source, &scope)?;
        let call_graph = self.callgraph_analyzer.analyze(source, &scope)?;
        
        Ok(StaticAnalysisResult {
            scope,
            dataflow,
            call_graph,
        })
    }
    
    fn analyze_dynamic(&mut self, source: &str) -> Result<DynamicAnalysisResult> {
        #[cfg(feature = "ai")]
        {
            if let Some(orchestrator) = &mut self.orchestrator {
                let properties = self.extract_runtime_properties(source, orchestrator)?;
                let frameworks = self.detect_frameworks(source)?;
                
                return Ok(DynamicAnalysisResult {
                    properties,
                    frameworks,
                });
            }
        }
        
        Ok(DynamicAnalysisResult::default())
    }
    
    fn extract_runtime_properties(
        &self,
        source: &str,
        orchestrator: &mut HybridJsOrchestrator,
    ) -> Result<HashMap<String, Value>> {
        // 执行脚本并捕获全局对象
        let result = orchestrator.execute(
            &format!(
                "{}\nJSON.stringify(Object.keys(this))",
                source
            )
        )?;
        
        // 解析结果
        let keys: Vec<String> = serde_json::from_str(&result)?;
        
        let mut properties = HashMap::new();
        for key in keys {
            properties.insert(key, Value::String("detected".into()));
        }
        
        Ok(properties)
    }
    
    fn detect_frameworks(&self, source: &str) -> Result<Vec<String>> {
        let mut frameworks = Vec::new();
        
        // 框架特征检测
        if source.contains("React.createElement") || source.contains("jsx") {
            frameworks.push("React".to_string());
        }
        if source.contains("Vue.") || source.contains("defineComponent") {
            frameworks.push("Vue".to_string());
        }
        if source.contains("angular.module") || source.contains("@angular") {
            frameworks.push("Angular".to_string());
        }
        
        Ok(frameworks)
    }
    
    fn combine_results(
        &self,
        static_result: StaticAnalysisResult,
        ast_info: JsAnalysisResult,
        dynamic_result: DynamicAnalysisResult,
    ) -> Result<HybridAnalysisResult> {
        // 计算置信度
        let confidence = self.calculate_confidence(&static_result, &ast_info, &dynamic_result);
        
        // 生成洞察
        let insights = self.generate_insights(
            &static_result,
            &ast_info,
            &dynamic_result,
        );
        
        Ok(HybridAnalysisResult {
            scope_info: static_result.scope,
            dataflow: static_result.dataflow,
            call_graph: static_result.call_graph,
            is_module: ast_info.is_module,
            is_typescript_jsx: ast_info.is_typescript_jsx,
            runtime_properties: dynamic_result.properties,
            detected_frameworks: dynamic_result.frameworks,
            confidence,
            mixed_insights: insights,
        })
    }
    
    fn calculate_confidence(
        &self,
        _static: &StaticAnalysisResult,
        _ast: &JsAnalysisResult,
        _dynamic: &DynamicAnalysisResult,
    ) -> f32 {
        // 简单的置信度计算
        0.85
    }
    
    fn generate_insights(
        &self,
        _static: &StaticAnalysisResult,
        ast: &JsAnalysisResult,
        dynamic: &DynamicAnalysisResult,
    ) -> Vec<String> {
        let mut insights = Vec::new();
        
        if ast.is_module {
            insights.push("代码为 ES6 模块，需要特殊处理".into());
        }
        
        if !dynamic.frameworks.is_empty() {
            insights.push(format!(
                "检测到使用框架：{}",
                dynamic.frameworks.join(", ")
            ));
        }
        
        if ast.is_typescript_jsx {
            insights.push("代码使用 TypeScript 和 JSX".into());
        }
        
        insights
    }
}
```

### 方案 B：策略化分析

根据代码特征选择合适的分析策略：

```rust
pub enum AnalysisStrategy {
    /// 轻量级：仅静态分析
    Static,
    
    /// 标准：静态 + AST 特征检测
    Standard,
    
    /// 完整：静态 + 动态 + 框架检测
    Full,
}

pub struct AnalysisConfig {
    pub strategy: AnalysisStrategy,
    pub timeout_ms: u64,
    pub cache_enabled: bool,
    pub framework_detection: bool,
}

impl Default for AnalysisConfig {
    fn default() -> Self {
        Self {
            strategy: AnalysisStrategy::Standard,
            timeout_ms: 5000,
            cache_enabled: true,
            framework_detection: true,
        }
    }
}

impl HybridJsAnalyzer {
    pub fn analyze_with_config(
        &mut self,
        source: &str,
        config: AnalysisConfig,
    ) -> Result<HybridAnalysisResult> {
        match config.strategy {
            AnalysisStrategy::Static => {
                self.analyze_static(source).map(|static_result| {
                    HybridAnalysisResult::from_static(static_result)
                })
            }
            AnalysisStrategy::Standard => {
                self.analyze(source)
            }
            AnalysisStrategy::Full => {
                // 启用所有功能
                self.analyze(source)
            }
        }
    }
}
```

## 集成步骤

### Step 1：更新 js-analyzer 的 lib.rs

```rust
// 在 js-analyzer/src/lib.rs 中
pub mod hybrid_analyzer;

pub use hybrid_analyzer::{HybridJsAnalyzer, HybridAnalysisResult, AnalysisStrategy};
```

### Step 2：创建混合分析器模块

创建 `js-analyzer/src/hybrid_analyzer.rs`：

```rust
use crate::*;
use browerai_core::*;
use anyhow::Result;

pub struct HybridJsAnalyzer {
    // ... 如上所示
}

impl HybridJsAnalyzer {
    // ... 实现如上所示
}
```

### Step 3：集成到分析管线

```rust
// 在上层应用中使用

use browerai_js_analyzer::HybridJsAnalyzer;

let mut analyzer = HybridJsAnalyzer::new();
let result = analyzer.analyze(js_code)?;

println!("Is Module: {}", result.is_module);
println!("Frameworks: {:?}", result.detected_frameworks);
println!("Call Graph: {:?}", result.call_graph);
```

## 性能和缓存

### 分析缓存

```rust
use std::collections::HashMap;

pub struct AnalysisCache {
    cache: HashMap<u64, HybridAnalysisResult>,
}

impl AnalysisCache {
    pub fn get_or_analyze(
        &mut self,
        source: &str,
        analyzer: &mut HybridJsAnalyzer,
    ) -> Result<HybridAnalysisResult> {
        let hash = calculate_hash(source);
        
        if let Some(cached) = self.cache.get(&hash) {
            return Ok(cached.clone());
        }
        
        let result = analyzer.analyze(source)?;
        self.cache.insert(hash, result.clone());
        Ok(result)
    }
}
```

### 增量分析

对于大型项目，支持增量分析避免重复工作：

```rust
pub struct IncrementalAnalyzer {
    previous_result: Option<HybridAnalysisResult>,
    change_range: Option<(usize, usize)>,
}

impl IncrementalAnalyzer {
    pub fn analyze_incremental(
        &mut self,
        source: &str,
        analyzer: &mut HybridJsAnalyzer,
    ) -> Result<HybridAnalysisResult> {
        if let Some((start, end)) = self.change_range {
            // 仅分析变化的范围
            let changed_source = &source[start..end];
            // ... 仅分析改动部分 ...
        } else {
            // 完整分析
            analyzer.analyze(source)
        }
    }
}
```

## 测试

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hybrid_analysis_basic() {
        let mut analyzer = HybridJsAnalyzer::new();
        let source = "import x from 'y'; export const a = 1;";
        
        let result = analyzer.analyze(source).unwrap();
        assert!(result.is_module);
    }
    
    #[test]
    fn test_framework_detection() {
        let mut analyzer = HybridJsAnalyzer::new();
        let source = "const el = React.createElement('div', null);";
        
        let result = analyzer.analyze(source).unwrap();
        assert!(result.detected_frameworks.contains(&"React".to_string()));
    }
}
```

## 环境变量

```bash
# 分析管线的 JS 策略
export BROWERAI_ANALYSIS_JS_POLICY=balanced

# 分析超时时间（毫秒）
export BROWERAI_ANALYSIS_TIMEOUT=5000

# 启用分析缓存
export BROWERAI_ANALYSIS_CACHE=true
```

## 下一步

1. 实现完整的动态分析功能
2. 添加更多框架检测逻辑
3. 性能基准测试
4. 集成到 intelligent-rendering 管线
5. 添加分析结果的可视化

## 相关文档

- [混合 JS 编排器集成指南](./HYBRID_JS_ORCHESTRATION_INTEGRATION.md)
- [快速参考](./HYBRID_JS_QUICK_REFERENCE.md)
- [Renderer 集成指南](./RENDERER_INTEGRATION_GUIDE.md)
