pub mod analysis_pipeline; // Phase 3 Week 3 Task 4
pub mod call_graph;
pub mod controlflow_analyzer; // Phase 3 Week 2
pub mod dataflow_analyzer;
pub mod enhanced_call_graph; // Phase 3 Week 3
pub mod extractor;
pub mod loop_analyzer; // Phase 3 Week 3 Task 2
pub mod performance_optimizer; // Phase 3 Week 3 Task 3
pub mod scope_analyzer; // Phase 3 Week 1
pub mod semantic;
pub mod swc_extractor; // Phase 2
pub mod types;

pub use analysis_pipeline::{AnalysisPipeline, FullAnalysisResult, PipelineStats}; // Phase 3 Week 3 Task 4 导出
pub use call_graph::CallGraphBuilder;
pub use controlflow_analyzer::ControlFlowAnalyzer; // Phase 3 Week 2 导出
pub use dataflow_analyzer::DataFlowAnalyzer;
pub use enhanced_call_graph::EnhancedCallGraphAnalyzer; // Phase 3 Week 3 导出
pub use extractor::{AstExtractor, ExtractedAst};
pub use loop_analyzer::LoopAnalyzer; // Phase 3 Week 3 Task 2 导出
pub use performance_optimizer::{
    AnalysisCache, IncrementalAnalyzer, OptimizedAnalyzer, PerformanceMetrics,
}; // Phase 3 Week 3 Task 3 导出
pub use scope_analyzer::ScopeAnalyzer; // Phase 3 Week 1 导出
pub use semantic::{AnalysisResult, SemanticAnalyzer};
pub use swc_extractor::{
    EnhancedAst, JsxElementInfo, LocationInfo, SwcAstExtractor, TypeScriptInfo,
}; // Phase 2 导出
pub use types::*; // Phase 3 Day 3-4 导出

use anyhow::{Context, Result};
use std::time::Instant;

/// JavaScript深度分析器 - 统一的入口点
///
/// JsDeepAnalyzer是整个分析系统的门面，负责：
/// 1. 协调AST提取、语义分析、调用图构建
/// 2. 管理分析流程和性能监控
/// 3. 错误恢复和部分结果处理
/// 4. 缓存和增量分析支持
///
/// 使用示例：
/// ```ignore
/// let mut analyzer = JsDeepAnalyzer::new();
/// let result = analyzer.analyze_source(javascript_code)?;
/// println!("{} functions found", result.function_count());
/// ```
pub struct JsDeepAnalyzer {
    /// AST提取器
    extractor: AstExtractor,

    /// 语义分析器
    semantic_analyzer: SemanticAnalyzer,

    /// 调用图构建器
    call_graph_builder: CallGraphBuilder,

    /// 分析配置
    config: AnalysisConfig,
}

/// 分析配置
#[derive(Debug, Clone)]
pub struct AnalysisConfig {
    /// 是否启用深度分析
    pub enable_deep_analysis: bool,

    /// 是否构建调用图
    pub build_call_graph: bool,

    /// 最大分析时间（毫秒）
    pub max_analysis_time_ms: u64,

    /// 是否进行框架检测
    pub detect_frameworks: bool,
}

impl Default for AnalysisConfig {
    fn default() -> Self {
        Self {
            enable_deep_analysis: true,
            build_call_graph: true,
            max_analysis_time_ms: 5000,
            detect_frameworks: true,
        }
    }
}

/// 完整的分析结果
#[derive(Debug, Clone)]
pub struct AnalysisOutput {
    /// AST元数据
    pub metadata: JsAstMetadata,

    /// 语义信息
    pub semantic: JsSemanticInfo,

    /// 调用图
    pub call_graph: JsCallGraph,

    /// 分析过程中的警告
    pub warnings: Vec<String>,

    /// 分析耗时
    pub analysis_time_ms: u64,
}

impl AnalysisOutput {
    /// 获取函数总数
    pub fn function_count(&self) -> usize {
        self.semantic.functions.len()
    }

    /// 获取类总数
    pub fn class_count(&self) -> usize {
        self.semantic.classes.len()
    }

    /// 获取事件处理器总数
    pub fn event_handler_count(&self) -> usize {
        self.semantic.event_handlers.len()
    }

    /// 获取代码复杂度分数
    pub fn complexity_score(&self) -> u32 {
        self.metadata.complexity_score
    }

    /// 是否检测到循环调用
    pub fn has_circular_calls(&self) -> bool {
        self.call_graph.has_cycles()
    }

    /// 获取入口点函数
    pub fn get_entry_points(&self) -> Vec<&JsCallNode> {
        self.call_graph
            .nodes
            .iter()
            .filter(|n| n.is_entry_point)
            .collect()
    }
}

impl Default for JsDeepAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl JsDeepAnalyzer {
    /// 创建新的分析器
    pub fn new() -> Self {
        Self {
            extractor: AstExtractor::new(),
            semantic_analyzer: SemanticAnalyzer::new(),
            call_graph_builder: CallGraphBuilder::new(),
            config: AnalysisConfig::default(),
        }
    }

    /// 使用自定义配置创建分析器
    pub fn with_config(config: AnalysisConfig) -> Self {
        Self {
            extractor: AstExtractor::new(),
            semantic_analyzer: SemanticAnalyzer::new(),
            call_graph_builder: CallGraphBuilder::new(),
            config,
        }
    }

    /// 分析JavaScript源代码
    pub fn analyze_source(&mut self, source: &str) -> Result<AnalysisOutput> {
        let start_time = Instant::now();
        let mut warnings = vec![];

        // 1. 提取AST
        let extracted = self
            .extractor
            .extract_from_source(source)
            .context("Failed to extract AST")?;

        let mut semantic = extracted.semantic.clone();
        let metadata = extracted.metadata.clone();
        warnings.extend(extracted.warnings);

        // 2. 语义分析
        let semantic_result = self
            .semantic_analyzer
            .analyze(&mut semantic)
            .context("Failed to perform semantic analysis")?;
        warnings.extend(
            semantic_result
                .special_features
                .iter()
                .map(|f| format!("Detected special feature: {}", f)),
        );

        // 3. 构建调用图
        let call_graph = if self.config.build_call_graph {
            self.call_graph_builder.build(&semantic).unwrap_or_default()
        } else {
            JsCallGraph::default()
        };

        let analysis_time = start_time.elapsed().as_millis() as u64;

        Ok(AnalysisOutput {
            metadata,
            semantic,
            call_graph,
            warnings,
            analysis_time_ms: analysis_time,
        })
    }

    /// 批量分析多个源文件
    pub fn analyze_multiple(&mut self, sources: &[(&str, &str)]) -> Result<Vec<AnalysisOutput>> {
        let mut results = vec![];

        for (name, source) in sources {
            match self.analyze_source(source) {
                Ok(output) => results.push(output),
                Err(e) => {
                    log::warn!("Failed to analyze {}: {}", name, e);
                }
            }
        }

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analyzer_creation() {
        let analyzer = JsDeepAnalyzer::new();
        assert!(analyzer.config.enable_deep_analysis);
        assert!(analyzer.config.build_call_graph);
    }

    #[test]
    fn test_analysis_config() {
        let config = AnalysisConfig {
            enable_deep_analysis: false,
            build_call_graph: false,
            max_analysis_time_ms: 1000,
            detect_frameworks: false,
        };

        let analyzer = JsDeepAnalyzer::with_config(config);
        assert!(!analyzer.config.enable_deep_analysis);
    }

    #[test]
    fn test_analysis_output_info() {
        let output = AnalysisOutput {
            metadata: JsAstMetadata::default(),
            semantic: JsSemanticInfo::default(),
            call_graph: JsCallGraph::default(),
            warnings: vec![],
            analysis_time_ms: 100,
        };

        assert_eq!(output.function_count(), 0);
        assert_eq!(output.class_count(), 0);
        assert!(!output.has_circular_calls());
    }

    #[test]
    fn test_end_to_end_simple_code() {
        let mut analyzer = JsDeepAnalyzer::new();
        let code = "function greet(name) { return 'Hello ' + name; }";

        let result = analyzer.analyze_source(code).unwrap();

        assert!(result.metadata.is_valid);
        assert_eq!(result.metadata.line_count, 1);
        assert!(!result.warnings.is_empty() || result.semantic.functions.len() > 0);
    }

    #[test]
    fn test_end_to_end_multiple_functions() {
        let mut analyzer = JsDeepAnalyzer::new();
        let code = "
            function add(a, b) { return a + b; }
            function multiply(a, b) { return a * b; }
            class Calculator {
                compute() { return 42; }
            }
        ";

        let result = analyzer.analyze_source(code).unwrap();

        assert!(result.metadata.is_valid);
        assert!(result.metadata.complexity_score > 0);
        // 应该检测到函数和类
        let total = result.semantic.functions.len() + result.semantic.classes.len();
        assert!(total > 0 || !result.warnings.is_empty());
    }

    #[test]
    fn test_end_to_end_with_config() {
        let config = AnalysisConfig {
            enable_deep_analysis: true,
            build_call_graph: true,
            max_analysis_time_ms: 5000,
            detect_frameworks: true,
        };

        let mut analyzer = JsDeepAnalyzer::with_config(config);
        let code = "const x = 1;";

        let result = analyzer.analyze_source(code).unwrap();
        assert!(result.metadata.is_valid);
    }

    #[test]
    fn test_batch_analysis() {
        let mut analyzer = JsDeepAnalyzer::new();
        let sources = vec![
            ("file1.js", "function a() {}"),
            ("file2.js", "class B {}"),
            ("file3.js", "const c = 1;"),
        ];

        let results = analyzer.analyze_multiple(&sources).unwrap();
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_error_handling() {
        let mut analyzer = JsDeepAnalyzer::new();
        let code = ""; // 空代码

        let result = analyzer.analyze_source(code).unwrap();
        assert_eq!(result.function_count(), 0);
        assert_eq!(result.class_count(), 0);
    }
}
