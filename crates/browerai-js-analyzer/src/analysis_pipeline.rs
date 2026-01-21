//! Full Analysis Pipeline
//!
//! Orchestrates the complete analysis workflow integrating all analyzers
//! with caching and performance optimization.

use super::controlflow_analyzer::ControlFlowAnalyzer;
use super::dataflow_analyzer::DataFlowAnalyzer;
use super::extractor::AstExtractor;
use super::loop_analyzer::LoopAnalyzer;
use super::performance_optimizer::{OptimizedAnalyzer, PerformanceMetrics};
use super::scope_analyzer::ScopeAnalyzer;
use super::unified_call_graph::UnifiedCallGraphBuilder;
use anyhow::Result;
use std::time::Instant;

/// Complete analysis results from all analyzers
#[derive(Debug, Clone)]
pub struct FullAnalysisResult {
    /// Whether analysis was cached
    pub cached: bool,
    /// Time taken (milliseconds)
    pub time_ms: f64,
    /// AST extraction successful
    pub ast_valid: bool,
    /// Scope tree created
    pub scope_count: usize,
    /// Data flow nodes analyzed
    pub dataflow_nodes: usize,
    /// Control flow paths found
    pub cfg_nodes: usize,
    /// Loops detected
    pub loop_count: usize,
    /// Function call edges found
    pub call_edges: usize,
}

/// Full analysis pipeline orchestrator
pub struct AnalysisPipeline {
    optimizer: OptimizedAnalyzer,
    ast_extractor: AstExtractor,
    scope_analyzer: ScopeAnalyzer,
    dataflow_analyzer: DataFlowAnalyzer,
    cfg_analyzer: ControlFlowAnalyzer,
    loop_analyzer: LoopAnalyzer,
    call_graph_analyzer: UnifiedCallGraphBuilder,
}

impl AnalysisPipeline {
    /// Create new analysis pipeline
    pub fn new() -> Self {
        Self {
            optimizer: OptimizedAnalyzer::new(),
            ast_extractor: AstExtractor::new(),
            scope_analyzer: ScopeAnalyzer::new(),
            dataflow_analyzer: DataFlowAnalyzer::new(),
            cfg_analyzer: ControlFlowAnalyzer::new(),
            loop_analyzer: LoopAnalyzer::new(),
            call_graph_analyzer: UnifiedCallGraphBuilder::new(),
        }
    }

    /// Run complete analysis pipeline
    pub fn analyze(&mut self, source: &str) -> Result<FullAnalysisResult> {
        let start = Instant::now();

        // Check cache
        let source_hash = super::performance_optimizer::hash_string(source);
        let cache_key = "full_analysis".to_string();

        if let Some(_cached) = self.optimizer.cache(&cache_key, source_hash) {
            self.optimizer.record_cache_hit();
            return Ok(FullAnalysisResult {
                cached: true,
                time_ms: 0.0,
                ast_valid: false,
                scope_count: 0,
                dataflow_nodes: 0,
                cfg_nodes: 0,
                loop_count: 0,
                call_edges: 0,
            });
        }

        self.optimizer.record_cache_miss();

        // Extract AST
        let ast = self.ast_extractor.extract_from_source(source)?;
        let ast_valid = ast.metadata.is_valid;

        // Analyze scopes
        let scope_tree = self.scope_analyzer.analyze(&ast)?;
        let scope_count = scope_tree.scopes.len();

        // Data flow analysis
        let data_flow = self.dataflow_analyzer.analyze(&ast, &scope_tree)?;
        let dataflow_nodes = data_flow.nodes.len();

        // Control flow analysis
        let control_flow = self.cfg_analyzer.analyze(&ast)?;
        let cfg_nodes = control_flow.nodes.len();
        let loop_count = control_flow.loops.len();

        // Loop analysis
        let _loop_analyses =
            self.loop_analyzer
                .analyze(&ast, &scope_tree, &data_flow, &control_flow)?;

        // Call graph analysis - use semantic info from extracted AST
        let call_graph = self.call_graph_analyzer.build(&ast.semantic)?;
        let call_edges = call_graph.nodes.len();

        let time_ms = start.elapsed().as_secs_f64() * 1000.0;
        self.optimizer.record_analysis(time_ms);

        // Store in cache
        self.optimizer.cache_put(
            cache_key,
            std::sync::Arc::from(format!("analyzed:{}:{}", ast_valid, scope_count)),
            source_hash,
        );

        Ok(FullAnalysisResult {
            cached: false,
            time_ms,
            ast_valid,
            scope_count,
            dataflow_nodes,
            cfg_nodes,
            loop_count,
            call_edges,
        })
    }

    /// Get performance metrics
    pub fn metrics(&self) -> PerformanceMetrics {
        self.optimizer.metrics()
    }

    /// Reset pipeline state
    pub fn reset(&mut self) {
        self.optimizer.reset();
    }

    /// Get pipeline statistics
    pub fn stats(&mut self) -> PipelineStats {
        let metrics = self.optimizer.metrics();
        PipelineStats {
            total_analyses: metrics.analysis_count,
            cache_hit_rate: metrics.cache_hit_rate(),
            avg_time_ms: metrics.avg_time_ms(),
            cache_size: self.optimizer.cache_stats().size,
        }
    }
}

impl Default for AnalysisPipeline {
    fn default() -> Self {
        Self::new()
    }
}

/// Pipeline statistics
#[derive(Debug, Clone)]
pub struct PipelineStats {
    pub total_analyses: usize,
    pub cache_hit_rate: f64,
    pub avg_time_ms: f64,
    pub cache_size: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_creation() {
        let mut pipeline = AnalysisPipeline::new();
        let stats = pipeline.stats();
        assert_eq!(stats.total_analyses, 0);
    }

    #[test]
    fn test_simple_analysis() {
        let mut pipeline = AnalysisPipeline::new();
        let code = "function test() { return 42; }";

        let result = pipeline.analyze(code);
        assert!(result.is_ok(), "Analysis should succeed");

        let r = result.expect("Analysis result should be available");
        assert!(!r.cached);
        assert!(r.ast_valid);
    }

    #[test]
    fn test_cache_hit() {
        let mut pipeline = AnalysisPipeline::new();
        let code = "function test() { return 42; }";

        let r1 = pipeline
            .analyze(code)
            .expect("First analysis should succeed");
        assert!(!r1.cached);

        let r2 = pipeline
            .analyze(code)
            .expect("Second analysis should succeed");
        assert!(r2.cached);
    }

    #[test]
    fn test_metrics_recording() {
        let mut pipeline = AnalysisPipeline::new();
        let code = "function test() { return 42; }";

        pipeline.analyze(code).expect("Analysis should succeed");

        let stats = pipeline.stats();
        assert!(stats.total_analyses > 0);
    }

    #[test]
    fn test_complex_code_analysis() {
        let mut pipeline = AnalysisPipeline::new();
        let code = r#"
            function outer() {
                function inner() {
                    for (let i = 0; i < 10; i++) {
                        console.log(i);
                    }
                }
                inner();
            }
        "#;

        let result = pipeline.analyze(code);
        assert!(result.is_ok(), "Analysis should succeed");

        let r = result.expect("Analysis result should be available");
        assert!(r.scope_count > 0);
        assert!(r.cfg_nodes > 0);
    }

    #[test]
    fn test_pipeline_reset() {
        let mut pipeline = AnalysisPipeline::new();
        let code = "function test() { return 42; }";

        pipeline.analyze(code).expect("Analysis should succeed");
        assert!(pipeline.stats().total_analyses > 0);

        pipeline.reset();
        assert_eq!(pipeline.stats().total_analyses, 0);
    }
}
