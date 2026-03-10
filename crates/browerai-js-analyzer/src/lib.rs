//! BrowerAI JavaScript Analyzer
//!
//! 7阶段分析管道：
//! 1. Scope Analysis - 作用域分析
//! 2. SWC Transformer - SWC转换器
//! 3. Dataflow Analysis - 数据流分析
//! 4. CFG - 控制流图
//! 5. CallGraph - 调用图
//! 6. Loop Analysis - 循环分析
//! 7. Unified Analysis - 统一分析层
//!
//! # 示例
//! ```
//! use browerai_js_analyzer::JsAnalyzer;
//!
//! let mut analyzer = JsAnalyzer::new();
//! let js = "function foo() { let x = 1; return x + 2; }";
//! let result = analyzer.analyze(js).unwrap();
//! ```

#![warn(missing_docs)]

use browerai_core::Result;
use browerai_js_parser::{JsParser, JsAst};

pub mod scope;
pub mod swc;
pub mod dataflow;
pub mod cfg;
pub mod callgraph;
pub mod loop_analysis;
pub mod unified;

pub use scope::{ScopeAnalyzer, ScopeTree, ScopeKind};
pub use cfg::{ControlFlowGraph, BasicBlock, BranchKind};
pub use callgraph::{CallGraph, CallSite, FunctionId};
pub use dataflow::{DataflowAnalyzer, VariableState};
pub use loop_analysis::{LoopAnalyzer, LoopInfo, LoopKind};
pub use unified::{UnifiedAnalysis, AnalysisSummary};

/// JavaScript 分析器 - 7阶段管道
pub struct JsAnalyzer {
    /// JS解析器
    parser: JsParser,
    /// 作用域分析器
    scope_analyzer: ScopeAnalyzer,
    /// CFG构建器
    cfg_builder: cfg::CfgBuilder,
    /// 调用图构建器
    callgraph_builder: callgraph::CallGraphBuilder,
    /// 数据流分析器
    dataflow_analyzer: DataflowAnalyzer,
    /// 循环分析器
    loop_analyzer: LoopAnalyzer,
    /// 统一分析器
    unified_analyzer: UnifiedAnalysis,
}

impl JsAnalyzer {
    /// 创建新的分析器
    pub fn new() -> Self {
        Self {
            parser: JsParser::new(),
            scope_analyzer: ScopeAnalyzer::new(),
            cfg_builder: cfg::CfgBuilder::new(),
            callgraph_builder: callgraph::CallGraphBuilder::new(),
            dataflow_analyzer: DataflowAnalyzer::new(),
            loop_analyzer: LoopAnalyzer::new(),
            unified_analyzer: UnifiedAnalysis::new(),
        }
    }

    /// 分析JavaScript代码（完整7阶段管道）
    pub fn analyze(&mut self, code: &str) -> Result<AnalysisResult> {
        // Stage 1: 解析
        let ast = self.parser.parse_string(code)?;
        
        // Stage 2: 作用域分析
        let scope_tree = self.scope_analyzer.analyze(&ast)?;
        
        // Stage 3: CFG构建
        let cfg = self.cfg_builder.build(&ast)?;
        
        // Stage 4: 调用图构建
        let callgraph = self.callgraph_builder.build(&ast)?;
        
        // Stage 5: 数据流分析
        let dataflow = self.dataflow_analyzer.analyze(&ast, &cfg)?;
        
        // Stage 6: 循环分析
        let loops = self.loop_analyzer.analyze(&ast, &cfg)?;
        
        // Stage 7: 统一分析
        let summary = self.unified_analyzer.summarize(&AnalysisInput {
            ast: &ast,
            scope_tree: &scope_tree,
            cfg: &cfg,
            callgraph: &callgraph,
            dataflow: &dataflow,
            loops: &loops,
        })?;
        
        Ok(AnalysisResult {
            ast,
            scope_tree,
            cfg,
            callgraph,
            dataflow,
            loops,
            summary,
        })
    }

    /// 快速分析（仅解析和作用域）
    pub fn analyze_quick(&mut self, code: &str) -> Result<QuickAnalysisResult> {
        let ast = self.parser.parse_string(code)?;
        let scope_tree = self.scope_analyzer.analyze(&ast)?;
        
        Ok(QuickAnalysisResult {
            function_count: ast.function_decls.len(),
            variable_count: ast.variable_decls.len(),
            scope_depth: scope_tree.max_depth(),
            global_variables: scope_tree.global_variables(),
        })
    }

    /// 获取作用域分析器
    pub fn scope_analyzer(&self) -> &ScopeAnalyzer {
        &self.scope_analyzer
    }

    /// 获取CFG构建器
    pub fn cfg_builder(&self) -> &cfg::CfgBuilder {
        &self.cfg_builder
    }

    /// 获取调用图构建器
    pub fn callgraph_builder(&self) -> &callgraph::CallGraphBuilder {
        &self.callgraph_builder
    }
}

impl Default for JsAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

// Note: Analyzer trait requires Send + Sync, but JsParser contains Interner which is not Send
// For now, we don't implement Analyzer trait directly. Use JsAnalyzer::analyze() instead.

/// 分析结果（完整）
#[derive(Debug, Clone)]
pub struct AnalysisResult {
    /// AST
    pub ast: JsAst,
    /// 作用域树
    pub scope_tree: ScopeTree,
    /// 控制流图
    pub cfg: ControlFlowGraph,
    /// 调用图
    pub callgraph: CallGraph,
    /// 数据流结果
    pub dataflow: dataflow::DataflowResult,
    /// 循环信息
    pub loops: Vec<LoopInfo>,
    /// 分析摘要
    pub summary: AnalysisSummary,
}

/// 快速分析结果
#[derive(Debug, Clone)]
pub struct QuickAnalysisResult {
    /// 函数数量
    pub function_count: usize,
    /// 变量数量
    pub variable_count: usize,
    /// 作用域深度
    pub scope_depth: usize,
    /// 全局变量
    pub global_variables: Vec<String>,
}

/// 统一分析输入
#[derive(Debug)]
pub struct AnalysisInput<'a> {
    /// AST
    pub ast: &'a JsAst,
    /// 作用域树
    pub scope_tree: &'a ScopeTree,
    /// CFG
    pub cfg: &'a ControlFlowGraph,
    /// 调用图
    pub callgraph: &'a CallGraph,
    /// 数据流结果
    pub dataflow: &'a dataflow::DataflowResult,
    /// 循环信息
    pub loops: &'a [LoopInfo],
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analyze_simple_js() {
        let mut analyzer = JsAnalyzer::new();
        let js = r#"
            function add(a, b) {
                return a + b;
            }
            let result = add(1, 2);
        "#;
        
        let result = analyzer.analyze(js).unwrap();
        
        assert_eq!(result.ast.function_decls.len(), 1);
        assert!(!result.scope_tree.is_empty());
        assert!(!result.cfg.is_empty());
    }

    #[test]
    fn test_analyze_quick() {
        let mut analyzer = JsAnalyzer::new();
        let js = "function test() { let x = 1; }";
        
        let result = analyzer.analyze_quick(js).unwrap();
        
        assert_eq!(result.function_count, 1);
    }

    #[test]
    fn test_complex_analysis() {
        let mut analyzer = JsAnalyzer::new();
        let js = r#"
            function factorial(n) {
                if (n <= 1) return 1;
                return n * factorial(n - 1);
            }
            
            for (let i = 0; i < 10; i++) {
                console.log(factorial(i));
            }
        "#;
        
        let result = analyzer.analyze(js).unwrap();
        
        assert_eq!(result.summary.metrics.function_count, 1);
    }
}
