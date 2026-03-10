//! Unified Analysis - 统一分析层
//!
//! 整合所有分析阶段的结果，提供：
//! - 综合分析摘要
//! - 代码质量评估
//! - 性能瓶颈识别
//! - 优化建议生成

use browerai_core::Result;
use crate::{
    AnalysisInput,
    cfg::ControlFlowGraph,
    dataflow::DataflowResult,
};

/// 统一分析器
#[derive(Debug, Clone, Default)]
pub struct UnifiedAnalysis;

impl UnifiedAnalysis {
    /// 创建新的统一分析器
    pub fn new() -> Self {
        Self
    }

    /// 生成分析摘要
    pub fn summarize(&self, input: &AnalysisInput) -> Result<AnalysisSummary> {
        // 计算各项指标
        let cyclomatic_complexity = self.calculate_cyclomatic_complexity(input.cfg);
        let cognitive_complexity = self.calculate_cognitive_complexity(input);
        let maintainability_index = self.calculate_maintainability_index(input);
        
        // 识别问题
        let issues = self.identify_issues(input);
        
        // 生成优化建议
        let optimizations = self.generate_optimizations(input);
        
        // 评估安全风险
        let security_risks = self.assess_security_risks(input);
        
        // 性能分析
        let performance = self.analyze_performance(input);
        
        Ok(AnalysisSummary {
            metrics: CodeMetrics {
                function_count: input.ast.function_decls.len(),
                variable_count: input.ast.variable_decls.len(),
                scope_depth: input.scope_tree.max_depth(),
                cyclomatic_complexity,
                cognitive_complexity,
                maintainability_index,
            },
            issues,
            optimizations,
            security_risks,
            performance,
        })
    }

    /// 计算圈复杂度 (Cyclomatic Complexity)
    fn calculate_cyclomatic_complexity(&self, cfg: &ControlFlowGraph) -> u32 {
        // V(G) = E - N + 2P
        // E = 边数, N = 节点数, P = 连通分量数
        let edges = cfg.edge_count() as u32;
        let nodes = cfg.node_count() as u32;
        let components = 1; // 简化：假设只有一个连通分量
        
        edges.saturating_sub(nodes).saturating_add(2 * components)
    }

    /// 计算认知复杂度 (Cognitive Complexity)
    fn calculate_cognitive_complexity(&self, input: &AnalysisInput) -> u32 {
        let mut complexity = 0;
        
        // 嵌套深度贡献
        complexity += input.scope_tree.max_depth() as u32 * 2;
        
        // 循环贡献
        complexity += input.loops.len() as u32 * 3;
        
        // 递归贡献
        if input.callgraph.has_recursive_calls() {
            complexity += 5;
        }
        
        // 全局变量使用
        let global_vars = input.scope_tree.global_variables().len() as u32;
        complexity += global_vars;
        
        complexity
    }

    /// 计算可维护性指数 (Maintainability Index)
    fn calculate_maintainability_index(&self, input: &AnalysisInput) -> f64 {
        // 简化版MI计算
        // MI = 171 - 5.2 * ln(Halstead Volume) - 0.23 * CC - 16.2 * ln(Lines of Code)
        
        let cc = self.calculate_cyclomatic_complexity(input.cfg) as f64;
        let loc = input.ast.statement_count() as f64 + 1.0;
        
        // 简化的Halstead Volume估计
        let halstead_volume = (input.ast.function_decls.len() + input.ast.variable_decls.len()) as f64 * 10.0;
        
        let mi = 171.0 - 5.2 * halstead_volume.ln() - 0.23 * cc - 16.2 * loc.ln();
        
        mi.max(0.0).min(100.0)
    }

    /// 识别代码问题
    fn identify_issues(&self, input: &AnalysisInput) -> Vec<CodeIssue> {
        let mut issues = Vec::new();
        
        // 检查过深的嵌套
        let max_depth = input.scope_tree.max_depth();
        if max_depth > 5 {
            issues.push(CodeIssue {
                severity: IssueSeverity::Warning,
                category: IssueCategory::Complexity,
                message: format!("Deep nesting detected: {} levels", max_depth),
                suggestion: "Consider refactoring nested functions".to_string(),
            });
        }

        // 检查过多全局变量
        let global_vars = input.scope_tree.global_variables();
        if global_vars.len() > 10 {
            issues.push(CodeIssue {
                severity: IssueSeverity::Warning,
                category: IssueCategory::Maintainability,
                message: format!("Too many global variables: {}", global_vars.len()),
                suggestion: "Consider encapsulating globals in modules".to_string(),
            });
        }

        // 检查递归调用
        if input.callgraph.has_recursive_calls() {
            let recursive_funcs = input.callgraph.get_recursive_functions();
            for func in recursive_funcs {
                issues.push(CodeIssue {
                    severity: IssueSeverity::Info,
                    category: IssueCategory::Performance,
                    message: format!("Recursive function: {}", func.name),
                    suggestion: "Consider iterative approach for large inputs".to_string(),
                });
            }
        }

        // 检查未使用变量
        let unused = input.dataflow.get_unused_variables();
        for var in unused {
            issues.push(CodeIssue {
                severity: IssueSeverity::Info,
                category: IssueCategory::Maintainability,
                message: format!("Potentially unused variable: {}", var),
                suggestion: "Remove if not needed".to_string(),
            });
        }

        issues
    }

    /// 生成优化建议
    fn generate_optimizations(&self, input: &AnalysisInput) -> Vec<OptimizationSuggestion> {
        let mut suggestions = Vec::new();

        // 建议1: 减少嵌套
        if input.scope_tree.max_depth() > 3 {
            suggestions.push(OptimizationSuggestion {
                priority: Priority::Medium,
                category: OptimizationCategory::Readability,
                description: "Reduce function nesting depth".to_string(),
                expected_impact: "Improved code readability and testability".to_string(),
            });
        }

        // 建议2: 模块化
        if input.ast.function_decls.len() > 20 {
            suggestions.push(OptimizationSuggestion {
                priority: Priority::High,
                category: OptimizationCategory::Architecture,
                description: "Split large module into smaller modules".to_string(),
                expected_impact: "Better code organization and maintainability".to_string(),
            });
        }

        // 建议3: 循环优化
        if !input.loops.is_empty() {
            suggestions.push(OptimizationSuggestion {
                priority: Priority::Low,
                category: OptimizationCategory::Performance,
                description: "Review loops for optimization opportunities".to_string(),
                expected_impact: "Potential performance improvement".to_string(),
            });
        }

        // 建议4: 消除重复代码
        let orphan_count = input.callgraph.get_orphan_functions().len();
        if orphan_count > 5 {
            suggestions.push(OptimizationSuggestion {
                priority: Priority::Medium,
                category: OptimizationCategory::Maintainability,
                description: "Review unused functions".to_string(),
                expected_impact: "Reduced code size".to_string(),
            });
        }

        suggestions
    }

    /// 评估安全风险
    fn assess_security_risks(&self, input: &AnalysisInput) -> Vec<SecurityRisk> {
        let mut risks = Vec::new();

        // 检查eval使用（简化检测）
        let has_eval = input.scope_tree.global_variables()
            .iter()
            .any(|v| v == "eval");
        
        if has_eval {
            risks.push(SecurityRisk {
                level: RiskLevel::High,
                category: SecurityCategory::CodeInjection,
                description: "Potential use of eval() detected".to_string(),
                mitigation: "Avoid using eval(); use safer alternatives".to_string(),
            });
        }

        // 检查全局变量污染
        let global_count = input.scope_tree.global_variables().len();
        if global_count > 20 {
            risks.push(SecurityRisk {
                level: RiskLevel::Medium,
                category: SecurityCategory::NamespacePollution,
                description: "Excessive global variables".to_string(),
                mitigation: "Encapsulate code in modules or IIFE".to_string(),
            });
        }

        risks
    }

    /// 性能分析
    fn analyze_performance(&self, input: &AnalysisInput) -> PerformanceAnalysis {
        PerformanceAnalysis {
            estimated_time_complexity: self.estimate_time_complexity(input),
            estimated_space_complexity: self.estimate_space_complexity(input),
            bottleneck_functions: self.identify_bottlenecks(input),
        }
    }

    /// 估计时间复杂度
    fn estimate_time_complexity(&self, input: &AnalysisInput) -> String {
        let loop_count = input.loops.len();
        let max_nesting = input.loops.iter()
            .map(|l| l.nesting_depth)
            .max()
            .unwrap_or(0);

        match (loop_count, max_nesting) {
            (0, _) => "O(1)".to_string(),
            (1..=3, 0) => "O(n)".to_string(),
            (_, 1) => "O(n)".to_string(),
            (_, 2) => "O(n²)".to_string(),
            (_, 3) => "O(n³)".to_string(),
            _ => "O(n^?)".to_string(),
        }
    }

    /// 估计空间复杂度
    fn estimate_space_complexity(&self, input: &AnalysisInput) -> String {
        let var_count = input.ast.variable_decls.len();
        
        if var_count < 10 {
            "O(1)".to_string()
        } else if var_count < 100 {
            "O(n)".to_string()
        } else {
            "O(n) or higher".to_string()
        }
    }

    /// 识别性能瓶颈
    fn identify_bottlenecks(&self, input: &AnalysisInput) -> Vec<String> {
        let mut bottlenecks = Vec::new();

        // 递归函数可能是瓶颈
        if input.callgraph.has_recursive_calls() {
            bottlenecks.push("Recursive functions may cause stack overflow".to_string());
        }

        // 深层嵌套循环
        let deep_loops: Vec<_> = input.loops.iter()
            .filter(|l| l.nesting_depth >= 2)
            .collect();
        
        if !deep_loops.is_empty() {
            bottlenecks.push(format!(
                "{} deeply nested loops detected",
                deep_loops.len()
            ));
        }

        bottlenecks
    }
}

/// 分析摘要
#[derive(Debug, Clone)]
pub struct AnalysisSummary {
    /// 代码指标
    pub metrics: CodeMetrics,
    /// 代码问题
    pub issues: Vec<CodeIssue>,
    /// 优化建议
    pub optimizations: Vec<OptimizationSuggestion>,
    /// 安全风险
    pub security_risks: Vec<SecurityRisk>,
    /// 性能分析
    pub performance: PerformanceAnalysis,
}

/// 代码指标
#[derive(Debug, Clone)]
pub struct CodeMetrics {
    /// 函数数量
    pub function_count: usize,
    /// 变量数量
    pub variable_count: usize,
    /// 作用域深度
    pub scope_depth: usize,
    /// 圈复杂度
    pub cyclomatic_complexity: u32,
    /// 认知复杂度
    pub cognitive_complexity: u32,
    /// 可维护性指数 (0-100)
    pub maintainability_index: f64,
}

impl CodeMetrics {
    /// 获取可维护性等级
    pub fn maintainability_level(&self) -> MaintainabilityLevel {
        match self.maintainability_index {
            0.0..=20.0 => MaintainabilityLevel::VeryLow,
            20.1..=40.0 => MaintainabilityLevel::Low,
            40.1..=60.0 => MaintainabilityLevel::Moderate,
            60.1..=80.0 => MaintainabilityLevel::Good,
            _ => MaintainabilityLevel::Excellent,
        }
    }
}

/// 可维护性等级
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaintainabilityLevel {
    VeryLow,
    Low,
    Moderate,
    Good,
    Excellent,
}

/// 代码问题
#[derive(Debug, Clone)]
pub struct CodeIssue {
    /// 严重程度
    pub severity: IssueSeverity,
    /// 问题类别
    pub category: IssueCategory,
    /// 问题描述
    pub message: String,
    /// 改进建议
    pub suggestion: String,
}

/// 问题严重程度
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum IssueSeverity {
    Error,
    Warning,
    Info,
}

/// 问题类别
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IssueCategory {
    Complexity,
    Maintainability,
    Performance,
    Security,
    Style,
}

/// 优化建议
#[derive(Debug, Clone)]
pub struct OptimizationSuggestion {
    /// 优先级
    pub priority: Priority,
    /// 优化类别
    pub category: OptimizationCategory,
    /// 描述
    pub description: String,
    /// 预期效果
    pub expected_impact: String,
}

/// 优先级
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Priority {
    Low,
    Medium,
    High,
    Critical,
}

/// 优化类别
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationCategory {
    Performance,
    Readability,
    Maintainability,
    Architecture,
}

/// 安全风险
#[derive(Debug, Clone)]
pub struct SecurityRisk {
    /// 风险等级
    pub level: RiskLevel,
    /// 安全类别
    pub category: SecurityCategory,
    /// 描述
    pub description: String,
    /// 缓解措施
    pub mitigation: String,
}

/// 风险等级
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

/// 安全类别
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SecurityCategory {
    CodeInjection,
    DataExposure,
    NamespacePollution,
    PrototypePollution,
}

/// 性能分析
#[derive(Debug, Clone)]
pub struct PerformanceAnalysis {
    /// 估计时间复杂度
    pub estimated_time_complexity: String,
    /// 估计空间复杂度
    pub estimated_space_complexity: String,
    /// 瓶颈函数
    pub bottleneck_functions: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use browerai_js_parser::JsAst;

    #[test]
    fn test_unified_analysis_creation() {
        let analyzer = UnifiedAnalysis::new();
        assert!(analyzer.summarize(&AnalysisInput {
            ast: &JsAst::default(),
            scope_tree: &ScopeTree::default(),
            cfg: &ControlFlowGraph::new(),
            callgraph: &CallGraph::new(),
            dataflow: &DataflowResult::default(),
            loops: &[],
        }).is_ok());
    }

    #[test]
    fn test_maintainability_level() {
        let metrics = CodeMetrics {
            function_count: 10,
            variable_count: 20,
            scope_depth: 3,
            cyclomatic_complexity: 10,
            cognitive_complexity: 15,
            maintainability_index: 75.0,
        };
        
        assert_eq!(metrics.maintainability_level(), MaintainabilityLevel::Good);
    }

    #[test]
    fn test_issue_severity_ordering() {
        // Error is most severe, so it should be "less" in ordering (if we think of it as priority)
        // Or we can just check they are different
        assert_ne!(IssueSeverity::Error, IssueSeverity::Warning);
        assert_ne!(IssueSeverity::Warning, IssueSeverity::Info);
        assert!(IssueSeverity::Error < IssueSeverity::Warning); // Error has higher priority
    }
}

impl Default for DataflowResult {
    fn default() -> Self {
        use crate::dataflow::DataflowAnalysisType;
        Self {
            analysis_type: DataflowAnalysisType::ReachingDefinitions,
            variable_states: std::collections::HashMap::new(),
            constants: std::collections::HashMap::new(),
        }
    }
}
