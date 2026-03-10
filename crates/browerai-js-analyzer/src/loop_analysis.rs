//! Loop Analysis - 循环分析
//!
//! 分析JavaScript代码中的循环结构，包括：
//! - 循环类型检测 (for, while, do-while, for-in, for-of)
//! - 嵌套循环深度
//! - 循环复杂度
//! - 无限循环检测

use browerai_core::Result;
use browerai_js_parser::JsAst;
use crate::cfg::ControlFlowGraph;

/// 循环分析器
#[derive(Debug, Clone, Default)]
pub struct LoopAnalyzer {
    /// 最大嵌套深度
    max_nesting_depth: usize,
}

impl LoopAnalyzer {
    /// 创建新的循环分析器
    pub fn new() -> Self {
        Self {
            max_nesting_depth: 0,
        }
    }

    /// 分析循环
    pub fn analyze(&mut self, _ast: &JsAst, _cfg: &ControlFlowGraph) -> Result<Vec<LoopInfo>> {
        let loops = Vec::new();
        
        // 简化实现：基于AST中的函数声明分析循环
        // 实际实现需要遍历AST中的所有循环语句
        
        // 这里我们创建一个示例循环信息
        // 实际应该从AST中提取循环节点
        
        Ok(loops)
    }

    /// 快速分析 - 仅统计循环数量
    pub fn analyze_quick(&self, _ast: &JsAst) -> LoopQuickStats {
        // 简化实现
        LoopQuickStats {
            total_loops: 0,
            for_loops: 0,
            while_loops: 0,
            do_while_loops: 0,
            for_in_loops: 0,
            for_of_loops: 0,
            max_nesting_depth: 0,
            potentially_infinite: 0,
        }
    }

    /// 检测无限循环
    pub fn detect_infinite_loops(&self, _loops: &[LoopInfo]) -> Vec<usize> {
        // 简化实现
        // 实际应该分析循环条件和循环体内的控制流
        Vec::new()
    }

    /// 计算循环复杂度
    pub fn calculate_complexity(&self, loops: &[LoopInfo]) -> LoopComplexity {
        let total_loops = loops.len();
        let max_depth = loops.iter()
            .map(|l| l.nesting_depth)
            .max()
            .unwrap_or(0);
        
        // 计算嵌套循环数量
        let nested_loops = loops.iter()
            .filter(|l| l.nesting_depth > 0)
            .count();
        
        LoopComplexity {
            total_loops,
            max_nesting_depth: max_depth,
            nested_loops,
            score: self.compute_complexity_score(total_loops, max_depth, nested_loops),
        }
    }

    /// 计算复杂度分数
    fn compute_complexity_score(&self, total: usize, max_depth: usize, nested: usize) -> u32 {
        let base_score = total as u32 * 10;
        let nesting_penalty = (max_depth as u32).pow(2) * 5;
        let nested_bonus = nested as u32 * 15;
        
        base_score + nesting_penalty + nested_bonus
    }
}

/// 循环信息
#[derive(Debug, Clone)]
pub struct LoopInfo {
    /// 循环ID
    pub id: usize,
    /// 循环类型
    pub kind: LoopKind,
    /// 循环变量（如果有）
    pub loop_variable: Option<String>,
    /// 迭代对象（for-in/for-of）
    pub iterable: Option<String>,
    /// 起始行号
    pub start_line: Option<usize>,
    /// 结束行号
    pub end_line: Option<usize>,
    /// 嵌套深度
    pub nesting_depth: usize,
    /// 父循环ID
    pub parent_loop: Option<usize>,
    /// 是否有break语句
    pub has_break: bool,
    /// 是否有continue语句
    pub has_continue: bool,
    /// 是否有return语句
    pub has_return: bool,
    /// 条件复杂度（简单/复杂）
    pub condition_complexity: ConditionComplexity,
}

impl LoopInfo {
    /// 创建新的循环信息
    pub fn new(id: usize, kind: LoopKind) -> Self {
        Self {
            id,
            kind,
            loop_variable: None,
            iterable: None,
            start_line: None,
            end_line: None,
            nesting_depth: 0,
            parent_loop: None,
            has_break: false,
            has_continue: false,
            has_return: false,
            condition_complexity: ConditionComplexity::Simple,
        }
    }

    /// 是否是无限循环
    pub fn is_potentially_infinite(&self) -> bool {
        // 简化判断：没有break/return的while(true)或for(;;)
        match self.kind {
            LoopKind::While | LoopKind::For => {
                !self.has_break && !self.has_return
            }
            _ => false,
        }
    }

    /// 是否是嵌套循环
    pub fn is_nested(&self) -> bool {
        self.nesting_depth > 0
    }
}

/// 循环类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoopKind {
    /// for循环
    For,
    /// while循环
    While,
    /// do-while循环
    DoWhile,
    /// for-in循环
    ForIn,
    /// for-of循环
    ForOf,
}

impl std::fmt::Display for LoopKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoopKind::For => write!(f, "for"),
            LoopKind::While => write!(f, "while"),
            LoopKind::DoWhile => write!(f, "do-while"),
            LoopKind::ForIn => write!(f, "for-in"),
            LoopKind::ForOf => write!(f, "for-of"),
        }
    }
}

/// 条件复杂度
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConditionComplexity {
    /// 简单条件（单一比较）
    Simple,
    /// 中等复杂度（逻辑组合）
    Moderate,
    /// 复杂条件（函数调用、复杂表达式）
    Complex,
}

/// 循环快速统计
#[derive(Debug, Clone, Default)]
pub struct LoopQuickStats {
    /// 总循环数
    pub total_loops: usize,
    /// for循环数
    pub for_loops: usize,
    /// while循环数
    pub while_loops: usize,
    /// do-while循环数
    pub do_while_loops: usize,
    /// for-in循环数
    pub for_in_loops: usize,
    /// for-of循环数
    pub for_of_loops: usize,
    /// 最大嵌套深度
    pub max_nesting_depth: usize,
    /// 潜在无限循环数
    pub potentially_infinite: usize,
}

impl LoopQuickStats {
    /// 是否有循环
    pub fn has_loops(&self) -> bool {
        self.total_loops > 0
    }

    /// 是否有嵌套循环
    pub fn has_nested_loops(&self) -> bool {
        self.max_nesting_depth > 0
    }

    /// 获取循环密度（每函数平均循环数）
    pub fn loop_density(&self, function_count: usize) -> f64 {
        if function_count == 0 {
            0.0
        } else {
            self.total_loops as f64 / function_count as f64
        }
    }
}

/// 循环复杂度
#[derive(Debug, Clone)]
pub struct LoopComplexity {
    /// 总循环数
    pub total_loops: usize,
    /// 最大嵌套深度
    pub max_nesting_depth: usize,
    /// 嵌套循环数量
    pub nested_loops: usize,
    /// 复杂度分数
    pub score: u32,
}

impl LoopComplexity {
    /// 获取复杂度等级
    pub fn level(&self) -> ComplexityLevel {
        match self.score {
            0..=20 => ComplexityLevel::Low,
            21..=50 => ComplexityLevel::Medium,
            51..=100 => ComplexityLevel::High,
            _ => ComplexityLevel::VeryHigh,
        }
    }
}

/// 复杂度等级
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComplexityLevel {
    /// 低复杂度
    Low,
    /// 中等复杂度
    Medium,
    /// 高复杂度
    High,
    /// 非常高复杂度
    VeryHigh,
}

impl std::fmt::Display for ComplexityLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ComplexityLevel::Low => write!(f, "Low"),
            ComplexityLevel::Medium => write!(f, "Medium"),
            ComplexityLevel::High => write!(f, "High"),
            ComplexityLevel::VeryHigh => write!(f, "Very High"),
        }
    }
}

/// 循环优化建议
#[derive(Debug, Clone)]
pub struct LoopOptimization {
    /// 循环ID
    pub loop_id: usize,
    /// 建议类型
    pub suggestion_type: OptimizationType,
    /// 建议描述
    pub description: String,
    /// 潜在收益
    pub benefit: OptimizationBenefit,
}

/// 优化类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationType {
    /// 循环展开
    Unroll,
    /// 循环不变量外提
    InvariantHoisting,
    /// 强度削减
    StrengthReduction,
    /// 向量化
    Vectorization,
    /// 避免重复计算
    AvoidRecomputation,
}

/// 优化收益
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationBenefit {
    /// 低收益
    Low,
    /// 中等收益
    Medium,
    /// 高收益
    High,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_loop_analyzer_creation() {
        let mut analyzer = LoopAnalyzer::new();
        let ast = JsAst {
            function_decls: vec![],
            variable_decls: vec![],
            ..Default::default()
        };
        let cfg = ControlFlowGraph::new();
        
        let loops = analyzer.analyze(&ast, &cfg).unwrap();
        assert!(loops.is_empty());
    }

    #[test]
    fn test_loop_quick_stats() {
        let stats = LoopQuickStats {
            total_loops: 5,
            for_loops: 3,
            while_loops: 2,
            max_nesting_depth: 2,
            ..Default::default()
        };
        
        assert!(stats.has_loops());
        assert!(stats.has_nested_loops());
        assert_eq!(stats.loop_density(2), 2.5);
    }

    #[test]
    fn test_loop_complexity() {
        let complexity = LoopComplexity {
            total_loops: 10,
            max_nesting_depth: 3,
            nested_loops: 5,
            score: 75,
        };
        
        assert_eq!(complexity.level(), ComplexityLevel::High);
    }

    #[test]
    fn test_loop_info() {
        let mut loop_info = LoopInfo::new(0, LoopKind::For);
        loop_info.nesting_depth = 2;
        loop_info.has_break = true;
        
        assert!(loop_info.is_nested());
        assert!(!loop_info.is_potentially_infinite());
    }
}
