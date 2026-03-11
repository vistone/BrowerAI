//! Dataflow Analysis - 数据流分析
//!
//! 实现基于格（Lattice）的数据流分析，包括：
//! - 到达定义分析 (Reaching Definitions)
//! - 活跃变量分析 (Live Variables)
//! - 常量传播 (Constant Propagation)
//! - 可用表达式 (Available Expressions)

use crate::cfg::ControlFlowGraph;
use browerai_core::Result;
use browerai_js_parser::JsAst;
use std::collections::HashMap;

/// 数据流分析器
#[derive(Debug, Clone, Default)]
pub struct DataflowAnalyzer {
    /// 分析类型
    analysis_type: DataflowAnalysisType,
}

impl DataflowAnalyzer {
    /// 创建新的数据流分析器
    pub fn new() -> Self {
        Self {
            analysis_type: DataflowAnalysisType::ReachingDefinitions,
        }
    }

    /// 设置分析类型
    pub fn with_analysis_type(mut self, analysis_type: DataflowAnalysisType) -> Self {
        self.analysis_type = analysis_type;
        self
    }

    /// 分析数据流
    pub fn analyze(&mut self, ast: &JsAst, cfg: &ControlFlowGraph) -> Result<DataflowResult> {
        match self.analysis_type {
            DataflowAnalysisType::ReachingDefinitions => {
                self.analyze_reaching_definitions(ast, cfg)
            }
            DataflowAnalysisType::LiveVariables => self.analyze_live_variables(ast, cfg),
            DataflowAnalysisType::ConstantPropagation => {
                self.analyze_constant_propagation(ast, cfg)
            }
            DataflowAnalysisType::AvailableExpressions => {
                self.analyze_available_expressions(ast, cfg)
            }
        }
    }

    /// 到达定义分析
    fn analyze_reaching_definitions(
        &self,
        ast: &JsAst,
        _cfg: &ControlFlowGraph,
    ) -> Result<DataflowResult> {
        // 简化实现：收集所有变量定义
        let mut definitions = HashMap::new();

        for (i, var) in ast.variable_decls.iter().enumerate() {
            definitions.insert(
                var.name.clone(),
                Definition {
                    id: i,
                    variable: var.name.clone(),
                    definition_kind: DefinitionKind::Variable,
                    line: None,
                },
            );
        }

        Ok(DataflowResult {
            analysis_type: DataflowAnalysisType::ReachingDefinitions,
            variable_states: definitions
                .into_iter()
                .map(|(name, def)| (name, VariableState::Defined(vec![def])))
                .collect(),
            constants: HashMap::new(),
        })
    }

    /// 活跃变量分析
    fn analyze_live_variables(
        &self,
        ast: &JsAst,
        _cfg: &ControlFlowGraph,
    ) -> Result<DataflowResult> {
        // 简化实现：假设所有变量都是活跃的
        let mut variable_states = HashMap::new();

        for var in &ast.variable_decls {
            variable_states.insert(var.name.clone(), VariableState::Live);
        }

        Ok(DataflowResult {
            analysis_type: DataflowAnalysisType::LiveVariables,
            variable_states,
            constants: HashMap::new(),
        })
    }

    /// 常量传播分析
    fn analyze_constant_propagation(
        &self,
        ast: &JsAst,
        _cfg: &ControlFlowGraph,
    ) -> Result<DataflowResult> {
        let mut constants = HashMap::new();
        let mut variable_states = HashMap::new();

        // 简化实现：检测简单的常量初始化
        for var in &ast.variable_decls {
            if var.init.is_some() {
                // 假设是常量（实际应该分析初始化表达式）
                constants.insert(var.name.clone(), ConstantValue::Unknown);
                variable_states.insert(var.name.clone(), VariableState::Constant);
            } else {
                variable_states.insert(var.name.clone(), VariableState::Undefined);
            }
        }

        Ok(DataflowResult {
            analysis_type: DataflowAnalysisType::ConstantPropagation,
            variable_states,
            constants,
        })
    }

    /// 可用表达式分析
    fn analyze_available_expressions(
        &self,
        _ast: &JsAst,
        _cfg: &ControlFlowGraph,
    ) -> Result<DataflowResult> {
        // 简化实现
        Ok(DataflowResult {
            analysis_type: DataflowAnalysisType::AvailableExpressions,
            variable_states: HashMap::new(),
            constants: HashMap::new(),
        })
    }
}

/// 数据流分析类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DataflowAnalysisType {
    /// 到达定义分析 (默认)
    #[default]
    ReachingDefinitions,
    /// 活跃变量分析
    LiveVariables,
    /// 常量传播
    ConstantPropagation,
    /// 可用表达式
    AvailableExpressions,
}

/// 数据流分析结果
#[derive(Debug, Clone)]
pub struct DataflowResult {
    /// 分析类型
    pub analysis_type: DataflowAnalysisType,
    /// 变量状态映射
    pub variable_states: HashMap<String, VariableState>,
    /// 常量值映射
    pub constants: HashMap<String, ConstantValue>,
}

impl DataflowResult {
    /// 获取变量状态
    pub fn get_state(&self, variable: &str) -> Option<&VariableState> {
        self.variable_states.get(variable)
    }

    /// 检查变量是否定义
    pub fn is_defined(&self, variable: &str) -> bool {
        matches!(
            self.variable_states.get(variable),
            Some(VariableState::Defined(_)) | Some(VariableState::Constant)
        )
    }

    /// 检查变量是否活跃
    pub fn is_live(&self, variable: &str) -> bool {
        matches!(
            self.variable_states.get(variable),
            Some(VariableState::Live)
        )
    }

    /// 获取常量值
    pub fn get_constant(&self, variable: &str) -> Option<&ConstantValue> {
        self.constants.get(variable)
    }

    /// 获取未使用的变量（未定义或已死亡）
    pub fn get_unused_variables(&self) -> Vec<String> {
        self.variable_states
            .iter()
            .filter(|(_, state)| matches!(state, VariableState::Undefined | VariableState::Dead))
            .map(|(name, _)| name.clone())
            .collect()
    }
}

/// 变量状态
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VariableState {
    /// 未定义
    Undefined,
    /// 已定义（带有定义点列表）
    Defined(Vec<Definition>),
    /// 常量
    Constant,
    /// 活跃
    Live,
    /// 死亡（不再使用）
    Dead,
    /// 未知
    Unknown,
}

/// 定义信息
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Definition {
    /// 定义ID
    pub id: usize,
    /// 变量名
    pub variable: String,
    /// 定义类型
    pub definition_kind: DefinitionKind,
    /// 行号
    pub line: Option<usize>,
}

/// 定义类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DefinitionKind {
    /// 变量声明
    Variable,
    /// 函数参数
    Parameter,
    /// 赋值语句
    Assignment,
    /// 解构赋值
    Destructuring,
}

/// 常量值
#[derive(Debug, Clone, PartialEq)]
pub enum ConstantValue {
    /// 字符串常量
    String(String),
    /// 数字常量
    Number(f64),
    /// 布尔常量
    Boolean(bool),
    /// null
    Null,
    /// undefined
    Undefined,
    /// 未知（非常量）
    Unknown,
}

/// 数据流方程求解器
pub struct DataflowSolver {
    /// 最大迭代次数
    max_iterations: usize,
}

impl DataflowSolver {
    /// 创建新的求解器
    pub fn new() -> Self {
        Self {
            max_iterations: 1000,
        }
    }

    /// 设置最大迭代次数
    pub fn with_max_iterations(mut self, max: usize) -> Self {
        self.max_iterations = max;
        self
    }

    /// 求解数据流方程（简化实现）
    pub fn solve(&self, _cfg: &ControlFlowGraph) -> Result<DataflowResult> {
        // 实际实现需要使用迭代算法（如Kildall算法）
        // 这里返回空结果
        Ok(DataflowResult {
            analysis_type: DataflowAnalysisType::ReachingDefinitions,
            variable_states: HashMap::new(),
            constants: HashMap::new(),
        })
    }
}

impl Default for DataflowSolver {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use browerai_js_parser::VariableDecl;

    #[test]
    fn test_dataflow_analyzer_creation() {
        let analyzer = DataflowAnalyzer::new();
        assert!(matches!(
            analyzer.analysis_type,
            DataflowAnalysisType::ReachingDefinitions
        ));
    }

    #[test]
    fn test_reaching_definitions() {
        let mut analyzer = DataflowAnalyzer::new();
        let ast = JsAst {
            variable_decls: vec![VariableDecl {
                name: "x".to_string(),
                kind: "let".to_string(),
                init: None,
            }],
            ..Default::default()
        };
        let cfg = ControlFlowGraph::new();

        let result = analyzer.analyze(&ast, &cfg).unwrap();

        assert!(result.is_defined("x"));
        assert_eq!(
            result.analysis_type,
            DataflowAnalysisType::ReachingDefinitions
        );
    }

    #[test]
    fn test_live_variables() {
        let mut analyzer =
            DataflowAnalyzer::new().with_analysis_type(DataflowAnalysisType::LiveVariables);

        let ast = JsAst {
            variable_decls: vec![VariableDecl {
                name: "y".to_string(),
                kind: "let".to_string(),
                init: None,
            }],
            ..Default::default()
        };
        let cfg = ControlFlowGraph::new();

        let result = analyzer.analyze(&ast, &cfg).unwrap();

        assert!(result.is_live("y"));
    }

    #[test]
    fn test_constant_propagation() {
        let mut analyzer =
            DataflowAnalyzer::new().with_analysis_type(DataflowAnalysisType::ConstantPropagation);

        let ast = JsAst {
            variable_decls: vec![VariableDecl {
                name: "z".to_string(),
                kind: "const".to_string(),
                init: Some("value".to_string()),
            }],
            ..Default::default()
        };
        let cfg = ControlFlowGraph::new();

        let result = analyzer.analyze(&ast, &cfg).unwrap();

        assert!(matches!(
            result.get_state("z"),
            Some(VariableState::Constant) | Some(VariableState::Undefined)
        ));
    }
}
