//! 数据流分析 (Data Flow Analysis)
//!
//! 追踪变量在程序中的定义和使用，识别数据依赖关系
//! 支持：def-use 链、变量追踪、污点分析、控制依赖

use anyhow::Result;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// 变量定义和使用
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct VarReference {
    pub variable: String,
    pub line: usize,
    pub is_definition: bool,
}

/// 数据流节点
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataFlowNode {
    pub id: usize,
    pub var_name: String,
    pub value: String,
    pub references: Vec<VarReference>,
    pub dependencies: Vec<usize>, // 依赖的其他节点
}

/// 污点分析信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaintInfo {
    pub variable: String,
    pub is_tainted: bool,
    pub taint_sources: Vec<String>,
    pub propagation_path: Vec<String>,
}

/// 数据流图
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DataFlowGraph {
    pub nodes: Vec<DataFlowNode>,
    pub edges: Vec<(usize, usize)>, // from_id -> to_id
    pub variable_map: HashMap<String, usize>, // var_name -> node_id
}

/// 数据流分析器
pub struct DataFlowAnalyzer {
    graph: DataFlowGraph,
    current_id: usize,
    taint_sources: HashSet<String>, // 已知的污染源
}

impl DataFlowAnalyzer {
    /// 创建新的数据流分析器
    pub fn new() -> Self {
        Self {
            graph: DataFlowGraph::default(),
            current_id: 0,
            taint_sources: [
                "eval",
                "document.write",
                "innerHTML",
                "appendChild",
                "insertAdjacentHTML",
                "fetch",
                "XMLHttpRequest",
                "location",
                "window.open",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect(),
        }
    }

    /// 分析代码的数据流
    pub fn analyze(&mut self, code: &str) -> Result<DataFlowAnalysisResult> {
        let mut result = DataFlowAnalysisResult::default();

        // 第一阶段：提取所有变量定义和使用
        self.extract_definitions_and_uses(code, &mut result)?;

        // 第二阶段：构建数据流图
        self.build_data_flow_graph(code, &mut result)?;

        // 第三阶段：污点分析
        self.taint_analysis(&mut result)?;

        // 第四阶段：识别关键变量
        self.identify_critical_variables(&mut result)?;

        Ok(result)
    }

    /// 提取定义和使用
    fn extract_definitions_and_uses(
        &mut self,
        code: &str,
        result: &mut DataFlowAnalysisResult,
    ) -> Result<()> {
        // 定义模式
        let def_patterns = vec![
            r"(?:var|let|const)\s+(\w+)\s*=",
            r"(\w+)\s*=\s*[^=]",  // 赋值（不是比较，简化版）
        ];

        for pattern in def_patterns {
            let re = Regex::new(pattern)?;
            for (line, code_line) in code.lines().enumerate() {
                for caps in re.captures_iter(code_line) {
                    if let Some(var) = caps.get(1) {
                        let var_name = var.as_str().to_string();
                        
                        result.definitions.push(VarReference {
                            variable: var_name.clone(),
                            line,
                            is_definition: true,
                        });

                        // 创建节点
                        self.create_node(var_name, code_line.to_string());
                    }
                }
            }
        }

        // 使用模式
        // 注意：移除了负后向断言，因为 Rust regex 不支持
        let use_patterns = vec![
            r"(\w+)\s*\+",      // 加法
            r"(\w+)\s*\-",      // 减法
            r"(\w+)\s*\*",      // 乘法
            r"(\w+)\s*\/",      // 除法
            r"\.replace\([^,]+,\s*(\w+)\)",   // 替换使用
            r"console\.log\(([^)]+)\)",       // 日志
        ];

        for pattern in use_patterns {
            let re = Regex::new(pattern)?;
            for (line, code_line) in code.lines().enumerate() {
                for caps in re.captures_iter(code_line) {
                    if let Some(var) = caps.get(1) {
                        let var_name = var.as_str().to_string();
                        
                        result.uses.push(VarReference {
                            variable: var_name,
                            line,
                            is_definition: false,
                        });
                    }
                }
            }
        }

        Ok(())
    }

    /// 构建数据流图
    fn build_data_flow_graph(
        &mut self,
        code: &str,
        result: &mut DataFlowAnalysisResult,
    ) -> Result<()> {
        // 遍历定义和使用，建立连接
        for def in &result.definitions {
            for use_ref in &result.uses {
                if def.variable == use_ref.variable && def.line < use_ref.line {
                    // 找到 def-use 链
                    if let (Some(def_id), Some(use_id)) = (
                        self.graph.variable_map.get(&def.variable),
                        self.graph.variable_map.get(&use_ref.variable),
                    ) {
                        self.graph.edges.push((*def_id, *use_id));

                        result.def_use_chains.push(DefUseChain {
                            variable: def.variable.clone(),
                            definition_line: def.line,
                            use_lines: vec![use_ref.line],
                        });
                    }
                }
            }
        }

        Ok(())
    }

    /// 污点分析
    fn taint_analysis(&self, result: &mut DataFlowAnalysisResult) -> Result<()> {
        // 识别污染源和污染传播
        for def in &result.definitions {
            let mut is_tainted = false;
            let mut taint_sources = Vec::new();

            for source in &self.taint_sources {
                if result
                    .definitions
                    .iter()
                    .any(|d| d.variable == def.variable && d.line == def.line)
                {
                    is_tainted = true;
                    taint_sources.push(source.clone());
                }
            }

            result.taints.push(TaintInfo {
                variable: def.variable.clone(),
                is_tainted,
                taint_sources,
                propagation_path: vec![],
            });
        }

        Ok(())
    }

    /// 识别关键变量
    fn identify_critical_variables(&self, result: &mut DataFlowAnalysisResult) -> Result<()> {
        // 关键变量：被多个地方使用，或被用于敏感操作
        let mut var_usage_count: HashMap<String, usize> = HashMap::new();

        for use_ref in &result.uses {
            *var_usage_count.entry(use_ref.variable.clone()).or_insert(0) += 1;
        }

        for (var, count) in var_usage_count {
            if count > 2 || self.taint_sources.iter().any(|s| var.contains(s)) {
                result.critical_variables.push(var);
            }
        }

        Ok(())
    }

    fn create_node(&mut self, var_name: String, value: String) {
        if !self.graph.variable_map.contains_key(&var_name) {
            let id = self.current_id;
            self.current_id += 1;

            self.graph.nodes.push(DataFlowNode {
                id,
                var_name: var_name.clone(),
                value,
                references: Vec::new(),
                dependencies: Vec::new(),
            });

            self.graph.variable_map.insert(var_name, id);
        }
    }
}

impl Default for DataFlowAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// 分析结果
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DataFlowAnalysisResult {
    pub definitions: Vec<VarReference>,
    pub uses: Vec<VarReference>,
    pub def_use_chains: Vec<DefUseChain>,
    pub taints: Vec<TaintInfo>,
    pub critical_variables: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DefUseChain {
    pub variable: String,
    pub definition_line: usize,
    pub use_lines: Vec<usize>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_definition_extraction() {
        let code = r#"
            var x = 10;
            let y = "hello";
            const z = x + 5;
        "#;

        let mut analyzer = DataFlowAnalyzer::new();
        let result = analyzer.analyze(code).unwrap();

        assert!(result.definitions.iter().any(|d| d.variable == "x"));
        assert!(result.definitions.iter().any(|d| d.variable == "y"));
        assert!(result.definitions.iter().any(|d| d.variable == "z"));
    }

    #[test]
    fn test_def_use_chain() {
        let code = r#"
            var x = 10;
            var y = x + 5;
            console.log(x);
        "#;

        let mut analyzer = DataFlowAnalyzer::new();
        let result = analyzer.analyze(code).unwrap();

        assert!(!result.def_use_chains.is_empty());
    }

    #[test]
    fn test_critical_variable_identification() {
        let code = r#"
            var sensitive = document.write("test");
            var other = 42;
            console.log(sensitive);
            console.log(sensitive);
            console.log(sensitive);
        "#;

        let mut analyzer = DataFlowAnalyzer::new();
        let result = analyzer.analyze(code).unwrap();

        // sensitive 应该被识别为关键变量（被使用多次）
        assert!(result.critical_variables.contains(&"sensitive".to_string()));
    }
}
