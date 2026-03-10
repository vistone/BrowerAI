//! Call Graph - 调用图
//!
//! 分析JavaScript代码中的函数调用关系，包括：
//! - 函数调用点 (CallSite)
//! - 调用者-被调用者关系
//! - 递归调用检测
//! - 调用深度分析

use browerai_core::Result;
use browerai_js_parser::JsAst;
use petgraph::graph::{DiGraph, NodeIndex};
use std::collections::HashMap;

/// 调用图
#[derive(Debug, Clone)]
pub struct CallGraph {
    /// 底层图结构
    graph: DiGraph<FunctionNode, CallEdge>,
    /// 函数名到节点索引的映射
    function_map: HashMap<String, NodeIndex>,
    /// 调用点列表
    call_sites: Vec<CallSite>,
}

impl CallGraph {
    /// 创建新的调用图
    pub fn new() -> Self {
        Self {
            graph: DiGraph::new(),
            function_map: HashMap::new(),
            call_sites: Vec::new(),
        }
    }

    /// 添加函数节点
    pub fn add_function(&mut self, func: FunctionNode) -> NodeIndex {
        let name = func.name.clone();
        let idx = self.graph.add_node(func);
        
        if !name.is_empty() {
            self.function_map.insert(name, idx);
        }
        
        idx
    }

    /// 添加调用边
    pub fn add_call(&mut self, from: NodeIndex, to: NodeIndex, call_site: CallSite) {
        let edge = CallEdge {
            call_site_id: self.call_sites.len(),
        };
        self.call_sites.push(call_site);
        self.graph.add_edge(from, to, edge);
    }

    /// 通过函数名查找节点
    pub fn find_function(&self, name: &str) -> Option<NodeIndex> {
        self.function_map.get(name).copied()
    }

    /// 获取函数节点
    pub fn get_function(&self, idx: NodeIndex) -> Option<&FunctionNode> {
        self.graph.node_weight(idx)
    }

    /// 获取调用者
    pub fn callers(&self, func_idx: NodeIndex) -> Vec<NodeIndex> {
        self.graph.neighbors_directed(func_idx, petgraph::Direction::Incoming).collect()
    }

    /// 获取被调用者
    pub fn callees(&self, func_idx: NodeIndex) -> Vec<NodeIndex> {
        self.graph.neighbors(func_idx).collect()
    }

    /// 检测递归调用
    pub fn has_recursive_calls(&self) -> bool {
        // 检查是否存在自环（直接递归）
        for node in self.graph.node_indices() {
            if self.graph.contains_edge(node, node) {
                return true;
            }
        }
        
        // 检查是否存在循环（间接递归）
        use petgraph::algo::is_cyclic_directed;
        is_cyclic_directed(&self.graph)
    }

    /// 获取递归函数
    pub fn get_recursive_functions(&self) -> Vec<&FunctionNode> {
        let mut recursive = Vec::new();
        
        for node in self.graph.node_indices() {
            // 直接递归
            if self.graph.contains_edge(node, node) {
                if let Some(func) = self.graph.node_weight(node) {
                    recursive.push(func);
                }
                continue;
            }
            
            // 间接递归 - 检查是否有路径回到自身
            use petgraph::algo::has_path_connecting;
            if has_path_connecting(&self.graph, node, node, None) {
                if let Some(func) = self.graph.node_weight(node) {
                    recursive.push(func);
                }
            }
        }
        
        recursive
    }

    /// 计算最大调用深度（简化实现）
    pub fn max_call_depth(&self) -> usize {
        // 简化：返回图的直径估计
        // 实际实现需要使用最长路径算法
        let node_count = self.graph.node_count();
        if node_count == 0 {
            0
        } else {
            // 估计值：假设调用链深度不超过节点数
            (node_count as f64).sqrt() as usize + 1
        }
    }

    /// 获取孤立函数（没有被调用的函数）
    pub fn get_orphan_functions(&self) -> Vec<&FunctionNode> {
        let mut orphans = Vec::new();
        
        for node in self.graph.node_indices() {
            let has_callers = self.callers(node).is_empty();
            let has_callees = self.callees(node).is_empty();
            
            // 没有被调用且没有调用其他函数（除了可能是入口函数）
            if has_callers && has_callees {
                if let Some(func) = self.graph.node_weight(node) {
                    orphans.push(func);
                }
            }
        }
        
        orphans
    }

    /// 获取入口函数（没有被其他函数调用的函数）
    pub fn get_entry_functions(&self) -> Vec<&FunctionNode> {
        let mut entries = Vec::new();
        
        for node in self.graph.node_indices() {
            if self.callers(node).is_empty() {
                if let Some(func) = self.graph.node_weight(node) {
                    entries.push(func);
                }
            }
        }
        
        entries
    }

    /// 函数数量
    pub fn function_count(&self) -> usize {
        self.graph.node_count()
    }

    /// 调用边数量
    pub fn call_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// 是否为空
    pub fn is_empty(&self) -> bool {
        self.graph.node_count() == 0
    }
}

impl Default for CallGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// 函数节点
#[derive(Debug, Clone)]
pub struct FunctionNode {
    /// 函数ID
    pub id: FunctionId,
    /// 函数名
    pub name: String,
    /// 参数数量
    pub param_count: usize,
    /// 是否是异步函数
    pub is_async: bool,
    /// 是否是生成器函数
    pub is_generator: bool,
    /// 所在行号
    pub line: Option<usize>,
    /// 所在列号
    pub column: Option<usize>,
}

impl FunctionNode {
    /// 创建新函数节点
    pub fn new(id: FunctionId, name: impl Into<String>) -> Self {
        Self {
            id,
            name: name.into(),
            param_count: 0,
            is_async: false,
            is_generator: false,
            line: None,
            column: None,
        }
    }
}

/// 函数ID
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FunctionId(pub usize);

/// 调用边
#[derive(Debug, Clone)]
pub struct CallEdge {
    /// 调用点ID
    call_site_id: usize,
}

/// 调用点
#[derive(Debug, Clone)]
pub struct CallSite {
    /// 调用点ID
    pub id: usize,
    /// 调用者函数ID
    pub caller: FunctionId,
    /// 被调用函数名（可能是动态的）
    pub callee_name: String,
    /// 参数数量
    pub arg_count: usize,
    /// 所在行号
    pub line: Option<usize>,
    /// 所在列号
    pub column: Option<usize>,
}

/// 调用图构建器
#[derive(Debug, Clone, Default)]
pub struct CallGraphBuilder {
    /// 下一个函数ID
    next_func_id: usize,
}

impl CallGraphBuilder {
    /// 创建新的调用图构建器
    pub fn new() -> Self {
        Self {
            next_func_id: 0,
        }
    }

    /// 从AST构建调用图
    pub fn build(&mut self, ast: &JsAst) -> Result<CallGraph> {
        let mut callgraph = CallGraph::new();
        
        // 添加所有函数节点
        for func in &ast.function_decls {
            let id = FunctionId(self.next_func_id);
            self.next_func_id += 1;
            
            let name = func.name.clone().unwrap_or_default();
            let mut node = FunctionNode::new(id, name);
            node.param_count = func.params.len();
            node.is_async = func.is_async;
            node.is_generator = func.is_generator;
            
            callgraph.add_function(node);
        }
        
        // 简化实现：假设所有函数都可能相互调用
        // 实际实现需要分析函数体内的调用表达式
        let func_indices: Vec<_> = callgraph.function_map.values().copied().collect();
        
        for (i, &caller) in func_indices.iter().enumerate() {
            for (j, &callee) in func_indices.iter().enumerate() {
                if i != j {
                    // 创建调用点
                    let call_site = CallSite {
                        id: callgraph.call_sites.len(),
                        caller: callgraph.get_function(caller)
                            .map(|f| f.id)
                            .unwrap_or(FunctionId(0)),
                        callee_name: callgraph.get_function(callee)
                            .map(|f| f.name.clone())
                            .unwrap_or_default(),
                        arg_count: 0,
                        line: None,
                        column: None,
                    };
                    
                    callgraph.add_call(caller, callee, call_site);
                }
            }
        }
        
        Ok(callgraph)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use browerai_js_parser::FunctionDecl;

    #[test]
    fn test_callgraph_creation() {
        let cg = CallGraph::new();
        assert!(cg.is_empty());
        assert_eq!(cg.function_count(), 0);
    }

    #[test]
    fn test_add_function() {
        let mut cg = CallGraph::new();
        let func = FunctionNode::new(FunctionId(0), "test");
        let idx = cg.add_function(func);
        
        assert_eq!(cg.function_count(), 1);
        assert!(cg.find_function("test").is_some());
    }

    #[test]
    fn test_recursive_detection() {
        let mut cg = CallGraph::new();
        
        let func1 = FunctionNode::new(FunctionId(0), "foo");
        let idx1 = cg.add_function(func1);
        
        // 添加自环（直接递归）
        cg.add_call(idx1, idx1, CallSite {
            id: 0,
            caller: FunctionId(0),
            callee_name: "foo".to_string(),
            arg_count: 0,
            line: None,
            column: None,
        });
        
        assert!(cg.has_recursive_calls());
    }

    #[test]
    fn test_callgraph_builder() {
        let mut builder = CallGraphBuilder::new();
        let ast = JsAst {
            function_decls: vec![
                FunctionDecl {
                    name: Some("foo".to_string()),
                    params: vec![],
                    is_async: false,
                    is_generator: false,
                    body: None,
                },
            ],
            variable_decls: vec![],
            ..Default::default()
        };
        
        let cg = builder.build(&ast).unwrap();
        
        assert_eq!(cg.function_count(), 1);
    }
}
