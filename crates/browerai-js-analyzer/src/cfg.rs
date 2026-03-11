//! Control Flow Graph - 控制流图
//!
//! 构建JavaScript代码的控制流图，包括：
//! - 基本块 (BasicBlock)
//! - 控制流边 (Control Flow Edges)
//! - 分支类型 (BranchKind)
//! - 可达性分析

use browerai_core::Result;
use browerai_js_parser::JsAst;
use petgraph::graph::{DiGraph, NodeIndex};
use std::collections::HashMap;

/// 控制流图
#[derive(Debug, Clone)]
pub struct ControlFlowGraph {
    /// 底层图结构
    graph: DiGraph<BasicBlock, BranchKind>,
    /// 入口节点
    entry: NodeIndex,
    /// 出口节点
    exit: NodeIndex,
    /// 节点映射（语句索引 -> 图节点）
    node_map: HashMap<usize, NodeIndex>,
}

impl ControlFlowGraph {
    /// 创建新的CFG
    pub fn new() -> Self {
        let mut graph = DiGraph::new();
        let entry = graph.add_node(BasicBlock::entry());
        let exit = graph.add_node(BasicBlock::exit());

        Self {
            graph,
            entry,
            exit,
            node_map: HashMap::new(),
        }
    }

    /// 添加入口边
    pub fn add_entry_edge(&mut self, target: NodeIndex, kind: BranchKind) {
        self.graph.add_edge(self.entry, target, kind);
    }

    /// 添加基本块
    pub fn add_block(&mut self, block: BasicBlock) -> NodeIndex {
        self.graph.add_node(block)
    }

    /// 添加控制流边
    pub fn add_edge(&mut self, from: NodeIndex, to: NodeIndex, kind: BranchKind) {
        self.graph.add_edge(from, to, kind);
    }

    /// 获取入口节点
    pub fn entry(&self) -> NodeIndex {
        self.entry
    }

    /// 获取出口节点
    pub fn exit(&self) -> NodeIndex {
        self.exit
    }

    /// 获取节点映射数量
    pub fn node_map_len(&self) -> usize {
        self.node_map.len()
    }

    /// 获取基本块
    pub fn get_block(&self, index: NodeIndex) -> Option<&BasicBlock> {
        self.graph.node_weight(index)
    }

    /// 获取可变基本块
    pub fn get_block_mut(&mut self, index: NodeIndex) -> Option<&mut BasicBlock> {
        self.graph.node_weight_mut(index)
    }

    /// 获取后继节点
    pub fn successors(&self, index: NodeIndex) -> Vec<NodeIndex> {
        self.graph.neighbors(index).collect()
    }

    /// 获取前驱节点
    pub fn predecessors(&self, index: NodeIndex) -> Vec<NodeIndex> {
        self.graph
            .neighbors_directed(index, petgraph::Direction::Incoming)
            .collect()
    }

    /// 检查是否为空
    pub fn is_empty(&self) -> bool {
        self.graph.node_count() <= 2 // 只有entry和exit
    }

    /// 获取节点数量
    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    /// 获取边数量
    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// 可达性分析 - 检查从入口是否可以到达某个节点
    pub fn is_reachable(&self, target: NodeIndex) -> bool {
        use petgraph::visit::Bfs;

        let mut bfs = Bfs::new(&self.graph, self.entry);
        while let Some(node) = bfs.next(&self.graph) {
            if node == target {
                return true;
            }
        }
        false
    }

    /// 查找死代码（不可达的节点）
    pub fn find_dead_code(&self) -> Vec<NodeIndex> {
        let mut dead = Vec::new();

        for node in self.graph.node_indices() {
            if node != self.entry && !self.is_reachable(node) {
                dead.push(node);
            }
        }

        dead
    }

    /// 获取所有基本块
    pub fn blocks(&self) -> impl Iterator<Item = (NodeIndex, &BasicBlock)> {
        self.graph
            .node_indices()
            .map(move |idx| (idx, self.graph.node_weight(idx).unwrap()))
    }
}

impl Default for ControlFlowGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// 基本块
#[derive(Debug, Clone)]
pub struct BasicBlock {
    /// 块ID
    pub id: usize,
    /// 块标签（用于调试）
    pub label: Option<String>,
    /// 语句列表
    pub statements: Vec<String>,
    /// 是否是入口块
    pub is_entry: bool,
    /// 是否是出口块
    pub is_exit: bool,
}

impl BasicBlock {
    /// 创建新基本块
    pub fn new(id: usize) -> Self {
        Self {
            id,
            label: None,
            statements: Vec::new(),
            is_entry: false,
            is_exit: false,
        }
    }

    /// 创建入口块
    pub fn entry() -> Self {
        Self {
            id: 0,
            label: Some("entry".to_string()),
            statements: Vec::new(),
            is_entry: true,
            is_exit: false,
        }
    }

    /// 创建出口块
    pub fn exit() -> Self {
        Self {
            id: 1,
            label: Some("exit".to_string()),
            statements: Vec::new(),
            is_entry: false,
            is_exit: true,
        }
    }

    /// 添加语句
    pub fn add_statement(&mut self, stmt: impl Into<String>) {
        self.statements.push(stmt.into());
    }

    /// 获取语句数量
    pub fn statement_count(&self) -> usize {
        self.statements.len()
    }

    /// 是否为空
    pub fn is_empty(&self) -> bool {
        self.statements.is_empty()
    }
}

/// 分支类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BranchKind {
    /// 无条件跳转
    Unconditional,
    /// 条件为真时跳转
    True,
    /// 条件为假时跳转
    False,
    /// 循环回边
    LoopBack,
    /// 跳出循环
    Break,
    /// 继续循环
    Continue,
    /// 返回
    Return,
    /// 异常
    Exception,
    /// 合并（多路径汇合）
    Merge,
}

/// CFG构建器
#[derive(Debug, Clone, Default)]
pub struct CfgBuilder {
    /// 下一个块ID
    next_block_id: usize,
}

impl CfgBuilder {
    /// 创建新的CFG构建器
    pub fn new() -> Self {
        Self {
            next_block_id: 2, // 0和1留给entry和exit
        }
    }

    /// 从AST构建CFG
    pub fn build(&mut self, ast: &JsAst) -> Result<ControlFlowGraph> {
        let mut cfg = ControlFlowGraph::new();

        // 创建主函数块
        let main_block = self.create_block("main");
        let main_idx = cfg.add_block(main_block);

        // 连接入口到主块
        cfg.add_entry_edge(main_idx, BranchKind::Unconditional);

        // 为主块添加语句
        if let Some(block) = cfg.get_block_mut(main_idx) {
            for func in &ast.function_decls {
                if let Some(ref name) = func.name {
                    block.add_statement(format!("function {}", name));
                }
            }
            for var in &ast.variable_decls {
                block.add_statement(format!("var {}", var.name));
            }
        }

        // 连接主块到出口
        cfg.add_edge(main_idx, cfg.exit(), BranchKind::Unconditional);

        Ok(cfg)
    }

    /// 创建新基本块
    fn create_block(&mut self, label: impl Into<String>) -> BasicBlock {
        let id = self.next_block_id;
        self.next_block_id += 1;

        BasicBlock {
            id,
            label: Some(label.into()),
            statements: Vec::new(),
            is_entry: false,
            is_exit: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cfg_creation() {
        let cfg = ControlFlowGraph::new();

        assert_eq!(cfg.node_count(), 2); // entry + exit
        assert!(cfg.is_empty());
    }

    #[test]
    fn test_add_blocks() {
        let mut cfg = ControlFlowGraph::new();
        let block = BasicBlock::new(2);
        let idx = cfg.add_block(block);

        assert_eq!(cfg.node_count(), 3);
        assert!(cfg.get_block(idx).is_some());
    }

    #[test]
    fn test_reachability() {
        let mut cfg = ControlFlowGraph::new();
        let block = BasicBlock::new(2);
        let idx = cfg.add_block(block);

        // 未连接时不可达
        assert!(!cfg.is_reachable(idx));

        // 连接后可达
        cfg.add_entry_edge(idx, BranchKind::Unconditional);
        assert!(cfg.is_reachable(idx));
    }

    #[test]
    fn test_cfg_builder() {
        let mut builder = CfgBuilder::new();
        let ast = JsAst {
            function_decls: vec![],
            variable_decls: vec![],
            ..Default::default()
        };

        let cfg = builder.build(&ast).unwrap();

        assert!(cfg.node_count() >= 2);
    }
}
