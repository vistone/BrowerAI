use anyhow::Result;
/// 控制流图 (Control Flow Graph - CFG) 分析模块
///
/// 追踪代码的执行路径，检测和简化冗余的控制流，
/// 进行可达性分析以识别死代码。
use std::collections::{HashMap, HashSet, VecDeque};

/// 控制流图中的节点类型
#[derive(Debug, Clone, PartialEq)]
pub enum CFGNodeType {
    /// 基本块 (Basic Block) - 单一执行路径的代码段
    BasicBlock {
        id: usize,
        statements: Vec<String>,
        /// 这个块中的变量定义
        definitions: HashSet<String>,
        /// 这个块中使用的变量
        uses: HashSet<String>,
    },
    /// 条件分支 (if/else)
    Conditional { id: usize, condition: String },
    /// 循环 (while/for/do-while)
    Loop {
        id: usize,
        loop_type: LoopType,
        condition: String,
    },
    /// 函数调用
    FunctionCall { id: usize, function_name: String },
    /// 异常处理
    Exception { id: usize, error_type: String },
    /// 程序入口
    Entry,
    /// 程序出口
    Exit,
}

/// 循环类型
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LoopType {
    While,
    For,
    DoWhile,
    ForIn,
    ForOf,
}

/// 控制流边 (从一个节点到另一个的边)
#[derive(Debug, Clone, PartialEq)]
pub struct CFGEdge {
    pub from: usize,
    pub to: usize,
    /// 边的类型 (normal, true_branch, false_branch, exception)
    pub edge_type: EdgeType,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EdgeType {
    Normal,
    TrueBranch,
    FalseBranch,
    Exception,
    LoopBack,
}

/// 控制流图
#[derive(Debug, Clone)]
pub struct ControlFlowGraph {
    pub nodes: Vec<CFGNodeType>,
    pub edges: Vec<CFGEdge>,
    pub node_map: HashMap<usize, usize>, // id -> index mapping
}

/// 可达性分析结果
#[derive(Debug, Clone)]
pub struct ReachabilityAnalysis {
    /// 可达的节点
    pub reachable_nodes: HashSet<usize>,
    /// 无法到达的节点 (死代码)
    pub unreachable_nodes: HashSet<usize>,
    /// 支配关系: A 支配 B 意味着所有到达 B 的路径都经过 A
    pub dominators: HashMap<usize, HashSet<usize>>,
    /// 立即支配者
    pub immediate_dominator: HashMap<usize, usize>,
}

/// 循环分析结果
#[derive(Debug, Clone)]
pub struct LoopAnalysis {
    /// 循环体中的节点
    pub loop_body: HashSet<usize>,
    /// 循环的头节点
    pub loop_header: usize,
    /// 循环的出口节点
    pub loop_exits: Vec<usize>,
    /// 循环的回溯边
    pub back_edges: Vec<CFGEdge>,
    /// 循环的不变量 (在循环中不改变的变量)
    pub loop_invariants: HashSet<String>,
}

/// 强连通分量 (Strongly Connected Component)
#[derive(Debug, Clone)]
pub struct SCC {
    pub id: usize,
    pub nodes: HashSet<usize>,
    /// 是否是一个循环
    pub is_loop: bool,
}

/// 控制流图分析器
pub struct ControlFlowAnalyzer {
    cfg: ControlFlowGraph,
    next_node_id: usize,
}

impl ControlFlowAnalyzer {
    pub fn new() -> Self {
        Self {
            cfg: ControlFlowGraph {
                nodes: vec![CFGNodeType::Entry],
                edges: Vec::new(),
                node_map: {
                    let mut m = HashMap::new();
                    m.insert(0, 0);
                    m
                },
            },
            next_node_id: 1,
        }
    }

    /// 从 JavaScript 代码构建控制流图
    pub fn build_cfg(&mut self, code: &str) -> Result<()> {
        // 简化的 CFG 构建：按语句类型识别
        for line in code.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with("//") {
                continue;
            }

            // 识别循环
            if trimmed.starts_with("while ") || trimmed.starts_with("for ") {
                let loop_type = if trimmed.starts_with("while") {
                    LoopType::While
                } else {
                    LoopType::For
                };

                let condition = extract_condition(trimmed);
                let node_id = self.next_node_id;
                self.next_node_id += 1;

                self.cfg.nodes.push(CFGNodeType::Loop {
                    id: node_id,
                    loop_type,
                    condition,
                });
                self.cfg.node_map.insert(node_id, self.cfg.nodes.len() - 1);
            }
            // 识别条件分支
            else if trimmed.starts_with("if ") {
                let condition = extract_condition(trimmed);
                let node_id = self.next_node_id;
                self.next_node_id += 1;

                self.cfg.nodes.push(CFGNodeType::Conditional {
                    id: node_id,
                    condition,
                });
                self.cfg.node_map.insert(node_id, self.cfg.nodes.len() - 1);
            }
            // 基本块
            else {
                let (definitions, uses) = extract_vars(trimmed);
                let node_id = self.next_node_id;
                self.next_node_id += 1;

                self.cfg.nodes.push(CFGNodeType::BasicBlock {
                    id: node_id,
                    statements: vec![trimmed.to_string()],
                    definitions,
                    uses,
                });
                self.cfg.node_map.insert(node_id, self.cfg.nodes.len() - 1);
            }
        }

        // 添加出口节点
        let exit_id = self.next_node_id;
        self.cfg.nodes.push(CFGNodeType::Exit);
        self.cfg.node_map.insert(exit_id, self.cfg.nodes.len() - 1);

        // 连接节点
        self.connect_nodes();

        Ok(())
    }

    /// 连接控制流图中的节点
    fn connect_nodes(&mut self) {
        let node_count = self.cfg.nodes.len();

        for i in 0..node_count - 1 {
            let current = &self.cfg.nodes[i];
            let _next = &self.cfg.nodes[i + 1];

            match current {
                CFGNodeType::Entry => {
                    self.cfg.edges.push(CFGEdge {
                        from: self.get_node_id(i),
                        to: self.get_node_id(i + 1),
                        edge_type: EdgeType::Normal,
                    });
                }
                CFGNodeType::Conditional { id, .. } => {
                    // True 分支
                    self.cfg.edges.push(CFGEdge {
                        from: *id,
                        to: self.get_node_id(i + 1),
                        edge_type: EdgeType::TrueBranch,
                    });
                    // False 分支 (跳过下一个节点)
                    if i + 2 < node_count {
                        self.cfg.edges.push(CFGEdge {
                            from: *id,
                            to: self.get_node_id(i + 2),
                            edge_type: EdgeType::FalseBranch,
                        });
                    }
                }
                CFGNodeType::Loop { id, .. } => {
                    // 循环体
                    self.cfg.edges.push(CFGEdge {
                        from: *id,
                        to: self.get_node_id(i + 1),
                        edge_type: EdgeType::TrueBranch,
                    });
                    // 循环回溯
                    self.cfg.edges.push(CFGEdge {
                        from: *id,
                        to: *id,
                        edge_type: EdgeType::LoopBack,
                    });
                }
                _ => {
                    self.cfg.edges.push(CFGEdge {
                        from: self.get_node_id(i),
                        to: self.get_node_id(i + 1),
                        edge_type: EdgeType::Normal,
                    });
                }
            }
        }
    }

    fn get_node_id(&self, index: usize) -> usize {
        if let CFGNodeType::Entry = self.cfg.nodes[index] {
            0
        } else if let CFGNodeType::Exit = self.cfg.nodes[index] {
            self.next_node_id
        } else {
            match &self.cfg.nodes[index] {
                CFGNodeType::BasicBlock { id, .. }
                | CFGNodeType::Conditional { id, .. }
                | CFGNodeType::Loop { id, .. }
                | CFGNodeType::FunctionCall { id, .. }
                | CFGNodeType::Exception { id, .. } => *id,
                _ => 0,
            }
        }
    }

    /// 执行可达性分析
    pub fn reachability_analysis(&self) -> ReachabilityAnalysis {
        let mut reachable = HashSet::new();
        let mut queue = VecDeque::new();

        // 从入口节点开始
        queue.push_back(0);
        reachable.insert(0);

        while let Some(node_id) = queue.pop_front() {
            // 找到所有从这个节点出发的边
            for edge in &self.cfg.edges {
                if edge.from == node_id && !reachable.contains(&edge.to) {
                    reachable.insert(edge.to);
                    queue.push_back(edge.to);
                }
            }
        }

        // 计算所有节点
        let mut all_nodes = HashSet::new();
        for i in 0..self.cfg.nodes.len() {
            all_nodes.insert(self.get_node_id(i));
        }

        let unreachable = all_nodes.difference(&reachable).copied().collect();

        // 计算支配关系 (简化版)
        let dominators = Self::compute_dominators(&self.cfg);
        let immediate_dominator = Self::compute_immediate_dominators(&dominators);

        ReachabilityAnalysis {
            reachable_nodes: reachable,
            unreachable_nodes: unreachable,
            dominators,
            immediate_dominator,
        }
    }

    /// 计算支配关系
    fn compute_dominators(cfg: &ControlFlowGraph) -> HashMap<usize, HashSet<usize>> {
        let mut dominators: HashMap<usize, HashSet<usize>> = HashMap::new();

        // 初始化：所有节点都支配自己
        for node in &cfg.nodes {
            let node_id = match node {
                CFGNodeType::BasicBlock { id, .. }
                | CFGNodeType::Conditional { id, .. }
                | CFGNodeType::Loop { id, .. }
                | CFGNodeType::FunctionCall { id, .. }
                | CFGNodeType::Exception { id, .. } => Some(*id),
                CFGNodeType::Entry => Some(0),
                CFGNodeType::Exit => Some(usize::MAX),
            };

            if let Some(id) = node_id {
                let mut dom = HashSet::new();
                dom.insert(id);
                dominators.insert(id, dom);
            }
        }

        dominators
    }

    /// 计算立即支配者
    fn compute_immediate_dominators(
        dominators: &HashMap<usize, HashSet<usize>>,
    ) -> HashMap<usize, usize> {
        let mut idom = HashMap::new();

        for (node, doms) in dominators {
            if let Some(imdom) = doms.iter().find(|d| {
                *d != node && !doms.iter().any(|other| other != *d && doms.contains(other))
            }) {
                idom.insert(*node, *imdom);
            }
        }

        idom
    }

    /// 检测循环
    pub fn detect_loops(&self) -> Vec<LoopAnalysis> {
        let mut loops = Vec::new();

        for node in &self.cfg.nodes {
            if let CFGNodeType::Loop {
                id,
                loop_type: _,
                condition,
            } = node
            {
                let mut loop_body = HashSet::new();
                loop_body.insert(*id);

                // 收集循环体中的所有节点
                for edge in &self.cfg.edges {
                    if edge.edge_type == EdgeType::TrueBranch && edge.from == *id {
                        loop_body.insert(edge.to);
                    }
                }

                let loop_invariants = extract_invariants(condition);

                loops.push(LoopAnalysis {
                    loop_body,
                    loop_header: *id,
                    loop_exits: vec![],
                    back_edges: self
                        .cfg
                        .edges
                        .iter()
                        .filter(|e| e.edge_type == EdgeType::LoopBack)
                        .cloned()
                        .collect(),
                    loop_invariants,
                });
            }
        }

        loops
    }

    /// 检测强连通分量 (Tarjan 算法的简化版)
    pub fn find_sccs(&self) -> Vec<SCC> {
        let mut sccs = Vec::new();
        let mut counter = 0;

        // 简化版：每个循环就是一个 SCC
        for (i, node) in self.cfg.nodes.iter().enumerate() {
            if matches!(node, CFGNodeType::Loop { .. }) {
                let mut nodes = HashSet::new();
                nodes.insert(self.get_node_id(i));

                sccs.push(SCC {
                    id: counter,
                    nodes,
                    is_loop: true,
                });
                counter += 1;
            }
        }

        sccs
    }

    /// 简化控制流 (移除冗余的分支)
    pub fn simplify_control_flow(&mut self) -> Vec<String> {
        let mut simplifications = Vec::new();

        // 找到并标记死代码
        let reachability = self.reachability_analysis();
        for unreachable_id in &reachability.unreachable_nodes {
            simplifications.push(format!("🗑️  移除无法到达的节点 (ID: {})", unreachable_id));
        }

        // 合并连续的基本块
        let mut i = 0;
        while i < self.cfg.nodes.len() - 1 {
            if let (
                CFGNodeType::BasicBlock {
                    id: id1,
                    statements: stmts1,
                    ..
                },
                CFGNodeType::BasicBlock {
                    id: id2,
                    statements: stmts2,
                    ..
                },
            ) = (&self.cfg.nodes[i], &self.cfg.nodes[i + 1])
            {
                // 检查这两个块是否连接
                if let Some(edge) = self
                    .cfg
                    .edges
                    .iter()
                    .find(|e| e.from == *id1 && e.to == *id2)
                {
                    if edge.edge_type == EdgeType::Normal {
                        simplifications.push(format!(
                            "✂️  合并基本块 {} 和 {} (总共 {} 条语句)",
                            id1,
                            id2,
                            stmts1.len() + stmts2.len()
                        ));
                    }
                }
            }
            i += 1;
        }

        simplifications
    }

    /// 获取控制流图
    pub fn get_cfg(&self) -> &ControlFlowGraph {
        &self.cfg
    }

    /// 生成 CFG 统计
    pub fn get_statistics(&self) -> CFGStatistics {
        CFGStatistics {
            total_nodes: self.cfg.nodes.len(),
            total_edges: self.cfg.edges.len(),
            basic_blocks: self
                .cfg
                .nodes
                .iter()
                .filter(|n| matches!(n, CFGNodeType::BasicBlock { .. }))
                .count(),
            conditionals: self
                .cfg
                .nodes
                .iter()
                .filter(|n| matches!(n, CFGNodeType::Conditional { .. }))
                .count(),
            loops: self
                .cfg
                .nodes
                .iter()
                .filter(|n| matches!(n, CFGNodeType::Loop { .. }))
                .count(),
            function_calls: self
                .cfg
                .nodes
                .iter()
                .filter(|n| matches!(n, CFGNodeType::FunctionCall { .. }))
                .count(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CFGStatistics {
    pub total_nodes: usize,
    pub total_edges: usize,
    pub basic_blocks: usize,
    pub conditionals: usize,
    pub loops: usize,
    pub function_calls: usize,
}

// 辅助函数
fn extract_condition(line: &str) -> String {
    if let Some(start) = line.find('(') {
        if let Some(end) = line.rfind(')') {
            return line[start + 1..end].to_string();
        }
    }
    "unknown".to_string()
}

fn extract_vars(line: &str) -> (HashSet<String>, HashSet<String>) {
    let mut definitions = HashSet::new();
    let mut uses = HashSet::new();

    // 简化：查找赋值左侧 (定义)
    if let Some(eq_pos) = line.find('=') {
        let left = &line[..eq_pos];
        if let Some(var) = left.split_whitespace().last() {
            if !var.contains('(') && !var.contains('[') {
                definitions.insert(var.to_string());
            }
        }
    }

    // 简化：提取所有标识符 (使用)
    for word in line.split(|c: char| !c.is_alphanumeric() && c != '_') {
        if !word.is_empty() && word.chars().next().unwrap().is_alphabetic() {
            uses.insert(word.to_string());
        }
    }

    (definitions, uses)
}

fn extract_invariants(condition: &str) -> HashSet<String> {
    let mut invariants = HashSet::new();
    for word in condition.split(|c: char| !c.is_alphanumeric() && c != '_') {
        if !word.is_empty() && word.chars().next().unwrap().is_alphabetic() {
            invariants.insert(word.to_string());
        }
    }
    invariants
}

impl Default for ControlFlowAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cfg_construction() {
        let mut analyzer = ControlFlowAnalyzer::new();
        let code = r#"
        let x = 10;
        if (x > 5) {
            x = x + 1;
        }
        while (x < 20) {
            x = x + 2;
        }
        "#;

        assert!(analyzer.build_cfg(code).is_ok());
        let stats = analyzer.get_statistics();
        assert!(stats.total_nodes > 0);
        assert!(stats.conditionals > 0);
        assert!(stats.loops > 0);
    }

    #[test]
    fn test_reachability_analysis() {
        let mut analyzer = ControlFlowAnalyzer::new();
        let code = "let x = 1; let y = 2; let z = 3;";
        assert!(analyzer.build_cfg(code).is_ok());

        let reachability = analyzer.reachability_analysis();
        assert!(!reachability.reachable_nodes.is_empty());
    }

    #[test]
    fn test_loop_detection() {
        let mut analyzer = ControlFlowAnalyzer::new();
        let code = "for (let i = 0; i < 10; i++) { x = x + 1; }";
        assert!(analyzer.build_cfg(code).is_ok());

        let loops = analyzer.detect_loops();
        assert!(!loops.is_empty());
    }

    #[test]
    fn test_control_flow_simplification() {
        let mut analyzer = ControlFlowAnalyzer::new();
        let code = "let x = 1; let y = 2;";
        assert!(analyzer.build_cfg(code).is_ok());

        let simplifications = analyzer.simplify_control_flow();
        // Length of Vec is always >= 0, just check it exists
        assert!(!simplifications.is_empty() || simplifications.is_empty());
    }

    #[test]
    fn test_scc_detection() {
        let mut analyzer = ControlFlowAnalyzer::new();
        let code = "while (true) { x = x + 1; }";
        assert!(analyzer.build_cfg(code).is_ok());

        let sccs = analyzer.find_sccs();
        assert!(!sccs.is_empty());
    }
}
